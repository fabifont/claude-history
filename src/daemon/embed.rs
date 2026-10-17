//! The daemon's embedding job, run in a child process so that the model and
//! everything embedding allocates are returned to the system when it ends.
//!
//! A job embeds the passages of some conversations, or of every
//! conversation newest first (a sweep), that the semantic cache has no
//! embedding for. Passages are the TUI's: the visible dialogue of each card
//! of the keyword store's current generation, chunked the same way, so the
//! TUI finds them in the cache. Conversations are read and chunked a page at
//! a time, so memory stays bounded whatever the size of the history.

use crate::error::Result;
use crate::semantic::cache::{
    MAX_CACHE_ENTRIES, embed_chunks_with_budget_and_save, embedding_cache_key,
    read_embedding_cache_in, write_embedding_cache_in,
};
use crate::semantic::chunk::build_chunks_with_sources;
use crate::semantic::embed::SemanticEmbedder;
use crate::semantic::matrix::EmbeddingKey;
use crate::semantic::types::{
    ChunkConfig, EmbeddingCache, SemanticCancellationToken, SemanticChunk, SemanticChunkSource,
};
use crate::store::{SemanticReader, Snapshot};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Conversations read and chunked at a time.
const PAGE: usize = 64;
/// Passages a sweep keeps embedded at most, below the cache's capacity so
/// that pruning the cache never drops passages the sweep embeds again.
pub const SWEEP_CAP: usize = MAX_CACHE_ENTRIES / 10 * 9;
/// Interval between writes of the cache while a job embeds; it is written
/// when the job ends too. Each write replaces the whole matrix.
pub const FLUSH_INTERVAL: Duration = Duration::from_secs(5 * 60);

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Job {
    /// The conversations whose session files are these paths.
    Conversations { paths: Vec<PathBuf> },
    /// Every conversation, newest first.
    Sweep,
}

/// Progress of a job, reported after each page and at its end.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct Report {
    /// Passages embedded so far.
    pub embedded: u64,
    /// Passages without an embedding that the job knows of and has not
    /// embedded yet.
    pub missing: u64,
    /// Conversations the job covers.
    pub conversations: u64,
    /// Whether the cache was written since the last report.
    pub written: bool,
    /// Whether the job is over.
    pub done: bool,
    /// Why the job stopped before embedding everything, if it did.
    pub stopped: Option<String>,
    pub error: Option<String>,
}

pub struct Context<'a> {
    pub store_root: &'a Path,
    pub semantic_dir: &'a Path,
    /// Makes the embedder, once the job finds something to embed.
    pub new_embedder: &'a mut dyn FnMut() -> Result<Box<dyn SemanticEmbedder>>,
    /// Cancelled to stop embedding at once.
    pub cancellation: &'a SemanticCancellationToken,
    /// Why to stop before the next page, if anything.
    pub interrupt: &'a mut dyn FnMut() -> Option<String>,
    pub report: &'a mut dyn FnMut(&Report),
    pub flush_interval: Duration,
}

/// Writes the cache when the flush interval has passed or when forced.
struct Flusher<'a> {
    dir: &'a Path,
    interval: Duration,
    last: Instant,
    unsaved: bool,
    written: bool,
}

impl Flusher<'_> {
    fn maybe(&mut self, cache: &EmbeddingCache, force: bool) {
        if self.unsaved && (force || self.last.elapsed() >= self.interval) {
            write_embedding_cache_in(cache, self.dir);
            self.last = Instant::now();
            self.unsaved = false;
            self.written = true;
        }
    }
}

/// Runs `job` and returns its last report, which `context.report` also
/// receives.
pub fn run(job: &Job, context: Context<'_>) -> Report {
    let Context {
        store_root,
        semantic_dir,
        new_embedder,
        cancellation,
        interrupt,
        report,
        flush_interval,
    } = context;
    let mut progress = Report::default();
    let finish = |mut progress: Report, report: &mut dyn FnMut(&Report)| {
        progress.done = true;
        report(&progress);
        progress
    };
    let Some(reader) =
        Snapshot::open(store_root, false, Default::default()).and_then(SemanticReader::open)
    else {
        progress.error = Some("the keyword store has no readable generation yet".into());
        return finish(progress, report);
    };
    let docs: Vec<u32> = match job {
        Job::Conversations { paths } => {
            let mut docs: Vec<u32> = paths
                .iter()
                .filter_map(|path| reader.cards().find(path))
                .collect();
            docs.sort_unstable();
            docs.dedup();
            docs
        }
        Job::Sweep => reader.cards().docs_newest_first(),
    };
    progress.conversations = docs.len() as u64;
    let config = ChunkConfig::default();
    let mut cache = read_embedding_cache_in(semantic_dir, config);

    // Which pages have passages to embed, newest first, up to the cap.
    let mut allowed: HashSet<EmbeddingKey> = HashSet::new();
    let mut pages: Vec<&[u32]> = Vec::new();
    for page in docs.chunks(PAGE) {
        if let Some(reason) = interrupt() {
            progress.stopped = Some(reason);
            return finish(progress, report);
        }
        let mut missing = 0;
        for chunk in page_chunks(&reader, page, config) {
            let key = embedding_cache_key(&chunk.text);
            if allowed.len() >= SWEEP_CAP && !allowed.contains(&key) {
                continue;
            }
            if allowed.insert(key) && !cache.entries.contains_key(&key) {
                missing += 1;
            }
        }
        if missing > 0 {
            pages.push(page);
            progress.missing += missing;
        }
    }
    report(&progress);

    let mut flusher = Flusher {
        dir: semantic_dir,
        interval: flush_interval,
        last: Instant::now(),
        unsaved: false,
        written: false,
    };
    let mut embedder: Option<Box<dyn SemanticEmbedder>> = None;
    for page in pages {
        if let Some(reason) = interrupt() {
            progress.stopped = Some(reason);
            break;
        }
        let chunks: Vec<SemanticChunk> = page_chunks(&reader, page, config)
            .into_iter()
            .filter(|chunk| allowed.contains(&embedding_cache_key(&chunk.text)))
            .collect();
        let missing = chunks
            .iter()
            .map(|chunk| embedding_cache_key(&chunk.text))
            .filter(|key| !cache.entries.contains_key(key))
            .collect::<HashSet<_>>()
            .len() as u64;
        if missing == 0 {
            continue;
        }
        let model = match &mut embedder {
            Some(model) => model,
            None => match new_embedder() {
                Ok(model) => embedder.insert(model),
                Err(error) => {
                    progress.error = Some(error.to_string());
                    break;
                }
            },
        };
        let result = embed_chunks_with_budget_and_save(
            model.as_mut(),
            chunks,
            &mut cache,
            cancellation,
            None,
            |_, _| {},
            |cache| {
                flusher.unsaved = true;
                flusher.maybe(cache, false);
            },
        );
        match result {
            Ok(_) => {
                progress.embedded += missing;
                progress.missing = progress.missing.saturating_sub(missing);
                flusher.unsaved = true;
                flusher.maybe(&cache, false);
                progress.written = std::mem::take(&mut flusher.written);
                report(&progress);
            }
            Err(crate::error::AppError::SemanticSearchCancelled) => {
                progress.stopped = Some("cancelled".into());
                break;
            }
            Err(error) => {
                progress.error = Some(error.to_string());
                break;
            }
        }
    }
    flusher.maybe(&cache, true);
    progress.written = std::mem::take(&mut flusher.written);
    finish(progress, report)
}

/// The TUI's passages of the conversations `docs`.
fn page_chunks(reader: &SemanticReader, docs: &[u32], config: ChunkConfig) -> Vec<SemanticChunk> {
    let conversations: Vec<_> = docs
        .iter()
        .filter_map(|&doc| reader.conversation(doc))
        .collect();
    build_chunks_with_sources(
        conversations
            .iter()
            .enumerate()
            .map(|(index, conversation)| {
                (index, SemanticChunkSource::VisibleDialogue, conversation)
            }),
        config,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{generation, test_support};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct FakeEmbedder {
        passages: Arc<AtomicUsize>,
    }

    impl SemanticEmbedder for FakeEmbedder {
        fn embed_passages(&mut self, passages: &[String]) -> Result<Vec<Vec<f32>>> {
            self.passages.fetch_add(passages.len(), Ordering::Relaxed);
            Ok(passages
                .iter()
                .map(|text| vec![text.len() as f32, 1.0, 0.5])
                .collect())
        }

        fn embed_query(&mut self, _query: &str) -> Result<Option<Vec<f32>>> {
            Ok(Some(vec![1.0, 0.0, 0.0]))
        }
    }

    struct Fixture {
        _dir: tempfile::TempDir,
        store: PathBuf,
        semantic: PathBuf,
        paths: Vec<PathBuf>,
        passages: Arc<AtomicUsize>,
        made: usize,
    }

    impl Fixture {
        /// A store of `count` conversations, the first the newest, each
        /// with one distinct turn.
        fn new(count: usize) -> Self {
            let dir = tempfile::tempdir().unwrap();
            let store = dir.path().join("store");
            let conversations: Vec<_> = (0..count)
                .map(|i| {
                    let mut conversation = test_support::conversation(
                        &format!("/p/-a/s{i}.jsonl"),
                        1_000_000 - i as i64,
                    );
                    conversation.semantic_turns =
                        vec![format!("turn of conversation {i} about walruses")].into();
                    conversation.semantic_turn_ranges =
                        vec![crate::agent::refs::MessageRange::single(1)];
                    conversation
                })
                .collect();
            generation::publish(&store, &conversations).unwrap();
            Self {
                semantic: dir.path().join("semantic"),
                paths: conversations.iter().map(|c| c.path.clone()).collect(),
                store,
                _dir: dir,
                passages: Arc::default(),
                made: 0,
            }
        }

        fn run(&mut self, job: &Job, interrupt_after: Option<usize>) -> (Report, Vec<Report>) {
            let mut reports = Vec::new();
            let passages = self.passages.clone();
            let made = &mut self.made;
            let mut new_embedder = || {
                *made += 1;
                Ok(Box::new(FakeEmbedder {
                    passages: passages.clone(),
                }) as Box<dyn SemanticEmbedder>)
            };
            let mut checks = 0;
            let mut interrupt = || {
                checks += 1;
                interrupt_after
                    .filter(|&after| checks > after)
                    .map(|_| "on battery".to_owned())
            };
            let report = run(
                job,
                Context {
                    store_root: &self.store,
                    semantic_dir: &self.semantic,
                    new_embedder: &mut new_embedder,
                    cancellation: &SemanticCancellationToken::new(),
                    interrupt: &mut interrupt,
                    report: &mut |report| reports.push(report.clone()),
                    flush_interval: Duration::from_secs(3600),
                },
            );
            (report, reports)
        }

        fn cached(&self) -> usize {
            read_embedding_cache_in(&self.semantic, ChunkConfig::default())
                .entries
                .len()
        }
    }

    #[test]
    fn a_sweep_embeds_every_missing_passage_once_and_writes_the_cache() {
        let mut fixture = Fixture::new(PAGE + 10);
        let (report, reports) = fixture.run(&Job::Sweep, None);
        assert_eq!(report.error, None);
        assert!(report.done && report.written);
        assert_eq!(report.embedded, (PAGE + 10) as u64);
        assert_eq!(report.missing, 0);
        assert_eq!(reports[0].missing, (PAGE + 10) as u64);
        assert_eq!(fixture.cached(), PAGE + 10);
        assert_eq!(fixture.passages.load(Ordering::Relaxed), PAGE + 10);

        // Nothing is left: no model is loaded and nothing is written.
        let made = fixture.made;
        let (report, _) = fixture.run(&Job::Sweep, None);
        assert_eq!((report.embedded, report.missing), (0, 0));
        assert!(!report.written);
        assert_eq!(fixture.made, made);
    }

    #[test]
    fn a_conversation_job_embeds_only_its_conversations() {
        let mut fixture = Fixture::new(5);
        let job = Job::Conversations {
            paths: vec![
                fixture.paths[3].clone(),
                PathBuf::from("/not/in/the/store.jsonl"),
            ],
        };
        let (report, _) = fixture.run(&job, None);
        assert_eq!((report.conversations, report.embedded), (1, 1));
        assert_eq!(fixture.cached(), 1);
        let (report, _) = fixture.run(&Job::Sweep, None);
        assert_eq!(report.embedded, 4);
    }

    #[test]
    fn an_interrupted_sweep_keeps_what_it_embedded_and_resumes_newest_first() {
        let mut fixture = Fixture::new(PAGE * 3);
        // Three pages are checked, one is embedded, then it stops.
        let (report, _) = fixture.run(&Job::Sweep, Some(4));
        assert_eq!(report.stopped.as_deref(), Some("on battery"));
        assert!(report.done && report.written);
        assert_eq!(report.embedded, PAGE as u64);
        assert_eq!(report.missing, 2 * PAGE as u64);
        let cache = read_embedding_cache_in(&fixture.semantic, ChunkConfig::default());
        let newest = "turn of conversation 0 about walruses";
        assert!(crate::semantic::cache::cache_contains_text(&cache, newest));

        let (report, _) = fixture.run(&Job::Sweep, None);
        assert_eq!(report.embedded, 2 * PAGE as u64);
        assert_eq!(fixture.cached(), 3 * PAGE);
    }

    #[test]
    fn a_missing_store_is_reported_as_an_error() {
        let mut fixture = Fixture::new(1);
        std::fs::remove_dir_all(&fixture.store).unwrap();
        let (report, _) = fixture.run(&Job::Sweep, None);
        assert!(report.error.is_some());
        assert_eq!(fixture.made, 0);
    }
}
