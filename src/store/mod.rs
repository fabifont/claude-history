//! Disk snapshot of the conversation list and its lexical index, readable
//! before any history is loaded.
//!
//! The snapshot is a generation directory under `v3/roots-<hash>/` in the
//! cache dir (see `default_root` and `generation`). The TUI lists the
//! generation's cards, searches its index, and swaps to each newer generation
//! that `reconcile` publishes.

pub mod card;
pub mod generation;
pub mod lexindex;
pub mod reconcile;
pub mod search;
pub mod segments;
pub mod semantic;
pub mod text;

pub use generation::{Generation, ProjectView};

use crate::history::Conversation;
use crate::time_filter::TimeFilter;
use chrono::{DateTime, Local};
use std::borrow::Cow;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

/// Store root: `v3/roots-<hash>/` beside the other caches. The cache dir is
/// namespaced by `CLAUDE_CONFIG_DIR` like the other caches, and the hash
/// covers the resolved Pi and OMP session roots, so a snapshot is only read
/// by runs that load the same sources. Tests get no default root, so they
/// never touch the user's cache.
pub fn default_root() -> Option<PathBuf> {
    if cfg!(test) {
        return None;
    }
    let pi = crate::history::pi_loader::session_root().ok();
    let omp = crate::history::omp_loader::session_root().ok();
    let namespace = source_roots_namespace(
        pi.as_ref().map(|root| root.path.as_path()),
        omp.as_ref().map(|root| root.path.as_path()),
    );
    crate::history::cache::cache_base_dir().map(|base| base.join("v3").join(namespace))
}

/// Directory name identifying the Pi and OMP session roots, canonicalized
/// when they exist.
fn source_roots_namespace(pi: Option<&Path>, omp: Option<&Path>) -> String {
    let mut hasher = blake3::Hasher::new();
    for root in [pi, omp] {
        match root {
            Some(path) => {
                let resolved = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
                let bytes = resolved.as_os_str().as_encoded_bytes();
                hasher.update(&[1]);
                hasher.update(&(bytes.len() as u64).to_le_bytes());
                hasher.update(bytes);
            }
            None => {
                hasher.update(&[0]);
            }
        }
    }
    format!("roots-{}", &hasher.finalize().to_hex()[..16])
}

/// Cards of one generation.
#[derive(Clone)]
pub struct Snapshot {
    generation: Arc<Generation>,
    show_last: bool,
    time: TimeFilter,
    /// Doc of each `CardRow::key_hash`, built on first lookup.
    by_key: Arc<OnceLock<HashMap<u64, u32>>>,
}

impl Snapshot {
    /// Opens the current generation under `root`; `None` when there is no
    /// usable one.
    pub fn open(root: &Path, show_last: bool, time: TimeFilter) -> Option<Self> {
        Some(Self {
            generation: Arc::new(Generation::open(root)?),
            show_last,
            time,
            by_key: Arc::default(),
        })
    }

    pub fn show_last(&self) -> bool {
        self.show_last
    }

    /// Whether `timestamp` is inside the time filter.
    pub fn in_time(&self, timestamp: DateTime<Local>) -> bool {
        !self.time.is_active() || self.time.matches(timestamp)
    }

    /// `CardRow::key_hash` of card `doc`.
    pub fn key_hash(&self, doc: u32) -> u64 {
        self.generation.rows()[doc as usize].key_hash
    }

    /// Doc of every card by `CardRow::key_hash`.
    pub fn docs_by_key(&self) -> &HashMap<u64, u32> {
        self.by_key.get_or_init(|| {
            self.generation
                .rows()
                .iter()
                .enumerate()
                .map(|(doc, row)| (row.key_hash, doc as u32))
                .collect()
        })
    }

    /// Whether a card has `CardRow::key_hash` `key`.
    pub fn contains_key(&self, key: u64) -> bool {
        self.docs_by_key().contains_key(&key)
    }

    /// Doc of the card stored at `path`.
    pub fn find(&self, path: &Path) -> Option<u32> {
        let path = path.to_str()?;
        let doc = *self.docs_by_key().get(&generation::key_hash(path))?;
        let row = &self.generation.rows()[doc as usize];
        (self.generation.str(row.path) == path).then_some(doc)
    }

    /// Doc of the card whose session id is `session_id`, ignoring case.
    pub fn find_session(&self, session_id: &str) -> Option<u32> {
        self.generation
            .rows()
            .iter()
            .position(|row| {
                self.generation
                    .str(row.session_id)
                    .eq_ignore_ascii_case(session_id)
            })
            .map(|doc| doc as u32)
    }

    /// Whether the cards come from more than one source.
    pub fn has_multiple_sources(&self) -> bool {
        let mut rows = self.generation.rows().iter();
        rows.next()
            .is_some_and(|first| rows.any(|row| row.source != first.source))
    }

    pub fn generation_name(&self) -> &str {
        self.generation.name()
    }

    /// Number of cards; doc ids are `0..len()`.
    pub fn len(&self) -> usize {
        self.generation.rows().len()
    }

    /// Ids of the newest `n` cards inside the time filter whose project
    /// passes `project_visible`.
    pub fn head(&self, n: usize, project_visible: impl FnMut(&ProjectView) -> bool) -> Vec<u32> {
        let visible = self.project_visibility(project_visible);
        self.generation.head(n, |row| {
            visible[row.project_id as usize] && in_time(&self.time, row)
        })
    }

    /// For every card, whether it is inside the time filter and its project
    /// passes `project_visible`.
    pub fn scope(&self, project_visible: impl FnMut(&ProjectView) -> bool) -> Vec<bool> {
        let visible = self.project_visibility(project_visible);
        self.generation
            .rows()
            .iter()
            .map(|row| visible[row.project_id as usize] && in_time(&self.time, row))
            .collect()
    }

    fn project_visibility(
        &self,
        mut project_visible: impl FnMut(&ProjectView) -> bool,
    ) -> Vec<bool> {
        (0..self.generation.projects().len() as u32)
            .map(|id| project_visible(&self.generation.project(id)))
            .collect()
    }

    /// Card `doc` as a list-only conversation.
    pub fn card(&self, doc: u32) -> Conversation {
        self.generation.conversation(doc, self.show_last)
    }

    /// The preview card `doc` shows.
    pub fn preview(&self, doc: u32) -> &str {
        let row = &self.generation.rows()[doc as usize];
        self.generation.str(if self.show_last {
            row.preview_last
        } else {
            row.preview_first
        })
    }
}

fn in_time(time: &TimeFilter, row: &card::CardRow) -> bool {
    !time.is_active()
        || chrono::DateTime::from_timestamp_millis(row.timestamp_ms)
            .is_some_and(|timestamp| time.matches(timestamp))
}

/// A generation's cards, suffix index and text, ready for lexical search.
pub struct SearchIndex {
    cards: Snapshot,
    index: segments::IndexSet,
    text: text::TextSegments,
    docs: search::SearchDocs,
}

impl SearchIndex {
    /// Opens the index and text of the generation behind `cards`; `None`
    /// when it has none usable.
    pub fn open(cards: Snapshot) -> Option<Self> {
        let (index, text) = cards.generation.open_search_files()?;
        Some(Self {
            docs: search::SearchDocs::from_generation(&cards.generation),
            cards,
            index,
            text,
        })
    }

    pub fn cards(&self) -> &Snapshot {
        &self.cards
    }

    /// Starts `query` over the documents for which `scope` holds.
    pub fn search<'a>(
        &'a self,
        query: &str,
        now: DateTime<Local>,
        scope: &(dyn Fn(u32) -> bool + Sync),
    ) -> search::Ranking<'a> {
        search::search(&self.index, &self.docs, self, query, now, scope)
    }

    /// `Conversation::full_text` of card `doc`, decompressed from its frame.
    pub fn text(&self, doc: u32) -> String {
        let row = &self.cards.generation.rows()[doc as usize];
        self.text.read(row.text).unwrap_or_default()
    }
}

impl search::TextSource for SearchIndex {
    fn full_text(&self, doc: u32) -> Cow<'_, str> {
        Cow::Owned(self.text(doc))
    }
}

#[cfg(test)]
mod tests {
    use super::source_roots_namespace;
    use std::path::Path;

    #[test]
    fn snapshot_namespace_follows_pi_and_omp_roots() {
        let pi = Path::new("/tmp/claude-history-test/pi-sessions");
        let other_pi = Path::new("/tmp/claude-history-test/other-pi-sessions");
        let omp = Path::new("/tmp/claude-history-test/omp-sessions");

        let base = source_roots_namespace(Some(pi), Some(omp));
        assert!(base.starts_with("roots-"));
        assert_eq!(base, source_roots_namespace(Some(pi), Some(omp)));
        for other in [
            source_roots_namespace(Some(other_pi), Some(omp)),
            source_roots_namespace(Some(pi), Some(other_pi)),
            source_roots_namespace(Some(omp), Some(pi)),
            source_roots_namespace(None, Some(omp)),
            source_roots_namespace(Some(pi), None),
        ] {
            assert_ne!(base, other);
        }
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use crate::history::{Conversation, Source};
    use chrono::{Local, TimeZone};
    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    /// A Claude conversation stored at `path` with `timestamp_ms`.
    pub(crate) fn conversation(path: &str, timestamp_ms: i64) -> Conversation {
        let path = PathBuf::from(path);
        let session_id = path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or_default()
            .to_owned();
        let project_name = path
            .parent()
            .and_then(Path::file_name)
            .and_then(|name| name.to_str())
            .and_then(|name| name.rsplit('-').next())
            .map(str::to_owned);
        let empty: Arc<str> = Arc::from("");
        Conversation {
            source: Source::Claude,
            session_id,
            path,
            index: 0,
            timestamp: Local.timestamp_millis_opt(timestamp_ms).unwrap(),
            preview: "first".into(),
            preview_first: "first".into(),
            preview_last: "last".into(),
            full_text: "full text".into(),
            agent_search_text: empty.clone(),
            semantic_route_text: empty.clone(),
            semantic_turns: Arc::from(Vec::new()),
            semantic_turn_ranges: Vec::new(),
            project_name,
            project_path: None,
            cwd: None,
            message_count: 3,
            parse_errors: Vec::new(),
            summary: None,
            custom_title: None,
            model: None,
            total_tokens: 42,
            duration_minutes: None,
        }
    }
}
