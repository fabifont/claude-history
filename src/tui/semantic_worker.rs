use crate::error::{AppError, Result};
use crate::history::Conversation;
use crate::search::literal::{build_literal_corpus, exact_fallback};
use crate::search::query::ParsedQuery;
use crate::semantic::cache::write_embedding_cache;
use crate::semantic::embed::SemanticEmbedder;
use crate::semantic::fastembed::{EmbedPriority, FastembedEmbedder};
use crate::semantic::index::{
    SemanticIndexCandidate, SemanticIndexProgress, SemanticIndexRequest, SemanticIndexResponse,
    SemanticIndexState,
};
use crate::semantic::types::{
    DEFAULT_EMBEDDING_BATCH_SIZE, SemanticCancellationToken, SemanticChunkIdentity,
    SemanticExplanation, SemanticQuality, SemanticRationaleKind, SemanticScoreBreakdown,
};
use crate::store::SearchIndex;
use crate::tui::app::{SemanticProgress, SemanticResultMetadata};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::mpsc;

#[derive(Clone)]
pub enum SemanticWorkerCommand {
    UpdateCorpus {
        corpus_version: u64,
        conversations: Arc<Vec<Arc<Conversation>>>,
        /// Lexical index whose doc ids are the indices of `conversations`,
        /// whose cards carry no text, for quoted-only queries.
        texts: Option<Arc<SearchIndex>>,
    },
    UpdateScope {
        corpus_version: u64,
        scope_version: u64,
        indices: Arc<Vec<usize>>,
    },
    Search {
        generation: u64,
        query: ParsedQuery,
        corpus_version: u64,
        scope_version: u64,
        prewarm: bool,
    },
}

#[derive(Clone)]
struct SemanticSearchRequest {
    generation: u64,
    query: ParsedQuery,
    corpus_version: u64,
    scope_version: u64,
    prewarm: bool,
}

pub enum SemanticSearchMessage {
    Progress {
        generation: u64,
        progress: SemanticProgress,
    },
    Complete(SemanticSearchResponse),
    /// Progress of embedding the passages the corpus has no embedding for,
    /// which runs behind searches.
    Indexing(SemanticIndexing),
    /// The worker has nothing left to do for the searches up to
    /// `generation`: each is answered and indexing has finished.
    Idle {
        generation: u64,
    },
}

/// Passages embedded since indexing started, of those it has found to embed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SemanticIndexing {
    pub completed: usize,
    pub total: usize,
}

pub struct SemanticSearchResponse {
    pub generation: u64,
    pub filtered: Vec<usize>,
    pub metadata: HashMap<usize, SemanticResultMetadata>,
    pub error: Option<String>,
    pub progress: SemanticProgress,
    pub prewarm: bool,
}

/// Makes the embedder, on the worker thread, when a search first needs it.
pub type EmbedderFactory = Box<dyn FnOnce() -> Result<Box<dyn SemanticEmbedder>> + Send>;

pub fn spawn_semantic_worker() -> (
    mpsc::Sender<SemanticWorkerCommand>,
    mpsc::Receiver<SemanticSearchMessage>,
    SemanticCancellationToken,
) {
    spawn_semantic_worker_with(Box::new(|| {
        let embedder = FastembedEmbedder::new_quiet(EmbedPriority::Background)?;
        Ok(Box::new(embedder) as Box<dyn SemanticEmbedder>)
    }))
}

pub fn spawn_semantic_worker_with(
    new_embedder: EmbedderFactory,
) -> (
    mpsc::Sender<SemanticWorkerCommand>,
    mpsc::Receiver<SemanticSearchMessage>,
    SemanticCancellationToken,
) {
    let (cmd_tx, cmd_rx) = mpsc::channel::<SemanticWorkerCommand>();
    let (res_tx, res_rx) = mpsc::channel::<SemanticSearchMessage>();
    let cancellation = SemanticCancellationToken::new();
    let worker_cancellation = cancellation.clone();

    std::thread::Builder::new()
        .name("semantic-search-worker".into())
        .spawn(move || {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .thread_name(|_| "semantic-rank-worker".to_string())
                .build()
                .expect("failed to create semantic ranking thread pool");
            pool.install(|| run_semantic_worker(cmd_rx, res_tx, worker_cancellation, new_embedder));
        })
        .expect("failed to spawn semantic search worker thread");

    (cmd_tx, res_rx, cancellation)
}

/// The embedder, made on first use.
enum Embedder {
    Unmade(Option<EmbedderFactory>),
    Ready(Box<dyn SemanticEmbedder>),
    Failed(String),
}

impl Embedder {
    fn get(&mut self) -> std::result::Result<&mut dyn SemanticEmbedder, String> {
        if let Self::Unmade(factory) = self {
            let factory = factory.take().expect("factory runs once");
            *self = match factory() {
                Ok(embedder) => Self::Ready(embedder),
                Err(error) => Self::Failed(error.to_string()),
            };
        }
        match self {
            Self::Ready(embedder) => Ok(embedder.as_mut()),
            Self::Failed(error) => Err(error.clone()),
            Self::Unmade(_) => unreachable!("made above"),
        }
    }

    fn is_unmade(&self) -> bool {
        matches!(self, Self::Unmade(_))
    }
}

/// Embedding of the passages a search found without embeddings, done in
/// slices between commands so that searches never wait for it.
struct Indexing {
    /// The search whose results are refreshed once indexing finishes.
    request: SemanticSearchRequest,
    /// Whether `request` has results shown. Until it has, it is ranked
    /// again after each slice and answered once anything matches.
    answered: bool,
    completed: usize,
    /// Passages to embed per slice, sized so a slice takes about
    /// `SLICE_TARGET`.
    budget: usize,
    last_save: std::time::Instant,
}

impl Indexing {
    /// How long a slice should take: short enough that a command waits
    /// little for the running slice, long enough that planning each slice
    /// costs little next to embedding.
    const SLICE_TARGET: std::time::Duration = std::time::Duration::from_secs(2);
    /// Interval between writes of the embedding cache while indexing; the
    /// cache is also written when indexing finishes.
    const SAVE_INTERVAL: std::time::Duration = std::time::Duration::from_secs(30);

    fn new(request: SemanticSearchRequest) -> Self {
        Self {
            request,
            answered: true,
            completed: 0,
            budget: DEFAULT_EMBEDDING_BATCH_SIZE,
            last_save: std::time::Instant::now(),
        }
    }

    fn progress(&self, missing: usize) -> SemanticIndexing {
        SemanticIndexing {
            completed: self.completed,
            total: self.completed + missing,
        }
    }

    fn resize(&mut self, embedded: usize, elapsed: std::time::Duration) {
        if embedded == 0 || elapsed.is_zero() {
            return;
        }
        let per_second = embedded as f64 / elapsed.as_secs_f64();
        let budget = (per_second * Self::SLICE_TARGET.as_secs_f64()) as usize;
        self.budget = budget.clamp(DEFAULT_EMBEDDING_BATCH_SIZE / 2, 4096);
    }

    fn save(&mut self, cache: &crate::semantic::types::EmbeddingCache) {
        if self.last_save.elapsed() >= Self::SAVE_INTERVAL {
            write_embedding_cache(cache);
            self.last_save = std::time::Instant::now();
        }
    }
}

fn run_semantic_worker(
    cmd_rx: mpsc::Receiver<SemanticWorkerCommand>,
    res_tx: mpsc::Sender<SemanticSearchMessage>,
    cancellation_source: SemanticCancellationToken,
    new_embedder: EmbedderFactory,
) {
    let mut worker = SemanticWorkerState::default();
    let mut state = SemanticIndexState::new();
    let mut embedder = Embedder::Unmade(Some(new_embedder));
    let mut indexing: Option<Indexing> = None;
    let mut idle_reported = true;

    loop {
        if indexing.is_none() && !idle_reported {
            idle_reported = true;
            let _ = res_tx.send(SemanticSearchMessage::Idle {
                generation: worker.last_search_generation,
            });
        }
        let command = if indexing.is_some() {
            match cmd_rx.try_recv() {
                Ok(command) => Some(command),
                Err(mpsc::TryRecvError::Empty) => None,
                Err(mpsc::TryRecvError::Disconnected) => break,
            }
        } else {
            match cmd_rx.recv() {
                Ok(command) => Some(command),
                Err(_) => break,
            }
        };
        let Some(command) = command else {
            if let Some(current) = indexing.as_mut() {
                let done = index_slice(
                    &worker,
                    &mut state,
                    &mut embedder,
                    current,
                    &cancellation_source,
                    &res_tx,
                );
                if done {
                    indexing = None;
                }
            }
            continue;
        };

        let mut request = worker.apply_command(command, &cancellation_source);
        while let Ok(pending) = cmd_rx.try_recv() {
            if let Some(search) = worker.apply_command(pending, &cancellation_source) {
                request = Some(search);
            }
        }

        let Some(request) = request else {
            continue;
        };
        idle_reported = false;
        let Some(version_state) = worker.version_state_for(&request) else {
            continue;
        };
        match version_state {
            VersionState::Future => {
                worker.pending_search = Some(request);
                continue;
            }
            VersionState::Stale => continue,
            VersionState::Current => {}
        }

        let cancellation = cancellation_source.child();
        if request.query.is_effectively_empty() && !request.prewarm {
            let _ = res_tx.send(SemanticSearchMessage::Complete(SemanticSearchResponse {
                generation: request.generation,
                filtered: Vec::new(),
                metadata: HashMap::new(),
                error: None,
                progress: SemanticProgress::Idle,
                prewarm: request.prewarm,
            }));
            continue;
        }
        if request.query.is_quoted_only() {
            let response = match &worker.texts {
                Some(texts) => indexed_literal_semantic_response(
                    request.generation,
                    request.prewarm,
                    texts,
                    worker.corpus.as_ref(),
                    worker.scope.as_ref(),
                    &request.query,
                ),
                None => exact_literal_semantic_response(
                    request.generation,
                    request.prewarm,
                    worker.corpus.as_ref(),
                    worker.scope.as_ref(),
                    &request.query,
                ),
            };
            let _ = res_tx.send(SemanticSearchMessage::Complete(response));
            continue;
        }
        let full_corpus = worker.full_corpus_candidates();
        let scope = worker.semantic_candidates();
        let index_request = SemanticIndexRequest {
            query: request.query.semantic_text(),
            literal_filters: request.query.literals(),
            full_corpus: &full_corpus,
            scope: &scope,
            corpus_version: request.corpus_version,
            prewarm: request.prewarm,
            include_chunk_hits: false,
        };
        let response = match state.has_chunks(&index_request, &cancellation) {
            Ok(true) => {
                if embedder.is_unmade() {
                    let _ = res_tx.send(SemanticSearchMessage::Progress {
                        generation: request.generation,
                        progress: SemanticProgress::InitializingModel,
                    });
                }
                match embedder.get() {
                    Ok(model) => rank_embedded(
                        request.generation,
                        &index_request,
                        &request.query,
                        &mut state,
                        model,
                        &cancellation,
                    )
                    .map(|(response, missing)| {
                        if missing == 0 {
                            return Some(response);
                        }
                        let current =
                            indexing.get_or_insert_with(|| Indexing::new(request.clone()));
                        let unanswered = response.filtered.is_empty() && !request.prewarm;
                        if !request.prewarm || current.request.prewarm {
                            current.request = request.clone();
                            current.answered = !unanswered;
                        }
                        let progress = current.progress(missing);
                        let _ = res_tx.send(SemanticSearchMessage::Indexing(progress));
                        if unanswered {
                            // Nothing embedded so far matches; the search is
                            // answered once something does.
                            let _ = res_tx.send(SemanticSearchMessage::Progress {
                                generation: request.generation,
                                progress: SemanticProgress::Embedding {
                                    completed: progress.completed,
                                    total: progress.total,
                                },
                            });
                            return None;
                        }
                        Some(response)
                    })
                    .unwrap_or_else(|error| {
                        Some(failed_semantic_response(
                            request.generation,
                            request.prewarm,
                            error.to_string(),
                        ))
                    }),
                    Err(error) => Some(failed_semantic_response(
                        request.generation,
                        request.prewarm,
                        error,
                    )),
                }
            }
            Ok(false) => Some(match state.clear_empty(&index_request, &cancellation) {
                Ok(()) => empty_semantic_response(request.generation, request.prewarm),
                Err(error) => {
                    failed_semantic_response(request.generation, request.prewarm, error.to_string())
                }
            }),
            Err(error) => Some(failed_semantic_response(
                request.generation,
                request.prewarm,
                error.to_string(),
            )),
        };

        if let Some(response) = response {
            let _ = res_tx.send(SemanticSearchMessage::Complete(response));
        }
    }
}

/// Embeds one slice of the passages still without embeddings. Returns
/// whether indexing is over: every passage is embedded, and the search it
/// serves has been answered again with all of them, or embedding failed.
fn index_slice(
    worker: &SemanticWorkerState,
    state: &mut SemanticIndexState,
    embedder: &mut Embedder,
    indexing: &mut Indexing,
    cancellation_source: &SemanticCancellationToken,
    res_tx: &mpsc::Sender<SemanticSearchMessage>,
) -> bool {
    let Ok(model) = embedder.get() else {
        return true;
    };
    let full_corpus = worker.full_corpus_candidates();
    let index_request = SemanticIndexRequest {
        query: "",
        literal_filters: &[],
        full_corpus: &full_corpus,
        scope: &full_corpus,
        corpus_version: worker.corpus_version,
        prewarm: true,
        include_chunk_hits: false,
    };
    let cancellation = cancellation_source.child();
    let started = std::time::Instant::now();
    let budget = indexing.budget;
    let slice = state.index_slice(&index_request, model, &cancellation, budget, |cache| {
        indexing.save(cache)
    });
    let response = match slice {
        Ok(response) => response,
        // A command is waiting; the next slice continues after it.
        Err(AppError::SemanticSearchCancelled) => return false,
        Err(_) => {
            write_embedding_cache(&state.cache);
            return true;
        }
    };
    indexing.completed += response.embedded_chunk_count;
    indexing.resize(response.embedded_chunk_count, started.elapsed());
    if response.missing_chunk_count > 0 {
        let _ = res_tx.send(SemanticSearchMessage::Indexing(
            indexing.progress(response.missing_chunk_count),
        ));
        if !indexing.answered
            && let Some(response) =
                rank_current(worker, state, model, &indexing.request, &cancellation)
            && !response.filtered.is_empty()
        {
            indexing.answered = true;
            let _ = res_tx.send(SemanticSearchMessage::Complete(response));
        }
        return false;
    }

    write_embedding_cache(&state.cache);
    if let Some(response) = rank_current(worker, state, model, &indexing.request, &cancellation) {
        let _ = res_tx.send(SemanticSearchMessage::Complete(response));
    }
    true
}

/// Ranks `request` over the passages embedded so far, unless it is a
/// prewarm or the corpus or scope has moved on from it.
fn rank_current(
    worker: &SemanticWorkerState,
    state: &mut SemanticIndexState,
    model: &mut dyn SemanticEmbedder,
    request: &SemanticSearchRequest,
    cancellation: &SemanticCancellationToken,
) -> Option<SemanticSearchResponse> {
    if request.prewarm
        || !matches!(
            worker.version_state_for(request),
            Some(VersionState::Current)
        )
    {
        return None;
    }
    let full_corpus = worker.full_corpus_candidates();
    let scope = worker.semantic_candidates();
    let index_request = SemanticIndexRequest {
        query: request.query.semantic_text(),
        literal_filters: request.query.literals(),
        full_corpus: &full_corpus,
        scope: &scope,
        corpus_version: request.corpus_version,
        prewarm: false,
        include_chunk_hits: false,
    };
    rank_embedded(
        request.generation,
        &index_request,
        &request.query,
        state,
        model,
        cancellation,
    )
    .ok()
    .map(|(response, _)| response)
}

#[derive(Default)]
struct SemanticWorkerState {
    corpus_version: u64,
    scope_version: u64,
    scope_corpus_version: u64,
    corpus: Arc<Vec<Arc<Conversation>>>,
    texts: Option<Arc<SearchIndex>>,
    scope: Arc<Vec<usize>>,
    pending_search: Option<SemanticSearchRequest>,
    /// Generation of the newest search received.
    last_search_generation: u64,
}

enum VersionState {
    Current,
    Future,
    Stale,
}

impl SemanticWorkerState {
    fn apply_command(
        &mut self,
        command: SemanticWorkerCommand,
        cancellation: &SemanticCancellationToken,
    ) -> Option<SemanticSearchRequest> {
        match command {
            SemanticWorkerCommand::UpdateCorpus {
                corpus_version,
                conversations,
                texts,
            } => {
                if corpus_version >= self.corpus_version {
                    self.corpus_version = corpus_version;
                    self.corpus = conversations;
                    self.texts = texts;
                    cancellation.cancel();
                }
                self.take_ready_pending()
            }
            SemanticWorkerCommand::UpdateScope {
                corpus_version,
                scope_version,
                indices,
            } => {
                if scope_version >= self.scope_version {
                    self.scope_corpus_version = corpus_version;
                    self.scope_version = scope_version;
                    self.scope = indices;
                    cancellation.cancel();
                }
                self.take_ready_pending()
            }
            SemanticWorkerCommand::Search {
                generation,
                query,
                corpus_version,
                scope_version,
                prewarm,
            } => {
                cancellation.cancel();
                self.last_search_generation = self.last_search_generation.max(generation);
                Some(SemanticSearchRequest {
                    generation,
                    query,
                    corpus_version,
                    scope_version,
                    prewarm,
                })
            }
        }
    }

    fn take_ready_pending(&mut self) -> Option<SemanticSearchRequest> {
        let request = self.pending_search.take()?;
        match self.version_state_for(&request) {
            Some(VersionState::Current) | Some(VersionState::Stale) => Some(request),
            Some(VersionState::Future) | None => {
                self.pending_search = Some(request);
                None
            }
        }
    }

    fn version_state_for(&self, request: &SemanticSearchRequest) -> Option<VersionState> {
        if self.scope_corpus_version != self.corpus_version {
            return Some(VersionState::Future);
        }
        if self.corpus_version == request.corpus_version
            && self.scope_version == request.scope_version
        {
            Some(VersionState::Current)
        } else if self.corpus_version < request.corpus_version
            || self.scope_version < request.scope_version
        {
            Some(VersionState::Future)
        } else {
            Some(VersionState::Stale)
        }
    }

    fn full_corpus_candidates(&self) -> Vec<SemanticIndexCandidate> {
        self.corpus
            .iter()
            .enumerate()
            .map(|(index, conversation)| SemanticIndexCandidate {
                index,
                source: crate::semantic::types::SemanticChunkSource::VisibleDialogue,
                conversation: conversation.clone(),
            })
            .collect()
    }

    fn semantic_candidates(&self) -> Vec<SemanticIndexCandidate> {
        self.scope
            .iter()
            .filter_map(|&index| {
                self.corpus
                    .get(index)
                    .map(|conversation| SemanticIndexCandidate {
                        index,
                        source: crate::semantic::types::SemanticChunkSource::VisibleDialogue,
                        conversation: conversation.clone(),
                    })
            })
            .collect()
    }
}

/// Answers `request` from the passages embedded so far, embedding none.
/// Also returns how many passages of the corpus have no embedding yet.
fn rank_embedded(
    generation: u64,
    request: &SemanticIndexRequest<'_>,
    parsed: &ParsedQuery,
    state: &mut SemanticIndexState,
    embedder: &mut dyn SemanticEmbedder,
    cancellation: &SemanticCancellationToken,
) -> Result<(SemanticSearchResponse, usize)> {
    let response = state.refresh_or_prewarm_with_budget(
        request,
        embedder,
        cancellation,
        Some(0),
        |_| {},
        |_| {},
    )?;
    let missing = response.missing_chunk_count;
    Ok((
        semantic_search_response(generation, response, request.full_corpus, parsed),
        missing,
    ))
}

fn semantic_search_response(
    generation: u64,
    response: SemanticIndexResponse,
    _conversations: &[SemanticIndexCandidate],
    _parsed: &ParsedQuery,
) -> SemanticSearchResponse {
    let filtered = response
        .hits
        .iter()
        .map(|hit| hit.conversation_index)
        .collect::<Vec<_>>();
    let metadata = response
        .hits
        .into_iter()
        .map(|hit| {
            let explanation = hit.explanation;
            (
                hit.conversation_index,
                SemanticResultMetadata {
                    score_breakdown: hit.score_breakdown,
                    explanation,
                },
            )
        })
        .collect();

    SemanticSearchResponse {
        generation,
        filtered,
        metadata,
        error: None,
        progress: semantic_progress(response.progress),
        prewarm: response.prewarm,
    }
}

fn empty_semantic_response(generation: u64, prewarm: bool) -> SemanticSearchResponse {
    SemanticSearchResponse {
        generation,
        filtered: Vec::new(),
        metadata: HashMap::new(),
        error: None,
        progress: SemanticProgress::EmptyCorpus,
        prewarm,
    }
}

fn exact_literal_semantic_response(
    generation: u64,
    prewarm: bool,
    conversations: &[Arc<Conversation>],
    scope: &[usize],
    parsed: &ParsedQuery,
) -> SemanticSearchResponse {
    let plain_conversations = conversations
        .iter()
        .map(|conversation| conversation.as_ref().clone())
        .collect::<Vec<_>>();
    let corpus = build_literal_corpus(&plain_conversations);
    let scope = scope
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>();
    let filtered = exact_fallback(&plain_conversations, &corpus, parsed.literals(), |index| {
        scope.contains(&index)
    });

    let metadata = filtered
        .iter()
        .filter_map(|index| {
            conversations.get(*index).map(|conversation| {
                (
                    *index,
                    exact_literal_metadata(conversation, &conversation.full_text, parsed),
                )
            })
        })
        .collect();

    SemanticSearchResponse {
        generation,
        filtered,
        metadata,
        error: None,
        progress: SemanticProgress::Complete,
        prewarm,
    }
}

/// Like `exact_literal_semantic_response`, for cards whose text is in
/// `texts`: the index finds the matches, newest first, and the text of the
/// first `EVIDENCE_LIMIT` is read for their evidence.
fn indexed_literal_semantic_response(
    generation: u64,
    prewarm: bool,
    texts: &SearchIndex,
    conversations: &[Arc<Conversation>],
    scope: &[usize],
    parsed: &ParsedQuery,
) -> SemanticSearchResponse {
    const EVIDENCE_LIMIT: usize = 200;
    let mut in_scope = vec![false; conversations.len()];
    for &index in scope {
        if let Some(slot) = in_scope.get_mut(index) {
            *slot = true;
        }
    }
    let scope = |doc: u32| in_scope.get(doc as usize).copied().unwrap_or(false);
    let mut ranking = texts.search(parsed.raw(), chrono::Local::now(), &scope, false);
    let filtered: Vec<usize> = ranking
        .complete()
        .iter()
        .map(|hit| hit.doc as usize)
        .filter(|&doc| doc < conversations.len())
        .collect();
    let metadata = filtered
        .iter()
        .enumerate()
        .map(|(rank, &index)| {
            let conversation = &conversations[index];
            let text = if rank < EVIDENCE_LIMIT {
                texts.text(index as u32)
            } else {
                String::new()
            };
            (index, exact_literal_metadata(conversation, &text, parsed))
        })
        .collect();
    SemanticSearchResponse {
        generation,
        filtered,
        metadata,
        error: None,
        progress: SemanticProgress::Complete,
        prewarm,
    }
}

fn literal_evidence_preview(
    conversation: &Conversation,
    full_text: &str,
    parsed: &ParsedQuery,
    visible_preview: &str,
) -> String {
    parsed
        .literals()
        .iter()
        .filter(|literal| !literal.matches(visible_preview))
        .chain(parsed.literals().iter())
        .find_map(|literal| {
            literal
                .match_ranges(full_text)
                .first()
                .map(|range| evidence_window(full_text, *range))
                .or_else(|| {
                    conversation.project_name.as_deref().and_then(|project| {
                        literal
                            .match_ranges(project)
                            .first()
                            .map(|range| evidence_window(project, *range))
                    })
                })
        })
        .unwrap_or_else(|| conversation.preview.to_string())
}

fn evidence_window(text: &str, range: (usize, usize)) -> String {
    const CONTEXT_CHARS: usize = 80;
    let (start, end) = range;
    let prefix_start = text[..start]
        .char_indices()
        .rev()
        .nth(CONTEXT_CHARS)
        .map(|(index, _)| index)
        .unwrap_or(0);
    let suffix_end = text[end..]
        .char_indices()
        .nth(CONTEXT_CHARS)
        .map(|(index, _)| end + index)
        .unwrap_or(text.len());
    let mut preview = String::new();
    if prefix_start > 0 {
        preview.push('…');
    }
    preview.push_str(text[prefix_start..suffix_end].trim());
    if suffix_end < text.len() {
        preview.push('…');
    }
    preview
}

fn exact_literal_metadata(
    conversation: &Conversation,
    full_text: &str,
    parsed: &ParsedQuery,
) -> SemanticResultMetadata {
    SemanticResultMetadata {
        score_breakdown: SemanticScoreBreakdown {
            hybrid: 0.0,
            semantic: 0.0,
            lexical: 0.0,
        },
        explanation: SemanticExplanation {
            quality: SemanticQuality::Strong,
            quality_label: "exact",
            matched_terms: parsed
                .literals()
                .iter()
                .map(|literal| literal.text().to_string())
                .collect(),
            evidence_preview: literal_evidence_preview(conversation, full_text, parsed, ""),
            rationale_kind: SemanticRationaleKind::LexicalBoosted,
            chunk: SemanticChunkIdentity {
                conversation_index: conversation.index,
                source: crate::semantic::types::SemanticChunkSource::VisibleDialogue,
                session: conversation
                    .path
                    .file_stem()
                    .and_then(|stem| stem.to_str())
                    .unwrap_or("?")
                    .to_string(),
                chunk_index: 0,
                message_range: crate::agent::refs::MessageRange::single(1),
            },
        },
    }
}

fn failed_semantic_response(
    generation: u64,
    prewarm: bool,
    error: String,
) -> SemanticSearchResponse {
    SemanticSearchResponse {
        generation,
        filtered: Vec::new(),
        metadata: HashMap::new(),
        error: Some(error),
        progress: SemanticProgress::Failed,
        prewarm,
    }
}

fn semantic_progress(progress: SemanticIndexProgress) -> SemanticProgress {
    match progress {
        SemanticIndexProgress::Embedding { completed, total } => {
            SemanticProgress::Embedding { completed, total }
        }
        SemanticIndexProgress::CacheReady => SemanticProgress::CacheReady,
        SemanticIndexProgress::Ranking => SemanticProgress::Ranking,
        SemanticIndexProgress::Complete => SemanticProgress::Complete,
        SemanticIndexProgress::EmptyCorpus => SemanticProgress::EmptyCorpus,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::history::Conversation;
    use crate::semantic::test_fixtures::{SemanticConversationFixture, beta_hit_metadata};
    use chrono::{Duration as ChronoDuration, Local};
    use std::time::Duration;

    const WORKER_RESPONSE_TIMEOUT: Duration = Duration::from_secs(10);

    fn conversation(path: &str, semantic_turns: Vec<&str>) -> Conversation {
        SemanticConversationFixture::new(path, semantic_turns).build()
    }

    fn corpus(conversations: Vec<Conversation>) -> Arc<Vec<Arc<Conversation>>> {
        Arc::new(conversations.into_iter().map(Arc::new).collect())
    }

    fn send_worker_setup(
        tx: &mpsc::Sender<SemanticWorkerCommand>,
        semantic_turns: Vec<&str>,
        query: &str,
        prewarm: bool,
    ) {
        tx.send(SemanticWorkerCommand::UpdateCorpus {
            corpus_version: 1,
            conversations: corpus(vec![conversation(
                "/projects/project-a/session-a.jsonl",
                semantic_turns,
            )]),
            texts: None,
        })
        .expect("send corpus");
        tx.send(SemanticWorkerCommand::UpdateScope {
            corpus_version: 1,
            scope_version: 1,
            indices: Arc::new(vec![0]),
        })
        .expect("send scope");
        tx.send(SemanticWorkerCommand::Search {
            generation: 1,
            query: ParsedQuery::parse(query),
            corpus_version: 1,
            scope_version: 1,
            prewarm,
        })
        .expect("send semantic request");
    }

    fn recv_empty_complete(rx: &mpsc::Receiver<SemanticSearchMessage>) -> SemanticSearchResponse {
        let message = rx
            .recv_timeout(WORKER_RESPONSE_TIMEOUT)
            .expect("empty response");
        match message {
            SemanticSearchMessage::Complete(response) => {
                assert!(response.filtered.is_empty());
                assert!(response.metadata.is_empty());
                assert_eq!(response.error, None);
                response
            }
            SemanticSearchMessage::Progress { progress, .. } => {
                panic!("unexpected progress before empty response: {progress:?}");
            }
            SemanticSearchMessage::Indexing(indexing) => {
                panic!("unexpected indexing before empty response: {indexing:?}");
            }
            SemanticSearchMessage::Idle { generation } => {
                panic!("unexpected idle before empty response: {generation}");
            }
        }
    }

    /// Embeds passages and queries by whether they mention alpha or beta,
    /// and holds passage embedding while `hold` is set.
    struct FakeEmbedder {
        hold: Arc<std::sync::atomic::AtomicBool>,
    }

    impl FakeEmbedder {
        fn vector(text: &str) -> Vec<f32> {
            vec![
                f32::from(u8::from(text.contains("alpha"))),
                f32::from(u8::from(text.contains("beta"))),
                0.1,
            ]
        }
    }

    impl SemanticEmbedder for FakeEmbedder {
        fn embed_passages(&mut self, passages: &[String]) -> Result<Vec<Vec<f32>>> {
            while self.hold.load(std::sync::atomic::Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(1));
            }
            Ok(passages
                .iter()
                .map(|passage| Self::vector(passage))
                .collect())
        }

        fn embed_query(&mut self, query: &str) -> Result<Option<Vec<f32>>> {
            Ok(Some(Self::vector(query)))
        }
    }

    fn spawn_fake_worker() -> (
        mpsc::Sender<SemanticWorkerCommand>,
        mpsc::Receiver<SemanticSearchMessage>,
        Arc<std::sync::atomic::AtomicBool>,
    ) {
        let hold = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let embedder_hold = hold.clone();
        let (tx, rx, _cancellation) = spawn_semantic_worker_with(Box::new(move || {
            Ok(Box::new(FakeEmbedder {
                hold: embedder_hold,
            }) as Box<dyn SemanticEmbedder>)
        }));
        (tx, rx, hold)
    }

    fn send_corpus(
        tx: &mpsc::Sender<SemanticWorkerCommand>,
        version: u64,
        conversations: Vec<Conversation>,
    ) {
        let indices = (0..conversations.len()).collect();
        tx.send(SemanticWorkerCommand::UpdateCorpus {
            corpus_version: version,
            conversations: corpus(conversations),
            texts: None,
        })
        .expect("send corpus");
        tx.send(SemanticWorkerCommand::UpdateScope {
            corpus_version: version,
            scope_version: version,
            indices: Arc::new(indices),
        })
        .expect("send scope");
    }

    fn send_search(
        tx: &mpsc::Sender<SemanticWorkerCommand>,
        generation: u64,
        version: u64,
        query: &str,
        prewarm: bool,
    ) {
        tx.send(SemanticWorkerCommand::Search {
            generation,
            query: ParsedQuery::parse(query),
            corpus_version: version,
            scope_version: version,
            prewarm,
        })
        .expect("send semantic request");
    }

    fn recv(rx: &mpsc::Receiver<SemanticSearchMessage>) -> SemanticSearchMessage {
        rx.recv_timeout(WORKER_RESPONSE_TIMEOUT)
            .expect("worker message")
    }

    /// Receives until indexing finishes, returning the searches answered on
    /// the way.
    fn recv_until_indexed(
        rx: &mpsc::Receiver<SemanticSearchMessage>,
    ) -> Vec<SemanticSearchResponse> {
        let mut responses = Vec::new();
        loop {
            match recv(rx) {
                SemanticSearchMessage::Complete(response) => responses.push(response),
                SemanticSearchMessage::Idle { .. } => return responses,
                SemanticSearchMessage::Indexing(_) | SemanticSearchMessage::Progress { .. } => {}
            }
        }
    }

    #[test]
    fn maps_domain_hits_to_original_indices_and_metadata() {
        let (expected_score_breakdown, expected_explanation) = beta_hit_metadata(42, "session-b");
        let response = semantic_search_response(
            7,
            SemanticIndexResponse {
                hits: vec![crate::semantic::types::SemanticHit::new(
                    expected_score_breakdown.clone(),
                    expected_explanation.clone(),
                )],
                chunk_hits: Vec::new(),
                indexed_chunk_count: 1,
                missing_chunk_count: 0,
                embedded_chunk_count: 0,
                query_embedding_returned: true,
                progress: SemanticIndexProgress::Complete,
                prewarm: false,
            },
            &[],
            &ParsedQuery::parse("beta"),
        );

        assert_eq!(response.generation, 7);
        assert_eq!(response.filtered, vec![42]);
        assert_eq!(response.progress, SemanticProgress::Complete);
        let metadata = &response.metadata[&42];
        assert_eq!(metadata.score_breakdown, expected_score_breakdown);
        assert_eq!(metadata.explanation, expected_explanation);
    }

    #[test]
    fn empty_visible_dialogue_returns_before_embedder_initialization() {
        let (tx, rx, _cancellation) = spawn_semantic_worker();
        send_worker_setup(&tx, vec![], "alpha", false);
        let response = recv_empty_complete(&rx);
        assert_eq!(response.progress, SemanticProgress::EmptyCorpus);
    }

    #[test]
    fn empty_quoted_search_returns_idle_without_embedding() {
        let (tx, rx, _cancellation) = spawn_semantic_worker();
        send_worker_setup(&tx, vec!["visible dialogue"], "\"\"", false);
        let response = recv_empty_complete(&rx);
        assert_eq!(response.progress, SemanticProgress::Idle);
    }

    #[test]
    fn prewarm_answers_at_once_and_indexes_in_the_background() {
        let (tx, rx, _hold) = spawn_fake_worker();
        send_worker_setup(&tx, vec!["visible dialogue"], "\"\"", true);

        let SemanticSearchMessage::Progress { progress, .. } = recv(&rx) else {
            panic!("expected model initialization first");
        };
        assert_eq!(progress, SemanticProgress::InitializingModel);
        let SemanticSearchMessage::Indexing(indexing) = recv(&rx) else {
            panic!("expected indexing to start");
        };
        assert_eq!(indexing.completed, 0);
        assert!(indexing.total > 0);
        let SemanticSearchMessage::Complete(response) = recv(&rx) else {
            panic!("expected the prewarm to be answered");
        };
        assert!(response.prewarm);
        assert_eq!(response.error, None);
        assert_ne!(response.progress, SemanticProgress::Idle);
        assert!(recv_until_indexed(&rx).is_empty());
    }

    #[test]
    fn search_waiting_for_indexing_is_answered_when_it_finishes() {
        let (tx, rx, _hold) = spawn_fake_worker();
        send_corpus(
            &tx,
            1,
            vec![conversation(
                "/projects/a/session-a.jsonl",
                vec!["alpha notes"],
            )],
        );
        send_search(&tx, 1, 1, "alpha", false);

        let responses = recv_until_indexed(&rx);
        assert_eq!(responses.len(), 1, "answered once, after indexing");
        assert_eq!(responses[0].generation, 1);
        assert_eq!(responses[0].filtered, vec![0]);
    }

    #[test]
    fn search_is_answered_once_indexed_conversations_match() {
        let (tx, rx, _hold) = spawn_fake_worker();
        let mut conversations = vec![conversation(
            "/projects/a/session-a.jsonl",
            vec!["alpha notes"],
        )];
        conversations.extend((1..=DEFAULT_EMBEDDING_BATCH_SIZE * 2).map(|index| {
            let text = format!("beta {index}");
            conversation(&format!("/projects/b/session-{index}.jsonl"), vec![&text])
        }));
        send_corpus(&tx, 1, conversations);
        send_search(&tx, 1, 1, "alpha", false);

        let mut indexing = None;
        let mut first_answer_during = None;
        let mut responses = Vec::new();
        loop {
            match recv(&rx) {
                SemanticSearchMessage::Complete(response) => {
                    first_answer_during = first_answer_during.or(indexing);
                    responses.push(response);
                }
                SemanticSearchMessage::Indexing(progress) => indexing = Some(progress),
                SemanticSearchMessage::Idle { .. } => break,
                SemanticSearchMessage::Progress { .. } => {}
            }
        }
        let first_answer_during = first_answer_during.expect("answered while indexing");
        assert!(first_answer_during.completed > 0);
        assert!(first_answer_during.completed < first_answer_during.total);
        assert_eq!(responses.len(), 2, "answered, then refreshed");
        assert!(
            responses
                .iter()
                .all(|response| response.generation == 1 && response.filtered[0] == 0)
        );
    }

    #[test]
    fn search_ranks_embedded_conversations_while_indexing_new_ones() {
        let (tx, rx, hold) = spawn_fake_worker();
        let alpha = || conversation("/projects/a/session-a.jsonl", vec!["alpha notes"]);
        send_corpus(&tx, 1, vec![alpha()]);
        send_search(&tx, 1, 1, "", true);
        recv_until_indexed(&rx);

        hold.store(true, std::sync::atomic::Ordering::SeqCst);
        send_corpus(
            &tx,
            2,
            vec![
                alpha(),
                conversation("/projects/b/session-b.jsonl", vec!["beta notes"]),
            ],
        );
        send_search(&tx, 2, 2, "alpha", false);
        let first = loop {
            match recv(&rx) {
                SemanticSearchMessage::Complete(response) => break response,
                SemanticSearchMessage::Idle { .. } => panic!("indexing ran while held"),
                _ => {}
            }
        };
        assert_eq!(first.generation, 2);
        assert_eq!(first.filtered, vec![0]);

        hold.store(false, std::sync::atomic::Ordering::SeqCst);
        let refreshed = recv_until_indexed(&rx);
        assert_eq!(refreshed.len(), 1, "refreshed once indexing finishes");
        assert_eq!(refreshed[0].generation, 2);
        assert_eq!(refreshed[0].filtered[0], 0);
    }

    #[test]
    fn quoted_only_search_uses_literal_fallback_without_embedding() {
        let (tx, rx, _cancellation) = spawn_semantic_worker();
        send_worker_setup(&tx, vec![], "\"title sentinel\"", false);
        let message = rx
            .recv_timeout(WORKER_RESPONSE_TIMEOUT)
            .expect("literal response");

        match message {
            SemanticSearchMessage::Complete(response) => {
                assert_eq!(response.filtered, vec![0]);
                assert!(
                    response.metadata[&0]
                        .explanation
                        .evidence_preview
                        .contains("title sentinel")
                );
                assert_eq!(response.error, None);
                assert_eq!(response.progress, SemanticProgress::Complete);
            }
            SemanticSearchMessage::Progress { progress, .. } => {
                panic!("unexpected progress before literal response: {progress:?}");
            }
            SemanticSearchMessage::Indexing(indexing) => {
                panic!("unexpected indexing before literal response: {indexing:?}");
            }
            SemanticSearchMessage::Idle { generation } => {
                panic!("unexpected idle before literal response: {generation}");
            }
        }
    }

    #[test]
    fn quoted_only_search_of_cards_reads_matches_from_the_index() {
        let mut old = conversation("/projects/project-a/session-a.jsonl", vec![]);
        old.timestamp = Local::now() - ChronoDuration::days(1);
        let mut newest = conversation("/projects/project-a/session-b.jsonl", vec![]);
        newest.timestamp = Local::now();
        let mut hidden = conversation("/projects/project-a/session-c.jsonl", vec![]);
        hidden.timestamp = Local::now() + ChronoDuration::days(1);
        let mut unrelated = conversation("/projects/project-a/session-d.jsonl", vec![]);
        unrelated.full_text = "nothing to see".into();
        let root = tempfile::tempdir().unwrap();
        crate::store::generation::publish(root.path(), &[old, newest, hidden, unrelated]).unwrap();
        let snapshot = crate::store::Snapshot::open(
            root.path(),
            false,
            crate::time_filter::TimeFilter::default(),
        )
        .unwrap();
        let cards = crate::store::SemanticCards::load(&snapshot, None).unwrap();
        let index = SearchIndex::open(snapshot).unwrap();
        let paths: Vec<_> = cards
            .conversations()
            .iter()
            .map(|card| card.path.display().to_string())
            .collect();
        let doc = |name: &str| paths.iter().position(|path| path.ends_with(name)).unwrap();
        assert!(
            cards
                .conversations()
                .iter()
                .all(|card| card.full_text.is_empty())
        );

        let response = indexed_literal_semantic_response(
            1,
            false,
            &index,
            cards.conversations(),
            &[
                doc("session-a.jsonl"),
                doc("session-b.jsonl"),
                doc("session-d.jsonl"),
            ],
            &ParsedQuery::parse("\"title sentinel\""),
        );

        assert_eq!(
            response.filtered,
            vec![doc("session-b.jsonl"), doc("session-a.jsonl")]
        );
        for index in &response.filtered {
            assert!(
                response.metadata[index]
                    .explanation
                    .evidence_preview
                    .contains("title sentinel")
            );
        }
        assert_eq!(response.progress, SemanticProgress::Complete);
    }

    #[test]
    fn quoted_only_search_respects_scope_and_orders_newest_first() {
        let mut old = conversation("/projects/project-a/session-a.jsonl", vec![]);
        old.timestamp = Local::now() - ChronoDuration::days(1);
        let mut newest = conversation("/projects/project-a/session-b.jsonl", vec![]);
        newest.timestamp = Local::now();
        let mut hidden = conversation("/projects/project-a/session-c.jsonl", vec![]);
        hidden.timestamp = Local::now() + ChronoDuration::days(1);
        let (tx, rx, _cancellation) = spawn_semantic_worker();
        tx.send(SemanticWorkerCommand::UpdateCorpus {
            corpus_version: 1,
            conversations: corpus(vec![old, newest, hidden]),
            texts: None,
        })
        .expect("send corpus");
        tx.send(SemanticWorkerCommand::UpdateScope {
            corpus_version: 1,
            scope_version: 1,
            indices: Arc::new(vec![0, 1]),
        })
        .expect("send scope");
        tx.send(SemanticWorkerCommand::Search {
            generation: 1,
            query: ParsedQuery::parse("\"title sentinel\""),
            corpus_version: 1,
            scope_version: 1,
            prewarm: false,
        })
        .expect("send semantic request");

        match rx
            .recv_timeout(WORKER_RESPONSE_TIMEOUT)
            .expect("literal response")
        {
            SemanticSearchMessage::Complete(response) => {
                assert_eq!(response.filtered, vec![1, 0]);
                assert!(
                    response.metadata[&1]
                        .explanation
                        .evidence_preview
                        .contains("title sentinel")
                );
                assert!(
                    response.metadata[&0]
                        .explanation
                        .evidence_preview
                        .contains("title sentinel")
                );
            }
            SemanticSearchMessage::Progress { progress, .. } => {
                panic!("unexpected progress before literal response: {progress:?}");
            }
            SemanticSearchMessage::Indexing(indexing) => {
                panic!("unexpected indexing before literal response: {indexing:?}");
            }
            SemanticSearchMessage::Idle { generation } => {
                panic!("unexpected idle before literal response: {generation}");
            }
        }
    }

    #[test]
    fn worker_builds_distinct_full_corpus_and_scoped_candidates() {
        let mut worker = SemanticWorkerState::default();
        let cancellation = SemanticCancellationToken::new();
        worker.apply_command(
            SemanticWorkerCommand::UpdateCorpus {
                corpus_version: 1,
                conversations: corpus(vec![
                    conversation("/projects/project-a/session-a.jsonl", vec!["hidden"]),
                    conversation("/projects/project-a/session-b.jsonl", vec!["visible"]),
                ]),
                texts: None,
            },
            &cancellation,
        );
        worker.apply_command(
            SemanticWorkerCommand::UpdateScope {
                corpus_version: 1,
                scope_version: 1,
                indices: Arc::new(vec![1]),
            },
            &cancellation,
        );

        let full_corpus = worker.full_corpus_candidates();
        let candidates = worker.semantic_candidates();

        assert_eq!(full_corpus.len(), 2);
        assert_eq!(full_corpus[0].index, 0);
        assert_eq!(full_corpus[1].index, 1);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].index, 1);
        assert_eq!(
            candidates[0].conversation.semantic_turns.to_vec(),
            vec!["visible"]
        );
    }

    #[test]
    fn stale_search_is_not_current_after_scope_advances() {
        let mut worker = SemanticWorkerState::default();
        let cancellation = SemanticCancellationToken::new();
        worker.apply_command(
            SemanticWorkerCommand::UpdateCorpus {
                corpus_version: 1,
                conversations: corpus(vec![conversation(
                    "/projects/project-a/session-a.jsonl",
                    vec!["visible"],
                )]),
                texts: None,
            },
            &cancellation,
        );
        worker.apply_command(
            SemanticWorkerCommand::UpdateScope {
                corpus_version: 1,
                scope_version: 2,
                indices: Arc::new(vec![0]),
            },
            &cancellation,
        );
        let request = SemanticSearchRequest {
            generation: 1,
            query: ParsedQuery::parse("visible"),
            corpus_version: 1,
            scope_version: 1,
            prewarm: false,
        };

        assert!(matches!(
            worker.version_state_for(&request),
            Some(VersionState::Stale)
        ));
    }
}
