//! Disk snapshot of the conversation list and its lexical index, readable
//! before any history is loaded.
//!
//! The snapshot is a generation directory under `v3/roots-<hash>/` in the
//! cache dir (see `default_root` and `generation`). The TUI lists the
//! generation's cards, searches its index, and swaps to each newer generation
//! that `reconcile` publishes.

pub mod card;
pub mod generation;
pub mod legacy;
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
use rayon::prelude::*;
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
    cache_base_dir().map(|base| base.join("v3").join(namespace))
}

/// Cache directory for this config root: `~/.cache/claude-history`, or a
/// `config-<hash>` directory inside it when CLAUDE_CONFIG_DIR is set, to
/// avoid cross-config cache collisions.
pub(crate) fn cache_base_dir() -> Option<PathBuf> {
    let base = home::home_dir()?.join(".cache").join("claude-history");
    if let Ok(config_dir) = std::env::var("CLAUDE_CONFIG_DIR") {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        std::hash::Hash::hash(&config_dir, &mut hasher);
        let hash = std::hash::Hasher::finish(&hasher);
        Some(base.join(format!("config-{:016x}", hash)))
    } else {
        Some(base)
    }
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

/// What `load_conversations` reads besides the cards.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Contents {
    /// Only what the cards hold, as the list shows them.
    Cards,
    /// The cards with their full text and semantic data, as the parser
    /// produces them.
    Full,
}

/// Every conversation, newest first, from the store: a reconcile pass first
/// brings the current generation up to date with the session files (or
/// waits for the process doing so), then its cards are read with
/// `contents`. Without a cache dir, or when the store cannot be read, the
/// session files are parsed instead.
pub fn load_conversations(
    show_last: bool,
    contents: Contents,
) -> crate::error::Result<Vec<Conversation>> {
    load_conversations_where(show_last, contents, &|_| true)
}

/// The conversations of `load_conversations` whose card `keep` accepts.
/// Contents beyond the card are read only for those.
pub fn load_conversations_where(
    show_last: bool,
    contents: Contents,
    keep: &(dyn Fn(&Conversation) -> bool + Sync),
) -> crate::error::Result<Vec<Conversation>> {
    if let Some(conversations) =
        default_root().and_then(|root| load_from_store(&root, show_last, contents, keep))
    {
        return Ok(conversations);
    }
    let mut conversations = load_in_memory(show_last)?;
    conversations.retain(|conversation| keep(conversation));
    Ok(conversations)
}

/// Every conversation the environment's instances hold, parsed in memory
/// and newest first: the corpus of a run without a store. Fails when no
/// instance has a history to read.
pub fn load_in_memory(show_last: bool) -> crate::error::Result<Vec<Conversation>> {
    let sources = crate::history::sources::Registry::from_env();
    if !sources
        .adapters()
        .iter()
        .any(|adapter| adapter.instance().root.exists())
    {
        let root = crate::history::get_claude_projects_root()?;
        return Err(crate::error::AppError::ProjectsDirNotFound(
            root.display().to_string(),
        ));
    }
    let mut conversations = reconcile::parse_everything(&sources);
    conversations.sort_by(|left, right| right.timestamp.cmp(&left.timestamp));
    for (index, conversation) in conversations.iter_mut().enumerate() {
        conversation.index = index;
        conversation.preview = if show_last {
            conversation.preview_last.clone()
        } else {
            conversation.preview_first.clone()
        };
    }
    Ok(conversations)
}

fn load_from_store(
    root: &Path,
    show_last: bool,
    contents: Contents,
    keep: &(dyn Fn(&Conversation) -> bool + Sync),
) -> Option<Vec<Conversation>> {
    let sources = crate::history::sources::Registry::from_env();
    for _ in 0..2 {
        let outcome = reconcile::reconcile(root, &sources).ok()?;
        if outcome.publish != generation::Publish::Busy {
            break;
        }
        generation::PublishLock::wait(root).ok()?;
    }
    let cards = Snapshot::open(root, show_last, TimeFilter::default())?;
    read_conversations(&cards, contents, keep)
}

/// Every card of `cards` that `keep` accepts, with `contents`, in doc order.
fn read_conversations(
    cards: &Snapshot,
    contents: Contents,
    keep: &(dyn Fn(&Conversation) -> bool + Sync),
) -> Option<Vec<Conversation>> {
    let generation = &cards.generation;
    let files = match contents {
        Contents::Cards => None,
        Contents::Full => Some((
            generation.open_text_files()?,
            generation.open_semantic_files()?,
        )),
    };
    // A card whose contents cannot be read fails the whole load.
    let read: Option<Vec<Option<Conversation>>> = (0..generation.rows().len() as u32)
        .into_par_iter()
        .map(|doc| {
            let mut conversation = cards.card(doc);
            if !keep(&conversation) {
                return Some(None);
            }
            if let Some((text, semantic)) = &files {
                let row = &generation.rows()[doc as usize];
                conversation.full_text = text.read(row.text)?.into();
                let sem = &generation.sems()[doc as usize];
                if sem.hash != 0 {
                    semantic::SemanticDoc::decode(&semantic.read_bytes(sem.blob)?)?
                        .apply(&mut conversation);
                }
            }
            Some(Some(conversation))
        })
        .collect();
    Some(read?.into_iter().flatten().collect())
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

    /// Ids of the newest `n` listed cards inside the time filter whose
    /// project passes `project_visible`.
    pub fn head(&self, n: usize, project_visible: impl FnMut(&ProjectView) -> bool) -> Vec<u32> {
        let visible = self.project_visibility(project_visible);
        self.generation.head(n, |row| {
            listed(row) && visible[row.project_id as usize] && in_time(&self.time, row)
        })
    }

    /// For every card, whether it is listed, inside the time filter and its
    /// project passes `project_visible`.
    pub fn scope(&self, project_visible: impl FnMut(&ProjectView) -> bool) -> Vec<bool> {
        let visible = self.project_visibility(project_visible);
        self.generation
            .rows()
            .iter()
            .map(|row| listed(row) && visible[row.project_id as usize] && in_time(&self.time, row))
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

/// Whether the list shows the card: automated and child conversations stay
/// out of it.
fn listed(row: &card::CardRow) -> bool {
    row.flags & card::card_flags::HIDDEN == 0
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

/// Every card of a generation, in doc order, as a conversation carrying its
/// semantic data, for semantic search.
pub struct SemanticCards {
    generation: String,
    conversations: Arc<Vec<Arc<Conversation>>>,
    /// `(CardRow::key_hash, SemRow::hash)` of each card.
    hashes: Vec<(u64, u64)>,
}

impl SemanticCards {
    /// Reads the semantic data of every card of `cards`. Data a card shares
    /// with the same card in `previous` is reused rather than read, so equal
    /// turns stay the same allocation. `None` when the generation's semantic
    /// segments are missing or invalid.
    pub fn load(cards: &Snapshot, previous: Option<&SemanticCards>) -> Option<Self> {
        let generation = &cards.generation;
        let segments = generation.open_semantic_files()?;
        let reusable: HashMap<u64, (u64, &Arc<Conversation>)> = previous
            .map(|previous| {
                previous
                    .hashes
                    .iter()
                    .zip(previous.conversations.iter())
                    .map(|(&(key, sem), conversation)| (key, (sem, conversation)))
                    .collect()
            })
            .unwrap_or_default();
        let hashes: Vec<(u64, u64)> = generation
            .rows()
            .iter()
            .zip(generation.sems())
            .map(|(row, sem)| (row.key_hash, sem.hash))
            .collect();
        let conversations = (0..hashes.len())
            .into_par_iter()
            .map(|doc| {
                let mut card = cards.card(doc as u32);
                let (key, hash) = hashes[doc];
                match reusable.get(&key) {
                    Some(&(previous_hash, previous)) if previous_hash == hash => {
                        card.semantic_turns = previous.semantic_turns.clone();
                        card.semantic_turn_ranges = previous.semantic_turn_ranges.clone();
                        card.semantic_route_text = previous.semantic_route_text.clone();
                        card.agent_search_text = previous.agent_search_text.clone();
                    }
                    _ if hash == 0 => {}
                    _ => {
                        let bytes = segments.read_bytes(generation.sems()[doc].blob)?;
                        semantic::SemanticDoc::decode(&bytes)?.apply(&mut card);
                    }
                }
                Some(Arc::new(card))
            })
            .collect::<Option<Vec<_>>>()?;
        Some(Self {
            generation: generation.name().to_owned(),
            conversations: Arc::new(conversations),
            hashes,
        })
    }

    /// Cards named as generation `generation`, without one on disk.
    #[cfg(test)]
    pub fn from_conversations(generation: &str, conversations: Vec<Conversation>) -> Self {
        Self {
            generation: generation.to_owned(),
            hashes: vec![(0, 0); conversations.len()],
            conversations: Arc::new(conversations.into_iter().map(Arc::new).collect()),
        }
    }

    pub fn generation_name(&self) -> &str {
        &self.generation
    }

    /// Conversation of each card; the index of card `doc` is `doc`.
    pub fn conversations(&self) -> &Arc<Vec<Arc<Conversation>>> {
        &self.conversations
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
    use super::{SemanticCards, Snapshot, generation, test_support};
    use crate::time_filter::TimeFilter;
    use std::path::Path;
    use std::sync::Arc;

    #[test]
    fn semantic_cards_carry_semantic_data_and_reuse_unchanged_turns() {
        let dir = tempfile::tempdir().unwrap();
        let mut conversations: Vec<_> = (0..3)
            .map(|i| test_support::conversation(&format!("/p/-a/s{i}.jsonl"), 1_000 - i))
            .collect();
        conversations[0].semantic_turns = vec!["zero".to_owned()].into();
        conversations[0].semantic_turn_ranges = vec![crate::agent::refs::MessageRange::single(1)];
        conversations[1].semantic_turns = vec!["one".to_owned()].into();
        conversations[1].semantic_route_text = "route".into();
        conversations[1].agent_search_text = "agent".into();
        generation::publish(dir.path(), &conversations).unwrap();
        let open = || Snapshot::open(dir.path(), false, TimeFilter::default()).unwrap();

        let first = SemanticCards::load(&open(), None).unwrap();
        let cards = first.conversations();
        assert_eq!(cards.len(), 3);
        for (doc, card) in cards.iter().enumerate() {
            assert_eq!(card.index, doc);
            assert_eq!(card.path, conversations[doc].path);
            assert_eq!(card.semantic_turns, conversations[doc].semantic_turns);
            assert_eq!(
                card.semantic_turn_ranges,
                conversations[doc].semantic_turn_ranges
            );
            assert_eq!(
                card.semantic_route_text,
                conversations[doc].semantic_route_text
            );
            assert_eq!(card.agent_search_text, conversations[doc].agent_search_text);
        }

        conversations[0].semantic_turns = vec!["changed".to_owned()].into();
        conversations[1].custom_title = Some("renamed".to_owned());
        generation::publish(dir.path(), &conversations).unwrap();
        let next = SemanticCards::load(&open(), Some(&first)).unwrap();
        assert_ne!(next.generation_name(), first.generation_name());
        let cards = next.conversations();
        assert_eq!(&*cards[0].semantic_turns, ["changed".to_owned()]);
        assert!(Arc::ptr_eq(
            &cards[1].semantic_turns,
            &first.conversations()[1].semantic_turns
        ));
        assert_eq!(cards[1].custom_title.as_deref(), Some("renamed"));
    }

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
    use crate::history::Conversation;
    use crate::history::model::SourceKind;
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
            source: SourceKind::Claude,
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
            provenance: Default::default(),
        }
    }
}
