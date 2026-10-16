use crate::config::KeyBindings;
use crate::history::{Conversation, format_short_name_from_path, process_conversation_file};
use crate::search;
#[cfg(test)]
use crate::semantic::types::{SemanticExplanation, SemanticScoreBreakdown};
#[cfg(test)]
use crate::tui::semantic_worker::{SemanticSearchMessage, SemanticWorkerCommand};
#[cfg(test)]
use crate::tui::ui;
use crate::tui::viewer::ToolDisplayMode;
#[cfg(test)]
use crate::tui::viewer::ToolOutputId;
#[cfg(test)]
use chrono::Local;
use crossterm::event::{KeyCode, KeyModifiers};
use ratatui::prelude::*;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc;
use std::time::Duration;

mod dialog_state;
mod input_controller;
mod list_state;
mod search_state;
mod types;
mod view_state;
#[allow(unused_imports)]
use search_state::{
    IndexScope, SearchCommand, SearchResponse, SemanticSearchState, spawn_search_worker,
};
#[allow(unused_imports)]
pub use types::{
    Action, AppMode, DialogMode, LIST_LINES_PER_ITEM, ListSearchMode, Readiness, ReconcileState,
    SemanticProgress, SemanticResultMetadata, TuiSearchOptions, ViewSearchMode, ViewState,
    list_lines_per_item,
};

/// Snapshot cards materialized for the list while loading. Bounded so the
/// first frame costs the same whatever the corpus size.
const SNAPSHOT_ROWS: usize = 500;

/// App state
pub struct App {
    /// All loaded conversations, shared with the search worker
    conversations: Arc<Vec<Conversation>>,
    /// Shared semantic conversation snapshot for background workers
    semantic_conversations_snapshot: Arc<Vec<Arc<Conversation>>>,
    /// Version of the semantic corpus snapshot
    semantic_corpus_version: u64,
    /// Version of the semantic scope snapshot
    semantic_scope_version: u64,
    /// Last corpus version sent to the semantic worker
    semantic_sent_corpus_version: u64,
    /// Last scope signature sent to the semantic worker
    semantic_sent_scope_signature: Option<(u64, Arc<Vec<usize>>)>,
    /// Indices into conversations, sorted by current score
    filtered: Vec<usize>,
    /// Currently selected index into filtered (None if no results)
    selected: Option<usize>,
    /// Session path to restore after replacing and re-filtering the corpus
    selection_anchor: Option<PathBuf>,
    /// Current search query
    query: String,
    /// Cursor position in query (character index, not byte)
    cursor_pos: usize,
    /// Which parts of the list are usable
    readiness: Readiness,
    /// Snapshot whose cards are shown while loading
    snapshot: Option<crate::store::Snapshot>,
    /// Generation index lexical queries run against, when there is one
    search_index: Option<Arc<crate::store::SearchIndex>>,
    /// For each card of the snapshot or index, its index in `conversations`
    doc_rows: Vec<Option<usize>>,
    /// Loaded conversations held back while snapshot cards are shown
    pending: Option<Vec<Conversation>>,
    /// Current dialog overlay (confirm, menu)
    dialog_mode: DialogMode,
    /// Main app mode (list or view)
    app_mode: AppMode,
    /// Status message with timestamp for auto-clear
    status_message: Option<(String, std::time::Instant)>,
    /// Persistent view setting: tool display mode
    tool_display: ToolDisplayMode,
    /// Persistent view setting: whether to show thinking blocks
    show_thinking: bool,
    /// Persistent view setting: whether to show timing information
    show_timing: bool,
    /// Whether the app is running in single file mode (direct input, no list)
    single_file_mode: bool,
    /// Configurable keybindings
    keys: KeyBindings,
    /// Whether workspace filter is active (only show current project's conversations)
    workspace_filter: bool,
    /// The encoded project directory name for the current workspace (for filtering)
    current_project_dir_name: Option<String>,
    /// Exact project names hidden from list-mode display
    excluded_projects: HashSet<String>,
    /// Channel to send commands to the background search worker
    search_tx: mpsc::Sender<SearchCommand>,
    /// Channel to receive results from the background search worker
    search_rx: mpsc::Receiver<SearchResponse>,
    /// Monotonic generation counter for search requests
    search_generation: u64,
    /// Whether a search is currently in-flight on the worker thread
    search_in_flight: bool,
    /// Whether `filtered` is the first page of an index search still being
    /// completed
    search_page_shown: bool,
    /// Current list search mode
    list_search_mode: ListSearchMode,
    /// Semantic TUI state
    semantic_search: SemanticSearchState,
    /// Cached lexical evidence produced outside the render path
    lexical_evidence: HashMap<usize, search::LexicalEvidence>,
}

struct AppParts {
    conversations: Arc<Vec<Conversation>>,
    semantic_conversations_snapshot: Arc<Vec<Arc<Conversation>>>,
    filtered: Vec<usize>,
    selected: Option<usize>,
    readiness: Readiness,
    app_mode: AppMode,
    tool_display: ToolDisplayMode,
    show_thinking: bool,
    single_file_mode: bool,
    keys: KeyBindings,
    workspace_filter: bool,
    current_project_dir_name: Option<String>,
    excluded_projects: HashSet<String>,
    search_tx: mpsc::Sender<SearchCommand>,
    search_rx: mpsc::Receiver<SearchResponse>,
    list_search_mode: ListSearchMode,
    semantic_search: SemanticSearchState,
}

impl App {
    fn from_parts(parts: AppParts) -> Self {
        Self {
            semantic_conversations_snapshot: parts.semantic_conversations_snapshot,
            semantic_corpus_version: 1,
            semantic_scope_version: 0,
            semantic_sent_corpus_version: 0,
            semantic_sent_scope_signature: None,
            conversations: parts.conversations,
            filtered: parts.filtered,
            selected: parts.selected,
            selection_anchor: None,
            query: String::new(),
            cursor_pos: 0,
            readiness: parts.readiness,
            snapshot: None,
            search_index: None,
            doc_rows: Vec::new(),
            pending: None,
            dialog_mode: DialogMode::None,
            app_mode: parts.app_mode,
            status_message: None,
            tool_display: parts.tool_display,
            show_thinking: parts.show_thinking,
            show_timing: false,
            single_file_mode: parts.single_file_mode,
            keys: parts.keys,
            workspace_filter: parts.workspace_filter,
            current_project_dir_name: parts.current_project_dir_name,
            excluded_projects: parts.excluded_projects,
            search_tx: parts.search_tx,
            search_rx: parts.search_rx,
            search_generation: 0,
            search_in_flight: false,
            search_page_shown: false,
            list_search_mode: parts.list_search_mode,
            semantic_search: parts.semantic_search,
            lexical_evidence: HashMap::new(),
        }
    }

    fn semantic_snapshot(conversations: &[Conversation]) -> Arc<Vec<Arc<Conversation>>> {
        Arc::new(conversations.iter().cloned().map(Arc::new).collect())
    }

    fn list_search_mode_from_options(search_options: TuiSearchOptions) -> ListSearchMode {
        search_options.default_mode
    }

    fn semantic_search_state(available: bool) -> SemanticSearchState {
        SemanticSearchState {
            available,
            ..Default::default()
        }
    }

    /// Create a new app with all conversations pre-loaded
    #[allow(dead_code)]
    pub fn new(
        conversations: Vec<Conversation>,
        tool_display: ToolDisplayMode,
        show_thinking: bool,
        keys: KeyBindings,
        exclude_projects: Vec<String>,
    ) -> Self {
        Self::new_with_options(
            conversations,
            tool_display,
            show_thinking,
            keys,
            exclude_projects,
            TuiSearchOptions::default(),
        )
    }

    #[allow(dead_code)]
    pub fn new_with_options(
        conversations: Vec<Conversation>,
        tool_display: ToolDisplayMode,
        show_thinking: bool,
        keys: KeyBindings,
        exclude_projects: Vec<String>,
        search_options: TuiSearchOptions,
    ) -> Self {
        let excluded_projects = exclude_projects.into_iter().collect();
        let filtered = list_state::filter_conversation_indices(
            0..conversations.len(),
            &conversations,
            &excluded_projects,
            false,
            None,
        );
        let selected = if filtered.is_empty() { None } else { Some(0) };
        let (search_tx, search_rx) = spawn_search_worker();
        let semantic_conversations_snapshot = Self::semantic_snapshot(&conversations);
        let conversations = Arc::new(conversations);
        let _ = search_tx.send(SearchCommand::UpdateData {
            conversations: conversations.clone(),
        });

        Self::from_parts(AppParts {
            semantic_conversations_snapshot,
            conversations,
            filtered,
            selected,
            readiness: Readiness::ready(),
            app_mode: AppMode::List,
            tool_display,
            show_thinking,
            single_file_mode: false,
            keys,
            workspace_filter: false,
            current_project_dir_name: None,
            excluded_projects,
            search_tx,
            search_rx,
            list_search_mode: Self::list_search_mode_from_options(search_options),
            semantic_search: Self::semantic_search_state(true),
        })
    }

    /// Create a new app in loading state
    pub fn new_loading_with_options(
        tool_display: ToolDisplayMode,
        show_thinking: bool,
        keys: KeyBindings,
        workspace_filter: bool,
        current_project_dir_name: Option<String>,
        exclude_projects: Vec<String>,
        search_options: TuiSearchOptions,
    ) -> Self {
        let (search_tx, search_rx) = spawn_search_worker();

        Self::from_parts(AppParts {
            semantic_conversations_snapshot: Arc::default(),
            conversations: Arc::default(),
            filtered: Vec::new(),
            selected: None,
            readiness: Readiness::default(),
            app_mode: AppMode::List,
            tool_display,
            show_thinking,
            single_file_mode: false,
            keys,
            workspace_filter,
            current_project_dir_name,
            excluded_projects: exclude_projects.into_iter().collect(),
            search_tx,
            search_rx,
            list_search_mode: Self::list_search_mode_from_options(search_options),
            semantic_search: Self::semantic_search_state(true),
        })
    }

    /// Create a new app for viewing a single file directly
    pub fn new_single_file(
        path: PathBuf,
        tool_display: ToolDisplayMode,
        show_thinking: bool,
        keys: KeyBindings,
    ) -> Self {
        let (search_tx, search_rx) = spawn_search_worker();

        // Parse using the same parser as the main list
        let modified = std::fs::metadata(&path).and_then(|m| m.modified()).ok();

        let mut conversations = Vec::new();
        let mut filtered = Vec::new();
        let mut selected = None;

        if let Ok(Some(mut conv)) = process_conversation_file(path.clone(), modified, None) {
            // Set project_name the same way as the loader does
            let project_path = conv.cwd.clone().unwrap_or_else(|| path.clone());
            conv.project_name = Some(format_short_name_from_path(&project_path));

            conversations.push(conv);
            filtered.push(0);
            selected = Some(0);
        }

        Self::from_parts(AppParts {
            semantic_conversations_snapshot: Self::semantic_snapshot(&conversations),
            conversations: Arc::new(conversations),
            filtered,
            selected,
            readiness: Readiness::ready(),
            app_mode: AppMode::View(ViewState::initial(
                path,
                tool_display,
                show_thinking,
                false,
                0,
            )),
            tool_display,
            show_thinking,
            single_file_mode: true,
            keys,
            workspace_filter: false,
            current_project_dir_name: None,
            excluded_projects: HashSet::new(),
            search_tx,
            search_rx,
            list_search_mode: ListSearchMode::Lexical,
            semantic_search: Self::semantic_search_state(false),
        })
    }

    pub fn keys(&self) -> &KeyBindings {
        &self.keys
    }

    /// Shows the newest cards of `snapshot` until loading finishes. Loaded
    /// batches are held back meanwhile and swapped in by `finish_loading`.
    pub fn show_snapshot(&mut self, snapshot: crate::store::Snapshot) {
        self.doc_rows = vec![None; snapshot.len()];
        self.conversations = Arc::default();
        self.snapshot = Some(snapshot);
        self.pending = Some(Vec::new());
        self.show_snapshot_cards();
    }

    /// Re-reads the snapshot cards for the current scope.
    fn show_snapshot_cards(&mut self) {
        let Some(snapshot) = &self.snapshot else {
            return;
        };
        let workspace = self.workspace_scope();
        let docs = snapshot.head(SNAPSHOT_ROWS, |project| {
            list_state::card_project_visible(project, &self.excluded_projects, &workspace)
        });
        let filtered = docs.into_iter().map(|doc| self.card_row(doc)).collect();
        self.apply_filtered(filtered);
        self.readiness.list = true;
    }

    fn workspace_scope(&self) -> list_state::WorkspaceScope<'_> {
        list_state::WorkspaceScope::new(
            self.current_project_dir_name
                .as_deref()
                .filter(|_| self.workspace_filter),
        )
    }

    /// Index in `conversations` of snapshot card `doc`, materializing it on
    /// first use.
    fn card_row(&mut self, doc: u32) -> usize {
        if let Some(row) = self.doc_rows[doc as usize] {
            return row;
        }
        let snapshot = self.snapshot.as_ref().expect("cards come from a snapshot");
        let mut card = snapshot.card(doc);
        let row = self.conversations.len();
        card.index = row;
        Arc::make_mut(&mut self.conversations).push(card);
        self.doc_rows[doc as usize] = Some(row);
        row
    }

    /// Opens the index of the snapshot being shown, so lexical queries run
    /// before loading finishes. Returns whether one was opened.
    pub fn open_search_index(&mut self) -> bool {
        let Some(snapshot) = &self.snapshot else {
            return false;
        };
        let Some(index) = crate::store::SearchIndex::open(snapshot.clone()) else {
            return false;
        };
        self.search_index = Some(Arc::new(index));
        self.send_index_scope();
        self.readiness.lexical = true;
        if !self.query.trim().is_empty() {
            self.dispatch_search();
        }
        true
    }

    /// Switches to the index of `snapshot` if it is a newer generation than
    /// the one in use, and reruns the query on it keeping the selection.
    /// Only applies once loading has finished; until then the snapshot shown
    /// stays in use.
    pub fn swap_search_index(&mut self, snapshot: crate::store::Snapshot) -> bool {
        if self.pending.is_some()
            || self
                .search_index
                .as_ref()
                .is_some_and(|index| index.cards().generation_name() == snapshot.generation_name())
        {
            return false;
        }
        let Some(index) = crate::store::SearchIndex::open(snapshot) else {
            return false;
        };
        self.search_index = Some(Arc::new(index));
        self.locate_docs();
        self.send_index_scope();
        self.readiness.lexical = true;
        if !self.query.trim().is_empty() {
            self.selection_anchor = self.get_selected_path();
            self.dispatch_search();
        }
        true
    }

    /// Maps the index's cards to the loaded conversations.
    fn locate_docs(&mut self) {
        self.doc_rows = match &self.search_index {
            Some(index) => index.cards().locate(&self.conversations),
            None => Vec::new(),
        };
    }

    /// Sends the index and the cards in scope to the search worker.
    pub(super) fn send_index_scope(&mut self) {
        let Some(index) = &self.search_index else {
            return;
        };
        let workspace = self.workspace_scope();
        let mut docs = index.cards().scope(|project| {
            list_state::card_project_visible(project, &self.excluded_projects, &workspace)
        });
        if self.pending.is_none() {
            for (doc, row) in docs.iter_mut().zip(&self.doc_rows) {
                *doc &= row.is_some();
            }
        }
        let _ = self
            .search_tx
            .send(SearchCommand::UseIndex(Some(IndexScope {
                index: index.clone(),
                docs: Arc::new(docs),
            })));
    }

    /// Conversation rows of index cards `docs`, in order. Cards without a
    /// loaded conversation are dropped.
    pub(super) fn rows_of_docs(&mut self, docs: &[usize]) -> Vec<usize> {
        if self.pending.is_some() {
            docs.iter().map(|&doc| self.card_row(doc as u32)).collect()
        } else {
            docs.iter()
                .filter_map(|&doc| self.doc_rows.get(doc).copied().flatten())
                .collect()
        }
    }

    /// Stops searching the index; queries then run over the loaded
    /// conversations, which must be loaded.
    pub(super) fn drop_search_index(&mut self) {
        if self.search_index.take().is_some() {
            self.doc_rows.clear();
            let _ = self.search_tx.send(SearchCommand::UseIndex(None));
        }
    }

    /// Append a batch of conversations during loading
    /// Note: Does NOT precompute search text - that's deferred to finish_loading
    pub fn append_conversations(&mut self, new_convs: Vec<Conversation>) {
        if let Some(pending) = &mut self.pending {
            pending.extend(new_convs);
            self.readiness.reconcile = ReconcileState::Running {
                loaded: pending.len(),
            };
            return;
        }
        let start_idx = self.conversations.len();
        Arc::make_mut(&mut self.conversations).extend(new_convs);
        let end_idx = self.conversations.len();

        let new_filtered = self.filter_indices(start_idx..end_idx);
        self.filtered.extend(new_filtered);

        // Select first item if nothing selected yet
        if self.selected.is_none() && !self.filtered.is_empty() {
            self.selected = Some(0);
        }

        self.readiness.reconcile = ReconcileState::Running {
            loaded: self.conversations.len(),
        };
    }

    /// Mark loading as complete: sort, share the corpus with the search
    /// workers, and transition to ready. Replaces any snapshot cards, keeping
    /// the selected session.
    pub fn finish_loading(&mut self) {
        if let Some(loaded) = self.pending.take() {
            self.snapshot = None;
            if self.selection_anchor.is_none() {
                self.selection_anchor = self.get_selected_path();
            }
            self.conversations = Arc::new(loaded);
            self.filtered.clear();
            self.selected = None;
        }
        let conversations = Arc::make_mut(&mut self.conversations);

        // Sort all conversations by timestamp (newest first)
        conversations.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));

        // Reindex after sorting
        for (idx, conv) in conversations.iter_mut().enumerate() {
            conv.index = idx;
        }

        self.refresh_search_data();

        self.readiness = Readiness::ready();

        if self.query.trim().is_empty() {
            // Apply filter (handles exclusions and workspace filter)
            self.update_filter();
            self.prewarm_semantic_cache();
        } else {
            // Keep the list populated until the worker answers the query.
            let anchor = self.selection_anchor.clone();
            self.apply_filtered(self.filter_indices(0..self.conversations.len()));
            self.selection_anchor = anchor;
            self.dispatch_search();
        }
    }

    /// Replace the loaded corpus while preserving the current picker state.
    pub fn replace_conversations(&mut self, mut conversations: Vec<Conversation>) {
        let mut paths = HashSet::new();
        conversations.retain(|conversation| {
            let path = conversation
                .path
                .canonicalize()
                .unwrap_or_else(|_| conversation.path.clone());
            paths.insert(path)
        });
        conversations.sort_by(|left, right| right.timestamp.cmp(&left.timestamp));
        for (index, conversation) in conversations.iter_mut().enumerate() {
            conversation.index = index;
        }

        self.selection_anchor = self.get_selected_path();
        self.conversations = Arc::new(conversations);
        self.filtered.clear();
        self.selected = None;
        // The index holds the corpus before the refresh.
        self.drop_search_index();
        self.refresh_search_data();

        if self.query.trim().is_empty() {
            self.update_filter();
            self.prewarm_semantic_cache();
        } else {
            self.dispatch_search();
        }
    }

    pub fn set_status_message(&mut self, message: String) {
        self.status_message = Some((message, std::time::Instant::now()));
    }

    /// Consume the app and return its conversations
    pub fn into_conversations(self) -> Arc<Vec<Conversation>> {
        self.conversations
    }

    pub fn readiness(&self) -> &Readiness {
        &self.readiness
    }

    /// Whether the full corpus is still loading, so edits wait.
    pub fn is_loading(&self) -> bool {
        self.readiness.reconcile != ReconcileState::Done
    }

    pub(crate) fn shared_conversations(&self) -> Arc<Vec<Conversation>> {
        self.conversations.clone()
    }

    // Getters for UI access
    pub fn filtered(&self) -> &[usize] {
        &self.filtered
    }

    pub fn conversations(&self) -> &[Conversation] {
        &self.conversations
    }

    pub fn selected(&self) -> Option<usize> {
        self.selected
    }

    pub fn query(&self) -> &str {
        &self.query
    }

    pub fn dialog_mode(&self) -> &DialogMode {
        &self.dialog_mode
    }

    #[cfg(test)]
    pub fn set_dialog_mode_for_test(&mut self, dialog_mode: DialogMode) {
        self.dialog_mode = dialog_mode;
    }

    pub fn app_mode(&self) -> &AppMode {
        &self.app_mode
    }

    pub fn status_message(&self) -> Option<&(String, std::time::Instant)> {
        self.status_message.as_ref()
    }

    /// Returns how long until the active status message expires, if any
    pub fn status_message_remaining(&self) -> Option<Duration> {
        const STATUS_TTL: Duration = Duration::from_secs(3);
        self.status_message
            .as_ref()
            .and_then(|(_, instant)| STATUS_TTL.checked_sub(instant.elapsed()))
    }

    pub fn cursor_pos(&self) -> usize {
        self.cursor_pos
    }

    pub fn is_single_file_mode(&self) -> bool {
        self.single_file_mode
    }

    pub(crate) fn show_timing(&self) -> bool {
        self.show_timing
    }

    pub fn workspace_filter(&self) -> bool {
        self.workspace_filter
    }

    pub fn list_search_mode(&self) -> ListSearchMode {
        self.list_search_mode
    }

    pub fn has_project_context(&self) -> bool {
        self.current_project_dir_name.is_some()
    }

    #[cfg(test)]
    pub fn semantic_search_available(&self) -> bool {
        self.semantic_search.available
    }

    pub fn semantic_toggle_available(&self) -> bool {
        self.semantic_search.available
            && !self
                .keys
                .rename
                .matches(KeyCode::Char('t'), KeyModifiers::CONTROL)
            && !self
                .keys
                .delete
                .matches(KeyCode::Char('t'), KeyModifiers::CONTROL)
            && !self
                .keys
                .resume
                .matches(KeyCode::Char('t'), KeyModifiers::CONTROL)
            && !self
                .keys
                .fork
                .matches(KeyCode::Char('t'), KeyModifiers::CONTROL)
            && !self
                .keys
                .refresh
                .matches(KeyCode::Char('t'), KeyModifiers::CONTROL)
    }

    /// Handle a left-click in list mode: select the conversation under the cursor.
    /// Returns true if the click landed on a list item — the caller is expected to
    /// then transition into view mode (matching the Enter-key behavior).
    pub fn handle_list_click(&mut self, row: u16, frame_area: Rect) -> bool {
        if !matches!(self.app_mode, AppMode::List)
            || self.dialog_mode != DialogMode::None
            || !self.readiness.list
        {
            return false;
        }

        // Mirror the layout in render_list_mode: outer 1px border, then split
        // [search bar (2), list (Min 1), bottom bar (1)] — or omit the bottom
        // bar when the inner area is < 4 lines tall.
        let inner_height = frame_area.height.saturating_sub(2);
        let list_y = frame_area.y.saturating_add(1).saturating_add(2);
        let list_height = if inner_height < 4 {
            inner_height.saturating_sub(2)
        } else {
            inner_height.saturating_sub(3)
        };

        if list_height == 0 || row < list_y || row >= list_y.saturating_add(list_height) {
            return false;
        }

        let lines_per_item = list_lines_per_item(self.list_search_mode, &self.query);
        let items_per_page = (list_height as usize) / lines_per_item;
        if items_per_page == 0 {
            return false;
        }

        let offset = match self.selected {
            Some(sel) => (sel / items_per_page) * items_per_page,
            None => 0,
        };
        let relative_row = (row - list_y) as usize;
        let relative_idx = relative_row / lines_per_item;
        let new_idx = offset + relative_idx;
        if new_idx < self.filtered.len() {
            self.clear_selection_anchor();
            self.selected = Some(new_idx);
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod semantic_test_helpers;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod interaction_tests;

#[cfg(test)]
mod snapshot_tests;
