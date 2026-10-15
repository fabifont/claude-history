//! Disk snapshot of the conversation list, readable before any history is
//! loaded.
//!
//! The snapshot is a generation directory under `v3/` in the cache dir (see
//! `generation`). The TUI renders its first frame from the newest cards and
//! replaces them with fully loaded conversations once loading finishes.

pub mod card;
pub mod generation;

pub use generation::{Generation, ProjectView, Publish};

use crate::history::Conversation;
use crate::time_filter::TimeFilter;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// Store root: `v3/` beside the other caches, namespaced like them by
/// `CLAUDE_CONFIG_DIR`. Tests get no default root, so they never touch the
/// user's cache.
pub fn default_root() -> Option<PathBuf> {
    if cfg!(test) {
        return None;
    }
    crate::history::cache::cache_base_dir().map(|base| base.join("v3"))
}

/// Cards of the current generation, shown until the full corpus is loaded.
pub struct Snapshot {
    generation: Generation,
    show_last: bool,
    time: TimeFilter,
}

impl Snapshot {
    /// Opens the current generation under `root`; `None` when there is no
    /// usable one.
    pub fn open(root: &Path, show_last: bool, time: TimeFilter) -> Option<Self> {
        Some(Self {
            generation: Generation::open(root)?,
            show_last,
            time,
        })
    }

    pub fn generation_name(&self) -> &str {
        self.generation.name()
    }

    /// The newest `n` cards inside the time filter whose project passes
    /// `project_visible`, as list-only conversations indexed from 0.
    pub fn conversations(
        &self,
        n: usize,
        mut project_visible: impl FnMut(&ProjectView) -> bool,
    ) -> Vec<Conversation> {
        let generation = &self.generation;
        let visible: Vec<bool> = (0..generation.projects().len() as u32)
            .map(|id| project_visible(&generation.project(id)))
            .collect();
        let time = &self.time;
        generation
            .head(n, |row| {
                visible[row.project_id as usize]
                    && (!time.is_active()
                        || chrono::DateTime::from_timestamp_millis(row.timestamp_ms)
                            .is_some_and(|timestamp| time.matches(timestamp)))
            })
            .into_iter()
            .enumerate()
            .map(|(index, row)| {
                let mut conversation = generation.conversation(row, self.show_last);
                conversation.index = index;
                conversation
            })
            .collect()
    }
}

/// Writes a generation for `conversations` (the complete corpus, loaded
/// starting at `loaded_since`) unless `CURRENT` already matches it.
pub fn write_snapshot(
    root: &Path,
    conversations: &[Conversation],
    loaded_since: SystemTime,
) -> std::io::Result<Publish> {
    let fingerprints: Vec<_> = conversations
        .iter()
        .map(|conversation| generation::fingerprint(&conversation.path, loaded_since))
        .collect();
    let manifest = generation::build_manifest(conversations, &fingerprints);
    generation::publish(root, &manifest)
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::card::Fingerprint;
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
            search_text_lower: "full text".into(),
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

    pub(crate) fn unknown_fingerprints(count: usize) -> Vec<Fingerprint> {
        vec![Fingerprint::default(); count]
    }
}
