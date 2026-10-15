//! Disk snapshot of the conversation list, readable before any history is
//! loaded.
//!
//! The snapshot is a generation directory under `v3/roots-<hash>/` in the
//! cache dir (see `default_root` and `generation`). The TUI renders its first frame from the newest cards and
//! replaces them with fully loaded conversations once loading finishes.

pub mod card;
pub mod generation;
#[cfg_attr(not(test), allow(dead_code))]
pub mod lexindex;

pub use generation::{Generation, ProjectView, Publish};

use crate::history::Conversation;
use crate::time_filter::TimeFilter;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

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
