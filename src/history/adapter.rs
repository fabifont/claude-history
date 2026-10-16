//! Source adapters: one per configured history root.
//!
//! An adapter owns everything source-specific about its instance: listing
//! its conversations with one `stat` per artifact (`discover`), picking the
//! history a conversation's key selects (`resolve`), and decoding that
//! history into a `Transcript`. Everything downstream works on transcripts.
//! It also plans the actions the source's own tool offers (`actions`).
//!
//! Discovery feeds reconcile's fingerprint diff, so it must be cheap and must
//! say when it is incomplete: a conversation that a partial or degraded
//! listing did not report may still exist, and its stored state is kept.

use super::actions::{Capabilities, Resume, Target};
use super::model::{ConversationKey, Event, SourceInstance, Transcript};
use std::any::Any;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Modification times this close to the start of a scan are not trusted:
/// the file may still be changing, so its stamp is unknown.
pub const MTIME_SLACK: Duration = Duration::from_secs(2);

/// What discovery saw of a conversation's artifacts. Equal stamps mean an
/// unchanged conversation, except that an unknown modification time (0)
/// never counts as unchanged.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Stamp {
    pub size: u64,
    /// Modification time in epoch nanoseconds, or 0 when unknown.
    pub mtime_ns: i64,
}

impl Stamp {
    /// Stamp of a file with `size` and `modified` time, listed by a scan
    /// that started at `scan_started`.
    pub fn of(size: u64, modified: Option<SystemTime>, scan_started: SystemTime) -> Self {
        let threshold = scan_started.checked_sub(MTIME_SLACK).unwrap_or(UNIX_EPOCH);
        let mtime_ns = modified
            .filter(|modified| *modified < threshold)
            .and_then(|modified| modified.duration_since(UNIX_EPOCH).ok())
            .map_or(0, |since_epoch| {
                i64::try_from(since_epoch.as_nanos()).unwrap_or(0)
            });
        Self { size, mtime_ns }
    }

    pub fn is_known(&self) -> bool {
        self.mtime_ns != 0
    }
}

/// Device and inode of a file.
pub type FileId = (u64, u64);

/// One conversation found by discovery, resolved to its selected history.
#[derive(Clone, Debug)]
pub struct Located {
    pub key: ConversationKey,
    /// The conversation's own artifact.
    pub path: PathBuf,
    /// Size of `path`.
    pub size: u64,
    /// Modification time of `path`.
    pub modified: Option<SystemTime>,
    /// Identity of the file behind `path`, so a file reached through two
    /// roots is found once.
    pub file_id: Option<FileId>,
    /// State of every artifact the history is decoded from.
    pub stamp: Stamp,
    /// What the adapter resolved about the history beyond `path`, for its
    /// own `decode_into`.
    pub native: Option<Native>,
}

/// An adapter's own resolution of a conversation, opaque to everything else.
#[derive(Clone)]
pub struct Native(pub Arc<dyn Any + Send + Sync>);

impl std::fmt::Debug for Native {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Native")
    }
}

/// How much of an instance a discovery pass saw.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub enum Completeness {
    #[default]
    Complete,
    /// These directories could not be listed.
    Partial(HashSet<PathBuf>),
    /// The instance's root could not be listed.
    Degraded,
}

impl Completeness {
    /// Whether the pass may have missed a conversation whose own artifact is
    /// `path`.
    pub fn may_miss(&self, path: &Path) -> bool {
        match self {
            Self::Complete => false,
            Self::Degraded => true,
            Self::Partial(dirs) => path.parent().is_some_and(|dir| dirs.contains(dir)),
        }
    }
}

/// Every conversation of an instance that one pass found.
#[derive(Clone, Debug, Default)]
pub struct Discovery {
    pub found: Vec<Located>,
    pub completeness: Completeness,
}

impl Discovery {
    /// Records that `dir` could not be listed.
    pub fn missed_dir(&mut self, dir: PathBuf) {
        match &mut self.completeness {
            Completeness::Degraded => {}
            Completeness::Partial(dirs) => {
                dirs.insert(dir);
            }
            Completeness::Complete => self.completeness = Completeness::Partial([dir].into()),
        }
    }
}

pub trait Adapter: Send + Sync {
    fn instance(&self) -> &SourceInstance;

    /// Native id of the conversation whose own artifact is `path`, which
    /// lies under the instance root: by default the path relative to it.
    fn native_id_of(&self, path: &Path) -> Option<String> {
        Some(
            path.strip_prefix(&self.instance().root)
                .unwrap_or(path)
                .to_string_lossy()
                .into_owned(),
        )
    }

    /// Lists every conversation of the instance, with one `stat` per
    /// artifact. `scan_started` dates the pass for `Stamp::of`.
    fn discover(&self, scan_started: SystemTime) -> Discovery;

    /// Locates the history the conversation `native_id` selects, as
    /// `discover` would list it.
    fn resolve(&self, native_id: &str, scan_started: SystemTime) -> crate::error::Result<Located>;

    /// Decodes the history `located` selects, passing each event to
    /// `events` in order as it is decoded; the transcript returned holds
    /// no events. `None` when the artifacts hold no conversation.
    fn decode_into(
        &self,
        located: &Located,
        events: &mut dyn FnMut(Event),
    ) -> crate::error::Result<Option<Transcript>>;

    /// Decodes the history `located` selects; `None` when its artifacts
    /// hold no conversation.
    fn decode(&self, located: &Located) -> crate::error::Result<Option<Transcript>> {
        let mut decoded = Vec::new();
        let transcript = self.decode_into(located, &mut |event| decoded.push(event))?;
        Ok(transcript.map(|mut transcript| {
            transcript.events = decoded;
            transcript
        }))
    }

    fn capabilities(&self) -> Capabilities;

    /// Plans resuming `target` in the tool that wrote it, or forking it
    /// into a new conversation. `default_args` are the user's extra
    /// arguments for tools that take them.
    fn resume(
        &self,
        target: Target<'_>,
        fork: bool,
        default_args: &[String],
    ) -> crate::error::Result<Resume>;

    /// Deletes the conversation whose own artifact is `path`.
    fn delete(&self, path: &Path) -> crate::error::Result<()>;

    /// Sets the title of the conversation whose own artifact is `path`.
    fn rename(&self, path: &Path, title: &str) -> crate::error::Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stamps_of_files_modified_near_the_scan_are_unknown() {
        let now = SystemTime::now();
        assert!(!Stamp::of(1, Some(now - Duration::from_secs(1)), now).is_known());
        assert!(!Stamp::of(1, None, now).is_known());
        let settled = Stamp::of(1, Some(now - Duration::from_secs(60)), now);
        assert!(settled.is_known());
        assert_eq!(settled.size, 1);
    }

    #[test]
    fn only_unlisted_directories_may_hide_conversations() {
        let mut discovery = Discovery::default();
        assert!(!discovery.completeness.may_miss(Path::new("/r/a/s.jsonl")));
        discovery.missed_dir(PathBuf::from("/r/a"));
        assert!(discovery.completeness.may_miss(Path::new("/r/a/s.jsonl")));
        assert!(!discovery.completeness.may_miss(Path::new("/r/b/s.jsonl")));
        assert!(Completeness::Degraded.may_miss(Path::new("/r/b/s.jsonl")));
    }
}
