//! What the daemon reports about itself, in a JSON file it replaces
//! atomically whenever its state changes. `daemon status` reads it, and a
//! launch reads it to see which binary the running daemon is.

use super::identity::BinaryIdentity;
use serde::{Deserialize, Serialize};
use std::io;
use std::path::Path;

/// Times are epoch milliseconds.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Status {
    pub pid: u32,
    pub binary: BinaryIdentity,
    pub started_ms: i64,
    /// What the daemon is doing: `idle`, `scanning`, `embedding`,
    /// `paused on battery`, or `exited` with `exit_reason`.
    pub state: String,
    pub exit_reason: Option<String>,
    pub last_quick_pass_ms: Option<i64>,
    pub last_full_pass_ms: Option<i64>,
    /// Last time a pass published a new generation of the keyword store.
    pub last_publish_ms: Option<i64>,
    /// Last time embeddings were written to the semantic cache.
    pub last_embed_ms: Option<i64>,
    /// Passages the last full check found without an embedding.
    pub backlog: Option<u64>,
    /// Conversations that changed and are embedded once they go quiet.
    pub pending_conversations: usize,
    /// Passages embedded since the daemon started.
    pub embedded: u64,
    /// CPU time the daemon and its finished pass and embedding processes
    /// have used, in milliseconds.
    #[serde(default)]
    pub cpu_ms: u64,
    pub last_error: Option<StatusError>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct StatusError {
    pub at_ms: i64,
    pub message: String,
}

impl Status {
    pub fn new(pid: u32, binary: BinaryIdentity, now_ms: i64) -> Self {
        Self {
            pid,
            binary,
            started_ms: now_ms,
            state: "starting".to_owned(),
            exit_reason: None,
            last_quick_pass_ms: None,
            last_full_pass_ms: None,
            last_publish_ms: None,
            last_embed_ms: None,
            backlog: None,
            pending_conversations: 0,
            embedded: 0,
            cpu_ms: 0,
            last_error: None,
        }
    }

    pub fn read(path: &Path) -> Option<Self> {
        serde_json::from_slice(&std::fs::read(path).ok()?).ok()
    }

    /// Replaces the file at `path` with this status.
    pub fn write(&self, path: &Path) -> io::Result<()> {
        let dir = path
            .parent()
            .ok_or_else(|| io::Error::other("status path has no parent"))?;
        std::fs::create_dir_all(dir)?;
        let mut temp = tempfile::NamedTempFile::new_in(dir)?;
        serde_json::to_writer_pretty(&mut temp, self).map_err(io::Error::other)?;
        temp.persist(path).map_err(|error| error.error)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_round_trips_through_its_file() {
        let dir = tempfile::tempdir().unwrap();
        let binary = dir.path().join("bin");
        std::fs::write(&binary, "x").unwrap();
        let path = dir.path().join("daemon").join("status.json");
        let mut status = Status::new(7, BinaryIdentity::of(&binary).unwrap(), 1_000);
        status.backlog = Some(12);
        status.last_error = Some(StatusError {
            at_ms: 2_000,
            message: "embedding failed".into(),
        });
        status.write(&path).unwrap();
        assert_eq!(Status::read(&path), Some(status.clone()));
        status.state = "idle".into();
        status.write(&path).unwrap();
        assert_eq!(Status::read(&path).unwrap().state, "idle");
        assert_eq!(
            std::fs::read_dir(path.parent().unwrap()).unwrap().count(),
            1
        );
        std::fs::write(&path, "{").unwrap();
        assert_eq!(Status::read(&path), None);
    }
}
