//! What the tool that wrote a conversation can do with it, as its adapter
//! plans it: resume or fork it in that tool, delete it, rename it.

use crate::error::{AppError, Result};
use crate::history::Provenance;
use crate::history::model::SourceKind;
use crate::history::sources::Registry;
use std::path::{Path, PathBuf};
use std::process::Command;

/// The actions an adapter supports for its conversations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Capabilities {
    pub resume: bool,
    pub fork: bool,
    pub delete: bool,
    pub rename: bool,
}

impl Capabilities {
    pub const ALL: Self = Self {
        resume: true,
        fork: true,
        delete: true,
        rename: true,
    };

    pub const NONE: Self = Self {
        resume: false,
        fork: false,
        delete: false,
        rename: false,
    };
}

/// The actions the adapter of `source` supports for a conversation of
/// `provenance`: none for an orphaned one, which its tool no longer has.
pub fn capabilities_of(source: SourceKind, provenance: Provenance) -> Capabilities {
    if provenance.orphaned {
        return Capabilities::NONE;
    }
    // Capabilities belong to the kind of source, not to an instance.
    Registry::with_roots(None, None, None)
        .for_source(source)
        .capabilities()
}

/// Why `doing` ("Deleting", ...) is unavailable for a conversation of
/// `source` and `provenance`.
pub fn refusal(source: SourceKind, provenance: Provenance, doing: &str) -> String {
    if provenance.orphaned {
        format!(
            "{doing} is unavailable: {} no longer has this conversation's current history",
            source.list_label()
        )
    } else {
        format!("{doing} {} sessions is not supported", source.list_label())
    }
}

/// A conversation to act on.
#[derive(Clone, Copy, Debug)]
pub struct Target<'a> {
    /// The conversation's own artifact.
    pub path: &'a Path,
    /// The project directory the conversation is listed under.
    pub project: Option<&'a Path>,
}

/// How to resume a conversation: copy its session into place, if needed,
/// then run `command`.
#[derive(Debug)]
pub struct Resume {
    pub copy: Option<SessionCopy>,
    pub command: Command,
}

/// Copies a Claude session (its transcript and its session directory of
/// tool results and subagents) into another project directory.
#[derive(Debug, Eq, PartialEq)]
pub struct SessionCopy {
    pub transcript: PathBuf,
    pub session_id: String,
    pub from_dir: PathBuf,
    pub to_dir: PathBuf,
}

impl SessionCopy {
    pub fn apply(&self) -> Result<()> {
        std::fs::create_dir_all(&self.to_dir).map_err(AppError::Io)?;
        let target = self.to_dir.join(self.transcript.file_name().unwrap());
        std::fs::copy(&self.transcript, &target).map_err(AppError::Io)?;

        let session_dir = self.from_dir.join(&self.session_id);
        if session_dir.is_dir() {
            copy_dir_recursive(&session_dir, &self.to_dir.join(&self.session_id))?;
        }

        // File history (~/.claude/file-history/<uuid>/) is global, not
        // per project: Claude Code finds it by session id.
        Ok(())
    }
}

fn copy_dir_recursive(src: &Path, dst: &Path) -> Result<()> {
    std::fs::create_dir_all(dst).map_err(AppError::Io)?;
    for entry in std::fs::read_dir(src).map_err(AppError::Io)? {
        let entry = entry.map_err(AppError::Io)?;
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());
        if src_path.is_dir() {
            copy_dir_recursive(&src_path, &dst_path)?;
        } else {
            std::fs::copy(&src_path, &dst_path).map_err(AppError::Io)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn session_copies_bring_the_session_directory() {
        let dir = tempfile::tempdir().unwrap();
        let from = dir.path().join("from");
        fs::create_dir_all(from.join("s/subagents")).unwrap();
        fs::write(from.join("s.jsonl"), "{}\n").unwrap();
        fs::write(from.join("s/subagents/agent-1.jsonl"), "{}\n").unwrap();
        let to = dir.path().join("to");

        SessionCopy {
            transcript: from.join("s.jsonl"),
            session_id: "s".to_owned(),
            from_dir: from,
            to_dir: to.clone(),
        }
        .apply()
        .unwrap();

        assert!(to.join("s.jsonl").is_file());
        assert!(to.join("s/subagents/agent-1.jsonl").is_file());
    }

    #[test]
    fn every_source_has_an_acting_adapter() {
        let registry = Registry::with_roots(None, None, None);
        for source in [
            SourceKind::Claude,
            SourceKind::Pi,
            SourceKind::Omp,
            SourceKind::Codex,
        ] {
            let adapter = registry.for_source(source);
            assert_eq!(adapter.instance().kind, source);
            let capabilities = adapter.capabilities();
            assert_eq!(capabilities.rename, source != SourceKind::Codex);
            assert!(capabilities.resume && capabilities.fork && capabilities.delete);
        }
    }
}
