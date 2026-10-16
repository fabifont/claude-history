//! Claude Code: one JSONL file per session in each project directory of a
//! `projects` root.

use crate::history::adapter::{Adapter, Completeness, Discovery};
use crate::history::model::{SourceInstance, SourceKind};
use rayon::prelude::*;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

pub struct ClaudeAdapter {
    instance: SourceInstance,
}

impl ClaudeAdapter {
    /// The adapter of a `projects` directory.
    pub fn new(root: PathBuf) -> Self {
        Self {
            instance: SourceInstance {
                kind: SourceKind::Claude,
                root,
            },
        }
    }
}

impl Adapter for ClaudeAdapter {
    fn instance(&self) -> &SourceInstance {
        &self.instance
    }

    fn discover(&self, scan_started: SystemTime) -> Discovery {
        let mut discovery = Discovery::default();
        let files = list_sessions(&self.instance.root, &mut discovery);
        discovery.found = super::locate_files(&self.instance, files, scan_started);
        discovery
    }
}

/// Session files: `*.jsonl` directly inside each project directory, except
/// subagent transcripts (`agent-*`). A missing root holds none.
fn list_sessions(root: &Path, discovery: &mut Discovery) -> Vec<PathBuf> {
    let entries = match fs::read_dir(root) {
        Ok(entries) => entries,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Vec::new(),
        Err(_) => {
            discovery.completeness = Completeness::Degraded;
            return Vec::new();
        }
    };
    let mut dirs = Vec::new();
    for entry in entries {
        match entry {
            Ok(entry) => dirs.push(entry.path()),
            Err(_) => discovery.completeness = Completeness::Degraded,
        }
    }
    let listed: Vec<(PathBuf, io::Result<Vec<PathBuf>>)> = dirs
        .into_par_iter()
        .filter(|dir| dir.is_dir())
        .map(|dir| {
            let files = list_project_dir(&dir);
            (dir, files)
        })
        .collect();
    let mut files = Vec::new();
    for (dir, listed) in listed {
        match listed {
            Ok(listed) => files.extend(listed),
            Err(_) => discovery.missed_dir(dir),
        }
    }
    files
}

fn list_project_dir(dir: &Path) -> io::Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if is_session_file(&path) {
            files.push(path);
        }
    }
    Ok(files)
}

fn is_session_file(path: &Path) -> bool {
    path.extension()
        .is_some_and(|extension| extension == "jsonl")
        && !path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with("agent-"))
}
