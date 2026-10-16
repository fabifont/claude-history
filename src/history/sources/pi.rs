//! Pi and OMP: one JSONL session file per conversation, in per-project
//! directories of a session root or directly in a flat one.

use crate::history::adapter::{Adapter, Completeness, Discovery};
use crate::history::model::{SourceInstance, SourceKind};
use crate::history::omp_loader::{self, OmpSessionRoot};
use crate::history::pi_loader::{self, PiSessionRoot};
use std::path::Path;
use std::time::SystemTime;

/// The adapter of a Pi or an OMP session root.
pub struct PiAdapter {
    instance: SourceInstance,
    flat: bool,
}

impl PiAdapter {
    pub fn pi(root: PiSessionRoot) -> Self {
        Self::new(SourceKind::Pi, root.path, root.flat)
    }

    pub fn omp(root: OmpSessionRoot) -> Self {
        Self::new(SourceKind::Omp, root.path, root.flat)
    }

    fn new(kind: SourceKind, root: std::path::PathBuf, flat: bool) -> Self {
        Self {
            instance: SourceInstance { kind, root },
            flat,
        }
    }
}

impl Adapter for PiAdapter {
    fn instance(&self) -> &SourceInstance {
        &self.instance
    }

    fn discover(&self, scan_started: SystemTime) -> Discovery {
        let root = self.instance.root.clone();
        let files = match self.instance.kind {
            SourceKind::Omp => omp_loader::discover_files(&OmpSessionRoot {
                path: root,
                flat: self.flat,
            }),
            _ => pi_loader::discover_files(&PiSessionRoot {
                path: root,
                flat: self.flat,
            }),
        };
        match files {
            Ok(files) => Discovery {
                found: super::locate_files(&self.instance, files, scan_started),
                completeness: Completeness::Complete,
            },
            Err(_) => Discovery {
                found: Vec::new(),
                completeness: Completeness::Degraded,
            },
        }
    }
}

/// Whether OMP sessions under `root` are OMP's own format rather than Pi's.
pub(crate) fn omp_owned(root: &Path) -> bool {
    root.components()
        .any(|component| matches!(component.as_os_str().to_str(), Some(".omp" | "omp")))
}
