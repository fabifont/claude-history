//! The adapters of each source, and the registry of the instances the
//! environment configures.

pub mod claude;
pub mod pi;

use super::adapter::{Adapter, FileId, Located, Stamp};
use super::model::{ConversationKey, SourceInstance, SourceKind};
use super::omp_loader::{self, OmpSessionRoot};
use super::pi_loader::{self, PiSessionRoot};
use rayon::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;

/// The adapters of every configured history root, in a fixed order: Claude,
/// Pi, OMP.
#[derive(Clone, Default)]
pub struct Registry {
    adapters: Vec<Arc<dyn Adapter>>,
}

impl Registry {
    /// The roots the environment selects (`CLAUDE_CONFIG_DIR`, the Pi and
    /// OMP settings), whether or not they exist.
    pub fn from_env() -> Self {
        Self::with_roots(
            super::get_claude_projects_root().ok(),
            pi_loader::session_root().ok(),
            omp_loader::session_root().ok(),
        )
    }

    /// Adapters for a Claude `projects` directory and Pi and OMP session
    /// roots.
    pub fn with_roots(
        claude: Option<PathBuf>,
        pi: Option<PiSessionRoot>,
        omp: Option<OmpSessionRoot>,
    ) -> Self {
        let mut adapters: Vec<Arc<dyn Adapter>> = Vec::new();
        if let Some(root) = claude {
            adapters.push(Arc::new(claude::ClaudeAdapter::new(root)));
        }
        if let Some(root) = pi {
            adapters.push(Arc::new(pi::PiAdapter::pi(root)));
        }
        if let Some(root) = omp {
            adapters.push(Arc::new(pi::PiAdapter::omp(root)));
        }
        Self { adapters }
    }

    pub fn adapters(&self) -> &[Arc<dyn Adapter>] {
        &self.adapters
    }

    /// Adapters of `kind`.
    pub fn of_kind(&self, kind: SourceKind) -> impl Iterator<Item = &Arc<dyn Adapter>> {
        self.adapters
            .iter()
            .filter(move |adapter| adapter.instance().kind == kind)
    }
}

/// `Located` for each of `paths`, an instance's session files, one per
/// conversation, stated in parallel. Files that cannot be stated are left
/// out.
fn locate_files(
    instance: &SourceInstance,
    paths: Vec<PathBuf>,
    scan_started: SystemTime,
) -> Vec<Located> {
    paths
        .into_par_iter()
        .filter_map(|path| locate_file(instance, path, scan_started))
        .collect()
}

fn locate_file(
    instance: &SourceInstance,
    path: PathBuf,
    scan_started: SystemTime,
) -> Option<Located> {
    let metadata = fs::metadata(&path).ok()?;
    let modified = metadata.modified().ok();
    let size = metadata.len();
    Some(Located {
        key: ConversationKey {
            instance: instance.clone(),
            native_id: native_id(&instance.root, &path),
        },
        stamp: Stamp::of(size, modified, scan_started),
        file_id: file_id(&metadata),
        path,
        size,
        modified,
    })
}

/// Native id of a session file: its path relative to the instance root.
fn native_id(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .into_owned()
}

#[cfg(unix)]
fn file_id(metadata: &fs::Metadata) -> Option<FileId> {
    use std::os::unix::fs::MetadataExt;
    Some((metadata.dev(), metadata.ino()))
}

#[cfg(not(unix))]
fn file_id(_metadata: &fs::Metadata) -> Option<FileId> {
    None
}
