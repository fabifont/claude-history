//! The adapters of each source, and the registry of the instances the
//! environment configures.

pub mod claude;
pub mod codex;
pub mod pi;

use super::adapter::{Adapter, FileId, Located, Stamp};
use super::model::{ConversationKey, Event, SourceInstance, SourceKind, Transcript};
use super::omp_loader::{self, OmpSessionRoot};
use super::pi_loader::{self, PiSessionRoot};
use crate::error::Result;
use rayon::prelude::*;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;

/// The adapters of every configured history root, in a fixed order: Claude,
/// Pi, OMP, Codex.
#[derive(Clone, Default)]
pub struct Registry {
    adapters: Vec<Arc<dyn Adapter>>,
}

impl Registry {
    /// The roots the environment selects (`CLAUDE_CONFIG_DIR`, the Pi and
    /// OMP settings, `CODEX_HOME`), whether or not they exist.
    pub fn from_env() -> Self {
        Self::with_roots(
            super::get_claude_projects_root().ok(),
            pi_loader::session_root().ok(),
            omp_loader::session_root().ok(),
        )
        .with_codex(codex::home_from_env())
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

    /// Adds the adapter of a Codex home.
    pub fn with_codex(mut self, home: Option<PathBuf>) -> Self {
        if let Some(home) = home {
            self.adapters.push(Arc::new(codex::CodexAdapter::new(home)));
        }
        self
    }

    pub fn adapters(&self) -> &[Arc<dyn Adapter>] {
        &self.adapters
    }

    /// The conversation whose own artifact is `path`, in the first instance
    /// whose root holds it.
    pub fn key_of(&self, path: &Path) -> Option<ConversationKey> {
        self.adapters.iter().find_map(|adapter| {
            let instance = adapter.instance();
            if !path.starts_with(&instance.root) {
                return None;
            }
            Some(ConversationKey {
                instance: instance.clone(),
                native_id: adapter.native_id_of(path)?,
            })
        })
    }

    /// The adapter that acts on conversations `source` wrote: its first
    /// configured instance, else one without a root.
    pub fn for_source(&self, source: SourceKind) -> Arc<dyn Adapter> {
        if let Some(adapter) = self
            .adapters
            .iter()
            .find(|adapter| adapter.instance().kind == source)
        {
            return adapter.clone();
        }
        match source {
            SourceKind::Claude => Arc::new(claude::ClaudeAdapter::new(PathBuf::new())),
            SourceKind::Pi => Arc::new(pi::PiAdapter::pi(PiSessionRoot {
                path: PathBuf::new(),
                flat: false,
            })),
            SourceKind::Omp => Arc::new(pi::PiAdapter::omp(OmpSessionRoot {
                path: PathBuf::new(),
                flat: false,
            })),
            SourceKind::Codex => Arc::new(codex::CodexAdapter::with_sqlite_home(
                PathBuf::new(),
                PathBuf::new(),
            )),
        }
    }

    /// The adapter of `instance`.
    pub fn get(&self, instance: &SourceInstance) -> Option<&Arc<dyn Adapter>> {
        self.adapters
            .iter()
            .find(|adapter| adapter.instance() == instance)
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

/// `Located` for the session file `native_id` names in `instance`.
fn resolve_file(
    instance: &SourceInstance,
    native_id: &str,
    scan_started: SystemTime,
) -> Result<Located> {
    let path = instance.root.join(native_id);
    let metadata = fs::metadata(&path)?;
    Ok(located(instance, path, &metadata, scan_started))
}

fn locate_file(
    instance: &SourceInstance,
    path: PathBuf,
    scan_started: SystemTime,
) -> Option<Located> {
    let metadata = fs::metadata(&path).ok()?;
    Some(located(instance, path, &metadata, scan_started))
}

fn located(
    instance: &SourceInstance,
    path: PathBuf,
    metadata: &fs::Metadata,
    scan_started: SystemTime,
) -> Located {
    let modified = metadata.modified().ok();
    let size = metadata.len();
    Located {
        key: ConversationKey {
            instance: instance.clone(),
            native_id: native_id(&instance.root, &path),
        },
        stamp: Stamp::of(size, modified, scan_started),
        file_id: file_id(metadata),
        path,
        size,
        modified,
        native: None,
    }
}

/// Native id of a session file: its path relative to the instance root.
fn native_id(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .into_owned()
}

/// Decodes a session file outside any configured root: a Codex rollout as
/// the thread of the Codex home holding it, else recognizing its source
/// from its contents (as OMP when it starts with a session header and
/// `omp` is set). `modified` dates a session that records no activity time.
pub fn decode_path(path: &Path, modified: Option<SystemTime>, omp: bool) -> Result<Transcript> {
    if let Some(transcript) = codex::view::decode_path(path) {
        return transcript;
    }
    let instance = SourceInstance {
        kind: SourceKind::Claude,
        root: path.parent().map(Path::to_path_buf).unwrap_or_default(),
    };
    let located = Located {
        key: ConversationKey {
            native_id: native_id(&instance.root, path),
            instance,
        },
        path: path.to_path_buf(),
        size: 0,
        modified,
        file_id: None,
        stamp: Stamp::default(),
        native: None,
    };
    let mut decoded = Vec::new();
    let transcript = decode_file(&located, omp, &mut |event| decoded.push(event))?
        .expect("a file that is not a Pi session decodes as a Claude session");
    Ok(Transcript {
        events: decoded,
        ..transcript
    })
}

/// Decodes the session file `located` names: as a Pi or OMP session when it
/// starts with a session header (always as OMP when `omp` is set), else as
/// a Claude session. `None` when it holds neither.
fn decode_file(
    located: &Located,
    omp: bool,
    events: &mut dyn FnMut(Event),
) -> Result<Option<Transcript>> {
    let path = &located.path;
    if super::pi::detect_source(path)?.is_some() {
        let mut reader = Counted::new(BufReader::new(File::open(path)?));
        let expected = omp.then_some(SourceKind::Omp);
        if let Some(projection) = super::pi::parse_reader(&mut reader, expected)? {
            return Ok(Some(pi::transcript_of(
                located.key.clone(),
                path,
                projection,
                reader.bytes,
                located.modified,
                events,
            )));
        }
    }
    let reader = BufReader::new(File::open(path)?);
    claude::decode_reader(located.key.clone(), path, reader, located.modified, events).map(Some)
}

/// A reader that counts the bytes consumed through it.
struct Counted<R> {
    inner: R,
    bytes: u64,
}

impl<R> Counted<R> {
    fn new(inner: R) -> Self {
        Self { inner, bytes: 0 }
    }
}

impl<R: Read> Read for Counted<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let read = self.inner.read(buf)?;
        self.bytes += read as u64;
        Ok(read)
    }
}

impl<R: BufRead> BufRead for Counted<R> {
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        self.inner.fill_buf()
    }

    fn consume(&mut self, amount: usize) {
        self.bytes += amount as u64;
        self.inner.consume(amount);
    }
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
