//! Opens conversations by key: resolves the history the key selects
//! through its instance's adapter and decodes it, reusing the transcript
//! decoded last while the history's artifacts are unchanged. The
//! transcripts of the conversations opened most recently are kept.

use super::adapter::{Located, Stamp};
use super::model::{ConversationKey, Revision, Transcript};
use super::sources::Registry;
use crate::error::{AppError, Result};
use std::collections::HashMap;
use std::io;
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::SystemTime;

/// How many decoded transcripts a service keeps.
const KEPT: usize = 8;

type Opened = HashMap<ConversationKey, (u64, Stamp, Arc<Transcript>)>;

pub struct HistoryService {
    registry: Registry,
    opened: Mutex<(u64, Opened)>,
}

/// The service over the instances the environment configures, shared by
/// the whole process.
pub fn shared() -> &'static HistoryService {
    static SHARED: OnceLock<HistoryService> = OnceLock::new();
    SHARED.get_or_init(HistoryService::default)
}

/// The instances the environment configures.
impl Default for HistoryService {
    fn default() -> Self {
        Self::new(Registry::from_env())
    }
}

impl HistoryService {
    pub fn new(registry: Registry) -> Self {
        Self {
            registry,
            opened: Mutex::default(),
        }
    }

    /// The selected history of conversation `key`.
    pub fn open(&self, key: &ConversationKey) -> Result<(Revision, Arc<Transcript>)> {
        let Some(adapter) = self.registry.get(&key.instance) else {
            return Err(not_found(&format!(
                "no {} history is configured at {}",
                key.instance.kind.label(),
                key.instance.root.display()
            )));
        };
        let located = adapter.resolve(&key.native_id, SystemTime::now())?;
        if let Some(transcript) = self.reusable(key, &located) {
            return Ok((transcript.revision, transcript));
        }
        let Some(transcript) = adapter.decode(&located)? else {
            return Err(not_found(&format!(
                "{} holds no conversation",
                located.path.display()
            )));
        };
        let transcript = Arc::new(transcript);
        if located.stamp.is_known() {
            let mut opened = self.lock();
            let (clock, kept) = &mut *opened;
            *clock += 1;
            if kept.len() >= KEPT && !kept.contains_key(key) {
                let oldest = kept
                    .iter()
                    .min_by_key(|(_, (used, ..))| *used)
                    .map(|(key, _)| key.clone());
                if let Some(oldest) = oldest {
                    kept.remove(&oldest);
                }
            }
            kept.insert(key.clone(), (*clock, located.stamp, transcript.clone()));
        }
        Ok((transcript.revision, transcript))
    }

    /// The conversation whose own artifact is `path`: through the instance
    /// whose root holds it, else decoded as a loose session file.
    pub fn open_path(&self, path: &Path) -> Result<(Revision, Arc<Transcript>)> {
        match self.registry.key_of(path) {
            Some(key) => self.open(&key),
            None => {
                let transcript = Arc::new(super::sources::decode_path(path, None, false)?);
                Ok((transcript.revision, transcript))
            }
        }
    }

    fn reusable(&self, key: &ConversationKey, located: &Located) -> Option<Arc<Transcript>> {
        let mut opened = self.lock();
        let (clock, kept) = &mut *opened;
        let (used, stamp, transcript) = kept.get_mut(key)?;
        if !located.stamp.is_known() || *stamp != located.stamp {
            return None;
        }
        *clock += 1;
        *used = *clock;
        Some(transcript.clone())
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, (u64, Opened)> {
        self.opened
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

fn not_found(message: &str) -> AppError {
    AppError::Io(io::Error::new(io::ErrorKind::NotFound, message.to_owned()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::history::model::SourceInstance;
    use std::fs;
    use std::path::PathBuf;
    use std::time::Duration;

    const USER: &str = r#"{"type":"user","message":{"role":"user","content":"hello"}}"#;

    fn settle(path: &Path) {
        let old = SystemTime::now() - Duration::from_secs(60);
        fs::File::options()
            .write(true)
            .open(path)
            .unwrap()
            .set_modified(old)
            .unwrap();
    }

    fn fixture() -> (tempfile::TempDir, HistoryService, ConversationKey) {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("projects");
        fs::create_dir_all(root.join("-p")).unwrap();
        let path = root.join("-p/s.jsonl");
        fs::write(&path, format!("{USER}\n")).unwrap();
        settle(&path);
        let service = HistoryService::new(Registry::with_roots(Some(root.clone()), None, None));
        let key = ConversationKey {
            instance: SourceInstance {
                kind: crate::history::model::SourceKind::Claude,
                root,
            },
            native_id: "-p/s.jsonl".to_owned(),
        };
        (dir, service, key)
    }

    #[test]
    fn unchanged_histories_reuse_the_decoded_transcript() {
        let (_dir, service, key) = fixture();
        let (revision, first) = service.open(&key).unwrap();
        let (again, second) = service.open(&key).unwrap();
        assert_eq!(revision, again);
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(first.events.len(), 1);
    }

    #[test]
    fn changed_histories_are_decoded_again() {
        let (_dir, service, key) = fixture();
        let (revision, _) = service.open(&key).unwrap();
        let path = key.instance.root.join(&key.native_id);
        fs::write(&path, format!("{USER}\n{USER}\n")).unwrap();
        settle(&path);
        let (changed, transcript) = service.open(&key).unwrap();
        assert_ne!(revision, changed);
        assert_eq!(transcript.events.len(), 2);
    }

    #[test]
    fn paths_open_through_the_instance_that_holds_them() {
        let (dir, service, key) = fixture();
        let path = key.instance.root.join(&key.native_id);
        let (_, transcript) = service.open_path(&path).unwrap();
        assert_eq!(transcript.key, key);

        let loose = dir.path().join("loose.jsonl");
        fs::write(&loose, format!("{USER}\n")).unwrap();
        let (_, transcript) = service.open_path(&loose).unwrap();
        assert_eq!(transcript.path(), loose);
    }

    #[test]
    fn missing_histories_are_not_found() {
        let (_dir, service, mut key) = fixture();
        key.native_id = "-p/gone.jsonl".to_owned();
        assert!(matches!(
            service.open(&key),
            Err(AppError::Io(error)) if error.kind() == io::ErrorKind::NotFound
        ));
        key.instance.root = PathBuf::from("/elsewhere");
        assert!(matches!(
            service.open(&key),
            Err(AppError::Io(error)) if error.kind() == io::ErrorKind::NotFound
        ));
    }
}
