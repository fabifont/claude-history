//! Shared cached loading for Pi and OMP session roots.

use super::cache::{self, PiCacheEntry};
use super::{Conversation, Source, format_short_name_from_path};
use crate::cli::DebugLevel;
use crate::error::Result;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

#[allow(clippy::too_many_arguments)]
pub(super) fn load(
    root: &Path,
    files: Vec<PathBuf>,
    source: Source,
    show_last: bool,
    debug_level: Option<DebugLevel>,
    cached: Option<HashMap<String, PiCacheEntry>>,
    persist: impl FnOnce(HashMap<String, PiCacheEntry>),
    mut parse: impl FnMut(PathBuf, Option<SystemTime>) -> Result<Option<Conversation>>,
    mut emit: impl FnMut(Vec<Conversation>),
) -> Result<()> {
    let mut dirty = cached.is_none();
    let mut cached = cached.unwrap_or_default();
    let mut files: Vec<_> = files
        .into_iter()
        .filter_map(|path| {
            let metadata = std::fs::metadata(&path).ok()?;
            let modified = metadata.modified().ok();
            let key = path
                .strip_prefix(root)
                .unwrap_or(&path)
                .to_string_lossy()
                .into_owned();
            let hit = modified.is_some_and(|mtime| {
                cached.get(&key).is_some_and(|entry| {
                    cache::entry_matches(&entry.metadata, metadata.len(), mtime)
                })
            });
            dirty |= !hit;
            Some((path, key, metadata.len(), modified, hit))
        })
        .collect();
    dirty |= files.len() != cached.len();
    files.sort_by(|left, right| right.3.cmp(&left.3).then_with(|| left.0.cmp(&right.0)));

    let mut updated = HashMap::new();
    let mut batch = Vec::new();
    let mut first = true;
    for (path, key, size, modified, hit) in files {
        let conversation = if hit {
            let entry = cached.remove(&key).expect("validated cache entry");
            if entry.metadata.is_empty {
                if dirty {
                    updated.insert(key, entry);
                }
                continue;
            }
            let session_id = entry.session_id.clone();
            let project_path = entry.project_path.clone();
            let mut conversation = if dirty {
                let conversation = cache::conversation_from_entry(&entry.metadata, path, show_last);
                updated.insert(key.clone(), entry);
                conversation
            } else {
                cache::conversation_from_owned_entry(entry.metadata, path, show_last)
            };
            conversation.source = source;
            conversation.session_id = session_id;
            conversation.cwd = Some(project_path.clone());
            conversation.project_path = Some(project_path.clone());
            conversation.project_name = Some(format_short_name_from_path(&project_path));
            Some(conversation)
        } else {
            match parse(path.clone(), modified) {
                Ok(Some(conversation)) if conversation.source == source => Some(conversation),
                Err(error) => {
                    crate::debug::warn(
                        debug_level,
                        &format!("Failed to parse session {}: {error}", path.display()),
                    );
                    None
                }
                Ok(_) => {
                    if let Some(mtime) = modified {
                        updated.insert(
                            key.clone(),
                            PiCacheEntry {
                                metadata: cache::empty_entry(size, mtime),
                                session_id: String::new(),
                                project_path: PathBuf::new(),
                            },
                        );
                    }
                    None
                }
            }
        };
        let Some(mut conversation) = conversation else {
            continue;
        };
        conversation.preview = if show_last {
            conversation.preview_last.clone()
        } else {
            conversation.preview_first.clone()
        };
        let project_path = conversation
            .cwd
            .clone()
            .unwrap_or_else(|| PathBuf::from("unknown"));
        conversation.project_name = Some(format_short_name_from_path(&project_path));
        conversation.project_path = Some(project_path.clone());
        if dirty
            && !hit
            && let Some(mtime) = modified
        {
            updated.insert(
                key,
                PiCacheEntry {
                    metadata: cache::entry_from_conversation(&conversation, size, mtime),
                    session_id: conversation.session_id.clone(),
                    project_path,
                },
            );
        }
        batch.push(conversation);
        if first || batch.len() >= 32 {
            emit(std::mem::take(&mut batch));
            first = false;
        }
    }
    if !batch.is_empty() {
        emit(batch);
    }
    if dirty {
        persist(updated);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    #[test]
    fn empty_sessions_are_cached_without_repeated_parsing_or_writes() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("empty.jsonl");
        std::fs::write(&path, "").unwrap();
        let cache = RefCell::new(None);
        load(
            directory.path(),
            vec![path.clone()],
            Source::Pi,
            false,
            None,
            None,
            |entries| *cache.borrow_mut() = Some(entries),
            |_, _| Ok(None),
            |_| panic!("empty session must not be emitted"),
        )
        .unwrap();
        load(
            directory.path(),
            vec![path],
            Source::Pi,
            false,
            None,
            cache.into_inner(),
            |_| panic!("negative cache hit must not be rewritten"),
            |_, _| panic!("negative cache hit must not be reparsed"),
            |_| panic!("negative cache hit must not be emitted"),
        )
        .unwrap();
    }

    #[test]
    fn warm_cache_skips_writes_and_streams_before_persistence() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("session.jsonl");
        std::fs::copy(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pi/v1.jsonl"),
            &path,
        )
        .unwrap();
        let cache = RefCell::new(None);
        let events = RefCell::new(Vec::new());
        let mut first = Vec::new();
        load(
            directory.path(),
            vec![path.clone()],
            Source::Pi,
            true,
            None,
            None,
            |entries| {
                events.borrow_mut().push("persist");
                *cache.borrow_mut() = Some(entries);
            },
            |path, modified| super::super::parser::process_conversation_file(path, modified, None),
            |batch| {
                events.borrow_mut().push("batch");
                first.extend(batch);
            },
        )
        .unwrap();
        assert_eq!(*events.borrow(), ["batch", "persist"]);
        assert_eq!(first.len(), 1);
        let cached = cache.into_inner().unwrap();
        let mut warm = Vec::new();
        load(
            directory.path(),
            vec![path.clone()],
            Source::Pi,
            true,
            None,
            Some(cached.clone()),
            |_| panic!("unchanged cache must not be rewritten"),
            |_, _| panic!("unchanged session must not be parsed"),
            |batch| warm.extend(batch),
        )
        .unwrap();
        assert_eq!(warm.len(), 1);
        assert_eq!(warm[0].full_text, first[0].full_text);
        assert_eq!(warm[0].session_id, first[0].session_id);
        assert_eq!(warm[0].preview, first[0].preview);

        // A replacement with the same file count still invalidates the cache.
        let replacement = directory.path().join("replacement.jsonl");
        std::fs::rename(&path, &replacement).unwrap();
        let updated = RefCell::new(None);
        load(
            directory.path(),
            vec![replacement],
            Source::Pi,
            true,
            None,
            Some(cached.clone()),
            |entries| *updated.borrow_mut() = Some(entries),
            |path, modified| super::super::parser::process_conversation_file(path, modified, None),
            |_| {},
        )
        .unwrap();
        let updated = updated.into_inner().unwrap();
        assert!(!updated.contains_key("session.jsonl"));
        assert!(updated.contains_key("replacement.jsonl"));

        load(
            directory.path(),
            Vec::new(),
            Source::Pi,
            true,
            None,
            Some(cached),
            |entries| assert!(entries.is_empty()),
            |_, _| panic!("deleted sessions must not be parsed"),
            |_| panic!("deleted sessions must not be emitted"),
        )
        .unwrap();
    }
}
