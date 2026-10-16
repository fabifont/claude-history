//! Shared loading for Pi and OMP session roots.

use super::{Conversation, Source, format_short_name_from_path};
use crate::cli::DebugLevel;
use crate::error::Result;
use std::path::PathBuf;
use std::time::SystemTime;

/// Parses `files` newest first, emitting the conversations of `source` in
/// batches.
pub(super) fn load(
    files: Vec<PathBuf>,
    source: Source,
    show_last: bool,
    debug_level: Option<DebugLevel>,
    mut parse: impl FnMut(PathBuf, Option<SystemTime>) -> Result<Option<Conversation>>,
    mut emit: impl FnMut(Vec<Conversation>),
) -> Result<()> {
    let mut files: Vec<_> = files
        .into_iter()
        .filter_map(|path| {
            let modified = std::fs::metadata(&path).ok()?.modified().ok();
            Some((path, modified))
        })
        .collect();
    files.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));

    let mut batch = Vec::new();
    let mut first = true;
    for (path, modified) in files {
        let mut conversation = match parse(path.clone(), modified) {
            Ok(Some(conversation)) if conversation.source == source => conversation,
            Err(error) => {
                crate::debug::warn(
                    debug_level,
                    &format!("Failed to parse session {}: {error}", path.display()),
                );
                continue;
            }
            Ok(_) => continue,
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
        conversation.project_path = Some(project_path);
        batch.push(conversation);
        if first || batch.len() >= 32 {
            emit(std::mem::take(&mut batch));
            first = false;
        }
    }
    if !batch.is_empty() {
        emit(batch);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sessions_stream_newest_first_and_skip_other_sources() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("session.jsonl");
        std::fs::copy(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pi/v1.jsonl"),
            &path,
        )
        .unwrap();
        let empty = directory.path().join("empty.jsonl");
        std::fs::write(&empty, "").unwrap();
        let mut batches = Vec::new();
        load(
            vec![empty, path.clone()],
            Source::Pi,
            true,
            None,
            |path, modified| super::super::parser::process_conversation_file(path, modified, None),
            |batch| batches.push(batch),
        )
        .unwrap();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].len(), 1);
        assert_eq!(batches[0][0].path, path);
        assert_eq!(batches[0][0].preview, batches[0][0].preview_last);

        let mut emitted = 0;
        load(
            vec![path],
            Source::Omp,
            true,
            None,
            |path, modified| super::super::parser::process_conversation_file(path, modified, None),
            |batch| emitted += batch.len(),
        )
        .unwrap();
        assert_eq!(emitted, 0);
    }
}
