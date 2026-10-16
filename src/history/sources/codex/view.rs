//! Codex rollouts opened by path, outside any configured instance.

use super::{CodexAdapter, discover};
use crate::error::{AppError, Result};
use crate::history::adapter::Adapter;
use crate::history::model::Transcript;
use std::path::Path;
use std::time::SystemTime;

/// The thread of the rollout at `path`, decoded through the Codex home
/// holding it. `None` when `path` is not a rollout of a Codex home.
pub fn decode_path(path: &Path) -> Option<Result<Transcript>> {
    let home = discover::home_of(path)?;
    let thread_id = discover::rollout_of(path, false)?.thread_id;
    let adapter = CodexAdapter::new(home.to_path_buf());
    Some(
        adapter
            .resolve(&thread_id, SystemTime::now())
            .and_then(|located| adapter.decode(&located))
            .and_then(|transcript| {
                transcript.ok_or_else(|| {
                    AppError::Io(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("not a Codex rollout: {}", path.display()),
                    ))
                })
            }),
    )
}
