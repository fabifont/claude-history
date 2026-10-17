//! Codex: threads whose history is one or more JSONL rollouts under a Codex
//! home (`CODEX_HOME`, else `~/.codex`), chosen with the help of the state
//! database (see `state_db`).
//!
//! A thread's native id is its thread id. Codex's own CLI acts on threads
//! by id, with `CODEX_HOME` selecting the home.

mod decode;
pub mod discover;
mod head;
mod state_db;
pub mod view;

use crate::error::{AppError, Result};
use crate::history::actions::{Capabilities, Launch, Launcher, Resume, Target};
use crate::history::adapter::{Adapter, Completeness, Discovery, Located, Native};
use crate::history::model::{Event, SourceInstance, SourceKind, Transcript};
use head::{Scan, Thread};
use rayon::prelude::*;
use state_db::StateDb;
use std::collections::HashMap;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::time::SystemTime;

/// The adapter of a Codex home.
pub struct CodexAdapter {
    instance: SourceInstance,
    /// Directory of the state database, when not where Codex would look.
    sqlite_home: Option<PathBuf>,
}

impl CodexAdapter {
    /// The adapter of the Codex home at `home`, with its state database
    /// where Codex would look for it.
    pub fn new(home: PathBuf) -> Self {
        Self {
            instance: SourceInstance {
                kind: SourceKind::Codex,
                root: home,
            },
            sqlite_home: None,
        }
    }

    pub fn with_sqlite_home(home: PathBuf, sqlite_home: PathBuf) -> Self {
        Self {
            sqlite_home: Some(sqlite_home),
            ..Self::new(home)
        }
    }

    /// The directory of the state database: the `sqlite_home` of the
    /// home's `config.toml`, else `CODEX_SQLITE_HOME`, else the home.
    /// Directories whose files hold thread state beside the rollouts: the
    /// home, with its session index, and the state database's directory.
    pub fn state_dirs(&self) -> Vec<PathBuf> {
        let mut dirs = vec![self.instance.root.clone()];
        let sqlite = self.sqlite_home();
        if sqlite != self.instance.root {
            dirs.push(sqlite);
        }
        dirs
    }

    fn sqlite_home(&self) -> PathBuf {
        let home = &self.instance.root;
        self.sqlite_home
            .clone()
            .or_else(|| configured_sqlite_home(home))
            .or_else(|| env_path("CODEX_SQLITE_HOME"))
            .unwrap_or_else(|| home.clone())
    }

    /// Every rollout of the home with the state that chooses among them,
    /// limited to the state database row of `only` when given. An error
    /// means the listing or the database could not be read now.
    fn scan(&self, only: Option<&str>) -> std::result::Result<Scan, AppError> {
        let rollouts = discover::list(&self.instance.root)?;
        let rows = match StateDb::open(&self.sqlite_home()).map_err(sqlite_error)? {
            None => HashMap::new(),
            Some(db) => match only {
                None => db.rows().map_err(sqlite_error)?,
                Some(id) => db
                    .row(id)
                    .map_err(sqlite_error)?
                    .map(|row| (id.to_owned(), row))
                    .into_iter()
                    .collect(),
            },
        };
        let names = discover::session_names(&self.instance.root);
        Ok(Scan::new(rollouts, rows, names))
    }

    fn located(&self, thread: Thread, scan_started: SystemTime) -> Located {
        let head = thread.head_link();
        Located {
            key: crate::history::model::ConversationKey {
                instance: self.instance.clone(),
                native_id: thread.id.clone(),
            },
            path: head.path.clone(),
            size: head.size,
            modified: head.modified,
            file_id: head.file_id,
            stamp: thread.stamp(scan_started),
            native: Some(Native(Arc::new(thread))),
        }
    }

    /// The thread `located` resolved, resolving it again when it carries
    /// none.
    fn thread_of(&self, located: &Located) -> Result<Arc<Thread>> {
        if let Some(thread) = located
            .native
            .as_ref()
            .and_then(|native| native.0.clone().downcast::<Thread>().ok())
        {
            return Ok(thread);
        }
        let located = self.resolve(&located.key.native_id, SystemTime::now())?;
        Ok(located
            .native
            .and_then(|native| native.0.downcast::<Thread>().ok())
            .expect("resolve carries the thread"))
    }
}

/// `$CODEX_HOME`, else `~/.codex`.
pub fn home_from_env() -> Option<PathBuf> {
    env_path("CODEX_HOME").or_else(|| home::home_dir().map(|home| home.join(".codex")))
}

fn env_path(name: &str) -> Option<PathBuf> {
    let value = std::env::var_os(name).filter(|value| !value.is_empty())?;
    let path = PathBuf::from(value);
    if path.is_absolute() {
        Some(path)
    } else {
        std::env::current_dir().ok().map(|cwd| cwd.join(path))
    }
}

/// `sqlite_home` of the home's `config.toml`.
fn configured_sqlite_home(home: &Path) -> Option<PathBuf> {
    let text = std::fs::read_to_string(home.join("config.toml")).ok()?;
    let config: toml::Table = toml::from_str(&text).ok()?;
    let path = PathBuf::from(config.get("sqlite_home")?.as_str()?);
    Some(if path.is_absolute() {
        path
    } else {
        home.join(path)
    })
}

fn sqlite_error(error: rusqlite::Error) -> AppError {
    AppError::Io(std::io::Error::other(format!(
        "Codex state database unavailable: {error}"
    )))
}

/// The thread id of the rollout at `path`.
fn thread_id_of(path: &Path) -> Result<String> {
    discover::rollout_of(path, false)
        .map(|rollout| rollout.thread_id)
        .ok_or_else(|| {
            AppError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("not a Codex rollout: {}", path.display()),
            ))
        })
}

/// The Codex CLI acting on the home of the rollout at `path`, else on
/// `home`.
fn codex_command(launcher: &Launcher, path: &Path, home: &Path) -> Command {
    let mut command = launcher.command();
    let home = discover::home_of(path).unwrap_or(home);
    if !home.as_os_str().is_empty() {
        command.env("CODEX_HOME", home);
    }
    command
}

impl Adapter for CodexAdapter {
    fn instance(&self) -> &SourceInstance {
        &self.instance
    }

    fn native_id_of(&self, path: &Path) -> Option<String> {
        discover::rollout_of(path, false).map(|rollout| rollout.thread_id)
    }

    fn discover(&self, scan_started: SystemTime) -> Discovery {
        let scan = match self.scan(None) {
            Ok(scan) => scan,
            Err(_) => {
                return Discovery {
                    found: Vec::new(),
                    completeness: Completeness::Degraded,
                };
            }
        };
        let found = scan
            .thread_ids()
            .into_par_iter()
            .filter_map(|id| scan.thread(id))
            .map(|thread| self.located(thread, scan_started))
            .collect();
        Discovery {
            found,
            completeness: Completeness::Complete,
        }
    }

    fn resolve(&self, native_id: &str, scan_started: SystemTime) -> Result<Located> {
        let thread = self
            .scan(Some(native_id))?
            .thread(native_id)
            .ok_or_else(|| {
                AppError::Io(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("no Codex rollout of thread {native_id}"),
                ))
            })?;
        Ok(self.located(thread, scan_started))
    }

    fn decode_into(
        &self,
        located: &Located,
        events: &mut dyn FnMut(Event),
    ) -> Result<Option<Transcript>> {
        let thread = self.thread_of(located)?;
        decode::decode(located.key.clone(), &thread, events)
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            rename: false,
            ..Capabilities::ALL
        }
    }

    /// `codex [default_args] resume|fork <thread id>` in the conversation's
    /// project.
    fn resume(&self, target: Target<'_>, fork: bool, launch: &Launch) -> Result<Resume> {
        let id = thread_id_of(target.path)?;
        let mut command = codex_command(&launch.launcher, target.path, &self.instance.root);
        command.args(&launch.default_args);
        command.arg(if fork { "fork" } else { "resume" });
        command.arg(id);
        if let Some(project) = target.project.filter(|project| project.is_dir()) {
            command.current_dir(project);
        }
        Ok(Resume {
            copy: None,
            command,
        })
    }

    /// `codex delete --force <thread id>`, which removes every rollout of
    /// the thread and its state.
    fn delete(&self, path: &Path, launcher: &Launcher) -> Result<()> {
        let id = thread_id_of(path)?;
        launcher.check(SourceKind::Codex)?;
        let output = codex_command(launcher, path, &self.instance.root)
            .args([OsString::from("delete"), "--force".into(), id.into()])
            .stdin(std::process::Stdio::null())
            .output()
            .map_err(AppError::Io)?;
        if output.status.success() {
            return Ok(());
        }
        let stderr = String::from_utf8_lossy(&output.stderr);
        Err(AppError::Io(std::io::Error::other(format!(
            "codex delete failed ({}): {}",
            output.status,
            stderr.trim()
        ))))
    }

    fn rename(&self, _path: &Path, _title: &str) -> Result<()> {
        Err(AppError::Io(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "Codex threads cannot be renamed from here",
        )))
    }
}

#[cfg(test)]
pub(crate) mod tests;
