//! The opt-in background daemon, which keeps the keyword store and the
//! semantic cache up to date so that launches find them warm.
//!
//! With `[semantic] daemon = true`, every launch starts it unless one runs.
//! It notices changed session files by polling (see `watch`), publishes
//! them to the store as a launch would, and embeds conversations once they
//! have gone quiet, in a child process that exits when done (see `embed`).
//! It exits by itself when its binary is replaced, when no launch happened
//! for days, and when the config no longer enables it.
//!
//! Its files live in `daemon/` of the cache directory: the instance lock,
//! `status.json`, the log, and `launched`, whose modification time is the
//! last launch.

pub mod embed;
pub mod identity;
pub mod lock;
pub mod log;
pub mod power;
pub mod process;
pub mod schedule;
mod service;
pub mod status;
pub mod watch;

use crate::cli::DaemonCommand;
use crate::error::{AppError, Result};
use identity::BinaryIdentity;
use status::Status;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// How long a starting daemon waits for the lock, which a daemon it
/// replaces holds until it has stopped.
const LOCK_WAIT: Duration = Duration::from_secs(60);
/// How long `daemon stop` waits for the daemon to exit.
const STOP_WAIT: Duration = Duration::from_secs(15);

/// The daemon's files.
#[derive(Clone, Debug)]
pub struct Paths {
    pub dir: PathBuf,
    pub lock: PathBuf,
    pub status: PathBuf,
    pub log: PathBuf,
    pub launched: PathBuf,
}

impl Paths {
    /// The files in the cache directory, which tests never use.
    pub fn from_env() -> Option<Self> {
        if cfg!(test) {
            return None;
        }
        crate::store::cache_base_dir().map(|base| Self::in_dir(base.join("daemon")))
    }

    pub fn in_dir(dir: PathBuf) -> Self {
        Self {
            lock: dir.join("daemon.lock"),
            status: dir.join("status.json"),
            log: dir.join("daemon.log"),
            launched: dir.join("launched"),
            dir,
        }
    }

    /// Time of the last launch.
    pub fn last_launch(&self) -> Option<SystemTime> {
        std::fs::metadata(&self.launched).ok()?.modified().ok()
    }

    /// Records a launch now.
    fn touch_launched(&self) -> std::io::Result<()> {
        std::fs::create_dir_all(&self.dir)?;
        let file = std::fs::File::options()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&self.launched)?;
        file.set_modified(SystemTime::now())
    }
}

fn enabled() -> bool {
    crate::config::load_config()
        .ok()
        .and_then(|config| config.semantic)
        .unwrap_or_default()
        .daemon()
}

pub fn is_running(paths: &Paths) -> bool {
    lock::is_held(&paths.lock)
}

/// Whether a daemon runs for the cache directory of this environment, so
/// that the TUI leaves embedding to it.
pub fn running() -> bool {
    Paths::from_env().is_some_and(|paths| is_running(&paths))
}

/// Run on every launch: records it and, when the config enables the
/// daemon, starts one unless one of this binary runs. A daemon of another
/// binary is stopped and replaced. With `wake`, a running daemon checks
/// for passages to embed at once, for a person about to search. Never
/// fails the launch.
pub fn on_launch(wake: bool) {
    let Some(paths) = Paths::from_env() else {
        return;
    };
    if !enabled() {
        return;
    }
    if let Err(error) = ensure_running(&paths, wake) {
        let _ = crate::debug_log::log_debug(&format!("Daemon: {error}"));
    }
}

fn ensure_running(paths: &Paths, wake: bool) -> std::io::Result<()> {
    paths.touch_launched()?;
    let current = BinaryIdentity::current()?;
    if is_running(paths) {
        // A daemon that has not written its status yet is starting.
        let Some(status) =
            Status::read(&paths.status).filter(|status| status.exit_reason.is_none())
        else {
            return Ok(());
        };
        if status.binary.same_build(&current) {
            if wake {
                process::wake(status.pid)?;
            }
            return Ok(());
        }
        // The new daemon waits for the lock until this one exits.
        process::terminate(status.pid)?;
    }
    process::spawn_detached(&current.path, &["daemon", "run"])
}

pub fn execute(command: DaemonCommand) -> Result<()> {
    let paths = Paths::from_env()
        .ok_or_else(|| AppError::ConfigError("no home directory for the daemon's files".into()))?;
    match command {
        DaemonCommand::Status => {
            print!("{}", describe(&paths, enabled(), now_ms()));
            Ok(())
        }
        DaemonCommand::Stop => stop(&paths),
        DaemonCommand::Start => start(&paths),
        DaemonCommand::Run => service::run(&paths),
        DaemonCommand::Embed => service::embed_child(),
        DaemonCommand::Pass => service::pass_child(),
    }
}

fn start(paths: &Paths) -> Result<()> {
    if !enabled() {
        return Err(AppError::ConfigError(format!(
            "the daemon is disabled; set `daemon = true` under [semantic] in {}",
            config_path_display()
        )));
    }
    if is_running(paths) {
        println!("The daemon is already running.");
        return Ok(());
    }
    ensure_running(paths, false)?;
    for _ in 0..50 {
        if is_running(paths) {
            println!("Started the daemon.");
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    Err(AppError::ConfigError(format!(
        "the daemon did not start; see {}",
        paths.log.display()
    )))
}

fn stop(paths: &Paths) -> Result<()> {
    if !is_running(paths) {
        println!("The daemon is not running.");
        return Ok(());
    }
    let status = Status::read(&paths.status)
        .ok_or_else(|| AppError::ConfigError("the daemon runs but has no status file".into()))?;
    process::terminate(status.pid)?;
    let deadline = std::time::Instant::now() + STOP_WAIT;
    while std::time::Instant::now() < deadline {
        if !is_running(paths) {
            println!("Stopped the daemon (pid {}).", status.pid);
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    Err(AppError::ConfigError(format!(
        "the daemon (pid {}) did not stop within {} seconds",
        status.pid,
        STOP_WAIT.as_secs()
    )))
}

fn config_path_display() -> String {
    crate::config::get_config_path()
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| "the config file".to_owned())
}

pub(crate) fn now_ms() -> i64 {
    ms_of(SystemTime::now())
}

pub(crate) fn ms_of(time: SystemTime) -> i64 {
    time.duration_since(UNIX_EPOCH).map_or(0, |since| {
        i64::try_from(since.as_millis()).unwrap_or(i64::MAX)
    })
}

/// `daemon status` output.
fn describe(paths: &Paths, enabled: bool, now_ms: i64) -> String {
    use std::fmt::Write;
    let running = is_running(paths);
    let status = Status::read(&paths.status);
    let mut out = String::new();
    let config = if enabled { "enabled" } else { "disabled" };
    match (&status, running) {
        (Some(status), true) => {
            let _ = writeln!(
                out,
                "Daemon: running (pid {}, {} {})",
                status.pid,
                status.binary.version,
                status.binary.path.display()
            );
        }
        (_, true) => out.push_str("Daemon: running\n"),
        (Some(status), false) => {
            let reason = status.exit_reason.as_deref().unwrap_or("it was killed");
            let _ = writeln!(out, "Daemon: not running (last exit: {reason})");
        }
        (None, false) => out.push_str("Daemon: not running\n"),
    }
    let _ = writeln!(out, "Config: {config} ([semantic] daemon)");
    if let Some(status) = status {
        let ago = |ms: Option<i64>| ms.map_or_else(|| "never".to_owned(), |ms| ago(now_ms, ms));
        if running {
            let _ = writeln!(out, "State: {}", status.state);
            let _ = writeln!(out, "Started: {}", ago(Some(status.started_ms)));
            let _ = writeln!(
                out,
                "CPU: {:.1} s since start, its pass and embedding processes included",
                status.cpu_ms as f64 / 1000.0
            );
            if let Some(rss) = process::resident_bytes(status.pid) {
                let _ = writeln!(out, "Memory: {:.1} MB resident", rss as f64 / 1e6);
            }
            let _ = writeln!(
                out,
                "Quick pass: every {} s, last {}",
                status.quick_interval_secs,
                ago(status.last_quick_pass_ms)
            );
        }
        let _ = writeln!(out, "Full pass: {}", ago(status.last_full_pass_ms));
        let _ = writeln!(out, "Last publish: {}", ago(status.last_publish_ms));
        let _ = writeln!(out, "Last embedding: {}", ago(status.last_embed_ms));
        let backlog = status
            .backlog
            .map_or_else(|| "not checked yet".to_owned(), |n| format!("{n} passages"));
        let _ = writeln!(out, "Backlog: {backlog}");
        let _ = writeln!(
            out,
            "Pending: {} conversations waiting to go quiet",
            status.pending_conversations
        );
        let _ = writeln!(out, "Embedded: {} passages since start", status.embedded);
        if let Some(error) = status.last_error {
            let _ = writeln!(
                out,
                "Last error: {} ({})",
                error.message,
                ago(Some(error.at_ms))
            );
        }
    }
    let _ = writeln!(out, "Log: {}", paths.log.display());
    for line in log::tail(&paths.log, LOG_LINES).unwrap_or_default() {
        let _ = writeln!(out, "  {line}");
    }
    out
}

/// Lines of the log `daemon status` shows.
const LOG_LINES: usize = 5;

fn ago(now_ms: i64, ms: i64) -> String {
    let secs = (now_ms - ms).max(0) / 1000;
    match secs {
        0..60 => format!("{secs} s ago"),
        60..3600 => format!("{} min ago", secs / 60),
        3600..86_400 => format!("{} h {} min ago", secs / 3600, secs % 3600 / 60),
        _ => format!("{} days ago", secs / 86_400),
    }
}

/// Where the daemon writes the embedding cache.
fn semantic_dir() -> Option<PathBuf> {
    crate::semantic::cache::semantic_cache_dir()
}

/// Whether `root` holds a generation of the store.
fn store_exists(root: &Path) -> bool {
    crate::store::generation::Generation::current_name(root).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_describes_a_stopped_daemon_and_its_last_state() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::in_dir(dir.path().join("daemon"));
        let out = describe(&paths, false, 0);
        assert!(out.starts_with("Daemon: not running\nConfig: disabled"));

        let binary = BinaryIdentity::current().unwrap();
        let mut status = Status::new(42, binary, 0);
        status.exit_reason = Some("no launch for 3 days".into());
        status.backlog = Some(12);
        status.last_full_pass_ms = Some(1_000);
        status.last_error = Some(status::StatusError {
            at_ms: 0,
            message: "boom".into(),
        });
        status.write(&paths.status).unwrap();
        let out = describe(&paths, true, 7_201_000);
        assert!(out.contains("not running (last exit: no launch for 3 days)"));
        assert!(out.contains("Full pass: 2 h 0 min ago"));
        assert!(out.contains("Backlog: 12 passages"));
        assert!(out.contains("Last error: boom (2 h 0 min ago)"));
        assert!(!out.contains("State:"));
    }

    #[test]
    fn a_launch_is_recorded_in_the_launched_file() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::in_dir(dir.path().join("daemon"));
        assert_eq!(paths.last_launch(), None);
        paths.touch_launched().unwrap();
        let first = paths.last_launch().unwrap();
        std::fs::File::options()
            .write(true)
            .open(&paths.launched)
            .unwrap()
            .set_modified(first - Duration::from_secs(3600))
            .unwrap();
        paths.touch_launched().unwrap();
        assert!(paths.last_launch().unwrap() >= first);
    }
}
