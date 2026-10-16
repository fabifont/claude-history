//! Progress of a reconcile pass, shared with the TUIs reading the store.
//!
//! `PROGRESS` in the store root holds one line while a pass that parses at
//! least `MIN_FILES` files runs:
//!
//! ```text
//! indexing <pid> <parsed> <to parse>
//! ```
//!
//! It is rewritten at most every `INTERVAL`, and only when the count
//! changed, by a thread of the reconcile process, and removed when the pass
//! ends. A pass that parses fewer files never writes it. A pass that fails leaves `failed <message>` for the TUI that started
//! it. The next pass removes whatever an earlier process left.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::thread::JoinHandle;
use std::time::Duration;

const FILE: &str = "PROGRESS";
const TMP: &str = "PROGRESS.tmp";
/// Fewest files a pass parses for its progress to be shown. Warm passes
/// parse the few sessions that changed since the last one; they leave the
/// store nearly complete, and the time they take depends more on the scan
/// than on the files parsed.
pub const MIN_FILES: usize = 50;
/// Shortest time between two writes.
const INTERVAL: Duration = Duration::from_millis(250);

/// Files parsed out of those a running pass parses.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Indexing {
    pub done: usize,
    pub total: usize,
}

/// What `PROGRESS` says.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Status {
    Indexing(Indexing),
    Failed(String),
}

/// Reads `PROGRESS` under `root`. A progress line of a process that has
/// exited is `None`.
pub fn read(root: &Path) -> Option<Status> {
    let text = fs::read_to_string(root.join(FILE)).ok()?;
    let line = text.lines().next()?;
    if let Some(message) = line.strip_prefix("failed ") {
        return Some(Status::Failed(message.to_owned()));
    }
    let mut fields = line.strip_prefix("indexing ")?.split(' ');
    let pid: u32 = fields.next()?.parse().ok()?;
    let done = fields.next()?.parse().ok()?;
    let total = fields.next()?.parse().ok()?;
    alive(pid).then_some(Status::Indexing(Indexing { done, total }))
}

/// Removes `PROGRESS` under `root`, if there is one.
pub fn clear(root: &Path) {
    let _ = fs::remove_file(root.join(FILE));
}

/// Leaves `message` for the TUI that started the failed pass.
pub fn record_failure(root: &Path, message: &str) {
    let line = message.lines().next().unwrap_or_default();
    let _ = write(root, &format!("failed {line}\n"));
}

fn write(root: &Path, line: &str) -> io::Result<()> {
    let tmp = root.join(TMP);
    fs::write(&tmp, line)?;
    fs::rename(&tmp, root.join(FILE))
}

#[cfg(unix)]
fn alive(pid: u32) -> bool {
    let Ok(pid) = libc::pid_t::try_from(pid) else {
        return false;
    };
    // SAFETY: signal 0 only checks that the process exists.
    pid > 0
        && (unsafe { libc::kill(pid, 0) } == 0
            || io::Error::last_os_error().raw_os_error() == Some(libc::EPERM))
}

#[cfg(not(unix))]
fn alive(_pid: u32) -> bool {
    true
}

/// Writes the progress of one pass from a thread of its own until dropped,
/// then removes what it wrote.
pub struct Reporter {
    done: Arc<AtomicUsize>,
    stop: Option<Sender<()>>,
    thread: Option<JoinHandle<()>>,
}

impl Reporter {
    /// Starts reporting a pass that parses `total` files, or `None` when
    /// there are too few to show.
    pub fn start(root: &Path, total: usize) -> Option<Self> {
        if total < MIN_FILES {
            return None;
        }
        let done = Arc::new(AtomicUsize::new(0));
        let (stop, stopped) = mpsc::channel::<()>();
        let root: PathBuf = root.to_path_buf();
        let counter = done.clone();
        let thread = std::thread::Builder::new()
            .name("reconcile-progress".into())
            .spawn(move || {
                let pid = std::process::id();
                let mut written = None;
                loop {
                    let parsed = counter.load(Ordering::Relaxed).min(total);
                    if written != Some(parsed)
                        && write(&root, &format!("indexing {pid} {parsed} {total}\n")).is_ok()
                    {
                        written = Some(parsed);
                    }
                    if stopped.recv_timeout(INTERVAL) != Err(RecvTimeoutError::Timeout) {
                        break;
                    }
                }
                if written.is_some() {
                    clear(&root);
                }
            })
            .ok();
        Some(Self {
            done,
            stop: Some(stop),
            thread,
        })
    }

    /// Counts files as they are parsed.
    pub fn counter(&self) -> &AtomicUsize {
        &self.done
    }
}

impl Drop for Reporter {
    fn drop(&mut self) {
        drop(self.stop.take());
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_pass_with_few_files_writes_nothing() {
        let dir = tempfile::tempdir().unwrap();
        assert!(Reporter::start(dir.path(), MIN_FILES - 1).is_none());
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 0);
    }

    #[test]
    fn a_large_pass_shows_its_progress_and_removes_it_when_done() {
        let dir = tempfile::tempdir().unwrap();
        let reporter = Reporter::start(dir.path(), MIN_FILES).unwrap();
        reporter.counter().fetch_add(3, Ordering::Relaxed);
        let expected = Status::Indexing(Indexing {
            done: 3,
            total: MIN_FILES,
        });
        let shown = (0..200).any(|_| {
            std::thread::sleep(Duration::from_millis(10));
            read(dir.path()) == Some(expected.clone())
        });
        assert!(shown);
        drop(reporter);
        assert_eq!(read(dir.path()), None);
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 0);
    }

    #[test]
    fn progress_of_an_exited_process_is_ignored() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "indexing 2147483600 1 2\n").unwrap();
        assert_eq!(read(dir.path()), None);
        let pid = std::process::id();
        write(dir.path(), &format!("indexing {pid} 1 2\n")).unwrap();
        assert_eq!(
            read(dir.path()),
            Some(Status::Indexing(Indexing { done: 1, total: 2 }))
        );
    }

    #[test]
    fn a_failure_is_kept_until_cleared() {
        let dir = tempfile::tempdir().unwrap();
        record_failure(dir.path(), "disk full\nmore detail");
        assert_eq!(
            read(dir.path()),
            Some(Status::Failed("disk full".to_owned()))
        );
        clear(dir.path());
        assert_eq!(read(dir.path()), None);
    }
}
