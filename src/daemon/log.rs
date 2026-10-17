//! The daemon's log: timestamped lines appended to a file that is rotated
//! to `<name>.1` once it passes a size cap, so the two files together never
//! hold much more than twice the cap.

use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

/// Size past which the log is rotated.
pub const LOG_CAP: u64 = 1 << 20;

pub struct Log {
    path: PathBuf,
    cap: u64,
    file: Option<File>,
    size: u64,
    /// Whether the process's stderr follows the log file.
    capture_stderr: bool,
}

impl Log {
    /// The log at `path`, which also receives what the process writes to
    /// stderr, such as panic messages and native library output.
    pub fn open(path: &Path) -> Self {
        Self::with_cap(path, LOG_CAP, true)
    }

    fn with_cap(path: &Path, cap: u64, capture_stderr: bool) -> Self {
        let mut log = Self {
            path: path.to_path_buf(),
            cap,
            file: None,
            size: 0,
            capture_stderr,
        };
        log.reopen();
        log
    }

    fn reopen(&mut self) {
        if let Some(dir) = self.path.parent() {
            let _ = fs::create_dir_all(dir);
        }
        self.file = File::options()
            .create(true)
            .append(true)
            .open(&self.path)
            .ok();
        self.size = fs::metadata(&self.path).map_or(0, |metadata| metadata.len());
        #[cfg(unix)]
        if self.capture_stderr
            && let Some(file) = &self.file
        {
            use std::os::fd::AsRawFd;
            // SAFETY: points the process's stderr at the open log file.
            unsafe {
                libc::dup2(file.as_raw_fd(), libc::STDERR_FILENO);
            }
        }
    }

    /// Appends `message` as one line with the local time.
    pub fn line(&mut self, message: &str) {
        if self.size >= self.cap {
            self.rotate();
        }
        let line = format!(
            "{} {message}\n",
            chrono::Local::now().format("%Y-%m-%dT%H:%M:%S%.3f%z")
        );
        if let Some(file) = &mut self.file
            && file.write_all(line.as_bytes()).is_ok()
        {
            self.size += line.len() as u64;
        }
    }

    fn rotate(&mut self) {
        self.file = None;
        let _ = fs::rename(&self.path, rotated(&self.path));
        self.reopen();
    }
}

/// Where the previous log goes when the log is rotated.
pub fn rotated(path: &Path) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(".1");
    path.with_file_name(name)
}

/// Lines of the log at `path`, the last `count` of them.
pub fn tail(path: &Path, count: usize) -> io::Result<Vec<String>> {
    let text = fs::read_to_string(path)?;
    let lines: Vec<&str> = text.lines().collect();
    Ok(lines[lines.len().saturating_sub(count)..]
        .iter()
        .map(|line| (*line).to_owned())
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_log_rotates_past_its_cap_and_keeps_one_old_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("daemon.log");
        // Tests keep the test runner's stderr.
        let mut log = Log::with_cap(&path, 200, false);
        for i in 0..40 {
            log.line(&format!("message {i}"));
        }
        let current = fs::metadata(&path).unwrap().len();
        let old = fs::metadata(rotated(&path)).unwrap().len();
        assert!(current < 300 && old < 300, "{current} {old}");
        let last = tail(&path, 1).unwrap();
        assert!(last[0].ends_with("message 39"), "{last:?}");
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 2);
    }
}
