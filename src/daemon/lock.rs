//! Single-instance lock of the daemon.
//!
//! The lock is an advisory `flock` on a file in the daemon directory, held
//! for the daemon's whole life. The kernel releases it when the process
//! exits, however it exits, so a crashed daemon never keeps a new one from
//! starting; the file itself carries no state and is never removed.

use std::fs::{self, File, TryLockError};
use std::io;
use std::path::Path;
use std::time::{Duration, Instant};

/// The held lock; dropping it releases it.
#[derive(Debug)]
pub struct InstanceLock {
    _file: File,
}

fn open(path: &Path) -> io::Result<File> {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    File::options()
        .create(true)
        .truncate(false)
        .write(true)
        .open(path)
}

impl InstanceLock {
    /// Takes the lock at `path` without waiting; `None` when another
    /// process holds it.
    pub fn try_acquire(path: &Path) -> io::Result<Option<Self>> {
        let file = open(path)?;
        match file.try_lock() {
            Ok(()) => Ok(Some(Self { _file: file })),
            Err(TryLockError::WouldBlock) => Ok(None),
            Err(TryLockError::Error(error)) => Err(error),
        }
    }

    /// Takes the lock at `path`, trying again until `wait` has passed: a
    /// daemon being replaced holds it while it finishes its work, and a
    /// launch checking whether a daemon runs holds it for an instant.
    pub fn acquire_within(path: &Path, wait: Duration) -> io::Result<Option<Self>> {
        let deadline = Instant::now() + wait;
        loop {
            if let Some(lock) = Self::try_acquire(path)? {
                return Ok(Some(lock));
            }
            if Instant::now() >= deadline {
                return Ok(None);
            }
            std::thread::sleep(Duration::from_millis(100));
        }
    }
}

/// Whether a process holds the lock at `path`. Checking takes a shared lock
/// for an instant when nobody holds it.
pub fn is_held(path: &Path) -> bool {
    let Ok(file) = File::open(path) else {
        return false;
    };
    match file.try_lock_shared() {
        Ok(()) => false,
        Err(TryLockError::WouldBlock) => true,
        Err(TryLockError::Error(_)) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_holder_at_a_time_and_released_on_drop() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("daemon").join("daemon.lock");
        assert!(!is_held(&path));
        let lock = InstanceLock::try_acquire(&path).unwrap().unwrap();
        assert!(is_held(&path));
        assert!(InstanceLock::try_acquire(&path).unwrap().is_none());
        let waited = Instant::now();
        assert!(
            InstanceLock::acquire_within(&path, Duration::from_millis(250))
                .unwrap()
                .is_none()
        );
        assert!(waited.elapsed() >= Duration::from_millis(250));
        drop(lock);
        assert!(!is_held(&path));
        assert!(InstanceLock::try_acquire(&path).unwrap().is_some());
    }

    #[test]
    fn a_lock_file_left_behind_does_not_hold_the_lock() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("daemon.lock");
        fs::write(&path, "4242\n").unwrap();
        assert!(!is_held(&path));
        assert!(InstanceLock::try_acquire(&path).unwrap().is_some());
    }

    /// A process killed while holding the lock releases it: the kernel
    /// drops `flock` locks with the last descriptor.
    #[cfg(unix)]
    #[test]
    fn a_killed_holder_releases_the_lock() {
        use std::ffi::CString;
        use std::os::unix::ffi::OsStrExt;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("daemon.lock");
        let c_path = CString::new(path.as_os_str().as_bytes()).unwrap();
        let mut pipe = [0; 2];
        // SAFETY: creates a pipe into the two descriptors of `pipe`.
        assert_eq!(unsafe { libc::pipe(pipe.as_mut_ptr()) }, 0);
        // SAFETY: the child calls only async-signal-safe functions (open,
        // flock, write, pause, _exit) before it is killed.
        let pid = unsafe { libc::fork() };
        assert!(pid >= 0);
        if pid == 0 {
            unsafe {
                let fd = libc::open(c_path.as_ptr(), libc::O_CREAT | libc::O_RDWR, 0o644);
                if fd < 0 || libc::flock(fd, libc::LOCK_EX) != 0 {
                    libc::_exit(1);
                }
                libc::write(pipe[1], b"x".as_ptr().cast(), 1);
                loop {
                    libc::pause();
                }
            }
        }
        let mut byte = 0u8;
        // SAFETY: reads one byte from the pipe the child writes once it
        // holds the lock.
        unsafe {
            libc::close(pipe[1]);
            assert_eq!(libc::read(pipe[0], (&mut byte as *mut u8).cast(), 1), 1);
            libc::close(pipe[0]);
        }
        assert!(is_held(&path));
        assert!(InstanceLock::try_acquire(&path).unwrap().is_none());
        // SAFETY: kills and reaps the child forked above.
        unsafe {
            libc::kill(pid, libc::SIGKILL);
            libc::waitpid(pid, std::ptr::null_mut(), 0);
        }
        assert!(!is_held(&path));
        assert!(InstanceLock::try_acquire(&path).unwrap().is_some());
    }
}
