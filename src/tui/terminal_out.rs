//! The terminal the TUI draws on, kept apart from the process's stderr.
//!
//! The TUI draws on the terminal that stderr pointed at when it started.
//! While it runs, file descriptor 2 points at a log file instead, so text that
//! native libraries (such as ONNX Runtime warnings) or child processes write
//! to stderr lands in `~/.local/state/claude-history/tui-stderr.log` rather
//! than over the screen.

use std::io::{self, Write};
#[cfg(unix)]
use std::sync::atomic::{AtomicI32, Ordering};

/// The terminal's file descriptor while stderr is redirected, or -1.
#[cfg(unix)]
static TERMINAL_FD: AtomicI32 = AtomicI32::new(-1);

/// Writes to the terminal: the saved descriptor while stderr is redirected,
/// stderr otherwise.
pub struct TerminalOut;

impl Write for TerminalOut {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        #[cfg(unix)]
        {
            let fd = TERMINAL_FD.load(Ordering::Acquire);
            if fd >= 0 {
                // SAFETY: `fd` stays open until the redirect that set it is
                // dropped, which clears it first; the TUI draws and drops the
                // redirect on the same thread.
                let written = unsafe { libc::write(fd, buf.as_ptr().cast(), buf.len()) };
                return if written < 0 {
                    Err(io::Error::last_os_error())
                } else {
                    Ok(written as usize)
                };
            }
        }
        io::stderr().write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        io::stderr().flush()
    }
}

/// Points stderr at the log file until dropped, keeping the terminal for
/// `TerminalOut`. Does nothing when stderr cannot be redirected.
pub struct StderrRedirect {
    #[cfg(unix)]
    saved: Option<i32>,
}

impl StderrRedirect {
    pub fn new() -> Self {
        #[cfg(unix)]
        {
            Self {
                saved: redirect_stderr(),
            }
        }
        #[cfg(not(unix))]
        {
            Self {}
        }
    }
}

#[cfg(unix)]
fn redirect_stderr() -> Option<i32> {
    use std::os::fd::AsRawFd;

    let log = crate::debug_log::create_state_file("tui-stderr.log").or_else(|| {
        std::fs::OpenOptions::new()
            .write(true)
            .open("/dev/null")
            .ok()
    })?;
    let _ = io::stderr().flush();
    // SAFETY: plain descriptor calls; every descriptor opened here is closed
    // on failure or by `Drop`.
    unsafe {
        let saved = libc::fcntl(2, libc::F_DUPFD_CLOEXEC, 3);
        if saved < 0 {
            return None;
        }
        if libc::dup2(log.as_raw_fd(), 2) < 0 {
            libc::close(saved);
            return None;
        }
        TERMINAL_FD.store(saved, Ordering::Release);
        Some(saved)
    }
}

impl Drop for StderrRedirect {
    fn drop(&mut self) {
        #[cfg(unix)]
        if let Some(saved) = self.saved.take() {
            TERMINAL_FD.store(-1, Ordering::Release);
            // SAFETY: `saved` is the descriptor `redirect_stderr` duplicated.
            unsafe {
                libc::dup2(saved, 2);
                libc::close(saved);
            }
        }
    }
}
