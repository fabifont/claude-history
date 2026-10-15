//! Startup timing marks, printed to stderr when benchmarking is enabled.

use std::io::Write;
use std::path::PathBuf;
use std::sync::OnceLock;
use std::time::Instant;

static START: OnceLock<Instant> = OnceLock::new();
static SINK: OnceLock<Option<Sink>> = OnceLock::new();

enum Sink {
    Stderr,
    File(PathBuf),
}

/// Starts the clock. Marks go to stderr when `force` is true or
/// `CLAUDE_HISTORY_BENCH=1`. Any other value of `CLAUDE_HISTORY_BENCH` is a
/// file path that marks are appended to, which keeps them off the TUI's
/// stderr.
pub(crate) fn start(force: bool) {
    START.get_or_init(Instant::now);
    SINK.get_or_init(|| {
        if force {
            return Some(Sink::Stderr);
        }
        match std::env::var_os("CLAUDE_HISTORY_BENCH") {
            None => None,
            Some(value) if value.is_empty() || value == "1" => Some(Sink::Stderr),
            Some(value) => Some(Sink::File(value.into())),
        }
    });
}

fn emit(line: std::fmt::Arguments) {
    match SINK.get() {
        Some(Some(Sink::Stderr)) => eprintln!("{line}"),
        Some(Some(Sink::File(path))) => {
            if let Ok(mut file) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
            {
                let _ = writeln!(file, "{line}");
            }
        }
        _ => {}
    }
}

/// Reports the elapsed time since `start` under `label`.
pub(crate) fn mark(label: &str) {
    let elapsed = START.get_or_init(Instant::now).elapsed();
    emit(format_args!(
        "bench {label} {:.1}ms",
        elapsed.as_secs_f64() * 1e3
    ));
}

/// Reports the process's peak resident set size.
pub(crate) fn mark_peak_rss() {
    if !matches!(SINK.get(), Some(Some(_))) {
        return;
    }
    if let Some(bytes) = peak_rss_bytes() {
        emit(format_args!(
            "bench peak_rss {:.1}MB",
            bytes as f64 / (1024.0 * 1024.0)
        ));
    }
}

#[cfg(unix)]
fn peak_rss_bytes() -> Option<u64> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    // SAFETY: getrusage fully initializes `usage` when it returns 0.
    let usage = unsafe {
        if libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) != 0 {
            return None;
        }
        usage.assume_init()
    };
    let max_rss = u64::try_from(usage.ru_maxrss).ok()?;
    // macOS reports bytes, Linux reports kilobytes.
    if cfg!(target_os = "macos") {
        Some(max_rss)
    } else {
        Some(max_rss * 1024)
    }
}

#[cfg(not(unix))]
fn peak_rss_bytes() -> Option<u64> {
    None
}
