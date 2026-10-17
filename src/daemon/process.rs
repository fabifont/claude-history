//! Process plumbing of the daemon: starting it detached from the launching
//! terminal, lowering its priority, and turning termination signals into
//! messages.

use std::io;
use std::path::Path;
use std::process::{Command, Stdio};

/// Starts `exe args` as a daemon: in a new session with no controlling
/// terminal, with stdio on `/dev/null` and `/` as its working directory,
/// and reparented to init, so it survives the launching terminal and is
/// never left a zombie of the launching process.
#[cfg(unix)]
pub fn spawn_detached(exe: &Path, args: &[&str]) -> io::Result<()> {
    use std::os::unix::process::CommandExt;
    let mut command = Command::new(exe);
    command
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    // SAFETY: the closure runs in the forked child before exec and calls
    // only async-signal-safe functions. Its first fork leaves an
    // intermediate process that exits at once; the grandchild, which goes
    // on to exec, belongs to no session of the launching terminal.
    unsafe {
        command.pre_exec(|| {
            match libc::fork() {
                -1 => return Err(io::Error::last_os_error()),
                0 => {}
                _ => libc::_exit(0),
            }
            libc::setsid();
            libc::chdir(c"/".as_ptr());
            Ok(())
        });
    }
    // The intermediate process exits at once; reap it.
    command.spawn()?.wait()?;
    Ok(())
}

#[cfg(not(unix))]
pub fn spawn_detached(_exe: &Path, _args: &[&str]) -> io::Result<()> {
    Err(io::Error::other(
        "the background daemon is supported on macOS and Linux",
    ))
}

/// Lowers the scheduling priority of the calling thread, and of the threads
/// it starts afterwards where they inherit it: nice 10, and the utility QoS
/// class on macOS.
pub fn lower_priority() {
    #[cfg(unix)]
    // SAFETY: changes the scheduling priority of this process (on Linux,
    // of the calling thread, which threads started later inherit).
    unsafe {
        libc::setpriority(libc::PRIO_PROCESS, 0, 10);
    }
    #[cfg(target_os = "macos")]
    // SAFETY: changes only the calling thread's QoS class.
    unsafe {
        libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_UTILITY, 0);
    }
}

/// Makes the global rayon pool one thread at lowered priority. Threads on
/// macOS do not inherit the QoS class of the thread starting them, so the
/// pool's thread lowers its own.
pub fn single_low_priority_pool() {
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .thread_name(|_| "daemon-worker".to_owned())
        .start_handler(|_| lower_priority())
        .build_global();
}

/// Signals the daemon handles: termination, and `WAKE`.
#[cfg(unix)]
const SIGNALS: [libc::c_int; 4] = [libc::SIGTERM, libc::SIGINT, libc::SIGHUP, WAKE];

/// The signal a launch sends a running daemon to have it check for work.
#[cfg(unix)]
pub const WAKE: i32 = libc::SIGUSR1;
#[cfg(not(unix))]
pub const WAKE: i32 = 10;

/// Blocks the termination signals in the calling thread, and so in every
/// thread it starts afterwards, and starts a thread that passes each one
/// that arrives to `on_signal`. Call it before starting other threads.
#[cfg(unix)]
pub fn forward_signals(on_signal: impl Fn(i32) + Send + 'static) -> io::Result<()> {
    // SAFETY: builds a signal set and sets the calling thread's mask.
    let set = unsafe {
        let mut set = std::mem::zeroed::<libc::sigset_t>();
        libc::sigemptyset(&mut set);
        for signal in SIGNALS {
            libc::sigaddset(&mut set, signal);
        }
        if libc::pthread_sigmask(libc::SIG_BLOCK, &set, std::ptr::null_mut()) != 0 {
            return Err(io::Error::last_os_error());
        }
        set
    };
    std::thread::Builder::new()
        .name("daemon-signals".to_owned())
        .spawn(move || {
            loop {
                let mut signal = 0;
                // SAFETY: waits for one of the blocked signals in `set`.
                if unsafe { libc::sigwait(&set, &mut signal) } != 0 {
                    return;
                }
                on_signal(signal);
            }
        })?;
    Ok(())
}

#[cfg(not(unix))]
pub fn forward_signals(_on_signal: impl Fn(i32) + Send + 'static) -> io::Result<()> {
    Ok(())
}

/// Asks process `pid` to stop.
pub fn terminate(pid: u32) -> io::Result<()> {
    #[cfg(unix)]
    return signal(pid, libc::SIGTERM);
    #[cfg(not(unix))]
    return signal(pid, 15);
}

/// Sends `WAKE` to process `pid`.
pub fn wake(pid: u32) -> io::Result<()> {
    signal(pid, WAKE)
}

#[cfg(unix)]
fn signal(pid: u32, signal: i32) -> io::Result<()> {
    let pid = libc::pid_t::try_from(pid).map_err(io::Error::other)?;
    // SAFETY: sends `signal` to `pid`.
    if unsafe { libc::kill(pid, signal) } != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

#[cfg(not(unix))]
fn signal(_pid: u32, _signal: i32) -> io::Result<()> {
    Err(io::Error::other(
        "the background daemon is supported on macOS and Linux",
    ))
}

/// Resident memory of process `pid` in bytes.
#[cfg(target_os = "macos")]
pub fn resident_bytes(pid: u32) -> Option<u64> {
    let mut info = std::mem::MaybeUninit::<libc::rusage_info_v4>::uninit();
    // SAFETY: proc_pid_rusage fully initializes `info` when it returns 0.
    let info = unsafe {
        if libc::proc_pid_rusage(
            libc::pid_t::try_from(pid).ok()?,
            libc::RUSAGE_INFO_V4,
            info.as_mut_ptr().cast(),
        ) != 0
        {
            return None;
        }
        info.assume_init()
    };
    Some(info.ri_resident_size)
}

#[cfg(target_os = "linux")]
pub fn resident_bytes(pid: u32) -> Option<u64> {
    let statm = std::fs::read_to_string(format!("/proc/{pid}/statm")).ok()?;
    let pages: u64 = statm.split_whitespace().nth(1)?.parse().ok()?;
    // SAFETY: sysconf has no preconditions.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    Some(pages * u64::try_from(page).ok()?)
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
pub fn resident_bytes(_pid: u32) -> Option<u64> {
    None
}
