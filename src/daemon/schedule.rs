//! When the daemon does what. Everything here takes the time as an
//! argument, so the scheduling is tested without waiting.

use std::collections::HashSet;
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime};

/// Quick pass interval while sessions are changing.
pub const QUICK_ACTIVE: Duration = Duration::from_secs(30);
/// Quick pass interval after the first pass without changes.
pub const QUICK_BASE: Duration = Duration::from_secs(60);
/// Longest quick pass interval, reached by doubling while nothing changes.
pub const QUICK_IDLE_MAX: Duration = Duration::from_secs(5 * 60);
/// Interval of the full pass, which lists every session file.
pub const FULL_INTERVAL: Duration = Duration::from_secs(20 * 60);
/// Delay before the first semantic sweep, so a launch that starts the
/// daemon is not competing with it.
pub const SWEEP_DELAY: Duration = Duration::from_secs(30);
/// Interval of the semantic sweep, which checks every conversation for
/// passages without embeddings.
pub const SWEEP_INTERVAL: Duration = Duration::from_secs(6 * 60 * 60);
/// Delay before an interrupted or refused embedding job runs again.
pub const JOB_RETRY: Duration = Duration::from_secs(2 * 60);
/// How long a session must go unmodified before it is embedded: an agent
/// appends to its session all the time while it works.
pub const QUIET: Duration = Duration::from_secs(60);
/// The daemon exits after this long without a launch.
pub const IDLE_EXIT: Duration = Duration::from_secs(3 * 24 * 60 * 60);

/// What is due at one moment.
#[derive(Debug, Default, Eq, PartialEq)]
pub struct Due {
    pub quick: bool,
    pub full: bool,
    pub sweep: bool,
}

#[derive(Debug)]
pub struct Schedule {
    quick_interval: Duration,
    next_quick: Instant,
    next_full: Instant,
    next_sweep: Instant,
    /// Earliest time another embedding job may start.
    next_job: Instant,
}

impl Schedule {
    /// A schedule starting at `now` with a full pass.
    pub fn new(now: Instant) -> Self {
        Self {
            quick_interval: QUICK_BASE,
            next_quick: now + QUICK_BASE,
            next_full: now,
            next_sweep: now + SWEEP_DELAY,
            next_job: now,
        }
    }

    pub fn quick_interval(&self) -> Duration {
        self.quick_interval
    }

    pub fn due(&self, now: Instant) -> Due {
        Due {
            quick: now >= self.next_quick,
            full: now >= self.next_full,
            sweep: now >= self.next_sweep,
        }
    }

    /// Whether an embedding job may start at `now`.
    pub fn job_allowed(&self, now: Instant) -> bool {
        now >= self.next_job
    }

    /// Records a quick pass at `now`: the next comes sooner while files
    /// change and later while they do not.
    pub fn quick_done(&mut self, now: Instant, changed: bool) {
        self.quick_interval = if changed {
            QUICK_ACTIVE
        } else if self.quick_interval < QUICK_BASE {
            QUICK_BASE
        } else {
            (self.quick_interval * 2).min(QUICK_IDLE_MAX)
        };
        self.next_quick = now + self.quick_interval;
    }

    /// Records a full pass at `now`, which counts as a quick pass too.
    pub fn full_done(&mut self, now: Instant, changed: bool) {
        self.next_full = now + FULL_INTERVAL;
        self.quick_done(now, changed);
    }

    /// Records the end of a sweep: the next after the interval when it
    /// finished, soon when it was interrupted.
    pub fn sweep_done(&mut self, now: Instant, finished: bool) {
        self.next_sweep = now + if finished { SWEEP_INTERVAL } else { JOB_RETRY };
    }

    /// Asks for a sweep at `now`, as after the store was rebuilt.
    pub fn sweep_now(&mut self, now: Instant) {
        self.next_sweep = self.next_sweep.min(now);
    }

    /// Brings the next full pass forward to `QUICK_ACTIVE` after `now`, as
    /// when a pass found another process publishing.
    pub fn full_soon(&mut self, now: Instant) {
        self.next_full = self.next_full.min(now + QUICK_ACTIVE);
    }

    /// Holds embedding jobs back until `JOB_RETRY` after `now`.
    pub fn job_refused(&mut self, now: Instant) {
        self.next_job = now + JOB_RETRY;
    }

    /// When the daemon next has something to do, given the time pending
    /// work becomes ready, if any. While `jobs_blocked`, as while a job
    /// runs or the store is not current, only passes wake it.
    pub fn next_wake(&self, ready_at: Option<Instant>, jobs_blocked: bool) -> Instant {
        let mut wake = self.next_quick.min(self.next_full);
        if !jobs_blocked {
            wake = wake.min(self.next_sweep.max(self.next_job));
            if let Some(ready_at) = ready_at {
                wake = wake.min(ready_at.max(self.next_job));
            }
        }
        wake
    }
}

/// Conversations that changed and wait to go quiet before they are
/// embedded.
#[derive(Debug, Default)]
pub struct Pending {
    paths: HashSet<PathBuf>,
}

impl Pending {
    pub fn add(&mut self, paths: impl IntoIterator<Item = PathBuf>) {
        self.paths.extend(paths);
    }

    pub fn len(&self) -> usize {
        self.paths.len()
    }

    /// Takes the conversations unmodified for `QUIET` at `now`, given each
    /// one's modification time (`None`: removed, and dropped). Returns them
    /// with how long until the next one goes quiet.
    pub fn take_quiet(
        &mut self,
        now: SystemTime,
        modified: impl Fn(&PathBuf) -> Option<SystemTime>,
    ) -> (Vec<PathBuf>, Option<Duration>) {
        let mut ready = Vec::new();
        let mut next: Option<Duration> = None;
        self.paths.retain(|path| {
            let Some(modified) = modified(path) else {
                return false;
            };
            let quiet_for = now.duration_since(modified).unwrap_or(Duration::ZERO);
            if quiet_for >= QUIET {
                ready.push(path.clone());
                return false;
            }
            let wait = QUIET - quiet_for;
            next = Some(next.map_or(wait, |next| next.min(wait)));
            true
        });
        ready.sort();
        (ready, next)
    }
}

/// Why the daemon exits.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ExitReason {
    /// `daemon stop` or another termination signal.
    Stopped,
    /// The executable was replaced or upgraded; the next launch starts it.
    BinaryChanged,
    /// No launch for `IDLE_EXIT`.
    Idle,
    /// The configuration no longer enables the daemon.
    Disabled,
    /// Something it cannot recover from.
    Failed(String),
}

impl std::fmt::Display for ExitReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Stopped => f.write_str("stopped"),
            Self::BinaryChanged => f.write_str("the claude-history binary changed"),
            Self::Idle => write!(
                f,
                "no launch for {} days",
                IDLE_EXIT.as_secs() / (24 * 60 * 60)
            ),
            Self::Disabled => f.write_str("disabled in the config"),
            Self::Failed(message) => write!(f, "failed: {message}"),
        }
    }
}

/// Whether the daemon has gone `IDLE_EXIT` without a launch at `now`. A
/// launch never recorded counts from `started`.
pub fn idle(now: SystemTime, last_launch: Option<SystemTime>, started: SystemTime) -> bool {
    let since = last_launch.map_or(started, |launch| launch.max(started));
    now.duration_since(since)
        .is_ok_and(|elapsed| elapsed >= IDLE_EXIT)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_new_schedule_starts_with_a_full_pass_and_delays_the_sweep() {
        let start = Instant::now();
        let schedule = Schedule::new(start);
        assert_eq!(
            schedule.due(start),
            Due {
                quick: false,
                full: true,
                sweep: false
            }
        );
        assert!(schedule.due(start + SWEEP_DELAY).sweep);
    }

    #[test]
    fn quick_passes_speed_up_while_files_change_and_back_off_when_idle() {
        let mut now = Instant::now();
        let mut schedule = Schedule::new(now);
        schedule.full_done(now, false);
        let mut intervals = Vec::new();
        for changed in [true, true, false, false, false, false, false, true, false] {
            now += schedule.quick_interval();
            assert!(schedule.due(now).quick);
            schedule.quick_done(now, changed);
            intervals.push(schedule.quick_interval().as_secs());
            assert!(!schedule.due(now + schedule.quick_interval() / 2).quick);
        }
        assert_eq!(intervals, [30, 30, 60, 120, 240, 300, 300, 30, 60]);
    }

    #[test]
    fn full_passes_come_every_interval() {
        let start = Instant::now();
        let mut schedule = Schedule::new(start);
        schedule.full_done(start, false);
        assert!(
            !schedule
                .due(start + FULL_INTERVAL - Duration::from_secs(1))
                .full
        );
        let next = start + FULL_INTERVAL;
        assert!(schedule.due(next).full);
        schedule.full_done(next, false);
        assert!(!schedule.due(next + Duration::from_secs(1)).full);
    }

    #[test]
    fn sweeps_repeat_after_finishing_and_retry_soon_when_interrupted() {
        let start = Instant::now();
        let mut schedule = Schedule::new(start);
        let at = start + SWEEP_DELAY;
        schedule.sweep_done(at, false);
        assert!(schedule.due(at + JOB_RETRY).sweep);
        schedule.sweep_done(at, true);
        assert!(!schedule.due(at + JOB_RETRY).sweep);
        assert!(schedule.due(at + SWEEP_INTERVAL).sweep);
        schedule.sweep_now(at);
        assert!(schedule.due(at).sweep);
    }

    #[test]
    fn a_busy_store_brings_the_full_pass_forward() {
        let start = Instant::now();
        let mut schedule = Schedule::new(start);
        schedule.full_done(start, false);
        schedule.full_soon(start);
        assert!(!schedule.due(start + QUICK_ACTIVE / 2).full);
        assert!(schedule.due(start + QUICK_ACTIVE).full);
    }

    #[test]
    fn the_daemon_wakes_for_the_earliest_work() {
        let start = Instant::now();
        let mut schedule = Schedule::new(start);
        schedule.full_done(start, false);
        schedule.sweep_done(start, true);
        let quick = start + QUICK_BASE * 2;
        assert_eq!(schedule.next_wake(None, false), quick);
        let ready = start + Duration::from_secs(10);
        assert_eq!(schedule.next_wake(Some(ready), false), ready);
        assert_eq!(schedule.next_wake(Some(ready), true), quick);
        // An overdue sweep that cannot start does not wake the daemon.
        let overdue = start + SWEEP_INTERVAL * 2;
        assert!(schedule.due(overdue).sweep);
        assert_eq!(schedule.next_wake(None, true), quick);
        schedule.job_refused(start);
        assert_eq!(schedule.next_wake(Some(ready), false), start + JOB_RETRY);
        assert!(!schedule.job_allowed(ready));
        assert!(schedule.job_allowed(start + JOB_RETRY));
    }

    #[test]
    fn conversations_are_ready_once_quiet() {
        let now = SystemTime::now();
        let mut pending = Pending::default();
        let a = PathBuf::from("/a.jsonl");
        let b = PathBuf::from("/b.jsonl");
        let gone = PathBuf::from("/gone.jsonl");
        pending.add([a.clone(), b.clone(), gone.clone()]);
        let modified = |path: &PathBuf| match path.to_str() {
            Some("/a.jsonl") => Some(now - QUIET),
            Some("/b.jsonl") => Some(now - Duration::from_secs(20)),
            _ => None,
        };
        let (ready, next) = pending.take_quiet(now, modified);
        assert_eq!(ready, vec![a]);
        assert_eq!(next, Some(QUIET - Duration::from_secs(20)));
        assert_eq!(pending.len(), 1);
        let (ready, next) = pending.take_quiet(now + Duration::from_secs(40), modified);
        assert_eq!(ready, vec![b]);
        assert_eq!(next, None);
        assert_eq!(pending.len(), 0);
    }

    #[test]
    fn the_daemon_is_idle_after_days_without_a_launch() {
        let started = SystemTime::now();
        let day = Duration::from_secs(24 * 60 * 60);
        assert!(!idle(started + 2 * day, None, started));
        assert!(idle(started + IDLE_EXIT, None, started));
        let launch = started + 2 * day;
        assert!(!idle(started + IDLE_EXIT, Some(launch), started));
        assert!(idle(launch + IDLE_EXIT, Some(launch), started));
        // A launch recorded before this daemon started counts from its start.
        assert!(!idle(started + 2 * day, Some(started - 10 * day), started));
    }
}
