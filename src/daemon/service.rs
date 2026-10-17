//! The daemon process and its embedding child.

use super::embed::{self, Job, Report};
use super::identity::BinaryIdentity;
use super::lock::InstanceLock;
use super::log::Log;
use super::schedule::{ExitReason, Pending, Schedule};
use super::status::{Status, StatusError};
use super::watch::{HOT_WINDOW, Watcher};
use super::{LOCK_WAIT, Paths, ms_of, now_ms, power, process, semantic_dir, store_exists};
use crate::config::SemanticConfig;
use crate::error::{AppError, Result};
use crate::history::adapter::MTIME_SLACK;
use crate::history::model::SourceKind;
use crate::history::sources::Registry;
use crate::semantic::cache::EmbedLock;
use crate::semantic::embed::SemanticEmbedder;
use crate::semantic::fastembed::{EmbedPriority, FastembedEmbedder};
use crate::semantic::types::SemanticCancellationToken;
use crate::store::generation::Publish;
use crate::store::reconcile::{self, Targets};
use serde::{Deserialize, Serialize};
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::time::{Duration, Instant, SystemTime};

/// How long the embedding child may take to stop before it is killed.
const CHILD_STOP_WAIT: Duration = Duration::from_secs(10);
/// Shortest wait between two rounds of the main loop.
const MIN_SLEEP: Duration = Duration::from_secs(1);
/// How long a battery reading is reused.
const BATTERY_RECHECK: Duration = Duration::from_secs(60);

enum Event {
    Signal,
    /// A launch asked the daemon to check for work.
    Wake,
    Report(Report),
    JobExited,
}

/// What the daemon sends its embedding child on stdin.
#[derive(Deserialize, Serialize)]
struct EmbedRequest {
    job: Job,
    store_root: PathBuf,
    semantic_dir: PathBuf,
    pause_on_battery: bool,
}

/// What the daemon sends its pass child on stdin: the files and
/// instances of a targeted pass, or neither for a full pass.
#[derive(Deserialize, Serialize)]
struct PassRequest {
    paths: Option<Vec<PathBuf>>,
    instances: Vec<String>,
}

/// What a pass child reports on stdout.
#[derive(Debug, Default, Deserialize, Serialize)]
struct PassReport {
    /// Another process was publishing.
    busy: bool,
    published: Option<String>,
    scanned: usize,
    parsed: usize,
    /// Session files parsed into a conversation.
    changed: Vec<PathBuf>,
    error: Option<String>,
}

/// Whether to hold embedding back for battery power, read at most once a
/// `BATTERY_RECHECK`.
struct Battery {
    checked: Option<(Instant, bool)>,
}

impl Battery {
    fn new() -> Self {
        Self { checked: None }
    }

    fn on_battery(&mut self, pause: bool) -> bool {
        if !pause {
            return false;
        }
        match self.checked {
            Some((at, on)) if at.elapsed() < BATTERY_RECHECK => on,
            _ => {
                let on = power::on_battery().unwrap_or(false);
                self.checked = Some((Instant::now(), on));
                on
            }
        }
    }
}

struct RunningJob {
    job: Job,
    child: Child,
    last: Option<Report>,
}

struct Daemon {
    paths: Paths,
    log: Log,
    identity: BinaryIdentity,
    root: PathBuf,
    semantic_dir: PathBuf,
    config: SemanticConfig,
    watcher: Watcher,
    primed: bool,
    schedule: Schedule,
    pending: Pending,
    /// Whether the last pass published or found the store current, rather
    /// than finding another process publishing.
    store_current: bool,
    /// Start of the last pass that left the store current: files unmodified
    /// since are in the store as they are.
    last_pass: Option<SystemTime>,
    /// When pending conversations next go quiet.
    ready_at: Option<Instant>,
    job: Option<RunningJob>,
    battery: Battery,
    started: SystemTime,
    status: Status,
    written: Option<Status>,
    events: Sender<Event>,
}

/// `daemon run`: the daemon itself, until it exits.
pub fn run(paths: &Paths) -> Result<()> {
    let (events, inbox) = mpsc::channel();
    let signals = events.clone();
    process::forward_signals(move |signal| {
        let event = if signal == process::WAKE {
            Event::Wake
        } else {
            Event::Signal
        };
        let _ = signals.send(event);
    })?;
    process::lower_priority();
    let Some(_lock) = InstanceLock::acquire_within(&paths.lock, LOCK_WAIT)? else {
        // Another daemon runs.
        return Ok(());
    };
    let mut log = Log::open(&paths.log);
    let identity = BinaryIdentity::current()?;
    let (Some(root), Some(semantic_dir)) = (crate::store::default_root(), semantic_dir()) else {
        log.line("exiting: no home directory");
        return Err(AppError::ConfigError("no home directory".into()));
    };
    log.line(&format!(
        "started: pid {}, {} {}",
        std::process::id(),
        identity.version,
        identity.path.display()
    ));
    let now = Instant::now();
    let started = SystemTime::now();
    let mut daemon = Daemon {
        status: Status::new(std::process::id(), identity.clone(), ms_of(started)),
        paths: paths.clone(),
        log,
        identity,
        root,
        semantic_dir,
        config: SemanticConfig::default(),
        watcher: Watcher::from_env(),
        primed: false,
        schedule: Schedule::new(now),
        pending: Pending::default(),
        store_current: false,
        last_pass: None,
        ready_at: None,
        job: None,
        battery: Battery::new(),
        started,
        written: None,
        events,
    };
    let reason = daemon.run_until_exit(&inbox);
    daemon.shutdown(&reason);
    Ok(())
}

impl Daemon {
    fn run_until_exit(&mut self, inbox: &Receiver<Event>) -> ExitReason {
        loop {
            if let Some(reason) = self.exit_reason() {
                return reason;
            }
            let now = Instant::now();
            self.passes(now);
            self.jobs(now);
            self.write_status();
            let jobs_blocked = self.job.is_some() || !self.store_current;
            let wake = self.schedule.next_wake(self.ready_at, jobs_blocked);
            // Work that is due but cannot proceed never makes the loop spin.
            let timeout = wake
                .saturating_duration_since(Instant::now())
                .max(MIN_SLEEP);
            match inbox.recv_timeout(timeout) {
                Ok(Event::Signal) => return ExitReason::Stopped,
                Ok(Event::Wake) => self.schedule.sweep_now(Instant::now()),
                Ok(Event::Report(report)) => self.on_report(report),
                Ok(Event::JobExited) => self.on_job_exit(),
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => {
                    return ExitReason::Failed("event channel closed".into());
                }
            }
        }
    }

    fn exit_reason(&mut self) -> Option<ExitReason> {
        if !self.identity.still_installed() {
            return Some(ExitReason::BinaryChanged);
        }
        match crate::config::load_config() {
            Ok(config) => {
                let semantic = config.semantic.unwrap_or_default();
                if !semantic.daemon() {
                    return Some(ExitReason::Disabled);
                }
                self.config = semantic;
            }
            // An unreadable config is kept as it was until it is fixed.
            Err(error) => self.error(&format!("config: {error}")),
        }
        let last_launch = self.paths.last_launch();
        if super::schedule::idle(SystemTime::now(), last_launch, self.started) {
            return Some(ExitReason::Idle);
        }
        None
    }

    fn passes(&mut self, now: Instant) {
        let due = self.schedule.due(now);
        if due.full {
            self.full_pass(now);
        } else if due.quick {
            self.quick_pass(now);
        }
        self.status.quick_interval_secs = self.schedule.quick_interval().as_secs();
    }

    fn full_pass(&mut self, now: Instant) {
        self.status.last_full_pass_ms = Some(now_ms());
        if !store_exists(&self.root) {
            // A launch builds the store; the daemon only keeps it current.
            self.store_current = false;
            self.schedule.full_done(now, false);
            return;
        }
        let today = chrono::Local::now().date_naive();
        if !self.primed {
            let hot = crate::store::Snapshot::open(&self.root, false, Default::default())
                .map(|cards| {
                    let since = SystemTime::now()
                        .checked_sub(HOT_WINDOW)
                        .unwrap_or(SystemTime::UNIX_EPOCH);
                    cards.paths_modified_since(since)
                })
                .unwrap_or_default();
            self.watcher.prime(hot, today);
            self.primed = true;
            self.log.line(&format!(
                "watching {} recently modified files",
                self.watcher.hot_len()
            ));
        }
        let started = SystemTime::now();
        let result = self.pass(None);
        // Recorded first: a pass that finds the store busy asks for the
        // next one sooner.
        self.schedule.full_done(now, false);
        if self.after_pass(result, started, "full") {
            self.schedule.quick_done(now, true);
        }
    }

    fn quick_pass(&mut self, now: Instant) {
        self.status.last_quick_pass_ms = Some(now_ms());
        if !self.primed {
            self.schedule.quick_done(now, false);
            return;
        }
        let started = SystemTime::now();
        let changes = self
            .watcher
            .quick_pass(started, chrono::Local::now().date_naive());
        if changes.is_empty() {
            if self.store_current {
                self.last_pass = Some(started);
            }
            self.schedule.quick_done(now, false);
            return;
        }
        let result = self.pass(Some(&changes.targets));
        self.schedule.quick_done(now, false);
        if self.after_pass(result, started, "quick") {
            self.schedule.quick_done(now, true);
        }
    }

    /// Runs a reconcile pass over `targets`, or a full pass, in a child
    /// process: parsing leaves the allocator holding tens of megabytes,
    /// which the daemon would otherwise keep while it idles.
    fn pass(&self, targets: Option<&Targets>) -> std::io::Result<PassReport> {
        let mut child = Command::new(&self.identity.path)
            .args(["daemon", "pass"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()?;
        let request = PassRequest {
            paths: targets.map(|targets| targets.paths.clone()),
            instances: targets
                .map(|targets| {
                    targets
                        .instances
                        .iter()
                        .map(|kind| kind.label().to_owned())
                        .collect()
                })
                .unwrap_or_default(),
        };
        let mut stdin = child.stdin.take().expect("piped stdin");
        serde_json::to_writer(&mut stdin, &request).map_err(std::io::Error::other)?;
        drop(stdin);
        let mut output = String::new();
        std::io::Read::read_to_string(
            &mut child.stdout.take().expect("piped stdout"),
            &mut output,
        )?;
        let status = child.wait()?;
        let report: PassReport = serde_json::from_str(&output).map_err(|_| {
            std::io::Error::other(format!("the pass process ended early: {status}"))
        })?;
        match report.error {
            Some(error) => Err(std::io::Error::other(error)),
            None => Ok(report),
        }
    }

    /// Takes in the outcome of a pass that started at `started`; returns
    /// whether it found changed conversations.
    fn after_pass(
        &mut self,
        result: std::io::Result<PassReport>,
        started: SystemTime,
        kind: &str,
    ) -> bool {
        let outcome = match result {
            Ok(outcome) => outcome,
            Err(error) => {
                self.error(&format!("{kind} pass: {error}"));
                self.store_current = false;
                self.schedule.full_soon(Instant::now());
                return false;
            }
        };
        if outcome.busy {
            // A launch is publishing; what this pass saw is looked at again.
            self.log
                .line(&format!("{kind} pass: another process is publishing"));
            self.store_current = false;
            self.schedule.full_soon(Instant::now());
            return false;
        }
        self.store_current = true;
        self.last_pass = Some(started);
        if let Some(name) = &outcome.published {
            self.status.last_publish_ms = Some(now_ms());
            self.log.line(&format!(
                "{kind} pass published {name}: {} files, {} parsed",
                outcome.scanned, outcome.parsed
            ));
        }
        // Reported once more by the next quick pass, which costs a stat:
        // a write while this pass parsed is not missed.
        for path in &outcome.changed {
            self.watcher.add_hot(path.clone(), false);
        }
        let changed = !outcome.changed.is_empty();
        self.pending.add(outcome.changed);
        self.status.pending_conversations = self.pending.len();
        changed
    }

    /// When `path` was last modified, if it is in the store as it is now;
    /// a file modified since the last pass counts as modified now.
    fn settled_since(&self, path: &Path) -> Option<SystemTime> {
        let modified = std::fs::metadata(path).ok()?.modified().ok()?;
        let passed = self
            .last_pass
            .is_some_and(|pass| modified + MTIME_SLACK <= pass);
        Some(if passed { modified } else { SystemTime::now() })
    }

    fn jobs(&mut self, now: Instant) {
        if self.job.is_some() {
            self.ready_at = None;
            return;
        }
        let (ready, next) = if self.store_current {
            let mut pending = std::mem::take(&mut self.pending);
            let taken = pending.take_quiet(SystemTime::now(), |path| self.settled_since(path));
            self.pending = pending;
            taken
        } else {
            (Vec::new(), None)
        };
        self.ready_at = next.map(|wait| now + wait);
        let job = if !ready.is_empty() {
            Job::Conversations { paths: ready }
        } else if self.store_current && self.schedule.due(now).sweep {
            Job::Sweep
        } else {
            self.status.state = "idle".into();
            return;
        };
        let refused = if !self.schedule.job_allowed(now) {
            Some(None)
        } else if self.battery.on_battery(self.config.pause_on_battery()) {
            self.status.state = "paused on battery".into();
            Some(None)
        } else if EmbedLock::is_held(&self.semantic_dir) {
            Some(Some("another process is embedding"))
        } else {
            None
        };
        if let Some(reason) = refused {
            if let Some(reason) = reason {
                self.log.line(&format!("embedding deferred: {reason}"));
            }
            self.requeue(&job);
            self.ready_at = Some(now);
            if self.schedule.job_allowed(now) {
                self.schedule.job_refused(now);
            }
            return;
        }
        match self.spawn_job(&job) {
            Ok(running) => {
                self.status.state = "embedding".into();
                self.job = Some(running);
            }
            Err(error) => {
                self.error(&format!("starting the embedding process: {error}"));
                self.requeue(&job);
                self.schedule.job_refused(now);
            }
        }
        self.status.pending_conversations = self.pending.len();
    }

    fn requeue(&mut self, job: &Job) {
        if let Job::Conversations { paths } = job {
            self.pending.add(paths.iter().cloned());
        }
    }

    fn spawn_job(&mut self, job: &Job) -> std::io::Result<RunningJob> {
        let mut child = Command::new(&self.identity.path)
            .args(["daemon", "embed"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()?;
        let request = EmbedRequest {
            job: job.clone(),
            store_root: self.root.clone(),
            semantic_dir: self.semantic_dir.clone(),
            pause_on_battery: self.config.pause_on_battery(),
        };
        let mut stdin = child.stdin.take().expect("piped stdin");
        serde_json::to_writer(&mut stdin, &request).map_err(std::io::Error::other)?;
        drop(stdin);
        let stdout = child.stdout.take().expect("piped stdout");
        let events = self.events.clone();
        std::thread::Builder::new()
            .name("daemon-job".to_owned())
            .spawn(move || {
                for line in std::io::BufReader::new(stdout).lines() {
                    let Ok(line) = line else { break };
                    if let Ok(report) = serde_json::from_str(&line) {
                        let _ = events.send(Event::Report(report));
                    }
                }
                let _ = events.send(Event::JobExited);
            })?;
        let what = match job {
            Job::Conversations { paths } => format!("{} conversations", paths.len()),
            Job::Sweep => "sweep".to_owned(),
        };
        self.log.line(&format!("embedding: {what}"));
        Ok(RunningJob {
            job: job.clone(),
            child,
            last: None,
        })
    }

    fn on_report(&mut self, report: Report) {
        let Some(job) = &mut self.job else {
            return;
        };
        let before = job.last.as_ref().map_or(0, |last| last.embedded);
        self.status.embedded += report.embedded.saturating_sub(before);
        if report.written {
            self.status.last_embed_ms = Some(now_ms());
        }
        if job.job == Job::Sweep {
            self.status.backlog = Some(report.missing);
        }
        job.last = Some(report);
    }

    fn on_job_exit(&mut self) {
        let Some(mut job) = self.job.take() else {
            return;
        };
        let exit = job.child.wait();
        let now = Instant::now();
        let last = job.last.unwrap_or_default();
        let finished = last.done && last.stopped.is_none() && last.error.is_none();
        if let Some(error) = &last.error {
            self.error(&format!("embedding: {error}"));
        } else if !last.done {
            let status = exit.map_or_else(|error| error.to_string(), |status| status.to_string());
            self.error(&format!("the embedding process ended early: {status}"));
        }
        self.log.line(&format!(
            "embedded {} passages, {} missing{}",
            last.embedded,
            last.missing,
            last.stopped
                .as_ref()
                .map(|reason| format!(", stopped: {reason}"))
                .unwrap_or_default()
        ));
        match &job.job {
            Job::Sweep => self.schedule.sweep_done(now, finished),
            Job::Conversations { .. } if !finished => {
                self.requeue(&job.job);
                self.schedule.job_refused(now);
            }
            Job::Conversations { .. } => {}
        }
        self.status.state = "idle".into();
        self.status.pending_conversations = self.pending.len();
    }

    fn error(&mut self, message: &str) {
        self.log.line(&format!("error: {message}"));
        self.status.last_error = Some(StatusError {
            at_ms: now_ms(),
            message: message.to_owned(),
        });
    }

    fn write_status(&mut self) {
        self.status.cpu_ms =
            u64::try_from(crate::bench::cpu_time_with_children().as_millis()).unwrap_or(u64::MAX);
        if self.written.as_ref() == Some(&self.status) {
            return;
        }
        if let Err(error) = self.status.write(&self.paths.status) {
            self.log
                .line(&format!("error: writing the status: {error}"));
        }
        self.written = Some(self.status.clone());
    }

    fn shutdown(&mut self, reason: &ExitReason) {
        if let Some(mut job) = self.job.take() {
            let _ = process::terminate(job.child.id());
            let deadline = Instant::now() + CHILD_STOP_WAIT;
            while Instant::now() < deadline {
                if matches!(job.child.try_wait(), Ok(Some(_))) {
                    break;
                }
                std::thread::sleep(Duration::from_millis(100));
            }
            if !matches!(job.child.try_wait(), Ok(Some(_))) {
                let _ = job.child.kill();
                let _ = job.child.wait();
            }
        }
        self.status.state = "exited".into();
        self.status.exit_reason = Some(reason.to_string());
        self.write_status();
        self.log.line(&format!("exiting: {reason}"));
    }
}

/// `daemon pass`: runs the reconcile pass the daemon sends on stdin and
/// reports it as JSON on stdout.
pub fn pass_child() -> Result<()> {
    process::lower_priority();
    process::single_low_priority_pool();
    let request: PassRequest = serde_json::from_reader(std::io::stdin().lock())
        .map_err(|error| AppError::ConfigError(format!("pass request: {error}")))?;
    let root = crate::store::default_root()
        .ok_or_else(|| AppError::ConfigError("no home directory".into()))?;
    let sources = Registry::from_env();
    let result = match request.paths {
        None => reconcile::reconcile(&root, &sources),
        Some(paths) => {
            let instances = [
                SourceKind::Claude,
                SourceKind::Pi,
                SourceKind::Omp,
                SourceKind::Codex,
            ]
            .into_iter()
            .filter(|kind| request.instances.iter().any(|name| name == kind.label()))
            .collect();
            reconcile::reconcile_targets(&root, &sources, &Targets { paths, instances })
        }
    };
    let report = match result {
        Ok(outcome) => PassReport {
            busy: outcome.publish == Publish::Busy,
            published: match outcome.publish {
                Publish::Published(name) => Some(name),
                _ => None,
            },
            scanned: outcome.scanned,
            parsed: outcome.parsed,
            changed: outcome.changed,
            error: None,
        },
        Err(error) => PassReport {
            error: Some(error.to_string()),
            ..PassReport::default()
        },
    };
    serde_json::to_writer(std::io::stdout().lock(), &report)
        .map_err(|error| AppError::ConfigError(error.to_string()))?;
    Ok(())
}

/// `daemon embed`: runs the job the daemon sends on stdin and reports its
/// progress as JSON lines on stdout.
pub fn embed_child() -> Result<()> {
    let cancellation = SemanticCancellationToken::new();
    let cancel = cancellation.clone();
    process::forward_signals(move |_| cancel.cancel())?;
    process::lower_priority();
    let request: EmbedRequest = serde_json::from_reader(std::io::stdin().lock())
        .map_err(|error| AppError::ConfigError(format!("embedding request: {error}")))?;
    let mut stdout = std::io::stdout().lock();
    let mut report = |report: &Report| {
        let _ = serde_json::to_writer(&mut stdout, report);
        let _ = writeln!(stdout);
        let _ = stdout.flush();
    };
    let Some(_lock) = EmbedLock::try_acquire(&request.semantic_dir)? else {
        report(&Report {
            done: true,
            stopped: Some("another process is embedding".into()),
            ..Report::default()
        });
        return Ok(());
    };
    let mut battery = Battery::new();
    let mut interrupt = || {
        if cancellation.is_cancelled() {
            Some("stopped".to_owned())
        } else if battery.on_battery(request.pause_on_battery) {
            Some("on battery".to_owned())
        } else {
            None
        }
    };
    let mut new_embedder = || {
        FastembedEmbedder::new_quiet(EmbedPriority::Daemon)
            .map(|embedder| Box::new(embedder) as Box<dyn SemanticEmbedder>)
    };
    embed::run(
        &request.job,
        embed::Context {
            store_root: &request.store_root,
            semantic_dir: &request.semantic_dir,
            new_embedder: &mut new_embedder,
            cancellation: &cancellation,
            interrupt: &mut interrupt,
            report: &mut report,
            flush_interval: embed::FLUSH_INTERVAL,
        },
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn battery_readings_are_skipped_when_not_pausing() {
        let mut battery = Battery::new();
        assert!(!battery.on_battery(false));
        assert!(battery.checked.is_none());
    }

    #[test]
    fn an_embedding_request_round_trips() {
        let request = EmbedRequest {
            job: Job::Conversations {
                paths: vec![PathBuf::from("/p/a.jsonl")],
            },
            store_root: PathBuf::from("/store"),
            semantic_dir: PathBuf::from("/semantic"),
            pause_on_battery: true,
        };
        let json = serde_json::to_string(&request).unwrap();
        let back: EmbedRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.job, request.job);
        assert_eq!(back.store_root, request.store_root);
        let sweep: EmbedRequest = serde_json::from_str(
            r#"{"job":{"kind":"sweep"},"store_root":"/s","semantic_dir":"/d","pause_on_battery":false}"#,
        )
        .unwrap();
        assert_eq!(sweep.job, Job::Sweep);
    }
}
