//! The daemon's quick pass: finds session files that changed since the
//! last pass from directory times instead of listing every directory,
//! which a full reconcile pass does.
//!
//! - The modification time of each project directory is read, in one
//!   listing of its root. A new session file changes its project
//!   directory's, and only then is that directory listed.
//! - Files modified recently (the hot set) are stated every pass, as are
//!   the files that changed since. When a Claude session changes, its
//!   `<session>/subagents/` directory is listed for sidecars, which are
//!   written while their session is.
//! - Codex threads span rollouts, a session index and a state database, so
//!   a change to any of these marks Codex for a listing of its own: the
//!   state files, today's and yesterday's rollout directories, the archive
//!   and the hot rollouts are stated.
//!
//! What the quick pass misses, such as a file changed outside the hot set
//! without a new file beside it, the periodic full pass finds.

use super::dirs::child_dirs;
use crate::history::model::SourceKind;
use crate::history::omp_loader::OmpSessionRoot;
use crate::history::pi_loader::PiSessionRoot;
use crate::store::reconcile::Targets;
use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

/// Size and modification time of a file or directory.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Stamp {
    pub size: u64,
    pub modified: Option<SystemTime>,
}

pub fn stamp(path: &Path) -> Option<Stamp> {
    let metadata = fs::metadata(path).ok()?;
    Some(Stamp {
        size: metadata.len(),
        modified: metadata.modified().ok(),
    })
}

/// How long a file stays in the hot set after it was last modified.
pub const HOT_WINDOW: Duration = Duration::from_secs(24 * 60 * 60);
/// Files modified before this, at `now`, leave the hot set.
fn cutoff(now: SystemTime) -> SystemTime {
    now.checked_sub(HOT_WINDOW)
        .unwrap_or(SystemTime::UNIX_EPOCH)
}

fn is_hot(stamp: Stamp, cutoff: SystemTime) -> bool {
    stamp.modified.is_none_or(|modified| modified >= cutoff)
}

/// Files modified this close to their directory's earlier modification time
/// are taken as new when the directory changes, for file systems whose
/// times are coarse.
const DIR_SLACK: Duration = Duration::from_secs(2);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Layout {
    /// `<project>/<session>.jsonl`, with sidecars in
    /// `<project>/<session>/subagents/agent-*.jsonl`.
    Claude,
    /// `<project>/*.jsonl`, or `*.jsonl` in the root when flat.
    Pi { flat: bool },
}

/// One root of session files.
struct Tree {
    root: PathBuf,
    layout: Layout,
    /// Modification time of each project directory, or of the root itself
    /// when flat, when last compared.
    dirs: HashMap<PathBuf, Option<SystemTime>>,
}

impl Tree {
    fn new(root: PathBuf, layout: Layout) -> Self {
        Self {
            root,
            layout,
            dirs: HashMap::new(),
        }
    }

    /// The current modification time of each project directory; none when
    /// the root is gone, and an error when it cannot be read.
    fn current_dirs(&self) -> io::Result<HashMap<PathBuf, Option<SystemTime>>> {
        let dirs = if matches!(self.layout, Layout::Pi { flat: true }) {
            fs::metadata(&self.root)
                .map(|metadata| HashMap::from([(self.root.clone(), metadata.modified().ok())]))
        } else {
            child_dirs(&self.root)
        };
        match dirs {
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(HashMap::new()),
            dirs => dirs,
        }
    }

    /// Session files directly in project directory `dir`.
    fn session_files(&self, dir: &Path) -> Vec<PathBuf> {
        let Ok(entries) = fs::read_dir(dir) else {
            return Vec::new();
        };
        entries
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| self.is_session_file(path))
            .collect()
    }

    fn is_session_file(&self, path: &Path) -> bool {
        let jsonl = path
            .extension()
            .is_some_and(|extension| extension == "jsonl");
        match self.layout {
            Layout::Claude => jsonl && !is_sidecar_name(path),
            Layout::Pi { .. } => jsonl,
        }
    }
}

fn is_sidecar_name(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.starts_with("agent-"))
}

/// The sidecars of the Claude session at `session`.
fn sidecars(session: &Path) -> Vec<PathBuf> {
    let Some(stem) = session.file_stem() else {
        return Vec::new();
    };
    let dir = session.with_file_name(stem).join("subagents");
    let Ok(entries) = fs::read_dir(dir) else {
        return Vec::new();
    };
    entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            is_sidecar_name(path)
                && path
                    .extension()
                    .is_some_and(|extension| extension == "jsonl")
        })
        .collect()
}

/// Whether `path` is a file that changes when its database is only read:
/// SQLite updates the `-shm` index of a WAL database for readers too, so
/// watching it would make every pass, which reads the database, report the
/// next one.
fn changes_on_read(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.ends_with("-shm"))
}

/// The Codex home and the files that change with its threads.
struct Codex {
    home: PathBuf,
    /// Directories whose top-level files hold thread state.
    state_dirs: Vec<PathBuf>,
    /// Stamp of each path watched besides the hot rollouts.
    stamps: HashMap<PathBuf, Option<Stamp>>,
}

impl Codex {
    /// Paths stated each pass: the state directories' files, the archive,
    /// and the rollout directories of `today` and the day before.
    fn watched(&self, today: chrono::NaiveDate) -> Vec<PathBuf> {
        let mut paths = Vec::new();
        for dir in &self.state_dirs {
            if let Ok(entries) = fs::read_dir(dir) {
                paths.extend(
                    entries
                        .flatten()
                        .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_file()))
                        .map(|entry| entry.path())
                        .filter(|path| !changes_on_read(path)),
                );
            }
        }
        let sessions = self
            .home
            .join(crate::history::sources::codex::discover::SESSIONS);
        paths.push(sessions.clone());
        paths.push(
            self.home
                .join(crate::history::sources::codex::discover::ARCHIVED_SESSIONS),
        );
        for day in [today, today.pred_opt().unwrap_or(today)] {
            paths.push(sessions.join(day.format("%Y/%m/%d").to_string()));
        }
        paths
    }
}

/// A file of the hot set.
#[derive(Clone, Copy, Debug)]
struct Hot {
    kind: SourceKind,
    /// Stamp when last stated; `None` makes the next pass report it.
    stamp: Option<Stamp>,
}

/// State of the quick pass.
pub struct Watcher {
    trees: Vec<(SourceKind, Tree)>,
    codex: Option<Codex>,
    hot: HashMap<PathBuf, Hot>,
}

/// What a quick pass found.
#[derive(Debug, Default, Eq, PartialEq)]
pub struct Changes {
    /// For a targeted reconcile pass.
    pub targets: Targets,
}

impl Changes {
    pub fn is_empty(&self) -> bool {
        self.targets.is_empty()
    }
}

impl Watcher {
    /// A watcher of the roots the environment configures, as
    /// `Registry::from_env` reads them.
    pub fn from_env() -> Self {
        let codex = crate::history::sources::codex::home_from_env().map(|home| {
            let adapter = crate::history::sources::codex::CodexAdapter::new(home.clone());
            (home, adapter.state_dirs())
        });
        Self::with_roots(
            crate::history::get_claude_projects_root().ok(),
            crate::history::pi_loader::session_root().ok(),
            crate::history::omp_loader::session_root().ok(),
            codex,
        )
    }

    pub fn with_roots(
        claude: Option<PathBuf>,
        pi: Option<PiSessionRoot>,
        omp: Option<OmpSessionRoot>,
        codex: Option<(PathBuf, Vec<PathBuf>)>,
    ) -> Self {
        let mut trees = Vec::new();
        if let Some(root) = claude {
            trees.push((SourceKind::Claude, Tree::new(root, Layout::Claude)));
        }
        if let Some(root) = pi {
            let layout = Layout::Pi { flat: root.flat };
            trees.push((SourceKind::Pi, Tree::new(root.path, layout)));
        }
        if let Some(root) = omp {
            let layout = Layout::Pi { flat: root.flat };
            trees.push((SourceKind::Omp, Tree::new(root.path, layout)));
        }
        Self {
            trees,
            codex: codex.map(|(home, state_dirs)| Codex {
                home,
                state_dirs,
                stamps: HashMap::new(),
            }),
            hot: HashMap::new(),
        }
    }

    /// Records the current state of every root and project directory and
    /// of `hot`, the files modified recently, as the state the next pass
    /// compares with. A full pass follows, which finds whatever changed
    /// before this.
    pub fn prime(&mut self, hot: Vec<PathBuf>, today: chrono::NaiveDate) {
        for (_, tree) in &mut self.trees {
            tree.dirs = tree.current_dirs().unwrap_or_default();
        }
        if let Some(codex) = &mut self.codex {
            codex.stamps = codex
                .watched(today)
                .into_iter()
                .map(|path| {
                    let stamp = stamp(&path);
                    (path, stamp)
                })
                .collect();
        }
        self.hot.clear();
        for path in hot {
            self.add_hot(path, true);
        }
        // Of the files the store lists, those since removed or last
        // modified before the window stay out.
        let cutoff = cutoff(SystemTime::now());
        self.hot
            .retain(|_, hot| hot.stamp.is_some_and(|stamp| is_hot(stamp, cutoff)));
    }

    /// Adds `path` to the hot set; unless `stated`, the next pass reports
    /// it.
    pub fn add_hot(&mut self, path: PathBuf, stated: bool) {
        let Some(kind) = self.kind_of(&path) else {
            return;
        };
        let stamp = if stated { stamp(&path) } else { None };
        self.hot.insert(path, Hot { kind, stamp });
    }

    fn kind_of(&self, path: &Path) -> Option<SourceKind> {
        self.trees
            .iter()
            .find(|(_, tree)| path.starts_with(&tree.root))
            .map(|(kind, _)| *kind)
            .or_else(|| {
                self.codex
                    .as_ref()
                    .filter(|codex| path.starts_with(&codex.home))
                    .map(|_| SourceKind::Codex)
            })
    }

    /// Files in the hot set.
    pub fn hot_len(&self) -> usize {
        self.hot.len()
    }

    /// Times each part of a quick pass with `time(name, count, run)`,
    /// for `--bench-quick-pass`.
    pub fn bench_parts(
        &self,
        today: chrono::NaiveDate,
        mut time: impl FnMut(&str, usize, &dyn Fn()),
    ) {
        for (kind, tree) in &self.trees {
            time(&format!("{kind:?} dirs"), tree.dirs.len(), &|| {
                let _ = tree.current_dirs();
            });
        }
        let hot: Vec<&PathBuf> = self.hot.keys().collect();
        time("hot files", hot.len(), &|| {
            for path in &hot {
                stamp(path);
            }
        });
        if let Some(codex) = &self.codex {
            time("codex watched", codex.watched(today).len(), &|| {
                for path in codex.watched(today) {
                    stamp(&path);
                }
            });
        }
    }

    /// Finds what changed since the last pass, at `now` on day `today`.
    pub fn quick_pass(&mut self, now: SystemTime, today: chrono::NaiveDate) -> Changes {
        let mut changed: Vec<PathBuf> = Vec::new();
        let mut codex_changed = false;

        let hot = &self.hot;
        for (_, tree) in &mut self.trees {
            // A root that cannot be read is compared again next pass.
            let Ok(current) = tree.current_dirs() else {
                continue;
            };
            for (dir, modified) in &current {
                // Files modified since the directory last changed are new or
                // were written since; a directory not seen before is new
                // with all its files. Hot files are stated below.
                let since = match tree.dirs.get(dir) {
                    Some(before) if before == modified => continue,
                    Some(before) => {
                        before.map(|before| before.checked_sub(DIR_SLACK).unwrap_or(before))
                    }
                    None => None,
                };
                for file in tree.session_files(dir) {
                    if hot.contains_key(&file) {
                        continue;
                    }
                    let fresh = since.is_none_or(|since| {
                        stamp(&file)
                            .and_then(|stamp| stamp.modified)
                            .is_none_or(|modified| modified >= since)
                    });
                    if fresh {
                        changed.push(file);
                    }
                }
            }
            tree.dirs = current;
        }

        if let Some(codex) = &mut self.codex {
            for path in codex.watched(today) {
                let current = stamp(&path);
                let recorded = codex.stamps.insert(path, current);
                codex_changed |= recorded.is_none_or(|recorded| recorded != current);
            }
        }

        let cutoff = cutoff(now);
        let mut sessions_changed = Vec::new();
        self.hot.retain(|path, hot| {
            let current = stamp(path);
            if current != hot.stamp {
                hot.stamp = current;
                match hot.kind {
                    SourceKind::Codex => codex_changed = true,
                    _ => {
                        changed.push(path.clone());
                        if hot.kind == SourceKind::Claude && !is_sidecar_name(path) {
                            sessions_changed.push(path.clone());
                        }
                    }
                }
            }
            current.is_some_and(|stamp| is_hot(stamp, cutoff))
        });
        for session in sessions_changed {
            for sidecar in sidecars(&session) {
                let current = stamp(&sidecar);
                let recorded = self.hot.get(&sidecar).and_then(|hot| hot.stamp);
                if recorded != current {
                    changed.push(sidecar.clone());
                    self.hot.insert(
                        sidecar,
                        Hot {
                            kind: SourceKind::Claude,
                            stamp: current,
                        },
                    );
                }
            }
        }
        // New files join the hot set as they are now; removed and cold ones
        // stay out.
        for path in &changed {
            if !self.hot.contains_key(path)
                && let Some(kind) = self.kind_of(path)
                && let Some(stamp) = stamp(path)
                && is_hot(stamp, cutoff)
            {
                self.hot.insert(
                    path.clone(),
                    Hot {
                        kind,
                        stamp: Some(stamp),
                    },
                );
            }
        }

        changed.sort();
        changed.dedup();
        Changes {
            targets: Targets {
                paths: changed,
                instances: if codex_changed {
                    vec![SourceKind::Codex]
                } else {
                    Vec::new()
                },
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fixture {
        dir: tempfile::TempDir,
        watcher: Watcher,
    }

    fn today() -> chrono::NaiveDate {
        chrono::Local::now().date_naive()
    }

    impl Fixture {
        fn new() -> Self {
            let dir = tempfile::tempdir().unwrap();
            for sub in ["claude", "pi", "codex/sessions"] {
                fs::create_dir_all(dir.path().join(sub)).unwrap();
            }
            let watcher = Watcher::with_roots(
                Some(dir.path().join("claude")),
                Some(PiSessionRoot {
                    path: dir.path().join("pi"),
                    flat: false,
                }),
                None,
                Some((dir.path().join("codex"), vec![dir.path().join("codex")])),
            );
            Self { dir, watcher }
        }

        fn path(&self, relative: &str) -> PathBuf {
            self.dir.path().join(relative)
        }

        /// Writes `relative`, dated `age` ago, and dates its directory the
        /// same so the change it made is in the past too.
        fn write(&self, relative: &str, age: Duration) -> PathBuf {
            let path = self.path(relative);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(&path, format!("{relative}\n")).unwrap();
            set_age(&path, age);
            path
        }

        fn append(&self, path: &Path) {
            let mut file = fs::File::options().append(true).open(path).unwrap();
            std::io::Write::write_all(&mut file, b"more\n").unwrap();
        }

        fn pass(&mut self) -> Changes {
            self.watcher.quick_pass(SystemTime::now(), today())
        }
    }

    fn set_age(path: &Path, age: Duration) {
        let time = SystemTime::now() - age;
        let file = fs::File::open(path).unwrap();
        file.set_times(fs::FileTimes::new().set_modified(time).set_accessed(time))
            .unwrap();
    }

    fn age_dir(path: &Path, age: Duration) {
        let time = SystemTime::now() - age;
        let dir = fs::File::open(path).unwrap();
        dir.set_times(fs::FileTimes::new().set_modified(time).set_accessed(time))
            .unwrap();
    }

    const HOUR: Duration = Duration::from_secs(3600);
    const WEEK: Duration = Duration::from_secs(7 * 24 * 3600);

    #[test]
    fn an_unchanged_tree_reports_nothing() {
        let mut fixture = Fixture::new();
        let old = fixture.write("claude/-p/old.jsonl", WEEK);
        let recent = fixture.write("claude/-p/recent.jsonl", HOUR);
        fixture.write("pi/--p--/s.jsonl", WEEK);
        age_dir(&fixture.path("claude/-p"), HOUR);
        fixture.watcher.prime(vec![recent], today());
        assert_eq!(fixture.watcher.hot_len(), 1);
        assert!(fixture.pass().is_empty());
        assert!(fixture.pass().is_empty());
        drop(old);
    }

    #[test]
    fn appends_to_hot_files_and_new_sessions_are_found() {
        let mut fixture = Fixture::new();
        fixture.write("claude/-p/old.jsonl", WEEK);
        let recent = fixture.write("claude/-p/recent.jsonl", HOUR);
        age_dir(&fixture.path("claude/-p"), HOUR);
        age_dir(&fixture.path("claude"), HOUR);
        fixture.watcher.prime(vec![recent.clone()], today());

        fixture.append(&recent);
        assert_eq!(fixture.pass().targets.paths, vec![recent.clone()]);
        assert!(fixture.pass().is_empty());

        // A new session in a known project, and one in a new project.
        let new = fixture.path("claude/-p/new.jsonl");
        fs::write(&new, "x\n").unwrap();
        let elsewhere = fixture.path("claude/-q/other.jsonl");
        fs::create_dir_all(elsewhere.parent().unwrap()).unwrap();
        fs::write(&elsewhere, "x\n").unwrap();
        let pi = fixture.path("pi/--p--/s.jsonl");
        fs::create_dir_all(pi.parent().unwrap()).unwrap();
        fs::write(&pi, "x\n").unwrap();
        let mut found = fixture.pass().targets.paths;
        found.sort();
        let mut expected = vec![new.clone(), elsewhere.clone(), pi.clone()];
        expected.sort();
        assert_eq!(found, expected);

        // New files are hot from now on.
        fixture.append(&elsewhere);
        assert_eq!(fixture.pass().targets.paths, vec![elsewhere]);
    }

    #[test]
    fn sidecars_are_found_when_their_session_changes() {
        let mut fixture = Fixture::new();
        let session = fixture.write("claude/-p/s1.jsonl", HOUR);
        fixture.write("claude/-p/s1/tool-results/out.txt", HOUR);
        age_dir(&fixture.path("claude/-p"), HOUR);
        fixture.watcher.prime(vec![session.clone()], today());

        let sidecar = fixture.path("claude/-p/s1/subagents/agent-a1.jsonl");
        fs::create_dir_all(sidecar.parent().unwrap()).unwrap();
        fs::write(&sidecar, "x\n").unwrap();
        fs::write(sidecar.with_extension("meta.json"), "{}").unwrap();
        fixture.append(&session);
        // Paths sort by component: `s1` before `s1.jsonl`.
        assert_eq!(
            fixture.pass().targets.paths,
            vec![sidecar.clone(), session.clone()]
        );

        fixture.append(&sidecar);
        assert_eq!(fixture.pass().targets.paths, vec![sidecar]);
    }

    #[test]
    fn removed_hot_files_are_reported_once_and_cold_files_leave_the_set() {
        let mut fixture = Fixture::new();
        let gone = fixture.write("claude/-p/gone.jsonl", HOUR);
        let cooling = fixture.write("claude/-p/cooling.jsonl", 2 * HOUR);
        age_dir(&fixture.path("claude/-p"), HOUR);
        fixture
            .watcher
            .prime(vec![gone.clone(), cooling.clone()], today());
        fs::remove_file(&gone).unwrap();
        age_dir(&fixture.path("claude/-p"), HOUR);
        assert_eq!(fixture.pass().targets.paths, vec![gone]);
        assert_eq!(fixture.watcher.hot_len(), 1);

        set_age(&cooling, 2 * HOT_WINDOW);
        assert_eq!(fixture.pass().targets.paths, vec![cooling]);
        assert_eq!(fixture.watcher.hot_len(), 0);
        assert!(fixture.pass().is_empty());
    }

    #[test]
    fn codex_state_and_rollouts_mark_codex_for_a_listing() {
        let mut fixture = Fixture::new();
        let day = today().format("%Y/%m/%d").to_string();
        let rollout = fixture.write(&format!("codex/sessions/{day}/rollout-a.jsonl"), HOUR);
        let db = fixture.write("codex/state_5.sqlite", HOUR);
        let shm = fixture.write("codex/state_5.sqlite-shm", HOUR);
        fixture.watcher.prime(vec![rollout.clone()], today());
        assert!(fixture.pass().is_empty());

        // Readers of the database, passes among them, update its index.
        fixture.append(&shm);
        assert!(fixture.pass().is_empty());

        fixture.append(&rollout);
        let changes = fixture.pass();
        assert!(changes.targets.paths.is_empty());
        assert_eq!(changes.targets.instances, vec![SourceKind::Codex]);
        assert!(fixture.pass().is_empty());

        fixture.append(&db);
        assert_eq!(fixture.pass().targets.instances, vec![SourceKind::Codex]);

        fixture.write(
            &format!("codex/sessions/{day}/rollout-b.jsonl"),
            Duration::ZERO,
        );
        assert_eq!(fixture.pass().targets.instances, vec![SourceKind::Codex]);
        assert!(fixture.pass().is_empty());
    }

    #[test]
    fn priming_leaves_out_removed_and_cold_files() {
        let mut fixture = Fixture::new();
        let recent = fixture.write("claude/-p/recent.jsonl", HOUR);
        let cold = fixture.write("claude/-p/cold.jsonl", WEEK);
        let gone = fixture.path("claude/-p/gone.jsonl");
        fixture.watcher.prime(vec![recent, cold, gone], today());
        assert_eq!(fixture.watcher.hot_len(), 1);
    }

    #[test]
    fn files_added_unstated_are_reported_by_the_next_pass() {
        let mut fixture = Fixture::new();
        let path = fixture.write("claude/-p/s.jsonl", HOUR);
        age_dir(&fixture.path("claude/-p"), HOUR);
        fixture.watcher.prime(Vec::new(), today());
        fixture.watcher.add_hot(path.clone(), false);
        assert_eq!(fixture.pass().targets.paths, vec![path]);
        assert!(fixture.pass().is_empty());
    }
}
