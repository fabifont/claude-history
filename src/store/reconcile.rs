//! Brings the store up to date with the native session files.
//!
//! One pass takes `LOCK` without waiting (a busy lock means another process
//! is reconciling, and readers keep the current generation), lists every
//! session file of every source with one `stat` each, and compares them with
//! the fingerprints of `CURRENT`. Only new or changed files are parsed; a
//! pass that finds nothing to do writes nothing. Changes are published as
//! delta generations (see `generation::publish_update`), newest files first
//! and in batches, so a cold build shows the newest conversations early.
//! Segments are merged afterwards when `MergePolicy` asks for it.
//!
//! A source or directory that cannot be listed is degraded: its cards and
//! ignored files are kept as they are rather than dropped.

use super::card::Fingerprint;
use super::generation::{
    self, Doc, Generation, MergePolicy, Publish, PublishLock, Update, Written,
};
use super::legacy;
use super::progress::{self, Reporter};
use crate::history::Conversation;
use crate::history::adapter::{Completeness, FileId, Located};
use crate::history::index::Indexer;
use crate::history::model::SourceKind;
use crate::history::sources::Registry;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant, SystemTime};

/// Newest files parsed and published before any other.
const FIRST_BATCH: usize = 200;
/// Parsing time after which the conversations parsed so far are published.
const BATCH_TIME: Duration = Duration::from_secs(2);

/// What one pass did.
#[derive(Debug)]
pub struct Outcome {
    /// `Busy` when another process holds the lock, `Unchanged` when nothing
    /// was written, else the last generation published.
    pub publish: Publish,
    /// Generations published, merges included.
    pub published: usize,
    pub written: Written,
    /// Session files found.
    pub scanned: usize,
    /// Files parsed because they were new or changed.
    pub parsed: usize,
    /// Old caches removed after publishing.
    pub removed: legacy::Removed,
}

/// Runs one pass over the instances of `sources` into the store at `root`.
pub fn reconcile(root: &Path, sources: &Registry) -> io::Result<Outcome> {
    reconcile_with(root, sources, MergePolicy::DEFAULT)
}

pub fn reconcile_with(root: &Path, sources: &Registry, policy: MergePolicy) -> io::Result<Outcome> {
    let mut outcome = Outcome {
        publish: Publish::Busy,
        published: 0,
        written: Written::default(),
        scanned: 0,
        parsed: 0,
        removed: legacy::Removed::default(),
    };
    let Some(lock) = PublishLock::try_acquire(root)? else {
        return Ok(outcome);
    };
    outcome.publish = Publish::Unchanged;
    // Left by an earlier process that stopped without removing it.
    progress::clear(root);
    let scan = scan(sources);
    outcome.scanned = scan.files.len();
    crate::bench::mark("reconcile_scanned");

    let mut base = open_base(root);
    let cold = base.is_none();
    let plan = diff(base.as_ref(), &scan);
    let mut keep = plan.keep;
    let mut ignored = plan.ignored;
    let mut pending = plan.parse;
    // Newest first, so the first batches cover what the list shows first.
    pending.sort_by(|a, b| {
        b.located
            .modified
            .cmp(&a.located.modified)
            .then_with(|| a.located.path.cmp(&b.located.path))
    });
    let reporter = Reporter::start(root, pending.len());
    let parsed = AtomicUsize::new(0);
    let counter = reporter.as_ref().map_or(&parsed, Reporter::counter);
    let mut first = true;
    loop {
        let batch = take_batch(&mut pending, first, sources, counter);
        first = false;
        outcome.parsed += batch.len();
        let docs = apply_batch(base.as_ref(), &mut keep, &mut ignored, batch);
        let (publish, written) = generation::publish_update(
            &lock,
            Update {
                base: base.as_ref(),
                docs,
                ignored: ignored.clone(),
            },
        )?;
        add_written(&mut outcome, publish, written);
        if pending.is_empty() {
            break;
        }
        // Later batches build on what was just published.
        base = open_base(root);
        let Some(current) = &base else {
            return Err(io::Error::other("published generation does not open"));
        };
        keep = (0..current.rows().len() as u32)
            .map(|row| (row, current.fingerprints()[row as usize]))
            .collect();
        ignored = current.ignored_entries();
    }

    if let Some(current) = open_base(root) {
        let policy = if cold { MergePolicy::FULL } else { policy };
        let (publish, written) = generation::compact(&lock, &current, policy)?;
        add_written(&mut outcome, publish, written);
    }
    if outcome.published > 0
        && let Some(cache) = legacy::cache_dir_of(root)
    {
        outcome.removed = legacy::remove_old_caches(cache);
    }
    drop(reporter);
    drop(lock);
    Ok(outcome)
}

fn add_written(outcome: &mut Outcome, publish: Publish, written: Written) {
    if let Publish::Published(_) = &publish {
        outcome.published += 1;
        outcome.written.bytes += written.bytes;
        outcome.written.docs += written.docs;
        outcome.publish = publish;
    }
}

/// `CURRENT` when it opens with its search files.
fn open_base(root: &Path) -> Option<Generation> {
    Generation::open(root).filter(|generation| generation.has_search_files())
}

/// A conversation found by the scan.
#[derive(Clone, Debug)]
struct Found {
    located: Located,
    /// Other sources whose root holds this same file.
    also: Vec<SourceKind>,
    fingerprint: Fingerprint,
}

impl Found {
    fn source(&self) -> SourceKind {
        self.located.key.instance.kind
    }
}

#[derive(Default)]
struct Scan {
    files: Vec<Found>,
    /// How much of each instance the scan saw.
    completeness: Vec<(SourceKind, Completeness)>,
}

impl Scan {
    /// Whether the scan may have missed `path` of `source`.
    fn misses(&self, path: &Path, source: SourceKind) -> bool {
        self.completeness
            .iter()
            .any(|(kind, completeness)| *kind == source && completeness.may_miss(path))
    }
}

/// Every conversation of every instance. A file reachable through more than
/// one instance is kept once, under the first in registry order.
fn scan(sources: &Registry) -> Scan {
    let started = SystemTime::now();
    let mut scan = Scan::default();
    let discoveries: Vec<_> = sources
        .adapters()
        .par_iter()
        .map(|adapter| (adapter.instance().kind, adapter.discover(started)))
        .collect();
    // A file listed by several sources, as when the Pi and OMP roots are
    // one directory, is parsed once and kept for whichever source it holds.
    let mut ids: HashMap<FileId, usize> = HashMap::new();
    let mut paths: HashMap<PathBuf, usize> = HashMap::new();
    for (kind, discovery) in discoveries {
        scan.completeness.push((kind, discovery.completeness));
        for located in discovery.found {
            let next = scan.files.len();
            let kept = match located.file_id {
                Some(id) => *ids.entry(id).or_insert(next),
                None => *paths.entry(located.path.clone()).or_insert(next),
            };
            if kept == next {
                scan.files.push(Found {
                    fingerprint: located.stamp.into(),
                    located,
                    also: Vec::new(),
                });
            } else {
                let kept = &mut scan.files[kept];
                if kept.source() != kind && !kept.also.contains(&kind) {
                    kept.also.push(kind);
                }
            }
        }
    }
    scan
}

/// Every conversation of every instance of `sources`, parsed in memory.
pub fn parse_everything(sources: &Registry) -> Vec<Conversation> {
    parse_all(scan(sources).files, sources, &AtomicUsize::new(0))
        .into_iter()
        .filter_map(|(_, parsed)| match parsed {
            Parsed::Conversation(conversation) => Some(*conversation),
            Parsed::Ignored | Parsed::Gone | Parsed::Failed => None,
        })
        .collect()
}

/// The scan compared with the base generation.
struct Plan {
    /// Base rows to carry over, with their fingerprints. Rows of files that
    /// are about to be parsed stay until their batch replaces them.
    keep: Vec<(u32, Fingerprint)>,
    ignored: Vec<(String, Fingerprint)>,
    /// New files, changed files and files whose fingerprint is unknown.
    parse: Vec<Found>,
}

fn diff(base: Option<&Generation>, scan: &Scan) -> Plan {
    let mut plan = Plan {
        keep: Vec::new(),
        ignored: Vec::new(),
        parse: Vec::new(),
    };
    let Some(base) = base else {
        plan.parse = scan.files.clone();
        return plan;
    };
    let rows = base.rows();
    let fingerprints = base.fingerprints();
    let by_path: HashMap<&str, u32> = rows
        .iter()
        .enumerate()
        .map(|(row, card)| (base.str(card.path), row as u32))
        .collect();
    let ignored: Vec<(String, Fingerprint)> = base.ignored_entries();
    let ignored_by_path: HashMap<&str, Fingerprint> = ignored
        .iter()
        .map(|(path, fingerprint)| (path.as_str(), *fingerprint))
        .collect();

    let mut seen_rows = vec![false; rows.len()];
    let mut seen_ignored: HashSet<&str> = HashSet::new();
    for found in &scan.files {
        let path = found.located.path.to_string_lossy();
        let known = found.located.stamp.is_known();
        let row = by_path.get(path.as_ref()).copied();
        if let Some(row) = row {
            seen_rows[row as usize] = true;
            if known && fingerprints[row as usize] == found.fingerprint {
                plan.keep.push((row, found.fingerprint));
                continue;
            }
            // Shown as it was until the file is parsed again.
            plan.keep.push((row, fingerprints[row as usize]));
        }
        if let Some((&key, &fingerprint)) = ignored_by_path.get_key_value(path.as_ref()) {
            seen_ignored.insert(key);
            if row.is_none() && known && fingerprint == found.fingerprint {
                plan.ignored.push((key.to_owned(), fingerprint));
                continue;
            }
        }
        plan.parse.push(found.clone());
    }
    for (row, card) in rows.iter().enumerate() {
        if !seen_rows[row]
            && let Some(source) = crate::store::card::source_from_u8(card.source)
            && scan.misses(Path::new(base.str(card.path)), source)
        {
            plan.keep.push((row as u32, fingerprints[row]));
        }
    }
    for (path, fingerprint) in &ignored {
        if !seen_ignored.contains(path.as_str())
            && [
                SourceKind::Claude,
                SourceKind::Pi,
                SourceKind::Omp,
                SourceKind::Codex,
            ]
            .into_iter()
            .any(|source| scan.misses(Path::new(path), source))
        {
            plan.ignored.push((path.clone(), *fingerprint));
        }
    }
    plan
}

/// What parsing one file gave.
enum Parsed {
    Conversation(Box<Conversation>),
    /// Holds no conversation of its source.
    Ignored,
    /// Removed since the scan.
    Gone,
    /// Could not be read; tried again next pass.
    Failed,
}

/// Removes and parses the next batch from `pending`: the first
/// `FIRST_BATCH` files on the first call, then as many as parse within
/// `BATCH_TIME`. Each parsed file is counted in `parsed`.
fn take_batch(
    pending: &mut Vec<Found>,
    first: bool,
    sources: &Registry,
    parsed: &AtomicUsize,
) -> Vec<(Found, Parsed)> {
    if first {
        let files: Vec<Found> = pending.drain(..pending.len().min(FIRST_BATCH)).collect();
        return parse_all(files, sources, parsed);
    }
    let started = Instant::now();
    let chunk = rayon::current_num_threads().max(1) * 8;
    let mut batch = Vec::new();
    while !pending.is_empty() && started.elapsed() < BATCH_TIME {
        let files: Vec<Found> = pending.drain(..pending.len().min(chunk)).collect();
        batch.extend(parse_all(files, sources, parsed));
    }
    batch
}

fn parse_all(files: Vec<Found>, sources: &Registry, count: &AtomicUsize) -> Vec<(Found, Parsed)> {
    files
        .into_par_iter()
        .map(|found| {
            let parsed = parse(&found, sources);
            count.fetch_add(1, Ordering::Relaxed);
            (found, parsed)
        })
        .collect()
}

/// Decodes `found` through its instance's adapter and indexes it. A file
/// that another source wrote is listed only when that source's root holds
/// it too, except under a Claude root.
fn parse(found: &Found, sources: &Registry) -> Parsed {
    let Some(adapter) = sources.get(&found.located.key.instance) else {
        return Parsed::Failed;
    };
    // Events stream into the indexer, so no file's events are held at once.
    let mut indexer = Indexer::default();
    let transcript = match adapter.decode_into(&found.located, &mut |event| indexer.push(&event)) {
        Ok(Some(transcript)) => transcript,
        Ok(None) => return Parsed::Ignored,
        Err(crate::error::AppError::Io(error)) if error.kind() == io::ErrorKind::NotFound => {
            return Parsed::Gone;
        }
        Err(_) => return Parsed::Failed,
    };
    let source = found.source();
    if source != SourceKind::Claude
        && transcript.source != source
        && !found.also.contains(&transcript.source)
    {
        return Parsed::Ignored;
    }
    match indexer.finish(&transcript) {
        Ok(conversation) => Parsed::Conversation(Box::new(conversation)),
        Err(_) => Parsed::Ignored,
    }
}

/// The docs of the next generation: `keep` without the rows `batch`
/// replaces, plus the batch's conversations. Updates `ignored` with the
/// batch's outcome.
fn apply_batch(
    base: Option<&Generation>,
    keep: &mut Vec<(u32, Fingerprint)>,
    ignored: &mut Vec<(String, Fingerprint)>,
    batch: Vec<(Found, Parsed)>,
) -> Vec<Doc> {
    let parsed_paths: HashSet<String> = batch
        .iter()
        .map(|(found, _)| found.located.path.to_string_lossy().into_owned())
        .collect();
    ignored.retain(|(path, _)| !parsed_paths.contains(path));
    let row_of: HashMap<&str, u32> = base
        .map(|base| {
            keep.iter()
                .map(|&(row, _)| (base.str(base.rows()[row as usize].path), row))
                .collect()
        })
        .unwrap_or_default();

    let mut replaced: HashSet<u32> = HashSet::new();
    let mut failed: Vec<(u32, Fingerprint)> = Vec::new();
    let mut new = Vec::new();
    for (found, parsed) in batch {
        let path = found.located.path.to_string_lossy().into_owned();
        let row = row_of.get(path.as_str()).copied();
        if let Some(row) = row {
            replaced.insert(row);
        }
        let unknown = Fingerprint {
            size: found.located.size,
            mtime_ns: 0,
        };
        match parsed {
            Parsed::Conversation(conversation) => new.push(Doc::New {
                conversation,
                fingerprint: found.fingerprint,
            }),
            Parsed::Ignored => ignored.push((path, found.fingerprint)),
            Parsed::Gone => {}
            Parsed::Failed => match row {
                Some(row) => failed.push((row, unknown)),
                None => ignored.push((path, unknown)),
            },
        }
    }
    keep.retain(|(row, _)| !replaced.contains(row));
    keep.extend(failed);
    let mut docs: Vec<Doc> = keep
        .iter()
        .map(|&(row, fingerprint)| Doc::Keep { row, fingerprint })
        .collect();
    docs.extend(new);
    docs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::history::omp_loader::OmpSessionRoot;
    use crate::history::pi_loader::PiSessionRoot;
    use crate::store::lexindex::Field;
    use std::fs;
    use std::time::Duration;

    struct Fixture {
        _dir: tempfile::TempDir,
        root: PathBuf,
        projects: PathBuf,
        pi: PiSessionRoot,
        omp: OmpSessionRoot,
    }

    impl Fixture {
        fn new() -> Self {
            let dir = tempfile::tempdir().unwrap();
            let projects = dir.path().join("claude").join("projects");
            let pi = dir.path().join("pi").join("sessions");
            let omp = dir.path().join(".omp").join("agent").join("sessions");
            for path in [&projects, &pi.join("--tmp-pi--"), &omp.join("--tmp-omp--")] {
                fs::create_dir_all(path).unwrap();
            }
            let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
            fs::copy(
                fixtures.join("pi/v1.jsonl"),
                pi.join("--tmp-pi--").join("one.jsonl"),
            )
            .unwrap();
            fs::copy(
                fixtures.join("omp/v3.jsonl"),
                omp.join("--tmp-omp--").join("two.jsonl"),
            )
            .unwrap();
            Self {
                root: dir.path().join("store"),
                pi: PiSessionRoot {
                    path: pi,
                    flat: false,
                },
                omp: OmpSessionRoot {
                    path: omp,
                    flat: false,
                },
                projects,
                _dir: dir,
            }
        }

        /// Writes a Claude session in project `project` with `text` as its
        /// only prompt, dated an hour ago so its fingerprint is known.
        fn claude(&self, project: &str, name: &str, text: &str) -> PathBuf {
            let dir = self.projects.join(project);
            fs::create_dir_all(&dir).unwrap();
            let path = dir.join(name);
            let line = serde_json::json!({
                "type": "user",
                "timestamp": "2024-01-01T00:00:00Z",
                "cwd": "/tmp/work",
                "message": {"role": "user", "content": text}
            });
            fs::write(&path, format!("{line}\n")).unwrap();
            settle(&path);
            path
        }

        fn run(&self) -> Outcome {
            let sources = Registry::with_roots(
                Some(self.projects.clone()),
                Some(self.pi.clone()),
                Some(self.omp.clone()),
            );
            reconcile(&self.root, &sources).unwrap()
        }

        fn current(&self) -> Generation {
            Generation::open(&self.root).unwrap()
        }

        fn paths(&self) -> Vec<String> {
            let generation = self.current();
            let mut paths: Vec<String> = generation
                .rows()
                .iter()
                .map(|row| generation.str(row.path).to_owned())
                .collect();
            paths.sort();
            paths
        }

        fn text_of(&self, path: &Path) -> String {
            let generation = self.current();
            let (_, text) = generation.open_search_files(Field::Body).unwrap();
            let row = generation
                .rows()
                .iter()
                .find(|row| generation.str(row.path) == path.to_string_lossy())
                .unwrap();
            text.read(row.text).unwrap()
        }
    }

    /// Dates `path` an hour back, past the window in which fingerprints are
    /// unknown.
    fn settle(path: &Path) {
        let file = fs::File::options().write(true).open(path).unwrap();
        file.set_modified(SystemTime::now() - Duration::from_secs(3600))
            .unwrap();
    }

    fn settle_all(dir: &Path) {
        for entry in fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                settle_all(&path);
            } else {
                settle(&path);
            }
        }
    }

    #[test]
    fn cold_build_reads_every_source_like_the_loader() {
        let fixture = Fixture::new();
        let a = fixture.claude("-tmp-work", "a.jsonl", "hello alpha");
        fixture.claude("-tmp-work", "agent-1.jsonl", "subagent");
        let empty = fixture.projects.join("-tmp-work").join("empty.jsonl");
        fs::write(&empty, "").unwrap();
        let outcome = fixture.run();
        assert!(matches!(outcome.publish, Publish::Published(_)));

        let generation = fixture.current();
        let sources: Vec<_> = generation.rows().iter().map(|row| row.source).collect();
        assert_eq!(generation.rows().len(), 3, "{:?}", fixture.paths());
        for source in [SourceKind::Claude, SourceKind::Pi, SourceKind::Omp] {
            assert!(sources.contains(&crate::store::card::source_to_u8(source)));
        }
        assert!(!fixture.paths().iter().any(|path| path.contains("agent-1")));
        assert_eq!(
            generation.ignored_entries()[0].0,
            empty.to_string_lossy().as_ref()
        );
        let row = generation
            .rows()
            .iter()
            .position(|row| generation.str(row.path) == a.to_string_lossy())
            .unwrap();
        let card = generation.conversation(row as u32, false);
        assert_eq!(card.project_path, Some(PathBuf::from("/tmp/work")));
        assert_eq!(card.project_name.as_deref(), Some("work"));
        assert!(fixture.text_of(&a).contains("hello alpha"));
    }

    #[test]
    fn sessions_of_a_root_pi_and_omp_share_are_kept_for_their_source() {
        let mut fixture = Fixture::new();
        let shared = fixture.pi.path.clone();
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let omp = shared.join("--tmp-pi--").join("omp.jsonl");
        fs::copy(fixtures.join("omp/v3.jsonl"), &omp).unwrap();
        settle_all(&shared);
        fixture.omp = OmpSessionRoot {
            path: shared,
            flat: false,
        };
        fixture.run();

        let generation = fixture.current();
        let source_of = |path: &Path| {
            generation
                .rows()
                .iter()
                .find(|row| generation.str(row.path) == path.to_string_lossy())
                .map(|row| row.source)
        };
        assert_eq!(
            source_of(&omp),
            Some(crate::store::card::source_to_u8(SourceKind::Omp))
        );
        assert_eq!(generation.rows().len(), 2, "{:?}", fixture.paths());
        let unchanged = fixture.run();
        assert_eq!(unchanged.publish, Publish::Unchanged);
    }

    #[test]
    fn unchanged_files_write_nothing_and_parse_nothing() {
        let fixture = Fixture::new();
        fixture.claude("-tmp-work", "a.jsonl", "hello alpha");
        fs::write(fixture.projects.join("-tmp-work").join("empty.jsonl"), "").unwrap();
        settle_all(fixture._dir.path());
        fixture.run();
        let current = fs::read_to_string(fixture.root.join("CURRENT")).unwrap();

        let listing = || {
            let mut names: Vec<_> = fs::read_dir(&fixture.root)
                .unwrap()
                .map(|entry| entry.unwrap().file_name())
                .collect();
            names.sort();
            names
        };
        let before = listing();
        let outcome = fixture.run();
        assert_eq!(outcome.publish, Publish::Unchanged);
        assert_eq!(outcome.parsed, 0);
        assert_eq!(outcome.written.bytes, 0);
        assert_eq!(
            fs::read_to_string(fixture.root.join("CURRENT")).unwrap(),
            current
        );
        assert_eq!(listing(), before);
        assert!(!fixture.root.join("PROGRESS").exists());
    }

    #[test]
    fn progress_left_by_a_stopped_pass_is_removed() {
        let fixture = Fixture::new();
        fixture.claude("-tmp-work", "a.jsonl", "hello alpha");
        fs::create_dir_all(&fixture.root).unwrap();
        progress::record_failure(&fixture.root, "stopped");
        fixture.run();
        assert_eq!(progress::read(&fixture.root), None);
        assert!(!fixture.root.join("PROGRESS").exists());
    }

    #[test]
    fn an_appended_session_publishes_one_delta_doc() {
        let fixture = Fixture::new();
        let a = fixture.claude("-tmp-work", "a.jsonl", "hello alpha");
        for i in 0..20 {
            fixture.claude("-tmp-other", &format!("{i}.jsonl"), "filler words");
        }
        settle_all(fixture._dir.path());
        fixture.run();

        let line = serde_json::json!({
            "type": "user",
            "timestamp": "2024-01-02T00:00:00Z",
            "message": {"role": "user", "content": "appended zebra"}
        });
        let mut file = fs::File::options().append(true).open(&a).unwrap();
        std::io::Write::write_all(&mut file, format!("{line}\n").as_bytes()).unwrap();
        drop(file);
        settle(&a);

        let outcome = fixture.run();
        assert!(matches!(outcome.publish, Publish::Published(_)));
        assert_eq!(outcome.parsed, 1);
        assert_eq!(outcome.written.docs, 1);
        assert!(fixture.text_of(&a).contains("appended zebra"));
        let generation = fixture.current();
        assert_eq!(generation.segs().len(), 2);
        let (index, _) = generation.open_search_files(Field::Body).unwrap();
        assert_eq!(
            index
                .prefix_tf("zebra")
                .iter()
                .filter(|&&tf| tf > 0)
                .count(),
            1
        );
        assert_eq!(
            index
                .prefix_tf("alpha")
                .iter()
                .filter(|&&tf| tf > 0)
                .count(),
            1
        );
    }

    #[test]
    fn recently_modified_files_are_parsed_again() {
        let fixture = Fixture::new();
        let a = fixture.claude("-tmp-work", "a.jsonl", "hello alpha");
        settle_all(fixture._dir.path());
        fs::File::options()
            .write(true)
            .open(&a)
            .unwrap()
            .set_modified(SystemTime::now())
            .unwrap();
        fixture.run();
        let outcome = fixture.run();
        assert_eq!(outcome.parsed, 1);
        assert_eq!(outcome.publish, Publish::Unchanged);
    }

    #[test]
    fn removed_files_are_dropped_unless_their_directory_is_unreadable() {
        let fixture = Fixture::new();
        let a = fixture.claude("-tmp-work", "a.jsonl", "hello alpha");
        let b = fixture.claude("-tmp-gone", "b.jsonl", "hello beta");
        let c = fixture.claude("-tmp-locked", "c.jsonl", "hello gamma");
        settle_all(fixture._dir.path());
        fixture.run();
        assert_eq!(fixture.paths().len(), 5);

        fs::remove_file(&b).unwrap();
        #[cfg(unix)]
        let locked = {
            use std::os::unix::fs::PermissionsExt;
            let dir = c.parent().unwrap().to_path_buf();
            fs::set_permissions(&dir, fs::Permissions::from_mode(0o000)).unwrap();
            fs::read_dir(&dir).is_err().then_some(dir)
        };
        fixture.run();
        let paths = fixture.paths();
        #[cfg(unix)]
        if let Some(dir) = locked {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&dir, fs::Permissions::from_mode(0o755)).unwrap();
            assert!(paths.contains(&c.to_string_lossy().into_owned()));
        }
        assert!(paths.contains(&a.to_string_lossy().into_owned()));
        assert!(!paths.contains(&b.to_string_lossy().into_owned()));
    }

    #[test]
    fn codex_threads_are_indexed_with_their_provenance() {
        use crate::store::card::{self, card_flags};
        let fixture = Fixture::new();
        let codex = crate::history::sources::codex::tests::home_copy(fixture._dir.path());
        settle_all(fixture._dir.path());
        let sources = Registry::with_roots(
            Some(fixture.projects.clone()),
            Some(fixture.pi.clone()),
            Some(fixture.omp.clone()),
        )
        .with_codex(Some(codex.clone()));
        reconcile(&fixture.root, &sources).unwrap();

        let generation = fixture.current();
        let codex_rows: Vec<usize> = (0..generation.rows().len())
            .filter(|&row| generation.rows()[row].source == card::source_to_u8(SourceKind::Codex))
            .collect();
        assert_eq!(codex_rows.len(), 9);
        let flags_of = |thread: &str| {
            let row = codex_rows
                .iter()
                .map(|&row| &generation.rows()[row])
                .find(|row| generation.str(row.session_id) == thread)
                .unwrap();
            row.flags & card_flags::PROVENANCE
        };
        assert_eq!(
            flags_of("01a00000-0000-7000-8000-000000000006"),
            card_flags::AUTOMATED
        );
        assert_eq!(
            flags_of("01a00000-0000-7000-8000-000000000007"),
            card_flags::CHILD
        );
        assert_eq!(
            flags_of("01a00000-0000-7000-8000-000000000009"),
            card_flags::ORPHANED
        );
        assert_eq!(flags_of("01a00000-0000-7000-8000-000000000002"), 0);
        let snapshot =
            crate::store::Snapshot::open(&fixture.root, false, Default::default()).unwrap();
        assert_eq!(
            snapshot.head(usize::MAX, false, &[], None, |_| true).len(),
            generation.rows().len() - 2
        );
        assert_eq!(
            snapshot.head(usize::MAX, true, &[], None, |_| true).len(),
            generation.rows().len()
        );

        let unchanged = reconcile(&fixture.root, &sources).unwrap();
        assert_eq!(unchanged.publish, Publish::Unchanged);
        assert_eq!(unchanged.parsed, 0);
        assert_eq!(unchanged.written.bytes, 0);

        // A name set in Codex reaches the card through the state database.
        let db = rusqlite::Connection::open(codex.join("state_5.sqlite")).unwrap();
        db.execute(
            "UPDATE threads SET name = 'Renamed walrus' WHERE id = ?1",
            ["01a00000-0000-7000-8000-000000000002"],
        )
        .unwrap();
        let renamed = reconcile(&fixture.root, &sources).unwrap();
        assert_eq!(renamed.parsed, 1);
        let generation = fixture.current();
        assert!(generation.rows().iter().any(|row| {
            row.flags & card_flags::CUSTOM_TITLE != 0
                && generation.str(row.custom_title) == "Renamed walrus"
        }));

        // A locked database keeps every Codex card as it is.
        db.execute_batch("PRAGMA locking_mode = EXCLUSIVE; BEGIN EXCLUSIVE;")
            .unwrap();
        let locked = reconcile(&fixture.root, &sources).unwrap();
        assert_eq!(locked.publish, Publish::Unchanged);
        assert_eq!(fixture.current().rows().len(), generation.rows().len());
        db.execute_batch("COMMIT;").unwrap();
    }

    #[test]
    fn a_held_lock_skips_the_pass() {
        let fixture = Fixture::new();
        fixture.claude("-tmp-work", "a.jsonl", "hello alpha");
        let held = PublishLock::try_acquire(&fixture.root).unwrap().unwrap();
        assert_eq!(fixture.run().publish, Publish::Busy);
        assert!(Generation::open(&fixture.root).is_none());
        drop(held);
        assert!(matches!(fixture.run().publish, Publish::Published(_)));
    }

    #[test]
    fn a_large_cold_build_publishes_the_newest_first_then_one_base() {
        let fixture = Fixture::new();
        let count = FIRST_BATCH + 50;
        for i in 0..count {
            let path = fixture.claude("-tmp-many", &format!("{i:04}.jsonl"), &format!("doc{i}"));
            let file = fs::File::options().write(true).open(&path).unwrap();
            file.set_modified(SystemTime::now() - Duration::from_secs(3600 + i as u64))
                .unwrap();
        }
        let outcome = fixture.run();
        assert!(outcome.published >= 3, "{outcome:?}");
        assert!(!fixture.root.join("PROGRESS").exists());
        let generation = fixture.current();
        assert_eq!(generation.rows().len(), count + 2);
        assert_eq!(generation.segs().len(), 1);
        let files = fs::read_dir(fixture.root.join(generation.name()))
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .filter(|name| name.starts_with("delta-"))
            .count();
        assert_eq!(files, 0);
        let (index, _) = generation.open_search_files(Field::Body).unwrap();
        assert_eq!(
            index.prefix_tf("doc1").iter().filter(|&&tf| tf > 0).count(),
            111
        );
    }
}
