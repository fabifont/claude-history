//! Immutable generation directories: build, publish, open and collect.
//!
//! Layout under the store root:
//!
//! ```text
//! LOCK        held by the process that publishes or collects
//! CURRENT     name of the live generation, e.g. "g000012"
//! g000012/
//!   manifest  header | CardRow[] | ProjectRow[] | Fingerprint[] | string heap
//! ```
//!
//! A generation is built in `gNNN.tmp/`, synced and renamed into place before
//! `CURRENT` is replaced through `CURRENT.tmp`, so readers only ever see
//! complete generations. Published files are never modified; collection
//! unlinks whole directories, which leaves existing memory maps valid.

use super::card::{
    CardRow, Fingerprint, Header, MAGIC, NO_DURATION, PREVIEW_CAP, ProjectRow, StrRef, VERSION,
    cap_preview, card_flags, project_flags, source_from_u8, source_to_u8,
};
use crate::history::{Conversation, Source};
use chrono::{Local, TimeZone};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::{self, File};
use std::io::{self, Write};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const CURRENT: &str = "CURRENT";
const MANIFEST: &str = "manifest";
const LOCK: &str = "LOCK";
/// Generations kept behind `CURRENT` for readers that opened them earlier.
const KEEP_BEHIND: u64 = 2;
/// Files modified this close to the start of loading may have changed after
/// the loader read them.
const MTIME_SLACK: Duration = Duration::from_secs(2);

/// A validated, memory-mapped generation manifest.
pub struct Generation {
    name: String,
    map: memmap2::Mmap,
    rows: Range<usize>,
    projects: Range<usize>,
    fingerprints: Range<usize>,
    heap: Range<usize>,
}

/// A project as seen by card filters.
pub struct ProjectView<'a> {
    pub name: Option<&'a str>,
    pub workspace_key: &'a str,
    /// `workspace_key` is a canonical path rather than a Claude directory name.
    pub key_is_path: bool,
}

impl Generation {
    /// Opens the generation named by `root/CURRENT`. Returns `None` when there
    /// is none or it fails validation, in which case callers load from
    /// scratch.
    pub fn open(root: &Path) -> Option<Self> {
        // A concurrent writer can collect the generation between reading
        // CURRENT and opening its manifest; CURRENT names a newer one by then.
        Self::open_current(root).or_else(|| Self::open_current(root))
    }

    fn open_current(root: &Path) -> Option<Self> {
        let name = fs::read_to_string(root.join(CURRENT)).ok()?;
        let name = name.trim();
        generation_number(name)?;
        let file = File::open(root.join(name).join(MANIFEST)).ok()?;
        // SAFETY: published manifests are never written again and are only
        // removed by unlinking, so the mapped bytes cannot change underneath.
        let map = unsafe { memmap2::Mmap::map(&file) }.ok()?;
        Self::validate(name.to_string(), map)
    }

    fn validate(name: String, map: memmap2::Mmap) -> Option<Self> {
        let header_len = std::mem::size_of::<Header>();
        let header: Header = bytemuck::try_pod_read_unaligned(map.get(..header_len)?).ok()?;
        if header.magic != MAGIC
            || header.version != VERSION
            || header.row_size as usize != std::mem::size_of::<CardRow>()
        {
            return None;
        }
        let file_len = map.len() as u64;
        let section = |off: u64, count: u64, size: usize, min_start: u64| -> Option<Range<usize>> {
            if off < min_start || !off.is_multiple_of(8) {
                return None;
            }
            let end = off.checked_add(count.checked_mul(size as u64)?)?;
            (end <= file_len).then_some(off as usize..end as usize)
        };
        let rows = section(
            header.rows_off,
            u64::from(header.row_count),
            std::mem::size_of::<CardRow>(),
            header_len as u64,
        )?;
        let projects = section(
            header.projects_off,
            u64::from(header.project_count),
            std::mem::size_of::<ProjectRow>(),
            rows.end as u64,
        )?;
        let fingerprints = section(
            header.fingerprints_off,
            u64::from(header.row_count),
            std::mem::size_of::<Fingerprint>(),
            projects.end as u64,
        )?;
        let heap = section(header.heap_off, header.heap_len, 1, fingerprints.end as u64)?;

        let generation = Self {
            name,
            map,
            rows,
            projects,
            fingerprints,
            heap,
        };
        let heap_len = header.heap_len;
        let in_heap = |r: StrRef| r.end() <= heap_len;
        let rows: &[CardRow] =
            bytemuck::try_cast_slice(&generation.map[generation.rows.clone()]).ok()?;
        let projects: &[ProjectRow] =
            bytemuck::try_cast_slice(&generation.map[generation.projects.clone()]).ok()?;
        let _: &[Fingerprint] =
            bytemuck::try_cast_slice(&generation.map[generation.fingerprints.clone()]).ok()?;
        let rows_valid = rows.iter().all(|row| {
            row.project_id < header.project_count
                && source_from_u8(row.source).is_some()
                && [
                    row.path,
                    row.session_id,
                    row.custom_title,
                    row.summary,
                    row.model,
                    row.cwd,
                    row.preview_first,
                    row.preview_last,
                ]
                .into_iter()
                .all(in_heap)
        });
        let projects_valid = projects.iter().all(|project| {
            [project.name, project.path, project.workspace_key]
                .into_iter()
                .all(in_heap)
        });
        (rows_valid && projects_valid).then_some(generation)
    }

    /// Directory name of this generation, e.g. `g000012`.
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn rows(&self) -> &[CardRow] {
        bytemuck::cast_slice(&self.map[self.rows.clone()])
    }

    pub fn projects(&self) -> &[ProjectRow] {
        bytemuck::cast_slice(&self.map[self.projects.clone()])
    }

    /// Resolves a heap string. Invalid UTF-8 reads as empty.
    pub fn str(&self, r: StrRef) -> &str {
        let start = self.heap.start + r.off as usize;
        std::str::from_utf8(&self.map[start..start + r.len as usize]).unwrap_or("")
    }

    fn manifest_bytes(&self) -> &[u8] {
        &self.map
    }

    pub fn project(&self, project_id: u32) -> ProjectView<'_> {
        let project = &self.projects()[project_id as usize];
        ProjectView {
            name: (project.flags & project_flags::NAME != 0).then(|| self.str(project.name)),
            workspace_key: self.str(project.workspace_key),
            key_is_path: project.flags & project_flags::PATH_KEY != 0,
        }
    }

    /// Indices of the first `n` rows accepted by `keep`, newest first.
    pub fn head(&self, n: usize, mut keep: impl FnMut(&CardRow) -> bool) -> Vec<u32> {
        self.rows()
            .iter()
            .enumerate()
            .filter(|(_, row)| keep(row))
            .take(n)
            .map(|(index, _)| index as u32)
            .collect()
    }

    /// Builds a list-only `Conversation` for row `index`: everything the list
    /// renders and the actions need, without search text.
    pub fn conversation(&self, index: u32, show_last: bool) -> Conversation {
        let row = &self.rows()[index as usize];
        let project = &self.projects()[row.project_id as usize];
        let optional =
            |flag: u8, r: StrRef| (row.flags & flag != 0).then(|| self.str(r).to_owned());
        let preview_first: Arc<str> = self.str(row.preview_first).into();
        let preview_last: Arc<str> = self.str(row.preview_last).into();
        let empty: Arc<str> = Arc::from("");
        Conversation {
            source: source_from_u8(row.source).unwrap_or(Source::Claude),
            session_id: self.str(row.session_id).to_owned(),
            path: PathBuf::from(self.str(row.path)),
            index: index as usize,
            timestamp: Local
                .timestamp_millis_opt(row.timestamp_ms)
                .single()
                .unwrap_or_else(Local::now),
            preview: if show_last {
                preview_last.clone()
            } else {
                preview_first.clone()
            },
            preview_first,
            preview_last,
            full_text: empty.clone(),
            agent_search_text: empty.clone(),
            semantic_route_text: empty.clone(),
            semantic_turns: Arc::from(Vec::new()),
            semantic_turn_ranges: Vec::new(),
            search_text_lower: empty,
            project_name: (project.flags & project_flags::NAME != 0)
                .then(|| self.str(project.name).to_owned()),
            project_path: (project.flags & project_flags::PATH != 0)
                .then(|| PathBuf::from(self.str(project.path))),
            cwd: optional(card_flags::CWD, row.cwd).map(PathBuf::from),
            message_count: row.message_count as usize,
            parse_errors: Vec::new(),
            summary: optional(card_flags::SUMMARY, row.summary),
            custom_title: optional(card_flags::CUSTOM_TITLE, row.custom_title),
            model: optional(card_flags::MODEL, row.model),
            total_tokens: row.total_tokens,
            duration_minutes: (row.duration_minutes != NO_DURATION)
                .then_some(u64::from(row.duration_minutes)),
        }
    }
}

/// Parses `gNNN` into its generation number.
fn generation_number(name: &str) -> Option<u64> {
    let digits = name.strip_prefix('g')?;
    if digits.is_empty() || digits.len() > 18 || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok()
}

/// Fingerprint of the file behind `path`. Files modified after
/// `loaded_since` (less `MTIME_SLACK`) get an unknown mtime, so a later run
/// sees them as changed.
pub fn fingerprint(path: &Path, loaded_since: SystemTime) -> Fingerprint {
    let Ok(metadata) = fs::metadata(path) else {
        return Fingerprint::default();
    };
    let threshold = loaded_since.checked_sub(MTIME_SLACK).unwrap_or(UNIX_EPOCH);
    let mtime_ns = metadata
        .modified()
        .ok()
        .filter(|modified| *modified < threshold)
        .and_then(|modified| modified.duration_since(UNIX_EPOCH).ok())
        .map_or(0, |since_epoch| {
            i64::try_from(since_epoch.as_nanos()).unwrap_or(0)
        });
    Fingerprint {
        size: metadata.len(),
        mtime_ns,
    }
}

/// String heap that stores each distinct string once.
#[derive(Default)]
struct Heap {
    bytes: Vec<u8>,
    index: HashMap<String, StrRef>,
}

impl Heap {
    fn push(&mut self, text: &str) -> StrRef {
        if text.is_empty() {
            return StrRef::default();
        }
        if let Some(&r) = self.index.get(text) {
            return r;
        }
        let r = StrRef {
            off: u32::try_from(self.bytes.len()).expect("manifest string heap exceeds 4 GiB"),
            len: text.len() as u32,
        };
        self.bytes.extend_from_slice(text.as_bytes());
        self.index.insert(text.to_owned(), r);
        r
    }

    fn push_optional(&mut self, text: Option<&str>, flag: u8, flags: &mut u8) -> StrRef {
        match text {
            Some(text) => {
                *flags |= flag;
                self.push(text)
            }
            None => StrRef::default(),
        }
    }
}

/// What the workspace filter compares for `conversation`; see
/// `ProjectRow::workspace_key`.
fn workspace_key(
    conversation: &Conversation,
    canonical: &mut HashMap<PathBuf, String>,
) -> (String, bool) {
    if conversation.source == Source::Claude {
        let dir = conversation
            .path
            .parent()
            .and_then(|parent| parent.file_name())
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        return (dir, false);
    }
    let key = conversation
        .project_path
        .as_ref()
        .or(conversation.cwd.as_ref())
        .map(|path| {
            canonical
                .entry(path.clone())
                .or_insert_with(|| {
                    path.canonicalize()
                        .unwrap_or_else(|_| path.clone())
                        .to_string_lossy()
                        .into_owned()
                })
                .clone()
        })
        .unwrap_or_default();
    (key, true)
}

fn path_str(path: &Path) -> std::borrow::Cow<'_, str> {
    path.to_string_lossy()
}

/// Serializes `conversations` and their `fingerprints` (in the same order)
/// into manifest bytes. Output depends only on the inputs, so an unchanged
/// corpus produces an identical manifest.
pub fn build_manifest(conversations: &[Conversation], fingerprints: &[Fingerprint]) -> Vec<u8> {
    assert_eq!(conversations.len(), fingerprints.len());
    let mut order: Vec<usize> = (0..conversations.len()).collect();
    order.sort_by(|&a, &b| {
        let (a, b) = (&conversations[a], &conversations[b]);
        b.timestamp
            .cmp(&a.timestamp)
            .then_with(|| a.path.cmp(&b.path))
    });

    // Project ids follow the sorted project keys, so they only change when
    // the set of projects does.
    type ProjectKey = (String, bool, Option<String>, Option<PathBuf>);
    let mut canonical = HashMap::new();
    let project_keys: Vec<ProjectKey> = conversations
        .iter()
        .map(|conversation| {
            let (key, key_is_path) = workspace_key(conversation, &mut canonical);
            (
                key,
                key_is_path,
                conversation.project_name.clone(),
                conversation.project_path.clone(),
            )
        })
        .collect();
    let project_ids: BTreeMap<&ProjectKey, u32> = project_keys
        .iter()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .zip(0..)
        .collect();

    let mut heap = Heap::default();
    let projects: Vec<ProjectRow> = project_ids
        .keys()
        .map(|(key, key_is_path, name, path)| {
            let mut flags = if *key_is_path {
                project_flags::PATH_KEY
            } else {
                0
            };
            let name = heap.push_optional(name.as_deref(), project_flags::NAME, &mut flags);
            let path = heap.push_optional(
                path.as_deref().map(path_str).as_deref(),
                project_flags::PATH,
                &mut flags,
            );
            ProjectRow {
                name,
                path,
                workspace_key: heap.push(key),
                flags,
                _pad: [0; 7],
            }
        })
        .collect();

    let mut rows: Vec<CardRow> = Vec::with_capacity(order.len());
    let mut row_fingerprints = Vec::with_capacity(order.len());
    for &index in &order {
        let conversation = &conversations[index];
        let project_id = project_ids[&project_keys[index]];

        let path = path_str(&conversation.path);
        let mut flags = 0;
        let custom_title = heap.push_optional(
            conversation.custom_title.as_deref(),
            card_flags::CUSTOM_TITLE,
            &mut flags,
        );
        let summary = heap.push_optional(
            conversation.summary.as_deref(),
            card_flags::SUMMARY,
            &mut flags,
        );
        let model =
            heap.push_optional(conversation.model.as_deref(), card_flags::MODEL, &mut flags);
        let cwd = heap.push_optional(
            conversation.cwd.as_deref().map(path_str).as_deref(),
            card_flags::CWD,
            &mut flags,
        );
        rows.push(CardRow {
            key_hash: u64::from_le_bytes(
                blake3::hash(path.as_bytes()).as_bytes()[..8]
                    .try_into()
                    .expect("8-byte slice"),
            ),
            timestamp_ms: conversation.timestamp.timestamp_millis(),
            total_tokens: conversation.total_tokens,
            path: heap.push(&path),
            session_id: heap.push(&conversation.session_id),
            custom_title,
            summary,
            model,
            cwd,
            preview_first: heap.push(cap_preview(&conversation.preview_first)),
            preview_last: heap.push(cap_preview(&conversation.preview_last)),
            project_id,
            message_count: u32::try_from(conversation.message_count).unwrap_or(u32::MAX),
            duration_minutes: conversation
                .duration_minutes
                .map_or(NO_DURATION, |minutes| {
                    u32::try_from(minutes).unwrap_or(NO_DURATION - 1)
                }),
            source: source_to_u8(conversation.source),
            flags,
            _pad: [0; 2],
        });
        row_fingerprints.push(fingerprints[index]);
    }
    debug_assert!(
        rows.iter()
            .all(|row| row.preview_first.len as usize <= PREVIEW_CAP)
    );

    let align = |offset: usize| offset.div_ceil(8) * 8;
    let rows_off = std::mem::size_of::<Header>();
    let projects_off = align(rows_off + std::mem::size_of_val(rows.as_slice()));
    let fingerprints_off = align(projects_off + std::mem::size_of_val(projects.as_slice()));
    let heap_off = align(fingerprints_off + std::mem::size_of_val(row_fingerprints.as_slice()));
    let header = Header {
        magic: MAGIC,
        version: VERSION,
        row_size: std::mem::size_of::<CardRow>() as u32,
        row_count: rows.len() as u32,
        project_count: projects.len() as u32,
        rows_off: rows_off as u64,
        projects_off: projects_off as u64,
        fingerprints_off: fingerprints_off as u64,
        heap_off: heap_off as u64,
        heap_len: heap.bytes.len() as u64,
    };

    let mut out = Vec::with_capacity(heap_off + heap.bytes.len());
    out.extend_from_slice(bytemuck::bytes_of(&header));
    out.extend_from_slice(bytemuck::cast_slice(&rows));
    out.resize(projects_off, 0);
    out.extend_from_slice(bytemuck::cast_slice(&projects));
    out.resize(fingerprints_off, 0);
    out.extend_from_slice(bytemuck::cast_slice(&row_fingerprints));
    out.resize(heap_off, 0);
    out.extend_from_slice(&heap.bytes);
    out
}

#[derive(Debug, Eq, PartialEq)]
pub enum Publish {
    /// `CURRENT` already holds an identical manifest; nothing was written.
    Unchanged,
    /// A new generation with this name is now `CURRENT`.
    Published(String),
    /// Another process holds the lock and is publishing.
    Busy,
}

fn current_matches(root: &Path, manifest: &[u8]) -> bool {
    Generation::open(root).is_some_and(|current| current.manifest_bytes() == manifest)
}

/// Publishes `manifest` as a new generation unless `CURRENT` already holds
/// the same bytes. Writes nothing when unchanged.
pub fn publish(root: &Path, manifest: &[u8]) -> io::Result<Publish> {
    if current_matches(root, manifest) {
        return Ok(Publish::Unchanged);
    }
    fs::create_dir_all(root)?;
    let lock = File::options()
        .create(true)
        .truncate(false)
        .write(true)
        .open(root.join(LOCK))?;
    match lock.try_lock() {
        Ok(()) => {}
        Err(fs::TryLockError::WouldBlock) => return Ok(Publish::Busy),
        Err(fs::TryLockError::Error(err)) => return Err(err),
    }
    // Another process may have published the same corpus while we waited.
    if current_matches(root, manifest) {
        return Ok(Publish::Unchanged);
    }

    let existing = generation_dirs(root)?;
    let number = existing
        .iter()
        .map(|(number, _)| number + 1)
        .max()
        .unwrap_or(1);
    let name = format!("g{number:06}");
    let tmp = root.join(format!("{name}.tmp"));
    if tmp.exists() {
        fs::remove_dir_all(&tmp)?;
    }
    fs::create_dir(&tmp)?;
    write_synced(&tmp.join(MANIFEST), manifest)?;
    sync_dir(&tmp);
    fs::rename(&tmp, root.join(&name))?;
    sync_dir(root);
    let current_tmp = root.join(format!("{CURRENT}.tmp"));
    write_synced(&current_tmp, format!("{name}\n").as_bytes())?;
    fs::rename(&current_tmp, root.join(CURRENT))?;
    sync_dir(root);

    collect(root, number, &existing);
    drop(lock);
    Ok(Publish::Published(name))
}

/// Published generation directories and their numbers.
fn generation_dirs(root: &Path) -> io::Result<Vec<(u64, PathBuf)>> {
    let mut dirs = Vec::new();
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let name = entry.file_name();
        if let Some(number) = name.to_str().and_then(generation_number) {
            dirs.push((number, entry.path()));
        }
    }
    Ok(dirs)
}

/// Removes generations more than `KEEP_BEHIND` older than `current` and
/// leftover build directories. Callers hold `LOCK`.
fn collect(root: &Path, current: u64, existing: &[(u64, PathBuf)]) {
    for (number, path) in existing {
        if number + KEEP_BEHIND < current {
            let _ = fs::remove_dir_all(path);
        }
    }
    let Ok(entries) = fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        if name
            .to_str()
            .and_then(|name| name.strip_suffix(".tmp"))
            .is_some_and(|name| generation_number(name).is_some())
        {
            let _ = fs::remove_dir_all(entry.path());
        }
    }
}

pub(super) fn write_synced(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let mut file = File::create(path)?;
    file.write_all(bytes)?;
    file.sync_all()
}

/// Makes renames inside `dir` durable. Not every platform can open a
/// directory for syncing, so failures are ignored.
fn sync_dir(dir: &Path) {
    if let Ok(dir) = File::open(dir) {
        let _ = dir.sync_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::test_support::{conversation, unknown_fingerprints};

    fn sample() -> Vec<Conversation> {
        let mut older = conversation("/p/-Users-me-app/older.jsonl", 1_000);
        older.custom_title = Some("Older title".into());
        older.preview_first = "é".repeat(PREVIEW_CAP).into();
        let mut newer = conversation("/p/-Users-me-app/newer.jsonl", 2_000);
        newer.summary = Some("Newer summary".into());
        newer.duration_minutes = Some(7);
        newer.model = Some("claude-opus".into());
        vec![older, newer]
    }

    fn publish_sample(root: &Path, conversations: &[Conversation]) -> Publish {
        let manifest = build_manifest(conversations, &unknown_fingerprints(conversations.len()));
        publish(root, &manifest).unwrap()
    }

    #[test]
    fn round_trips_cards_newest_first() {
        let dir = tempfile::tempdir().unwrap();
        let conversations = sample();
        assert!(matches!(
            publish_sample(dir.path(), &conversations),
            Publish::Published(_)
        ));

        let generation = Generation::open(dir.path()).unwrap();
        assert_eq!(generation.rows().len(), 2);
        let newest = generation.conversation(0, false);
        assert_eq!(newest.path, PathBuf::from("/p/-Users-me-app/newer.jsonl"));
        assert_eq!(newest.summary.as_deref(), Some("Newer summary"));
        assert_eq!(newest.custom_title, None);
        assert_eq!(newest.duration_minutes, Some(7));
        assert_eq!(newest.model.as_deref(), Some("claude-opus"));
        assert_eq!(newest.timestamp, conversations[1].timestamp);
        assert_eq!(newest.project_name.as_deref(), Some("app"));

        let older = generation.conversation(1, false);
        assert_eq!(older.custom_title.as_deref(), Some("Older title"));
        assert_eq!(older.preview.len(), PREVIEW_CAP);
        assert_eq!(older.duration_minutes, None);
        assert_eq!(&*generation.conversation(1, true).preview, "last");

        let project = generation.project(generation.rows()[0].project_id);
        assert_eq!(project.workspace_key, "-Users-me-app");
        assert!(!project.key_is_path);
    }

    #[test]
    fn head_walks_rows_in_order_until_n_match() {
        let dir = tempfile::tempdir().unwrap();
        let conversations: Vec<_> = (0..10)
            .map(|i| conversation(&format!("/p/proj/{i}.jsonl"), i64::from(i) * 1000))
            .collect();
        publish_sample(dir.path(), &conversations);
        let generation = Generation::open(dir.path()).unwrap();
        let head = generation.head(3, |row| row.timestamp_ms % 2000 == 0);
        let times: Vec<_> = head
            .iter()
            .map(|&i| generation.rows()[i as usize].timestamp_ms)
            .collect();
        assert_eq!(times, vec![8000, 6000, 4000]);
    }

    #[test]
    fn unchanged_corpus_writes_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let conversations = sample();
        assert_eq!(
            publish_sample(dir.path(), &conversations),
            Publish::Published("g000001".into())
        );
        let before = fs::metadata(dir.path().join(CURRENT))
            .unwrap()
            .modified()
            .unwrap();
        let mut reversed = conversations.clone();
        reversed.reverse();
        assert_eq!(publish_sample(dir.path(), &reversed), Publish::Unchanged);
        let after = fs::metadata(dir.path().join(CURRENT))
            .unwrap()
            .modified()
            .unwrap();
        assert_eq!(before, after);
        assert_eq!(generation_dirs(dir.path()).unwrap().len(), 1);
    }

    #[test]
    fn keeps_two_generations_behind_current() {
        let dir = tempfile::tempdir().unwrap();
        let mut conversations = sample();
        for round in 0..5 {
            conversations[0].message_count = round;
            publish_sample(dir.path(), &conversations);
        }
        let mut numbers: Vec<_> = generation_dirs(dir.path())
            .unwrap()
            .into_iter()
            .map(|(number, _)| number)
            .collect();
        numbers.sort();
        assert_eq!(numbers, vec![3, 4, 5]);
        assert_eq!(Generation::open(dir.path()).unwrap().name(), "g000005");
    }

    #[test]
    fn an_open_generation_survives_collection() {
        let dir = tempfile::tempdir().unwrap();
        let mut conversations = sample();
        publish_sample(dir.path(), &conversations);
        let first = Generation::open(dir.path()).unwrap();
        for round in 1..5 {
            conversations[0].message_count = round;
            publish_sample(dir.path(), &conversations);
        }
        assert!(!dir.path().join(first.name()).exists());
        assert_eq!(first.conversation(1, false).message_count, 3);
    }

    #[test]
    fn rejects_corrupt_or_foreign_manifests() {
        let dir = tempfile::tempdir().unwrap();
        publish_sample(dir.path(), &sample());
        let manifest_path = dir.path().join("g000001").join(MANIFEST);
        let good = fs::read(&manifest_path).unwrap();

        let corrupt = |mutate: &dyn Fn(&mut Vec<u8>)| {
            let mut bytes = good.clone();
            mutate(&mut bytes);
            fs::remove_file(&manifest_path).unwrap();
            fs::write(&manifest_path, &bytes).unwrap();
            Generation::open(dir.path()).is_none()
        };
        assert!(corrupt(&|bytes| bytes[0] = b'X'), "magic");
        assert!(corrupt(&|bytes| bytes[8] = 99), "version");
        assert!(
            corrupt(&|bytes| bytes.truncate(bytes.len() - 1)),
            "heap past end"
        );
        assert!(corrupt(&|bytes| bytes.truncate(40)), "short header");
        let rows_off = std::mem::size_of::<Header>();
        // CardRow::path.len of the first row.
        let path_len = rows_off + 24 + 4;
        assert!(
            corrupt(&|bytes| bytes[path_len..path_len + 4].copy_from_slice(&u32::MAX.to_le_bytes())),
            "string ref out of bounds"
        );
        // CardRow::project_id of the first row.
        let project_id = rows_off + 88;
        assert!(
            corrupt(&|bytes| bytes[project_id..project_id + 4].copy_from_slice(&9u32.to_le_bytes())),
            "project id out of range"
        );
        assert!(!corrupt(&|_| {}), "restored manifest opens");

        fs::write(dir.path().join(CURRENT), "../elsewhere").unwrap();
        assert!(Generation::open(dir.path()).is_none());
        assert!(Generation::open(&dir.path().join("missing")).is_none());
    }

    #[test]
    fn concurrent_publishers_never_expose_partial_generations() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().to_path_buf();
        std::thread::scope(|scope| {
            for writer in 0..4 {
                let root = root.clone();
                scope.spawn(move || {
                    let mut conversations = sample();
                    for round in 0..10 {
                        conversations[0].message_count = writer * 100 + round;
                        publish_sample(&root, &conversations);
                    }
                });
            }
            let root = root.clone();
            scope.spawn(move || {
                for _ in 0..200 {
                    if let Some(generation) = Generation::open(&root) {
                        assert_eq!(generation.rows().len(), 2);
                    }
                }
            });
        });
        assert!(Generation::open(&root).is_some());
        assert!(generation_dirs(&root).unwrap().len() <= 3);
    }

    #[test]
    fn fingerprints_mark_recently_modified_files_unknown() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("session.jsonl");
        fs::write(&path, "{}\n").unwrap();
        let recent = fingerprint(&path, SystemTime::now());
        assert_eq!(
            recent,
            Fingerprint {
                size: 3,
                mtime_ns: 0
            }
        );
        let later = fingerprint(&path, SystemTime::now() + Duration::from_secs(60));
        assert_eq!(later.size, 3);
        assert_ne!(later.mtime_ns, 0);
        assert_eq!(
            fingerprint(&dir.path().join("missing"), SystemTime::now()),
            Fingerprint::default()
        );
    }

    /// `head` and `open` cost must not grow with the corpus beyond the
    /// validation pass. Run with `--ignored --nocapture` to see timings.
    #[test]
    #[ignore]
    fn open_and_head_timing_on_large_manifest() {
        let dir = tempfile::tempdir().unwrap();
        for count in [1_000usize, 10_000, 100_000] {
            let conversations: Vec<_> = (0..count)
                .map(|i| {
                    let mut c = conversation(&format!("/p/proj-{}/{i}.jsonl", i % 500), i as i64);
                    c.preview_first = format!("first preview {i} {}", "x".repeat(300)).into();
                    c.preview_last = format!("last preview {i} {}", "y".repeat(300)).into();
                    c
                })
                .collect();
            let manifest = build_manifest(&conversations, &unknown_fingerprints(count));
            let root = dir.path().join(count.to_string());
            publish(&root, &manifest).unwrap();
            let started = std::time::Instant::now();
            let generation = Generation::open(&root).unwrap();
            let opened = started.elapsed();
            let head = generation.head(500, |_| true);
            let cards: Vec<_> = head
                .iter()
                .map(|&i| generation.conversation(i, true))
                .collect();
            let total = started.elapsed();
            assert_eq!(cards.len(), 500);
            eprintln!(
                "rows {count:>6} manifest {:>6.1} MB open {:>6.2} ms open+head(500) {:>6.2} ms",
                manifest.len() as f64 / 1e6,
                opened.as_secs_f64() * 1e3,
                total.as_secs_f64() * 1e3
            );
        }
    }
}
