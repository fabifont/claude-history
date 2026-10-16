//! Immutable generation directories: build, publish, open and collect.
//!
//! Layout under the store root:
//!
//! ```text
//! LOCK            held by the process that publishes or collects
//! CURRENT         name of the live generation, e.g. "g000012"
//! g000012/
//!   manifest      header | CardRow[] | ProjectRow[] | Fingerprint[] |
//!                 IndexLoc[] | SemRow[] | IndexSegRow[] | IgnoredRow[] |
//!                 string heap
//!   text-NNNN.zst conversation text, one zstd frame per card (see `text`)
//!   sem-NNNN.zst  semantic data, one zstd frame per card (see `semantic`)
//!   base.*        suffix index segment 0 (see `lexindex`)
//!   delta-NNNN.*  further index segments (see `segments`)
//! ```
//!
//! A generation is built in `gNNN.tmp/`, synced and renamed into place before
//! `CURRENT` is replaced through `CURRENT.tmp`, so readers only ever see
//! complete generations, whose cards, text and index always agree. Published
//! files are never modified, so a generation shares the unchanged text and
//! index segments of its predecessor as hard links. Collection unlinks whole
//! directories, which leaves existing memory maps valid.

use super::card::{
    BlobRef, CardRow, Fingerprint, Header, IgnoredRow, IndexLoc, IndexSegRow, MAGIC, NO_DURATION,
    PREVIEW_CAP, ProjectRow, SemRow, StrRef, VERSION, cap_preview, card_flags, project_flags,
    source_from_u8, source_to_u8,
};
use super::lexindex::{self, LexIndex};
use super::segments::IndexSet;
use super::semantic::{self, SemanticDoc};
use super::text::{self, TextSegments};
use crate::history::{Conversation, Source};
use crate::text_match::normalize_for_search;
use chrono::{Local, TimeZone};
use std::borrow::Cow;
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
const KEEP_BEHIND: u64 = 1;
/// Files modified this close to the start of loading may have changed after
/// the loader read them.
const MTIME_SLACK: Duration = Duration::from_secs(2);

/// A validated, memory-mapped generation manifest.
pub struct Generation {
    name: String,
    dir: PathBuf,
    map: memmap2::Mmap,
    rows: Range<usize>,
    projects: Range<usize>,
    fingerprints: Range<usize>,
    locs: Range<usize>,
    sems: Range<usize>,
    segs: Range<usize>,
    ignored: Range<usize>,
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

    /// Name of the generation `root/CURRENT` points at, without opening
    /// it.
    pub fn current_name(root: &Path) -> Option<String> {
        let name = fs::read_to_string(root.join(CURRENT)).ok()?;
        let name = name.trim();
        generation_number(name)?;
        Some(name.to_string())
    }

    fn open_current(root: &Path) -> Option<Self> {
        let name = Self::current_name(root)?;
        let name = name.as_str();
        let dir = root.join(name);
        let file = File::open(dir.join(MANIFEST)).ok()?;
        // SAFETY: published manifests are never written again and are only
        // removed by unlinking, so the mapped bytes cannot change underneath.
        let map = unsafe { memmap2::Mmap::map(&file) }.ok()?;
        Self::validate(name.to_string(), dir, map)
    }

    fn validate(name: String, dir: PathBuf, map: memmap2::Mmap) -> Option<Self> {
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
        let locs = section(
            header.locs_off,
            u64::from(header.row_count),
            std::mem::size_of::<IndexLoc>(),
            fingerprints.end as u64,
        )?;
        let sems = section(
            header.sems_off,
            u64::from(header.row_count),
            std::mem::size_of::<SemRow>(),
            locs.end as u64,
        )?;
        let segs = section(
            header.segs_off,
            u64::from(header.seg_count),
            std::mem::size_of::<IndexSegRow>(),
            sems.end as u64,
        )?;
        let ignored = section(
            header.ignored_off,
            u64::from(header.ignored_count),
            std::mem::size_of::<IgnoredRow>(),
            segs.end as u64,
        )?;
        let heap = section(header.heap_off, header.heap_len, 1, ignored.end as u64)?;

        let generation = Self {
            name,
            dir,
            map,
            rows,
            projects,
            fingerprints,
            locs,
            sems,
            segs,
            ignored,
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
        let _: &[IndexLoc] =
            bytemuck::try_cast_slice(&generation.map[generation.locs.clone()]).ok()?;
        let _: &[SemRow] =
            bytemuck::try_cast_slice(&generation.map[generation.sems.clone()]).ok()?;
        let segs: &[IndexSegRow] =
            bytemuck::try_cast_slice(&generation.map[generation.segs.clone()]).ok()?;
        let ignored: &[IgnoredRow] =
            bytemuck::try_cast_slice(&generation.map[generation.ignored.clone()]).ok()?;
        let segs_valid = segs.windows(2).all(|pair| pair[0].id < pair[1].id);
        let ignored_valid = ignored.iter().all(|row| in_heap(row.path));
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
        (rows_valid && projects_valid && segs_valid && ignored_valid).then_some(generation)
    }

    /// Directory name of this generation, e.g. `g000012`.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Opens this generation's index segments and text segments. `None`
    /// when any is missing or invalid.
    pub fn open_search_files(&self) -> Option<(IndexSet, TextSegments)> {
        let index = IndexSet::open(&self.dir, self.segs(), self.locs())?;
        let text = TextSegments::open(&self.dir, text::TEXT, self.rows().iter().map(|r| r.text))?;
        Some((index, text))
    }

    /// Opens this generation's semantic segments. `None` when one is
    /// missing or invalid.
    pub fn open_semantic_files(&self) -> Option<TextSegments> {
        TextSegments::open(&self.dir, text::SEM, self.sems().iter().map(|r| r.blob))
    }

    pub fn rows(&self) -> &[CardRow] {
        bytemuck::cast_slice(&self.map[self.rows.clone()])
    }

    /// Native file state of each card.
    pub fn fingerprints(&self) -> &[Fingerprint] {
        bytemuck::cast_slice(&self.map[self.fingerprints.clone()])
    }

    /// Index location of each card.
    pub fn locs(&self) -> &[IndexLoc] {
        bytemuck::cast_slice(&self.map[self.locs.clone()])
    }

    /// Semantic data location of each card.
    pub fn sems(&self) -> &[SemRow] {
        bytemuck::cast_slice(&self.map[self.sems.clone()])
    }

    pub fn segs(&self) -> &[IndexSegRow] {
        bytemuck::cast_slice(&self.map[self.segs.clone()])
    }

    pub fn ignored(&self) -> &[IgnoredRow] {
        bytemuck::cast_slice(&self.map[self.ignored.clone()])
    }

    /// Ignored files as `(path, fingerprint)`, sorted by path.
    pub fn ignored_entries(&self) -> Vec<(String, Fingerprint)> {
        self.ignored()
            .iter()
            .map(|row| (self.str(row.path).to_owned(), row.fingerprint))
            .collect()
    }

    pub fn projects(&self) -> &[ProjectRow] {
        bytemuck::cast_slice(&self.map[self.projects.clone()])
    }

    /// Resolves a heap string. Invalid UTF-8 reads as empty.
    pub fn str(&self, r: StrRef) -> &str {
        let start = self.heap.start + r.off as usize;
        std::str::from_utf8(&self.map[start..start + r.len as usize]).unwrap_or("")
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
#[cfg(test)]
pub fn fingerprint(path: &Path, loaded_since: SystemTime) -> Fingerprint {
    let Ok(metadata) = fs::metadata(path) else {
        return Fingerprint::default();
    };
    fingerprint_from(metadata.len(), metadata.modified().ok(), loaded_since)
}

/// `fingerprint` of a file with `size` and `modified` time.
pub fn fingerprint_from(
    size: u64,
    modified: Option<SystemTime>,
    loaded_since: SystemTime,
) -> Fingerprint {
    let threshold = loaded_since.checked_sub(MTIME_SLACK).unwrap_or(UNIX_EPOCH);
    let mtime_ns = modified
        .filter(|modified| *modified < threshold)
        .and_then(|modified| modified.duration_since(UNIX_EPOCH).ok())
        .map_or(0, |since_epoch| {
            i64::try_from(since_epoch.as_nanos()).unwrap_or(0)
        });
    Fingerprint { size, mtime_ns }
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

/// Project of a card: workspace key, whether the key is a canonical path,
/// `Conversation::project_name` and `Conversation::project_path`. Project
/// ids follow the sorted keys, so they only change when the set of projects
/// does.
type ProjectKey = (String, bool, Option<String>, Option<String>);

/// Everything a card stores about its conversation, independent of where
/// its text and index live. Equal data means an identical card.
#[derive(Clone, Debug, PartialEq)]
pub struct CardData {
    project: ProjectKey,
    timestamp_ms: i64,
    total_tokens: u64,
    text_hash: u64,
    sem_hash: u64,
    path: String,
    session_id: String,
    custom_title: Option<String>,
    summary: Option<String>,
    model: Option<String>,
    cwd: Option<String>,
    preview_first: String,
    preview_last: String,
    message_count: u32,
    duration_minutes: u32,
    source: u8,
}

impl CardData {
    /// `canonical` caches canonicalized project paths across calls;
    /// `sem_hash` is the `SemRow::hash` of the conversation's semantic data.
    pub fn from_conversation(
        conversation: &Conversation,
        canonical: &mut HashMap<PathBuf, String>,
        sem_hash: u64,
    ) -> Self {
        let (key, key_is_path) = workspace_key(conversation, canonical);
        Self {
            project: (
                key,
                key_is_path,
                conversation.project_name.clone(),
                conversation
                    .project_path
                    .as_deref()
                    .map(|path| path_str(path).into_owned()),
            ),
            timestamp_ms: conversation.timestamp.timestamp_millis(),
            total_tokens: conversation.total_tokens,
            text_hash: hash64(conversation.full_text.as_bytes()),
            sem_hash,
            path: path_str(&conversation.path).into_owned(),
            session_id: conversation.session_id.clone(),
            custom_title: conversation.custom_title.clone(),
            summary: conversation.summary.clone(),
            model: conversation.model.clone(),
            cwd: conversation
                .cwd
                .as_deref()
                .map(|path| path_str(path).into_owned()),
            preview_first: cap_preview(&conversation.preview_first).to_owned(),
            preview_last: cap_preview(&conversation.preview_last).to_owned(),
            message_count: u32::try_from(conversation.message_count).unwrap_or(u32::MAX),
            duration_minutes: conversation
                .duration_minutes
                .map_or(NO_DURATION, |minutes| {
                    u32::try_from(minutes).unwrap_or(NO_DURATION - 1)
                }),
            source: source_to_u8(conversation.source),
        }
    }

    fn from_row(generation: &Generation, index: usize) -> Self {
        let row = &generation.rows()[index];
        let project = &generation.projects()[row.project_id as usize];
        let str = |r: StrRef| generation.str(r).to_owned();
        let card = |flag: u8, r: StrRef| (row.flags & flag != 0).then(|| str(r));
        let project_field = |flag: u8, r: StrRef| (project.flags & flag != 0).then(|| str(r));
        Self {
            project: (
                str(project.workspace_key),
                project.flags & project_flags::PATH_KEY != 0,
                project_field(project_flags::NAME, project.name),
                project_field(project_flags::PATH, project.path),
            ),
            timestamp_ms: row.timestamp_ms,
            total_tokens: row.total_tokens,
            text_hash: row.text_hash,
            sem_hash: generation.sems()[index].hash,
            path: str(row.path),
            session_id: str(row.session_id),
            custom_title: card(card_flags::CUSTOM_TITLE, row.custom_title),
            summary: card(card_flags::SUMMARY, row.summary),
            model: card(card_flags::MODEL, row.model),
            cwd: card(card_flags::CWD, row.cwd),
            preview_first: str(row.preview_first),
            preview_last: str(row.preview_last),
            message_count: row.message_count,
            duration_minutes: row.duration_minutes,
            source: row.source,
        }
    }
}

/// A card with where its text, semantic data and index doc live.
struct Entry {
    card: CardData,
    fingerprint: Fingerprint,
    text: BlobRef,
    sem: BlobRef,
    loc: IndexLoc,
}

pub(super) fn hash64(bytes: &[u8]) -> u64 {
    u64::from_le_bytes(
        blake3::hash(bytes).as_bytes()[..8]
            .try_into()
            .expect("8-byte slice"),
    )
}

/// Serializes `entries` (sorted here, newest first then by path) with the
/// index segments `segs` and the `ignored` files. Output depends only on the
/// inputs.
fn build_manifest(
    entries: &mut [Entry],
    segs: &[IndexSegRow],
    ignored: &[(String, Fingerprint)],
) -> Vec<u8> {
    entries.sort_by(|a, b| {
        b.card
            .timestamp_ms
            .cmp(&a.card.timestamp_ms)
            .then_with(|| a.card.path.cmp(&b.card.path))
    });
    let project_ids: BTreeMap<&ProjectKey, u32> = entries
        .iter()
        .map(|entry| &entry.card.project)
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
            let path = heap.push_optional(path.as_deref(), project_flags::PATH, &mut flags);
            ProjectRow {
                name,
                path,
                workspace_key: heap.push(key),
                flags,
                _pad: [0; 7],
            }
        })
        .collect();

    let mut rows: Vec<CardRow> = Vec::with_capacity(entries.len());
    for Entry { card, text, .. } in entries.iter() {
        let mut flags = 0;
        let custom_title = heap.push_optional(
            card.custom_title.as_deref(),
            card_flags::CUSTOM_TITLE,
            &mut flags,
        );
        let summary = heap.push_optional(card.summary.as_deref(), card_flags::SUMMARY, &mut flags);
        let model = heap.push_optional(card.model.as_deref(), card_flags::MODEL, &mut flags);
        let cwd = heap.push_optional(card.cwd.as_deref(), card_flags::CWD, &mut flags);
        rows.push(CardRow {
            key_hash: key_hash(&card.path),
            timestamp_ms: card.timestamp_ms,
            total_tokens: card.total_tokens,
            text_hash: card.text_hash,
            text: *text,
            path: heap.push(&card.path),
            session_id: heap.push(&card.session_id),
            custom_title,
            summary,
            model,
            cwd,
            preview_first: heap.push(&card.preview_first),
            preview_last: heap.push(&card.preview_last),
            project_id: project_ids[&card.project],
            message_count: card.message_count,
            duration_minutes: card.duration_minutes,
            source: card.source,
            flags,
            _pad: [0; 2],
        });
    }
    debug_assert!(
        rows.iter()
            .all(|row| row.preview_first.len as usize <= PREVIEW_CAP)
    );
    let fingerprints: Vec<Fingerprint> = entries.iter().map(|entry| entry.fingerprint).collect();
    let locs: Vec<IndexLoc> = entries.iter().map(|entry| entry.loc).collect();
    let sems: Vec<SemRow> = entries
        .iter()
        .map(|entry| SemRow {
            blob: entry.sem,
            hash: entry.card.sem_hash,
        })
        .collect();
    let ignored: Vec<IgnoredRow> = ignored
        .iter()
        .map(|(path, fingerprint)| IgnoredRow {
            path: heap.push(path),
            fingerprint: *fingerprint,
        })
        .collect();

    let align = |offset: usize| offset.div_ceil(8) * 8;
    let rows_off = std::mem::size_of::<Header>();
    let projects_off = align(rows_off + std::mem::size_of_val(rows.as_slice()));
    let fingerprints_off = align(projects_off + std::mem::size_of_val(projects.as_slice()));
    let locs_off = align(fingerprints_off + std::mem::size_of_val(fingerprints.as_slice()));
    let sems_off = align(locs_off + std::mem::size_of_val(locs.as_slice()));
    let segs_off = align(sems_off + std::mem::size_of_val(sems.as_slice()));
    let ignored_off = align(segs_off + std::mem::size_of_val(segs));
    let heap_off = align(ignored_off + std::mem::size_of_val(ignored.as_slice()));
    let header = Header {
        magic: MAGIC,
        version: VERSION,
        row_size: std::mem::size_of::<CardRow>() as u32,
        row_count: rows.len() as u32,
        project_count: projects.len() as u32,
        rows_off: rows_off as u64,
        projects_off: projects_off as u64,
        fingerprints_off: fingerprints_off as u64,
        locs_off: locs_off as u64,
        sems_off: sems_off as u64,
        segs_off: segs_off as u64,
        ignored_off: ignored_off as u64,
        heap_off: heap_off as u64,
        heap_len: heap.bytes.len() as u64,
        seg_count: segs.len() as u32,
        ignored_count: ignored.len() as u32,
    };

    let mut out = Vec::with_capacity(heap_off + heap.bytes.len());
    out.extend_from_slice(bytemuck::bytes_of(&header));
    out.extend_from_slice(bytemuck::cast_slice(&rows));
    out.resize(projects_off, 0);
    out.extend_from_slice(bytemuck::cast_slice(&projects));
    out.resize(fingerprints_off, 0);
    out.extend_from_slice(bytemuck::cast_slice(&fingerprints));
    out.resize(locs_off, 0);
    out.extend_from_slice(bytemuck::cast_slice(&locs));
    out.resize(sems_off, 0);
    out.extend_from_slice(bytemuck::cast_slice(&sems));
    out.resize(segs_off, 0);
    out.extend_from_slice(bytemuck::cast_slice(segs));
    out.resize(ignored_off, 0);
    out.extend_from_slice(bytemuck::cast_slice(&ignored));
    out.resize(heap_off, 0);
    out.extend_from_slice(&heap.bytes);
    out
}

/// `CardRow::key_hash` of the conversation stored at `path`.
pub fn key_hash(path: &str) -> u64 {
    hash64(path.as_bytes())
}

#[derive(Debug, Eq, PartialEq)]
pub enum Publish {
    /// `CURRENT` already holds the same state; nothing was written.
    Unchanged,
    /// A new generation with this name is now `CURRENT`.
    Published(String),
    /// Another process holds the lock and is publishing.
    Busy,
}

/// Exclusive right to publish and collect generations under one root, held
/// until dropped.
pub struct PublishLock {
    root: PathBuf,
    _file: File,
}

impl PublishLock {
    /// Takes `LOCK` under `root` without waiting; `None` when another
    /// process or thread holds it.
    pub fn try_acquire(root: &Path) -> io::Result<Option<Self>> {
        fs::create_dir_all(root)?;
        let file = File::options()
            .create(true)
            .truncate(false)
            .write(true)
            .open(root.join(LOCK))?;
        match file.try_lock() {
            Ok(()) => Ok(Some(Self {
                root: root.to_path_buf(),
                _file: file,
            })),
            Err(fs::TryLockError::WouldBlock) => Ok(None),
            Err(fs::TryLockError::Error(err)) => Err(err),
        }
    }
}

/// One conversation of the next generation.
pub enum Doc {
    /// Row `row` of the base generation, unchanged apart from `fingerprint`.
    Keep { row: u32, fingerprint: Fingerprint },
    /// A conversation parsed from its native file.
    New {
        conversation: Box<Conversation>,
        fingerprint: Fingerprint,
    },
}

/// The next state of the store: every conversation and every ignored file.
pub struct Update<'a> {
    /// `CURRENT` when it opened with its search files, else `None`.
    pub base: Option<&'a Generation>,
    pub docs: Vec<Doc>,
    /// Ignored files and their fingerprints.
    pub ignored: Vec<(String, Fingerprint)>,
}

/// What a publish wrote.
#[derive(Debug, Default, Clone, Copy)]
pub struct Written {
    /// Bytes of new files; linked files are not counted.
    pub bytes: u64,
    /// Conversations whose text and index were written.
    pub docs: usize,
}

/// Publishes `update` as a new generation under the lock's root. New
/// conversations identical to a base card keep that card. When nothing
/// differs from `base`, nothing is written. Otherwise the new conversations'
/// text goes into one new text segment, their semantic data into one new
/// semantic segment and their index into one new index segment; unchanged
/// segments of `base` are linked, not copied.
pub fn publish_update(lock: &PublishLock, update: Update) -> io::Result<(Publish, Written)> {
    let Update {
        base,
        docs,
        mut ignored,
    } = update;
    let base_rows: &[CardRow] = base.map_or(&[], |base| base.rows());
    let by_key: HashMap<u64, u32> = if docs.iter().any(|doc| matches!(doc, Doc::New { .. })) {
        base_rows
            .iter()
            .enumerate()
            .map(|(row, card)| (card.key_hash, row as u32))
            .collect()
    } else {
        HashMap::new()
    };

    let mut canonical = HashMap::new();
    let mut kept: Vec<(u32, Fingerprint)> = Vec::new();
    let mut added: Vec<(CardData, Fingerprint, Box<Conversation>, Vec<u8>)> = Vec::new();
    for doc in docs {
        match doc {
            Doc::Keep { row, fingerprint } => kept.push((row, fingerprint)),
            Doc::New {
                conversation,
                fingerprint,
            } => {
                let sem = SemanticDoc::of(&conversation).encode();
                let card = CardData::from_conversation(
                    &conversation,
                    &mut canonical,
                    semantic::hash(&sem),
                );
                let same = base
                    .zip(by_key.get(&key_hash(&card.path)))
                    .filter(|(base, row)| CardData::from_row(base, **row as usize) == card);
                match same {
                    Some((_, &row)) => kept.push((row, fingerprint)),
                    None => added.push((card, fingerprint, conversation, sem)),
                }
            }
        }
    }
    kept.sort_unstable_by_key(|&(row, _)| row);
    kept.dedup_by_key(|(row, _)| *row);
    ignored.sort_by(|a, b| a.0.cmp(&b.0));
    ignored.dedup_by(|a, b| a.0 == b.0);

    if let Some(base) = base
        && added.is_empty()
        && kept.len() == base_rows.len()
        && kept
            .iter()
            .all(|&(row, fingerprint)| base.fingerprints()[row as usize] == fingerprint)
        && base.ignored_entries() == ignored
    {
        return Ok((Publish::Unchanged, Written::default()));
    }

    let mut entries: Vec<Entry> = Vec::with_capacity(kept.len() + added.len());
    let mut segs: Vec<IndexSegRow> = Vec::new();
    let mut text_segs: BTreeSet<u32> = BTreeSet::new();
    let mut sem_segs: BTreeSet<u32> = BTreeSet::new();
    if let Some(base) = base {
        let locs = base.locs();
        let sems = base.sems();
        for &(row, fingerprint) in &kept {
            let card = &base_rows[row as usize];
            let sem = sems[row as usize].blob;
            if card.text.len > 0 {
                text_segs.insert(card.text.seg);
            }
            if sem.len > 0 {
                sem_segs.insert(sem.seg);
            }
            entries.push(Entry {
                card: CardData::from_row(base, row as usize),
                fingerprint,
                text: card.text,
                sem,
                loc: locs[row as usize],
            });
        }
        let used: BTreeSet<u32> = kept
            .iter()
            .map(|&(row, _)| locs[row as usize].seg)
            .collect();
        segs.extend(base.segs().iter().filter(|seg| used.contains(&seg.id)));
    }

    let staged = Staged::new(lock)?;
    if let Some(base) = base {
        for &seg in &text_segs {
            staged.link(&base.dir, &text::segment_file(seg))?;
        }
        for &seg in &sem_segs {
            staged.link(&base.dir, &text::kind_file(text::SEM, seg))?;
        }
        for seg in &segs {
            for file in lexindex::segment_files(seg.id) {
                staged.link(&base.dir, &file)?;
            }
        }
    }
    let mut written = Written::default();
    if !added.is_empty() {
        let text_seg = text_segs.last().map_or(0, |seg| seg + 1);
        let index_seg = segs.last().map_or(0, |seg| seg.id + 1);
        let texts: Vec<&str> = added
            .iter()
            .map(|(_, _, conversation, _)| &*conversation.full_text)
            .collect();
        let text_path = text::segment_path(&staged.dir, text_seg);
        let refs = text::write_segment(&text_path, text_seg, &texts)?;
        written.bytes += fs::metadata(&text_path)?.len();
        let sem_refs = if added.iter().any(|(_, _, _, sem)| !sem.is_empty()) {
            let sem_seg = sem_segs.last().map_or(0, |seg| seg + 1);
            let sems: Vec<&[u8]> = added.iter().map(|(_, _, _, sem)| sem.as_slice()).collect();
            let path = staged.dir.join(text::kind_file(text::SEM, sem_seg));
            let refs = text::write_segment(&path, sem_seg, &sems)?;
            written.bytes += fs::metadata(&path)?.len();
            refs
        } else {
            vec![BlobRef::default(); added.len()]
        };
        let built = lexindex::build(texts.len(), |doc| {
            Cow::Owned(normalize_for_search(texts[doc as usize]))
        });
        built.write(&staged.dir, index_seg)?;
        written.bytes += built.disk_bytes();
        written.docs = added.len();
        drop(texts);
        segs.push(IndexSegRow {
            id: index_seg,
            n_docs: added.len() as u32,
        });
        for (doc, (((card, fingerprint, _, _), text), sem)) in
            added.into_iter().zip(refs).zip(sem_refs).enumerate()
        {
            entries.push(Entry {
                card,
                fingerprint,
                text,
                sem,
                loc: IndexLoc {
                    seg: index_seg,
                    doc: doc as u32,
                },
            });
        }
    }
    let manifest = build_manifest(&mut entries, &segs, &ignored);
    written.bytes += manifest.len() as u64;
    Ok((staged.publish(&manifest)?, written))
}

/// When a generation's segments are merged; see `compact`.
#[derive(Clone, Copy, Debug)]
pub struct MergePolicy {
    /// Merge all deltas into one when there are more than this many.
    pub max_deltas: usize,
    /// Merge everything into a new base when delta docs plus dead base docs
    /// exceed this share of the base.
    pub base_share: f64,
    /// Rewrite a text segment when more than this share of its bytes belong
    /// to no card.
    pub text_garbage: f64,
}

impl MergePolicy {
    pub const DEFAULT: Self = Self {
        max_deltas: 4,
        base_share: 0.10,
        text_garbage: 0.40,
    };

    /// Merges everything there is to merge.
    pub const FULL: Self = Self {
        max_deltas: 0,
        base_share: 0.0,
        text_garbage: 0.0,
    };
}

/// Merges index segments of `base` (which must be `CURRENT` with its search
/// files) as `policy` asks: everything into a new base, or all deltas into
/// one delta. Text and semantic segments with too much dead data are
/// rewritten by copying the live frames. Postings are merged as they are,
/// without reading text.
/// Writes nothing when no threshold is crossed.
pub fn compact(
    lock: &PublishLock,
    base: &Generation,
    policy: MergePolicy,
) -> io::Result<(Publish, Written)> {
    let rows = base.rows();
    let locs = base.locs();
    let segs = base.segs();
    let mut live = vec![0u32; segs.len()];
    for loc in locs {
        if let Ok(at) = segs.binary_search_by_key(&loc.seg, |seg| seg.id) {
            live[at] += 1;
        }
    }
    let base_seg = segs.iter().position(|seg| seg.id == 0);
    let base_docs = base_seg.map_or(0, |at| segs[at].n_docs);
    let base_dead = base_seg.map_or(0, |at| segs[at].n_docs - live[at]);
    let deltas: Vec<usize> = (0..segs.len()).filter(|&at| Some(at) != base_seg).collect();
    let delta_docs: u64 = deltas.iter().map(|&at| u64::from(segs[at].n_docs)).sum();
    let full = (!deltas.is_empty() || base_dead > 0)
        && (delta_docs + u64::from(base_dead)) as f64 > policy.base_share * f64::from(base_docs);
    let merge_deltas = !full && deltas.len() > policy.max_deltas.max(1);

    let text_refs: Vec<BlobRef> = rows.iter().map(|row| row.text).collect();
    let sem_refs: Vec<BlobRef> = base.sems().iter().map(|row| row.blob).collect();
    let rewrite_text = garbage_segments(&base.dir, text::TEXT, &text_refs, policy, full);
    let rewrite_sem = garbage_segments(&base.dir, text::SEM, &sem_refs, policy, full);
    if !full && !merge_deltas && rewrite_text.is_empty() && rewrite_sem.is_empty() {
        return Ok((Publish::Unchanged, Written::default()));
    }
    let Some((index, text)) = base.open_search_files() else {
        return Err(io::Error::other("generation has no usable search files"));
    };
    let Some(sem) = base.open_semantic_files() else {
        return Err(io::Error::other("generation has no usable semantic files"));
    };

    let staged = Staged::new(lock)?;
    let mut written = Written::default();
    let mut new_locs: Vec<IndexLoc> = locs.to_vec();
    let mut new_segs: Vec<IndexSegRow> = Vec::new();
    let sources = index.sources();
    let merged: Vec<usize> = if full {
        (0..segs.len()).collect()
    } else if merge_deltas {
        deltas
    } else {
        Vec::new()
    };
    if !merged.is_empty() {
        let id = if full {
            0
        } else {
            segs.last().map_or(1, |seg| seg.id + 1)
        };
        // New doc ids follow the card order.
        let mut next = 0u32;
        let mut new_doc = vec![lexindex::DEAD; rows.len()];
        for (row, loc) in locs.iter().enumerate() {
            if merged.iter().any(|&at| segs[at].id == loc.seg) {
                new_doc[row] = next;
                new_locs[row] = IndexLoc { seg: id, doc: next };
                next += 1;
            }
        }
        let remaps: Vec<Vec<u32>> = merged
            .iter()
            .map(|&at| {
                sources[at]
                    .1
                    .iter()
                    .map(|&row| {
                        if row == lexindex::DEAD {
                            lexindex::DEAD
                        } else {
                            new_doc[row as usize]
                        }
                    })
                    .collect()
            })
            .collect();
        let inputs: Vec<(&LexIndex, &[u32])> = merged
            .iter()
            .zip(&remaps)
            .map(|(&at, remap)| (sources[at].0, remap.as_slice()))
            .collect();
        let built = lexindex::merge(&inputs, next as usize);
        built.write(&staged.dir, id)?;
        written.bytes += built.disk_bytes();
        written.docs = next as usize;
        for (at, seg) in segs.iter().enumerate() {
            if !merged.contains(&at) && live[at] > 0 {
                for file in lexindex::segment_files(seg.id) {
                    staged.link(&base.dir, &file)?;
                }
                new_segs.push(*seg);
            }
        }
        new_segs.push(IndexSegRow { id, n_docs: next });
        new_segs.sort_unstable_by_key(|seg| seg.id);
    } else {
        for (at, seg) in segs.iter().enumerate() {
            if live[at] > 0 {
                for file in lexindex::segment_files(seg.id) {
                    staged.link(&base.dir, &file)?;
                }
                new_segs.push(*seg);
            }
        }
    }

    let new_text = carry_segments(
        &staged,
        &base.dir,
        text::TEXT,
        &text,
        &text_refs,
        &rewrite_text,
        &mut written,
    )?;
    let new_sem = carry_segments(
        &staged,
        &base.dir,
        text::SEM,
        &sem,
        &sem_refs,
        &rewrite_sem,
        &mut written,
    )?;

    let fingerprints = base.fingerprints();
    let mut entries: Vec<Entry> = (0..rows.len())
        .map(|row| Entry {
            card: CardData::from_row(base, row),
            fingerprint: fingerprints[row],
            text: new_text[row],
            sem: new_sem[row],
            loc: new_locs[row],
        })
        .collect();
    let manifest = build_manifest(&mut entries, &new_segs, &base.ignored_entries());
    written.bytes += manifest.len() as u64;
    Ok((staged.publish(&manifest)?, written))
}

/// The `kind` segments in `dir` that `refs` use and that have more than
/// `policy.text_garbage` of their bytes in no frame of `refs`. A single
/// segment is only rewritten by a full merge, which is what accumulates its
/// garbage.
fn garbage_segments(
    dir: &Path,
    kind: &str,
    refs: &[BlobRef],
    policy: MergePolicy,
    full: bool,
) -> BTreeSet<u32> {
    let mut live: BTreeMap<u32, u64> = BTreeMap::new();
    for r in refs.iter().filter(|r| r.len > 0) {
        *live.entry(r.seg).or_default() += u64::from(r.len);
    }
    if !full && live.len() <= 1 {
        return BTreeSet::new();
    }
    live.iter()
        .filter(|&(&seg, &live)| {
            let size = fs::metadata(dir.join(text::kind_file(kind, seg))).map_or(0, |m| m.len());
            size > 0 && (size - live.min(size)) as f64 > policy.text_garbage * size as f64
        })
        .map(|(&seg, _)| seg)
        .collect()
}

/// Makes the `kind` segments `refs` use part of `staged`: linked from
/// `from`, except the `rewrite` ones, whose live frames are copied from
/// `segments` into one new segment. Returns the refs into the new layout.
fn carry_segments(
    staged: &Staged,
    from: &Path,
    kind: &str,
    segments: &TextSegments,
    refs: &[BlobRef],
    rewrite: &BTreeSet<u32>,
    written: &mut Written,
) -> io::Result<Vec<BlobRef>> {
    let used: BTreeSet<u32> = refs.iter().filter(|r| r.len > 0).map(|r| r.seg).collect();
    for &seg in &used {
        if !rewrite.contains(&seg) {
            staged.link(from, &text::kind_file(kind, seg))?;
        }
    }
    let mut new_refs = refs.to_vec();
    if !rewrite.is_empty() {
        let seg = used.last().map_or(0, |seg| seg + 1);
        let moved: Vec<usize> = (0..refs.len())
            .filter(|&at| refs[at].len > 0 && rewrite.contains(&refs[at].seg))
            .collect();
        let frames: Vec<&[u8]> = moved
            .iter()
            .map(|&at| segments.frame(refs[at]).unwrap_or_default())
            .collect();
        let path = staged.dir.join(text::kind_file(kind, seg));
        let moved_refs = text::write_frames(&path, seg, &frames)?;
        written.bytes += fs::metadata(&path)?.len();
        for (at, r) in moved.into_iter().zip(moved_refs) {
            new_refs[at] = r;
        }
    }
    Ok(new_refs)
}

/// A generation being built in `gNNN.tmp/` under a held lock.
struct Staged<'a> {
    lock: &'a PublishLock,
    existing: Vec<(u64, PathBuf)>,
    number: u64,
    dir: PathBuf,
}

impl<'a> Staged<'a> {
    fn new(lock: &'a PublishLock) -> io::Result<Self> {
        let root = lock.root.as_path();
        let existing = generation_dirs(root)?;
        let number = existing
            .iter()
            .map(|(number, _)| number + 1)
            .max()
            .unwrap_or(1);
        let dir = root.join(format!("g{number:06}.tmp"));
        if dir.exists() {
            fs::remove_dir_all(&dir)?;
        }
        fs::create_dir(&dir)?;
        Ok(Self {
            lock,
            existing,
            number,
            dir,
        })
    }

    /// Makes `from/name` part of this generation. Published files are never
    /// modified, so generations share them as hard links; the copy fallback
    /// covers filesystems without links.
    fn link(&self, from: &Path, name: &str) -> io::Result<()> {
        let (source, target) = (from.join(name), self.dir.join(name));
        if fs::hard_link(&source, &target).is_err() {
            fs::copy(&source, &target)?;
            File::open(&target)?.sync_all()?;
        }
        Ok(())
    }

    /// Writes `manifest`, renames the directory into place, points `CURRENT`
    /// at it and collects old generations.
    fn publish(self, manifest: &[u8]) -> io::Result<Publish> {
        let root = self.lock.root.as_path();
        write_synced(&self.dir.join(MANIFEST), manifest)?;
        sync_dir(&self.dir);
        let name = format!("g{:06}", self.number);
        fs::rename(&self.dir, root.join(&name))?;
        sync_dir(root);
        let current_tmp = root.join(format!("{CURRENT}.tmp"));
        write_synced(&current_tmp, format!("{name}\n").as_bytes())?;
        fs::rename(&current_tmp, root.join(CURRENT))?;
        sync_dir(root);
        collect(root, self.number, &self.existing);
        Ok(Publish::Published(name))
    }
}

/// Publishes `conversations` with unknown fingerprints as the complete
/// corpus, on top of `CURRENT`.
#[cfg(test)]
pub fn publish(root: &Path, conversations: &[Conversation]) -> io::Result<Publish> {
    let Some(lock) = PublishLock::try_acquire(root)? else {
        return Ok(Publish::Busy);
    };
    publish_all(&lock, conversations).map(|(publish, _)| publish)
}

#[cfg(test)]
pub fn publish_all(
    lock: &PublishLock,
    conversations: &[Conversation],
) -> io::Result<(Publish, Written)> {
    let base = Generation::open(&lock.root).filter(|base| base.open_search_files().is_some());
    publish_update(
        lock,
        Update {
            base: base.as_ref(),
            docs: conversations
                .iter()
                .map(|conversation| Doc::New {
                    conversation: Box::new(conversation.clone()),
                    fingerprint: Fingerprint::default(),
                })
                .collect(),
            ignored: Vec::new(),
        },
    )
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
    use crate::store::test_support::conversation;

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
        publish(root, conversations).unwrap()
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
    fn publishes_text_and_index_with_the_cards() {
        let dir = tempfile::tempdir().unwrap();
        let mut conversations = sample();
        conversations[0].full_text = "older text about caches".into();
        conversations[1].full_text = "".into();
        publish_sample(dir.path(), &conversations);
        let generation = Generation::open(dir.path()).unwrap();
        let (index, text) = generation.open_search_files().unwrap();
        let rows = generation.rows();
        assert_eq!(text.read(rows[0].text).as_deref(), Some(""));
        assert_eq!(
            text.read(rows[1].text).as_deref(),
            Some("older text about caches")
        );
        assert_eq!(index.prefix_tf("cach"), vec![0, 1]);
    }

    #[test]
    fn changed_text_or_missing_search_files_publish_again() {
        let dir = tempfile::tempdir().unwrap();
        let mut conversations = sample();
        publish_sample(dir.path(), &conversations);
        conversations[0].full_text = "different text, same cards".into();
        assert_eq!(
            publish_sample(dir.path(), &conversations),
            Publish::Published("g000002".into())
        );
        assert_eq!(
            publish_sample(dir.path(), &conversations),
            Publish::Unchanged
        );

        fs::remove_file(dir.path().join("g000002").join(lexindex::POST_FILE)).unwrap();
        assert_eq!(
            publish_sample(dir.path(), &conversations),
            Publish::Published("g000003".into())
        );
        fs::remove_file(text::segment_path(&dir.path().join("g000003"), 0)).unwrap();
        assert_eq!(
            publish_sample(dir.path(), &conversations),
            Publish::Published("g000004".into())
        );
    }

    #[test]
    fn keeps_one_generation_behind_current() {
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
        assert_eq!(numbers, vec![4, 5]);
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
        let path_len = rows_off + std::mem::offset_of!(CardRow, path) + 4;
        assert!(
            corrupt(&|bytes| bytes[path_len..path_len + 4].copy_from_slice(&u32::MAX.to_le_bytes())),
            "string ref out of bounds"
        );
        // CardRow::project_id of the first row.
        let project_id = rows_off + std::mem::offset_of!(CardRow, project_id);
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
    fn held_publish_lock_excludes_other_publishers_until_dropped() {
        let dir = tempfile::tempdir().unwrap();
        let conversations = sample();
        let lock = PublishLock::try_acquire(dir.path()).unwrap().unwrap();
        assert!(PublishLock::try_acquire(dir.path()).unwrap().is_none());
        assert_eq!(publish(dir.path(), &conversations).unwrap(), Publish::Busy);
        assert!(Generation::open(dir.path()).is_none());

        assert_eq!(
            publish_all(&lock, &conversations).unwrap().0,
            Publish::Published("g000001".into())
        );
        assert_eq!(
            publish_all(&lock, &conversations).unwrap().0,
            Publish::Unchanged
        );
        assert!(PublishLock::try_acquire(dir.path()).unwrap().is_none());
        drop(lock);
        assert!(PublishLock::try_acquire(dir.path()).unwrap().is_some());
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
        assert!(generation_dirs(&root).unwrap().len() <= 2);
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

    fn texts_of(generation: &Generation) -> Vec<String> {
        let (_, text) = generation.open_search_files().unwrap();
        generation
            .rows()
            .iter()
            .map(|row| text.read(row.text).unwrap())
            .collect()
    }

    /// Every card's tf for `word`, checked against an index built from
    /// scratch over the same texts.
    fn assert_index_matches_text(generation: &Generation, words: &[&str]) {
        let (index, _) = generation.open_search_files().unwrap();
        let lowered: Vec<String> = texts_of(generation)
            .iter()
            .map(|text| normalize_for_search(text))
            .collect();
        let dir = tempfile::tempdir().unwrap();
        lexindex::build(lowered.len(), |doc| Cow::Borrowed(&lowered[doc as usize]))
            .write(dir.path(), 0)
            .unwrap();
        let fresh = LexIndex::open(dir.path(), 0, lowered.len() as u32).unwrap();
        for word in words {
            assert_eq!(index.prefix_tf(word), fresh.prefix_tf(word), "{word:?}");
            assert_eq!(index.inner_tf(word), fresh.inner_tf(word), "{word:?}");
        }
    }

    fn sems_of(generation: &Generation) -> Vec<SemanticDoc> {
        let sem = generation.open_semantic_files().unwrap();
        generation
            .sems()
            .iter()
            .map(|row| SemanticDoc::decode(&sem.read_bytes(row.blob).unwrap()).unwrap())
            .collect()
    }

    /// Semantic data of `conversations` in card order.
    fn expected_sems(conversations: &[Conversation]) -> Vec<SemanticDoc> {
        let mut sorted: Vec<&Conversation> = conversations.iter().collect();
        sorted.sort_by(|a, b| b.timestamp.cmp(&a.timestamp).then(a.path.cmp(&b.path)));
        sorted.into_iter().map(SemanticDoc::of).collect()
    }

    /// Conversations with text, and semantic turns on every other one.
    fn corpus(n: usize) -> Vec<Conversation> {
        (0..n)
            .map(|i| {
                let mut c = conversation(&format!("/p/proj-{}/{i}.jsonl", i % 3), i as i64 * 1000);
                c.full_text = format!("alpha{i} common topic{} 测试{i}", i % 4).into();
                if i % 2 == 0 {
                    c.semantic_turns = vec![format!("turn {i}"), "second".to_owned()].into();
                    c.semantic_turn_ranges = vec![
                        crate::agent::refs::MessageRange::single(1),
                        crate::agent::refs::MessageRange::single(2),
                    ];
                    c.semantic_route_text = format!("route {i}").into();
                }
                c
            })
            .collect()
    }

    const WORDS: &[&str] = &[
        "alpha", "alpha1", "alpha12", "common", "topic", "topic2", "changed", "al", "测试", "试",
    ];

    #[test]
    fn changes_publish_as_a_delta_that_links_unchanged_segments() {
        let dir = tempfile::tempdir().unwrap();
        let mut conversations = corpus(20);
        publish_sample(dir.path(), &conversations);
        let first = Generation::open(dir.path()).unwrap();
        assert_eq!(first.segs(), &[IndexSegRow { id: 0, n_docs: 20 }]);

        conversations[3].full_text = "changed text".into();
        conversations[3].timestamp = conversations[19].timestamp + chrono::Duration::seconds(1);
        conversations.remove(7);
        let lock = PublishLock::try_acquire(dir.path()).unwrap().unwrap();
        let (publish, written) = publish_all(&lock, &conversations).unwrap();
        assert_eq!(publish, Publish::Published("g000002".into()));
        assert_eq!(written.docs, 1);

        let second = Generation::open(dir.path()).unwrap();
        assert_eq!(
            second.segs(),
            &[
                IndexSegRow { id: 0, n_docs: 20 },
                IndexSegRow { id: 1, n_docs: 1 }
            ]
        );
        assert_eq!(second.rows().len(), 19);
        assert_eq!(second.locs()[0], IndexLoc { seg: 1, doc: 0 });
        assert_eq!(texts_of(&second)[0], "changed text");
        assert!(
            texts_of(&second)
                .iter()
                .all(|text| !text.contains("alpha3 "))
        );
        assert_index_matches_text(&second, WORDS);
        assert_eq!(sems_of(&second), expected_sems(&conversations));

        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            for file in [
                "base.fst",
                "base.post",
                "base.short",
                "text-0000.zst",
                "sem-0000.zst",
            ] {
                let a = fs::metadata(dir.path().join("g000001").join(file)).unwrap();
                let b = fs::metadata(dir.path().join("g000002").join(file)).unwrap();
                assert_eq!(a.ino(), b.ino(), "{file} is linked");
            }
        }
        // The new text and index are as large as the changed conversation.
        assert!(written.bytes < 6144, "{written:?}");
    }

    #[test]
    fn semantic_changes_publish_a_new_card_with_the_same_text() {
        let dir = tempfile::tempdir().unwrap();
        let mut conversations = corpus(6);
        publish_sample(dir.path(), &conversations);
        conversations[1].semantic_turns = vec!["new turn".to_owned()].into();
        conversations[1].semantic_turn_ranges = vec![crate::agent::refs::MessageRange::single(4)];
        conversations[2].agent_search_text = "subagent text".into();
        let lock = PublishLock::try_acquire(dir.path()).unwrap().unwrap();
        let (publish, written) = publish_all(&lock, &conversations).unwrap();
        assert_eq!(publish, Publish::Published("g000002".into()));
        assert_eq!(written.docs, 2);
        let next = Generation::open(dir.path()).unwrap();
        assert_eq!(sems_of(&next), expected_sems(&conversations));
        assert_eq!(
            next.sems()[2].hash,
            0,
            "conversation 3 has no semantic data"
        );
        assert_eq!(next.sems()[2].blob, BlobRef::default());
        assert_eq!(
            publish_all(&lock, &conversations).unwrap().0,
            Publish::Unchanged
        );
    }

    #[test]
    fn fingerprint_only_changes_rewrite_just_the_manifest() {
        let dir = tempfile::tempdir().unwrap();
        let conversations = corpus(5);
        publish_sample(dir.path(), &conversations);
        let lock = PublishLock::try_acquire(dir.path()).unwrap().unwrap();
        let base = Generation::open(dir.path()).unwrap();
        let touched = Fingerprint {
            size: 9,
            mtime_ns: 77,
        };
        let docs = (0..5)
            .map(|row| Doc::Keep {
                row,
                fingerprint: if row == 2 {
                    touched
                } else {
                    Fingerprint::default()
                },
            })
            .collect();
        let ignored = vec![("/p/empty.jsonl".to_owned(), touched)];
        let (publish, written) = publish_update(
            &lock,
            Update {
                base: Some(&base),
                docs,
                ignored: ignored.clone(),
            },
        )
        .unwrap();
        assert_eq!(publish, Publish::Published("g000002".into()));
        assert_eq!(written.docs, 0);
        let next = Generation::open(dir.path()).unwrap();
        assert_eq!(next.fingerprints()[2], touched);
        assert_eq!(next.ignored_entries(), ignored);
        assert_eq!(next.segs(), base.segs());

        let docs = (0..5)
            .map(|row| Doc::Keep {
                row,
                fingerprint: next.fingerprints()[row as usize],
            })
            .collect();
        let (publish, _) = publish_update(
            &lock,
            Update {
                base: Some(&next),
                docs,
                ignored,
            },
        )
        .unwrap();
        assert_eq!(publish, Publish::Unchanged);
    }

    #[test]
    fn compaction_merges_deltas_then_everything_into_a_base() {
        let dir = tempfile::tempdir().unwrap();
        let mut conversations = corpus(100);
        publish_sample(dir.path(), &conversations);
        let lock = PublishLock::try_acquire(dir.path()).unwrap().unwrap();
        for round in 0..5 {
            conversations[round * 3].full_text = format!("changed round {round}").into();
            publish_all(&lock, &conversations).unwrap();
        }
        let generation = Generation::open(dir.path()).unwrap();
        assert_eq!(generation.segs().len(), 6);
        assert_index_matches_text(&generation, WORDS);
        let before = texts_of(&generation);

        let (publish, _) = compact(&lock, &generation, MergePolicy::DEFAULT).unwrap();
        assert!(matches!(publish, Publish::Published(_)));
        let merged = Generation::open(dir.path()).unwrap();
        assert_eq!(
            merged.segs(),
            &[
                IndexSegRow { id: 0, n_docs: 100 },
                IndexSegRow { id: 6, n_docs: 5 }
            ]
        );
        assert_eq!(texts_of(&merged), before);
        assert_index_matches_text(&merged, WORDS);
        assert_eq!(sems_of(&merged), expected_sems(&conversations));
        assert_eq!(
            compact(&lock, &merged, MergePolicy::DEFAULT).unwrap().0,
            Publish::Unchanged
        );

        let (publish, _) = compact(&lock, &merged, MergePolicy::FULL).unwrap();
        assert!(matches!(publish, Publish::Published(_)));
        let full = Generation::open(dir.path()).unwrap();
        assert_eq!(full.segs(), &[IndexSegRow { id: 0, n_docs: 100 }]);
        assert!(
            full.locs()
                .iter()
                .enumerate()
                .all(|(row, loc)| loc.seg == 0 && loc.doc == row as u32)
        );
        assert_eq!(texts_of(&full), before);
        assert_index_matches_text(&full, WORDS);
        assert_eq!(sems_of(&full), expected_sems(&conversations));
    }

    #[test]
    fn compaction_rewrites_text_segments_that_are_mostly_dead() {
        let dir = tempfile::tempdir().unwrap();
        let mut conversations = corpus(10);
        publish_sample(dir.path(), &conversations);
        let lock = PublishLock::try_acquire(dir.path()).unwrap().unwrap();
        for conversation in &mut conversations[..6] {
            conversation.full_text = format!("{} again", conversation.full_text).into();
            conversation.semantic_turns =
                vec![format!("{} again", conversation.path.display())].into();
        }
        publish_all(&lock, &conversations).unwrap();
        let generation = Generation::open(dir.path()).unwrap();
        let before = texts_of(&generation);
        compact(&lock, &generation, MergePolicy::FULL).unwrap();
        let full = Generation::open(dir.path()).unwrap();
        let files: BTreeSet<String> = fs::read_dir(full.dir.clone())
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .collect();
        assert!(!files.contains("text-0000.zst"), "{files:?}");
        assert!(files.contains("text-0001.zst") && files.contains("text-0002.zst"));
        assert!(!files.contains("sem-0000.zst"), "{files:?}");
        assert!(files.contains("sem-0001.zst") && files.contains("sem-0002.zst"));
        assert_eq!(texts_of(&full), before);
        assert_eq!(sems_of(&full), expected_sems(&conversations));
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
            let root = dir.path().join(count.to_string());
            publish(&root, &conversations).unwrap();
            let manifest = fs::read(root.join("g000001").join(MANIFEST)).unwrap();
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
