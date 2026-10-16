//! Which rollouts make up a thread's history, as Codex would resolve it.
//!
//! A thread can have several rollouts: a revert starts a new one under the
//! same thread id, and a fork or revert may continue a prefix of another
//! (`history_base`). The state database names the rollout a thread uses;
//! for a paginated thread that is the only one Codex will open. Without a
//! usable row the head is the newest rollout no other rollout of the thread
//! continues.

use super::discover::{self, Rollout};
use super::state_db::Row;
use crate::history::adapter::{FileId, Stamp};
use crate::history::model::HeadStatus;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// Longest `history_base` chain followed.
const MAX_CHAIN: usize = 16;

/// One rollout of a thread's history.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Link {
    pub path: PathBuf,
    /// Bytes of the rollout that belong to the history, when only a prefix
    /// does.
    pub end: Option<u64>,
    /// Whether the rollout is another thread's, whose prefix this thread
    /// was forked from.
    pub inherited: bool,
    pub size: u64,
    pub modified: Option<SystemTime>,
    pub file_id: Option<FileId>,
}

/// A thread resolved to the rollouts its history is read from.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Thread {
    pub id: String,
    pub head: HeadStatus,
    /// Whether the head rollout is in `archived_sessions`.
    pub archived: bool,
    /// The name the user or Codex gave the thread.
    pub name: Option<String>,
    /// Oldest base first; the head rollout last.
    pub chain: Vec<Link>,
    /// Rollout id of a `history_base` that could not be found.
    pub missing_base: Option<String>,
    /// The state database row the head was chosen with.
    pub row: Option<Row>,
}

impl Thread {
    pub fn head_link(&self) -> &Link {
        self.chain.last().expect("a thread has a head rollout")
    }

    /// What discovery saw of every file the history is read from and of the
    /// state that chose them. The modification time is a digest of all of
    /// it, unknown when any file's is.
    pub fn stamp(&self, scan_started: SystemTime) -> Stamp {
        let size = self.chain.iter().map(|link| link.size).sum();
        let mut hasher = blake3::Hasher::new();
        for link in &self.chain {
            let file = Stamp::of(link.size, link.modified, scan_started);
            if !file.is_known() {
                return Stamp { size, mtime_ns: 0 };
            }
            hash_bytes(&mut hasher, link.path.as_os_str().as_encoded_bytes());
            hasher.update(&link.size.to_le_bytes());
            hasher.update(&file.mtime_ns.to_le_bytes());
            hasher.update(&link.end.unwrap_or(u64::MAX).to_le_bytes());
            hasher.update(&[u8::from(link.inherited)]);
        }
        if let Some(row) = &self.row {
            hash_bytes(&mut hasher, row.rollout_path.as_os_str().as_encoded_bytes());
            hasher.update(&row.updated_at.to_le_bytes());
            hasher.update(&[u8::from(row.paginated)]);
        }
        hash_bytes(
            &mut hasher,
            self.name.as_deref().unwrap_or_default().as_bytes(),
        );
        hasher.update(&[match self.head {
            HeadStatus::Current => 0,
            HeadStatus::Inferred => 1,
            HeadStatus::Orphaned => 2,
        }]);
        let digest = u64::from_le_bytes(hasher.finalize().as_bytes()[..8].try_into().unwrap());
        Stamp {
            size,
            // Positive and nonzero, so never mistaken for unknown.
            mtime_ns: (digest >> 1).max(1) as i64,
        }
    }
}

fn hash_bytes(hasher: &mut blake3::Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

/// File state of a rollout.
#[derive(Clone, Copy, Debug)]
struct FileState {
    size: u64,
    modified: Option<SystemTime>,
    file_id: Option<FileId>,
}

impl FileState {
    fn of(path: &Path) -> Option<Self> {
        let metadata = fs::metadata(path).ok()?;
        Some(Self {
            size: metadata.len(),
            modified: metadata.modified().ok(),
            file_id: file_id(&metadata),
        })
    }
}

#[cfg(unix)]
fn file_id(metadata: &fs::Metadata) -> Option<FileId> {
    use std::os::unix::fs::MetadataExt;
    Some((metadata.dev(), metadata.ino()))
}

#[cfg(not(unix))]
fn file_id(_metadata: &fs::Metadata) -> Option<FileId> {
    None
}

/// Every rollout of a Codex home with the state that chooses among them.
pub struct Scan {
    rollouts: Vec<Rollout>,
    files: Vec<Option<FileState>>,
    /// Rollouts of each thread, by index.
    threads: HashMap<String, Vec<usize>>,
    by_rollout: HashMap<String, usize>,
    rows: HashMap<String, Row>,
    names: HashMap<String, String>,
}

impl Scan {
    /// Groups `rollouts` by thread, stating each in parallel. A rollout
    /// listed both live and archived (mid-move) counts once, as live.
    pub fn new(
        mut rollouts: Vec<Rollout>,
        rows: HashMap<String, Row>,
        names: HashMap<String, String>,
    ) -> Self {
        rollouts.sort_by(|a, b| {
            (a.archived, &a.rollout_id, &a.path).cmp(&(b.archived, &b.rollout_id, &b.path))
        });
        let mut seen = HashSet::new();
        rollouts.retain(|rollout| seen.insert(rollout_name(&rollout.path).to_owned()));
        let files = rollouts
            .par_iter()
            .map(|rollout| FileState::of(&rollout.path))
            .collect();
        let mut threads: HashMap<String, Vec<usize>> = HashMap::new();
        let mut by_rollout = HashMap::new();
        for (index, rollout) in rollouts.iter().enumerate() {
            threads
                .entry(rollout.thread_id.clone())
                .or_default()
                .push(index);
            by_rollout
                .entry(rollout.rollout_id.clone())
                .or_insert(index);
        }
        Self {
            rollouts,
            files,
            threads,
            by_rollout,
            rows,
            names,
        }
    }

    pub fn thread_ids(&self) -> Vec<&str> {
        self.threads.keys().map(String::as_str).collect()
    }

    /// The thread `id` resolved to its rollouts; `None` when it has none
    /// that can be stated.
    pub fn thread(&self, id: &str) -> Option<Thread> {
        let candidates: Vec<usize> = self
            .threads
            .get(id)?
            .iter()
            .copied()
            .filter(|&index| self.files[index].is_some())
            .collect();
        let row = self.rows.get(id);
        let (head, status) = self.head(&candidates, row)?;
        let mut chain = vec![self.link(head, None, false)];
        let mut missing_base = None;
        let mut visited = HashSet::from([head]);
        let mut current = head;
        while chain.len() < MAX_CHAIN {
            let Ok(start) = discover::read_start(&self.rollouts[current].path) else {
                break;
            };
            let Some(base) = start.history_base else {
                break;
            };
            let Some(&index) = self
                .by_rollout
                .get(&base.rollout_id)
                .filter(|&&index| self.files[index].is_some())
            else {
                missing_base = Some(base.rollout_id);
                break;
            };
            if !visited.insert(index) {
                break;
            }
            let inherited = self.rollouts[index].thread_id != id;
            chain.push(self.link(index, Some(base.end_byte_offset), inherited));
            current = index;
        }
        chain.reverse();
        Some(Thread {
            id: id.to_owned(),
            head: status,
            archived: self.rollouts[head].archived,
            name: row
                .and_then(|row| row.name.clone())
                .or_else(|| self.names.get(id).cloned()),
            chain,
            missing_base,
            row: row.cloned(),
        })
    }

    /// The head rollout among `candidates` and how it was chosen.
    fn head(&self, candidates: &[usize], row: Option<&Row>) -> Option<(usize, HeadStatus)> {
        if let Some(row) = row {
            let named = rollout_name(&row.rollout_path);
            if let Some(&index) = candidates
                .iter()
                .find(|&&index| rollout_name(&self.rollouts[index].path) == named)
            {
                return Some((index, HeadStatus::Current));
            }
            if row.paginated {
                let newest = candidates.iter().copied().max_by(|&a, &b| {
                    self.rollouts[a]
                        .rollout_id
                        .cmp(&self.rollouts[b].rollout_id)
                })?;
                return Some((newest, HeadStatus::Orphaned));
            }
        }
        if let [only] = candidates {
            return Some((*only, HeadStatus::Inferred));
        }
        let continued: HashSet<String> = candidates
            .iter()
            .filter_map(|&index| discover::read_start(&self.rollouts[index].path).ok())
            .filter_map(|start| start.history_base.map(|base| base.rollout_id))
            .collect();
        let newest = |indices: &mut dyn Iterator<Item = usize>| {
            indices.max_by(|&a, &b| {
                self.rollouts[a]
                    .rollout_id
                    .cmp(&self.rollouts[b].rollout_id)
            })
        };
        newest(
            &mut candidates
                .iter()
                .copied()
                .filter(|&index| !continued.contains(&self.rollouts[index].rollout_id)),
        )
        .or_else(|| newest(&mut candidates.iter().copied()))
        .map(|index| (index, HeadStatus::Inferred))
    }

    fn link(&self, index: usize, end: Option<u64>, inherited: bool) -> Link {
        let file = self.files[index].expect("linked rollouts are stated");
        Link {
            path: self.rollouts[index].path.clone(),
            end,
            inherited,
            size: file.size,
            modified: file.modified,
            file_id: file.file_id,
        }
    }
}

/// A rollout's file name without the `.zst` a compressed copy adds, which
/// identifies it wherever its home is mounted.
fn rollout_name(path: &Path) -> &str {
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    name.strip_suffix(".zst").unwrap_or(name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    const A: &str = "01a08547-6fb3-7790-a95c-cfd12fc88f30";
    const A2: &str = "01a0854b-0d71-7fb2-a90b-5cf9c13ff6b8";
    const A3: &str = "01a0854c-0000-7000-8000-000000000000";
    const B: &str = "01a09000-0000-7000-8000-000000000000";

    struct Home(tempfile::TempDir);

    impl Home {
        fn new() -> Self {
            let home = tempfile::tempdir().unwrap();
            fs::create_dir_all(home.path().join("sessions/2026/09/09")).unwrap();
            Self(home)
        }

        /// Writes a rollout of `thread` (`rollout` after `_` when given)
        /// whose `session_meta` continues `base`.
        fn rollout(
            &self,
            thread: &str,
            rollout: Option<&str>,
            base: Option<(&str, u64)>,
        ) -> PathBuf {
            let suffix = rollout.map(|id| format!("_{id}")).unwrap_or_default();
            let path = self.0.path().join(format!(
                "sessions/2026/09/09/rollout-2026-09-09T11-27-26-{thread}{suffix}.jsonl"
            ));
            let base = base.map_or(String::new(), |(id, end)| {
                format!(
                    r#","history_base":{{"thread_id":"{id}","end_ordinal_exclusive":1,"end_byte_offset":{end}}}"#
                )
            });
            fs::write(
                &path,
                format!(
                    "{{\"timestamp\":\"2026-09-09T08:27:26.000Z\",\"type\":\"session_meta\",\"payload\":{{\"id\":\"{thread}\"{base}}}}}\n"
                ),
            )
            .unwrap();
            path
        }

        fn scan(&self, rows: HashMap<String, Row>) -> Scan {
            Scan::new(discover::list(self.0.path()).unwrap(), rows, HashMap::new())
        }
    }

    fn row(path: &Path, paginated: bool) -> Row {
        Row {
            rollout_path: path.to_path_buf(),
            paginated,
            name: None,
            updated_at: 1,
        }
    }

    #[test]
    fn the_state_row_names_the_current_rollout() {
        let home = Home::new();
        let original = home.rollout(A, None, None);
        home.rollout(A, Some(A2), Some((A, 10)));
        // A row written under another mount of the same home still matches.
        let elsewhere = Path::new("/elsewhere/.codex/sessions").join(original.file_name().unwrap());
        let scan = home.scan(HashMap::from([(A.to_owned(), row(&elsewhere, true))]));
        let thread = scan.thread(A).unwrap();
        assert_eq!(thread.head, HeadStatus::Current);
        assert_eq!(thread.chain.len(), 1);
        assert_eq!(thread.head_link().path, original);
    }

    #[test]
    fn a_paginated_row_naming_a_missing_rollout_orphans_the_thread() {
        let home = Home::new();
        home.rollout(A, None, None);
        let newest = home.rollout(A, Some(A2), None);
        let scan = home.scan(HashMap::from([(
            A.to_owned(),
            row(Path::new("/gone/rollout-x.jsonl"), true),
        )]));
        let thread = scan.thread(A).unwrap();
        assert_eq!(thread.head, HeadStatus::Orphaned);
        assert_eq!(thread.head_link().path, newest);
    }

    #[test]
    fn without_a_row_the_head_is_the_newest_leaf() {
        let home = Home::new();
        let original = home.rollout(A, None, None);
        let revert = home.rollout(A, Some(A2), Some((A, 10)));
        // Newer, but continued by nothing and continuing nothing: a leaf too.
        let other = home.rollout(A, Some(A3), None);
        let scan = home.scan(HashMap::new());
        let thread = scan.thread(A).unwrap();
        assert_eq!(thread.head, HeadStatus::Inferred);
        assert_eq!(thread.head_link().path, other);

        fs::remove_file(other).unwrap();
        // A legacy row naming a missing rollout does not orphan the thread.
        let rows = HashMap::from([(A.to_owned(), row(Path::new("/gone/x.jsonl"), false))]);
        let thread = home.scan(rows).thread(A).unwrap();
        assert_eq!(thread.head, HeadStatus::Inferred);
        assert_eq!(
            thread
                .chain
                .iter()
                .map(|link| (link.path.clone(), link.end, link.inherited))
                .collect::<Vec<_>>(),
            vec![(original, Some(10), false), (revert, None, false)]
        );
    }

    #[test]
    fn bases_of_other_threads_are_inherited_and_missing_bases_reported() {
        let home = Home::new();
        let parent = home.rollout(A, None, None);
        home.rollout(B, None, Some((A, 5)));
        let scan = home.scan(HashMap::new());
        let thread = scan.thread(B).unwrap();
        assert_eq!(thread.chain[0].path, parent);
        assert!(thread.chain[0].inherited);
        assert_eq!(thread.missing_base, None);

        fs::remove_file(parent).unwrap();
        let thread = home.scan(HashMap::new()).thread(B).unwrap();
        assert_eq!(thread.chain.len(), 1);
        assert_eq!(thread.missing_base.as_deref(), Some(A));
    }

    #[test]
    fn stamps_change_with_any_chained_file_or_the_row() {
        let home = Home::new();
        let parent = home.rollout(A, None, None);
        home.rollout(B, None, Some((A, 5)));
        let later = SystemTime::now() + Duration::from_secs(60);
        let rows = |name: Option<&str>| {
            let mut row = row(Path::new("/x"), false);
            row.name = name.map(str::to_owned);
            HashMap::from([(B.to_owned(), row)])
        };
        let stamp = |rows| home.scan(rows).thread(B).unwrap().stamp(later);
        let before = stamp(rows(None));
        assert!(before.is_known());
        assert_eq!(before, stamp(rows(None)));
        assert_ne!(before, stamp(rows(Some("Named"))));

        fs::write(&parent, "{}\n{}\n").unwrap();
        assert_ne!(before, stamp(rows(None)));
        // A file modified during the scan leaves the stamp unknown.
        assert!(
            !home
                .scan(rows(None))
                .thread(B)
                .unwrap()
                .stamp(SystemTime::now())
                .is_known()
        );
    }
}
