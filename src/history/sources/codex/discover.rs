//! Rollout files of a Codex home: where they live, what their names say,
//! and their first record.
//!
//! Rollouts are `sessions/YYYY/MM/DD/rollout-<local time>-<thread id>.jsonl`,
//! moved flat into `archived_sessions/` when archived, and possibly zstd
//! compressed to `.jsonl.zst` once idle. A thread that was reverted keeps
//! its id and continues in a new rollout named
//! `rollout-<time>-<thread id>_<rollout id>.jsonl`.

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Read};
use std::path::{Path, PathBuf};

pub const SESSIONS: &str = "sessions";
pub const ARCHIVED_SESSIONS: &str = "archived_sessions";
/// Thread names Codex appends to; the last entry for a thread wins.
const SESSION_INDEX: &str = "session_index.jsonl";
/// Longest first record read: `session_meta` carries the full system prompt.
const FIRST_LINE_CAP: u64 = 4 << 20;

/// One rollout file.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Rollout {
    pub path: PathBuf,
    pub thread_id: String,
    /// The thread id, or the id after `_` for a rollout started by a revert.
    pub rollout_id: String,
    pub archived: bool,
}

/// Thread and rollout id of a rollout file name.
pub fn parse_name(name: &str) -> Option<(&str, &str)> {
    let name = name.strip_suffix(".zst").unwrap_or(name);
    let core = name.strip_prefix("rollout-")?.strip_suffix(".jsonl")?;
    // `YYYY-MM-DDTHH-MM-SS-` precedes the ids.
    if core.get(19..20)? != "-" {
        return None;
    }
    let ids = core.get(20..)?;
    let (thread_id, rollout_id) = ids.split_once('_').unwrap_or((ids, ids));
    (is_uuid(thread_id) && is_uuid(rollout_id)).then_some((thread_id, rollout_id))
}

fn is_uuid(text: &str) -> bool {
    text.len() == 36
        && text.char_indices().all(|(i, c)| match i {
            8 | 13 | 18 | 23 => c == '-',
            _ => c.is_ascii_hexdigit(),
        })
}

/// The rollout `path` names, if it is one.
pub fn rollout_of(path: &Path, archived: bool) -> Option<Rollout> {
    let (thread_id, rollout_id) = parse_name(path.file_name()?.to_str()?)?;
    Some(Rollout {
        thread_id: thread_id.to_ascii_lowercase(),
        rollout_id: rollout_id.to_ascii_lowercase(),
        path: path.to_path_buf(),
        archived,
    })
}

/// The Codex home holding the rollout at `path`: the parent of its
/// `sessions` or `archived_sessions` directory.
pub fn home_of(path: &Path) -> Option<&Path> {
    path.ancestors().skip(1).find_map(|dir| {
        let name = dir.file_name()?;
        (name == SESSIONS || name == ARCHIVED_SESSIONS)
            .then(|| dir.parent())
            .flatten()
    })
}

/// Every rollout under `home`. An error means a directory that exists
/// could not be listed, so rollouts may be missing.
pub fn list(home: &Path) -> io::Result<Vec<Rollout>> {
    let mut rollouts = Vec::new();
    for (dir, archived, depth) in [(SESSIONS, false, 3), (ARCHIVED_SESSIONS, true, 0)] {
        match walk(&home.join(dir), archived, depth, &mut rollouts) {
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            result => result?,
        }
    }
    Ok(rollouts)
}

/// Rollouts in `dir` and, `depth` levels down (`YYYY/MM/DD`), its
/// directories.
fn walk(dir: &Path, archived: bool, depth: usize, out: &mut Vec<Rollout>) -> io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if let Some(rollout) = rollout_of(&path, archived) {
            out.push(rollout);
        } else if depth > 0 && entry.file_type()?.is_dir() {
            walk(&path, archived, depth - 1, out)?;
        }
    }
    Ok(())
}

/// Thread names from `session_index.jsonl`, the last entry of each thread.
pub fn session_names(home: &Path) -> HashMap<String, String> {
    #[derive(serde::Deserialize)]
    struct Entry {
        id: String,
        thread_name: Option<String>,
    }
    let mut names = HashMap::new();
    let Ok(file) = File::open(home.join(SESSION_INDEX)) else {
        return names;
    };
    for line in BufReader::new(file).lines() {
        let Ok(line) = line else { break };
        let Ok(entry) = serde_json::from_str::<Entry>(&line) else {
            continue;
        };
        let name = entry.thread_name.unwrap_or_default().trim().to_owned();
        if name.is_empty() {
            names.remove(&entry.id);
        } else {
            names.insert(entry.id, name);
        }
    }
    names
}

/// A reader over a rollout's JSONL, decompressing `.zst` rollouts.
pub fn open(path: &Path) -> io::Result<Box<dyn BufRead + Send>> {
    let file = File::open(path)?;
    if path.extension().is_some_and(|extension| extension == "zst") {
        Ok(Box::new(BufReader::new(zstd::Decoder::new(file)?)))
    } else {
        Ok(Box::new(BufReader::new(file)))
    }
}

/// What a rollout's first record says about where its history starts.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Start {
    /// Prefix of another rollout this one continues.
    pub history_base: Option<HistoryBase>,
}

/// `session_meta.history_base`: the first `end_byte_offset` bytes of the
/// rollout `rollout_id` precede this rollout's own records.
#[derive(Clone, Debug, Eq, PartialEq, serde::Deserialize)]
pub struct HistoryBase {
    /// Named `thread_id` on disk, but a rollout id.
    #[serde(rename = "thread_id")]
    pub rollout_id: String,
    pub end_byte_offset: u64,
}

/// Reads the first record of the rollout at `path`. Only a record that
/// mentions `history_base` is parsed.
pub fn read_start(path: &Path) -> io::Result<Start> {
    let mut line = Vec::new();
    open(path)?
        .take(FIRST_LINE_CAP)
        .read_until(b'\n', &mut line)?;
    if memchr::memmem::find(&line, b"\"history_base\":").is_none() {
        return Ok(Start::default());
    }
    #[derive(serde::Deserialize)]
    struct Meta {
        history_base: Option<HistoryBase>,
    }
    #[derive(serde::Deserialize)]
    struct Line {
        payload: Meta,
    }
    let line: Line = serde_json::from_slice(&line).map_err(io::Error::other)?;
    Ok(Start {
        history_base: line.payload.history_base.map(|mut base| {
            base.rollout_id.make_ascii_lowercase();
            base
        }),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const THREAD: &str = "01a08547-6fb3-7790-a95c-cfd12fc88f30";
    const REVERT: &str = "01a0854b-0d71-7fb2-a90b-5cf9c13ff6b8";

    #[test]
    fn names_carry_the_thread_and_rollout_ids() {
        let plain = format!("rollout-2026-09-09T11-27-26-{THREAD}.jsonl");
        assert_eq!(parse_name(&plain), Some((THREAD, THREAD)));
        let reverted = format!("rollout-2026-09-09T11-31-23-{THREAD}_{REVERT}.jsonl.zst");
        assert_eq!(parse_name(&reverted), Some((THREAD, REVERT)));
        for other in [
            "history.jsonl",
            &format!("rollout-2026-09-09T11-27-26-{THREAD}.json"),
            "rollout-2026-09-09T11-27-26-not-a-uuid.jsonl",
            &format!("rollout-2026-09-09-{THREAD}.jsonl"),
        ] {
            assert_eq!(parse_name(other), None, "{other}");
        }
    }

    #[test]
    fn rollouts_know_their_home() {
        let path = Path::new("/h/.codex/sessions/2026/09/09/rollout-x.jsonl");
        assert_eq!(home_of(path), Some(Path::new("/h/.codex")));
        let archived = Path::new("/h/.codex/archived_sessions/rollout-x.jsonl");
        assert_eq!(home_of(archived), Some(Path::new("/h/.codex")));
        assert_eq!(home_of(Path::new("/h/other/rollout-x.jsonl")), None);
    }

    #[test]
    fn listing_finds_dated_and_archived_rollouts() {
        let dir = tempfile::tempdir().unwrap();
        let day = dir.path().join("sessions/2026/09/09");
        fs::create_dir_all(&day).unwrap();
        fs::create_dir_all(dir.path().join(ARCHIVED_SESSIONS)).unwrap();
        let name = format!("rollout-2026-09-09T11-27-26-{THREAD}.jsonl");
        fs::write(day.join(&name), "").unwrap();
        fs::write(day.join("notes.txt"), "").unwrap();
        let archived = format!("rollout-2026-09-09T11-31-23-{THREAD}_{REVERT}.jsonl");
        fs::write(dir.path().join(ARCHIVED_SESSIONS).join(&archived), "").unwrap();
        let mut rollouts = list(dir.path()).unwrap();
        rollouts.sort_by(|a, b| a.path.cmp(&b.path));
        assert_eq!(rollouts.len(), 2);
        assert!(rollouts[0].archived);
        assert_eq!(rollouts[0].rollout_id, REVERT);
        assert!(!rollouts[1].archived);
        assert_eq!(rollouts[1].thread_id, THREAD);
        assert!(list(&dir.path().join("missing")).unwrap().is_empty());
    }

    #[test]
    fn the_last_session_index_name_wins() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join(SESSION_INDEX),
            [
                r#"{"id":"a","thread_name":"First","updated_at":"2026-01-01T00:00:00Z"}"#,
                r#"{"id":"a","thread_name":" Second ","updated_at":"2026-01-02T00:00:00Z"}"#,
                r#"{"id":"b","thread_name":"Gone","updated_at":"2026-01-01T00:00:00Z"}"#,
                r#"{"id":"b","thread_name":"","updated_at":"2026-01-02T00:00:00Z"}"#,
                "not json",
            ]
            .join("\n"),
        )
        .unwrap();
        let names = session_names(dir.path());
        assert_eq!(names.get("a").map(String::as_str), Some("Second"));
        assert!(!names.contains_key("b"));
    }

    #[test]
    fn starts_read_the_history_base_of_plain_and_compressed_rollouts() {
        let dir = tempfile::tempdir().unwrap();
        let meta = format!(
            r#"{{"timestamp":"t","ordinal":0,"type":"session_meta","payload":{{"id":"{THREAD}","base_instructions":{{"text":"say \"history_base\": no"}},"history_base":{{"thread_id":"{THREAD}","end_ordinal_exclusive":3,"end_byte_offset":120}}}}}}"#
        );
        let plain = dir.path().join("plain.jsonl");
        fs::write(&plain, format!("{meta}\n{{}}\n")).unwrap();
        let compressed = dir.path().join("packed.jsonl.zst");
        fs::write(
            &compressed,
            zstd::encode_all(format!("{meta}\n").as_bytes(), 3).unwrap(),
        )
        .unwrap();
        let expected = Some(HistoryBase {
            rollout_id: THREAD.to_owned(),
            end_byte_offset: 120,
        });
        assert_eq!(read_start(&plain).unwrap().history_base, expected);
        assert_eq!(read_start(&compressed).unwrap().history_base, expected);

        let quoted = dir.path().join("quoted.jsonl");
        fs::write(
            &quoted,
            r#"{"type":"session_meta","payload":{"base_instructions":{"text":"\"history_base\": x"}}}"#,
        )
        .unwrap();
        assert_eq!(read_start(&quoted).unwrap(), Start::default());
    }
}
