//! Codex's thread metadata database, `state_<N>.sqlite`, read-only.
//!
//! Codex keeps the database beside its rollouts (or in `CODEX_SQLITE_HOME`)
//! and bumps `<N>` on incompatible schema changes, so the newest file is
//! opened and its `threads` columns are checked rather than trusted. Rows
//! are authoritative for which rollout a paginated thread currently uses
//! and for its name; everything else comes from the rollouts.

use rusqlite::{Connection, OpenFlags, OptionalExtension};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// How long a read waits for a writer before the database counts as busy.
const BUSY_TIMEOUT: Duration = Duration::from_millis(50);

/// What a `threads` row says about one thread.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Row {
    /// Rollout the thread currently uses.
    pub rollout_path: PathBuf,
    /// Whether the thread's history is paginated, which makes `rollout_path`
    /// the only rollout Codex will open for it.
    pub paginated: bool,
    /// Name the user or Codex gave the thread.
    pub name: Option<String>,
    /// Last update, in whatever unit the schema stores.
    pub updated_at: i64,
}

/// An open state database.
pub struct StateDb {
    connection: Connection,
    /// The `SELECT` list for a row, adapted to the columns present.
    columns: String,
}

impl StateDb {
    /// Opens the newest `state_<N>.sqlite` in `dir`. `Ok(None)` when there
    /// is none, or when its `threads` table lacks the columns a row needs.
    /// Errors mean the database exists but cannot be read now (busy,
    /// locked or corrupt).
    pub fn open(dir: &Path) -> rusqlite::Result<Option<Self>> {
        let Some(path) = newest(dir) else {
            return Ok(None);
        };
        let connection = Connection::open_with_flags(
            &path,
            OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )?;
        connection.busy_timeout(BUSY_TIMEOUT)?;
        let mut present = Vec::new();
        {
            let mut statement =
                connection.prepare("SELECT name FROM pragma_table_info('threads')")?;
            let mut rows = statement.query([])?;
            while let Some(row) = rows.next()? {
                present.push(row.get::<_, String>(0)?);
            }
        }
        let has = |column: &str| present.iter().any(|name| name == column);
        if !has("id") || !has("rollout_path") {
            return Ok(None);
        }
        let optional = |column: &str, fallback: &str| {
            if has(column) {
                column.to_owned()
            } else {
                fallback.to_owned()
            }
        };
        let updated_at = if has("updated_at_ms") {
            "updated_at_ms".to_owned()
        } else {
            optional("updated_at", "0")
        };
        let columns = format!(
            "id, rollout_path, {}, {}, {}",
            optional("history_mode", "'legacy'"),
            optional("name", "NULL"),
            updated_at,
        );
        Ok(Some(Self {
            connection,
            columns,
        }))
    }

    /// Every row, by thread id.
    pub fn rows(&self) -> rusqlite::Result<HashMap<String, Row>> {
        let mut statement = self
            .connection
            .prepare(&format!("SELECT {} FROM threads", self.columns))?;
        let mut rows = statement.query([])?;
        let mut by_id = HashMap::new();
        while let Some(row) = rows.next()? {
            let (id, parsed) = parse(row)?;
            by_id.insert(id, parsed);
        }
        Ok(by_id)
    }

    /// The row of thread `id`.
    pub fn row(&self, id: &str) -> rusqlite::Result<Option<Row>> {
        self.connection
            .query_row(
                &format!("SELECT {} FROM threads WHERE id = ?1", self.columns),
                [id],
                parse,
            )
            .optional()
            .map(|row| row.map(|(_, row)| row))
    }
}

fn parse(row: &rusqlite::Row<'_>) -> rusqlite::Result<(String, Row)> {
    let name: Option<String> = row.get(3)?;
    Ok((
        row.get(0)?,
        Row {
            rollout_path: PathBuf::from(row.get::<_, String>(1)?),
            paginated: row.get::<_, Option<String>>(2)?.as_deref() == Some("paginated"),
            name: name
                .map(|name| name.trim().to_owned())
                .filter(|name| !name.is_empty()),
            updated_at: row.get::<_, Option<i64>>(4)?.unwrap_or(0),
        },
    ))
}

/// The `state_<N>.sqlite` in `dir` with the highest `<N>`.
fn newest(dir: &Path) -> Option<PathBuf> {
    std::fs::read_dir(dir)
        .ok()?
        .filter_map(|entry| {
            let name = entry.ok()?.file_name();
            let version: u32 = name
                .to_str()?
                .strip_prefix("state_")?
                .strip_suffix(".sqlite")?
                .parse()
                .ok()?;
            Some((version, dir.join(name)))
        })
        .max_by_key(|(version, _)| *version)
        .map(|(_, path)| path)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Creates `state_<version>.sqlite` in `dir` with Codex's `threads`
    /// columns that the adapter reads, plus `rows` of
    /// `(id, rollout_path, history_mode, name, updated_at_ms)`.
    pub(crate) fn create(
        dir: &Path,
        version: u32,
        rows: &[(&str, &Path, &str, Option<&str>, i64)],
    ) -> PathBuf {
        let path = dir.join(format!("state_{version}.sqlite"));
        let connection = Connection::open(&path).unwrap();
        connection
            .execute_batch(
                "CREATE TABLE threads (
                    id TEXT PRIMARY KEY,
                    rollout_path TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    source TEXT NOT NULL,
                    title TEXT NOT NULL,
                    archived INTEGER NOT NULL DEFAULT 0,
                    first_user_message TEXT NOT NULL DEFAULT '',
                    updated_at_ms INTEGER,
                    history_mode TEXT NOT NULL DEFAULT 'legacy',
                    name TEXT
                );",
            )
            .unwrap();
        for (id, rollout_path, history_mode, name, updated_at_ms) in rows {
            connection
                .execute(
                    "INSERT INTO threads (id, rollout_path, created_at, updated_at, source,
                        title, updated_at_ms, history_mode, name)
                     VALUES (?1, ?2, 0, ?3 / 1000, 'cli', '', ?3, ?4, ?5)",
                    rusqlite::params![
                        id,
                        rollout_path.to_string_lossy(),
                        updated_at_ms,
                        history_mode,
                        name
                    ],
                )
                .unwrap();
        }
        path
    }

    #[test]
    fn opens_the_newest_database_and_reads_rows() {
        let dir = tempfile::tempdir().unwrap();
        create(
            dir.path(),
            4,
            &[("old", Path::new("/r/old.jsonl"), "legacy", None, 1)],
        );
        create(
            dir.path(),
            5,
            &[
                (
                    "a",
                    Path::new("/r/a.jsonl"),
                    "paginated",
                    Some(" Named "),
                    7,
                ),
                ("b", Path::new("/r/b.jsonl"), "legacy", Some(""), 8),
            ],
        );
        let db = StateDb::open(dir.path()).unwrap().unwrap();
        let rows = db.rows().unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(
            rows["a"],
            Row {
                rollout_path: PathBuf::from("/r/a.jsonl"),
                paginated: true,
                name: Some("Named".to_owned()),
                updated_at: 7,
            }
        );
        assert_eq!(rows["b"].name, None);
        assert!(!rows["b"].paginated);
        assert_eq!(db.row("a").unwrap(), Some(rows["a"].clone()));
        assert_eq!(db.row("missing").unwrap(), None);
    }

    #[test]
    fn a_missing_database_or_foreign_schema_is_absent() {
        let dir = tempfile::tempdir().unwrap();
        assert!(StateDb::open(dir.path()).unwrap().is_none());
        let connection = Connection::open(dir.path().join("state_9.sqlite")).unwrap();
        connection
            .execute_batch("CREATE TABLE threads (thread TEXT);")
            .unwrap();
        drop(connection);
        assert!(StateDb::open(dir.path()).unwrap().is_none());
    }

    #[test]
    fn older_schemas_without_optional_columns_still_read() {
        let dir = tempfile::tempdir().unwrap();
        let connection = Connection::open(dir.path().join("state_5.sqlite")).unwrap();
        connection
            .execute_batch(
                "CREATE TABLE threads (id TEXT, rollout_path TEXT, updated_at INTEGER);
                 INSERT INTO threads VALUES ('a', '/r/a.jsonl', 3);",
            )
            .unwrap();
        drop(connection);
        let rows = StateDb::open(dir.path()).unwrap().unwrap().rows().unwrap();
        assert_eq!(rows["a"].updated_at, 3);
        assert!(!rows["a"].paginated);
    }

    #[test]
    fn a_locked_database_is_an_error_not_an_absence() {
        let dir = tempfile::tempdir().unwrap();
        let path = create(dir.path(), 5, &[]);
        let writer = Connection::open(&path).unwrap();
        writer
            .execute_batch("PRAGMA locking_mode = EXCLUSIVE; BEGIN EXCLUSIVE;")
            .unwrap();
        let opened = StateDb::open(dir.path()).and_then(|db| match db {
            Some(db) => db.rows().map(|_| ()),
            None => Ok(()),
        });
        assert!(opened.is_err());
        writer.execute_batch("COMMIT;").unwrap();
    }
}
