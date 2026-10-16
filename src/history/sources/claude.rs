//! Claude Code: one JSONL file per session in each project directory of a
//! `projects` root.
//!
//! Records are Claude's `LogEntry` wire format, which only this module
//! decodes into events.

use crate::claude::{AgentContent, ContentBlock, LogEntry, TokenUsage, UserContent};
use crate::error::{AppError, Result};
use crate::history::ParseError;
use crate::history::actions::{Capabilities, Launch, Launcher, Resume, SessionCopy, Target};
use crate::history::adapter::{Adapter, Completeness, Discovery, Located};
use crate::history::model::{
    ConversationKey, Event, EventKind, ExecutionMode, HeadStatus, Message, Origin, Part,
    Relationship, Revision, Role, Segment, SessionMeta, SourceInstance, SourceKind, TitleKind,
    Transcript, Usage, Visibility,
};
use crate::history::path::decode_project_dir_name_to_path;
use chrono::{DateTime, Utc};
use rayon::prelude::*;
use std::collections::VecDeque;
use std::fs;
use std::io::{self, BufRead};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

pub struct ClaudeAdapter {
    instance: SourceInstance,
}

impl ClaudeAdapter {
    /// The adapter of a `projects` directory.
    pub fn new(root: PathBuf) -> Self {
        Self {
            instance: SourceInstance {
                kind: SourceKind::Claude,
                root,
            },
        }
    }
}

impl Adapter for ClaudeAdapter {
    fn instance(&self) -> &SourceInstance {
        &self.instance
    }

    fn discover(&self, scan_started: SystemTime) -> Discovery {
        let mut discovery = Discovery::default();
        let files = list_sessions(&self.instance.root, &mut discovery);
        discovery.found = super::locate_files(&self.instance, files, scan_started);
        discovery
    }

    fn resolve(&self, native_id: &str, scan_started: SystemTime) -> Result<Located> {
        super::resolve_file(&self.instance, native_id, scan_started)
    }

    fn decode_into(
        &self,
        located: &Located,
        events: &mut dyn FnMut(Event),
    ) -> Result<Option<Transcript>> {
        super::decode_file(located, false, events)
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities::ALL
    }

    fn resume(&self, target: Target<'_>, fork: bool, launch: &Launch) -> Result<Resume> {
        let cwd = std::env::current_dir().map_err(|e| {
            AppError::Io(io::Error::new(
                io::ErrorKind::NotFound,
                format!("Failed to get current directory: {}", e),
            ))
        })?;
        resume_plan(target, &cwd, fork, launch)
    }

    /// Deletes every transcript of the session, in any project, with its
    /// session directory.
    fn delete(&self, path: &Path, _launcher: &Launcher) -> Result<()> {
        let uuid = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        crate::history::delete_session_by_uuid(uuid).map(|_| ())
    }

    fn rename(&self, path: &Path, title: &str) -> Result<()> {
        crate::history::append_session_rename(path, title)
    }
}

#[derive(Debug, PartialEq, Eq)]
enum ResumeAction {
    Run { current_dir: PathBuf },
    CopyToCurrent { cwd_projects_dir: PathBuf },
}

/// Claude resumes a session only from the project directory that holds it.
/// Run there when `cwd` or the conversation's project maps to that
/// directory; otherwise (and to fork into another project) copy the session
/// into `cwd`'s project directory first.
fn resolve_resume_action(
    selected_path: &Path,
    project_path: Option<&Path>,
    cwd: &Path,
    fork_session: bool,
) -> Result<ResumeAction> {
    let conv_projects_dir = selected_path.parent().ok_or_else(|| {
        AppError::ClaudeExecutionError(
            "Cannot determine conversation's project directory".to_string(),
        )
    })?;
    let cwd_projects_dir = crate::history::get_claude_projects_dir(cwd)?;
    let project_dir = project_path.filter(|p| p.exists() && p.is_dir());

    let same_dir = |dir: &Path| crate::history::path::same_path(dir, conv_projects_dir);
    if project_dir.is_none() || (fork_session && !same_dir(&cwd_projects_dir)) {
        return Ok(ResumeAction::CopyToCurrent { cwd_projects_dir });
    }

    if same_dir(&cwd_projects_dir) {
        return Ok(ResumeAction::Run {
            current_dir: cwd.to_path_buf(),
        });
    }

    let project_dir = project_dir.unwrap();
    let project_projects_dir = crate::history::get_claude_projects_dir(project_dir)?;
    if same_dir(&project_projects_dir) {
        Ok(ResumeAction::Run {
            current_dir: project_dir.to_path_buf(),
        })
    } else {
        Ok(ResumeAction::CopyToCurrent { cwd_projects_dir })
    }
}

fn resume_plan(
    target: Target<'_>,
    cwd: &Path,
    fork_session: bool,
    launch: &Launch,
) -> Result<Resume> {
    let default_args = &launch.default_args;
    let selected_path = target.path;
    let conversation_id = selected_path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .ok_or_else(|| {
            AppError::ClaudeExecutionError("Conversation filename is not valid Unicode".to_string())
        })?
        .to_owned();

    let conv_projects_dir = selected_path.parent().ok_or_else(|| {
        AppError::ClaudeExecutionError(
            "Cannot determine conversation's project directory".to_string(),
        )
    })?;

    let mut command = launch.launcher.command();
    command.args(["--resume", &conversation_id]);
    match resolve_resume_action(selected_path, target.project, cwd, fork_session)? {
        ResumeAction::CopyToCurrent { cwd_projects_dir } => {
            command.args(default_args);
            command.current_dir(cwd);
            Ok(Resume {
                copy: Some(SessionCopy {
                    transcript: selected_path.to_path_buf(),
                    session_id: conversation_id,
                    from_dir: conv_projects_dir.to_path_buf(),
                    to_dir: cwd_projects_dir,
                }),
                command,
            })
        }
        ResumeAction::Run { current_dir } => {
            if fork_session {
                command.arg("--fork-session");
            }
            command.args(default_args);
            command.current_dir(current_dir);
            Ok(Resume {
                copy: None,
                command,
            })
        }
    }
}

/// Session files: `*.jsonl` directly inside each project directory, except
/// subagent transcripts (`agent-*`). A missing root holds none.
fn list_sessions(root: &Path, discovery: &mut Discovery) -> Vec<PathBuf> {
    let entries = match fs::read_dir(root) {
        Ok(entries) => entries,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Vec::new(),
        Err(_) => {
            discovery.completeness = Completeness::Degraded;
            return Vec::new();
        }
    };
    let mut dirs = Vec::new();
    for entry in entries {
        match entry {
            Ok(entry) => dirs.push(entry.path()),
            Err(_) => discovery.completeness = Completeness::Degraded,
        }
    }
    let listed: Vec<(PathBuf, io::Result<Vec<PathBuf>>)> = dirs
        .into_par_iter()
        .filter(|dir| dir.is_dir())
        .map(|dir| {
            let files = list_project_dir(&dir);
            (dir, files)
        })
        .collect();
    let mut files = Vec::new();
    for (dir, listed) in listed {
        match listed {
            Ok(listed) => files.extend(listed),
            Err(_) => discovery.missed_dir(dir),
        }
    }
    files
}

fn list_project_dir(dir: &Path) -> io::Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if is_session_file(&path) {
            files.push(path);
        }
    }
    Ok(files)
}

fn is_session_file(path: &Path) -> bool {
    path.extension()
        .is_some_and(|extension| extension == "jsonl")
        && !path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with("agent-"))
}

/// Decodes a Claude session from `reader`, the contents of `path`, passing
/// its events to `events`. The session was last active at its latest
/// message, else when it was `modified`.
pub(crate) fn decode_reader(
    key: ConversationKey,
    path: &Path,
    reader: impl BufRead,
    modified: Option<SystemTime>,
    events: &mut dyn FnMut(Event),
) -> Result<Transcript> {
    let mut lines = Lines::new(reader);
    // The current line, plus up to 2 lines after it for parse error context.
    let mut window: VecDeque<String> = VecDeque::with_capacity(3);
    let mut context_before: VecDeque<String> = VecDeque::with_capacity(2);
    for _ in 0..2 {
        match lines.next_line()? {
            Some(line) => window.push_back(line),
            None => break,
        }
    }
    let mut records = Records::default();
    let mut activity = super::LastActivity::default();
    let mut diagnostics = Vec::new();
    let mut line_index: usize = 0;
    while let Some(line) = window.pop_front() {
        if let Some(next) = lines.next_line()? {
            window.push_back(next);
        }
        if !line.trim().is_empty() {
            match serde_json::from_str::<LogEntry>(&line) {
                Ok(entry) => {
                    if let Some(event) = records.push(entry, line_number(line_index)) {
                        activity.observe(&event);
                        events(event);
                    }
                }
                Err(error) => diagnostics.push(ParseError {
                    line_number: line_index + 1,
                    line_content: line.clone(),
                    error_message: error.to_string(),
                    context_before: context_before.iter().cloned().collect(),
                    context_after: window.iter().cloned().collect(),
                }),
            }
        }
        context_before.push_back(line);
        if context_before.len() > 2 {
            context_before.pop_front();
        }
        line_index += 1;
    }

    let segments = vec![Segment {
        artifact: path.to_path_buf(),
        start: 0,
        end: lines.bytes,
        inherited: false,
    }];
    let project = records.cwd.clone().unwrap_or_else(|| {
        path.parent()
            .and_then(Path::file_name)
            .map(|dir| decode_project_dir_name_to_path(&dir.to_string_lossy()))
            .unwrap_or_default()
    });
    let meta = SessionMeta {
        session_id: path
            .file_stem()
            .and_then(|name| name.to_str())
            .unwrap_or_default()
            .to_owned(),
        cwd: records.cwd.take(),
        project: Some(project),
        model: records.model.take(),
        summary: records.summary.take(),
        custom_title: records.custom_title.take(),
        updated_at: activity
            .at()
            .or_else(|| modified.map(DateTime::<Utc>::from)),
    };
    Ok(Transcript {
        revision: Revision::of(SourceKind::Claude, &segments, &[mtime_ns(modified)]),
        key,
        source: SourceKind::Claude,
        head: HeadStatus::Current,
        mode: ExecutionMode::Unknown,
        relationship: Relationship::Root,
        meta,
        segments,
        records: records.count,
        events: Vec::new(),
        diagnostics,
    })
}

/// Decodes `reader` as the Claude session at `path`, outside any root.
#[cfg(test)]
pub(crate) fn decode_test_reader(path: &Path, reader: impl BufRead) -> Result<Transcript> {
    let instance = SourceInstance {
        kind: SourceKind::Claude,
        root: PathBuf::new(),
    };
    let key = ConversationKey {
        instance,
        native_id: path.to_string_lossy().into_owned(),
    };
    let mut decoded = Vec::new();
    let mut transcript = decode_reader(key, path, reader, None, &mut |event| decoded.push(event))?;
    transcript.events = decoded;
    Ok(transcript)
}

fn line_number(index: usize) -> Origin {
    Origin {
        segment: 0,
        line: u32::try_from(index + 1).unwrap_or(u32::MAX),
    }
}

pub(super) fn mtime_ns(modified: Option<SystemTime>) -> i64 {
    modified
        .and_then(|modified| modified.duration_since(SystemTime::UNIX_EPOCH).ok())
        .map_or(0, |since| i64::try_from(since.as_nanos()).unwrap_or(0))
}

/// Lines of a reader as `BufRead::lines` splits them, counting the bytes
/// read.
struct Lines<R> {
    reader: R,
    bytes: u64,
}

impl<R: BufRead> Lines<R> {
    fn new(reader: R) -> Self {
        Self { reader, bytes: 0 }
    }

    fn next_line(&mut self) -> io::Result<Option<String>> {
        let mut line = String::new();
        let read = self.reader.read_line(&mut line)?;
        if read == 0 {
            return Ok(None);
        }
        self.bytes += read as u64;
        if line.ends_with('\n') {
            line.pop();
            if line.ends_with('\r') {
                line.pop();
            }
        }
        Ok(Some(line))
    }
}

/// Converts Claude-format records into events, in order, collecting the
/// session metadata they define: the first user record's cwd, the first
/// assistant model, the first summary unless a generated title replaces
/// it, and the last custom title. A file-history snapshot, which records
/// file states for undo rather than the conversation, yields no event.
#[derive(Default)]
pub(crate) struct Records {
    count: usize,
    cwd: Option<PathBuf>,
    model: Option<String>,
    summary: Option<String>,
    custom_title: Option<String>,
}

impl Records {
    pub(crate) fn push(&mut self, entry: LogEntry, origin: Origin) -> Option<Event> {
        self.count += 1;
        let (timestamp, kind) = match entry {
            LogEntry::User {
                message,
                timestamp,
                cwd,
                parent_tool_use_id,
                usage,
                ..
            } => {
                if self.cwd.is_none()
                    && let Some(cwd) = cwd
                {
                    self.cwd = Some(PathBuf::from(cwd));
                }
                let parts = match message.content {
                    UserContent::String(text) => vec![Part::Text(text)],
                    UserContent::Blocks(blocks) => parts(blocks),
                };
                (
                    timestamp,
                    EventKind::Message(Message {
                        role: Role::User,
                        id: None,
                        parts,
                        model: None,
                        usage: usage.map(usage_of),
                        parent_tool_use_id,
                    }),
                )
            }
            LogEntry::Assistant {
                message,
                timestamp,
                parent_tool_use_id,
                ..
            } => {
                if self.model.is_none() {
                    self.model.clone_from(&message.model);
                }
                (
                    timestamp,
                    EventKind::Message(Message {
                        role: Role::Assistant,
                        id: message.id,
                        parts: parts(message.content),
                        model: message.model,
                        usage: message.usage.map(usage_of),
                        parent_tool_use_id,
                    }),
                )
            }
            LogEntry::Summary { summary } => {
                if self.summary.is_none() {
                    self.summary = Some(summary.clone());
                }
                (None, title(TitleKind::Summary, summary))
            }
            LogEntry::AiTitle { ai_title } => {
                let trimmed = ai_title.trim();
                if !trimmed.is_empty() {
                    self.summary = Some(trimmed.to_owned());
                }
                (None, title(TitleKind::Generated, ai_title))
            }
            LogEntry::CustomTitle { custom_title } => {
                let trimmed = custom_title.trim();
                self.custom_title = (!trimmed.is_empty()).then(|| trimmed.to_owned());
                (None, title(TitleKind::Custom, custom_title))
            }
            LogEntry::Progress { data, .. } => (None, subagent(&data)),
            LogEntry::FileHistorySnapshot { .. } => return None,
            LogEntry::System { .. }
            | LogEntry::AgentName { .. }
            | LogEntry::PermissionMode { .. }
            | LogEntry::Unknown => (None, EventKind::Other),
        };
        Some(Event {
            origin,
            timestamp: timestamp.as_deref().and_then(parse_timestamp),
            visibility: Visibility::Dialogue,
            kind,
        })
    }
}

fn title(kind: TitleKind, text: String) -> EventKind {
    EventKind::Title { kind, text }
}

/// The subagent message a progress record reports, if it reports one.
fn subagent(data: &serde_json::Value) -> EventKind {
    let Some(progress) = crate::claude::parse_agent_progress(data) else {
        return EventKind::Other;
    };
    let role = match progress.message.message_type.as_str() {
        "user" => Role::User,
        "assistant" => Role::Assistant,
        _ => return EventKind::Other,
    };
    let AgentContent::Blocks(blocks) = progress.message.message.content;
    EventKind::Subagent {
        agent_id: progress.agent_id,
        message: Message {
            role,
            id: None,
            parts: parts(blocks),
            model: None,
            usage: None,
            parent_tool_use_id: None,
        },
    }
}

fn parts(blocks: Vec<ContentBlock>) -> Vec<Part> {
    blocks
        .into_iter()
        .map(|block| match block {
            ContentBlock::Text { text } => Part::Text(text),
            ContentBlock::ToolUse { id, name, input } => Part::ToolCall { id, name, input },
            ContentBlock::ToolResult {
                tool_use_id,
                content,
            } => Part::ToolResult {
                call_id: tool_use_id,
                output: content,
            },
            ContentBlock::Thinking { thinking, .. } => Part::Thinking(thinking),
            ContentBlock::Image { .. } => Part::Image,
            ContentBlock::Other => Part::Other,
        })
        .collect()
}

fn usage_of(usage: TokenUsage) -> Usage {
    Usage {
        input: usage.input_tokens,
        output: usage.output_tokens,
        cache_creation: usage.cache_creation_input_tokens,
        cache_read: usage.cache_read_input_tokens,
    }
}

pub(super) fn parse_timestamp(text: &str) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(text)
        .ok()
        .map(|timestamp| timestamp.with_timezone(&Utc))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode_with_mtime(lines: &[&str], modified: SystemTime) -> Transcript {
        let path = Path::new("/projects/-work/session.jsonl");
        let key = ConversationKey {
            instance: SourceInstance {
                kind: SourceKind::Claude,
                root: PathBuf::new(),
            },
            native_id: path.to_string_lossy().into_owned(),
        };
        let contents = lines.join("\n");
        decode_reader(key, path, contents.as_bytes(), Some(modified), &mut |_| {}).unwrap()
    }

    #[test]
    fn last_activity_is_the_newest_message_not_the_file_mtime() {
        let modified = SystemTime::now();
        let transcript = decode_with_mtime(
            &[
                r#"{"type":"user","timestamp":"2026-10-15T14:50:00Z","message":{"role":"user","content":"hi"}}"#,
                r#"{"type":"assistant","timestamp":"2026-10-15T14:51:00Z","message":{"role":"assistant","content":[{"type":"tool_use","id":"t","name":"Bash","input":{}}]}}"#,
                r#"{"type":"user","timestamp":"2026-10-15T14:51:56Z","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"t","content":"ok"}]}}"#,
                r#"{"type":"system","subtype":"turn_duration","timestamp":"2026-10-15T14:52:30Z"}"#,
                r#"{"type":"system","subtype":"away_summary","timestamp":"2026-10-16T14:44:00Z"}"#,
            ],
            modified,
        );
        assert_eq!(
            transcript.meta.updated_at.unwrap().to_rfc3339(),
            "2026-10-15T14:51:56+00:00"
        );
    }

    #[test]
    fn a_session_without_message_times_was_last_active_at_its_mtime() {
        let modified = SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(1_700_000_000);
        let transcript = decode_with_mtime(
            &[r#"{"type":"user","message":{"role":"user","content":"hi"}}"#],
            modified,
        );
        assert_eq!(
            transcript.meta.updated_at,
            Some(DateTime::<Utc>::from(modified))
        );
    }

    #[test]
    fn resume_runs_the_configured_command() {
        let cwd = tempfile::tempdir().unwrap();
        let selected_path = crate::history::get_claude_projects_dir(cwd.path())
            .unwrap()
            .join("12345678-1234-4234-9234-123456789abc.jsonl");
        let target = Target {
            path: &selected_path,
            project: Some(cwd.path()),
        };
        let launch = Launch {
            launcher: Launcher::configured(
                &[
                    "nix".to_owned(),
                    "run".to_owned(),
                    ".#claude".to_owned(),
                    "--".to_owned(),
                ],
                "[resume]",
            ),
            default_args: vec!["--verbose".to_owned()],
        };

        let resume = resume_plan(target, cwd.path(), true, &launch).unwrap();

        assert!(resume.copy.is_none());
        assert_eq!(resume.command.get_program(), "nix");
        assert_eq!(
            resume.command.get_args().collect::<Vec<_>>(),
            [
                "run",
                ".#claude",
                "--",
                "--resume",
                "12345678-1234-4234-9234-123456789abc",
                "--fork-session",
                "--verbose"
            ]
        );
    }

    #[test]
    fn resume_action_uses_cwd_when_it_maps_to_selected_project_dir() {
        let cwd = tempfile::tempdir().unwrap();
        let stale_project = tempfile::tempdir().unwrap();
        let stale_project_path = stale_project.path().to_path_buf();
        let selected_path = crate::history::get_claude_projects_dir(cwd.path())
            .unwrap()
            .join("12345678-1234-4234-9234-123456789abc.jsonl");

        let action = resolve_resume_action(
            &selected_path,
            Some(stale_project_path.as_path()),
            cwd.path(),
            false,
        )
        .unwrap();

        assert_eq!(
            action,
            ResumeAction::Run {
                current_dir: cwd.path().to_path_buf()
            }
        );
    }

    #[test]
    fn resume_action_uses_project_path_when_it_maps_to_selected_project_dir() {
        let cwd = tempfile::tempdir().unwrap();
        let project = tempfile::tempdir().unwrap();
        let project_path = project.path().to_path_buf();
        let selected_path = crate::history::get_claude_projects_dir(project.path())
            .unwrap()
            .join("12345678-1234-4234-9234-123456789abc.jsonl");

        let action = resolve_resume_action(
            &selected_path,
            Some(project_path.as_path()),
            cwd.path(),
            false,
        )
        .unwrap();

        assert_eq!(
            action,
            ResumeAction::Run {
                current_dir: project.path().to_path_buf()
            }
        );
    }

    #[test]
    fn resume_action_copies_selected_transcript_when_project_path_maps_elsewhere() {
        let cwd = tempfile::tempdir().unwrap();
        let selected_project = tempfile::tempdir().unwrap();
        let stale_project = tempfile::tempdir().unwrap();
        let stale_project_path = stale_project.path().to_path_buf();
        let selected_path = crate::history::get_claude_projects_dir(selected_project.path())
            .unwrap()
            .join("12345678-1234-4234-9234-123456789abc.jsonl");
        let cwd_projects_dir = crate::history::get_claude_projects_dir(cwd.path()).unwrap();

        let action = resolve_resume_action(
            &selected_path,
            Some(stale_project_path.as_path()),
            cwd.path(),
            false,
        )
        .unwrap();

        assert_eq!(action, ResumeAction::CopyToCurrent { cwd_projects_dir });
    }
}
