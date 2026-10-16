//! Claude Code: one JSONL file per session in each project directory of a
//! `projects` root.
//!
//! Records are Claude's `LogEntry` wire format, which only this module
//! decodes into events. Pi and OMP sessions are projected into the same
//! records, so their adapters share the conversion (`Records`).

use crate::claude::{AgentContent, ContentBlock, LogEntry, TokenUsage, UserContent};
use crate::error::Result;
use crate::history::ParseError;
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

    fn decode(&self, located: &Located) -> Result<Option<Transcript>> {
        super::decode_file(located, false)
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

/// Decodes a Claude session from `reader`, the contents of `path`.
pub(crate) fn decode_reader(
    key: ConversationKey,
    path: &Path,
    reader: impl BufRead,
    modified: Option<SystemTime>,
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
    let mut diagnostics = Vec::new();
    let mut line_index: usize = 0;
    while let Some(line) = window.pop_front() {
        if let Some(next) = lines.next_line()? {
            window.push_back(next);
        }
        if !line.trim().is_empty() {
            match serde_json::from_str::<LogEntry>(&line) {
                Ok(entry) => records.push(entry, line_number(line_index)),
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
        updated_at: modified.map(DateTime::<Utc>::from),
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
        events: records.events,
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
    decode_reader(key, path, reader, None)
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

/// Claude-format records converted into events, in order, with the session
/// metadata they define: the first user record's cwd, the first assistant
/// model unless a `Model` note names one, the first summary unless a
/// generated title replaces it, and the last custom title.
#[derive(Default)]
pub(crate) struct Records {
    pub events: Vec<Event>,
    pub count: usize,
    pub cwd: Option<PathBuf>,
    pub model: Option<String>,
    pub summary: Option<String>,
    pub custom_title: Option<String>,
}

impl Records {
    pub fn push(&mut self, entry: LogEntry, origin: Origin) {
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
            LogEntry::PiMetadata {
                label,
                text,
                timestamp,
                searchable,
                usage,
            } => {
                if label == "Model" && !text.is_empty() {
                    self.model = Some(text.clone());
                }
                (
                    timestamp,
                    EventKind::Note {
                        label,
                        text,
                        searchable,
                        usage: usage.map(usage_of),
                    },
                )
            }
            LogEntry::Progress { data, .. } => (None, subagent(&data)),
            LogEntry::FileHistorySnapshot { .. }
            | LogEntry::System { .. }
            | LogEntry::AgentName { .. }
            | LogEntry::PermissionMode { .. }
            | LogEntry::Unknown => (None, EventKind::Other),
        };
        self.events.push(Event {
            origin,
            timestamp: timestamp.as_deref().and_then(parse_timestamp),
            visibility: Visibility::Dialogue,
            kind,
        });
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
