//! Codex threads as the log entries the viewer, exports and display render.
//!
//! The entries are the thread's dialogue. A thread forked from another
//! starts with the history it was forked from, followed by a note marking
//! where its own records begin.

use super::{CodexAdapter, discover};
use crate::claude::{
    AssistantMessage, ContentBlock, LogEntry, TokenUsage, UserContent, UserMessage,
};
use crate::error::{AppError, Result};
use crate::history::adapter::Adapter;
use crate::history::model::{Event, EventKind, Message, Part, Role, Transcript, Usage, Visibility};
use serde_json::json;
use std::path::Path;
use std::time::SystemTime;

const AGENT: &str = "Codex";

/// The thread of the rollout at `path`, decoded through the Codex home
/// holding it. `None` when `path` is not a rollout of a Codex home.
pub fn decode_path(path: &Path) -> Option<Result<Transcript>> {
    let home = discover::home_of(path)?;
    let thread_id = discover::rollout_of(path, false)?.thread_id;
    let adapter = CodexAdapter::new(home.to_path_buf());
    Some(
        adapter
            .resolve(&thread_id, SystemTime::now())
            .and_then(|located| adapter.decode(&located))
            .and_then(|transcript| {
                transcript.ok_or_else(|| {
                    AppError::Io(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("not a Codex rollout: {}", path.display()),
                    ))
                })
            }),
    )
}

/// The log entries of the thread of the rollout at `path`, numbered by
/// their line in the rollout that holds them. `None` when `path` is not a
/// rollout of a Codex home.
pub fn log_entries(path: &Path) -> Result<Option<Vec<(usize, LogEntry)>>> {
    match decode_path(path) {
        None => Ok(None),
        Some(transcript) => Ok(Some(entries_of(&transcript?))),
    }
}

fn entries_of(transcript: &Transcript) -> Vec<(usize, LogEntry)> {
    let inherited = |event: &Event| {
        transcript
            .segments
            .get(usize::from(event.origin.segment))
            .is_some_and(|segment| segment.inherited)
    };
    let mut entries = Vec::new();
    let mut in_base = false;
    for event in &transcript.events {
        let from_base = inherited(event);
        let shown = match event.visibility {
            Visibility::Dialogue => true,
            Visibility::Inherited => from_base,
            Visibility::Injected => false,
        };
        if !shown {
            continue;
        }
        if in_base && !from_base {
            entries.push((
                event.origin.line as usize,
                LogEntry::PiMetadata {
                    label: "Fork".to_owned(),
                    text: "The conversation continues from here".to_owned(),
                    timestamp: None,
                    searchable: true,
                    usage: None,
                },
            ));
        }
        in_base = from_base;
        let timestamp = event.timestamp.map(|timestamp| timestamp.to_rfc3339());
        if let Some(entry) = entry_of(&event.kind, timestamp) {
            entries.push((event.origin.line as usize, entry));
        }
    }
    entries
}

fn entry_of(kind: &EventKind, timestamp: Option<String>) -> Option<LogEntry> {
    match kind {
        EventKind::Message(message) => Some(message_entry(message, None, timestamp)),
        EventKind::Subagent { agent_id, message } => {
            Some(message_entry(message, Some(agent_id), timestamp))
        }
        EventKind::Note {
            label,
            text,
            searchable,
            usage,
        } => Some(LogEntry::PiMetadata {
            label: label.clone(),
            text: text.clone(),
            timestamp,
            searchable: *searchable,
            usage: usage.as_ref().map(token_usage),
        }),
        EventKind::Title { .. } | EventKind::Other => None,
    }
}

fn message_entry(message: &Message, agent: Option<&str>, timestamp: Option<String>) -> LogEntry {
    let content = message.parts.iter().filter_map(block_of).collect();
    match message.role {
        Role::User => LogEntry::User {
            message: UserMessage {
                role: "user".to_owned(),
                content: UserContent::Blocks(content),
            },
            timestamp,
            uuid: None,
            cwd: None,
            parent_tool_use_id: message.parent_tool_use_id.clone(),
            usage: None,
        },
        Role::Assistant => LogEntry::Assistant {
            message: AssistantMessage {
                role: "assistant".to_owned(),
                content,
                model: message.model.clone(),
                usage: message.usage.as_ref().map(token_usage),
                id: None,
            },
            agent: Some(agent.unwrap_or(AGENT).to_owned()),
            timestamp,
            uuid: None,
            parent_tool_use_id: message.parent_tool_use_id.clone(),
        },
    }
}

fn block_of(part: &Part) -> Option<ContentBlock> {
    Some(match part {
        Part::Text(text) => ContentBlock::Text { text: text.clone() },
        Part::Thinking(thinking) => ContentBlock::Thinking {
            thinking: thinking.clone(),
            signature: String::new(),
        },
        Part::ToolCall { id, name, input } => ContentBlock::ToolUse {
            id: id.clone(),
            name: name.clone(),
            input: input.clone(),
        },
        Part::ToolResult { call_id, output } => ContentBlock::ToolResult {
            tool_use_id: call_id.clone(),
            content: output.clone(),
        },
        Part::Image => ContentBlock::Image { source: json!({}) },
        Part::Other => return None,
    })
}

fn token_usage(usage: &Usage) -> TokenUsage {
    TokenUsage {
        input_tokens: usage.input,
        output_tokens: usage.output,
        cache_creation_input_tokens: usage.cache_creation,
        cache_read_input_tokens: usage.cache_read,
    }
}
