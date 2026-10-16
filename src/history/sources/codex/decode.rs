//! Rollout records into transcript events.
//!
//! Every rollout line is `{timestamp, ordinal?, type, payload}`. The first
//! is the rollout's `session_meta`. A legacy rollout records the dialogue
//! as `event_msg` user, agent and reasoning messages; a paginated one as
//! `event_msg` `item_completed` turn items. Both record what the model saw
//! as `response_item`s, which carry the tool calls and their outputs; the
//! messages among them repeat the dialogue plus injected context, so they
//! count as injected. `token_count` events report cumulative usage.

use super::discover;
use super::head::{Link, Thread};
use crate::error::Result;
use crate::history::ParseError;
use crate::history::model::{
    ConversationKey, Event, EventKind, ExecutionMode, Message, Origin, Part, Relationship,
    Revision, Role, Segment, SessionMeta, SourceKind, Transcript, Usage, Visibility,
};
use crate::history::sources::claude::{mtime_ns, parse_timestamp};
use chrono::{DateTime, Utc};
use serde::Deserialize;
use serde_json::Value;
use serde_json::value::RawValue;
use std::io::BufRead;
use std::path::PathBuf;

#[derive(Deserialize)]
struct Line<'a> {
    #[serde(default)]
    timestamp: Option<String>,
    #[serde(default)]
    ordinal: Option<u64>,
    #[serde(rename = "type")]
    kind: String,
    #[serde(borrow)]
    payload: &'a RawValue,
}

#[derive(Default, Deserialize)]
struct Meta {
    #[serde(default)]
    timestamp: Option<String>,
    #[serde(default)]
    cwd: Option<PathBuf>,
    #[serde(default)]
    source: Option<Value>,
    #[serde(default)]
    parent_thread_id: Option<String>,
    #[serde(default)]
    history_mode: Option<String>,
    /// Records before this ordinal are the parent's context a subagent
    /// starts with.
    #[serde(default)]
    subagent_history_start_ordinal: Option<u64>,
}

impl Meta {
    fn paginated(&self) -> bool {
        self.history_mode.as_deref() == Some("paginated")
    }
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum EventMsg {
    UserMessage {
        #[serde(default)]
        message: String,
        #[serde(default)]
        images: Option<Vec<Value>>,
        #[serde(default)]
        local_images: Option<Vec<Value>>,
    },
    AgentMessage {
        #[serde(default)]
        message: String,
    },
    AgentReasoning {
        #[serde(default)]
        text: String,
    },
    AgentReasoningRawContent {
        #[serde(default)]
        text: String,
    },
    ItemCompleted {
        item: TurnItem,
    },
    TokenCount {
        #[serde(default)]
        info: Option<TokenInfo>,
    },
    #[serde(other)]
    Other,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum TurnItem {
    UserMessage {
        #[serde(default)]
        content: Vec<UserInput>,
    },
    AgentMessage {
        #[serde(default)]
        id: Option<String>,
        #[serde(default)]
        content: Vec<AgentContent>,
    },
    Reasoning {
        #[serde(default)]
        id: Option<String>,
        #[serde(default)]
        summary_text: Vec<String>,
        #[serde(default)]
        raw_content: Vec<String>,
    },
    #[serde(other)]
    Other,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum UserInput {
    Text {
        text: String,
    },
    Image {},
    LocalImage {},
    #[serde(other)]
    Other,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum AgentContent {
    Text {
        text: String,
    },
    #[serde(other)]
    Other,
}

#[derive(Deserialize)]
struct TokenInfo {
    total_token_usage: TokenUsage,
}

#[derive(Clone, Copy, Default, Deserialize, Eq, PartialEq)]
struct TokenUsage {
    #[serde(default)]
    input_tokens: u64,
    #[serde(default)]
    cached_input_tokens: u64,
    #[serde(default)]
    cache_write_input_tokens: u64,
    #[serde(default)]
    output_tokens: u64,
}

impl TokenUsage {
    fn total(&self) -> u64 {
        self.input_tokens + self.output_tokens
    }

    /// Usage of the calls between `previous` and `self`, cumulative totals
    /// of the same thread. A smaller total starts a new count.
    fn since(&self, previous: &Self) -> Usage {
        let delta = if self.total() < previous.total() {
            *self
        } else {
            Self {
                input_tokens: self.input_tokens.saturating_sub(previous.input_tokens),
                cached_input_tokens: self
                    .cached_input_tokens
                    .saturating_sub(previous.cached_input_tokens),
                cache_write_input_tokens: self
                    .cache_write_input_tokens
                    .saturating_sub(previous.cache_write_input_tokens),
                output_tokens: self.output_tokens.saturating_sub(previous.output_tokens),
            }
        };
        // Input tokens include the cached ones.
        Usage {
            input: delta
                .input_tokens
                .saturating_sub(delta.cached_input_tokens)
                .saturating_sub(delta.cache_write_input_tokens),
            output: delta.output_tokens,
            cache_creation: delta.cache_write_input_tokens,
            cache_read: delta.cached_input_tokens,
        }
    }
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ResponseItem {
    Message {
        #[serde(default)]
        role: String,
        #[serde(default)]
        content: Vec<ContentItem>,
    },
    FunctionCall {
        #[serde(default)]
        id: Option<String>,
        name: String,
        #[serde(default)]
        namespace: Option<String>,
        #[serde(default)]
        arguments: String,
        call_id: String,
    },
    CustomToolCall {
        #[serde(default)]
        id: Option<String>,
        name: String,
        #[serde(default)]
        input: String,
        call_id: String,
    },
    LocalShellCall {
        #[serde(default)]
        id: Option<String>,
        #[serde(default)]
        call_id: Option<String>,
        #[serde(default)]
        action: Value,
    },
    WebSearchCall {
        #[serde(default)]
        id: Option<String>,
        #[serde(default)]
        action: Value,
    },
    ToolSearchCall {
        #[serde(default)]
        id: Option<String>,
        #[serde(default)]
        call_id: Option<String>,
        #[serde(default)]
        arguments: Value,
    },
    FunctionCallOutput {
        #[serde(default)]
        call_id: Option<String>,
        output: Value,
    },
    CustomToolCallOutput {
        #[serde(default)]
        call_id: Option<String>,
        output: Value,
    },
    ToolSearchOutput {
        #[serde(default)]
        call_id: Option<String>,
        #[serde(default)]
        tools: Value,
    },
    #[serde(other)]
    Other,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ContentItem {
    InputText {
        text: String,
    },
    OutputText {
        text: String,
    },
    InputImage {},
    #[serde(other)]
    Other,
}

#[derive(Deserialize)]
struct TurnContext {
    #[serde(default)]
    model: Option<String>,
}

#[derive(Deserialize)]
struct Compacted {
    #[serde(default)]
    message: String,
}

/// Decodes the history of `thread`, passing its events to `events`.
/// `None` when the head rollout does not start with a `session_meta`.
pub fn decode(
    key: ConversationKey,
    thread: &Thread,
    events: &mut dyn FnMut(Event),
) -> Result<Option<Transcript>> {
    let mut decoder = Decoder {
        events,
        segments: Vec::new(),
        diagnostics: Vec::new(),
        records: 0,
        usage: TokenUsage::default(),
        model: None,
        head_meta: None,
        last_timestamp: None,
    };
    if let Some(missing) = &thread.missing_base {
        decoder.diagnostics.push(diagnostic(
            0,
            String::new(),
            format!("history base rollout {missing} not found"),
        ));
    }
    let head = thread.chain.len() - 1;
    for (index, link) in thread.chain.iter().enumerate() {
        decoder.segment(link, index == head)?;
    }
    let Some(meta) = decoder.head_meta.take() else {
        return Ok(None);
    };
    let head_link = thread.head_link();
    let updated_at = decoder
        .last_timestamp
        .as_deref()
        .or(meta.timestamp.as_deref())
        .and_then(parse_timestamp)
        .or_else(|| head_link.modified.map(DateTime::<Utc>::from));
    let mtimes: Vec<i64> = thread
        .chain
        .iter()
        .map(|link| mtime_ns(link.modified))
        .collect();
    Ok(Some(Transcript {
        revision: Revision::of(SourceKind::Codex, &decoder.segments, &mtimes),
        key,
        source: SourceKind::Codex,
        head: thread.head,
        mode: mode_of(meta.source.as_ref()),
        relationship: relationship_of(&meta),
        meta: SessionMeta {
            session_id: thread.id.clone(),
            cwd: meta.cwd.clone(),
            project: meta.cwd.clone(),
            model: decoder.model,
            summary: None,
            custom_title: thread.name.clone(),
            updated_at,
        },
        segments: decoder.segments,
        events: Vec::new(),
        records: decoder.records,
        diagnostics: decoder.diagnostics,
    }))
}

/// `exec` runs are scripted; the TUI, IDE and app sessions are a person's.
fn mode_of(source: Option<&Value>) -> ExecutionMode {
    match source.map_or(Some("vscode"), Value::as_str) {
        Some("cli" | "vscode") => ExecutionMode::Interactive,
        Some("exec" | "mcp") => ExecutionMode::Automated,
        _ => ExecutionMode::Unknown,
    }
}

/// Subagents of every kind are children, of the thread that spawned them
/// when the rollout names it.
fn relationship_of(meta: &Meta) -> Relationship {
    let subagent = meta
        .source
        .as_ref()
        .and_then(|source| source.get("subagent"));
    let spawned_by = subagent
        .and_then(|subagent| subagent.get("thread_spawn"))
        .and_then(|spawn| spawn.get("parent_thread_id"))
        .and_then(Value::as_str);
    match (subagent, spawned_by.or(meta.parent_thread_id.as_deref())) {
        (_, Some(parent)) => Relationship::Child {
            parent: parent.to_owned(),
        },
        (Some(_), None) => Relationship::Child {
            parent: String::new(),
        },
        (None, None) => Relationship::Root,
    }
}

struct Decoder<'a> {
    events: &'a mut dyn FnMut(Event),
    segments: Vec<Segment>,
    diagnostics: Vec<ParseError>,
    records: usize,
    /// Cumulative usage of the last `token_count`, across segments.
    usage: TokenUsage,
    model: Option<String>,
    head_meta: Option<Meta>,
    /// Timestamp of the head rollout's last record.
    last_timestamp: Option<String>,
}

impl Decoder<'_> {
    /// Decodes the rollout of `link`, up to its end.
    fn segment(&mut self, link: &Link, is_head: bool) -> Result<()> {
        let segment = u16::try_from(self.segments.len()).unwrap_or(u16::MAX);
        let mut reader = discover::open(&link.path)?;
        let mut line = Vec::new();
        let mut consumed: u64 = 0;
        let mut meta: Option<Meta> = None;
        let mut line_number: u32 = 0;
        loop {
            if link.end.is_some_and(|end| consumed >= end) {
                break;
            }
            line.clear();
            let read = reader.read_until(b'\n', &mut line)?;
            if read == 0 {
                break;
            }
            consumed += read as u64;
            line_number += 1;
            let complete = line.last() == Some(&b'\n');
            let text = line.trim_ascii();
            if text.is_empty() {
                continue;
            }
            self.records += 1;
            let record: Line<'_> = match serde_json::from_slice(text) {
                Ok(record) => record,
                Err(error) => {
                    let message = if complete {
                        format!("malformed Codex rollout record: {error}")
                    } else {
                        "incomplete last record, still being written".to_owned()
                    };
                    self.diagnostics.push(diagnostic(
                        line_number as usize,
                        String::from_utf8_lossy(text).into_owned(),
                        message,
                    ));
                    continue;
                }
            };
            let origin = Origin {
                segment,
                line: line_number,
            };
            if meta.is_none() {
                if record.kind != "session_meta" {
                    break;
                }
                meta = Some(serde_json::from_str(record.payload.get()).unwrap_or_default());
                continue;
            }
            let file_meta = meta.as_ref().expect("set above");
            let ordinal = record.ordinal.unwrap_or(u64::from(line_number - 1));
            let inherited = link.inherited
                || file_meta
                    .subagent_history_start_ordinal
                    .is_some_and(|start| ordinal < start);
            if is_head && !inherited {
                self.last_timestamp.clone_from(&record.timestamp);
            }
            let visibility = if inherited {
                Visibility::Inherited
            } else {
                Visibility::Dialogue
            };
            let timestamp = record.timestamp.as_deref().and_then(parse_timestamp);
            let paginated = file_meta.paginated();
            for (visibility, kind) in self.kinds(&record, paginated, visibility) {
                (self.events)(Event {
                    origin,
                    timestamp,
                    visibility,
                    kind,
                });
            }
        }
        self.segments.push(Segment {
            artifact: link.path.clone(),
            start: 0,
            end: consumed,
            inherited: link.inherited,
        });
        if is_head {
            self.head_meta = meta;
        }
        Ok(())
    }

    /// The events of one record, each with its visibility.
    fn kinds(
        &mut self,
        record: &Line<'_>,
        paginated: bool,
        visibility: Visibility,
    ) -> Vec<(Visibility, EventKind)> {
        let payload = record.payload.get();
        let kind = match record.kind.as_str() {
            "event_msg" => match serde_json::from_str(payload) {
                Ok(event) => self.event_msg(event, paginated),
                Err(_) => EventKind::Other,
            },
            "response_item" => match serde_json::from_str(payload) {
                Ok(ResponseItem::Message { role, content }) => {
                    let visibility = match visibility {
                        Visibility::Dialogue => Visibility::Injected,
                        other => other,
                    };
                    let role = if role == "assistant" {
                        Role::Assistant
                    } else {
                        Role::User
                    };
                    let parts = content.into_iter().filter_map(content_part).collect();
                    return vec![(visibility, message(role, None, parts))];
                }
                Ok(item) => response_item(item),
                Err(_) => EventKind::Other,
            },
            "turn_context" => {
                if self.model.is_none()
                    && let Ok(context) = serde_json::from_str::<TurnContext>(payload)
                {
                    self.model = context.model;
                }
                EventKind::Other
            }
            "compacted" => EventKind::Note {
                label: "Compaction".to_owned(),
                text: serde_json::from_str::<Compacted>(payload)
                    .map(|compacted| compacted.message)
                    .unwrap_or_default(),
                searchable: false,
                usage: None,
            },
            _ => EventKind::Other,
        };
        vec![(visibility, kind)]
    }

    fn event_msg(&mut self, event: EventMsg, paginated: bool) -> EventKind {
        match event {
            EventMsg::ItemCompleted { item } if paginated => turn_item(item),
            EventMsg::UserMessage {
                message: text,
                images,
                local_images,
            } if !paginated => {
                let mut parts = Vec::new();
                if !text.is_empty() {
                    parts.push(Part::Text(text));
                }
                let images = images.map_or(0, |images| images.len())
                    + local_images.map_or(0, |images| images.len());
                parts.extend(std::iter::repeat_n(Part::Image, images));
                message(Role::User, None, parts)
            }
            EventMsg::AgentMessage { message: text } if !paginated => {
                message(Role::Assistant, None, vec![Part::Text(text)])
            }
            EventMsg::AgentReasoning { text } | EventMsg::AgentReasoningRawContent { text }
                if !paginated && !text.is_empty() =>
            {
                message(Role::Assistant, None, vec![Part::Thinking(text)])
            }
            EventMsg::TokenCount { info: Some(info) } => {
                let usage = info.total_token_usage.since(&self.usage);
                self.usage = info.total_token_usage;
                EventKind::Note {
                    label: "Token count".to_owned(),
                    text: String::new(),
                    searchable: false,
                    usage: Some(usage),
                }
            }
            _ => EventKind::Other,
        }
    }
}

fn turn_item(item: TurnItem) -> EventKind {
    match item {
        TurnItem::UserMessage { content } => {
            let parts = content
                .into_iter()
                .map(|input| match input {
                    UserInput::Text { text } => Part::Text(text),
                    UserInput::Image {} | UserInput::LocalImage {} => Part::Image,
                    UserInput::Other => Part::Other,
                })
                .collect();
            message(Role::User, None, parts)
        }
        TurnItem::AgentMessage { id, content } => {
            let parts = content
                .into_iter()
                .map(|content| match content {
                    AgentContent::Text { text } => Part::Text(text),
                    AgentContent::Other => Part::Other,
                })
                .collect();
            message(Role::Assistant, id, parts)
        }
        TurnItem::Reasoning {
            id,
            summary_text,
            raw_content,
        } => {
            let text = summary_text
                .into_iter()
                .chain(raw_content)
                .filter(|text| !text.is_empty())
                .collect::<Vec<_>>()
                .join("\n\n");
            if text.is_empty() {
                EventKind::Other
            } else {
                message(Role::Assistant, id, vec![Part::Thinking(text)])
            }
        }
        TurnItem::Other => EventKind::Other,
    }
}

fn response_item(item: ResponseItem) -> EventKind {
    let call = |id: Option<String>, call_id: Option<String>, name: &str, input: Value| {
        message(
            Role::Assistant,
            id.clone(),
            vec![Part::ToolCall {
                id: call_id.or(id).unwrap_or_default(),
                name: name.to_owned(),
                input,
            }],
        )
    };
    let result = |call_id: Option<String>, output: Value| {
        message(
            Role::User,
            None,
            vec![Part::ToolResult {
                call_id: call_id.unwrap_or_default(),
                output: Some(tool_output(output)),
            }],
        )
    };
    match item {
        ResponseItem::FunctionCall {
            id,
            name,
            namespace,
            arguments,
            call_id,
        } => {
            let name = match namespace {
                Some(namespace) if !namespace.is_empty() => format!("{namespace}__{name}"),
                _ => name,
            };
            let input = serde_json::from_str(&arguments).unwrap_or(Value::String(arguments));
            call(id, Some(call_id), &name, input)
        }
        ResponseItem::CustomToolCall {
            id,
            name,
            input,
            call_id,
        } => call(
            id,
            Some(call_id),
            &name,
            serde_json::json!({ "input": input }),
        ),
        ResponseItem::LocalShellCall {
            id,
            call_id,
            action,
        } => call(id, call_id, "local_shell", action),
        ResponseItem::WebSearchCall { id, action } => call(id, None, "web_search", action),
        ResponseItem::ToolSearchCall {
            id,
            call_id,
            arguments,
        } => call(id, call_id, "tool_search", arguments),
        ResponseItem::FunctionCallOutput { call_id, output }
        | ResponseItem::CustomToolCallOutput { call_id, output } => result(call_id, output),
        ResponseItem::ToolSearchOutput { call_id, tools } => result(call_id, tools),
        ResponseItem::Message { .. } | ResponseItem::Other => EventKind::Other,
    }
}

/// A tool output as the shared model renders it: a string, or text blocks.
fn tool_output(output: Value) -> Value {
    match output {
        Value::Array(items) => Value::Array(
            items
                .into_iter()
                .filter_map(|item| {
                    let text = item.get("text").and_then(Value::as_str)?;
                    Some(serde_json::json!({ "type": "text", "text": text }))
                })
                .collect(),
        ),
        // An older payload shape: `{content, success}`.
        Value::Object(mut object) => match object.remove("content") {
            Some(content) => tool_output(content),
            None => Value::Object(object),
        },
        other => other,
    }
}

fn content_part(item: ContentItem) -> Option<Part> {
    match item {
        ContentItem::InputText { text } | ContentItem::OutputText { text } => {
            Some(Part::Text(text))
        }
        ContentItem::InputImage {} => Some(Part::Image),
        ContentItem::Other => None,
    }
}

fn message(role: Role, id: Option<String>, parts: Vec<Part>) -> EventKind {
    EventKind::Message(Message {
        role,
        id,
        parts,
        model: None,
        usage: None,
        parent_tool_use_id: None,
    })
}

fn diagnostic(line_number: usize, line_content: String, error_message: String) -> ParseError {
    ParseError {
        line_number,
        line_content,
        error_message,
        context_before: Vec::new(),
        context_after: Vec::new(),
    }
}
