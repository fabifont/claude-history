//! Folds a `Transcript` into the fields the conversation list and search
//! index store: previews, full text, own text, semantic turns and their
//! message ranges, route text, token totals and duration.

use super::messages::{MessageOrdinals, Placement, authored_text, extract_skill_preview};
use super::model::{
    Event, EventKind, ExecutionMode, HeadStatus, Message, Part, Relationship, Role, Transcript,
    Usage, Visibility,
};
use super::parser::{is_clear_only_conversation, normalize_whitespace};
use super::{Conversation, Provenance};
use crate::agent::refs::MessageRange;
use crate::agent::transcript::{
    AgentMessageRole, agent_search_text_from_parts, bounded_tool_result_text,
};
use crate::semantic::filter::{SemanticTurnRole, filter_turn};
use chrono::{DateTime, Local, Utc};
use std::collections::HashMap;
use std::path::PathBuf;

/// Why a transcript yields no conversation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Skipped {
    /// Only the records a `/clear` leaves behind.
    ClearOnly,
    /// No searchable text, or no message to preview.
    Empty {
        search_parts: usize,
        preview_parts: usize,
    },
}

/// The conversation `transcript` lists as.
pub fn index(transcript: &Transcript) -> Result<Conversation, Skipped> {
    let mut indexer = Indexer::default();
    for event in &transcript.events {
        indexer.push(event);
    }
    indexer.finish(transcript)
}

/// Folds a transcript's events, in order, into a conversation.
#[derive(Default)]
pub struct Indexer {
    all_parts: Vec<String>,
    agent_search_parts: Vec<String>,
    /// What the person typed, part by part; see `Conversation::own_text`.
    own_parts: Vec<String>,
    semantic_turns: Vec<String>,
    semantic_turn_ranges: Vec<MessageRange>,
    preview_parts: Vec<String>,
    user_messages: Vec<String>,
    ordinals: MessageOrdinals,
    // Streamed assistant messages repeat their usage; the last one counts.
    usage_by_message: HashMap<String, Usage>,
    assistant_id_semantic_indices: HashMap<String, usize>,
    assistant_id_preview_indices: HashMap<String, usize>,
    anonymous_tokens: u64,
    first_timestamp: Option<DateTime<Utc>>,
    last_timestamp: Option<DateTime<Utc>>,
}

impl Indexer {
    pub fn push(&mut self, event: &Event) {
        if event.visibility != Visibility::Dialogue {
            return;
        }
        let placement = self.ordinals.place(event);
        match &event.kind {
            EventKind::Message(message) => {
                if let Some(timestamp) = event.timestamp {
                    self.first_timestamp.get_or_insert(timestamp);
                    self.last_timestamp = Some(timestamp);
                }
                match (message.role, &message.usage, &message.id) {
                    (Role::Assistant, Some(usage), Some(id)) => {
                        self.usage_by_message.insert(id.clone(), *usage);
                    }
                    (_, Some(usage), _) => self.anonymous_tokens += usage.total(),
                    (_, None, _) => {}
                }
                let preview_text = text_of(&message.parts);
                let search_text = search_text_of(&message.parts);
                if !search_text.is_empty() {
                    self.all_parts.push(search_text);
                }
                match message.role {
                    Role::User => {
                        if !preview_text.is_empty() {
                            self.user_messages.push(preview_text.clone());
                        }
                        let Placement::Message(ordinal) = placement else {
                            return;
                        };
                        if message.parent_tool_use_id.is_none() {
                            self.own_parts.extend(message.parts.iter().filter_map(
                                |part| match part {
                                    Part::Text(text) => authored_text(text),
                                    _ => None,
                                },
                            ));
                        }
                        // Skill invocations preview as the command line
                        // (e.g. "/consult how to do X?") instead of the tags.
                        let effective_preview = extract_skill_preview(&preview_text)
                            .unwrap_or_else(|| preview_text.clone());
                        if !effective_preview.is_empty() {
                            if let Some(turn) = filter_turn(SemanticTurnRole::User, &preview_text) {
                                self.semantic_turns.push(turn);
                                self.semantic_turn_ranges
                                    .push(MessageRange::single(ordinal));
                            }
                            self.preview_parts.push(effective_preview);
                        }
                    }
                    Role::Assistant => {
                        let (Placement::Message(ordinal) | Placement::Replaces(ordinal)) =
                            placement
                        else {
                            return;
                        };
                        let message_range = MessageRange::single(ordinal);
                        let semantic_turn = filter_turn(SemanticTurnRole::Assistant, &preview_text);
                        if let Some(id) = message.id.as_deref() {
                            if let Some(existing) =
                                self.assistant_id_semantic_indices.get(id).copied()
                            {
                                if let Some(turn) = semantic_turn {
                                    self.semantic_turns[existing] = turn;
                                    self.semantic_turn_ranges[existing] = message_range;
                                } else {
                                    self.semantic_turns[existing].clear();
                                }
                            } else if let Some(turn) = semantic_turn {
                                self.assistant_id_semantic_indices
                                    .insert(id.to_owned(), self.semantic_turns.len());
                                self.semantic_turns.push(turn);
                                self.semantic_turn_ranges.push(message_range);
                            }

                            if !preview_text.is_empty() {
                                if let Some(existing) =
                                    self.assistant_id_preview_indices.get(id).copied()
                                {
                                    self.preview_parts[existing] = preview_text;
                                } else {
                                    self.assistant_id_preview_indices
                                        .insert(id.to_owned(), self.preview_parts.len());
                                    self.preview_parts.push(preview_text);
                                }
                            }
                        } else if !preview_text.is_empty() {
                            if let Some(turn) = semantic_turn {
                                self.semantic_turns.push(turn);
                                self.semantic_turn_ranges.push(message_range);
                            }
                            self.preview_parts.push(preview_text);
                        }
                    }
                }
            }
            EventKind::Note { text, usage, .. } => {
                if let Some(usage) = usage {
                    self.anonymous_tokens += usage.total();
                }
                if let Placement::Message(ordinal) = placement
                    && !text.is_empty()
                {
                    self.all_parts.push(text.clone());
                    if let Some(turn) = filter_turn(SemanticTurnRole::User, text) {
                        self.semantic_turns.push(turn);
                        self.semantic_turn_ranges
                            .push(MessageRange::single(ordinal));
                    }
                }
            }
            EventKind::Subagent { message, .. } => {
                if let Placement::Message(_) = placement {
                    let text = agent_search_text_from_parts(agent_role(message), &message.parts);
                    if !text.is_empty() {
                        self.agent_search_parts.push(text);
                    }
                }
            }
            EventKind::Title { .. } | EventKind::Other => {}
        }
    }

    /// The conversation of `transcript`, whose events were pushed.
    pub fn finish(self, transcript: &Transcript) -> Result<Conversation, Skipped> {
        let Self {
            all_parts,
            agent_search_parts,
            own_parts,
            semantic_turns,
            semantic_turn_ranges,
            preview_parts,
            user_messages,
            ordinals,
            usage_by_message,
            anonymous_tokens,
            first_timestamp,
            last_timestamp,
            ..
        } = self;
        if is_clear_only_conversation(&user_messages) {
            return Err(Skipped::ClearOnly);
        }
        if all_parts.is_empty() || preview_parts.is_empty() {
            return Err(Skipped::Empty {
                search_parts: all_parts.len(),
                preview_parts: preview_parts.len(),
            });
        }

        let meta = &transcript.meta;
        let timestamp = meta
            .updated_at
            .map(DateTime::<Local>::from)
            .unwrap_or_else(Local::now);

        // Previews skip leading assistant messages: preview_parts starts at the
        // first user message.
        let preview_first = preview_parts
            .iter()
            .take(3)
            .cloned()
            .collect::<Vec<_>>()
            .join(" ... ");
        let preview_last = preview_parts
            .iter()
            .rev()
            .take(3)
            .cloned()
            .collect::<Vec<_>>()
            .join(" ... ");

        let mut full_text = all_parts.join(" ");
        if let Some(summary) = &meta.summary {
            full_text = format!("{summary} {full_text}");
        }
        if let Some(custom_title) = &meta.custom_title {
            full_text = format!("{custom_title} {full_text}");
        }

        let preview_first: std::sync::Arc<str> = normalize_whitespace(&preview_first).into();
        let preview_last = normalize_whitespace(&preview_last);
        let full_text = normalize_whitespace(&full_text);
        let agent_search_text = normalize_whitespace(&agent_search_parts.join(" "));
        let semantic_route_text = super::semantic_route_text(&full_text, &agent_search_text);

        let provenance = Provenance {
            automated: transcript.mode == ExecutionMode::Automated,
            child: matches!(transcript.relationship, Relationship::Child { .. }),
            orphaned: transcript.head == HeadStatus::Orphaned,
        };
        // A child conversation's prompts come from the conversation that
        // started it.
        let own_text = if provenance.child {
            String::new()
        } else {
            normalize_whitespace(&own_parts.join(" "))
        };
        // Conversations hidden by default are not embedded.
        let (semantic_turns, semantic_turn_ranges): (Vec<_>, Vec<_>) = semantic_turns
            .into_iter()
            .zip(semantic_turn_ranges)
            .filter(|(turn, _)| !turn.is_empty() && !provenance.hidden_by_default())
            .unzip();

        let total_tokens =
            usage_by_message.values().map(Usage::total).sum::<u64>() + anonymous_tokens;

        let duration_minutes = match (first_timestamp, last_timestamp) {
            (Some(first), Some(last)) => {
                let minutes = last.signed_duration_since(first).num_minutes();
                (minutes > 0).then_some(minutes as u64)
            }
            _ => None,
        };

        let project_path = meta
            .project
            .clone()
            .unwrap_or_else(|| PathBuf::from("unknown"));
        Ok(Conversation {
            source: transcript.source,
            session_id: meta.session_id.clone(),
            path: transcript.path().to_path_buf(),
            index: 0,
            timestamp,
            preview: preview_first.clone(),
            preview_first,
            preview_last: preview_last.into(),
            full_text: full_text.into(),
            own_text: own_text.into(),
            agent_search_text: agent_search_text.into(),
            semantic_route_text: semantic_route_text.into(),
            semantic_turns: semantic_turns.into(),
            semantic_turn_ranges,
            project_name: Some(super::format_short_name_from_path(&project_path)),
            project_path: Some(project_path),
            cwd: meta.cwd.clone(),
            message_count: ordinals.count(),
            parse_errors: transcript.diagnostics.clone(),
            summary: meta.summary.clone(),
            custom_title: meta.custom_title.clone(),
            model: meta.model.clone(),
            total_tokens,
            duration_minutes,
            provenance,
        })
    }
}

fn agent_role(message: &Message) -> AgentMessageRole {
    match message.role {
        Role::User => AgentMessageRole::User,
        Role::Assistant => AgentMessageRole::Assistant,
    }
}

/// Text parts, as previews and semantic turns show them.
fn text_of(parts: &[Part]) -> String {
    parts
        .iter()
        .filter_map(|part| match part {
            Part::Text(text) => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Text parts plus tool result text, as full-text search sees them.
fn search_text_of(parts: &[Part]) -> String {
    parts
        .iter()
        .filter_map(|part| match part {
            Part::Text(text) => Some(text.clone()),
            Part::ToolResult {
                output: Some(output),
                ..
            } => bounded_tool_result_text(output),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join(" ")
}
