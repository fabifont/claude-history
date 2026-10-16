//! Folds a `Transcript` into the fields the conversation list and search
//! index store: previews, full text, semantic turns and their message
//! ranges, route text, token totals and duration.

use super::Conversation;
use super::messages::{MessageOrdinals, Placement, extract_skill_preview};
use super::model::{EventKind, Message, Part, Role, Transcript, Usage, Visibility};
use super::parser::{is_clear_only_conversation, normalize_whitespace};
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
    let mut all_parts = Vec::new();
    let mut agent_search_parts = Vec::new();
    let mut semantic_turns = Vec::new();
    let mut semantic_turn_ranges = Vec::new();
    let mut preview_parts = Vec::new();
    let mut user_messages = Vec::new();
    let mut ordinals = MessageOrdinals::new();
    // Streamed assistant messages repeat their usage; the last one counts.
    let mut usage_by_message: HashMap<&str, Usage> = HashMap::new();
    let mut assistant_id_semantic_indices: HashMap<&str, usize> = HashMap::new();
    let mut assistant_id_preview_indices: HashMap<&str, usize> = HashMap::new();
    let mut anonymous_tokens: u64 = 0;
    let mut first_timestamp: Option<DateTime<Utc>> = None;
    let mut last_timestamp: Option<DateTime<Utc>> = None;

    for event in &transcript.events {
        if event.visibility != Visibility::Dialogue {
            continue;
        }
        let placement = ordinals.place(event);
        match &event.kind {
            EventKind::Message(message) => {
                if let Some(timestamp) = event.timestamp {
                    first_timestamp.get_or_insert(timestamp);
                    last_timestamp = Some(timestamp);
                }
                match (message.role, &message.usage, &message.id) {
                    (Role::Assistant, Some(usage), Some(id)) => {
                        usage_by_message.insert(id, *usage);
                    }
                    (_, Some(usage), _) => anonymous_tokens += usage.total(),
                    (_, None, _) => {}
                }
                let preview_text = text_of(&message.parts);
                let search_text = search_text_of(&message.parts);
                if !search_text.is_empty() {
                    all_parts.push(search_text);
                }
                match message.role {
                    Role::User => {
                        if !preview_text.is_empty() {
                            user_messages.push(preview_text.clone());
                        }
                        let Placement::Message(ordinal) = placement else {
                            continue;
                        };
                        // Skill invocations preview as the command line
                        // (e.g. "/consult how to do X?") instead of the tags.
                        let effective_preview = extract_skill_preview(&preview_text)
                            .unwrap_or_else(|| preview_text.clone());
                        if !effective_preview.is_empty() {
                            if let Some(turn) = filter_turn(SemanticTurnRole::User, &preview_text) {
                                semantic_turns.push(turn);
                                semantic_turn_ranges.push(MessageRange::single(ordinal));
                            }
                            preview_parts.push(effective_preview);
                        }
                    }
                    Role::Assistant => {
                        let (Placement::Message(ordinal) | Placement::Replaces(ordinal)) =
                            placement
                        else {
                            continue;
                        };
                        let message_range = MessageRange::single(ordinal);
                        let semantic_turn = filter_turn(SemanticTurnRole::Assistant, &preview_text);
                        if let Some(id) = message.id.as_deref() {
                            if let Some(existing) = assistant_id_semantic_indices.get(id).copied() {
                                if let Some(turn) = semantic_turn {
                                    semantic_turns[existing] = turn;
                                    semantic_turn_ranges[existing] = message_range;
                                } else {
                                    semantic_turns[existing].clear();
                                }
                            } else if let Some(turn) = semantic_turn {
                                assistant_id_semantic_indices.insert(id, semantic_turns.len());
                                semantic_turns.push(turn);
                                semantic_turn_ranges.push(message_range);
                            }

                            if !preview_text.is_empty() {
                                if let Some(existing) =
                                    assistant_id_preview_indices.get(id).copied()
                                {
                                    preview_parts[existing] = preview_text;
                                } else {
                                    assistant_id_preview_indices.insert(id, preview_parts.len());
                                    preview_parts.push(preview_text);
                                }
                            }
                        } else if !preview_text.is_empty() {
                            if let Some(turn) = semantic_turn {
                                semantic_turns.push(turn);
                                semantic_turn_ranges.push(message_range);
                            }
                            preview_parts.push(preview_text);
                        }
                    }
                }
            }
            EventKind::Note { text, usage, .. } => {
                if let Some(usage) = usage {
                    anonymous_tokens += usage.total();
                }
                if let Placement::Message(ordinal) = placement
                    && !text.is_empty()
                {
                    all_parts.push(text.clone());
                    if let Some(turn) = filter_turn(SemanticTurnRole::User, text) {
                        semantic_turns.push(turn);
                        semantic_turn_ranges.push(MessageRange::single(ordinal));
                    }
                }
            }
            EventKind::Subagent { message, .. } => {
                if let Placement::Message(_) = placement {
                    let text = agent_search_text_from_parts(agent_role(message), &message.parts);
                    if !text.is_empty() {
                        agent_search_parts.push(text);
                    }
                }
            }
            EventKind::Title { .. } | EventKind::Other => {}
        }
    }

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

    let (semantic_turns, semantic_turn_ranges): (Vec<_>, Vec<_>) = semantic_turns
        .into_iter()
        .zip(semantic_turn_ranges)
        .filter(|(turn, _)| !turn.is_empty())
        .unzip();

    let total_tokens = usage_by_message.values().map(Usage::total).sum::<u64>() + anonymous_tokens;

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
    })
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
