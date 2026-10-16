//! What a reader is shown of a transcript, in order: the entries the
//! viewer, exports and `--render` present.
//!
//! A conversation forked from another starts with the history it was
//! forked from, followed by a note marking where its own records begin.
//! Context the source injected for the model is not shown.

use super::model::{Event, EventKind, Message, SourceKind, Transcript, Visibility};
use chrono::{DateTime, Utc};

#[derive(Clone, Copy, Debug)]
pub struct Entry<'a> {
    pub timestamp: Option<DateTime<Utc>>,
    pub kind: EntryKind<'a>,
}

#[derive(Clone, Copy, Debug)]
pub enum EntryKind<'a> {
    /// A message of the conversation, or of a subagent it started when
    /// `Message::parent_tool_use_id` is set. `speaker` names the
    /// assistant.
    Message {
        message: &'a Message,
        speaker: &'a str,
    },
    /// A message of a subagent, reported while the subagent runs.
    Subagent {
        agent_id: &'a str,
        message: &'a Message,
    },
    /// A labeled record of the source; only searchable notes are shown.
    Note {
        label: &'a str,
        text: &'a str,
        searchable: bool,
    },
    /// A record that shows nothing. It still ends a run of tool calls
    /// the viewer summarizes as one.
    Hidden,
}

const FORK_LABEL: &str = "Fork";
const FORK_TEXT: &str = "The conversation continues from here";

/// The entries of `transcript`.
pub fn entries(transcript: &Transcript) -> Vec<Entry<'_>> {
    let speaker = transcript.source.assistant_name();
    // Codex records bookkeeping between a turn's tool calls (command
    // output, turn context, titles), none of it shown; it reports what a
    // thread it spawned said once that thread finishes.
    let codex = transcript.source == SourceKind::Codex;
    let inherited = |event: &Event| {
        transcript
            .segments
            .get(usize::from(event.origin.segment))
            .is_some_and(|segment| segment.inherited)
    };
    let mut entries = Vec::with_capacity(transcript.events.len());
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
            entries.push(Entry {
                timestamp: None,
                kind: EntryKind::Note {
                    label: FORK_LABEL,
                    text: FORK_TEXT,
                    searchable: true,
                },
            });
        }
        in_base = from_base;
        let kind = match &event.kind {
            EventKind::Message(message) => EntryKind::Message { message, speaker },
            EventKind::Subagent { agent_id, message } if codex => EntryKind::Message {
                message,
                speaker: agent_id,
            },
            EventKind::Subagent { agent_id, message } => EntryKind::Subagent { agent_id, message },
            EventKind::Note {
                label,
                text,
                searchable,
                ..
            } => EntryKind::Note {
                label,
                text,
                searchable: *searchable,
            },
            EventKind::Title { .. } | EventKind::Other if codex => continue,
            EventKind::Title { .. } | EventKind::Other => EntryKind::Hidden,
        };
        entries.push(Entry {
            timestamp: event.timestamp,
            kind,
        });
    }
    entries
}
