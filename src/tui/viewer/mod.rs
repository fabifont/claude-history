//! Conversation viewer rendering for TUI display.
//!
//! This module renders conversation JSONL files to `Vec<RenderedLine>` for display
//! in the TUI viewer. It produces styled spans that ratatui can render directly,
//! without using ANSI escape codes.

use crate::history::model::Transcript;
use crate::history::view::{self, Entry, EntryKind};
use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;

use crate::tui::theme::{self, Theme};

mod commands;
mod entry;

pub(crate) use commands::process_command_message;
mod ledger;
mod markdown;
mod output;
mod style;
mod summary;
mod timing;
mod tools;

pub use output::{LineStyle, RenderedLine};

use entry::render_entry;
use summary::{
    PendingToolSummary, entry_shows_nothing, flush_tool_summary, tool_only_assistant_summary,
    user_entry_is_only_tool_results,
};
use tools::make_tool_summary_output_id;

/// Width of the focus gutter indicator (▌ + space)
pub const GUTTER_WIDTH: usize = 2;

pub(crate) const NAME_WIDTH: usize = 9;
/// Width of timestamp prefix when timing is enabled (space + Mon DD HH:MM + space)
pub(crate) const TIMESTAMP_WIDTH: usize = 14;

/// Get the current theme (cached after first detection)
fn th() -> &'static Theme {
    theme::detect_theme()
}

/// Maximum body lines shown in truncated tool call mode
const TRUNCATED_BODY_LINES: usize = 3;
/// Maximum result lines shown in truncated tool result mode
const TRUNCATED_RESULT_LINES: usize = 4;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ToolOutputId(pub String);

/// Controls how tool calls and results are displayed
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ToolDisplayMode {
    /// Tool calls and results are not rendered at all
    Off,
    #[default]
    Summary,
    /// Each tool call on one line, without results
    Calls,
    Truncated,
    Full,
}

impl ToolDisplayMode {
    /// Cycle to the next mode: Off → Summary → Calls → Truncated → Full → Off
    pub fn next(self) -> Self {
        match self {
            Self::Off => Self::Summary,
            Self::Summary => Self::Calls,
            Self::Calls => Self::Truncated,
            Self::Truncated => Self::Full,
            Self::Full => Self::Off,
        }
    }

    /// Whether runs of tool calls collapse into rows that expand on click
    pub fn groups_tools(self) -> bool {
        matches!(self, Self::Summary | Self::Calls)
    }

    /// Whether full or truncated tool details should be rendered
    pub fn shows_details(self) -> bool {
        matches!(self, Self::Truncated | Self::Full)
    }

    /// Whether tools should be included in exported text
    pub fn is_visible(self) -> bool {
        self.shows_details()
    }

    /// Fixed-width label for the status bar (3 chars each)
    pub fn status_label(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Summary => "sum",
            Self::Calls => "cal",
            Self::Truncated => "trn",
            Self::Full => "all",
        }
    }
}

/// Options for rendering a conversation
pub struct RenderOptions {
    pub tool_display: ToolDisplayMode,
    pub show_thinking: bool,
    pub show_timing: bool,
    pub content_width: usize,
    pub expanded_tool_outputs: BTreeSet<ToolOutputId>,
}

/// Tracks the line range of a single message (User or Assistant entry) in the rendered output
#[derive(Clone, Debug)]
pub struct MessageRange {
    /// Index of the JSONL entry (line number in the file, 0-based, counting only parsed entries)
    pub entry_index: usize,
    /// Start line in rendered output (inclusive)
    pub start_line: usize,
    /// End line in rendered output (exclusive, excludes trailing blank)
    pub end_line: usize,
}

/// Result of rendering a conversation
pub struct RenderedConversation {
    pub lines: Vec<RenderedLine>,
    pub messages: Vec<MessageRange>,
}

/// Format a timestamp as Mon DD HH:MM in local time
fn format_timestamp(timestamp: chrono::DateTime<chrono::Utc>) -> String {
    timestamp
        .with_timezone(&chrono::Local)
        .format("%b %d %H:%M")
        .to_string()
}

/// An entry of the conversation shown, with its position among the
/// transcript's entries: the identity tool outputs and message ranges
/// refer to.
#[derive(Debug)]
pub struct RenderableEntry<'a> {
    pub entry_index: usize,
    entry: Entry<'a>,
}

/// The conversation whose session file is `path`.
pub fn open_conversation(path: &Path) -> std::io::Result<Arc<Transcript>> {
    crate::history::service::shared()
        .open_path(path)
        .map(|(_, transcript)| transcript)
        .map_err(|error| std::io::Error::other(error.to_string()))
}

pub fn renderable_entries(transcript: &Transcript) -> Vec<RenderableEntry<'_>> {
    view::entries(transcript)
        .into_iter()
        .enumerate()
        .map(|(entry_index, entry)| RenderableEntry { entry_index, entry })
        .collect()
}

pub fn render_parsed_conversation(
    entries: &[RenderableEntry],
    options: &RenderOptions,
) -> RenderedConversation {
    let mut lines = Vec::new();
    let mut messages = Vec::new();
    let mut pending_tool_summary: Option<PendingToolSummary> = None;

    for (parsed_idx, parsed) in entries.iter().enumerate() {
        if options.tool_display.groups_tools()
            && try_extend_or_start_pending_summary(
                &mut lines,
                &mut messages,
                &mut pending_tool_summary,
                entries,
                parsed_idx,
                options,
            )
        {
            continue;
        }

        flush_tool_summary(
            &mut lines,
            &mut messages,
            &mut pending_tool_summary,
            entries,
            options,
        );

        render_entry_with_range(&mut lines, &mut messages, parsed, options);
    }

    flush_tool_summary(
        &mut lines,
        &mut messages,
        &mut pending_tool_summary,
        entries,
        options,
    );

    postprocess_blank_lines(&mut lines, &mut messages);

    RenderedConversation { lines, messages }
}

/// Handle a parsed entry while in summary tool-display mode.
///
/// Returns `true` when the entry was absorbed into (or started) a pending
/// summary group and should be skipped by the normal render path.
fn try_extend_or_start_pending_summary(
    lines: &mut Vec<RenderedLine>,
    messages: &mut Vec<MessageRange>,
    pending: &mut Option<PendingToolSummary>,
    entries: &[RenderableEntry],
    parsed_idx: usize,
    options: &RenderOptions,
) -> bool {
    let parsed = &entries[parsed_idx];
    let entry_index = parsed.entry_index;
    let entry = &parsed.entry;

    if let Some((parent_id, speaker, timestamp, summary)) =
        tool_only_assistant_summary(entry, options)
    {
        match pending {
            Some(p) if p.parent_id.as_deref() == parent_id && p.speaker == speaker => {
                p.last_parsed_idx = parsed_idx;
                p.summary.merge(summary);
            }
            _ => {
                flush_tool_summary(lines, messages, pending, entries, options);
                *pending = Some(PendingToolSummary {
                    id: make_tool_summary_output_id(entry_index, parent_id),
                    first_entry_index: entry_index,
                    first_parsed_idx: parsed_idx,
                    last_parsed_idx: parsed_idx,
                    parent_id: parent_id.map(str::to_string),
                    speaker: speaker.to_owned(),
                    timestamp,
                    summary,
                });
            }
        }
        return true;
    }

    // Records that render nothing sit between a tool result and the next
    // call; they must not split one run of tool calls into several rows.
    if pending.is_some() && entry_shows_nothing(entry, options) {
        return true;
    }

    if user_entry_is_only_tool_results(entry, options) {
        if let Some(p) = pending {
            p.last_parsed_idx = parsed_idx;
        }
        return true;
    }

    false
}

/// Render one parsed entry and, if it produced a navigable message,
/// append a `MessageRange` that excludes any trailing blank line.
fn render_entry_with_range(
    lines: &mut Vec<RenderedLine>,
    messages: &mut Vec<MessageRange>,
    parsed: &RenderableEntry,
    options: &RenderOptions,
) {
    let entry_index = parsed.entry_index;
    let entry = &parsed.entry;
    let is_message = match entry.kind {
        EntryKind::Message { .. } => true,
        EntryKind::Subagent { .. } => options.show_thinking,
        EntryKind::Note { .. } | EntryKind::Hidden => false,
    };

    let start_line = lines.len();
    render_entry(lines, entry_index, entry, options);
    let end_line = lines.len();

    if !is_message {
        return;
    }
    if let Some(range) =
        message_range_excluding_trailing_blank(lines, start_line, end_line, entry_index)
    {
        messages.push(range);
    }
}

/// If the rendered slice produced any non-blank lines, return a
/// `MessageRange` whose `end_line` excludes a trailing blank.
fn message_range_excluding_trailing_blank(
    lines: &[RenderedLine],
    start_line: usize,
    end_line: usize,
    entry_index: usize,
) -> Option<MessageRange> {
    if end_line <= start_line {
        return None;
    }
    let effective_end = if lines.get(end_line - 1).is_some_and(|l| l.spans.is_empty()) {
        end_line - 1
    } else {
        end_line
    };
    if effective_end <= start_line {
        return None;
    }
    Some(MessageRange {
        entry_index,
        start_line,
        end_line: effective_end,
    })
}

/// Collapse consecutive blank rendered lines and remap message ranges so
/// they continue to point at their original visible content.
///
/// Multiple render helpers each push a trailing blank line, which can
/// produce adjacent blanks when a tool result emits empty output. The
/// dedup pass removes any blank line whose immediate predecessor is also
/// blank, and the remap pass shifts every range start/end onto the new
/// line indices, clamping ranges that ended on a removed blank.
fn postprocess_blank_lines(lines: &mut Vec<RenderedLine>, messages: &mut Vec<MessageRange>) {
    let mut removed = vec![false; lines.len()];
    let mut i = 1;
    while i < lines.len() {
        if lines[i].spans.is_empty() && lines[i - 1].spans.is_empty() {
            removed[i] = true;
        }
        i += 1;
    }

    // Build index mapping: old line index -> new line index. Removed
    // entries get the index they would collapse onto; they are never
    // dereferenced for surviving ranges because the remap below walks
    // backward off any removed terminator first.
    let mut new_index = Vec::with_capacity(lines.len());
    let mut offset = 0usize;
    for (idx, &is_removed) in removed.iter().enumerate() {
        if is_removed {
            new_index.push(idx - offset);
            offset += 1;
        } else {
            new_index.push(idx - offset);
        }
    }
    let total_after = lines.len() - offset;

    // Compact in place.
    let mut write = 0;
    for (read, &is_removed) in removed.iter().enumerate() {
        if !is_removed {
            if write != read {
                lines.swap(write, read);
            }
            write += 1;
        }
    }
    lines.truncate(total_after);

    for msg in messages.iter_mut() {
        msg.start_line = new_index[msg.start_line];
        if msg.end_line > 0 && msg.end_line <= new_index.len() {
            // end_line is exclusive — find the new index of the last
            // non-removed line before it and add 1.
            let mut last = msg.end_line - 1;
            while last > msg.start_line && removed[last] {
                last -= 1;
            }
            msg.end_line = new_index[last] + 1;
        } else if msg.end_line == new_index.len() {
            msg.end_line = total_after;
        }
        msg.end_line = msg.end_line.min(total_after);
        msg.start_line = msg.start_line.min(msg.end_line);
    }

    messages.retain(|m| m.start_line < m.end_line);
}

/// Render a conversation to lines for display in the TUI viewer
pub fn render_conversation(
    transcript: &Transcript,
    options: &RenderOptions,
) -> RenderedConversation {
    render_parsed_conversation(&renderable_entries(transcript), options)
}

#[cfg(test)]
mod tests;
