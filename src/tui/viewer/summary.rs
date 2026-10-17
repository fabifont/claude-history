use crate::history::model::{Part, Role};
use crate::history::view::{Entry, EntryKind};
use chrono::{DateTime, Utc};
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use super::ledger::{LedgerRow, NameCol, push_row};
use super::style::assistant_label;
use super::timing::TimingSlot;
use super::tools::{
    ToolCallRenderSpec, ToolOutputKind, ToolResultRenderSpec, make_tool_output_id,
    render_tool_call, render_tool_result, tool_result_display_text,
};
use super::*;

pub(super) struct PendingToolSummary {
    pub(super) id: ToolOutputId,
    pub(super) first_entry_index: usize,
    pub(super) first_parsed_idx: usize,
    pub(super) last_parsed_idx: usize,
    pub(super) parent_id: Option<String>,
    pub(super) speaker: String,
    pub(super) timestamp: Option<DateTime<Utc>>,
    pub(super) summary: ToolActivitySummary,
}

#[derive(Default)]
pub(super) struct ToolActivitySummary {
    searched_patterns: usize,
    searched_file_patterns: usize,
    read_files: usize,
    shell_commands: usize,
    edited_files: usize,
    wrote_files: usize,
    agents: usize,
    fetched_urls: usize,
    web_searches: usize,
    other_tools: usize,
}

impl ToolActivitySummary {
    fn add_call(&mut self, name: &str) {
        match name {
            "Bash" => self.shell_commands += 1,
            "Read" => self.read_files += 1,
            "Grep" => self.searched_patterns += 1,
            "Glob" => self.searched_file_patterns += 1,
            "Edit" => self.edited_files += 1,
            "Write" => self.wrote_files += 1,
            "Task" => self.agents += 1,
            "WebFetch" => self.fetched_urls += 1,
            "WebSearch" => self.web_searches += 1,
            _ => self.other_tools += 1,
        }
    }

    pub(super) fn merge(&mut self, other: Self) {
        self.searched_patterns += other.searched_patterns;
        self.searched_file_patterns += other.searched_file_patterns;
        self.read_files += other.read_files;
        self.shell_commands += other.shell_commands;
        self.edited_files += other.edited_files;
        self.wrote_files += other.wrote_files;
        self.agents += other.agents;
        self.fetched_urls += other.fetched_urls;
        self.web_searches += other.web_searches;
        self.other_tools += other.other_tools;
    }

    pub(super) fn is_empty(&self) -> bool {
        self.searched_patterns
            + self.searched_file_patterns
            + self.read_files
            + self.shell_commands
            + self.edited_files
            + self.wrote_files
            + self.agents
            + self.fetched_urls
            + self.web_searches
            + self.other_tools
            == 0
    }

    /// Whether the group is one call to a tool without its own summary
    /// wording, which reads better as the call itself than as "Called 1 tool".
    fn is_single_other_tool(&self) -> bool {
        self.other_tools == 1
            && Self {
                other_tools: 0,
                ..*self
            }
            .is_empty()
    }

    fn sentence(&self) -> String {
        let mut parts = Vec::new();
        push_summary_item(
            &mut parts,
            self.searched_patterns,
            "Searched for",
            "pattern",
        );
        push_summary_item(
            &mut parts,
            self.searched_file_patterns,
            "Searched for",
            "file pattern",
        );
        push_summary_item(&mut parts, self.read_files, "read", "file");
        push_summary_item(&mut parts, self.shell_commands, "ran", "shell command");
        push_summary_item(&mut parts, self.edited_files, "edited", "file");
        push_summary_item(&mut parts, self.wrote_files, "wrote", "file");
        push_summary_item(&mut parts, self.agents, "started", "agent");
        push_summary_item(&mut parts, self.fetched_urls, "fetched", "URL");
        push_summary_item(&mut parts, self.web_searches, "searched", "web");
        push_summary_item(&mut parts, self.other_tools, "called", "tool");
        capitalize_first(parts.join(", "))
    }
}

fn capitalize_first(text: String) -> String {
    let mut chars = text.chars();
    let Some(first) = chars.next() else {
        return text;
    };
    first.to_uppercase().chain(chars).collect()
}

fn push_summary_item(parts: &mut Vec<String>, count: usize, verb: &str, noun: &str) {
    if count == 0 {
        return;
    }
    let suffix = if count == 1 { "" } else { "s" };
    parts.push(format!("{verb} {count} {noun}{suffix}"));
}

pub(super) fn render_tool_activity_summary(
    lines: &mut Vec<RenderedLine>,
    label: &str,
    label_color: (u8, u8, u8),
    dimmed: bool,
    timing: TimingSlot<'_>,
    text: String,
    tool_output_id: Option<&ToolOutputId>,
) {
    let content = vec![(
        text,
        LineStyle {
            fg: Some(th().tool_text),
            dimmed: true,
            ..Default::default()
        },
    )];
    push_row(
        lines,
        LedgerRow {
            timing,
            name: NameCol::Label {
                text: label,
                color: label_color,
                bold: false,
                dimmed,
            },
            separator_dimmed: dimmed,
            tool_output_id,
            clickable: tool_output_id.is_some(),
        },
        content,
    );
}

pub(super) fn summarize_tool_calls(parts: &[Part]) -> ToolActivitySummary {
    let mut summary = ToolActivitySummary::default();
    for part in parts {
        if let Part::ToolCall { name, .. } = part {
            summary.add_call(name);
        }
    }
    summary
}

fn assistant_parts_are_tool_only(parts: &[Part], show_thinking: bool) -> bool {
    parts.iter().all(|part| {
        matches!(part, Part::ToolCall { .. })
            || (!show_thinking && matches!(part, Part::Thinking(_)))
    })
}

pub(super) fn tool_only_assistant_summary<'a>(
    entry: &Entry<'a>,
    options: &RenderOptions,
) -> Option<(
    Option<&'a str>,
    &'a str,
    Option<DateTime<Utc>>,
    ToolActivitySummary,
)> {
    let EntryKind::Message { message, speaker } = entry.kind else {
        return None;
    };
    if message.role != Role::Assistant {
        return None;
    }

    if message.parent_tool_use_id.is_some() && !options.show_thinking {
        return None;
    }
    if message.parts.is_empty()
        || !assistant_parts_are_tool_only(&message.parts, options.show_thinking)
    {
        return None;
    }

    let summary = summarize_tool_calls(&message.parts);
    (!summary.is_empty()).then_some((
        message.parent_tool_use_id.as_deref(),
        speaker,
        entry.timestamp,
        summary,
    ))
}

pub(super) fn user_entry_is_only_tool_results(entry: &Entry<'_>, options: &RenderOptions) -> bool {
    let EntryKind::Message { message, .. } = entry.kind else {
        return false;
    };
    if message.role != Role::User {
        return false;
    }

    if message.parent_tool_use_id.is_some() && !options.show_thinking {
        return false;
    }

    !message.parts.is_empty()
        && message
            .parts
            .iter()
            .all(|part| matches!(part, Part::ToolResult { .. }))
}

fn render_summary_group_details(
    lines: &mut Vec<RenderedLine>,
    entries: &[RenderableEntry],
    pending: &PendingToolSummary,
    options: &RenderOptions,
) {
    let first_line = lines.len();
    let mut rendered_any = false;
    let pad_timing = TimingSlot::from_show_timing(options.show_timing);
    let label = assistant_label(pending.parent_id.as_deref(), &pending.speaker);
    for parsed in &entries[pending.first_parsed_idx..=pending.last_parsed_idx] {
        let EntryKind::Message { message, .. } = parsed.entry.kind else {
            continue;
        };
        let parent_tool_use_id = &message.parent_tool_use_id;
        if parent_tool_use_id.as_deref() != pending.parent_id.as_deref() {
            continue;
        }
        match message.role {
            Role::Assistant => {
                for (block_idx, block) in message.parts.iter().enumerate() {
                    if let Part::ToolCall { id, name, input } = block {
                        if rendered_any {
                            lines.push(RenderedLine::new(vec![]));
                        }
                        let output_id = make_tool_output_id(
                            parsed.entry_index,
                            parent_tool_use_id.as_deref(),
                            block_idx,
                            ToolOutputKind::ToolCall,
                            Some(id),
                        );
                        let expanded = options.expanded_tool_outputs.contains(&output_id);
                        render_tool_call(
                            lines,
                            &ToolCallRenderSpec {
                                name,
                                input,
                                label: &label,
                                label_color: th().accent_dim,
                                dimmed: true,
                                content_width: options.content_width,
                                timing: pad_timing,
                                tool_display: ToolDisplayMode::Truncated,
                                tool_output_id: &output_id,
                                expanded,
                            },
                        );
                        rendered_any = true;
                    }
                }
            }
            Role::User => {
                for (block_idx, block) in message.parts.iter().enumerate() {
                    if let Part::ToolResult {
                        output: content,
                        call_id: tool_use_id,
                    } = block
                    {
                        if rendered_any {
                            lines.push(RenderedLine::new(vec![]));
                        }
                        let output_id = make_tool_output_id(
                            parsed.entry_index,
                            parent_tool_use_id.as_deref(),
                            block_idx,
                            ToolOutputKind::ToolResult,
                            Some(tool_use_id),
                        );
                        let expanded = options.expanded_tool_outputs.contains(&output_id);
                        let content_str = tool_result_display_text(content.as_ref());
                        render_tool_result(
                            lines,
                            &ToolResultRenderSpec {
                                text: &content_str,
                                content_width: options.content_width,
                                timing: pad_timing,
                                tool_display: ToolDisplayMode::Truncated,
                                tool_output_id: &output_id,
                                expanded,
                            },
                        );
                        rendered_any = true;
                    }
                }
            }
        }
    }
    if let Some(line) = lines.get_mut(first_line) {
        line.tool_output_id = Some(pending.id.clone());
        line.clickable = true;
    }
}

/// The summary text for a pending group; see [`summary_line`].
fn group_summary_line(
    entries: &[RenderableEntry],
    pending: &PendingToolSummary,
    width: usize,
) -> String {
    let parts = entries[pending.first_parsed_idx..=pending.last_parsed_idx]
        .iter()
        .filter_map(|parsed| match parsed.entry.kind {
            EntryKind::Message { message, .. }
                if message.role == Role::Assistant
                    && message.parent_tool_use_id.as_deref() == pending.parent_id.as_deref() =>
            {
                Some(message)
            }
            _ => None,
        })
        .flat_map(|message| &message.parts);
    summary_line(&pending.summary, parts, width)
}

/// The summary text for the calls among `parts`. A single call to a tool
/// without its own summary wording shows as the call itself on one line cut
/// to `width`, which says more than "Called 1 tool".
pub(super) fn summary_line<'a>(
    summary: &ToolActivitySummary,
    mut parts: impl Iterator<Item = &'a Part>,
    width: usize,
) -> String {
    let single_call = parts.find_map(|part| match part {
        Part::ToolCall { name, input, .. } => Some((name, input)),
        _ => None,
    });
    match single_call {
        Some((name, input)) if summary.is_single_other_tool() => {
            let header = crate::tool_format::format_tool_call(name, input, 0).header;
            truncate_end(&header, width)
        }
        _ => summary.sentence(),
    }
}

/// `text` cut to `max_width` columns, ending in `…` when cut.
fn truncate_end(text: &str, max_width: usize) -> String {
    if UnicodeWidthStr::width(text) <= max_width {
        return text.to_string();
    }
    let mut out = String::new();
    let mut width = 0;
    for ch in text.chars() {
        let ch_width = UnicodeWidthChar::width(ch).unwrap_or(0);
        if width + ch_width + 1 > max_width {
            break;
        }
        out.push(ch);
        width += ch_width;
    }
    out.push('…');
    out
}

pub(super) fn flush_tool_summary(
    lines: &mut Vec<RenderedLine>,
    messages: &mut Vec<MessageRange>,
    pending: &mut Option<PendingToolSummary>,
    entries: &[RenderableEntry],
    options: &RenderOptions,
) {
    let Some(pending) = pending.take() else {
        return;
    };

    let start_line = lines.len();
    let label = assistant_label(pending.parent_id.as_deref(), &pending.speaker);
    let ts = if options.show_timing {
        pending.timestamp.map(format_timestamp)
    } else {
        None
    };
    let timing = match ts.as_deref() {
        Some(ts) => TimingSlot::Stamp(ts),
        None => TimingSlot::Disabled,
    };
    if options.expanded_tool_outputs.contains(&pending.id) {
        render_summary_group_details(lines, entries, &pending, options);
    } else {
        render_tool_activity_summary(
            lines,
            &label,
            th().accent_dim,
            pending.parent_id.is_some(),
            timing,
            group_summary_line(entries, &pending, options.content_width),
            Some(&pending.id),
        );
    }

    let end_line = lines.len();
    if end_line > start_line {
        messages.push(MessageRange {
            entry_index: pending.first_entry_index,
            start_line,
            end_line,
        });
        lines.push(RenderedLine::new(vec![]));
    }
}
