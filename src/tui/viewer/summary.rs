use crate::history::model::{Message, Part, Role};
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
    /// Tools without their own summary wording, by name in first-call order,
    /// with their call counts.
    other_tools: Vec<(String, usize)>,
}

impl ToolActivitySummary {
    fn add_call(&mut self, name: &str) {
        match name {
            "Bash" | "bash" => self.shell_commands += 1,
            "Read" | "read" => self.read_files += 1,
            "Grep" | "grep" => self.searched_patterns += 1,
            "Glob" | "find" => self.searched_file_patterns += 1,
            "Edit" | "edit" => self.edited_files += 1,
            "Write" | "write" => self.wrote_files += 1,
            "Task" => self.agents += 1,
            "WebFetch" | "fetch_content" => self.fetched_urls += 1,
            "WebSearch" | "web_search" => self.web_searches += 1,
            _ => self.add_other_tool(name, 1),
        }
    }

    fn add_other_tool(&mut self, name: &str, count: usize) {
        match self.other_tools.iter_mut().find(|(seen, _)| seen == name) {
            Some((_, seen_count)) => *seen_count += count,
            None => self.other_tools.push((name.to_string(), count)),
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
        for (name, count) in &other.other_tools {
            self.add_other_tool(name, *count);
        }
    }

    pub(super) fn is_empty(&self) -> bool {
        self.other_tools.is_empty() && self.known_tool_calls() == 0
    }

    fn known_tool_calls(&self) -> usize {
        self.searched_patterns
            + self.searched_file_patterns
            + self.read_files
            + self.shell_commands
            + self.edited_files
            + self.wrote_files
            + self.agents
            + self.fetched_urls
            + self.web_searches
    }

    /// Whether the group is one call to a tool without its own summary
    /// wording, which reads better as the call itself than as "Called 1 tool".
    fn is_single_other_tool(&self) -> bool {
        matches!(self.other_tools.as_slice(), [(_, 1)]) && self.known_tool_calls() == 0
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
        if !self.other_tools.is_empty() {
            let names: Vec<String> = self
                .other_tools
                .iter()
                .map(|(name, count)| match count {
                    1 => name.clone(),
                    _ => format!("{name} ×{count}"),
                })
                .collect();
            parts.push(format!("called {}", names.join(", ")));
        }
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
    rows: Vec<String>,
    tool_output_id: Option<&ToolOutputId>,
) {
    for (i, text) in rows.into_iter().enumerate() {
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
                timing: if i == 0 {
                    timing
                } else {
                    timing.continuation()
                },
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

/// Whether the entry renders nothing: a record with no display, or an
/// assistant message of only thinking while thinking is hidden.
pub(super) fn entry_shows_nothing(entry: &Entry<'_>, options: &RenderOptions) -> bool {
    match entry.kind {
        EntryKind::Message { message, .. } => {
            !options.show_thinking
                && message.role == Role::Assistant
                && !message.parts.is_empty()
                && message
                    .parts
                    .iter()
                    .all(|part| matches!(part, Part::Thinking(_)))
        }
        ref kind => kind.shows_nothing(),
    }
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

/// The group's messages of `role`, with the entry index each came from.
fn group_messages<'a>(
    entries: &'a [RenderableEntry],
    pending: &'a PendingToolSummary,
    role: Role,
) -> impl Iterator<Item = (usize, &'a Message)> {
    entries[pending.first_parsed_idx..=pending.last_parsed_idx]
        .iter()
        .filter_map(move |parsed| match parsed.entry.kind {
            EntryKind::Message { message, .. }
                if message.role == role
                    && message.parent_tool_use_id.as_deref() == pending.parent_id.as_deref() =>
            {
                Some((parsed.entry_index, message))
            }
            _ => None,
        })
}

/// The summary line for a pending group; see [`summary_line`].
fn group_summary_line(
    entries: &[RenderableEntry],
    pending: &PendingToolSummary,
    width: usize,
) -> String {
    let parts =
        group_messages(entries, pending, Role::Assistant).flat_map(|(_, message)| &message.parts);
    summary_line(&pending.summary, parts, width)
}

/// Calls mode: each call of the group on its own clickable row. A clicked row
/// expands into that call and its result; clicking the call's first line
/// collapses it again.
fn render_calls_group(
    lines: &mut Vec<RenderedLine>,
    entries: &[RenderableEntry],
    pending: &PendingToolSummary,
    label: &str,
    timing: TimingSlot<'_>,
    options: &RenderOptions,
) {
    let parent_id = pending.parent_id.as_deref();
    let mut rendered_any = false;
    let mut after_expanded = false;
    for (entry_index, message) in group_messages(entries, pending, Role::Assistant) {
        for (block_idx, block) in message.parts.iter().enumerate() {
            let Part::ToolCall { id, name, input } = block else {
                continue;
            };
            let row_id = make_tool_output_id(
                entry_index,
                parent_id,
                block_idx,
                ToolOutputKind::CallRow,
                Some(id),
            );
            let row_timing = if rendered_any {
                timing.continuation()
            } else {
                timing
            };
            let expanded = options.expanded_tool_outputs.contains(&row_id);
            if after_expanded {
                lines.push(RenderedLine::new(vec![]));
            }
            if expanded {
                let first_line = lines.len();
                render_call_details(
                    lines,
                    entries,
                    pending,
                    (entry_index, block_idx, id, name, input),
                    label,
                    row_timing,
                    options,
                );
                if let Some(line) = lines.get_mut(first_line) {
                    line.tool_output_id = Some(row_id);
                    line.clickable = true;
                }
            } else {
                render_tool_activity_summary(
                    lines,
                    label,
                    th().accent_dim,
                    parent_id.is_some(),
                    row_timing,
                    vec![call_line(name, input, options.content_width)],
                    Some(&row_id),
                );
            }
            rendered_any = true;
            after_expanded = expanded;
        }
    }
}

/// One call of a group in truncated form, followed by its result when the
/// group holds it.
fn render_call_details(
    lines: &mut Vec<RenderedLine>,
    entries: &[RenderableEntry],
    pending: &PendingToolSummary,
    (entry_index, block_idx, call_id, name, input): (usize, usize, &str, &str, &serde_json::Value),
    label: &str,
    timing: TimingSlot<'_>,
    options: &RenderOptions,
) {
    let parent_id = pending.parent_id.as_deref();
    let call_output_id = make_tool_output_id(
        entry_index,
        parent_id,
        block_idx,
        ToolOutputKind::ToolCall,
        Some(call_id),
    );
    render_tool_call(
        lines,
        &ToolCallRenderSpec {
            name,
            input,
            label,
            label_color: th().accent_dim,
            dimmed: true,
            content_width: options.content_width,
            timing,
            tool_display: ToolDisplayMode::Truncated,
            tool_output_id: &call_output_id,
            expanded: options.expanded_tool_outputs.contains(&call_output_id),
        },
    );

    let result = group_messages(entries, pending, Role::User).find_map(|(entry_index, message)| {
        message
            .parts
            .iter()
            .enumerate()
            .find_map(|(block_idx, part)| match part {
                Part::ToolResult {
                    output,
                    call_id: result_call_id,
                } if result_call_id == call_id => Some((entry_index, block_idx, output)),
                _ => None,
            })
    });
    let Some((entry_index, block_idx, output)) = result else {
        return;
    };
    let result_output_id = make_tool_output_id(
        entry_index,
        parent_id,
        block_idx,
        ToolOutputKind::ToolResult,
        Some(call_id),
    );
    lines.push(RenderedLine::new(vec![]));
    render_tool_result(
        lines,
        &ToolResultRenderSpec {
            text: &tool_result_display_text(output.as_ref()),
            content_width: options.content_width,
            timing: timing.continuation(),
            tool_display: ToolDisplayMode::Truncated,
            tool_output_id: &result_output_id,
            expanded: options.expanded_tool_outputs.contains(&result_output_id),
        },
    );
}

/// The collapsed rows for the calls among `parts`: each call on one line in
/// calls mode, otherwise the single summary line.
pub(super) fn summary_rows<'a>(
    summary: &ToolActivitySummary,
    parts: impl Iterator<Item = &'a Part>,
    options: &RenderOptions,
) -> Vec<String> {
    let width = options.content_width;
    if options.tool_display != ToolDisplayMode::Calls {
        return vec![summary_line(summary, parts, width)];
    }
    parts
        .filter_map(|part| match part {
            Part::ToolCall { name, input, .. } => Some(call_line(name, input, width)),
            _ => None,
        })
        .collect()
}

/// A tool call as one formatted line cut to `width`. A call with more to
/// show than its header, such as a multi-line command, ends with how many
/// lines it hides.
fn call_line(name: &str, input: &serde_json::Value, width: usize) -> String {
    let formatted = crate::tool_format::format_tool_call(name, input, 0);
    let hidden = formatted
        .body
        .as_deref()
        .map_or(0, |body| body.lines().count());
    if hidden == 0 {
        return truncate_end(&formatted.header, width);
    }
    let noun = if hidden == 1 { "line" } else { "lines" };
    let suffix = format!(" ({hidden} more {noun})");
    let header_width = width.saturating_sub(UnicodeWidthStr::width(suffix.as_str()));
    format!("{}{suffix}", truncate_end(&formatted.header, header_width))
}

/// The summary text for the calls among `parts`. A single call to a tool
/// without its own summary wording shows as the call itself on one line cut
/// to `width`, which says more than "Called clock__sleep". Any other text is
/// cut to `width` too, since tool names make it unbounded.
fn summary_line<'a>(
    summary: &ToolActivitySummary,
    mut parts: impl Iterator<Item = &'a Part>,
    width: usize,
) -> String {
    let single_call = parts.find_map(|part| match part {
        Part::ToolCall { name, input, .. } => Some((name, input)),
        _ => None,
    });
    match single_call {
        Some((name, input)) if summary.is_single_other_tool() => call_line(name, input, width),
        _ => truncate_end(&summary.sentence(), width),
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
    if options.tool_display == ToolDisplayMode::Calls {
        render_calls_group(lines, entries, &pending, &label, timing, options);
    } else if options.expanded_tool_outputs.contains(&pending.id) {
        render_summary_group_details(lines, entries, &pending, options);
    } else {
        render_tool_activity_summary(
            lines,
            &label,
            th().accent_dim,
            pending.parent_id.is_some(),
            timing,
            vec![group_summary_line(entries, &pending, options.content_width)],
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
