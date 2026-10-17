use std::borrow::Cow;

use crate::history::model::{Message, Part, Role};
use crate::history::view::{Entry, EntryKind};

use super::RenderedLine;

use super::commands::process_command_message;
use super::ledger::{render_ledger_block_styled, render_ledger_block_styled_dimmed};
use super::markdown::{apply_thinking_style, render_markdown_to_lines};
use super::style::subagent_label;
use super::summary::{render_tool_activity_summary, summarize_tool_calls, summary_line};
use super::timing::{RowTiming, TimingSlot};
use super::tools::{
    ToolCallRenderSpec, ToolOutputKind, ToolResultRenderSpec, format_tool_result_content,
    make_tool_output_id, render_dimmed_tool_result_body, render_subagent_tool_result_header,
    render_tool_call, render_tool_result, tool_result_display_text,
};
use super::*;

/// Classify an entry and dispatch to the matching focused render
/// function. This is the only entry point used by the rest of the
/// viewer; per-kind rendering lives in `render_user_message`,
/// `render_assistant_message`, and `render_agent_progress_message`.
pub(super) fn render_entry(
    lines: &mut Vec<RenderedLine>,
    entry_index: usize,
    entry: &Entry<'_>,
    options: &RenderOptions,
) {
    match entry.kind {
        EntryKind::Hidden
        | EntryKind::Note {
            searchable: false, ..
        } => {}
        EntryKind::Note {
            label,
            text,
            searchable: true,
        } => {
            let content = [Part::Text(text.to_owned())];
            let ctx = EntryCtx {
                style: MessageStyle::for_pi_metadata(label),
                parent_id: None,
                entry_index,
                options,
            };
            let ts = entry_timestamp(options, entry.timestamp);
            render_user_message(
                lines,
                &ctx,
                RowTiming::new(options.show_timing, ts.as_deref()),
                &content,
            );
        }
        EntryKind::Subagent { agent_id, message } => {
            if options.show_thinking {
                render_agent_progress_message(lines, entry_index, agent_id, message, options);
            }
        }
        EntryKind::Message { message, speaker } => {
            if message.parent_tool_use_id.is_some() && !options.show_thinking {
                return;
            }
            let parent_id = message.parent_tool_use_id.as_deref();
            let style = match message.role {
                Role::User => MessageStyle::for_user(parent_id, &message.parts),
                Role::Assistant => MessageStyle::for_assistant(parent_id, speaker),
            };
            let ctx = EntryCtx {
                style,
                parent_id,
                entry_index,
                options,
            };
            let ts = entry_timestamp(options, entry.timestamp);
            let timing = RowTiming::new(options.show_timing, ts.as_deref());
            match message.role {
                Role::User => render_user_message(lines, &ctx, timing, &message.parts),
                Role::Assistant => render_assistant_message(lines, &ctx, timing, &message.parts),
            }
        }
    }
}

fn entry_timestamp(
    options: &RenderOptions,
    timestamp: Option<chrono::DateTime<chrono::Utc>>,
) -> Option<String> {
    timestamp
        .filter(|_| options.show_timing)
        .map(format_timestamp)
}

/// Captures the visual style and label for a single message render pass.
///
/// Each message kind (top-level user, top-level assistant, subagent
/// user/assistant, agent_progress user/assistant) maps to one of these
/// styles. Centralizing them removes the per-block "is this nested?"
/// branches that previously lived inside the render functions.
struct MessageStyle<'a> {
    label: Cow<'a, str>,
    label_color: (u8, u8, u8),
    /// Whether the label and content render dimmed (subagent / skill).
    dimmed: bool,
    /// Whether the first text-block label is bold.
    bold: bool,
    /// Whether the message renders as a nested/subagent message —
    /// controls the special tool-result header and skips thinking
    /// blocks. Distinct from `dimmed` because skill-mode user messages
    /// are dimmed but not nested.
    is_subagent: bool,
}

impl<'a> MessageStyle<'a> {
    fn for_user(parent_id: Option<&'a str>, parts: &[Part]) -> Self {
        if let Some(p) = parent_id {
            return Self {
                label: Cow::Owned(subagent_label(p)),
                label_color: th().text_primary,
                dimmed: true,
                bold: false,
                is_subagent: true,
            };
        }
        let is_skill = parts.iter().any(|part| {
            matches!(part, Part::Text(text)
                if text.trim().starts_with("Base directory for this skill:"))
        });
        Self {
            label: Cow::Borrowed("You"),
            label_color: th().text_primary,
            dimmed: is_skill,
            bold: !is_skill,
            is_subagent: false,
        }
    }

    fn for_assistant(parent_id: Option<&'a str>, speaker: &'a str) -> Self {
        match parent_id {
            Some(p) => Self {
                label: Cow::Owned(subagent_label(p)),
                label_color: th().accent,
                dimmed: true,
                bold: false,
                is_subagent: true,
            },
            None => Self {
                label: Cow::Borrowed(speaker),
                label_color: th().accent,
                dimmed: false,
                bold: true,
                is_subagent: false,
            },
        }
    }

    fn for_pi_metadata(label: &'a str) -> Self {
        Self {
            label: Cow::Borrowed(label),
            label_color: th().text_secondary,
            dimmed: true,
            bold: false,
            is_subagent: false,
        }
    }

    fn for_agent_user(short_id: &str) -> Self {
        Self {
            label: Cow::Owned(format!("↳{}", short_id)),
            label_color: th().text_primary,
            dimmed: true,
            bold: false,
            is_subagent: true,
        }
    }

    fn for_agent_assistant(short_id: &str) -> Self {
        Self {
            label: Cow::Owned(format!("↳{}", short_id)),
            label_color: th().accent,
            dimmed: true,
            bold: false,
            is_subagent: true,
        }
    }
}

/// Per-entry rendering context: style + identity + immutable options.
/// Held by reference so the focused render functions read it without
/// taking ownership of the style and so the per-row timing cursor can
/// be mutated independently.
struct EntryCtx<'a> {
    style: MessageStyle<'a>,
    parent_id: Option<&'a str>,
    entry_index: usize,
    options: &'a RenderOptions,
}

// ---------------------------------------------------------------------
// Focused per-kind render functions.
//
// These replace the previous `MessageRenderer` orchestrator. Each one
// owns the block-pipeline template for one entry kind and pushes a
// trailing blank only when something rendered. The assistant template
// ordering (text → tool summary → tool calls → thinking) is the
// documented compatibility contract — it does not match raw JSON block
// order.
// ---------------------------------------------------------------------

/// User template: text (first), then tool results (second).
fn render_user_message(
    lines: &mut Vec<RenderedLine>,
    ctx: &EntryCtx<'_>,
    mut timing: RowTiming<'_>,
    parts: &[Part],
) {
    let mut printed = step_user_text(lines, ctx, &mut timing, parts);
    if ctx.options.tool_display.shows_details() {
        printed |= step_user_tool_results(lines, ctx, &mut timing, parts);
    }
    if printed {
        lines.push(RenderedLine::new(vec![]));
    }
}

/// Assistant template: text → tool summary → tool calls → thinking.
/// Order is intentional and matches pre-refactor behavior, not the
/// raw JSON block order.
fn render_assistant_message(
    lines: &mut Vec<RenderedLine>,
    ctx: &EntryCtx<'_>,
    mut timing: RowTiming<'_>,
    blocks: &[Part],
) {
    let mut printed = step_assistant_text(lines, ctx, &mut timing, blocks);
    printed |= step_tool_summary(lines, ctx, &mut timing, blocks);
    if ctx.options.tool_display.shows_details() {
        printed |= step_tool_calls(lines, ctx, &mut timing, blocks);
    }
    printed |= step_thinking(lines, ctx, &mut timing, blocks);
    if printed {
        lines.push(RenderedLine::new(vec![]));
    }
}

/// agent_progress user template: aggregated text, then tool results.
fn render_agent_progress_user_message(
    lines: &mut Vec<RenderedLine>,
    ctx: &EntryCtx<'_>,
    mut timing: RowTiming<'_>,
    blocks: &[Part],
) {
    let mut printed = step_aggregated_text(lines, ctx, &mut timing, blocks);
    if ctx.options.tool_display.shows_details() {
        printed |= step_agent_tool_results(lines, ctx, blocks);
    }
    if printed {
        lines.push(RenderedLine::new(vec![]));
    }
}

/// agent_progress assistant template: aggregated text, then tool
/// summary, then tool calls. agent_progress assistant entries do not
/// render thinking blocks.
fn render_agent_progress_assistant_message(
    lines: &mut Vec<RenderedLine>,
    ctx: &EntryCtx<'_>,
    mut timing: RowTiming<'_>,
    blocks: &[Part],
) {
    let mut printed = step_aggregated_text(lines, ctx, &mut timing, blocks);
    printed |= step_tool_summary(lines, ctx, &mut timing, blocks);
    if ctx.options.tool_display.shows_details() {
        printed |= step_tool_calls(lines, ctx, &mut timing, blocks);
    }
    if printed {
        lines.push(RenderedLine::new(vec![]));
    }
}

// ---- block-pipeline steps ----------------------------------------------

fn step_user_text(
    lines: &mut Vec<RenderedLine>,
    ctx: &EntryCtx<'_>,
    timing: &mut RowTiming<'_>,
    parts: &[Part],
) -> bool {
    let texts: Vec<String> = parts
        .iter()
        .filter_map(|part| match part {
            Part::Text(text) => process_command_message(text),
            _ => None,
        })
        .collect();
    let text = (!texts.is_empty()).then(|| texts.join("\n\n"));
    let Some(text) = text else { return false };
    let md_lines = render_markdown_to_lines(&text, ctx.options.content_width);
    if ctx.style.dimmed {
        render_ledger_block_styled_dimmed(
            lines,
            &ctx.style.label,
            ctx.style.label_color,
            md_lines,
            timing.pad(),
        );
    } else {
        render_ledger_block_styled(
            lines,
            &ctx.style.label,
            ctx.style.label_color,
            ctx.style.bold,
            md_lines,
            timing.take_once(),
        );
    }
    // Top-level slot is now spent regardless of the branch taken.
    let _ = timing.take_once();
    true
}

fn step_assistant_text(
    lines: &mut Vec<RenderedLine>,
    ctx: &EntryCtx<'_>,
    timing: &mut RowTiming<'_>,
    blocks: &[Part],
) -> bool {
    let mut printed = false;
    for block in blocks {
        let Part::Text(text) = block else {
            continue;
        };
        if text.trim().is_empty() {
            continue;
        }
        let md_lines = render_markdown_to_lines(text, ctx.options.content_width);
        if ctx.style.dimmed {
            render_ledger_block_styled_dimmed(
                lines,
                &ctx.style.label,
                ctx.style.label_color,
                md_lines,
                timing.pad(),
            );
            let _ = timing.take_once();
        } else {
            render_ledger_block_styled(
                lines,
                &ctx.style.label,
                ctx.style.label_color,
                ctx.style.bold,
                md_lines,
                timing.take_once(),
            );
        }
        printed = true;
    }
    printed
}

/// Aggregated-text step used by agent_progress entries: joins all
/// `ContentBlock::Text` blocks and renders them as one styled block.
fn step_aggregated_text(
    lines: &mut Vec<RenderedLine>,
    ctx: &EntryCtx<'_>,
    timing: &mut RowTiming<'_>,
    blocks: &[Part],
) -> bool {
    let texts: Vec<&str> = blocks
        .iter()
        .filter_map(|b| match b {
            Part::Text(text) => Some(text.as_str()),
            _ => None,
        })
        .collect();
    if texts.is_empty() {
        return false;
    }
    let combined = texts.join("\n\n");
    let md_lines = render_markdown_to_lines(&combined, ctx.options.content_width);
    render_ledger_block_styled_dimmed(
        lines,
        &ctx.style.label,
        ctx.style.label_color,
        md_lines,
        timing.pad(),
    );
    let _ = timing.take_once();
    true
}

fn step_tool_summary(
    lines: &mut Vec<RenderedLine>,
    ctx: &EntryCtx<'_>,
    timing: &mut RowTiming<'_>,
    blocks: &[Part],
) -> bool {
    if !ctx.options.tool_display.is_summary() {
        return false;
    }
    let summary = summarize_tool_calls(blocks);
    if summary.is_empty() {
        return false;
    }
    render_tool_activity_summary(
        lines,
        &ctx.style.label,
        th().accent_dim,
        ctx.style.is_subagent,
        timing.consume(),
        summary_line(&summary, blocks.iter(), ctx.options.content_width),
        None,
    );
    true
}

fn step_tool_calls(
    lines: &mut Vec<RenderedLine>,
    ctx: &EntryCtx<'_>,
    timing: &mut RowTiming<'_>,
    blocks: &[Part],
) -> bool {
    let mut printed = false;
    for (block_idx, block) in blocks.iter().enumerate() {
        let Part::ToolCall { id, name, input } = block else {
            continue;
        };
        let output_id = make_tool_output_id(
            ctx.entry_index,
            ctx.parent_id,
            block_idx,
            ToolOutputKind::ToolCall,
            Some(id),
        );
        let expanded = ctx.options.expanded_tool_outputs.contains(&output_id);
        let row_timing = if ctx.style.is_subagent {
            timing.pad()
        } else {
            timing.consume()
        };
        render_tool_call(
            lines,
            &ToolCallRenderSpec {
                name,
                input,
                label: &ctx.style.label,
                label_color: th().accent_dim,
                dimmed: ctx.style.dimmed,
                content_width: ctx.options.content_width,
                timing: row_timing,
                tool_display: ctx.options.tool_display,
                tool_output_id: &output_id,
                expanded,
            },
        );
        printed = true;
    }
    printed
}

fn step_thinking(
    lines: &mut Vec<RenderedLine>,
    ctx: &EntryCtx<'_>,
    timing: &mut RowTiming<'_>,
    blocks: &[Part],
) -> bool {
    if !ctx.options.show_thinking || ctx.style.is_subagent {
        return false;
    }
    let mut printed = false;
    for block in blocks {
        let Part::Thinking(thinking) = block else {
            continue;
        };
        if thinking.is_empty() {
            continue;
        }
        let md_lines = render_markdown_to_lines(thinking, ctx.options.content_width);
        let styled_lines = apply_thinking_style(md_lines);
        render_ledger_block_styled(
            lines,
            "Thinking",
            th().accent_dim,
            false,
            styled_lines,
            timing.consume(),
        );
        printed = true;
    }
    printed
}

fn step_user_tool_results(
    lines: &mut Vec<RenderedLine>,
    ctx: &EntryCtx<'_>,
    timing: &mut RowTiming<'_>,
    blocks: &[Part],
) -> bool {
    let mut printed = false;
    let tool_results = collect_tool_result_rows(
        ctx,
        blocks,
        if ctx.style.is_subagent {
            format_tool_result_content
        } else {
            tool_result_display_text
        },
    );
    for row in tool_results {
        let row_timing = if ctx.style.is_subagent {
            timing.pad()
        } else {
            timing.consume()
        };
        if ctx.style.is_subagent {
            render_dimmed_tool_result_row(lines, ctx, &row, row_timing);
        } else {
            render_normal_tool_result_row(lines, ctx, &row, row_timing);
        }
        printed = true;
    }
    printed
}

/// Tool-result step for agent_progress user blocks. Always renders
/// as a dimmed subagent result regardless of `style.is_subagent`,
/// because agent_progress user content always uses the dimmed
/// subagent result presentation.
fn step_agent_tool_results(
    lines: &mut Vec<RenderedLine>,
    ctx: &EntryCtx<'_>,
    blocks: &[Part],
) -> bool {
    let mut printed = false;
    let tool_results = collect_tool_result_rows(ctx, blocks, format_tool_result_content);
    for row in tool_results {
        let row_timing = TimingSlot::from_show_timing(ctx.options.show_timing);
        render_dimmed_tool_result_row(lines, ctx, &row, row_timing);
        printed = true;
    }
    printed
}

struct ToolResultRenderRow {
    output_id: ToolOutputId,
    expanded: bool,
    content: String,
}

fn collect_tool_result_rows(
    ctx: &EntryCtx<'_>,
    blocks: &[Part],
    mut content_text: impl FnMut(Option<&serde_json::Value>) -> String,
) -> Vec<ToolResultRenderRow> {
    let mut rows = Vec::new();
    for (block_idx, block) in blocks.iter().enumerate() {
        let Part::ToolResult {
            output: content,
            call_id: tool_use_id,
        } = block
        else {
            continue;
        };
        let output_id = make_tool_output_id(
            ctx.entry_index,
            ctx.parent_id,
            block_idx,
            ToolOutputKind::ToolResult,
            Some(tool_use_id),
        );
        rows.push(ToolResultRenderRow {
            expanded: ctx.options.expanded_tool_outputs.contains(&output_id),
            output_id,
            content: content_text(content.as_ref()),
        });
    }
    rows
}

fn render_normal_tool_result_row(
    lines: &mut Vec<RenderedLine>,
    ctx: &EntryCtx<'_>,
    row: &ToolResultRenderRow,
    timing: TimingSlot<'_>,
) {
    render_tool_result(
        lines,
        &ToolResultRenderSpec {
            text: &row.content,
            content_width: ctx.options.content_width,
            timing,
            tool_display: ctx.options.tool_display,
            tool_output_id: &row.output_id,
            expanded: row.expanded,
        },
    );
}

fn render_dimmed_tool_result_row(
    lines: &mut Vec<RenderedLine>,
    ctx: &EntryCtx<'_>,
    row: &ToolResultRenderRow,
    timing: TimingSlot<'_>,
) {
    render_subagent_tool_result_header(lines, timing);
    render_dimmed_tool_result_body(
        lines,
        ctx.options,
        &row.output_id,
        row.expanded,
        &row.content,
        timing,
    );
}

/// Get a truncated agent ID for display (max 7 characters)
fn short_agent_id(agent_id: &str) -> &str {
    &agent_id[..agent_id.len().min(7)]
}

/// Render an agent (subagent) progress message by classifying the
/// nested message and dispatching to the matching focused render
/// function. The `agent_id` is reused as the synthetic `parent_id`
/// for tool output IDs so expanded-tool keys stay stable across
/// re-renders.
fn render_agent_progress_message(
    lines: &mut Vec<RenderedLine>,
    entry_index: usize,
    agent_id: &str,
    message: &Message,
    options: &RenderOptions,
) {
    let short_id = short_agent_id(agent_id);
    let style = match message.role {
        Role::User => MessageStyle::for_agent_user(short_id),
        Role::Assistant => MessageStyle::for_agent_assistant(short_id),
    };
    let ctx = EntryCtx {
        style,
        parent_id: Some(agent_id),
        entry_index,
        options,
    };
    let timing = RowTiming::column_only(options.show_timing);
    match message.role {
        Role::User => render_agent_progress_user_message(lines, &ctx, timing, &message.parts),
        Role::Assistant => {
            render_agent_progress_assistant_message(lines, &ctx, timing, &message.parts)
        }
    }
}
