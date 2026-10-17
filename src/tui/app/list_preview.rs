use super::{App, AppMode, line_matches_query};
use crate::tui::ui::ViewSearchQuery;
use crate::tui::viewer::{
    MessageRange, RenderOptions, RenderedLine, ToolDisplayMode, open_conversation,
    render_conversation,
};
use std::collections::{BTreeSet, HashSet};
use std::path::PathBuf;

/// The selected search result rendered as the viewer would, for the preview
/// pane beside or below the list.
pub struct ListPreview {
    key: PreviewKey,
    pub lines: Vec<RenderedLine>,
    pub messages: Vec<MessageRange>,
    /// Query and semantic evidence `anchor` was found for
    anchored_for: (String, Option<String>),
    /// Line the search result is at, if found
    pub anchor: Option<usize>,
}

#[derive(PartialEq)]
struct PreviewKey {
    path: PathBuf,
    content_width: usize,
    tool_display: ToolDisplayMode,
    show_thinking: bool,
    show_timing: bool,
}

impl ListPreview {
    /// First line to show in a pane `height` rows tall: the start of the
    /// message holding the match when that keeps the match in the upper two
    /// thirds, else the match a third of the way down.
    pub fn scroll_offset(&self, height: usize) -> usize {
        let Some(anchor) = self.anchor else {
            return 0;
        };
        let message_start = self
            .messages
            .iter()
            .find(|m| m.start_line <= anchor && anchor < m.end_line)
            .map(|m| m.start_line)
            .filter(|&start| anchor - start < height * 2 / 3);
        let offset = message_start.unwrap_or_else(|| anchor.saturating_sub(height / 3));
        offset.min(self.lines.len().saturating_sub(height))
    }
}

impl App {
    /// Whether the list shows a preview of the selected result.
    pub fn list_preview_wanted(&self) -> bool {
        matches!(self.app_mode, AppMode::List)
            && self.readiness.list
            && !self.query.trim().is_empty()
            && self.selected.is_some()
    }

    pub fn list_preview(&self) -> Option<&ListPreview> {
        let preview = self.list_preview.as_ref()?;
        let path = &self.conversations[*self.filtered.get(self.selected?)?].path;
        (&preview.key.path == path).then_some(preview)
    }

    /// Renders the selected result at `content_width` unless the preview
    /// already holds it, and finds the query in it.
    pub fn fill_list_preview(&mut self, content_width: usize) {
        if !self.list_preview_wanted() {
            return;
        }
        let Some(&conv_idx) = self.selected.and_then(|sel| self.filtered.get(sel)) else {
            return;
        };
        let key = PreviewKey {
            path: self.conversations[conv_idx].path.clone(),
            content_width,
            tool_display: self.tool_display,
            show_thinking: self.show_thinking,
            show_timing: self.show_timing,
        };

        if self.list_preview.as_ref().is_none_or(|p| p.key != key) {
            let options = RenderOptions {
                tool_display: key.tool_display,
                show_thinking: key.show_thinking,
                show_timing: key.show_timing,
                content_width,
                expanded_tool_outputs: BTreeSet::new(),
            };
            // A conversation that cannot be opened previews as empty.
            let rendered = open_conversation(&key.path)
                .ok()
                .map(|transcript| render_conversation(&transcript, &options));
            let (lines, messages) = rendered.map(|r| (r.lines, r.messages)).unwrap_or_default();
            self.list_preview = Some(ListPreview {
                key,
                lines,
                messages,
                anchored_for: Default::default(),
                anchor: None,
            });
        }

        let evidence = self
            .semantic_result_metadata_for_selection()
            .map(|metadata| metadata.explanation.evidence_preview.clone());
        let anchored_for = (self.query.clone(), evidence);
        let preview = self.list_preview.as_mut().expect("preview filled above");
        if preview.anchored_for != anchored_for {
            // The semantic evidence is what the row shows, so it wins over
            // the first literal match.
            let query = ViewSearchQuery::parse(&self.query);
            preview.anchor = anchored_for
                .1
                .as_deref()
                .and_then(|evidence| evidence_anchor(&preview.lines, &preview.messages, evidence))
                .or_else(|| {
                    preview
                        .lines
                        .iter()
                        .position(|line| line_matches_query(line, &query))
                });
            preview.anchored_for = anchored_for;
        }
    }
}

/// Words of `text` long enough to tell passages apart, lowercased.
fn distinctive_words(text: &str) -> HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|word| word.chars().count() >= 4)
        .map(str::to_lowercase)
        .collect()
}

fn line_text(line: &RenderedLine) -> String {
    line.spans.iter().map(|(text, _)| text.as_str()).collect()
}

/// Line of the message sharing the most words with `evidence`, at the line
/// of it sharing the most. `None` when no message shares enough to be the
/// passage the evidence was taken from.
fn evidence_anchor(
    lines: &[RenderedLine],
    messages: &[MessageRange],
    evidence: &str,
) -> Option<usize> {
    let wanted = distinctive_words(evidence);
    let line_words: Vec<HashSet<String>> = lines
        .iter()
        .map(|line| distinctive_words(&line_text(line)))
        .collect();
    let shared = |range: std::ops::Range<usize>| {
        line_words[range]
            .iter()
            .flatten()
            .filter(|word| wanted.contains(*word))
            .collect::<HashSet<_>>()
            .len()
    };

    let (count, message) = messages
        .iter()
        .map(|m| (shared(m.start_line..m.end_line), m))
        .max_by_key(|(count, m)| (*count, std::cmp::Reverse(m.start_line)))?;
    if count < wanted.len().min(3) || count == 0 {
        return None;
    }
    (message.start_line..message.end_line)
        .max_by_key(|&line| (shared(line..line + 1), std::cmp::Reverse(line)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lines(texts: &[&str]) -> Vec<RenderedLine> {
        texts
            .iter()
            .map(|text| RenderedLine::new(vec![(text.to_string(), Default::default())]))
            .collect()
    }

    fn message(entry_index: usize, start_line: usize, end_line: usize) -> MessageRange {
        MessageRange {
            entry_index,
            start_line,
            end_line,
        }
    }

    #[test]
    fn evidence_anchors_on_the_message_it_was_taken_from() {
        let lines = lines(&[
            "the cache is warm",
            "",
            "background daemon keeps",
            "semantic embeddings warm while idle",
            "",
            "unrelated reply",
        ]);
        let messages = [message(0, 0, 1), message(1, 2, 4), message(2, 5, 6)];

        let anchor = evidence_anchor(
            &lines,
            &messages,
            "…a background daemon keeps semantic embeddings warm…",
        );

        assert_eq!(anchor, Some(2));
    }

    #[test]
    fn evidence_sharing_too_little_anchors_nowhere() {
        let lines = lines(&["the cache is warm", "unrelated reply"]);
        let messages = [message(0, 0, 1), message(1, 1, 2)];

        assert_eq!(
            evidence_anchor(&lines, &messages, "daemon embeddings reconcile"),
            None
        );
    }
}
