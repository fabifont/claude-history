//! Renders the list's preview of the selected search result off the UI
//! thread.
//!
//! The UI sends the newest request; the worker drops requests superseded
//! before it gets to them and stops rendering one superseded midway.
//! Rendered conversations are kept by revision, so returning to one, or
//! typing on, only re-finds the match.

use crate::history::model::Revision;
use crate::tui::ui::ViewSearchQuery;
use crate::tui::viewer::{
    MessageRange, RenderOptions, RenderedLine, ToolDisplayMode, render_parsed_conversation_until,
    renderable_entries,
};
use std::collections::{BTreeSet, HashMap};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::{Duration, Instant};

/// Bytes of rendered conversations kept. A conversation larger than this is
/// still shown, then dropped once another one is rendered.
const CACHE_BYTES: usize = 64 * 1024 * 1024;
/// Rendered conversations kept, whatever their size.
const CACHE_ENTRIES: usize = 32;

/// A conversation as rendered for one pane width and set of view settings.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DocKey {
    pub path: PathBuf,
    pub content_width: usize,
    pub tool_display: ToolDisplayMode,
    pub show_thinking: bool,
    pub show_timing: bool,
}

/// A rendered conversation.
pub struct PreviewDoc {
    pub lines: Vec<RenderedLine>,
    pub messages: Vec<MessageRange>,
    /// Text of each line, its spans joined
    pub texts: Vec<Box<str>>,
    bytes: usize,
}

impl PreviewDoc {
    fn new(lines: Vec<RenderedLine>, messages: Vec<MessageRange>) -> Self {
        let texts: Vec<Box<str>> = lines
            .iter()
            .map(|line| {
                line.spans
                    .iter()
                    .map(|(text, _)| text.as_str())
                    .collect::<String>()
                    .into_boxed_str()
            })
            .collect();
        let span_bytes: usize = lines
            .iter()
            .flat_map(|line| &line.spans)
            .map(|(text, style)| text.capacity() + std::mem::size_of_val(style) + 24)
            .sum();
        let bytes = span_bytes
            + texts.iter().map(|text| text.len() + 16).sum::<usize>()
            + lines.len() * std::mem::size_of::<RenderedLine>()
            + messages.len() * std::mem::size_of::<MessageRange>();
        Self {
            lines,
            messages,
            texts,
            bytes,
        }
    }

    /// First line to show in a pane `height` rows tall: the start of the
    /// message holding `anchor` when that keeps it in the upper two thirds,
    /// else `anchor` a third of the way down.
    pub fn scroll_offset(&self, anchor: Option<usize>, height: usize) -> usize {
        let Some(anchor) = anchor else {
            return 0;
        };
        let message_start = self
            .message_at(anchor)
            .map(|m| m.start_line)
            .filter(|&start| anchor - start < height * 2 / 3);
        let offset = message_start.unwrap_or_else(|| anchor.saturating_sub(height / 3));
        offset.min(self.lines.len().saturating_sub(height))
    }

    fn message_at(&self, line: usize) -> Option<&MessageRange> {
        let idx = self.messages.partition_point(|m| m.end_line <= line);
        self.messages
            .get(idx)
            .filter(|m| m.start_line <= line && line < m.end_line)
    }
}

/// What the preview shows: a conversation, and what to find in it.
#[derive(Clone, Debug, PartialEq)]
pub struct Want {
    pub key: DocKey,
    pub query: String,
    /// Semantic evidence shown in the row, which the match is found by
    /// before the query
    pub evidence: Option<String>,
}

#[derive(Clone)]
pub enum Outcome {
    Ready {
        doc: Arc<PreviewDoc>,
        /// Line the search result is at, if found
        anchor: Option<usize>,
    },
    Failed(String),
}

pub struct Reply {
    pub seq: u64,
    pub want: Want,
    pub outcome: Outcome,
}

enum Message {
    /// Show this; supersedes everything sent before
    Show(u64, Want),
    /// Render this into the cache while nothing is to be shown, unless a
    /// newer `Show` arrives
    Prefetch(u64, DocKey),
}

pub struct PreviewWorker {
    tx: Sender<Message>,
    rx: Receiver<Reply>,
    /// Sequence number of the newest `Show`
    latest: Arc<AtomicU64>,
}

impl PreviewWorker {
    /// Starts the worker. `wake` runs whenever a reply is ready.
    pub fn spawn(wake: Box<dyn Fn() + Send>) -> Self {
        let (tx, worker_rx) = mpsc::channel();
        let (worker_tx, rx) = mpsc::channel();
        let latest = Arc::new(AtomicU64::new(0));
        let worker_latest = latest.clone();
        std::thread::Builder::new()
            .name("tui-preview".into())
            .spawn(move || {
                crate::syntax::warm();
                run(&worker_rx, &worker_tx, &worker_latest, &*wake);
            })
            .expect("spawn preview worker");
        Self { tx, rx, latest }
    }

    pub fn show(&self, seq: u64, want: Want) {
        self.latest.store(seq, Ordering::Release);
        let _ = self.tx.send(Message::Show(seq, want));
    }

    pub fn prefetch(&self, key: DocKey) {
        let seq = self.latest.load(Ordering::Acquire);
        let _ = self.tx.send(Message::Prefetch(seq, key));
    }

    pub fn try_recv(&self) -> Option<Reply> {
        self.rx.try_recv().ok()
    }

    pub fn recv_until(&self, deadline: Instant) -> Option<Reply> {
        let timeout = deadline.saturating_duration_since(Instant::now());
        self.rx.recv_timeout(timeout).ok()
    }
}

fn run(rx: &Receiver<Message>, tx: &Sender<Reply>, latest: &AtomicU64, wake: &dyn Fn()) {
    let mut cache = DocCache::default();
    let mut prefetch: Option<(u64, DocKey)> = None;
    loop {
        // Block only when there is nothing left to do.
        let first = if prefetch.is_none() {
            match rx.recv() {
                Ok(message) => Some(message),
                Err(_) => return,
            }
        } else {
            None
        };
        let mut show = None;
        for message in first.into_iter().chain(rx.try_iter()) {
            match message {
                Message::Show(seq, want) => {
                    show = Some((seq, want));
                    prefetch = None;
                }
                Message::Prefetch(seq, key) => prefetch = Some((seq, key)),
            }
        }

        if let Some((seq, want)) = show {
            let stop = || latest.load(Ordering::Acquire) != seq;
            let Some(outcome) = build(&mut cache, &want, &stop) else {
                continue;
            };
            if stop() || tx.send(Reply { seq, want, outcome }).is_err() {
                continue;
            }
            wake();
        } else if let Some((seq, key)) = prefetch.take() {
            let stop = || latest.load(Ordering::Acquire) != seq;
            let _ = document(&mut cache, &key, &stop);
        }
    }
}

/// `want` rendered and its match found, or `None` once `stop` returns true.
fn build(cache: &mut DocCache, want: &Want, stop: &dyn Fn() -> bool) -> Option<Outcome> {
    let doc = match document(cache, &want.key, stop)? {
        Ok(doc) => doc,
        Err(error) => return Some(Outcome::Failed(error)),
    };
    let anchor = want
        .evidence
        .as_deref()
        .and_then(|evidence| evidence_anchor(&doc, evidence))
        .or_else(|| query_anchor(&doc, &want.query));
    Some(Outcome::Ready { doc, anchor })
}

/// The conversation at `key` rendered, from the cache when its revision is
/// unchanged.
fn document(
    cache: &mut DocCache,
    key: &DocKey,
    stop: &dyn Fn() -> bool,
) -> Option<Result<Arc<PreviewDoc>, String>> {
    let started = Instant::now();
    let (revision, transcript) = match crate::history::service::shared().open_path(&key.path) {
        Ok(opened) => opened,
        Err(error) => return Some(Err(error.to_string())),
    };
    if let Some(doc) = cache.get(key, revision) {
        return Some(Ok(doc));
    }
    if stop() {
        return None;
    }
    let decoded = started.elapsed();
    let options = RenderOptions {
        tool_display: key.tool_display,
        show_thinking: key.show_thinking,
        show_timing: key.show_timing,
        content_width: key.content_width,
        expanded_tool_outputs: BTreeSet::new(),
    };
    let rendered =
        render_parsed_conversation_until(&renderable_entries(&transcript), &options, stop)?;
    let doc = Arc::new(PreviewDoc::new(rendered.lines, rendered.messages));
    if crate::bench::enabled() {
        crate::bench::mark(&format!(
            "preview_render open={:.1}ms total={:.1}ms lines={} bytes={}",
            decoded.as_secs_f64() * 1e3,
            started.elapsed().as_secs_f64() * 1e3,
            doc.lines.len(),
            doc.bytes
        ));
    }
    cache.insert(key.clone(), revision, doc.clone());
    Some(Ok(doc))
}

/// Rendered conversations, most recently used last, bounded by bytes.
#[derive(Default)]
struct DocCache {
    entries: Vec<(DocKey, Revision, Arc<PreviewDoc>)>,
}

impl DocCache {
    fn get(&mut self, key: &DocKey, revision: Revision) -> Option<Arc<PreviewDoc>> {
        let idx = self.entries.iter().position(|(k, _, _)| k == key)?;
        let entry = self.entries.remove(idx);
        if entry.1 != revision {
            return None;
        }
        let doc = entry.2.clone();
        self.entries.push(entry);
        Some(doc)
    }

    fn insert(&mut self, key: DocKey, revision: Revision, doc: Arc<PreviewDoc>) {
        self.entries.retain(|(k, _, _)| *k != key);
        self.entries.push((key, revision, doc));
        let mut bytes: usize = self.entries.iter().map(|(_, _, doc)| doc.bytes).sum();
        // The newest entry stays even when it alone is over budget.
        while self.entries.len() > 1 && (bytes > CACHE_BYTES || self.entries.len() > CACHE_ENTRIES)
        {
            bytes -= self.entries.remove(0).2.bytes;
        }
    }
}

/// First line matching `query`.
fn query_anchor(doc: &PreviewDoc, query: &str) -> Option<usize> {
    let query = ViewSearchQuery::parse(query);
    if query.is_empty() {
        return None;
    }
    doc.texts.iter().position(|text| query.has_match(text))
}

/// Most words compared from evidence; more add little to telling passages
/// apart.
const EVIDENCE_WORDS: usize = 64;

/// Line of the message sharing the most words with `evidence`, at the line
/// of it sharing the most; earliest on ties. `None` when no message shares
/// enough to be the passage the evidence was taken from.
fn evidence_anchor(doc: &PreviewDoc, evidence: &str) -> Option<usize> {
    let mut wanted: HashMap<String, u32> = HashMap::new();
    for word in distinctive_words(evidence) {
        let next = wanted.len() as u32;
        if next as usize == EVIDENCE_WORDS {
            break;
        }
        wanted.entry(word.to_lowercase()).or_insert(next);
    }
    if wanted.is_empty() {
        return None;
    }

    let mut lowered = String::new();
    let masks: Vec<u64> = doc
        .texts
        .iter()
        .map(|text| {
            distinctive_words(text).fold(0u64, |mask, word| {
                lowered.clear();
                lowered.extend(word.chars().flat_map(char::to_lowercase));
                wanted
                    .get(lowered.as_str())
                    .map_or(mask, |&bit| mask | 1 << bit)
            })
        })
        .collect();
    let shared = |range: std::ops::Range<usize>| {
        masks[range]
            .iter()
            .fold(0u64, |acc, mask| acc | mask)
            .count_ones() as usize
    };

    let (count, message) = doc
        .messages
        .iter()
        .map(|m| (shared(m.start_line..m.end_line), m))
        .max_by_key(|(count, m)| (*count, std::cmp::Reverse(m.start_line)))?;
    if count == 0 || count < wanted.len().min(3) {
        return None;
    }
    (message.start_line..message.end_line)
        .max_by_key(|&line| (masks[line].count_ones(), std::cmp::Reverse(line)))
}

/// Words of `text` long enough to tell passages apart.
fn distinctive_words(text: &str) -> impl Iterator<Item = &str> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|word| word.chars().nth(3).is_some())
}

/// How long the UI waits for a preview before drawing without it, so most
/// appear in the frame that selects them.
pub const SAME_FRAME_BUDGET: Duration = Duration::from_millis(20);

/// How long a preview may load before the pane says so.
pub const LOADING_NOTICE_AFTER: Duration = Duration::from_millis(150);

#[cfg(test)]
mod tests {
    use super::*;

    fn doc(texts: &[&str], messages: &[(usize, usize)]) -> PreviewDoc {
        PreviewDoc::new(
            texts
                .iter()
                .map(|text| RenderedLine::new(vec![(text.to_string(), Default::default())]))
                .collect(),
            messages
                .iter()
                .enumerate()
                .map(|(entry_index, &(start_line, end_line))| MessageRange {
                    entry_index,
                    start_line,
                    end_line,
                })
                .collect(),
        )
    }

    #[test]
    fn evidence_anchors_on_the_message_it_was_taken_from() {
        let doc = doc(
            &[
                "the cache is warm",
                "",
                "background daemon keeps",
                "semantic embeddings warm while idle",
                "",
                "unrelated reply",
            ],
            &[(0, 1), (2, 4), (5, 6)],
        );

        let anchor = evidence_anchor(&doc, "…a background daemon keeps semantic embeddings warm…");

        assert_eq!(anchor, Some(2));
    }

    #[test]
    fn evidence_matches_words_case_insensitively() {
        let doc = doc(&["intro", "Daemon Keeps Embeddings"], &[(0, 1), (1, 2)]);

        assert_eq!(evidence_anchor(&doc, "daemon keeps embeddings"), Some(1));
    }

    #[test]
    fn evidence_sharing_too_little_anchors_nowhere() {
        let doc = doc(&["the cache is warm", "unrelated reply"], &[(0, 1), (1, 2)]);

        assert_eq!(evidence_anchor(&doc, "daemon embeddings reconcile"), None);
    }

    #[test]
    fn query_anchors_on_the_first_matching_line() {
        let doc = doc(
            &["set up the tunnel", "rotate the token", "token again"],
            &[],
        );

        assert_eq!(query_anchor(&doc, "token"), Some(1));
        assert_eq!(query_anchor(&doc, "absent"), None);
    }

    #[test]
    fn scroll_starts_at_the_message_when_the_match_stays_high() {
        let doc = doc(&["x"; 40], &[(0, 10), (10, 40)]);

        assert_eq!(doc.scroll_offset(Some(14), 12), 10);
        // Too deep into its message: the match goes a third of the way down.
        assert_eq!(doc.scroll_offset(Some(30), 12), 26);
        assert_eq!(doc.scroll_offset(None, 12), 0);
    }

    #[test]
    fn cache_drops_least_recently_used_past_its_budget() {
        let key = |n: usize| DocKey {
            path: PathBuf::from(format!("/{n}")),
            content_width: 60,
            tool_display: ToolDisplayMode::default(),
            show_thinking: false,
            show_timing: false,
        };
        let revision = Revision([0; 32]);
        let mut cache = DocCache::default();
        for n in 0..CACHE_ENTRIES + 1 {
            cache.insert(key(n), revision, Arc::new(doc(&["x"], &[])));
        }

        assert!(cache.get(&key(0), revision).is_none());
        assert!(cache.get(&key(1), revision).is_some());
        assert!(cache.get(&key(1), Revision([1; 32])).is_none());
    }
}
