use super::{App, AppMode};
use crate::tui::preview_worker::{
    DocKey, LOADING_NOTICE_AFTER, Outcome, PreviewDoc, PreviewWorker, Want,
};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// The list's preview of the selected search result, rendered by a worker.
#[derive(Default)]
pub struct ListPreviewState {
    worker: Option<PreviewWorker>,
    /// Sequence number of the newest request
    seq: u64,
    /// What the newest request asked for, and when
    requested: Option<(Want, Instant)>,
    /// Newest reply to arrive for the requested conversation
    shown: Option<(Want, Outcome)>,
    /// Selection the newest request was made for, to prefetch in the
    /// direction the selection moves
    requested_row: Option<usize>,
    /// Direction between the last two requested selections
    direction: isize,
}

/// What the preview pane draws.
pub enum PreviewView<'a> {
    /// The selected conversation, at `anchor`. It may be one reply behind
    /// the query or pane width while the newer one renders.
    Ready {
        doc: &'a Arc<PreviewDoc>,
        anchor: Option<usize>,
    },
    /// The selected conversation could not be opened.
    Failed(&'a str),
    /// The selected conversation is rendering; `notice` once that has taken
    /// long enough to say so.
    Loading { notice: bool },
}

impl App {
    /// Starts rendering previews off the UI thread. `wake` runs whenever one
    /// is ready to draw.
    pub fn start_list_preview_worker(&mut self, wake: Box<dyn Fn() + Send>) {
        self.list_preview.worker = Some(PreviewWorker::spawn(wake));
    }

    /// Whether the list shows a preview of the selected result.
    pub fn list_preview_wanted(&self) -> bool {
        matches!(self.app_mode, AppMode::List)
            && self.readiness.list
            && !self.query.trim().is_empty()
            && self.selected.is_some()
    }

    fn list_preview_key(&self, row: usize, content_width: usize) -> Option<DocKey> {
        let &conv_idx = self.filtered.get(row)?;
        Some(DocKey {
            path: self.conversations[conv_idx].path.clone(),
            content_width,
            tool_display: self.tool_display,
            show_thinking: self.show_thinking,
            show_timing: self.show_timing,
        })
    }

    /// Asks for the selected result rendered at `content_width`, unless it
    /// already was, and waits up to `budget` for it so most previews appear
    /// in the frame that selects them.
    pub fn fill_list_preview(&mut self, content_width: usize, budget: Duration) {
        self.receive_list_previews(None);
        if !self.list_preview_wanted() {
            return;
        }
        let Some(row) = self.selected else {
            return;
        };
        let Some(key) = self.list_preview_key(row, content_width) else {
            return;
        };
        let want = Want {
            key,
            query: self.query.clone(),
            evidence: self
                .semantic_result_metadata_for_selection()
                .map(|metadata| metadata.explanation.evidence_preview.clone()),
        };
        if self
            .list_preview
            .requested
            .as_ref()
            .is_some_and(|(requested, _)| *requested == want)
        {
            return;
        }
        if self.list_preview.worker.is_none() {
            self.start_list_preview_worker(Box::new(|| {}));
        }

        let state = &mut self.list_preview;
        if let Some(previous) = state.requested_row
            && previous != row
        {
            state.direction = (row as isize - previous as isize).signum();
        }
        state.requested_row = Some(row);
        state.seq += 1;
        state.requested = Some((want.clone(), Instant::now()));
        if let Some(worker) = &state.worker {
            worker.show(state.seq, want);
        }
        self.receive_list_previews(Some(Instant::now() + budget));
    }

    /// Takes the replies that arrived, waiting until `deadline` for the one
    /// to the newest request.
    fn receive_list_previews(&mut self, deadline: Option<Instant>) {
        let mut answered = false;
        loop {
            let state = &mut self.list_preview;
            let Some(worker) = &state.worker else {
                return;
            };
            let reply = match deadline {
                Some(deadline) if !answered => worker.recv_until(deadline),
                _ => worker.try_recv(),
            };
            let Some(reply) = reply else {
                break;
            };
            // An older reply still shows the right conversation while the
            // newest renders, e.g. one keystroke behind on the query.
            let requested = state.requested.as_ref().map(|(want, _)| want);
            if requested.is_some_and(|want| want.key.path == reply.want.key.path) {
                answered |= reply.seq == state.seq;
                state.shown = Some((reply.want, reply.outcome));
            }
        }
        if answered {
            self.prefetch_list_preview();
        }
    }

    /// Renders the result the selection is likely to move to next into the
    /// worker's cache.
    fn prefetch_list_preview(&self) {
        let state = &self.list_preview;
        let (Some((want, _)), Some(row), Some(worker)) =
            (&state.requested, state.requested_row, &state.worker)
        else {
            return;
        };
        let direction = if state.direction == 0 {
            1
        } else {
            state.direction
        };
        if let Some(key) = row
            .checked_add_signed(direction)
            .and_then(|next| self.list_preview_key(next, want.key.content_width))
        {
            worker.prefetch(key);
        }
    }

    pub fn list_preview(&self) -> Option<PreviewView<'_>> {
        if !self.list_preview_wanted() {
            return None;
        }
        let state = &self.list_preview;
        let selected_path = &self.conversations[*self.filtered.get(self.selected?)?].path;
        let Some((requested, since)) = state
            .requested
            .as_ref()
            .filter(|(want, _)| want.key.path == *selected_path)
        else {
            // Requested in the next frame.
            return Some(PreviewView::Loading { notice: false });
        };
        match &state.shown {
            Some((want, outcome)) if want.key.path == requested.key.path => Some(match outcome {
                Outcome::Ready { doc, anchor } => PreviewView::Ready {
                    doc,
                    anchor: *anchor,
                },
                Outcome::Failed(error) => PreviewView::Failed(error),
            }),
            _ => Some(PreviewView::Loading {
                notice: since.elapsed() >= LOADING_NOTICE_AFTER,
            }),
        }
    }

    /// How long until the pane should say a preview is loading, while it
    /// does not yet.
    pub fn list_preview_notice_in(&self) -> Option<Duration> {
        let Some(PreviewView::Loading { notice: false }) = self.list_preview() else {
            return None;
        };
        let (_, since) = self.list_preview.requested.as_ref()?;
        Some(LOADING_NOTICE_AFTER.saturating_sub(since.elapsed()))
    }
}
