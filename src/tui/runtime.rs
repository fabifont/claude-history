use super::app::{Action, App, AppMode, DialogMode, TuiSearchOptions};
use super::ui;
use crate::cli::DebugLevel;
use crate::config::KeyBindings;
use crate::debug_log;
use crate::error::{AppError, Result};
use crate::history::{Conversation, LoaderMessage};
use crate::time_filter::TimeFilter;
use crate::tui::viewer::ToolDisplayMode;
use crossterm::event::{
    self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind, MouseButton,
    MouseEventKind,
};
use crossterm::terminal::{self, EnterAlternateScreen, LeaveAlternateScreen};
use ratatui::prelude::*;
use std::io::{self, Stderr};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

struct TerminalGuard {
    terminal: Terminal<CrosstermBackend<Stderr>>,
}

impl TerminalGuard {
    fn new() -> Result<Self> {
        terminal::enable_raw_mode().map_err(|e| AppError::Io(io::Error::other(e)))?;

        let mut stderr = io::stderr();
        if let Err(e) = crossterm::execute!(stderr, EnterAlternateScreen, EnableMouseCapture) {
            let _ = terminal::disable_raw_mode();
            return Err(AppError::Io(io::Error::other(e)));
        }

        let backend = CrosstermBackend::new(stderr);
        let terminal = match Terminal::new(backend) {
            Ok(t) => t,
            Err(e) => {
                let _ = terminal::disable_raw_mode();
                let _ =
                    crossterm::execute!(io::stderr(), DisableMouseCapture, LeaveAlternateScreen);
                return Err(AppError::Io(io::Error::other(e)));
            }
        };

        Ok(Self { terminal })
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = terminal::disable_raw_mode();
        let _ = crossterm::execute!(
            self.terminal.backend_mut(),
            DisableMouseCapture,
            LeaveAlternateScreen
        );
    }
}

const NAME_WIDTH: usize = 9;

fn calculate_content_width(frame_width: usize, show_timing: bool) -> usize {
    let timing_width = if show_timing {
        crate::tui::viewer::TIMESTAMP_WIDTH
    } else {
        0
    };
    frame_width.saturating_sub(NAME_WIDTH + 3 + crate::tui::viewer::GUTTER_WIDTH + timing_width)
}

struct FrameState {
    frame_area: Rect,
    viewport_height: usize,
    content_width: usize,
}

enum EventLoopResult<T> {
    Continue,
    Break,
    Return(T),
}

/// Everything the list event loop waits on, merged into one channel so the
/// loop can block until there is work.
enum RuntimeEvent {
    Input(Event),
    InputError(io::Error),
    Loader(LoaderMessage),
    LoaderDisconnected,
}

/// How long the input thread waits for terminal input before checking
/// whether it should stop.
const INPUT_POLL_INTERVAL: Duration = Duration::from_millis(20);

/// Forwards terminal input to the runtime channel. Dropping it stops and
/// joins the thread, so no input is consumed after the TUI exits.
struct InputReader {
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl InputReader {
    fn spawn(tx: Sender<RuntimeEvent>) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let thread_stop = stop.clone();
        let handle = std::thread::Builder::new()
            .name("tui-input".into())
            .spawn(move || {
                while !thread_stop.load(Ordering::Relaxed) {
                    let event = match event::poll(INPUT_POLL_INTERVAL) {
                        Ok(false) => continue,
                        Ok(true) => event::read().map(RuntimeEvent::Input),
                        Err(err) => Err(err),
                    };
                    let event = event.unwrap_or_else(RuntimeEvent::InputError);
                    let failed = matches!(event, RuntimeEvent::InputError(_));
                    if tx.send(event).is_err() || failed {
                        return;
                    }
                }
            })
            .ok();
        Self { stop, handle }
    }
}

impl Drop for InputReader {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// Forwards loader messages to the runtime channel until the loader finishes.
fn forward_loader(rx: Receiver<LoaderMessage>, tx: Sender<RuntimeEvent>) {
    std::thread::spawn(move || {
        for message in rx {
            let last = matches!(message, LoaderMessage::Done | LoaderMessage::Fatal(_));
            if tx.send(RuntimeEvent::Loader(message)).is_err() || last {
                return;
            }
        }
        let _ = tx.send(RuntimeEvent::LoaderDisconnected);
    });
}

fn read_event(wait: Duration) -> Result<Option<Event>> {
    if !event::poll(wait).map_err(|e| AppError::Io(io::Error::other(e)))? {
        return Ok(None);
    }

    event::read()
        .map(Some)
        .map_err(|e| AppError::Io(io::Error::other(e)))
}

fn prepare_frame(app: &mut App, terminal: &mut Terminal<CrosstermBackend<Stderr>>) -> FrameState {
    let frame_area = terminal.get_frame().area();
    let viewport_height = frame_area.height.saturating_sub(3) as usize;
    let content_width = calculate_content_width(frame_area.width as usize, app.show_timing());

    app.check_view_resize(content_width, viewport_height);
    let viewport_height = match app.app_mode() {
        AppMode::View(state) => {
            ui::view_layout_rects(frame_area, app, state).content.height as usize
        }
        AppMode::List => viewport_height,
    };

    FrameState {
        frame_area,
        viewport_height,
        content_width,
    }
}

fn draw_frame(app: &App, terminal: &mut Terminal<CrosstermBackend<Stderr>>) -> Result<()> {
    terminal.draw(|frame| ui::render(frame, app))?;
    Ok(())
}

fn handle_event<F>(
    app: &mut App,
    frame_state: &FrameState,
    ev: Event,
    allow_list_click_enter: bool,
    mut on_action: F,
) -> EventLoopResult<Option<Action>>
where
    F: FnMut(&mut App, Action) -> EventLoopResult<Option<Action>>,
{
    let key = match ev {
        Event::Key(k) if k.kind == KeyEventKind::Press => k,
        Event::Mouse(m) => {
            match m.kind {
                MouseEventKind::ScrollDown => {
                    app.scroll_mouse(3, frame_state.viewport_height);
                }
                MouseEventKind::ScrollUp => {
                    app.scroll_mouse(-3, frame_state.viewport_height);
                }
                MouseEventKind::Down(MouseButton::Left) => {
                    if app.handle_view_click(
                        m.row,
                        frame_state.frame_area,
                        frame_state.viewport_height,
                    ) {
                        return EventLoopResult::Break;
                    }
                    if allow_list_click_enter
                        && app.handle_list_click(m.row, frame_state.frame_area)
                    {
                        app.enter_view_mode(frame_state.content_width);
                        return EventLoopResult::Break;
                    }
                }
                MouseEventKind::Moved => {
                    app.handle_view_mouse_move(m.row, frame_state.frame_area);
                }
                _ => {}
            }
            return EventLoopResult::Continue;
        }
        _ => return EventLoopResult::Continue,
    };

    if allow_list_click_enter
        && matches!(app.app_mode(), AppMode::List)
        && *app.dialog_mode() == DialogMode::None
        && key.code == KeyCode::Enter
        && !app.is_loading()
        && app.selected().is_some()
    {
        app.enter_view_mode(frame_state.content_width);
        return EventLoopResult::Break;
    }

    let Some(action) = app.handle_key(key.code, key.modifiers, frame_state.viewport_height) else {
        return EventLoopResult::Continue;
    };
    on_action(app, action)
}

/// Loader bookkeeping for the initial load and for user-triggered refreshes.
#[derive(Default)]
struct LoadState {
    active: bool,
    refresh_buffer: Option<Vec<Conversation>>,
    refresh_failed: bool,
}

impl LoadState {
    fn start(&mut self, rx: Receiver<LoaderMessage>, tx: &Sender<RuntimeEvent>) {
        self.active = true;
        forward_loader(rx, tx.clone());
    }

    /// Applies one loader event. Returns true when the initial load finished.
    fn apply(&mut self, app: &mut App, event: RuntimeEvent) -> Result<bool> {
        match event {
            RuntimeEvent::Loader(LoaderMessage::Fatal(err)) => {
                self.active = false;
                if self.refresh_buffer.take().is_some() {
                    app.set_status_message(format!("Refresh failed: {err}"));
                } else {
                    return Err(err);
                }
            }
            RuntimeEvent::Loader(LoaderMessage::ProjectError) => {
                if self.refresh_buffer.is_some() {
                    self.refresh_failed = true;
                }
            }
            RuntimeEvent::Loader(LoaderMessage::Batch(convs)) => {
                if let Some(buffer) = &mut self.refresh_buffer {
                    buffer.extend(convs);
                } else {
                    app.append_conversations(convs);
                }
            }
            RuntimeEvent::Loader(LoaderMessage::Done) => {
                self.active = false;
                if let Some(conversations) = self.refresh_buffer.take() {
                    if self.refresh_failed {
                        app.set_status_message(
                            "Refresh incomplete; keeping existing sessions".to_string(),
                        );
                    } else {
                        let old_count = app.conversations().len();
                        app.replace_conversations(conversations);
                        let new_count = app.conversations().len();
                        app.set_status_message(format!(
                            "Refreshed sessions: {old_count} -> {new_count}"
                        ));
                    }
                    self.refresh_failed = false;
                } else {
                    app.finish_loading();
                    if app.conversations().is_empty() {
                        return Err(AppError::NoHistoryFound("selected scope".to_string()));
                    }
                    return Ok(true);
                }
            }
            RuntimeEvent::LoaderDisconnected => {
                self.active = false;
                if self.refresh_buffer.take().is_some() {
                    app.set_status_message("Refresh failed; keeping existing sessions".to_string());
                    self.refresh_failed = false;
                } else if app.is_loading() {
                    app.finish_loading();
                    if app.conversations().is_empty() {
                        return Err(AppError::NoHistoryFound("selected scope".to_string()));
                    }
                }
            }
            RuntimeEvent::Input(_) | RuntimeEvent::InputError(_) => {}
        }
        Ok(false)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn run_with_loader(
    show_last: bool,
    debug_level: Option<DebugLevel>,
    time_filter: TimeFilter,
    tool_display: ToolDisplayMode,
    show_thinking: bool,
    keys: KeyBindings,
    workspace_filter: bool,
    current_project_dir_name: Option<String>,
    exclude_projects: Vec<String>,
    search_options: TuiSearchOptions,
) -> Result<(Action, Arc<Vec<Conversation>>)> {
    let mut guard = TerminalGuard::new()?;
    let (events_tx, events_rx) = mpsc::channel();
    let _input = InputReader::spawn(events_tx.clone());
    let mut load = LoadState::default();
    load.start(
        crate::history::load_all_conversations_streaming(show_last, debug_level, time_filter),
        &events_tx,
    );
    let mut app = App::new_loading_with_options(
        tool_display,
        show_thinking,
        keys,
        workspace_filter,
        current_project_dir_name,
        exclude_projects,
        search_options,
    );

    let mut reported_first_rows = false;
    let mut report_load_done = false;

    loop {
        let frame_state = prepare_frame(&mut app, &mut guard.terminal);
        draw_frame(&app, &mut guard.terminal)?;
        if !reported_first_rows && app.filtered().len() >= BENCH_FIRST_ROWS {
            crate::bench::mark("first_rows");
            reported_first_rows = true;
        }
        if std::mem::take(&mut report_load_done) {
            crate::bench::mark("load_done");
            crate::bench::mark_peak_rss();
        }
        if app.receive_search_results() {
            draw_frame(&app, &mut guard.terminal)?;
        }

        // Search results arrive on the workers' own channels, so poll for
        // them while a search runs; otherwise sleep until an event arrives
        // or the status message expires.
        let wait = if app.has_search_work_in_flight() {
            Some(Duration::from_millis(8))
        } else {
            app.status_message_remaining()
        };
        let received = match wait {
            Some(wait) => events_rx.recv_timeout(wait),
            None => events_rx.recv().map_err(|_| RecvTimeoutError::Disconnected),
        };
        let mut next = match received {
            Ok(event) => Some(event),
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => unreachable!("events_tx is held by the loop"),
        };

        // Apply queued loader batches for up to 8 ms before redrawing, but
        // redraw right after each input event.
        let batch_started = Instant::now();
        while let Some(event) = next.take() {
            let ev = match event {
                RuntimeEvent::Input(ev) => ev,
                RuntimeEvent::InputError(err) => return Err(AppError::Io(err)),
                loader_event => {
                    report_load_done |= load.apply(&mut app, loader_event)?;
                    if batch_started.elapsed() < Duration::from_millis(8) {
                        next = events_rx.try_recv().ok();
                    }
                    continue;
                }
            };

            let event_result =
                handle_event(
                    &mut app,
                    &frame_state,
                    ev,
                    true,
                    |app, action| match action {
                        Action::Delete(ref path) => {
                            let source = app
                                .get_selected_source()
                                .unwrap_or(crate::history::Source::Claude);
                            let result = match source {
                                crate::history::Source::Claude => {
                                    let uuid =
                                        path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
                                    crate::history::delete_session_by_uuid(uuid).map(|_| ())
                                }
                                crate::history::Source::Pi => {
                                    crate::history::pi_loader::delete_session(path)
                                }
                                crate::history::Source::Omp => {
                                    crate::history::omp_loader::delete_session(path)
                                }
                            };
                            match result {
                                Ok(()) => {
                                    app.remove_selected_from_list();
                                    app.exit_view_mode();
                                    EventLoopResult::Continue
                                }
                                Err(e) => {
                                    let _ = debug_log::log_debug(&format!(
                                        "Failed to delete session {}: {}",
                                        path.display(),
                                        e
                                    ));
                                    EventLoopResult::Continue
                                }
                            }
                        }
                        _ => EventLoopResult::Return(Some(action)),
                    },
                );

            match event_result {
                EventLoopResult::Continue | EventLoopResult::Break => {}
                EventLoopResult::Return(Some(Action::Refresh)) => {
                    if !load.active {
                        load.refresh_buffer = Some(Vec::new());
                        load.refresh_failed = false;
                        load.start(
                            crate::history::load_all_conversations_streaming(
                                show_last,
                                debug_level,
                                time_filter,
                            ),
                            &events_tx,
                        );
                        app.set_status_message("Refreshing sessions...".to_string());
                    }
                }
                EventLoopResult::Return(Some(action)) => {
                    return Ok((action, app.into_conversations()));
                }
                EventLoopResult::Return(None) => {}
            }
        }
    }
}

pub fn run_single_file(
    path: PathBuf,
    tool_display: ToolDisplayMode,
    show_thinking: bool,
    keys: KeyBindings,
) -> Result<()> {
    let mut guard = TerminalGuard::new()?;
    let mut app = App::new_single_file(path, tool_display, show_thinking, keys);

    loop {
        let frame_state = prepare_frame(&mut app, &mut guard.terminal);
        draw_frame(&app, &mut guard.terminal)?;

        let Some(ev) = read_event(Duration::from_secs(3600))? else {
            continue;
        };
        let event_result = handle_event(
            &mut app,
            &frame_state,
            ev,
            false,
            |_, action| match action {
                Action::Quit => EventLoopResult::Return(Some(Action::Quit)),
                _ => EventLoopResult::Continue,
            },
        );

        match event_result {
            EventLoopResult::Continue => {}
            EventLoopResult::Break => continue,
            EventLoopResult::Return(Some(Action::Quit)) => return Ok(()),
            EventLoopResult::Return(None) => {}
            EventLoopResult::Return(Some(_)) => {}
        }
    }
}

/// Rows that must be available before the list counts as populated.
const BENCH_FIRST_ROWS: usize = 50;

/// Runs the list startup path against an offscreen terminal and prints timing
/// marks: first populated frame, loading finished, and results for `query`.
#[allow(clippy::too_many_arguments)]
pub fn bench_startup(
    show_last: bool,
    debug_level: Option<DebugLevel>,
    time_filter: TimeFilter,
    keys: KeyBindings,
    workspace_filter: bool,
    current_project_dir_name: Option<String>,
    exclude_projects: Vec<String>,
    query: &str,
) -> Result<()> {
    use crate::bench;
    use ratatui::backend::TestBackend;

    const WIDTH: u16 = 120;
    const HEIGHT: u16 = 40;

    let mut terminal = Terminal::new(TestBackend::new(WIDTH, HEIGHT))
        .map_err(|e| AppError::Io(io::Error::other(e)))?;
    let mut draw = |app: &App| {
        terminal
            .draw(|frame| ui::render(frame, app))
            .map(|_| ())
            .map_err(|e| AppError::Io(io::Error::other(e)))
    };

    let rx = crate::history::load_all_conversations_streaming(show_last, debug_level, time_filter);
    let mut app = App::new_loading_with_options(
        ToolDisplayMode::Hidden,
        false,
        keys,
        workspace_filter,
        current_project_dir_name,
        exclude_projects,
        TuiSearchOptions::default(),
    );
    draw(&app)?;
    bench::mark("first_frame");

    let mut first_rows = false;
    for message in rx {
        match message {
            LoaderMessage::Fatal(err) => return Err(err),
            LoaderMessage::ProjectError => {}
            LoaderMessage::Batch(convs) => {
                app.append_conversations(convs);
                if !first_rows && app.filtered().len() >= BENCH_FIRST_ROWS {
                    draw(&app)?;
                    bench::mark("first_rows");
                    first_rows = true;
                }
            }
            LoaderMessage::Done => break,
        }
    }
    bench::mark("load_received");

    app.finish_loading();
    draw(&app)?;
    if !first_rows {
        bench::mark("first_rows");
    }
    bench::mark("load_done");

    app.replace_query(query);
    while app.has_search_work_in_flight() {
        if !app.receive_search_results() {
            std::thread::sleep(Duration::from_millis(1));
        }
    }
    draw(&app)?;
    bench::mark("search_ready");
    eprintln!(
        "bench conversations {} results {}",
        app.conversations().len(),
        app.filtered().len()
    );
    bench::mark_peak_rss();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_width_keeps_timing_off_behavior() {
        assert_eq!(calculate_content_width(60, false), 46);
    }

    #[test]
    fn content_width_reserves_timestamp_prefix_when_timing_is_on() {
        assert_eq!(calculate_content_width(60, true), 32);
    }
}
