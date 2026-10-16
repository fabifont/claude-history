use super::*;
use crate::store::test_support::{conversation, unknown_fingerprints};
use crate::time_filter::TimeFilter;
use chrono::TimeZone;

const HOUR_MS: i64 = 3_600_000;

fn publish(conversations: &[Conversation]) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let corpus = crate::store::generation::Corpus::new(
        conversations,
        unknown_fingerprints(conversations.len()),
    );
    crate::store::generation::publish(dir.path(), &corpus).unwrap();
    dir
}

fn loading_app(workspace: Option<&str>, excluded: Vec<&str>) -> App {
    App::new_loading_with_options(
        ToolDisplayMode::Truncated,
        false,
        KeyBindings::default(),
        workspace.is_some(),
        workspace.map(str::to_string),
        excluded.into_iter().map(str::to_string).collect(),
        TuiSearchOptions::default(),
    )
}

fn show(app: &mut App, root: &tempfile::TempDir, time: TimeFilter) {
    let snapshot = crate::store::Snapshot::open(root.path(), true, time).unwrap();
    app.show_snapshot(snapshot);
}

fn wait_for_search(app: &mut App) {
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while app.has_search_work_in_flight() && std::time::Instant::now() < deadline {
        app.receive_search_results();
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(!app.has_search_work_in_flight(), "search did not finish");
}

fn type_query(app: &mut App, query: &str) {
    for c in query.chars() {
        app.handle_key(KeyCode::Char(c), KeyModifiers::NONE, 20);
    }
}

fn listed_paths(app: &App) -> Vec<String> {
    app.filtered()
        .iter()
        .map(|&index| app.conversations()[index].path.display().to_string())
        .collect()
}

fn corpus() -> Vec<Conversation> {
    vec![
        conversation("/p/-tmp-alpha/a.jsonl", HOUR_MS),
        conversation("/p/-tmp-beta/b.jsonl", 2 * HOUR_MS),
        conversation("/p/-tmp-alpha/c.jsonl", 3 * HOUR_MS),
    ]
}

#[test]
fn snapshot_rows_show_before_loading_and_swap_keeps_selection() {
    let root = publish(&corpus());
    let mut app = loading_app(None, vec![]);
    show(&mut app, &root, TimeFilter::default());

    assert!(app.readiness().list);
    assert!(app.is_loading());
    assert_eq!(
        listed_paths(&app),
        vec![
            "/p/-tmp-alpha/c.jsonl",
            "/p/-tmp-beta/b.jsonl",
            "/p/-tmp-alpha/a.jsonl"
        ]
    );
    assert_eq!(&*app.conversations()[0].preview, "last");
    app.select_next();

    let mut loaded = corpus();
    loaded.push(conversation("/p/-tmp-beta/d.jsonl", 4 * HOUR_MS));
    app.append_conversations(loaded);
    assert_eq!(app.conversations().len(), 3, "batches wait for the swap");
    assert_eq!(
        app.readiness().reconcile,
        ReconcileState::Running { loaded: 4 }
    );

    app.finish_loading();
    assert_eq!(*app.readiness(), Readiness::ready());
    assert_eq!(app.conversations().len(), 4);
    assert_eq!(&*app.conversations()[0].full_text, "full text");
    assert_eq!(
        app.get_selected_path(),
        Some(PathBuf::from("/p/-tmp-beta/b.jsonl"))
    );
}

#[test]
fn query_typed_over_snapshot_runs_after_loading() {
    let mut corpus = corpus();
    corpus[1].full_text = "needle".into();
    corpus[1].search_text_lower = "needle".into();
    let root = publish(&corpus);
    let mut app = loading_app(None, vec![]);
    show(&mut app, &root, TimeFilter::default());

    for c in "needle".chars() {
        app.handle_key(KeyCode::Char(c), KeyModifiers::NONE, 20);
    }
    assert_eq!(app.query(), "needle");
    assert!(!app.has_search_work_in_flight());
    assert_eq!(app.filtered().len(), 3);

    app.append_conversations(corpus);
    app.finish_loading();
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while app.has_search_work_in_flight() && std::time::Instant::now() < deadline {
        app.receive_search_results();
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(listed_paths(&app), vec!["/p/-tmp-beta/b.jsonl"]);
}

#[test]
fn snapshot_rows_honor_workspace_exclusions_and_time_window() {
    let mut corpus = corpus();
    corpus.push(conversation("/p/-tmp-gamma/e.jsonl", 5 * HOUR_MS));
    let root = publish(&corpus);

    let mut app = loading_app(Some("-tmp-alpha"), vec!["gamma"]);
    let time = TimeFilter {
        after: Some(chrono::Local.timestamp_millis_opt(2 * HOUR_MS).unwrap()),
        before: None,
    };
    show(&mut app, &root, time);
    assert_eq!(listed_paths(&app), vec!["/p/-tmp-alpha/c.jsonl"]);

    app.handle_key(KeyCode::Tab, KeyModifiers::NONE, 20);
    assert!(!app.workspace_filter());
    assert_eq!(
        listed_paths(&app),
        vec!["/p/-tmp-alpha/c.jsonl", "/p/-tmp-beta/b.jsonl"]
    );
}

#[test]
fn snapshot_rows_open_resume_and_fork_but_wait_to_edit() {
    let root = publish(&corpus());
    let mut app = loading_app(None, vec![]);
    show(&mut app, &root, TimeFilter::default());
    let keys = *app.keys();
    let selected = PathBuf::from("/p/-tmp-alpha/c.jsonl");

    assert!(matches!(
        app.handle_key(keys.resume.code, keys.resume.modifiers, 20),
        Some(Action::Resume(path)) if path == selected
    ));
    assert!(matches!(
        app.handle_key(keys.fork.code, keys.fork.modifiers, 20),
        Some(Action::ForkResume(path)) if path == selected
    ));
    assert!(app.handle_list_click(4, Rect::new(0, 0, 80, 30)));

    app.handle_key(keys.delete.code, keys.delete.modifiers, 20);
    app.handle_key(keys.rename.code, keys.rename.modifiers, 20);
    assert_eq!(*app.dialog_mode(), DialogMode::None);
}

#[test]
fn without_snapshot_rows_wait_for_loading() {
    let mut app = loading_app(None, vec![]);
    app.append_conversations(corpus());
    assert!(!app.readiness().list);
    assert!(!app.handle_list_click(4, Rect::new(0, 0, 80, 30)));
    app.finish_loading();
    assert!(app.readiness().list);
    assert_eq!(listed_paths(&app)[0], "/p/-tmp-alpha/c.jsonl");
}

#[test]
fn snapshot_pi_rows_match_workspace_by_canonical_project_path() {
    let current = std::env::current_dir().unwrap();
    let mut here = conversation("/pi/sessions/here.jsonl", HOUR_MS);
    here.source = crate::history::Source::Pi;
    here.project_path = Some(current.clone());
    let mut elsewhere = conversation("/pi/sessions/elsewhere.jsonl", 2 * HOUR_MS);
    elsewhere.source = crate::history::Source::Pi;
    elsewhere.project_path = Some(PathBuf::from("/tmp/elsewhere-project"));
    let root = publish(&[here, elsewhere]);

    let project_dir = crate::history::convert_path_to_project_dir_name(&current);
    let mut app = loading_app(Some(&project_dir), vec![]);
    show(&mut app, &root, TimeFilter::default());
    assert_eq!(listed_paths(&app), vec!["/pi/sessions/here.jsonl"]);
}

fn needle_corpus() -> Vec<Conversation> {
    let mut corpus = corpus();
    corpus[0].full_text = format!("{} hidden needle here", "padding ".repeat(400)).into();
    corpus[2].full_text = "needle in the newest".into();
    for conversation in &mut corpus {
        conversation.search_text_lower =
            crate::search::normalize_for_search(&conversation.full_text).into();
    }
    corpus
}

#[test]
fn index_answers_queries_before_loading_with_evidence_from_frames() {
    let root = publish(&needle_corpus());
    let mut app = loading_app(None, vec![]);
    show(&mut app, &root, TimeFilter::default());
    assert!(!app.readiness().lexical);
    assert!(app.open_search_index());
    assert!(app.readiness().lexical);
    assert!(app.is_loading());

    type_query(&mut app, "needle");
    wait_for_search(&mut app);
    assert_eq!(
        listed_paths(&app),
        vec!["/p/-tmp-alpha/c.jsonl", "/p/-tmp-alpha/a.jsonl"]
    );
    let hidden = app.filtered()[1];
    let evidence = app
        .lexical_evidence(hidden)
        .expect("evidence from the frame");
    let (start, end) = evidence.context_ranges[0];
    assert_eq!(&evidence.text[start..end], "needle");
    assert!(evidence.text.len() < 4000);

    // Loading keeps the index and maps its results to loaded conversations.
    app.append_conversations(needle_corpus());
    app.finish_loading();
    wait_for_search(&mut app);
    assert_eq!(
        listed_paths(&app),
        vec!["/p/-tmp-alpha/c.jsonl", "/p/-tmp-alpha/a.jsonl"]
    );
    assert_eq!(
        &*app.conversations()[app.filtered()[0]].full_text,
        "needle in the newest"
    );
}

#[test]
fn index_results_match_in_memory_search_after_loading() {
    let corpus = needle_corpus();
    let root = publish(&corpus);
    let mut app = loading_app(None, vec![]);
    show(&mut app, &root, TimeFilter::default());
    assert!(app.open_search_index());
    app.append_conversations(corpus.clone());
    app.finish_loading();

    let mut in_memory = loading_app(None, vec![]);
    in_memory.append_conversations(corpus);
    in_memory.finish_loading();
    for query in [
        "needle",
        "full",
        "\"hidden needle\"",
        "padding needle",
        "zzz",
    ] {
        app.replace_query(query);
        wait_for_search(&mut app);
        in_memory.replace_query(query);
        wait_for_search(&mut in_memory);
        assert_eq!(listed_paths(&app), listed_paths(&in_memory), "{query}");
    }
}

#[test]
fn index_scope_follows_workspace_filter() {
    let root = publish(&needle_corpus());
    let mut app = loading_app(Some("-tmp-beta"), vec![]);
    show(&mut app, &root, TimeFilter::default());
    assert!(app.open_search_index());
    type_query(&mut app, "full");
    wait_for_search(&mut app);
    assert_eq!(listed_paths(&app), vec!["/p/-tmp-beta/b.jsonl"]);

    app.handle_key(KeyCode::Tab, KeyModifiers::NONE, 20);
    wait_for_search(&mut app);
    assert_eq!(listed_paths(&app), vec!["/p/-tmp-beta/b.jsonl"]);
    assert!(!app.workspace_filter());
    type_query(&mut app, " text");
    wait_for_search(&mut app);
    assert_eq!(listed_paths(&app), vec!["/p/-tmp-beta/b.jsonl"]);
}

#[test]
fn newer_generation_swaps_in_and_reruns_the_query_keeping_selection() {
    let corpus = needle_corpus();
    let root = publish(&corpus);
    let mut app = loading_app(None, vec![]);
    show(&mut app, &root, TimeFilter::default());
    assert!(app.open_search_index());
    let mut loaded = corpus.clone();
    loaded.push(conversation("/p/-tmp-beta/d.jsonl", 4 * HOUR_MS));
    loaded[3].full_text = "late needle".into();
    loaded[3].search_text_lower = "late needle".into();
    app.append_conversations(loaded.clone());
    app.finish_loading();
    type_query(&mut app, "needle");
    wait_for_search(&mut app);
    // The new conversation is not in the index yet.
    assert_eq!(
        listed_paths(&app),
        vec!["/p/-tmp-alpha/c.jsonl", "/p/-tmp-alpha/a.jsonl"]
    );
    app.select_next();

    let same = crate::store::Snapshot::open(root.path(), true, TimeFilter::default()).unwrap();
    assert!(!app.swap_search_index(same), "same generation");

    let corpus = crate::store::generation::Corpus::new(&loaded, unknown_fingerprints(loaded.len()));
    crate::store::generation::publish(root.path(), &corpus).unwrap();
    let newer = crate::store::Snapshot::open(root.path(), true, TimeFilter::default()).unwrap();
    assert!(app.swap_search_index(newer));
    wait_for_search(&mut app);
    assert_eq!(
        listed_paths(&app),
        vec![
            "/p/-tmp-beta/d.jsonl",
            "/p/-tmp-alpha/c.jsonl",
            "/p/-tmp-alpha/a.jsonl"
        ]
    );
    assert_eq!(
        app.get_selected_path(),
        Some(PathBuf::from("/p/-tmp-alpha/a.jsonl"))
    );
}

#[test]
fn snapshot_without_index_files_waits_for_loading() {
    let root = publish(&needle_corpus());
    let generation = std::fs::read_to_string(root.path().join("CURRENT")).unwrap();
    std::fs::remove_file(root.path().join(generation.trim()).join("base.fst")).unwrap();
    let mut app = loading_app(None, vec![]);
    show(&mut app, &root, TimeFilter::default());
    assert!(!app.open_search_index());
    type_query(&mut app, "needle");
    assert!(!app.has_search_work_in_flight());

    app.append_conversations(needle_corpus());
    app.finish_loading();
    wait_for_search(&mut app);
    assert_eq!(
        listed_paths(&app),
        vec!["/p/-tmp-alpha/c.jsonl", "/p/-tmp-alpha/a.jsonl"]
    );
}
