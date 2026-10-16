use super::*;
use crate::store::test_support::conversation;
use crate::time_filter::TimeFilter;
use chrono::TimeZone;

const HOUR_MS: i64 = 3_600_000;

fn publish(conversations: &[Conversation]) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    crate::store::generation::publish(dir.path(), conversations).unwrap();
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
fn snapshot_rows_show_first_and_index_lists_the_rest_keeping_selection() {
    let root = publish(&corpus());
    let mut app = loading_app(None, vec![]);
    show(&mut app, &root, TimeFilter::default());

    assert!(app.readiness().list);
    assert!(!app.is_loading());
    assert_eq!(
        listed_paths(&app),
        vec![
            "/p/-tmp-alpha/c.jsonl",
            "/p/-tmp-beta/b.jsonl",
            "/p/-tmp-alpha/a.jsonl"
        ]
    );
    assert_eq!(&*app.conversations()[0].preview, "last");
    assert_eq!(&*app.conversations()[0].full_text, "");
    app.select_next();

    assert!(app.open_search_index());
    assert_eq!(listed_paths(&app).len(), 3);
    assert_eq!(
        app.get_selected_path(),
        Some(PathBuf::from("/p/-tmp-beta/b.jsonl"))
    );
}

#[test]
fn query_typed_before_the_index_opens_runs_on_it() {
    let mut corpus = corpus();
    corpus[1].full_text = "needle".into();
    let root = publish(&corpus);
    let mut app = loading_app(None, vec![]);
    show(&mut app, &root, TimeFilter::default());

    type_query(&mut app, "needle");
    assert_eq!(app.query(), "needle");
    assert!(!app.has_search_work_in_flight());
    assert_eq!(app.filtered().len(), 3);

    assert!(app.open_search_index());
    wait_for_search(&mut app);
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
fn snapshot_rows_open_resume_fork_and_edit() {
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
    assert_eq!(
        app.conversation_at(&selected).map(|c| c.session_id),
        Some("c".to_string())
    );

    app.handle_key(keys.delete.code, keys.delete.modifiers, 20);
    assert_eq!(*app.dialog_mode(), DialogMode::ConfirmDelete);
}

#[test]
fn without_a_generation_rows_wait() {
    let mut app = loading_app(None, vec![]);
    assert!(!app.readiness().list);
    assert!(app.is_loading());
    assert!(!app.handle_list_click(4, Rect::new(0, 0, 80, 30)));
    app.show_conversations(corpus());
    assert!(app.readiness().list);
    assert_eq!(listed_paths(&app)[0], "/p/-tmp-alpha/c.jsonl");
}

#[test]
fn snapshot_pi_rows_match_workspace_by_canonical_project_path() {
    let current = std::env::current_dir().unwrap();
    let mut here = conversation("/pi/sessions/here.jsonl", HOUR_MS);
    here.source = crate::history::model::SourceKind::Pi;
    here.project_path = Some(current.clone());
    let mut elsewhere = conversation("/pi/sessions/elsewhere.jsonl", 2 * HOUR_MS);
    elsewhere.source = crate::history::model::SourceKind::Pi;
    elsewhere.project_path = Some(PathBuf::from("/tmp/elsewhere-project"));
    let root = publish(&[here, elsewhere]);

    let project_dir = crate::history::convert_path_to_project_dir_name(&current);
    let mut app = loading_app(Some(&project_dir), vec![]);
    show(&mut app, &root, TimeFilter::default());
    assert_eq!(listed_paths(&app), vec!["/pi/sessions/here.jsonl"]);
    assert!(!app.has_multiple_sources());
}

fn needle_corpus() -> Vec<Conversation> {
    let mut corpus = corpus();
    corpus[0].full_text = format!("{} hidden needle here", "padding ".repeat(400)).into();
    corpus[2].full_text = "needle in the newest".into();
    corpus
}

#[test]
fn index_answers_queries_with_evidence_from_frames() {
    let root = publish(&needle_corpus());
    let mut app = loading_app(None, vec![]);
    show(&mut app, &root, TimeFilter::default());
    assert!(!app.readiness().lexical);
    assert!(app.open_search_index());
    assert!(app.readiness().lexical);

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

    // Rows the worker left without evidence get it from their frames when
    // shown.
    app.lexical_evidence.clear();
    app.fill_visible_evidence(10);
    let evidence = app.lexical_evidence(hidden).expect("filled from the frame");
    let (start, end) = evidence.context_ranges[0];
    assert_eq!(&evidence.text[start..end], "needle");
}

#[test]
fn index_results_match_in_memory_search() {
    let corpus = needle_corpus();
    let root = publish(&corpus);
    let mut app = loading_app(None, vec![]);
    show(&mut app, &root, TimeFilter::default());
    assert!(app.open_search_index());

    let mut in_memory = loading_app(None, vec![]);
    in_memory.show_conversations(corpus);
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
    type_query(&mut app, "needle");
    wait_for_search(&mut app);
    assert_eq!(
        listed_paths(&app),
        vec!["/p/-tmp-alpha/c.jsonl", "/p/-tmp-alpha/a.jsonl"]
    );
    app.select_next();

    let same = crate::store::Snapshot::open(root.path(), true, TimeFilter::default()).unwrap();
    assert!(!app.swap_generation(same), "same generation");

    let mut newer = corpus.clone();
    newer.push(conversation("/p/-tmp-beta/d.jsonl", 4 * HOUR_MS));
    newer[3].full_text = "late needle".into();
    crate::store::generation::publish(root.path(), &newer).unwrap();
    let newer = crate::store::Snapshot::open(root.path(), true, TimeFilter::default()).unwrap();
    assert!(app.swap_generation(newer));
    assert_eq!(app.generation_len(), Some(4));
    // The rows shown stay listed while the query runs on the new index.
    assert_eq!(
        listed_paths(&app),
        vec!["/p/-tmp-alpha/c.jsonl", "/p/-tmp-alpha/a.jsonl"]
    );
    assert_eq!(
        app.get_selected_path(),
        Some(PathBuf::from("/p/-tmp-alpha/a.jsonl"))
    );
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
fn snapshot_without_index_files_browses_until_a_generation_has_them() {
    let root = publish(&needle_corpus());
    let generation = std::fs::read_to_string(root.path().join("CURRENT")).unwrap();
    std::fs::remove_file(root.path().join(generation.trim()).join("base.fst")).unwrap();
    let mut app = loading_app(None, vec![]);
    show(&mut app, &root, TimeFilter::default());
    assert!(!app.open_search_index());
    type_query(&mut app, "needle");
    assert!(!app.has_search_work_in_flight());
    assert_eq!(app.filtered().len(), 3);

    let mut newer = needle_corpus();
    newer[1].custom_title = Some("retitled".to_string());
    crate::store::generation::publish(root.path(), &newer).unwrap();
    let newer = crate::store::Snapshot::open(root.path(), true, TimeFilter::default()).unwrap();
    assert!(app.swap_generation(newer));
    wait_for_search(&mut app);
    assert_eq!(
        listed_paths(&app),
        vec!["/p/-tmp-alpha/c.jsonl", "/p/-tmp-alpha/a.jsonl"]
    );
}

#[test]
fn deleted_rows_stay_hidden_until_a_generation_drops_them() {
    let corpus = needle_corpus();
    let root = publish(&corpus);
    let mut app = loading_app(None, vec![]);
    show(&mut app, &root, TimeFilter::default());
    assert!(app.open_search_index());
    assert!(!app.take_reconcile_request());

    app.remove_selected_from_list();
    assert!(app.take_reconcile_request());
    assert_eq!(
        listed_paths(&app),
        vec!["/p/-tmp-beta/b.jsonl", "/p/-tmp-alpha/a.jsonl"]
    );
    type_query(&mut app, "needle");
    wait_for_search(&mut app);
    assert_eq!(listed_paths(&app), vec!["/p/-tmp-alpha/a.jsonl"]);

    // A generation published before the delete still holds the session.
    let mut stale = corpus.clone();
    stale[1].custom_title = Some("stale".to_string());
    crate::store::generation::publish(root.path(), &stale).unwrap();
    let stale = crate::store::Snapshot::open(root.path(), true, TimeFilter::default()).unwrap();
    assert!(app.swap_generation(stale));
    wait_for_search(&mut app);
    assert_eq!(listed_paths(&app), vec!["/p/-tmp-alpha/a.jsonl"]);

    crate::store::generation::publish(root.path(), &corpus[..2]).unwrap();
    let pruned = crate::store::Snapshot::open(root.path(), true, TimeFilter::default()).unwrap();
    assert!(app.swap_generation(pruned));
    assert!(app.removed.is_empty());
}

#[test]
fn renamed_rows_show_the_new_title_until_the_generation_has_it() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("abc123.jsonl");
    std::fs::write(
        &path,
        r#"{"type":"user","timestamp":"2024-01-01T00:00:00Z","message":{"role":"user","content":"hello body"}}"#.to_string() + "\n",
    )
    .unwrap();
    let mut session = conversation(path.to_str().unwrap(), 10 * HOUR_MS);
    session.full_text = "hello body".into();
    let mut corpus = corpus();
    corpus.push(session.clone());
    let root = publish(&corpus);
    let mut app = loading_app(None, vec![]);
    show(&mut app, &root, TimeFilter::default());
    assert!(app.open_search_index());
    assert_eq!(app.get_selected_path(), Some(path.clone()));

    app.start_rename();
    app.handle_rename_key(KeyCode::Char('u'), KeyModifiers::CONTROL);
    for c in "fresh".chars() {
        app.handle_rename_key(KeyCode::Char(c), KeyModifiers::empty());
    }
    app.handle_rename_key(KeyCode::Enter, KeyModifiers::empty());
    assert!(app.take_reconcile_request());
    let row = app.filtered()[0];
    assert_eq!(
        app.conversations()[row].custom_title.as_deref(),
        Some("fresh")
    );
    assert_eq!(app.get_selected_path(), Some(path.clone()));

    // A generation published before the rename keeps the overlay.
    let mut stale = corpus.clone();
    stale[0].custom_title = Some("stale".to_string());
    crate::store::generation::publish(root.path(), &stale).unwrap();
    let stale = crate::store::Snapshot::open(root.path(), true, TimeFilter::default()).unwrap();
    assert!(app.swap_generation(stale));
    let row = app.filtered()[0];
    assert_eq!(
        app.conversations()[row].custom_title.as_deref(),
        Some("fresh")
    );

    session.custom_title = Some("fresh".to_string());
    *corpus.last_mut().unwrap() = session;
    crate::store::generation::publish(root.path(), &corpus).unwrap();
    let renamed = crate::store::Snapshot::open(root.path(), true, TimeFilter::default()).unwrap();
    assert!(app.swap_generation(renamed));
    assert!(app.renamed.is_empty());
}

#[test]
fn uuid_query_finds_cards_beyond_the_first_rows() {
    let root = publish(&corpus());
    let mut app = loading_app(None, vec![]);
    show(&mut app, &root, TimeFilter::default());
    assert_eq!(app.find_or_load_uuid("B"), Some(1));
    assert_eq!(
        app.conversations()[1].path,
        PathBuf::from("/p/-tmp-beta/b.jsonl")
    );
}

#[test]
fn semantic_results_map_to_cards_by_path() {
    let root = publish(&corpus());
    let mut app = loading_app(Some("-tmp-alpha"), vec![]);
    show(&mut app, &root, TimeFilter::default());
    let mut loaded = corpus();
    loaded.push(conversation("/p/-tmp-alpha/unpublished.jsonl", 9 * HOUR_MS));
    app.semantic_corpus.loaded = Some(Arc::new(crate::store::SemanticCards::from_conversations(
        "loaded", loaded,
    )));

    assert_eq!(*app.semantic_scope_indices(), vec![0, 2, 3]);
    let (rows, _) = app.semantic_rows(vec![2, 3, 0], HashMap::new());
    let paths: Vec<_> = rows
        .iter()
        .map(|&row| app.conversations()[row].path.display().to_string())
        .collect();
    assert_eq!(
        paths,
        vec!["/p/-tmp-alpha/c.jsonl", "/p/-tmp-alpha/a.jsonl"]
    );
}

#[test]
fn semantic_corpus_is_read_from_the_generation_and_follows_swaps() {
    let mut corpus = corpus();
    corpus[0].semantic_turns = vec!["alpha turn".to_string()].into();
    corpus[1].semantic_turns = vec!["beta turn".to_string()].into();
    let root = publish(&corpus);
    let mut app = loading_app(None, vec![]);
    show(&mut app, &root, TimeFilter::default());
    app.removed
        .insert(crate::store::generation::key_hash("/p/-tmp-beta/b.jsonl"));

    assert!(!app.ensure_semantic_corpus());
    wait_for_search(&mut app);
    let first = app.semantic_corpus.loaded.clone().unwrap();
    let turns = |cards: &crate::store::SemanticCards, name: &str| {
        cards
            .conversations()
            .iter()
            .find(|card| card.path.ends_with(name))
            .map(|card| card.semantic_turns.clone())
            .unwrap()
    };
    assert_eq!(&*turns(&first, "a.jsonl"), ["alpha turn".to_string()]);
    assert!(Arc::ptr_eq(
        &app.semantic_conversations_snapshot,
        first.conversations()
    ));
    let scope = app.semantic_scope_indices();
    let scoped: Vec<_> = scope
        .iter()
        .map(|&index| first.conversations()[index].path.display().to_string())
        .collect();
    assert_eq!(
        scoped,
        vec!["/p/-tmp-alpha/c.jsonl", "/p/-tmp-alpha/a.jsonl"]
    );

    corpus[2].semantic_turns = vec!["gamma turn".to_string()].into();
    crate::store::generation::publish(root.path(), &corpus).unwrap();
    let next = crate::store::Snapshot::open(root.path(), true, TimeFilter::default()).unwrap();
    assert!(app.swap_generation(next));
    assert!(
        app.ensure_semantic_corpus(),
        "the older corpus serves meanwhile"
    );
    wait_for_search(&mut app);
    let second = app.semantic_corpus.loaded.clone().unwrap();
    assert_ne!(second.generation_name(), first.generation_name());
    assert_eq!(&*turns(&second, "c.jsonl"), ["gamma turn".to_string()]);
    assert!(Arc::ptr_eq(
        &turns(&second, "a.jsonl"),
        &turns(&first, "a.jsonl")
    ));
}

#[test]
fn typed_uuid_selects_its_card() {
    let mut corpus = corpus();
    corpus.push(conversation(
        "/p/-tmp-beta/6dd62dc0-8b64-4f5e-a651-5bd95216478e.jsonl",
        HOUR_MS / 2,
    ));
    let root = publish(&corpus);
    let mut app = loading_app(None, vec![]);
    show(&mut app, &root, TimeFilter::default());
    assert!(app.open_search_index());
    type_query(&mut app, "6dd62dc0-8b64-4f5e-a651-5bd95216478e");
    wait_for_search(&mut app);
    assert_eq!(
        listed_paths(&app),
        vec!["/p/-tmp-beta/6dd62dc0-8b64-4f5e-a651-5bd95216478e.jsonl"]
    );
}

#[test]
fn hidden_kinds_toggle_in_and_out_of_the_list_and_its_search() {
    let mut corpus = corpus();
    corpus[0].full_text = "needle".into();
    let mut exec = conversation("/p/-tmp-alpha/exec.jsonl", 4 * HOUR_MS);
    exec.provenance.automated = true;
    exec.full_text = "needle".into();
    let mut child = conversation("/p/-tmp-beta/child.jsonl", 5 * HOUR_MS);
    child.provenance.child = true;
    corpus.extend([exec, child]);
    let root = publish(&corpus);
    let mut app = loading_app(None, vec![]);
    show(&mut app, &root, TimeFilter::default());
    assert!(app.open_search_index());
    let shown = [
        "/p/-tmp-alpha/c.jsonl",
        "/p/-tmp-beta/b.jsonl",
        "/p/-tmp-alpha/a.jsonl",
    ];
    assert_eq!(listed_paths(&app), shown);

    let toggle = app.keys().all_kinds;
    app.handle_key(toggle.code, toggle.modifiers, 20);
    assert!(app.all_kinds());
    assert_eq!(
        listed_paths(&app),
        [
            "/p/-tmp-beta/child.jsonl",
            "/p/-tmp-alpha/exec.jsonl",
            "/p/-tmp-alpha/c.jsonl",
            "/p/-tmp-beta/b.jsonl",
            "/p/-tmp-alpha/a.jsonl",
        ]
    );

    type_query(&mut app, "needle");
    wait_for_search(&mut app);
    assert_eq!(
        listed_paths(&app),
        ["/p/-tmp-alpha/exec.jsonl", "/p/-tmp-alpha/a.jsonl"]
    );

    app.handle_key(toggle.code, toggle.modifiers, 20);
    wait_for_search(&mut app);
    assert!(!app.all_kinds());
    assert_eq!(listed_paths(&app), ["/p/-tmp-alpha/a.jsonl"]);

    app.handle_key(KeyCode::Esc, KeyModifiers::NONE, 20);
    assert_eq!(listed_paths(&app), shown);
}
