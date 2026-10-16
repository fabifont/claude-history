use super::*;
use crate::history::index;
use crate::history::model::{
    EventKind, ExecutionMode, HeadStatus, Message, Part, Relationship, Role, Visibility,
};
use crate::history::sources::Registry;

const LEGACY: &str = "0198f000-0000-7000-8000-000000000001";
const PAGED: &str = "01a00000-0000-7000-8000-000000000002";
const FORK: &str = "01a00000-0000-7000-8000-000000000003";
const REVERT: &str = "01a00000-0000-7000-8000-000000000004";
const REVERT2: &str = "01a00000-0000-7000-8000-000000000044";
const TORN: &str = "01a00000-0000-7000-8000-000000000005";
const EXEC: &str = "01a00000-0000-7000-8000-000000000006";
const SUB: &str = "01a00000-0000-7000-8000-000000000007";
const ARCH: &str = "01a00000-0000-7000-8000-000000000008";
const ORPHAN: &str = "01a00000-0000-7000-8000-000000000009";

fn fixture_home() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/codex")
}

/// The fixture home with a state database in `db_dir` that names the
/// threads' rollouts the way Codex does.
fn adapter(db_dir: &Path) -> CodexAdapter {
    create_state_db(&fixture_home(), db_dir);
    CodexAdapter::with_sqlite_home(fixture_home(), db_dir.to_path_buf())
}

/// A copy of the fixture home in `dir`, with its state database, for
/// tests that change it.
pub(crate) fn home_copy(dir: &Path) -> PathBuf {
    fn copy(from: &Path, to: &Path) {
        std::fs::create_dir_all(to).unwrap();
        for entry in std::fs::read_dir(from).unwrap() {
            let path = entry.unwrap().path();
            let target = to.join(path.file_name().unwrap());
            if path.is_dir() {
                copy(&path, &target);
            } else {
                std::fs::copy(&path, &target).unwrap();
            }
        }
    }
    let home = dir.join("codex");
    copy(&fixture_home(), &home);
    create_state_db(&home, &home);
    home
}

/// Writes `state_5.sqlite` in `db_dir` with a row of every paginated
/// thread of `home`; the orphaned one names a rollout that is gone.
fn create_state_db(home: &Path, db_dir: &Path) {
    let rollouts = discover::list(home).unwrap();
    let path_of = |thread: &str| {
        rollouts
            .iter()
            .filter(|rollout| rollout.thread_id == thread)
            .map(|rollout| rollout.path.clone())
            .max()
            .unwrap()
    };
    let paths: Vec<(&str, PathBuf)> = [PAGED, FORK, REVERT, TORN, EXEC, SUB, ARCH]
        .into_iter()
        .map(|thread| (thread, path_of(thread)))
        .chain([(
            ORPHAN,
            home.join(format!(
                "sessions/2026/09/02/rollout-2026-09-02T14-30-00-{ORPHAN}_{ORPHAN}.jsonl"
            )),
        )])
        .collect();
    let rows: Vec<_> = paths
        .iter()
        .map(|(thread, path)| {
            let name = (*thread == PAGED).then_some("Named in Codex");
            (
                *thread,
                path.as_path(),
                "paginated",
                name,
                1_788_000_000_000,
            )
        })
        .collect();
    state_db::tests::create(db_dir, 5, &rows);
}

fn decoded(adapter: &CodexAdapter, thread: &str) -> Transcript {
    let located = adapter.resolve(thread, far_future()).unwrap();
    adapter.decode(&located).unwrap().unwrap()
}

fn far_future() -> SystemTime {
    SystemTime::now() + std::time::Duration::from_secs(3600)
}

fn texts(transcript: &Transcript, visibility: Visibility) -> Vec<String> {
    transcript
        .events
        .iter()
        .filter(|event| event.visibility == visibility)
        .filter_map(|event| match &event.kind {
            EventKind::Message(Message { parts, .. }) => Some(parts),
            _ => None,
        })
        .flatten()
        .filter_map(|part| match part {
            Part::Text(text) | Part::Thinking(text) => Some(text.clone()),
            _ => None,
        })
        .collect()
}

fn mentions(transcript: &Transcript, visibility: Visibility, word: &str) -> bool {
    texts(transcript, visibility)
        .iter()
        .any(|text| text.contains(word))
}

#[test]
fn discovery_resolves_every_thread_of_the_home() {
    let db = tempfile::tempdir().unwrap();
    let adapter = adapter(db.path());
    let discovery = adapter.discover(far_future());
    assert_eq!(discovery.completeness, Completeness::Complete);
    let mut found: Vec<(String, HeadStatus, usize)> = discovery
        .found
        .iter()
        .map(|located| {
            let thread = adapter.thread_of(located).unwrap();
            assert_eq!(located.path, thread.head_link().path);
            assert!(located.stamp.is_known());
            (
                located.key.native_id.clone(),
                thread.head,
                thread.chain.len(),
            )
        })
        .collect();
    found.sort_by(|a, b| a.0.cmp(&b.0));
    let expected = [
        (LEGACY, HeadStatus::Inferred, 1),
        (PAGED, HeadStatus::Current, 1),
        (FORK, HeadStatus::Current, 2),
        (REVERT, HeadStatus::Current, 2),
        (TORN, HeadStatus::Current, 1),
        (EXEC, HeadStatus::Current, 1),
        (SUB, HeadStatus::Current, 1),
        (ARCH, HeadStatus::Current, 1),
        (ORPHAN, HeadStatus::Orphaned, 1),
    ];
    let mut expected: Vec<_> = expected
        .into_iter()
        .map(|(id, head, links)| (id.to_owned(), head, links))
        .collect();
    expected.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(found, expected);

    let resolved = adapter.resolve(REVERT, far_future()).unwrap();
    let listed = discovery
        .found
        .iter()
        .find(|located| located.key.native_id == REVERT)
        .unwrap();
    assert_eq!(resolved.stamp, listed.stamp);
    assert!(resolved.path.to_string_lossy().contains(REVERT2));
}

#[test]
fn legacy_rollouts_decode_event_messages_and_function_calls() {
    let db = tempfile::tempdir().unwrap();
    let transcript = decoded(&adapter(db.path()), LEGACY);
    assert_eq!(transcript.source, SourceKind::Codex);
    assert_eq!(transcript.mode, ExecutionMode::Interactive);
    assert_eq!(transcript.relationship, Relationship::Root);
    assert_eq!(transcript.meta.session_id, LEGACY);
    assert_eq!(
        transcript.meta.custom_title.as_deref(),
        Some("Zebra listing")
    );
    assert_eq!(transcript.meta.model.as_deref(), Some("gpt-5-codex"));
    assert_eq!(
        transcript.meta.cwd.as_deref(),
        Some(Path::new("/work/demo"))
    );
    assert_eq!(
        transcript.meta.updated_at.unwrap().to_rfc3339(),
        "2026-09-01T06:00:20+00:00"
    );
    assert!(transcript.diagnostics.is_empty());
    assert_eq!(
        texts(&transcript, Visibility::Dialogue),
        [
            "How do I list the zebra files?",
            "Thinking about **listing** files",
            "Run `ls zebra` to see zebra-stripes.txt.",
        ]
    );
    assert!(mentions(&transcript, Visibility::Injected, "AGENTS.md"));
    let parts: Vec<&Part> = transcript
        .events
        .iter()
        .filter_map(|event| match &event.kind {
            EventKind::Message(message) => Some(&message.parts),
            _ => None,
        })
        .flatten()
        .collect();
    assert!(parts.contains(&&Part::ToolCall {
        id: "call_legacy_1".to_owned(),
        name: "shell".to_owned(),
        input: serde_json::json!({"command": ["ls", "zebra"], "workdir": "/work/demo"}),
    }));
    assert!(parts.contains(&&Part::ToolResult {
        call_id: "call_legacy_1".to_owned(),
        output: Some(serde_json::json!("zebra-stripes.txt\n")),
    }));

    let conversation = index::index(&transcript).unwrap();
    // Cumulative token counts, with a repeated one counted once.
    assert_eq!(conversation.total_tokens, 2500 + 90);
    assert!(conversation.full_text.contains("zebra-stripes.txt"));
    assert!(!conversation.full_text.contains("legacy injected context"));
    assert_eq!(
        conversation.own_text.as_ref(),
        "How do I list the zebra files?"
    );
    assert_eq!(
        conversation.preview_first.as_ref(),
        "How do I list the zebra files? ... Run `ls zebra` to see zebra-stripes.txt."
    );
}

#[test]
fn paginated_rollouts_decode_turn_items_and_custom_tools() {
    let db = tempfile::tempdir().unwrap();
    let transcript = decoded(&adapter(db.path()), PAGED);
    assert_eq!(transcript.head, HeadStatus::Current);
    assert_eq!(
        transcript.meta.custom_title.as_deref(),
        Some("Named in Codex")
    );
    assert_eq!(
        texts(&transcript, Visibility::Dialogue),
        [
            "Explain the walrus operator",
            "Recalling **PEP 572**",
            "The walrus operator assigns inside expressions.",
            "Now show a narwhal example",
            "Here is a narwhal example.",
        ]
    );
    let result = transcript
        .events
        .iter()
        .find_map(|event| match &event.kind {
            EventKind::Message(message) if message.role == Role::User => {
                message.parts.iter().find_map(|part| match part {
                    Part::ToolResult { output, .. } => output.clone(),
                    _ => None,
                })
            }
            _ => None,
        });
    assert_eq!(
        result,
        Some(serde_json::json!([
            {"type": "text", "text": "Script completed\n"},
            {"type": "text", "text": "walrus ok"},
        ]))
    );
    let conversation = index::index(&transcript).unwrap();
    assert_eq!(conversation.total_tokens, 7000 + 160);
    assert!(conversation.full_text.starts_with("Named in Codex "));
    assert!(conversation.full_text.contains("walrus ok"));
    assert_eq!(
        conversation.own_text.as_ref(),
        "Explain the walrus operator Now show a narwhal example"
    );
}

#[test]
fn forks_inherit_the_base_prefix_and_reverts_keep_their_own() {
    let db = tempfile::tempdir().unwrap();
    let adapter = adapter(db.path());

    let fork = decoded(&adapter, FORK);
    assert_eq!(fork.segments.len(), 2);
    assert!(fork.segments[0].inherited);
    assert_eq!(fork.segments[0].end, 3787);
    assert!(mentions(&fork, Visibility::Inherited, "walrus"));
    assert!(!mentions(&fork, Visibility::Inherited, "narwhal"));
    assert!(mentions(&fork, Visibility::Dialogue, "ocelot"));
    let conversation = index::index(&fork).unwrap();
    assert!(!conversation.full_text.contains("walrus"));
    assert!(!conversation.own_text.contains("walrus"));
    assert!(conversation.own_text.contains("ocelot"));
    // Only the fork's own usage, beyond the base's cumulative count.
    assert_eq!(conversation.total_tokens, (4000 + 140) - (3000 + 100));

    let revert = decoded(&adapter, REVERT);
    assert_eq!(revert.segments.len(), 2);
    assert!(!revert.segments[0].inherited);
    let dialogue = texts(&revert, Visibility::Dialogue).join(" ");
    assert!(dialogue.contains("heron"));
    assert!(dialogue.contains("flamingo"));
    assert!(!dialogue.contains("pelican"));
}

#[test]
fn a_torn_last_line_is_a_soft_diagnostic() {
    let db = tempfile::tempdir().unwrap();
    let transcript = decoded(&adapter(db.path()), TORN);
    assert_eq!(transcript.diagnostics.len(), 1);
    assert!(
        transcript.diagnostics[0]
            .error_message
            .contains("still being written")
    );
    assert!(
        index::index(&transcript)
            .unwrap()
            .full_text
            .contains("capybaras")
    );
}

#[test]
fn exec_runs_are_automated_and_subagents_children() {
    let db = tempfile::tempdir().unwrap();
    let adapter = adapter(db.path());

    let exec = decoded(&adapter, EXEC);
    assert_eq!(exec.mode, ExecutionMode::Automated);
    assert_eq!(exec.relationship, Relationship::Root);
    let conversation = index::index(&exec).unwrap();
    assert!(conversation.provenance.automated);
    // A subagent's report reaches its parent as a subagent message.
    assert!(
        conversation
            .agent_search_text
            .contains("iguana check passed")
    );
    // Lexically searchable, but not embedded.
    assert!(conversation.full_text.contains("iguana"));
    assert!(conversation.semantic_turns.is_empty());

    let sub = decoded(&adapter, SUB);
    assert_eq!(
        sub.relationship,
        Relationship::Child {
            parent: PAGED.to_owned()
        }
    );
    assert_eq!(sub.mode, ExecutionMode::Unknown);
    assert!(mentions(&sub, Visibility::Inherited, "walrus"));
    assert!(!mentions(&sub, Visibility::Dialogue, "walrus"));
    assert!(mentions(&sub, Visibility::Dialogue, "armadillo"));
    // The task from the parent is the subagent's prompt.
    assert!(mentions(
        &sub,
        Visibility::Dialogue,
        "Task name: /root/armadillo_search"
    ));
    let conversation = index::index(&sub).unwrap();
    assert!(conversation.provenance.child && conversation.provenance.hidden_by_default());
    assert!(conversation.semantic_turns.is_empty());
    // The parent wrote the subagent's prompt.
    assert_eq!(conversation.own_text.as_ref(), "");

    let archived = adapter.resolve(ARCH, far_future()).unwrap();
    assert!(adapter.thread_of(&archived).unwrap().archived);
    assert!(mentions(
        &decoded(&adapter, ARCH),
        Visibility::Dialogue,
        "tapir"
    ));
    assert_eq!(decoded(&adapter, ORPHAN).head, HeadStatus::Orphaned);
}

#[test]
fn a_busy_state_database_degrades_discovery_without_losing_anything() {
    let db = tempfile::tempdir().unwrap();
    let adapter = adapter(db.path());
    let writer = rusqlite::Connection::open(db.path().join("state_5.sqlite")).unwrap();
    writer
        .execute_batch("PRAGMA locking_mode = EXCLUSIVE; BEGIN EXCLUSIVE;")
        .unwrap();
    let discovery = adapter.discover(far_future());
    assert!(discovery.found.is_empty());
    assert_eq!(discovery.completeness, Completeness::Degraded);
    assert!(adapter.resolve(PAGED, far_future()).is_err());
    writer.execute_batch("COMMIT;").unwrap();
}

#[test]
fn without_a_state_database_heads_come_from_the_rollouts() {
    let db = tempfile::tempdir().unwrap();
    let adapter = CodexAdapter::with_sqlite_home(fixture_home(), db.path().to_path_buf());
    let discovery = adapter.discover(far_future());
    assert_eq!(discovery.found.len(), 9);
    let revert = adapter.resolve(REVERT, far_future()).unwrap();
    assert!(revert.path.to_string_lossy().contains(REVERT2));
    let paged = decoded(&adapter, PAGED);
    assert_eq!(paged.head, HeadStatus::Inferred);
    assert_eq!(paged.meta.custom_title.as_deref(), Some("Index name"));
    assert_eq!(decoded(&adapter, ORPHAN).head, HeadStatus::Inferred);
}

#[test]
fn registries_key_rollouts_by_thread_id() {
    let home = fixture_home();
    let registry = Registry::with_roots(None, None, None).with_codex(Some(home.clone()));
    let path = discover::list(&home)
        .unwrap()
        .into_iter()
        .find(|rollout| rollout.rollout_id == REVERT2)
        .unwrap()
        .path;
    let key = registry.key_of(&path).unwrap();
    assert_eq!(key.instance.kind, SourceKind::Codex);
    assert_eq!(key.native_id, REVERT);
    assert_eq!(registry.key_of(&home.join("session_index.jsonl")), None);
}

#[test]
fn actions_run_the_codex_cli_on_the_rollouts_home() {
    let home = fixture_home();
    let adapter = CodexAdapter::with_sqlite_home(PathBuf::new(), PathBuf::new());
    let path = home.join(format!(
        "sessions/2026/09/01/rollout-2026-09-01T10-00-00-{PAGED}.jsonl"
    ));
    let project = tempfile::tempdir().unwrap();
    let target = Target {
        path: &path,
        project: Some(project.path()),
    };
    let args = |command: &Command| -> Vec<String> {
        command
            .get_args()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect()
    };
    let resume = adapter
        .resume(target, false, &["--yolo".to_owned()])
        .unwrap();
    assert!(resume.copy.is_none());
    assert_eq!(resume.command.get_program(), "codex");
    assert_eq!(args(&resume.command), ["--yolo", "resume", PAGED]);
    assert_eq!(resume.command.get_current_dir(), Some(project.path()));
    assert_eq!(
        resume.command.get_envs().collect::<Vec<_>>(),
        [(std::ffi::OsStr::new("CODEX_HOME"), Some(home.as_os_str()))]
    );
    let fork = adapter.resume(target, true, &[]).unwrap();
    assert_eq!(args(&fork.command), ["fork", PAGED]);
    assert!(!adapter.capabilities().rename);
    assert!(adapter.rename(&path, "x").is_err());
}

#[test]
fn viewed_threads_show_the_dialogue_and_where_a_fork_begins() {
    use crate::history::view::{self, Entry, EntryKind};
    fn text_of(entry: &Entry<'_>) -> String {
        let message = match entry.kind {
            EntryKind::Message { message, .. } | EntryKind::Subagent { message, .. } => message,
            EntryKind::Note { label, .. } => return format!("[{label}]"),
            EntryKind::Hidden => return String::new(),
        };
        message
            .parts
            .iter()
            .filter_map(|part| match part {
                Part::Text(text) => Some(text.clone()),
                _ => None,
            })
            .collect()
    }
    let dir = tempfile::tempdir().unwrap();
    let home = home_copy(dir.path());
    let path_of = |thread: &str| {
        discover::list(&home)
            .unwrap()
            .into_iter()
            .find(|rollout| rollout.thread_id == thread)
            .unwrap()
            .path
    };

    let fork = super::view::decode_path(&path_of(FORK)).unwrap().unwrap();
    let fork = view::entries(&fork);
    let texts: Vec<String> = fork.iter().map(text_of).collect();
    let walrus = texts.iter().filter(|text| text.contains("walrus")).count();
    assert_eq!(
        walrus, 2,
        "the base's question and answer, once each: {texts:?}"
    );
    let boundary = texts.iter().position(|text| text == "[Fork]").unwrap();
    let ocelot = texts
        .iter()
        .position(|text| text.contains("ocelot"))
        .unwrap();
    assert!(boundary < ocelot);
    assert!(
        texts[..boundary]
            .iter()
            .all(|text| !text.contains("ocelot"))
    );
    assert!(fork.iter().any(|entry| matches!(
        entry.kind,
        EntryKind::Message { message, speaker: "Codex" } if message.role == Role::Assistant
    )));

    let sub = super::view::decode_path(&path_of(SUB)).unwrap().unwrap();
    let sub = view::entries(&sub);
    let texts: Vec<String> = sub.iter().map(text_of).collect();
    assert!(texts.iter().any(|text| text.contains("armadillo")));
    assert!(!texts.iter().any(|text| text == "[Fork]"));
    assert!(!texts.iter().any(|text| text.contains("walrus")));

    let claude = dir.path().join("session.jsonl");
    std::fs::write(&claude, "{}\n").unwrap();
    assert!(super::view::decode_path(&claude).is_none());
}

/// The thread of a rollout holding `records` after its `session_meta`, in
/// a Codex home under `dir`.
fn rollout_thread(dir: &Path, meta: &str, records: &[&str]) -> Transcript {
    let id = "01a00000-0000-7000-8000-0000000000aa";
    let day = dir.join("sessions/2026/09/03");
    std::fs::create_dir_all(&day).unwrap();
    let path = day.join(format!("rollout-2026-09-03T10-00-00-{id}.jsonl"));
    let meta = format!(
        r#"{{"timestamp":"2026-09-03T10:00:00.000Z","type":"session_meta","payload":{{"id":"{id}","timestamp":"2026-09-03T10:00:00.000Z","cwd":"/work/demo","source":"cli"{meta}}}}}"#
    );
    let mut contents = std::iter::once(meta.as_str())
        .chain(records.iter().copied())
        .collect::<Vec<_>>()
        .join("\n");
    contents.push('\n');
    std::fs::write(&path, contents).unwrap();
    view::decode_path(&path).unwrap().unwrap()
}

fn record(time: &str, payload: &str) -> String {
    format!(r#"{{"timestamp":"2026-09-03T{time}.000Z","type":"event_msg","payload":{payload}}}"#)
}

#[test]
fn threads_were_last_active_at_their_own_newest_message() {
    let dir = tempfile::tempdir().unwrap();
    let records = [
        record("10:30:00", r#"{"type":"user_message","message":"copied"}"#),
        record("10:30:00", r#"{"type":"agent_message","message":"copied"}"#),
        record("10:00:05", r#"{"type":"user_message","message":"own"}"#),
        record("10:00:10", r#"{"type":"agent_message","message":"own"}"#),
        record("10:05:00", r#"{"type":"token_count","info":null}"#),
    ];
    let records: Vec<&str> = records.iter().map(String::as_str).collect();
    let transcript = rollout_thread(
        dir.path(),
        r#","subagent_history_start_ordinal":3"#,
        &records,
    );
    assert!(mentions(&transcript, Visibility::Inherited, "copied"));
    assert_eq!(
        transcript.meta.updated_at.unwrap().to_rfc3339(),
        "2026-09-03T10:00:10+00:00"
    );

    let inherited_only = rollout_thread(
        dir.path(),
        r#","subagent_history_start_ordinal":3"#,
        &records[..2],
    );
    assert_eq!(
        inherited_only.meta.updated_at.unwrap().to_rfc3339(),
        "2026-09-03T10:30:00+00:00"
    );
}

#[test]
fn threads_without_messages_were_last_active_at_their_last_record() {
    let dir = tempfile::tempdir().unwrap();
    let context = r#"{"timestamp":"2026-09-03T10:01:00.000Z","type":"turn_context","payload":{"cwd":"/work/demo","model":"gpt-5-codex"}}"#;
    let transcript = rollout_thread(dir.path(), "", &[context]);
    assert_eq!(
        transcript.meta.updated_at.unwrap().to_rfc3339(),
        "2026-09-03T10:01:00+00:00"
    );
}
