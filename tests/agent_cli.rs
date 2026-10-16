use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn binary() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_claude-history"))
}

/// The binary with `HOME` at `home`, so its caches and the Codex home it
/// reads stay out of the user's.
fn command(home: &Path) -> Command {
    let mut command = Command::new(binary());
    command
        .env("HOME", home)
        .env_remove("CODEX_HOME")
        .env_remove("CODEX_SQLITE_HOME");
    command
}

fn run(config: &Path, args: &[&str]) -> Output {
    command(&config.join("home"))
        .env("CLAUDE_CONFIG_DIR", config)
        .env(
            "PI_CODING_AGENT_SESSION_DIR",
            config.join("empty-agent-sessions"),
        )
        .args(args)
        .output()
        .expect("run claude-history")
}

fn run_pi(config: &Path, sessions: &Path, args: &[&str]) -> Output {
    command(&config.join("home"))
        .env("CLAUDE_CONFIG_DIR", config)
        .env("PI_CODING_AGENT_SESSION_DIR", sessions)
        .args(args)
        .output()
        .expect("run claude-history with Pi sessions")
}

fn project(config: &Path) -> PathBuf {
    let project = config.join("projects").join("-tmp-agent-phase3-tests");
    std::fs::create_dir_all(&project).expect("create project");
    project
}

fn write_transcript(path: &Path, needle: &str) {
    write_transcript_at(path, needle, "2026-07-20");
}

/// Backdate a transcript's modification time.
///
/// Conversation timestamps come from the file's mtime (see
/// `history::parser`), not from the records inside it, so time filtering can
/// only be exercised by changing the mtime. `stamp` is `YYYYMMDDhhmm`.
fn set_modified(path: &Path, stamp: &str) {
    let status = Command::new("touch")
        .args(["-t", stamp])
        .arg(path)
        .status()
        .expect("run touch");
    assert!(status.success(), "touch -t {stamp} failed");
}

fn write_transcript_at(path: &Path, needle: &str, date: &str) {
    let user = serde_json::json!({
        "type": "user",
        "timestamp": format!("{date}T00:00:00Z"),
        "cwd": "/tmp/agent-phase3-tests",
        "message": {"role": "user", "content": needle}
    });
    let assistant = serde_json::json!({
        "type": "assistant",
        "timestamp": format!("{date}T00:00:01Z"),
        "message": {"role": "assistant", "content": [{"type": "text", "text": "answer"}]}
    });
    std::fs::write(path, format!("{user}\n{assistant}\n")).expect("write transcript");
}

fn first_ref(output: &[u8]) -> String {
    String::from_utf8_lossy(output)
        .split_whitespace()
        .find_map(|field| field.strip_prefix("ref="))
        .expect("search ref")
        .trim_end_matches(|character: char| !character.is_ascii_hexdigit())
        .to_string()
}

#[test]
fn pi_sessions_support_agent_search_read_and_outline_without_claude_storage() {
    let config = tempfile::tempdir().expect("config");
    let sessions = tempfile::tempdir().expect("sessions");
    std::fs::copy(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pi/v3-branched.jsonl"),
        sessions.path().join("pi.jsonl"),
    )
    .expect("copy Pi fixture");

    let search = run_pi(
        config.path(),
        sessions.path(),
        &["agent", "search", "--lexical", "active root question"],
    );
    assert!(
        search.status.success(),
        "{}",
        String::from_utf8_lossy(&search.stderr)
    );
    let search_text = String::from_utf8_lossy(&search.stdout);
    assert!(search_text.contains("uuid=01912345-6789-7abc-8def-0123456789ab"));
    assert!(!search_text.contains("ABANDONED_BRANCH_SENTINEL"));
    let reference = first_ref(&search.stdout);

    let read = run_pi(
        config.path(),
        sessions.path(),
        &["agent", "read", &reference],
    );
    assert!(
        read.status.success(),
        "{}",
        String::from_utf8_lossy(&read.stderr)
    );
    let read_text = String::from_utf8_lossy(&read.stdout);
    assert!(read_text.contains("active root question"));
    assert!(!read_text.contains("compaction summary searchable"));
    assert!(!read_text.contains("branch summary searchable"));
    assert!(!read_text.contains("ABANDONED_BRANCH_SENTINEL"));

    let outline = run_pi(
        config.path(),
        sessions.path(),
        &["agent", "outline", &reference],
    );
    assert!(
        outline.status.success(),
        "{}",
        String::from_utf8_lossy(&outline.stderr)
    );
    assert!(String::from_utf8_lossy(&outline.stdout).contains("active root question"));
}

#[test]
fn omp_sessions_support_agent_search_and_direct_render() {
    let config = tempfile::tempdir().expect("config");
    let sessions = tempfile::tempdir().expect("sessions");
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/omp/v3.jsonl");
    std::fs::copy(&fixture, sessions.path().join("omp.jsonl")).expect("copy OMP fixture");

    let search = run_pi(
        config.path(),
        sessions.path(),
        &["agent", "search", "--lexical", "OMP active question"],
    );
    assert!(
        search.status.success(),
        "{}",
        String::from_utf8_lossy(&search.stderr)
    );
    let search_text = String::from_utf8_lossy(&search.stdout);
    assert!(search_text.contains("uuid=omp_session_custom_id"));
    assert!(!search_text.contains("OMP_ABANDONED_SENTINEL"));
    let reference = first_ref(&search.stdout);

    let read = run_pi(
        config.path(),
        sessions.path(),
        &["agent", "read", &reference],
    );
    assert!(
        read.status.success(),
        "{}",
        String::from_utf8_lossy(&read.stderr)
    );
    assert!(String::from_utf8_lossy(&read.stdout).contains("OMP active answer"));

    let rendered = command(&config.path().join("home"))
        .args(["--no-color", "--render"])
        .arg(fixture)
        .output()
        .expect("render OMP fixture");
    assert!(rendered.status.success());
    let rendered = String::from_utf8_lossy(&rendered.stdout);
    assert!(rendered.contains("OMP"));
    assert!(rendered.contains("OMP active question"));
    assert!(!rendered.contains("OMP_ABANDONED_SENTINEL"));
    assert!(!rendered.contains("Mode change"));
}

#[test]
fn direct_render_supports_pi_active_branch() {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pi/v3-branched.jsonl");
    let home = tempfile::tempdir().expect("home");
    let output = command(home.path())
        .args(["--no-color", "--render"])
        .arg(path)
        .output()
        .expect("render Pi fixture");

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let rendered = String::from_utf8_lossy(&output.stdout);
    assert!(rendered.contains("Pi"));
    assert!(rendered.contains("active root question"));
    for metadata in [
        "Branch summary",
        "Compaction",
        "Thinking level",
        "Model",
        "Label",
    ] {
        assert!(!rendered.contains(metadata));
    }
    assert!(!rendered.contains("ABANDONED_BRANCH_SENTINEL"));
}

#[test]
fn malformed_and_missing_refs_have_structured_stderr_and_nonzero_exit() {
    let config = tempfile::tempdir().expect("config");
    project(config.path());

    let invalid = run(config.path(), &["agent", "read", "not-a-ref"]);
    assert!(!invalid.status.success());
    assert!(invalid.stdout.is_empty());
    assert!(
        String::from_utf8_lossy(&invalid.stderr)
            .starts_with("protocol agent-error kind=invalid-ref ref=not-a-ref")
    );

    let missing = run(config.path(), &["agent", "read", "ch_12345678"]);
    assert!(!missing.status.success());
    assert!(
        String::from_utf8_lossy(&missing.stderr)
            .starts_with("protocol agent-error kind=not-found ref=ch_12345678")
    );
}

#[test]
fn target_transcript_and_range_failures_have_precise_kinds() {
    let config = tempfile::tempdir().expect("config");
    let transcript = project(config.path()).join("12345678-1234-4234-9234-123456789abc.jsonl");
    write_transcript(&transcript, "phase three needle");

    let search = run(
        config.path(),
        &["agent", "search", "--lexical", "phase three needle"],
    );
    assert!(
        search.status.success(),
        "{}",
        String::from_utf8_lossy(&search.stderr)
    );
    let reference = first_ref(&search.stdout);

    let range = run(
        config.path(),
        &["agent", "read", &format!("{reference}:m99")],
    );
    assert!(!range.status.success());
    assert!(String::from_utf8_lossy(&range.stderr).starts_with(&format!(
        "protocol agent-error kind=out-of-range ref={reference}"
    )));

    std::fs::write(&transcript, "{malformed\n").expect("malform transcript");
    let malformed = run(config.path(), &["agent", "read", &reference]);
    assert!(!malformed.status.success());
    assert!(
        String::from_utf8_lossy(&malformed.stderr).starts_with(&format!(
            "protocol agent-error kind=malformed-transcript ref={reference}"
        ))
    );
}

#[test]
fn search_reports_partial_warnings_and_preserves_compact_success_output() {
    let config = tempfile::tempdir().expect("config");
    let project = project(config.path());
    write_transcript(
        &project.join("12345678-1234-4234-9234-123456789abc.jsonl"),
        "warning contract needle",
    );
    std::fs::write(
        project.join("87654321-1234-4234-9234-123456789abc.jsonl"),
        "{malformed\n",
    )
    .expect("write malformed transcript");

    let output = run(
        config.path(),
        &["agent", "search", "--lexical", "warning contract needle"],
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stderr.is_empty());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.starts_with("protocol agent-search mode=lexical"));
    assert!(stdout.contains("protocol agent-warning kind=malformed-transcript ref=ch_"));
    assert!(stdout.contains("read ref=ch_"));
}

#[test]
fn ref_only_commands_parse_only_the_selected_transcript() {
    let config = tempfile::tempdir().expect("config");
    let project = project(config.path());
    let selected = project.join("12345678-1234-4234-9234-123456789abc.jsonl");
    write_transcript(&selected, "selected transcript needle");

    let search = run(
        config.path(),
        &["agent", "search", "--lexical", "selected transcript needle"],
    );
    assert!(search.status.success());
    let reference = first_ref(&search.stdout);
    std::fs::write(
        project.join("87654321-1234-4234-9234-123456789abc.jsonl"),
        "{malformed\n",
    )
    .expect("write unrelated malformed transcript");

    let outline = run(config.path(), &["agent", "outline", &reference]);

    assert!(
        outline.status.success(),
        "{}",
        String::from_utf8_lossy(&outline.stderr)
    );
    assert!(outline.stderr.is_empty());
    let stdout = String::from_utf8_lossy(&outline.stdout);
    assert!(stdout.contains("m1 role=user"));
    assert!(stdout.contains("m2 role=assistant"));
    assert!(!stdout.contains("malformed-transcript"));
}

#[test]
fn selected_partial_transcript_recovers_records_and_reports_warning() {
    let config = tempfile::tempdir().expect("config");
    let transcript = project(config.path()).join("12345678-1234-4234-9234-123456789abc.jsonl");
    write_transcript(&transcript, "partial transcript needle");
    let search = run(
        config.path(),
        &["agent", "search", "--lexical", "partial transcript needle"],
    );
    assert!(search.status.success());
    let reference = first_ref(&search.stdout);
    let content = std::fs::read_to_string(&transcript).expect("read transcript");
    let (first, second) = content.split_once('\n').expect("two records");
    std::fs::write(&transcript, format!("{first}\n{{malformed\n{second}"))
        .expect("write partial transcript");

    let recovered_search = run(
        config.path(),
        &["agent", "search", "--lexical", "partial transcript needle"],
    );
    assert!(recovered_search.status.success());
    let search_stdout = String::from_utf8_lossy(&recovered_search.stdout);
    assert!(search_stdout.contains("focus=m1..m1"));
    assert!(search_stdout.contains("kind=malformed-transcript"));

    let within = run(
        config.path(),
        &[
            "agent",
            "within",
            &reference,
            "partial transcript needle",
            "--lexical",
        ],
    );
    assert!(within.status.success());
    assert!(String::from_utf8_lossy(&within.stdout).contains("focus=m1..m1"));

    let read = run(
        config.path(),
        &["agent", "read", &format!("{reference}:m1")],
    );
    assert!(read.status.success());
    assert!(String::from_utf8_lossy(&read.stdout).contains("partial transcript needle"));

    let outline = run(config.path(), &["agent", "outline", &reference]);

    assert!(outline.status.success());
    let stdout = String::from_utf8_lossy(&outline.stdout);
    assert!(stdout.contains("warnings=1"));
    assert!(stdout.contains("kind=malformed-transcript"));
    assert!(stdout.contains("m1 role=user"));
    assert!(stdout.contains("m2 role=assistant"));

    let bounded = run(
        config.path(),
        &["agent", "read", &reference, "--budget", "180"],
    );
    assert!(bounded.status.success());
    let bounded_stdout = String::from_utf8_lossy(&bounded.stdout);
    assert!(bounded_stdout.chars().count() <= 180);
    assert!(bounded_stdout.contains("warnings=1"));
    assert!(bounded_stdout.contains("continue read"));
    assert_eq!(bounded_stdout.lines().count(), 2);
}

#[test]
fn agent_filesystem_failures_use_io_envelope() {
    let config = tempfile::tempdir().expect("config");
    std::fs::write(config.path().join("projects"), "not a directory").expect("write projects file");

    let output = run(config.path(), &["agent", "search", "--lexical", "needle"]);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).starts_with("protocol agent-error kind=io"));
}

#[test]
fn search_time_range_narrows_the_corpus_without_reporting_skips() {
    let config = tempfile::tempdir().expect("config");
    let project = project(config.path());

    let recent = project.join("11111111-1111-4111-9111-111111111111.jsonl");
    let old = project.join("22222222-2222-4222-9222-222222222222.jsonl");
    write_transcript_at(&recent, "time filter needle", "2026-07-20");
    write_transcript_at(&old, "time filter needle", "2020-01-15");
    set_modified(&recent, "202607200000");
    set_modified(&old, "202001150000");

    let unfiltered = run(
        config.path(),
        &["agent", "search", "--lexical", "time filter needle"],
    );
    assert!(
        unfiltered.status.success(),
        "{}",
        String::from_utf8_lossy(&unfiltered.stderr)
    );
    let unfiltered_stdout = String::from_utf8_lossy(&unfiltered.stdout);
    assert!(unfiltered_stdout.contains("uuid=11111111"));
    assert!(unfiltered_stdout.contains("uuid=22222222"));

    let filtered = run(
        config.path(),
        &[
            "agent",
            "search",
            "--lexical",
            "time filter needle",
            "--since",
            "2026-01-01",
        ],
    );
    assert!(
        filtered.status.success(),
        "{}",
        String::from_utf8_lossy(&filtered.stderr)
    );
    let filtered_stdout = String::from_utf8_lossy(&filtered.stdout);
    assert!(filtered_stdout.contains("uuid=11111111"));
    assert!(
        !filtered_stdout.contains("uuid=22222222"),
        "out-of-window conversation still returned: {filtered_stdout}"
    );

    // Key discovery walks the projects directory independently of the time
    // filter, so an unfiltered key list would report every excluded
    // conversation as a skipped transcript and claim partial coverage.
    assert!(
        !filtered_stdout.contains("kind=skipped"),
        "filtered-out conversations were reported as skipped: {filtered_stdout}"
    );

    // The converse: narrowing the key list must not hide diagnostics for files
    // that are inside the window but failed to parse, or a filtered search would
    // claim full coverage it does not have.
    let unparseable = project.join("33333333-3333-4333-9333-333333333333.jsonl");
    std::fs::write(&unparseable, "{malformed\n").expect("write malformed transcript");
    set_modified(&unparseable, "202607200000");

    let with_malformed = run(
        config.path(),
        &[
            "agent",
            "search",
            "--lexical",
            "time filter needle",
            "--since",
            "2026-01-01",
        ],
    );
    assert!(with_malformed.status.success());
    let with_malformed_stdout = String::from_utf8_lossy(&with_malformed.stdout);
    assert!(
        with_malformed_stdout.contains("kind=malformed-transcript"),
        "in-window malformed transcript was silently dropped: {with_malformed_stdout}"
    );
}

#[test]
fn search_rejects_an_inverted_time_range() {
    let config = tempfile::tempdir().expect("config");
    let transcript = project(config.path()).join("33333333-3333-4333-9333-333333333333.jsonl");
    write_transcript(&transcript, "inverted range needle");

    let output = run(
        config.path(),
        &[
            "agent",
            "search",
            "--lexical",
            "inverted range needle",
            "--after",
            "2026-07-20",
            "--before",
            "2026-01-01",
        ],
    );

    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr)
            .starts_with("protocol agent-error kind=out-of-range"),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn direct_uuid_and_pi_filename_inputs_preserve_agent_recipes() {
    let config = tempfile::tempdir().unwrap();
    let sessions = tempfile::tempdir().unwrap();
    let uuid = "01912345-6789-7abc-8def-0123456789ab";
    let stem = format!("2026-09-08T20-20-22-361Z_{uuid}");
    std::fs::copy(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pi/v3-branched.jsonl"),
        sessions.path().join(format!("{stem}.jsonl")),
    )
    .unwrap();
    std::fs::write(
        config.path().join("settings.json"),
        serde_json::json!({"sessionDir": sessions.path()}).to_string(),
    )
    .unwrap();
    let invoke = |args: &[&str]| {
        let output = command(&config.path().join("home"))
            .env("CLAUDE_CONFIG_DIR", config.path())
            .env("PI_CODING_AGENT_DIR", config.path())
            .env_remove("PI_CODING_AGENT_SESSION_DIR")
            .env_remove("OMP_PROFILE")
            .env_remove("PI_PROFILE")
            .current_dir(config.path())
            .args(args)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8(output.stdout).unwrap()
    };
    for identity in [uuid.to_owned(), stem.clone(), format!("{stem}.jsonl")] {
        let outline = invoke(&["agent", "outline", &identity]);
        assert!(outline.contains("active root question"));
        let read = invoke(&[
            "agent",
            "read",
            &format!("{identity}:m1..m2"),
            "--focus",
            &format!("{uuid}:m1"),
        ]);
        assert!(read.contains("active root question"));
        assert!(!read.contains("ABANDONED_BRANCH_SENTINEL"));
        let within = invoke(&[
            "agent",
            "within",
            &identity,
            "active root question",
            "--exact",
        ]);
        let handle = first_ref(within.as_bytes());
        let opaque = invoke(&["agent", "outline", &handle]);
        assert_eq!(outline, opaque);
        let anchor = within
            .split_whitespace()
            .find_map(|field| field.strip_prefix("anchors="))
            .unwrap()
            .split(',')
            .next()
            .unwrap();
        let anchored = invoke(&["agent", "read", &identity, "--anchor", anchor]);
        assert!(anchored.contains("active root question"));
    }
}

#[test]
fn duplicate_uuid_cli_errors_offer_resolvable_handles() {
    let config = tempfile::tempdir().unwrap();
    let uuid = "12345678-1234-4234-9234-123456789abc";
    for name in ["-tmp-first", "-tmp-second"] {
        let dir = config.path().join("projects").join(name);
        std::fs::create_dir_all(&dir).unwrap();
        write_transcript(&dir.join(format!("{uuid}.jsonl")), name);
    }
    for args in [
        vec!["agent", "outline", uuid],
        vec!["agent", "read", uuid],
        vec!["agent", "within", uuid, "answer"],
    ] {
        let output = run(config.path(), &args);
        assert!(!output.status.success());
        let error = String::from_utf8_lossy(&output.stderr);
        assert!(error.contains("kind=ambiguous-ref"));
        assert!(error.contains("retry%20with"));
        assert!(error.contains("-tmp-first"));
        assert!(error.contains("-tmp-second"));
    }
    let missing = run(
        config.path(),
        &["agent", "outline", "00000000-0000-0000-0000-000000000000"],
    );
    assert!(!missing.status.success());
    assert!(String::from_utf8_lossy(&missing.stderr).contains("kind=not-found"));
}

#[test]
fn uuid_qualified_focus_matches_handle_focus_across_conversations() {
    let config = tempfile::tempdir().unwrap();
    let dir = project(config.path());
    let first = "12345678-1234-4234-9234-123456789abc";
    let second = "87654321-1234-4234-9234-123456789abc";
    for uuid in [first, second] {
        write_transcript(
            &dir.join(format!("{uuid}.jsonl")),
            &"bounded evidence ".repeat(100),
        );
    }
    let outline = run(config.path(), &["agent", "outline", second]);
    assert!(outline.status.success());
    let handle = first_ref(&outline.stdout);
    let mut results = Vec::new();
    for focus in [format!("{second}:m1"), format!("{handle}:m1")] {
        let output = run(
            config.path(),
            &[
                "agent", "read", first, second, "--focus", &focus, "--budget", "1200",
            ],
        );
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        results.push(output.stdout);
    }
    assert_eq!(results[0], results[1]);
}

#[test]
fn lexical_and_exact_search_omit_semantic_breakdowns() {
    let config = tempfile::tempdir().expect("config");
    write_transcript(
        &project(config.path()).join("12345678-1234-4234-9234-123456789abc.jsonl"),
        "cache warming",
    );
    for mode in ["lexical", "exact"] {
        let output = run(
            config.path(),
            &["agent", "search", "--mode", mode, "cache warming"],
        );
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let text = String::from_utf8_lossy(&output.stdout);
        assert!(text.contains("hit project="));
        for atom in [" hybrid=", " semantic=", " lexical="] {
            assert!(!text.contains(atom), "{text}");
        }
    }
}
