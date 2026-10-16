//! Pi and OMP: one JSONL session file per conversation, in per-project
//! directories of a session root or directly in a flat one.

use crate::error::Result;
use crate::history::ParseError;
use crate::history::actions::{Capabilities, Launch, Launcher, Resume, Target};
use crate::history::adapter::{Adapter, Completeness, Discovery, Located};
use crate::history::model::{
    ConversationKey, Event, EventKind, ExecutionMode, HeadStatus, Relationship, Revision, Role,
    Segment, SessionMeta, SourceInstance, SourceKind, Transcript,
};
use crate::history::omp_loader::{self, OmpSessionRoot};
use crate::history::pi::PiSession;
use crate::history::pi_loader::{self, PiSessionRoot};
use chrono::{DateTime, Utc};
use std::path::Path;
use std::process::Command;
use std::time::SystemTime;

/// The adapter of a Pi or an OMP session root.
pub struct PiAdapter {
    instance: SourceInstance,
    flat: bool,
}

impl PiAdapter {
    pub fn pi(root: PiSessionRoot) -> Self {
        Self::new(SourceKind::Pi, root.path, root.flat)
    }

    pub fn omp(root: OmpSessionRoot) -> Self {
        Self::new(SourceKind::Omp, root.path, root.flat)
    }

    fn new(kind: SourceKind, root: std::path::PathBuf, flat: bool) -> Self {
        Self {
            instance: SourceInstance { kind, root },
            flat,
        }
    }
}

impl Adapter for PiAdapter {
    fn instance(&self) -> &SourceInstance {
        &self.instance
    }

    fn discover(&self, scan_started: SystemTime) -> Discovery {
        let root = self.instance.root.clone();
        let files = match self.instance.kind {
            SourceKind::Omp => omp_loader::discover_files(&OmpSessionRoot {
                path: root,
                flat: self.flat,
            }),
            _ => pi_loader::discover_files(&PiSessionRoot {
                path: root,
                flat: self.flat,
            }),
        };
        match files {
            Ok(files) => Discovery {
                found: super::locate_files(&self.instance, files, scan_started),
                completeness: Completeness::Complete,
            },
            Err(_) => Discovery {
                found: Vec::new(),
                completeness: Completeness::Degraded,
            },
        }
    }

    fn resolve(&self, native_id: &str, scan_started: SystemTime) -> Result<Located> {
        super::resolve_file(&self.instance, native_id, scan_started)
    }

    fn decode_into(
        &self,
        located: &Located,
        events: &mut dyn FnMut(Event),
    ) -> Result<Option<Transcript>> {
        let omp_owned = self.instance.kind == SourceKind::Omp && omp_owned(&self.instance.root);
        super::decode_file(located, omp_owned, events)
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities::ALL
    }

    fn resume(&self, target: Target<'_>, fork: bool, launch: &Launch) -> Result<Resume> {
        let launcher = &launch.launcher;
        let command = match self.instance.kind {
            SourceKind::Omp => {
                build_omp_resume_command(launcher, target.path, target.project, fork)
            }
            _ => build_pi_resume_command(launcher, target.path, target.project, fork),
        };
        Ok(Resume {
            copy: None,
            command,
        })
    }

    fn delete(&self, path: &Path, _launcher: &Launcher) -> Result<()> {
        match self.instance.kind {
            SourceKind::Omp => omp_loader::delete_session(path),
            _ => pi_loader::delete_session(path),
        }
    }

    fn rename(&self, path: &Path, title: &str) -> Result<()> {
        match self.instance.kind {
            SourceKind::Omp => crate::history::pi::append_omp_session_rename(path, title),
            _ => crate::history::pi::append_session_rename(path, title),
        }
    }
}

/// Pi resumes a session file in place, or forks it into a new one.
fn build_pi_resume_command(
    launcher: &Launcher,
    selected_path: &Path,
    project_path: Option<&Path>,
    fork_session: bool,
) -> Command {
    let mut command = launcher.command();
    command.arg(if fork_session { "--fork" } else { "--session" });
    command.arg(selected_path);
    if !fork_session && let Some(project_path) = project_path {
        command.current_dir(project_path);
    }
    command
}

fn build_omp_resume_command(
    launcher: &Launcher,
    selected_path: &Path,
    project_path: Option<&Path>,
    fork_session: bool,
) -> Command {
    let mut command = launcher.command();
    command.arg(if fork_session { "--fork" } else { "--resume" });
    command.arg(selected_path);
    if !fork_session && let Some(project_path) = project_path {
        command.current_dir(project_path);
    }
    command
}

/// Whether OMP sessions under `root` are OMP's own format rather than Pi's.
pub(crate) fn omp_owned(root: &Path) -> bool {
    root.components()
        .any(|component| matches!(component.as_os_str().to_str(), Some(".omp" | "omp")))
}

/// The transcript of a Pi or OMP session: the events of its active branch,
/// each at the line of the record it was decoded from, passed to `events`.
/// The model is the first assistant model unless a model change names one,
/// the title is the last one set, and the session was last active at its
/// latest message.
pub(super) fn transcript_of(
    key: ConversationKey,
    path: &Path,
    session: PiSession,
    bytes: u64,
    modified: Option<SystemTime>,
    events: &mut dyn FnMut(Event),
) -> Transcript {
    let source = session.source;
    let records = session.events.len();
    let mut model = None;
    let mut custom_title = None;
    let mut activity = super::LastActivity::default();
    for event in session.events {
        activity.observe(&event);
        match &event.kind {
            EventKind::Message(message) => {
                if model.is_none() && message.role == Role::Assistant {
                    model.clone_from(&message.model);
                }
            }
            EventKind::Note { label, text, .. } if label == "Model" && !text.is_empty() => {
                model = Some(text.clone());
            }
            EventKind::Title { text, .. } => {
                let trimmed = text.trim();
                custom_title = (!trimmed.is_empty()).then(|| trimmed.to_owned());
            }
            _ => {}
        }
        events(event);
    }
    let updated_at = activity
        .at()
        .or_else(|| super::claude::parse_timestamp(&session.header.timestamp))
        .or_else(|| modified.map(DateTime::<Utc>::from));
    let diagnostics = session
        .malformed_lines
        .into_iter()
        .map(|line_number| ParseError {
            line_number,
            line_content: String::new(),
            error_message: format!("malformed {} JSONL record", source.label()),
            context_before: Vec::new(),
            context_after: Vec::new(),
        })
        .collect();
    let segments = vec![Segment {
        artifact: path.to_path_buf(),
        start: 0,
        end: bytes,
        inherited: false,
    }];
    let cwd = session.header.cwd;
    Transcript {
        revision: Revision::of(source, &segments, &[super::claude::mtime_ns(modified)]),
        key,
        source,
        head: HeadStatus::Current,
        mode: ExecutionMode::Unknown,
        relationship: Relationship::Root,
        meta: SessionMeta {
            session_id: session.header.id,
            cwd: Some(cwd.clone()),
            project: Some(cwd),
            model,
            summary: None,
            custom_title,
            updated_at,
        },
        segments,
        // The session header is a record too.
        records: records + 1,
        events: Vec::new(),
        diagnostics,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn transcript_with_mtime(lines: &[&str], modified: SystemTime) -> Transcript {
        let contents = lines.join("\n");
        let session = crate::history::pi::parse_reader(contents.as_bytes(), None)
            .unwrap()
            .unwrap();
        let path = Path::new("/sessions/session.jsonl");
        let key = ConversationKey {
            instance: SourceInstance {
                kind: SourceKind::Pi,
                root: PathBuf::new(),
            },
            native_id: path.to_string_lossy().into_owned(),
        };
        transcript_of(key, path, session, 0, Some(modified), &mut |_| {})
    }

    const HEADER: &str = r#"{"type":"session","version":3,"id":"s","timestamp":"2026-10-15T14:00:00Z","cwd":"/tmp"}"#;

    #[test]
    fn last_activity_is_the_newest_message_not_the_file_mtime() {
        let transcript = transcript_with_mtime(
            &[
                HEADER,
                r#"{"type":"message","id":"u","parentId":null,"timestamp":"2026-10-15T14:50:00Z","message":{"role":"user","content":"hi"}}"#,
                r#"{"type":"message","id":"a","parentId":"u","timestamp":"2026-10-15T14:51:56Z","message":{"role":"assistant","content":[{"type":"text","text":"hello"}]}}"#,
                r#"{"type":"session_info","id":"n","parentId":"a","timestamp":"2026-10-16T14:44:00Z","name":"title"}"#,
            ],
            SystemTime::now(),
        );
        assert_eq!(
            transcript.meta.updated_at.unwrap().to_rfc3339(),
            "2026-10-15T14:51:56+00:00"
        );
    }

    #[test]
    fn a_session_without_message_times_was_last_active_at_its_header() {
        let transcript = transcript_with_mtime(
            &[
                HEADER,
                r#"{"type":"message","id":"u","parentId":null,"message":{"role":"user","content":"hi"}}"#,
            ],
            SystemTime::now(),
        );
        assert_eq!(
            transcript.meta.updated_at.unwrap().to_rfc3339(),
            "2026-10-15T14:00:00+00:00"
        );
    }

    #[test]
    fn pi_resume_and_fork_commands_use_native_paths_without_claude_defaults() {
        let path = PathBuf::from("/tmp/pi sessions/session.jsonl");
        let cwd = PathBuf::from("/tmp/project");
        let pi = Launcher::default_for(SourceKind::Pi);
        let resume = build_pi_resume_command(&pi, &path, Some(cwd.as_path()), false);
        assert_eq!(resume.get_program(), std::ffi::OsStr::new("pi"));
        assert_eq!(
            resume.get_args().collect::<Vec<_>>(),
            vec![std::ffi::OsStr::new("--session"), path.as_os_str()]
        );
        assert_eq!(resume.get_current_dir(), Some(cwd.as_path()));

        let fork = build_pi_resume_command(&pi, &path, Some(cwd.as_path()), true);
        assert_eq!(
            fork.get_args().collect::<Vec<_>>(),
            vec![std::ffi::OsStr::new("--fork"), path.as_os_str()]
        );
        assert_eq!(fork.get_current_dir(), None);
    }

    #[test]
    fn omp_resume_and_fork_commands_use_native_paths() {
        let path = PathBuf::from("/tmp/omp sessions/session.jsonl");
        let cwd = PathBuf::from("/tmp/project");
        let omp = Launcher::default_for(SourceKind::Omp);
        let resume = build_omp_resume_command(&omp, &path, Some(cwd.as_path()), false);
        assert_eq!(resume.get_program(), std::ffi::OsStr::new("omp"));
        assert_eq!(
            resume.get_args().collect::<Vec<_>>(),
            vec![std::ffi::OsStr::new("--resume"), path.as_os_str()]
        );
        assert_eq!(resume.get_current_dir(), Some(cwd.as_path()));

        let fork = build_omp_resume_command(&omp, &path, Some(cwd.as_path()), true);
        assert_eq!(
            fork.get_args().collect::<Vec<_>>(),
            vec![std::ffi::OsStr::new("--fork"), path.as_os_str()]
        );
        assert_eq!(fork.get_current_dir(), None);
    }
}
