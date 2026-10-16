//! What the tool that wrote a conversation can do with it, as its adapter
//! plans it: resume or fork it in that tool, delete it, rename it.

use crate::error::{AppError, Result};
use crate::history::Provenance;
use crate::history::model::SourceKind;
use crate::history::sources::Registry;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;

/// The actions an adapter supports for its conversations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Capabilities {
    pub resume: bool,
    pub fork: bool,
    pub delete: bool,
    pub rename: bool,
}

impl Capabilities {
    pub const ALL: Self = Self {
        resume: true,
        fork: true,
        delete: true,
        rename: true,
    };

    pub const NONE: Self = Self {
        resume: false,
        fork: false,
        delete: false,
        rename: false,
    };
}

/// The actions the adapter of `source` supports for a conversation of
/// `provenance`: none for an orphaned one, which its tool no longer has.
pub fn capabilities_of(source: SourceKind, provenance: Provenance) -> Capabilities {
    if provenance.orphaned {
        return Capabilities::NONE;
    }
    // Capabilities belong to the kind of source, not to an instance.
    Registry::with_roots(None, None, None)
        .for_source(source)
        .capabilities()
}

/// Why `doing` ("Deleting", ...) is unavailable for a conversation of
/// `source` and `provenance`.
pub fn refusal(source: SourceKind, provenance: Provenance, doing: &str) -> String {
    if provenance.orphaned {
        format!(
            "{doing} is unavailable: {} no longer has this conversation's current history",
            source.list_label()
        )
    } else {
        format!("{doing} {} sessions is not supported", source.list_label())
    }
}

/// A conversation to act on.
#[derive(Clone, Copy, Debug)]
pub struct Target<'a> {
    /// The conversation's own artifact.
    pub path: &'a Path,
    /// The project directory the conversation is listed under.
    pub project: Option<&'a Path>,
}

/// The program an agent is started with, and the arguments that precede the
/// agent's own: the agent's binary by default, or the user's wrapper.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Launcher {
    pub program: PathBuf,
    pub args: Vec<String>,
    /// Where the user configured it, for errors; `None` for the default.
    pub configured_at: Option<&'static str>,
}

impl Launcher {
    /// The agent's own binary, found on `PATH`.
    pub fn default_for(source: SourceKind) -> Self {
        Self {
            program: PathBuf::from(source.label()),
            args: Vec::new(),
            configured_at: None,
        }
    }

    /// The configured `command`, its first element the program: a name
    /// found on `PATH`, or a path. `~/` starts a path in the home
    /// directory, and a relative path is relative to the current directory.
    pub fn configured(command: &[String], configured_at: &'static str) -> Self {
        let (program, args) = command
            .split_first()
            .expect("configured commands are not empty");
        Self {
            program: absolute_program(program),
            args: args.to_vec(),
            configured_at: Some(configured_at),
        }
    }

    /// The program with its leading arguments, ready for the agent's own.
    pub fn command(&self) -> Command {
        let mut command = Command::new(&self.program);
        command.args(&self.args);
        command
    }

    /// Fails, naming the program and where it was configured, when it does
    /// not exist or is not executable.
    pub fn check(&self, source: SourceKind) -> Result<()> {
        if find_program(&self.program, std::env::var_os("PATH")).is_some() {
            return Ok(());
        }
        let program = self.program.display();
        let found = if has_separator(&self.program) {
            "is not an executable file"
        } else {
            "was not found on PATH"
        };
        let hint = match self.configured_at {
            Some(key) => format!("check `command` under {key} in the config file"),
            None => format!(
                "if you start {} through a wrapper, set `command` under {} in the config file",
                source.assistant_name(),
                config_table(source)
            ),
        };
        Err(AppError::AgentLaunch(format!(
            "`{program}` {found}; {hint}"
        )))
    }
}

/// The config table whose `command` launches `source`.
pub fn config_table(source: SourceKind) -> &'static str {
    match source {
        SourceKind::Claude => "[resume]",
        SourceKind::Pi => "[resume.pi]",
        SourceKind::Omp => "[resume.omp]",
        SourceKind::Codex => "[resume.codex]",
    }
}

/// How to start the agent of a conversation: its launcher, and the user's
/// extra arguments for agents that take them.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Launch {
    pub launcher: Launcher,
    pub default_args: Vec<String>,
}

fn has_separator(program: &Path) -> bool {
    program.components().count() > 1
}

fn absolute_program(program: &str) -> PathBuf {
    let path = match program.strip_prefix("~/") {
        Some(rest) => match home::home_dir() {
            Some(home) => home.join(rest),
            None => PathBuf::from(program),
        },
        None => PathBuf::from(program),
    };
    if path.is_relative() && has_separator(&path) {
        // The agent runs in the conversation's project, not here.
        if let Ok(cwd) = std::env::current_dir() {
            return cwd.join(path);
        }
    }
    path
}

/// The executable `program` names: itself when it is a path, else the
/// first match in the directories of `path_var`.
fn find_program(program: &Path, path_var: Option<OsString>) -> Option<PathBuf> {
    if has_separator(program) {
        return is_executable(program).then(|| program.to_path_buf());
    }
    let path_var = path_var?;
    std::env::split_paths(&path_var)
        .filter(|dir| !dir.as_os_str().is_empty())
        .flat_map(|dir| candidates(&dir.join(program)))
        .find(|candidate| is_executable(candidate))
}

#[cfg(unix)]
fn candidates(path: &Path) -> Vec<PathBuf> {
    vec![path.to_path_buf()]
}

/// `path`, and `path` with each extension of `PATHEXT`.
#[cfg(not(unix))]
fn candidates(path: &Path) -> Vec<PathBuf> {
    let mut candidates = vec![path.to_path_buf()];
    let extensions = std::env::var("PATHEXT").unwrap_or_else(|_| ".COM;.EXE;.BAT;.CMD".into());
    for extension in extensions.split(';').filter(|ext| !ext.is_empty()) {
        let mut candidate = path.as_os_str().to_owned();
        candidate.push(extension);
        candidates.push(PathBuf::from(candidate));
    }
    candidates
}

#[cfg(unix)]
fn is_executable(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    std::fs::metadata(path)
        .is_ok_and(|metadata| metadata.is_file() && metadata.permissions().mode() & 0o111 != 0)
}

#[cfg(not(unix))]
fn is_executable(path: &Path) -> bool {
    path.is_file()
}

/// How to resume a conversation: copy its session into place, if needed,
/// then run `command`.
#[derive(Debug)]
pub struct Resume {
    pub copy: Option<SessionCopy>,
    pub command: Command,
}

/// Copies a Claude session (its transcript and its session directory of
/// tool results and subagents) into another project directory.
#[derive(Debug, Eq, PartialEq)]
pub struct SessionCopy {
    pub transcript: PathBuf,
    pub session_id: String,
    pub from_dir: PathBuf,
    pub to_dir: PathBuf,
}

impl SessionCopy {
    pub fn apply(&self) -> Result<()> {
        std::fs::create_dir_all(&self.to_dir).map_err(AppError::Io)?;
        let target = self.to_dir.join(self.transcript.file_name().unwrap());
        std::fs::copy(&self.transcript, &target).map_err(AppError::Io)?;

        let session_dir = self.from_dir.join(&self.session_id);
        if session_dir.is_dir() {
            copy_dir_recursive(&session_dir, &self.to_dir.join(&self.session_id))?;
        }

        // File history (~/.claude/file-history/<uuid>/) is global, not
        // per project: Claude Code finds it by session id.
        Ok(())
    }
}

fn copy_dir_recursive(src: &Path, dst: &Path) -> Result<()> {
    std::fs::create_dir_all(dst).map_err(AppError::Io)?;
    for entry in std::fs::read_dir(src).map_err(AppError::Io)? {
        let entry = entry.map_err(AppError::Io)?;
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());
        if src_path.is_dir() {
            copy_dir_recursive(&src_path, &dst_path)?;
        } else {
            std::fs::copy(&src_path, &dst_path).map_err(AppError::Io)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn session_copies_bring_the_session_directory() {
        let dir = tempfile::tempdir().unwrap();
        let from = dir.path().join("from");
        fs::create_dir_all(from.join("s/subagents")).unwrap();
        fs::write(from.join("s.jsonl"), "{}\n").unwrap();
        fs::write(from.join("s/subagents/agent-1.jsonl"), "{}\n").unwrap();
        let to = dir.path().join("to");

        SessionCopy {
            transcript: from.join("s.jsonl"),
            session_id: "s".to_owned(),
            from_dir: from,
            to_dir: to.clone(),
        }
        .apply()
        .unwrap();

        assert!(to.join("s.jsonl").is_file());
        assert!(to.join("s/subagents/agent-1.jsonl").is_file());
    }

    fn executable(path: &Path) {
        fs::write(path, "#!/bin/sh\n").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(path, fs::Permissions::from_mode(0o755)).unwrap();
        }
    }

    #[test]
    fn launchers_default_to_the_agent_binary() {
        for (source, program) in [
            (SourceKind::Claude, "claude"),
            (SourceKind::Pi, "pi"),
            (SourceKind::Omp, "omp"),
            (SourceKind::Codex, "codex"),
        ] {
            let command = Launcher::default_for(source).command();
            assert_eq!(command.get_program(), program);
            assert_eq!(command.get_args().count(), 0);
        }
    }

    #[test]
    fn configured_launchers_lead_with_their_arguments() {
        let command = ["mise", "exec", "--", "claude"].map(String::from);
        let launcher = Launcher::configured(&command, "[resume]");
        let mut built = launcher.command();
        built.arg("--resume");
        assert_eq!(built.get_program(), "mise");
        assert_eq!(
            built.get_args().collect::<Vec<_>>(),
            ["exec", "--", "claude", "--resume"]
        );
    }

    #[test]
    fn configured_programs_resolve_home_and_relative_paths() {
        let home = home::home_dir().unwrap();
        let launcher = Launcher::configured(&["~/bin/claude".to_owned()], "[resume]");
        assert_eq!(launcher.program, home.join("bin/claude"));
        let launcher = Launcher::configured(&["bin/claude".to_owned()], "[resume]");
        assert_eq!(
            launcher.program,
            std::env::current_dir().unwrap().join("bin/claude")
        );
        let launcher = Launcher::configured(&["my-claude".to_owned()], "[resume]");
        assert_eq!(launcher.program, Path::new("my-claude"));
    }

    #[test]
    fn programs_are_found_on_path_or_by_path() {
        let dir = tempfile::tempdir().unwrap();
        let bin = dir.path().join("bin");
        fs::create_dir(&bin).unwrap();
        executable(&bin.join("my-claude"));
        fs::write(bin.join("data"), "").unwrap();
        let path_var = Some(std::env::join_paths([dir.path(), &bin]).unwrap());

        assert_eq!(
            find_program(Path::new("my-claude"), path_var.clone()),
            Some(bin.join("my-claude"))
        );
        assert_eq!(find_program(Path::new("missing"), path_var.clone()), None);
        assert_eq!(
            find_program(&bin.join("my-claude"), None),
            Some(bin.join("my-claude"))
        );
        assert_eq!(find_program(&bin.join("missing"), path_var.clone()), None);
        #[cfg(unix)]
        assert_eq!(find_program(Path::new("data"), path_var), None);
    }

    #[test]
    fn missing_programs_name_their_config_key() {
        let missing = Launcher::configured(
            &["/nonexistent/claude-history-test/agent".to_owned()],
            "[resume.pi]",
        );
        let message = missing.check(SourceKind::Pi).unwrap_err().to_string();
        assert!(
            message.contains("/nonexistent/claude-history-test/agent"),
            "{message}"
        );
        assert!(message.contains("is not an executable file"), "{message}");
        assert!(message.contains("[resume.pi]"), "{message}");

        let default = Launcher {
            program: PathBuf::from("claude-history-test-missing-agent"),
            ..Launcher::default_for(SourceKind::Codex)
        };
        let message = default.check(SourceKind::Codex).unwrap_err().to_string();
        assert!(message.contains("was not found on PATH"), "{message}");
        assert!(message.contains("[resume.codex]"), "{message}");
    }

    #[test]
    fn every_source_has_an_acting_adapter() {
        let registry = Registry::with_roots(None, None, None);
        for source in [
            SourceKind::Claude,
            SourceKind::Pi,
            SourceKind::Omp,
            SourceKind::Codex,
        ] {
            let adapter = registry.for_source(source);
            assert_eq!(adapter.instance().kind, source);
            let capabilities = adapter.capabilities();
            assert_eq!(capabilities.rename, source != SourceKind::Codex);
            assert!(capabilities.resume && capabilities.fork && capabilities.delete);
        }
    }
}
