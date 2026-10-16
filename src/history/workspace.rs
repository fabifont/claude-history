//! The workspace scope: which conversations belong to the directory
//! claude-history runs in. The TUI's project scope, `--local`, local agent
//! search and `--debug-search --local` all decide membership here.

use std::path::{Path, PathBuf};

use super::Conversation;
use super::model::SourceKind;
use super::path::{convert_path_to_project_dir_name, is_same_project, path_is_within, tilde_path};

/// The directory the workspace scope is anchored to.
#[derive(Clone, Debug)]
pub struct Workspace {
    dir: PathBuf,
    /// `dir` with symlinks resolved, which is what non-Claude project keys
    /// are compared against.
    canonical_dir: PathBuf,
    /// Claude's project directory name for `dir`.
    project_dir_name: String,
    /// `dir` with the home directory written as `~`.
    display: String,
}

impl Workspace {
    pub fn new(dir: &Path) -> Self {
        Self {
            dir: dir.to_path_buf(),
            canonical_dir: dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf()),
            project_dir_name: convert_path_to_project_dir_name(dir),
            display: tilde_path(dir, home::home_dir().as_deref()),
        }
    }

    /// The workspace of the process's working directory.
    pub fn current() -> Option<Self> {
        std::env::current_dir().ok().map(|dir| Self::new(&dir))
    }

    /// The workspace directory with the home directory written as `~`.
    pub fn display(&self) -> &str {
        &self.display
    }

    /// Whether a conversation belongs to the workspace: its project is the
    /// workspace (for Claude, stored under the workspace's project directory
    /// or one of its worktrees'), or any cwd it recorded is the workspace or
    /// a directory below it.
    pub fn contains(&self, conversation: &Conversation) -> bool {
        let project_matches = if conversation.source == SourceKind::Claude {
            conversation
                .path
                .parent()
                .and_then(Path::file_name)
                .is_some_and(|name| self.contains_project_key(&name.to_string_lossy(), false))
        } else {
            conversation
                .project_path
                .as_ref()
                .or(conversation.cwd.as_ref())
                .is_some_and(|path| {
                    let path = path.canonicalize().unwrap_or_else(|_| path.clone());
                    self.contains_project_key(&path.to_string_lossy(), true)
                })
        };
        project_matches || conversation.cwds.iter().any(|cwd| self.contains_cwd(cwd))
    }

    /// Whether a project belongs to the workspace, given its workspace key:
    /// a Claude project directory name, or a canonical project path when
    /// `key_is_path`.
    pub fn contains_project_key(&self, key: &str, key_is_path: bool) -> bool {
        if key_is_path {
            path_is_within(Path::new(key), &self.canonical_dir)
        } else {
            is_same_project(key, &self.project_dir_name)
        }
    }

    /// Whether a recorded cwd is the workspace or a directory below it.
    pub fn contains_cwd(&self, cwd: &Path) -> bool {
        path_is_within(cwd, &self.dir) || path_is_within(cwd, &self.canonical_dir)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn conversation(source: SourceKind, project_dir_name: &str, cwds: &[&str]) -> Conversation {
        let cwds: Vec<PathBuf> = cwds.iter().map(PathBuf::from).collect();
        let mut conversation = crate::search::test_fixtures::one_message_conversation(
            "text",
            chrono::Local::now(),
            None,
            None,
            None,
        );
        conversation.path = PathBuf::from(format!(
            "/home/u/.claude/projects/{project_dir_name}/s.jsonl"
        ));
        conversation.source = source;
        conversation.project_path = cwds.first().cloned();
        conversation.cwd = cwds.first().cloned();
        conversation.cwds = cwds;
        conversation
    }

    fn claude(project_dir_name: &str, cwds: &[&str]) -> Conversation {
        conversation(SourceKind::Claude, project_dir_name, cwds)
    }

    fn workspace(dir: &str) -> Workspace {
        Workspace::new(Path::new(dir))
    }

    #[test]
    fn matches_claude_session_stored_under_workspace_or_worktree() {
        let ws = workspace("/nonexistent/group/repo-a");
        assert!(ws.contains(&claude("-nonexistent-group-repo-a", &[])));
        assert!(ws.contains(&claude("-nonexistent-group-repo-a--worktrees-fix", &[])));
        assert!(!ws.contains(&claude("-nonexistent-group-repo-b", &[])));
    }

    #[test]
    fn matches_session_that_moved_into_workspace() {
        let ws = workspace("/nonexistent/group/repo-a");
        let moved = claude(
            "-nonexistent-group",
            &[
                "/nonexistent/group",
                "/nonexistent/group/repo-b",
                "/nonexistent/group/repo-a/docs",
            ],
        );
        assert!(ws.contains(&moved));
        let sibling_only = claude(
            "-nonexistent-group",
            &["/nonexistent/group", "/nonexistent/group/repo-b"],
        );
        assert!(!ws.contains(&sibling_only));
    }

    #[test]
    fn cwd_must_match_on_component_boundary() {
        let ws = workspace("/nonexistent/group/repo");
        let other = claude("-nonexistent-group", &["/nonexistent/group/repo-other"]);
        assert!(!ws.contains(&other));
    }

    #[test]
    fn non_claude_sessions_match_by_cwd_below_workspace() {
        let ws = workspace("/nonexistent/group");
        for source in [SourceKind::Pi, SourceKind::Omp, SourceKind::Codex] {
            let inside = conversation(source, "ignored", &["/nonexistent/group/repo-a"]);
            assert!(ws.contains(&inside), "{source:?}");
            let outside = conversation(source, "ignored", &["/nonexistent/groupie"]);
            assert!(!ws.contains(&outside), "{source:?}");
        }
        let codex_moved = conversation(
            SourceKind::Codex,
            "ignored",
            &["/nonexistent/other", "/nonexistent/group/repo-a"],
        );
        assert!(ws.contains(&codex_moved));
    }

    #[test]
    fn project_keys_match_by_name_or_path() {
        let ws = workspace("/nonexistent/group");
        assert!(ws.contains_project_key("-nonexistent-group--worktrees-x", false));
        assert!(!ws.contains_project_key("-nonexistent-group-repo-a", false));
        assert!(ws.contains_project_key("/nonexistent/group/repo-a", true));
        assert!(!ws.contains_project_key("/nonexistent/group-b", true));
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_workspace_matches_cwds_through_either_path() {
        let temp = tempfile::tempdir().unwrap();
        let real = temp.path().join("real");
        std::fs::create_dir(&real).unwrap();
        let link = temp.path().join("link");
        std::os::unix::fs::symlink(&real, &link).unwrap();
        let ws = Workspace::new(&link);
        assert!(ws.contains_cwd(&link.join("sub")));
        assert!(ws.contains_cwd(&real.canonicalize().unwrap().join("sub")));
    }
}
