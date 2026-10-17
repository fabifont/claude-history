//! Conversation loading and project discovery.
//!
//! This module handles loading conversations from Claude project directories,
//! both synchronously and via streaming for the TUI.

use super::Project;
use super::messages::parts_count_as_message;
use super::model::{EventKind, Part, Role, SourceKind};
use super::path::decode_project_dir_name;
use crate::agent::transcript::bounded_tool_result_text;
use crate::error::{AppError, Result};
use rayon::prelude::*;
use std::fs::read_dir;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum DeleteEmptyScope {
    All,
    Local,
}

#[derive(Debug, Clone)]
pub struct EmptyTranscript {
    pub path: PathBuf,
    pub session_id: String,
    pub project_name: String,
    pub user_messages: usize,
    pub line_count: usize,
    pub preview: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct DeleteEmptySummary {
    pub candidates: Vec<EmptyTranscript>,
    pub deleted: usize,
}

pub fn find_jsonl_by_uuid(uuid: &str) -> Result<Option<PathBuf>> {
    let matches = find_all_jsonl_by_uuid(uuid)?;
    Ok(matches.into_iter().next())
}

/// Find all session JSONL files by UUID across all projects.
/// A session may exist in multiple project directories due to cross-project forking.
fn find_all_jsonl_by_uuid(uuid: &str) -> Result<Vec<PathBuf>> {
    let root = super::get_claude_projects_root()?;
    if !root.exists() {
        return Ok(Vec::new());
    }

    let filename = format!("{}.jsonl", uuid);
    let mut matches = Vec::new();

    for entry in read_dir(&root)? {
        let entry = entry?;
        let project_dir = entry.path();
        if !project_dir.is_dir() {
            continue;
        }
        let candidate = project_dir.join(&filename);
        if candidate.exists() {
            matches.push(candidate);
        }
    }

    Ok(matches)
}

/// Delete a session by UUID across all projects.
/// Removes both the .jsonl file and the session subdirectory (tool-results/,
/// subagents/), keeping the `/btw` forks among its subagents.
/// Returns the number of files deleted.
pub fn delete_session_by_uuid(uuid: &str) -> Result<usize> {
    // Validate format to prevent path traversal
    if uuid.is_empty() || !uuid.chars().all(|c| c.is_ascii_alphanumeric() || c == '-') {
        return Err(AppError::SessionNotFound(uuid.to_owned()));
    }

    let matches = find_all_jsonl_by_uuid(uuid)?;
    if matches.is_empty() {
        return Err(AppError::SessionNotFound(uuid.to_owned()));
    }

    let count = matches.len();
    for jsonl_path in &matches {
        std::fs::remove_file(jsonl_path)?;

        // Also remove the session subdirectory if it exists
        if let Some(project_dir) = jsonl_path.parent() {
            let session_dir = project_dir.join(uuid);
            if session_dir.is_dir() {
                super::sources::claude::remove_session_dir(&session_dir)?;
            }
        }
    }

    Ok(count)
}

pub fn delete_empty_transcripts(
    scope: DeleteEmptyScope,
    delete: bool,
) -> Result<DeleteEmptySummary> {
    let candidates = find_empty_transcripts(scope)?;
    let mut deleted = 0;

    if delete {
        for transcript in &candidates {
            std::fs::remove_file(&transcript.path)?;
            if let Some(project_dir) = transcript.path.parent() {
                let session_dir = project_dir.join(&transcript.session_id);
                if session_dir.is_dir() {
                    super::sources::claude::remove_session_dir(&session_dir)?;
                }
            }
            deleted += 1;
        }
    }

    Ok(DeleteEmptySummary {
        candidates,
        deleted,
    })
}

fn find_empty_transcripts(scope: DeleteEmptyScope) -> Result<Vec<EmptyTranscript>> {
    let root = super::get_claude_projects_root()?;
    if !root.exists() {
        return Err(AppError::ProjectsDirNotFound(root.display().to_string()));
    }

    let projects = match scope {
        DeleteEmptyScope::All => list_projects(&root)?,
        DeleteEmptyScope::Local => {
            let current_dir = std::env::current_dir()?;
            let project_dir_name = super::convert_path_to_project_dir_name(&current_dir);
            let project_dir = root.join(&project_dir_name);
            if !project_dir.exists() {
                return Ok(Vec::new());
            }
            vec![Project {
                name: project_dir_name,
                display_name: current_dir.display().to_string(),
                modified: SystemTime::UNIX_EPOCH,
            }]
        }
    };

    let mut candidates: Vec<EmptyTranscript> = projects
        .par_iter()
        .flat_map(|project| {
            let project_dir = root.join(&project.name);
            let entries = match read_dir(project_dir) {
                Ok(entries) => entries,
                Err(_) => return Vec::new(),
            };

            entries
                .filter_map(|entry| {
                    let path = entry.ok()?.path();
                    let filename = path.file_name()?.to_str()?;
                    if path.extension().and_then(|s| s.to_str()) != Some("jsonl")
                        || filename.starts_with("agent-")
                    {
                        return None;
                    }

                    empty_transcript_from_path(&path, &project.display_name)
                        .ok()
                        .flatten()
                })
                .collect::<Vec<_>>()
        })
        .collect();

    candidates.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(candidates)
}

/// The text a user message is found by: its text and the bounded text of
/// its tool results.
fn user_search_text(parts: &[Part]) -> String {
    parts
        .iter()
        .filter_map(|part| match part {
            Part::Text(text) => Some(text.clone()),
            Part::ToolResult {
                output: Some(output),
                ..
            } => bounded_tool_result_text(output),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn empty_transcript_from_path(path: &Path, project_name: &str) -> Result<Option<EmptyTranscript>> {
    let transcript = super::sources::decode_path(path, None, false)?;
    if transcript.source != SourceKind::Claude {
        return Ok(None);
    }
    let line_count = transcript.records + transcript.diagnostics.len();
    let mut user_messages = 0;
    let mut assistant_messages = 0;
    let mut preview = None;

    for event in &transcript.events {
        let message = match &event.kind {
            EventKind::Message(message) | EventKind::Subagent { message, .. } => message,
            _ => continue,
        };
        match (&event.kind, message.role) {
            (EventKind::Message(_), Role::User) => {
                let text = user_search_text(&message.parts);
                if !text.trim().is_empty() {
                    user_messages += 1;
                    if preview.is_none() {
                        preview = Some(super::parser::normalize_whitespace(&text));
                    }
                }
            }
            (_, Role::Assistant) if parts_count_as_message(&message.parts) => {
                assistant_messages += 1;
            }
            _ => {}
        }
    }

    if assistant_messages > 0 {
        return Ok(None);
    }

    let session_id = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or_default()
        .to_owned();

    Ok(Some(EmptyTranscript {
        path: path.to_owned(),
        session_id,
        project_name: project_name.to_owned(),
        user_messages,
        line_count,
        preview,
    }))
}

/// List all projects that contain conversation files
pub fn list_projects(root: &Path) -> Result<Vec<Project>> {
    let entries = read_dir(root)?;

    let mut projects: Vec<Project> = entries
        .par_bridge()
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();

            if !path.is_dir() {
                return None;
            }

            // Check if project has any non-agent .jsonl files
            let has_conversations = read_dir(&path).ok()?.any(|e| {
                e.ok()
                    .map(|e| {
                        let path = e.path();
                        let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
                        path.extension().map(|s| s == "jsonl").unwrap_or(false)
                            && !name.starts_with("agent-")
                    })
                    .unwrap_or(false)
            });

            if !has_conversations {
                return None;
            }

            let name = path.file_name()?.to_string_lossy().to_string();
            // Heuristic decode: convert encoded directory name back to readable path
            // The encoding replaces non-alphanumeric chars (except -) with -
            // So / becomes -, but _ also becomes -, and __ becomes --
            // We convert single dashes to / but preserve double dashes as _
            let display_name = decode_project_dir_name(&name);
            let modified = entry
                .metadata()
                .ok()?
                .modified()
                .ok()
                .unwrap_or(SystemTime::UNIX_EPOCH);

            Some(Project {
                name,
                display_name,
                modified,
            })
        })
        .collect();

    // Sort by recently modified
    projects.sort_by_key(|a| std::cmp::Reverse(a.modified));

    Ok(projects)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::io::Write;

    fn result(output: Option<serde_json::Value>) -> Part {
        Part::ToolResult {
            call_id: "id".into(),
            output,
        }
    }

    #[test]
    fn user_search_text_includes_tool_result_string() {
        let parts = [
            Part::Text("hello".into()),
            result(Some(json!("tool output here"))),
        ];
        let text = user_search_text(&parts);
        assert!(text.contains("hello"));
        assert!(text.contains("tool output here"));
    }

    #[test]
    fn user_search_text_includes_tool_result_array() {
        let parts = [result(Some(json!([
            {"type": "text", "text": "line one"},
            {"type": "text", "text": "line two"}
        ])))];
        let text = user_search_text(&parts);
        assert!(text.contains("line one"));
        assert!(text.contains("line two"));
    }

    #[test]
    fn user_search_text_ignores_non_text_blocks_in_array() {
        let parts = [result(Some(json!([
            {"type": "text", "text": "visible"},
            {"type": "image", "source": {"data": "base64..."}}
        ])))];
        let text = user_search_text(&parts);
        assert!(text.contains("visible"));
        assert!(!text.contains("base64"));
    }

    #[test]
    fn user_search_text_handles_missing_and_empty_results() {
        assert_eq!(user_search_text(&[result(None)]), "");
        assert_eq!(user_search_text(&[result(Some(json!("")))]), "");
    }

    fn write_transcript(lines: &[&str]) -> tempfile::NamedTempFile {
        let mut file = tempfile::Builder::new()
            .suffix(".jsonl")
            .tempfile()
            .unwrap();
        for line in lines {
            writeln!(file, "{line}").unwrap();
        }
        file
    }

    #[test]
    fn empty_transcript_detects_user_only_command_session() {
        let file = write_transcript(&[
            r#"{"type":"user","message":{"role":"user","content":"<command-name>/status</command-name>"}}"#,
        ]);

        let transcript = empty_transcript_from_path(file.path(), "project")
            .unwrap()
            .expect("user-only transcript should be empty");

        assert_eq!(transcript.user_messages, 1);
        assert_eq!(transcript.line_count, 1);
        assert_eq!(transcript.project_name, "project");
        assert_eq!(
            transcript.preview.as_deref(),
            Some("<command-name>/status</command-name>")
        );
    }

    #[test]
    fn empty_transcript_ignores_transcript_with_assistant_message() {
        let file = write_transcript(&[
            r#"{"type":"user","message":{"role":"user","content":"hi"}}"#,
            r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"hello"}]}}"#,
        ]);

        let transcript = empty_transcript_from_path(file.path(), "project").unwrap();

        assert!(transcript.is_none());
    }

    #[test]
    fn empty_transcript_includes_metadata_only_file() {
        let file = write_transcript(&[r#"{"type":"summary","summary":"Only metadata"}"#]);

        let transcript = empty_transcript_from_path(file.path(), "project")
            .unwrap()
            .expect("metadata-only transcript should be empty");

        assert_eq!(transcript.user_messages, 0);
        assert_eq!(transcript.line_count, 1);
        assert_eq!(transcript.preview, None);
    }
}
