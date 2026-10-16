//! Claude conversation history loading and parsing.
//!
//! This module provides functionality for:
//! - Loading conversations from Claude project directories
//! - Parsing JSONL conversation files
//! - Encoding/decoding project directory paths
//!
//! # Module Structure
//!
//! - `loader` - Loading conversations from directories
//! - `model` - The normalized transcript model adapters decode into
//! - `parser` - Parsing individual JSONL files
//! - `path` - Path encoding/decoding utilities

pub mod actions;
pub mod adapter;
pub mod index;
mod loader;
mod messages;
pub mod model;
pub mod omp_loader;
pub mod parser;
pub mod path;
pub mod pi;
pub mod pi_loader;
mod rename;
pub mod service;
pub mod sources;
pub mod view;
mod workspace;

pub(crate) use messages::{MessageOrdinals, Placement};

use crate::error::{AppError, Result};
use chrono::{DateTime, Local};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::SystemTime;

// Re-export public API
pub use loader::{
    DeleteEmptyScope, delete_empty_transcripts, delete_session_by_uuid, find_jsonl_by_uuid,
};
pub(crate) use messages::{authored_text, extract_skill_preview, retained_user_text};
pub(crate) use parser::process_conversation_file;
pub use path::{convert_path_to_project_dir_name, format_short_name_from_path, is_same_project};
pub use rename::append_session_rename;
pub use workspace::Workspace;

/// Represents a JSONL parsing error with context for debugging
#[derive(Clone, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct ParseError {
    pub line_number: usize,
    pub line_content: String,
    pub error_message: String,
    /// Lines before the error (up to 2)
    pub context_before: Vec<String>,
    /// Lines after the error (up to 2)
    pub context_after: Vec<String>,
}

#[derive(Clone)]
pub struct Conversation {
    pub source: model::SourceKind,
    pub session_id: String,
    pub path: PathBuf,
    pub index: usize,
    pub timestamp: DateTime<Local>,
    pub preview: Arc<str>,
    /// Preview showing first 3 messages (used when show_last=false)
    pub preview_first: Arc<str>,
    /// Preview showing last 3 messages (used when show_last=true)
    pub preview_last: Arc<str>,
    pub full_text: Arc<str>,
    /// What the person typed: the authored text parts of top-level user
    /// messages (see `messages::authored_text`), without tool results,
    /// injected or inherited records, or subagent prompts. Empty for a child
    /// conversation, whose prompts another conversation wrote.
    pub own_text: Arc<str>,
    pub agent_search_text: Arc<str>,
    pub semantic_route_text: Arc<str>,
    pub semantic_turns: Arc<[String]>,
    pub semantic_turn_ranges: Vec<crate::agent::refs::MessageRange>,
    pub project_name: Option<String>,
    pub project_path: Option<PathBuf>,
    /// The working directory extracted from the JSONL file (the actual cwd)
    pub cwd: Option<PathBuf>,
    /// Every distinct working directory the conversation recorded, in order
    /// of first appearance and at most `model::MAX_CWDS`; the first is `cwd`.
    /// A session that changes directory partway records each one.
    pub cwds: Vec<PathBuf>,
    /// Number of user and assistant messages in the conversation
    pub message_count: usize,
    /// Parse errors encountered while processing this conversation file
    pub parse_errors: Vec<ParseError>,
    /// Summary/title of the conversation (from type=summary JSONL entry)
    pub summary: Option<String>,
    /// Custom session title set by user via /rename (from type=custom-title JSONL entry)
    pub custom_title: Option<String>,
    /// Model name from assistant messages (e.g., "claude-opus-4-5-20251101")
    pub model: Option<String>,
    /// Total tokens used in the conversation (input + output + cache)
    pub total_tokens: u64,
    /// Conversation duration in minutes (from first to last message)
    pub duration_minutes: Option<u64>,
    pub provenance: Provenance,
    /// Session id of the conversation this one was started or branched
    /// from, in the same source; see `is_parent_of`.
    pub parent: Option<String>,
}

/// How a conversation came about, as its source records it.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Provenance {
    /// Run by a tool or script rather than a person.
    pub automated: bool,
    /// Started by another conversation, such as a subagent.
    pub child: bool,
    /// A side conversation the person branched off another one, such as a
    /// Claude Code `/btw` fork. Listed like any conversation, but its
    /// source's tool reaches it only through the one it came from.
    pub branch: bool,
    /// The source names a current history that no longer exists, so the
    /// source's own tool cannot act on this one.
    pub orphaned: bool,
}

impl Provenance {
    /// Whether the conversation stays out of the list and of semantic
    /// search unless asked for.
    pub fn hidden_by_default(&self) -> bool {
        self.automated || self.child
    }

    /// The tag marking a conversation that is not an ordinary root in the
    /// list.
    pub fn list_label(&self) -> Option<&'static str> {
        if self.child {
            Some("sub")
        } else if self.branch {
            Some("fork")
        } else if self.automated {
            Some("auto")
        } else {
            None
        }
    }
}

/// Whether `parent` is the conversation `child` names as its parent: the
/// same source and session id, and for Claude, whose sessions can be copied
/// between projects with their subagents and forks, the session directory
/// holding `child`.
pub fn is_parent_of(parent: &Conversation, child: &Conversation) -> bool {
    child.source == parent.source
        && child.parent.as_deref() == Some(parent.session_id.as_str())
        && parent_in_place(parent.source, &parent.path, &child.path)
}

/// The path half of `is_parent_of`, for callers holding only the paths.
pub(crate) fn parent_in_place(
    source: model::SourceKind,
    parent_path: &std::path::Path,
    child_path: &std::path::Path,
) -> bool {
    source != model::SourceKind::Claude
        || sources::claude::parent_transcript(child_path).as_deref() == Some(parent_path)
}

pub(crate) fn semantic_route_text(full_text: &str, agent_search_text: &str) -> String {
    const ROUTE_EVIDENCE_CHARS: usize = 1_000;
    const ROUTE_EVIDENCE_SEGMENTS: usize = 4;
    const ROUTE_KEYWORDS: usize = 100;
    const ROUTE_KEYWORD_CHARS: usize = 800;

    let mut text = full_text.to_string();
    if !agent_search_text.is_empty() {
        text.push(' ');
        text.push_str(agent_search_text);
    }
    if text.is_empty() {
        return String::new();
    }
    let keywords = semantic_route_keywords(&text, ROUTE_KEYWORDS)
        .chars()
        .take(ROUTE_KEYWORD_CHARS)
        .collect::<String>();
    let excerpt = evenly_spaced_excerpt(&text, ROUTE_EVIDENCE_CHARS, ROUTE_EVIDENCE_SEGMENTS);
    [keywords, excerpt]
        .into_iter()
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

fn semantic_route_keywords(text: &str, limit: usize) -> String {
    const STOP_WORDS: &[&str] = &[
        "about", "after", "also", "been", "before", "being", "could", "does", "from", "have",
        "into", "just", "more", "only", "other", "should", "some", "than", "that", "their",
        "there", "these", "they", "this", "through", "using", "very", "what", "when", "where",
        "which", "while", "with", "would", "your",
    ];

    let normalized = text.to_lowercase();
    let mut words = std::collections::HashMap::<&str, (usize, usize)>::new();
    for (position, word) in normalized
        .split(|character: char| !character.is_alphanumeric())
        .filter(|word| {
            let chars = word.chars().count();
            (4..=40).contains(&chars) && !STOP_WORDS.contains(word)
        })
        .enumerate()
    {
        let entry = words.entry(word).or_insert((0, position));
        entry.0 += 1;
    }
    let mut ranked = words.into_iter().collect::<Vec<_>>();
    ranked.sort_unstable_by(|(left_word, left), (right_word, right)| {
        right
            .0
            .cmp(&left.0)
            .then_with(|| right_word.chars().count().cmp(&left_word.chars().count()))
            .then_with(|| left.1.cmp(&right.1))
    });
    ranked
        .into_iter()
        .take(limit)
        .map(|(word, _)| word)
        .collect::<Vec<_>>()
        .join(" ")
}

fn evenly_spaced_excerpt(text: &str, max_chars: usize, segments: usize) -> String {
    let char_count = text.chars().count();
    if char_count <= max_chars || segments <= 1 {
        return text.chars().take(max_chars).collect();
    }
    let segment_chars = max_chars / segments;
    let last_start = char_count.saturating_sub(segment_chars);
    (0..segments)
        .map(|index| {
            let start = last_start * index / (segments - 1);
            text.chars()
                .skip(start)
                .take(segment_chars)
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("\n...\n")
}

pub struct Project {
    pub name: String,         // directory name (encoded)
    pub display_name: String, // heuristic decoded path
    pub modified: SystemTime,
}

/// Get the root Claude projects directory (~/.claude/projects)
/// Respects CLAUDE_CONFIG_DIR env variable if set.
pub fn get_claude_projects_root() -> Result<PathBuf> {
    let claude_dir = if let Ok(config_dir) = std::env::var("CLAUDE_CONFIG_DIR") {
        PathBuf::from(config_dir)
    } else {
        let home_dir = home::home_dir().ok_or_else(|| {
            AppError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "Could not determine home directory",
            ))
        })?;
        home_dir.join(".claude")
    };

    Ok(claude_dir.join("projects"))
}

/// Get the Claude projects directory for the current working directory
pub fn get_claude_projects_dir(current_dir: &std::path::Path) -> Result<PathBuf> {
    let converted = convert_path_to_project_dir_name(current_dir);
    Ok(get_claude_projects_root()?.join(converted))
}
