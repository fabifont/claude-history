//! The app's own model of a conversation's history.
//!
//! A source adapter (see `adapter`) decodes its native files into a
//! `Transcript`: the selected history of one conversation as a list of
//! `Event`s, with the session metadata the source records and the identity
//! of the revision it was decoded from. Indexing, reading and rendering work
//! on this model rather than on any source's wire format.

use super::ParseError;
use chrono::{DateTime, Utc};
use serde_json::Value;
use std::path::PathBuf;

/// The tool that wrote a history.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum SourceKind {
    Claude,
    Pi,
    Omp,
}

impl SourceKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::Claude => "claude",
            Self::Pi => "pi",
            Self::Omp => "omp",
        }
    }

    pub fn list_label(self) -> &'static str {
        match self {
            Self::Claude => "CC",
            Self::Pi => "Pi",
            Self::Omp => "OMP",
        }
    }
}

/// One configured history root of a source, such as the `projects`
/// directory of a `CLAUDE_CONFIG_DIR`. Two roots of the same kind are
/// different instances, and a conversation's identity includes its
/// instance.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct SourceInstance {
    pub kind: SourceKind,
    pub root: PathBuf,
}

/// A conversation within its source instance. `native_id` is the adapter's
/// own stable id for it: the session file's path relative to the instance
/// root for sources that keep one file per session.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ConversationKey {
    pub instance: SourceInstance,
    pub native_id: String,
}

/// Identity of the history a transcript was decoded from: a digest of the
/// artifact ranges read and the state of their files. Equal revisions mean
/// the same events.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Revision(pub [u8; 32]);

impl Revision {
    /// Digest of `segments` with the modification time of each artifact in
    /// epoch nanoseconds (0 when unknown).
    pub fn of(kind: SourceKind, segments: &[Segment], mtimes_ns: &[i64]) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(kind.label().as_bytes());
        for (segment, mtime_ns) in segments.iter().zip(mtimes_ns) {
            let path = segment.artifact.as_os_str().as_encoded_bytes();
            hasher.update(&(path.len() as u64).to_le_bytes());
            hasher.update(path);
            hasher.update(&segment.start.to_le_bytes());
            hasher.update(&segment.end.to_le_bytes());
            hasher.update(&[u8::from(segment.inherited)]);
            hasher.update(&mtime_ns.to_le_bytes());
        }
        Self(*hasher.finalize().as_bytes())
    }
}

/// How the history was chosen among the source's candidates for the
/// conversation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HeadStatus {
    /// The source names this history as the current one.
    Current,
    /// Chosen by the adapter's own rule, without the source's word for it.
    Inferred,
    /// The source names a current history that no longer exists; this is
    /// the newest candidate left.
    Orphaned,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExecutionMode {
    Interactive,
    /// Run by a tool or script rather than a person.
    Automated,
    Unknown,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Relationship {
    Root,
    /// Started by another conversation of the same instance.
    Child {
        parent: String,
    },
    Unknown,
}

/// A byte range of one artifact that forms part of a conversation's history.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Segment {
    pub artifact: PathBuf,
    pub start: u64,
    /// End of the bytes read.
    pub end: u64,
    /// Whether the range belongs to the history this conversation was forked
    /// from rather than to the conversation itself.
    pub inherited: bool,
}

/// Where an event was decoded from.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Origin {
    /// Index into `Transcript::segments`.
    pub segment: u16,
    /// 1-based line of the record in the decoded stream of that segment.
    pub line: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Role {
    User,
    Assistant,
}

/// Who an event is meant for.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Visibility {
    /// Part of the conversation.
    Dialogue,
    /// Context the source injected for the model, such as instructions files.
    Injected,
    /// History of the conversation this one was forked from.
    Inherited,
}

/// Token counts of one model call.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Usage {
    pub input: u64,
    pub output: u64,
    pub cache_creation: u64,
    pub cache_read: u64,
}

impl Usage {
    pub fn total(&self) -> u64 {
        self.input + self.output + self.cache_creation + self.cache_read
    }
}

/// One piece of a message, in the order the source recorded them.
#[derive(Clone, Debug, PartialEq)]
pub enum Part {
    Text(String),
    Thinking(String),
    ToolCall {
        id: String,
        name: String,
        input: Value,
    },
    ToolResult {
        call_id: String,
        output: Option<Value>,
    },
    /// An image, which has no text.
    Image,
    /// A piece of a kind the model does not interpret.
    Other,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Message {
    pub role: Role,
    /// Id of a streamed message: a later message with the same id is a newer
    /// version of it.
    pub id: Option<String>,
    pub parts: Vec<Part>,
    pub model: Option<String>,
    pub usage: Option<Usage>,
    /// Tool call that started the subagent this message belongs to.
    pub parent_tool_use_id: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TitleKind {
    /// A summary the source wrote for the conversation.
    Summary,
    /// A title the source generated.
    Generated,
    /// A title the user set.
    Custom,
}

#[derive(Clone, Debug, PartialEq)]
pub enum EventKind {
    Message(Message),
    /// A message of a subagent, reported to the parent conversation while
    /// the subagent runs.
    Subagent {
        agent_id: String,
        message: Message,
    },
    /// A labeled record of the source, such as a compaction summary or a
    /// model change. Searchable notes count as messages.
    Note {
        label: String,
        text: String,
        searchable: bool,
        usage: Option<Usage>,
    },
    /// The conversation's title or summary changed to `text`, as recorded.
    Title {
        kind: TitleKind,
        text: String,
    },
    /// A record the model does not interpret.
    Other,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Event {
    pub origin: Origin,
    pub timestamp: Option<DateTime<Utc>>,
    pub visibility: Visibility,
    pub kind: EventKind,
}

/// What the source records about the session as a whole, resolved by the
/// adapter from its records under the source's own rules.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SessionMeta {
    /// The source's id for the session.
    pub session_id: String,
    /// Working directory the session ran in.
    pub cwd: Option<PathBuf>,
    /// Project the conversation is listed under.
    pub project: Option<PathBuf>,
    pub model: Option<String>,
    pub summary: Option<String>,
    pub custom_title: Option<String>,
    /// When the conversation was last active, which orders the list.
    pub updated_at: Option<DateTime<Utc>>,
}

/// The selected history of one conversation.
#[derive(Clone, Debug)]
pub struct Transcript {
    pub key: ConversationKey,
    /// The tool that wrote the history. Usually the instance's kind, but one
    /// directory can be the root of several sources.
    pub source: SourceKind,
    pub revision: Revision,
    pub head: HeadStatus,
    pub mode: ExecutionMode,
    pub relationship: Relationship,
    pub meta: SessionMeta,
    /// Artifact ranges the events were decoded from; the conversation's own
    /// artifact comes last.
    pub segments: Vec<Segment>,
    pub events: Vec<Event>,
    /// Records decoded, whether or not each became an event. With
    /// `diagnostics` this tells a history whose every record is malformed.
    pub records: usize,
    /// Records that could not be decoded.
    pub diagnostics: Vec<ParseError>,
}

impl Transcript {
    /// The conversation's own artifact.
    pub fn path(&self) -> &std::path::Path {
        self.segments
            .last()
            .map_or(std::path::Path::new(""), |segment| &segment.artifact)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segment(end: u64) -> Segment {
        Segment {
            artifact: PathBuf::from("/tmp/a.jsonl"),
            start: 0,
            end,
            inherited: false,
        }
    }

    #[test]
    fn revisions_change_with_the_range_read_and_the_file_state() {
        let base = Revision::of(SourceKind::Claude, &[segment(10)], &[5]);
        assert_eq!(base, Revision::of(SourceKind::Claude, &[segment(10)], &[5]));
        assert_ne!(base, Revision::of(SourceKind::Claude, &[segment(11)], &[5]));
        assert_ne!(base, Revision::of(SourceKind::Claude, &[segment(10)], &[6]));
        assert_ne!(base, Revision::of(SourceKind::Pi, &[segment(10)], &[5]));
    }
}
