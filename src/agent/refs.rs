use crate::agent::diagnostic::{AgentError, AgentErrorKind};
use crate::error::{AppError, Result};
use crate::history::Conversation;
use crate::history::model::SourceKind;
use crate::history::sources::Registry;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Namespace of refs digesting a conversation's source instance and native
/// id.
const REF_NAMESPACE: &str = "agent-v2";
/// Namespace of the path-based refs Claude conversations had before refs
/// digested the source instance; they still resolve.
const LEGACY_REF_NAMESPACE: &str = "agent-v1";
const PROJECT_NAMESPACE: &str = "agent-project-v1";
const MIN_EMITTED_DIGEST_HEX_LEN: usize = 12;
const PROJECT_DIGEST_HEX_LEN: usize = 16;
const MIN_PREFIX_HEX_LEN: usize = 8;
const DIGEST_HEX_LEN: usize = 32;
const UUID_HEX_LEN: usize = 32;
const UUID_LEN: usize = 36;
const FNV_OFFSET: u128 = 0x6c62272e07bb014262b821756295c58d;
const FNV_PRIME: u128 = 0x0000000001000000000000000000013b;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AgentConversationRef {
    uuid: String,
    digest_hex: String,
    emitted_digest_hex_len: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum ConversationRefInput {
    Digest(String),
    Uuid(String),
    Filename(String),
}

impl AgentConversationRef {
    fn new(uuid: String, digest: u128) -> Self {
        Self {
            uuid,
            digest_hex: format!("{digest:032x}"),
            emitted_digest_hex_len: MIN_EMITTED_DIGEST_HEX_LEN,
        }
    }

    fn with_emitted_digest_hex_len(mut self, len: usize) -> Self {
        self.emitted_digest_hex_len = len;
        self
    }

    pub fn canonical(&self) -> String {
        format!("ch_{}", &self.digest_hex[..self.emitted_digest_hex_len])
    }

    pub fn full_ref(&self) -> String {
        format!("ch_{}", self.digest_hex)
    }

    pub fn uuid(&self) -> String {
        self.uuid.clone()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AgentConversationKey {
    pub source: SourceKind,
    pub project_dir_name: String,
    pub session_filename: String,
    pub session_id: String,
    pub path: PathBuf,
    /// Root of the source instance holding the conversation, canonical
    /// where it resolves.
    pub root: PathBuf,
    /// The source adapter's id of the conversation within `root`.
    pub native_id: String,
}

impl AgentConversationKey {
    /// Key of the Claude session file `path`, at
    /// `<projects root>/<project_dir_name>/<session_filename>`.
    pub fn new(
        project_dir_name: impl Into<String>,
        session_filename: impl Into<String>,
        path: PathBuf,
    ) -> Self {
        let project_dir_name = project_dir_name.into();
        let session_filename = session_filename.into();
        let root = path
            .parent()
            .and_then(Path::parent)
            .map(canonical_root)
            .unwrap_or_default();
        Self {
            source: SourceKind::Claude,
            session_id: session_filename
                .strip_suffix(".jsonl")
                .unwrap_or(&session_filename)
                .to_owned(),
            native_id: format!("{project_dir_name}/{session_filename}"),
            project_dir_name,
            session_filename,
            path,
            root,
        }
    }

    /// Key of the session file `path` under `root`, the root of a source
    /// instance keeping one file per session.
    pub fn in_instance(
        source: SourceKind,
        root: &Path,
        project_dir_name: String,
        session_id: String,
        path: PathBuf,
    ) -> Self {
        let session_filename = path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        let native_id = path
            .strip_prefix(root)
            .unwrap_or(&path)
            .to_string_lossy()
            .into_owned();
        Self {
            source,
            project_dir_name,
            session_filename,
            session_id,
            root: canonical_root(root),
            native_id,
            path,
        }
    }

    /// Key of the conversation stored at `conversation.path`, identified
    /// within the first configured instance whose root holds it.
    pub fn from_conversation(conversation: &Conversation) -> Result<Self> {
        Self::from_conversation_in(conversation, &Registry::from_env())
    }

    pub fn from_conversation_in(conversation: &Conversation, registry: &Registry) -> Result<Self> {
        let project_dir_name = if conversation.source != SourceKind::Claude {
            let project = conversation
                .project_path
                .as_deref()
                .or(conversation.cwd.as_deref())
                .ok_or_else(|| {
                    AppError::ConfigError(format!(
                        "{} conversation has no project path",
                        conversation.source.label()
                    ))
                })?;
            project
                .canonicalize()
                .unwrap_or_else(|_| project.to_path_buf())
                .to_string_lossy()
                .into_owned()
        } else {
            conversation
                .path
                .parent()
                .and_then(|p| p.file_name())
                .and_then(|n| n.to_str())
                .ok_or_else(|| {
                    AppError::ConfigError("conversation path has no project directory".to_string())
                })?
                .to_string()
        };
        let session_filename = conversation
            .path
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| {
                AppError::ConfigError("conversation path has no session filename".to_string())
            })?
            .to_string();
        let (root, native_id) = match registry.key_of(&conversation.path) {
            Some(key) => (canonical_root(&key.instance.root), key.native_id),
            None => (
                conversation
                    .path
                    .parent()
                    .map(canonical_root)
                    .unwrap_or_default(),
                session_filename.clone(),
            ),
        };
        Ok(Self {
            source: conversation.source,
            project_dir_name,
            session_filename,
            session_id: conversation.session_id.clone(),
            path: conversation.path.clone(),
            root,
            native_id,
        })
    }

    /// The conversation's ref: a digest of its source instance and native
    /// id, so it holds while the adapter keeps the id (a Codex thread
    /// across reverts and archiving).
    pub fn conversation_ref(&self) -> AgentConversationRef {
        let root = self.root.to_string_lossy();
        let digest = digest_parts([REF_NAMESPACE, self.source.label(), &root, &self.native_id]);
        AgentConversationRef::new(self.uuid(), digest)
    }

    /// The ref the conversation had when refs digested its project and file
    /// name rather than its source instance.
    pub fn legacy_ref(&self) -> AgentConversationRef {
        let digest = match self.source {
            SourceKind::Claude => digest_parts([
                LEGACY_REF_NAMESPACE,
                &self.project_dir_name,
                &self.session_filename,
            ]),
            SourceKind::Pi => digest_parts([
                "agent-pi-v1",
                self.source.label(),
                &self.project_dir_name,
                &self.session_id,
                &self.session_filename,
            ]),
            SourceKind::Omp => digest_parts([
                "agent-omp-v1",
                self.source.label(),
                &self.project_dir_name,
                &self.session_id,
                &self.session_filename,
            ]),
            SourceKind::Codex => {
                digest_parts(["agent-codex-v1", self.source.label(), &self.session_id])
            }
        };
        AgentConversationRef::new(self.uuid(), digest)
    }

    fn uuid(&self) -> String {
        match self.source {
            SourceKind::Claude => session_uuid(&self.session_filename)
                .filter(|uuid| is_uuid(uuid))
                .unwrap_or("none")
                .to_ascii_lowercase(),
            _ => self.session_id.clone(),
        }
    }

    pub fn project_id(&self) -> String {
        let identity = match self.source {
            SourceKind::Claude => format!("{PROJECT_NAMESPACE}\0{}", self.project_dir_name),
            SourceKind::Pi => format!("agent-pi-project-v1\0{}", self.project_dir_name),
            SourceKind::Omp => format!("agent-omp-project-v1\0{}", self.project_dir_name),
            SourceKind::Codex => format!("agent-codex-project-v1\0{}", self.project_dir_name),
        };
        format!(
            "pr_{}",
            &blake3::hash(identity.as_bytes()).to_hex()[..PROJECT_DIGEST_HEX_LEN]
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedConversation {
    pub key: AgentConversationKey,
    pub reference: AgentConversationRef,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct MessageRange {
    pub start: usize,
    pub end: usize,
}

impl MessageRange {
    pub fn single(message: usize) -> Self {
        Self {
            start: message,
            end: message,
        }
    }

    pub fn contains(&self, other: &MessageRange) -> bool {
        self.start <= other.start && self.end >= other.end
    }

    pub fn union(&self, other: &MessageRange) -> Self {
        Self {
            start: self.start.min(other.start),
            end: self.end.max(other.end),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReadRef {
    pub conversation: String,
    pub range: Option<MessageRange>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FocusRef {
    pub conversation: Option<String>,
    pub range: MessageRange,
}

pub fn parse_read_ref(input: &str) -> Result<ReadRef> {
    let (conversation, range) = match input.split_once(':') {
        Some((conversation, range)) => (conversation, Some(parse_message_range(range)?)),
        None => (input, None),
    };
    validate_conversation_ref(conversation)?;
    Ok(ReadRef {
        conversation: conversation.to_string(),
        range,
    })
}

pub fn parse_focus_ref(input: &str) -> Result<FocusRef> {
    if let Some((conversation, range)) = input.split_once(':') {
        validate_conversation_ref(conversation)?;
        Ok(FocusRef {
            conversation: Some(conversation.to_string()),
            range: parse_message_range(range)?,
        })
    } else {
        Ok(FocusRef {
            conversation: None,
            range: parse_message_range(input)?,
        })
    }
}

pub fn validate_resolved_focus_in_ranges(
    read_refs: &[(ReadRef, ResolvedConversation)],
    focus: &FocusRef,
    focus_conversation: Option<&ResolvedConversation>,
) -> Result<()> {
    let target_ref = if let Some(focus_conversation) = focus_conversation {
        focus_conversation.reference.full_ref()
    } else {
        let Some((_, first)) = read_refs.first() else {
            return Err(
                AgentError::invalid_ref("focus", "focus requires at least one read ref").into(),
            );
        };
        let first_ref = first.reference.full_ref();
        if read_refs
            .iter()
            .any(|(_, resolved)| resolved.reference.full_ref() != first_ref)
        {
            return Err(AgentError::invalid_ref(
                "focus",
                "bare focus is ambiguous for multiple conversations; use ch_...:mN",
            )
            .into());
        }
        first_ref
    };

    let contained = read_refs.iter().any(|(read_ref, resolved)| {
        resolved.reference.full_ref() == target_ref
            && read_ref
                .range
                .unwrap_or(MessageRange {
                    start: 1,
                    end: usize::MAX,
                })
                .contains(&focus.range)
    });

    if contained {
        Ok(())
    } else {
        Err(AgentError::out_of_range(
            focus.conversation.as_deref(),
            format!(
                "focus m{}..m{} is outside the requested read range",
                focus.range.start, focus.range.end
            ),
        )
        .into())
    }
}

pub fn resolve_conversation_ref(
    keys: &[AgentConversationKey],
    reference: &str,
) -> Result<ResolvedConversation> {
    let input = validate_conversation_ref(reference)?;
    let mut matches: Vec<&AgentConversationKey> = keys
        .iter()
        .filter(|key| match &input {
            ConversationRefInput::Digest(prefix) => {
                key.conversation_ref().digest_hex.starts_with(prefix)
            }
            ConversationRefInput::Uuid(uuid) => key.uuid().eq_ignore_ascii_case(uuid),
            ConversationRefInput::Filename(stem) => {
                key.session_filename.strip_suffix(".jsonl") == Some(stem.as_str())
            }
        })
        .collect();
    if matches.is_empty()
        && let ConversationRefInput::Digest(prefix) = &input
    {
        matches = legacy_matches(keys, prefix);
    }

    finish_resolution(
        reference,
        matches
            .into_iter()
            .map(|key| resolved_conversation_for_key(keys, key))
            .collect(),
    )
}

/// Keys whose legacy ref starts with `prefix`, for refs issued before refs
/// digested the source instance.
fn legacy_matches<'a>(
    keys: &'a [AgentConversationKey],
    prefix: &str,
) -> Vec<&'a AgentConversationKey> {
    keys.iter()
        .filter(|key| key.legacy_ref().digest_hex.starts_with(prefix))
        .collect()
}

pub fn resolved_conversation_for_key(
    keys: &[AgentConversationKey],
    key: &AgentConversationKey,
) -> ResolvedConversation {
    let base = keys
        .iter()
        .map(AgentConversationKey::conversation_ref)
        .collect::<Vec<_>>();
    ResolvedConversation {
        key: key.clone(),
        reference: unique_emitted_reference(&key.conversation_ref(), &base),
    }
}

pub fn resolved_conversations_for_keys(keys: &[AgentConversationKey]) -> Vec<ResolvedConversation> {
    let base = keys
        .iter()
        .map(AgentConversationKey::conversation_ref)
        .collect::<Vec<_>>();
    keys.iter()
        .cloned()
        .zip(unique_emitted_references(&base))
        .map(|(key, reference)| ResolvedConversation { key, reference })
        .collect()
}

fn unique_emitted_references(base: &[AgentConversationRef]) -> Vec<AgentConversationRef> {
    let mut sorted = (0..base.len()).collect::<Vec<_>>();
    sorted.sort_unstable_by(|left, right| base[*left].digest_hex.cmp(&base[*right].digest_hex));

    let mut lengths = vec![MIN_EMITTED_DIGEST_HEX_LEN; base.len()];
    for (position, index) in sorted.iter().copied().enumerate() {
        let neighboring_prefix = position
            .checked_sub(1)
            .map(|previous| common_digest_prefix(&base[index], &base[sorted[previous]]))
            .into_iter()
            .chain(
                sorted
                    .get(position + 1)
                    .map(|next| common_digest_prefix(&base[index], &base[*next])),
            )
            .max()
            .unwrap_or(0);
        lengths[index] = neighboring_prefix
            .saturating_add(1)
            .clamp(MIN_EMITTED_DIGEST_HEX_LEN, DIGEST_HEX_LEN);
    }

    base.iter()
        .zip(lengths)
        .map(|(reference, len)| reference.clone().with_emitted_digest_hex_len(len))
        .collect()
}

fn common_digest_prefix(left: &AgentConversationRef, right: &AgentConversationRef) -> usize {
    left.digest_hex
        .bytes()
        .zip(right.digest_hex.bytes())
        .take_while(|(left, right)| left == right)
        .count()
}

fn unique_emitted_reference(
    reference: &AgentConversationRef,
    base: &[AgentConversationRef],
) -> AgentConversationRef {
    let len = (MIN_EMITTED_DIGEST_HEX_LEN..=DIGEST_HEX_LEN)
        .find(|len| {
            base.iter()
                .filter(|candidate| candidate.digest_hex[..*len] == reference.digest_hex[..*len])
                .count()
                == 1
        })
        .unwrap_or(DIGEST_HEX_LEN);
    reference.clone().with_emitted_digest_hex_len(len)
}

fn finish_resolution(
    reference: &str,
    matches: Vec<ResolvedConversation>,
) -> Result<ResolvedConversation> {
    match matches.as_slice() {
        [resolved] => Ok(resolved.clone()),
        [] => Err(AgentError::new(
            AgentErrorKind::NotFound,
            Some(reference),
            format!("conversation ref {reference} was not found in discovered sessions; check the full UUID or exact filename and configured session storage roots"),
        )
        .into()),
        _ => {
            let candidates = matches
                .iter()
                .map(|m| {
                    format!(
                        "{} project={} source={} directory={} file={}",
                        m.reference.canonical(),
                        m.key.project_id(),
                        m.key.source.label(),
                        m.key.project_dir_name,
                        m.key.session_filename
                    )
                })
                .collect::<Vec<_>>()
                .join("\n  ");
            Err(AgentError::new(
                AgentErrorKind::AmbiguousRef,
                Some(reference),
                format!("ambiguous conversation ref; retry with one of these ch_ handles; candidates: {candidates}"),
            )
            .into())
        }
    }
}

pub fn conversation_keys_from_conversations(
    conversations: &[Conversation],
) -> Result<Vec<AgentConversationKey>> {
    conversations
        .iter()
        .map(AgentConversationKey::from_conversation)
        .collect()
}

fn validate_conversation_ref(reference: &str) -> Result<ConversationRefInput> {
    if let Some(hex) = reference.strip_prefix("ch_") {
        if hex.len() < MIN_PREFIX_HEX_LEN {
            return Err(AgentError::invalid_ref(
                reference,
                format!(
                    "conversation ref is too short; use at least {MIN_PREFIX_HEX_LEN} hex characters"
                ),
            )
            .into());
        }
        if hex.len() > DIGEST_HEX_LEN {
            return Err(AgentError::invalid_ref(
                reference,
                format!(
                    "conversation ref is too long; use at most {DIGEST_HEX_LEN} hex characters"
                ),
            )
            .into());
        }
        if !hex.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err(AgentError::invalid_ref(
                reference,
                "conversation ref must contain hexadecimal digits",
            )
            .into());
        }
        return Ok(ConversationRefInput::Digest(hex.to_ascii_lowercase()));
    }

    if is_uuid(reference) {
        return Ok(ConversationRefInput::Uuid(reference.to_ascii_lowercase()));
    }
    let stem = reference.strip_suffix(".jsonl").unwrap_or(reference);
    if !stem.contains(['/', '\\'])
        && (is_uuid(stem)
            || stem
                .rsplit_once('_')
                .is_some_and(|(prefix, uuid)| !prefix.is_empty() && is_uuid(uuid)))
    {
        return Ok(ConversationRefInput::Filename(stem.to_owned()));
    }
    Err(AgentError::invalid_ref(reference, "use a full UUID, UUID-based session basename/filename (not a path), or ref=ch_... from agent output").into())
}

/// `root`, canonical where it resolves, so a ref does not depend on how the
/// root is spelled (a symlinked home, a relative `CLAUDE_CONFIG_DIR`).
/// Resolved once per root and thread.
fn canonical_root(root: &Path) -> PathBuf {
    thread_local! {
        static ROOTS: RefCell<HashMap<PathBuf, PathBuf>> = RefCell::default();
    }
    ROOTS.with_borrow_mut(|roots| {
        roots
            .entry(root.to_path_buf())
            .or_insert_with(|| root.canonicalize().unwrap_or_else(|_| root.to_path_buf()))
            .clone()
    })
}

fn session_uuid(session_filename: &str) -> Option<&str> {
    session_filename.strip_suffix(".jsonl")
}

fn is_uuid(value: &str) -> bool {
    value.len() == UUID_LEN
        && value.chars().enumerate().all(|(index, c)| match index {
            8 | 13 | 18 | 23 => c == '-',
            _ => c.is_ascii_hexdigit(),
        })
        && value.chars().filter(|c| c.is_ascii_hexdigit()).count() == UUID_HEX_LEN
}

fn parse_message_range(input: &str) -> Result<MessageRange> {
    if input.contains("...") {
        return Err(
            AgentError::invalid_ref(input, "invalid message range; use mN or mN..mM").into(),
        );
    }
    if let Some((start, end)) = input.split_once("..") {
        if start.is_empty() || end.is_empty() {
            return Err(AgentError::invalid_ref(
                input,
                "open-ended message ranges are not supported",
            )
            .into());
        }
        let start = parse_message_number(start)?;
        let end = parse_message_number(end)?;
        if start > end {
            return Err(
                AgentError::invalid_ref(input, "message range start must be before end").into(),
            );
        }
        Ok(MessageRange { start, end })
    } else {
        Ok(MessageRange::single(parse_message_number(input)?))
    }
}

fn parse_message_number(input: &str) -> Result<usize> {
    let Some(number) = input.strip_prefix('m') else {
        return Err(AgentError::invalid_ref(input, "invalid message ref; expected mN").into());
    };
    if number.is_empty() || !number.chars().all(|c| c.is_ascii_digit()) {
        return Err(AgentError::invalid_ref(input, "invalid message ref; expected mN").into());
    }
    let parsed = number
        .parse::<usize>()
        .map_err(|_| AgentError::invalid_ref(input, "invalid message ref; expected mN"))?;
    if parsed == 0 {
        return Err(AgentError::invalid_ref(input, "message refs are 1-based").into());
    }
    Ok(parsed)
}

fn digest_parts<'a>(parts: impl IntoIterator<Item = &'a str>) -> u128 {
    let mut hash = FNV_OFFSET;
    for part in parts {
        for byte in (part.len() as u64).to_le_bytes() {
            hash = (hash ^ byte as u128).wrapping_mul(FNV_PRIME);
        }
        for byte in part.as_bytes() {
            hash = (hash ^ *byte as u128).wrapping_mul(FNV_PRIME);
        }
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(project: &str, filename: &str) -> AgentConversationKey {
        AgentConversationKey::new(
            project,
            filename,
            PathBuf::from(format!("/{project}/{filename}")),
        )
    }

    fn agent_error_kind(error: AppError) -> AgentErrorKind {
        match error {
            AppError::Agent(error) => error.kind,
            error => panic!("expected agent error, got {error}"),
        }
    }

    fn pi(native_id: &str, session_id: &str) -> AgentConversationKey {
        let path = PathBuf::from(format!("/sessions/{native_id}"));
        AgentConversationKey {
            source: SourceKind::Pi,
            project_dir_name: "/tmp/project".to_owned(),
            session_filename: path.file_name().unwrap().to_str().unwrap().to_owned(),
            session_id: session_id.to_owned(),
            path,
            root: PathBuf::from("/sessions"),
            native_id: native_id.to_owned(),
        }
    }

    #[test]
    fn refs_digest_the_source_instance_and_native_id() {
        let pi = pi(
            "--tmp-project--/2024_custom_id_with_underscores.jsonl",
            "custom_id_with_underscores",
        );
        let omp = AgentConversationKey {
            source: SourceKind::Omp,
            ..pi.clone()
        };
        let other_root = AgentConversationKey {
            root: PathBuf::from("/other"),
            ..pi.clone()
        };
        let moved_project = AgentConversationKey {
            project_dir_name: "/tmp/other".to_owned(),
            ..pi.clone()
        };

        assert_eq!(pi.conversation_ref().uuid(), "custom_id_with_underscores");
        assert_ne!(
            pi.conversation_ref().full_ref(),
            omp.conversation_ref().full_ref()
        );
        assert_ne!(
            pi.conversation_ref().full_ref(),
            other_root.conversation_ref().full_ref()
        );
        // The project a session ran in is not part of its identity.
        assert_eq!(
            pi.conversation_ref().full_ref(),
            moved_project.conversation_ref().full_ref()
        );
        assert_ne!(
            pi.legacy_ref().full_ref(),
            moved_project.legacy_ref().full_ref()
        );
        assert_ne!(pi.conversation_ref(), pi.legacy_ref());
        assert!(pi.conversation_ref().canonical().starts_with("ch_"));
    }

    #[test]
    fn codex_refs_hold_across_rollouts_of_a_thread() {
        let thread = "019a0000-0000-7000-8000-000000000001";
        let codex = |path: &str| AgentConversationKey {
            source: SourceKind::Codex,
            project_dir_name: "/tmp/project".to_owned(),
            session_filename: format!("rollout-{thread}.jsonl"),
            session_id: thread.to_owned(),
            path: PathBuf::from(path),
            root: PathBuf::from("/home/.codex"),
            native_id: thread.to_owned(),
        };
        let live = codex("/home/.codex/sessions/2026/10/16/rollout-a.jsonl");
        let reverted = codex("/home/.codex/sessions/2026/10/17/rollout-b.jsonl");
        let archived = codex("/home/.codex/archived_sessions/rollout-b.jsonl");

        assert_eq!(live.conversation_ref(), reverted.conversation_ref());
        assert_eq!(live.conversation_ref(), archived.conversation_ref());
    }

    #[test]
    fn legacy_refs_resolve_to_the_current_ref() {
        let claude = key("project-a", "12345678-1234-4234-9234-123456789abc.jsonl");
        let pi = pi("--tmp--/2024_session.jsonl", "session");
        let keys = vec![claude.clone(), pi.clone()];
        for key in &keys {
            let legacy = key.legacy_ref();
            let resolved = resolve_conversation_ref(&keys, &legacy.canonical()).unwrap();
            assert_eq!(resolved.key, *key);
            assert_eq!(
                resolved.reference.full_ref(),
                key.conversation_ref().full_ref()
            );
            let full = resolve_conversation_ref(&keys, &legacy.full_ref()).unwrap();
            assert_eq!(full.key, *key);
        }
    }

    #[test]
    fn pi_refs_distinguish_duplicate_session_ids_in_one_project() {
        let first = pi("--tmp-project--/first.jsonl", "copied-session");
        let second = pi("--tmp-project--/second.jsonl", "copied-session");

        assert_ne!(
            first.conversation_ref().full_ref(),
            second.conversation_ref().full_ref()
        );
        let keys = vec![first, second];
        for resolved in resolved_conversations_for_keys(&keys) {
            assert_eq!(
                resolve_conversation_ref(&keys, &resolved.reference.canonical())
                    .unwrap()
                    .key
                    .path,
                resolved.key.path
            );
        }
    }

    #[test]
    fn emitted_refs_extend_colliding_minimum_prefixes() {
        let mut first = key("project-a", "one.jsonl").conversation_ref();
        first.digest_hex = "aaaaaaaaaaaa0bbbbbbbbbbbbbbbbbbb".to_string();
        let mut second = key("project-b", "two.jsonl").conversation_ref();
        second.digest_hex = "aaaaaaaaaaaa1ccccccccccccccccccc".to_string();

        let emitted = unique_emitted_references(&[first, second]);

        assert_eq!(emitted[0].canonical(), "ch_aaaaaaaaaaaa0");
        assert_eq!(emitted[1].canonical(), "ch_aaaaaaaaaaaa1");
    }

    #[test]
    fn emitted_refs_keep_documented_twelve_hex_minimum() {
        let refs = unique_emitted_references(&[
            key("project-a", "one.jsonl").conversation_ref(),
            key("project-b", "two.jsonl").conversation_ref(),
        ]);

        assert!(
            refs.iter()
                .all(|reference| reference.canonical().len() == 15)
        );
    }

    #[test]
    fn bulk_resolution_matches_individual_resolution() {
        let keys = (0..500)
            .map(|index| {
                key(
                    &format!("project-{}", index % 17),
                    &format!("{index}.jsonl"),
                )
            })
            .collect::<Vec<_>>();

        let bulk = resolved_conversations_for_keys(&keys);
        let individual = keys
            .iter()
            .map(|key| resolved_conversation_for_key(&keys, key))
            .collect::<Vec<_>>();

        assert_eq!(bulk, individual);
    }

    #[test]
    fn duplicate_full_digests_emit_full_refs() {
        let mut first = key("project-a", "one.jsonl").conversation_ref();
        first.digest_hex = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string();
        let mut second = key("project-b", "two.jsonl").conversation_ref();
        second.digest_hex = first.digest_hex.clone();

        let emitted = unique_emitted_references(&[first, second]);

        assert!(
            emitted
                .iter()
                .all(|reference| reference.canonical() == "ch_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        );
    }

    #[test]
    fn project_identity_distinguishes_duplicate_uuids() {
        let filename = "12345678-1234-4234-9234-123456789abc.jsonl";
        assert_ne!(
            key("project-a", filename).project_id(),
            key("project-b", filename).project_id()
        );
    }

    #[test]
    fn canonical_ref_is_internal_digest_ref() {
        let key = key("project-a", "12345678-1234-4234-9234-123456789abc.jsonl");
        let reference = key.conversation_ref();
        assert_eq!(reference.canonical().len(), "ch_".len() + 12);
        assert_eq!(reference.full_ref().len(), "ch_".len() + DIGEST_HEX_LEN);
        assert_eq!(reference.uuid(), "12345678-1234-4234-9234-123456789abc");
        assert_eq!(reference, key.conversation_ref());
        // Refs agents were given before keep their digest.
        assert_eq!(key.legacy_ref().canonical(), "ch_98bd40760a01");
    }

    #[test]
    fn same_uuid_across_projects_has_distinct_internal_refs() {
        let first =
            key("project-a", "12345678-1234-4234-9234-123456789abc.jsonl").conversation_ref();
        let second =
            key("project-b", "12345678-1234-4234-9234-123456789abc.jsonl").conversation_ref();
        assert_ne!(first.full_ref(), second.full_ref());
        assert_eq!(first.uuid(), second.uuid());
    }

    #[test]
    fn resolves_uuid_and_reports_missing_and_duplicate_identities() {
        let uuid = "12345678-1234-4234-9234-123456789abc";
        let first = key("project-a", &format!("{uuid}.jsonl"));
        let keys = vec![first.clone()];
        assert_eq!(
            resolve_conversation_ref(&keys, &uuid.to_uppercase())
                .unwrap()
                .key,
            first
        );
        assert_eq!(
            agent_error_kind(resolve_conversation_ref(&[], uuid).unwrap_err()),
            AgentErrorKind::NotFound
        );
        let keys = vec![first, key("project-b", &format!("{uuid}.jsonl"))];
        let error = resolve_conversation_ref(&keys, uuid).unwrap_err();
        let detail = error.to_string();
        assert!(detail.contains("retry with"));
        for key in &keys {
            assert!(detail.contains(&key.project_id()));
            assert!(detail.contains(&key.project_dir_name));
            let handle = key.conversation_ref().canonical();
            assert!(detail.contains(&handle));
            assert_eq!(resolve_conversation_ref(&keys, &handle).unwrap().key, *key);
        }
        assert_eq!(agent_error_kind(error), AgentErrorKind::AmbiguousRef);
    }

    #[test]
    fn resolves_pi_filenames_exactly_and_preserves_ranges_and_focus() {
        let uuid = "01a082ad-c8d8-7149-8d3e-dd8f74bb7ed4";
        let stem = format!("2026-09-08T20-20-22-361Z_{uuid}");
        let mut pi = key("project-a", &format!("{stem}.jsonl"));
        pi.source = SourceKind::Pi;
        pi.session_id = uuid.to_owned();
        let keys = vec![pi.clone()];
        for input in [uuid.to_owned(), stem.clone(), format!("{stem}.jsonl")] {
            assert_eq!(resolve_conversation_ref(&keys, &input).unwrap().key, pi);
            let read = parse_read_ref(&format!("{input}:m2..m4")).unwrap();
            assert_eq!(read.range, Some(MessageRange { start: 2, end: 4 }));
            let resolved = resolve_conversation_ref(&keys, &read.conversation).unwrap();
            let focus = parse_focus_ref(&format!("{uuid}:m3")).unwrap();
            validate_resolved_focus_in_ranges(&[(read, resolved.clone())], &focus, Some(&resolved))
                .unwrap();
        }
        assert_eq!(
            agent_error_kind(
                resolve_conversation_ref(&keys, &format!("wrong_{uuid}")).unwrap_err()
            ),
            AgentErrorKind::NotFound
        );
        assert!(parse_read_ref(&format!("/tmp/{stem}.jsonl")).is_err());
        let duplicate = AgentConversationKey {
            project_dir_name: "project-b".into(),
            ..pi.clone()
        };
        assert_eq!(
            agent_error_kind(resolve_conversation_ref(&[pi, duplicate], &stem).unwrap_err()),
            AgentErrorKind::AmbiguousRef
        );
    }

    #[test]
    fn resolves_internal_digest_prefix() {
        let keys = vec![
            key("project-a", "12345678-1234-4234-9234-123456789abc.jsonl"),
            key("project-b", "87654321-1234-4234-9234-123456789abc.jsonl"),
        ];
        let internal = key("project-a", "12345678-1234-4234-9234-123456789abc.jsonl")
            .conversation_ref()
            .digest_hex;
        let resolved = resolve_conversation_ref(&keys, &format!("ch_{}", &internal[..8])).unwrap();
        assert_eq!(
            resolved.key.session_filename,
            "12345678-1234-4234-9234-123456789abc.jsonl"
        );
    }

    #[test]
    fn ambiguous_prefix_reports_canonical_candidates() {
        let first = key("project-a", "12345678-1234-4234-9234-123456789abc.jsonl");
        let second = key("project-c", "12345678-ffff-4234-9234-123456789abc.jsonl");
        let first_ref = first.conversation_ref();
        let second_ref = second.conversation_ref();
        let err = finish_resolution(
            &first_ref.canonical(),
            vec![
                ResolvedConversation {
                    key: first,
                    reference: first_ref.clone(),
                },
                ResolvedConversation {
                    key: second,
                    reference: second_ref,
                },
            ],
        )
        .unwrap_err();
        let message = err.to_string();
        assert!(message.contains("ambiguous conversation ref"));
        assert!(message.contains(&first_ref.canonical()));
        assert!(message.contains("12345678-ffff-4234-9234-123456789abc.jsonl"));
    }

    #[test]
    fn maps_ref_failures_to_precise_kinds() {
        let keys = vec![key(
            "project-a",
            "12345678-1234-4234-9234-123456789abc.jsonl",
        )];
        assert_eq!(
            agent_error_kind(parse_read_ref("bad-ref").unwrap_err()),
            AgentErrorKind::InvalidRef
        );
        assert_eq!(
            agent_error_kind(resolve_conversation_ref(&keys, "ch_12345678").unwrap_err()),
            AgentErrorKind::NotFound
        );

        let first = keys[0].clone();
        let second = key("project-c", "12345678-ffff-4234-9234-123456789abc.jsonl");
        let first_ref = first.conversation_ref();
        let second_ref = second.conversation_ref();
        assert_eq!(
            agent_error_kind(
                finish_resolution(
                    &first_ref.canonical(),
                    vec![
                        ResolvedConversation {
                            key: first,
                            reference: first_ref,
                        },
                        ResolvedConversation {
                            key: second,
                            reference: second_ref,
                        },
                    ],
                )
                .unwrap_err(),
            ),
            AgentErrorKind::AmbiguousRef
        );
    }

    #[test]
    fn read_ref_rejects_invalid_forms() {
        assert!(
            parse_read_ref("ch_1234567")
                .unwrap_err()
                .to_string()
                .contains("too short")
        );
        assert!(
            parse_read_ref("ch_12345678:m1..")
                .unwrap_err()
                .to_string()
                .contains("open-ended")
        );
        assert!(
            parse_read_ref("ch_12345678:m..m2")
                .unwrap_err()
                .to_string()
                .contains("invalid message ref")
        );
        assert!(
            parse_read_ref("ch_12345678:m3..m2")
                .unwrap_err()
                .to_string()
                .contains("start must be before end")
        );
        assert!(
            parse_read_ref("ch_12345678:1")
                .unwrap_err()
                .to_string()
                .contains("expected mN")
        );
    }

    #[test]
    fn validates_focus_inside_read_ranges() {
        let reads = vec![parse_read_ref("ch_12345678:m2..m5").unwrap()];
        let resolved = ResolvedConversation {
            key: key("project-a", "one.jsonl"),
            reference: key("project-a", "one.jsonl").conversation_ref(),
        };
        let resolved_reads = vec![(reads[0].clone(), resolved)];
        validate_resolved_focus_in_ranges(&resolved_reads, &parse_focus_ref("m3").unwrap(), None)
            .unwrap();
        let err = validate_resolved_focus_in_ranges(
            &resolved_reads,
            &parse_focus_ref("m6").unwrap(),
            None,
        )
        .unwrap_err();
        assert!(err.to_string().contains("outside"));
    }

    #[test]
    fn bare_focus_is_rejected_for_multiple_conversations() {
        let reads = vec![
            parse_read_ref("ch_12345678:m1..m5").unwrap(),
            parse_read_ref("ch_87654321:m1..m5").unwrap(),
        ];
        let first = ResolvedConversation {
            key: key("project-a", "one.jsonl"),
            reference: key("project-a", "one.jsonl").conversation_ref(),
        };
        let second = ResolvedConversation {
            key: key("project-b", "two.jsonl"),
            reference: key("project-b", "two.jsonl").conversation_ref(),
        };
        let resolved_reads = vec![(reads[0].clone(), first), (reads[1].clone(), second)];
        let err = validate_resolved_focus_in_ranges(
            &resolved_reads,
            &parse_focus_ref("m2").unwrap(),
            None,
        )
        .unwrap_err();
        assert!(err.to_string().contains("bare focus is ambiguous"));
    }
}
