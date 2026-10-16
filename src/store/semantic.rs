//! What semantic search reads about a conversation, stored per card in a
//! generation's `sem-NNNN.zst` segments.
//!
//! The data is only read by semantic search and the commands that rank or
//! route conversations by meaning, so list and lexical search never map
//! these segments.

use crate::agent::refs::MessageRange;
use crate::history::Conversation;
use serde::{Deserialize, Serialize};

/// `Conversation::semantic_turns` with their message ranges, plus the
/// route text and agent search text.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct SemanticDoc {
    pub turns: Vec<String>,
    pub ranges: Vec<MessageRange>,
    pub route_text: String,
    pub agent_text: String,
}

impl SemanticDoc {
    pub fn of(conversation: &Conversation) -> Self {
        Self {
            turns: conversation.semantic_turns.to_vec(),
            ranges: conversation.semantic_turn_ranges.clone(),
            route_text: conversation.semantic_route_text.to_string(),
            agent_text: conversation.agent_search_text.to_string(),
        }
    }

    /// Whether there is nothing to store.
    pub fn is_empty(&self) -> bool {
        self.turns.is_empty()
            && self.ranges.is_empty()
            && self.route_text.is_empty()
            && self.agent_text.is_empty()
    }

    /// Encoded form stored in a frame; empty data encodes to no bytes.
    pub fn encode(&self) -> Vec<u8> {
        if self.is_empty() {
            return Vec::new();
        }
        bincode::serialize(self).expect("semantic data serializes")
    }

    /// Reads `encode` output. No bytes decode to empty data.
    #[cfg(test)]
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        if bytes.is_empty() {
            return Some(Self::default());
        }
        bincode::deserialize(bytes).ok()
    }
}

/// `SemRow::hash` of encoded data: 0 for none.
pub fn hash(encoded: &[u8]) -> u64 {
    if encoded.is_empty() {
        return 0;
    }
    super::generation::hash64(encoded)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_and_empty_data_has_no_bytes() {
        assert!(SemanticDoc::default().encode().is_empty());
        assert_eq!(SemanticDoc::decode(&[]), Some(SemanticDoc::default()));
        assert_eq!(hash(&[]), 0);
        let doc = SemanticDoc {
            turns: vec!["first turn".into(), "second é".into()],
            ranges: vec![MessageRange::single(1), MessageRange::single(3)],
            route_text: "route".into(),
            agent_text: String::new(),
        };
        let bytes = doc.encode();
        assert_ne!(hash(&bytes), 0);
        assert_eq!(SemanticDoc::decode(&bytes), Some(doc));
        assert_eq!(SemanticDoc::decode(b"\xff"), None);
    }
}
