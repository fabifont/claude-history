//! What semantic search reads about a conversation, stored per card in a
//! generation's `sem-NNNN.zst` segments, with the parse errors the agent
//! commands report.
//!
//! The data is only read by semantic search and the commands that load
//! whole conversations, so list and lexical search never map these
//! segments.

use crate::agent::refs::MessageRange;
use crate::history::{Conversation, ParseError};
use serde::{Deserialize, Serialize};

/// `Conversation::semantic_turns` with their message ranges, plus the
/// route text, agent search text and parse errors.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct SemanticDoc {
    pub turns: Vec<String>,
    pub ranges: Vec<MessageRange>,
    pub route_text: String,
    pub agent_text: String,
    pub parse_errors: Vec<ParseError>,
}

impl SemanticDoc {
    pub fn of(conversation: &Conversation) -> Self {
        Self {
            turns: conversation.semantic_turns.to_vec(),
            ranges: conversation.semantic_turn_ranges.clone(),
            route_text: conversation.semantic_route_text.to_string(),
            agent_text: conversation.agent_search_text.to_string(),
            parse_errors: conversation.parse_errors.clone(),
        }
    }

    /// Whether there is nothing to store.
    pub fn is_empty(&self) -> bool {
        self.turns.is_empty()
            && self.ranges.is_empty()
            && self.route_text.is_empty()
            && self.agent_text.is_empty()
            && self.parse_errors.is_empty()
    }

    /// Encoded form stored in a frame; empty data encodes to no bytes.
    pub fn encode(&self) -> Vec<u8> {
        if self.is_empty() {
            return Vec::new();
        }
        bincode::serialize(self).expect("semantic data serializes")
    }

    /// Sets the semantic fields of `conversation` to this data.
    pub fn apply(self, conversation: &mut Conversation) {
        conversation.semantic_turns = self.turns.into();
        conversation.semantic_turn_ranges = self.ranges;
        conversation.semantic_route_text = self.route_text.into();
        conversation.agent_search_text = self.agent_text.into();
        conversation.parse_errors = self.parse_errors;
    }

    /// Reads `encode` output. No bytes decode to empty data.
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
            parse_errors: vec![ParseError {
                line_number: 2,
                line_content: "{".into(),
                error_message: "EOF".into(),
                context_before: vec!["{}".into()],
                context_after: Vec::new(),
            }],
        };
        let bytes = doc.encode();
        assert_ne!(hash(&bytes), 0);
        assert_eq!(SemanticDoc::decode(&bytes), Some(doc));
        assert_eq!(SemanticDoc::decode(b"\xff"), None);
    }
}
