use crate::history::Conversation;
use crate::text_match::{is_word_start, next_lowercase_candidate, nfc};
use rayon::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaseMode {
    Sensitive,
    Insensitive,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Literal {
    text: String,
    case_mode: CaseMode,
}

impl Literal {
    pub fn new(text: String) -> Self {
        // Conversation texts are NFC (see `history::index`).
        let text = nfc(&text).into_owned();
        let case_mode = if text.chars().any(char::is_uppercase) {
            CaseMode::Sensitive
        } else {
            CaseMode::Insensitive
        };
        Self { text, case_mode }
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn case_mode(&self) -> CaseMode {
        self.case_mode
    }

    /// Whether the literal occurs in `text` starting at a word start.
    pub fn matches(&self, text: &str) -> bool {
        !self.find_ranges(text, true).is_empty()
    }

    /// Byte ranges of the non-overlapping occurrences in `text` that start at
    /// a word start.
    pub fn match_ranges(&self, text: &str) -> Vec<(usize, usize)> {
        self.find_ranges(text, false)
    }

    /// Occurrences start at a word start: the start of `text` or after a
    /// non-alphanumeric character, so `_`, `.`, `/`, `-` and whitespace all
    /// begin words and `streaming` matches in `load_streaming` but not in
    /// `livestreaming`.
    fn find_ranges(&self, text: &str, first_only: bool) -> Vec<(usize, usize)> {
        if self.text.is_empty() {
            return Vec::new();
        }
        match self.case_mode {
            CaseMode::Sensitive => find_word_start_ranges(text, &self.text, first_only),
            CaseMode::Insensitive => find_case_insensitive_ranges(text, &self.text, first_only),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LiteralCorpusEntry {
    pub index: usize,
    pub text: String,
}

pub fn build_literal_corpus(conversations: &[Conversation]) -> Vec<LiteralCorpusEntry> {
    build_literal_corpus_with(conversations, false)
}

pub fn build_agent_literal_corpus(conversations: &[Conversation]) -> Vec<LiteralCorpusEntry> {
    build_literal_corpus_with(conversations, true)
}

fn build_literal_corpus_with(
    conversations: &[Conversation],
    include_agent_text: bool,
) -> Vec<LiteralCorpusEntry> {
    conversations
        .par_iter()
        .enumerate()
        .map(|(index, conversation)| LiteralCorpusEntry {
            index,
            text: literal_text(conversation, include_agent_text),
        })
        .collect()
}

pub fn matches_all_literals(text: &str, literals: &[Literal]) -> bool {
    literals.iter().all(|literal| literal.matches(text))
}

pub fn conversation_matches_all_literals(
    conversation: &Conversation,
    literals: &[Literal],
) -> bool {
    matches_all_literals(&literal_text(conversation, false), literals)
}

pub fn match_literal_ranges(text: &str, literals: &[Literal]) -> Vec<(usize, usize)> {
    literals
        .iter()
        .flat_map(|literal| literal.match_ranges(text))
        .collect()
}

pub fn exact_fallback(
    conversations: &[Conversation],
    corpus: &[LiteralCorpusEntry],
    literals: &[Literal],
    scope: impl Fn(usize) -> bool + Sync,
) -> Vec<usize> {
    if literals.is_empty() {
        return Vec::new();
    }

    let mut matches = corpus
        .par_iter()
        .filter(|entry| scope(entry.index) && matches_all_literals(&entry.text, literals))
        .map(|entry| (entry.index, conversations[entry.index].timestamp))
        .collect::<Vec<_>>();

    matches.sort_unstable_by(|a, b| b.1.cmp(&a.1));
    matches.into_iter().map(|(index, _)| index).collect()
}

fn find_word_start_ranges(text: &str, needle: &str, first_only: bool) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let mut start = 0;
    while let Some(pos) = text[start..].find(needle) {
        let range_start = start + pos;
        if is_word_start(text, range_start) {
            let range_end = range_start + needle.len();
            ranges.push((range_start, range_end));
            if first_only {
                break;
            }
            start = range_end;
        } else {
            start = range_start + text[range_start..].chars().next().map_or(1, char::len_utf8);
        }
    }
    ranges
}

fn find_case_insensitive_ranges(text: &str, needle: &str, first_only: bool) -> Vec<(usize, usize)> {
    let needle_chars: Vec<char> = needle.chars().flat_map(char::to_lowercase).collect();
    let Some(&first) = needle_chars.first() else {
        return Vec::new();
    };

    let mut ranges = Vec::new();
    let mut from = 0;
    while let Some(start) = next_lowercase_candidate(text, from, first) {
        let rest = &text[start..];
        let end = (is_word_start(text, start)
            && rest.chars().next().and_then(|ch| ch.to_lowercase().next()) == Some(first))
        .then(|| folded_match_end(rest, &needle_chars))
        .flatten();
        match end {
            Some(end) => {
                ranges.push((start, start + end));
                if first_only {
                    break;
                }
                from = start + end;
            }
            None => from = start + rest.chars().next().map_or(1, char::len_utf8),
        }
    }
    ranges
}

/// Byte length of the shortest prefix of `text` whose lowercase folding
/// starts with `needle`, if any.
fn folded_match_end(text: &str, needle: &[char]) -> Option<usize> {
    let mut matched = 0;
    for (offset, ch) in text.char_indices() {
        for folded in ch.to_lowercase() {
            if folded != needle[matched] {
                return None;
            }
            matched += 1;
            if matched == needle.len() {
                return Some(offset + ch.len_utf8());
            }
        }
    }
    None
}

fn literal_text(conversation: &Conversation, include_agent_text: bool) -> String {
    let mut text = String::new();
    push_part(&mut text, Some(&conversation.full_text));
    if include_agent_text {
        push_part(&mut text, Some(&conversation.agent_search_text));
    }
    push_part(&mut text, conversation.project_name.as_deref());
    text
}

fn push_part(text: &mut String, part: Option<&str>) {
    let Some(part) = part else {
        return;
    };
    if part.is_empty() {
        return;
    }
    if !text.is_empty() {
        text.push(' ');
    }
    text.push_str(part);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text_match::starts_word;

    /// Char-by-char reference for `find_case_insensitive_ranges`.
    fn reference_case_insensitive_ranges(text: &str, needle: &str) -> Vec<(usize, usize)> {
        let needle_chars: Vec<char> = needle.chars().flat_map(char::to_lowercase).collect();
        let Some(&first) = needle_chars.first() else {
            return Vec::new();
        };
        let mut ranges = Vec::new();
        let mut resume = 0;
        let mut prev = None;
        for (start, ch) in text.char_indices() {
            let word_start = prev.is_none_or(|prev| starts_word(prev, Some(ch)));
            prev = Some(ch);
            if start < resume || !word_start || ch.to_lowercase().next() != Some(first) {
                continue;
            }
            if let Some(end) = folded_match_end(&text[start..], &needle_chars) {
                ranges.push((start, start + end));
                resume = start + end;
            }
        }
        ranges
    }

    #[test]
    fn case_insensitive_ranges_match_the_char_by_char_reference() {
        const PIECES: &[&str] = &[
            "a", "A", "b", "ab", "AB", "İ", "i", "I", "K", "\u{212A}", "k", "é", "É", "_", "-",
            " ", ".", "测", "测试", "ß", "ǅ", "1",
        ];
        let mut state = 0x2545_F491_4F6C_DD1D_u64;
        let mut next = |n: usize| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % n as u64) as usize
        };
        for _ in 0..3000 {
            let text: String = (0..next(40)).map(|_| PIECES[next(PIECES.len())]).collect();
            let needle: String = (0..1 + next(3))
                .map(|_| PIECES[next(PIECES.len())])
                .collect();
            assert_eq!(
                find_case_insensitive_ranges(&text, &needle, false),
                reference_case_insensitive_ranges(&text, &needle),
                "{text:?} {needle:?}"
            );
        }
    }
    use crate::search::test_fixtures::one_message_conversation;
    use chrono::{Duration, Local};

    fn make_conv_full(
        text: &str,
        project: Option<&str>,
        title: Option<&str>,
        summary: Option<&str>,
        timestamp: chrono::DateTime<Local>,
    ) -> Conversation {
        one_message_conversation(text, timestamp, summary, title, project)
    }

    #[test]
    fn literal_uses_smart_case() {
        assert!(Literal::new("exact_phrase".to_string()).matches("EXACT_PHRASE"));
        assert!(!Literal::new("Exact_Phrase".to_string()).matches("exact_phrase"));
        assert!(Literal::new("Exact_Phrase".to_string()).matches("Exact_Phrase"));
    }

    #[test]
    fn literal_ranges_use_smart_case() {
        let insensitive = Literal::new("exact_phrase".to_string());
        let sensitive = Literal::new("Exact_Phrase".to_string());

        assert_eq!(insensitive.match_ranges("EXACT_PHRASE"), vec![(0, 12)]);
        assert_eq!(sensitive.match_ranges("exact_phrase"), Vec::new());
        assert_eq!(sensitive.match_ranges("Exact_Phrase"), vec![(0, 12)]);
    }

    #[test]
    fn insensitive_literal_ranges_are_original_text_boundaries() {
        let text = "pre İSTANBUL post";
        let literal = Literal::new("i\u{307}stanbul".to_string());
        let ranges = literal.match_ranges(text);

        assert_eq!(ranges, vec![(4, 13)]);
        assert_eq!(&text[ranges[0].0..ranges[0].1], "İSTANBUL");
    }

    #[test]
    fn insensitive_literal_match_uses_smart_case() {
        let literal = Literal::new("i\u{307}stanbul".to_string());

        assert!(literal.matches("pre İSTANBUL post"));
        assert!(!literal.matches("pre constantinople post"));
    }

    #[test]
    fn literal_matches_only_at_word_starts() {
        for case in [
            Literal::new("streaming".to_string()),
            Literal::new("Streaming".to_string()),
        ] {
            assert!(!case.matches("LiveStreaming livestreaming"));
            assert_eq!(case.match_ranges("liveStreaming"), Vec::new());
        }

        let literal = Literal::new("conversations_streaming".to_string());
        assert!(literal.matches("load_all_conversations_streaming()"));
        assert!(!literal.matches("subconversations_streaming"));
        for text in [
            "a_streaming",
            "a.streaming",
            "a/streaming",
            "a-streaming",
            "streaming",
        ] {
            assert!(
                Literal::new("streaming".to_string()).matches(text),
                "{text}"
            );
        }
        assert!(Literal::new("ünchen".to_string()).matches("(ünchen)"));
        assert!(!Literal::new("nchen".to_string()).matches("münchen"));
    }

    #[test]
    fn literal_matches_at_letter_digit_transitions() {
        for (needle, text) in [
            ("1.2.3", "bump v1.2.3"),
            ("264", "H264"),
            ("V", "2V"),
            ("8", "UTF8"),
        ] {
            let literal = Literal::new(needle.to_string());
            assert!(literal.matches(text), "{needle:?} in {text:?}");
        }
        assert_eq!(
            Literal::new("264".to_string()).match_ranges("h264 x264"),
            [(1, 4), (6, 9)]
        );
        for (needle, text) in [
            ("Config", "AppConfig"),
            ("config", "appconfig"),
            ("64", "h264"),
        ] {
            assert!(
                !Literal::new(needle.to_string()).matches(text),
                "{needle:?} in {text:?}"
            );
        }
    }

    #[test]
    fn literal_ranges_skip_mid_word_occurrences() {
        let literal = Literal::new("log".to_string());

        assert_eq!(
            literal.match_ranges("catalog log blog.log"),
            vec![(8, 11), (17, 20)]
        );
        assert_eq!(
            Literal::new("Log".to_string()).match_ranges("catalog Log blogLog"),
            vec![(8, 11)]
        );
    }

    #[test]
    fn corpus_preserves_punctuation_and_metadata() {
        let now = Local::now();
        let conversations = vec![make_conv_full(
            "body_with_under_score and punctuation: yes",
            Some("project_name/raw-value"),
            Some("Title: Raw_Value"),
            Some("Summary.with punctuation"),
            now,
        )];

        let corpus = build_literal_corpus(&conversations);

        assert!(corpus[0].text.contains("body_with_under_score"));
        assert!(corpus[0].text.contains("punctuation: yes"));
        assert!(corpus[0].text.contains("project_name/raw-value"));
        assert!(corpus[0].text.contains("Title: Raw_Value"));
        assert!(corpus[0].text.contains("Summary.with punctuation"));
    }

    #[test]
    fn corpus_does_not_duplicate_title_and_summary() {
        let now = Local::now();
        let conversations = vec![make_conv_full(
            "body sentinel",
            None,
            Some("TitleA"),
            Some("SummaryB"),
            now,
        )];

        let corpus = build_literal_corpus(&conversations);

        assert!(!corpus[0].text.contains("SummaryB TitleA"));
    }

    #[test]
    fn exact_fallback_returns_scoped_matches_newest_first() {
        let now = Local::now();
        let conversations = vec![
            make_conv_full("needle phrase", None, None, None, now - Duration::days(1)),
            make_conv_full("needle phrase", None, None, None, now),
            make_conv_full("needle phrase", None, None, None, now - Duration::hours(1)),
        ];
        let corpus = build_literal_corpus(&conversations);
        let literal = Literal::new("needle phrase".to_string());

        let results = exact_fallback(&conversations, &corpus, &[literal], |index| index != 1);

        assert_eq!(results, vec![2, 0]);
    }

    #[test]
    fn exact_fallback_requires_all_literals() {
        let now = Local::now();
        let conversations = vec![
            make_conv_full("alpha beta", None, None, None, now),
            make_conv_full("alpha gamma", None, None, None, now),
        ];
        let corpus = build_literal_corpus(&conversations);
        let literals = vec![
            Literal::new("alpha".to_string()),
            Literal::new("beta".to_string()),
        ];

        let results = exact_fallback(&conversations, &corpus, &literals, |_| true);

        assert_eq!(results, vec![0]);
    }
}
