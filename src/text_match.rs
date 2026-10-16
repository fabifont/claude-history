use crate::case_fold::{drops_following_dot_above, push_folded};
use icu_normalizer::ComposingNormalizerBorrowed;
use std::borrow::Cow;

fn is_cjk_punctuation(c: char) -> bool {
    matches!(
        c,
        '\u{3000}'
            | '\u{3001}'
            | '\u{3002}'
            | '\u{3008}'
            | '\u{3009}'
            | '\u{300A}'
            | '\u{300B}'
            | '\u{300C}'
            | '\u{300D}'
            | '\u{300E}'
            | '\u{300F}'
            | '\u{3010}'
            | '\u{3011}'
            | '\u{3014}'
            | '\u{3015}'
            | '\u{3016}'
            | '\u{3017}'
            | '\u{FF01}'
            | '\u{FF08}'
            | '\u{FF09}'
            | '\u{FF0C}'
            | '\u{FF1A}'
            | '\u{FF1B}'
            | '\u{FF1F}'
            | '\u{201C}'
            | '\u{201D}'
            | '\u{2018}'
            | '\u{2019}'
            | '\u{2014}'
            | '\u{2026}'
            | '\u{00B7}'
    )
}

/// `text` in Unicode Normalization Form C, so composed and decomposed
/// accents compare equal.
pub fn nfc(text: &str) -> Cow<'_, str> {
    ComposingNormalizerBorrowed::new_nfc().normalize(text)
}

/// The form search terms and texts compare in: NFC, separators as spaces,
/// and full case folding (see `case_fold`).
pub fn normalize_for_search(text: &str) -> String {
    let text = nfc(text);
    let mut out = String::with_capacity(text.len());
    text.chars().for_each(|ch| push_normalized(&mut out, ch));
    out
}

fn push_normalized(out: &mut String, ch: char) {
    if is_word_separator(ch) {
        out.push(' ');
    } else {
        push_folded(out, ch);
    }
}

/// Byte ranges of the occurrences of `word` in `text`, both search-normalized,
/// that keyword search counts: those at a word start, or all of them when
/// `anywhere` (the substring fallback of queries with CJK text). Occurrences
/// do not overlap.
pub fn word_match_ranges(text: &str, word: &str, anywhere: bool) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    if word.is_empty() {
        return ranges;
    }
    let mut start = 0;
    while let Some(pos) = text[start..].find(word) {
        let at = start + pos;
        if anywhere || is_word_start(text, at) {
            ranges.push((at, at + word.len()));
            start = at + word.len();
        } else {
            start = at + text[at..].chars().next().map_or(1, char::len_utf8);
        }
    }
    ranges
}

/// A text in NFC with its search-normalized form, for finding query words
/// the way keyword search does and mapping the matches back to the text.
pub struct NormalizedText<'a> {
    raw: &'a str,
    normalized: String,
}

impl<'a> NormalizedText<'a> {
    /// `raw` should be in NFC, as loaded conversation texts are; the
    /// normalized form is then `normalize_for_search(raw)`.
    pub fn new(raw: &'a str) -> Self {
        let mut normalized = String::with_capacity(raw.len());
        raw.chars()
            .for_each(|ch| push_normalized(&mut normalized, ch));
        Self { raw, normalized }
    }

    pub fn raw(&self) -> &'a str {
        self.raw
    }

    pub fn normalized(&self) -> &str {
        &self.normalized
    }

    /// Byte ranges in the raw text, sorted, of the matches of the words of
    /// `query` (search-normalized, words separated by spaces). Words match
    /// at word starts, or anywhere when the query has CJK text, as in
    /// keyword search.
    pub fn query_ranges(&self, query: &str) -> Vec<(usize, usize)> {
        let anywhere = contains_cjk(query);
        let mut ranges: Vec<(usize, usize)> = query
            .split_whitespace()
            .flat_map(|word| word_match_ranges(&self.normalized, word, anywhere))
            .collect();
        ranges.sort_unstable();
        self.raw_ranges(&ranges)
    }

    /// Byte ranges in the raw text of the chars that `ranges`, byte ranges in
    /// the normalized form, came from.
    pub fn raw_ranges(&self, ranges: &[(usize, usize)]) -> Vec<(usize, usize)> {
        // Each range's start and last byte, in normalized order; even
        // entries are starts and odd ones ends.
        let mut points: Vec<(usize, usize)> = ranges
            .iter()
            .enumerate()
            .flat_map(|(i, &(start, end))| [(start, 2 * i), (end.max(start + 1) - 1, 2 * i + 1)])
            .collect();
        points.sort_unstable();
        let mut mapped = vec![self.raw.len(); points.len()];
        let mut points = points.into_iter().peekable();
        let mut folded = String::new();
        let mut at = 0;
        for (raw_at, ch) in self.raw.char_indices() {
            // Folding a char depends on the char folded before it.
            let previous = folded.chars().next_back();
            folded.clear();
            folded.extend(previous);
            let carried = folded.len();
            push_normalized(&mut folded, ch);
            let next = at + folded.len() - carried;
            while let Some(&(point, index)) = points.peek()
                && point < next
            {
                mapped[index] = if index % 2 == 0 {
                    raw_at
                } else {
                    raw_at + ch.len_utf8()
                };
                points.next();
            }
            at = next;
        }
        mapped.chunks(2).map(|pair| (pair[0], pair[1])).collect()
    }
}

pub fn is_word_separator(c: char) -> bool {
    c.is_whitespace() || c == '_' || c == '-' || c == '/' || is_cjk_punctuation(c)
}

/// Whether `pos` starts a word: it follows nothing or a non-alphanumeric
/// char. A combining dot above that folding drops (see `case_fold`) belongs
/// to the word before it, so raw and normalized text agree on word starts.
pub fn is_word_start(text: &str, pos: usize) -> bool {
    let mut before = text[..pos].chars().rev();
    match before.next() {
        None => true,
        Some('\u{307}') => !before.next().is_some_and(drops_following_dot_above),
        Some(c) => !c.is_alphanumeric(),
    }
}

/// ASCII chars that a non-ASCII char lowercases to, with that char's first
/// UTF-8 byte: `İ` (U+0130) and the Kelvin sign (U+212A).
const ASCII_FROM_NON_ASCII: [(char, u8); 2] = [('i', 0xC4), ('k', 0xE2)];

/// Byte offset of the first char at or after `from` whose lowercase form
/// may start with `first`, a lowercase char. Every char whose
/// `to_lowercase()` starts with `first` is found; some others are too, so
/// callers check each one. Offsets are char boundaries.
pub fn next_lowercase_candidate(text: &str, from: usize, first: char) -> Option<usize> {
    let haystack = &text.as_bytes()[from..];
    let found = if first.is_ascii() {
        let lower = first as u8;
        let upper = first.to_ascii_uppercase() as u8;
        match ASCII_FROM_NON_ASCII
            .iter()
            .find(|(ascii, _)| *ascii == first)
        {
            Some(&(_, lead)) => memchr::memchr3(lower, upper, lead, haystack),
            None => memchr::memchr2(lower, upper, haystack),
        }
    } else {
        // Only non-ASCII chars lowercase to non-ASCII ones.
        haystack.iter().position(|&byte| byte >= 0xC0)
    };
    found.map(|offset| from + offset)
}

pub fn contains_prefix_match(text: &str, word: &str) -> bool {
    let mut start = 0;
    while let Some(pos) = text[start..].find(word) {
        let actual_pos = start + pos;
        if is_word_start(text, actual_pos) {
            return true;
        }
        start = actual_pos + text[actual_pos..].chars().next().map_or(1, char::len_utf8);
    }
    false
}

/// Returns `s` cut to at most `max` bytes on a char boundary.
pub fn truncate_on_char_boundary(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

pub fn contains_cjk(text: &str) -> bool {
    text.chars().any(|c| ('\u{4E00}'..='\u{9FFF}').contains(&c))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_listed_non_ascii_chars_lowercase_to_ascii() {
        for c in (0x80..=0x10FFFF).filter_map(char::from_u32) {
            let Some(lower) = c.to_lowercase().next().filter(char::is_ascii) else {
                continue;
            };
            let mut utf8 = [0; 4];
            c.encode_utf8(&mut utf8);
            assert!(
                ASCII_FROM_NON_ASCII.contains(&(lower, utf8[0])),
                "{c:?} lowercases to {lower:?}"
            );
        }
    }

    #[test]
    fn lowercase_candidates_include_every_char_folding_to_first() {
        let text = "aİbKcIkéÉ_x测\u{212A}i";
        for first in ['i', 'k', 'é', 'x', '测', '_'] {
            let mut from = 0;
            let mut found = Vec::new();
            while let Some(at) = next_lowercase_candidate(text, from, first) {
                found.push(at);
                from = at + text[at..].chars().next().unwrap().len_utf8();
            }
            for (at, c) in text.char_indices() {
                if c.to_lowercase().next() == Some(first) {
                    assert!(found.contains(&at), "{first:?} misses {c:?}");
                }
            }
        }
    }

    #[test]
    fn normalizes_search_separators_and_case() {
        assert_eq!(
            normalize_for_search("HARDENED_RUNTIME/main-worktree"),
            "hardened runtime main worktree"
        );
    }

    #[test]
    fn normalizes_composition_and_case_folding() {
        assert_eq!(
            normalize_for_search("Cafe\u{301} İstanbul STRAẞE ΛΟΓΟΣ"),
            normalize_for_search("café istanbul strasse λογος")
        );
    }

    #[test]
    fn normalized_text_maps_matches_back_to_raw_chars() {
        let raw = "İstanbul ﬁle STRAẞE i\u{307}stanbul ok";
        let text = NormalizedText::new(raw);
        assert_eq!(text.normalized, normalize_for_search(raw));
        let found = |query: &str| {
            text.query_ranges(query)
                .into_iter()
                .map(|(start, end)| &raw[start..end])
                .collect::<Vec<_>>()
        };
        assert_eq!(found("istanbul"), ["İstanbul", "i\u{307}stanbul"]);
        assert_eq!(found("stanbul"), Vec::<&str>::new());
        assert_eq!(found("fi"), ["ﬁ"]);
        assert_eq!(found("strasse ok"), ["STRAẞE", "ok"]);
        assert_eq!(found("ss"), Vec::<&str>::new());
    }

    #[test]
    fn word_matches_follow_word_starts_unless_anywhere() {
        assert_eq!(word_match_ranges("v1.1.1 x", "1.1", false), [(3, 6)]);
        assert_eq!(word_match_ranges("aab ab", "ab", false), [(4, 6)]);
        assert_eq!(word_match_ranges("aab ab", "ab", true), [(1, 3), (4, 6)]);
    }

    #[test]
    fn normalizes_cjk_punctuation() {
        assert_eq!(
            normalize_for_search("SIGTERM\u{FF0C}\u{5C5E}\u{4E8E}"),
            "sigterm \u{5C5E}\u{4E8E}"
        );
    }

    #[test]
    fn prefix_match_respects_word_start() {
        assert!(contains_prefix_match("redaction plan", "red"));
        assert!(!contains_prefix_match("fired plan", "red"));
        assert!(contains_prefix_match("bump to v1.1.1", "1.1"));
        assert!(!is_word_start("i\u{307}stanbul", 3));
        assert!(!is_word_start("\u{FB01}\u{307}x", 5));
        assert!(is_word_start("x\u{307}stanbul", 3));
    }
}
