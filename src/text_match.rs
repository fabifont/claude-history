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

pub fn normalize_for_search(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        if is_word_separator(ch) {
            out.push(' ');
        } else {
            out.extend(ch.to_lowercase());
        }
    }
    out
}

pub fn is_word_separator(c: char) -> bool {
    c.is_whitespace() || c == '_' || c == '-' || c == '/' || is_cjk_punctuation(c)
}

pub fn is_word_start(text: &str, pos: usize) -> bool {
    pos == 0
        || text[..pos]
            .chars()
            .next_back()
            .is_some_and(|c| !c.is_alphanumeric())
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
    }
}
