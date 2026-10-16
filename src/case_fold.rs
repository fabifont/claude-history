//! Full Unicode case folding (CaseFolding.txt statuses C and F), applied one
//! char at a time to text in Normalization Form C.
//!
//! Folding agrees with `char::to_lowercase` except for the chars in `FOLDS`,
//! the Cherokee letters, which fold to their uppercase forms, and `İ`, which
//! folds to a plain `i` so `istanbul` finds `İstanbul`. A combining dot above
//! (U+0307) directly after a folded `i` or `j` is dropped, which makes the
//! lowercase form of `İ` (`i` + U+0307) fold the same way.

/// Appends the case folding of `ch` to `out`.
pub fn push_folded(out: &mut String, ch: char) {
    if ch.is_ascii() {
        out.push(ch.to_ascii_lowercase());
        return;
    }
    if let Some(&folded) = two_byte_folds().get(ch as usize - 0x80)
        && folded != '\0'
    {
        out.push(folded);
        return;
    }
    push_folded_uncached(out, ch);
}

/// The folding of each two-byte UTF-8 char (U+0080 to U+07FF) that folds to
/// one char regardless of the text before it, or `'\0'`.
fn two_byte_folds() -> &'static [char] {
    static FOLDED: std::sync::OnceLock<Vec<char>> = std::sync::OnceLock::new();
    FOLDED.get_or_init(|| {
        ('\u{80}'..='\u{7FF}')
            .map(|ch| {
                if ch == '\u{307}' {
                    return '\0';
                }
                let mut folded = String::new();
                push_folded_uncached(&mut folded, ch);
                let mut chars = folded.chars();
                match (chars.next(), chars.next()) {
                    (Some(folded), None) => folded,
                    _ => '\0',
                }
            })
            .collect()
    })
}

fn push_folded_uncached(out: &mut String, ch: char) {
    match ch {
        // Kana, CJK ideographs and Hangul syllables have no case.
        '\u{3040}'..='\u{9FFF}' | '\u{AC00}'..='\u{D7A3}' => out.push(ch),
        '\u{130}' => out.push('i'),
        '\u{307}' if out.ends_with(['i', 'j']) => {}
        '\u{13A0}'..='\u{13F5}' => out.push(ch),
        '\u{13F8}'..='\u{13FD}' => out.push(offset(ch, -8)),
        '\u{AB70}'..='\u{ABBF}' => out.push(offset(ch, 0x13A0 - 0xAB70)),
        _ => match FOLDS.binary_search_by_key(&ch, |&(from, _)| from) {
            Ok(index) => out.push_str(FOLDS[index].1),
            Err(_) => out.extend(ch.to_lowercase()),
        },
    }
}

/// Whether a combining dot above (U+0307) right after `ch` folds away: the
/// folding of `ch` ends with `i` or `j`.
pub fn drops_following_dot_above(ch: char) -> bool {
    let mut folded = String::new();
    push_folded(&mut folded, ch);
    folded.ends_with(['i', 'j'])
}

fn offset(ch: char, by: i32) -> char {
    char::from_u32((ch as i32 + by) as u32).unwrap_or(ch)
}

/// Chars whose full case folding differs from their lowercase form, other
/// than `İ` and Cherokee, sorted by char.
const FOLDS: &[(char, &str)] = &[
    ('\u{B5}', "\u{3BC}"),
    ('\u{DF}', "ss"),
    ('\u{149}', "\u{2BC}n"),
    ('\u{17F}', "s"),
    ('\u{1F0}', "j\u{30C}"),
    ('\u{345}', "\u{3B9}"),
    ('\u{390}', "\u{3B9}\u{308}\u{301}"),
    ('\u{3B0}', "\u{3C5}\u{308}\u{301}"),
    ('\u{3C2}', "\u{3C3}"),
    ('\u{3D0}', "\u{3B2}"),
    ('\u{3D1}', "\u{3B8}"),
    ('\u{3D5}', "\u{3C6}"),
    ('\u{3D6}', "\u{3C0}"),
    ('\u{3F0}', "\u{3BA}"),
    ('\u{3F1}', "\u{3C1}"),
    ('\u{3F5}', "\u{3B5}"),
    ('\u{587}', "\u{565}\u{582}"),
    ('\u{1C80}', "\u{432}"),
    ('\u{1C81}', "\u{434}"),
    ('\u{1C82}', "\u{43E}"),
    ('\u{1C83}', "\u{441}"),
    ('\u{1C84}', "\u{442}"),
    ('\u{1C85}', "\u{442}"),
    ('\u{1C86}', "\u{44A}"),
    ('\u{1C87}', "\u{463}"),
    ('\u{1C88}', "\u{A64B}"),
    ('\u{1E96}', "h\u{331}"),
    ('\u{1E97}', "t\u{308}"),
    ('\u{1E98}', "w\u{30A}"),
    ('\u{1E99}', "y\u{30A}"),
    ('\u{1E9A}', "a\u{2BE}"),
    ('\u{1E9B}', "\u{1E61}"),
    ('\u{1E9E}', "ss"),
    ('\u{1F50}', "\u{3C5}\u{313}"),
    ('\u{1F52}', "\u{3C5}\u{313}\u{300}"),
    ('\u{1F54}', "\u{3C5}\u{313}\u{301}"),
    ('\u{1F56}', "\u{3C5}\u{313}\u{342}"),
    ('\u{1F80}', "\u{1F00}\u{3B9}"),
    ('\u{1F81}', "\u{1F01}\u{3B9}"),
    ('\u{1F82}', "\u{1F02}\u{3B9}"),
    ('\u{1F83}', "\u{1F03}\u{3B9}"),
    ('\u{1F84}', "\u{1F04}\u{3B9}"),
    ('\u{1F85}', "\u{1F05}\u{3B9}"),
    ('\u{1F86}', "\u{1F06}\u{3B9}"),
    ('\u{1F87}', "\u{1F07}\u{3B9}"),
    ('\u{1F88}', "\u{1F00}\u{3B9}"),
    ('\u{1F89}', "\u{1F01}\u{3B9}"),
    ('\u{1F8A}', "\u{1F02}\u{3B9}"),
    ('\u{1F8B}', "\u{1F03}\u{3B9}"),
    ('\u{1F8C}', "\u{1F04}\u{3B9}"),
    ('\u{1F8D}', "\u{1F05}\u{3B9}"),
    ('\u{1F8E}', "\u{1F06}\u{3B9}"),
    ('\u{1F8F}', "\u{1F07}\u{3B9}"),
    ('\u{1F90}', "\u{1F20}\u{3B9}"),
    ('\u{1F91}', "\u{1F21}\u{3B9}"),
    ('\u{1F92}', "\u{1F22}\u{3B9}"),
    ('\u{1F93}', "\u{1F23}\u{3B9}"),
    ('\u{1F94}', "\u{1F24}\u{3B9}"),
    ('\u{1F95}', "\u{1F25}\u{3B9}"),
    ('\u{1F96}', "\u{1F26}\u{3B9}"),
    ('\u{1F97}', "\u{1F27}\u{3B9}"),
    ('\u{1F98}', "\u{1F20}\u{3B9}"),
    ('\u{1F99}', "\u{1F21}\u{3B9}"),
    ('\u{1F9A}', "\u{1F22}\u{3B9}"),
    ('\u{1F9B}', "\u{1F23}\u{3B9}"),
    ('\u{1F9C}', "\u{1F24}\u{3B9}"),
    ('\u{1F9D}', "\u{1F25}\u{3B9}"),
    ('\u{1F9E}', "\u{1F26}\u{3B9}"),
    ('\u{1F9F}', "\u{1F27}\u{3B9}"),
    ('\u{1FA0}', "\u{1F60}\u{3B9}"),
    ('\u{1FA1}', "\u{1F61}\u{3B9}"),
    ('\u{1FA2}', "\u{1F62}\u{3B9}"),
    ('\u{1FA3}', "\u{1F63}\u{3B9}"),
    ('\u{1FA4}', "\u{1F64}\u{3B9}"),
    ('\u{1FA5}', "\u{1F65}\u{3B9}"),
    ('\u{1FA6}', "\u{1F66}\u{3B9}"),
    ('\u{1FA7}', "\u{1F67}\u{3B9}"),
    ('\u{1FA8}', "\u{1F60}\u{3B9}"),
    ('\u{1FA9}', "\u{1F61}\u{3B9}"),
    ('\u{1FAA}', "\u{1F62}\u{3B9}"),
    ('\u{1FAB}', "\u{1F63}\u{3B9}"),
    ('\u{1FAC}', "\u{1F64}\u{3B9}"),
    ('\u{1FAD}', "\u{1F65}\u{3B9}"),
    ('\u{1FAE}', "\u{1F66}\u{3B9}"),
    ('\u{1FAF}', "\u{1F67}\u{3B9}"),
    ('\u{1FB2}', "\u{1F70}\u{3B9}"),
    ('\u{1FB3}', "\u{3B1}\u{3B9}"),
    ('\u{1FB4}', "\u{3AC}\u{3B9}"),
    ('\u{1FB6}', "\u{3B1}\u{342}"),
    ('\u{1FB7}', "\u{3B1}\u{342}\u{3B9}"),
    ('\u{1FBC}', "\u{3B1}\u{3B9}"),
    ('\u{1FBE}', "\u{3B9}"),
    ('\u{1FC2}', "\u{1F74}\u{3B9}"),
    ('\u{1FC3}', "\u{3B7}\u{3B9}"),
    ('\u{1FC4}', "\u{3AE}\u{3B9}"),
    ('\u{1FC6}', "\u{3B7}\u{342}"),
    ('\u{1FC7}', "\u{3B7}\u{342}\u{3B9}"),
    ('\u{1FCC}', "\u{3B7}\u{3B9}"),
    ('\u{1FD2}', "\u{3B9}\u{308}\u{300}"),
    ('\u{1FD3}', "\u{3B9}\u{308}\u{301}"),
    ('\u{1FD6}', "\u{3B9}\u{342}"),
    ('\u{1FD7}', "\u{3B9}\u{308}\u{342}"),
    ('\u{1FE2}', "\u{3C5}\u{308}\u{300}"),
    ('\u{1FE3}', "\u{3C5}\u{308}\u{301}"),
    ('\u{1FE4}', "\u{3C1}\u{313}"),
    ('\u{1FE6}', "\u{3C5}\u{342}"),
    ('\u{1FE7}', "\u{3C5}\u{308}\u{342}"),
    ('\u{1FF2}', "\u{1F7C}\u{3B9}"),
    ('\u{1FF3}', "\u{3C9}\u{3B9}"),
    ('\u{1FF4}', "\u{3CE}\u{3B9}"),
    ('\u{1FF6}', "\u{3C9}\u{342}"),
    ('\u{1FF7}', "\u{3C9}\u{342}\u{3B9}"),
    ('\u{1FFC}', "\u{3C9}\u{3B9}"),
    ('\u{FB00}', "ff"),
    ('\u{FB01}', "fi"),
    ('\u{FB02}', "fl"),
    ('\u{FB03}', "ffi"),
    ('\u{FB04}', "ffl"),
    ('\u{FB05}', "st"),
    ('\u{FB06}', "st"),
    ('\u{FB13}', "\u{574}\u{576}"),
    ('\u{FB14}', "\u{574}\u{565}"),
    ('\u{FB15}', "\u{574}\u{56B}"),
    ('\u{FB16}', "\u{57E}\u{576}"),
    ('\u{FB17}', "\u{574}\u{56D}"),
];

#[cfg(test)]
mod tests {
    use super::*;

    fn fold(text: &str) -> String {
        let mut out = String::new();
        text.chars().for_each(|ch| push_folded(&mut out, ch));
        out
    }

    #[test]
    fn two_byte_table_folds_like_the_uncached_rules() {
        for ch in '\u{80}'..='\u{7FF}' {
            for before in ["", "i", "x"] {
                let mut cached = before.to_string();
                push_folded(&mut cached, ch);
                let mut uncached = before.to_string();
                push_folded_uncached(&mut uncached, ch);
                assert_eq!(cached, uncached, "{before:?} {ch:?}");
            }
        }
    }

    #[test]
    fn caseless_ranges_have_no_folding() {
        for ch in ('\u{3040}'..='\u{9FFF}').chain('\u{AC00}'..='\u{D7A3}') {
            assert!(ch.to_lowercase().eq([ch]), "{ch:?}");
            assert!(
                FOLDS.binary_search_by_key(&ch, |&(from, _)| from).is_err(),
                "{ch:?}"
            );
        }
    }

    #[test]
    fn folds_sorted_and_differ_from_lowercase() {
        assert!(FOLDS.windows(2).all(|w| w[0].0 < w[1].0));
        for &(from, to) in FOLDS {
            assert_ne!(from.to_lowercase().collect::<String>(), to, "{from:?}");
        }
    }

    #[test]
    fn folding_is_stable_and_ignores_lowercasing() {
        for ch in (0..=0x10FFFF).filter_map(char::from_u32) {
            let folded = fold(&ch.to_string());
            assert_eq!(fold(&folded), folded, "{ch:?} folds twice differently");
            let lower: String = ch.to_lowercase().collect();
            assert_eq!(fold(&lower), folded, "{ch:?} and its lowercase differ");
            assert_eq!(
                folded.chars().next().map(char::is_alphanumeric),
                Some(ch.is_alphanumeric()),
                "{ch:?} changes word starts"
            );
        }
    }

    #[test]
    fn folds_case_variants_together() {
        for (a, b) in [
            ("straße", "STRASSE"),
            ("STRAẞE", "strasse"),
            ("İstanbul", "istanbul"),
            ("i\u{307}stanbul", "ISTANBUL"),
            ("λογος", "ΛΟΓΟΣ"),
            ("ﬁle", "FILE"),
            ("Ꮳ", "ꮳ"),
        ] {
            assert_eq!(fold(a), fold(b), "{a} vs {b}");
        }
    }
}
