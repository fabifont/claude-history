//! Syntax highlighting for code blocks using syntect.

use crate::tui::theme;
use std::sync::OnceLock;
use syntect::easy::HighlightLines;
use syntect::highlighting::{FontStyle, ThemeSet};
use syntect::parsing::SyntaxSet;
use syntect::util::LinesWithEndings;

static SYNTAX_SET: OnceLock<SyntaxSet> = OnceLock::new();
static THEME_SET: OnceLock<ThemeSet> = OnceLock::new();

fn get_syntax_set() -> &'static SyntaxSet {
    SYNTAX_SET.get_or_init(SyntaxSet::load_defaults_newlines)
}

fn get_theme_set() -> &'static ThemeSet {
    THEME_SET.get_or_init(ThemeSet::load_defaults)
}

/// Loads the syntax and theme definitions and compiles the grammars of
/// languages common in conversations, so the first highlight does not pay
/// for them. Needs the TUI theme to be detected already.
pub fn warm() {
    for (code, lang) in [
        ("fn main() {}", "rust"),
        ("const a = 1;", "typescript"),
        ("def f(): pass", "python"),
        ("echo hi", "bash"),
        ("{\"a\": 1}", "json"),
        ("a: 1", "yaml"),
        ("# a", "markdown"),
    ] {
        let _ = highlight_code_tui(code, lang);
    }
}

/// Normalize common language aliases to their canonical names
fn normalize_language(lang: &str) -> &str {
    // Take only the first token (handle "rust,ignore" or "rust title=x")
    let lang = lang.split([',', ' ']).next().unwrap_or(lang).trim();

    match lang.to_lowercase().as_str() {
        "js" => "javascript",
        "ts" => "typescript",
        "sh" | "shell" => "bash",
        "yml" => "yaml",
        "py" => "python",
        "rb" => "ruby",
        "md" => "markdown",
        "dockerfile" => "Dockerfile",
        _ => lang,
    }
}

/// Highlighted token with styling information
pub struct HighlightedToken {
    pub text: String,
    pub fg: (u8, u8, u8),
    pub bold: bool,
    pub italic: bool,
}

/// Highlight code and return styled tokens per line.
/// Returns None if language is unknown.
pub fn highlight_code_tui(code: &str, lang: &str) -> Option<Vec<Vec<HighlightedToken>>> {
    let ps = get_syntax_set();
    let ts = get_theme_set();

    let lang = normalize_language(lang);
    let syntax = ps.find_syntax_by_token(lang)?;
    let theme = ts.themes.get(theme::detect_theme().syntect_theme)?;

    let mut highlighter = HighlightLines::new(syntax, theme);
    let mut lines = Vec::new();

    for line in LinesWithEndings::from(code) {
        let ranges = highlighter.highlight_line(line, ps).ok()?;
        let tokens: Vec<HighlightedToken> = ranges
            .into_iter()
            .map(|(style, text)| HighlightedToken {
                text: text.to_string(),
                fg: (style.foreground.r, style.foreground.g, style.foreground.b),
                bold: style.font_style.contains(FontStyle::BOLD),
                italic: style.font_style.contains(FontStyle::ITALIC),
            })
            .collect();
        lines.push(tokens);
    }

    Some(lines)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_language() {
        assert_eq!(normalize_language("js"), "javascript");
        assert_eq!(normalize_language("ts"), "typescript");
        assert_eq!(normalize_language("sh"), "bash");
        assert_eq!(normalize_language("rust"), "rust");
        assert_eq!(normalize_language("rust,ignore"), "rust");
        assert_eq!(normalize_language("rust title=x"), "rust");
    }

    #[test]
    fn test_highlight_known_language() {
        let code = "let x = 1;";
        let result = highlight_code_tui(code, "rust");
        assert!(result.is_some());
        let lines = result.unwrap();
        assert!(!lines.is_empty());
        let total_tokens: usize = lines.iter().map(|l| l.len()).sum();
        assert!(total_tokens > 1, "Expected multiple tokens for syntax");
    }

    #[test]
    fn test_highlight_unknown_language() {
        let result = highlight_code_tui("some code", "unknown_language_xyz");
        assert!(result.is_none());
    }

    #[test]
    fn test_highlight_with_alias() {
        let result = highlight_code_tui("const x = 1;", "js");
        assert!(result.is_some());
    }
}
