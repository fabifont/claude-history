//! Tool-specific formatting for nicer display of tool calls.
//!
//! Instead of showing raw JSON, this module formats each tool's input
//! in a human-readable way that highlights the most relevant information.

use serde_json::Value;

/// Formatted tool call representation
pub struct FormattedToolCall {
    /// The header line (e.g., "Task (Explore): description" or "$ command")
    pub header: String,
    /// Optional continuation lines (e.g., prompt text, diff lines)
    pub body: Option<String>,
}

/// Format a tool call for display
///
/// The `max_width` parameter controls line wrapping for tools with long content (e.g., Bash commands).
pub fn format_tool_call(name: &str, input: &Value, max_width: usize) -> FormattedToolCall {
    match name {
        "Task" => format_task(input),
        "Bash" => format_bash(input, max_width),
        "Read" => format_read(input),
        "Grep" => format_grep(input),
        "Glob" => format_glob(input),
        "Edit" => format_edit(input),
        "Write" => format_write(input),
        "WebFetch" => format_web_fetch(input),
        "WebSearch" => format_web_search(input),
        _ => format_fallback(name, input, max_width),
    }
}

fn format_task(input: &Value) -> FormattedToolCall {
    let subagent_type = input
        .get("subagent_type")
        .and_then(|v| v.as_str())
        .unwrap_or("agent");
    let description = input
        .get("description")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let prompt = input.get("prompt").and_then(|v| v.as_str());

    FormattedToolCall {
        header: format!("Task ({}): {}", subagent_type, description),
        body: prompt.map(|p| p.to_string()),
    }
}

fn format_bash(input: &Value, max_width: usize) -> FormattedToolCall {
    let command = input.get("command").and_then(|v| v.as_str()).unwrap_or("");
    wrap_after_prefix("Bash: ", command, max_width)
}

/// Lay out `prefix` + `text` as a header line, wrapping the text to
/// `max_width` and spilling extra lines into the body. A `max_width` of 0
/// disables wrapping.
fn wrap_after_prefix(prefix: &str, text: &str, max_width: usize) -> FormattedToolCall {
    // Available width for the text (accounting for prefix on first line)
    let available_width = max_width.saturating_sub(prefix.len());

    let wrapped: Vec<_> = if text.contains('\n') {
        text.lines()
            .flat_map(|line| {
                if available_width == 0 || line.chars().count() <= available_width {
                    vec![line.to_string()]
                } else {
                    textwrap::wrap(line, available_width)
                        .into_iter()
                        .map(|cow| cow.into_owned())
                        .collect()
                }
            })
            .collect()
    } else if available_width == 0 || text.chars().count() <= available_width {
        return FormattedToolCall {
            header: format!("{}{}", prefix, text),
            body: None,
        };
    } else {
        textwrap::wrap(text, available_width)
            .into_iter()
            .map(|cow| cow.into_owned())
            .collect()
    };

    if wrapped.len() <= 1 {
        return FormattedToolCall {
            header: format!("{}{}", prefix, text),
            body: None,
        };
    }

    // First line goes in header, rest in body
    let header = format!("{}{}", prefix, wrapped[0]);
    let body = wrapped[1..].join("\n");

    FormattedToolCall {
        header,
        body: Some(body),
    }
}

fn format_read(input: &Value) -> FormattedToolCall {
    let file_path = input
        .get("file_path")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let offset = input.get("offset").and_then(|v| v.as_u64());
    let limit = input.get("limit").and_then(|v| v.as_u64());

    let header = match (offset, limit) {
        (Some(o), Some(l)) => format!("Read: {}:{}-{}", file_path, o, o + l),
        (Some(o), None) => format!("Read: {}:{}", file_path, o),
        _ => format!("Read: {}", file_path),
    };

    FormattedToolCall { header, body: None }
}

fn format_grep(input: &Value) -> FormattedToolCall {
    let pattern = input.get("pattern").and_then(|v| v.as_str()).unwrap_or("");
    let path = input.get("path").and_then(|v| v.as_str());
    let glob = input.get("glob").and_then(|v| v.as_str());

    let location = match (path, glob) {
        (Some(p), Some(g)) => format!("{}/{}", p, g),
        (Some(p), None) => p.to_string(),
        (None, Some(g)) => g.to_string(),
        (None, None) => ".".to_string(),
    };

    FormattedToolCall {
        header: format!("Grep: \"{}\" in {}", pattern, location),
        body: None,
    }
}

fn format_glob(input: &Value) -> FormattedToolCall {
    let pattern = input.get("pattern").and_then(|v| v.as_str()).unwrap_or("");
    let path = input.get("path").and_then(|v| v.as_str());

    let header = match path {
        Some(p) => format!("Glob: {} in {}", pattern, p),
        None => format!("Glob: {}", pattern),
    };

    FormattedToolCall { header, body: None }
}

fn format_edit(input: &Value) -> FormattedToolCall {
    let file_path = input
        .get("file_path")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let old_string = input.get("old_string").and_then(|v| v.as_str());
    let new_string = input.get("new_string").and_then(|v| v.as_str());

    let body = match (old_string, new_string) {
        (Some(old), Some(new)) => {
            let mut diff = String::new();
            for line in old.lines() {
                diff.push_str(&format!("- {}\n", line));
            }
            for line in new.lines() {
                diff.push_str(&format!("+ {}\n", line));
            }
            // Remove trailing newline
            if diff.ends_with('\n') {
                diff.pop();
            }
            Some(diff)
        }
        _ => None,
    };

    FormattedToolCall {
        header: format!("Edit: {}", file_path),
        body,
    }
}

fn format_write(input: &Value) -> FormattedToolCall {
    let file_path = input
        .get("file_path")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    FormattedToolCall {
        header: format!("Write: {}", file_path),
        body: None,
    }
}

fn format_web_fetch(input: &Value) -> FormattedToolCall {
    let url = input.get("url").and_then(|v| v.as_str()).unwrap_or("");
    let prompt = input.get("prompt").and_then(|v| v.as_str());

    FormattedToolCall {
        header: format!("Fetch: {}", url),
        body: prompt.map(|p| p.to_string()),
    }
}

fn format_web_search(input: &Value) -> FormattedToolCall {
    let query = input.get("query").and_then(|v| v.as_str()).unwrap_or("");

    FormattedToolCall {
        header: format!("Search: \"{}\"", query),
        body: None,
    }
}

fn format_fallback(name: &str, input: &Value, max_width: usize) -> FormattedToolCall {
    wrap_after_prefix(&format!("{}: ", name), &compact_json(input), max_width)
}

/// Single-line JSON with a space after separators, e.g.
/// `{ "duration_ms": 20000, "tags": ["a", "b"] }`.
fn compact_json(value: &Value) -> String {
    match value {
        Value::Object(map) if map.is_empty() => "{}".to_string(),
        Value::Object(map) => {
            let fields: Vec<String> = map
                .iter()
                .map(|(key, value)| {
                    format!("{}: {}", Value::from(key.as_str()), compact_json(value))
                })
                .collect();
            format!("{{ {} }}", fields.join(", "))
        }
        Value::Array(items) => {
            let items: Vec<String> = items.iter().map(compact_json).collect();
            format!("[{}]", items.join(", "))
        }
        scalar => scalar.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_format_task() {
        let input = json!({
            "subagent_type": "Explore",
            "description": "Find the bug",
            "prompt": "Look for issues in the code"
        });
        let result = format_tool_call("Task", &input, 80);
        assert_eq!(result.header, "Task (Explore): Find the bug");
        assert_eq!(result.body, Some("Look for issues in the code".to_string()));
    }

    #[test]
    fn test_format_bash() {
        let input = json!({
            "command": "git status",
            "description": "Check repo status"
        });
        let result = format_tool_call("Bash", &input, 80);
        assert_eq!(result.header, "Bash: git status");
        assert_eq!(result.body, None);
    }

    #[test]
    fn test_format_bash_wrapping() {
        let long_command = "cargo build --release --features 'feature1 feature2 feature3' --target x86_64-unknown-linux-gnu";
        let input = json!({
            "command": long_command
        });
        // With width 40, command should wrap (available width is 40 - 6 = 34 for command text)
        let result = format_tool_call("Bash", &input, 40);
        assert!(result.header.starts_with("Bash: cargo"));
        assert!(
            result.body.is_some(),
            "Long command should have body for continuation"
        );
    }

    #[test]
    fn test_format_bash_no_wrap_when_fits() {
        let input = json!({
            "command": "ls -la"
        });
        let result = format_tool_call("Bash", &input, 80);
        assert_eq!(result.header, "Bash: ls -la");
        assert_eq!(result.body, None);
    }

    #[test]
    fn test_format_bash_multiline_command_has_body() {
        let input = json!({
            "command": "one\ntwo"
        });
        let result = format_tool_call("Bash", &input, 80);
        assert_eq!(result.header, "Bash: one");
        assert_eq!(result.body, Some("two".to_string()));
    }

    #[test]
    fn test_format_bash_wraps_long_lines_in_multiline_command() {
        let input = json!({
            "command": "alpha beta gamma\nnext"
        });
        let result = format_tool_call("Bash", &input, 12);
        assert_eq!(result.header, "Bash: alpha");
        assert_eq!(result.body, Some("beta\ngamma\nnext".to_string()));
    }

    #[test]
    fn test_format_read_with_range() {
        let input = json!({
            "file_path": "/src/main.rs",
            "offset": 100,
            "limit": 50
        });
        let result = format_tool_call("Read", &input, 80);
        assert_eq!(result.header, "Read: /src/main.rs:100-150");
    }

    #[test]
    fn test_format_grep() {
        let input = json!({
            "pattern": "fn main",
            "path": "src",
            "glob": "*.rs"
        });
        let result = format_tool_call("Grep", &input, 80);
        assert_eq!(result.header, "Grep: \"fn main\" in src/*.rs");
    }

    #[test]
    fn test_format_fallback_is_single_line_json() {
        let input = json!({ "duration_ms": 20000, "tags": ["a", "b"], "opts": {} });
        let result = format_tool_call("clock__sleep", &input, 80);
        assert_eq!(
            result.header,
            r#"clock__sleep: { "duration_ms": 20000, "opts": {}, "tags": ["a", "b"] }"#
        );
        assert_eq!(result.body, None);
    }

    #[test]
    fn test_format_fallback_wraps_long_input() {
        let input = json!({ "query": "one two three four five six seven" });
        let result = format_tool_call("mcp__search", &input, 30);
        assert!(result.header.starts_with("mcp__search: {"));
        assert!(result.body.is_some());
    }

    #[test]
    fn test_format_edit() {
        let input = json!({
            "file_path": "/src/lib.rs",
            "old_string": "old code",
            "new_string": "new code"
        });
        let result = format_tool_call("Edit", &input, 80);
        assert_eq!(result.header, "Edit: /src/lib.rs");
        assert_eq!(result.body, Some("- old code\n+ new code".to_string()));
    }
}
