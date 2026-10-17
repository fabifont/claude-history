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
/// Claude Code tool names and Pi's lowercase ones (`bash`, `read`, ...)
/// share formatters; Pi names files with `path` where Claude uses `file_path`.
pub fn format_tool_call(name: &str, input: &Value, max_width: usize) -> FormattedToolCall {
    match name {
        "Task" => format_task(input),
        "Bash" | "bash" => format_bash(input, max_width),
        "Read" | "read" => format_read(input),
        "Grep" | "grep" => format_grep(input),
        "Glob" => format_glob("Glob", input),
        "find" => format_glob("Find", input),
        "Edit" | "edit" => format_edit(input),
        "Write" | "write" => format_write(input),
        "WebFetch" | "fetch_content" => format_web_fetch(input),
        "WebSearch" | "web_search" => format_web_search(input),
        "codemode" => wrap_after_prefix("Codemode: ", str_or_empty(input, "code"), max_width),
        "ls" => FormattedToolCall {
            header: format!(
                "List: {}",
                input.get("path").and_then(|v| v.as_str()).unwrap_or(".")
            ),
            body: None,
        },
        "suba" => format_labeled("Suba", input, "name", "task"),
        "suba_send" => format_labeled("Suba send", input, "id", "message"),
        "suba_resume" => format_labeled("Suba resume", input, "id", "task"),
        "suba_finish" => format_labeled("Suba finish", input, "id", "result"),
        "suba_list" => format_labeled("Suba list", input, "status", ""),
        "process" => format_process(input),
        "Skill" if str_field(input, "skill").is_some() => format_skill(input, max_width),
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

/// The file a tool call targets, under Claude's or Pi's key.
fn file_path(input: &Value) -> &str {
    ["file_path", "path"]
        .iter()
        .find_map(|key| input.get(key).and_then(|v| v.as_str()))
        .unwrap_or("")
}

fn format_read(input: &Value) -> FormattedToolCall {
    let file_path = file_path(input);
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

fn format_glob(label: &str, input: &Value) -> FormattedToolCall {
    let pattern = input.get("pattern").and_then(|v| v.as_str()).unwrap_or("");
    let path = input.get("path").and_then(|v| v.as_str());

    let header = match path {
        Some(p) => format!("{}: {} in {}", label, pattern, p),
        None => format!("{}: {}", label, pattern),
    };

    FormattedToolCall { header, body: None }
}

fn str_field<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(|v| v.as_str())
}

fn format_edit(input: &Value) -> FormattedToolCall {
    let file_path = file_path(input);

    // Claude edits one span; Pi passes a list of `edits`.
    let replacements: Vec<(&str, &str)> = match input.get("edits").and_then(|v| v.as_array()) {
        Some(edits) => edits
            .iter()
            .filter_map(|e| Some((str_field(e, "oldText")?, str_field(e, "newText")?)))
            .collect(),
        None => str_field(input, "old_string")
            .zip(str_field(input, "new_string"))
            .into_iter()
            .collect(),
    };

    let body = (!replacements.is_empty()).then(|| {
        let mut diff = Vec::new();
        for (old, new) in replacements {
            diff.extend(old.lines().map(|line| format!("- {}", line)));
            diff.extend(new.lines().map(|line| format!("+ {}", line)));
        }
        diff.join("\n")
    });

    FormattedToolCall {
        header: format!("Edit: {}", file_path),
        body,
    }
}

fn format_write(input: &Value) -> FormattedToolCall {
    let file_path = file_path(input);

    FormattedToolCall {
        header: format!("Write: {}", file_path),
        body: None,
    }
}

fn format_web_fetch(input: &Value) -> FormattedToolCall {
    // Pi's fetch_content takes either one `url` or a list of `urls`.
    let url = match input.get("urls").and_then(|v| v.as_array()) {
        Some(urls) => strings(urls).join(", "),
        None => str_or_empty(input, "url").to_string(),
    };
    let prompt = input.get("prompt").and_then(|v| v.as_str());

    FormattedToolCall {
        header: format!("Fetch: {}", url),
        body: prompt.map(|p| p.to_string()),
    }
}

fn format_web_search(input: &Value) -> FormattedToolCall {
    // Pi's web_search takes either one `query` or a list of `queries`.
    let queries = match input.get("queries").and_then(|v| v.as_array()) {
        Some(queries) => strings(queries),
        None => vec![str_or_empty(input, "query")],
    };
    let quoted: Vec<String> = queries.iter().map(|q| format!("\"{}\"", q)).collect();

    FormattedToolCall {
        header: format!("Search: {}", quoted.join(", ")),
        body: None,
    }
}

fn str_or_empty<'a>(input: &'a Value, key: &str) -> &'a str {
    input.get(key).and_then(|v| v.as_str()).unwrap_or("")
}

fn strings(values: &[Value]) -> Vec<&str> {
    values.iter().filter_map(Value::as_str).collect()
}

/// `label: <subject>` with the long text under `body_key` as the body.
fn format_labeled(
    label: &str,
    input: &Value,
    subject_key: &str,
    body_key: &str,
) -> FormattedToolCall {
    let subject = str_or_empty(input, subject_key);
    let header = if subject.is_empty() {
        label.to_string()
    } else {
        format!("{}: {}", label, subject)
    };
    let body = input
        .get(body_key)
        .and_then(|v| v.as_str())
        .filter(|text| !text.is_empty())
        .map(str::to_string);
    FormattedToolCall { header, body }
}

/// Pi's process tool: `Process <action>: <name or id>`, with a started
/// process's command as the body.
/// `Skill: /name args`, as the skill would be typed as a slash command.
fn format_skill(input: &Value, max_width: usize) -> FormattedToolCall {
    let command = format!(
        "/{} {}",
        str_or_empty(input, "skill"),
        str_or_empty(input, "args")
    );
    wrap_after_prefix("Skill: ", command.trim_end(), max_width)
}

fn format_process(input: &Value) -> FormattedToolCall {
    let action = str_or_empty(input, "action");
    let subject = ["name", "id"]
        .iter()
        .map(|key| str_or_empty(input, key))
        .find(|value| !value.is_empty())
        .unwrap_or("");
    let label = format!("Process {}", action);
    let mut formatted = format_labeled(label.trim_end(), input, "", "command");
    if !subject.is_empty() {
        formatted.header = format!("{}: {}", formatted.header, subject);
    }
    formatted
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

/// Display text for a tool result's content, shared by every renderer.
///
/// Content blocks become their text, with images, documents and loaded tools
/// as one-line placeholders instead of their base64 data or JSON. A tool
/// error drops its wrapper tags, and a one-line JSON string is pretty-printed.
pub fn format_tool_result(content: Option<&Value>) -> String {
    match content {
        None => "<no content>".to_string(),
        Some(Value::String(text)) => format_result_string(text),
        Some(Value::Array(blocks)) if blocks.iter().all(is_content_block) => blocks
            .iter()
            .map(format_result_block)
            .collect::<Vec<_>>()
            .join("\n\n"),
        Some(value) => pretty_json(value),
    }
}

fn is_content_block(value: &Value) -> bool {
    value.get("type").is_some_and(Value::is_string)
}

fn format_result_string(text: &str) -> String {
    let trimmed = text.trim();
    if let Some(error) = strip_tag(trimmed, "tool_use_error") {
        return format!("Error: {}", error);
    }
    // Claude Code's notice for output too large to keep inline.
    if let Some(notice) = strip_tag(trimmed, "persisted-output") {
        return notice.to_string();
    }
    if !trimmed.contains('\n')
        && let Ok(value @ (Value::Object(_) | Value::Array(_))) =
            serde_json::from_str::<Value>(trimmed)
    {
        return pretty_json(&value);
    }
    text.to_string()
}

/// The trimmed contents of `text` when it is wholly wrapped in `<tag>`.
fn strip_tag<'a>(text: &'a str, tag: &str) -> Option<&'a str> {
    text.strip_prefix(&format!("<{}>", tag))?
        .strip_suffix(&format!("</{}>", tag))
        .map(str::trim)
}

fn format_result_block(block: &Value) -> String {
    let block_type = block.get("type").and_then(Value::as_str).unwrap_or("");
    match block_type {
        "text" | "input_text" | "output_text" => block
            .get("text")
            .and_then(Value::as_str)
            .map(format_result_string)
            .unwrap_or_default(),
        "image" | "input_image" => media_placeholder("image", media_type(block)),
        "document" => media_placeholder("document", media_type(block)),
        "tool_reference" => format!(
            "Loaded tool: {}",
            block
                .get("tool_name")
                .and_then(Value::as_str)
                .unwrap_or("?")
        ),
        _ => match block.get("text").and_then(Value::as_str) {
            Some(text) => format_result_string(text),
            None => compact_json(block),
        },
    }
}

/// The media type of an image or document block: Claude's
/// `source.media_type`, Pi's `mimeType`, or the type in a `data:` URL.
fn media_type(block: &Value) -> Option<&str> {
    let str_at = |pointer: &str| block.pointer(pointer).and_then(Value::as_str);
    str_at("/source/media_type")
        .or_else(|| str_at("/media_type"))
        .or_else(|| str_at("/mimeType"))
        .or_else(|| {
            str_at("/image_url")?
                .strip_prefix("data:")?
                .split([';', ','])
                .next()
        })
}

fn media_placeholder(kind: &str, media_type: Option<&str>) -> String {
    match media_type {
        Some(media_type) => format!("[{}: {}]", kind, media_type),
        None => format!("[{}]", kind),
    }
}

fn pretty_json(value: &Value) -> String {
    serde_json::to_string_pretty(value).unwrap_or_else(|_| "<invalid content>".to_string())
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
    fn test_format_pi_tools() {
        let bash = format_tool_call("bash", &json!({ "command": "ls", "timeout": 30 }), 80);
        assert_eq!(bash.header, "Bash: ls");
        let read = format_tool_call(
            "read",
            &json!({ "path": "a.rs", "offset": 5, "limit": 10 }),
            80,
        );
        assert_eq!(read.header, "Read: a.rs:5-15");
        let find = format_tool_call("find", &json!({ "pattern": "*.rs", "path": "src" }), 80);
        assert_eq!(find.header, "Find: *.rs in src");
        let edit = format_tool_call(
            "edit",
            &json!({ "path": "a.rs", "edits": [
                { "oldText": "a", "newText": "b" },
                { "oldText": "c", "newText": "d" }
            ] }),
            80,
        );
        assert_eq!(edit.header, "Edit: a.rs");
        assert_eq!(edit.body, Some("- a\n+ b\n- c\n+ d".to_string()));
    }

    #[test]
    fn test_format_tool_result_placeholders() {
        let claude = json!([
            { "type": "image", "source": { "type": "base64", "media_type": "image/png", "data": "iVBOR" } },
            { "type": "text", "text": "caption" },
            { "type": "document", "source": { "media_type": "application/pdf", "data": "JVBER" } },
            { "type": "tool_reference", "tool_name": "Monitor" }
        ]);
        assert_eq!(
            format_tool_result(Some(&claude)),
            "[image: image/png]\n\ncaption\n\n[document: application/pdf]\n\nLoaded tool: Monitor"
        );
        let pi = json!([{ "type": "image", "data": "/9j/", "mimeType": "image/jpeg" }]);
        assert_eq!(format_tool_result(Some(&pi)), "[image: image/jpeg]");
        let codex = json!([{ "type": "input_image", "image_url": "data:image/png;base64,iVBOR" }]);
        assert_eq!(format_tool_result(Some(&codex)), "[image: image/png]");
    }

    #[test]
    fn test_format_tool_result_strings() {
        let error = json!("<tool_use_error>String to replace not found in file.</tool_use_error>");
        assert_eq!(
            format_tool_result(Some(&error)),
            "Error: String to replace not found in file."
        );
        let one_line_json = json!(r#"{"ok":true,"path":"a.md"}"#);
        assert_eq!(
            format_tool_result(Some(&one_line_json)),
            "{\n  \"ok\": true,\n  \"path\": \"a.md\"\n}"
        );
        let persisted = json!(
            "<persisted-output>\nOutput too large (48KB). Full output saved to: /t/a.txt\n\nPreview (first 2KB):\nhello\n</persisted-output>"
        );
        assert_eq!(
            format_tool_result(Some(&persisted)),
            "Output too large (48KB). Full output saved to: /t/a.txt\n\nPreview (first 2KB):\nhello"
        );
        let text = json!("[exited with code 0]");
        assert_eq!(format_tool_result(Some(&text)), "[exited with code 0]");
        assert_eq!(format_tool_result(Some(&json!([1, 2]))), "[\n  1,\n  2\n]");
        assert_eq!(format_tool_result(None), "<no content>");
    }

    #[test]
    fn test_format_skill_as_slash_command() {
        let skill = |input: Value| format_tool_call("Skill", &input, 0);
        assert_eq!(
            skill(json!({ "skill": "phased-implement", "args": "add dark mode" })).header,
            "Skill: /phased-implement add dark mode"
        );
        assert_eq!(
            skill(json!({ "skill": "consult-llm", "args": "" })).header,
            "Skill: /consult-llm"
        );
        assert_eq!(skill(json!({ "skill": "commit" })).header, "Skill: /commit");
        let multiline = skill(json!({ "skill": "review", "args": "first\nsecond" }));
        assert_eq!(multiline.header, "Skill: /review first");
        assert_eq!(multiline.body.as_deref(), Some("second"));
    }

    #[test]
    fn test_format_more_pi_tools() {
        let header = |name: &str, input: Value| format_tool_call(name, &input, 0).header;
        assert_eq!(header("ls", json!({ "path": "src" })), "List: src");
        assert_eq!(
            header(
                "web_search",
                json!({ "queries": ["a b", "c"], "numResults": 5 })
            ),
            r#"Search: "a b", "c""#
        );
        assert_eq!(
            header(
                "fetch_content",
                json!({ "urls": ["https://a", "https://b"] })
            ),
            "Fetch: https://a, https://b"
        );
        let suba = format_tool_call("suba", &json!({ "name": "review", "task": "Check it" }), 0);
        assert_eq!(suba.header, "Suba: review");
        assert_eq!(suba.body.as_deref(), Some("Check it"));
        assert_eq!(
            header("suba_list", json!({ "status": "all" })),
            "Suba list: all"
        );
        let process = format_tool_call(
            "process",
            &json!({ "action": "start", "name": "smoke", "command": "make test" }),
            0,
        );
        assert_eq!(process.header, "Process start: smoke");
        assert_eq!(process.body.as_deref(), Some("make test"));
        assert_eq!(
            header("process", json!({ "action": "list" })),
            "Process list"
        );
        let codemode =
            format_tool_call("codemode", &json!({ "code": "const r = 1;\nreturn r;" }), 0);
        assert_eq!(codemode.header, "Codemode: const r = 1;");
        assert_eq!(codemode.body.as_deref(), Some("return r;"));
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
