#!/usr/bin/env python3
"""Synthetic search QA corpus for claude-history.

Writes Claude, Pi, OMP and Codex transcripts with known content under a
throwaway HOME, plus a config with exclusions. Working directories are
under /home/user/qa (the Cua sandbox user's home) and are created, so
`--local` can be run from them:

    python3 -I corpus.py <home> [--filler N] [--now EPOCH]

Every conversation is labelled (`[A1]`, `[K3]`, ...) in its first user
message and session id, so search results can be mapped back to labels.
`--filler N` adds N extra Claude conversations of neutral vocabulary, enough
to make a cold index build visible. Ages are relative to `--now` (default:
the current time).

`labels.json` in the HOME maps each label to its session file stem and
session id.
"""

import argparse
import json
import os
import random
import time
import uuid
from datetime import datetime, timezone

NS = uuid.UUID("6f1c2d3e-4b5a-4c6d-8e7f-0a1b2c3d4e5f")
HOUR = 3600
DAY = 24 * HOUR


def sid(label):
    return str(uuid.uuid5(NS, label))


def iso(epoch):
    return (
        datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.")
        + f"{int(epoch * 1000) % 1000:03d}Z"
    )


# Each conversation: label, source, project cwd, age in seconds, messages,
# and optional title, summary, tool output and kind.
# Messages alternate user/assistant, starting with the user.
C = []


def conv(label, text, *, source="claude", cwd="/home/user/qa/alpha", age=None,
         title=None, summary=None, tool=None, more=(), kind=None, sid_override=None,
         raw=()):
    C.append(dict(
        label=label, source=source, cwd=cwd,
        age=age if age is not None else (len(C) + 1) * HOUR,
        messages=[f"[{label}] {text}", f"Noted for {label}.", *more],
        title=title, summary=summary, tool=tool, kind=kind,
        session_id=sid_override or sid(label), raw=list(raw),
    ))


# A: single words, prefixes, case, mid-word substrings.
conv("A1", "The authentication middleware rejects expired tokens.")
conv("A2", "We should authorize admin users before rendering.")
conv("A3", "The fired employee record was archived.")
conv("A4", "CONFIG loader reads Config files and config overrides.")
conv("A5", "Draft the redaction plan for logs.")
conv("A6", "Restart the service using systemctl.")

# B: multiple words, order, adjacency, repetition.
conv("B1", "error handling in the parser is fragile", cwd="/home/user/qa/beta")
conv("B2", "handling of each failure is done later, the error comes back", cwd="/home/user/qa/beta")
conv("B3", "error only here, nothing else", cwd="/home/user/qa/beta")
conv("B4", "the handling error appears in logs", cwd="/home/user/qa/beta")
conv("B5", "cache cache cache cache miss", cwd="/home/user/qa/beta")
conv("B6", "cache miss once", cwd="/home/user/qa/beta")

# S: separators.
conv("S1", "set API_KEY in env", cwd="/home/user/qa/gamma")
conv("S2", "the api key was rotated", cwd="/home/user/qa/gamma")
conv("S3", "use the api-key header", cwd="/home/user/qa/gamma")
conv("S4", "look at src/store/lexindex.rs and src/search/query.rs", cwd="/home/user/qa/gamma")
conv("S5", "call std::collections::HashMap::new() here", cwd="/home/user/qa/gamma")
conv("S6", "parseHTTPResponse and getUserName are camelCase", cwd="/home/user/qa/gamma")
conv("S7", "version v1.2.3 then 2.0 release and ipv6 address", cwd="/home/user/qa/gamma")
conv("S8", "visit https://example.com/docs/getting-started?lang=en#intro", cwd="/home/user/qa/gamma")
conv("S9", "email raine.qa@example.org about it", cwd="/home/user/qa/gamma")
conv("S10", "edit Cargo.toml and README.md and main.rs", cwd="/home/user/qa/gamma")
conv("S11", "load_all_conversations_streaming() is slow", cwd="/home/user/qa/gamma")
conv("S12", "subconversations_streaming is unrelated", cwd="/home/user/qa/gamma")
conv("S13", "AppConfig struct holds settings", cwd="/home/user/qa/gamma")
conv("S14", "bump to v1.1.1 now", cwd="/home/user/qa/gamma")

# L: quoted literals.
conv("L1", "deploy with DEPLOYMENT_TOKEN set", cwd="/home/user/qa/delta")
conv("L2", "deployment token is missing", cwd="/home/user/qa/delta")
conv("L3", "The Quick Brown Fox jumps", cwd="/home/user/qa/delta")
conv("L4", "the quick brown fox naps", cwd="/home/user/qa/delta")
conv("L5", "punctuation: foo(bar, baz); end", cwd="/home/user/qa/delta")
conv("L6", "paths x.foo.bar and foo-bar too", cwd="/home/user/qa/delta")

# U: unicode.
conv("U1", "Café crème with naïve résumé", cwd="/home/user/qa/epsilon")
conv("U2", "Café decomposed resumé text", cwd="/home/user/qa/epsilon")
conv("U3", "Die Straße ist lang", cwd="/home/user/qa/epsilon")
conv("U4", "THE STRASSE IS LONG", cwd="/home/user/qa/epsilon")
conv("U5", "İstanbul ve DİYARBAKIR şehirleri", cwd="/home/user/qa/epsilon")
conv("U6", "temperature 300 Kelvin measured", cwd="/home/user/qa/epsilon")
conv("U7", "party 🎉 emoji🚀rocket and 👍 thumbs", cwd="/home/user/qa/epsilon")
conv("U8", "ÉCOLE ÉTÉ uppercase accents", cwd="/home/user/qa/epsilon")

# K: CJK.
conv("K1", "这是测试会话的内容", cwd="/home/user/qa/zeta")
conv("K2", "运行测试。然后会话结束", cwd="/home/user/qa/zeta")
conv("K3", "调用API接口和JSON数据", cwd="/home/user/qa/zeta")
conv("K4", "これはひらがなとカタカナのテストです", cwd="/home/user/qa/zeta")
conv("K5", "한국어 검색 테스트입니다", cwd="/home/user/qa/zeta")
conv("K6", "testing the fired 测试", cwd="/home/user/qa/zeta")

# W: long words.
conv("W1", "pneumonoultramicroscopicsilicovolcanoconiosis is long", cwd="/home/user/qa/eta")
conv("W2", "pneumonoultramicroscopicsilicovolcanoconiosiz typo", cwd="/home/user/qa/eta")
conv("W3", "一二三四五六七八九十百千 numbers", cwd="/home/user/qa/eta")
conv("W4", "x一二三四五六七八九十百千 prefixed", cwd="/home/user/qa/eta")

# Q: UUID lookup. Q1's body mentions Q2's session id.
conv("Q1", f"see session {sid('Q2')} for details", cwd="/home/user/qa/theta")
conv("Q2", "the referenced session about otters", cwd="/home/user/qa/theta")

# F: fields. Same age band so field weight dominates freshness.
conv("F1", "migration notes", cwd="/home/user/qa/fields", title="zanzibar migration", age=10 * DAY)
conv("F2", "overview notes", cwd="/home/user/qa/fields", summary="zanzibar overview", age=10 * DAY + 60)
conv("F3", "project notes", cwd="/home/user/qa/zanzibar", age=10 * DAY + 120)
conv("F4", "body mentions zanzibar once", cwd="/home/user/qa/fields", age=10 * DAY + 180)
conv("F5", "tool notes", cwd="/home/user/qa/fields", tool="grep found zanzibar.txt", age=10 * DAY + 240)

# R: recency with identical content.
conv("R1", "quokka habitat survey", cwd="/home/user/qa/iota", age=1 * HOUR)
conv("R2", "quokka habitat survey", cwd="/home/user/qa/iota", age=3 * DAY)
conv("R3", "quokka habitat survey", cwd="/home/user/qa/iota", age=30 * DAY)

# X: scope.
conv("X1", "wombat in an excluded path", cwd="/home/user/qa/excluded/sub")
conv("X2", "wombat in a hidden project", cwd="/home/user/qa/hiddenproj")
conv("X3", "wombat visible everywhere", cwd="/home/user/qa/alpha")
conv("X4", "wombat exec run", source="codex", kind="exec", cwd="/home/user/qa/alpha")
conv("X5", "wombat subagent run", source="codex", kind="subagent", cwd="/home/user/qa/alpha")
conv("X6", "wombat old one", cwd="/home/user/qa/alpha", age=20 * DAY)

# P: one per source.
conv("P1", "claude platypus notes", cwd="/home/user/qa/multi")
conv("P2", "pi platypus notes with kangaroo_route", source="pi", cwd="/home/user/qa/multi",
     title="Pi platypus title")
conv("P3", "omp platypus notes", source="omp", cwd="/home/user/qa/multi", title="OMP platypus title")
conv("P4", "codex platypus notes", source="codex", cwd="/home/user/qa/multi",
     tool="zebra-stripes.txt listed")

# H: highlighting beyond the preview, and viewer search.
filler = [f"filler message number {n} about unrelated gardening" for n in range(8)]
conv("H1", "starting a long conversation", cwd="/home/user/qa/hl",
     more=[*filler, "the hidden needle_token appears late with Ünïcödé and 测试会话 here",
           "another needle line", "needle again in the last line, NEEDLE upper"])
conv("H2", "short one", cwd="/home/user/qa/hl",
     more=[*filler, "late mention of İstanbul and Kelvin and Straße"])

# Matches hidden in the middle of a long conversation, outside the preview,
# so the list shows evidence snippets.
conv("H3", "opening topic about travel", cwd="/home/user/qa/hl",
     more=[*filler[:4],
           "middle: the İstanbul trip, DİYARBAKIR too, 5 \u212Aelvin, Straße name, "
           "ünïcödé word, needle_token value, Config object, v1.1.1 tag",
           *filler[4:], "closing [H3]"])
conv("H4", "opening topic about cjk", cwd="/home/user/qa/hl",
     more=[*filler[:4], "中间这是测试会话内容 and 调用JSON数据", *filler[4:],
                     "closing [H4]"])

# G: Greek final sigma.
conv("G1", "ΛΟΓΟΣ ΣΟΦΟΣ uppercase", cwd="/home/user/qa/epsilon")
conv("G2", "λογος σοφος lowercase", cwd="/home/user/qa/epsilon")

# Y: user-only search. Messages alternate user/assistant, so replies sit at
# odd positions. `raw` records follow the plain messages: (role,
# content) where role is user, assistant, or a source-specific kind.
skill_call = ("<command-message>consult</command-message>\n<command-name>/consult</command-name>"
              "\n<command-args>jaguarundi question</command-args>")
conv("Y1", "how do I configure ocelot logging", cwd="/home/user/qa/useronly",
     title="title quoll", summary="summary numbat", tool="tool output caracal",
     more=["user continues", "ocelot replies mention lynx"],
     raw=[("user", "follow up on tapir<system-reminder>reminder serval</system-reminder> done"),
          ("user", "<local-command-stdout>stdout margay</local-command-stdout>"),
          ("user", skill_call),
          ("user", "Base directory for this skill: /x\n\nskill body kinkajou"),
          ("user", "[Request interrupted by user]"),
          ("user", "Stop hook feedback:\nhook sloth")])
conv("Y2", "the bug: [Image upload fails for pangolin", cwd="/home/user/qa/useronly")
conv("Y3", "explain the <system-reminder> tag handling for axolotl", cwd="/home/user/qa/useronly")
conv("Y4", "pi user wording okapi", source="pi", cwd="/home/user/qa/useronly",
     more=["more pi words", "pi reply bonobo"],
     raw=[("custom", "custom message gibbon"),
          ("user", '<skill name="merge" location="/s/SKILL.md">\nbody tamarin\n</skill>\n\n--keep marmoset')])
conv("Y5", "codex user narwhal", source="codex", cwd="/home/user/qa/useronly",
     more=["more codex words", "codex reply manatee"],
     raw=[("injected", "# AGENTS.md instructions for /x\nagents beluga"),
          ("injected", "<environment_context>env dugong</environment_context>")])
conv("Y6", "subagent prompt walrus", source="codex", kind="subagent", cwd="/home/user/qa/useronly")
conv("Y7", "omp user wording gecko", source="omp", cwd="/home/user/qa/useronly",
     more=["more omp words", "omp reply iguana"])
conv("Y8", "first question about vicuna", cwd="/home/user/qa/useronly",
     more=["user continues", *(f"assistant vicuna detail {n} " + "padding words " * 30 for n in range(1)),
           "unrelated follow up", *filler, "vicuna vicuna vicuna assistant again"])

# E: adjacency when the second query word starts with a 4-byte char.
conv("E1", "fiesta 🎉 time", cwd="/home/user/qa/misc", age=5 * DAY)
conv("E2", "fiesta then later 🎉", cwd="/home/user/qa/misc", age=3 * DAY)
conv("E3", "fiesta hats and 🎉", cwd="/home/user/qa/misc", age=4 * DAY)
conv("E4", "🎉 confetti at the fiesta", cwd="/home/user/qa/misc", age=1 * DAY)

# M: misc edge cases.
conv("M1", "aaa.aaa.aaa repeating dotted", cwd="/home/user/qa/misc")
conv("M2", "**bold** and (paren) and `code_span` markers", cwd="/home/user/qa/misc")
conv("M3", "C++ and C# and .NET languages", cwd="/home/user/qa/misc")
conv("M4", "smart “quotes” and ‘single’ and naïve—dash", cwd="/home/user/qa/misc")
conv("Y9", "[Images] of capybara and [Image #2] with pademelon", cwd="/home/user/qa/useronly")


def lines_claude(c, t0):
    out = []
    if c["summary"]:
        out.append({"type": "summary", "summary": c["summary"], "leafUuid": "leaf"})
    for i, text in enumerate(c["messages"]):
        ts = iso(t0 + i)
        role = "user" if i % 2 == 0 else "assistant"
        base = {"type": role, "uuid": str(uuid.uuid5(NS, f"{c['label']}-{i}")),
                "timestamp": ts, "cwd": c["cwd"], "sessionId": c["session_id"]}
        if role == "user":
            base["message"] = {"role": "user", "content": text}
        else:
            base["message"] = {"role": "assistant", "content": [{"type": "text", "text": text}],
                               "model": "claude-test"}
        out.append(base)
    if c["tool"]:
        n = len(c["messages"])
        out.append({"type": "assistant", "uuid": str(uuid.uuid5(NS, c["label"] + "-tu")),
                    "timestamp": iso(t0 + n), "cwd": c["cwd"], "sessionId": c["session_id"],
                    "message": {"role": "assistant", "model": "claude-test", "content": [
                        {"type": "tool_use", "id": "toolu_qa", "name": "Bash",
                         "input": {"command": "grep -r ."}}]}})
        out.append({"type": "user", "uuid": str(uuid.uuid5(NS, c["label"] + "-tr")),
                    "timestamp": iso(t0 + n + 1), "cwd": c["cwd"], "sessionId": c["session_id"],
                    "message": {"role": "user", "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_qa", "content": c["tool"]}]}})
    for j, (role, content) in enumerate(c.get("raw", [])):
        message = ({"role": "user", "content": content} if role == "user" else
                   {"role": "assistant", "content": [{"type": "text", "text": content}],
                    "model": "claude-test"})
        out.append({"type": role, "uuid": str(uuid.uuid5(NS, f"{c['label']}-raw{j}")),
                    "timestamp": iso(t0 + len(c["messages"]) + 2 + j), "cwd": c["cwd"],
                    "sessionId": c["session_id"], "message": message})
    if c["title"]:
        out.append({"type": "custom-title", "customTitle": c["title"], "sessionId": c["session_id"]})
    return out


def lines_pi(c, t0, omp):
    out = []
    if omp:
        out.append({"type": "title", "v": 1, "title": c["title"] or "", "source": "user",
                    "updatedAt": iso(t0), "pad": ""})
    out.append({"type": "session", "version": 3, "id": c["session_id"], "timestamp": iso(t0),
                "cwd": c["cwd"]})
    parent = None
    for i, text in enumerate(c["messages"]):
        mid = f"m{i}"
        ms = int((t0 + i) * 1000)
        if i % 2 == 0:
            content = [{"type": "text", "text": text}] if omp else text
            msg = {"role": "user", "content": content, "timestamp": ms}
        else:
            msg = {"role": "assistant", "content": [{"type": "text", "text": text}],
                   "provider": "anthropic", "model": "claude-test",
                   "usage": {"input": 1, "output": 1, "cacheRead": 0, "cacheWrite": 0},
                   "timestamp": ms}
        out.append({"type": "message", "id": mid, "parentId": parent, "timestamp": iso(t0 + i),
                    "message": msg})
        parent = mid
    for j, (role, content) in enumerate(c.get("raw", [])):
        mid = f"r{j}"
        ts = iso(t0 + len(c["messages"]) + j)
        if role == "custom":
            out.append({"type": "custom_message", "id": mid, "parentId": parent, "timestamp": ts,
                        "customType": "notice", "content": [{"type": "text", "text": content}],
                        "display": True})
        else:
            out.append({"type": "message", "id": mid, "parentId": parent, "timestamp": ts,
                        "message": {"role": "user", "content": content,
                                    "timestamp": int((t0 + len(c["messages"]) + j) * 1000)}})
        parent = mid
    if c["title"] and not omp:
        out.append({"type": "session_info", "id": "name", "parentId": parent,
                    "timestamp": iso(t0 + len(c["messages"])), "name": c["title"]})
    return out


def lines_codex(c, t0):
    source = {"exec": "exec",
              "subagent": {"subagent": {"thread_spawn": {
                  "parent_thread_id": sid("P4"), "depth": 1, "agent_path": None,
                  "agent_nickname": "Euclid", "agent_role": "explorer"}}}}.get(c["kind"], "cli")
    out = [{"timestamp": iso(t0), "type": "session_meta", "payload": {
        "id": c["session_id"], "session_id": c["session_id"], "timestamp": iso(t0),
        "cwd": c["cwd"], "originator": "codex_cli_rs", "cli_version": "0.160.0",
        "source": source, "model_provider": "openai",
        "base_instructions": {"text": "You are Codex."}}},
        {"timestamp": iso(t0), "type": "turn_context", "payload": {
            "cwd": c["cwd"], "model": "gpt-5-codex", "approval_policy": "on-request"}}]
    for _, content in c.get("raw", []):
        out.append({"timestamp": iso(t0), "type": "response_item", "payload": {
            "type": "message", "role": "user",
            "content": [{"type": "input_text", "text": content}]}})
    for i, text in enumerate(c["messages"]):
        ts = iso(t0 + i)
        if i % 2 == 0:
            out.append({"timestamp": ts, "type": "response_item", "payload": {
                "type": "message", "role": "user",
                "content": [{"type": "input_text", "text": text}]}})
            out.append({"timestamp": ts, "type": "event_msg", "payload": {
                "type": "user_message", "message": text, "images": [], "local_images": []}})
            if i == 0 and c["tool"]:
                out.append({"timestamp": ts, "type": "response_item", "payload": {
                    "type": "function_call", "name": "shell",
                    "arguments": json.dumps({"command": ["ls"], "workdir": c["cwd"]}),
                    "call_id": "call_qa"}})
                out.append({"timestamp": ts, "type": "response_item", "payload": {
                    "type": "function_call_output", "call_id": "call_qa", "output": c["tool"]}})
        else:
            out.append({"timestamp": ts, "type": "event_msg", "payload": {
                "type": "agent_message", "message": text}})
            out.append({"timestamp": ts, "type": "response_item", "payload": {
                "type": "message", "role": "assistant",
                "content": [{"type": "output_text", "text": text}]}})
    return out


def write_jsonl(path, records, mtime):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.utime(path, (mtime, mtime))


def dash(cwd):
    return cwd.replace("/", "-")


FILLER_WORDS = ("garden lattice harbor meadow copper lantern orchard pebble "
                "thistle velvet willow ember fjord quarry saffron timber").split()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("home")
    ap.add_argument("--filler", type=int, default=0)
    ap.add_argument("--now", type=float, default=time.time())
    args = ap.parse_args()
    home, now = args.home, args.now
    labels = {}
    convs = list(C)
    rng = random.Random(7)
    for n in range(args.filler):
        body = " ".join(rng.choice(FILLER_WORDS) for _ in range(rng.randint(20, 400)))
        convs.append(dict(label=f"Z{n}", source="claude", cwd=f"/home/user/qa/filler{n % 40}",
                          age=40 * DAY + n * 60, messages=[body, "ok " + body[:80]],
                          title=None, summary=None, tool=None, kind=None,
                          session_id=sid(f"Z{n}"), raw=[]))
    for c in convs:
        t0 = now - c["age"]
        last = t0 + len(c["messages"]) + 2
        s = c["session_id"]
        if c["source"] == "claude":
            path = f"{home}/.claude/projects/{dash(c['cwd'])}/{s}.jsonl"
            write_jsonl(path, lines_claude(c, t0), last)
        elif c["source"] in ("pi", "omp"):
            root = ".pi" if c["source"] == "pi" else ".omp"
            stamp = datetime.fromtimestamp(t0, timezone.utc).strftime("%Y-%m-%dT%H-%M-%S-000Z")
            path = f"{home}/{root}/agent/sessions/-{dash(c['cwd'])}--/{stamp}_{s}.jsonl"
            write_jsonl(path, lines_pi(c, t0, c["source"] == "omp"), last)
        else:
            d = datetime.fromtimestamp(t0, timezone.utc)
            path = (f"{home}/.codex/sessions/{d:%Y/%m/%d}/"
                    f"rollout-{d:%Y-%m-%dT%H-%M-%S}-{s}.jsonl")
            write_jsonl(path, lines_codex(c, t0), last)
        labels[c["label"]] = {"stem": os.path.basename(path)[:-len(".jsonl")],
                              "session_id": s, "source": c["source"], "cwd": c["cwd"]}
    os.makedirs(f"{home}/.config/claude-history", exist_ok=True)
    with open(f"{home}/.config/claude-history/config.toml", "w") as f:
        f.write('[tui]\nexclude_paths = ["/home/user/qa/excluded/"]\n'
                'exclude_projects = ["hiddenproj"]\n')
    for cwd in {c["cwd"] for c in convs}:
        os.makedirs(cwd, exist_ok=True)
    with open(f"{home}/labels.json", "w") as f:
        json.dump(labels, f, indent=1)
    print(f"wrote {len(convs)} conversations under {home}")


if __name__ == "__main__":
    main()
