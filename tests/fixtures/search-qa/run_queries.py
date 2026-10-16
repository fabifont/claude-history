#!/usr/bin/env python3
"""Runs queries.json against a seeded Cua sandbox (see sandbox.sh) and
compares three surfaces with the expected labels:

- oracle: `--debug-search` (the in-memory scorer, every kind, top 30)
- agent:  `agent search --all-kinds` result conversations
- tui:    the list in a headless tmux TUI (the generation index)

    python3 -I run_queries.py <session> [--ids 1,2,3] [--json out.json]
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SANDBOX = os.path.join(HERE, "sandbox.sh")
# Never listed in the TUI by default: excluded path, excluded project, Codex
# exec and subagent.
TUI_HIDDEN = {"X1", "X2", "X4", "X5", "Y6"}


def sh(session, *args, check=True):
    out = subprocess.run([SANDBOX, session, *args], capture_output=True, text=True)
    if check and out.returncode != 0:
        raise RuntimeError(out.stderr)
    return out.stdout + out.stderr


def load_labels(session):
    text = sh(session, "run", "--version")  # ensures the container answers
    del text
    raw = subprocess.run(
        ["docker", "exec", "-u", "user", container(session), "cat",
         "/home/user/qa-home/labels.json"], capture_output=True, text=True, check=True).stdout
    return json.loads(raw)


def container(session):
    root = subprocess.run(["git", "-C", HERE, "rev-parse", "--show-toplevel"],
                          capture_output=True, text=True, check=True).stdout.strip()
    return subprocess.run([os.path.join(root, "scripts/cua-sandbox"), "container", session],
                          capture_output=True, text=True, check=True).stdout.strip()


def expand(query, labels):
    def sub(m):
        label, cut = m.group(1), m.group(2)
        value = labels[label]["session_id"]
        return value[: int(cut)] if cut else value
    return re.sub(r"@([A-Z]+\d+)(?::(\d+))?", sub, query)


def oracle(session, query, by_stem):
    out = sh(session, "run", "--debug-search", query, check=False)
    found = []
    for line in out.splitlines():
        m = re.match(r"#\s*\d+ score=\S+ freshness=\S+ \| .* \| (\S+) \| .* ago$", line)
        if m:
            found.append(by_stem.get(m.group(1), m.group(1)))
    return found


def agent(session, query, by_id, user_only=False):
    extra = ["--user-only"] if user_only else []
    out = sh(session, "run", "agent", "search", query, "--all-kinds", "--no-budget",
             "--top", "100", *extra, check=False)
    found = []
    for line in out.splitlines():
        m = re.match(r"conversation rank=\d+ .*uuid=(\S+)", line)
        if m:
            found.append(by_id.get(m.group(1), m.group(1)))
    return found


TITLES = {"zanzibar migration": "F1", "Pi platypus title": "P2", "OMP platypus title": "P3",
          "title quoll": "Y1",
          # Evidence snippets that replace the preview.
          "middle: the": "H3", "v1.1.1 tag": "H3", "中间这是": "H4"}
# Cards whose preview shows neither the label nor a title.
CARDS = {("hl", 13): "H1", ("hl", 11): "H2"}


def tui_rows(screen):
    """Labels of the result cards on `screen`, top to bottom."""
    labels = []
    lines = screen.splitlines()
    for i, line in enumerate(lines):
        header = re.search(r"▌ \S+\s+· (\S+).*?(\d+) msgs", line)
        if not header:
            continue
        card = line + " " + (lines[i + 1] if i + 1 < len(lines) else "")
        m = re.search(r"\[([A-Z]+\d+)\]", card)
        label = m.group(1) if m else CARDS.get((header.group(1), int(header.group(2))))
        if label is None:
            label = next((v for k, v in TITLES.items() if k in card), "?")
        labels.append(label)
    return labels


USER_ONLY = [False]


def settled_screen(session, timeout=15):
    """The TUI screen once it stops changing."""
    time.sleep(0.5)
    previous = sh(session, "screen")
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(0.4)
        screen = sh(session, "screen")
        if screen == previous:
            return screen
        previous = screen
    return previous


def tui(session, query, user_only=False):
    if USER_ONLY[0] != user_only:
        sh(session, "keys", "C-v")
        USER_ONLY[0] = user_only
        time.sleep(0.3)
    # Ctrl+U clears the query. Escape would merge with a backlog of typed
    # keys into an Alt chord when the TUI lags behind.
    sh(session, "keys", "C-u")
    sh(session, "keys", "-l", query)
    screen = settled_screen(session)
    header = screen.splitlines()[1] if len(screen.splitlines()) > 1 else ""
    m = re.search(r"(\d+)/(\d+)\s*│", header)
    count = int(m.group(2)) if m else None
    if "No matches" in screen or count == 0:
        return [], 0, screen
    return tui_rows(screen), count, screen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--ids")
    ap.add_argument("--json")
    ap.add_argument("--no-tui", action="store_true")
    args = ap.parse_args()
    labels = load_labels(args.session)
    by_stem = {v["stem"]: k for k, v in labels.items()}
    by_id = {v["session_id"]: k for k, v in labels.items()}
    queries = json.load(open(os.path.join(HERE, "queries.json")))
    if args.ids:
        wanted = {int(i) for i in args.ids.split(",")}
        queries = [q for q in queries if q["id"] in wanted]
    if not args.no_tui:
        subprocess.run(["docker", "exec", "-u", "user", container(args.session), "tmux",
                        "kill-session", "-t", "claude-history-qa"], capture_output=True)
        subprocess.run(["docker", "exec", "-u", "user", container(args.session), "tmux",
                        "new-session", "-d", "-x", "200", "-y", "150", "-s", "claude-history-qa",
                        "cd /home/user/qa && env -u XDG_CONFIG_HOME -u XDG_CACHE_HOME "
                        "-u CLAUDE_CONFIG_DIR -u CODEX_HOME HOME=/home/user/qa-home "
                        "LD_LIBRARY_PATH=/home/user/claude-history-run/lib "
                        "/home/user/claude-history-run/claude-history"], check=True)
        time.sleep(3)
    results = []
    for q in queries:
        query = expand(q["q"], labels)
        expect = q["expect"]
        uo = q.get("user_only", False)
        # --debug-search has no user-only mode.
        o = None if uo else oracle(args.session, query, by_stem)
        a = agent(args.session, query, by_id, uo)
        t, count, screen = (None, None, "") if args.no_tui else tui(args.session, query, uo)
        exp_oracle = q.get("oracle", expect)
        # agent search --all-kinds honours exclude_paths and exclude_projects.
        exp_agent = q.get("agent", [] if expect == "ALL"
                          else [x for x in exp_oracle if x not in ("X1", "X2")])
        exp_tui = expect if expect == "ALL" else [x for x in expect if x not in TUI_HIDDEN]

        def ok(actual, wanted):
            if actual is None:
                return None
            if wanted == "ALL":
                # Browsing: every visible conversation, newest first.
                return len(actual) >= 30
            if q.get("order"):
                return actual == wanted
            return sorted(actual) == sorted(wanted)

        def ok_agent(actual, wanted):
            return sorted(actual) == sorted(wanted)

        # Agent search lists conversations in keyword search rank order, as
        # the TUI does.
        agent_vs_oracle = None
        if a is not None and o is not None and "oracle" not in q:
            agent_vs_oracle = a == [x for x in o if x in a]

        tui_vs_oracle = None
        # An "oracle" override marks queries where the TUI deliberately differs
        # from --debug-search, such as a pasted UUID jumping to its session.
        if t is not None and o is not None and "oracle" not in q:
            visible_oracle = [x for x in o if x not in TUI_HIDDEN]
            n = min(len(t), len(visible_oracle))
            tui_vs_oracle = t[:n] == visible_oracle[:n] and (
                count == len(visible_oracle) or len(o) >= 30)
        row = dict(id=q["id"], group=q["group"], query=query, expect=expect,
                   user=q.get("user"), oracle=o, agent=a, tui=t, tui_count=count,
                   oracle_ok=ok(o, exp_oracle), agent_ok=ok_agent(a, exp_agent),
                   tui_ok=ok(t, exp_tui), tui_vs_oracle=tui_vs_oracle,
                   agent_vs_oracle=agent_vs_oracle, note=q.get("note"))
        results.append(row)
        flag = "PASS" if all(v in (True, None) for v in
                              (row["oracle_ok"], row["agent_ok"], row["tui_ok"], tui_vs_oracle,
                               agent_vs_oracle)) else "FAIL"
        print(f"{flag} #{q['id']:3} {query[:40]!r:44} exp={expect} oracle={o} agent={a} "
              f"tui={t}({count})", flush=True)
        if flag == "FAIL" and t is not None and os.environ.get("QA_SCREENS"):
            print(screen)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.exit(main())
