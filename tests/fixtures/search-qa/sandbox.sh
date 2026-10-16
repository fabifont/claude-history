#!/usr/bin/env bash
# Runs claude-history against the search QA corpus inside a Cua sandbox
# session prepared with `scripts/cua-sandbox build-deploy <session>`.
#
#   sandbox.sh <session> seed [--filler N]   write the corpus into a fresh HOME
#   sandbox.sh <session> run [args...]       run the CLI with that HOME
#   sandbox.sh <session> tui [--cwd DIR] [args...]
#                                            launch the TUI in Kitty under tmux
#   sandbox.sh <session> keys <tmux keys...> send keys to the TUI
#   sandbox.sh <session> screen              print the TUI screen as text
#
# The corpus HOME is /home/user/qa-home and working directories are under
# /home/user/qa, both inside the disposable container.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(git -C "$here" rev-parse --show-toplevel)"
session="${1:?session name required}"
command="${2:?command required}"
shift 2
container="$("$root/scripts/cua-sandbox" container "$session")"
qa_home=/home/user/qa-home
binary=/home/user/claude-history-run/claude-history
tmux_session=claude-history-qa
env_prefix="env -u XDG_CONFIG_HOME -u XDG_CACHE_HOME -u XDG_DATA_HOME \
-u CLAUDE_CONFIG_DIR -u CODEX_HOME HOME=$qa_home \
LD_LIBRARY_PATH=/home/user/claude-history-run/lib"

case "$command" in
  seed)
    docker exec -u user "$container" rm -rf "$qa_home" /home/user/qa
    docker exec -i -u user "$container" mkdir -p "$qa_home"
    docker exec -i -u user "$container" python3 -I - "$qa_home" "$@" <"$here/corpus.py"
    ;;
  run)
    docker exec -u user -w /home/user/qa "$container" $env_prefix "$binary" "$@"
    ;;
  tui)
    cwd=/home/user/qa
    if [[ "${1:-}" == --cwd ]]; then
      cwd="$2"
      shift 2
    fi
    docker exec -u user "$container" tmux kill-session -t "$tmux_session" 2>/dev/null || true
    docker exec -d -u user -e DISPLAY=:100 "$container" \
      kitty --title "claude-history search QA" \
      --override "font_family=JetBrainsMono Nerd Font Mono" \
      --override font_size=11 \
      --override background=#0b0f14 --override foreground=#d8dee9 \
      --override remember_window_size=no \
      --override initial_window_width=1200 --override initial_window_height=650 \
      tmux new-session -s "$tmux_session" \
      bash -c "cd '$cwd' && exec $env_prefix '$binary' \"\$@\"" claude-history "$@"
    ;;
  keys)
    docker exec -u user "$container" tmux send-keys -t "$tmux_session" "$@"
    ;;
  screen)
    docker exec -u user "$container" tmux capture-pane -p -t "$tmux_session"
    ;;
  *)
    echo "unknown command: $command" >&2
    exit 1
    ;;
esac
