#!/usr/bin/env bash
# rag_auto_ingest.sh — refresh the memory archive when it has fallen behind.
#
# Designed for a `hermes cron` no_agent job (the watchdog pattern):
#   * prints NOTHING on a clean refresh, or when nothing has changed
#   * prints an error line ONLY when something is actually wrong
#
# "Fallen behind" is checked, not assumed: when every session corpus is already at its source
# watermark this exits before doing any work, so a 30-minute schedule costs one status query
# instead of an ingest.
#
# The work lives in scripts/cron/rag-refresh.sh next to this file. This script pins nothing
# about the machine:
#
#   RAG_DIR        archive directory (contains rag.db)   [default: ~/.wetnoodle]
#   RAG_PYTHON     interpreter with numpy + the checkout [default: <checkout>/.venv/bin/python]
#   RAG_OLLAMA     embedder base URL                     [default: http://127.0.0.1:11434]
#   RAG_STALE_HOURS  hours without a good refresh before a dead embedder is reported [default 12]
#   WETNOODLE_SRC  checkout root                         [default: parent of this script]

set -uo pipefail

SRC="${WETNOODLE_SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

# Host safety: an unpinned ingest pinned every core of a laptop and crashed its
# gateway (2026-10-19). Re-exec once, pinned to a few cores at low priority.
# The absolute path matters: the shim that starts this may run from any cwd.
if [ -z "${WETNOODLE_LIMITED:-}" ] && command -v taskset >/dev/null 2>&1; then
    WETNOODLE_LIMITED=1 exec nice -n "${WETNOODLE_NICE:-19}" taskset -c "${WETNOODLE_CPUS:-0-3}" \
        bash "$SRC/scripts/cron/rag_auto_ingest.sh" "$@"
fi

export RAG_DIR="${RAG_DIR:-$HOME/.wetnoodle}"
export RAG_PYTHON="${RAG_PYTHON:-$SRC/.venv/bin/python}"
export RAG_OLLAMA="${RAG_OLLAMA:-http://127.0.0.1:11434}"
STATE="$RAG_DIR/.last_refresh_ok"
STALE_HOURS="${RAG_STALE_HOURS:-12}"

if [ ! -f "$SRC/scripts/cron/rag-refresh.sh" ]; then
    echo "RAG refresh: checkout not found at $SRC (set WETNOODLE_SRC)"
    exit 1
fi
if [ ! -x "$RAG_PYTHON" ]; then
    echo "RAG refresh: python missing at $RAG_PYTHON (set RAG_PYTHON)"
    exit 1
fi

# An unreachable embedder is usually transient (machine asleep, service restarting) and is not
# worth a message. A permanently dead one is a silent staler: the archive stops growing and
# nothing says so. Report only once the last good refresh is older than RAG_STALE_HOURS.
if ! curl -s --max-time 5 "$RAG_OLLAMA/" 2>/dev/null | grep -q Ollama; then
    last=$(cat "$STATE" 2>/dev/null || echo 0)
    case "$last" in ''|*[!0-9]*) last=0 ;; esac
    if [ "$last" -gt 0 ]; then
        age_h=$(( ($(date +%s) - last) / 3600 ))
        if [ "$age_h" -ge "$STALE_HOURS" ]; then
            echo "RAG refresh: no embedder at $RAG_OLLAMA; archive is ${age_h}h stale"
            exit 1
        fi
    fi
    exit 0
fi

# Nothing to do is not the same as failed: ask before spending minutes of CPU.
status=$("$RAG_PYTHON" -E -s -m wetnoodle.ingest --status 2>&1)
if [ $? -ne 0 ]; then
    echo "RAG refresh: status query failed"
    printf '%s\n' "$status" | tail -5
    exit 1
fi
if ! printf '%s' "$status" | grep -q '"fresh": false'; then
    exit 0
fi

bash "$SRC/scripts/cron/rag-refresh.sh" "$@"
rc=$?
if [ "$rc" -eq 0 ]; then
    mkdir -p "$RAG_DIR" 2>/dev/null
    date +%s >"$STATE" 2>/dev/null
fi
exit "$rc"
