#!/usr/bin/env bash
# rag_auto_ingest.sh — refresh the memory archive when it has fallen behind.
#
# Designed for a `hermes cron` no_agent job (the watchdog pattern):
#   * prints NOTHING on a clean refresh, or when nothing has changed
#   * prints an error line ONLY when something is actually wrong
#
# "Fallen behind" is checked, not assumed. If every session corpus is at its source watermark,
# the skill pass still runs: skills carry their own hashes and the status report says nothing
# about them, so a tick with fresh sessions can still have a SKILL.md to index.
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

# `-m wetnoodle` resolves against the working directory and this script is documented to run
# from anywhere, including cron's empty cwd. Without this every tick failed with
# "No module named 'wetnoodle'".
cd "$SRC" || { echo "RAG refresh: cannot cd to $SRC"; exit 1; }

status=$("$RAG_PYTHON" -E -s -m wetnoodle.ingest --status 2>&1)
rc=$?
if [ "$rc" -ne 0 ]; then
    echo "RAG refresh: status query failed"
    printf '%s\n' "$status" | tail -5
    exit 1
fi

# An unreadable source is the failure this watchdog exists to surface. A schema change makes
# fresh null, and grepping only for "fresh": false would stay silent about it forever.
if printf '%s' "$status" | grep -q '"broken": true'; then
    echo "RAG refresh: a corpus source exists but could not be read"
    printf '%s' "$status" | grep -B3 '"broken": true' | head -12
    exit 1
fi

behind=0
printf '%s' "$status" | grep -q '"fresh": false' && behind=1

# Nothing to do and no embedder is not a problem; work outstanding and no embedder is, but only
# once it has been that way longer than RAG_STALE_HOURS. A quiet week must not raise an alarm
# just because the machine was idle.
ollama_up=0
curl -s --max-time 5 "$RAG_OLLAMA/" 2>/dev/null | grep -q Ollama && ollama_up=1

if [ "$ollama_up" -eq 0 ]; then
    if [ "$behind" -eq 1 ]; then
        last=$(cat "$STATE" 2>/dev/null || echo 0)
        case "$last" in ''|*[!0-9]*) last=0 ;; esac
        if [ "$last" -gt 0 ]; then
            age_h=$(( ($(date +%s) - last) / 3600 ))
            if [ "$age_h" -ge "$STALE_HOURS" ]; then
                echo "RAG refresh: no embedder at $RAG_OLLAMA; archive is ${age_h}h stale"
                exit 1
            fi
        fi
    fi
    exit 0
fi

if [ "$behind" -eq 1 ]; then
    bash "$SRC/scripts/cron/rag-refresh.sh" "$@"   # sessions and skills together
    rc=$?
else
    # Sessions are current; a new or edited SKILL.md may still need indexing.
    out=$("$RAG_PYTHON" -E -s -m wetnoodle.ingest_skills 2>&1)
    rc=$?
    if [ "$rc" -eq 4 ]; then
        exit 0                      # another ingest holds the lock: a skip, not a failure
    fi
    if [ "$rc" -ne 0 ]; then
        printf '%s\n' "$out" | tail -8
        exit "$rc"
    fi
fi

if [ "$rc" -eq 0 ]; then
    # Refresh the project facts. Deterministic, skips anything unchanged, and never fails
    # the ingest: a missing curated row costs recall quality, not the index itself.
    "$RAG_PYTHON" -E -s -m wetnoodle.curate >/dev/null 2>&1 || true
    mkdir -p "$RAG_DIR" 2>/dev/null
    date +%s >"$STATE" 2>/dev/null
fi
exit "$rc"
