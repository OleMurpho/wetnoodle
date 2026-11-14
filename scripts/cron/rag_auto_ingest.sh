#!/usr/bin/env bash
# rag_auto_ingest.sh — refresh the memory archive when it has fallen behind.
#
# Designed for a `hermes cron` no_agent job (the watchdog pattern):
#   * prints NOTHING on a clean, successful refresh  -> the user sees no message
#   * prints an error line ONLY when something broke -> the user gets alerted
#
# The work lives in scripts/cron/rag-refresh.sh next to this file. This script
# pins nothing about the machine: the store comes from RAG_DIR (or the profile's
# configured db_dir, which the generated shim passes in), the interpreter from
# RAG_PYTHON, and the checkout from this file's own location.
#
#   RAG_DIR      archive directory (contains rag.db)      [default: ~/.wetnoodle]
#   RAG_PYTHON   interpreter with numpy + the checkout    [default: <checkout>/.venv/bin/python]
#   RAG_OLLAMA   embedder base URL                        [default: http://127.0.0.1:11434]
#   WETNOODLE_SRC  checkout root                          [default: parent of this script]

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

if [ ! -f "$SRC/scripts/cron/rag-refresh.sh" ]; then
    echo "RAG refresh: checkout not found at $SRC (set WETNOODLE_SRC)"
    exit 1
fi
if [ ! -x "$RAG_PYTHON" ]; then
    echo "RAG refresh: python missing at $RAG_PYTHON (set RAG_PYTHON)"
    exit 1
fi

# Ollama absence is transient (machine asleep, service restarting) and not
# actionable in chat. Stay quiet and let the next tick pick it up.
if ! curl -s --max-time 5 "$RAG_OLLAMA/" 2>/dev/null | grep -q Ollama; then
    exit 0
fi

exec bash "$SRC/scripts/cron/rag-refresh.sh" "$@"
