#!/usr/bin/env bash
# refresh-with-ollama.sh — the safe way to run the index refresh on a machine you also use.
#
# Why this exists: on 2026-10-19 a default install pinned all 20 threads of a laptop, the WSL VM grew
# until Windows stopped responding, and the Hermes gateway crashed. The fix is not a smaller corpus; it
# is telling the embedder how much of the machine it may have. This wrapper:
#
#   * starts Ollama itself, pinned to WETNOODLE_CPUS at nice 19, with ONE parallel request;
#   * runs the ingest under the same pin, so a hand-run is as polite as a scheduled one;
#   * stops the daemon afterwards, so a cron job does not leave a resident process on a laptop.
#
#   WETNOODLE_CPUS=0-3   cores to use (default 0-3; on a 4-core box use 0-1)
#   WETNOODLE_NICE=19    scheduling priority (19 = lowest)
#   WETNOODLE_KEEP_DAEMON=1  leave Ollama running afterwards (only if you use it for other work)
#
# On macOS there is no `taskset`; the wrapper degrades to `nice` alone and says so. Windows users should
# run this inside WSL, not from a Windows shell.
set -uo pipefail

CPUS="${WETNOODLE_CPUS:-0-3}"
NICE_N="${WETNOODLE_NICE:-19}"
HERE="$(cd "$(dirname "$0")" && pwd)"
HOME_DIR="${HOME:-/root}"
OLLAMA_BIN="${OLLAMA_BIN:-$(command -v ollama || echo "$HOME_DIR/.local/bin/ollama")}"
LIMIT=()

if command -v taskset >/dev/null 2>&1; then
    LIMIT=(taskset -c "$CPUS")
elif [ -n "${WETNOODLE_CPUS:-}" ]; then
    echo "refresh: taskset not found (macOS?) — CPU pinning skipped, only nice applies" >&2
fi

# Is a daemon already listening? If so it is not ours and we must not kill it.
STARTED=0
if ! curl -sf -m 3 "${RAG_OLLAMA:-http://127.0.0.1:11434}/" >/dev/null 2>&1; then
    if [ ! -x "$OLLAMA_BIN" ]; then
        echo "refresh: ollama not found at $OLLAMA_BIN — install from https://ollama.com/download" >&2
        exit 1
    fi
    # OLLAMA_NUM_PARALLEL=1 + a single loaded model is what keeps the resident footprint small.
    # setsid puts the daemon in its own process group, so cleanup can kill exactly what we started.
    # A bare `pkill -f "ollama serve"` would also kill a daemon the user is already relying on.
    if command -v setsid >/dev/null 2>&1; then
        OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=1 \
            setsid nice -n "$NICE_N" "${LIMIT[@]}" "$OLLAMA_BIN" serve \
            >"$HOME_DIR/.ollama-serve.log" 2>&1 &
    else
        OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=1 \
            nohup nice -n "$NICE_N" "${LIMIT[@]}" "$OLLAMA_BIN" serve \
            >"$HOME_DIR/.ollama-serve.log" 2>&1 &
    fi
    DAEMON_PID=$!
    STARTED=1
    for _ in $(seq 1 40); do
        curl -sf -m 2 "${RAG_OLLAMA:-http://127.0.0.1:11434}/" >/dev/null 2>&1 && break
        sleep 0.5
    done
    if ! curl -sf -m 3 "${RAG_OLLAMA:-http://127.0.0.1:11434}/" >/dev/null 2>&1; then
        echo "refresh: ollama did not start; see $HOME_DIR/.ollama-serve.log" >&2
        exit 1
    fi
fi

nice -n "$NICE_N" "${LIMIT[@]}" bash "$HERE/rag-refresh.sh" "$@"
rc=$?

if [ "$STARTED" = 1 ] && [ "${WETNOODLE_KEEP_DAEMON:-0}" != "1" ]; then
    # Kill our own process group only. On-demand: a cron job must not leave a resident daemon on a laptop.
    kill -- "-$DAEMON_PID" 2>/dev/null || kill "$DAEMON_PID" 2>/dev/null || true
    for _ in $(seq 1 10); do
        kill -0 "$DAEMON_PID" 2>/dev/null || break
        sleep 0.2
    done
fi
exit $rc
