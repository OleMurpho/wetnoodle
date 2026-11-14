#!/usr/bin/env bash
# rag-refresh.sh — keep the index current. Run it every two hours, not every thirty minutes.
#
# Silent on success, loud on failure: a cron job that prints on every run trains you to ignore it.
# Override with RAG_DIR / RAG_PYTHON / RAG_OLLAMA.
#
# HOST SAFETY. An unbounded ingest pinned every core of a laptop and crashed the machine it was
# indexing (docs/troubleshooting.md). So this script limits itself before doing any work: if `nice` and
# `taskset` are available it re-executes, once, pinned to WETNOODLE_CPUS (default 0-3) at nice 19.
# That means a hand-run from a shell is as polite as a scheduled one. Prefer
# cron/refresh-with-ollama.sh, which also owns the daemon's lifecycle.
set -uo pipefail

CPUS="${WETNOODLE_CPUS:-0-3}"
NICE_N="${WETNOODLE_NICE:-19}"
if [ -z "${WETNOODLE_LIMITED:-}" ] && command -v taskset >/dev/null 2>&1; then
    WETNOODLE_LIMITED=1 exec nice -n "$NICE_N" taskset -c "$CPUS" bash "$0" "$@"
elif [ -z "${WETNOODLE_LIMITED:-}" ] && command -v nice >/dev/null 2>&1; then
    WETNOODLE_LIMITED=1 exec nice -n "$NICE_N" bash "$0" "$@"   # macOS: no taskset, still deprioritised
fi

RAG_DIR="${RAG_DIR:-$HOME/.wetnoodle}"
SRC="${WETNOODLE_SRC:-$(cd "$(dirname "$0")/../.." && pwd)}"
mkdir -p "$RAG_DIR" 2>/dev/null || { echo "rag-refresh: cannot create store $RAG_DIR"; exit 1; }
export RAG_DIR

# Job runtime isolation. A cron job inherits the environment of whatever launched it, and a host process
# may export PYTHONPATH for its own interpreter. Left alone, that makes this venv import a foreign numpy
# and die with an ABI error that looks like corruption. Clear it, choose the interpreter explicitly, and
# run with -E -s. NOT -I: it implies -P, which stops prepending the working directory, so `-m wetnoodle`
# would no longer resolve.
unset PYTHONPATH PYTHONHOME
PY="${RAG_PYTHON:-$SRC/.venv/bin/python}"

cd "$SRC" || { echo "rag-refresh: cannot cd to $SRC"; exit 1; }
[ -x "$PY" ] || { echo "rag-refresh: python missing at $PY (set RAG_PYTHON)"; exit 1; }
"$PY" -I -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null \
  || { echo "rag-refresh: $PY is not Python 3.11+ (set RAG_PYTHON)"; exit 1; }

out=$("$PY" -E -s -m wetnoodle.ingest 2>&1); rc=$?
[ $rc -ne 0 ] && { echo "rag-refresh: session ingest failed (rc=$rc)"; echo "$out"; exit $rc; }

out=$("$PY" -E -s -m wetnoodle.ingest_skills 2>&1); rc=$?
[ $rc -ne 0 ] && { echo "rag-refresh: skill ingest failed (rc=$rc)"; echo "$out"; exit $rc; }

exit 0
