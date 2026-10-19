#!/usr/bin/env bash
# bootstrap.sh — first run: venv, dependencies, Ollama, model, doctor, first ingest.
# Does nothing that needs root. Re-runnable: every step is idempotent.
set -uo pipefail

# Isolation: an inherited PYTHONPATH from another interpreter (agent runtimes set one) makes this
# venv import a foreign numpy and fail. Ignore all PYTHON* variables for everything below.
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE
export PYTHONNOUSERSITE=1
cd "$(dirname "$0")/.."
ROOT=$(pwd)

say() { printf '\n== %s\n' "$1"; }

say "python"
PY=${RAG_PYTHON:-$(command -v python3 || true)}
[ -n "$PY" ] || { echo "python3 not found"; exit 1; }
"$PY" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' \
  || { echo "need Python 3.11+, found $("$PY" -V)"; exit 1; }
"$PY" -V

say "venv + dependencies"
[ -d .venv ] || "$PY" -m venv .venv
./.venv/bin/pip install --quiet --upgrade pip
./.venv/bin/pip install --quiet -r requirements.txt
echo "installed: $(./.venv/bin/pip list 2>/dev/null | grep -iE '^(numpy|mcp) ' | tr '\n' ' ')"

say "ollama"
if ! curl -fsS --max-time 3 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  echo "no local Ollama daemon on 127.0.0.1:11434."
  echo "  install:  curl -fsSL https://ollama.com/install.sh | sh     (Linux)"
  echo "            https://ollama.com/download                        (macOS / Windows)"
  echo "Cloud Ollama cannot embed (its /api/embed returns 401), so a LOCAL daemon is required."
  exit 1
fi
MODEL=${RAG_EMBED_MODEL:-nomic-embed-text}
if ! curl -fsS http://127.0.0.1:11434/api/tags | grep -q "\"$MODEL"; then
  echo "pulling $MODEL (274 MB, CPU is fine)"
  ollama pull "$MODEL" || { echo "ollama pull failed"; exit 1; }
fi

say "doctor"
./.venv/bin/python doctor.py || { echo "doctor found problems above; fix them and rerun"; exit 1; }

say "first ingest"
./.venv/bin/python ingest.py --full
./.venv/bin/python ingest_skills.py

say "done"
cat <<'EOS'
Try it:
  ./.venv/bin/python query.py "a phrase you know is in your history"

Then, to let your agent use it, register the MCP server (see README "MCP registration") and add the
refresh job (see README "Keeping it current"). Run the tests first if you like:
  ./.venv/bin/python -E -m unittest discover -s tests -q
EOS
