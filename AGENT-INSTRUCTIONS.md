# Agent instructions

You are setting up `wetnoodle` for a Hermes user. Work through these steps in order and **stop at the first
failure** — `doctor.py` names the fix, so do not improvise alternatives.

1. **Confirm the target.** `RAG_HERMES_HOME` should be the user's Hermes install (default `~/.hermes`) and
   `RAG_DIR` the store (default `~/.wetnoodle`). Say both out loud before writing anything.
2. **Check Python.** `python3 -V` must be 3.11+. If not, stop and report the version.
3. **Run `./scripts/bootstrap.sh`.** It creates the venv, installs `requirements.txt`, verifies the local
   Ollama daemon, pulls `nomic-embed-text`, runs `doctor.py`, and does a first ingest. It is idempotent.
4. **Read `doctor.py` output, not the exit code alone.** Every `[fail]` line carries its own remedy. A `[warn]`
   on `session corpora` means no Hermes sessions were found — check the path before continuing.
5. **Verify the ingest.** `python ingest.py --status` must show `chunks > 0` and one corpus per profile that
   has a `state.db`. A corpus tagged frozen means its source database was deleted; leave it.
6. **Search once, for real.** `python query.py "<a phrase you know is in the history>"`. If it returns
   nothing, stop: the index is empty or the model is wrong, and every later step will lie to you.
7. **Run the smoke test.** `python -m pytest tests/ -q`. It needs no Ollama and proves discover → chunk →
   store → search → render → export/import. If it fails, the install is broken, not the archive.
8. **Write eval questions before claiming quality.** Copy `eval/questions.example.json` to
   `eval/questions.local.json` and replace it with 8–10 questions about *this* archive, each with a marker you
   verified with `--verify-markers`. Then run `python eval_recall.py`. Report the MRR you measured — do not
   report the README's number as if it were theirs.
9. **Register the MCP server** using the absolute path to `.venv/bin/python` (see README). Tell the user a
   gateway restart or a new session is needed before the tools appear; do not restart it yourself.
10. **Schedule the refresh** (`cron/rag-refresh.sh`, every 30 min) and remind the user that the archive grows
    only as far as `ingest.py` has run — a stale archive answers with stale confidence.

## Rules

- Never commit `rag.db`, `*.jsonl`, `imports/`, or `eval/questions.local.json`. `scripts/hygiene.sh` is the
  gate; run it before any push.
- Never change `RAG_EMBED_MODEL` on a populated store without `ingest.py --full`: the dimension mismatch reads
  like corruption.
- Never paste the user's transcripts into a report. Quote a short span, or describe it.
- Do not add an LLM to the ingestion path. Determinism and zero token cost are the point.
