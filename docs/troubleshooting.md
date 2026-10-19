# Troubleshooting — traps that cost real time

Each entry is a bug that actually happened, not a hypothetical.

## `ModuleNotFoundError: No module named 'rag_core'` from cron

The job inherits `PYTHONPATH` from whatever launched it. If a host process exports its own runtime, this
venv can import a foreign `numpy` and die with an ABI error that looks like corruption.

```bash
unset PYTHONPATH PYTHONHOME
"$PY" -E -s ingest.py        # NOT -I
```

**`-I` is the trap**: it implies `-P`, which stops prepending the script's own directory, so a script that
imports a sibling module (`import rag_core`) fails even though the file is right there. `-E -s` isolates the
environment without breaking sibling imports. `cron/rag-refresh.sh` gets this right; copy it.

## `dim mismatch` after everything worked yesterday

`RAG_EMBED_MODEL` changed. The model pins the vector dimension (768 for `nomic-embed-text`), so old and new
vectors cannot be compared. Either set the old model back or rebuild: `python ingest.py --full`.

## `database is locked`

Two writers, or a reader holding a long transaction. The store runs WAL with a 5 s busy timeout; if you still
hit it, you have two ingests racing — let the cron job own writes.

## `/api/embed` returns 401 (or `/v1/embeddings` 404)

You are pointed at Ollama **Cloud**. It serves chat models only; embedding is a local-daemon feature. Install
Ollama locally and pull the model. `doctor.py` distinguishes the two failures for you.

## Long chunks return identical embeddings

`/api/embed` **silently truncates** input beyond the model's context (2,048 tokens for `nomic-embed-text`).
Truncated text doesn't error — it embeds the prefix, so two documents differing only past the limit get
cosine ≈ 1.0. Chunks here are capped at 1,200 characters (~500 tokens), well inside the limit; if you raise
the cap, measure the token count first, character count is not a proxy for it.

## Search returns nothing

In order: the index is empty (`ingest.py --status`), the query dim disagrees with the store (above), or the
corpus filter names a corpus that does not exist (`--status` lists them).

## Search works but the right chunk is not in the top 8

Generic queries let the biggest corpora win: a 2,700-chunk profile scores a little higher than a 780-chunk one
on vague wording, even when the small corpus holds the answer. Pass `corpus=` when you know where the answer
lives. Per-corpus quotas were tried as a fix and measured **worse** (MRR 0.453 vs 0.478, hit@8 8/10 vs 9/10) —
they surface a corpus's highest-*scoring* chunks, not its answering chunk.

## The eval gate fails and you are tempted to lower the floor

Don't. The floor is a regression detector set just under the measured baseline; lowering it means the gate can
never catch anything. Fix retrieval, or add the failing question to the backlog with its measured rank.

## An LLM rewrote a document and dropped content

If consolidation is ever added to the write path, verify the rewrite before accepting it: headings still
present, length within 60–160 % of the original, each source note's body actually contained in the output,
code fences balanced. On any failure, append under a "consolidated" heading instead of merging — losing a note
silently is worse than an untidy file.
