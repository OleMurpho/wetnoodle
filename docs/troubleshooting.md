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

## Running on a laptop, or inside WSL

**The failure, measured.** On 2026-10-19 an ordinary install on a 13th-gen i7 laptop (20 threads, 7.6 GiB given
to the WSL2 VM) ran its scheduled refresh at 22:00. The embedding pass took every core at normal priority,
`VmmemWSL` ballooned, CPU sat at 100 %, Windows stopped responding, and the Hermes gateway on the host
hard-crashed. Recovery was manual: pause the job, `pkill -f "ollama serve"`, cap `VmmemWSL` affinity to 4 cores
in Task Manager. Load average right after the crash was 1.73 / 3.47 / 1.84 on a machine that should idle at 0.1.

The corpus was **1,083 chunks**. This is not a big-archive problem; it is a default-behaviour problem. Ollama
spreads one embedding request across every core it can see, and a VM reports all of the host's threads while
owning only part of its memory.

**What to run instead.** Never point cron at `ingest.py`, and prefer the wrapper:

```bash
17 */2 * * * /path/to/wetnoodle/cron/refresh-with-ollama.sh
```

It starts Ollama itself with `OLLAMA_NUM_PARALLEL=1`, pinned to `WETNOODLE_CPUS` (default `0-3`) at
`WETNOODLE_NICE` (default 19), runs the refresh under the same pin, and stops the daemon afterwards. Plain
`cron/rag-refresh.sh` also pins itself before doing any work, so a hand-run is polite too. Inside the ingest,
`RAG_NUM_THREADS` (default `min(4, nproc)`) caps threads per request and `RAG_BATCH` (default 16) caps passages
per request.

**Check before you schedule.** `python doctor.py` now reports the shape of the machine and warns when it is
thread-rich and memory-poor, when it is WSL, and when nothing caps affinity:

```
[ok  ] machine: 12 core(s), 15.5 GiB, WSL, affinity 12
[warn] machine shape: WSL: 12 host thread(s) visible, but this VM owns only 15.5 GiB ...
[warn] cpu affinity: 12 cores and no affinity limit — one embedding request will use all of them.
```

**If the host is already unresponsive.** From another terminal: pause the cron job, then
`pkill -f "ollama serve"`, then re-run under the pin:
`taskset -c 0-3 nice -n 19 ./cron/rag-refresh.sh`. On Windows, capping `VmmemWSL` affinity in Task Manager
buys you the seconds you need to get a terminal.

**Cadence.** Two hours, not thirty minutes. Nothing in the design needs half-hourly freshness — the archive is
read, not watched — and every extra run is another chance to catch the machine at load. A refresh is ~70 s per
1,000 new chunks, so a two-hourly job that usually finds nothing is nearly free.

**GPU is not yet the answer.** Moving the embedder to a GPU may be the real fix, but it has not been measured on
a host with a competing GPU workload, and VRAM is shared. Until someone measures it, the pin is the fix.

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
