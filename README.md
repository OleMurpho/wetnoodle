# wetnoodle

A local, session-aware memory archive for [Hermes Agent](https://hermes-agent.nousresearch.com/docs).

Hermes keeps a separate SQLite session database per profile. `wetnoodle` indexes **all of them** — plus every
`SKILL.md` you have — into one small vector store, so an agent can answer *"what did we decide about X three
weeks ago"* instead of only seeing what is still in its context window. Everything stays on your machine: a
local `sqlite` file and a local Ollama embedding model, no API keys, no cloud store.

```
you ──▶ Hermes sessions (per profile, per SQLite db) ──┐
                                                       ├─▶ wetnoodle ingest ──▶ rag.db (vectors + text)
you ──▶ skills (*/SKILL.md) ───────────────────────────┘                              │
                                                                                      ▼
                                             agent (MCP tool) / query.py ◀── search ───┘
```

## Try it without installing anything

**→ https://olemurpho.github.io/wetnoodle/**

A working index of **5,677 PubMed abstracts** that you can search in the browser. Every hit is a real paper —
title, journal, year, PMID — so any answer can be checked against its source instead of taken on trust.

Two search paths are exposed, and the page says which is which:

- **Typed queries** use BM25 (keyword) in JavaScript, because a browser has no embedding model.
- **The twelve examples** run the real embedding search: their query vectors were computed locally by
  `nomic-embed-text`, truncated to 256 dimensions and int8-quantised, so the cosine ranking you see is the
  same maths the archive uses. The page states the rank of the abstract each example came from.

That curation is disclosed on the page: of 46 titles sampled across the corpus, 30 retrieved their own
abstract in the top three, and the examples are drawn from those 30. It is a demo of the mechanism, not a
benchmark — [Honest numbers](#honest-numbers) is the benchmark, on a real archive with a held-out set.

## Why this shape

- **Multi-corpus, per-corpus watermarks.** Each profile's database keeps its own incremental cursor, so a
  profile whose message ids lag another's still ingests correctly. One shared watermark cannot do that.
- **Sessions *and* procedures.** Conversation history answers "what did we say"; skills answer "how do we do
  this here". Both are searched by the same tool.
- **Small.** numpy + `sqlite3` + the MCP SDK. No Chroma, no LlamaIndex, no vector server. A 7,000-chunk
  archive is ~40 MB and indexes in a couple of minutes on CPU.
- **Nothing is thrown away.** A profile you delete keeps its corpus as *frozen* history rather than vanishing
  from the archive.
- **Measurable.** `eval_recall.py` scores retrieval quality (see [Honest numbers](#honest-numbers)).

## Requirements

| | |
|---|---|
| Python | 3.11+ |
| Ollama | **local daemon**, `127.0.0.1:11434` |
| Embedding model | `nomic-embed-text` (274 MB, CPU is fine) |
| Disk | ~1 MB per 200 chunks |

Ollama Cloud cannot be used: its `/api/embed` returns `401` and `/v1/embeddings` `404` for embedding models.
A local daemon is a hard requirement, and that is deliberate — your archive should not need a key.

## Quickstart

```bash
git clone https://github.com/OleMurpho/wetnoodle.git && cd wetnoodle
./scripts/bootstrap.sh          # venv, deps, model, doctor, first ingest
./.venv/bin/python query.py "the phrase you are looking for"
```

`bootstrap.sh` is idempotent and needs no root. If it stops, `python doctor.py` names the exact fix.

## MCP registration

Give the agent the two tools (`rag_search`, `rag_audit`). In your Hermes config:

```json
"mcp_servers": {
  "wetnoodle": {
    "command": "/absolute/path/to/wetnoodle/.venv/bin/python",
    "args": ["/absolute/path/to/wetnoodle/mcp_server.py"]
  }
}
```

Use the **absolute** interpreter path: a system `python` will not have `numpy`/`mcp`. Restart the gateway
(or start a new session) and the tools appear as `mcp_wetnoodle_rag_search` and `mcp_wetnoodle_rag_audit`.

## Keeping it current

```bash
*/30 * * * * /absolute/path/to/wetnoodle/cron/rag-refresh.sh
```

Silent on success, loud on failure. It is safe to run while the agent is working: the store is WAL-mode and
ingest only appends. In Hermes you can schedule the same script with the `cronjob` tool instead of crontab.

## Commands

| Command | What it does |
|---|---|
| `python ingest.py` | incremental ingest of every session corpus |
| `python ingest.py --status` | per-corpus counts and watermarks |
| `python ingest.py --full` | rebuild available corpora from scratch |
| `python ingest.py --purge-missing` | drop corpora whose source database is gone |
| `python ingest_skills.py` | index every `SKILL.md` |
| `python query.py "text" [--k N] [--corpus C]` | search from the shell |
| `python eval_recall.py` | retrieval gate: MRR, hit@1, hit@k |
| `python eval_recall.py --verify-markers` | is each eval marker actually present? |
| `python export_store.py --corpus X --out f.jsonl` | corpus → text (portable, diffable) |
| `python import_store.py f.jsonl --corpus peer` | text → corpus (re-embedded locally) |
| `python doctor.py` | preflight every dependency |
| `python -m pytest tests/ -q` | smoke test, no Ollama needed |

## Tuning

All optional, all environment variables:

| Variable | Default | Effect |
|---|---|---|
| `RAG_DIR` | `~/.wetnoodle` | where `rag.db` lives |
| `RAG_HERMES_HOME` | `~/.hermes` | Hermes install to index |
| `RAG_EMBED_MODEL` | `nomic-embed-text` | **pins vector dim**; changing it requires `ingest.py --full` |
| `RAG_OLLAMA` | `http://127.0.0.1:11434` | daemon URL |
| `RAG_SKILLS_DIRS` | discovered | override skill roots (`:`-separated) |
| `RAG_MAX_HIT_CHARS` | `800` | per-hit cap in tool output |
| `RAG_MAX_TOTAL_CHARS` | `6000` | cap across the whole result set |
| `RAG_HYBRID` | `1` | keyword rerank inside vector candidates |
| `RAG_VEC_WEIGHT` | `0.7` | vector share of the rerank blend |
| `RAG_PER_CORPUS` | `0` | per-corpus slot quotas — **measured worse, off** (see below) |

## Honest numbers

Retrieval quality is measured, not assumed. On the reference archive (7,465 chunks, 4 corpora, 10
hand-written questions with unique markers):

| configuration | MRR | hit@1 | hit@8 |
|---|---|---|---|
| vector only | 0.442 | 2/10 | 9/10 |
| + keyword rerank in the candidate set | **0.478** | 2/10 | **9/10** |
| + per-corpus quotas | 0.453 | 2/10 | 8/10 |

Read that as: **search works, and it is not good enough.** Only 2 of 10 questions answer at rank 1. The
`0.45` floor in `eval_recall.py` is a *regression* gate — it catches a change that makes retrieval worse; it
does not certify quality. The quality target is MRR ≥ 0.6 and it is not met yet.

Two findings that shaped the code, both reproducible with the harness above:

- **Keyword terms may reorder candidates, never introduce them.** Whole-corpus rank fusion scored 0.512 MRR
  but dropped hit@8 to 7/10: a word appearing in 177 chunks ties at maximum keyword score, and fusing that
  tied noise promotes arbitrary chunks above good vector hits.
- **Per-corpus quotas are not the fix for cross-corpus dilution.** The dilution is real — the one miss ranks
  4th when restricted to its own corpus — but allocating slots surfaced each corpus's highest-*scoring*
  chunks, not its answering chunk. Use `corpus=` when you know where the answer lives.

Write your own `eval/questions.local.json` against your own history. Ten good questions will tell you more
about this system than any benchmark number borrowed from someone else's archive.

## Known limits

- **Paraphrase queries underperform identifier queries.** Embeddings drift from literal tokens; the keyword
  rerank only partly compensates.
- **Chunks are raw conversation.** Unlike a curated wiki, half a chunk can be boilerplate. Quality is bounded
  by what was said.
- **One embedding model per store.** Switching models without `--full` is a dimension mismatch, which reads
  like corruption.
- **No reranker model, no query expansion, no agentic retrieval.** All three are plausible next steps, none
  are implemented.
- **Not multi-user.** It indexes the Hermes install it can see; there is no auth layer.

## Prior art

The marker-based recall scoring, bounded recall output, text export/rebuild and the job-runtime isolation
pattern were inspired by [tomRumi/hermes-memory-rag](https://github.com/tomRumi/hermes-memory-rag) (MIT),
which solves the adjacent problem of curated project memory. No code was copied; the mechanisms were
reimplemented against this store and independently measured. See `docs/troubleshooting.md` for the traps that
came out of that comparison.

## Licence

MIT — see [LICENSE](LICENSE). Use it, fork it, ship it.
