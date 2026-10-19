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
benchmark — [Does it actually find the right answer?](#does-it-actually-find-the-right-answer) is the benchmark, on a real archive with a held-out set.

## Why this shape

- **Multi-corpus, per-corpus watermarks.** Each profile's database keeps its own incremental cursor, so a
  profile whose message ids lag another's still ingests correctly. One shared watermark cannot do that.
- **Sessions *and* procedures.** Conversation history answers "what did we say"; skills answer "how do we do
  this here". Both are searched by the same tool.
- **Small.** numpy + `sqlite3` + the MCP SDK. No Chroma, no LlamaIndex, no vector server. A 7,000-chunk
  archive is ~40 MB and indexes in a couple of minutes on CPU.
- **Nothing is thrown away.** A profile you delete keeps its corpus as *frozen* history rather than vanishing
  from the archive.
- **Measurable.** `eval_recall.py` scores retrieval quality (see [Does it actually find the right answer?](#does-it-actually-find-the-right-answer)).

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
| `python eval_recall.py` | retrieval gate: where each known answer comes back |
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

## Does it actually find the right answer?

We wrote 46 questions about things that really happened in this archive. Each one carries a distinctive
detail — a command, a hostname, a number — that appears in exactly one passage. Then we asked the only
question that matters: when you search, does the passage holding that detail come back **first**?

Nineteen of those questions were locked away while the system was built, and never used for any decision.
Opening the envelope gives this:

![](docs/img/held-out.svg)

Six of nineteen puts the right passage first. Nine when an optional re-reader goes back over the twenty
candidates it was handed (see the next section). The grey bar is not a result — it is the ceiling: in
seventeen of the nineteen, the right passage was already sitting in those twenty candidates.

That ceiling is the important number. Search is not failing to *find* things; it is failing to *choose*
between near-misses. Nearly everything is recoverable, and the hard part is telling the one right passage
from the nineteen that sound like it.

The twenty-seven questions used while building tell the same story at a different scale:

![](docs/img/dev.svg)

In plain terms: **plain search answers about one question in three, and a second pass over the candidates
gets you to about one in two.** On the hardest questions — the ones whose detail appears in three passages or
fewer, so lucky guessing cannot help — plain search is close to useless (2 of 8) and the second pass roughly
doubles it. This is an honest 5 out of 10, not a victory lap.

If you want the exact figures, the splits, and how each number was produced: [`docs/models.md`](docs/models.md).

## The two optional upgrades, and why they are not the default

Both were built and measured. Neither ships switched on, for different reasons.

**A careful re-read of the candidates** (a cross-encoder, ~570 MB, CPU, no account). It lifts the held-out
figure from 6 to 9 of 19. It costs a second or two per search and one extra dependency, so it stays an
optional install rather than the default:

```bash
pip install fastembed        # then the snippet in docs/models.md
```

**A model reading all twenty candidates at once and picking the best.** This is the one that actually works
well — 15 of 19 held out, and 7 of 8 on the hardest questions — but it needs a model with real reasoning
ability. On a 4 GB graphics card the smallest capable model spills to the processor and takes forty seconds
a *trivial* turn; a 1.7 B model that does fit scored *worse than plain search* (4 of 19 against 6). So it is
in this repo as an optional, off-by-default, **local-only** module, and the honest advice is: leave it off
unless you have a card that fits a 7 B model comfortably.

```bash
export RAG_JUDGE_MODEL=<a local model that fits>     # off by default; no network call when unset
```

A hosted model address is refused outright, with no override. The re-read happens on every search, and
anyone who can message an agent using this archive can trigger a search — so a rented model would mean
strangers spending your account. Local-only is a safety property here, not a preference.

## Ideas that did not survive being measured

Every one of these was built, run against the same questions, and dropped when it did not help:

![](docs/img/rejected.svg)

- **A short context line above every passage.** This is published advice for exactly the problem we had — a
  title buried in a long passage — and it made things *worse*: seven fewer questions answered first. Boilerplate
  repeated across thousands of passages competes with the words that actually matter.
- **The session's opening topic instead of a full context line.** Still three fewer.
- **Reading fifty candidates instead of twenty.** One fewer, at roughly three times the cost per search.
- **Blending the careful scorer's opinion with the original score.** Exactly nothing.
- **Keyword boosting across the whole corpus.** Two fewer questions found, because a word that appears
  everywhere outranks the words that appear once.
- **A bigger embedding model (4B instead of 0.6B).** It raises a ceiling that was not the problem. The
  answer was already in the twenty candidates most of the time; the missing piece was choosing between them.

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
