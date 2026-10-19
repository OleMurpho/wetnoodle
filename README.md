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

## How a search actually works

**The model that retrieves is the model that judges.** There is no reranker and no judge endpoint anywhere in
this design — not local, not cloud.

1. `rag_search` returns a **pool of candidates**: a dozen passages, each with a ref, a date, a corpus and the
   part of the passage where your query's own words appear.
2. The agent reads them and calls `rag_open` on the ones that actually answer the question.
3. The agent answers from those passages, or says it could not find it.

Passage-finding is left to vectors, which are good at proposing a neighbourhood. Choosing between near-misses
is left to the model that asked — the one already loaded, already holding the conversation, and already paid
for. That model can compare a question against twenty passages; a reranker compares each passage against the
question alone, and a separate judge model would add an endpoint, a bill, and a copy of your conversation
leaving the machine.

The practical consequence: **answer quality depends on the model you point at it.** A strong model reads the
pool and picks well. A weak one takes the first hit. The system tells it not to.

## Try it without installing anything

**→ https://olemurpho.github.io/wetnoodle/**

A working index of **5,677 PubMed abstracts** that you can search in the browser. Every hit is a real paper —
title, journal, year, PMID — so any answer can be checked against its source instead of taken on trust.

Two search paths are exposed, and the page says which is which:

- **Typed queries** use BM25 (keyword) in JavaScript, because a browser has no embedding model.
- **The twelve examples** run the real embedding search: their query vectors were computed locally by
  `nomic-embed-text`, truncated to 256 dimensions and int8-quantised. It is the same cosine maths the archive
  uses, at lower precision and with no keyword rerank — the ranking you see is close to the archive's, not
  identical. The page states the rank of the abstract each example came from.

That curation is disclosed on the page. We drew 59 titles across the corpus, kept the 46 long enough to be a
fair query, and 30 of those 46 retrieved their own abstract in the top three; the examples are drawn from
those 30. It is a demo of the mechanism, not a benchmark — [Does it actually find the right answer?](#does-it-actually-find-the-right-answer) is the benchmark, on a real archive with a held-out set.

## Why this shape

- **Multi-corpus, per-corpus watermarks.** Each profile's database keeps its own incremental cursor, so a
  profile whose message ids lag another's still ingests correctly. One shared watermark cannot do that.
- **Sessions *and* procedures.** Conversation history answers "what did we say"; skills answer "how do we do
  this here". Both are searched by the same tool.
- **Small.** numpy + `sqlite3` + the MCP SDK. No Chroma, no LlamaIndex, no vector server. A 7,000-chunk
  archive is ~40 MB, and re-indexing it took about seven minutes on a modest four-year-old graphics card
  (16 passages a second at roughly a thousand characters each). On the processor alone it is several times
  slower.
- **Nothing is thrown away.** A profile you delete keeps its corpus as *frozen* history rather than vanishing
  from the archive.
- **Measurable.** `eval_recall.py` scores retrieval quality (see [Does it actually find the right answer?](#does-it-actually-find-the-right-answer)).

## Requirements

| | |
|---|---|
| Python | 3.11+ |
| Ollama | **local daemon**, `127.0.0.1:11434` |
| Embedding model | `nomic-embed-text` (274 MB) — **required**, not optional |
| Disk | ~1 MB per 200 chunks |

**This is the one dependency that cannot be swapped out or skipped.** There is no fallback to keyword search,
no hosted substitute, and no reduced mode: without a local embedding model, `ingest.py` cannot index and
`query.py` cannot search. That is deliberate — an archive of your conversations should not need an API key.

Ollama Cloud will not do: its `/api/embed` returns `401` and `/v1/embeddings` `404` for embedding models. It
has to be a daemon on your own machine, at `127.0.0.1:11434` by default.

The demo page on GitHub is the single exception, and only because a web page cannot run the model:
**its typed searches are keyword-only**, and its twelve examples replay vectors that were computed locally
beforehand. The demo shows what the maths does; it is not the product running.

## Quickstart

```bash
git clone https://github.com/OleMurpho/wetnoodle.git && cd wetnoodle
./scripts/bootstrap.sh          # venv, deps, model, doctor, first ingest
./.venv/bin/python query.py "the phrase you are looking for"
```

`bootstrap.sh` is idempotent, needs no root, and clears any inherited `PYTHONPATH` before doing anything — a
foreign `PYTHONPATH` makes this venv import the wrong numpy. If it stops, `python doctor.py` names the exact fix.

## MCP registration

Give the agent the three tools — `rag_search` (candidates), `rag_open` (the ones it chose), `rag_audit`
(health). Add this to your Hermes `config.yaml` — it is YAML,
so no braces and no quotes around the keys:

```yaml
mcp_servers:
  wetnoodle:
    command: /absolute/path/to/wetnoodle/.venv/bin/python
    args: [/absolute/path/to/wetnoodle/mcp_server.py]
```

Use the **absolute** interpreter path: a system `python` will not have `numpy`/`mcp`. Restart the gateway
(or start a new session) and the tools appear as `mcp_wetnoodle_rag_search`, `mcp_wetnoodle_rag_open` and
`mcp_wetnoodle_rag_audit`.

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
| `python query.py "text" [-k N] [--corpus C]` | search from the shell |
| `python eval_recall.py` | retrieval gate: where each known answer comes back |
| `python eval_recall.py --verify-markers` | is each eval marker actually present? |
| `python export_store.py --corpus X --out f.jsonl` | corpus → text (portable, diffable) |
| `python import_store.py f.jsonl --corpus peer` | text → corpus (re-embedded locally) |
| `python doctor.py` | preflight every dependency |
| `python -E -m unittest discover -s tests -q` | smoke test, no Ollama needed (`-E` ignores a stray `PYTHONPATH`) |

## Settings

Defaults are right for almost everyone. These exist for relocating the store, not for experimenting:

| Variable | Default | Effect |
|---|---|---|
| `RAG_DIR` | `~/.wetnoodle` | where `rag.db` lives |
| `RAG_HERMES_HOME` | `~/.hermes` | the Hermes install to index |
| `RAG_EMBED_MODEL` | `nomic-embed-text` | **pins vector dim**; changing it requires `ingest.py --full` |
| `RAG_OLLAMA` | `http://127.0.0.1:11434` | the local daemon |
| `RAG_SKILLS_DIRS` | discovered | override skill roots (`:`-separated) |
| `RAG_MAX_HIT_CHARS` | `800` | per-hit cap in tool output |
| `RAG_MAX_TOTAL_CHARS` | `6000` | cap across the whole result set |

The retrieval knobs behind these (`RAG_HYBRID`, `RAG_VEC_WEIGHT`, per-corpus quotas, model prefixes) are
documented in [`docs/models.md`](docs/models.md) with the measurement that settled each one. Every one of
them is measured worse or neutral than the default, so there is no reason to reach for them.

## Does it actually find the right answer?

We wrote 46 questions about things that really happened in this archive. Each one carries a distinctive
detail — a command, a hostname, a number — that appears in exactly one passage. Then we asked the only
question that matters: when you search, does the passage holding that detail come back **first**?

Nineteen of those questions were locked away while the system was built, and never used for any decision.
Every figure in this section is measured on a fixed 2,000-passage comparison set, so each change is compared
like for like; the full archive behaves the same way (see the next section). Opening the envelope gives this:

![](docs/img/held-out.svg)

Six of nineteen puts the right passage first. The grey bar is not a result — it is the ceiling: in seventeen
of the nineteen, the right passage was already sitting among the twenty candidates search had found.

That ceiling is the important number, and it is what the design is built around. Search does not fail to
*find* things — it fails to *choose* between near-misses. So the choosing is given to the model doing the
asking, which is the one thing in the stack that can hold twenty passages and a conversation at once.
Retrieval supplies; the calling model selects.

The twenty-seven questions used while building tell the same story at a different scale:

![](docs/img/dev.svg)

In plain terms: **raw ranking answers about one question in three**, which is why nothing is meant to rely on
the first hit. On the hardest questions — detail present in three passages or fewer — it puts the right
one first **once in eight**, and inside the top three twice in eight. That is the stratum no amount of
ranking polish fixes, and it is why the answer is meant to come from a model reading a pool, not from
position one. That gap is
the entire reason `rag_search` returns candidates instead of an answer.

If you want the exact figures, the splits, and how each number was produced: [`docs/models.md`](docs/models.md).

## Ideas that did not survive being measured

Every row here was built, run against the same questions, and dropped when it did not help:

![](docs/img/rejected.svg)

- **A short context line above every passage.** This is published advice for exactly the problem we had — a
  title buried in a long passage — and it made things *worse*: seven fewer questions answered first. Boilerplate
  repeated across thousands of passages competes with the words that actually matter.
- **The session's opening topic instead of a full context line.** Still three fewer.
- **A gate that forces rare identifier terms to match.** One fewer: the questions that need it are the ones
  where the identifier is the *answer*, so it is not in the query to gate on.
- **Removing all 1,570 duplicate passages** (20% of the archive). Exactly no change — duplicates were not
  stealing rank, they were simply occupying slots nobody was fighting over.
- **Keyword boosting across the whole corpus.** Two fewer questions found, because a word that appears
  everywhere outranks the words that appear once.

Two more were abandoned before they finished, so they carry no number:

- **Ranking fifty candidates instead of twenty** cost about three times as much per search and had still not
  cleared ten questions after forty-six minutes.
- **A 4B embedding model instead of 0.6B** ran at 0.8 passages a second — nearly three hours for one index —
  and the ceiling measurement says a bigger embedder raises a limit that was not the constraint.

## Known limits

- **Paraphrase queries underperform identifier queries.** Embeddings drift from literal tokens; the keyword
  rerank only partly compensates.
- **Chunks are raw conversation.** Unlike a curated wiki, half a chunk can be boilerplate. Quality is bounded
  by what was said.
- **One embedding model per store.** Switching models without `--full` is a dimension mismatch, which reads
  like corruption.
- **No reranker, no query expansion, no agentic retrieval.** All three are plausible next steps and none is
  implemented. A second scoring pass was measured at roughly one extra question in nineteen; it is recorded in
  `docs/models.md` and is not part of this kit.
- **Not multi-user.** It indexes the Hermes install it can see; there is no auth layer.

## Other projects, and what we took from them

The marker-based recall scoring, bounded recall output, text export/rebuild and the job-runtime isolation
pattern came from [tomRumi/hermes-memory-rag](https://github.com/tomRumi/hermes-memory-rag) (MIT),
which solves the adjacent problem of curated project memory. No code was copied; the mechanisms were
reimplemented against this store and independently measured. See `docs/troubleshooting.md` for the traps that
came out of that comparison.

## Licence

MIT — see [LICENSE](LICENSE). Use it, fork it, ship it.
