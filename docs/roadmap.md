# Roadmap: harvest only what improves this product

A memory layer for one person's archive: one user, one machine, Hermes transcripts plus
notes, embedded by local Ollama, queried by agents that must not repeat a corrected fact.

Nothing enters this list because a competitor has it. Each entry states the evidence that
it would help *here*, and the measurement that can kill it.

## Admitted — with evidence

### 1. Contradiction queue (`contradictions.py`)
**Evidence:** supersession works (verified: a wrong answer ranked #1, retired, correction
promoted to #1) but nothing in the system ever *calls* it. Today the only trigger is a human
noticing. The archive is known to contain at least two live contradictions from one HVAC
thread. A mechanism with no trigger is dead weight.
**Build:** at ingest and on demand, find near-neighbour pairs (cosine >= 0.90) that share rare
terms, exclude continuations in the same session, and record them as `pending` in a
`contradictions` table. Nothing is rewritten until a human or agent resolves a pair.
**Win condition:** it surfaces the known HVAC pair, and a spot-check of the top pairs finds
mostly genuine disagreements rather than restatements.

### 2. FTS fusion, swept rather than guessed
**Evidence:** the misses are disproportionately exact-identifier questions — "Which command
pulls the embedding model", "What is the store's chunk cap in characters", "What is the local
IP of the second machine", "Which event id marks an unexpected shutdown". Embeddings are worst
exactly there; BM25 is strongest there. Our single earlier fusion test used one arbitrary
weighting and lost, which is not evidence that fusion loses.
**Build:** sweep vector/FTS weights on dev only, confirm the winner once on test.
**Win condition:** dev MRR above 0.465. If the sweep cannot beat the keyword re-order path,
fusion is dead here and the roadmap says so permanently.

### 3. Importance, populated or deleted
**Evidence:** none yet. The column exists and does nothing because every row is 0.5.
**Build:** rules over signals already in the data (user constraints, notes, skills, decisions,
prose, tool output), then a dev sweep of `RAG_IMPORTANCE_WEIGHT`.
**Win condition:** measurable dev gain. Otherwise the column and the knob are removed — an
unproven scoring term is a bug waiting to misfire.

### 4. Provenance in results (30 lines)
**Evidence:** the niche is auditability; a hit you cannot trace is a hit you cannot trust. The
data is already in the row, just not exposed.
**Win condition:** every result names corpus, session, message span and date.

## Rejected now — and why

| Feature (theirs) | Why not here |
|---|---|
| Working-memory TTL eviction | We are an archive. Forgetting is the failure mode, not the feature. |
| `context_for(prompt)` injection | Hermes already injects memory each turn; a second injector would duplicate it. No evidence of a gap. |
| Readable consolidation via LLM | Nights of CPU on 6 cores, unproven gain, and it makes the archive unreproducible. Reconsider only if retrieval of facts (not passages) is shown to matter. |
| Temporal triples / entity extraction | Large build with no query in our set that needs it. Would be aspiration, not improvement. |
| Binary vector quantisation | No measurable benefit at 8 k chunks; their own database is 2.5x larger than ours. |
| Sync, cloud, multi-user | Contradicts the product. |

## Results — measured, and mostly negative

Both ranking experiments were swept on dev and then confirmed once on the held-out set.
Neither survived.

| Experiment | dev MRR | held-out MRR | verdict |
|---|---|---|---|
| baseline | 0.465 | 0.591 | — |
| keyword-dominant fusion (`RAG_VEC_WEIGHT` 0.7 -> 0.3) | **0.591** | 0.604 | no transfer, not shipped |
| corpus-derived importance (`RAG_IMPORTANCE_WEIGHT` 0.10) | **0.521** | 0.596 (hit@8 15 -> 14) | no transfer, knob stays 0.0 |
| contradiction queue | — | — | **killed** |

**Contradiction queue: killed.** It produced 1,634 candidates on this archive. Spot-checking
the top of the queue found no true contradictions — what it actually catches is the same
document ingested twice (overlapping skill corpora: `macos-apple-tools` vs `macos-computer-use`,
`article-illustration` vs `baoyu-content-illustration`) and chunks of one long message that share
an opening. Lexical heuristics cannot separate "same subject, different claim" from "same subject,
reworded" — cosine saturates, and number/negation asymmetry fires on dates and paths. Deleted
rather than shipped behind a flag.

**The real finding.** Two independent changes each produced a large dev gain (+0.126 and +0.056)
and then disappeared on held-out data. That is the signature of an exhausted dev set: 27 questions
tuned against for weeks will accept almost anything. Any further tuning against it is self-deception.
The next eval work is a fresh held-out set drawn from sessions that postdate this tuning, not
another sweep.

**Importance**: the column stays because dropping a SQLite column means rebuilding the table for
zero behavioural gain; it defaults to 0.5 and the weight stays 0.0, so it is inert. That is a
compromise, stated rather than hidden.

## The fresh held-out set (2026-10-25)

`~/rag_memory/eval/questions.fresh.json` — 32 questions, **report only, never tuned against**.

Construction, so it can be repeated honestly:
1. Pool = every chunk whose session is not the tuning session and whose text is >= 320 chars.
2. Random sample, seed 42 — no hand-picking of questions whose answers were already known.
3. One question per sampled chunk, written to be answered *by* that chunk, with a short
   marker string as the answer.
4. Mechanical validation, not judgement: every marker must appear in its target chunk, must
   appear in **at most 6** chunks archive-wide (or it cannot prove which chunk answered), and
   must not leak into the question text. Eight markers failed the first rule and were tightened
   to rarer phrases before the set was frozen.

Not committed to this repository: the markers name private projects, hosts and paths.

**Reference baseline (default settings, live index): MRR 0.454, hit@1 12/32, hit@8 20/32.**

That is *below* dev (0.465) and well below the old held-out set (0.591), which confirms the
suspicion that the older sets had been tuned into being easy. From here, a change is worth
keeping only if it moves this number, not the old ones.

## Protocol
1. Branch, off by default or kill criterion attached.
2. Dev sweep (27 questions). Only the winner proceeds.
3. One confirmation on the held-out 19. Never tuned on test.
4. The number it failed to beat goes in the commit message either way.
