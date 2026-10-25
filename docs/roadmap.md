# Roadmap: a memory layer for one person's archive

Not a general memory framework. The target is one user, one machine, one growing pile of
Hermes transcripts, skills and hand-written notes, embedded locally by Ollama, queried by
agents that must not confidently repeat something they corrected last week.

That constraint decides what gets built. Features are admitted on evidence from this
archive's own eval harness, and every one of them carries a kill criterion.

## Where we differ by design, not by feature count

| | Framework-shaped (mnemosyne-oss/mnemosyne) | Archive-shaped (this) |
|---|---|---|
| Input | `remember()` — an agent must decide what to store | ingest the whole session database, nothing is pre-filtered |
| Install | 360 MB: onnxruntime, sqlite-vec, llama-cpp-python | numpy + the Ollama you already run |
| Corrections | version chains in a temporal graph | `superseded_by` + a readable audit trail with reasons |
| Summaries | consolidated by the framework, opaque | notes you can read, edit and re-ingest |
| Cloud | optional sync | none, ever |
| Claims | benchmark tables | `eval_recall.py` on this corpus, dev and held-out |

The niche is auditability: every hit can be traced to a source message, every correction
records why, and nothing is rewritten silently.

## Admitted, in order

### P0 — close the gap that makes supersession inert
**1. Contradiction queue** (`contradictions.py`, ~150 lines)
At ingest, compare each new chunk against existing ones; cosine >= 0.90 with shared rare
terms (or same session and same subject line) becomes a `contradictions` row: pending.
`mnemo review` lists them; `mnemo supersede <a> <b> --reason` resolves. No LLM, no writes
to the index until a human or agent decides.
*Why:* supersession works but nothing calls it. This is the missing trigger.
*Kill criterion:* if a week of real traffic produces more than ~10 false positives per
true contradiction, widen the thresholds or drop it.

**2. Importance that is actually populated** (`importance_rules.py`, ~70 lines)
Rules over signals we already have, not vibes: user-stated constraints 0.9, notes 0.9,
skills 0.8, agent decisions 0.7, agent prose 0.5, tool output 0.35.
Sweep `RAG_IMPORTANCE_WEIGHT` over {0, 0.05, 0.1, 0.2} on **dev only**, confirm the winner
once on test.
*Kill criterion:* no dev MRR gain over 0.465 at any weight => delete the column and the
knob. A feature that cannot pay for its complexity is a liability.

### P1 — quality, still cheap
**3. Decay for session chatter only** (~50 lines)
`score -= w * age_days / 90` applied to session chunks, never to notes or skills. The
archive is allowed to remember; conversations are allowed to fade.
*Kill criterion:* any hit@8 loss on dev => off by default.

**4. Weighted FTS fusion, swept properly** (~60 lines)
Our earlier fusion test was a single unlucky weighting. Grid over vector/FTS/importance on
dev, then one confirmation on test. BM25 should help exact identifiers — model names,
error codes, package versions — which is exactly where embeddings are weakest here.
*Kill criterion:* no dev gain => keep the keyword re-order path and record why fusion lost.

### P2 — the things that make it a product, not a script
**5. `context_for(prompt)` MCP tool** (~80 lines)
Given the agent's prompt, return 5 durable facts + 3 recent decisions + anything pending
review. This is our answer to their "working memory", minus TTL eviction: we never evict,
we rank.
**6. Readable consolidation** (`consolidate.py`, ~200 lines, Ollama)
Per session, distil to atomic statements with source refs, written as markdown into the
`notes` corpus so a human can read, fix and re-ingest. Nightly, `nice -19`, capped. Their
version is a framework-internal pass; ours is a document you can audit and edit.
**7. Temporal triples from notes** (~250 lines, Ollama extraction)
(subject, predicate, object, valid_from, valid_to) built from the notes corpus, where
supersession sets `valid_to`. Answers "when did X change" and "what did we decide about Y"
without re-reading transcripts.
**8. Provenance in every result** (~30 lines)
Every hit reports corpus, session, message span and date. Already in the row; expose it.

### Not admitted
- **Binary vector quantisation (MIB).** At 8 k chunks it saves nothing we can measure and
  their own database is 2.5x larger than ours.
- **Sync of any kind.** Contradicts the point.
- **LLM in the write path by default.** Slow on 6 cores and it makes the archive
  unreproducible. Consolidation is opt-in and writes readable notes, not embeddings.
- **Working-memory TTL eviction.** We are an archive; forgetting is the failure mode.

## Measurement protocol, applied to every entry above
1. Change lands on a branch, off by default or with a kill criterion attached.
2. Dev sweep, 27 questions. Only the winner proceeds.
3. One confirmation run on the held-out 19. Never tuned on test.
4. Result recorded in the commit message, including the number it failed to beat.
5. Nothing merges to `main` while the live index still has to migrate.
