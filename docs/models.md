# Choosing an embedding model

The store is model-agnostic, but the **dimension** and the **query format** are not. Changing model means
re-embedding everything (`ingest.py --full`), and using the wrong query prefix costs real accuracy.

## Measured on a real archive

Setup: 2,000 chunks sampled from a 7,467-chunk Hermes archive (every answering chunk plus 1,955 random
distractors), 46 questions, 27 dev / 19 held out, identical chunks for both models, each model embedding its
own queries with its own required prefix. "strict" = questions whose answer string appears in ≤3 chunks,
i.e. precise factual recall rather than loose topical recall.

| | nomic-embed-text (137M, 768d) | Qwen3-Embedding-0.6B (0.6B, 1024d) |
|---|---|---|
| dev hit@1 | 9/27 (33%) | **10/27 (37%)** |
| test hit@1 | 6/19 (32%) | **7/19 (37%)** |
| dev strict hit@1 | 0/4 | **1/4** |
| test strict hit@1 | 1/8 (12%) | **3/8 (38%)** |
| test strict hit@3 | 2/8 (25%) | **5/8 (62%)** |
| throughput on CPU (Ryzen 5 5500) | ~14 docs/s | ~5 docs/s |

Both models are mediocre in absolute terms on this corpus: 37% hit@1 is not a good memory system. The gain
that matters is on the strict stratum, where precise facts nearly stop being retrievable with nomic.

## What did NOT work (measured, dev set)

| change | result | why it lost |
|---|---|---|
| centring the embeddings (remove the mean direction) | 8/27 vs 9/27 | vectors are ~0.80 aligned with the mean, but that shared component carries topic signal, not just noise |
| whitening (PCA scaling, p=0.5 and 1.0) | 9–10/27 | amplifies low-variance noise as fast as signal |
| pseudo-relevance feedback (dense query expansion) | 12–13/27 | top-5 documents are often from the wrong topic and drag the query with them |
| reciprocal rank fusion instead of a z-score blend | 11/27 | ties in the lexical channel turn RRF into array-order noise |
| removing 1,547 exact-duplicate chunks | 12/27 vs 13/27 | duplicates of a *correct* chunk are free hits; dedup mostly removes those |
| message-level chunks (1 message = 1 unit) | 8/27 vs 13/27 | single conversational turns are too thin to match on |
| per-corpus slot quotas | 0.453 vs 0.478 MRR | allocates slots to a corpus's highest-*scoring* chunks, not its answering chunk |

The one lexical change that helped on dev (real BM25 with IDF, TF saturation and length normalisation,
10 → 13/27) **did not replicate on the held-out set** (9 → 8/19). Treat it as neutral: it is in the code
because it is the correct estimator, not because it was measured to win.

## Recommendation

- Default `nomic-embed-text`: fastest on CPU, smallest vectors, adequate for "what did we discuss" recall.
- Switch to `qwen3-embedding:0.6b` when precise-fact recall matters more than indexing speed:
  `RAG_EMBED_MODEL=qwen3-embedding:0.6b` then `python ingest.py --full`.
- Above 0.6B the CPU cost becomes impractical (4B ≈ 1 doc/s, hours per rebuild); at that point the honest
  alternative is a cross-encoder reranker over the top-20 rather than a bigger embedder.
- If you change model, delete the old `rag.db` rather than mixing: a dimension mismatch surfaces as
  "corruption" and wastes an afternoon.

## Reranking (the biggest single win, measured)

Two-stage retrieval: the embedder's job is **recall** — get the answer into a candidate list — and a
cross-encoder's job is **precision** — order that list by reading query and passage jointly, which a
bi-encoder cannot do at index time. Cost is per-query latency, not an index rebuild.

Tested with `BAAI/bge-reranker-v2-m3-int8` via `fastembed` (ONNX, 570 MB, CPU — no torch), reranking the
top-20 candidates, same 2,000-chunk corpus and question sets:

| | retriever only | + cross-encoder | oracle (answer in top-20) |
|---|---|---|---|
| nomic, dev hit@1 | 9/27 (33%) | **13/27 (48%)** | 19/27 (70%) |
| nomic, dev hit@3 | 15/27 (56%) | **17/27 (63%)** | — |
| Qwen3-0.6B, dev hit@1 | 11/27 (41%) | **14/27 (52%)** | 24/27 (89%) |
| Qwen3-0.6B, test hit@1 | 6/19 (32%) | **9/19 (47%)** | 17/19 (89%) |
| Qwen3-0.6B, test hit@3 | 12/19 (63%) | **15/19 (79%)** | — |
| Qwen3-0.6B, test strict hit@1 | 2/8 (25%) | **4/8 (50%)** | — |

Read the oracle column carefully: with Qwen3 the answering passage is in the top 20 **89%** of the time, yet
the reranker only converts about half of that into rank 1. So past this point the **reranker** is the
bottleneck, not the embedder. A bigger embedder raises the ceiling; it does not fix the conversion.

Two things that did **not** work here, both measured:

- **Reranking 50 candidates instead of 20.** It costs ~3× per query and on a 6-core CPU it failed to clear ten
  questions in 46 minutes. More candidates only help if the reranker can afford to read them.
- **Blending the cross-encoder score with the retriever score** — included in the harness, no gain over the
  cross-encoder alone.

If you want to push further, spend it on a stronger reranker (fp32 `BAAI/bge-reranker-v2-m3` or
`jinaai/jina-reranker-v2-base-multilingual`), not on a larger embedder.

## Listwise LLM reranking — the measured answer to "the reranker is the bottleneck"

A cross-encoder scores each (query, document) pair in isolation. A listwise judge reads the query and all
candidates in **one prompt** and ranks them against each other, so it costs one call per query instead of N
scorings. On the same corpus, same splits, same top-20 pool:

| | dev hit@1 | test hit@1 | test hit@3 | test strict hit@1 |
|---|---|---|---|---|
| retriever only (Qwen3-0.6B) | 11/27 (41%) | 6/19 (32%) | 12/19 (63%) | 2/8 (25%) |
| + cross-encoder (bge-reranker-v2-m3-int8) | 14/27 (52%) | 9/19 (47%) | 15/19 (79%) | 4/8 (50%) |
| + listwise judge | **19/27 (70%)** | **15/19 (79%)** | **16/19 (84%)** | **7/8 (88%)** |
| oracle ceiling | 24/27 (89%) | 17/19 (89%) | — | — |

Combined: **34/46 (74%)** against 23/46 (50%) for the cross-encoder and 16/46 (35%) for dense alone. The judge
converts 15 of the 17 answerable questions, i.e. it is working at the ceiling the retriever supplies. Three
unrelated judges (`deepseek-v4.1-flash`, `glm-5.3-flash`, `kimi-k3`) all landed on 15/19 at MRR 0.819, a 68B
reasoning model managed 13/19, and a 20B model 11/19 — so judge quality is a real variable and a small model is
not a free substitute.

Enable it with:

```bash
export RAG_JUDGE_MODEL=qwen3:4b        # any OpenAI-compatible chat model
```

Unset by default: with no `RAG_JUDGE_MODEL`, search is plain retrieval and no network call is made.

**Hosted endpoints are refused, with no override.** The judge runs on every search, so a hosted base URL turns
every search into a billable request — including searches triggered by anyone who can message an agent using
this archive. A non-local `RAG_JUDGE_BASE_URL` disables the judge and warns on stderr; no flag enables it.
The measurements below used hosted models for the comparison only; the shipped default is local.

**Small local models do not do this job.** `qwen3:1.7b` as the judge scored *worse than no judge at all* —
6/19 → 4/19 on the held-out set, with parseable answers, so it is a capability limit rather than a parser
bug. Listwise comparison of twenty passages is not a 2B-scale task; budget for the largest model that fits.

### Three ways this measurement was wrong before it was right

All three produced plausible-looking numbers and would have led to the wrong conclusion. They are the reason
the table above is trustworthy:

- **Snippets truncated to 420 chars.** Answering chunks average ~1,044 chars, so the judge often could not see
  the fact it was being asked to rank. 11/27 → 13/27 just from showing it the whole chunk.
- **Reading every integer in the reply.** A judge answering `1) 7  2) 3` was scored as ranking 1 then 2 — the
  opposite of what it said, because list markers are digits too. This inflated one model to 14/19; with an
  honest parser the same runs score 12/19. The prompt now demands a final `ANSWER: [...]` line.
- **A 700-token output cap.** Reasoning models spend that on thinking and return *empty* content with
  `finish_reason=length`, which silently falls back to the retriever's order. One judge looked like 7/19 when
  the cap was the fault; with 2,500 tokens it scored 13/19.

Failure policy in `judge.py`: any error, timeout, or unparseable reply returns the retriever's order unchanged.
A judge outage degrades recall to the dense baseline; it never breaks search and never invents a result.

## What you actually get on a full archive

The tables above use a fixed 2,000-chunk comparison set so every change is comparable. On the real 7,674-chunk
archive, with the same 46 questions, the numbers are:

| configuration | dev (27) | held out (19) | answer present in top-20 |
|---|---|---|---|
| plain search (this kit's default) | 13/27 (48%) | 9/19 (47%) | 16/19 (84%) |
| + cross-encoder, blended (optional) | — | 10/19 (53%) | 16/19 (84%) |
| + listwise judge (optional, needs a capable local model) | — | 15/19 (79%)* | 17/19 (89%)* |

\* measured on the 2,000-chunk set with a hosted model, before the local-only rule. Kept for reference: it is
the ceiling a capable re-reader reaches, not something this kit will spend for you.

The full archive scores slightly better than the subset because more context is available, and the plain-search
gap between dev and held-out (48% vs 47%) is small — the two splits behave the same way, which is the sign that
the number is real rather than tuned.

**Why the cross-encoder is not the default.** It costs one extra dependency (fastembed, ~570 MB of ONNX) and a
second or two per search for six points on the held-out set. Reasonable if you want it; not reasonable to force
on everyone.

**Why the judge is not the default either.** Measured on this machine: `qwen3:4b` would not fit a 4 GB card
(18% GPU / 82% CPU, 41 s for a trivial prompt), and `qwen3:1.7b`, which did fit, scored **worse than plain
search** (4/19 against 6/19) — listwise comparison of twenty passages is not a small-model task. It stays in
the repo, off, for machines that can run a 7B model comfortably.
