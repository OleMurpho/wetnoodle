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
