#!/usr/bin/env python3
"""Extensive, pre-registered comparison of retrieval configurations.

Decision rule, fixed before the run:
  ship iff  dMRR >= +0.040 on the fresh 32
       and  paired-bootstrap 95% CI excludes 0
       and  hit@8 does not drop
       and  old held-out set does not regress by more than 0.02
  Family of 4 candidate configs => Bonferroni alpha ~ 0.0125.

Sign convention: d = config - baseline. Positive means the config is better.
"""
import functools
import json
import os
import random
import statistics

import rag_core as core

SETS = {
    "fresh": os.path.expanduser("~/rag_memory/eval/questions.fresh.json"),
    "old_test": os.path.expanduser("~/rag_memory/eval/questions.test.json"),
}
CONFIGS = [
    ("baseline", 0.70, 0.00),
    ("fusion", 0.30, 0.00),
    ("importance", 0.70, 0.10),
    ("both", 0.30, 0.10),
    ("mid", 0.50, 0.00),
]
_raw_embed = core.embed_query


@functools.lru_cache(maxsize=512)
def _cached(text):
    return _raw_embed(text)


core.embed_query = _cached


def load(path):
    d = json.load(open(path, encoding="utf-8"))
    return d["questions"] if isinstance(d, dict) else d


def rank_of(q, k=8):
    hits = core.search(q["question"], k=k)
    marker = q["marker"].lower()
    for i, h in enumerate(hits, 1):
        if marker in (h.get("content") or "").lower():
            return i
    return None


def rr(r):
    return 0.0 if not r else 1.0 / r


def boot(deltas, n=20000, seed=7):
    random.seed(seed)
    m = sorted(statistics.mean([random.choice(deltas) for _ in deltas]) for _ in range(n))
    return m[int(0.025 * n)], m[int(0.975 * n)]


def binom_p(k, n):
    """Two-sided exact McNemar p-value from k successes in n discordant pairs."""
    if n == 0:
        return 1.0
    from math import comb
    tail = sum(comb(n, i) for i in range(min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main():
    tables = {}
    for set_name, path in SETS.items():
        qs = load(path)
        print(f"\n=== {set_name}: {len(qs)} questions ===", flush=True)
        per = {}
        for label, vec, imp in CONFIGS:
            core.VEC_WEIGHT, core.IMPORTANCE_WEIGHT = vec, imp
            ranks = [rank_of(q) for q in qs]
            per[label] = ranks
            print(f"  {label:<11} vec{vec:.2f} imp{imp:.2f}  MRR {statistics.mean(rr(r) for r in ranks):.3f}"
                  f"  hit@1 {sum(1 for r in ranks if r == 1)}/{len(qs)}"
                  f"  hit@8 {sum(1 for r in ranks if r)}/{len(qs)}", flush=True)
        tables[set_name] = (qs, per)

    print("\n=== paired analysis (d = config - baseline; positive = config better) ===")
    for set_name, (qs, per) in tables.items():
        base = per["baseline"]
        print(f"\n  [{set_name}]  baseline MRR {statistics.mean(rr(r) for r in base):.3f}")
        for label, _v, _i in CONFIGS[1:]:
            oth = per[label]
            d = [rr(o) - rr(b) for b, o in zip(base, oth)]
            lo, hi = boot(d)
            wins = sum(1 for x in d if x > 0)
            losses = sum(1 for x in d if x < 0)
            gained = sum(1 for b, o in zip(base, oth) if o and not b)
            lost = sum(1 for b, o in zip(base, oth) if b and not o)
            # leave-one-out sensitivity on dMRR
            loo = [statistics.mean(d[:i] + d[i + 1:]) for i in range(len(d))]
            print(f"    {label:<11} dMRR {statistics.mean(d):+.3f}  CI [{lo:+.3f},{hi:+.3f}]"
                  f"  win/loss {wins}/{losses}  hit@8 +{gained}/-{lost} (p={binom_p(gained, gained + lost):.2f})"
                  f"  LOO {min(loo):+.3f}..{max(loo):+.3f}")

    qs, per = tables["fresh"]
    print("\n=== per-question where any config differs from baseline (fresh) ===")
    for i, q in enumerate(qs):
        row = {c[0]: per[c[0]][i] for c in CONFIGS}
        if len(set(row.values())) > 1:
            print(f"  {q['id']}  " + "  ".join(f"{k}={row[k] or 'MISS'}" for k in row)
                  + f"  | {q['question'][:58]}")


if __name__ == "__main__":
    main()
