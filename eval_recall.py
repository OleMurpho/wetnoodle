#!/usr/bin/env python3.11
"""eval_recall.py — does search actually find the thing, and how high does it rank?

A gate, not a demo. Each question names a `marker`: a short string that appears only in the chunk that
should answer it. The score is the RANK the marker came back at, measured against the real search path
that agents use — so "chunks exist and are fresh" (what rag_audit proves) stops being mistaken for
"retrieval works".

Harvested from hermes-memory-rag's scripts/recall_score.py: marker-in-content rather than filename, and
the position rather than a pass/fail. Difference: one interpreter, because our search and our questions
live in the same process.

    python eval_recall.py                          # score eval/questions.local.json
    python eval_recall.py --verify-markers         # is each marker uniquely present in the index?
    python eval_recall.py --min-mrr 0.7 --json     # gate form, for CI

Exit 0 when mean reciprocal rank >= --min-mrr (default 0.45); exit 1 otherwise.

READ THIS BEFORE TOUCHING THE FLOOR. 0.45 is a REGRESSION gate, set just under the measured baseline
(MRR 0.478 over 10 questions, 2026-10-18, candidate-rerank retrieval). It exists to catch a change that
makes retrieval worse, not to certify it as good. The quality target is MRR >= 0.6 and it is tracked as a
backlog item; lowering this floor to make a change pass is the one thing that makes the gate worthless.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load(path):
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return data["questions"] if isinstance(data, dict) else data


def verify_markers(questions, core):
    """How many chunks contain each marker. 0 => the question is invalid, not a retrieval failure."""
    import sqlite3
    con = sqlite3.connect("file:%s?mode=ro" % core.RAG_DB, uri=True)
    try:
        out = []
        for q in questions:
            like = "%" + q["marker"] + "%"
            n = con.execute("SELECT COUNT(*) FROM chunks WHERE content LIKE ?", (like,)).fetchone()[0]
            out.append((q["id"], n))
        return out
    finally:
        con.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                        "eval", "questions.local.json"))
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--min-mrr", type=float, default=0.45)
    ap.add_argument("--verify-markers", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    import rag_core as core
    questions = load(a.questions)

    if a.verify_markers:
        rows = verify_markers(questions, core)
        bad = [i for i, n in rows if n == 0]
        for i, n in rows:
            print("  %-14s marker in %d chunk(s)" % (i, n))
        print("\n%d/%d markers present%s" % (len(rows) - len(bad), len(rows),
              "" if not bad else "; NOT IN INDEX: " + ", ".join(bad)))
        return 0 if not bad else 1

    rows, rr = [], []
    for q in questions:
        try:
            hits = core.search(q["question"], k=max(1, a.k),
                               corpus=q.get("corpus") or None)
        except (ValueError, RuntimeError) as e:
            print("search failed: %s" % e, file=sys.stderr)
            return 2
        rank = None
        for idx, h in enumerate(hits, 1):
            if q["marker"].lower() in (h.get("content") or "").lower():
                rank = idx
                break
        rows.append({"id": q["id"], "question": q["question"], "rank": rank,
                     "top": (hits[0]["content"][:90].replace("\n", " ") if hits else "")})
        rr.append(0.0 if rank is None else 1.0 / rank)

    mrr = sum(rr) / len(rr) if rr else 0.0
    h1 = sum(1 for r in rows if r["rank"] == 1)
    hk = sum(1 for r in rows if r["rank"] is not None)

    if a.json:
        print(json.dumps({"mrr": round(mrr, 4), "hit_at_1": h1, "hit_at_k": hk,
                          "k": a.k, "n": len(rows), "rows": rows}, indent=1))
    else:
        print("%-14s %-6s %s" % ("id", "rank", "question"))
        for r in rows:
            print("%-14s %-6s %s" % (r["id"], r["rank"] if r["rank"] else "MISS", r["question"][:88]))
            if r["rank"] is None:
                print("                        top hit was: %s" % r["top"])
        print("\nMRR %.3f | hit@1 %d/%d | hit@%d %d/%d | floor %.2f -> %s"
              % (mrr, h1, len(rows), a.k, hk, len(rows), a.min_mrr,
                 "PASS" if mrr >= a.min_mrr else "FAIL"))
    return 0 if mrr >= a.min_mrr else 1


if __name__ == "__main__":
    sys.exit(main())
