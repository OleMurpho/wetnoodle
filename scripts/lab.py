#!/usr/bin/env python3
"""lab.py — regenerate docs/lab.js, the two worked sections on the demo page.

Both sections are measured, not written by hand:

  judge  questions whose source abstract IS in the top eight but NOT first, so the page can show what the
         tool actually hands a model — a pool of near-misses — and let the visitor make the call.
  fail   questions whose source abstract is NOT in the top five at all. Three of those, printed with the
         rank where the answer really sits. A demo that hides its misses is advertising.

Questions are typed by hand, embedded with the local nomic-embed-text at 768 dims, truncated to the corpus's
256, L2-normalised, and ranked by cosine against every vector in docs/corpus.js. Rerun after any corpus
change; the assertions below refuse to emit a page whose numbers do not say what the page claims.

    python scripts/lab.py
"""
import base64
import json
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.environ.get("RAG_HOME", os.path.expanduser("~/rag_memory")))
os.environ["PYTHONPATH"] = ""
import rag_core as core  # noqa: E402

JUDGE = [
    ("Does cutting salt help people with high blood pressure?", 0),
    ("Should mildly symptomatic patients with a blocked basilar artery have clot removal?", 1880),
    ("Which blood or spinal fluid tests detect Alzheimer's pathology?", 2350),
]
FAIL = [
    ("Which signalling molecule rises as kidney disease gets worse?", 1410),
    ("How many mothers died in Slovakia over the last two decades?", 3290),
    ("Which biologic works best for rheumatoid arthritis after TNF failure?", 3760),
]


def load_corpus():
    raw = open(os.path.join(ROOT, "docs/corpus.js"), encoding="utf-8").read()
    data = json.loads(raw.split("=", 1)[1].rstrip().rstrip(";"))
    m = []
    for b64, scale in zip(data["vectors"], data["scales"]):
        v = np.frombuffer(base64.b64decode(b64), dtype=np.int8).astype(np.float32) * scale
        n = np.linalg.norm(v)
        m.append(v / n if n else v)
    return data, np.stack(m)


def qvec(text):
    v = np.asarray(core.embed_documents([text])[0], dtype=np.float32)[:256]
    n = np.linalg.norm(v)
    return v / n if n else v


def entry(data, V, question, answer, depth):
    scores = V @ qvec(question)
    order = [int(i) for i in np.argsort(-scores)]
    return {"q": question, "answer": answer, "rank": order.index(answer) + 1,
            "picks": order[:depth]}


def main():
    data, V = load_corpus()
    lab = {"judge": [], "fail": []}
    for k, (q, a) in enumerate(JUDGE):
        e = entry(data, V, q, a, 8)
        assert e["rank"] <= 8, f"judge question is not a pool case: {q} -> {e['rank']}"
        assert e["rank"] > 1, f"judge question is trivial (answer already first): {q}"
        random.Random(k).shuffle(e["picks"])          # fixed seed: the page must be reproducible
        lab["judge"].append(e)
    for q, a in FAIL:
        e = entry(data, V, q, a, 5)
        assert e["rank"] > 5, f"fail question actually won: {q} -> {e['rank']}"
        lab["fail"].append(e)
    for group in lab.values():
        for e in group:
            e["source_title"] = data["docs"][e["answer"]].split("\n", 1)[0]
            e["source_meta"] = data["meta"][e["answer"]]
    path = os.path.join(ROOT, "docs/lab.js")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("// Precomputed by scripts/lab.py — real queries, real nomic-embed-text vectors, "
                 "ranked against corpus.js.\n")
        fh.write("window.WETNOODLE_LAB = " + json.dumps(lab).replace("</", "<\\/") + ";\n")
    print("wrote %s (%d bytes)" % (path, os.path.getsize(path)))
    for group, rows in lab.items():
        for e in rows:
            print("  %-5s rank %-4d %s" % (group, e["rank"], e["q"][:64]))


if __name__ == "__main__":
    main()
