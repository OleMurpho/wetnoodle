#!/usr/bin/env python3.11
"""import_store.py — the other half of federation: ingest a JSONL export as a corpus.

    python import_store.py peer.jsonl --corpus import_peer
    python import_store.py sample.jsonl --db /tmp/copy.db --dry-run

Text-only exports are re-embedded here with the local Ollama model (so the two indexes stay comparable);
exports made with --with-vectors are inserted as-is and need no Ollama. Read-only in spirit: it only ever
APPENDS a corpus, and --dry-run proves what would happen without touching the store.
"""
import argparse
import base64
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rag_core as core

BATCH = 32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--corpus", required=True, help="target corpus name, e.g. import_peer")
    ap.add_argument("--db", default="", help="target database (default: the live one)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if a.db:
        core.RAG_DB = a.db
    rows = []
    with open(a.jsonl, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        print("nothing to import"); return 1

    corpus = a.corpus if a.corpus.startswith("import_") else "import_" + a.corpus
    have_vectors = all("vector_b64" in r for r in rows)
    print("import: %d chunk(s) -> corpus %r%s%s"
          % (len(rows), corpus, " (vectors included)" if have_vectors else " (will embed locally)",
             " [dry run]" if a.dry_run else ""))
    if a.dry_run:
        print("would write to %s" % core.RAG_DB); return 0

    core.init_db()
    dim = core.get_dim() or 0
    inserted = 0
    for start in range(0, len(rows), BATCH):
        batch = rows[start:start + BATCH]
        if have_vectors:
            vecs = [list(base64.b64decode(r["vector_b64"])) for r in batch]
        else:
            import numpy as np
            vecs = [v.tobytes() for v in core.embed_documents([r["content"] for r in batch])]
        payload = []
        for r, v in zip(batch, vecs):
            emb = v if isinstance(v, (bytes, bytearray)) else bytes(v)
            dim = len(emb) // 4
            payload.append((r.get("msg_start", 0), r.get("msg_end", 0), r.get("session_id", "import"),
                            r.get("role", "unknown"), r.get("ts", 0), r.get("source", a.jsonl),
                            r["content"], emb, corpus))
        core.commit_ingest(payload, {}, dim or 0)
        inserted += len(payload)
        print("  %d/%d" % (inserted, len(rows)))
    core._CACHE["mat"] = None; core._CACHE["key"] = None
    print("imported %d chunk(s) into corpus %r" % (inserted, corpus))
    return 0


if __name__ == "__main__":
    sys.exit(main())
