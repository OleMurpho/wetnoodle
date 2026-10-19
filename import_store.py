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
    import sqlite3 as _sq
    _con = _sq.connect(core.RAG_DB); existing_rows = _con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]; _con.close()

    # Work out the dimension this import will actually write, once, before writing anything.
    if have_vectors and rows:
        out_dim = len(base64.b64decode(rows[0]["vector_b64"])) // 4
    elif rows:
        out_dim = len(core.embed_documents([rows[0]["content"]])[0])
    else:
        out_dim = dim

    # An export made with another embedding model is not interchangeable: different dimension, vectors never
    # comparable to this store's. Refuse rather than write, because the old behaviour took each row's dim
    # from the payload and rewrote the store's meta to match, leaving a store that looked healthy and
    # returned nonsense.
    if dim and existing_rows and out_dim and out_dim != dim:
        print("refusing to import: these vectors are %d-dim but the store already holds %d-dim vectors "
              "(export made with a different embedding model). Re-embed with ingest.py --full instead."
              % (out_dim, dim))
        return 2

    inserted = 0
    for start in range(0, len(rows), BATCH):
        batch = rows[start:start + BATCH]
        if have_vectors:
            vecs = [base64.b64decode(r["vector_b64"]) for r in batch]   # raw float32 bytes, not ints
        else:
            import numpy as np
            vecs = [v.tobytes() for v in core.embed_documents([r["content"] for r in batch])]
        payload = []
        for r, v in zip(batch, vecs):
            emb = v if isinstance(v, (bytes, bytearray)) else bytes(v)
            payload.append((r.get("msg_start", 0), r.get("msg_end", 0), r.get("session_id", "import"),
                            r.get("role", "unknown"), r.get("ts", 0), r.get("source", a.jsonl),
                            r["content"], emb, corpus))
        core.commit_ingest(payload, {}, out_dim or dim or 0)
        inserted += len(payload)
        print("  %d/%d" % (inserted, len(rows)))
    core._CACHE["mat"] = None; core._CACHE["key"] = None
    print("imported %d chunk(s) into corpus %r" % (inserted, corpus))
    return 0


if __name__ == "__main__":
    sys.exit(main())
