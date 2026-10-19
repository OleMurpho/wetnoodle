#!/usr/bin/env python3.11
"""export_store.py — write a corpus to JSONL so it can be read, diffed, kept in git, or handed to
another machine without shipping a 37 MB SQLite file.

    python export_store.py --corpus <profile> --out profile.jsonl
    python export_store.py --corpus all --limit 200 --out sample.jsonl
    python export_store.py --corpus <profile> --with-vectors --out full.jsonl

Text only by default: the other side re-embeds with its own Ollama, which is what keeps two indexes
comparable. `--with-vectors` embeds nothing and needs no Ollama (base64 float32 per line), for moving an
index to a machine that has no embedding model.
"""
import argparse
import base64
import json
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rag_core as core


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="all")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = everything")
    ap.add_argument("--with-vectors", action="store_true")
    a = ap.parse_args()

    con = sqlite3.connect("file:%s?mode=ro" % core.RAG_DB, uri=True)
    sql = ("SELECT corpus, session_id, msg_start, msg_end, role, ts, source, content, embedding "
           "FROM chunks")
    args = []
    if a.corpus != "all":
        sql += " WHERE corpus=?"
        args.append(a.corpus)
    sql += " ORDER BY ts"
    if a.limit:
        sql += " LIMIT %d" % a.limit

    n = 0
    with open(a.out, "w", encoding="utf-8") as fh, con:
        for corpus, sid, ms, me, role, ts, source, content, emb in con.execute(sql, args):
            row = {"corpus": corpus, "session_id": sid, "msg_start": ms, "msg_end": me,
                   "role": role, "ts": ts, "source": source, "content": content}
            if a.with_vectors and emb:
                row["vector_b64"] = base64.b64encode(emb).decode()
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    size = os.path.getsize(a.out)
    print("exported %d chunk(s) from corpus=%s to %s (%.1f KB)%s"
          % (n, a.corpus, a.out, size / 1024.0, ", with vectors" if a.with_vectors else ", text only"))
    return 0 if n else 1


if __name__ == "__main__":
    sys.exit(main())
