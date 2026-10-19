#!/usr/bin/env python
"""
query.py — CLI retrieval over the RAG memory archive.
  python query.py "what did we decide about battery chemistry" [-k 8] [--corpus skill] [--json]
"""
import argparse
import json
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=8)
    ap.add_argument("--corpus", default="", help="restrict to one corpus, e.g. 'skill'")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    import rag_core as core
    try:
        hits = core.search(args.query, k=args.k, corpus=args.corpus or None)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if args.json:
        for h in hits:
            h["date"] = core.fmt_date(h["ts"])
            h.pop("embedding", None)
        print(json.dumps(hits, indent=1))
        return
    if not hits:
        if args.corpus:
            print(f"no hits in corpus {args.corpus!r} — check `python ingest.py --status` for corpus names")
        else:
            print("no hits — index empty? run ingest.py first")
        return

    def header(i, h, body):
        date = core.fmt_date(h["ts"])
        return (f"{i:2d}. [{h['score']:.3f}] {date} | {h.get('corpus','?')} | {h['session_id']} | "
                f"{h['role']} | src={h['source']}\n    " + body.replace("\n", " ⏎ "))

    import render
    blocks, shown, available, _ = render.bounded(hits, header=header, hit_chars=500)
    print(f"query: {args.query!r}  ({available} hits)\n" + "-" * 72)
    print("\n".join(blocks) + render.note(shown, available))
    print(f"audit: {json.dumps(core.audit())}")


if __name__ == "__main__":
    main()