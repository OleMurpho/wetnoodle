#!/usr/bin/env python
"""ingest.py — build/update the RAG index from EVERY Hermes session archive.

Each source corpus keeps its own watermark, so profiles whose message ids are
far apart are all ingested incrementally and correctly. The old single
`watermark_msg_id` could not do this.

  python ingest.py                     # incremental across all corpora
  python ingest.py --status            # per-corpus state, no work
  python ingest.py --corpus <profile> # one corpus only
  python ingest.py --full              # rebuild every AVAILABLE corpus
  python ingest.py --full --corpus X   # rebuild just X
  python ingest.py --purge-missing     # drop chunks of corpora whose source is gone

A corpus whose source database has been deleted (a removed profile) is FROZEN:
its history stays searchable but it is never re-ingested. --full deliberately
does NOT touch frozen corpora, so a routine rebuild cannot silently discard
history. Use --purge-missing when you actually want that gone.
"""
from __future__ import annotations

import json
import sys
import time

import rag_core as core


def status():
    a = core.audit()
    print(json.dumps(a, indent=1))
    return a


def run(full=False, only=None, purge_missing=False):
    core.init_db()
    sources = core.discover_sources()
    sessions = {k: v for k, v in sources.items() if v["kind"] == "sessions"}
    frozen = sorted(
        k for k, v in (core.audit().get("corpora") or {}).items()
        if v.get("chunks") and not v.get("available") and k != "skill"
    )

    if purge_missing and frozen:
        rows = []
        core.commit_ingest(rows, {}, core.get_dim(), delete_corpora=frozen)
        core._CACHE.clear()
        print(f"purged {len(frozen)} frozen corpus/corpora: {frozen}")
        frozen = []

    if only:
        if only == "skill":
            print("corpus 'skill' is owned by ingest_skills.py — run that instead.")
            return 0
        if only not in sessions:
            print(f"no such session corpus: {only!r}. available: {sorted(sessions)}")
            return 2
        sessions = {only: sessions[only]}

    if not sessions:
        print("no session corpora available on disk.")
        if frozen:
            print(f"frozen (source deleted, history kept): {frozen}")
        return 0

    print(f"session corpora: {sorted(sessions)}")
    if frozen:
        print(f"frozen, skipped   : {frozen}")

    watermarks = {}
    grand_chunks = 0

    for corpus, desc in sorted(sessions.items()):
        mark = 0 if full else core.get_watermark(corpus)
        t0 = time.time()
        try:
            chunks = core.build_chunks_for(corpus, desc, skip_through=mark)
        except FileNotFoundError:
            print(f"  - {corpus}: source vanished mid-run, skipped")
            continue
        except Exception as e:
            print(f"  ! {corpus}: build failed ({type(e).__name__}: {e})")
            continue

        if not chunks:
            print(f"  = {corpus}: up to date (watermark {mark})")
            continue

        texts = [c["content"] for c in chunks]
        mats = core.embed_documents(texts)
        rows = [
            (
                c["msg_start"], c["msg_end"], c["session_id"], c["role"],
                c["ts"], c["source"], c["content"], mat.tobytes(), c["corpus"],
            )
            for c, mat in zip(chunks, mats)
        ]
        new_mark = max(c["msg_end"] for c in chunks)
        core.commit_ingest(
            rows,
            {corpus: new_mark},
            int(mats.shape[1]),
            delete_corpora=[corpus] if full else (),
        )
        watermarks[corpus] = new_mark
        grand_chunks += len(rows)
        print(
            f"  + {corpus}: {len(rows)} chunk(s), watermark {mark} -> {new_mark} "
            f"in {time.time() - t0:.1f}s"
        )

    core._CACHE.clear()  # never reuse a stale in-process matrix
    if grand_chunks:
        print(f"committed {grand_chunks} chunk(s) across {len(watermarks)} corpus/corpora.")
    else:
        print("already up to date.")
    return 0


if __name__ == "__main__":
    if "--status" in sys.argv:
        status()
        sys.exit(0)

    only = None
    if "--corpus" in sys.argv:
        i = sys.argv.index("--corpus")
        if i + 1 < len(sys.argv):
            only = sys.argv[i + 1]
    sys.exit(run(full="--full" in sys.argv, only=only,
                 purge_missing="--purge-missing" in sys.argv))
