#!/usr/bin/env python
"""ingest.py — build/update the RAG index from EVERY Hermes session archive.

Each source corpus keeps its own watermark, so profiles whose message ids are
far apart are all ingested incrementally and correctly. The old single
`watermark_msg_id` could not do this.

  python ingest.py                     # incremental across all corpora
  python ingest.py --status            # per-corpus state, no work
  python ingest.py --corpus <profile> # one corpus only
  python -m wetnoodle.ingest --full    # rebuild every AVAILABLE corpus
  python -m wetnoodle.ingest --full --corpus X   # rebuild just X
  python -m wetnoodle.ingest --purge-missing     # drop chunks of corpora whose source is gone
  python -m wetnoodle.ingest --reembed # re-embed every stored chunk and memory item

A corpus whose source database has been deleted (a removed profile) is FROZEN:
its history stays searchable but it is never re-ingested. --full deliberately
does NOT touch frozen corpora, so a routine rebuild cannot silently discard
history. Use --purge-missing when you actually want that gone.
"""
from __future__ import annotations

import json
import sys
import time

from . import rag_core as core


def status():
    a = core.audit()
    print(json.dumps(a, indent=1))
    return a


def _exclusive_lock():
    """One ingest at a time. Watermarks are read before work starts, so two overlapping
    runs read the same mark and commit the same chunks twice — the duplicates models.md
    reports are consistent with exactly that."""
    return core.exclusive_lock()


def run(full=False, only=None, purge_missing=False):
    core.init_db()
    lock = _exclusive_lock()
    if lock is None:
        print("another ingest is already running (see .ingest.lock); nothing done.")
        return 4
    if not purge_missing:
        conflict = core.model_conflict()
        if conflict:
            print(conflict)
            return 3
    sources = core.discover_sources()
    sessions = {k: v for k, v in sources.items() if v["kind"] == "sessions"}
    frozen = sorted(
        k for k, v in (core.audit().get("corpora") or {}).items()
        if v.get("chunks") and not v.get("available")
        and k not in core.DERIVED_CORPORA and k != "skill"
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
    failures = []

    for corpus, desc in sorted(sessions.items()):
        mark = 0 if full else core.get_watermark(corpus)
        t0 = time.time()
        try:
            chunks = core.build_chunks_for(corpus, desc, skip_through=mark)
        except FileNotFoundError:
            print(f"  - {corpus}: source vanished mid-run, skipped")
            continue
        except Exception as e:
            failures.append(corpus)
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
    if failures:
        # Exit nonzero: a scheduled refresh only speaks up when the job fails, so reporting
        # success here means a broken corpus stays silently frozen.
        print(f"incomplete: {len(failures)} corpus/corpora failed: {', '.join(sorted(failures))}")
        return 1
    return 0


if __name__ == "__main__":
    if "--status" in sys.argv:
        status()
        sys.exit(0)

    if "--reembed" in sys.argv:
        core.init_db()
        # Same lock as ingest: a cron run on the old model must not add old-width chunks
        # half way through the rewrite.
        lock = _exclusive_lock()
        if lock is None:
            print("another ingest or re-embed is running (see .ingest.lock); nothing done.")
            sys.exit(4)
        stored = core.stored_model()
        if stored and stored == core.MODEL and "--force" not in sys.argv:
            print(f"store is already on {stored} ({core.get_dim()}-dim); nothing to re-embed. "
                  f"Pass --force to rewrite the vectors anyway.")
            sys.exit(0)
        n = core.reembed()
        print(f"re-embedded {n} chunk(s) with {core.MODEL} ({core.get_dim()}-dim)")
        sys.exit(0)

    only = None
    if "--corpus" in sys.argv:
        i = sys.argv.index("--corpus")
        if i + 1 < len(sys.argv):
            only = sys.argv[i + 1]
    sys.exit(run(full="--full" in sys.argv, only=only,
                 purge_missing="--purge-missing" in sys.argv))
