#!/usr/bin/env python
"""ingest_skills.py — index Hermes SKILL.md documents into the RAG archive.

Skills are first-class memory: they encode procedures the agent should be able
to recall *semantically*, not just by name. This indexes them into the same
`chunks` table that `rag_search` reads, under a synthetic session id
`skill:<name>`, without disturbing the session-chunk corpus.

Why synthetic ids / zero msg ids:
  `audit()` and `load_index()` derive the ingest watermark from `MAX(msg_end)`.
  Skill rows therefore store `msg_start = msg_end = 0`, which is always below
  any real message id, so adding skills can never make the session corpus look
  fresher than it is.

Idempotent: each skill's rows are deleted and re-inserted in one transaction.
A sha256 of the source file is kept in `meta` as `skill_hash:<name>`, so
unchanged skills are skipped unless --all is passed.

Usage:
    python ingest_skills.py            # refresh changed/new skills only
    python ingest_skills.py --all      # force re-embed every skill
    python ingest_skills.py --list     # show what is indexed
    python ingest_skills.py --prune    # drop rows for skills that no longer exist
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
import sys
import time

import rag_core as core


def skill_roots():
    """Skill roots, resolved by the core so there is one source of truth."""
    return list(core.discover_sources().get("skill", {}).get("roots", ()))


def discover():
    """{skill_name: path} for every SKILL.md found, first root wins."""
    found: dict[str, str] = {}
    for root in skill_roots():
        if not os.path.isdir(root):
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            if "SKILL.md" not in filenames:
                continue
            name = os.path.basename(dirpath)
            found.setdefault(name, os.path.join(dirpath, "SKILL.md"))
    return found


def _chunk_skill(name, text, mtime):
    """Split a SKILL.md into <=MAX_CHARS pieces with chunk metadata."""
    pieces = []
    for piece in core.split_piece(text.strip()):
        if piece.strip():
            pieces.append(piece)
    out = []
    for i, piece in enumerate(pieces):
        out.append(
            {
                "msg_start": 0,
                "msg_end": 0,
                "session_id": f"skill:{name}",
                "role": "skill",
                "ts": mtime,
                "source": "skill",
                "corpus": "skill",
                "content": f"[skill] {name} (part {i + 1}/{len(pieces)})\n{piece}",
            }
        )
    return out


def _delete_skill(con, name):
    con.execute("DELETE FROM chunks WHERE session_id=?", (f"skill:{name}",))


def run(force=False, prune=False):
    core.init_db()
    skills = discover()
    print(f"skills found: {len(skills)} across {len(skill_roots())} root(s)")

    con = sqlite3.connect(core.RAG_DB)
    con.execute("PRAGMA busy_timeout=5000")

    if prune:
        indexed = {
            r[0].split(":", 1)[1]
            for r in con.execute(
                "SELECT DISTINCT session_id FROM chunks WHERE source='skill'"
            )
        }
        stale = sorted(indexed - set(skills))
        for name in stale:
            _delete_skill(con, name)
            con.execute("DELETE FROM meta WHERE key=?", (f"skill_hash:{name}",))
        con.commit()
        print(f"pruned {len(stale)} stale skill(s): {stale}")

    todo = []
    for name, path in sorted(skills.items()):
        try:
            raw = open(path, encoding="utf-8", errors="replace").read()
        except OSError as e:
            print(f"  ! {name}: unreadable ({e})")
            continue
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        prev = con.execute(
            "SELECT value FROM meta WHERE key=?", (f"skill_hash:{name}",)
        ).fetchone()
        if not force and prev and prev[0] == digest:
            continue
        todo.append((name, path, raw, digest))

    if not todo:
        if not skills:
            print("no SKILL.md files found — check RAG_SKILLS_DIRS or RAG_HERMES_HOME.")
        else:
            print(f"all {len(skills)} skill(s) already indexed and unchanged (use --all to force).")
        con.close()
        return

    total_chunks = 0
    for name, path, raw, digest in todo:
        chunks = _chunk_skill(name, raw, os.path.getmtime(path))
        if not chunks:
            continue
        t0 = time.time()
        mats = core.embed_documents([c["content"] for c in chunks])
        rows = [
            (
                c["msg_start"],
                c["msg_end"],
                c["session_id"],
                c["role"],
                c["ts"],
                c["source"],
                c["content"],
                mat.tobytes(),
                c["corpus"],
            )
            for c, mat in zip(chunks, mats)
        ]
        try:
            with con:  # one transaction per skill: delete + insert + hash
                _delete_skill(con, name)
                con.executemany(
                    "INSERT INTO chunks(msg_start, msg_end, session_id, role, ts, "
                    "source, content, embedding, corpus) VALUES(?,?,?,?,?,?,?,?,?)",
                    rows,
                )
                con.execute(
                    "INSERT INTO meta(key, value) VALUES(?,?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (f"skill_hash:{name}", digest),
                )
        except sqlite3.Error as e:
            print(f"  ! {name}: commit failed ({e})")
            continue
        total_chunks += len(rows)
        print(f"  + {name}: {len(rows)} chunk(s) in {time.time() - t0:.1f}s")

    con.close()
    core._CACHE.clear()  # never serve a stale matrix after new rows
    print(f"indexed {len(todo)} skill(s), {total_chunks} chunk(s).")


def show():
    con = sqlite3.connect(f"file:{core.RAG_DB}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT session_id, COUNT(*), MAX(ts) FROM chunks WHERE source='skill' "
        "GROUP BY session_id ORDER BY session_id"
    ).fetchall()
    print(f"skills indexed: {len(rows)}")
    for sid, n, ts in rows:
        print(f"  {sid:<46} {n:>3} chunk(s)  {core.fmt_date(ts)}")
    con.close()


if __name__ == "__main__":
    if "--list" in sys.argv:
        show()
    else:
        run(force="--all" in sys.argv, prune="--prune" in sys.argv)
