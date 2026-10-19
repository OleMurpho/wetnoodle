#!/usr/bin/env python3
"""doctor.py — check every dependency wetnoodle has, before it wastes your time.

Exit 0 when it can work, 1 when it cannot. Each failure prints the command that fixes it.

    python doctor.py            # full check, including a real embedding round trip
    python doctor.py --quiet    # only the failures
"""
import json
import os
import sqlite3
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rag_core as core

OK, WARN, FAIL = "ok  ", "warn", "fail"
problems = []


def report(level, name, detail=""):
    if level == FAIL:
        problems.append(name)
    if level != OK and "--quiet" in sys.argv:
        print(f"[{level}] {name}: {detail}")
    else:
        print(f"[{level}] {name}{': ' + detail if detail else ''}")


def main():
    # python
    report(OK if sys.version_info >= (3, 11) else FAIL, "python",
           "%d.%d.%d%s" % (*sys.version_info[:3], "" if sys.version_info >= (3, 11) else " (need 3.11+)"))

    # numpy
    try:
        import numpy
        report(OK, "numpy", numpy.__version__)
    except ImportError:
        report(FAIL, "numpy", "pip install -r requirements.txt")

    # ollama daemon
    tags = None
    try:
        with urllib.request.urlopen(core.OLLAMA + "/api/tags", timeout=5) as r:
            tags = json.loads(r.read().decode())
        report(OK, "ollama daemon", core.OLLAMA)
    except (urllib.error.URLError, OSError) as e:
        report(FAIL, "ollama daemon", f"{core.OLLAMA} unreachable ({e}) — install from https://ollama.com/download")

    # embedding model present
    if tags:
        names = {m.get("name", "") for m in tags.get("models", [])}
        have = any(n == core.MODEL or n.startswith(core.MODEL + ":") for n in names)
        report(OK if have else FAIL, "embedding model", core.MODEL if have else
               f"{core.MODEL} not pulled — ollama pull {core.MODEL}")

    # a real embed round trip: the only check that proves the stack works
    if not problems:
        try:
            v = core.embed_query("wetnoodle doctor probe")
            report(OK if v.size else FAIL, "embed round trip", f"dim {v.size}")
            if v.size and abs(float((v * v).sum()) - 1.0) > 1e-3:
                report(WARN, "embed normalisation", "query vector is not unit length; cosine ranking will skew")
        except Exception as e:
            report(FAIL, "embed round trip", f"{type(e).__name__}: {e}")

    # hermes home and sources
    report(OK if os.path.isdir(core.HERMES_HOME) else FAIL, "hermes home", core.HERMES_HOME)
    sources = core.discover_sources()
    sessions = {k: v for k, v in sources.items() if v["kind"] == "sessions"}
    report(OK if sessions else WARN, "session corpora", f"{len(sessions)}: {sorted(sessions)}")
    skill_roots = [r for r in sources.get("skill", {}).get("roots", ()) if os.path.isdir(r)]
    n_skills = sum(1 for r in skill_roots for _, _, fs in os.walk(r) for f in fs if f == "SKILL.md")
    report(OK if n_skills else WARN, "skill documents", f"{n_skills} SKILL.md under {len(skill_roots)} root(s)")

    # each session db readable read-only (WAL tolerant)
    for name, desc in sorted(sessions.items()):
        try:
            con = sqlite3.connect(f"file:{desc['path']}?mode=ro", uri=True, timeout=5)
            n = con.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            con.close()
            report(OK, f"db {name}", f"{n} messages")
        except sqlite3.Error as e:
            report(FAIL, f"db {name}", f"{e} (is the file a Hermes state.db?)")

    # store writable
    try:
        os.makedirs(core.RAG_DIR, exist_ok=True)
        probe = os.path.join(core.RAG_DIR, ".doctor-probe")
        open(probe, "w").write("x")
        os.remove(probe)
        report(OK, "store dir", core.RAG_DIR)
    except OSError as e:
        report(FAIL, "store dir", f"{core.RAG_DIR}: {e}")

    if os.path.exists(core.RAG_DB):
        try:
            con = sqlite3.connect(f"file:{core.RAG_DB}?mode=ro", uri=True)
            ok = con.execute("PRAGMA quick_check").fetchone()[0]
            con.close()
            report(OK if ok == "ok" else FAIL, "store integrity", ok)
        except sqlite3.Error as e:
            report(FAIL, "store integrity", str(e))

    print()
    if problems:
        print(f"{len(problems)} problem(s): {', '.join(problems)}")
        return 1
    print("all checks passed — run: python ingest.py --full")
    return 0


if __name__ == "__main__":
    sys.exit(main())
