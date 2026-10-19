#!/usr/bin/env python3
"""test_smoke.py — the plumbing test, runnable with NO Ollama and no real archive.

A synthetic Hermes home is built in a temp dir (two messages plus one SKILL.md), the embedding functions
are replaced with a deterministic bag-of-words vector, and the full path is exercised: discover -> chunk ->
store -> search -> render -> export -> import -> search again. That proves the wiring, not the model's
semantic quality — quality is what eval_recall.py measures against a real archive.

    python -m pytest tests/ -q        # or:
    python tests/test_smoke.py
"""
import json
import os
import sqlite3
import sys
import tempfile
import unittest
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import rag_core as core
import render

DIM = 64


def _vec(text):
    """Stable across processes (crc32, not hash()) so the test can be rerun against a stored db."""
    v = np.zeros(DIM, dtype=np.float32)
    for tok in [t for t in "".join(c if c.isalnum() else " " for c in text.lower()).split() if len(t) > 2]:
        v[zlib.crc32(tok.encode()) % DIM] += 1.0
    n = float(np.linalg.norm(v))
    return v / n if n else v


def make_session_db(path, messages):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE sessions(id TEXT PRIMARY KEY, source TEXT, created_at REAL)")
    con.execute("CREATE TABLE messages(id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, "
                "content TEXT, timestamp REAL)")
    con.execute("INSERT INTO sessions VALUES('s1','discord',0)")
    con.executemany("INSERT INTO messages(id,session_id,role,content,timestamp) VALUES(?,?,?,?,?)",
                    [(i, "s1", role, text, 1700000000 + i) for i, (role, text) in enumerate(messages, 1)])
    con.commit()
    con.close()


class SmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="wetnoodle-test-")
        cls.home = os.path.join(cls.tmp, "hermes")
        make_session_db(os.path.join(cls.home, "profiles", "probe", "state.db"), [
            ("user", "how do I rotate the widget calibration knob"),
            ("assistant", "Hold the caliper steady for four seconds, then rotate the knob clockwise twice."),
        ])
        skill_dir = os.path.join(cls.home, "profiles", "probe", "skills", "widget-repair")
        os.makedirs(skill_dir, exist_ok=True)
        open(os.path.join(skill_dir, "SKILL.md"), "w").write(
            "# Widget repair\n\nUse the brass caliper when the knob binds. Never force it.\n")

        cls._old = (core.HERMES_HOME, core.STATE_DB, core.DEFAULT_SKILL_ROOTS,
                    core.RAG_DIR, core.RAG_DB, core.embed_documents, core.embed_query)
        core.HERMES_HOME = cls.home
        core.STATE_DB = os.path.join(cls.home, "state.db")
        core.DEFAULT_SKILL_ROOTS = (os.path.join(cls.home, "profiles", "probe", "skills"),
                                    os.path.join(cls.home, "skills"))
        core.RAG_DIR = os.path.join(cls.tmp, "store")
        os.makedirs(core.RAG_DIR, exist_ok=True)   # sqlite will not create a db inside a missing dir
        core.RAG_DB = os.path.join(core.RAG_DIR, "rag.db")
        core.embed_documents = lambda texts: np.stack([_vec(t) for t in texts])
        core.embed_query = lambda q: _vec(q)
        core._CACHE.clear()

    @classmethod
    def tearDownClass(cls):
        (core.HERMES_HOME, core.STATE_DB, core.DEFAULT_SKILL_ROOTS,
         core.RAG_DIR, core.RAG_DB, core.embed_documents, core.embed_query) = cls._old
        core._CACHE.clear()

    def test_1_discovery_finds_the_profile_and_skills(self):
        sources = core.discover_sources()
        self.assertIn("probe", sources)
        self.assertEqual(sources["probe"]["kind"], "sessions")

    def test_2_ingest_then_search(self):
        core.init_db()
        desc = core.discover_sources()["probe"]
        chunks = core.build_chunks_for("probe", desc)
        self.assertTrue(chunks, "no chunks built from the synthetic session")
        mats = core.embed_documents([c["content"] for c in chunks])
        rows = [(c["msg_start"], c["msg_end"], c["session_id"], c["role"], c["ts"], c["source"],
                 c["content"], m.tobytes(), c["corpus"]) for c, m in zip(chunks, mats)]
        core.commit_ingest(rows, {"probe": max(c["msg_end"] for c in chunks)}, int(mats.shape[1]))
        core._CACHE.clear()

        hits = core.search("rotate the calibration knob", k=3)
        self.assertTrue(hits)
        self.assertEqual(hits[0]["corpus"], "probe")
        self.assertLessEqual(len(hits), 3)

        # watermark advanced, so a second incremental run finds nothing new
        self.assertGreater(core.get_watermark("probe"), 0)
        self.assertEqual(core.build_chunks_for("probe", desc, skip_through=core.get_watermark("probe")), [])

    def test_3_audit_reports_the_corpus(self):
        a = core.audit()
        self.assertGreaterEqual(a["chunks"], 1)
        self.assertIn("probe", a["corpora"])

    def test_4_render_holds_the_budget(self):
        hits = [{"content": "x" * 5000, "score": 0.9, "ts": 1700000000, "corpus": "probe",
                 "session_id": "s1", "role": "user", "msg_start": 1, "msg_end": 2, "source": "discord"}
                for _ in range(30)]
        blocks, shown, available, cut = render.bounded(
            hits, header=lambda i, h, body: f"{i}. {body}", hit_chars=100, total_chars=400)
        self.assertEqual(available, 30)
        self.assertLess(shown, 30, "budget did not cut the list")
        self.assertTrue(cut)
        self.assertLessEqual(sum(len(b) for b in blocks), 400 + 200)

    def test_5_export_import_round_trip(self):
        jl = os.path.join(self.tmp, "export.jsonl")
        con = sqlite3.connect(core.RAG_DB)
        rows = con.execute("SELECT corpus,session_id,msg_start,msg_end,role,ts,source,content FROM chunks").fetchall()
        con.close()
        with open(jl, "w") as fh:
            for r in rows:
                fh.write(json.dumps(dict(zip(
                    ("corpus", "session_id", "msg_start", "msg_end", "role", "ts", "source", "content"), r))) + "\n")
        self.assertGreater(len(rows), 0)

        import import_store
        copy_db = os.path.join(self.tmp, "copy.db")
        old = core.RAG_DB
        core.RAG_DB = copy_db
        try:
            saved = sys.argv
            sys.argv = ["import_store.py", jl, "--corpus", "import_probe"]
            try:
                self.assertEqual(import_store.main(), 0)
            finally:
                sys.argv = saved
            hits = core.search("caliper knob clockwise", k=3, corpus="import_probe")
            self.assertTrue(hits, "imported corpus is not searchable")
            self.assertEqual(hits[0]["corpus"], "import_probe")
        finally:
            core.RAG_DB = old
            core._CACHE.clear()


    def test_6_corpus_filter_actually_scopes(self):
        """corpus= used to be accepted and ignored, so a scoped search returned the whole archive."""
        import rag_core as core
        hits = core.search("widget repair", k=5, corpus="probe")
        self.assertTrue(hits, "expected hits inside the corpus")
        self.assertTrue(all(h.get("corpus") == "probe" for h in hits),
                        f"corpus filter leaked: {[h.get('corpus') for h in hits]}")
        self.assertEqual(core.search("widget repair", k=5, corpus="no-such-corpus"), [])

    def test_7_judging_loop_snippet_and_open(self):
        """The window, the refs, and the failure path the agent depends on."""
        import render
        import rag_core as core
        text = "boilerplate " * 40 + "the RANK GUARD rejects inverted relaxation " + "tail " * 40
        sn = render.snippet(text, "what guards against inverted relaxation", 200)
        self.assertIn("RANK GUARD", sn, "snippet must show the matching window, not the head")
        self.assertLessEqual(len(sn), 202)
        self.assertEqual(render.snippet("short", "anything", 200), "short")
        hits = core.search("widget repair", k=3)
        self.assertTrue(hits)
        opened, missing = core.get_chunks(["c%d" % hits[0]["id"]])
        self.assertEqual(len(opened), 1)
        self.assertEqual(opened[0]["content"], hits[0]["content"])
        self.assertEqual(missing, [])
        _, missing = core.get_chunks(["c999999999"])
        self.assertEqual(missing, [999999999])

    def test_8_dates_are_local_not_utc(self):
        """A UTC label on a local epoch put every passage hours in the future."""
        import datetime
        import time
        import rag_core as core
        now = time.time()
        shown = datetime.datetime.strptime(core.fmt_date(now - 60), "%Y-%m-%d %H:%M")
        drift = abs((datetime.datetime.now().replace(second=0, microsecond=0) - shown).total_seconds())
        self.assertLess(drift, 600, f"fmt_date drifts {drift/60:.0f} min from local time")
        self.assertEqual(core.fmt_date(0), "?")

if __name__ == "__main__":
    unittest.main(verbosity=2)
