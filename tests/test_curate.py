#!/usr/bin/env python3
"""test_curate.py — what the curator must and must not turn into a fact.

No Ollama and no real archive: a three-row chunks table in memory stands in for the
transcript, and the assertions are about the rules rather than the wording. The rules
that matter are the negative ones — a project is recognised by structure (a path under a
web root, a home directory, his own GitHub repository), never by a name appearing in prose,
because otherwise the index fills with `node_modules`, `bashrc` and third-party libraries.

    python -E -m unittest discover -s tests -q
"""
import os
import sqlite3
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wetnoodle import curate  # noqa: E402

SCHEMA = """
CREATE TABLE chunks(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    msg_start INTEGER, msg_end INTEGER, session_id TEXT, role TEXT, ts REAL,
    source TEXT, content TEXT, embedding BLOB, corpus TEXT, importance REAL,
    superseded_by INTEGER, superseded_at REAL
);
"""

ROWS = [
    ("s1", "assistant+user", 1000.0,
     "I need you to familiarize yourself with a different project called STORMHOLD. "
     "It's a local home manager that utilizes XAMPP, at the file address "
     "C:\\xampp\\htdocs\\stormhold."),
    ("s2", "assistant", 2000.0,
     "STORMHOLD is a multi-app home manager with calendar, library and recipes. "
     "The repo is github.com/OleMurpho/stormhold."),
    ("s3", "assistant", 3000.0,
     "Check ~/node_modules and ~/bashrc, then read C:\\Program Files\\Oracle\\VirtualBox\\VBoxManage.exe."),
    ("s4", "assistant", 4000.0,
     "A library: github.com/comfyanonymous/ComfyUI is what you want."),
]


def _db():
    con = sqlite3.connect(":memory:")
    con.executescript(SCHEMA)
    for session, role, ts, content in ROWS:
        con.execute(
            "INSERT INTO chunks(session_id, role, ts, source, content, corpus, importance) "
            "VALUES(?,?,?,?,?,?,?)", (session, role, ts, "discord", content, "discord", 0.5))
    con.commit()
    return con


class CurateFacts(unittest.TestCase):
    def setUp(self):
        self.facts = curate.collect(_db(), min_mentions=1)

    def test_project_with_path_and_repo_is_recognised(self):
        self.assertIn("stormhold", self.facts)
        fact = self.facts["stormhold"]
        self.assertEqual(fact["name"], "STORMHOLD")
        self.assertTrue(any("htdocs" in p.lower() for p in fact["paths"]))
        self.assertEqual(fact["repo"], "github.com/OleMurpho/stormhold")

    def test_rendered_fact_answers_where_and_what(self):
        text = curate.render(self.facts["stormhold"])
        self.assertIn("C:\\xampp\\htdocs\\stormhold", text)
        self.assertIn("github.com/OleMurpho/stormhold", text)
        self.assertIn("XAMPP", text)

    def test_directories_and_binaries_are_not_projects(self):
        for junk in ("node_modules", "bashrc", "program", "vboxmanage", "oracle"):
            self.assertNotIn(junk, self.facts)

    def test_third_party_repositories_are_not_his_projects(self):
        self.assertNotIn("comfyui", self.facts)

    def test_mention_count_is_enforced(self):
        # STORMHOLD appears in two sessions; demanding three drops it.
        self.assertNotIn("stormhold", curate.collect(_db(), min_mentions=3))


if __name__ == "__main__":
    unittest.main()
