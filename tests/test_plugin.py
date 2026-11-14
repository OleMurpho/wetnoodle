#!/usr/bin/env python3
"""test_plugin.py — the memory-provider layer, which had no tests at all.

The provider imports `agent.memory_provider` from the Hermes runtime, so this file stubs
that one class and loads the plugin by path. Everything else is the real module: the real
gate, the real memory store, the real tier helpers. No Ollama needed — where an embedding
would be called, the store falls back to an unembedded row, which is what happens on a
machine without a daemon anyway.

    python -E -m unittest discover -s tests -q
"""
import importlib
import importlib.util
import os
import sqlite3
import sys
import tempfile
import types
import unittest
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[1] / "plugin"
# The plugin's own modules (memory_store, tiers, coreload) are imported flat, the way Hermes
# loads them from the profile's plugins directory.
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))


def _install_agent_stub():
    """The provider subclasses Hermes's MemoryProvider; the base class is not used here."""
    if "agent.memory_provider" in sys.modules:
        return
    agent = types.ModuleType("agent")
    provider = types.ModuleType("agent.memory_provider")

    class MemoryProvider:  # minimal shape: plugins call super().__init__ at most
        def __init__(self, *args, **kwargs):
            pass

    provider.MemoryProvider = MemoryProvider
    agent.memory_provider = provider
    sys.modules["agent"] = agent
    sys.modules["agent.memory_provider"] = provider


def _load_plugin():
    _install_agent_stub()
    if "wn_plugin" in sys.modules:
        return sys.modules["wn_plugin"]
    spec = importlib.util.spec_from_file_location(
        "wn_plugin", PLUGIN_DIR / "__init__.py",
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["wn_plugin"] = module
    spec.loader.exec_module(module)
    return module


def _hit(score, content, cid=1):
    return {"id": cid, "score": score, "content": content, "ts": 1.0,
            "corpus": "mnemosyne", "role": "user"}


class GateTests(unittest.TestCase):
    """The injection gate is the only thing standing between a weak match and the context."""

    def setUp(self):
        plugin = _load_plugin()
        self.provider = plugin.WetnoodleMemoryProvider({
            "repo": str(PLUGIN_DIR.parent), "db_dir": tempfile.mkdtemp(),
            "score_floor": 0.62, "margin": 0.0, "min_terms": 2,
        })
        self.provider._last_query = "gateway restart"

    def test_gates_on_the_best_cosine_not_the_first_hit(self):
        """Hybrid search orders by blended score, so hits[0] is not always the best cosine.

        Gating on hits[0] dropped the whole block whenever the strongest cosine sat second,
        which is exactly the ordering hybrid produces.
        """
        hits = [_hit(0.55, "the gateway restart procedure", 1),
                _hit(0.90, "restart the gateway after editing config", 2)]
        self.assertTrue(self.provider._format(hits),
                        "a 0.90 cosine must clear a 0.62 floor whatever its position")

    def test_everything_below_the_floor_is_silent(self):
        hits = [_hit(0.55, "the gateway restart procedure", 1),
                _hit(0.40, "restart the gateway", 2)]
        self.assertEqual(self.provider._format(hits), "")

    def test_margin_drops_a_near_tie(self):
        self.provider._cfg["margin"] = 0.05
        hits = [_hit(0.90, "restart the gateway", 1), _hit(0.89, "the gateway restart", 2)]
        self.assertEqual(self.provider._format(hits), "",
                         "two indistinguishable hits should not both be injected")

    def test_margin_disabled_keeps_the_near_tie(self):
        self.provider._cfg["margin"] = 0.0
        hits = [_hit(0.90, "restart the gateway", 1), _hit(0.89, "the gateway restart", 2)]
        self.assertTrue(self.provider._format(hits))


class ScopeTests(unittest.TestCase):
    def setUp(self):
        plugin = _load_plugin()
        self.plugin = plugin
        self.provider = plugin.WetnoodleMemoryProvider({
            "repo": str(PLUGIN_DIR.parent), "db_dir": tempfile.mkdtemp(),
            "corpora": "mnemosyne,skill", "exclude_corpora": "",
        })

    def test_scope_is_passed_into_search_rather_than_filtering_results(self):
        """Filtering after the fact spent k on chunks the caller could not see."""
        seen = {}

        class FakeCore:
            @staticmethod
            def search(query, **kwargs):
                seen.update(kwargs)
                return []

        self.provider._import_core = lambda: FakeCore
        self.provider._search("anything")
        self.assertEqual(seen["corpora"], ["mnemosyne", "skill"])
        self.assertEqual(seen["exclude_corpora"], [])

    def test_scope_reaches_get_chunks(self):
        import inspect
        core = importlib.import_module("wetnoodle.rag_core")
        params = inspect.signature(core.get_chunks).parameters
        self.assertIn("corpora", params)
        self.assertIn("exclude_corpora", params)


class MemoryStoreTests(unittest.TestCase):
    def setUp(self):
        self.store_mod = importlib.import_module("memory_store")
        self.store = self.store_mod.MemoryStore(
            db_path=os.path.join(tempfile.mkdtemp(), "rag.db"),
            repo=PLUGIN_DIR.parent,
        )

    def test_remove_treats_wildcards_literally(self):
        """`_` matches any character in LIKE, so an unescaped old_text deleted neighbours."""
        self.store.add("memory", "alpha_underscore entry")
        self.store.add("memory", "alphaxunderscore other entry")
        removed = self.store.remove("memory", "alpha_underscore entry")
        self.assertEqual(removed, 1)
        texts = [r["text"] for r in self.store.entries("memory")]
        self.assertEqual(texts, ["alphaxunderscore other entry"])

    def test_record_entries_leaves_exactly_the_new_set(self):
        """record_entries mirrors a whole target, so the second call must replace the first."""
        from unittest import mock
        with mock.patch.object(self.store_mod, "_live_store", return_value=self.store):
            self.store_mod.record_entries("memory", ["first fact", "second fact"])
            self.store_mod.record_entries("memory", ["third fact"])
        live = [r["text"] for r in self.store.entries("memory")]
        self.assertEqual(live, ["third fact"])

    def test_embedding_uses_the_configured_repo(self):
        """A hard-coded ~/wetnoodle made embedding fail silently anywhere else."""
        self.assertEqual(Path(self.store.repo), PLUGIN_DIR.parent)


class TierTests(unittest.TestCase):
    def test_evidence_terms_is_empty_for_short_words(self):
        """An empty set used to produce `AND ()`, a syntax error that killed promotion."""
        tiers = importlib.import_module("tiers")
        self.assertEqual(tiers.evidence_terms("a b cc ddd"), [])
        self.assertEqual(tiers.evidence_terms(""), [])

    def test_evidence_terms_is_deterministic(self):
        tiers = importlib.import_module("tiers")
        text = "gateway restarts systemd service socket listener timeout"
        self.assertEqual(tiers.evidence_terms(text), tiers.evidence_terms(text))
        self.assertEqual(tiers.evidence_terms(text), ["gateway", "listener", "restarts", "service"])


if __name__ == "__main__":
    unittest.main()
