"""Embedding-backed memory store.

Lives inside the wetnoodle archive (rag.db) as its own table so the ingest
pipeline, which only ever touches chunks and meta, cannot disturb it.

One row per memory entry. Each row carries a local nomic embedding, so entries
are retrievable by meaning as well as by exact text. MEMORY.md and USER.md are
rendered from this table and written back as plain exports.
"""

from __future__ import annotations

import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    from coreload import load_core
except ImportError:  # package-style import
    from .coreload import load_core

def _profile_home() -> Path:
    """Profile directory this process acts for; memory_tool is authoritative, env is fallback."""
    try:
        from tools import memory_tool
        d = Path(memory_tool.get_memory_dir())
        if d.name == "memories":
            return d.parent
    except Exception:
        pass
    env = os.environ.get("HERMES_HOME")
    if env and Path(env).is_dir():
        return Path(env)
    return Path.home() / ".hermes"


def _db_path() -> Path:
    override = os.environ.get("WETNOODLE_DB_DIR")
    return (Path(override) / "rag.db") if override else (_profile_home() / "rag" / "rag.db")


def __getattr__(name):
    """Resolve per profile on each access; constants were shared across profiles."""
    if name == "DB_PATH":
        return _db_path()
    if name == "DB_DIR":
        return _db_path().parent
    if name == "DEFAULT_MEMORIES_DIR":
        return _profile_home() / "memories"
    raise AttributeError(name)


REPO = Path(os.environ.get("WETNOODLE_REPO", str(Path.home() / "wetnoodle")))

SEPARATOR = "\n\u00a7\n"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_items(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,              -- 'memory' | 'user'
    text TEXT NOT NULL,
    ts REAL NOT NULL,
    pinned INTEGER NOT NULL DEFAULT 0,
    source TEXT NOT NULL DEFAULT '',
    superseded_by INTEGER,
    embedding BLOB,
    dim INTEGER
);
CREATE INDEX IF NOT EXISTS idx_memory_kind ON memory_items(kind, superseded_by);
"""


class MemoryStore:
    def __init__(self, db_path: Optional[Path] = None, repo: Optional[Path] = None):
        self.db_path = Path(db_path or _db_path())
        self.repo = Path(repo or REPO)
        self._core = None
        self._ensure()

    # ------------------------------------------------------------------ plumbing
    def _ensure(self) -> None:
        con = sqlite3.connect(self.db_path)
        try:
            con.executescript(_SCHEMA)
            con.commit()
        finally:
            con.close()

    def _con(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.execute("PRAGMA busy_timeout=5000")
        return con

    def _embed_core(self):
        if self._core is None:
            import sys
            import numpy as np  # noqa: F401  (import check only)
            os.environ.setdefault("RAG_DIR", str(self.db_path.parent))
            self._core = load_core(Path.home() / "wetnoodle")
        return self._core

    def embed(self, text: str) -> Optional[bytes]:
        """Embed one entry with the local model; None when Ollama is unavailable."""
        try:
            core = self._embed_core()
            mat = core.embed_documents([text])[0]
            return mat.astype("float32").tobytes()
        except Exception:
            return None

    # ------------------------------------------------------------------ writes
    def add(self, kind: str, text: str, *, pinned: int = 0, source: str = "tool") -> int:
        text = text.strip()
        if not text:
            raise ValueError("empty memory text")
        blob = self.embed(text)
        con = self._con()
        try:
            with con:
                cur = con.execute(
                    "INSERT INTO memory_items(kind, text, ts, pinned, source, embedding, dim) "
                    "VALUES(?,?,?,?,?,?,?)",
                    (kind, text, time.time(), int(pinned), source, blob,
                     len(blob) // 4 if blob else None),
                )
                return int(cur.lastrowid)
        finally:
            con.close()

    def replace(self, kind: str, old_text: str, new_text: str) -> int:
        """Replace the entry containing old_text. Returns rows changed (0 or 1)."""
        new_text = new_text.strip()
        con = self._con()
        try:
            row = con.execute(
                "SELECT id, text FROM memory_items WHERE kind=? AND superseded_by IS NULL "
                "AND text LIKE ? ORDER BY pinned DESC, ts DESC LIMIT 1",
                (kind, f"%{old_text.strip()}%"),
            ).fetchone()
            if not row:
                return 0
            blob = self.embed(new_text)
            with con:
                con.execute(
                    "UPDATE memory_items SET text=?, ts=?, embedding=?, dim=?, source='tool' WHERE id=?",
                    (new_text, time.time(), blob, len(blob) // 4 if blob else None, row[0]),
                )
            return 1
        finally:
            con.close()

    def patch(self, kind: str, old_text: str, new_text: str) -> int:
        """Replace one span inside the entry that contains old_text (builtin patch semantics)."""
        con = self._con()
        try:
            row = con.execute(
                "SELECT id, text FROM memory_items WHERE kind=? AND superseded_by IS NULL "
                "AND text LIKE ? ORDER BY pinned DESC, ts DESC LIMIT 1",
                (kind, f"%{old_text.strip()}%"),
            ).fetchone()
            if not row:
                return 0
            updated = row[1].replace(old_text, new_text, 1)
            blob = self.embed(updated)
            with con:
                con.execute(
                    "UPDATE memory_items SET text=?, ts=?, embedding=?, dim=?, source='tool' WHERE id=?",
                    (updated, time.time(), blob, len(blob) // 4 if blob else None, row[0]),
                )
            return 1
        finally:
            con.close()

    def remove(self, kind: str, old_text: str) -> int:
        con = self._con()
        try:
            rows = con.execute(
                "SELECT id FROM memory_items WHERE kind=? AND superseded_by IS NULL AND text LIKE ?",
                (kind, f"%{old_text.strip()}%"),
            ).fetchall()
            if not rows:
                return 0
            with con:
                con.executemany("DELETE FROM memory_items WHERE id=?", [(r[0],) for r in rows])
            return len(rows)
        finally:
            con.close()

    # ------------------------------------------------------------------ reads
    def entries(self, kind: str) -> List[Dict[str, Any]]:
        con = self._con()
        try:
            rows = con.execute(
                "SELECT id, text, ts, pinned, source FROM memory_items "
                "WHERE kind=? AND superseded_by IS NULL ORDER BY pinned DESC, ts ASC",
                (kind,),
            ).fetchall()
        finally:
            con.close()
        return [{"id": r[0], "text": r[1], "ts": r[2], "pinned": r[3], "source": r[4]} for r in rows]

    def render(self, kind: str, char_limit: int) -> str:
        """Render entries as the injected block, preserving order, honouring the budget.

        Oldest first so the block reads the way the file always has. Entries that do
        not fit are dropped from the view, not from the store.
        """
        out, used = [], 0
        for e in self.entries(kind):
            piece = e["text"].strip()
            cost = len(piece) + len(SEPARATOR)
            if used + cost > char_limit:
                continue
            out.append(piece)
            used += cost
        return SEPARATOR.join(out)

    def search(self, query: str, k: int = 5, kind: Optional[str] = None) -> List[Dict[str, Any]]:
        import numpy as np
        core = self._embed_core()
        try:
            qv = core.embed_query(query).astype("float32")
        except Exception:
            return []
        con = self._con()
        try:
            sql = ("SELECT id, kind, text, ts, pinned, embedding FROM memory_items "
                   "WHERE superseded_by IS NULL AND embedding IS NOT NULL")
            args: Iterable[Any] = ()
            if kind:
                sql += " AND kind=?"
                args = (kind,)
            rows = con.execute(sql, args).fetchall()
        finally:
            con.close()
        if not rows:
            return []
        mat = np.vstack([np.frombuffer(r[5], dtype="float32") for r in rows])
        sims = mat @ qv
        order = np.argsort(-sims)[:k]
        return [
            {"id": rows[i][0], "kind": rows[i][1], "text": rows[i][2],
             "ts": rows[i][3], "pinned": rows[i][4], "score": float(sims[i])}
            for i in order
        ]

    # ------------------------------------------------------------------ exports
    def export_files(self, memories_dir: Path, limits: Dict[str, int]) -> None:
        """Write MEMORY.md and USER.md from the store. Best effort, never fatal."""
        memories_dir = Path(memories_dir)
        memories_dir.mkdir(parents=True, exist_ok=True)
        for kind, name in (("memory", "MEMORY.md"), ("user", "USER.md")):
            body = self.render(kind, limits.get(kind, 2200))
            try:
                (memories_dir / name).write_text(body, encoding="utf-8")
            except Exception:
                pass

    def count(self, kind: Optional[str] = None) -> int:
        con = self._con()
        try:
            if kind:
                return con.execute(
                    "SELECT count(*) FROM memory_items WHERE kind=? AND superseded_by IS NULL",
                    (kind,)).fetchone()[0]
            return con.execute("SELECT count(*) FROM memory_items WHERE superseded_by IS NULL").fetchone()[0]
        finally:
            con.close()


# ---------------------------------------------------------------- builtin bridge
LIMITS = {"memory": 2200, "user": 1375}


def _live_store() -> "MemoryStore":
    """Store bound to this profile's own archive."""
    return MemoryStore(db_path=_db_path(), repo=REPO)


def sync_files_from_store(memories_dir: Optional[Path] = None,
                          limits: Optional[Dict[str, int]] = None) -> bool:
    """Rewrite MEMORY.md / USER.md from the archive.

    Called by the builtin store immediately before it loads from disk, which makes the
    archive authoritative while every downstream behaviour (block rendering, limits,
    threat scanning, the memory tool) stays Hermes' own code.
    Returns True when the archive held entries and the files were refreshed.
    """
    try:
        store = _live_store()
        if store.count() == 0:
            return False
        target_dir = Path(memories_dir or os.environ.get("WETNOODLE_MEMORIES_DIR")
                          or (_profile_home() / "memories"))
        store.export_files(target_dir, limits or LIMITS)
        return True
    except Exception:
        return False


def record_entries(kind: str, entries: List[str], *, source: str = "builtin") -> int:
    """Mirror the builtin store's entry list for one target into the archive.

    The file stays the write path; this makes the archive an exact copy of it,
    with an embedding per entry so the same text is searchable by meaning.
    """
    if kind not in ("memory", "user"):
        return 0
    entries = [e.strip() for e in entries if e and e.strip()]
    store = _live_store()
    con = store._con()
    try:
        with con:
            # hide rather than delete: an entry that no longer fits the injected block stays
            # recoverable in the archive instead of being silently destroyed
            con.execute("UPDATE memory_items SET superseded_by=-1 WHERE kind=?", (kind,))
    finally:
        con.close()
    if not entries:
        return 0
    try:
        blobs = store._embed_core().embed_documents(entries) if entries else []
    except Exception:
        blobs = []
    con = store._con()
    try:
        with con:
            for i, text in enumerate(entries):
                blob = blobs[i].astype("float32").tobytes() if i < len(blobs) else None
                row = con.execute(
                    "SELECT id FROM memory_items WHERE kind=? AND text=? "
                    "ORDER BY id DESC LIMIT 1", (kind, text)).fetchone()
                if row:
                    con.execute(
                        "UPDATE memory_items SET superseded_by=NULL, ts=?, source=?, "
                        "embedding=COALESCE(?, embedding) WHERE id=?",
                        (time.time(), source, blob, row[0]))
                else:
                    con.execute(
                        "INSERT INTO memory_items(kind, text, ts, pinned, source, embedding, dim) "
                        "VALUES(?,?,?,?,?,?,?)",
                        (kind, text, time.time(), 1, source, blob,
                         len(blob) // 4 if blob else None),
                    )
    finally:
        con.close()
    return len(entries)
