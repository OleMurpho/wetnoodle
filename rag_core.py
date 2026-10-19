"""
rag_core.py — ingestion + retrieval core for the Hermes RAG memory archive.

MULTI-SOURCE ARCHIVE
--------------------
One vector store, many source corpora. Every Hermes profile keeps its own
session database, so the archive indexes all of them rather than pretending the
default profile is the only agent:

  corpus 'default'     <hermes>/state.db            (may be absent — a removed
                                                    profile leaves a frozen corpus)
  corpus '<profile>'   <hermes>/profiles/<p>/state.db
  corpus 'skill'       every SKILL.md under the skill roots

Each corpus carries its OWN watermark, so a profile whose message ids are far
behind another's is still ingested incrementally and correctly. A single shared
watermark cannot do this — that is the bug this design replaces.

Embeddings : a local Ollama model (default nomic-embed-text, 768-dim) at 127.0.0.1:11434.
             Cloud Ollama cannot embed — /api/embed returns 401 — so this is a hard dependency.
Vector store: rag.db (L2-normalized float32 BLOB) + a `corpus` tag on every chunk.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
import urllib.request

import re

import numpy as np

HOME = os.path.expanduser("~")
HERMES_HOME = os.environ.get("RAG_HERMES_HOME") or os.path.join(HOME, ".hermes")
RAG_DIR = os.environ.get("RAG_DIR") or os.path.join(HOME, ".wetnoodle")
RAG_DB = os.path.join(RAG_DIR, "rag.db")
OLLAMA = os.environ.get("RAG_OLLAMA", "http://127.0.0.1:11434")
MODEL = os.environ.get("RAG_EMBED_MODEL", "nomic-embed-text")  # pins vector dim: changing it needs --full
BATCH = 32
SCHEMA_VERSION = 2

# Legacy single-corpus location (kept for messages/paths that predate multi-source).
STATE_DB = os.path.join(HERMES_HOME, "state.db")

# Prefixes are NOT cosmetic: each embedding model was trained with its own convention, and using the wrong
# one measurably degrades retrieval. So the defaults are chosen per model family, and both are overridable.
#   nomic-embed-text*            doc "search_document: "   query "search_query: "
#   qwen3-embedding*             doc (none)                query "Instruct: <task>\nQuery: <q>"
#   mxbai-embed-large*           doc (none)                query "Represent this sentence for searching relevant passages: "
#   bge-m3, snowflake-arctic-*   doc (none)                query (none)
def _prefixes(model):
    m = model.lower()
    if m.startswith("nomic"):
        return "search_document: ", "search_query: "
    if m.startswith("qwen3-embedding"):
        return "", ("Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
                    "Query: ")
    if m.startswith("mxbai"):
        return "", "Represent this sentence for searching relevant passages: "
    return "", ""


DOC_PREFIX, QUERY_PREFIX = _prefixes(MODEL)
DOC_PREFIX = os.environ.get("RAG_DOC_PREFIX", DOC_PREFIX)
QUERY_PREFIX = os.environ.get("RAG_QUERY_PREFIX", QUERY_PREFIX)

def _skill_roots():
    """Every skill root that exists: <hermes>/skills plus each profile's own skills dir."""
    roots = []
    profiles_dir = os.path.join(HERMES_HOME, "profiles")
    if os.path.isdir(profiles_dir):
        for name in sorted(os.listdir(profiles_dir)):
            candidate = os.path.join(profiles_dir, name, "skills")
            if os.path.isdir(candidate):
                roots.append(candidate)
    roots.append(os.path.join(HERMES_HOME, "skills"))
    return tuple(roots)


DEFAULT_SKILL_ROOTS = _skill_roots()


# --------------------------------------------------------------------------- #
# sources
# --------------------------------------------------------------------------- #
def discover_sources():
    """Map corpus key -> descriptor for every source that exists on disk.

    {'default': {'kind':'sessions','path':...},
     '<profile>': {'kind':'sessions','path':...},
     'skill': {'kind':'skills','roots':(...)}}
    """
    out = {}
    if os.path.exists(STATE_DB):
        out["default"] = {"kind": "sessions", "path": STATE_DB}

    profiles_dir = os.path.join(HERMES_HOME, "profiles")
    if os.path.isdir(profiles_dir):
        for name in sorted(os.listdir(profiles_dir)):
            p = os.path.join(profiles_dir, name, "state.db")
            if os.path.exists(p):
                out[name] = {"kind": "sessions", "path": p}

    # Imported session databases from OTHER machines. Drop a copy at imports/<name>/state.db and it
    # becomes corpus "import_<name>"; keeping it out of HERMES_HOME/profiles means a multiplexed gateway
    # never mistakes it for a servable profile. Prefer export_store.py/import_store.py (text, no db copy).
    imports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "imports")
    if os.path.isdir(imports_dir):
        for name in sorted(os.listdir(imports_dir)):
            p = os.path.join(imports_dir, name, "state.db")
            if os.path.exists(p):
                out["import_" + name] = {"kind": "sessions", "path": p}

    roots = tuple(
        p for p in (os.environ.get("RAG_SKILLS_DIRS", "").split(os.pathsep) or DEFAULT_SKILL_ROOTS)
        if p
    ) or DEFAULT_SKILL_ROOTS
    out["skill"] = {"kind": "skills", "roots": roots}
    return out


def source_label(corpus, desc):
    return desc.get("path") or ",".join(desc.get("roots", ())) or corpus



# --------------------------------------------------------------------------- #
# hybrid retrieval
# --------------------------------------------------------------------------- #
# Vector search alone is weak on identifier-shaped questions ("which error code",
# "what is the executable called"): the embedding of a paraphrase drifts away from
# the literal token that answers it. A keyword component fixes exactly that case
# without a second index — the chunks are already in `meta`, so term overlap is a
# substring count. The two rankings are fused with reciprocal rank fusion, which
# needs no score calibration between a cosine and a count.
HYBRID = os.environ.get("RAG_HYBRID", "1").strip() not in ("0", "false", "no", "")
VEC_WEIGHT = float(os.environ.get("RAG_VEC_WEIGHT", "0.7"))  # vector share of the rerank blend
# Fair representation across corpora. A generic query ("Unity CLI executable") scores a little higher in
# the biggest corpora, so chunks of 2,780 default-profile messages push the answering chunk of a 782-chunk
# corpus out of the top 8 even when it is plainly relevant. Allocating slots per corpus fixes that —
# the same structural idea as hermes-memory-rag's fixed 2 wiki + 3 code + 1 memory recall.
# MEASURED WORSE and therefore OFF by default: over the 10-question eval it scored MRR 0.453 / hit@8 8/10
# against 0.478 / 9/10 for plain candidate rerank, and it introduced a miss. The hypothesis was that the
# quota would surface a small corpus's answering chunk; it surfaced that corpus's *highest scoring* chunks
# instead, which are not the same thing. Kept as a documented, switchable experiment.
PER_CORPUS = os.environ.get("RAG_PER_CORPUS", "0").strip() not in ("0", "false", "no", "")
PER_CORPUS_QUOTA = max(1, int(os.environ.get("RAG_PER_CORPUS_QUOTA", "2") or 2))

_STOP = set("""the a an and or of to in on for is are was were what which how does do did when where who
with that this it its as at by from be been being not no but if then than so we i you they he she them our
your my me us can could would should will just also about into over after before between during under again
further once here there all any both each few more most other some such only own same too very now use used
using s t don""".split())


def _tokens(text):
    return [t for t in re.findall(r"[a-z0-9_.:+-]{3,}", (text or "").lower()) if t not in _STOP]


def keyword_scores(query, meta):
    """Fraction of distinct query terms present in each chunk. None when the query has no usable terms."""
    toks = _tokens(query)
    if not toks:
        return None
    out = np.empty(len(meta), dtype=np.float32)
    for i, m in enumerate(meta):
        c = (m.get("content") or "").lower()
        out[i] = sum(1 for t in toks if t in c) / float(len(toks))
    return out


# --------------------------------------------------------------------------- #
# ollama
# --------------------------------------------------------------------------- #
def _post(path, payload, timeout=120):
    if not ollama_up():
        raise RuntimeError(
            f"Ollama daemon not reachable at {OLLAMA} — start ollama.service"
        )
    req = urllib.request.Request(
        OLLAMA + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", "replace")[:200]
        except Exception:
            pass
        raise RuntimeError(f"Ollama API returned {e.code}: {body}")


def ollama_up(timeout=5):
    try:
        with urllib.request.urlopen(OLLAMA + "/", timeout=timeout) as r:
            return b"Ollama" in r.read()[:64]
    except Exception:
        return False


def _embed_batch(texts, tries=3):
    last = None
    for i in range(tries):
        try:
            d = _post("/api/embed", {"model": MODEL, "input": texts})
            embs = d.get("embeddings")
            if not embs:
                raise ValueError(f"no embeddings in response, keys={list(d)[:6]}")
            m = np.array(embs, dtype=np.float32)
            if m.shape[0] != len(texts):
                raise ValueError(f"embedding count {m.shape[0]} != input count {len(texts)}")
            n = np.linalg.norm(m, axis=1, keepdims=True)
            n[n == 0] = 1.0
            return m / n
        except RuntimeError:
            raise  # already-informative API/health errors are not transient
        except Exception as e:
            last = e
            time.sleep(1 + 2 * i)
    raise RuntimeError(f"embedding failed after {tries} attempts: {last}")


def embed_documents(texts):
    """Batched document embeddings -> (n, dim) float32, L2-normalized."""
    parts = []
    for i in range(0, len(texts), BATCH):
        parts.append(_embed_batch([DOC_PREFIX + t for t in texts[i : i + BATCH]]))
    return np.vstack(parts) if parts else np.zeros((0, 768), np.float32)


def embed_query(text):
    return _embed_batch([QUERY_PREFIX + text])[0]


# --------------------------------------------------------------------------- #
# chunking
# --------------------------------------------------------------------------- #
MAX_CHARS = 1100  # ~340 tokens — safely inside nomic-embed-text's 2048 ctx


def split_piece(text):
    """Split an oversized message into <=MAX_CHARS pieces on newline/space."""
    if len(text) <= MAX_CHARS:
        return [text]
    pieces, cur = [], ""

    def emit(s):
        nonlocal cur
        if cur:
            pieces.append(cur)
            cur = ""
        while len(s) > MAX_CHARS:
            cut = max(s.rfind("\n", 0, MAX_CHARS), s.rfind(" ", 0, MAX_CHARS))
            if cut <= 0:
                cut = MAX_CHARS
            pieces.append(s[:cut])
            s = s[cut:].lstrip()
        cur = s

    for para in text.split("\n"):
        cand = (cur + "\n" + para) if cur else para
        if len(cand) <= MAX_CHARS:
            cur = cand
        else:
            emit(para)
    if cur:
        pieces.append(cur)
    return [p for p in pieces if p.strip()]


def source_messages(path, skip_through=0):
    """User+assistant corpus rows from ONE session db, read-only, id-ordered:
    (id, session_id, role, ts, content, source)."""
    if not path or not os.path.exists(path):
        raise FileNotFoundError(path)
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        cur = con.cursor()
        cur.execute(
            "SELECT m.id, m.session_id, m.role, m.timestamp, substr(m.content,1,20000), s.source "
            "FROM messages m JOIN sessions s ON s.id = m.session_id "
            "WHERE m.role IN ('user','assistant') AND m.content IS NOT NULL "
            "AND length(trim(m.content)) > 0 AND m.id > ? ORDER BY m.id",
            (int(skip_through),),
        )
        return cur.fetchall()
    finally:
        con.close()


def build_chunks_for(corpus, desc, skip_through=0):
    """Merge small adjacent same-session messages into chunks of <=1200 chars,
    tagged with their corpus. Oversized messages are subdivided by split_piece."""
    chunks = []
    buf = []

    def flush():
        if not buf:
            return
        first, lastb = buf[0], buf[-1]
        roles = "+".join(sorted({b[2] for b in buf}))
        chunks.append(
            {
                "msg_start": first[0],
                "msg_end": lastb[0],
                "session_id": first[1],
                "role": roles,
                "ts": first[3],
                "source": first[5],
                "content": "\n".join(b[4] for b in buf),
                "corpus": corpus,
            }
        )
        buf.clear()

    for mid, sid, role, ts, content, src in source_messages(desc["path"], skip_through):
        for c in split_piece(content.strip()):
            if buf and (sid != buf[-1][1] or len("\n".join(b[4] for b in buf)) + len(c) > 1200):
                flush()
            buf.append((mid, sid, role, ts, c, src))
            if len(buf) >= 4:
                flush()
    flush()
    return chunks


# --------------------------------------------------------------------------- #
# db
# --------------------------------------------------------------------------- #
def init_db():
    con = sqlite3.connect(RAG_DB)
    cur = con.cursor()
    cur.execute("PRAGMA journal_mode=WAL")  # reader/writer non-blocking
    cur.execute("PRAGMA busy_timeout=5000")
    cur.execute(
        """CREATE TABLE IF NOT EXISTS chunks(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            msg_start INTEGER, msg_end INTEGER, session_id TEXT,
            role TEXT, ts REAL, source TEXT, content TEXT, embedding BLOB)"""
    )
    cur.execute("CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT)")

    # --- migration: single-corpus schema -> multi-source ------------------- #
    cols = [r[1] for r in cur.execute("PRAGMA table_info(chunks)")]
    if "corpus" not in cols:
        cur.execute("ALTER TABLE chunks ADD COLUMN corpus TEXT")
        # Everything already present came from the pre-multi-source layout:
        # skill rows were tagged source='skill', the rest from the default db.
        cur.execute("UPDATE chunks SET corpus='skill' WHERE source='skill'")
        cur.execute("UPDATE chunks SET corpus='default' WHERE corpus IS NULL")
        legacy = cur.execute(
            "SELECT value FROM meta WHERE key='watermark_msg_id'"
        ).fetchone()
        if legacy:
            cur.execute(
                "INSERT INTO meta(key, value) VALUES(?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (watermark_key("default"), legacy[0]),
            )

    cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_ts ON chunks(ts)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_session ON chunks(session_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_msg_start ON chunks(msg_start)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_msg_end ON chunks(msg_end)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_corpus ON chunks(corpus, msg_end)")
    cur.execute(
        "INSERT INTO meta(key, value) VALUES('schema_version',?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (str(SCHEMA_VERSION),),
    )
    con.commit()
    con.close()


def watermark_key(corpus):
    return f"watermark:{corpus}"


def set_meta(key, value):
    con = sqlite3.connect(RAG_DB)
    con.execute(
        "INSERT INTO meta(key, value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, str(value)),
    )
    con.commit()
    con.close()


def get_meta(key, default=None):
    con = sqlite3.connect(f"file:{RAG_DB}?mode=ro", uri=True)
    try:
        row = con.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else default
    finally:
        con.close()


def all_meta():
    con = sqlite3.connect(f"file:{RAG_DB}?mode=ro", uri=True)
    try:
        return {k: v for k, v in con.execute("SELECT key, value FROM meta")}
    finally:
        con.close()


def get_watermark(corpus):
    return int(get_meta(watermark_key(corpus), 0) or 0)


def commit_ingest(rows, watermarks, dim, delete_corpora=()):
    """Atomic ingest commit: chunks + per-corpus watermarks + model/dim meta land
    in ONE transaction (sqlite3 `with con:` = commit on success, rollback on
    error). An interrupted run leaves no partial state, so re-running cannot
    create duplicates. Callers must clear _CACHE afterwards."""
    con = sqlite3.connect(RAG_DB)
    cur = con.cursor()
    cur.execute("PRAGMA busy_timeout=5000")
    try:
        with con:
            for corpus in delete_corpora:
                cur.execute("DELETE FROM chunks WHERE corpus=?", (corpus,))
                cur.execute("DELETE FROM meta WHERE key=?", (watermark_key(corpus),))
            if rows:
                cur.executemany(
                    "INSERT INTO chunks(msg_start, msg_end, session_id, role, ts, "
                    "source, content, embedding, corpus) VALUES(?,?,?,?,?,?,?,?,?)",
                    rows,
                )
            for corpus, mark in (watermarks or {}).items():
                cur.execute(
                    "INSERT INTO meta(key, value) VALUES(?,?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (watermark_key(corpus), str(mark)),
                )
            for k, v in (
                ("model", MODEL),
                ("dim", dim),
                ("last_ingest", time.time()),
                ("schema_version", SCHEMA_VERSION),
            ):
                cur.execute(
                    "INSERT INTO meta(key, value) VALUES(?,?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (k, str(v)),
                )
    finally:
        con.close()


def get_dim():
    """Single source of truth for embedding dimension (set at ingest from the
    model's real output shape; 768 = nomic-embed-text)."""
    return int(get_meta("dim", 768) or 768)


_CACHE = {"key": None, "mat": None, "meta": None}


def load_index():
    """Matrix + metadata with in-process caching, keyed on index state
    (row count + max msg_end + max id). Reloads automatically when ingest
    advances. Past ~10k chunks, replace linear scan with sqlite-vec or FAISS."""
    con = sqlite3.connect(f"file:{RAG_DB}?mode=ro", uri=True)
    cur = con.cursor()
    cur.execute("SELECT IFNULL(COUNT(*),0), IFNULL(MAX(msg_end),0), IFNULL(MAX(id),0) FROM chunks")
    key = cur.fetchone()
    # .get(), not subscripts: callers invalidate with _CACHE.clear(), which removes the keys entirely.
    # A subscript here turns a legitimate invalidation into a KeyError on the next search.
    if _CACHE.get("key") == key and _CACHE.get("mat") is not None:
        con.close()
        return _CACHE["mat"], _CACHE["meta"]
    cur.execute(
        "SELECT id, session_id, msg_start, msg_end, role, ts, source, content, "
        "embedding, corpus FROM chunks ORDER BY ts"
    )
    rows = cur.fetchall()
    con.close()
    meta = [
        dict(id=r[0], session_id=r[1], msg_start=r[2], msg_end=r[3],
             role=r[4], ts=r[5], source=r[6], content=r[7], corpus=r[9])
        for r in rows
    ]
    dim = get_dim()
    for i, r in enumerate(rows):
        if len(r[8]) != dim * 4:
            raise RuntimeError(
                f"rag.db chunk id={r[0]} has vector size {len(r[8])}B, expected {dim * 4}B "
                "— embedding model changed; rebuild with ingest.py --full"
            )
    mat = (
        np.vstack([np.frombuffer(r[8], dtype=np.float32) for r in rows])
        if rows
        else np.zeros((0, dim), np.float32)
    )
    _CACHE.update(key=key, mat=mat, meta=meta)
    return mat, meta


def search(query, k=8, corpus=None, hybrid=None):
    if not (query or "").strip():
        raise ValueError("empty query")
    mat, meta = load_index()
    if not meta:
        return []
    q = embed_query(query)
    if q.shape[0] != mat.shape[1]:
        raise RuntimeError(
            f"query dim {q.shape[0]} != index dim {mat.shape[1]} — rebuild with ingest.py --full"
        )
    scores = mat @ q                      # both sides L2-normalised => this is cosine
    if corpus:
        # Scope first, then rank. This used to be accepted and ignored — it only skipped the quota path —
        # so a call that asked for one corpus silently returned the whole archive.
        mask = np.fromiter((str(m.get("corpus")) == corpus for m in meta), dtype=bool, count=len(meta))
        if not mask.any():
            return []
        scores = np.where(mask, scores, -np.inf)
    order = np.argsort(-scores)
    if corpus:
        order = order[np.isfinite(scores[order])]

    use_hybrid = HYBRID if hybrid is None else bool(hybrid)
    fused = None
    if use_hybrid:
        # Two-stage retrieval: keyword terms may REORDER the vector candidates, never introduce new
        # ones. Whole-corpus fusion lost recall@k (a term in 177 chunks ties at max keyword score, and
        # RRF then ranks that tied noise above good vector hits). Within a candidate set the vector top-k
        # is always present, so reordering cannot lose a hit — it can only move a marker upward.
        cand = order[:max(k * 8, 64)]
        kw = keyword_scores(query, meta)
        if kw is not None and kw.max() > 0:
            vc = scores[cand]
            kc = kw[cand]
            vspan = float(vc.max() - vc.min()) or 1.0
            kspan = float(kc.max() - kc.min()) or 1.0
            vn = (vc - vc.min()) / vspan
            kn = (kc - kc.min()) / kspan
            blend = VEC_WEIGHT * vn + (1.0 - VEC_WEIGHT) * kn
            fused = np.empty(len(scores), dtype=np.float32)
            fused[cand] = blend
            order = cand[np.argsort(-blend)]

    # Selection: either plain top-k, or quota-first then fill. Either way the returned order is
    # relevance order, so a caller never sees the selection mechanism.
    if corpus or not PER_CORPUS:
        chosen = [(int(i), 1) for i in order]
    else:
        seen, chosen = {}, []
        for i in order:                      # pass 1: up to the quota from each corpus
            c = meta[int(i)].get("corpus")
            if seen.get(c, 0) >= PER_CORPUS_QUOTA:
                continue
            seen[c] = seen.get(c, 0) + 1
            chosen.append((int(i), 1))
        taken = {i for i, _ in chosen}
        for i in order:                      # pass 2: fill anything left, in relevance order
            if int(i) not in taken:
                chosen.append((int(i), 2))
        chosen.sort(key=lambda t: (t[1], order.tolist().index(t[0])))

    out = []
    for i, _pass in chosen:
        m = dict(meta[i])
        m["score"] = round(float(scores[i]), 4)
        if fused is not None:
            m["fused"] = round(float(fused[i]), 5)
        out.append(m)
        if len(out) >= k:
            break
    return out


def fmt_date(ts):
    import datetime
    if not ts:
        return "?"
    return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


# --------------------------------------------------------------------------- #
# audit
# --------------------------------------------------------------------------- #
def get_chunks(ids):
    """Fetch chunks by id, preserving the caller's order. Used by the agent to open the candidates it chose.

    Refs are `c<rowid>`; a bare integer is accepted too. Missing ids are reported rather than skipped
    silently, because an agent that asked for a passage and got nothing back would quietly answer from
    memory instead.
    """
    wanted, seen = [], set()
    for raw in ids:
        m = re.fullmatch(r"c?(\d+)", str(raw).strip(), re.I)
        if m and int(m.group(1)) not in seen:
            seen.add(int(m.group(1)))
            wanted.append(int(m.group(1)))
    if not wanted:
        return [], []
    con = sqlite3.connect(f"file:{RAG_DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT id, msg_start, msg_end, session_id, role, ts, source, content, corpus "
        f"FROM chunks WHERE id IN ({','.join('?' * len(wanted))})", wanted
    ).fetchall()
    con.close()
    by_id = {r["id"]: dict(r) for r in rows}
    found = [by_id[i] for i in wanted if i in by_id]
    missing = [i for i in wanted if i not in by_id]
    return found, missing


def audit():
    """Index health report: per-corpus coverage + freshness, sizes, ollama."""
    con = sqlite3.connect(f"file:{RAG_DB}?mode=ro", uri=True)
    cur = con.cursor()
    cur.execute("SELECT IFNULL(COUNT(*),0), IFNULL(MAX(ts),0) FROM chunks")
    n_chunks, max_ts = cur.fetchone()
    cur.execute("SELECT COUNT(DISTINCT session_id) FROM chunks")
    n_sessions = cur.fetchone()[0]
    per = {
        r[0]: {"chunks": r[1], "max_msg_end": r[2] or 0, "latest_ts": r[3]}
        for r in cur.execute(
            "SELECT corpus, COUNT(*), MAX(msg_end), MAX(ts) FROM chunks GROUP BY corpus"
        )
    }
    con.close()

    sources = discover_sources()
    corpora = {}
    for key in sorted(set(sources) | set(per)):
        desc = sources.get(key)
        entry = dict(per.get(key, {"chunks": 0, "max_msg_end": 0, "latest_ts": 0}))
        entry["kind"] = (desc or {}).get("kind", "missing")
        entry["path"] = source_label(key, desc or {}) if desc else None
        entry["watermark"] = get_watermark(key)
        entry["latest_chunk_ts"] = fmt_date(entry.pop("latest_ts", 0))
        if desc and desc.get("kind") == "sessions":
            try:
                con = sqlite3.connect(f"file:{desc['path']}?mode=ro", uri=True)
                row = con.execute(
                    "SELECT MAX(timestamp), MAX(id), COUNT(*) FROM messages "
                    "WHERE role IN ('user','assistant')"
                ).fetchone()
                con.close()
                entry.update(
                    available=True,
                    source_max_ts=fmt_date(row[0]),
                    source_max_msg_id=row[1],
                    source_messages=row[2],
                    fresh=(entry["max_msg_end"] or 0) >= (row[1] or 0),
                )
            except sqlite3.Error as e:
                entry.update(available=False, error=f"{type(e).__name__}: {e}", fresh=None)
        elif desc:
            entry.update(available=True, fresh=None)
        else:
            # corpus has chunks but its source db is gone (removed profile)
            entry.update(
                available=False,
                fresh=None,
                error="source corpus missing — frozen; history retained",
            )
        corpora[key] = entry

    return {
        "chunks": n_chunks,
        "session_chunks": sum(v["chunks"] for k, v in per.items() if k != "skill"),
        "skill_chunks": per.get("skill", {}).get("chunks", 0),
        "corpora_count": len(corpora),
        "sessions_covered": n_sessions,
        "latest_chunk_ts": fmt_date(max_ts),
        "rag_db_MB": round(os.path.getsize(RAG_DB) / 1e6, 1),
        "model": MODEL,
        "dim": get_dim(),
        "ollama": ollama_up(),
        "schema_version": get_meta("schema_version"),
        "corpora": corpora,
    }
