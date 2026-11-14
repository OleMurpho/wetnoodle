#!/usr/bin/env python3
"""Three-tier memory rendering for MEMORY.md.

  long     stable facts, 2500 chars   — only entries that have matured (unchanged 14 days
                                        and confirmed in a later session); migrated entries
                                        are grandfathered so nothing disappears on upgrade
  middle   24h .. 14d, 1000 chars     — chronological index of recent sessions, plus facts
                                        that are still maturing
  short    last 24h, 1500 chars       — the live thread, newest first

MEMORY.md is a generated document: it is rebuilt from the archive and state.db. The builtin
Hermes store keeps reading/writing MEMORY.long.md (entries only) so its parser never sees
the generated sections.

  python3 tiers.py --render     print the three sections, write nothing
  python3 tiers.py --refresh    rewrite MEMORY.long.md, MEMORY.md and the JSON mirror
  python3 tiers.py --promote    run the maturity gate (unchanged 14d + confirmed)
  python3 tiers.py --status     counts per tier
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path

HOME = Path.home()


def _profile_home() -> Path:
    """The profile directory this process is acting for.

    memory_tool.get_memory_dir() is monkeypatched per profile, so it is the reliable
    source inside a multiplexed gateway; HERMES_HOME is the fallback for CLI runs.
    """
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
    return HOME / ".hermes"


def _paths() -> dict:
    prof = _profile_home()
    mem = prof / "memories"
    override = os.environ.get("WETNOODLE_DB_DIR")
    db = (Path(override) / "rag.db") if override else (prof / "rag" / "rag.db")
    return {
        "DB": db,
        "STATE": prof / "state.db",
        "MEM_DIR": mem,
        "COMPOSITE": mem / "MEMORY.md",
        "LONG_FILE": mem / "MEMORY.long.md",
        "MIRROR": mem / ".tiers.json",
    }


def _db() -> Path:
    return _paths()["DB"]


def _state() -> Path:
    return _paths()["STATE"]


def _mem_dir() -> Path:
    return _paths()["MEM_DIR"]


def _composite() -> Path:
    return _paths()["COMPOSITE"]


def _long_file() -> Path:
    return _paths()["LONG_FILE"]


def _mirror() -> Path:
    return _paths()["MIRROR"]


def __getattr__(name):
    """Resolve profile paths on every access.

    Module constants were captured once per process; in a multiplexed gateway that meant
    one profile's paths served every other profile and cross-wrote their memory files.
    """
    if name in ("DB", "STATE", "MEM_DIR", "COMPOSITE", "LONG_FILE", "MIRROR"):
        return _paths()[name]
    raise AttributeError(name)


BUDGET = {"long": 2500, "middle": 1000, "short": 1500}
SEP = "\n§\n"
MATURE_DAYS = 14
MIDDLE_WINDOW_DAYS = 14
SHORT_HOURS = 24

HDR = {
    "long": "LONG TERM — stable facts ({used}/{cap})",
    "middle": "MIDDLE TERM — open work, last 14 days ({used}/{cap})",
    "short": "SHORT TERM — last 24 hours ({used}/{cap})",
}


def _ro(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def ensure_schema() -> None:
    con = sqlite3.connect(_db())
    try:
        cols = [r[1] for r in con.execute("PRAGMA table_info(memory_items)")]
        with con:
            for name, decl in (("tier", "TEXT NOT NULL DEFAULT 'long'"),
                               ("first_seen", "REAL"),
                               ("last_changed", "REAL"),
                               ("confirmed_at", "REAL"),
                               ("text_hash", "TEXT")):
                if name not in cols:
                    con.execute(f"ALTER TABLE memory_items ADD COLUMN {name} {decl}")
            # grandfather everything that exists today: migration wrote the curated file, and
            # blanking memory on upgrade would be a data-loss bug, not a policy win
            con.execute("UPDATE memory_items SET tier='long' WHERE tier IS NULL OR tier=''")
            con.execute("UPDATE memory_items SET first_seen=ts WHERE first_seen IS NULL")
            con.execute("UPDATE memory_items SET last_changed=ts WHERE last_changed IS NULL")
            con.execute("UPDATE memory_items SET confirmed_at=ts WHERE confirmed_at IS NULL")
            for rid, text in con.execute("SELECT id, text FROM memory_items").fetchall():
                con.execute("UPDATE memory_items SET text_hash=? WHERE id=?",
                            (hashlib.sha1(text.strip().encode()).hexdigest(), rid))
    finally:
        con.close()


def rows(tier: str | None = None) -> list[dict]:
    con = _ro(_db())
    try:
        q = ("SELECT id, kind, text, ts, pinned, tier, first_seen, last_changed, confirmed_at "
             "FROM memory_items WHERE superseded_by IS NULL")
        args: tuple = ()
        if tier:
            q += " AND tier=?"
            args = (tier,)
        q += " ORDER BY pinned DESC, first_seen ASC"
        return [dict(zip(("id", "kind", "text", "ts", "pinned", "tier", "first_seen",
                          "last_changed", "confirmed_at"), r)) for r in con.execute(q, args)]
    finally:
        con.close()


def _fit(items: list[str], cap: int) -> tuple[list[str], int]:
    out, used = [], 0
    for it in items:
        cost = len(it) + len(SEP)
        if used + cost > cap:
            break
        out.append(it)
        used += cost
    return out, used


def render_long() -> tuple[str, int]:
    facts = [r["text"].strip() for r in rows("long") if r["kind"] == "memory"]
    kept, used = _fit(facts, BUDGET["long"])
    return SEP.join(kept), used


def _day_lines(window_from: float, window_to: float) -> list[str]:
    """One line per day in the window: date, volume, and two representative asks.

    Days are the right unit here, not sessions: a single chat session can run for weeks, so
    its last message is always today and it would never appear in a middle window.
    """
    con = _ro(_state())
    buckets: dict[str, list[str]] = {}
    try:
        for content, ts in con.execute(
                "SELECT content, timestamp FROM messages WHERE role='user' AND timestamp >= ? "
                "AND timestamp <= ? AND trim(coalesce(content,''))<>'' ORDER BY id",
                (window_from, window_to)):
            text = " ".join((content or "").split())
            low = text.lower()
            if len(text) < 20:
                continue
            if text.startswith("[") or "out-of-band" in low or low.startswith("gateway message origin"):
                continue          # attachment banners and injected scaffolding, not asks
            buckets.setdefault(time.strftime("%m-%d", time.localtime(ts)), []).append(text)
    finally:
        con.close()

    lines = []
    for day in sorted(buckets, reverse=True):
        msgs = buckets[day]
        def clip(t: str, n: int = 44) -> str:
            t = t.strip()
            if len(t) <= n:
                return t
            cut = t[:n].rsplit(" ", 1)[0]
            return (cut or t[:n]).rstrip(",;:") + "..."

        first = clip(msgs[0])
        longest = clip(max(msgs, key=len))
        parts = [first] if first == longest else [first, longest]
        lines.append(f"{day}  {len(msgs):>2} msgs  " + " | ".join(parts))
    return lines


def render_middle(current_session: str | None = None) -> tuple[str, int]:
    now = time.time()
    lines = _day_lines(now - MIDDLE_WINDOW_DAYS * 86400, now - SHORT_HOURS * 3600)
    # facts still maturing ride in the middle tier so a fresh fact is never invisible
    pending = [f"pending: {r['text'].strip()}" for r in rows("candidate") if r["kind"] == "memory"]
    kept, used = _fit(pending + lines, BUDGET["middle"])
    return "\n".join(kept), used


def render_short(current_session: str | None = None) -> tuple[str, int]:
    since = time.time() - SHORT_HOURS * 3600
    con = _ro(_state())
    try:
        cur = con.execute("SELECT session_id, role, content, timestamp FROM messages "
                          "WHERE role IN ('user','assistant') AND timestamp >= ? "
                          "ORDER BY id DESC LIMIT 400", (since,))
        lines = []
        for sid, role, content, ts in cur:
            text = " ".join((content or "").split())
            if not text:
                continue
            stamp = time.strftime("%H:%M", time.localtime(ts))
            tag = "me" if role == "assistant" else "you"
            lines.append(f"{stamp} {tag}: {text[:88]}")
    finally:
        con.close()
    kept, used = _fit(lines, BUDGET["short"])
    return "\n".join(kept), used


def compose(current_session: str | None = None) -> tuple[str, dict]:
    """Full MEMORY.md text plus the per-section sizes actually used."""
    ensure_schema()
    long_txt, long_used = render_long()
    mid_txt, mid_used = render_middle(current_session)
    short_txt, short_used = render_short(current_session)
    used = {"long": long_used, "middle": mid_used, "short": short_used}
    blocks = []
    for key, body in (("long", long_txt), ("middle", mid_txt), ("short", short_txt)):
        head = HDR[key].format(used=used[key], cap=BUDGET[key])
        blocks.append(f"## {head}\n{body}".rstrip())
    return "\n\n".join(blocks), used


def refresh(current_session: str | None = None) -> dict:
    """Write MEMORY.long.md (store entries), MEMORY.md (composite) and the mirror."""
    ensure_schema()
    _mem_dir().mkdir(parents=True, exist_ok=True)
    facts = [r["text"].strip() for r in rows("long") if r["kind"] == "memory"]
    kept, _ = _fit(facts, BUDGET["long"])
    _long_file().write_text(SEP.join(kept), encoding="utf-8")
    text, used = compose(current_session)
    _composite().write_text(text, encoding="utf-8")
    _mirror().write_text(json.dumps({"at": time.time(), "used": used, "budget": BUDGET,
                                  "total": sum(used.values())}, indent=1), encoding="utf-8")
    return used


def promote(min_days: int = MATURE_DAYS) -> list[tuple[int, str]]:
    """Move facts from candidate to long once unchanged min_days and seen again afterwards."""
    ensure_schema()
    now = time.time()
    con = _ro(_state())
    try:
        moved = []
        for r in rows("candidate"):
            first = r["first_seen"] or r["ts"]
            changed = r["last_changed"] or r["ts"]
            age_days = (now - first) / 86400
            stable_days = (now - changed) / 86400
            if age_days < min_days or stable_days < min_days:
                continue
            terms = {w.lower().strip(".,:;()[]'\"") for w in r["text"].split() if len(w) > 5}
            hit = con.execute("SELECT count(*) FROM messages WHERE timestamp > ? AND ("
                              + " OR ".join(["lower(content) LIKE ?"] * min(len(terms), 4)) + ")",
                              (changed, *[f"%{t}%" for t in list(terms)[:4]])).fetchone()[0]
            if not hit:
                continue
            c2 = sqlite3.connect(_db())
            try:
                with c2:
                    c2.execute("UPDATE memory_items SET tier='long', confirmed_at=? WHERE id=?",
                               (now, r["id"]))
            finally:
                c2.close()
            moved.append((r["id"], r["text"][:60]))
        return moved
    finally:
        con.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--promote", action="store_true")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    if a.render:
        text, used = compose()
        print(text)
        print(f"\n[total {sum(used.values())}/{sum(BUDGET.values())}]", file=os.sys.stderr)
        return 0
    if a.promote:
        moved = promote()
        for rid, snippet in moved:
            print(f"promoted #{rid}: {snippet}")
        print(f"{len(moved)} promoted")
        return 0
    if a.status:
        ensure_schema()
        for t in ("long", "candidate"):
            print(f"{t}: {len([r for r in rows(t)])}")
        print(json.loads(_mirror().read_text())["used"] if _mirror().exists() else "not rendered yet")
        return 0
    used = refresh()
    print(json.dumps(used))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
