#!/usr/bin/env python3
"""wetnoodle.curate — lift durable project facts out of the transcript into the index.

The archive ranks transcript chunks. That is the right store for "what did we say", and
the wrong one for "where does this project live": a fact stated once, months ago, sits at
a 0.59 cosine match behind a thousand later mentions of the same words, and a question
that expects an exact answer gets a pool of near-misses.

This module reads the archive and writes one chunk per project — path, repository, what it
is, when it was last discussed — into the same index under `corpus='curated'`. Nothing
downstream changes: `rag_search` ranks it with everything else, `rag_open` returns it, and
a curated row is written at importance 1.0 so it wins ties against chatter.

It is deliberately deterministic. No model decides what counts as a fact. A project is
recognised by structure — a path under an XAMPP/Apache root, a home-directory project, a
GitHub repository, or the phrase "project called X" — never by a name appearing in prose.
That rule is what stops the curator filling the index with `node_modules`, `bashrc` and
every other word that happens to follow a slash.

    python -m wetnoodle.curate --dry-run      # print what it would write
    python -m wetnoodle.curate                # write it
    python -m wetnoodle.curate --min-mentions 2

Re-running updates a project's row in place. A name that stops being mentioned keeps its
row: absence of new mention is not evidence it stopped existing.
"""
from __future__ import annotations

import argparse
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone

OWNER = "olemurpho"      # only his own repositories count as projects
CORPUS = "curated"
SOURCE = "curated"
IMPORTANCE = 1.0

# Directories and words that are never a project, however often they appear.
STOP = {
    "appdata", "windows", "program files", "programdata", "users", "temp", "tmp",
    "downloads", "desktop", "documents", "pictures", "music", "videos", "roaming", "local",
    "localappdata", "public", "node_modules", "dist", "build", "target", "bin", "lib",
    "src", "assets", "static", "vendor", "cache", "config", "data", "docs", "tests",
    "scripts", "skills", "plugins", "hermes", "wsl", "mnt", "home", "root", "opt", "usr",
    "var", "etc", "system32", "drivers", "packages", "flutter", "pocketbase", "codex",
    "python", "node", "npm", "cargo", "git", "ssh", "bashrc", "profile", "env",
    "project", "projects", "repo", "repos", "folder", "file", "files", "app", "apps",
    "thing", "stuff", "example", "test", "demo", "mlope", "olemurpho", "onedrive",
    "program", "oracle", "virtualbox", "vboxmanage", "ephemeral", "plugin_symlinks",
    "linux", "windows", "x64", "amd64", "binaries", "venv", "site-packages",
}
# Stack words used to enrich a description, only once the project is already recognised.
STACK = ("xampp", "php", "python", "flask", "django", "tauri", "rust", "flutter", "dart",
         "node", "react", "vue", "svelte", "docker", "arduino", "esp32", "php", "mysql",
         "sqlite", "bootstrap")
DESC_HINTS = (" is a ", " is an ", "project", " app", " tool", " manager", " service",
              " site", " library", " platform", " framework", " engine", " system",
              " dashboard", " library", "local", "personal")

RE_WIN = re.compile(r"[A-Za-z]:\\(?:[^\s\"'`<>|()\[\],;]+)")
RE_WSL = re.compile(r"/mnt/[a-z]/(?:[^\s\"'`<>|()\[\],;]+)")
RE_HOME = re.compile(r"~/([A-Za-z0-9_\-]{2,40})")
RE_GH = re.compile(r"github\.com/([A-Za-z0-9_.\-]+)/([A-Za-z0-9_.\-]+)")
RE_CALLED = re.compile(
    r"\bproject\s+(?:called|named)\s+[\"'`]?([A-Za-z][A-Za-z0-9_\-]{2,24})"
    r"|\bcalled\s+[\"'`]([A-Z][A-Za-z0-9_\-]{2,24})[\"'`]"
)
RE_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _clean(name: str | None):
    name = (name or "").strip().strip(".,;:'\"`()[]")
    if name.lower().endswith(".git"):
        name = name[:-4]
    low = name.lower()
    if not name or low in STOP:
        return None
    if "." in name:            # dotfiles, binaries, domains: not project names
        return None
    if not re.match(r"^[A-Za-z][A-Za-z0-9_\-]{2,29}$", name):
        return None
    return name


def _leaf(path: str):
    """The project-looking segment of a path, or None when there is none."""
    parts = [p for p in re.split(r"[\\/]+", path) if p]
    if not parts:
        return None
    for i, p in enumerate(parts):
        if p.lower() in ("htdocs", "www", "public_html", "sites") and i + 1 < len(parts):
            return parts[i + 1]
    return parts[-1]


def collect(con: sqlite3.Connection, min_mentions: int = 2):
    """Scan the archive. Only structurally-evidenced names become candidates."""
    rows = con.execute(
        "SELECT content, ts, session_id, role FROM chunks "
        "WHERE corpus != ? AND superseded_by IS NULL", (CORPUS,)
    ).fetchall()

    evidence = defaultdict(lambda: {"paths": Counter(), "repos": Counter(),
                                    "sessions": set(), "ts": 0, "casing": Counter(),
                                    "chunks": []})

    for content, ts, session, role in rows:
        text = content or ""
        hits = {}   # low -> display name, from structural evidence only

        for m in RE_WIN.findall(text) + RE_WSL.findall(text):
            depth = len([x for x in re.split(r"[\\/]+", m) if x])
            webroot = any(w in m.lower() for w in ("htdocs", "\\www\\", "public_html"))
            if depth > 6 and not webroot:
                continue
            leaf = _clean(_leaf(m))
            if leaf:
                hits.setdefault(leaf.lower(), leaf)
                evidence[leaf.lower()]["paths"][m.rstrip(".,;:")] += 1
        for m in RE_HOME.findall(text):
            leaf = _clean(m)
            if leaf:
                hits.setdefault(leaf.lower(), leaf)
                evidence[leaf.lower()]["paths"]["~/" + leaf] += 1
        for owner, repo in RE_GH.findall(text):
            if owner.lower() != OWNER:
                continue          # a third-party library, not one of his projects
            leaf = _clean(repo)
            if leaf:
                hits.setdefault(leaf.lower(), leaf)
                evidence[leaf.lower()]["repos"][f"github.com/{owner}/{leaf}"] += 1
        for m in RE_CALLED.finditer(text):
            leaf = _clean(m.group(1) or m.group(2) or "")
            if leaf:
                hits.setdefault(leaf.lower(), leaf)

        for low, display in hits.items():
            e = evidence[low]
            e["sessions"].add(session)
            e["ts"] = max(e["ts"], ts or 0)
            for m in re.finditer(rf"\b{re.escape(low)}\b", text, re.I):
                e["casing"][m.group(0)] += 1
            if len(e["chunks"]) < 12:
                e["chunks"].append((text, role, session))

    out = {}
    for low, e in evidence.items():
        if len(e["sessions"]) < min_mentions:
            continue
        if not e["paths"] and not e["repos"]:
            continue
        name = max(e["casing"].items(),
                   key=lambda kv: (sum(1 for ch in kv[0] if ch.isupper()), kv[1]))[0]
        out[low] = {
            "name": name,
            "paths": [p for p, _ in e["paths"].most_common(2)],
            "repo": e["repos"].most_common(1)[0][0] if e["repos"] else "",
            "desc": _describe(name, e["chunks"]),
            "stack": _stack(e["chunks"]),
            "ts": e["ts"],
            "sessions": len(e["sessions"]),
        }
    return out


def _describe(name: str, chunks) -> str:
    """The sentence that best says what this thing is, preferring the user's own words."""
    best, best_score = "", 0.0
    rx = re.compile(rf"\b{re.escape(name)}\b", re.I)
    for text, role, _ in chunks:
        for sent in RE_SENT_SPLIT.split(text):
            sent = sent.strip()
            if len(sent) < 12 or len(sent) > 240:
                continue
            if not re.match(r'^[A-Z0-9`"*\-(\[]', sent):
                continue          # a fragment cut out of a path or a word
            if "\n" in sent or "```" in sent or sent.count("`") > 2:
                continue          # shell output and fenced code are not descriptions
            if not rx.search(sent):
                continue
            low = sent.lower()
            score = 0.0
            if any(h in low for h in DESC_HINTS):
                score += 2
            if any(s in low for s in STACK):
                score += 2
            if "project" in low:
                score += 1
            if role.startswith("user") or role == "assistant+user":
                score += 1
            score -= len(sent) / 400.0      # shorter sentences read better as a summary
            if score > best_score:
                best, best_score = sent.strip(), score
    best = best.rstrip(".")
    if len(best) > 170:
        best = best[:167].rsplit(" ", 1)[0] + "…"
    return best


def _stack(chunks) -> list:
    seen = Counter()
    for text, _, _ in chunks:
        low = text.lower()
        for s in set(STACK):
            if s in low:
                seen[s] += 1
    return [s for s, _ in seen.most_common(4)]


def render(fact) -> str:
    """One line, dense enough to answer 'where is X' on its own."""
    name = fact["name"]
    head = f"{name} — {fact['desc']}" if fact["desc"] else name
    bits = [head]
    if fact["paths"]:
        bits.append("Path: " + " ; ".join(fact["paths"]))
    if fact["repo"]:
        bits.append("Repo: " + fact["repo"])
    if fact["stack"]:
        bits.append("Stack: " + ", ".join(fact["stack"][:3]))
    if fact["ts"]:
        when = datetime.fromtimestamp(fact["ts"], tz=timezone.utc).strftime("%Y-%m")
        bits.append(f"Last discussed {when} ({fact['sessions']} sessions)")
    text = ". ".join(bits).replace(" .", ".").replace("..", ".")
    return text + "." if not text.endswith(".") else text


def write(con: sqlite3.Connection, facts, dry_run=False) -> tuple[int, int, int]:
    """Insert new facts, update changed ones. Returns (inserted, updated, unchanged).

    A fact whose rendered text has not changed is not re-embedded: the curator runs on a
    timer, and an Ollama call per project per tick is a cost with no benefit.
    """
    from wetnoodle import rag_core as core

    # Identity lives in meta, not in the rendered text: a fact without a description renders
    # as "NAME. Path: …", and splitting that on an em dash invents a new name every run.
    known = {row[0][len("curated:"):]: int(row[1]) for row in con.execute(
        "SELECT key, value FROM meta WHERE key LIKE 'curated:%'")}
    rendered = [(low, render(f)) for low, f in facts.items()]
    if not rendered:
        return (0, 0, 0)

    live = {row[0]: row[1] for row in con.execute(
        "SELECT id, content FROM chunks WHERE corpus = ?", (CORPUS,))}
    changed = [(low, text) for low, text in rendered
               if known.get(low) not in live or live.get(known.get(low)) != text]
    unchanged = len(rendered) - len(changed)
    if not changed:
        return (0, 0, unchanged)

    if dry_run:
        new = upd = 0
        for low, text in changed:
            if known.get(low) in live:
                print(f"  ~ update {text}")
                upd += 1
            else:
                print(f"  + insert {text}")
                new += 1
        return (new, upd, unchanged)

    mats = core.embed_documents([t for _, t in changed])
    now = time.time()
    inserted = updated = 0
    for (low, text), mat in zip(changed, mats):
        blob = mat.tobytes()
        row_id: int | None = known.get(low)
        with con:
            if row_id in live:
                con.execute(
                    "UPDATE chunks SET content=?, embedding=?, ts=?, importance=? WHERE id=?",
                    (text, blob, now, IMPORTANCE, row_id),
                )
                updated += 1
            else:
                cur = con.execute(
                    "INSERT INTO chunks(msg_start, msg_end, session_id, role, ts, source, "
                    "content, embedding, corpus, importance) VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (0, 0, "curator", "memory", now, SOURCE, text, blob, CORPUS, IMPORTANCE),
                )
                row_id = int(cur.lastrowid or 0)
                inserted += 1
            con.execute(
                "INSERT INTO meta(key, value) VALUES(?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (f"curated:{low}", str(row_id)),
            )
    return (inserted, updated, unchanged)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Lift project facts out of the archive.")
    ap.add_argument("--dry-run", action="store_true", help="print what would be written")
    ap.add_argument("--min-mentions", type=int, default=2,
                    help="distinct sessions a name must appear in (default 2)")
    ap.add_argument("--limit", type=int, default=0, help="cap the number of facts written")
    args = ap.parse_args(argv)

    from wetnoodle import rag_core as core

    con = sqlite3.connect(core.RAG_DB)
    try:
        facts = collect(con, args.min_mentions)
        ranked = sorted(facts.items(),
                        key=lambda kv: (0 if kv[1]["repo"] else 1, -kv[1]["sessions"], kv[0]))
        if args.limit:
            ranked = ranked[: args.limit]
        if not ranked:
            print("no project facts found")
            return 0
        ins, upd, same = write(con, dict(ranked), dry_run=args.dry_run)
    finally:
        con.close()

    verb = "would write" if args.dry_run else "wrote"
    print(f"{verb} {ins} new and {upd} updated fact(s); {same} unchanged "
          f"({len(facts)} candidates)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
