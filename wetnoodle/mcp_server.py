#!/usr/bin/env python3.11
"""
mcp_server.py — MCP stdio server exposing the RAG memory archive to Hermes.

THE MODEL THAT RETRIEVES IS THE MODEL THAT JUDGES.

There is no reranker, no judge endpoint and no second model anywhere in this design. Retrieval does what
vectors are good at: propose a pool of plausible passages. Ranking what matters is done by the model that
asked the question — the one already loaded, already holding the conversation, already paid for. It reads
the candidates and calls `rag_open` on the ones that actually answer.

Why this is the right split rather than a compromise: a re-reader has to compare the question against twenty
passages. The calling model can do that, has context no external judge has, and costs nothing extra because
it is already in the turn. A separate judge model would add an endpoint, a bill, and a copy of your
conversation leaving the machine.

Tools: rag_search(query, pool, corpus) · rag_open(refs) · rag_audit()
Requires numpy + mcp in the same interpreter the MCP server runs on.
"""
import json
import os
import sqlite3
import sys


from . import rag_core as core
from . import render
try:  # mcp 1.x exposes FastMCP
    from mcp.server.fastmcp import FastMCP
except ImportError:  # mcp 2.x renamed it to MCPServer and kept the same tool/run API
    from mcp.server.mcpserver import MCPServer as FastMCP

mcp = FastMCP("wetnoodle")

# A snippet must be long enough to judge relevance from and short enough that twelve of them fit in a turn.
SNIPPET = max(120, int(os.environ.get("RAG_SNIPPET_CHARS", "320")))


def _scope():
    """Corpora this server may read, from the environment.

    The store is shared by every profile on the machine, so the MCP boundary is where "this
    agent may not read another profile's transcripts" has to be enforced. Empty means
    unrestricted, which is what a single-profile install wants.
    """
    split = lambda name: [c.strip() for c in os.environ.get(name, "").split(",") if c.strip()]
    return split("RAG_ALLOWED_CORPORA"), split("RAG_EXCLUDED_CORPORA")


@mcp.tool()
def rag_search(query: str, pool: int = 12, corpus: str = "") -> str:
    """Search the memory archive and get a POOL OF CANDIDATES — not an answer, and not a ranking to trust.

    You are the judge. Read the candidates, then call `rag_open` with the refs that actually answer the
    question. Do not answer from the first hit because it came first: measured on this archive, the right
    passage is somewhere in the top twenty 84% of the time but first only about a third of the time.

    Returns up to `pool` candidates (default 12, max 25), each with a ref like `c1234`, a date, a corpus, a
    similarity score and a ~320-character snippet. Scores are cosine similarity and are a coarse guide only.

    The archive spans MULTIPLE corpora: every Hermes profile's session database plus every indexed
    SKILL.md (corpus 'skill'). Leave `corpus` empty to search everything; name it when you know where the
    answer lives, because a generic query lets the biggest corpora win.
    """
    try:
        size = 12 if pool is None else int(pool)
        allow, deny = _scope()
        hits = core.search(
            query, k=max(1, min(size, 25)), corpus=(corpus or "").strip() or None,
            corpora=allow, exclude_corpora=deny,
        )
    except (ValueError, TypeError) as e:
        # A bad query and a bad pool both raise ValueError; do not blame the pool for both.
        return f"rag_search error: {e}"
    except sqlite3.Error as e:
        return (f"rag_search unavailable: no index at {core.RAG_DB} ({e}). "
                f"Run ingest.py first, or point RAG_DIR at a store that exists.")
    except RuntimeError as e:
        return f"rag_search unavailable: {e}"
    if not hits:
        scope = f" (corpus={corpus!r})" if corpus else ""
        return f"no hits{scope} — index empty for that scope? see rag_audit"

    lines = [f"{len(hits)} candidate(s) for: {query!r}",
             "You are the judge: pick the ones that answer it, then call rag_open(refs).", ""]
    for h in hits:
        body = render.snippet((h.get("content") or "").replace("\n", " ⏎ "), query, SNIPPET)
        # When the keyword blend reorders, the cosine no longer matches the order. Show the value that
        # produced the order, and the cosine too so a judge knows why a row sits where it does.
        shown = h.get("fused")
        if shown is None:
            grade = f"score {h['score']:.3f}"
        else:
            grade = f"score {shown:.3f} (cosine {h['score']:.3f})"
        lines.append(
            f"[c{h['id']}] {core.fmt_date(h['ts'])} | {h.get('corpus', '?')} | {h['role']} | "
            f"{grade}\n    {body}"
        )
    return "\n".join(lines)


@mcp.tool()
def rag_open(refs: list[str], max_chars: int = 0) -> str:
    """Open the candidates you selected, in full. Pass refs from rag_search, e.g. ["c1234","c1250"].

    This is the second half of judging: search proposes, you choose, this returns the whole passage so the
    answer can be quoted from the source instead of reconstructed from a snippet. Output stays bounded by
    RAG_MAX_HIT_CHARS per passage and RAG_MAX_TOTAL_CHARS overall; `max_chars` raises the total for a
    deliberate deep read.
    """
    try:
        budget = int(max_chars or 0)
    except (ValueError, TypeError):
        return f"rag_open error: max_chars must be a whole number; got {max_chars!r}"
    allow, deny = _scope()
    try:
        hits, missing = core.get_chunks(refs, corpora=allow, exclude_corpora=deny)
    except sqlite3.Error as e:
        return (f"rag_open unavailable: no index at {core.RAG_DB} ({e}). Run ingest.py first, "
                f"or point RAG_DIR at a store that exists.")
    except Exception as e:                      # noqa: BLE001 — a tool must not crash the server
        return f"rag_open error: {type(e).__name__}: {e}"
    if not hits:
        return (f"rag_open: no such chunk(s) {list(refs)} — refs are c<rowid>; a re-ingest can retire an id, "
                f"so run rag_search again if this was from an earlier session")

    def header(i, h, body):
        return (
            f"[c{h['id']}] {core.fmt_date(h['ts'])} | {h.get('corpus', '?')} | {h['session_id']} | "
            f"{h['role']} | msgs {h['msg_start']}-{h['msg_end']}\n{body}"
        )

    blocks, shown, available, _ = render.bounded(
        hits, header=header, hit_chars=render.OPEN_HIT_CHARS, total_chars=budget
    )
    out = "\n\n".join(blocks)
    if missing:
        out += f"\n\n[not found: {', '.join('c%d' % m for m in missing)}]"
    return out + render.note(shown, available)


@mcp.tool()
def rag_audit() -> str:
    """Index health: total and per-corpus chunk counts, per-corpus coverage and freshness against each
    source database, sizes, schema version, and Ollama reachability. Corpora whose source db was deleted
    show as frozen."""
    try:
        report = core.audit()
        allow, deny = _scope()
        if allow or deny:
            # Naming a corpus this caller may not read is a disclosure even without its contents.
            keep = lambda name: (not allow or name in allow) and name not in deny
            counts = report.get("corpora")
            if isinstance(counts, dict):
                permitted = {k: v for k, v in counts.items() if keep(k)}
                report["corpora"] = permitted
            elif isinstance(counts, list):
                permitted = [c for c in counts
                             if keep(c.get("corpus") if isinstance(c, dict) else c)]
                report["corpora"] = permitted
            else:
                permitted = {}
            # The aggregates leak what the per-corpus filter just hid: a chunk total larger
            # than the visible corpora is itself a disclosure. Recompute them from what the
            # caller may see, or drop the ones that cannot be recomputed.
            if isinstance(permitted, dict):
                total = sum(int(v.get("chunks", 0)) for v in permitted.values()
                            if isinstance(v, dict))
                report["chunks"] = total
                report["corpora_count"] = len(permitted)
                report["session_chunks"] = sum(int(v.get("chunks", 0)) for k, v in permitted.items()
                                               if k != "skill" and isinstance(v, dict))
                report["skill_chunks"] = int((permitted.get("skill") or {}).get("chunks", 0)) \
                    if isinstance(permitted.get("skill"), dict) else 0
            for hidden in ("sessions_covered", "rag_db_MB"):
                report.pop(hidden, None)
            # Never echo the deny list: this code's own reason for filtering the corpus map is
            # that naming a corpus discloses it. Report the shape of the scope, not its contents.
            if isinstance(permitted, dict):
                newest = max((v.get("latest_chunk_ts") or "" for v in permitted.values()
                              if isinstance(v, dict)), default="")
                if newest:
                    report["latest_chunk_ts"] = newest
            report["scope"] = {"scoped": True,
                               "allowed_corpora": len(allow) if allow else None,
                               "excluded_corpora": len(deny) if deny else 0}
        return json.dumps(report, indent=1)
    except sqlite3.Error as e:
        return (f'{{"error": "no index at {core.RAG_DB}: {e}", "hint": "run ingest.py, or point RAG_DIR '
                f'at a store that exists"}}')


if __name__ == "__main__":
    mcp.run(transport="stdio")
