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
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rag_core as core
import render
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("wetnoodle")

# A snippet must be long enough to judge relevance from and short enough that twelve of them fit in a turn.
SNIPPET = max(120, int(os.environ.get("RAG_SNIPPET_CHARS", "320")))


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
        hits = core.search(
            query, k=max(1, min(int(pool), 25)), corpus=(corpus or "").strip() or None
        )
    except ValueError as e:
        return f"rag_search error: {e}"
    except RuntimeError as e:
        return f"rag_search unavailable: {e}"
    if not hits:
        scope = f" (corpus={corpus!r})" if corpus else ""
        return f"no hits{scope} — index empty for that scope? see rag_audit"

    lines = [f"{len(hits)} candidate(s) for: {query!r}",
             "You are the judge: pick the ones that answer it, then call rag_open(refs).", ""]
    for h in hits:
        body = render.snippet((h.get("content") or "").replace("\n", " ⏎ "), query, SNIPPET)
        lines.append(
            f"[c{h['id']}] {core.fmt_date(h['ts'])} | {h.get('corpus', '?')} | {h['role']} | "
            f"score {h['score']:.3f}\n    {body}"
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
        hits, missing = core.get_chunks(refs)
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
        hits, header=header, total_chars=int(max_chars or 0)
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
    return json.dumps(core.audit(), indent=1)


if __name__ == "__main__":
    mcp.run(transport="stdio")
