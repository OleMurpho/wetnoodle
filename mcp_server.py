#!/usr/bin/env python3.11
"""
mcp_server.py — MCP stdio server exposing the RAG memory archive to Hermes.
SHARED MULTI-SOURCE ARCHIVE: one vector store indexes every Hermes profile's
session database plus all SKILL.md documents. Each corpus keeps its own
watermark, so no profile's history is silently skipped.

Tools: rag_search(query, k, corpus) · rag_audit()
Register under an mcp_servers entry of your choosing; tools then appear to the agent prefixed
with that name. Requires numpy + mcp in the same interpreter the MCP server runs on.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rag_core as core
import render
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("wetnoodle")


@mcp.tool()
def rag_search(query: str, k: int = 8, corpus: str = "", max_chars: int = 0) -> str:
    """Semantic search over the indexed Hermes memory archive. The archive spans
    MULTIPLE corpora: every Hermes profile's session database plus every
    indexed SKILL.md document (corpus 'skill').
    Returns top-k history chunks with date, session id, role, corpus and source.

    Leave `corpus` empty to search everything. Pass e.g. 'skill' to search only
    procedures/skills, or a profile name to search only that profile's history.

    Corpus matters: a generic query lets the biggest corpora win, so when you
    know where the answer lives, name it.

    Output is BOUNDED: RAG_MAX_HIT_CHARS (800) per hit and RAG_MAX_TOTAL_CHARS
    (6000) across the set, so a wide query cannot paste a context window full of
    transcript. When the list is cut, the reply says so. `max_chars` overrides the
    per-call total budget for a deliberate deep read."""
    try:
        hits = core.search(
            query, k=max(1, min(int(k), 25)), corpus=(corpus or "").strip() or None
        )
    except ValueError as e:
        return f"rag_search error: {e}"
    except RuntimeError as e:
        return f"rag_search unavailable: {e}"
    if not hits:
        scope = f" (corpus={corpus!r})" if corpus else ""
        return f"no hits{scope} — index empty for that scope? see rag_audit"

    def header(i, h, body):
        return (
            f"{i:2d}. [{h['score']:.3f}] {core.fmt_date(h['ts'])} | "
            f"{h.get('corpus', '?')} | {h['session_id']} | {h['role']} | "
            f"msgs {h['msg_start']}-{h['msg_end']} | src={h['source']}\n"
            f"{body}"
        )

    blocks, shown, available, _ = render.bounded(hits, header=header,
                                                 total_chars=int(max_chars or 0))
    return "\n\n".join(blocks) + render.note(shown, available)


@mcp.tool()
def rag_audit() -> str:
    """Index health: total and per-corpus chunk counts, per-corpus coverage and
    freshness against each source database, sizes, schema version, and Ollama
    reachability. Corpora whose source db was deleted show as frozen."""
    return json.dumps(core.audit(), indent=1)


if __name__ == "__main__":
    mcp.run(transport="stdio")