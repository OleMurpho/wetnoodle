#!/usr/bin/env python3.11
"""render.py — the one place that decides how much retrieved text is shown.

A hit is cheap to find and expensive to paste: every character of transcript that reaches a model costs
tokens on every turn it stays in context. So there are two caps with two different jobs:

  RAG_MAX_HIT_CHARS    (default 800)   per hit
  RAG_MAX_TOTAL_CHARS  (default 6000)  across the whole result set

The second is the one that matters. k=25 at 800 chars is 20,000 characters of somebody else's Tuesday if
nothing bounds it. Harvested from hermes-memory-rag's bounded recall (2 wiki + 3 code + 1 memory, 600
chars each): same principle, expressed as an env-tunable budget instead of a fixed shape.
"""
import os


def _int(name: str, default: int) -> int:
    try:
        return max(0, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


HIT_CHARS = _int("RAG_MAX_HIT_CHARS", 800)
TOTAL_CHARS = _int("RAG_MAX_TOTAL_CHARS", 6000)


def bounded(hits, *, header, hit_chars: int = 0, total_chars: int = 0):
    """Render hits under both caps.

    header(i, hit, body) -> str  formats one hit; body is already truncated.

    Returns (blocks, shown, available, truncated). The first hit is always included, even when it alone
    exceeds the total budget: returning nothing for a query that did match would be worse than being long.
    """
    hc = hit_chars or HIT_CHARS
    tc = total_chars or TOTAL_CHARS
    blocks, used = [], 0
    for i, h in enumerate(hits, 1):
        body = h.get("content") or ""
        if hc and len(body) > hc:
            body = body[:hc] + "…"
        block = header(i, h, body)
        if tc and blocks and used + len(block) > tc:
            break
        blocks.append(block)
        used += len(block)
    return blocks, len(blocks), len(hits), len(blocks) < len(hits)


def note(shown: int, available: int) -> str:
    """The line that tells a reader (or an agent) that the list was cut, and why."""
    if shown >= available:
        return ""
    return (f"\n\n[{shown} of {available} hits shown — bounded by RAG_MAX_TOTAL_CHARS="
            f"{TOTAL_CHARS}; raise it or narrow the query with `corpus`]")
