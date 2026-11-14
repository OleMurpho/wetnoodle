#!/usr/bin/env python3
"""wetnoodle.retire — take a chunk out of circulation.

A chunk is retired when a later one says the same thing better, or when it should not be
answered from at all. Retired chunks stay in the archive: search drops them, and with
successor forwarding enabled the replacement inherits the standing of the claim it replaced.

    python -m wetnoodle.retire 1234 --by 1250 --reason "the corrected figure"
    python -m wetnoodle.retire 1234 --reason "withdrawn, nothing replaces it"

This is a shell command on purpose. Supersession rewrites what the archive will answer from,
which is not a decision to hand an agent as a tool call; the MCP server does not expose it.
"""
from __future__ import annotations

import argparse
import re
import sys

from . import rag_core as core


def _ref(text: str) -> int:
    m = re.fullmatch(r"c?(\d+)", (text or "").strip(), re.I)
    if not m:
        raise SystemExit(f"not a chunk ref: {text!r} (expected c<rowid>, as rag_search prints)")
    return int(m.group(1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m wetnoodle.retire",
        description="Retire a chunk, optionally naming the chunk that replaces it.",
    )
    ap.add_argument("old", help="the ref to retire, e.g. c1234")
    ap.add_argument("--by", default=None, help="the ref that replaces it, e.g. c1250")
    ap.add_argument("--reason", default=None, help="why, recorded with the retirement")
    args = ap.parse_args(argv)

    old = _ref(args.old)
    new = _ref(args.by) if args.by else None

    found, _ = core.get_chunks([old])
    if not found:
        raise SystemExit(f"chunk c{old} does not exist")
    if new is not None:
        if new == old:
            raise SystemExit("a chunk cannot replace itself")
        # Refusing here matters: supersede() will happily point a chunk at an id that does not
        # exist, and the successor boost then silently never fires for it.
        found_new, _ = core.get_chunks([new])
        if not found_new:
            raise SystemExit(f"successor c{new} does not exist")

    if not core.supersede(old, new, args.reason):
        raise SystemExit(f"chunk c{old} was already retired, or the store is read-only")

    if new is None:
        print(f"retired c{old} with no successor: search will not return it again")
    else:
        print(f"retired c{old}, replaced by c{new}")
    if args.reason:
        print(f"reason recorded: {args.reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
