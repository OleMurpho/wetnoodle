"""Optional listwise reranking: one model call reorders the retriever's candidate pool.

Why this exists
---------------
Two-stage retrieval is standard: the retriever's job is recall (get the answer into a candidate list), the
reranker's job is precision (order that list). A bi-encoder cannot compare two documents, and a cross-encoder
compares only one (query, document) pair at a time. A listwise judge reads the query and every candidate in a
single prompt, so it can rank them against each other — and it costs one call per query instead of N scorings.

Measured on this repo's 46-question set (same 2,000-chunk corpus, held-out intact):

    dense only                    35% hit@1
    + cross-encoder (top-20)      50%
    + listwise judge (top-20)     74%      dev 70%, held-out 79%, strict stratum 7/8
    oracle ceiling (answer in pool) 89%

Which judge
-----------
Any OpenAI-compatible chat endpoint works, local or hosted. Reasoning models need output budget beyond their
thinking: with a 700-token cap two of them returned empty content and silently fell back to the retriever.

    RAG_JUDGE_MODEL=deepseek-v4.1-flash:cloud     # unset (default) = judge off, plain retrieval
    RAG_JUDGE_BASE_URL=https://ollama.com/v1      # default: the local Ollama's /v1
    RAG_JUDGE_API_KEY=...                          # omit for a local daemon
    RAG_JUDGE_CANDIDATES=20                        # pool size reordered in one call
    RAG_JUDGE_SNIPPET=1200                         # chars per candidate shown to the judge
    RAG_JUDGE_MAX_TOKENS=2500                      # must exceed the model's reasoning budget

Failure policy
--------------
Every error path returns the retriever's order unchanged. A judge outage degrades recall to the dense
baseline; it never breaks search and never invents results.
"""
import json
import os
import re
import sys
import urllib.request
from urllib.parse import urlparse


def _is_local(base):
    """Is this endpoint on this machine? Anything else spends the owner's money when a stranger queries."""
    host = (urlparse(base).hostname or "").lower()
    return host in ("localhost", "::1", "") or host.startswith("127.") or host.endswith(".local") \
        or host.endswith(".internal")


_WARNED = set()


def _warn_once(msg):
    if msg not in _WARNED:
        _WARNED.add(msg)
        sys.stderr.write("[judge] %s\n" % msg)


def config():
    model = (os.environ.get("RAG_JUDGE_MODEL") or "").strip()
    if not model:
        return None
    default_base = (os.environ.get("RAG_OLLAMA") or "http://127.0.0.1:11434").rstrip("/") + "/v1"
    base = (os.environ.get("RAG_JUDGE_BASE_URL") or default_base).rstrip("/")
    # A hosted endpoint bills per call, and this function is reached by any search — including searches
    # triggered by people who merely message an agent. So a non-local endpoint is refused unless the owner
    # has said, explicitly and in writing, that they want that. Default-deny, not default-allow.
    if not _is_local(base) and os.environ.get("RAG_JUDGE_ALLOW_REMOTE") != "1":
        _warn_once("RAG_JUDGE_BASE_URL=%s is not local; refusing to call it. Set RAG_JUDGE_ALLOW_REMOTE=1 "
                   "to accept that every search will be a billable request on that account." % base)
        return None
    return {
        "model": model,
        "base": base,
        "key": (os.environ.get("RAG_JUDGE_API_KEY") or "").strip(),
        "candidates": int(os.environ.get("RAG_JUDGE_CANDIDATES") or 20),
        "snippet": int(os.environ.get("RAG_JUDGE_SNIPPET") or 1200),
        "max_tokens": int(os.environ.get("RAG_JUDGE_MAX_TOKENS") or 2500),
        "timeout": float(os.environ.get("RAG_JUDGE_TIMEOUT") or 60),
    }


def enabled():
    return config() is not None


SYSTEM = (
    "You are a retrieval judge. You receive a query and numbered candidate passages. Decide which passages "
    "actually answer the query. A passage that merely mentions the same words but does not answer the "
    "question must rank below one that does. Output JSON only."
)


def parse_order(text, n):
    """Extract a ranking without being fooled by list markers.

    A model that answers "1) 19  2) 3" ranks 19 above 3; reading every integer would yield [1, 19, 2, 3] and
    score the ranking as 1, 2, 3 — the exact opposite of what it said. So prefer an explicit ANSWER: [...],
    then any bracketed array, and accept bare digits only when the reply is nothing but digits and separators.
    Returns [] when nothing trustworthy is found, which the caller treats as "no judgement".
    """
    if not text:
        return []
    for pattern in (r"ANSWER\s*:\s*\[([^\]]*)\]", r"\[([0-9,\s]+)\]"):
        m = re.search(pattern, text, re.I | re.S)
        if m:
            return _dedupe([int(x) for x in re.findall(r"\d+", m.group(1))], n)
    stripped = text.strip()
    if re.fullmatch(r"[0-9,\s\[\].]+", stripped):
        return _dedupe([int(x) for x in re.findall(r"\d+", stripped)], n)
    return []


def _dedupe(nums, n):
    seen, order = set(), []
    for v in nums:
        if 1 <= v <= n and v not in seen:
            seen.add(v)
            order.append(v)
    return order


def _ask(cfg, query, passages):
    listing = "\n".join(
        "%d. %s" % (i + 1, " ".join((p or "").split())[: cfg["snippet"]]) for i, p in enumerate(passages)
    )
    prompt = (
        "Query: %s\n\nCandidates:\n%s\n\n"
        "Return ONLY a JSON array of the candidate numbers that answer the query, best first, at most 3.\n"
        "Example: [3, 1]\n"
        "Your final line must be exactly: ANSWER: [3, 1] — numbers only, no prose after it." % (query, listing)
    )
    body = json.dumps(
        {
            "model": cfg["model"],
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            "max_tokens": cfg["max_tokens"],
            "temperature": 0,
        }
    ).encode()
    headers = {"Content-Type": "application/json"}
    if cfg["key"]:
        headers["Authorization"] = "Bearer " + cfg["key"]
    req = urllib.request.Request(cfg["base"] + "/chat/completions", data=body, headers=headers)
    payload = json.loads(urllib.request.urlopen(req, timeout=cfg["timeout"]).read())
    return payload["choices"][0]["message"].get("content") or ""


def rerank(query, hits, k=None):
    """Return `hits` reordered by the judge, truncated to k. Any failure returns the input unchanged."""
    cfg = config()
    if not cfg or len(hits) < 2:
        return hits[:k] if k else hits
    pool = hits[: cfg["candidates"]]
    try:
        order = parse_order(_ask(cfg, query, [h.get("content", "") for h in pool]), len(pool))
    except Exception:
        return hits[:k] if k else hits
    if not order:
        return hits[:k] if k else hits
    picked = [pool[i - 1] for i in order]
    chosen = {id(h) for h in picked}
    picked.extend(h for h in hits if id(h) not in chosen)   # anything the judge dropped keeps its order
    return picked[:k] if k else picked
