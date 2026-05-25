"""
crag_engine.py
--------------
Corrective RAG engine implementing:
  1. Multi-query expansion
  2. Hierarchical BM25 indexing (level-0 → level-1 → level-2)
  3. Tree-of-Thought (ToT) relevance evaluation (3-path scoring)
  4. Tavily fallback when internal docs score below threshold

No external LLM is invoked here directly – expansion uses lightweight
heuristics so the module stays self-contained and fast.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from .knowledge_base import KNOWLEDGE_BASE, get_all_docs, get_by_level, get_children

logger = logging.getLogger("mcp_server.crag")

# ── Constants ──────────────────────────────────────────────────────────────────
TOT_RELEVANCE_THRESHOLD = 5.0   # Minimum average ToT score to keep a chunk
MAX_CHUNKS_RETURNED     = 6     # Hard cap on returned chunks


# ── 1. Multi-Query Expansion ──────────────────────────────────────────────────

def expand_queries(query: str, n: int = 3) -> list[str]:
    """
    Generate N semantically varied query variants using lightweight heuristics.
    In production this would call an LLM; here we apply deterministic transforms
    to avoid adding a hard LLM dependency inside the resource.
    """
    base = query.strip().rstrip("?")
    variants = [base]

    # Variant 2: Rephrase with "explain" prefix
    if not base.lower().startswith(("explain", "what is", "how")):
        variants.append(f"explain {base}")

    # Variant 3: Add "with examples" suffix
    if len(variants) < n:
        variants.append(f"{base} with examples and code")

    # Variant 4: Keyword extraction (drop stop words)
    if len(variants) < n:
        stop = {"the", "a", "an", "is", "are", "of", "in", "to", "for", "and"}
        kws = [w for w in base.lower().split() if w not in stop]
        if kws:
            variants.append(" ".join(kws))

    unique: list[str] = []
    seen: set[str] = set()
    for v in variants:
        k = v.lower()
        if k not in seen:
            seen.add(k)
            unique.append(v)

    logger.debug("Query expansion: %s → %s", query, unique[:n])
    return unique[:n]


# ── 2. BM25-style scoring helper ──────────────────────────────────────────────

def _tokenise(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def _bm25_score(doc_tokens: list[str], query_tokens: list[str]) -> float:
    """Simplified BM25 term-frequency overlap score (no IDF for brevity)."""
    if not query_tokens or not doc_tokens:
        return 0.0
    doc_set = set(doc_tokens)
    hits = sum(1 for t in query_tokens if t in doc_set)
    return hits / len(query_tokens)


def _score_doc(doc: dict, query_tokens: list[str]) -> float:
    title_score  = _bm25_score(_tokenise(doc["title"]), query_tokens) * 2.0
    text_score   = _bm25_score(_tokenise(doc["text"]),  query_tokens)
    return title_score + text_score


# ── 3. Hierarchical Search ────────────────────────────────────────────────────

def hierarchical_search(queries: list[str], top_k: int = 8) -> list[dict]:
    """
    Top-down hierarchical retrieval:
      L0 summaries → filter top domains → expand to L1 → expand to L2
    Returns deduplicated list of (doc, score) sorted by relevance.
    """
    all_query_tokens: list[str] = []
    for q in queries:
        all_query_tokens.extend(_tokenise(q))

    scored: dict[str, tuple[dict, float]] = {}

    # Score L0 domains
    l0_docs = get_by_level(0)
    l0_scored = sorted(
        [(d, _score_doc(d, all_query_tokens)) for d in l0_docs],
        key=lambda x: x[1], reverse=True
    )

    for l0_doc, l0_s in l0_scored[:2]:          # take top-2 domains
        scored[l0_doc["id"]] = (l0_doc, l0_s)

        # Drill into L1 children
        for l1_doc in get_children(l0_doc["id"]):
            l1_s = _score_doc(l1_doc, all_query_tokens)
            scored[l1_doc["id"]] = (l1_doc, l1_s)

            # Drill into L2 children
            for l2_doc in get_children(l1_doc["id"]):
                l2_s = _score_doc(l2_doc, all_query_tokens)
                scored[l2_doc["id"]] = (l2_doc, l2_s)

    results = sorted(scored.values(), key=lambda x: x[1], reverse=True)
    logger.debug("Hierarchical search retrieved %d candidates", len(results))
    return [doc for doc, _ in results[:top_k]]


# ── 4. Tree-of-Thought (ToT) Evaluation ──────────────────────────────────────

def _tot_path_score(doc: dict, query_tokens: list[str], path: str) -> float:
    """
    Three reasoning paths for ToT evaluation:
      path='literal'   – exact keyword overlap
      path='semantic'  – bigram/phrase overlap
      path='structural'– level weighting (L2 preferred for grounding)
    Returns score 0-10.
    """
    text_tokens = _tokenise(doc["text"])
    title_tokens = _tokenise(doc["title"])

    if path == "literal":
        overlap = sum(1 for t in query_tokens if t in set(text_tokens + title_tokens))
        return min(10.0, (overlap / max(len(query_tokens), 1)) * 10)

    if path == "semantic":
        # Bigram overlap between query and doc title
        def bigrams(toks: list[str]) -> set[tuple[str, str]]:
            return {(toks[i], toks[i+1]) for i in range(len(toks)-1)}
        q_bi = bigrams(query_tokens)
        d_bi = bigrams(title_tokens + text_tokens[:30])
        if not q_bi:
            return 0.0
        inter = q_bi & d_bi
        return min(10.0, (len(inter) / len(q_bi)) * 10 * 2)

    if path == "structural":
        # Prefer deeper (more specific) docs; penalise if very short text
        level_bonus = {0: 2.0, 1: 5.0, 2: 8.0}
        base = level_bonus.get(doc.get("level", 0), 3.0)
        length_bonus = min(2.0, len(doc["text"]) / 400)
        return base + length_bonus

    return 0.0


def tot_evaluate(
    docs: list[dict],
    query: str,
    threshold: float = TOT_RELEVANCE_THRESHOLD,
) -> list[tuple[dict, float]]:
    """
    Run 3-path ToT on each candidate chunk.
    Returns list of (doc, avg_score) for chunks passing threshold.
    """
    query_tokens = _tokenise(query)
    results: list[tuple[dict, float]] = []

    for doc in docs:
        scores = {
            p: _tot_path_score(doc, query_tokens, p)
            for p in ("literal", "semantic", "structural")
        }
        avg = sum(scores.values()) / 3
        logger.debug(
            "ToT [%s] L%d '%s' → paths=%s avg=%.2f",
            doc["id"], doc.get("level", -1), doc["title"], scores, avg,
        )
        results.append((doc, avg))

    passing = [(d, s) for d, s in results if s >= threshold]
    logger.info(
        "ToT evaluation: %d/%d chunks passed threshold %.1f",
        len(passing), len(docs), threshold,
    )
    return sorted(passing, key=lambda x: x[1], reverse=True)


# ── 5. Tavily Fallback ────────────────────────────────────────────────────────

async def tavily_fallback(query: str) -> list[dict]:
    """
    Fetch web results via Tavily when internal docs are insufficient.
    Returns synthesised doc dicts compatible with the KB schema.
    """
    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        logger.warning("TAVILY_API_KEY not set – skipping web fallback")
        return []

    try:
        import httpx
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": api_key,
                    "query": query,
                    "search_depth": "basic",
                    "max_results": 3,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        results = []
        for r in data.get("results", []):
            results.append({
                "id": f"WEB-{hash(r.get('url',''))%10000:04d}",
                "level": 2,
                "title": r.get("title", "Web Result"),
                "text": r.get("content", r.get("snippet", "")),
                "source": "tavily",
                "url": r.get("url", ""),
            })
        logger.info("Tavily returned %d fallback chunks", len(results))
        return results

    except Exception as exc:  # noqa: BLE001
        logger.error("Tavily fallback failed: %s", exc)
        return []


# ── 6. Full CRAG Pipeline ─────────────────────────────────────────────────────

async def run_crag(query: str) -> str:
    """
    Full Corrective RAG pipeline.
    Returns a formatted string ready to be returned as an MCP resource.
    """
    logger.info("CRAG pipeline started for query: %s", query)

    # Step 1 – Multi-query expansion
    queries = expand_queries(query, n=3)

    # Step 2 – Hierarchical retrieval
    candidates = hierarchical_search(queries, top_k=10)

    # Step 3 – ToT evaluation
    scored = tot_evaluate(candidates, query)

    # Step 4 – Fallback if needed
    if not scored:
        logger.info("All chunks below ToT threshold – triggering Tavily fallback")
        web_docs = await tavily_fallback(query)
        if web_docs:
            scored = [(d, 5.0) for d in web_docs]   # accept all web results

    # Step 5 – Synthesise output
    if not scored:
        return (
            f"No relevant documentation found for: '{query}'.\n"
            "Please consult external resources or rephrase your query."
        )

    top_chunks = scored[:MAX_CHUNKS_RETURNED]
    lines: list[str] = [
        f"## CRAG Results for: '{query}'\n",
        f"*Expanded queries: {queries}*\n",
        f"*{len(top_chunks)} chunk(s) passed ToT evaluation*\n",
        "---\n",
    ]

    for rank, (doc, score) in enumerate(top_chunks, 1):
        source_tag = f"[{doc.get('source', 'internal')}]" if doc.get("source") else "[internal]"
        lines.append(
            f"### [{rank}] {doc['title']} {source_tag} (relevance: {score:.1f}/10)\n"
        )
        lines.append(doc["text"])
        if doc.get("url"):
            lines.append(f"\nSource: {doc['url']}")
        lines.append("\n\n---\n")

    return "\n".join(lines)
