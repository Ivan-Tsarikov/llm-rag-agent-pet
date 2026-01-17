from __future__ import annotations

from typing import Any, Dict

from src.agent.calc import safe_calc, CalcError
from src.rag.retriever import Retriever


MAX_QUERY_LEN = 2000
MAX_TOP_K = 10
MAX_HITS = 10
MAX_TEXT_CHARS = 2000
MAX_EXPR_LEN = 200


def search_docs_impl(retriever: Retriever, query: str, top_k: int = 5) -> Dict[str, Any]:
    q = (query or "").strip()
    if len(q) > MAX_QUERY_LEN:
        return {"error": f"Query too long (max {MAX_QUERY_LEN} chars)."}

    try:
        top_k_int = int(top_k)
    except (TypeError, ValueError):
        top_k_int = 5

    top_k_int = max(1, min(top_k_int, MAX_TOP_K))
    hits = retriever.search(q, top_k=top_k_int)
    limited_hits = hits[: min(MAX_HITS, len(hits))]

    return {
        "hits": [
            {
                "source_path": h.record.source_path,
                "chunk_id": h.record.chunk_id,
                "score": float(h.score),
                "text": (h.record.text or "")[:MAX_TEXT_CHARS],
            }
            for h in limited_hits
        ]
    }


def calc_impl(expression: str) -> Dict[str, Any]:
    expr = (expression or "").strip()
    if len(expr) > MAX_EXPR_LEN:
        return {"error": f"Expression too long (max {MAX_EXPR_LEN} chars)."}

    try:
        result = safe_calc(expr)
        return {"value": result.value, "formatted": result.formatted}
    except CalcError as exc:
        return {"error": str(exc)}
