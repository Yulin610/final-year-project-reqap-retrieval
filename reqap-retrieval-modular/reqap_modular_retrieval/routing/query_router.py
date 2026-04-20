"""
Per-query fusion routing (rule-based MVP).

Maps a query string to weights over BM25 / SPLADE / dense **score channels**
(keys must match RetrievedDoc.signals: bm25_score, splade_score, dense_score).

Used by DynamicFusionOurs instead of rank-only RRF for distribution-aware retrieval.
"""
from __future__ import annotations

from typing import Dict


def is_short_query(q: str) -> bool:
    """
    Heuristic short intent vs long structured text (see paper / eval Task B vs Task A).

    Spec (user-provided MVP):
      len(q.split()) <= 4 OR "event:" in q OR "\\n" not in q
    """
    if not q:
        return True
    if len(q.split()) <= 4:
        return True
    if "event:" in q:
        return True
    if "\n" not in q:
        return True
    return False


def is_structured_query(q: str) -> bool:
    """
    Long-form structured event (calendar / mail / multi-line payload), not short intent.

    Only evaluated when is_short_query is False (router order: short -> structured -> default).
    """
    if is_short_query(q):
        return False
    ql = q.lower()
    markers = (
        "calendar",
        "mail",
        "meeting",
        "reminder",
        "agenda",
        "inbox",
        "schedule",
    )
    if "\n" in q and len(q.split()) > 4:
        return True
    return any(m in ql for m in markers)


def route_query_fusion_weights(query: str) -> Dict[str, float]:
    """
    Returns normalized weights for weighted_sum_fuse signal keys.
    """
    if is_short_query(query):
        raw = {"bm25_score": 0.5, "splade_score": 0.3, "dense_score": 0.2}
    elif is_structured_query(query):
        raw = {"bm25_score": 0.2, "splade_score": 0.6, "dense_score": 0.2}
    else:
        raw = {"bm25_score": 0.3, "splade_score": 0.4, "dense_score": 0.3}

    s = sum(raw.values())
    if s <= 0:
        return {"bm25_score": 1.0 / 3, "splade_score": 1.0 / 3, "dense_score": 1.0 / 3}
    return {k: v / s for k, v in raw.items()}
