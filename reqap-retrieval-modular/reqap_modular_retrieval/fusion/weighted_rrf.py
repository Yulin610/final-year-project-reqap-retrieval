from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from ..core.types import RetrievedDoc


def weighted_rrf_fuse(
    lists: Dict[str, List[RetrievedDoc]],
    *,
    weights: Dict[str, float],
    k: int = 60,
) -> List[RetrievedDoc]:
    """
    Weighted Reciprocal Rank Fusion: score(d) = sum_b w_b / (k + rank_b(d)).
    """
    ranks: Dict[str, Dict[int, int]] = {}
    doc_map: Dict[int, RetrievedDoc] = {}

    for name, docs in lists.items():
        ranks[name] = {d.id: i + 1 for i, d in enumerate(docs)}
        for d in docs:
            doc_map.setdefault(d.id, d)

    scores = defaultdict(float)
    for doc_id in doc_map:
        s = 0.0
        for name, rmap in ranks.items():
            r = rmap.get(doc_id)
            if r is not None:
                w = float(weights.get(name, 1.0))
                s += w / (k + r)
        scores[doc_id] = s

    out: List[RetrievedDoc] = []
    for doc_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        d = doc_map[doc_id]
        out.append(
            RetrievedDoc(
                id=d.id,
                score=float(score),
                data=d.data,
                signals=dict(d.signals),
                derivation=list(d.derivation)
                + [{"method": "weighted_rrf", "k": k, "weights": dict(weights), "inputs": list(lists.keys())}],
            )
        )
    return out
