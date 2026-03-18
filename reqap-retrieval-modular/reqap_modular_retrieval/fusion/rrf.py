from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List

from ..core.types import RetrievedDoc


def rrf_fuse(
    lists: Dict[str, List[RetrievedDoc]],
    *,
    k: int = 60,
) -> List[RetrievedDoc]:
    ranks: Dict[str, Dict[int, int]] = {}
    doc_map: Dict[int, RetrievedDoc] = {}

    for name, docs in lists.items():
        ranks[name] = {d.id: i + 1 for i, d in enumerate(docs)}
        for d in docs:
            doc_map.setdefault(d.id, d)

    all_ids = set(doc_map.keys())
    scores = defaultdict(float)

    for doc_id in all_ids:
        s = 0.0
        for name, rmap in ranks.items():
            r = rmap.get(doc_id)
            if r is not None:
                s += 1.0 / (k + r)
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
                derivation=list(d.derivation) + [{"method": "rrf", "k": k, "inputs": list(lists.keys())}],
            )
        )
    return out

