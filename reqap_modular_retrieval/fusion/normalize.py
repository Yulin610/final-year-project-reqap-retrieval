from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

from ..core.types import RetrievedDoc


def minmax_by_key(docs: Iterable[RetrievedDoc], key: str) -> Dict[int, float]:
    vals: List[Tuple[int, float]] = []
    for d in docs:
        v = d.signals.get(key)
        if isinstance(v, (int, float)):
            vals.append((d.id, float(v)))

    if not vals:
        return {}

    scores = [v for _, v in vals]
    lo = min(scores)
    hi = max(scores)
    denom = (hi - lo) if hi != lo else 1.0
    return {doc_id: (v - lo) / denom for doc_id, v in vals}

