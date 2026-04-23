from __future__ import annotations

from typing import Dict, Iterable, List

from ..core.types import RetrievedDoc
from .normalize import minmax_by_key


def weighted_sum_fuse(
    docs: Iterable[RetrievedDoc],
    *,
    weights: Dict[str, float],
    normalize: bool = True,
) -> List[RetrievedDoc]:
    """
    Combine multiple signals into a single score using a weighted sum.

    Signals are expected in `doc.signals[signal_name]` (float).
    """
    docs_list = list(docs)
    if not docs_list:
        return []

    if normalize:
        norm_maps = {k: minmax_by_key(docs_list, k) for k in weights.keys()}
    else:
        norm_maps = {k: {d.id: float(d.signals.get(k, 0.0) or 0.0) for d in docs_list} for k in weights.keys()}

    out: List[RetrievedDoc] = []
    for d in docs_list:
        score = 0.0
        for key, w in weights.items():
            score += float(w) * float(norm_maps.get(key, {}).get(d.id, 0.0))
        out.append(
            RetrievedDoc(
                id=d.id,
                score=score,
                data=d.data,
                signals=dict(d.signals),
                derivation=list(d.derivation)
                + [{"method": "weighted_sum", "weights": dict(weights), "normalized": normalize}],
            )
        )
    out.sort(key=lambda x: x.score, reverse=True)
    return out

