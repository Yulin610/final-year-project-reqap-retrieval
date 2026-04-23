from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence


JsonDict = Dict[str, Any]


@dataclass(frozen=True)
class RetrievedDoc:
    """
    A unified retrieval result format.

    - `id` is the document/event id (string or int).
    - `score` is the final score after fusion/rerank for ranking.
    - `data` is the original doc payload (must include fields expected by downstream code).
    - `signals` holds per-retriever raw scores/ranks for debugging/fusion.
    """

    id: int
    score: float
    data: JsonDict
    signals: JsonDict = field(default_factory=dict)
    derivation: List[JsonDict] = field(default_factory=list)


class Retriever(Protocol):
    name: str

    def retrieve(
        self, query: str, *, top_k: int, threshold: float = 0.0
    ) -> List[RetrievedDoc]:
        ...


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, docs: Sequence[RetrievedDoc]) -> List[RetrievedDoc]:
        ...

