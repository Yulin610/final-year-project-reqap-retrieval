from __future__ import annotations

from typing import List

from reqap.retrieval.splade.sparse_retrieval import SparseRetrieval

from ..core.types import RetrievedDoc, Retriever


class SpladeRetriever(Retriever):
    name = "splade"

    def __init__(self, sparse_retrieval: SparseRetrieval, *, involve_model: bool = True):
        self._sr = sparse_retrieval
        self._involve_model = involve_model

    def retrieve(self, query: str, *, top_k: int, threshold: float = 0.0) -> List[RetrievedDoc]:
        raw, _bow = self._sr.retrieve(
            query,
            involve_model=self._involve_model,
            top_k=top_k,
            threshold=threshold,
        )

        out: List[RetrievedDoc] = []
        for d in raw:
            doc_id = int(d["id"])
            data = d.get("data", {k: v for k, v in d.items() if k not in {"id", "score", "derivation"}})
            out.append(
                RetrievedDoc(
                    id=doc_id,
                    score=float(d["score"]),
                    data=data,
                    signals={"splade_score": float(d["score"])},
                    derivation=list(d.get("derivation", [])) + [{"method": "splade", "score": float(d["score"])}],
                )
            )
        out.sort(key=lambda x: x.score, reverse=True)
        return out

