from __future__ import annotations

from typing import List

from reqap.retrieval.dense.dense_retrieval import DenseRetrieval

from ..core.types import RetrievedDoc, Retriever


class DenseFaissRetriever(Retriever):
    name = "dense_faiss"

    def __init__(self, dense_retrieval: DenseRetrieval):
        self._dr = dense_retrieval

    def retrieve(self, query: str, *, top_k: int, threshold: float = 0.0) -> List[RetrievedDoc]:
        raw = self._dr.retrieve(query, top_k=top_k, threshold=threshold)
        out: List[RetrievedDoc] = []
        for d in raw:
            doc_id = int(d["id"])
            data = {k: v for k, v in d.items() if k not in {"id", "score", "derivation"}}
            out.append(
                RetrievedDoc(
                    id=doc_id,
                    score=float(d["score"]),
                    data=data,
                    signals={"dense_score": float(d["score"])},
                    derivation=list(d.get("derivation", [])) + [{"method": "dense_faiss", "score": float(d["score"])}],
                )
            )
        out.sort(key=lambda x: x.score, reverse=True)
        return out

