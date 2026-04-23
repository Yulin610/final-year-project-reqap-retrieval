from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from loguru import logger

from ..core.types import RetrievedDoc
from ..fusion.weighted_sum import weighted_sum_fuse
from ..retrievers.bm25 import BM25Retriever
from ..retrievers.dense_adapter import DenseFaissRetriever


@dataclass
class BM25DenseParallelFusion:
    """Parallel BM25 + dense FAISS, then weighted-sum fusion (min-max per signal)."""

    bm25: BM25Retriever
    dense: DenseFaissRetriever
    weight_bm25: float = 0.5
    weight_dense: float = 0.5
    normalize: bool = True

    def retrieve(
        self,
        query: str,
        *,
        top_k_bm25: int = 1000,
        top_k_dense: int = 1000,
        threshold_bm25: float = 0.0,
        threshold_dense: float = 0.0,
        top_k_final: int = 0,
    ) -> List[RetrievedDoc]:
        bm25_docs = self.bm25.retrieve(query, top_k=top_k_bm25, threshold=threshold_bm25)
        dense_docs = self.dense.retrieve(query, top_k=top_k_dense, threshold=threshold_dense)

        merged: Dict[int, RetrievedDoc] = {}
        for d in bm25_docs:
            merged[d.id] = d
        for d in dense_docs:
            if d.id in merged:
                base = merged[d.id]
                sig = dict(base.signals)
                sig.update(d.signals)
                merged[d.id] = RetrievedDoc(
                    id=base.id,
                    score=base.score,
                    data=base.data,
                    signals=sig,
                    derivation=list(base.derivation) + list(d.derivation),
                )
            else:
                merged[d.id] = d

        fused = weighted_sum_fuse(
            merged.values(),
            weights={"bm25_score": self.weight_bm25, "dense_score": self.weight_dense},
            normalize=self.normalize,
        )
        if top_k_final and top_k_final > 0:
            fused = fused[:top_k_final]
        logger.debug(f"BM25||Dense fused={len(fused)}")
        return fused
