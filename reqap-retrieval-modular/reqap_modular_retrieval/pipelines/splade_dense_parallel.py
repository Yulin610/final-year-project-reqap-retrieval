from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from loguru import logger

from ..core.types import RetrievedDoc
from ..fusion.weighted_sum import weighted_sum_fuse
from ..retrievers.splade_adapter import SpladeRetriever
from ..retrievers.dense_adapter import DenseFaissRetriever


@dataclass
class SpladeDenseParallelFusion:
    """
    Parallel retrieval: SPLADE (sparse semantic) + Dense (deep semantic), then fuse.

    Default fusion is 7:3 (SPLADE:Dense) weighted sum after min-max normalization.
    """

    splade: SpladeRetriever
    dense: DenseFaissRetriever
    weight_splade: float = 0.7
    weight_dense: float = 0.3
    normalize: bool = True

    def retrieve(
        self,
        query: str,
        *,
        top_k_splade: int = 1000,
        top_k_dense: int = 1000,
        threshold_splade: float = 0.0,
        threshold_dense: float = 0.0,
        top_k_final: int = 0,
    ) -> List[RetrievedDoc]:
        splade_docs = self.splade.retrieve(query, top_k=top_k_splade, threshold=threshold_splade)
        dense_docs = self.dense.retrieve(query, top_k=top_k_dense, threshold=threshold_dense)

        merged: Dict[int, RetrievedDoc] = {}
        for d in splade_docs:
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
            weights={"splade_score": self.weight_splade, "dense_score": self.weight_dense},
            normalize=self.normalize,
        )

        if top_k_final and top_k_final > 0:
            fused = fused[:top_k_final]

        logger.debug(f"SPLADE||Dense fused={len(fused)} (splade={len(splade_docs)} dense={len(dense_docs)})")
        return fused

