from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from loguru import logger

from reqap.retrieval.splade.index_construction import CollectionDataset

from ..core.types import RetrievedDoc
from ..fusion.weighted_sum import weighted_sum_fuse
from ..retrievers.bm25 import BM25Retriever, default_event_to_text
from ..retrievers.splade_adapter import SpladeRetriever


@dataclass
class SpladeBM25Fusion:
    """
    SPLADE + BM25 fusion (modern statistical vs classic statistical).

    Motivation:
    - SPLADE can capture sparse semantic expansion and term co-occurrence patterns.
    - BM25 provides robust literal matching and tf-idf-ish statistical priors.
    """

    splade: SpladeRetriever
    collection: CollectionDataset
    bm25_index_dir: str
    weight_splade: float = 0.7
    weight_bm25: float = 0.3
    normalize: bool = True
    event_to_text: callable = default_event_to_text

    def __post_init__(self):
        self.bm25 = BM25Retriever(self.collection, index_dir=self.bm25_index_dir, event_to_text=self.event_to_text)

    def retrieve(
        self,
        query: str,
        *,
        top_k_splade: int = 1000,
        top_k_bm25: int = 1000,
        threshold_splade: float = 0.0,
        threshold_bm25: float = 0.0,
        top_k_final: int = 0,
    ) -> List[RetrievedDoc]:
        splade_docs = self.splade.retrieve(query, top_k=top_k_splade, threshold=threshold_splade)
        bm25_docs = self.bm25.retrieve(query, top_k=top_k_bm25, threshold=threshold_bm25)

        merged: Dict[int, RetrievedDoc] = {}
        for d in splade_docs:
            merged[d.id] = d
        for d in bm25_docs:
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
            weights={"splade_score": self.weight_splade, "bm25_score": self.weight_bm25},
            normalize=self.normalize,
        )

        if top_k_final and top_k_final > 0:
            fused = fused[:top_k_final]

        logger.debug(f"SPLADE||BM25 fused={len(fused)} (splade={len(splade_docs)} bm25={len(bm25_docs)})")
        return fused

