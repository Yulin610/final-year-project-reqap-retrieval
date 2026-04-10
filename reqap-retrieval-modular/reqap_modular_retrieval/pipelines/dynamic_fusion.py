from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from ..core.types import RetrievedDoc
from ..fusion.rrf import rrf_fuse
from ..fusion.weighted_rrf import weighted_rrf_fuse
from ..retrievers.bm25 import BM25Retriever
from ..retrievers.dense_adapter import DenseFaissRetriever
from ..retrievers.splade_adapter import SpladeRetriever
from .splade_then_dense import SpladeThenDenseRerank


@dataclass
class DynamicFusionOurs:
    """
    Query-frequency–aware routing (counts from QU dataset RETRIEVE(query=...)):
    - High-frequency: SPLADE || Dense parallel, fusion = Weighted RRF with dense:sparse = 7:3;
      semantic cache (sentence-transformer embedding similarity, LRU) for paraphrase hits.
    - Low-frequency: BM25 coarse recall → dense ST rerank on candidates → RRF(bm25, dense_rerank).
    """

    splade: SpladeRetriever
    dense: DenseFaissRetriever
    bm25: BM25Retriever
    splade_then_dense: SpladeThenDenseRerank
    retrieve_counts: Dict[str, int]
    freq_percentile_threshold: float = 75.0
    cache_max: int = 256
    semantic_cache_threshold: float = 0.88

    weight_dense_high: float = 0.3
    weight_splade_high: float = 0.7
    wrrf_k: int = 60

    _sem_cache: List[Tuple[np.ndarray, List[RetrievedDoc]]] = field(default_factory=list)
    _threshold: Optional[float] = field(default=None, init=False)

    def __post_init__(self):
        vals = sorted(self.retrieve_counts.values())
        if not vals:
            self._threshold = float("inf")
        else:
            self._threshold = float(
                np.percentile(np.array(vals, dtype=np.float64), self.freq_percentile_threshold)
            )

    def _is_high_freq(self, query_key: str) -> bool:
        c = self.retrieve_counts.get(query_key.strip(), 0)
        return c >= self._threshold and c > 0

    def _sem_cache_get(self, q_emb: np.ndarray) -> Optional[List[RetrievedDoc]]:
        q = q_emb.astype("float32").reshape(-1)
        n = float(np.linalg.norm(q) + 1e-12)
        q = q / n
        best_i = -1
        best_sim = -1.0
        for i, (e, _) in enumerate(self._sem_cache):
            sim = float(np.dot(q, e.reshape(-1)))
            if sim > best_sim:
                best_sim = sim
                best_i = i
        if best_i < 0 or best_sim < self.semantic_cache_threshold:
            return None
        entry = self._sem_cache.pop(best_i)
        self._sem_cache.append(entry)
        return list(entry[1])

    def _sem_cache_put(self, q_emb: np.ndarray, docs: List[RetrievedDoc]) -> None:
        q = q_emb.astype("float32").reshape(-1)
        q = q / (np.linalg.norm(q) + 1e-12)
        self._sem_cache.append((q.copy(), list(docs)))
        while len(self._sem_cache) > self.cache_max:
            self._sem_cache.pop(0)

    def retrieve(
        self,
        query: str,
        *,
        query_key: str,
        top_k_splade: int = 500,
        top_k_dense: int = 500,
        top_k_bm25_cold: int = 200,
        top_k_final: int = 100,
        threshold_splade: float = 0.0,
        threshold_dense: float = 0.0,
    ) -> List[RetrievedDoc]:
        high = self._is_high_freq(query_key)

        if high:
            q_emb = self.splade_then_dense.encode_query(query)
            hit = self._sem_cache_get(q_emb)
            if hit is not None:
                return hit[:top_k_final] if top_k_final else hit

            splade_docs = self.splade.retrieve(query, top_k=top_k_splade, threshold=threshold_splade)
            dense_docs = self.dense.retrieve(query, top_k=top_k_dense, threshold=threshold_dense)

            fused = weighted_rrf_fuse(
                {"splade": splade_docs, "dense": dense_docs},
                weights={"splade": self.weight_splade_high, "dense": self.weight_dense_high},
                k=self.wrrf_k,
            )
            if top_k_final and top_k_final > 0:
                fused = fused[:top_k_final]
            self._sem_cache_put(q_emb, fused)
            logger.debug(f"DynamicFusion HIGH freq key={query_key!r} wrrf fused={len(fused)}")
            return fused

        bm25_docs = self.bm25.retrieve(query, top_k=top_k_bm25_cold)
        dense_ranked = self.splade_then_dense.rerank_documents(query, bm25_docs)
        fused = rrf_fuse({"bm25": bm25_docs, "dense_rerank": dense_ranked}, k=self.wrrf_k)
        out = fused[:top_k_final] if top_k_final > 0 else fused
        logger.debug(f"DynamicFusion LOW freq key={query_key!r} bm25→dense rrf={len(out)}")
        return out
