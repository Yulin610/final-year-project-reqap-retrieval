from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from loguru import logger

from ..core.types import RetrievedDoc
from ..retrievers.bm25 import default_event_to_text
from ..retrievers.bm25 import BM25Retriever
from ..retrievers.dense_adapter import DenseFaissRetriever
from ..retrievers.splade_adapter import SpladeRetriever
from .splade_then_dense import SpladeThenDenseRerank


@dataclass
class DynamicFusionOurs:
    """
    Clean fusion (no Cross-Encoder):
    1) Parallel pre-retrieval: Dense/SPLADE topK=200 each, union by id.
    2) Local fusion on top_k_semantic (default 100) by backbone ranking:
       - Reciprocal Rank Fusion (RRF) over ranks from each retriever:
         score(d) = Σ_i 1 / (k + rank_i(d))
    3) return top_k_final
    """

    splade: SpladeRetriever
    dense: DenseFaissRetriever
    bm25: BM25Retriever
    splade_then_dense: SpladeThenDenseRerank
    retrieve_counts: Dict[str, int]  # kept for API compatibility

    # Stage sizes
    top_k_candidates: int = 1000
    top_k_semantic: int = 100

    # RRF constant (larger -> flatter; typical 60)
    rrf_k: int = 60

    @staticmethod
    def _normalize(score_map: Dict[int, float]) -> Dict[int, float]:
        if not score_map:
            return {}
        vals = list(score_map.values())
        v_min = min(vals)
        v_max = max(vals)
        denom = (v_max - v_min) + 1e-8
        return {doc_id: (v - v_min) / denom for doc_id, v in score_map.items()}

    def retrieve(
        self,
        query: str,
        *,
        query_key: str,
        top_k_splade: int = 500,
        top_k_dense: int = 500,
        top_k_bm25_cold: int = 1000,
        top_k_final: int = 100,
        threshold_splade: float = 0.0,
        threshold_dense: float = 0.0,
    ) -> List[RetrievedDoc]:
        _ = query_key  # intentionally unused in this fixed strategy
        _ = top_k_splade, top_k_dense  # kept for compatibility

        # Stage 1: parallel pre-retrieval from Dense / SPLADE (wider candidate pool)
        # 用户需求: 去掉 BM25，仅做 RRF(Dense + SPLADE), k=60
        pre_k = 200
        dense_docs = self.dense.retrieve(query, top_k=pre_k, threshold=threshold_dense)
        splade_docs_for_pool = self.splade.retrieve(query, top_k=pre_k, threshold=threshold_splade)

        if not dense_docs and not splade_docs_for_pool:
            return []

        # union by id,合并来自两路的 signals/derivation
        merged: Dict[int, RetrievedDoc] = {}
        for d in dense_docs:
            merged[d.id] = d
        for d in splade_docs_for_pool:
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

        # Stage 2: Reciprocal Rank Fusion (RRF) on ranks from each retriever list.
        # score(d) = Σ_i 1 / (rrf_k + rank_i(d)), where rank starts at 1.
        def _rank_map(docs: List[RetrievedDoc]) -> Dict[int, int]:
            return {d.id: i for i, d in enumerate(docs, start=1)}

        dense_rank = _rank_map(dense_docs)
        splade_rank = _rank_map(splade_docs_for_pool)

        def _rrf_score(doc_id: int) -> float:
            s = 0.0
            r = dense_rank.get(doc_id)
            if r is not None:
                s += 1.0 / (self.rrf_k + r)
            r = splade_rank.get(doc_id)
            if r is not None:
                s += 1.0 / (self.rrf_k + r)
            return s

        candidates: List[Tuple[RetrievedDoc, float]] = []
        for d in merged.values():
            candidates.append((d, _rrf_score(d.id)))
        candidates.sort(key=lambda t: t[1], reverse=True)

        k_candidates = len(candidates)
        k_sem = min(self.top_k_semantic, k_candidates)
        fused_pool = candidates[:k_sem]

        stage2_docs: List[RetrievedDoc] = []
        for d, s2 in fused_pool:
            sig = dict(d.signals)
            sig.update({"rrf_score": float(s2), "rrf_k": int(self.rrf_k)})
            stage2_docs.append(
                RetrievedDoc(
                    id=d.id,
                    score=float(s2),
                    data=d.data,
                    signals=sig,
                    derivation=list(d.derivation)
                    + [
                        {
                            "method": "rrf",
                            "rrf_k": int(self.rrf_k),
                            "top_k_pre": pre_k,
                            "top_k_candidates": k_candidates,
                            "top_k_semantic": k_sem,
                        }
                    ],
                )
            )

        out = stage2_docs[:top_k_final] if top_k_final and top_k_final > 0 else stage2_docs
        logger.debug(
            f"DynamicFusion(Dense+SPLADE RRF k={self.rrf_k}) "
            f"candidates={k_candidates} semantic={k_sem} returned={len(out)}"
        )
        return out
