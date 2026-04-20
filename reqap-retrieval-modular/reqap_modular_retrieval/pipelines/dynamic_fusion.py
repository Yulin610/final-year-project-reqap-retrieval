from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from loguru import logger

from ..core.types import RetrievedDoc
from ..fusion.weighted_sum import weighted_sum_fuse
from ..retrievers.bm25 import BM25Retriever
from ..retrievers.dense_adapter import DenseFaissRetriever
from ..retrievers.splade_adapter import SpladeRetriever
from ..routing.learned_router import LearnedRouterModel, extract_router_features, load_learned_router_model
from ..routing.query_router import route_query_fusion_weights
from .splade_then_dense import SpladeThenDenseRerank


@dataclass
class DynamicFusionOurs:
    """
    Query-conditioned mixture-of-retrievers (no Cross-Encoder):

    1) Parallel pre-retrieval: BM25 / SPLADE / Dense each with a wide top-k pool.
    2) Union candidates by doc id; each doc carries bm25_score / splade_score / dense_score in signals.
    3) Per-query **weighted score fusion** (min–max normalize each signal, then
       score = w_bm25 * bm25~ + w_splade * splade~ + w_dense * dense~) with weights from
       `route_query_fusion_weights(query)` (short / structured / default), unless grid-search
       overrides are set (`w1_bm25`, `w2_dense`, `w3_splade`).
    4) Truncate to top_k_semantic then top_k_final.
    """

    splade: SpladeRetriever
    dense: DenseFaissRetriever
    bm25: BM25Retriever
    splade_then_dense: SpladeThenDenseRerank
    retrieve_counts: Dict[str, int]  # kept for API compatibility with older configs

    top_k_candidates: int = 1000
    top_k_semantic: int = 100

    # Optional fixed weights for grid search (if all three set, router is bypassed).
    w1_bm25: Optional[float] = None
    w2_dense: Optional[float] = None
    w3_splade: Optional[float] = None
    learned_router_model_path: Optional[str] = None
    learned_router_model: Optional[LearnedRouterModel] = None

    @staticmethod
    def _merge_by_id(doc_lists: List[List[RetrievedDoc]]) -> List[RetrievedDoc]:
        merged: Dict[int, RetrievedDoc] = {}
        for docs in doc_lists:
            for d in docs:
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
        return list(merged.values())

    def _fusion_weights(self, query: str, merged_list: List[RetrievedDoc]) -> Dict[str, float]:
        if (
            self.w1_bm25 is not None
            and self.w2_dense is not None
            and self.w3_splade is not None
        ):
            raw = {
                "bm25_score": float(self.w1_bm25),
                "dense_score": float(self.w2_dense),
                "splade_score": float(self.w3_splade),
            }
            s = sum(raw.values())
            if s <= 0:
                return route_query_fusion_weights(query)
            return {k: v / s for k, v in raw.items()}

        # Fallback order:
        # 1) fixed grid weights, 2) learned router, 3) rule-based router.
        mdl = self.learned_router_model
        if mdl is None and self.learned_router_model_path:
            mdl = load_learned_router_model(self.learned_router_model_path)
            if mdl is not None:
                self.learned_router_model = mdl
        if mdl is not None:
            try:
                feats = extract_router_features(query, merged_list)
                return mdl.predict(feats)
            except Exception as e:
                logger.warning(f"Learned router predict failed, fallback to rules: {e}")
        return route_query_fusion_weights(query)

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
        _ = query_key  # callers pass QU/query_key; router uses full `query` string

        pre_k = int(min(max(200, top_k_splade), max(200, top_k_dense), 1000))
        bm25_k = int(min(max(pre_k, top_k_bm25_cold), 2000))

        dense_docs = self.dense.retrieve(query, top_k=pre_k, threshold=threshold_dense)
        splade_docs = self.splade.retrieve(query, top_k=pre_k, threshold=threshold_splade)
        bm25_docs = self.bm25.retrieve(query, top_k=bm25_k, threshold=0.0)

        if not dense_docs and not splade_docs and not bm25_docs:
            return []

        merged_list = self._merge_by_id([dense_docs, splade_docs, bm25_docs])
        weights = self._fusion_weights(query, merged_list)

        fused = weighted_sum_fuse(merged_list, weights=weights, normalize=True)

        k_sem = min(self.top_k_semantic, len(fused))
        fused = fused[:k_sem]

        out_docs: List[RetrievedDoc] = []
        for d in fused:
            sig = dict(d.signals)
            sig.update(
                {
                    "fusion_backend": "query_weighted_sum",
                    "fusion_weights": dict(weights),
                    "router_backend": (
                        "fixed_weights"
                        if (
                            self.w1_bm25 is not None
                            and self.w2_dense is not None
                            and self.w3_splade is not None
                        )
                        else ("learned_router" if self.learned_router_model or self.learned_router_model_path else "rule_router")
                    ),
                }
            )
            out_docs.append(
                RetrievedDoc(
                    id=d.id,
                    score=float(d.score),
                    data=d.data,
                    signals=sig,
                    derivation=list(d.derivation)
                    + [
                        {
                            "method": "query_weighted_sum",
                            "weights": dict(weights),
                            "top_k_pre": pre_k,
                            "top_k_bm25": bm25_k,
                            "top_k_semantic": k_sem,
                        }
                    ],
                )
            )

        final = out_docs[:top_k_final] if top_k_final and top_k_final > 0 else out_docs
        logger.debug(
            f"DynamicFusion(query-weighted sum) candidates={len(merged_list)} "
            f"semantic={k_sem} returned={len(final)}"
        )
        return final
