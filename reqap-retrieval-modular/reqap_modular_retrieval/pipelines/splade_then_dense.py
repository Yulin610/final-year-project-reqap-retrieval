from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from loguru import logger

try:
    from sentence_transformers import SentenceTransformer

    _ST_AVAILABLE = True
except Exception:
    SentenceTransformer = None
    _ST_AVAILABLE = False

from ..core.types import RetrievedDoc
from ..fusion.rrf import rrf_fuse
from ..retrievers.bm25 import default_event_to_text
from ..retrievers.splade_adapter import SpladeRetriever


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
    return a @ b.T


@dataclass
class SpladeThenDenseRerank:
    """
    Serial: SPLADE coarse recall -> dense (sentence-transformer) rerank on candidates.
    """

    splade: SpladeRetriever
    dense_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: Optional[str] = None
    event_to_text: callable = default_event_to_text

    def __post_init__(self):
        if not _ST_AVAILABLE:
            raise ImportError("sentence-transformers is required for Splade→Dense rerank.")
        self._device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = SentenceTransformer(self.dense_model_name, device=self._device)

    def encode_query(self, query: str) -> "np.ndarray":
        """Single L2-normalized query embedding, shape (1, dim), float32."""
        return self._model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")

    def rerank_documents(self, query: str, docs: List[RetrievedDoc]) -> List[RetrievedDoc]:
        """Cross-encoder style rerank of an arbitrary candidate list (e.g. BM25 or SPLADE)."""
        return self._rerank(query, docs)

    def _rerank(self, query: str, splade_docs: List[RetrievedDoc]) -> List[RetrievedDoc]:
        if not splade_docs:
            return []
        texts = [self.event_to_text(d.data) for d in splade_docs]
        q_emb = self._model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        d_emb = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        sims = _cosine_sim(q_emb.astype("float32"), d_emb.astype("float32"))[0]
        reranked: List[RetrievedDoc] = []
        for d, sim in zip(splade_docs, sims.tolist()):
            signals = dict(d.signals)
            signals["dense_rerank_score"] = float(sim)
            reranked.append(
                RetrievedDoc(
                    id=d.id,
                    score=float(sim),
                    data=d.data,
                    signals=signals,
                    derivation=list(d.derivation) + [{"method": "dense_rerank", "score": float(sim)}],
                )
            )
        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked

    def retrieve(
        self,
        query: str,
        *,
        top_k_splade: int = 200,
        top_k_final: int = 50,
        threshold_splade: float = 0.0,
    ) -> List[RetrievedDoc]:
        splade_docs = self.splade.retrieve(query, top_k=top_k_splade, threshold=threshold_splade)
        out = self._rerank(query, splade_docs)
        out = out[:top_k_final] if top_k_final > 0 else out
        logger.debug(f"SPLADE→Dense: splade={len(splade_docs)} out={len(out)}")
        return out

    def retrieve_rrf_splade_dense(
        self,
        query: str,
        *,
        top_k_splade: int = 200,
        top_k_final: int = 50,
        threshold_splade: float = 0.0,
        rrf_k: int = 60,
    ) -> List[RetrievedDoc]:
        splade_docs = self.splade.retrieve(query, top_k=top_k_splade, threshold=threshold_splade)
        dense_ranked = self._rerank(query, splade_docs)
        fused = rrf_fuse({"splade": splade_docs, "dense_rerank": dense_ranked}, k=rrf_k)
        return fused[:top_k_final] if top_k_final > 0 else fused
