from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
from loguru import logger

try:
    from sentence_transformers import SentenceTransformer

    _ST_AVAILABLE = True
except Exception:
    SentenceTransformer = None
    _ST_AVAILABLE = False

import torch

from reqap.retrieval.splade.index_construction import CollectionDataset

from ..core.types import RetrievedDoc
from ..retrievers.bm25 import BM25Retriever, default_event_to_text


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
    return a @ b.T


@dataclass
class BM25ThenDenseRerank:
    """
    Serial retrieval: BM25 recall -> Dense rerank over BM25 candidates.

    Use-case: cold/rare literal queries (proper nouns), where BM25 catches exact terms,
    and dense reranks by meaning within the BM25 candidate pool.
    """

    collection: CollectionDataset
    bm25_index_dir: str
    dense_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: Optional[str] = None
    event_to_text: callable = default_event_to_text

    def __post_init__(self):
        self.bm25 = BM25Retriever(self.collection, index_dir=self.bm25_index_dir, event_to_text=self.event_to_text)
        if not _ST_AVAILABLE:
            raise ImportError("sentence-transformers is required for BM25→Dense rerank. Install: pip install sentence-transformers")
        self._device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = SentenceTransformer(self.dense_model_name, device=self._device)

    def retrieve(
        self,
        query: str,
        *,
        top_k_bm25: int = 200,
        top_k_final: int = 50,
        bm25_threshold: float = 0.0,
    ) -> List[RetrievedDoc]:
        bm25_docs = self.bm25.retrieve(query, top_k=top_k_bm25, threshold=bm25_threshold)
        if not bm25_docs:
            return []

        texts: List[str] = [self.event_to_text(d.data) for d in bm25_docs]
        q_emb = self._model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        d_emb = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

        sims = _cosine_sim(q_emb.astype("float32"), d_emb.astype("float32"))[0]
        reranked: List[RetrievedDoc] = []
        for d, sim in zip(bm25_docs, sims.tolist()):
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
        out = reranked[:top_k_final] if top_k_final > 0 else reranked
        logger.debug(f"BM25→Dense: bm25={len(bm25_docs)} reranked={len(out)}")
        return out

