"""
Standalone retrieval module extracted from ReQAP.

This package focuses on first-stage retrieval (recall / fusion / rerank) and is
designed to be used alongside the existing `reqap` package.
"""

from .pipelines.bm25_then_dense import BM25ThenDenseRerank
from .pipelines.splade_dense_parallel import SpladeDenseParallelFusion
from .pipelines.splade_bm25_fusion import SpladeBM25Fusion

__all__ = [
    "BM25ThenDenseRerank",
    "SpladeDenseParallelFusion",
    "SpladeBM25Fusion",
]

