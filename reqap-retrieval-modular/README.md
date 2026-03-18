# Modular Retrieval for ReQAP

This folder is **a sibling** of the original `ReQAP-main/` code. It extracts and modularizes the **first-stage retrieval** part (recall / fusion / rerank) so you can plug in new retrieval logic without touching the Cross-Encoder / pattern pipeline.

## What the original code does (retrieval stage only)

In `ReQAP-main/reqap/retrieval/retrieval.py`:

- **SPLADE recall**: `SparseRetrieval.retrieve(query, ...)` returns many candidates.
- **Optional hybrid**: if `config.hybrid.enabled=true`, it also runs `DenseRetrieval` and fuses results via `HybridRetrieval` (`rrf`, `weighted_sum`, etc).
- The rest of the pipeline (**pattern mining + cross-encoder filtering**) happens after recall and is unchanged by this folder.

## New retrieval logic implemented here

- **BM25 → Dense (serial)**: `BM25ThenDenseRerank`
  - BM25 does literal matching, collects a candidate pool
  - Dense model reranks *only within that pool* (fast + robust for cold/rare words)
- **SPLADE + Dense (parallel, 7:3)**: `SpladeDenseParallelFusion`
  - SPLADE provides sparse semantic recall
  - Dense provides deep semantic recall
  - Fuse via normalized weighted sum (default 0.7 / 0.3)
- **SPLADE + BM25 (fusion)**: `SpladeBM25Fusion`
  - SPLADE as a “smart BM25 upgrade”
  - BM25 as classic statistical prior / literal fallback

## How to run

See `examples/run_modular_retrieval.py`.

You will need:

- Your existing `reqap` environment (same as the original project)
- A SPLADE index directory (the original project already builds this)
- A Dense FAISS index directory if you want SPLADE + Dense (parallel)
- A BM25 index directory (this folder can build it via `bm25s`)

