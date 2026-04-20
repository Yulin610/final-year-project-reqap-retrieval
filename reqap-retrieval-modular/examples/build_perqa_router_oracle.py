from __future__ import annotations

import argparse
import json
import os
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

from omegaconf import OmegaConf

from build_eval_indexes import resolved_splade_model_type_or_path
from eval_perqa_retrieval_export import load_queries_from_path
from reqap.retrieval.dense.dense_retrieval import DenseRetrieval
from reqap.retrieval.splade.index_construction import CollectionDataset
from reqap.retrieval.splade.models import Splade
from reqap.retrieval.splade.sparse_retrieval import SparseRetrieval
from reqap_modular_retrieval.fusion.weighted_sum import weighted_sum_fuse
from reqap_modular_retrieval.pipelines.dynamic_fusion import DynamicFusionOurs
from reqap_modular_retrieval.retrievers import BM25Retriever, DenseFaissRetriever, SpladeRetriever
from reqap_modular_retrieval.routing.learned_router import extract_router_features, grid_48


def _recall_at_k(pred: Sequence[int], rel: Set[int], k: int) -> float:
    if not rel:
        return 0.0
    top = pred[:k]
    return len(set(top) & rel) / float(len(rel))


def _mrr(pred: Sequence[int], rel: Set[int]) -> float:
    for i, d in enumerate(pred, start=1):
        if d in rel:
            return 1.0 / float(i)
    return 0.0


def _dcg(rels: Sequence[int]) -> float:
    s = 0.0
    for i, r in enumerate(rels, start=1):
        if r <= 0:
            continue
        s += 1.0 / (math.log2(i + 1.0))
    return s


def _ndcg_at_k(pred: Sequence[int], rel: Set[int], k: int) -> float:
    top = pred[:k]
    gains = [1 if d in rel else 0 for d in top]
    dcg = _dcg(gains)
    ideal = _dcg([1] * min(k, len(rel)))
    if ideal <= 0:
        return 0.0
    return dcg / ideal


def main() -> None:
    ap = argparse.ArgumentParser(description="Build per-query oracle labels for Task C learned router.")
    ap.add_argument("--queries-jsonl", required=True)
    ap.add_argument("--obs-csv", required=True)
    ap.add_argument("--splade-index", required=True)
    ap.add_argument("--dense-index", required=True)
    ap.add_argument("--bm25-index", required=True)
    ap.add_argument("--dense-model", required=True)
    ap.add_argument("--out-jsonl", required=True, help="Oracle labels + features output JSONL.")
    ap.add_argument("--top-k-final", type=int, default=100)
    ap.add_argument("--top-k-pool", type=int, default=500)
    args = ap.parse_args()

    qs = load_queries_from_path(args.queries_jsonl)
    if not qs:
        raise SystemExit("No queries loaded.")

    collection = CollectionDataset(data_path=args.obs_csv)
    splade_path = os.environ.get("PERQA_SPLADE_MODEL_TYPE_OR_PATH", "").strip() or resolved_splade_model_type_or_path()
    splade_cfg = OmegaConf.create(
        {
            "splade_model_type_or_path": splade_path,
            "splade_tokenizer_type": "bert-base-uncased",
        }
    )
    splade_model = Splade(splade_cfg.splade_model_type_or_path, agg="max")
    sparse = SparseRetrieval(
        splade_config=splade_cfg,
        model=splade_model,
        collection=collection,
        dim_voc=splade_model.output_dim,
        splade_index_path=args.splade_index,
    )
    splade = SpladeRetriever(sparse, involve_model=True)

    dense_cfg = {"dense_model_type_or_path": args.dense_model, "use_sentence_transformers": True}
    dense_native = DenseRetrieval(dense_config=dense_cfg, collection=collection, dense_index_path=args.dense_index)
    dense = DenseFaissRetriever(dense_native)
    bm25 = BM25Retriever(collection, index_dir=args.bm25_index)

    grid = grid_48()
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for idx, q in enumerate(qs):
            query = q["query"]
            rel = set(int(x) for x in q["relevant_ids"])
            dense_docs = dense.retrieve(query, top_k=args.top_k_pool, threshold=0.0)
            splade_docs = splade.retrieve(query, top_k=args.top_k_pool, threshold=0.0)
            bm25_docs = bm25.retrieve(query, top_k=max(200, args.top_k_pool), threshold=0.0)
            merged = DynamicFusionOurs._merge_by_id([dense_docs, splade_docs, bm25_docs])
            feats = extract_router_features(query, merged)

            best: Dict[str, Any] = {}
            best_key: Tuple[float, float, float] | None = None
            best_cls: int = -1
            for cls_id, (w_bm25, w_dense, w_splade) in enumerate(grid):
                ws = {"bm25_score": w_bm25, "dense_score": w_dense, "splade_score": w_splade}
                fused = weighted_sum_fuse(merged, weights=ws, normalize=True)[: args.top_k_final]
                pred_ids = [int(d.id) for d in fused]
                recall10 = _recall_at_k(pred_ids, rel, 10)
                mrr = _mrr(pred_ids, rel)
                ndcg10 = _ndcg_at_k(pred_ids, rel, 10)
                # Fixed, reproducible tie-break:
                # 1) max Recall@10
                # 2) tie -> max NDCG@10
                # 3) tie -> min w_bm25 (encourage semantic)
                key = (recall10, ndcg10, -w_bm25)
                if best_key is None or key > best_key:
                    best_key = key
                    best_cls = cls_id
                    best = {
                        "class_id": cls_id,
                        "w_bm25": w_bm25,
                        "w_dense": w_dense,
                        "w_splade": w_splade,
                        "Recall@10": recall10,
                        "NDCG@10": ndcg10,
                        "MRR": mrr,
                    }

            row = {
                "query_id": idx,
                "query": query,
                "query_key": q.get("query_key", ""),
                "num_relevant": len(rel),
                "features": feats,
                "oracle": best,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote oracle labels: {out_path} (queries={len(qs)}, grid={len(grid)}, classes={len(grid)})")


if __name__ == "__main__":
    main()
