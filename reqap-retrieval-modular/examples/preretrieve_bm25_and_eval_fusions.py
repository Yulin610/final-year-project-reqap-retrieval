"""
Pre-retrieval evaluation:
- Use BM25 to pre-retrieve a candidate pool (Top-500 / Top-1000) for each query.
- Run fusion pipelines restricted to that BM25 candidate pool:
  1) BM25 + SPLADE
  2) BM25 + Dense-Trieve
  3) SPLADE + Dense (Fixed)
  4) Dynamic Fusion (Ours)

Metrics are kept identical to `eval_three_pipelines.py`.

Prereqs:
  - prepare_retrieve_dev_eval.py (creates obs.csv + queries.jsonl)
  - build_eval_indexes.py (creates indices/splade + indices/bm25 + indices/dense)

Env vars (optional):
  - RETRIEVE_EVAL_WORKSPACE, RETRIEVE_DEV_JSONL, QU_DEV_JSONL, EVAL_MAX_QUERIES
  - PRE_BM25_KS="500,1000"  (default)
  - RETRIEVE_K=100          (default)
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

_EXAMPLES = Path(__file__).resolve().parent
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

try:
    from tqdm import tqdm

    _TQDM = tqdm
except Exception:
    _TQDM = None

from omegaconf import OmegaConf

from prepare_retrieve_dev_eval import prepare as prepare_retrieve_workspace
from qu_retrieve_counts import load_or_build_retrieve_query_counts

# ReQAP + 本模块
_ROOT_REQAP = r"C:\Users\23369\Desktop\final_work\ReQAP-main\ReQAP-main"
_ROOT_MOD = r"C:\Users\23369\Desktop\final_work\ReQAP-main\reqap-retrieval-modular"
sys.path.insert(0, _ROOT_REQAP)
sys.path.insert(0, _ROOT_MOD)

from reqap.retrieval.splade.models import Splade
from reqap.retrieval.splade.index_construction import CollectionDataset
from reqap.retrieval.splade.sparse_retrieval import SparseRetrieval
from reqap.retrieval.dense.dense_retrieval import DenseRetrieval

from reqap_modular_retrieval.core.types import RetrievedDoc
from reqap_modular_retrieval.fusion.weighted_sum import weighted_sum_fuse
from reqap_modular_retrieval.retrievers import BM25Retriever, SpladeRetriever, DenseFaissRetriever
from reqap_modular_retrieval.pipelines import DynamicFusionOurs
from reqap_modular_retrieval.pipelines.splade_then_dense import SpladeThenDenseRerank

# --- Paths ---
WORKSPACE = os.environ.get(
    "RETRIEVE_EVAL_WORKSPACE",
    r"C:\Users\23369\Desktop\final_work\data\retrieve\eval_workspace",
)
DEV_JSONL = os.environ.get(
    "RETRIEVE_DEV_JSONL",
    r"C:\Users\23369\Desktop\final_work\data\retrieve\dev_data.jsonl",
)
QU_DEV = os.environ.get(
    "QU_DEV_JSONL",
    r"C:\Users\23369\Desktop\final_work\data\qu\dev_data.jsonl",
)

OBS_CSV = os.path.join(WORKSPACE, "obs.csv")
if os.environ.get("PERQA_OBS_CSV"):
    OBS_CSV = os.environ["PERQA_OBS_CSV"]
QUERIES_JSONL = os.path.join(WORKSPACE, "queries.jsonl")
INDEX_ROOT = os.path.join(WORKSPACE, "indices")
SPLADE_INDEX = os.path.join(INDEX_ROOT, "splade")
DENSE_INDEX = os.path.join(INDEX_ROOT, "dense")
BM25_INDEX = os.path.join(INDEX_ROOT, "bm25")

RETRIEVE_K = int(os.environ.get("RETRIEVE_K", "100"))
AUTO_WEIGHT_GRID = os.environ.get("AUTO_WEIGHT_GRID", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
    "on",
}
WEIGHT_GRID_STEP = float(os.environ.get("WEIGHT_GRID_STEP", "0.2"))
WEIGHT_GRID_METRIC = os.environ.get("WEIGHT_GRID_METRIC", "MRR").strip()

QU_RETRIEVE_COUNTS_CACHE = os.environ.get(
    "QU_RETRIEVE_COUNTS_CACHE",
    os.path.join(WORKSPACE, "qu_dev_retrieve_counts.json"),
)


def load_queries() -> List[Dict[str, Any]]:
    qs: List[Dict[str, Any]] = []
    with open(QUERIES_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rk = row.get("query_key") or row["query"].split("\n", 1)[0].strip()
            qs.append(
                {
                    "qid": row["qid"],
                    "query": row["query"],
                    "query_key": rk,
                    "relevant_ids": set(row["relevant_ids"]),
                }
            )
    mx = os.environ.get("EVAL_MAX_QUERIES")
    if mx:
        qs = qs[: max(0, int(mx))]
    return qs


def hit_at_k(pred_ids: List[int], rel_ids: Set[int], k: int) -> float:
    pred = pred_ids[:k]
    return 1.0 if any(pid in rel_ids for pid in pred) else 0.0


def mrr(pred_ids: List[int], rel_ids: Set[int], max_depth: int) -> float:
    for rank, pid in enumerate(pred_ids[:max_depth], start=1):
        if pid in rel_ids:
            return 1.0 / rank
    return 0.0


def recall_at_k(pred_ids: List[int], rel_ids: Set[int], k: int) -> float:
    if not rel_ids:
        return 0.0
    pred = set(pred_ids[:k])
    return len(pred & rel_ids) / len(rel_ids)


def precision_at_k(pred_ids: List[int], rel_ids: Set[int], k: int) -> float:
    if k <= 0:
        return 0.0
    pred = pred_ids[:k]
    hits = sum(1 for pid in pred if pid in rel_ids)
    return hits / k


def ndcg_at_k(pred_ids: List[int], rel_ids: Set[int], k: int) -> float:
    pred = pred_ids[:k]
    gains = [1.0 if pid in rel_ids else 0.0 for pid in pred]

    def dcg(gs: List[float]) -> float:
        return sum(g / math.log2(i + 2) for i, g in enumerate(gs))

    dcg_val = dcg(gains)
    ideal = sorted(gains, reverse=True)
    idcg_val = dcg(ideal)
    return 0.0 if idcg_val == 0.0 else dcg_val / idcg_val


def _safe_mean(xs: List[float]) -> float:
    return float(mean(xs)) if xs else 0.0


def _cuda_peak_gb() -> Optional[float]:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return torch.cuda.max_memory_allocated() / (1024.0**3)
    except Exception:
        return None


def evaluate_model(model_label: str, retrieve_fn: Callable[[Dict[str, Any]], List[RetrievedDoc]]) -> Dict[str, Any]:
    qs = load_queries()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass

    h1, h5, h10, h50 = [], [], [], []
    mrrs, r50, p10, ndcg10 = [], [], [], []
    times = []

    it = _TQDM(qs, desc=f"Eval {model_label}", leave=False) if _TQDM else qs
    for q in it:
        t0 = time.time()
        docs = retrieve_fn(q)
        t1 = time.time()
        ids = [d.id for d in docs]
        rel = q["relevant_ids"]

        h1.append(hit_at_k(ids, rel, 1))
        h5.append(hit_at_k(ids, rel, 5))
        h10.append(hit_at_k(ids, rel, 10))
        h50.append(hit_at_k(ids, rel, 50))
        mrrs.append(mrr(ids, rel, RETRIEVE_K))
        r50.append(recall_at_k(ids, rel, 50))
        p10.append(precision_at_k(ids, rel, 10))
        ndcg10.append(ndcg_at_k(ids, rel, 10))
        times.append(t1 - t0)

    peak = _cuda_peak_gb()
    lat_ms = _safe_mean([t * 1000.0 for t in times])

    return {
        "Model": model_label,
        "num_queries": len(qs),
        "Hit@1": _safe_mean(h1),
        "Hit@5": _safe_mean(h5),
        "Hit@10": _safe_mean(h10),
        "Hit@50": _safe_mean(h50),
        "MRR": _safe_mean(mrrs),
        "Recall@50": _safe_mean(r50),
        "Precision@10": _safe_mean(p10),
        "NDCG@10": _safe_mean(ndcg10),
        "Avg. Latency": lat_ms,
        "GPU Memory": peak,
        "total_time_s": float(sum(times)),
    }


def _filter_to_pool(docs: Iterable[RetrievedDoc], pool_ids: Set[int]) -> List[RetrievedDoc]:
    out = [d for d in docs if d.id in pool_ids]
    out.sort(key=lambda x: x.score, reverse=True)
    return out


def _merge_by_id(primary: List[RetrievedDoc], secondary: List[RetrievedDoc]) -> Dict[int, RetrievedDoc]:
    merged: Dict[int, RetrievedDoc] = {d.id: d for d in primary}
    for d in secondary:
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
    return merged


def _write_candidate_jsonl(path: str, qs: List[Dict[str, Any]], pools: Dict[str, List[int]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for q in qs:
            qid = q["qid"]
            f.write(json.dumps({"qid": qid, "candidate_ids": pools.get(qid, [])}) + "\n")


def parse_pre_ks() -> List[int]:
    raw = os.environ.get("PRE_BM25_KS", "500,1000")
    out: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    out = sorted(set(out))
    return out or [500, 1000]


def _make_weight_grid(step: float) -> List[Tuple[float, float, float, float]]:
    # Use an integer grid that sums to 1.0 exactly at grid resolution.
    n = int(round(1.0 / step)) if step > 0 else 10
    n = max(1, n)
    out: List[Tuple[float, float, float, float]] = []
    for i in range(n + 1):
        for j in range(n - i + 1):
            for k in range(n - i - j + 1):
                l = n - i - j - k
                out.append((i / n, j / n, k / n, l / n))
    return out


def _evaluate_dynamic_weight_from_cache(
    cache_rows: List[Dict[str, Any]],
    weights: Tuple[float, float, float, float],
) -> Dict[str, float]:
    a, b, c, dlt = weights
    h1, h5, h10, h50 = [], [], [], []
    mrrs, r50, p10, ndcg10 = [], [], [], []

    for row in cache_rows:
        rel = row["relevant_ids"]
        docs = row["docs"]
        scored: List[Tuple[int, float]] = []
        for doc in docs:
            s = (
                a * doc["bm25_norm"]
                + b * doc["dense_norm"]
                + c * doc["splade_norm"]
                + dlt * doc["ce_norm"]
            )
            scored.append((doc["id"], float(s)))
        scored.sort(key=lambda x: x[1], reverse=True)
        ids = [x[0] for x in scored[:RETRIEVE_K]]

        h1.append(hit_at_k(ids, rel, 1))
        h5.append(hit_at_k(ids, rel, 5))
        h10.append(hit_at_k(ids, rel, 10))
        h50.append(hit_at_k(ids, rel, 50))
        mrrs.append(mrr(ids, rel, RETRIEVE_K))
        r50.append(recall_at_k(ids, rel, 50))
        p10.append(precision_at_k(ids, rel, 10))
        ndcg10.append(ndcg_at_k(ids, rel, 10))

    return {
        "Hit@1": _safe_mean(h1),
        "Hit@5": _safe_mean(h5),
        "Hit@10": _safe_mean(h10),
        "Hit@50": _safe_mean(h50),
        "MRR": _safe_mean(mrrs),
        "Recall@50": _safe_mean(r50),
        "Precision@10": _safe_mean(p10),
        "NDCG@10": _safe_mean(ndcg10),
    }


def main() -> None:
    os.makedirs(WORKSPACE, exist_ok=True)
    if not os.path.isfile(OBS_CSV) or not os.path.isfile(QUERIES_JSONL):
        print(f"Preparing workspace from {DEV_JSONL} …")
        prepare_retrieve_workspace(DEV_JSONL, WORKSPACE)

    if not os.path.isdir(SPLADE_INDEX) or not os.path.isdir(BM25_INDEX) or not os.path.isdir(DENSE_INDEX):
        print(f"Index not found under {INDEX_ROOT}. Run build_eval_indexes.py first.", file=sys.stderr)
        sys.exit(1)

    retrieve_counts = (
        load_or_build_retrieve_query_counts(QU_DEV, QU_RETRIEVE_COUNTS_CACHE) if os.path.isfile(QU_DEV) else {}
    )
    qs = load_queries()

    collection = CollectionDataset(data_path=OBS_CSV)

    splade_cfg = OmegaConf.create(
        {
            "splade_model_type_or_path": "naver/splade-cocondenser-ensembledistil",
            "splade_tokenizer_type": "bert-base-uncased",
        }
    )
    splade_model = Splade(splade_cfg.splade_model_type_or_path, agg="max")
    sparse = SparseRetrieval(
        splade_config=splade_cfg,
        model=splade_model,
        collection=collection,
        dim_voc=splade_model.output_dim,
        splade_index_path=SPLADE_INDEX,
    )
    splade = SpladeRetriever(sparse, involve_model=True)

    dense_cfg = {
        "dense_model_type_or_path": "sentence-transformers/all-MiniLM-L6-v2",
        "use_sentence_transformers": True,
    }
    dense_native = DenseRetrieval(dense_config=dense_cfg, collection=collection, dense_index_path=DENSE_INDEX)
    dense = DenseFaissRetriever(dense_native)

    bm25 = BM25Retriever(collection, index_dir=BM25_INDEX)
    splade_then_dense = SpladeThenDenseRerank(splade=splade)

    pre_ks = parse_pre_ks()
    for pre_k in pre_ks:
        print(f"\n=== BM25 pre-retrieval pool: top-{pre_k} ===")

        # Precompute BM25 pools ONCE per query (reused across all models)
        pools: Dict[str, List[int]] = {}  # qid -> candidate ids
        bm25_docs_cache: Dict[str, List[RetrievedDoc]] = {}  # qid -> bm25 docs (with scores)
        pool_ids_cache: Dict[str, Set[int]] = {}  # qid -> candidate id set

        # Recreate per `pre_k` to avoid semantic-cache cross-talk between pools.
        dynamic = DynamicFusionOurs(
            splade=splade,
            dense=dense,
            bm25=bm25,
            splade_then_dense=splade_then_dense,
            retrieve_counts=retrieve_counts,
        )

        print(f"Precomputing BM25 pools (top-{pre_k}) for {len(qs)} queries…")
        itq = _TQDM(qs, desc=f"BM25@{pre_k} pool", leave=True) if _TQDM else qs
        for q in itq:
            qid = q["qid"]
            bm25_docs = bm25.retrieve(q["query"], top_k=pre_k)
            bm25_docs_cache[qid] = bm25_docs
            pool_ids = {d.id for d in bm25_docs}
            pool_ids_cache[qid] = pool_ids
            pools[qid] = [d.id for d in bm25_docs]

        def _pool_for(q: Dict[str, Any]) -> Tuple[List[RetrievedDoc], Set[int]]:
            qid = q["qid"]
            return bm25_docs_cache[qid], pool_ids_cache[qid]

        # --- Model wrappers (restricted to BM25 pool) ---
        def bm25_splade(q: Dict[str, Any]) -> List[RetrievedDoc]:
            bm25_docs, pool_ids = _pool_for(q)
            splade_docs = _filter_to_pool(splade.retrieve(q["query"], top_k=pre_k), pool_ids)
            merged = _merge_by_id(splade_docs, bm25_docs)
            fused = weighted_sum_fuse(
                merged.values(),
                weights={"splade_score": 0.7, "bm25_score": 0.3},
                normalize=True,
            )
            return fused[:RETRIEVE_K]

        def bm25_dense(q: Dict[str, Any]) -> List[RetrievedDoc]:
            bm25_docs, pool_ids = _pool_for(q)
            dense_docs = _filter_to_pool(dense.retrieve(q["query"], top_k=pre_k), pool_ids)
            merged = _merge_by_id(bm25_docs, dense_docs)
            fused = weighted_sum_fuse(
                merged.values(),
                weights={"bm25_score": 0.5, "dense_score": 0.5},
                normalize=True,
            )
            return fused[:RETRIEVE_K]

        def splade_dense_fixed(q: Dict[str, Any]) -> List[RetrievedDoc]:
            bm25_docs, pool_ids = _pool_for(q)
            # pool defined by BM25, but SPLADE/Dense are computed and then filtered to that pool
            splade_docs = _filter_to_pool(splade.retrieve(q["query"], top_k=pre_k), pool_ids)
            dense_docs = _filter_to_pool(dense.retrieve(q["query"], top_k=pre_k), pool_ids)
            merged = _merge_by_id(splade_docs, dense_docs)
            fused = weighted_sum_fuse(
                merged.values(),
                weights={"splade_score": 0.7, "dense_score": 0.3},
                normalize=True,
            )
            return fused[:RETRIEVE_K]

        def dynamic_fusion(q: Dict[str, Any]) -> List[RetrievedDoc]:
            _, pool_ids = _pool_for(q)
            # Run the original algorithm, but finally restrict outputs to the BM25 pool.
            # (DynamicFusionOurs itself is not pool-aware.)
            docs = dynamic.retrieve(
                q["query"],
                query_key=q["query_key"],
                top_k_splade=pre_k,
                top_k_dense=pre_k,
                top_k_bm25_cold=pre_k,
                top_k_final=0,  # return full fused list before pool restriction
            )
            filtered = [d for d in docs if d.id in pool_ids]
            return filtered[:RETRIEVE_K]

        # Optional auto grid search for Dynamic Fusion final-score weights.
        if AUTO_WEIGHT_GRID:
            metric = WEIGHT_GRID_METRIC
            if metric not in {
                "Hit@1",
                "Hit@5",
                "Hit@10",
                "Hit@50",
                "MRR",
                "Recall@50",
                "Precision@10",
                "NDCG@10",
            }:
                raise ValueError(
                    f"Unsupported WEIGHT_GRID_METRIC={metric}. "
                    "Use one of: Hit@1, Hit@5, Hit@10, Hit@50, MRR, Recall@50, Precision@10, NDCG@10"
                )
            grid = _make_weight_grid(WEIGHT_GRID_STEP)
            print(
                f"Auto weight grid search enabled: step={WEIGHT_GRID_STEP}, "
                f"metric={metric}, combos={len(grid)}"
            )
            print("Precomputing CE + normalized feature cache for dynamic fusion grid search…")

            # Cache per-query normalized signals once, then iterate weights cheaply.
            dynamic._ensure_cross_encoder()
            total_docs = len(getattr(bm25, "_collection", []))
            cache_rows: List[Dict[str, Any]] = []
            itg = _TQDM(qs, desc="Build dynamic-cache", leave=True) if _TQDM else qs
            for q in itg:
                qid = q["qid"]
                sem_pool = bm25_docs_cache[qid][: dynamic.top_k_semantic]
                if not sem_pool:
                    cache_rows.append({"qid": qid, "relevant_ids": q["relevant_ids"], "docs": []})
                    continue
                sem_ids = {d.id for d in sem_pool}

                dense_ranked = splade_then_dense.rerank_documents(q["query"], sem_pool)
                dense_score_map = {d.id: float(d.score) for d in dense_ranked}

                splade_fetch_k = total_docs if total_docs > 0 else pre_k
                splade_docs_all = splade.retrieve(q["query"], top_k=splade_fetch_k)
                splade_score_map = {d.id: float(d.score) for d in splade_docs_all if d.id in sem_ids}

                bm25_sem_scores = {d.id: float(d.score) for d in sem_pool}
                bm25_norm = dynamic._normalize(bm25_sem_scores)
                dense_norm = dynamic._normalize(dense_score_map)
                splade_norm = dynamic._normalize(splade_score_map)

                pairs = [(q["query"], dynamic._doc_to_text(d)) for d in sem_pool]
                ce_scores = dynamic._cross_encoder.predict(pairs)
                ce_score_map = {d.id: float(ce) for d, ce in zip(sem_pool, ce_scores)}
                ce_norm = dynamic._normalize(ce_score_map)

                docs_cache = []
                for d in sem_pool:
                    did = d.id
                    docs_cache.append(
                        {
                            "id": did,
                            "bm25_norm": bm25_norm.get(did, 0.0),
                            "dense_norm": dense_norm.get(did, 0.0),
                            "splade_norm": splade_norm.get(did, 0.0),
                            "ce_norm": ce_norm.get(did, 0.0),
                        }
                    )
                cache_rows.append({"qid": qid, "relevant_ids": q["relevant_ids"], "docs": docs_cache})

            best_score = float("-inf")
            best_weights: Tuple[float, float, float, float] = (
                dynamic.w1_bm25,
                dynamic.w2_dense,
                dynamic.w3_splade,
                dynamic.w4_ce,
            )

            for a, b, c, dlt in grid:
                metric_row = _evaluate_dynamic_weight_from_cache(cache_rows, (a, b, c, dlt))
                score = float(metric_row.get(metric, 0.0))
                if score > best_score:
                    best_score = score
                    best_weights = (a, b, c, dlt)

            dynamic = DynamicFusionOurs(
                splade=splade,
                dense=dense,
                bm25=bm25,
                splade_then_dense=splade_then_dense,
                retrieve_counts=retrieve_counts,
                w1_bm25=best_weights[0],
                w2_dense=best_weights[1],
                w3_splade=best_weights[2],
                w4_ce=best_weights[3],
            )
            print(
                "Best dynamic weights found: "
                f"alpha={best_weights[0]:.2f}, beta={best_weights[1]:.2f}, "
                f"gamma={best_weights[2]:.2f}, delta={best_weights[3]:.2f}, "
                f"{metric}={best_score:.6f}"
            )

        def splade_pure(q: Dict[str, Any]) -> List[RetrievedDoc]:
            # Pure SPLADE retrieval with ReQAP's SPLADE model (no fusion).
            return splade.retrieve(q["query"], top_k=RETRIEVE_K)

        specs: List[Tuple[str, Callable[[Dict[str, Any]], List[RetrievedDoc]]]] = [
            ("SPLADE (Pure)", splade_pure),
            ("BM25 + SPLADE (BM25-pool)", bm25_splade),
            ("BM25 + Dense-Trieve (BM25-pool)", bm25_dense),
            ("SPLADE + Dense (Fixed, BM25-pool)", splade_dense_fixed),
            (
                "Dynamic Fusion (Ours, BM25-pool)"
                f"[a={dynamic.w1_bm25:.2f},b={dynamic.w2_dense:.2f},c={dynamic.w3_splade:.2f},d={dynamic.w4_ce:.2f}]",
                dynamic_fusion,
            ),
        ]

        rows: List[Dict[str, Any]] = []
        for name, fn in specs:
            rows.append(evaluate_model(name, fn))

        # Write outputs
        out_prefix = os.path.join(WORKSPACE, f"preretrieve_bm25_top{pre_k}")
        cand_path = out_prefix + "_candidates.jsonl"
        results_json = out_prefix + "_results_models.json"
        results_csv = out_prefix + "_results_models.csv"

        _write_candidate_jsonl(cand_path, qs, pools)
        with open(results_json, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)

        # CSV (same columns as eval_three_pipelines)
        cols = [
            "Model",
            "Hit@1",
            "Hit@5",
            "Hit@10",
            "Hit@50",
            "MRR",
            "Recall@50",
            "Precision@10",
            "NDCG@10",
            "Avg. Latency",
            "GPU Memory",
            "num_queries",
            "total_time_s",
        ]
        import csv

        with open(results_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in cols})

        print(f"Wrote candidates: {cand_path}")
        print(f"Wrote results   : {results_json}")
        print(f"Wrote results   : {results_csv}")


if __name__ == "__main__":
    main()

