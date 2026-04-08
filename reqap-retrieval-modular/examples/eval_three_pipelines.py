"""
多模型检索评测：dev_data.jsonl 语料 + QU dev_data.jsonl 统计 RETRIEVE 频率驱动 Dynamic Fusion。
查询频率仅扫描 data/qu/dev_data.jsonl，结果缓存到工作区 qu_dev_retrieve_counts.json（源文件变更后自动重扫）。
运行前：prepare_retrieve_dev_eval.py → build_eval_indexes.py（或设置 RETRIEVE_EVAL_WORKSPACE）。
可选环境变量：EVAL_MAX_QUERIES、RETRIEVE_DEV_JSONL、RETRIEVE_EVAL_WORKSPACE、QU_DEV_JSONL、QU_RETRIEVE_COUNTS_CACHE、FORCE_QU_RETRIEVE_COUNTS=1（强制重建频率缓存）。
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Dict, List, Optional, Set

_EXAMPLES = Path(__file__).resolve().parent
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

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

from reqap_modular_retrieval.retrievers import BM25Retriever, SpladeRetriever, DenseFaissRetriever
from reqap_modular_retrieval.pipelines import (
    BM25DenseParallelFusion,
    SpladeBM25Fusion,
    SpladeDenseParallelFusion,
    SpladeThenDenseRerank,
    DynamicFusionOurs,
)

# --- 路径 ---
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

RETRIEVE_K = 100

RESULTS_DIR = WORKSPACE
RESULTS_JSON_PATH = os.path.join(RESULTS_DIR, "results_models.json")
RESULTS_CSV_PATH = os.path.join(RESULTS_DIR, "results_models.csv")
RESULTS_MD_PATH = os.path.join(RESULTS_DIR, "results_table.md")
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

    def dcg(gs):
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
        return torch.cuda.max_memory_allocated() / (1024.0 ** 3)
    except Exception:
        return None


def evaluate_model(model_label: str, retrieve_fn: Callable[[Dict[str, Any]], List[Any]]) -> Dict[str, Any]:
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

    for q in qs:
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


def _fmt_cell(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        return f"{x:.4f}".rstrip("0").rstrip(".")
    return str(x)


def write_markdown_table(rows: List[Dict[str, Any]], path: str) -> None:
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
    ]
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c)
            if c == "GPU Memory":
                cells.append(f"{v:.2f} GB" if isinstance(v, float) else ("" if v is None else str(v)))
            elif c == "Avg. Latency":
                cells.append(f"{float(v):.2f} ms" if v is not None else "")
            elif c == "Model":
                cells.append(str(v))
            else:
                cells.append(_fmt_cell(v) if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    os.makedirs(WORKSPACE, exist_ok=True)
    if not os.path.isfile(OBS_CSV) or not os.path.isfile(QUERIES_JSONL):
        print(f"Preparing workspace from {DEV_JSONL} …")
        prepare_retrieve_workspace(DEV_JSONL, WORKSPACE)

    if not os.path.isdir(SPLADE_INDEX) or not os.path.isdir(BM25_INDEX):
        print(
            f"Index not found under {INDEX_ROOT}. Run build_eval_indexes.py (same WORKSPACE) first.",
            file=sys.stderr,
        )
        sys.exit(1)

    retrieve_counts = (
        load_or_build_retrieve_query_counts(QU_DEV, QU_RETRIEVE_COUNTS_CACHE)
        if os.path.isfile(QU_DEV)
        else {}
    )
    print(
        f"QU RETRIEVE query types: {len(retrieve_counts)} "
        f"(source={QU_DEV}, cache={QU_RETRIEVE_COUNTS_CACHE})"
    )

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

    tk = RETRIEVE_K
    pool = max(500, tk)

    p_bm25_dense_par = BM25DenseParallelFusion(bm25=bm25, dense=dense, weight_bm25=0.5, weight_dense=0.5)
    p_splade_bm25 = SpladeBM25Fusion(
        splade=splade, collection=collection, bm25_index_dir=BM25_INDEX, weight_splade=0.7, weight_bm25=0.3
    )
    p_splade_dense_fix = SpladeDenseParallelFusion(splade=splade, dense=dense, weight_splade=0.7, weight_dense=0.3)
    splade_then_dense = SpladeThenDenseRerank(splade=splade)
    dynamic = DynamicFusionOurs(
        splade=splade,
        dense=dense,
        bm25=bm25,
        splade_then_dense=splade_then_dense,
        retrieve_counts=retrieve_counts,
        freq_percentile_threshold=75.0,
    )

    specs: List[tuple[str, Callable[[Dict[str, Any]], List[Any]]]] = [
        ("BM25 (Pure)", lambda q: bm25.retrieve(q["query"], top_k=tk)),
        ("SPLADE (Pure)", lambda q: splade.retrieve(q["query"], top_k=tk)),
        ("Dense-Trieve (Pure)", lambda q: dense.retrieve(q["query"], top_k=tk)),
        (
            "BM25 + SPLADE",
            lambda q: p_splade_bm25.retrieve(q["query"], top_k_splade=pool, top_k_bm25=pool, top_k_final=tk),
        ),
        (
            "BM25 + Dense-Trieve",
            lambda q: p_bm25_dense_par.retrieve(q["query"], top_k_bm25=pool, top_k_dense=pool, top_k_final=tk),
        ),
        (
            "SPLADE + Dense (Fixed)",
            lambda q: p_splade_dense_fix.retrieve(
                q["query"], top_k_splade=pool, top_k_dense=pool, top_k_final=tk
            ),
        ),
        (
            "Dynamic Fusion (Ours)",
            lambda q: dynamic.retrieve(
                q["query"],
                query_key=q["query_key"],
                top_k_splade=pool,
                top_k_dense=pool,
                top_k_bm25_cold=max(200, tk),
                top_k_final=tk,
            ),
        ),
    ]

    results: List[Dict[str, Any]] = []
    for label, fn in specs:
        print(f"\n>>> Evaluating: {label}")
        row = evaluate_model(label, fn)
        results.append(row)
        gm = row["GPU Memory"]
        print(
            f"    Hit@10={row['Hit@10']:.4f} Hit@50={row['Hit@50']:.4f} "
            f"NDCG@10={row['NDCG@10']:.4f} latency={row['Avg. Latency']:.1f}ms "
            f"GPU={f'{gm:.2f}GB' if gm is not None else 'N/A'}"
        )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_json = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "workspace": WORKSPACE,
        "dev_jsonl": DEV_JSONL,
        "retrieve_k": RETRIEVE_K,
        "qu_dev_jsonl": QU_DEV,
        "qu_retrieve_counts_cache": QU_RETRIEVE_COUNTS_CACHE,
        "retrieve_query_vocab": len(retrieve_counts),
        "results": results,
    }
    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)

    csv_cols = [
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
    with open(RESULTS_CSV_PATH, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_cols)
        w.writeheader()
        for r in results:
            flat = {k: r.get(k, "") for k in csv_cols}
            if flat["GPU Memory"] is None:
                flat["GPU Memory"] = ""
            w.writerow(flat)

    write_markdown_table(results, RESULTS_MD_PATH)

    print(f"\n[Done] JSON: {RESULTS_JSON_PATH}\n       CSV: {RESULTS_CSV_PATH}\n       MD:  {RESULTS_MD_PATH}")


if __name__ == "__main__":
    main()
