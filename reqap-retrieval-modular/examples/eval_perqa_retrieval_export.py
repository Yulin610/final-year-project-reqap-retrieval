"""
对 export_queries_jsonl.py 产出的 queries.jsonl 做多路检索评测：Hit@k、MRR、Recall@50、P@10、NDCG@10 等（与 eval_three_pipelines 一致）。

运行前需已构建对应 split/persona 的 BM25 / SPLADE / dense 索引（见 perqa_benchmark_paths）。

环境变量：EVAL_MAX_QUERIES、RETRIEVE_K（默认 100）、PERQA_OBS_CSV / PERQA_SPLADE_INDEX 等可覆盖默认路径。

用法:
  python eval_perqa_retrieval_export.py --split dev --persona-id 0 --queries-jsonl queries.jsonl
  python eval_perqa_retrieval_export.py --queries-jsonl q.jsonl --obs-csv path/to_obs.csv --splade-index ... --dense-index ... --bm25-index ...
"""
from __future__ import annotations

import argparse
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

_MODULAR = _EXAMPLES.parent
if str(_MODULAR) not in sys.path:
    sys.path.insert(0, str(_MODULAR))

try:
    from omegaconf import OmegaConf
except Exception:  # optional dependency
    OmegaConf = None

from perqa_benchmark_paths import (
    perqa_bm25_index_dir,
    perqa_dense_index_dir,
    perqa_obs_csv,
    perqa_splade_index_dir,
    persona_folder_name,
    reqap_main_root,
)
from qu_retrieve_counts import load_or_build_retrieve_query_counts

_ROOT_REQAP = reqap_main_root()
if str(_ROOT_REQAP) not in sys.path:
    sys.path.insert(0, str(_ROOT_REQAP))

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


def load_queries_from_path(path: str) -> List[Dict[str, Any]]:
    qs: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
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
                    "relevant_ids": set(int(x) for x in row["relevant_ids"]),
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


def evaluate_model(
    model_label: str,
    retrieve_fn: Callable[[Dict[str, Any]], List[Any]],
    qs: List[Dict[str, Any]],
    retrieve_k: int,
) -> Dict[str, Any]:
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
        mrrs.append(mrr(ids, rel, retrieve_k))
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
    lines: List[str] = []
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Eval retrieval on exported PerQA queries.jsonl")
    ap.add_argument("--queries-jsonl", required=True, help="Path from export_queries_jsonl.py")
    ap.add_argument("--split", default="dev", help="Used with --persona-id for default index paths")
    ap.add_argument("--persona-id", type=int, default=0)
    ap.add_argument("--obs-csv", default="", help="Override corpus CSV (default: benchmark obs for split/persona)")
    ap.add_argument("--splade-index", default="", help="Override SPLADE index dir")
    ap.add_argument("--dense-index", default="", help="Override dense FAISS index dir")
    ap.add_argument("--bm25-index", default="", help="Override BM25 index dir")
    ap.add_argument(
        "--dense-model",
        default="",
        help="Dense encoder model type/path (overrides PERQA_DENSE_MODEL_TYPE_OR_PATH).",
    )
    ap.add_argument(
        "--qu-jsonl",
        default="",
        help="QU jsonl for Dynamic Fusion retrieve counts (default: reqap_sft qu_result for split/persona)",
    )
    ap.add_argument(
        "--qu-retrieve-counts-cache",
        default="",
        help="Cache path for retrieve counts JSON (default: beside queries-jsonl)",
    )
    ap.add_argument("--retrieve-k", type=int, default=None, help="Final top-k (default: env RETRIEVE_K or 100)")
    ap.add_argument("--out-dir", default="", help="Directory for results JSON/CSV/MD (default: beside queries-jsonl)")
    ap.add_argument(
        "--grid-search-dynamic",
        action="store_true",
        help="Grid-search Dynamic Fusion weights (no CE) and write dynamic_grid_search.{json,csv}",
    )
    ap.add_argument(
        "--grid-metric",
        default="NDCG@10",
        choices=["NDCG@10", "MRR", "Hit@10", "Hit@50", "Recall@50", "Precision@10"],
        help="Metric to maximize when selecting best weights (default: NDCG@10)",
    )
    args = ap.parse_args()

    queries_path = Path(args.queries_jsonl)
    if not queries_path.is_file():
        raise SystemExit(f"Missing queries: {queries_path}")

    obs_csv = os.environ.get("PERQA_OBS_CSV") or args.obs_csv or str(perqa_obs_csv(args.split, args.persona_id))
    splade_index = os.environ.get("PERQA_SPLADE_INDEX") or args.splade_index or str(
        perqa_splade_index_dir(args.split, args.persona_id)
    )
    dense_index = os.environ.get("PERQA_DENSE_INDEX") or args.dense_index or str(
        perqa_dense_index_dir(args.split, args.persona_id)
    )
    bm25_index = os.environ.get("PERQA_BM25_INDEX") or args.bm25_index or str(perqa_bm25_index_dir(args.split, args.persona_id))

    if not os.path.isfile(obs_csv):
        raise SystemExit(f"Missing OBS CSV: {obs_csv}")
    for label, p in [("splade_index", splade_index), ("dense_index", dense_index), ("bm25_index", bm25_index)]:
        if not os.path.isdir(p):
            raise SystemExit(f"Missing index directory ({label}): {p}")

    dense_model = os.environ.get("PERQA_DENSE_MODEL_TYPE_OR_PATH") or args.dense_model or "sentence-transformers/all-MiniLM-L6-v2"

    repo = reqap_main_root()
    folder = persona_folder_name(args.split, args.persona_id)
    qu_default = repo / "data" / "data" / "results" / "perqa" / "reqap_sft" / folder / "qu_result.jsonl"
    qu_path = Path(args.qu_jsonl) if args.qu_jsonl else qu_default
    cache_default = queries_path.parent / f"qu_retrieve_counts_{folder}.json"
    qu_cache = Path(args.qu_retrieve_counts_cache) if args.qu_retrieve_counts_cache else cache_default

    retrieve_k = args.retrieve_k
    if retrieve_k is None:
        retrieve_k = int(os.environ.get("RETRIEVE_K", "100"))

    qs = load_queries_from_path(str(queries_path))
    if not qs:
        raise SystemExit("No queries loaded.")

    collection = CollectionDataset(data_path=obs_csv)

    retrieve_counts: Dict[str, int] = {}
    if qu_path.is_file():
        retrieve_counts = load_or_build_retrieve_query_counts(str(qu_path), str(qu_cache))
        print(f"QU RETRIEVE query types: {len(retrieve_counts)} (source={qu_path}, cache={qu_cache})")
    else:
        print(f"No QU jsonl at {qu_path}; Dynamic Fusion uses empty retrieve_counts.")

    if OmegaConf is not None:
        splade_cfg = OmegaConf.create(
            {
                "splade_model_type_or_path": "naver/splade-cocondenser-ensembledistil",
                "splade_tokenizer_type": "bert-base-uncased",
            }
        )
    else:
        class _Cfg:
            splade_model_type_or_path = "naver/splade-cocondenser-ensembledistil"
            splade_tokenizer_type = "bert-base-uncased"

        splade_cfg = _Cfg()
    splade_model = Splade(splade_cfg.splade_model_type_or_path, agg="max")
    sparse = SparseRetrieval(
        splade_config=splade_cfg,
        model=splade_model,
        collection=collection,
        dim_voc=splade_model.output_dim,
        splade_index_path=splade_index,
    )
    splade = SpladeRetriever(sparse, involve_model=True)

    dense_cfg = {
        "dense_model_type_or_path": dense_model,
        "use_sentence_transformers": True,
    }
    dense_native = DenseRetrieval(dense_config=dense_cfg, collection=collection, dense_index_path=dense_index)
    dense = DenseFaissRetriever(dense_native)

    bm25 = BM25Retriever(collection, index_dir=bm25_index)

    tk = retrieve_k
    pool = max(500, tk)

    p_bm25_dense_par = BM25DenseParallelFusion(bm25=bm25, dense=dense, weight_bm25=0.5, weight_dense=0.5)
    p_splade_bm25 = SpladeBM25Fusion(
        splade=splade, collection=collection, bm25_index_dir=bm25_index, weight_splade=0.7, weight_bm25=0.3
    )
    p_splade_dense_fix = SpladeDenseParallelFusion(splade=splade, dense=dense, weight_splade=0.7, weight_dense=0.3)
    splade_then_dense = SpladeThenDenseRerank(splade=splade, dense_model_name=dense_model)
    dynamic = DynamicFusionOurs(
        splade=splade,
        dense=dense,
        bm25=bm25,
        splade_then_dense=splade_then_dense,
        retrieve_counts=retrieve_counts,
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

    out_dir = Path(args.out_dir) if args.out_dir else queries_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = queries_path.stem
    results_json_path = out_dir / f"{stem}_results_models.json"
    results_csv_path = out_dir / f"{stem}_results_models.csv"
    results_md_path = out_dir / f"{stem}_results_table.md"

    results: List[Dict[str, Any]] = []

    # Optionally grid-search DynamicFusion weights (no CE).
    if args.grid_search_dynamic:
        grid_rows: List[Dict[str, Any]] = []
        w_dense_list = [0.4, 0.5, 0.6]
        w_splade_list = [0.4, 0.5, 0.6]
        w_bm25_list = [0.0, 0.1]

        best_row: Optional[Dict[str, Any]] = None
        best_key: Optional[tuple] = None

        for w_dense in w_dense_list:
            for w_splade in w_splade_list:
                for w_bm25 in w_bm25_list:
                    label = f"Dynamic (grid) bm25={w_bm25} dense={w_dense} splade={w_splade}"
                    dyn = DynamicFusionOurs(
                        splade=splade,
                        dense=dense,
                        bm25=bm25,
                        splade_then_dense=splade_then_dense,
                        retrieve_counts=retrieve_counts,
                        w1_bm25=w_bm25,
                        w2_dense=w_dense,
                        w3_splade=w_splade,
                    )
                    fn = lambda q, _dyn=dyn: _dyn.retrieve(
                        q["query"],
                        query_key=q["query_key"],
                        top_k_splade=pool,
                        top_k_dense=pool,
                        top_k_bm25_cold=max(200, tk),
                        top_k_final=tk,
                    )

                    print(f"\n>>> Grid: {label}")
                    row = evaluate_model(label, fn, qs, retrieve_k)
                    row["_w_bm25"] = w_bm25
                    row["_w_dense"] = w_dense
                    row["_w_splade"] = w_splade
                    grid_rows.append(row)

                    metric = float(row.get(args.grid_metric, 0.0) or 0.0)
                    tie_mrr = float(row.get("MRR", 0.0) or 0.0)
                    tie_lat = float(row.get("Avg. Latency", 1e18) or 1e18)
                    key = (metric, tie_mrr, -tie_lat)
                    if best_key is None or key > best_key:
                        best_key = key
                        best_row = row

        grid_json_path = out_dir / f"{stem}_dynamic_grid_search.json"
        grid_csv_path = out_dir / f"{stem}_dynamic_grid_search.csv"
        with open(grid_json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "generated_at": datetime.now().isoformat(timespec="seconds"),
                    "grid_metric": args.grid_metric,
                    "weights": {"w_dense": w_dense_list, "w_splade": w_splade_list, "w_bm25": w_bm25_list},
                    "best": best_row,
                    "results": grid_rows,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        csv_cols = [
            "Model",
            "_w_bm25",
            "_w_dense",
            "_w_splade",
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
        with open(grid_csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=csv_cols)
            w.writeheader()
            for r in grid_rows:
                flat = {k: r.get(k, "") for k in csv_cols}
                if flat["GPU Memory"] is None:
                    flat["GPU Memory"] = ""
                w.writerow(flat)

        if best_row is not None:
            print(
                f"\n[Grid Best] metric={args.grid_metric}={best_row[args.grid_metric]:.4f} "
                f"MRR={best_row['MRR']:.4f} latency={best_row['Avg. Latency']:.1f}ms "
                f"(bm25={best_row['_w_bm25']}, dense={best_row['_w_dense']}, splade={best_row['_w_splade']})"
            )

    for label, fn in specs:
        print(f"\n>>> Evaluating: {label}")
        row = evaluate_model(label, fn, qs, retrieve_k)
        results.append(row)
        gm = row["GPU Memory"]
        print(
            f"    Hit@10={row['Hit@10']:.4f} Hit@50={row['Hit@50']:.4f} "
            f"NDCG@10={row['NDCG@10']:.4f} latency={row['Avg. Latency']:.1f}ms "
            f"GPU={f'{gm:.2f}GB' if gm is not None else 'N/A'}"
        )

    out_json = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "queries_jsonl": str(queries_path.resolve()),
        "obs_csv": obs_csv,
        "splade_index": splade_index,
        "dense_index": dense_index,
        "bm25_index": bm25_index,
        "split": args.split,
        "persona_id": args.persona_id,
        "retrieve_k": retrieve_k,
        "qu_jsonl": str(qu_path) if qu_path.is_file() else None,
        "qu_retrieve_counts_cache": str(qu_cache),
        "retrieve_query_vocab": len(retrieve_counts),
        "num_queries": len(qs),
        "results": results,
    }
    with open(results_json_path, "w", encoding="utf-8") as f:
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
    with open(results_csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_cols)
        w.writeheader()
        for r in results:
            flat = {k: r.get(k, "") for k in csv_cols}
            if flat["GPU Memory"] is None:
                flat["GPU Memory"] = ""
            w.writerow(flat)

    write_markdown_table(results, str(results_md_path))

    print(f"\n[Done] JSON: {results_json_path}\n       CSV: {results_csv_path}\n       MD:  {results_md_path}")


if __name__ == "__main__":
    main()
