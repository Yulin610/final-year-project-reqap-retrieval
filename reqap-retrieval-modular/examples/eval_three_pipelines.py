import os, json, time, math, sys, csv
from datetime import datetime

# 保证可以导入原始 ReQAP 代码和本模块
sys.path.insert(0, r"C:\Users\23369\Desktop\final_work\ReQAP-main\ReQAP-main")
sys.path.insert(0, r"C:\Users\23369\Desktop\final_work\ReQAP-main\reqap-retrieval-modular")
from statistics import mean
from typing import List, Set, Optional, Dict, Any

from omegaconf import OmegaConf

from reqap.retrieval.splade.models import Splade
from reqap.retrieval.splade.index_construction import CollectionDataset
from reqap.retrieval.splade.sparse_retrieval import SparseRetrieval
from reqap.retrieval.dense.dense_retrieval import DenseRetrieval

from reqap_modular_retrieval.retrievers import BM25Retriever, SpladeRetriever, DenseFaissRetriever
from reqap_modular_retrieval.pipelines import BM25ThenDenseRerank, SpladeDenseParallelFusion, SpladeBM25Fusion

ROOT = r"C:\Users\23369\Desktop\final_work"
OBS_CSV = os.path.join(ROOT, "eval_from_extract", "obs.csv")
QUERIES_JSONL = os.path.join(ROOT, "eval_from_extract", "queries.jsonl")
INDEX_ROOT = os.path.join(ROOT, "eval_from_extract", "indices")
SPLADE_INDEX = os.path.join(INDEX_ROOT, "splade")
DENSE_INDEX = os.path.join(INDEX_ROOT, "dense")
BM25_INDEX = os.path.join(INDEX_ROOT, "bm25")

# 检索需截断到此长度，以计算 Hit@100 / Recall@50 等
RETRIEVE_K = 100

RESULTS_DIR = os.path.join(ROOT, "eval_from_extract")
RESULTS_JSON_PATH = os.path.join(RESULTS_DIR, "results.json")
RESULTS_CSV_PATH = os.path.join(RESULTS_DIR, "results.csv")


def load_queries():
    qs = []
    with open(QUERIES_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            qs.append({
                "qid": row["qid"],
                "query": row["query"],
                "relevant_ids": set(row["relevant_ids"]),
            })
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


def _cuda_peak_gb_this_process() -> Optional[float]:
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        return torch.cuda.max_memory_allocated() / (1024.0 ** 3)
    except Exception:
        return None


def evaluate(name: str, retrieve_fn) -> Dict[str, Any]:
    qs = load_queries()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass

    hits1, hits5, hits10, hits100 = [], [], [], []
    mrrs, r50, p10, ndcg10 = [], [], [], []
    times = []

    for q in qs:
        t0 = time.time()
        docs = retrieve_fn(q["query"])
        t1 = time.time()
        ids = [d.id for d in docs]
        rel = q["relevant_ids"]

        hits1.append(hit_at_k(ids, rel, 1))
        hits5.append(hit_at_k(ids, rel, 5))
        hits10.append(hit_at_k(ids, rel, 10))
        hits100.append(hit_at_k(ids, rel, 100))
        mrrs.append(mrr(ids, rel, RETRIEVE_K))
        r50.append(recall_at_k(ids, rel, 50))
        p10.append(precision_at_k(ids, rel, 10))
        ndcg10.append(ndcg_at_k(ids, rel, 10))
        times.append(t1 - t0)

    peak_gb = _cuda_peak_gb_this_process()

    row = {
        "method": name,
        "num_queries": len(qs),
        "retrieve_list_len": RETRIEVE_K,
        "Hit@1": _safe_mean(hits1),
        "Hit@5": _safe_mean(hits5),
        "Hit@10": _safe_mean(hits10),
        "Hit@100": _safe_mean(hits100),
        "MRR": _safe_mean(mrrs),
        "Recall@50": _safe_mean(r50),
        "Precision@10": _safe_mean(p10),
        "NDCG@10": _safe_mean(ndcg10),
        "avg_latency_ms": _safe_mean([t * 1000.0 for t in times]),
        "total_time_s": float(sum(times)),
        "gpu_peak_memory_gb": peak_gb,
    }
    return row


def main():
    collection = CollectionDataset(data_path=OBS_CSV)

    # SPLADE
    splade_cfg = OmegaConf.create({
        "splade_model_type_or_path": "naver/splade-cocondenser-ensembledistil",
        "splade_tokenizer_type": "bert-base-uncased",
    })
    splade_model = Splade(splade_cfg.splade_model_type_or_path, agg="max")
    sparse = SparseRetrieval(
        splade_config=splade_cfg,
        model=splade_model,
        collection=collection,
        dim_voc=splade_model.output_dim,
        splade_index_path=SPLADE_INDEX,
    )
    splade = SpladeRetriever(sparse, involve_model=True)

    # Dense (FAISS)
    dense_cfg = {
        "dense_model_type_or_path": "sentence-transformers/all-MiniLM-L6-v2",
        "use_sentence_transformers": True,
    }
    dense_native = DenseRetrieval(dense_config=dense_cfg, collection=collection, dense_index_path=DENSE_INDEX)
    dense = DenseFaissRetriever(dense_native)

    # BM25
    bm25 = BM25Retriever(collection, index_dir=BM25_INDEX)

    # 三种 pipeline（最终列表长度需 >= 100 以支撑 Hit@100 / Recall@50）
    p_bm25_dense = BM25ThenDenseRerank(collection=collection, bm25_index_dir=BM25_INDEX)
    p_splade_dense = SpladeDenseParallelFusion(splade=splade, dense=dense, weight_splade=0.7, weight_dense=0.3)
    p_splade_bm25 = SpladeBM25Fusion(splade=splade, collection=collection, bm25_index_dir=BM25_INDEX,
                                     weight_splade=0.7, weight_bm25=0.3)

    tk = RETRIEVE_K
    results = []
    results.append(evaluate("BM25 -> Dense 串行", lambda q: p_bm25_dense.retrieve(q, top_k_bm25=max(200, tk), top_k_final=tk)))
    results.append(evaluate("SPLADE + Dense 并行 7:3", lambda q: p_splade_dense.retrieve(q, top_k_splade=500, top_k_dense=500, top_k_final=tk)))
    results.append(evaluate("SPLADE + BM25 融合", lambda q: p_splade_bm25.retrieve(q, top_k_splade=500, top_k_bm25=500, top_k_final=tk)))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "retrieve_list_len": RETRIEVE_K,
        "results": results,
    }
    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    fieldnames = [
        "method", "num_queries", "retrieve_list_len",
        "Hit@1", "Hit@5", "Hit@10", "Hit@100",
        "MRR", "Recall@50", "Precision@10", "NDCG@10",
        "avg_latency_ms", "total_time_s", "gpu_peak_memory_gb",
    ]
    with open(RESULTS_CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            flat = {k: row[k] for k in fieldnames}
            if flat["gpu_peak_memory_gb"] is None:
                flat["gpu_peak_memory_gb"] = ""
            writer.writerow(flat)

    print("\n[Eval finished] Result files:")
    print(f"- {RESULTS_JSON_PATH}")
    print(f"- {RESULTS_CSV_PATH}")
    print("\nSummary:")
    for r in results:
        vram = r["gpu_peak_memory_gb"]
        vram_s = f"{vram:.2f}GB" if vram is not None else "N/A (no CUDA)"
        print(
            f"{r['method']}: Hit@1={r['Hit@1']:.3f} Hit@10={r['Hit@10']:.3f} Hit@100={r['Hit@100']:.3f} "
            f"MRR={r['MRR']:.3f} Recall@50={r['Recall@50']:.3f} Precision@10={r['Precision@10']:.3f} "
            f"NDCG@10={r['NDCG@10']:.3f} avg_latency_ms={r['avg_latency_ms']:.1f} peakVRAM={vram_s}"
        )


if __name__ == "__main__":
    main()
