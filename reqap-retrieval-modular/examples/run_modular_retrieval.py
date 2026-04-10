import os
import sys

# Make both sibling folders importable:
# - original package:   ./ReQAP-main
# - this package:       ./reqap-retrieval-modular
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "ReQAP-main"))
sys.path.insert(0, os.path.join(ROOT, "reqap-retrieval-modular"))

from omegaconf import OmegaConf
from loguru import logger

from reqap.retrieval.splade.models import Splade
from reqap.retrieval.splade.index_construction import CollectionDataset
from reqap.retrieval.splade.sparse_retrieval import SparseRetrieval
from reqap.retrieval.dense.dense_retrieval import DenseRetrieval

from reqap_modular_retrieval.retrievers import BM25Retriever, SpladeRetriever, DenseFaissRetriever
from reqap_modular_retrieval.pipelines import BM25ThenDenseRerank, SpladeDenseParallelFusion, SpladeBM25Fusion


def main():
    # Adjust these paths for your persona/split
    obs_events_csv_path = r".\ReQAP-main\data\perqa\dev\persona_0\persona_0_obs.csv"
    splade_index_path = r".\ReQAP-main\indices\splade\persona_0.splade_index"
    dense_index_path = r".\ReQAP-main\indices\dense\persona_0.dense_index"
    bm25_index_path = r".\ReQAP-main\indices\bm25\persona_0.bm25_index"

    # Minimal SPLADE config (tokenizer/max_length used in model)
    splade_cfg = OmegaConf.create(
        {
            "splade_model_type_or_path": "naver/splade-cocondenser-ensembledistil",
            "splade_tokenizer_type": "bert-base-uncased",
        }
    )

    collection = CollectionDataset(data_path=obs_events_csv_path)

    # Build/load SPLADE retriever
    splade_model = Splade(splade_cfg.splade_model_type_or_path, agg="max")
    sparse = SparseRetrieval(
        splade_config=splade_cfg,
        model=splade_model,
        collection=collection,
        dim_voc=splade_model.output_dim,
        splade_index_path=splade_index_path,
    )
    splade = SpladeRetriever(sparse, involve_model=True)

    # Build/load Dense retriever (FAISS)
    dense_cfg = {"dense_model_type_or_path": "sentence-transformers/all-MiniLM-L6-v2", "use_sentence_transformers": True}
    dense_native = DenseRetrieval(dense_config=dense_cfg, collection=collection, dense_index_path=dense_index_path)
    dense = DenseFaissRetriever(dense_native)

    # Build/load BM25 index if missing
    if not os.path.exists(bm25_index_path):
        os.makedirs(os.path.dirname(bm25_index_path), exist_ok=True)
        bm25_builder = BM25Retriever(collection)
        bm25_builder.build(bm25_index_path, show_progress=True)

    query = "停车位置"

    logger.info("=== BM25 -> Dense (serial) ===")
    p1 = BM25ThenDenseRerank(collection=collection, bm25_index_dir=bm25_index_path)
    r1 = p1.retrieve(query, top_k_bm25=200, top_k_final=20)
    logger.info([{"id": d.id, "score": d.score} for d in r1[:5]])

    logger.info("=== SPLADE + Dense (parallel 7:3) ===")
    p2 = SpladeDenseParallelFusion(splade=splade, dense=dense, weight_splade=0.7, weight_dense=0.3)
    r2 = p2.retrieve(query, top_k_splade=500, top_k_dense=500, top_k_final=20)
    logger.info([{"id": d.id, "score": d.score} for d in r2[:5]])

    logger.info("=== SPLADE + BM25 (fusion) ===")
    p3 = SpladeBM25Fusion(splade=splade, collection=collection, bm25_index_dir=bm25_index_path, weight_splade=0.7, weight_bm25=0.3)
    r3 = p3.retrieve(query, top_k_splade=500, top_k_bm25=500, top_k_final=20)
    logger.info([{"id": d.id, "score": d.score} for d in r3[:5]])


if __name__ == "__main__":
    main()

