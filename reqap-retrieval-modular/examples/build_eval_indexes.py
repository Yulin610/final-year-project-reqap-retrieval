import os, sys

# 保证可以导入原始 ReQAP 代码和本模块
sys.path.insert(0, r"C:\Users\23369\Desktop\final_work\ReQAP-main\ReQAP-main")
sys.path.insert(0, r"C:\Users\23369\Desktop\final_work\ReQAP-main\reqap-retrieval-modular")
from omegaconf import OmegaConf
from loguru import logger

from reqap.retrieval.splade.models import Splade
from reqap.retrieval.splade.index_construction import CollectionDataset, SparseIndexing
from reqap.retrieval.dense.dense_retrieval import DenseRetrieval
from reqap_modular_retrieval.retrievers.bm25 import BM25Retriever

ROOT = r"C:\Users\23369\Desktop\final_work"
OBS_CSV = os.path.join(ROOT, "eval_from_extract", "obs.csv")
INDEX_ROOT = os.path.join(ROOT, "eval_from_extract", "indices")
SPLADE_INDEX = os.path.join(INDEX_ROOT, "splade")
DENSE_INDEX = os.path.join(INDEX_ROOT, "dense")
BM25_INDEX = os.path.join(INDEX_ROOT, "bm25")

def build_splade():
    from reqap.retrieval.splade.index_construction import CollectionDataLoader
    os.makedirs(SPLADE_INDEX, exist_ok=True)
    splade_cfg = OmegaConf.create({
        "splade_model_type_or_path": "naver/splade-cocondenser-ensembledistil",
        "splade_tokenizer_type": "bert-base-uncased",
        "splade_index_path": SPLADE_INDEX,
        "splade_max_length": 256,
        "splade_index_batch_size": 16,
        "splade_verbalize_events": False,
    })
    collection = CollectionDataset(data_path=OBS_CSV)
    model = Splade(splade_cfg.splade_model_type_or_path, agg="max")
    # Windows 上 num_workers=0 时不能设置 prefetch_factor
    dl = CollectionDataLoader(
        dataset=collection,
        tokenizer_type=splade_cfg.splade_tokenizer_type,
        max_length=splade_cfg.splade_max_length,
        batch_size=splade_cfg.splade_index_batch_size,
        shuffle=False,
        num_workers=0,
    )
    indexer = SparseIndexing(model=model, splade_config=splade_cfg, dim_voc=model.output_dim)
    indexer.run(dl)

def build_dense():
    os.makedirs(DENSE_INDEX, exist_ok=True)
    collection = CollectionDataset(data_path=OBS_CSV)
    dense_cfg = {
        "dense_model_type_or_path": "sentence-transformers/all-MiniLM-L6-v2",
        "use_sentence_transformers": True,
    }
    dr = DenseRetrieval(dense_config=dense_cfg, collection=collection)
    dr.build_index(DENSE_INDEX, batch_size=32)

def build_bm25():
    os.makedirs(BM25_INDEX, exist_ok=True)
    collection = CollectionDataset(data_path=OBS_CSV)
    bm25 = BM25Retriever(collection)
    bm25.build(BM25_INDEX, show_progress=True)

if __name__ == "__main__":
    os.makedirs(INDEX_ROOT, exist_ok=True)
    build_splade()
    build_dense()
    build_bm25()