import csv
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

# 保证可以导入原始 ReQAP 代码和本模块
_EXAMPLES = Path(__file__).resolve().parent
sys.path.insert(0, str(_EXAMPLES))
sys.path.insert(0, r"C:\Users\23369\Desktop\final_work\ReQAP-main\ReQAP-main")
sys.path.insert(0, r"C:\Users\23369\Desktop\final_work\ReQAP-main\reqap-retrieval-modular")

from prepare_retrieve_dev_eval import prepare as prepare_retrieve_workspace
from omegaconf import OmegaConf
from loguru import logger

from reqap.retrieval.splade.models import Splade
from reqap.retrieval.splade.index_construction import CollectionDataset, SparseIndexing
from reqap.retrieval.dense.dense_retrieval import DenseRetrieval
from reqap_modular_retrieval.retrievers.bm25 import BM25Retriever

# 默认与 retrieve 评测工作区一致；缺 obs/queries 时会从 dev_data.jsonl 自动生成（与 eval_three_pipelines 一致）
WORKSPACE = os.environ.get(
    "RETRIEVE_EVAL_WORKSPACE",
    r"C:\Users\23369\Desktop\final_work\data\retrieve\eval_workspace",
)
DEV_JSONL = os.environ.get(
    "RETRIEVE_DEV_JSONL",
    r"C:\Users\23369\Desktop\final_work\data\retrieve\dev_data.jsonl",
)
OBS_CSV = os.path.join(WORKSPACE, "obs.csv")
# Point directly at official benchmark obs (must match queries.jsonl persona/split):
#   set PERQA_OBS_CSV=.../dev_persona_0/dev_persona_0_obs.csv
if os.environ.get("PERQA_OBS_CSV"):
    OBS_CSV = os.environ["PERQA_OBS_CSV"]
QUERIES_JSONL = os.path.join(WORKSPACE, "queries.jsonl")
INDEX_ROOT = os.path.join(WORKSPACE, "indices")
SPLADE_INDEX = os.path.join(INDEX_ROOT, "splade")
DENSE_INDEX = os.path.join(INDEX_ROOT, "dense")
BM25_INDEX = os.path.join(INDEX_ROOT, "bm25")
LOCAL_DENSE_MODEL = (
    _EXAMPLES.parent.parent / "ReQAP-main" / "data" / "data" / "models" / "perqa" / "dense_adapted_phase1"
)
# Default local SPLADE checkpoint (Windows); used when env is unset or invalid (e.g. ".")
_DEFAULT_LOCAL_SPLADE = Path(r"C:\Users\23369\Desktop\final_work\splade_adapted_hn_phase1")


def _is_valid_local_hf_model_dir(p: Path) -> bool:
    return p.is_dir() and (p / "config.json").is_file()


def _resolve_splade_model_from_env() -> str:
    """
    PERQA_SPLADE_MODEL_TYPE_OR_PATH may be a HF hub id or a local folder.
    Reject ".", "..", and folders without config.json (common mistake: env set to ".").
    """
    raw = (os.environ.get("PERQA_SPLADE_MODEL_TYPE_OR_PATH") or "").strip()
    if raw in ("", ".", ".."):
        return ""
    p = Path(raw).expanduser()
    if p.is_dir():
        if _is_valid_local_hf_model_dir(p):
            return str(p)
        logger.warning(
            f"Ignoring PERQA_SPLADE_MODEL_TYPE_OR_PATH={raw!r}: not a HF model folder (missing config.json)."
        )
        return ""
    # Likely hub id (e.g. naver/splade-...); not a local path
    return raw


LOCAL_SPLADE_MODEL = _DEFAULT_LOCAL_SPLADE
_env_splade = (os.environ.get("PERQA_SPLADE_MODEL_TYPE_OR_PATH") or "").strip()
if _env_splade not in ("", ".", ".."):
    _p = Path(_env_splade).expanduser()
    if _is_valid_local_hf_model_dir(_p):
        LOCAL_SPLADE_MODEL = _p


def resolved_splade_model_type_or_path(*, log_choice: bool = False) -> str:
    """
    Final SPLADE checkpoint id (Hub or local dir). Must match between build_eval_indexes and eval_three_pipelines.
    """
    splade_model = _resolve_splade_model_from_env()
    if not splade_model and LOCAL_SPLADE_MODEL.is_dir() and _is_valid_local_hf_model_dir(LOCAL_SPLADE_MODEL):
        splade_model = str(LOCAL_SPLADE_MODEL)
        if log_choice:
            logger.info(f"Using local SPLADE model: {splade_model}")
    if not splade_model:
        splade_model = "naver/splade-cocondenser-ensembledistil"
    return splade_model


def build_splade():
    from reqap.retrieval.splade.index_construction import CollectionDataLoader

    os.makedirs(SPLADE_INDEX, exist_ok=True)
    splade_model = resolved_splade_model_type_or_path(log_choice=True)
    splade_cfg = OmegaConf.create(
        {
            "splade_model_type_or_path": splade_model,
            "splade_tokenizer_type": "bert-base-uncased",
            "splade_index_path": SPLADE_INDEX,
            "splade_max_length": 256,
            "splade_index_batch_size": 16,
            # Important: avoid indexing raw JSON strings; prefer text or key:value verbalization
            "splade_verbalize_events": True,
        }
    )
    collection = CollectionDataset(data_path=OBS_CSV)
    model = Splade(splade_cfg.splade_model_type_or_path, agg="max")
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
    dense_model = os.environ.get("PERQA_DENSE_MODEL_TYPE_OR_PATH")
    if not dense_model and LOCAL_DENSE_MODEL.is_dir():
        dense_model = str(LOCAL_DENSE_MODEL)
        logger.info(f"Using project local dense model: {dense_model}")
    if not dense_model:
        dense_model = "sentence-transformers/all-MiniLM-L6-v2"
    dense_cfg = {
        "dense_model_type_or_path": dense_model,
        "use_sentence_transformers": True,
    }
    dr = DenseRetrieval(dense_config=dense_cfg, collection=collection)
    dr.build_index(DENSE_INDEX, batch_size=32)


def build_bm25():
    os.makedirs(BM25_INDEX, exist_ok=True)
    collection = CollectionDataset(data_path=OBS_CSV)
    bm25 = BM25Retriever(collection)
    bm25.build(BM25_INDEX, show_progress=True)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _obs_stats(obs_csv: str) -> tuple[int, list[int]]:
    ids: list[int] = []
    with open(obs_csv, "r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        for row in rd:
            try:
                ids.append(int(row["id"]))
            except Exception:
                continue
    return len(ids), ids


def _count_splade_docs(index_dir: str) -> int:
    p = os.path.join(index_dir, "doc_ids.pkl")
    if not os.path.isfile(p):
        return -1
    with open(p, "rb") as f:
        return len(pickle.load(f))


def _count_dense_docs(index_dir: str) -> int:
    p = os.path.join(index_dir, "doc_ids.pkl")
    if not os.path.isfile(p):
        return -1
    with open(p, "rb") as f:
        return len(pickle.load(f))


def _count_bm25_docs(index_dir: str) -> int:
    p = os.path.join(index_dir, "doc_ids.json")
    if not os.path.isfile(p):
        return -1
    with open(p, "r", encoding="utf-8") as f:
        return len(json.load(f))


def write_workspace_fingerprint() -> None:
    """
    Persist workspace/index metadata for strict eval-time consistency checks.
    """
    n_docs, ids = _obs_stats(OBS_CSV)
    meta = {
        "workspace": WORKSPACE,
        "obs_csv": OBS_CSV,
        "obs_num_docs": n_docs,
        "obs_min_id": min(ids) if ids else None,
        "obs_max_id": max(ids) if ids else None,
        "obs_sha256": _sha256_file(OBS_CSV) if os.path.isfile(OBS_CSV) else None,
        "splade_model_type_or_path": resolved_splade_model_type_or_path(),
        "dense_model_type_or_path": os.environ.get("PERQA_DENSE_MODEL_TYPE_OR_PATH")
        or (str(LOCAL_DENSE_MODEL) if LOCAL_DENSE_MODEL.is_dir() else "sentence-transformers/all-MiniLM-L6-v2"),
        "index_doc_counts": {
            "splade": _count_splade_docs(SPLADE_INDEX),
            "dense": _count_dense_docs(DENSE_INDEX),
            "bm25": _count_bm25_docs(BM25_INDEX),
        },
    }
    out = os.path.join(INDEX_ROOT, "index_meta.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    logger.info(f"Wrote workspace fingerprint: {out}")


if __name__ == "__main__":
    os.makedirs(WORKSPACE, exist_ok=True)
    if not os.path.isfile(OBS_CSV) or not os.path.isfile(QUERIES_JSONL):
        if not os.path.isfile(DEV_JSONL):
            logger.error(
                f"Missing {OBS_CSV} and dev jsonl not found at {DEV_JSONL}. "
                f"Set RETRIEVE_DEV_JSONL or run prepare_retrieve_dev_eval.py."
            )
            sys.exit(1)
        logger.info(f"Preparing workspace from {DEV_JSONL} …")
        prepare_retrieve_workspace(DEV_JSONL, WORKSPACE)
    os.makedirs(INDEX_ROOT, exist_ok=True)
    build_splade()
    build_dense()
    build_bm25()
    write_workspace_fingerprint()
