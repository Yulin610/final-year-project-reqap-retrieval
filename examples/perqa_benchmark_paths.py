"""
Official PerQA benchmark layout under ReQAP-main (sibling of reqap-retrieval-modular).

Pre-retrieval corpus (BM25 / SPLADE / dense index) for one user + split is exactly:

    {benchmark_root}/{split}/{split}_persona_{id}/{split}_persona_{id}_obs.csv

Questions for that same setting live beside it:

    .../questions.json

Default benchmark_root tries (in order):
  1) ReQAP-main/data/data/benchmarks/perqa   (current repo layout)
  2) ReQAP-main/data/benchmarks/perqa        (if you symlink to match config yml)
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple


def examples_dir() -> Path:
    return Path(__file__).resolve().parent


def workspace_root() -> Path:
    """reqap-retrieval-modular 的父目录（常见为名为 ReQAP-main 的文件夹）。"""
    return examples_dir().parents[1]


def reqap_main_root() -> Path:
    """
    含有 `reqap/` 包与 `data/` 的 ReQAP 工程根目录。

    从本文件向上找：任一祖先下的 ``./reqap`` 或 ``ReQAP-main/reqap``。

    必须先匹配 ``anc/reqap``，再匹配 ``anc/ReQAP-main/reqap``；否则在
    ``.../ReQAP-main/ReQAP-main/reqap`` 这种嵌套目录存在时，会错误地把内层当成工程根。
    """
    here = examples_dir().resolve()
    for anc in here.parents:
        if (anc / "reqap").is_dir():
            return anc
        inner = anc / "ReQAP-main"
        if (inner / "reqap").is_dir():
            return inner
    return here.parents[2] / "ReQAP-main"


def perqa_benchmark_root() -> Path:
    primary = reqap_main_root() / "data" / "data" / "benchmarks" / "perqa"
    if primary.is_dir():
        return primary
    alt = reqap_main_root() / "data" / "benchmarks" / "perqa"
    if alt.is_dir():
        return alt
    return primary


def persona_folder_name(split: str, persona_id: int) -> str:
    return f"{split}_persona_{persona_id}"


def perqa_obs_csv(split: str, persona_id: int) -> Path:
    """Path to the obs CSV that should back BM25/SPLADE/dense for this split + persona."""
    name = persona_folder_name(split, persona_id)
    return perqa_benchmark_root() / split / name / f"{name}_obs.csv"


def perqa_questions_json(split: str, persona_id: int) -> Path:
    name = persona_folder_name(split, persona_id)
    return perqa_benchmark_root() / split / name / "questions.json"


def perqa_splade_index_dir(split: str, persona_id: int) -> Path:
    """SPLADE index folder matching rag.py naming: {split}_persona_{id}.splade_index"""
    name = persona_folder_name(split, persona_id)
    return reqap_main_root() / "data" / "data" / "splade_indices" / "perqa" / f"{name}.splade_index"


def perqa_dense_index_dir(split: str, persona_id: int) -> Path:
    """Default dense FAISS output dir (create with run_dense_index or build_eval_indexes)."""
    name = persona_folder_name(split, persona_id)
    return reqap_main_root() / "data" / "data" / "dense_indices" / "perqa" / f"{name}.dense_index"


def perqa_bm25_index_dir(split: str, persona_id: int) -> Path:
    """Default BM25 index directory for this split/persona."""
    name = persona_folder_name(split, persona_id)
    return reqap_main_root() / "data" / "data" / "bm25_indices" / "perqa" / name


def list_perqa_obs_datasets() -> List[Tuple[str, int, Path]]:
    """
    Scan benchmark tree for *_{split}_persona_{id}_obs.csv files.
    Returns list of (split, persona_id, path).
    """
    root = perqa_benchmark_root()
    if not root.is_dir():
        return []
    out: List[Tuple[str, int, Path]] = []
    for split_dir in sorted(root.iterdir()):
        if not split_dir.is_dir():
            continue
        split = split_dir.name
        if split not in ("train", "dev", "test"):
            continue
        for p in split_dir.iterdir():
            if not p.is_dir():
                continue
            prefix = f"{split}_persona_"
            if not p.name.startswith(prefix):
                continue
            suffix = p.name[len(prefix) :]
            if not suffix.isdigit():
                continue
            pid = int(suffix)
            obs = p / f"{p.name}_obs.csv"
            if obs.is_file():
                out.append((split, pid, obs))
    return sorted(out, key=lambda t: (t[0], t[1]))


if __name__ == "__main__":
    root = perqa_benchmark_root()
    print(f"perqa_benchmark_root: {root}")
    rows = list_perqa_obs_datasets()
    print(f"Found {len(rows)} obs corpora (split, persona_id, path):")
    for split, pid, p in rows:
        print(f"  {split:5}  persona_{pid}  {p}")
