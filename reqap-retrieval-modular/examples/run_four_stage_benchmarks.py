from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


EXAMPLES = Path(__file__).resolve().parent


def _run(cmd: List[str], *, env: Dict[str, str], cwd: Path, title: str) -> None:
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")
    print("+", " ".join(cmd))
    subprocess.run(cmd, env=env, cwd=str(cwd), check=True)


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _prepare_stage_perqa_workspace(stage_dir: Path, source_perqa_workspace: Path, queries_src: Path) -> Tuple[Path, Path]:
    ws = stage_dir / "perqa_workspace"
    ws.mkdir(parents=True, exist_ok=True)
    obs_src = source_perqa_workspace / "obs.csv"
    if not obs_src.is_file():
        raise FileNotFoundError(f"Missing source PerQA obs.csv: {obs_src}")
    shutil.copy2(obs_src, ws / "obs.csv")
    shutil.copy2(queries_src, ws / "queries.jsonl")
    q_eval = stage_dir / queries_src.name
    shutil.copy2(queries_src, q_eval)
    return ws, q_eval


def _load_results(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _dynamic_row(results_json: Dict[str, Any]) -> Dict[str, Any]:
    for r in results_json.get("results", []):
        if r.get("Model") == "Dynamic Fusion (Ours)":
            return r
    return {}


def _write_summary(out_root: Path, stage_meta: List[Dict[str, Any]]) -> None:
    summary = {"stages": stage_meta}
    json_path = out_root / "four_stage_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    cols = [
        "stage",
        "tag",
        "task",
        "model",
        "Hit@1",
        "MRR",
        "Recall@10",
        "Recall@50",
        "NDCG@10",
        "Avg. Latency",
    ]
    csv_path = out_root / "four_stage_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for st in stage_meta:
            for task_key in ("taskA", "taskB", "taskC"):
                row = st.get(task_key, {})
                if not row:
                    continue
                out = {k: "" for k in cols}
                out["stage"] = st["stage"]
                out["tag"] = st["tag"]
                out["task"] = task_key
                out["model"] = "Dynamic Fusion (Ours)"
                for m in ("Hit@1", "MRR", "Recall@10", "Recall@50", "NDCG@10", "Avg. Latency"):
                    out[m] = row.get(m, "")
                w.writerow(out)

    md_path = out_root / "four_stage_summary.md"
    lines = [
        "# Four-stage A/B/C benchmark summary",
        "",
        "| Stage | Tag | Task | Hit@1 | MRR | Recall@10 | Recall@50 | NDCG@10 | Avg. Latency |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for st in stage_meta:
        for task_key in ("taskA", "taskB", "taskC"):
            row = st.get(task_key, {})
            if not row:
                continue
            lines.append(
                f"| {st['stage']} | {st['tag']} | {task_key} | "
                f"{row.get('Hit@1', ''):.4f} | {row.get('MRR', ''):.4f} | "
                f"{row.get('Recall@10', ''):.4f} | {row.get('Recall@50', ''):.4f} | "
                f"{row.get('NDCG@10', ''):.4f} | {row.get('Avg. Latency', ''):.2f} |"
            )
    lines += ["", "## Stage configs", ""]
    for st in stage_meta:
        lines.append(f"- `{st['stage']}` {st['tag']}:")
        lines.append(f"  - splade_model: `{st.get('splade_model', '')}`")
        lines.append(f"  - dense_model: `{st.get('dense_model', '')}`")
        lines.append(f"  - dynamic_fixed_weights: `{st.get('dynamic_fixed_weights', {})}`")
        lines.append(f"  - learned_router_model: `{st.get('learned_router_model', '')}`")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _stage_run_all(
    py: str,
    *,
    tag: str,
    out_root: Path,
    base_env: Dict[str, str],
    splade_model: str,
    dense_model: str,
    dev_jsonl: str,
    qu_dev_jsonl: str,
    source_perqa_workspace: Path,
    queries_perqa_src: Path,
    perqa_obs_csv: str,
    taskC_extra_env: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    stage_dir = out_root / tag
    stage_dir.mkdir(parents=True, exist_ok=True)
    workspace_exact = stage_dir / "eval_workspace_exact"
    workspace_short = stage_dir / "eval_workspace_short"
    perqa_ws, perqa_q = _prepare_stage_perqa_workspace(stage_dir, source_perqa_workspace, queries_perqa_src)

    # IMPORTANT: keep Task A/B identical across stages; apply fixed/learned router ONLY to Task C.
    env_ab = dict(base_env)
    env_ab.update(
        {
            "PERQA_SPLADE_MODEL_TYPE_OR_PATH": splade_model,
            "PERQA_DENSE_MODEL_TYPE_OR_PATH": dense_model,
            "RETRIEVE_DEV_JSONL": dev_jsonl,
            "QU_DEV_JSONL": qu_dev_jsonl,
        }
    )

    # Task A
    _run(
        [py, "prepare_retrieve_dev_eval.py", "--dev-jsonl", dev_jsonl, "--workspace", str(workspace_exact)],
        env=env_ab,
        cwd=EXAMPLES,
        title=f"{tag} — Task A prepare",
    )
    env_a = dict(env_ab)
    env_a["RETRIEVE_EVAL_WORKSPACE"] = str(workspace_exact)
    _run([py, "build_eval_indexes.py"], env=env_a, cwd=EXAMPLES, title=f"{tag} — Task A build_eval_indexes")
    _run([py, "eval_three_pipelines.py", "--benchmark-profile", "exact"], env=env_a, cwd=EXAMPLES, title=f"{tag} — Task A eval")

    # Task B
    _run(
        [py, "build_short_query_eval.py", "--dev-jsonl", dev_jsonl, "--workspace", str(workspace_short)],
        env=env_ab,
        cwd=EXAMPLES,
        title=f"{tag} — Task B build_short_query_eval",
    )
    env_b = dict(env_ab)
    env_b["RETRIEVE_EVAL_WORKSPACE"] = str(workspace_short)
    _run([py, "build_eval_indexes.py"], env=env_b, cwd=EXAMPLES, title=f"{tag} — Task B build_eval_indexes")
    _run([py, "eval_three_pipelines.py", "--benchmark-profile", "short"], env=env_b, cwd=EXAMPLES, title=f"{tag} — Task B eval")

    # Task C
    env_c = dict(env_ab)
    env_c.update(
        {
            "PERQA_OBS_CSV": perqa_obs_csv,
            "RETRIEVE_EVAL_WORKSPACE": str(perqa_ws),
        }
    )
    if taskC_extra_env:
        env_c.update(taskC_extra_env)
    _run([py, "build_eval_indexes.py"], env=env_c, cwd=EXAMPLES, title=f"{tag} — Task C build_eval_indexes")
    env_c_eval = dict(env_c)
    env_c_eval.update(
        {
            "PERQA_SPLADE_INDEX": str(perqa_ws / "indices" / "splade"),
            "PERQA_DENSE_INDEX": str(perqa_ws / "indices" / "dense"),
            "PERQA_BM25_INDEX": str(perqa_ws / "indices" / "bm25"),
        }
    )
    _run(
        [
            py,
            "eval_perqa_retrieval_export.py",
            "--split",
            "dev",
            "--persona-id",
            "0",
            "--queries-jsonl",
            str(perqa_q),
        ],
        env=env_c_eval,
        cwd=EXAMPLES,
        title=f"{tag} — Task C eval",
    )

    taskA_json = _load_results(workspace_exact / "results_models.json")
    taskB_json = _load_results(workspace_short / "results_models.json")
    taskC_json = _load_results(stage_dir / f"{queries_perqa_src.stem}_results_models.json")
    return {
        "stage_dir": str(stage_dir),
        "taskA_json": str(workspace_exact / "results_models.json"),
        "taskB_json": str(workspace_short / "results_models.json"),
        "taskC_json": str(stage_dir / f"{queries_perqa_src.stem}_results_models.json"),
        "taskA": _dynamic_row(taskA_json),
        "taskB": _dynamic_row(taskB_json),
        "taskC": _dynamic_row(taskC_json),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run four-stage benchmark: baseline, trained, grid-best, learned-router.")
    ap.add_argument("--untrained-splade-model", required=True)
    ap.add_argument("--trained-splade-model", required=True)
    ap.add_argument("--dense-model", required=True)
    ap.add_argument("--dev-jsonl", required=True)
    ap.add_argument("--qu-dev-jsonl", required=True)
    ap.add_argument("--source-perqa-workspace", required=True, help="Workspace that already contains PerQA obs.csv.")
    ap.add_argument("--queries-perqa", default=str(EXAMPLES / "queries_dev_p0.jsonl"))
    ap.add_argument("--perqa-obs-csv", required=True)
    ap.add_argument("--out-root", required=True)
    args = ap.parse_args()

    py = sys.executable
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    source_perqa_workspace = Path(args.source_perqa_workspace).resolve()
    queries_perqa_src = Path(args.queries_perqa).resolve()

    base_env = os.environ.copy()
    base_env["RETRIEVE_DEV_JSONL"] = args.dev_jsonl
    base_env["QU_DEV_JSONL"] = args.qu_dev_jsonl

    stage_meta: List[Dict[str, Any]] = []

    # Stage 1: untrained SPLADE + trained Dense
    st1 = _stage_run_all(
        py,
        tag="stage1_untrained_splade_trained_dense",
        out_root=out_root,
        base_env=base_env,
        splade_model=args.untrained_splade_model,
        dense_model=args.dense_model,
        dev_jsonl=args.dev_jsonl,
        qu_dev_jsonl=args.qu_dev_jsonl,
        source_perqa_workspace=source_perqa_workspace,
        queries_perqa_src=queries_perqa_src,
        perqa_obs_csv=args.perqa_obs_csv,
    )
    st1.update(
        {
            "stage": "stage1",
            "tag": "untrained_splade+trained_dense",
            "splade_model": args.untrained_splade_model,
            "dense_model": args.dense_model,
            "dynamic_fixed_weights": {},
            "learned_router_model": "",
        }
    )
    stage_meta.append(st1)

    # Stage 2: trained SPLADE + trained Dense
    st2 = _stage_run_all(
        py,
        tag="stage2_trained_splade_trained_dense",
        out_root=out_root,
        base_env=base_env,
        splade_model=args.trained_splade_model,
        dense_model=args.dense_model,
        dev_jsonl=args.dev_jsonl,
        qu_dev_jsonl=args.qu_dev_jsonl,
        source_perqa_workspace=source_perqa_workspace,
        queries_perqa_src=queries_perqa_src,
        perqa_obs_csv=args.perqa_obs_csv,
    )
    st2.update(
        {
            "stage": "stage2",
            "tag": "trained_splade+trained_dense",
            "splade_model": args.trained_splade_model,
            "dense_model": args.dense_model,
            "dynamic_fixed_weights": {},
            "learned_router_model": "",
        }
    )
    stage_meta.append(st2)

    # Grid search on stage2 task-C workspace to get best fixed weights.
    stage2_dir = Path(st2["stage_dir"])
    stage2_perqa_ws = stage2_dir / "perqa_workspace"
    stage2_queries = stage2_dir / queries_perqa_src.name
    env_grid = dict(base_env)
    env_grid.update(
        {
            "PERQA_SPLADE_MODEL_TYPE_OR_PATH": args.trained_splade_model,
            "PERQA_DENSE_MODEL_TYPE_OR_PATH": args.dense_model,
            "PERQA_OBS_CSV": args.perqa_obs_csv,
            "PERQA_SPLADE_INDEX": str(stage2_perqa_ws / "indices" / "splade"),
            "PERQA_DENSE_INDEX": str(stage2_perqa_ws / "indices" / "dense"),
            "PERQA_BM25_INDEX": str(stage2_perqa_ws / "indices" / "bm25"),
        }
    )
    _run(
        [
            py,
            "eval_perqa_retrieval_export.py",
            "--split",
            "dev",
            "--persona-id",
            "0",
            "--queries-jsonl",
            str(stage2_queries),
            "--grid-search-dynamic",
            "--grid-metric",
            "Recall@10",
        ],
        env=env_grid,
        cwd=EXAMPLES,
        title="stage2 — taskC grid-search dynamic",
    )
    grid_json = stage2_dir / f"{queries_perqa_src.stem}_dynamic_grid_search.json"
    grid = _load_results(grid_json)
    best = grid.get("best", {})
    best_weights = {
        "bm25": float(best.get("_w_bm25", 0.1)),
        "dense": float(best.get("_w_dense", 0.4)),
        "splade": float(best.get("_w_splade", 0.5)),
    }

    # Stage 3: trained SPLADE + trained Dense + grid best fixed
    st3 = _stage_run_all(
        py,
        tag="stage3_trained_plus_grid_best",
        out_root=out_root,
        base_env=base_env,
        splade_model=args.trained_splade_model,
        dense_model=args.dense_model,
        dev_jsonl=args.dev_jsonl,
        qu_dev_jsonl=args.qu_dev_jsonl,
        source_perqa_workspace=source_perqa_workspace,
        queries_perqa_src=queries_perqa_src,
        perqa_obs_csv=args.perqa_obs_csv,
        taskC_extra_env={
            "DYNAMIC_FIXED_W_BM25": str(best_weights["bm25"]),
            "DYNAMIC_FIXED_W_DENSE": str(best_weights["dense"]),
            "DYNAMIC_FIXED_W_SPLADE": str(best_weights["splade"]),
        },
    )
    st3.update(
        {
            "stage": "stage3",
            "tag": "trained_splade+trained_dense+grid_best",
            "splade_model": args.trained_splade_model,
            "dense_model": args.dense_model,
            "dynamic_fixed_weights": best_weights,
            "learned_router_model": "",
            "grid_json": str(grid_json),
        }
    )
    stage_meta.append(st3)

    # Train learned router (Task C oracle from stage2 workspace/index).
    router_dir = out_root / "learned_router"
    router_dir.mkdir(parents=True, exist_ok=True)
    oracle_jsonl = router_dir / "taskC_oracle_labels.jsonl"
    router_model = router_dir / "taskC_learned_router_model.json"
    _run(
        [
            py,
            "build_perqa_router_oracle.py",
            "--queries-jsonl",
            str(stage2_queries),
            "--obs-csv",
            args.perqa_obs_csv,
            "--splade-index",
            str(stage2_perqa_ws / "indices" / "splade"),
            "--dense-index",
            str(stage2_perqa_ws / "indices" / "dense"),
            "--bm25-index",
            str(stage2_perqa_ws / "indices" / "bm25"),
            "--dense-model",
            args.dense_model,
            "--out-jsonl",
            str(oracle_jsonl),
        ],
        env=env_grid,
        cwd=EXAMPLES,
        title="train-router — build oracle labels",
    )
    _run(
        [
            py,
            "train_perqa_learned_router.py",
            "--oracle-jsonl",
            str(oracle_jsonl),
            "--out-model-json",
            str(router_model),
        ],
        env=env_grid,
        cwd=EXAMPLES,
        title="train-router — fit learned router",
    )

    # Stage 4: trained SPLADE + trained Dense + learned router
    st4 = _stage_run_all(
        py,
        tag="stage4_trained_plus_learned_router",
        out_root=out_root,
        base_env=base_env,
        splade_model=args.trained_splade_model,
        dense_model=args.dense_model,
        dev_jsonl=args.dev_jsonl,
        qu_dev_jsonl=args.qu_dev_jsonl,
        source_perqa_workspace=source_perqa_workspace,
        queries_perqa_src=queries_perqa_src,
        perqa_obs_csv=args.perqa_obs_csv,
        taskC_extra_env={"DYNAMIC_LEARNED_ROUTER_MODEL_PATH": str(router_model)},
    )
    st4.update(
        {
            "stage": "stage4",
            "tag": "trained_splade+trained_dense+learned_router",
            "splade_model": args.trained_splade_model,
            "dense_model": args.dense_model,
            "dynamic_fixed_weights": {},
            "learned_router_model": str(router_model),
            "oracle_jsonl": str(oracle_jsonl),
        }
    )
    stage_meta.append(st4)

    _write_summary(out_root, stage_meta)
    print(f"\nDone. Summary written under: {out_root}")


if __name__ == "__main__":
    main()
