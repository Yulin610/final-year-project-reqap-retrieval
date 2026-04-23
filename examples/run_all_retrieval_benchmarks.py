"""
一次性跑完三套检索评测（Task A Exact / Task B Short / Task C PerQA），并生成对应结果表。

顺序：
  1) Task A：prepare_retrieve_dev_eval（可选）→ build_eval_indexes → eval_three_pipelines --benchmark-profile exact
  2) Task B：build_short_query_eval（可选）→ build_eval_indexes → eval_three_pipelines --benchmark-profile short
  3) Task C：对 PerQA 工作区 build_eval_indexes（可选）→ eval_perqa_retrieval_export

换 SPLADE checkpoint 后必须重建各工作区的 SPLADE 索引；本脚本默认会跑 build_eval_indexes（三处）。

用法（PowerShell）:
  cd ...\\reqap-retrieval-modular\\examples
  python run_all_retrieval_benchmarks.py \\
    --splade-model "c:\\Users\\23369\\Desktop\\final_work\\models\\splade_bucket_mvp" \\
    --dense-model "c:\\Users\\23369\\Desktop\\final_work\\dense_adapted"

  # 只跑部分任务：
  python run_all_retrieval_benchmarks.py --splade-model ... --tasks A C
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_EXAMPLES = Path(__file__).resolve().parent


def _run(
    args: list[str],
    *,
    env: dict[str, str],
    cwd: Path,
    title: str,
) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")
    print("+", " ".join(args))
    subprocess.run(args, env=env, cwd=str(cwd), check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Task A/B/C retrieval benchmarks in one shot.")
    ap.add_argument(
        "--splade-model",
        required=True,
        help="SPLADE checkpoint dir or Hub id (must match indexing + eval).",
    )
    ap.add_argument(
        "--dense-model",
        required=True,
        help="Dense model dir or Hub id (must match indexing + eval).",
    )
    ap.add_argument(
        "--dev-jsonl",
        default=os.environ.get(
            "RETRIEVE_DEV_JSONL",
            r"c:\Users\23369\Desktop\final_work\data\retrieve\dev_data.jsonl",
        ),
        help="dev_data.jsonl for Task A/B prepare.",
    )
    ap.add_argument(
        "--workspace-exact",
        default=os.environ.get(
            "RETRIEVE_EVAL_WORKSPACE_EXACT",
            r"c:\Users\23369\Desktop\final_work\data\retrieve\eval_workspace",
        ),
        help="Task A workspace (obs.csv, queries.jsonl, indices/).",
    )
    ap.add_argument(
        "--workspace-short",
        default=os.environ.get(
            "RETRIEVE_EVAL_WORKSPACE_SHORT",
            r"c:\Users\23369\Desktop\final_work\data\retrieve\eval_workspace_short_query",
        ),
        help="Task B workspace.",
    )
    ap.add_argument(
        "--perqa-workspace",
        default=str(_EXAMPLES / "eval_workspace_dev_p0"),
        help="Task C workspace: must contain obs.csv, queries.jsonl; indices under indices/.",
    )
    ap.add_argument(
        "--queries-perqa",
        default=str(_EXAMPLES / "queries_dev_p0.jsonl"),
        help="PerQA queries jsonl for eval_perqa_retrieval_export.",
    )
    ap.add_argument(
        "--perqa-obs-csv",
        default="",
        help="Optional explicit Task C obs.csv path (recommended official benchmark obs to avoid id mismatch).",
    )
    ap.add_argument(
        "--qu-dev-jsonl",
        default=os.environ.get(
            "QU_DEV_JSONL",
            r"c:\Users\23369\Desktop\final_work\data\qu\dev_data.jsonl",
        ),
        help="QU dev jsonl (Dynamic Fusion retrieve counts).",
    )
    ap.add_argument(
        "--tasks",
        default="A,B,C",
        help="Comma-separated: A (exact), B (short), C (perqa). Example: A,B,C",
    )
    ap.add_argument(
        "--skip-data-prep",
        action="store_true",
        help="Do not run prepare_retrieve_dev_eval / build_short_query_eval (assume obs+queries exist).",
    )
    ap.add_argument(
        "--skip-index-build",
        action="store_true",
        help="Skip build_eval_indexes.py (only run eval). Indices must already match --splade-model.",
    )
    args = ap.parse_args()

    py = sys.executable
    cwd = _EXAMPLES
    tasks = {x.strip().upper() for x in args.tasks.split(",") if x.strip()}

    base = os.environ.copy()
    base["PERQA_SPLADE_MODEL_TYPE_OR_PATH"] = args.splade_model
    base["PERQA_DENSE_MODEL_TYPE_OR_PATH"] = args.dense_model
    base["RETRIEVE_DEV_JSONL"] = args.dev_jsonl
    base["QU_DEV_JSONL"] = args.qu_dev_jsonl

    # --- Task A ---
    if "A" in tasks:
        if not args.skip_data_prep:
            _run(
                [py, "prepare_retrieve_dev_eval.py", "--dev-jsonl", args.dev_jsonl, "--workspace", args.workspace_exact],
                env=dict(base),
                cwd=cwd,
                title="Task A — prepare (exact match obs + queries)",
            )
        if not args.skip_index_build:
            e = dict(base)
            e["RETRIEVE_EVAL_WORKSPACE"] = args.workspace_exact
            _run([py, "build_eval_indexes.py"], env=e, cwd=cwd, title="Task A — build_eval_indexes (BM25 / SPLADE / Dense)")
        e = dict(base)
        e["RETRIEVE_EVAL_WORKSPACE"] = args.workspace_exact
        _run(
            [py, "eval_three_pipelines.py", "--benchmark-profile", "exact"],
            env=e,
            cwd=cwd,
            title="Task A — eval_three_pipelines (results_table_task1_exact_match.md + full)",
        )

    # --- Task B ---
    if "B" in tasks:
        if not args.skip_data_prep:
            _run(
                [py, "build_short_query_eval.py", "--dev-jsonl", args.dev_jsonl, "--workspace", args.workspace_short],
                env=dict(base),
                cwd=cwd,
                title="Task B — build_short_query_eval",
            )
        if not args.skip_index_build:
            e = dict(base)
            e["RETRIEVE_EVAL_WORKSPACE"] = args.workspace_short
            _run([py, "build_eval_indexes.py"], env=e, cwd=cwd, title="Task B — build_eval_indexes")
        e = dict(base)
        e["RETRIEVE_EVAL_WORKSPACE"] = args.workspace_short
        _run(
            [py, "eval_three_pipelines.py", "--benchmark-profile", "short"],
            env=e,
            cwd=cwd,
            title="Task B — eval_three_pipelines (results_table_task2_short_query.md + full)",
        )

    # --- Task C ---
    if "C" in tasks:
        perqa = Path(args.perqa_workspace)
        obs = perqa / "obs.csv"
        qf = perqa / "queries.jsonl"
        if not obs.is_file() or not qf.is_file():
            raise SystemExit(
                f"Task C: missing {obs} or {qf}. Create/populate eval_workspace_dev_p0 first "
                f"(e.g. export PerQA queries + obs into this folder)."
            )
        qpath = Path(args.queries_perqa)
        if not qpath.is_file():
            raise SystemExit(f"Task C: missing queries file: {qpath}")

        if not args.skip_index_build:
            e = dict(base)
            e["RETRIEVE_EVAL_WORKSPACE"] = str(perqa.resolve())
            if args.perqa_obs_csv:
                e["PERQA_OBS_CSV"] = str(Path(args.perqa_obs_csv).resolve())
            else:
                e["PERQA_OBS_CSV"] = str(obs.resolve())
            _run([py, "build_eval_indexes.py"], env=e, cwd=cwd, title="Task C — build_eval_indexes (PerQA workspace)")

        e = dict(base)
        if args.perqa_obs_csv:
            e["PERQA_OBS_CSV"] = str(Path(args.perqa_obs_csv).resolve())
        else:
            e["PERQA_OBS_CSV"] = str(obs.resolve())
        e["PERQA_SPLADE_INDEX"] = str((perqa / "indices" / "splade").resolve())
        e["PERQA_DENSE_INDEX"] = str((perqa / "indices" / "dense").resolve())
        e["PERQA_BM25_INDEX"] = str((perqa / "indices" / "bm25").resolve())
        _run(
            [
                py,
                "eval_perqa_retrieval_export.py",
                "--split",
                "dev",
                "--persona-id",
                "0",
                "--queries-jsonl",
                str(qpath.resolve()),
            ],
            env=e,
            cwd=cwd,
            title="Task C — eval_perqa_retrieval_export (queries_*_results_table*.md)",
        )

    print("\n" + "=" * 72)
    print("Done. Expected outputs:")
    if "A" in tasks:
        print(f"  Task A: {args.workspace_exact}\\results_table.md")
        print(f"          {args.workspace_exact}\\results_table_task1_exact_match.md")
    if "B" in tasks:
        print(f"  Task B: {args.workspace_short}\\results_table.md")
        print(f"          {args.workspace_short}\\results_table_task2_short_query.md")
    if "C" in tasks:
        out_dir = qpath.parent
        stem = qpath.stem
        print(f"  Task C: {out_dir}\\{stem}_results_table.md")
        print(f"          {out_dir}\\{stem}_results_table_task3_perqa.md")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
