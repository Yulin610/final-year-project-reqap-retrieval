"""
Task 2 — Short Query Retrieval（类别级 / multi-label）

从同一份 dev_data.jsonl 构造：
  query = input[0]（短意图）
  relevant_ids = 所有与该 input[0] 相同的行的 doc id（行号）

obs.csv 与 Exact Match（Task 1）相同：每行 dev 仍为一条文档，便于复用 build_eval_indexes.py。

用法:
  python build_short_query_eval.py \\
    --dev-jsonl "C:\\...\\data\\retrieve\\dev_data.jsonl" \\
    --workspace "C:\\...\\data\\retrieve\\eval_workspace_short_query"

然后对该 workspace 设置 RETRIEVE_EVAL_WORKSPACE 后运行 build_eval_indexes.py 与 eval_three_pipelines.py，
并设置环境变量 RETRIEVE_BENCHMARK_PROFILE=short 以生成 Task 2 聚焦表。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

_EX = Path(__file__).resolve().parent
if str(_EX) not in sys.path:
    sys.path.insert(0, str(_EX))

from prepare_retrieve_dev_eval import row_to_query_text


def prepare_short_query_workspace(dev_jsonl: str, workspace: str) -> None:
    os.makedirs(workspace, exist_ok=True)
    obs_csv = os.path.join(workspace, "obs.csv")
    queries_jsonl = os.path.join(workspace, "queries.jsonl")

    rows: list[dict] = []
    with open(dev_jsonl, "r", encoding="utf-8-sig") as f_in:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    groups: dict[str, list[int]] = defaultdict(list)
    for doc_id, row in enumerate(rows):
        inp = row.get("input")
        if not isinstance(inp, list) or len(inp) < 1:
            continue
        key = str(inp[0]).strip()
        groups[key].append(doc_id)

    multi_label_groups = sum(1 for v in groups.values() if len(v) > 1)
    if multi_label_groups == 0:
        raise ValueError(
            "Short-query benchmark degenerated to single-label groups only. "
            "Expected at least one query_key with >1 relevant_ids."
        )

    meta_path = os.path.join(workspace, "short_query_benchmark_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f_meta:
        json.dump(
            {
                "task": "short_query_retrieval",
                "num_docs": len(rows),
                "num_unique_short_queries": len(groups),
                "keys": sorted(groups.keys()),
            },
            f_meta,
            ensure_ascii=False,
            indent=2,
        )

    with open(obs_csv, "w", encoding="utf-8", newline="") as f_obs, open(
        queries_jsonl, "w", encoding="utf-8"
    ) as f_q:
        w = csv.DictWriter(f_obs, fieldnames=["id", "event_type", "event_data"])
        w.writeheader()
        for doc_id, row in enumerate(rows):
            event_data = {
                "retrieve_dev": True,
                "positive": row.get("positive"),
                "input_type": row.get("input_type"),
                "label": row.get("label"),
                "text": row_to_query_text(row),
            }
            w.writerow(
                {
                    "id": doc_id,
                    "event_type": "retrieve_dev",
                    "event_data": json.dumps(event_data, ensure_ascii=False),
                }
            )

        for qi, key in enumerate(sorted(groups.keys())):
            rel = sorted(groups[key])
            f_q.write(
                json.dumps(
                    {
                        "qid": f"short-{qi}",
                        "query": key,
                        "query_key": key,
                        "relevant_ids": rel,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"Wrote {obs_csv}, {queries_jsonl}, {meta_path}")
    print(
        f"  docs={len(rows)}  unique_short_queries={len(groups)}  "
        f"multi_label_groups={multi_label_groups}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Task 2 short-query queries.jsonl + obs.csv (multi-label).")
    _dev_def = os.environ.get("RETRIEVE_DEV_JSONL", r"C:\Users\23369\Desktop\final_work\data\retrieve\dev_data.jsonl")
    _ws_def = os.environ.get(
        "RETRIEVE_EVAL_WORKSPACE",
        r"C:\Users\23369\Desktop\final_work\data\retrieve\eval_workspace_short_query",
    )
    ap.add_argument("--dev-jsonl", default=_dev_def, help="Same dev_data.jsonl as Task 1.")
    ap.add_argument(
        "--workspace",
        default=_ws_def,
        help="Output directory (obs.csv, queries.jsonl).",
    )
    args = ap.parse_args()
    prepare_short_query_workspace(args.dev_jsonl, args.workspace)


if __name__ == "__main__":
    main()
