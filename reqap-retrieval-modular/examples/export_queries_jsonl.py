"""
从 qu_result.jsonl 导出检索评测用 queries.jsonl：qid / query / query_key / relevant_ids（gold obs id 并集）。

gold 派生与 verify_perqa_gold_obs_selfcheck 相同（perqa_sql_gold.build_export_rows）。

依赖: pip install pandas duckdb
用法:
  python export_queries_jsonl.py --split dev --persona-id 0 -o queries_dev_p0.jsonl
  python export_queries_jsonl.py --split dev --persona-id 0 --max 500 --random --seed 42 -o sample.jsonl
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

_EX = Path(__file__).resolve().parent
if str(_EX) not in sys.path:
    sys.path.insert(0, str(_EX))

import pandas as pd

from perqa_benchmark_paths import persona_folder_name, reqap_main_root
from perqa_sql_gold import MiniQE, build_export_rows, prepare_simple_pool


def main() -> None:
    ap = argparse.ArgumentParser(description="Export PerQA queries JSONL for retrieval eval.")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--persona-id", type=int, default=0)
    ap.add_argument("-o", "--output", required=True, help="Output .jsonl path")
    ap.add_argument("--qu-result", default="", help="Override path to qu_result.jsonl")
    ap.add_argument("--max", type=int, default=None, help="Cap number of simple-SQL questions (after shuffle if --random)")
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    repo = reqap_main_root()
    folder = persona_folder_name(args.split, args.persona_id)
    obs_csv = repo / "data" / "data" / "benchmarks" / "perqa" / args.split / folder / f"{folder}_obs.csv"
    str_csv = repo / "data" / "data" / "benchmarks" / "perqa" / args.split / folder / f"{folder}_str.csv"
    qu_path = Path(args.qu_result) if args.qu_result else repo / "data" / "data" / "results" / "perqa" / "reqap_sft" / folder / "qu_result.jsonl"

    for p in (obs_csv, str_csv, qu_path):
        if not p.is_file():
            raise SystemExit(f"Missing: {p}")

    obs_df = pd.read_csv(obs_csv, converters={"event_data": json.loads})
    with contextlib.redirect_stdout(io.StringIO()):
        qe = MiniQE(str(obs_csv), str(str_csv))

    pool = prepare_simple_pool(
        qu_path,
        max_rows=args.max,
        randomize=args.random,
        seed=args.seed,
    )
    rows, skipped = build_export_rows(pool, obs_df, qe)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = {
        "output": str(out.resolve()),
        "split": args.split,
        "persona_id": args.persona_id,
        "pool_size": len(pool),
        "exported": len(rows),
        "skipped_no_pairs": skipped["no_pairs"],
        "skipped_empty_gold": skipped["empty_gold"],
        "random": args.random,
        "seed": args.seed,
        "max_cap": args.max,
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
