"""
自检：复现 CE gold_obs_event_ids 与 obs.csv 一致性（依赖 perqa_sql_gold）。

依赖: pip install pandas duckdb
用法:
  python verify_perqa_gold_obs_selfcheck.py --max 200 --random
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
from perqa_sql_gold import (
    MiniQE,
    load_jsonl,
    prepare_simple_pool,
    run_selfcheck_on_pool,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--persona-id", type=int, default=0)
    ap.add_argument("--max", type=int, default=25)
    ap.add_argument("--qu-result", default="")
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

    pool = prepare_simple_pool(qu_path, max_rows=args.max, randomize=args.random, seed=args.seed)
    if args.random:
        print(
            f"Random sample: pool_size after cap={len(pool)}, max={args.max}, seed={args.seed!r}"
        )
    else:
        print(f"Sequential: simple SQL, taking first max={args.max} -> {len(pool)} questions")

    all_simple = [r for r in load_jsonl(qu_path) if MiniQE.is_simple_query(r["sql_query"])]
    if len(all_simple) < args.max:
        print(
            f"Note: only {len(all_simple)} simple-SQL questions available (fewer than --max {args.max})."
        )

    failures, ok_lines = run_selfcheck_on_pool(pool, obs_df, qe)
    for line in ok_lines:
        print(line)
    print(f"\n--- summary: {len(pool)} instances checked ---")
    if failures:
        for fmsg in failures:
            print("FAIL", fmsg)
        raise SystemExit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
