"""
Align retrieval eval workspace with official PerQA benchmark files:

  - Copies  {split}_persona_{id}_obs.csv  →  WORKSPACE/obs.csv
  - Writes  questions.json              →  WORKSPACE/queries.jsonl
    (one line per question; relevant_ids default [] unless you merge gold later)

Then run build_eval_indexes.py / eval_three_pipelines.py against the same WORKSPACE.

Note: questions.json has no per-query gold obs row ids. For Hit@k you still need
relevant_ids (e.g. from SQL gold pipeline or a separate jsonl). Empty relevant_ids
makes Hit@* metrics all zero but indexing / latency tests are still valid.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

_EX = Path(__file__).resolve().parent
if str(_EX) not in sys.path:
    sys.path.insert(0, str(_EX))

from perqa_benchmark_paths import perqa_obs_csv, perqa_questions_json

WORKSPACE_DEFAULT = os.environ.get(
    "RETRIEVE_EVAL_WORKSPACE",
    str(_EX / "perqa_benchmark_eval_workspace"),
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Copy benchmark obs + build queries.jsonl for modular eval.")
    ap.add_argument("--split", default=os.environ.get("PERQA_SPLIT", "dev"), choices=("train", "dev", "test"))
    ap.add_argument("--persona-id", type=int, default=int(os.environ.get("PERQA_PERSONA_ID", "0")))
    ap.add_argument("--workspace", default=WORKSPACE_DEFAULT, help="Eval workspace (obs.csv + queries.jsonl)")
    ap.add_argument("--obs-only", action="store_true", help="Only copy obs.csv, skip queries.jsonl")
    args = ap.parse_args()

    src_obs = perqa_obs_csv(args.split, args.persona_id)
    if not src_obs.is_file():
        raise SystemExit(f"Missing benchmark obs: {src_obs}")

    ws = Path(args.workspace)
    ws.mkdir(parents=True, exist_ok=True)
    dst_obs = ws / "obs.csv"
    shutil.copy2(src_obs, dst_obs)
    print(f"Copied obs → {dst_obs}")

    if args.obs_only:
        return

    src_q = perqa_questions_json(args.split, args.persona_id)
    if not src_q.is_file():
        raise SystemExit(f"Missing questions.json: {src_q}")

    with open(src_q, "r", encoding="utf-8") as f:
        questions = json.load(f)
    if not isinstance(questions, list):
        raise SystemExit("questions.json must be a JSON array")

    out_q = ws / "queries.jsonl"
    with open(out_q, "w", encoding="utf-8") as f_out:
        for item in questions:
            qtext = (item.get("question") or "").strip()
            if not qtext:
                continue
            qid = item.get("id") or f"{args.split}_persona_{args.persona_id}-q_{item.get('q_id', len(questions))}"
            row = {
                "qid": qid,
                "query": qtext,
                "query_key": qtext.split("\n", 1)[0].strip()[:200],
                "relevant_ids": [],
            }
            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote queries → {out_q} (relevant_ids=[]; add gold for Hit@k)")
    print("Next: set RETRIEVE_EVAL_WORKSPACE to this folder, then build_eval_indexes.py / eval_three_pipelines.py")


if __name__ == "__main__":
    main()
