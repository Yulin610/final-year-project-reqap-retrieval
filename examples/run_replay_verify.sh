#!/usr/bin/env bash
set -Eeuo pipefail

#######################################
# Config (可按需改)
#######################################
REPO_ROOT="${REPO_ROOT:-/root/autodl-tmp/ReQAP-main}"
BUNDLE="${BUNDLE:-/root/autodl-tmp/four_stage_benchmarks_full}"
STAGE2="${STAGE2:-$BUNDLE/stage2_trained_splade_trained_dense}"
STAGE3="${STAGE3:-$BUNDLE/stage3_trained_plus_grid_best}"

SPLADE_MODEL="${SPLADE_MODEL:-/root/autodl-tmp/models/splade_bucket_mvp}"
DENSE_MODEL="${DENSE_MODEL:-/root/autodl-tmp/models/dense_adapted}"

PY_BIN="${PY_BIN:-python3}"
PERSONA_ID="${PERSONA_ID:-0}"
SPLIT="${SPLIT:-dev}"

# learned router路径
ROUTER_JSON="${ROUTER_JSON:-$BUNDLE/learned_router/taskC_learned_router_model.json}"

#######################################
# Error trap
#######################################
on_error() {
  local exit_code=$?
  local line_no=$1
  echo "[FATAL] Script failed at line ${line_no}, exit_code=${exit_code}" >&2
  echo "[FATAL] Check logs under: /tmp/perqa_replay_*.log (if created)" >&2
  exit "${exit_code}"
}
trap 'on_error ${LINENO}' ERR

#######################################
# Helpers
#######################################
die() { echo "[FATAL] $*" >&2; exit 1; }
info() { echo "[INFO] $*"; }
ok() { echo "[OK] $*"; }

need_file() { [[ -f "$1" ]] || die "Missing file: $1"; }
need_dir()  { [[ -d "$1" ]] || die "Missing directory: $1"; }

run_one() {
  local name="$1"; shift
  info "Running: $name"
  "$PY_BIN" "$EVAL" "$@" 2>&1 | tee "/tmp/perqa_replay_${name}.log"
  local out_json="$OUT/$name/queries_dev_p0_results_models.json"
  [[ -s "$out_json" ]] || die "Output missing or empty: $out_json"
  ok "$name done -> $out_json"
}

#######################################
# Resolve paths
#######################################
EXAMPLES="$REPO_ROOT/reqap-retrieval-modular/examples"
EVAL="$EXAMPLES/eval_perqa_retrieval_export.py"

QUERIES="$STAGE3/queries_dev_p0.jsonl"
GRID_JSON="$STAGE2/queries_dev_p0_dynamic_grid_search.json"

OBS_CSV="$REPO_ROOT/data/data/benchmarks/perqa/${SPLIT}/${SPLIT}_persona_${PERSONA_ID}/${SPLIT}_persona_${PERSONA_ID}_obs.csv"
QU_JSONL="$REPO_ROOT/data/data/results/perqa/reqap_sft/${SPLIT}_persona_${PERSONA_ID}/qu_result.jsonl"
QU_CACHE="$STAGE3/qu_retrieve_counts_${SPLIT}_persona_${PERSONA_ID}.json"

PERQA_SPLADE_INDEX="$STAGE3/perqa_workspace/indices/splade"
PERQA_DENSE_INDEX="$STAGE3/perqa_workspace/indices/dense"
PERQA_BM25_INDEX="$STAGE3/perqa_workspace/indices/bm25"

OUT="$STAGE3/replay_verify_$(date +%Y%m%d_%H%M%S)"

#######################################
# Preflight checks
#######################################
info "Preflight checks..."

command -v "$PY_BIN" >/dev/null 2>&1 || die "Python not found: $PY_BIN"
need_dir "$REPO_ROOT"
need_dir "$BUNDLE"
need_dir "$EXAMPLES"
need_file "$EVAL"

need_file "$QUERIES"
need_file "$GRID_JSON"
need_file "$OBS_CSV"
need_file "$QU_JSONL"
need_dir "$PERQA_SPLADE_INDEX"
need_dir "$PERQA_DENSE_INDEX"
need_dir "$PERQA_BM25_INDEX"
need_file "$ROUTER_JSON"

# 关键包/模块导入检查
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/reqap-retrieval-modular:$EXAMPLES${PYTHONPATH:+:$PYTHONPATH}"
"$PY_BIN" - <<'PY'
import importlib
mods = [
  "pandas","omegaconf","loguru","transformers",
  "reqap.retrieval.splade.models",
  "reqap_modular_retrieval.retrievers.bm25",
  "perqa_benchmark_paths","qu_retrieve_counts","build_eval_indexes"
]
for m in mods:
    importlib.import_module(m)
print("import-check-ok")
PY
ok "Dependency/module imports passed"

#######################################
# Export env
#######################################
mkdir -p "$OUT/rule_baseline" "$OUT/fixed_grid_best" "$OUT/learned_router"

export PERQA_SPLADE_MODEL_TYPE_OR_PATH="$SPLADE_MODEL"
export PERQA_DENSE_MODEL_TYPE_OR_PATH="$DENSE_MODEL"
export PERQA_OBS_CSV="$OBS_CSV"
export PERQA_SPLADE_INDEX="$PERQA_SPLADE_INDEX"
export PERQA_DENSE_INDEX="$PERQA_DENSE_INDEX"
export PERQA_BM25_INDEX="$PERQA_BM25_INDEX"

info "OUT=$OUT"

#######################################
# Run 3 modes
#######################################
unset DYNAMIC_FIXED_W_BM25 DYNAMIC_FIXED_W_DENSE DYNAMIC_FIXED_W_SPLADE DYNAMIC_LEARNED_ROUTER_MODEL_PATH || true
run_one rule_baseline \
  --split "$SPLIT" --persona-id "$PERSONA_ID" \
  --queries-jsonl "$QUERIES" \
  --obs-csv "$OBS_CSV" \
  --splade-index "$PERQA_SPLADE_INDEX" \
  --dense-index "$PERQA_DENSE_INDEX" \
  --bm25-index "$PERQA_BM25_INDEX" \
  --qu-jsonl "$QU_JSONL" \
  --qu-retrieve-counts-cache "$QU_CACHE" \
  --out-dir "$OUT/rule_baseline"

export DYNAMIC_FIXED_W_BM25=0.2 DYNAMIC_FIXED_W_DENSE=0.5 DYNAMIC_FIXED_W_SPLADE=0.6
run_one fixed_grid_best \
  --split "$SPLIT" --persona-id "$PERSONA_ID" \
  --queries-jsonl "$QUERIES" \
  --obs-csv "$OBS_CSV" \
  --splade-index "$PERQA_SPLADE_INDEX" \
  --dense-index "$PERQA_DENSE_INDEX" \
  --bm25-index "$PERQA_BM25_INDEX" \
  --qu-jsonl "$QU_JSONL" \
  --qu-retrieve-counts-cache "$QU_CACHE" \
  --out-dir "$OUT/fixed_grid_best"
unset DYNAMIC_FIXED_W_BM25 DYNAMIC_FIXED_W_DENSE DYNAMIC_FIXED_W_SPLADE

export DYNAMIC_LEARNED_ROUTER_MODEL_PATH="$ROUTER_JSON"
run_one learned_router \
  --split "$SPLIT" --persona-id "$PERSONA_ID" \
  --queries-jsonl "$QUERIES" \
  --obs-csv "$OBS_CSV" \
  --splade-index "$PERQA_SPLADE_INDEX" \
  --dense-index "$PERQA_DENSE_INDEX" \
  --bm25-index "$PERQA_BM25_INDEX" \
  --qu-jsonl "$QU_JSONL" \
  --qu-retrieve-counts-cache "$QU_CACHE" \
  --out-dir "$OUT/learned_router"
unset DYNAMIC_LEARNED_ROUTER_MODEL_PATH

#######################################
# Validate wiring + generate report
#######################################
"$PY_BIN" - <<PY
import json, os, sys, datetime, platform
out = "$OUT"
grid_json = "$GRID_JSON"
keys = ("Recall@10","Hit@1","MRR","NDCG@10")

def load_dynamic(path):
    d = json.load(open(path, encoding="utf-8"))
    row = next(r for r in d["results"] if r["Model"]=="Dynamic Fusion (Ours)")
    return d, {k: float(row[k]) for k in keys}

grid = json.load(open(grid_json, encoding="utf-8"))["best"]
grid_m = {k: float(grid[k]) for k in keys}

rb_d, rb_m = load_dynamic(f"{out}/rule_baseline/queries_dev_p0_results_models.json")
fg_d, fg_m = load_dynamic(f"{out}/fixed_grid_best/queries_dev_p0_results_models.json")
lr_d, lr_m = load_dynamic(f"{out}/learned_router/queries_dev_p0_results_models.json")

# 1) fixed provenance must be set
fw = (fg_d.get("dynamic_fixed_weights") or {})
if not (fw.get("bm25")==0.2 and fw.get("dense")==0.5 and fw.get("splade")==0.6):
    print("[FATAL] fixed_grid_best provenance mismatch:", fw, file=sys.stderr)
    sys.exit(2)

# 2) learned provenance must be set
lrp = lr_d.get("dynamic_learned_router_model_path")
if not lrp:
    print("[FATAL] learned_router model path missing in provenance", file=sys.stderr)
    sys.exit(3)

# 3) fixed should align grid best (tight tolerance)
tol = 1e-12
for k in keys:
    if abs(fg_m[k]-grid_m[k]) > tol:
        print(f"[FATAL] fixed_grid_best not aligned with grid[best] on {k}: {fg_m[k]} vs {grid_m[k]}", file=sys.stderr)
        sys.exit(4)

# 4) fixed should differ from rule (otherwise likely not wired)
if all(abs(fg_m[k]-rb_m[k]) < 1e-12 for k in keys):
    print("[FATAL] fixed_grid_best metrics identical to rule_baseline; wiring likely ineffective", file=sys.stderr)
    sys.exit(5)

report = {
  "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
  "host": platform.node(),
  "out_dir": out,
  "grid_best_source": grid_json,
  "grid_best": grid_m,
  "runs": [
    {"name":"rule_baseline","provenance":{"dynamic_fixed_weights": rb_d.get("dynamic_fixed_weights"),"dynamic_learned_router_model_path": rb_d.get("dynamic_learned_router_model_path")},"metrics": rb_m},
    {"name":"fixed_grid_best","provenance":{"dynamic_fixed_weights": fg_d.get("dynamic_fixed_weights"),"dynamic_learned_router_model_path": fg_d.get("dynamic_learned_router_model_path")},"metrics": fg_m},
    {"name":"learned_router","provenance":{"dynamic_fixed_weights": lr_d.get("dynamic_fixed_weights"),"dynamic_learned_router_model_path": lr_d.get("dynamic_learned_router_model_path")},"metrics": lr_m},
  ]
}

json_path = f"{out}/replay_report.json"
md_path = f"{out}/replay_report.md"

with open(json_path,"w",encoding="utf-8") as f:
    json.dump(report,f,ensure_ascii=False,indent=2)

lines = []
lines.append("# Replay Verification Report")
lines.append("")
lines.append(f"- OUT: `{out}`")
lines.append(f"- Grid source: `{grid_json}`")
lines.append("")
lines.append("## Grid Best Reference")
lines.append("")
lines.append("| Recall@10 | Hit@1 | MRR | NDCG@10 |")
lines.append("| --- | --- | --- | --- |")
lines.append(f"| {grid_m['Recall@10']:.12f} | {grid_m['Hit@1']:.12f} | {grid_m['MRR']:.12f} | {grid_m['NDCG@10']:.12f} |")
lines.append("")
lines.append("## Dynamic Fusion Comparison")
lines.append("")
lines.append("| Run | Recall@10 | Hit@1 | MRR | NDCG@10 | fixed_weights | learned_router_model |")
lines.append("| --- | --- | --- | --- | --- | --- | --- |")
for r in report["runs"]:
    lines.append(
      f"| {r['name']} | {r['metrics']['Recall@10']:.12f} | {r['metrics']['Hit@1']:.12f} | {r['metrics']['MRR']:.12f} | {r['metrics']['NDCG@10']:.12f} | `{r['provenance']['dynamic_fixed_weights']}` | `{r['provenance']['dynamic_learned_router_model_path']}` |"
    )

with open(md_path,"w",encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

print("[OK] Wrote:", json_path)
print("[OK] Wrote:", md_path)
print("[OK] Wiring validation passed")
PY

tar -czf "$OUT/replay_verify_archive.tar.gz" -C "$OUT" replay_report.md replay_report.json rule_baseline fixed_grid_best learned_router
ok "Archive: $OUT/replay_verify_archive.tar.gz"
echo "[DONE] OUT=$OUT"