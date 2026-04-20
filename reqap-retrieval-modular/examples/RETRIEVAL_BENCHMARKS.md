# Retrieval benchmark: three tasks (paper tables)

This repo separates **three retrieval regimes** so metrics match the label structure (single-label vs multi-label).

## Task 1 — Exact Match (retrieve-dev)

- **Query**: full structured event (`input[0]` + `input[1]`, same as current dev pipeline).
- **Label**: one relevant doc id per query (self-match).
- **Prepare**: `prepare_retrieve_dev_eval.py` → `obs.csv` + `queries.jsonl`.
- **Indexes**: `build_eval_indexes.py` with the same `RETRIEVE_EVAL_WORKSPACE`.
- **Eval**: `eval_three_pipelines.py` with `--benchmark-profile exact` (or `RETRIEVE_BENCHMARK_PROFILE=exact`).
- **Focused table**: `results_table_task1_exact_match.md` — Hit@1, MRR, Hit@5, Hit@10, latency.
- **Full table**: `results_table.md` (all columns including Recall@k).

## Task 2 — Short Query (category / multi-label)

- **Query**: `input[0]` only (short intent string).
- **Label**: **all** doc ids that share that `input[0]`.
- **Prepare**: `build_short_query_eval.py` (writes `obs.csv` + `queries.jsonl` + `short_query_benchmark_meta.json`). Use a **separate** workspace directory (e.g. `eval_workspace_short_query`).
- **Indexes**: `build_eval_indexes.py` with `RETRIEVE_EVAL_WORKSPACE` pointing at that workspace (same `obs.csv` schema as Task 1).
- **Eval**: `eval_three_pipelines.py` with `--benchmark-profile short`.
- **Focused table**: `results_table_task2_short_query.md` — Recall@10, Recall@50, NDCG@10, Precision@10, MRR, latency.
- **Do not** use Hit@1 as the primary metric for this task.

## Task 3 — PerQA

- **Queries**: exported `queries_dev_p*.jsonl` (natural-language questions, multiple obs ids).
- **Eval**: `eval_perqa_retrieval_export.py --queries-jsonl ...` with persona indices.
- **Focused table**: `*_results_table_task3_perqa.md` — Recall@10, Recall@50, NDCG@10, latency.
- **Full table**: `*_results_table.md`.

## Environment quick reference

| Variable | Role |
|----------|------|
| `RETRIEVE_EVAL_WORKSPACE` | Root with `obs.csv`, `queries.jsonl`, `indices/` |
| `RETRIEVE_BENCHMARK_PROFILE` | `exact` or `short` (focused MD for Task 1/2) |
| `RETRIEVE_DEV_JSONL` | Source `dev_data.jsonl` for Task 1 prepare scripts |
| `PERQA_SPLADE_MODEL_TYPE_OR_PATH` | Optional; if unset/invalid, eval falls back to `resolved_splade_model_type_or_path()` |

## Dynamic Fusion (implementation note)

`DynamicFusionOurs` uses **per-query weighted score fusion** (`reqap_modular_retrieval/routing/query_router.py` + `fusion/weighted_sum.py`): min–max normalize `bm25_score` / `splade_score` / `dense_score`, then combine with router weights (short vs structured vs default). Grid search in `eval_perqa_retrieval_export.py` can override with fixed `w1_bm25` / `w2_dense` / `w3_splade`.
