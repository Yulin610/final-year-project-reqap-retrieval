import argparse
import csv
import json
import os
import random
import re
from dataclasses import dataclass
from glob import glob
from typing import Dict, Iterable, List, Optional, Sequence

import torch
from sentence_transformers import InputExample, SentenceTransformer
from sentence_transformers.losses import MultipleNegativesRankingLoss
from torch.utils.data import DataLoader


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _parse_jsonl_lines(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _verbalize_obs_event(event_type: str, event_data_str: str) -> str:
    """
    Match DenseRetrieval._verbalize_event() as closely as possible.
    """
    try:
        event_data = (
            json.loads(event_data_str) if isinstance(event_data_str, str) else (event_data_str or {})
        )
    except Exception:
        event_data = {}

    parts = [f"Event type: {event_type}"]
    if isinstance(event_data, dict):
        for k, v in event_data.items():
            if v is None:
                continue
            if isinstance(v, (int, float, str)):
                parts.append(f"{k}: {v}")
    else:
        parts.append(str(event_data))
    return ". ".join(parts)


def _load_obs_id_to_text(obs_csv: str) -> Dict[int, str]:
    """
    Build mapping: obs_id -> verbalized event text.
    obs_csv must have columns: id,event_type,event_data (stringified JSON).
    """
    out: Dict[int, str] = {}
    with open(obs_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                doc_id = int(row["id"])
            except Exception:
                continue
            event_type = row.get("event_type", "") or ""
            event_data_str = row.get("event_data", "") or "{}"
            out[doc_id] = _verbalize_obs_event(event_type, event_data_str)
    return out


def _persona_id_from_filename(path: str) -> Optional[int]:
    """
    Expect patterns like queries_train_p0.jsonl / queries_train_p12.jsonl.
    """
    m = re.search(r"_p(\d+)\.jsonl$", os.path.basename(path))
    if not m:
        return None
    return int(m.group(1))


def build_stage1_examples(retrieve_train_jsonl: str, *, max_pos_pairs: Optional[int]) -> List[InputExample]:
    """
    Phase 1 uses retrieve/train_data.jsonl.
    With MultipleNegativesRankingLoss we only need positive (query, relevant_doc) pairs;
    negatives are handled implicitly by in-batch negatives.
    """
    examples: List[InputExample] = []
    for row in _parse_jsonl_lines(retrieve_train_jsonl):
        inp = row.get("input")
        if not isinstance(inp, list) or len(inp) < 2:
            continue
        positive = bool(row.get("positive"))
        if not positive:
            continue
        query = str(inp[0])
        doc = str(inp[1])
        examples.append(InputExample(texts=[query, doc]))
        if max_pos_pairs is not None and len(examples) >= max_pos_pairs:
            break
    return examples


def build_stage2_examples(
    queries_paths: Sequence[str],
    *,
    obs_csv_template: Optional[str],
    obs_csv_single: Optional[str],
    sample_relevant_k: int,
    max_queries: Optional[int],
    seed: int,
) -> List[InputExample]:
    if not obs_csv_template and not obs_csv_single:
        raise ValueError("Stage2 requires obs csv via obs_csv_template or obs_csv_single.")

    examples: List[InputExample] = []
    rng = random.Random(seed)

    # Cache obs_id_to_text per persona path.
    obs_cache: Dict[str, Dict[int, str]] = {}

    for qpath in queries_paths:
        persona_id = _persona_id_from_filename(qpath)
        if obs_csv_single:
            obs_csv = obs_csv_single
        else:
            if persona_id is None:
                raise ValueError(f"Cannot infer persona id from filename: {qpath}")
            if not obs_csv_template:
                raise ValueError("Missing obs_csv_template.")
            obs_csv = obs_csv_template.format(persona_id=persona_id)

        if obs_csv not in obs_cache:
            obs_cache[obs_csv] = _load_obs_id_to_text(obs_csv)
        obs_id_to_text = obs_cache[obs_csv]

        n_q = 0
        for row in _parse_jsonl_lines(qpath):
            n_q += 1
            if max_queries is not None and n_q > max_queries:
                break
            query = str(row.get("query", ""))
            rel_ids = row.get("relevant_ids", [])
            if not isinstance(rel_ids, list) and not isinstance(rel_ids, set):
                continue
            rel_ids_list = list(rel_ids)
            if not rel_ids_list:
                continue

            if len(rel_ids_list) > sample_relevant_k:
                rel_ids_list = rng.sample(rel_ids_list, sample_relevant_k)

            for rid in rel_ids_list:
                try:
                    did = int(rid)
                except Exception:
                    continue
                doc_text = obs_id_to_text.get(did)
                if doc_text is None:
                    continue
                examples.append(InputExample(texts=[query, doc_text]))
    return examples


def train_with_mnr(
    *,
    model: SentenceTransformer,
    examples: List[InputExample],
    epochs: int,
    batch_size: int,
    lr: float,
    warmup_ratio: float,
    weight_decay: float,
    output_dir: str,
    seed: int,
) -> None:
    if not examples:
        raise ValueError("No training examples.")

    _set_seed(seed)
    train_dataloader = DataLoader(examples, shuffle=True, batch_size=batch_size, drop_last=True)
    loss = MultipleNegativesRankingLoss(model)

    steps_per_epoch = max(1, len(train_dataloader))
    warmup_steps = int(steps_per_epoch * epochs * warmup_ratio)

    model.fit(
        train_objectives=[(train_dataloader, loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": lr, "weight_decay": weight_decay},
        output_path=output_dir,
        show_progress_bar=True,
        use_amp=torch.cuda.is_available(),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Train dense retriever (bi-encoder) with 2 stages.")
    ap.add_argument("--stage1-retrieve-train-jsonl", default="", help="Path to retrieve/train_data.jsonl")
    ap.add_argument("--stage1-base-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--stage1-output-dir", default="", help="Output dir for stage1 checkpoint/model.")
    ap.add_argument("--stage1-epochs", type=int, default=3)
    ap.add_argument("--stage1-batch-size", type=int, default=32)
    ap.add_argument("--stage1-lr", type=float, default=2e-5)
    ap.add_argument("--stage1-warmup-ratio", type=float, default=0.06)
    ap.add_argument("--stage1-weight-decay", type=float, default=0.01)
    ap.add_argument("--stage1-max-pos-pairs", type=int, default=0, help="0 means no cap")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--stage2-queries-train-glob", default="", help="Glob for queries_train_p*.jsonl (optional)")
    ap.add_argument("--stage2-obs-csv-single", default="", help="Use this obs.csv for all stage2 queries (optional)")
    ap.add_argument(
        "--stage2-obs-csv-template",
        default="",
        help="Template like '.../train_persona_{persona_id}/{persona_id}_obs.csv' (use {persona_id})",
    )
    ap.add_argument("--stage2-sample-relevant-k", type=int, default=10, help="Sample up to K relevant docs per query")
    ap.add_argument("--stage2-epochs", type=int, default=1)
    ap.add_argument("--stage2-batch-size", type=int, default=32)
    ap.add_argument("--stage2-lr", type=float, default=1e-5)
    ap.add_argument("--stage2-warmup-ratio", type=float, default=0.1)
    ap.add_argument("--stage2-weight-decay", type=float, default=0.01)
    ap.add_argument("--stage2-max-queries", type=int, default=0, help="0 means no cap")

    ap.add_argument("--final-output-dir", default="", help="Where to save final dense model")
    args = ap.parse_args()

    if not args.stage1_retrieve_train_jsonl:
        raise SystemExit("--stage1-retrieve-train-jsonl is required.")
    if not args.stage1_output_dir:
        raise SystemExit("--stage1-output-dir is required.")
    if not args.final_output_dir:
        raise SystemExit("--final-output-dir is required.")

    stage1_max = args.stage1_max_pos_pairs if args.stage1_max_pos_pairs and args.stage1_max_pos_pairs > 0 else None

    print("Stage 1: building positive pairs from retrieve/train_data.jsonl ...")
    stage1_examples = build_stage1_examples(args.stage1_retrieve_train_jsonl, max_pos_pairs=stage1_max)
    print(f"Stage 1: pos pairs = {len(stage1_examples)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(args.stage1_base_model, device=device)

    print("Stage 1: training (MultipleNegativesRankingLoss) ...")
    train_with_mnr(
        model=model,
        examples=stage1_examples,
        epochs=args.stage1_epochs,
        batch_size=args.stage1_batch_size,
        lr=args.stage1_lr,
        warmup_ratio=args.stage1_warmup_ratio,
        weight_decay=args.stage1_weight_decay,
        output_dir=args.stage1_output_dir,
        seed=args.seed,
    )
    model.save(args.stage1_output_dir)

    # Stage 2 (optional)
    if args.stage2_queries_train_glob.strip():
        query_paths = sorted(glob(args.stage2_queries_train_glob))
        if not query_paths:
            raise SystemExit(f"Stage2: glob matched nothing: {args.stage2_queries_train_glob}")

        obs_single = args.stage2_obs_csv_single.strip() or None
        obs_template = args.stage2_obs_csv_template.strip() or None
        max_queries = args.stage2_max_queries if args.stage2_max_queries and args.stage2_max_queries > 0 else None

        print("Stage 2: loading obs and building (query, relevant_event) pairs ...")
        stage2_examples = build_stage2_examples(
            query_paths,
            obs_csv_template=obs_template,
            obs_csv_single=obs_single,
            sample_relevant_k=args.stage2_sample_relevant_k,
            max_queries=max_queries,
            seed=args.seed,
        )
        print(f"Stage 2: pairs = {len(stage2_examples)}")

        print("Stage 2: training ...")
        stage2_out = os.path.join(args.final_output_dir, "stage2")
        train_with_mnr(
            model=model,
            examples=stage2_examples,
            epochs=args.stage2_epochs,
            batch_size=args.stage2_batch_size,
            lr=args.stage2_lr,
            warmup_ratio=args.stage2_warmup_ratio,
            weight_decay=args.stage2_weight_decay,
            output_dir=stage2_out,
            seed=args.seed,
        )

    print(f"Saving final dense model to: {args.final_output_dir}")
    os.makedirs(args.final_output_dir, exist_ok=True)
    model.save(args.final_output_dir)


if __name__ == "__main__":
    main()

