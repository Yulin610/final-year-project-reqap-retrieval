"""
Train SPLADE (sparse retriever) with an in-batch negatives contrastive objective.

Goal: mirror the dense Stage1/Stage2 idea, but for SPLADE:
  - Stage 1 (recommended): train on retrieve/train_data.jsonl positive pairs (query, doc)
  - Stage 2 (optional): adapt on queries_train_p*.jsonl by pairing (query, relevant_event_text)

Notes:
  - SPLADE is an MLM-based sparse encoder; regularization is important to keep representations sparse.
  - This script uses a simple FLOPS-style regularizer by default.

Output:
  - A HuggingFace-compatible directory (model + tokenizer) that can be loaded by
    `reqap.retrieval.splade.models.Splade(<output_dir>, agg="max")`.

Typical usage (Stage 1 only):
  python train_splade_retriever.py ^
    --stage1-retrieve-train-jsonl "C:\\Users\\...\\data\\retrieve\\train_data.jsonl" ^
    --stage1-base-model "naver/splade-cocondenser-ensembledistil" ^
    --final-output-dir "C:\\Users\\...\\splade_adapted_phase1" ^
    --epochs 2 --batch-size 16 --lr 2e-5

Bucket training (Task T1: L_short + L_long + λ L_sparse, shared encoder):
  python train_splade_retriever.py ^
    --stage1-retrieve-train-jsonl ... ^
    --bucket-training ^
    --final-output-dir ...

Stage 1 + Stage 2:
  python train_splade_retriever.py ^
    --stage1-retrieve-train-jsonl ... ^
    --stage2-queries-train-glob "C:\\Users\\...\\queries_train_p*.jsonl" ^
    --stage2-obs-csv-template "C:\\Users\\...\\train_persona_{persona_id}\\train_persona_{persona_id}_obs.csv" ^
    --final-output-dir ...
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
from glob import glob
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

_EXAMPLES = Path(__file__).resolve().parent
# workspace root is .../ReQAP-main (the folder that contains both ReQAP-main/ and reqap-retrieval-modular/)
_WORKSPACE = _EXAMPLES.parent.parent
_REQAP_MAIN = _WORKSPACE / "ReQAP-main"
_REQAP_RETRIEVAL_MODULAR = _WORKSPACE / "reqap-retrieval-modular"

# Ensure imports work no matter where the script is executed from.
for p in (_EXAMPLES, _REQAP_MAIN, _REQAP_RETRIEVAL_MODULAR):
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from reqap.retrieval.splade.models import Splade
from reqap.library.text import get_doc_text, get_query_text


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


def _persona_id_from_filename(path: str) -> Optional[int]:
    m = re.search(r"_p(\d+)\.jsonl$", os.path.basename(path))
    return int(m.group(1)) if m else None


def _load_obs_id_to_event_data_str(obs_csv: str) -> Dict[int, str]:
    """
    Build mapping obs_id -> event_data (string).
    This matches the default SPLADE indexing path in this repo where CollectionDataset uses row["event_data"].
    """
    out: Dict[int, str] = {}
    with open(obs_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                did = int(row["id"])
            except Exception:
                continue
            out[did] = get_doc_text(row.get("event_data", "") or "", verbalize_fallback=True)
    return out


def build_bucket_pairs_from_retrieve_train_jsonl(
    retrieve_train_jsonl: str,
    *,
    max_pos_pairs: Optional[int],
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Split Stage-1 positives into two query regimes (shared encoder, dual CE):

    - **short**: query = input[0] only (Task B / short-intent style); doc = verbalized payload.
    - **long**: query = input[0] + \"\\n\" + input[1] (Task A full event string); same doc.

    Bucket assignment for logging follows `is_short_query` on the **long** query string.
    """
    short_pairs: List[Tuple[str, str]] = []
    long_pairs: List[Tuple[str, str]] = []
    for row in _parse_jsonl_lines(retrieve_train_jsonl):
        inp = row.get("input")
        if not isinstance(inp, list) or len(inp) < 2:
            continue
        if not bool(row.get("positive")):
            continue
        q_short = get_query_text(inp[0])
        q_long = f"{inp[0]}\n{inp[1]}"
        d = get_doc_text(inp[1], verbalize_fallback=True)
        short_pairs.append((q_short, d))
        long_pairs.append((q_long, d))
        if max_pos_pairs is not None and len(short_pairs) >= max_pos_pairs:
            break
    return short_pairs, long_pairs


def build_stage1_pairs(retrieve_train_jsonl: str, *, max_pos_pairs: Optional[int]) -> List[Tuple[str, str]]:
    """
    From retrieve/train_data.jsonl: take only rows with positive=True, use (input[0], input[1]).
    """
    pairs: List[Tuple[str, str]] = []
    for row in _parse_jsonl_lines(retrieve_train_jsonl):
        inp = row.get("input")
        if not isinstance(inp, list) or len(inp) < 2:
            continue
        if not bool(row.get("positive")):
            continue
        q = get_query_text(inp[0])
        d = get_doc_text(inp[1], verbalize_fallback=True)
        pairs.append((q, d))
        if max_pos_pairs is not None and len(pairs) >= max_pos_pairs:
            break
    return pairs


def load_grouped_hardneg_from_retrieve_jsonl(
    retrieve_train_jsonl: str,
    *,
    max_queries: Optional[int],
    max_pos_per_query: Optional[int],
    max_neg_per_query: Optional[int],
    seed: int,
) -> Dict[str, Dict[str, List[str]]]:
    """
    Group retrieve/train_data.jsonl into:
      groups[query]["pos"] = [doc, ...]
      groups[query]["neg"] = [doc, ...]

    This uses BOTH positive:true and positive:false rows, enabling "1 pos + N hard neg" batches.
    """
    rng = random.Random(seed)
    groups: Dict[str, Dict[str, List[str]]] = {}

    for row in _parse_jsonl_lines(retrieve_train_jsonl):
        inp = row.get("input")
        if not isinstance(inp, list) or len(inp) < 2:
            continue
        query = get_query_text(inp[0])
        doc = get_doc_text(inp[1], verbalize_fallback=True).strip()
        if not query or not doc:
            continue
        g = groups.get(query)
        if g is None:
            g = {"pos": [], "neg": []}
            groups[query] = g
        if bool(row.get("positive")):
            if max_pos_per_query is None or len(g["pos"]) < max_pos_per_query:
                g["pos"].append(doc)
        else:
            if max_neg_per_query is None or len(g["neg"]) < max_neg_per_query:
                g["neg"].append(doc)

        if max_queries is not None and len(groups) >= max_queries:
            break

    groups = {q: v for q, v in groups.items() if v["pos"] and v["neg"]}
    for v in groups.values():
        rng.shuffle(v["pos"])
        rng.shuffle(v["neg"])
    return groups


def build_stage2_pairs(
    queries_paths: Sequence[str],
    *,
    obs_csv_template: Optional[str],
    obs_csv_single: Optional[str],
    sample_relevant_k: int,
    max_queries: Optional[int],
    seed: int,
) -> List[Tuple[str, str]]:
    if not obs_csv_template and not obs_csv_single:
        raise ValueError("Stage2 requires obs csv via obs_csv_template or obs_csv_single.")

    rng = random.Random(seed)
    pairs: List[Tuple[str, str]] = []

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
            obs_cache[obs_csv] = _load_obs_id_to_event_data_str(obs_csv)
        obs_id_to_text = obs_cache[obs_csv]

        n_q = 0
        for row in _parse_jsonl_lines(qpath):
            n_q += 1
            if max_queries is not None and n_q > max_queries:
                break
            query = get_query_text(row.get("query", ""))
            rel_ids = row.get("relevant_ids", [])
            if not query or not isinstance(rel_ids, (list, set, tuple)):
                continue
            rel_list = list(rel_ids)
            if not rel_list:
                continue
            if len(rel_list) > sample_relevant_k:
                rel_list = rng.sample(rel_list, sample_relevant_k)
            for rid in rel_list:
                try:
                    did = int(rid)
                except Exception:
                    continue
                doc = obs_id_to_text.get(did)
                if not doc:
                    continue
                pairs.append((query, doc))
    return pairs


class PairDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        return self.pairs[idx]


class HardNegativeDataset(Dataset):
    """
    Each item yields one query with 1 positive and N hard negatives.
    """

    def __init__(self, grouped_data: Dict[str, Dict[str, List[str]]], *, num_neg: int, seed: int):
        self.queries = list(grouped_data.keys())
        self.data = grouped_data
        self.num_neg = num_neg
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        q = self.queries[idx]
        pos_list = self.data[q]["pos"]
        neg_list = self.data[q]["neg"]
        pos = self.rng.choice(pos_list)
        if len(neg_list) >= self.num_neg:
            negs = self.rng.sample(neg_list, self.num_neg)
        else:
            negs = [self.rng.choice(neg_list) for _ in range(self.num_neg)]
        return {"query": q, "positive": pos, "negatives": negs}


def _flops_regularizer(reps: torch.Tensor) -> torch.Tensor:
    """
    FLOPS regularizer (SPLADE-style):
      sum_j ( mean_i |rep_ij| )^2
    reps: (bs, vocab)
    """
    return (reps.abs().mean(dim=0) ** 2).sum()


def train_splade_contrastive(
    *,
    model: Splade,
    tokenizer,
    pairs: List[Tuple[str, str]],
    output_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    warmup_ratio: float,
    weight_decay: float,
    max_length: int,
    reg_lambda: float,
    reg_on: str,
    seed: int,
) -> None:
    if not pairs:
        raise ValueError("No training pairs.")

    _set_seed(seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.train(True)

    ds = PairDataset(pairs)

    def collate(batch: List[Tuple[str, str]]):
        qs, ds_ = zip(*batch)
        q_tokens = tokenizer(
            list(qs),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        d_tokens = tokenizer(
            list(ds_),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return q_tokens, d_tokens

    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0, collate_fn=collate)

    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = max(1, len(dl) * epochs)
    warmup_steps = int(total_steps * warmup_ratio)
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for ep in range(epochs):
        running = 0.0
        for step, (q_tokens, d_tokens) in enumerate(dl, start=1):
            q_tokens = {k: v.to(device) for k, v in q_tokens.items()}
            d_tokens = {k: v.to(device) for k, v in d_tokens.items()}

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(q_kwargs=q_tokens, d_kwargs=d_tokens, score_batch=True)
                scores = out["score"]  # (bs, bs)
                labels = torch.arange(scores.size(0), device=device)
                loss_main = F.cross_entropy(scores, labels)

                loss_reg = torch.tensor(0.0, device=device)
                if reg_lambda > 0:
                    reg_target = reg_on.lower().strip()
                    if reg_target in {"q", "both"}:
                        loss_reg = loss_reg + _flops_regularizer(out["q_rep"])
                    if reg_target in {"d", "both"}:
                        loss_reg = loss_reg + _flops_regularizer(out["d_rep"])

                loss = loss_main + reg_lambda * loss_reg

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()

            running += float(loss_main.detach().cpu())
            if step % 200 == 0:
                avg = running / 200.0
                running = 0.0
                print(
                    json.dumps(
                        {
                            "epoch": ep + 1,
                            "step": step,
                            "loss": round(avg, 4),
                            "lr": float(sched.get_last_lr()[0]),
                        },
                        ensure_ascii=False,
                    )
                )

    outp = Path(output_dir)
    outp.mkdir(parents=True, exist_ok=True)
    # Save as HF model dir so Splade(...) can load it via AutoModelForMaskedLM.from_pretrained(output_dir)
    model.transformer_rep.transformer.save_pretrained(str(outp))
    tokenizer.save_pretrained(str(outp))
    (outp / "splade_meta.json").write_text(
        json.dumps(
            {
                "agg": getattr(model, "agg", None),
                "reg_lambda": reg_lambda,
                "reg_on": reg_on,
                "max_length": max_length,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def train_splade_bucket_contrastive(
    *,
    model: Splade,
    tokenizer,
    short_pairs: List[Tuple[str, str]],
    long_pairs: List[Tuple[str, str]],
    output_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    warmup_ratio: float,
    weight_decay: float,
    max_length: int,
    reg_lambda: float,
    reg_on: str,
    seed: int,
) -> None:
    """
    Bucket training (Task T1): shared SPLADE encoder, two in-batch contrastive heads per step:

        L = L_short + L_long + λ * L_sparse

    where L_sparse is the existing FLOPS regularizer applied to q/d reps from **both** forwards
    (mean of the two regularizer values).
    """
    if not short_pairs or not long_pairs:
        raise ValueError("Bucket training requires non-empty short and long pair lists.")

    _set_seed(seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.train(True)

    ds_s = PairDataset(short_pairs)
    ds_l = PairDataset(long_pairs)

    def collate(batch: List[Tuple[str, str]]):
        qs, ds_ = zip(*batch)
        q_tokens = tokenizer(
            list(qs),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        d_tokens = tokenizer(
            list(ds_),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return q_tokens, d_tokens

    dl_s = DataLoader(ds_s, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0, collate_fn=collate)
    dl_l = DataLoader(ds_l, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0, collate_fn=collate)
    if len(dl_s) == 0 or len(dl_l) == 0:
        raise ValueError(
            "Bucket training: increase data or lower --batch-size so both short and long buckets "
            "have at least one full batch."
        )

    steps_per_epoch = min(len(dl_s), len(dl_l))
    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = max(1, steps_per_epoch * epochs)
    warmup_steps = int(total_steps * warmup_ratio)
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for ep in range(epochs):
        it_s = iter(dl_s)
        it_l = iter(dl_l)
        running = 0.0
        for step in range(1, steps_per_epoch + 1):
            q_tokens_s, d_tokens_s = next(it_s)
            q_tokens_l, d_tokens_l = next(it_l)

            q_tokens_s = {k: v.to(device) for k, v in q_tokens_s.items()}
            d_tokens_s = {k: v.to(device) for k, v in d_tokens_s.items()}
            q_tokens_l = {k: v.to(device) for k, v in q_tokens_l.items()}
            d_tokens_l = {k: v.to(device) for k, v in d_tokens_l.items()}

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                out_s = model(q_kwargs=q_tokens_s, d_kwargs=d_tokens_s, score_batch=True)
                scores_s = out_s["score"]
                labels_s = torch.arange(scores_s.size(0), device=device)
                loss_s = F.cross_entropy(scores_s, labels_s)

                out_l = model(q_kwargs=q_tokens_l, d_kwargs=d_tokens_l, score_batch=True)
                scores_l = out_l["score"]
                labels_l = torch.arange(scores_l.size(0), device=device)
                loss_l = F.cross_entropy(scores_l, labels_l)

                loss_reg = torch.tensor(0.0, device=device)
                if reg_lambda > 0:
                    reg_target = reg_on.lower().strip()
                    for out in (out_s, out_l):
                        if reg_target in {"q", "both"}:
                            loss_reg = loss_reg + _flops_regularizer(out["q_rep"])
                        if reg_target in {"d", "both"}:
                            loss_reg = loss_reg + _flops_regularizer(out["d_rep"])
                    loss_reg = loss_reg / 2.0

                loss = loss_s + loss_l + reg_lambda * loss_reg

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()

            running += float((loss_s + loss_l).detach().cpu()) / 2.0
            if step % 200 == 0:
                avg = running / 200.0
                running = 0.0
                print(
                    json.dumps(
                        {
                            "epoch": ep + 1,
                            "step": step,
                            "loss_short+long_avg": round(avg, 4),
                            "lr": float(sched.get_last_lr()[0]),
                            "bucket": "short+long",
                        },
                        ensure_ascii=False,
                    )
                )

    outp = Path(output_dir)
    outp.mkdir(parents=True, exist_ok=True)
    model.transformer_rep.transformer.save_pretrained(str(outp))
    tokenizer.save_pretrained(str(outp))
    (outp / "splade_meta.json").write_text(
        json.dumps(
            {
                "agg": getattr(model, "agg", None),
                "reg_lambda": reg_lambda,
                "reg_on": reg_on,
                "max_length": max_length,
                "bucket_training": True,
                "short_pairs": len(short_pairs),
                "long_pairs": len(long_pairs),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def train_splade_hardneg(
    *,
    model: Splade,
    tokenizer,
    grouped: Dict[str, Dict[str, List[str]]],
    output_dir: str,
    epochs: int,
    batch_queries: int,
    num_neg: int,
    lr: float,
    warmup_ratio: float,
    weight_decay: float,
    max_length: int,
    reg_lambda: float,
    reg_on: str,
    seed: int,
) -> None:
    """
    Hard-negative in-batch training.

    For each step with B queries:
      q_batch = [q1..qB]
      d_batch = [pos1..posB, neg(q1)_1..neg(q1)_N, ..., neg(qB)_N]
    Score matrix shape: (B, B + B*N)
    Labels are [0..B-1], i.e. each query's positive is at the same index in the first B docs.
    """
    if not grouped:
        raise ValueError("No grouped data for hard-negative training.")

    _set_seed(seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.train(True)

    ds = HardNegativeDataset(grouped, num_neg=num_neg, seed=seed)

    def collate(batch: List[Dict[str, object]]):
        queries: List[str] = []
        docs: List[str] = []
        for item in batch:
            q = str(item["query"])
            pos = str(item["positive"])
            negs = list(item["negatives"])  # type: ignore[arg-type]
            queries.append(q)
            docs.append(pos)
            docs.extend([str(x) for x in negs])

        q_tokens = tokenizer(
            queries,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        d_tokens = tokenizer(
            docs,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return q_tokens, d_tokens, len(queries)

    dl = DataLoader(ds, batch_size=batch_queries, shuffle=True, drop_last=True, num_workers=0, collate_fn=collate)

    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = max(1, len(dl) * epochs)
    warmup_steps = int(total_steps * warmup_ratio)
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for ep in range(epochs):
        running = 0.0
        for step, (q_tokens, d_tokens, bq) in enumerate(dl, start=1):
            q_tokens = {k: v.to(device) for k, v in q_tokens.items()}
            d_tokens = {k: v.to(device) for k, v in d_tokens.items()}

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(q_kwargs=q_tokens, d_kwargs=d_tokens, score_batch=True)
                scores = out["score"]  # (Bq, Bd)
                labels = torch.arange(int(bq), device=device)
                loss_main = F.cross_entropy(scores, labels)

                loss_reg = torch.tensor(0.0, device=device)
                if reg_lambda > 0:
                    reg_target = reg_on.lower().strip()
                    if reg_target in {"q", "both"}:
                        loss_reg = loss_reg + _flops_regularizer(out["q_rep"])
                    if reg_target in {"d", "both"}:
                        loss_reg = loss_reg + _flops_regularizer(out["d_rep"])

                loss = loss_main + reg_lambda * loss_reg

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()

            running += float(loss_main.detach().cpu())
            if step % 200 == 0:
                avg = running / 200.0
                running = 0.0
                print(
                    json.dumps(
                        {
                            "epoch": ep + 1,
                            "step": step,
                            "loss": round(avg, 4),
                            "lr": float(sched.get_last_lr()[0]),
                            "batch_queries": int(bq),
                            "num_neg": int(num_neg),
                            "docs_per_step": int(bq) * (1 + int(num_neg)),
                        },
                        ensure_ascii=False,
                    )
                )

    outp = Path(output_dir)
    outp.mkdir(parents=True, exist_ok=True)
    model.transformer_rep.transformer.save_pretrained(str(outp))
    tokenizer.save_pretrained(str(outp))
    (outp / "splade_meta.json").write_text(
        json.dumps(
            {
                "agg": getattr(model, "agg", None),
                "reg_lambda": reg_lambda,
                "reg_on": reg_on,
                "max_length": max_length,
                "hard_negatives": {"batch_queries": batch_queries, "num_neg": num_neg},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Train SPLADE retriever with in-batch negatives + sparsity regularization.")
    ap.add_argument("--stage1-retrieve-train-jsonl", default="", help="Path to retrieve/train_data.jsonl (recommended).")
    ap.add_argument("--stage1-max-pos-pairs", type=int, default=0, help="0 means no cap.")
    ap.add_argument(
        "--stage1-use-hard-negatives",
        action="store_true",
        help="Use positive:false rows in retrieve/train_data.jsonl as hard negatives (requires pos+neg per query).",
    )
    ap.add_argument("--hn-batch-queries", type=int, default=8, help="Number of queries per step (B).")
    ap.add_argument("--hn-num-neg", type=int, default=4, help="Hard negatives per query (N).")
    ap.add_argument("--hn-max-queries", type=int, default=0, help="0 means no cap when grouping by query.")
    ap.add_argument("--hn-max-pos-per-query", type=int, default=0, help="0 means no cap.")
    ap.add_argument("--hn-max-neg-per-query", type=int, default=0, help="0 means no cap.")

    ap.add_argument("--stage2-queries-train-glob", default="", help="Glob for queries_train_p*.jsonl (optional).")
    ap.add_argument("--stage2-obs-csv-single", default="", help="Use one obs.csv for all queries (optional).")
    ap.add_argument(
        "--stage2-obs-csv-template",
        default="",
        help="Template with {persona_id} for obs csv (optional).",
    )
    ap.add_argument("--stage2-sample-relevant-k", type=int, default=10)
    ap.add_argument("--stage2-max-queries", type=int, default=0, help="0 means no cap.")

    ap.add_argument("--stage1-base-model", default="naver/splade-cocondenser-ensembledistil")
    ap.add_argument("--tokenizer", default="bert-base-uncased", help="Tokenizer for SPLADE training.")
    ap.add_argument("--agg", default="max", choices=["max", "sum"])

    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup-ratio", type=float, default=0.06)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-length", type=int, default=256)

    ap.add_argument("--reg-lambda", type=float, default=3e-5, help="Sparsity regularization coefficient.")
    ap.add_argument("--reg-on", default="both", choices=["q", "d", "both"], help="Apply sparsity reg on q/d/both.")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--final-output-dir", required=True, help="Output dir to save final SPLADE model.")
    ap.add_argument(
        "--bucket-training",
        action="store_true",
        help="Task T1: dual in-batch loss L_short+L_long on retrieve/train_data (short query=input[0], long=full event). "
        "Incompatible with --stage1-use-hard-negatives in this MVP.",
    )
    args = ap.parse_args()

    if args.bucket_training and not args.stage1_retrieve_train_jsonl.strip():
        raise SystemExit("--bucket-training requires --stage1-retrieve-train-jsonl")

    stage1_max = args.stage1_max_pos_pairs if args.stage1_max_pos_pairs and args.stage1_max_pos_pairs > 0 else None
    stage2_max_q = args.stage2_max_queries if args.stage2_max_queries and args.stage2_max_queries > 0 else None
    hn_max_queries = args.hn_max_queries if args.hn_max_queries and args.hn_max_queries > 0 else None
    hn_max_pos = args.hn_max_pos_per_query if args.hn_max_pos_per_query and args.hn_max_pos_per_query > 0 else None
    hn_max_neg = args.hn_max_neg_per_query if args.hn_max_neg_per_query and args.hn_max_neg_per_query > 0 else None

    pairs: List[Tuple[str, str]] = []
    grouped_hn: Dict[str, Dict[str, List[str]]] = {}

    bucket_short: List[Tuple[str, str]] = []
    bucket_long: List[Tuple[str, str]] = []

    if args.stage1_retrieve_train_jsonl.strip():
        if args.bucket_training and args.stage1_use_hard_negatives:
            raise SystemExit("Use either --bucket-training or --stage1-use-hard-negatives (not both) in this MVP.")
        if args.stage1_use_hard_negatives:
            print("Stage 1: grouping retrieve/train_data.jsonl for hard-negative training ...")
            grouped_hn = load_grouped_hardneg_from_retrieve_jsonl(
                args.stage1_retrieve_train_jsonl,
                max_queries=hn_max_queries,
                max_pos_per_query=hn_max_pos,
                max_neg_per_query=hn_max_neg,
                seed=args.seed,
            )
            print(f"Stage 1: grouped queries (with pos+neg) = {len(grouped_hn)}")
        elif args.bucket_training:
            print("Stage 1: bucket pairs (short vs long query) from retrieve/train_data.jsonl ...")
            bucket_short, bucket_long = build_bucket_pairs_from_retrieve_train_jsonl(
                args.stage1_retrieve_train_jsonl,
                max_pos_pairs=stage1_max,
            )
            print(f"Stage 1: short bucket pairs = {len(bucket_short)}  long bucket pairs = {len(bucket_long)}")
        else:
            print("Stage 1: building (query, doc) positive pairs from retrieve/train_data.jsonl ...")
            p1 = build_stage1_pairs(args.stage1_retrieve_train_jsonl, max_pos_pairs=stage1_max)
            print(f"Stage 1: pairs = {len(p1)}")
            pairs.extend(p1)

    if args.stage2_queries_train_glob.strip():
        qpaths = sorted(glob(args.stage2_queries_train_glob))
        if not qpaths:
            raise SystemExit(f"Stage2: glob matched nothing: {args.stage2_queries_train_glob}")
        print("Stage 2: building (query, relevant_event) pairs from queries_train_p*.jsonl ...")
        obs_single = args.stage2_obs_csv_single.strip() or None
        obs_template = args.stage2_obs_csv_template.strip() or None
        p2 = build_stage2_pairs(
            qpaths,
            obs_csv_template=obs_template,
            obs_csv_single=obs_single,
            sample_relevant_k=args.stage2_sample_relevant_k,
            max_queries=stage2_max_q,
            seed=args.seed,
        )
        print(f"Stage 2: pairs = {len(p2)}")
        pairs.extend(p2)

    if args.bucket_training and args.stage2_queries_train_glob.strip():
        raise SystemExit("In this MVP, --bucket-training cannot be combined with Stage 2 PerQA pairs.")

    if not pairs and not grouped_hn and not (args.bucket_training and bucket_short and bucket_long):
        raise SystemExit(
            "No training data: provide --stage1-retrieve-train-jsonl and/or --stage2-queries-train-glob "
            "(enable --stage1-use-hard-negatives for hard-neg batches, or --bucket-training for dual-bucket SPLADE)"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading base SPLADE model on {device} ...")
    model = Splade(args.stage1_base_model, agg=args.agg)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, clean_up_tokenization_spaces=True)

    if grouped_hn:
        print("Training SPLADE (hard negatives: 1 pos + N hard neg per query) ...")
        train_splade_hardneg(
            model=model,
            tokenizer=tokenizer,
            grouped=grouped_hn,
            output_dir=args.final_output_dir,
            epochs=args.epochs,
            batch_queries=args.hn_batch_queries,
            num_neg=args.hn_num_neg,
            lr=args.lr,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            max_length=args.max_length,
            reg_lambda=args.reg_lambda,
            reg_on=args.reg_on,
            seed=args.seed,
        )
    elif args.bucket_training:
        print("Training SPLADE (bucket: L_short + L_long + λ L_sparse) ...")
        train_splade_bucket_contrastive(
            model=model,
            tokenizer=tokenizer,
            short_pairs=bucket_short,
            long_pairs=bucket_long,
            output_dir=args.final_output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            max_length=args.max_length,
            reg_lambda=args.reg_lambda,
            reg_on=args.reg_on,
            seed=args.seed,
        )
    else:
        print("Training SPLADE (classic in-batch negatives) ...")
        train_splade_contrastive(
            model=model,
            tokenizer=tokenizer,
            pairs=pairs,
            output_dir=args.final_output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            max_length=args.max_length,
            reg_lambda=args.reg_lambda,
            reg_on=args.reg_on,
            seed=args.seed,
        )
    print(f"Saved final SPLADE model to: {args.final_output_dir}")


if __name__ == "__main__":
    main()

