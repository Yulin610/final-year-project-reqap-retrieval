from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from reqap_modular_retrieval.routing.learned_router import LearnedRouterModel, grid_48, save_learned_router_model


def _load_rows(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            rows.append(json.loads(ln))
    return rows


def _train_val_split(n: int, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    nv = int(round(n * val_ratio))
    nv = max(1, min(n - 1, nv))
    return idx[nv:], idx[:nv]


def _build_xy(rows: List[Dict], feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    x = np.zeros((len(rows), len(feature_names)), dtype=np.float64)
    y = np.zeros((len(rows),), dtype=np.int64)
    for i, r in enumerate(rows):
        feats: Dict[str, float] = r["features"]
        x[i, :] = [float(feats.get(k, 0.0)) for k in feature_names]
        o = r["oracle"]
        y[i] = int(o.get("class_id", -1))
    return x, y


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    s = np.sum(e, axis=1, keepdims=True)
    s[s <= 0] = 1.0
    return e / s


def main() -> None:
    ap = argparse.ArgumentParser(description="Train linear regression learned router for Task C.")
    ap.add_argument("--oracle-jsonl", required=True)
    ap.add_argument("--out-model-json", required=True)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ridge-l2", type=float, default=1e-3)
    args = ap.parse_args()

    rows = _load_rows(args.oracle_jsonl)
    if len(rows) < 10:
        raise SystemExit("Too few oracle rows to train.")

    feature_names = sorted(rows[0]["features"].keys())
    x, y = _build_xy(rows, feature_names)
    if np.any(y < 0):
        raise SystemExit("Oracle jsonl missing class_id; rebuild with updated build_perqa_router_oracle.py")
    train_idx, val_idx = _train_val_split(len(rows), args.val_ratio, args.seed)

    x_train = x[train_idx]
    y_train = y[train_idx]
    x_val = x[val_idx]
    y_val = y[val_idx]

    mu = x_train.mean(axis=0)
    sd = x_train.std(axis=0)
    sd[sd < 1e-12] = 1.0
    xn_train = (x_train - mu) / sd
    xn_val = (x_val - mu) / sd

    # Train a simple softmax classifier over 48 grid classes.
    num_classes = 48
    if int(np.max(y_train)) >= num_classes:
        raise SystemExit(f"class_id out of range: max={int(np.max(y_train))} expected < {num_classes}")
    l2 = float(args.ridge_l2)
    lr = 0.1
    steps = 800
    rng = np.random.default_rng(args.seed)
    w = rng.normal(scale=0.01, size=(num_classes, xn_train.shape[1])).astype(np.float64)
    b = np.zeros((num_classes,), dtype=np.float64)

    y_onehot = np.zeros((xn_train.shape[0], num_classes), dtype=np.float64)
    y_onehot[np.arange(xn_train.shape[0]), y_train] = 1.0

    for _ in range(steps):
        logits = xn_train @ w.T + b[None, :]
        p = _softmax(logits)
        grad_logits = (p - y_onehot) / float(xn_train.shape[0])
        grad_w = grad_logits.T @ xn_train + l2 * w
        grad_b = np.sum(grad_logits, axis=0)
        w -= lr * grad_w
        b -= lr * grad_b

    logits_val = xn_val @ w.T + b[None, :]
    pred_val = np.argmax(logits_val, axis=1)
    acc = float(np.mean(pred_val == y_val))

    model = LearnedRouterModel(
        model_type="softmax_classifier_48",
        feature_names=feature_names,
        weights_matrix=w.tolist(),
        bias_vector=b.tolist(),
        feature_mean=mu.tolist(),
        feature_std=sd.tolist(),
        metadata={
            "oracle_jsonl": str(Path(args.oracle_jsonl).resolve()),
            "num_rows": len(rows),
            "num_train": int(len(train_idx)),
            "num_val": int(len(val_idx)),
            "val_acc": acc,
            "ridge_l2": l2,
            "seed": int(args.seed),
            "num_classes": num_classes,
        },
        grid_table=grid_48(),
    )
    save_learned_router_model(model, args.out_model_json)
    print(f"Saved learned router model: {args.out_model_json}")
    print(f"Validation accuracy={acc:.4f}  (num_classes={num_classes})")


if __name__ == "__main__":
    main()
