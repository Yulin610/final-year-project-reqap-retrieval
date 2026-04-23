from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..core.types import RetrievedDoc


def _safe_float(x: object, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _normalize_weights(raw: Dict[str, float]) -> Dict[str, float]:
    bm = max(0.0, _safe_float(raw.get("bm25_score"), 0.0))
    sp = max(0.0, _safe_float(raw.get("splade_score"), 0.0))
    de = max(0.0, _safe_float(raw.get("dense_score"), 0.0))
    s = bm + sp + de
    if s <= 0:
        return {"bm25_score": 1.0 / 3, "splade_score": 1.0 / 3, "dense_score": 1.0 / 3}
    return {"bm25_score": bm / s, "splade_score": sp / s, "dense_score": de / s}


def _tokenize(text: str) -> List[str]:
    cur: List[str] = []
    out: List[str] = []
    for ch in text.lower():
        if ch.isalnum():
            cur.append(ch)
        else:
            if cur:
                out.append("".join(cur))
                cur = []
    if cur:
        out.append("".join(cur))
    return out


def _token_entropy(tokens: Sequence[str]) -> float:
    if not tokens:
        return 0.0
    freq: Dict[str, int] = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    n = float(len(tokens))
    h = 0.0
    for c in freq.values():
        p = c / n
        if p > 0:
            h -= p * math.log(p + 1e-12)
    return h


def _signal_scores(docs: Sequence[RetrievedDoc], key: str) -> List[float]:
    vals: List[float] = []
    for d in docs:
        vals.append(_safe_float(d.signals.get(key), 0.0))
    vals.sort(reverse=True)
    return vals


def _gap(scores: Sequence[float], i: int, j: int) -> float:
    if not scores:
        return 0.0
    a = scores[i] if i < len(scores) else scores[-1]
    b = scores[j] if j < len(scores) else scores[-1]
    return a - b


def extract_router_features(query: str, merged_docs: Sequence[RetrievedDoc]) -> Dict[str, float]:
    toks = _tokenize(query)
    uniq = len(set(toks))
    token_count = float(len(toks))
    char_count = float(len(query))
    nl_count = float(query.count("\n"))
    digit_ratio = (sum(ch.isdigit() for ch in query) / max(1, len(query)))
    upper_ratio = (sum(ch.isupper() for ch in query) / max(1, len(query)))

    bm25_scores = _signal_scores(merged_docs, "bm25_score")
    splade_scores = _signal_scores(merged_docs, "splade_score")
    dense_scores = _signal_scores(merged_docs, "dense_score")

    return {
        "bias": 1.0,
        "token_count": token_count,
        "char_count": char_count,
        "newline_count": nl_count,
        "avg_token_len": (sum(len(t) for t in toks) / max(1, len(toks))),
        "unique_token_ratio": (uniq / max(1, len(toks))),
        "token_entropy": _token_entropy(toks),
        "digit_ratio": float(digit_ratio),
        "upper_ratio": float(upper_ratio),
        "bm25_top1": bm25_scores[0] if bm25_scores else 0.0,
        "bm25_gap_1_2": _gap(bm25_scores, 0, 1),
        "bm25_gap_1_10": _gap(bm25_scores, 0, 9),
        "splade_top1": splade_scores[0] if splade_scores else 0.0,
        "splade_gap_1_2": _gap(splade_scores, 0, 1),
        "splade_gap_1_10": _gap(splade_scores, 0, 9),
        "dense_top1": dense_scores[0] if dense_scores else 0.0,
        "dense_gap_1_2": _gap(dense_scores, 0, 1),
        "dense_gap_1_10": _gap(dense_scores, 0, 9),
    }

def grid_48() -> List[Tuple[float, float, float]]:
    """Canonical 48-class weight table: (w_bm25, w_dense, w_splade)."""
    out: List[Tuple[float, float, float]] = []
    for w_dense in (0.2, 0.3, 0.4, 0.5):
        for w_splade in (0.3, 0.4, 0.5, 0.6):
            for w_bm25 in (0.1, 0.2, 0.3):
                out.append((w_bm25, w_dense, w_splade))
    return out


@dataclass
class LearnedRouterModel:
    model_type: str
    feature_names: List[str]
    # For model_type="softmax_classifier_48": W shape (48, d), b shape (48,)
    weights_matrix: List[List[float]]
    bias_vector: List[float]
    feature_mean: List[float]
    feature_std: List[float]
    metadata: Dict[str, object]
    grid_table: Optional[List[Tuple[float, float, float]]] = None

    def _zscore(self, features: Dict[str, float]) -> List[float]:
        x: List[float] = []
        for idx, name in enumerate(self.feature_names):
            v = _safe_float(features.get(name), 0.0)
            mu = self.feature_mean[idx] if idx < len(self.feature_mean) else 0.0
            sd = self.feature_std[idx] if idx < len(self.feature_std) else 1.0
            if abs(sd) < 1e-12:
                sd = 1.0
            x.append((v - mu) / sd)
        return x

    def predict_class(self, features: Dict[str, float]) -> int:
        if self.model_type != "softmax_classifier_48":
            raise ValueError(f"predict_class only supports softmax_classifier_48 (got {self.model_type})")
        x = self._zscore(features)
        best_i = 0
        best_logit = None
        for i, w in enumerate(self.weights_matrix):
            s = self.bias_vector[i] if i < len(self.bias_vector) else 0.0
            for j, xv in enumerate(x):
                s += (w[j] if j < len(w) else 0.0) * xv
            if best_logit is None or s > best_logit:
                best_logit = s
                best_i = i
        return int(best_i)

    def predict(self, features: Dict[str, float]) -> Dict[str, float]:
        if self.model_type == "softmax_classifier_48":
            table = self.grid_table or grid_48()
            cls = self.predict_class(features)
            cls = max(0, min(len(table) - 1, cls))
            w_bm25, w_dense, w_splade = table[cls]
            return _normalize_weights({"bm25_score": w_bm25, "dense_score": w_dense, "splade_score": w_splade})
        # Backward compatibility: older regression router models.
        x = self._zscore(features)
        out = [0.0, 0.0, 0.0]  # bm25, dense, splade
        for k in range(min(3, len(self.weights_matrix))):
            w = self.weights_matrix[k]
            s = self.bias_vector[k] if k < len(self.bias_vector) else 0.0
            for i, xv in enumerate(x):
                wi = w[i] if i < len(w) else 0.0
                s += wi * xv
            out[k] = s
        return _normalize_weights({"bm25_score": out[0], "dense_score": out[1], "splade_score": out[2]})

    def to_json(self) -> Dict[str, object]:
        return {
            "schema_version": 1,
            "model_type": self.model_type,
            "feature_names": self.feature_names,
            "weights_matrix": self.weights_matrix,
            "bias_vector": self.bias_vector,
            "feature_mean": self.feature_mean,
            "feature_std": self.feature_std,
            "metadata": self.metadata,
            "grid_table": self.grid_table,
        }

    @staticmethod
    def from_json(data: Dict[str, object]) -> "LearnedRouterModel":
        gt = data.get("grid_table")
        grid_table = None
        if isinstance(gt, list) and gt and isinstance(gt[0], (list, tuple)) and len(gt[0]) == 3:
            grid_table = [(float(a), float(b), float(c)) for a, b, c in gt]  # type: ignore[misc]
        return LearnedRouterModel(
            model_type=str(data.get("model_type", "linear_regression")),
            feature_names=[str(x) for x in data.get("feature_names", [])],
            weights_matrix=[[float(v) for v in row] for row in data.get("weights_matrix", [])],
            bias_vector=[float(v) for v in data.get("bias_vector", [])],
            feature_mean=[float(v) for v in data.get("feature_mean", [])],
            feature_std=[float(v) for v in data.get("feature_std", [])],
            metadata=dict(data.get("metadata", {})),
            grid_table=grid_table,
        )


def save_learned_router_model(model: LearnedRouterModel, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(model.to_json(), f, ensure_ascii=False, indent=2)


def load_learned_router_model(path: str) -> Optional[LearnedRouterModel]:
    p = Path(path)
    if not p.is_file():
        return None
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return LearnedRouterModel.from_json(data)
