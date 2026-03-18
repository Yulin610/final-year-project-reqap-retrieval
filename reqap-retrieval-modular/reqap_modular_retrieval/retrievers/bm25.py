from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from loguru import logger

from reqap.retrieval.splade.index_construction import CollectionDataset

from ..core.types import RetrievedDoc, Retriever


def _looks_cjk(ch: str) -> bool:
    o = ord(ch)
    return (
        0x4E00 <= o <= 0x9FFF  # CJK Unified Ideographs
        or 0x3400 <= o <= 0x4DBF  # CJK Extension A
        or 0x3040 <= o <= 0x30FF  # Japanese Hiragana/Katakana
        or 0xAC00 <= o <= 0xD7AF  # Hangul
    )


def tokenize_mixed(text: str) -> List[str]:
    """
    Simple tokenizer that works reasonably for both English and CJK.
    - English: split on non-alnum, lowercase
    - CJK: keep characters as tokens (helps literal matching for proper nouns)
    """
    tokens: List[str] = []
    buf: List[str] = []

    def flush_buf():
        nonlocal buf
        if buf:
            word = "".join(buf).lower()
            if word:
                tokens.append(word)
            buf = []

    for ch in text:
        if _looks_cjk(ch):
            flush_buf()
            if not ch.isspace():
                tokens.append(ch)
        elif ch.isalnum():
            buf.append(ch)
        else:
            flush_buf()

    flush_buf()
    return tokens


def default_event_to_text(event_row: Dict) -> str:
    event_type = event_row.get("event_type", "")
    event_data = event_row.get("event_data", "")
    if isinstance(event_data, str):
        try:
            event_data_obj = json.loads(event_data)
        except Exception:
            event_data_obj = {}
    else:
        event_data_obj = event_data or {}

    parts = [str(event_type)]
    if isinstance(event_data_obj, dict):
        for k, v in event_data_obj.items():
            if v is None:
                continue
            parts.append(f"{k} {v}")
    else:
        parts.append(str(event_data_obj))
    return " ".join(parts)


@dataclass
class BM25Index:
    bm25: "bm25s.BM25"
    doc_ids: List[int]  # aligned with corpus docs


class BM25Retriever(Retriever):
    name = "bm25"

    def __init__(
        self,
        collection: CollectionDataset,
        *,
        tokenizer: Callable[[str], List[str]] = tokenize_mixed,
        event_to_text: Callable[[Dict], str] = default_event_to_text,
        index_dir: Optional[str] = None,
        bm25_params: Optional[Dict] = None,
    ):
        self._collection = collection
        self._tokenizer = tokenizer
        self._event_to_text = event_to_text
        self._bm25_params = bm25_params or {"k1": 1.5, "b": 0.75, "method": "lucene", "backend": "auto"}
        self._doc_by_id: Dict[int, Dict] = {}
        for i in range(len(self._collection)):
            row = self._collection[i]["data"]
            self._doc_by_id[int(row["id"])] = row

        self._index: Optional[BM25Index] = None
        if index_dir:
            self.load(index_dir)

    def build(self, index_dir: str, *, show_progress: bool = True) -> None:
        import bm25s

        logger.info("Building BM25 index (bm25s)...")
        os.makedirs(index_dir, exist_ok=True)

        doc_ids: List[int] = []
        corpus_tokens: List[List[str]] = []
        corpus_texts: List[str] = []
        for i in range(len(self._collection)):
            doc = self._collection[i]["data"]
            doc_id = int(doc["id"])
            text = self._event_to_text(doc)
            tokens = self._tokenizer(text)
            doc_ids.append(doc_id)
            corpus_tokens.append(tokens)
            corpus_texts.append(text)

        # bm25s 在 save 时会把 corpus 写成 JSONL，仅接受 str/dict/list/tuple；勿传 int doc_id。
        bm25 = bm25s.BM25(**self._bm25_params)
        bm25.index(corpus_tokens, show_progress=show_progress)
        bm25.save(index_dir, corpus=corpus_texts)

        with open(os.path.join(index_dir, "doc_ids.json"), "w", encoding="utf-8") as f:
            json.dump(doc_ids, f, ensure_ascii=False, indent=2)

        self._index = BM25Index(bm25=bm25, doc_ids=doc_ids)
        logger.info(f"BM25 index built at {index_dir} with {len(doc_ids)} docs.")

    def load(self, index_dir: str) -> None:
        import bm25s

        bm25 = bm25s.BM25.load(index_dir, load_corpus=True, load_vocab=True)
        doc_ids_path = os.path.join(index_dir, "doc_ids.json")
        if os.path.exists(doc_ids_path):
            with open(doc_ids_path, "r", encoding="utf-8") as f:
                doc_ids = [int(x) for x in json.load(f)]
        else:
            # fallback: rely on saved corpus
            doc_ids = [int(x) for x in (bm25.corpus or [])]

        self._index = BM25Index(bm25=bm25, doc_ids=doc_ids)
        logger.info(f"Loaded BM25 index from {index_dir} with {len(doc_ids)} docs.")

    def retrieve(self, query: str, *, top_k: int, threshold: float = 0.0) -> List[RetrievedDoc]:
        if self._index is None:
            raise ValueError("BM25 index not loaded/built. Call build() or provide index_dir.")

        q_tokens = self._tokenizer(query)
        res = self._index.bm25.retrieve([q_tokens], corpus=self._index.doc_ids, k=top_k, sorted=True, show_progress=False)

        doc_ids: Sequence[int] = [int(x) for x in res.documents[0].tolist()]
        scores: Sequence[float] = [float(x) for x in res.scores[0].tolist()]

        out: List[RetrievedDoc] = []
        for doc_id, score in zip(doc_ids, scores):
            if score < threshold:
                continue
            data = self._doc_by_id.get(int(doc_id))
            if not data:
                continue
            out.append(
                RetrievedDoc(
                    id=int(doc_id),
                    score=float(score),
                    data=data,
                    signals={"bm25_score": float(score)},
                    derivation=[{"method": "bm25", "score": float(score)}],
                )
            )
        out.sort(key=lambda x: x.score, reverse=True)
        return out

