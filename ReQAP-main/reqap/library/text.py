from __future__ import annotations

import json
from typing import Any, Dict, Optional


def _verbalize_event_kv(event_dict: Dict[str, Any]) -> str:
    """
    Simple, stable event verbalization for lexical retrievers (BM25/SPLADE).
    Keeps the representation close to key:value text used in retrieve/* data.
    """
    parts = []
    for k, v in event_dict.items():
        if v is None:
            continue
        parts.append(f"{k}: {json.dumps(v, ensure_ascii=False)}")
    return ",\n".join(parts).replace("_", " ")


def get_doc_text(event_data: Any, *, verbalize_fallback: bool = True) -> str:
    """
    Unify document text extraction from different CSV/JSON representations.

    Supports:
    - raw string that is itself a JSON object string (e.g. obs.csv event_data column)
    - already-parsed dict
    - other types -> str(...)

    Priority:
    1) if JSON/dict has "text" field and it's non-empty -> return it
    2) else if verbalize_fallback -> return a key:value verbalization
    3) else return the raw string form
    """
    if isinstance(event_data, dict):
        txt = event_data.get("text")
        if isinstance(txt, str) and txt.strip():
            return txt.strip()
        return _verbalize_event_kv(event_data) if verbalize_fallback else json.dumps(event_data, ensure_ascii=False)

    if isinstance(event_data, str):
        s = event_data.strip()
        if not s:
            return ""
        try:
            obj = json.loads(s)
        except Exception:
            return s
        if isinstance(obj, dict):
            txt = obj.get("text")
            if isinstance(txt, str) and txt.strip():
                return txt.strip()
            return _verbalize_event_kv(obj) if verbalize_fallback else s
        return s

    return str(event_data)


def get_query_text(query: Any) -> str:
    """
    Unify query text extraction.
    Today queries_dev_p*.jsonl and retrieve/train_data.jsonl already store plain strings,
    but this keeps the interface symmetric and future-proof.
    """
    if query is None:
        return ""
    return str(query).strip()

