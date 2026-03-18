"""
扫描 QU dev_data.jsonl，统计 RETRIEVE(query="...") 出现次数，供 Dynamic Fusion 分桶。
支持缓存到 JSON 文件，避免重复扫描。
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, Iterable

_RETRIEVE = re.compile(r'RETRIEVE\s*\(\s*query\s*=\s*"((?:[^"\\]|\\.)*)"\s*\)')


def _unescape_inner(s: str) -> str:
    return s.replace("\\\\", "\\").replace('\\"', '"')


def _iter_json_strings(obj: Any) -> Iterable[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, list):
        for x in obj:
            yield from _iter_json_strings(x)
    elif isinstance(obj, dict):
        for x in obj.values():
            yield from _iter_json_strings(x)


def count_retrieve_queries(qu_dev_jsonl: str) -> Dict[str, int]:
    """从 QU dev_data.jsonl 中提取 RETRIEVE(query="...") 并计数。"""
    counts: Dict[str, int] = defaultdict(int)
    
    if not os.path.isfile(qu_dev_jsonl):
        print(f"警告：QU dev 文件不存在 {qu_dev_jsonl}")
        return {}
    
    try:
        with open(qu_dev_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for s in _iter_json_strings(row):
                    for m in _RETRIEVE.finditer(s):
                        inner = m.group(1)
                        q = _unescape_inner(inner).strip()
                        if q:
                            counts[q] += 1
    except Exception as e:
        print(f"错误读取 QU 文件 {qu_dev_jsonl}: {e}")
    
    return dict(counts)


def save_retrieve_query_counts(qu_dev_jsonl: str, cache_path: str) -> Dict[str, int]:
    """生成并保存 RETRIEVE 查询频率计数到 JSON 文件（与 load_or_build 使用相同带 meta 的格式）。"""
    return load_or_build_retrieve_query_counts(qu_dev_jsonl, cache_path, force_rebuild=True)


def load_retrieve_query_counts(cache_path: str) -> Dict[str, int]:
    """从缓存文件加载查询频率计数。"""
    if os.path.isfile(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict) and "counts" in payload:
                return dict(payload["counts"])
            if isinstance(payload, dict):
                return dict(payload)
        except Exception as e:
            print(f"加载缓存失败 {cache_path}: {e}")
    return {}


def load_or_build_retrieve_query_counts(
    source_jsonl: str,
    cache_path: str,
    *,
    force_rebuild: bool = False,
) -> Dict[str, int]:
    """
    从 QU dev_data.jsonl 统计 RETRIEVE 查询频率；若缓存存在且与源文件 mtime 一致则直接加载，避免每次启动全表扫描。
    缓存格式：{"meta": {"source_path", "source_mtime"}, "counts": {...}}
    环境变量 FORCE_QU_RETRIEVE_COUNTS=1 时强制重新统计并覆盖缓存。
    """
    source_jsonl = os.path.normpath(os.path.abspath(source_jsonl))
    if not os.path.isfile(source_jsonl):
        print(f"警告：QU 源文件不存在 {source_jsonl}")
        return {}

    force = force_rebuild or os.environ.get("FORCE_QU_RETRIEVE_COUNTS", "").strip() in (
        "1",
        "true",
        "yes",
    )

    def write_cache(counts: Dict[str, int]) -> None:
        d = os.path.dirname(cache_path)
        if d:
            os.makedirs(d, exist_ok=True)
        meta = {
            "source_path": source_jsonl,
            "source_mtime": os.path.getmtime(source_jsonl),
        }
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"meta": meta, "counts": counts}, f, ensure_ascii=False, indent=2)

    src_mtime = os.path.getmtime(source_jsonl)

    if not force and os.path.isfile(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict) and "counts" in payload:
                meta = payload.get("meta") or {}
                if (
                    os.path.normpath(os.path.abspath(str(meta.get("source_path", "")))) == source_jsonl
                    and float(meta.get("source_mtime", -1)) == src_mtime
                ):
                    c = dict(payload["counts"])
                    print(f"已加载查询频率缓存 {cache_path}（{len(c)} 个唯一查询，源未变更）")
                    return c
            elif isinstance(payload, dict) and payload and "meta" not in payload:
                # 旧版纯 counts 字典：仅当缓存不早于源文件时沿用
                if os.path.getmtime(cache_path) >= src_mtime:
                    c = {k: int(v) for k, v in payload.items() if isinstance(k, str)}
                    print(f"已加载查询频率缓存（旧格式）{cache_path}（{len(c)} 个唯一查询）")
                    return c
        except Exception as e:
            print(f"读取查询频率缓存失败，将重新统计: {e}")

    counts = count_retrieve_queries(source_jsonl)
    write_cache(counts)
    print(f"已写入查询频率缓存 {cache_path}（{len(counts)} 个唯一查询）")
    return counts
