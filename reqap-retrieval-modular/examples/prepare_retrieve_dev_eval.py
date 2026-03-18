"""
从 data/retrieve/dev_data.jsonl 生成 ReQAP 可用的 obs.csv 与 queries.jsonl（每行一条文档，qid 对应行号）。
"""
import csv
import json
import os
import sys

sys.path.insert(0, r"C:\Users\23369\Desktop\final_work\ReQAP-main\reqap-retrieval-modular")

DEV_JSONL_DEFAULT = r"C:\Users\23369\Desktop\final_work\data\retrieve\dev_data.jsonl"
WORKSPACE_DEFAULT = r"C:\Users\23369\Desktop\final_work\data\retrieve\eval_workspace"


def row_to_query_text(row: dict) -> str:
    inp = row.get("input")
    if isinstance(inp, list) and len(inp) >= 2:
        return f"{inp[0]}\n{inp[1]}"
    if isinstance(inp, list) and len(inp) == 1:
        return str(inp[0])
    return json.dumps(inp, ensure_ascii=False)


def row_to_query_key(row: dict) -> str:
    """与 QU 数据里 RETRIEVE(query=...) 对齐的短查询键（通常为 pattern 首段）。"""
    inp = row.get("input")
    if isinstance(inp, list) and len(inp) >= 1:
        return str(inp[0]).strip()
    return row_to_query_text(row).strip()[:200]


def prepare(dev_jsonl: str, workspace: str) -> None:
    os.makedirs(workspace, exist_ok=True)
    obs_csv = os.path.join(workspace, "obs.csv")
    queries_jsonl = os.path.join(workspace, "queries.jsonl")

    with open(dev_jsonl, "r", encoding="utf-8") as f_in, open(
        obs_csv, "w", encoding="utf-8", newline=""
    ) as f_obs, open(queries_jsonl, "w", encoding="utf-8") as f_q:
        w = csv.DictWriter(f_obs, fieldnames=["id", "event_type", "event_data"])
        w.writeheader()
        doc_id = 0
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            event_data = {
                "retrieve_dev": True,
                "positive": row.get("positive"),
                "input_type": row.get("input_type"),
                "label": row.get("label"),
                "text": row_to_query_text(row),
            }
            w.writerow(
                {
                    "id": doc_id,
                    "event_type": "retrieve_dev",
                    "event_data": json.dumps(event_data, ensure_ascii=False),
                }
            )
            f_q.write(
                json.dumps(
                    {
                        "qid": f"q{doc_id}",
                        "query": row_to_query_text(row),
                        "query_key": row_to_query_key(row),
                        "relevant_ids": [doc_id],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            doc_id += 1

    print(f"Wrote {obs_csv} and {queries_jsonl}")


if __name__ == "__main__":
    dev = os.environ.get("RETRIEVE_DEV_JSONL", DEV_JSONL_DEFAULT)
    ws = os.environ.get("RETRIEVE_EVAL_WORKSPACE", WORKSPACE_DEFAULT)
    prepare(dev, ws)
