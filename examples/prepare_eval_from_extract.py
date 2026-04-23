import json, csv, os, random, sys

# 保证可以导入原始 ReQAP 代码和本模块
sys.path.insert(0, r"C:\Users\23369\Desktop\final_work\ReQAP-main\ReQAP-main")
sys.path.insert(0, r"C:\Users\23369\Desktop\final_work\ReQAP-main\reqap-retrieval-modular")

ROOT = r"C:\Users\23369\Desktop\final_work"
EXTRACT_PATH = os.path.join(ROOT, "data", "extract", "dev_data_aliases.jsonl")

OUT_DIR = os.path.join(ROOT, "eval_from_extract")
OBS_CSV = os.path.join(OUT_DIR, "obs.csv")
QUERIES_JSONL = os.path.join(OUT_DIR, "queries.jsonl")

os.makedirs(OUT_DIR, exist_ok=True)

def main(max_samples=500):
    docs = []
    with open(EXTRACT_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            docs.append({"id": i, "input": row["input"], "output": row["output"]})
    random.seed(42)
    random.shuffle(docs)
    docs = docs[:max_samples]

    # 写成 ReQAP 的 obs.csv
    with open(OBS_CSV, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=["id", "event_type", "event_data"])
        writer.writeheader()
        for d in docs:
            event_type = "extract_event"
            event_data = {
                "raw_input": d["input"],
                "label": d["output"],
            }
            writer.writerow({
                "id": d["id"],
                "event_type": event_type,
                "event_data": json.dumps(event_data, ensure_ascii=False),
            })

    # 构造简单查询 & 标注
    with open(QUERIES_JSONL, "w", encoding="utf-8") as f_q:
        for d in docs:
            # 用 Attribute 和 label 构个 query，这里先保持英文
            text = d["input"]
            attr_line = text.split(", Event:")[0].replace("Attribute:", "").strip()
            q = f"What is the value of {attr_line}?"
            f_q.write(json.dumps({
                "qid": f"q{d['id']}",
                "query": q,
                "relevant_ids": [d["id"]],
            }, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()