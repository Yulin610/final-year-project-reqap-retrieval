import argparse
import json
import os
from typing import Dict, TextIO


def sanitize_tsv_text(s: str) -> str:
    """
    Lightning-IR TSV files must be one record per line, so we flatten newlines.
    """
    return s.replace("\r", " ").replace("\n", " ")


def open_out(path: str, mode: str) -> TextIO:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, mode, encoding="utf-8", newline="\n")


def process_split(
    jsonl_path: str,
    *,
    query2id: Dict[str, int],
    doc2id: Dict[str, int],
    out_queries_tsv: TextIO,
    out_corpus_tsv: TextIO,
    out_qrels_tsv: TextIO,
):
    next_qid = len(query2id)
    next_pid = len(doc2id)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            # Expected structure from your `retrieve/train_data.jsonl`
            # {
            #   "positive": bool,
            #   "input": [query_text, doc_text],
            #   ...
            # }
            input_pair = obj.get("input")
            if not isinstance(input_pair, list) or len(input_pair) < 2:
                raise ValueError(f"Bad `input` format in {jsonl_path}: {obj.keys()}")

            query_text = sanitize_tsv_text(str(input_pair[0]))
            doc_text = sanitize_tsv_text(str(input_pair[1]))

            is_positive = bool(obj.get("positive", False))
            if query_text not in query2id:
                query2id[query_text] = next_qid
                out_queries_tsv.write(f"{next_qid}\t{query_text}\n")
                next_qid += 1
            if doc_text not in doc2id:
                doc2id[doc_text] = next_pid
                out_corpus_tsv.write(f"{next_pid}\t{doc_text}\n")
                next_pid += 1

            if is_positive:
                qid = query2id[query_text]
                pid = doc2id[doc_text]
                out_qrels_tsv.write(f"{qid}\t{pid}\t1\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_jsonl",
        default=r"..\..\data\retrieve\train_data.jsonl",
        help="Path to retrieve/train_data.jsonl",
    )
    parser.add_argument(
        "--dev_jsonl",
        default=r"..\..\data\retrieve\dev_data.jsonl",
        help="Path to retrieve/dev_data.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        default=r"..\..\data\lightning_ir_ready\splade_retrieve",
        help="Output directory",
    )
    args = parser.parse_args()

    train_jsonl = os.path.abspath(args.train_jsonl)
    dev_jsonl = os.path.abspath(args.dev_jsonl)
    output_dir = os.path.abspath(args.output_dir)

    if not os.path.isfile(train_jsonl):
        raise FileNotFoundError(train_jsonl)
    if not os.path.isfile(dev_jsonl):
        raise FileNotFoundError(dev_jsonl)

    # Build a single shared mapping (query/doc ids) across train+dev
    query2id: Dict[str, int] = {}
    doc2id: Dict[str, int] = {}

    os.makedirs(output_dir, exist_ok=True)
    qrels_dir = os.path.join(output_dir, "qrels")
    os.makedirs(qrels_dir, exist_ok=True)

    queries_tsv_path = os.path.join(output_dir, "queries.tsv")
    corpus_tsv_path = os.path.join(output_dir, "corpus.tsv")
    train_qrels_path = os.path.join(qrels_dir, "train.tsv")
    dev_qrels_path = os.path.join(qrels_dir, "dev.tsv")

    with open(queries_tsv_path, "w", encoding="utf-8", newline="\n") as out_queries, open(
        corpus_tsv_path, "w", encoding="utf-8", newline="\n"
    ) as out_corpus, open(train_qrels_path, "w", encoding="utf-8", newline="\n") as out_train_qrels, open(
        dev_qrels_path, "w", encoding="utf-8", newline="\n"
    ) as out_dev_qrels:
        process_split(
            train_jsonl,
            query2id=query2id,
            doc2id=doc2id,
            out_queries_tsv=out_queries,
            out_corpus_tsv=out_corpus,
            out_qrels_tsv=out_train_qrels,
        )
        process_split(
            dev_jsonl,
            query2id=query2id,
            doc2id=doc2id,
            out_queries_tsv=out_queries,
            out_corpus_tsv=out_corpus,
            out_qrels_tsv=out_dev_qrels,
        )

    print("Prepared Lightning-IR data:")
    print(f"  queries.tsv: {queries_tsv_path}")
    print(f"  corpus.tsv : {corpus_tsv_path}")
    print(f"  qrels/train.tsv: {train_qrels_path}")
    print(f"  qrels/dev.tsv  : {dev_qrels_path}")
    print(f"  unique queries: {len(query2id)}")
    print(f"  unique docs   : {len(doc2id)}")


if __name__ == "__main__":
    main()

