"""
evaluate.py – Đánh giá hệ thống tìm kiếm trên tập test.

Yêu cầu file: data/processed/test_queries.json
Định dạng:
[
  {"query": "học máy", "relevant_doc_ids": [0, 5, 12, ...]},
  ...
]

Chạy:
    python scripts/evaluate.py
    python scripts/evaluate.py --method bm25 --top-k 10
"""
import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config
from src.evaluation import (
    average_precision,
    mean_average_precision,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from src.web.search_engine import SearchEngine

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Đánh giá hệ thống tìm kiếm")
    parser.add_argument("--method", default="bm25",
                        choices=["tfidf", "bm25", "bm25_expand"])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--test-file",
        type=str,
        default=str(config.PROC_DATA_DIR / "test_queries.json"),
        help="Đường dẫn file test queries JSON",
    )
    args = parser.parse_args()

    test_file = Path(args.test_file)
    if not test_file.exists():
        print(
            f"Không tìm thấy file test: {test_file}\n"
            "Tạo file theo định dạng: [{\"query\": \"...\", \"relevant_doc_ids\": [...]}, ...]"
        )
        sys.exit(1)

    with open(test_file, encoding="utf-8") as f:
        test_queries = json.load(f)

    print(f"Đã load {len(test_queries)} test queries.")

    engine = SearchEngine()
    engine.load()

    results_list  = []
    relevant_list = []

    for item in test_queries:
        query   = item["query"]
        relevant = set(item.get("relevant_doc_ids", []))

        result   = engine.search(query, method=args.method, top_k=args.top_k)
        retrieved = [r["doc_id"] for r in result["results"]]

        results_list.append(retrieved)
        relevant_list.append(relevant)

    # ── Tính metrics ──────────────────────────────────────────────────────────
    k = args.top_k
    p_at_k_vals  = [precision_at_k(r, rel, k) for r, rel in zip(results_list, relevant_list)]
    r_at_k_vals  = [recall_at_k(r, rel, k) for r, rel in zip(results_list, relevant_list)]
    ndcg_vals    = [ndcg_at_k(r, rel, k) for r, rel in zip(results_list, relevant_list)]
    map_score    = mean_average_precision(results_list, relevant_list)

    print(f"\n{'='*50}")
    print(f"Phương pháp : {args.method.upper()}")
    print(f"Top-K       : {k}")
    print(f"Số queries  : {len(test_queries)}")
    print(f"{'='*50}")
    print(f"MAP         : {map_score:.4f}")
    print(f"P@{k:<9}  : {sum(p_at_k_vals)/len(p_at_k_vals):.4f}")
    print(f"Recall@{k:<6}: {sum(r_at_k_vals)/len(r_at_k_vals):.4f}")
    print(f"NDCG@{k:<8}: {sum(ndcg_vals)/len(ndcg_vals):.4f}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
