"""
run_search.py – Script demo tìm kiếm từ command line.

Chạy:
    python scripts/run_search.py --query "học máy ứng dụng y tế"
    python scripts/run_search.py --query "du lịch hà nội" --method tfidf --top-k 5
"""
import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.web.search_engine import SearchEngine

logging.basicConfig(level=logging.WARNING)


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo CLI tìm kiếm tài liệu tiếng Việt")
    parser.add_argument("--query",  type=str, required=True, help="Truy vấn tìm kiếm")
    parser.add_argument("--method", type=str, default="bm25",
                        choices=["tfidf", "bm25", "bm25_expand"],
                        help="Phương pháp xếp hạng (mặc định: bm25)")
    parser.add_argument("--top-k",  type=int, default=10, help="Số kết quả (mặc định: 10)")
    args = parser.parse_args()

    engine = SearchEngine()
    engine.load()

    result = engine.search(args.query, method=args.method, top_k=args.top_k)

    print(f"\n{'='*60}")
    print(f"Query  : {result['query']}")
    if result.get("expanded_query"):
        print(f"Mở rộng: {result['expanded_query']}")
    print(f"Method : {result['method'].upper()}")
    print(f"Thời gian: {result['elapsed_ms']} ms")
    print(f"{'='*60}")

    if not result["results"]:
        print("Không tìm thấy kết quả.")
        return

    for i, item in enumerate(result["results"], start=1):
        title  = item.get("title", f"Doc #{item['doc_id']}")
        score  = item.get("score", 0)
        cat    = item.get("category", "")
        snippet = item.get("snippet", "")[:120]
        print(f"\n{i:2d}. [{score:.4f}] {title}")
        if cat:
            print(f"    Category: {cat}")
        if snippet:
            print(f"    {snippet}...")


if __name__ == "__main__":
    main()
