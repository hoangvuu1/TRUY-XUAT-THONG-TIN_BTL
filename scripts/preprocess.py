"""
preprocess.py – Tiền xử lý dữ liệu thô và lưu ra data/processed/.

Bước này:
1. Đọc file CSV từ data/raw/
2. Làm sạch (clean) và tokenize văn bản (song song bằng joblib)
3. Lưu kết quả ra data/processed/documents.jsonl
4. Lưu metadata (title, category, snippet) ra data/processed/metadata.pkl

Chạy:
    python scripts/preprocess.py
    python scripts/preprocess.py --max-docs 50000
    python scripts/preprocess.py --n-jobs 8
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from tqdm import tqdm

# Thêm root vào sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config
from src.processing import clean_text, tokenize, load_stopwords

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _process_one(raw_text: str, title: str, stopwords: frozenset, min_len: int) -> tuple:
    """Làm sạch và tokenize một tài liệu (chạy trong worker process).

    Returns:
        (clean_text, tokens, title) – tokens là list rỗng nếu không có nội dung.
    """
    clean   = clean_text(raw_text)
    tokens  = tokenize(clean, stopwords=stopwords, min_len=min_len)
    return clean, tokens, title


def main(
    max_docs: int = config.MAX_DOCS,
    n_jobs: int = config.N_JOBS,
) -> None:
    # ── Kiểm tra file đầu vào ─────────────────────────────────────────────────
    csv_path = config.RAW_CSV_FILE
    if not csv_path.exists():
        logger.error(
            "Không tìm thấy file CSV: %s\n"
            "Hãy download dataset từ Kaggle và đặt vào data/raw/",
            csv_path,
        )
        sys.exit(1)

    # ── Đọc CSV ───────────────────────────────────────────────────────────────
    logger.info("Đang đọc %s ...", csv_path)
    nrows: Optional[int] = max_docs if max_docs > 0 else None
    df = pd.read_csv(csv_path, nrows=nrows)
    logger.info("Đọc xong: %d hàng, cột: %s", len(df), df.columns.tolist())

    # Cột bắt buộc
    text_col  = "content" if "content" in df.columns else df.columns[0]
    title_col = "title"   if "title"   in df.columns else None

    # ── Load stopwords ────────────────────────────────────────────────────────
    # frozenset để serialise nhanh hơn khi gửi sang worker processes
    stopwords = frozenset(load_stopwords(config.STOPWORDS_FILE))

    # ── Chuẩn bị dữ liệu đầu vào cho parallel workers ────────────────────────
    records = df.to_dict("records")
    raw_texts = [str(r.get(text_col, "")) for r in records]
    titles    = [str(r.get(title_col, "")) if title_col else "" for r in records]
    categories = [str(r.get("topic", r.get("category", ""))) for r in records]
    sources    = [str(r.get("source", "")) for r in records]

    min_len = config.MIN_TOKEN_LEN

    logger.info(
        "Đang xử lý %d tài liệu với n_jobs=%s ...",
        len(records),
        n_jobs if n_jobs != -1 else "all CPUs",
    )

    # ── Song song hoá tokenize ────────────────────────────────────────────────
    results = joblib.Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        joblib.delayed(_process_one)(raw_text, title, stopwords, min_len)
        for raw_text, title in tqdm(
            zip(raw_texts, titles), total=len(raw_texts), desc="Tokenizing"
        )
    )

    # ── Ghi kết quả ra file ───────────────────────────────────────────────────
    config.PROC_DATA_DIR.mkdir(parents=True, exist_ok=True)

    processed_file = config.PROCESSED_FILE
    metadata_file  = config.PROC_DATA_DIR / "metadata.pkl"

    metadata  = []
    doc_count = 0

    with open(processed_file, "w", encoding="utf-8") as fout:
        for i, (clean, tokens, title) in enumerate(results):
            if not tokens:
                continue

            record = {"doc_id": doc_count, "tokens": tokens}
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

            snippet = clean[:200]
            meta = {
                "doc_id"  : doc_count,
                "title"   : clean_text(title) or f"Tài liệu {doc_count}",
                "snippet" : snippet,
                "category": categories[i],
                "source"  : sources[i],
            }
            metadata.append(meta)
            doc_count += 1

    # ── Lưu metadata ──────────────────────────────────────────────────────────
    joblib.dump(metadata, metadata_file)
    logger.info(
        "Hoàn tất! Đã xử lý %d tài liệu.\n"
        "  → Tokens   : %s\n"
        "  → Metadata : %s",
        doc_count,
        processed_file,
        metadata_file,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tiền xử lý dữ liệu tiếng Việt")
    parser.add_argument(
        "--max-docs", type=int, default=config.MAX_DOCS,
        help="Số tài liệu tối đa (0 = không giới hạn, mặc định: 0)"
    )
    parser.add_argument(
        "--n-jobs", type=int, default=config.N_JOBS,
        help="Số tiến trình song song (-1 = dùng hết CPU, mặc định: -1)"
    )
    args = parser.parse_args()
    main(max_docs=args.max_docs, n_jobs=args.n_jobs)
