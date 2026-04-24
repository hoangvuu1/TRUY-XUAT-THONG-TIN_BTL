"""
preprocess.py – Tiền xử lý dữ liệu thô và lưu ra data/processed/.

Bước này:
1. Đọc file CSV từ data/raw/
2. Làm sạch (clean) và tokenize văn bản
3. Lưu kết quả ra data/processed/documents.jsonl
4. Lưu metadata (title, category, snippet) ra data/processed/metadata.pkl

Chạy:
    python scripts/preprocess.py
    python scripts/preprocess.py --max-docs 5000
"""
import argparse
import json
import logging
import sys
from pathlib import Path

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


def main(max_docs: int = config.MAX_DOCS) -> None:
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
    df = pd.read_csv(csv_path, nrows=max_docs)
    logger.info("Đọc xong: %d hàng, cột: %s", len(df), df.columns.tolist())

    # Cột bắt buộc
    text_col  = "content" if "content" in df.columns else df.columns[0]
    title_col = "title"   if "title"   in df.columns else None

    # ── Load stopwords ────────────────────────────────────────────────────────
    stopwords = load_stopwords(config.STOPWORDS_FILE)

    # ── Tiền xử lý ────────────────────────────────────────────────────────────
    config.PROC_DATA_DIR.mkdir(parents=True, exist_ok=True)

    processed_file = config.PROCESSED_FILE
    metadata_file  = config.PROC_DATA_DIR / "metadata.pkl"

    metadata = []
    doc_count = 0

    logger.info("Đang xử lý %d tài liệu ...", len(df))
    with open(processed_file, "w", encoding="utf-8") as fout:
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing"):
            raw_text = str(row.get(text_col, ""))
            title    = str(row.get(title_col, "")) if title_col else ""

            clean   = clean_text(raw_text)
            tokens  = tokenize(clean, stopwords=stopwords, min_len=config.MIN_TOKEN_LEN)

            if not tokens:
                continue

            # Lưu tokens ra JSONL
            record = {"doc_id": doc_count, "tokens": tokens}
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

            # Lưu metadata
            snippet = clean[:200]
            meta = {
                "doc_id"  : doc_count,
                "title"   : clean_text(title) or f"Tài liệu {doc_count}",
                "snippet" : snippet,
                "category": str(row.get("category", "")),
                "source"  : str(row.get("source", "")),
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
        help=f"Số tài liệu tối đa (mặc định: {config.MAX_DOCS})"
    )
    args = parser.parse_args()
    main(max_docs=args.max_docs)
