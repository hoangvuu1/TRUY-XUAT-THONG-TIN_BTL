"""
build_index.py – Xây dựng Inverted Index + TF-IDF + BM25 từ dữ liệu đã xử lý.

Yêu cầu: Chạy scripts/preprocess.py trước.

Chạy:
    python scripts/build_index.py
"""
import json
import logging
import sys
from pathlib import Path

import joblib
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config
from src.indexing import InvertedIndex
from src.searching import BM25, TFIDFSearcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    processed_file = config.PROCESSED_FILE
    if not processed_file.exists():
        logger.error(
            "Chưa có dữ liệu đã xử lý: %s\n"
            "Hãy chạy: python scripts/preprocess.py",
            processed_file,
        )
        sys.exit(1)

    # ── Đọc dữ liệu đã tokenize ───────────────────────────────────────────────
    logger.info("Đọc dữ liệu từ %s ...", processed_file)
    records = []
    with open(processed_file, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    logger.info("Tổng số tài liệu: %d", len(records))

    # ── Xây dựng Inverted Index ───────────────────────────────────────────────
    logger.info("Đang xây dựng Inverted Index ...")
    inv_index = InvertedIndex()
    inv_index.build(
        (rec["doc_id"], rec["tokens"])
        for rec in tqdm(records, desc="Building index")
    )

    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    inv_index_path = config.INDEX_DIR / "inverted_index.pkl"
    inv_index.save(inv_index_path)

    # ── Xây dựng BM25 ────────────────────────────────────────────────────────
    logger.info("Đang tạo BM25 model ...")
    bm25 = BM25(inv_index, k1=config.BM25_K1, b=config.BM25_B)
    bm25.save(config.BM25_INDEX_FILE)

    # ── Xây dựng TF-IDF ──────────────────────────────────────────────────────
    logger.info("Đang fit TF-IDF ...")
    corpus = [" ".join(rec["tokens"]) for rec in records]
    tfidf = TFIDFSearcher(max_features=50_000)
    tfidf.fit(corpus)
    tfidf.save(config.TFIDF_VECTORIZER_FILE, config.TFIDF_MATRIX_FILE)

    # ── Copy metadata để web dùng ─────────────────────────────────────────────
    src_meta = config.PROC_DATA_DIR / "metadata.pkl"
    if src_meta.exists():
        import shutil
        shutil.copy(src_meta, config.DOC_METADATA_FILE)
        logger.info("Đã copy metadata → %s", config.DOC_METADATA_FILE)

    logger.info("✅ Build index hoàn tất! Tất cả file lưu trong: %s", config.INDEX_DIR)


if __name__ == "__main__":
    main()
