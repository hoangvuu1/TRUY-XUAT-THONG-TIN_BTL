"""
search_engine.py – Lớp bao bọc Search Engine (BM25 + TF-IDF + Query Expansion).

Load model/index từ đĩa và cung cấp interface tìm kiếm thống nhất.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib

from src import config
from src.processing import clean_text, tokenize, load_stopwords
from src.searching import BM25, TFIDFSearcher, QueryExpander

logger = logging.getLogger(__name__)


class SearchEngine:
    """Facade kết hợp BM25 / TF-IDF / Query Expansion.

    Usage:
        engine = SearchEngine()
        engine.load()                         # Load index từ đĩa
        results = engine.search("học máy", method="bm25", top_k=10)
    """

    def __init__(self) -> None:
        self.bm25: Optional[BM25] = None
        self.tfidf: Optional[TFIDFSearcher] = None
        self.expander: QueryExpander = QueryExpander()
        self.doc_metadata: List[Dict[str, Any]] = []   # title, snippet, category, ...
        self.stopwords: set = set()
        self._loaded = False

    # ──────────────────────────────────────────────────────────────────────────
    # Load
    # ──────────────────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load tất cả model/index từ đĩa.

        Nếu file chưa tồn tại, hệ thống vẫn khởi động (trả về kết quả rỗng).
        """
        self.stopwords = load_stopwords(config.STOPWORDS_FILE)

        # Load BM25
        if Path(config.BM25_INDEX_FILE).exists():
            try:
                self.bm25 = BM25.load(config.BM25_INDEX_FILE)
                logger.info("Đã load BM25 index.")
            except Exception as exc:
                logger.error("Lỗi load BM25: %s", exc)

        # Load TF-IDF
        if (
            Path(config.TFIDF_VECTORIZER_FILE).exists()
            and Path(config.TFIDF_MATRIX_FILE).exists()
        ):
            try:
                self.tfidf = TFIDFSearcher.load(
                    config.TFIDF_VECTORIZER_FILE, config.TFIDF_MATRIX_FILE
                )
                logger.info("Đã load TF-IDF model.")
            except Exception as exc:
                logger.error("Lỗi load TF-IDF: %s", exc)

        # Load metadata tài liệu
        if Path(config.DOC_METADATA_FILE).exists():
            try:
                self.doc_metadata = joblib.load(config.DOC_METADATA_FILE)
                logger.info("Đã load %d doc metadata.", len(self.doc_metadata))
            except Exception as exc:
                logger.error("Lỗi load metadata: %s", exc)

        self._loaded = True

    # ──────────────────────────────────────────────────────────────────────────
    # Search
    # ──────────────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        method: str = "bm25",
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """Tìm kiếm và trả về kết quả.

        Args:
            query : Chuỗi truy vấn người dùng nhập.
            method: Phương pháp xếp hạng: "tfidf" | "bm25" | "bm25_expand".
            top_k : Số kết quả trả về.

        Returns:
            dict chứa:
                - query         : truy vấn gốc
                - expanded_query: truy vấn sau mở rộng (nếu có)
                - method        : phương pháp dùng
                - results       : list[dict] (title, snippet, score, category, ...)
                - elapsed_ms    : thời gian xử lý (ms)
        """
        start = time.perf_counter()

        # Tiền xử lý truy vấn
        clean = clean_text(query)
        tokens = tokenize(clean, stopwords=self.stopwords, min_len=config.MIN_TOKEN_LEN)

        expanded_tokens = tokens
        if method == "bm25_expand":
            expanded_tokens = self.expander.expand(tokens)

        # Xếp hạng
        ranked: List[tuple] = []
        if method in ("bm25", "bm25_expand"):
            if self.bm25 is not None:
                ranked = self.bm25.get_top_k(expanded_tokens, k=top_k)
            else:
                logger.warning("BM25 chưa được load. Chạy build_index.py trước.")
        elif method == "tfidf":
            if self.tfidf is not None:
                query_str = " ".join(tokens)
                ranked = self.tfidf.get_top_k(query_str, k=top_k)
            else:
                logger.warning("TF-IDF chưa được load. Chạy build_index.py trước.")

        # Lấy metadata
        results = []
        for doc_id, score in ranked:
            if doc_id < len(self.doc_metadata):
                meta = dict(self.doc_metadata[doc_id])
            else:
                meta = {"doc_id": doc_id}
            meta["score"] = round(score, 4)
            meta["doc_id"] = doc_id
            results.append(meta)

        elapsed_ms = round((time.perf_counter() - start) * 1000, 1)

        return {
            "query": query,
            "expanded_query": " ".join(expanded_tokens) if method == "bm25_expand" else None,
            "method": method,
            "results": results,
            "elapsed_ms": elapsed_ms,
        }
