"""
index_builder.py – Xây dựng và lưu Inverted Index.

Inverted Index là cấu trúc dữ liệu cốt lõi của mọi search engine:
  term → { doc_id: term_frequency, ... }

Ngoài ra lưu thêm:
  - doc_lengths : dict[doc_id, int]  – số token trong mỗi tài liệu
  - avg_doc_length : float           – độ dài trung bình (dùng cho BM25)
  - doc_count : int                  – tổng số tài liệu
"""
from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import joblib

logger = logging.getLogger(__name__)


class InvertedIndex:
    """Inverted Index đơn giản, thuần Python.

    Attributes:
        index         : dict[term, dict[doc_id, tf]]
        doc_lengths   : dict[doc_id, int]
        avg_doc_length: float
        doc_count     : int
    """

    def __init__(self) -> None:
        # term → {doc_id: term_frequency}
        self.index: Dict[str, Dict[int, int]] = defaultdict(dict)
        # Độ dài (số token) của mỗi tài liệu
        self.doc_lengths: Dict[int, int] = {}
        self.avg_doc_length: float = 0.0
        self.doc_count: int = 0

    # ──────────────────────────────────────────────────────────────────────────
    # Xây dựng index
    # ──────────────────────────────────────────────────────────────────────────

    def build(self, tokenized_docs: Iterable[tuple[int, List[str]]]) -> None:
        """Xây dựng inverted index từ danh sách tài liệu đã tokenize.

        Args:
            tokenized_docs: Iterator của (doc_id, tokens).
                            VD: [(0, ['học_máy', 'là', ...]), (1, [...]), ...]
        """
        total_tokens = 0

        for doc_id, tokens in tokenized_docs:
            self.doc_lengths[doc_id] = len(tokens)
            total_tokens += len(tokens)
            self.doc_count += 1

            # Đếm tần suất từ trong tài liệu
            term_freq: Dict[str, int] = defaultdict(int)
            for token in tokens:
                term_freq[token] += 1

            # Cập nhật inverted index
            for term, freq in term_freq.items():
                self.index[term][doc_id] = freq

        self.avg_doc_length = (
            total_tokens / self.doc_count if self.doc_count > 0 else 0.0
        )
        logger.info(
            "Đã build index: %d terms, %d tài liệu, avg_len=%.1f",
            len(self.index),
            self.doc_count,
            self.avg_doc_length,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Lookup
    # ──────────────────────────────────────────────────────────────────────────

    def get_postings(self, term: str) -> Dict[int, int]:
        """Trả về posting list của một term.

        Args:
            term: Từ cần tra cứu.

        Returns:
            dict[doc_id, tf] – rỗng nếu term không tồn tại.
        """
        return self.index.get(term, {})

    def document_frequency(self, term: str) -> int:
        """Số tài liệu chứa term."""
        return len(self.index.get(term, {}))

    # ──────────────────────────────────────────────────────────────────────────
    # Lưu / Load
    # ──────────────────────────────────────────────────────────────────────────

    def save(self, filepath: str | Path) -> None:
        """Lưu index xuống đĩa bằng joblib.

        Args:
            filepath: Đường dẫn file xuất ra (.pkl).
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)
        logger.info("Đã lưu index → %s", path)

    @classmethod
    def load(cls, filepath: str | Path) -> "InvertedIndex":
        """Load index từ file .pkl.

        Args:
            filepath: Đường dẫn file đã lưu.

        Returns:
            InvertedIndex instance.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file index: {path}")
        obj = joblib.load(path)
        logger.info("Đã load index ← %s (%d terms)", path, len(obj.index))
        return obj
