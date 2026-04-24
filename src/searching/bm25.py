"""
bm25.py – Cài đặt thuật toán BM25 (Okapi BM25) thuần Python + NumPy.

Công thức:
    score(q, d) = Σ_{t∈q} IDF(t) × (tf(t,d) × (k1+1)) / (tf(t,d) + k1×(1 - b + b×|d|/avgdl))

    IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)  [Robertson IDF]

Tham số:
    k1 : float, mặc định 1.5  – điều chỉnh ảnh hưởng của term frequency
    b  : float, mặc định 0.75 – điều chỉnh ảnh hưởng của độ dài tài liệu

Tài liệu tham khảo:
    Robertson & Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond", 2009.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, List, Tuple

import joblib

from src.indexing.index_builder import InvertedIndex

logger = logging.getLogger(__name__)


class BM25:
    """BM25 scorer dựa trên InvertedIndex đã build sẵn.

    Usage:
        index = InvertedIndex.load("data/index/inverted_index.pkl")
        bm25  = BM25(index, k1=1.5, b=0.75)
        results = bm25.get_top_k(["học_máy", "ứng_dụng"], k=10)
    """

    def __init__(
        self,
        index: InvertedIndex,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.index = index
        self.k1 = k1
        self.b = b
        self._idf_cache: Dict[str, float] = {}

    # ──────────────────────────────────────────────────────────────────────────
    # IDF
    # ──────────────────────────────────────────────────────────────────────────

    def _idf(self, term: str) -> float:
        """Tính IDF của một term (Robertson IDF, không âm).

        IDF(t) = log((N - df + 0.5) / (df + 0.5) + 1)
        """
        if term not in self._idf_cache:
            n = self.index.doc_count
            df = self.index.document_frequency(term)
            self._idf_cache[term] = math.log((n - df + 0.5) / (df + 0.5) + 1)
        return self._idf_cache[term]

    # ──────────────────────────────────────────────────────────────────────────
    # Scoring
    # ──────────────────────────────────────────────────────────────────────────

    def score(self, query_tokens: List[str], doc_id: int) -> float:
        """Tính BM25 score cho một cặp (query, document).

        Args:
            query_tokens: Danh sách token của truy vấn.
            doc_id: ID tài liệu.

        Returns:
            BM25 score (float ≥ 0).
        """
        dl = self.index.doc_lengths.get(doc_id, 0)
        avgdl = self.index.avg_doc_length
        score = 0.0
        for term in query_tokens:
            tf = self.index.get_postings(term).get(doc_id, 0)
            if tf == 0:
                continue
            idf = self._idf(term)
            # Công thức BM25
            num = tf * (self.k1 + 1)
            den = tf + self.k1 * (1 - self.b + self.b * dl / (avgdl or 1))
            score += idf * (num / den)
        return score

    def get_scores(self, query_tokens: List[str]) -> Dict[int, float]:
        """Tính BM25 score cho tất cả tài liệu chứa ít nhất một query term.

        Args:
            query_tokens: Danh sách token truy vấn.

        Returns:
            dict[doc_id, score] – chỉ chứa tài liệu có score > 0.
        """
        candidate_docs: Dict[int, float] = {}
        for term in query_tokens:
            for doc_id in self.index.get_postings(term):
                if doc_id not in candidate_docs:
                    candidate_docs[doc_id] = self.score(query_tokens, doc_id)
        return candidate_docs

    def get_top_k(
        self, query_tokens: List[str], k: int = 10
    ) -> List[Tuple[int, float]]:
        """Trả về Top-K tài liệu theo BM25 score (giảm dần).

        Args:
            query_tokens: Danh sách token truy vấn.
            k: Số kết quả trả về.

        Returns:
            Danh sách (doc_id, score) sắp xếp giảm dần theo score.
        """
        scores = self.get_scores(query_tokens)
        top_k = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return top_k

    # ──────────────────────────────────────────────────────────────────────────
    # Lưu / Load
    # ──────────────────────────────────────────────────────────────────────────

    def save(self, filepath: str | Path) -> None:
        """Lưu BM25 model ra file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)
        logger.info("Đã lưu BM25 → %s", path)

    @classmethod
    def load(cls, filepath: str | Path) -> "BM25":
        """Load BM25 model từ file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy: {path}")
        return joblib.load(path)
