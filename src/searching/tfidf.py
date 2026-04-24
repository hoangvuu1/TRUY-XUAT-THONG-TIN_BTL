"""
tfidf.py – TF-IDF search dùng scikit-learn.

Dùng TfidfVectorizer để tạo ma trận TF-IDF, sau đó tính cosine similarity
giữa query vector và tất cả document vectors.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class TFIDFSearcher:
    """Tìm kiếm dựa trên TF-IDF + Cosine Similarity.

    Usage:
        searcher = TFIDFSearcher()
        searcher.fit(corpus_texts)          # corpus_texts: List[str] (đã tokenize)
        results = searcher.get_top_k("học máy ứng dụng", k=10)
    """

    def __init__(self, max_features: int = 50_000, n_jobs: int = -1) -> None:
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.vectorizer: TfidfVectorizer | None = None
        self.tfidf_matrix = None  # scipy sparse matrix

    # ──────────────────────────────────────────────────────────────────────────
    # Fit / Transform
    # ──────────────────────────────────────────────────────────────────────────

    def fit(self, corpus: List[str]) -> None:
        """Học TF-IDF từ corpus và xây dựng ma trận.

        Args:
            corpus: Danh sách chuỗi văn bản (mỗi phần tử = một tài liệu đã
                    được tokenize và join bằng khoảng trắng, ví dụ:
                    "học_máy ứng_dụng y_tế").
        """
        logger.info("Đang fit TF-IDF trên %d tài liệu ...", len(corpus))
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            sublinear_tf=True,  # Dùng log(tf) + 1 thay vì tf thuần
            ngram_range=(1, 2), # Unigram + Bigram
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        logger.info(
            "Đã fit TF-IDF: vocab=%d, ma trận=%s",
            len(self.vectorizer.vocabulary_),
            self.tfidf_matrix.shape,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Search
    # ──────────────────────────────────────────────────────────────────────────

    def get_scores(self, query: str) -> np.ndarray:
        """Tính cosine similarity giữa query và tất cả tài liệu.

        Args:
            query: Chuỗi truy vấn (đã tokenize & join).

        Returns:
            numpy array shape (n_docs,) chứa score của từng tài liệu.
        """
        if self.vectorizer is None or self.tfidf_matrix is None:
            raise RuntimeError("Chưa fit model. Gọi fit() trước.")
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        return scores

    def get_top_k(
        self, query: str, k: int = 10
    ) -> List[Tuple[int, float]]:
        """Trả về Top-K tài liệu theo TF-IDF cosine score.

        Args:
            query: Chuỗi truy vấn.
            k: Số kết quả trả về.

        Returns:
            Danh sách (doc_id, score) sắp xếp giảm dần.
        """
        scores = self.get_scores(query)
        top_k_idx = np.argsort(scores)[::-1][:k]
        return [(int(idx), float(scores[idx])) for idx in top_k_idx if scores[idx] > 0]

    # ──────────────────────────────────────────────────────────────────────────
    # Lưu / Load
    # ──────────────────────────────────────────────────────────────────────────

    def save(self, vectorizer_path: str | Path, matrix_path: str | Path) -> None:
        """Lưu vectorizer và ma trận TF-IDF."""
        for path in (Path(vectorizer_path), Path(matrix_path)):
            path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, vectorizer_path, compress=3)
        joblib.dump(self.tfidf_matrix, matrix_path, compress=3)
        logger.info("Đã lưu TF-IDF vectorizer → %s", vectorizer_path)
        logger.info("Đã lưu TF-IDF matrix    → %s", matrix_path)

    @classmethod
    def load(
        cls, vectorizer_path: str | Path, matrix_path: str | Path
    ) -> "TFIDFSearcher":
        """Load TFIDFSearcher từ file."""
        obj = cls()
        obj.vectorizer = joblib.load(vectorizer_path)
        obj.tfidf_matrix = joblib.load(matrix_path)
        logger.info("Đã load TF-IDF ← %s", vectorizer_path)
        return obj
