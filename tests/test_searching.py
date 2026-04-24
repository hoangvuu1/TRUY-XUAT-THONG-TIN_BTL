"""tests/test_searching.py – Unit tests cho BM25, TFIDFSearcher, QueryExpander."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.indexing.index_builder import InvertedIndex
from src.searching.bm25 import BM25
from src.searching.query_expansion import QueryExpander
from src.searching.tfidf import TFIDFSearcher


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture()
def index_with_docs() -> InvertedIndex:
    idx = InvertedIndex()
    docs = [
        (0, ["học_máy", "ứng_dụng", "y_tế", "học_máy", "ai"]),
        (1, ["hà_nội", "du_lịch", "ẩm_thực", "thăm_quan"]),
        (2, ["học_máy", "deep_learning", "ai", "mạng_nơ_ron"]),
        (3, ["kinh_tế", "tài_chính", "chứng_khoán"]),
        (4, ["y_tế", "bệnh_viện", "sức_khoẻ", "ai"]),
    ]
    idx.build(iter(docs))
    return idx


@pytest.fixture()
def bm25_model(index_with_docs) -> BM25:
    return BM25(index_with_docs, k1=1.5, b=0.75)


@pytest.fixture()
def tfidf_model() -> TFIDFSearcher:
    corpus = [
        "học_máy ứng_dụng y_tế học_máy ai",
        "hà_nội du_lịch ẩm_thực thăm_quan",
        "học_máy deep_learning ai mạng_nơ_ron",
        "kinh_tế tài_chính chứng_khoán",
        "y_tế bệnh_viện sức_khoẻ ai",
    ]
    searcher = TFIDFSearcher(max_features=1000)
    searcher.fit(corpus)
    return searcher


# ── BM25 Tests ────────────────────────────────────────────────────────────────

class TestBM25:
    def test_score_positive(self, bm25_model):
        score = bm25_model.score(["học_máy"], doc_id=0)
        assert score > 0

    def test_score_zero_for_no_match(self, bm25_model):
        score = bm25_model.score(["hà_nội"], doc_id=0)
        assert score == 0.0

    def test_get_top_k_returns_k(self, bm25_model):
        results = bm25_model.get_top_k(["học_máy", "ai"], k=3)
        assert len(results) <= 3

    def test_get_top_k_sorted_descending(self, bm25_model):
        results = bm25_model.get_top_k(["học_máy", "ai"], k=5)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_relevant_docs_ranked_first(self, bm25_model):
        results = bm25_model.get_top_k(["học_máy"], k=5)
        top_doc_ids = [doc_id for doc_id, _ in results]
        # doc 0 và doc 2 đều chứa "học_máy"
        assert 0 in top_doc_ids
        assert 2 in top_doc_ids

    def test_empty_query(self, bm25_model):
        results = bm25_model.get_top_k([], k=5)
        assert results == []

    def test_save_and_load(self, bm25_model, tmp_path):
        path = tmp_path / "bm25.pkl"
        bm25_model.save(path)
        loaded = BM25.load(path)
        r1 = bm25_model.get_top_k(["học_máy"], k=3)
        r2 = loaded.get_top_k(["học_máy"], k=3)
        assert r1 == r2


# ── TF-IDF Tests ──────────────────────────────────────────────────────────────

class TestTFIDFSearcher:
    def test_fit_creates_matrix(self, tfidf_model):
        assert tfidf_model.tfidf_matrix is not None
        assert tfidf_model.tfidf_matrix.shape[0] == 5

    def test_get_top_k(self, tfidf_model):
        results = tfidf_model.get_top_k("học_máy ai", k=3)
        assert isinstance(results, list)
        assert len(results) <= 3

    def test_scores_sorted(self, tfidf_model):
        results = tfidf_model.get_top_k("học_máy", k=5)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_not_fitted_raises(self):
        searcher = TFIDFSearcher()
        with pytest.raises(RuntimeError):
            searcher.get_scores("test")


# ── Query Expansion Tests ─────────────────────────────────────────────────────

class TestQueryExpander:
    def test_expand_known_term(self):
        expander = QueryExpander()
        result = expander.expand(["ai"])
        assert "ai" in result
        assert len(result) > 1  # Phải có từ đồng nghĩa

    def test_expand_unknown_term(self):
        expander = QueryExpander()
        result = expander.expand(["từ_không_có_trong_từ_điển"])
        assert result == ["từ_không_có_trong_từ_điển"]

    def test_no_duplicates(self):
        expander = QueryExpander()
        result = expander.expand(["ai", "học_máy"])
        assert len(result) == len(set(result))

    def test_original_tokens_preserved(self):
        expander = QueryExpander()
        tokens = ["học_máy", "y_tế"]
        result = expander.expand(tokens)
        for t in tokens:
            assert t in result

    def test_custom_synonyms(self):
        custom = {"python": ["lập_trình", "coding"]}
        expander = QueryExpander(synonyms=custom)
        result = expander.expand(["python"])
        assert "lập_trình" in result
