"""tests/test_indexing.py – Unit tests cho InvertedIndex."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.indexing.index_builder import InvertedIndex


@pytest.fixture()
def sample_index() -> InvertedIndex:
    idx = InvertedIndex()
    docs = [
        (0, ["học_máy", "ứng_dụng", "y_tế", "học_máy"]),
        (1, ["hà_nội", "du_lịch", "ẩm_thực"]),
        (2, ["học_máy", "deep_learning", "ai"]),
    ]
    idx.build(iter(docs))
    return idx


class TestInvertedIndex:
    def test_doc_count(self, sample_index):
        assert sample_index.doc_count == 3

    def test_doc_lengths(self, sample_index):
        assert sample_index.doc_lengths[0] == 4  # 4 tokens
        assert sample_index.doc_lengths[1] == 3

    def test_avg_doc_length(self, sample_index):
        # (4 + 3 + 3) / 3 = 10/3 ≈ 3.33
        assert abs(sample_index.avg_doc_length - (10 / 3)) < 0.01

    def test_postings(self, sample_index):
        postings = sample_index.get_postings("học_máy")
        assert 0 in postings
        assert postings[0] == 2  # xuất hiện 2 lần trong doc 0
        assert 2 in postings
        assert 1 not in postings

    def test_document_frequency(self, sample_index):
        assert sample_index.document_frequency("học_máy") == 2
        assert sample_index.document_frequency("hà_nội") == 1
        assert sample_index.document_frequency("không_tồn_tại") == 0

    def test_unknown_term(self, sample_index):
        assert sample_index.get_postings("xyz") == {}

    def test_save_and_load(self, sample_index, tmp_path):
        filepath = tmp_path / "test_index.pkl"
        sample_index.save(filepath)
        loaded = InvertedIndex.load(filepath)
        assert loaded.doc_count == sample_index.doc_count
        assert loaded.get_postings("học_máy") == sample_index.get_postings("học_máy")
