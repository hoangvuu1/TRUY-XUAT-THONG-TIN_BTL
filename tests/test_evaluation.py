"""tests/test_evaluation.py – Unit tests cho các metrics IR."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.metrics import (
    average_precision,
    mean_average_precision,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


class TestPrecisionAtK:
    def test_perfect(self):
        assert precision_at_k([1, 2, 3], {1, 2, 3}, k=3) == 1.0

    def test_zero(self):
        assert precision_at_k([4, 5, 6], {1, 2, 3}, k=3) == 0.0

    def test_partial(self):
        result = precision_at_k([1, 2, 3, 4, 5], {1, 3, 5}, k=5)
        assert abs(result - 0.6) < 1e-9

    def test_k_zero(self):
        assert precision_at_k([1, 2], {1, 2}, k=0) == 0.0


class TestRecallAtK:
    def test_perfect(self):
        assert recall_at_k([1, 2, 3], {1, 2, 3}, k=3) == 1.0

    def test_zero(self):
        assert recall_at_k([4, 5], {1, 2, 3}, k=5) == 0.0

    def test_partial(self):
        # 2 trong 4 relevant docs được tìm thấy
        result = recall_at_k([1, 2, 5, 6], {1, 2, 3, 4}, k=4)
        assert abs(result - 0.5) < 1e-9

    def test_empty_relevant(self):
        assert recall_at_k([1, 2], set(), k=5) == 0.0


class TestAveragePrecision:
    def test_perfect(self):
        assert average_precision([1, 2, 3], {1, 2, 3}) == 1.0

    def test_zero(self):
        assert average_precision([4, 5, 6], {1, 2, 3}) == 0.0

    def test_known_value(self):
        # Retrieved [1, 2, 3, 4, 5], relevant {1, 3}
        # Precision tại rank 1 (doc 1 relevant): 1/1 = 1.0
        # Precision tại rank 3 (doc 3 relevant): 2/3 ≈ 0.667
        # AP = (1.0 + 2/3) / 2 = 0.833...
        result = average_precision([1, 2, 3, 4, 5], {1, 3})
        assert abs(result - (1.0 + 2 / 3) / 2) < 1e-9

    def test_empty_relevant(self):
        assert average_precision([1, 2, 3], set()) == 0.0


class TestMAP:
    def test_single_query(self):
        map_score = mean_average_precision([[1, 2, 3]], [{1, 2, 3}])
        assert map_score == 1.0

    def test_multiple_queries(self):
        results   = [[1, 2, 3], [4, 5, 6]]
        relevant  = [{1, 2, 3}, {4, 5, 6}]
        assert mean_average_precision(results, relevant) == 1.0

    def test_empty(self):
        assert mean_average_precision([], []) == 0.0


class TestNDCG:
    def test_perfect(self):
        assert ndcg_at_k([1, 2, 3], {1, 2, 3}, k=3) == 1.0

    def test_zero(self):
        assert ndcg_at_k([4, 5, 6], {1, 2, 3}, k=3) == 0.0

    def test_empty_relevant(self):
        assert ndcg_at_k([1, 2], set(), k=5) == 0.0

    def test_partial(self):
        # Chỉ một trong ba kết quả là relevant
        result = ndcg_at_k([1, 2, 3], {1}, k=3)
        assert 0 < result <= 1.0
