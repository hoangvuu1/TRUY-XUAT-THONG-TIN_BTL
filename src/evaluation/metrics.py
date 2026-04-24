"""
metrics.py – Các metrics đánh giá hệ thống IR.

Triển khai các metrics chuẩn trong Information Retrieval:
  - Precision@K       : Tỉ lệ tài liệu liên quan trong Top-K kết quả
  - Recall@K          : Tỉ lệ tài liệu liên quan được tìm thấy trong Top-K
  - Average Precision : AP cho một truy vấn
  - MAP               : Mean Average Precision trên toàn bộ tập test
  - NDCG@K            : Normalized Discounted Cumulative Gain

Tài liệu tham khảo:
  Manning et al., "Introduction to Information Retrieval", Chapter 8.
"""
from __future__ import annotations

import math
from typing import List, Set


def precision_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
    """Precision@K – tỉ lệ tài liệu liên quan trong Top-K kết quả.

    Args:
        retrieved: Danh sách doc_id trả về theo thứ tự xếp hạng.
        relevant : Tập doc_id liên quan (ground truth).
        k        : Giá trị K.

    Returns:
        P@K ∈ [0, 1].

    Examples:
        >>> precision_at_k([1, 2, 3, 4, 5], {1, 3, 5}, k=5)
        0.6
    """
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for doc in top_k if doc in relevant)
    return hits / k


def recall_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
    """Recall@K – tỉ lệ tài liệu liên quan được tìm thấy trong Top-K.

    Args:
        retrieved: Danh sách doc_id trả về.
        relevant : Tập doc_id liên quan.
        k        : Giá trị K.

    Returns:
        R@K ∈ [0, 1]. Trả về 0 nếu tập relevant rỗng.
    """
    if not relevant or k <= 0:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for doc in top_k if doc in relevant)
    return hits / len(relevant)


def average_precision(retrieved: List[int], relevant: Set[int]) -> float:
    """Average Precision (AP) cho một truy vấn.

    AP = (1/|R|) × Σ_{k: retrieved[k]∈R} P@k

    Args:
        retrieved: Danh sách doc_id trả về theo thứ tự xếp hạng.
        relevant : Tập doc_id liên quan (ground truth).

    Returns:
        AP ∈ [0, 1]. Trả về 0 nếu tập relevant rỗng.

    Examples:
        >>> average_precision([1, 2, 3, 4, 5], {1, 3})
        0.5833...
    """
    if not relevant:
        return 0.0
    hits = 0
    sum_precision = 0.0
    for rank, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            hits += 1
            sum_precision += hits / rank
    return sum_precision / len(relevant)


def mean_average_precision(
    results_list: List[List[int]],
    relevant_list: List[Set[int]],
) -> float:
    """Mean Average Precision (MAP) trên tập nhiều truy vấn.

    MAP = (1/|Q|) × Σ AP(q)

    Args:
        results_list  : Danh sách kết quả cho từng query (list of lists).
        relevant_list : Danh sách tập relevant cho từng query (list of sets).

    Returns:
        MAP ∈ [0, 1].
    """
    if not results_list:
        return 0.0
    aps = [
        average_precision(retrieved, relevant)
        for retrieved, relevant in zip(results_list, relevant_list)
    ]
    return sum(aps) / len(aps)


def ndcg_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
    """Normalized Discounted Cumulative Gain @ K (binary relevance).

    DCG@K  = Σ_{i=1}^{K} rel_i / log2(i+1)
    IDCG@K = DCG của ranking hoàn hảo (relevant docs ở đầu)
    NDCG@K = DCG@K / IDCG@K

    Args:
        retrieved: Danh sách doc_id trả về.
        relevant : Tập doc_id liên quan.
        k        : Giá trị K.

    Returns:
        NDCG@K ∈ [0, 1].
    """
    if not relevant or k <= 0:
        return 0.0

    # DCG
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc in enumerate(retrieved[:k], start=1)
        if doc in relevant
    )

    # Ideal DCG – giả sử tất cả relevant docs được xếp hạng trước
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))

    return dcg / idcg if idcg > 0 else 0.0
