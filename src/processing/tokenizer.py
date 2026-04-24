"""
tokenizer.py – Tokenization tiếng Việt dùng thư viện underthesea.

underthesea.word_tokenize ghép các từ ghép bằng dấu gạch dưới,
ví dụ: "học máy" → "học_máy".

Tham khảo: https://github.com/undertheseanlp/underthesea
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Lazy-load underthesea để tránh lỗi nếu chưa cài
try:
    from underthesea import word_tokenize as _word_tokenize
    _UNDERTHESEA_AVAILABLE = True
except ImportError:
    _UNDERTHESEA_AVAILABLE = False
    logger.warning(
        "underthesea chưa được cài đặt. "
        "Chạy: pip install underthesea\n"
        "Tokenizer sẽ fallback về tách theo khoảng trắng."
    )


def load_stopwords(filepath: str | Path | None = None) -> set[str]:
    """Đọc danh sách stopwords từ file (mỗi dòng một từ).

    Args:
        filepath: Đường dẫn đến file stopwords. Nếu None, trả về set rỗng.

    Returns:
        Tập hợp các stopword.
    """
    if filepath is None:
        return set()
    path = Path(filepath)
    if not path.exists():
        logger.warning("Không tìm thấy file stopwords: %s", path)
        return set()
    with open(path, encoding="utf-8") as f:
        words = {line.strip() for line in f if line.strip()}
    logger.info("Đã load %d stopwords từ %s", len(words), path)
    return words


def tokenize(
    text: str,
    stopwords: set[str] | None = None,
    min_len: int = 2,
) -> list[str]:
    """Tách từ tiếng Việt và lọc token.

    Args:
        text: Văn bản đã làm sạch (chữ thường, không ký tự đặc biệt).
        stopwords: Tập stopwords cần loại bỏ.
        min_len: Độ dài tối thiểu của token (tính theo ký tự).

    Returns:
        Danh sách tokens sau khi tách từ và lọc.

    Examples:
        >>> tokenize("học máy là gì")
        ['học_máy', 'là', 'gì']
    """
    if not text:
        return []

    if _UNDERTHESEA_AVAILABLE:
        # format="text" trả về chuỗi các từ, phân cách bằng khoảng trắng
        # từ ghép được nối bằng dấu gạch dưới, VD: "học_máy"
        tokens = _word_tokenize(text, format="text").split()
    else:
        # Fallback đơn giản nếu chưa cài underthesea
        tokens = text.split()

    # Lọc stopwords và token quá ngắn
    sw = stopwords or set()
    tokens = [
        t for t in tokens
        if len(t) >= min_len and t not in sw
    ]

    return tokens
