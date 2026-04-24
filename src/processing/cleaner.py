"""
cleaner.py – Làm sạch văn bản tiếng Việt.

Các bước xử lý:
1. Chuyển về chữ thường
2. Loại bỏ HTML tags
3. Loại bỏ URL
4. Loại bỏ ký tự đặc biệt, giữ lại chữ, số, khoảng trắng và dấu tiếng Việt
5. Chuẩn hoá khoảng trắng
"""
import re
import unicodedata


# Pattern loại bỏ HTML tags
_HTML_TAG = re.compile(r"<[^>]+>")

# Pattern URL
_URL = re.compile(
    r"https?://\S+|www\.\S+",
    flags=re.IGNORECASE,
)

# Giữ lại: chữ cái (Latin + Unicode mở rộng cho tiếng Việt), chữ số, khoảng trắng
# Loại bỏ: ký tự đặc biệt, dấu câu
_KEEP_CHARS = re.compile(r"[^\w\s]", flags=re.UNICODE)

# Nhiều khoảng trắng liên tiếp → một khoảng trắng
_MULTI_SPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Làm sạch một đoạn văn bản thô.

    Args:
        text: Văn bản đầu vào (có thể chứa HTML, URL, ký tự đặc biệt).

    Returns:
        Văn bản đã làm sạch, chữ thường, không dư khoảng trắng.
    """
    if not isinstance(text, str):
        return ""

    # Chuẩn hoá Unicode (NFC: tổ hợp dấu thành ký tự đơn)
    text = unicodedata.normalize("NFC", text)

    # Chuyển thường
    text = text.lower()

    # Bỏ HTML
    text = _HTML_TAG.sub(" ", text)

    # Bỏ URL
    text = _URL.sub(" ", text)

    # Bỏ ký tự đặc biệt (giữ chữ, số, dấu tiếng Việt, khoảng trắng)
    text = _KEEP_CHARS.sub(" ", text)

    # Chuẩn hoá khoảng trắng
    text = _MULTI_SPACE.sub(" ", text).strip()

    return text
