"""tests/test_processing.py – Unit tests cho module processing."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.processing.cleaner import clean_text
from src.processing.tokenizer import tokenize


class TestCleanText:
    def test_lowercase(self):
        assert clean_text("Hello WORLD") == "hello world"

    def test_remove_html(self):
        result = clean_text("<p>xin chào</p>")
        assert "<p>" not in result
        assert "xin chào" in result

    def test_remove_url(self):
        result = clean_text("Xem tại https://example.com nhé")
        assert "https" not in result

    def test_normalize_spaces(self):
        result = clean_text("  hello    world  ")
        assert result == "hello world"

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_non_string(self):
        assert clean_text(None) == ""  # type: ignore[arg-type]

    def test_keep_vietnamese(self):
        result = clean_text("Học máy là gì?")
        assert "học" in result
        assert "máy" in result


class TestTokenize:
    def test_basic(self):
        tokens = tokenize("học máy là gì")
        assert isinstance(tokens, list)
        assert len(tokens) > 0

    def test_empty(self):
        assert tokenize("") == []

    def test_min_len_filter(self):
        # Token ngắn hơn min_len nên bị bỏ
        tokens = tokenize("a b c học máy", min_len=2)
        for t in tokens:
            assert len(t) >= 2

    def test_stopwords_filter(self):
        tokens = tokenize("học máy là gì", stopwords={"là", "gì"}, min_len=1)
        assert "là" not in tokens
        assert "gì" not in tokens
