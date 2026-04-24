"""
query_expansion.py – Mở rộng truy vấn (Query Expansion).

Chiến lược mặc định: Synonym Dictionary
  - Duy trì một dict từ đồng nghĩa tiếng Việt đơn giản
  - Mở rộng mỗi token bằng các từ đồng nghĩa tương ứng

Có thể mở rộng thêm:
  - Word co-occurrence matrix
  - PhoBERT / Word2Vec embeddings
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

# ── Từ điển đồng nghĩa đơn giản (tiếng Việt) ─────────────────────────────────
# Khoá là token (đã tokenize bằng underthesea, dấu _ cho từ ghép)
# Giá trị là danh sách các từ đồng nghĩa / liên quan
DEFAULT_SYNONYMS: Dict[str, List[str]] = {
    # Công nghệ
    "ai":                   ["trí_tuệ_nhân_tạo", "machine_learning", "học_máy", "deep_learning"],
    "trí_tuệ_nhân_tạo":     ["ai", "machine_learning", "học_máy"],
    "học_máy":              ["machine_learning", "trí_tuệ_nhân_tạo", "ai", "deep_learning"],
    "deep_learning":        ["học_sâu", "mạng_nơ_ron", "neural_network"],
    # Du lịch
    "du_lịch":              ["tham_quan", "khám_phá", "nghỉ_dưỡng", "đi_chơi"],
    "hà_nội":               ["thủ_đô", "hà_nội"],
    "tp_hcm":               ["sài_gòn", "hồ_chí_minh", "thành_phố_hồ_chí_minh"],
    # Y tế
    "bệnh_viện":            ["cơ_sở_y_tế", "phòng_khám", "trung_tâm_y_tế"],
    "bác_sĩ":               ["thầy_thuốc", "y_sĩ", "chuyên_gia_y_tế"],
    # Kinh tế
    "kinh_tế":              ["tài_chính", "thị_trường", "đầu_tư"],
    "chứng_khoán":          ["cổ_phiếu", "thị_trường_chứng_khoán", "sàn_giao_dịch"],
    # Giáo dục
    "đại_học":              ["trường_đại_học", "cao_đẳng", "trường_học"],
    "sinh_viên":            ["học_sinh", "học_viên", "người_học"],
    # Thể thao
    "bóng_đá":              ["football", "soccer", "sân_cỏ"],
    "olympic":              ["thế_vận_hội", "vận_động_viên"],
}


class QueryExpander:
    """Mở rộng truy vấn bằng từ điển đồng nghĩa.

    Usage:
        expander = QueryExpander()
        tokens = ["ai", "ứng_dụng"]
        expanded = expander.expand(tokens)
        # → ["ai", "ứng_dụng", "trí_tuệ_nhân_tạo", "machine_learning", "học_máy", "deep_learning"]
    """

    def __init__(
        self,
        synonyms: Dict[str, List[str]] | None = None,
        max_expansion: int = 5,
    ) -> None:
        """
        Args:
            synonyms: Từ điển đồng nghĩa tùy chỉnh. Nếu None dùng DEFAULT_SYNONYMS.
            max_expansion: Số từ đồng nghĩa tối đa thêm vào cho mỗi token.
        """
        self.synonyms: Dict[str, List[str]] = synonyms if synonyms is not None else DEFAULT_SYNONYMS
        self.max_expansion = max_expansion

    def expand(self, tokens: List[str]) -> List[str]:
        """Mở rộng danh sách token bằng từ đồng nghĩa.

        Giữ nguyên thứ tự token gốc, thêm từ đồng nghĩa vào cuối.
        Loại bỏ trùng lặp, giữ thứ tự xuất hiện.

        Args:
            tokens: Danh sách token của truy vấn gốc.

        Returns:
            Danh sách token đã mở rộng.
        """
        seen: set[str] = set()
        result: List[str] = []

        # Thêm token gốc trước
        for t in tokens:
            if t not in seen:
                result.append(t)
                seen.add(t)

        # Thêm từ đồng nghĩa
        for t in tokens:
            synonyms = self.synonyms.get(t, [])
            for syn in synonyms[: self.max_expansion]:
                if syn not in seen:
                    result.append(syn)
                    seen.add(syn)

        if len(result) > len(tokens):
            logger.debug(
                "Query expansion: %s → %s",
                tokens,
                result,
            )
        return result

    def load_custom_synonyms(self, filepath: str | Path) -> None:
        """Load từ điển đồng nghĩa từ file TSV (term\\tsynonym1,synonym2,...).

        Args:
            filepath: Đường dẫn file TSV.
        """
        path = Path(filepath)
        if not path.exists():
            logger.warning("Không tìm thấy file synonyms: %s", path)
            return
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) >= 2:
                    term = parts[0].strip()
                    syns = [s.strip() for s in parts[1].split(",") if s.strip()]
                    self.synonyms[term] = syns
        logger.info("Đã load synonyms từ %s", path)
