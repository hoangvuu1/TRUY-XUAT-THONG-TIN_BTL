"""
config.py – Cấu hình toàn cục của hệ thống.
Điều chỉnh các tham số tại đây trước khi chạy pipeline.
"""
import os
from pathlib import Path

# ── Đường dẫn gốc ─────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent

# ── Dữ liệu ──────────────────────────────────────────────────────────────────
DATA_DIR       = ROOT_DIR / "data"
RAW_DATA_DIR   = DATA_DIR / "raw"
PROC_DATA_DIR  = DATA_DIR / "processed"
INDEX_DIR      = DATA_DIR / "index"

# File đầu vào mặc định (đặt file CSV vào data/raw/)
RAW_CSV_FILE   = RAW_DATA_DIR / "vietnamese_news.csv"

# File sau xử lý
PROCESSED_FILE = PROC_DATA_DIR / "documents.jsonl"
STOPWORDS_FILE = PROC_DATA_DIR / "stopwords.txt"

# ── Index / Model ─────────────────────────────────────────────────────────────
BM25_INDEX_FILE         = INDEX_DIR / "bm25.pkl"
TFIDF_VECTORIZER_FILE   = INDEX_DIR / "tfidf_vectorizer.pkl"
TFIDF_MATRIX_FILE       = INDEX_DIR / "tfidf_matrix.pkl"
DOC_METADATA_FILE       = INDEX_DIR / "doc_metadata.pkl"

# ── Tham số tiền xử lý ────────────────────────────────────────────────────────
MAX_DOCS = int(os.getenv("MAX_DOCS", 0))       # 0 = không giới hạn (load toàn bộ dataset)
MIN_TOKEN_LEN = 2                              # Bỏ token ngắn hơn n ký tự
N_JOBS = int(os.getenv("N_JOBS", -1))          # Số tiến trình song song (-1 = dùng hết CPU)

# ── Tham số BM25 ──────────────────────────────────────────────────────────────
BM25_K1 = float(os.getenv("BM25_K1", 1.5))
BM25_B  = float(os.getenv("BM25_B", 0.75))

# ── Tham số tìm kiếm ──────────────────────────────────────────────────────────
TOP_K = int(os.getenv("TOP_K", 10))           # Số kết quả trả về

# ── Web server ────────────────────────────────────────────────────────────────
FLASK_HOST  = os.getenv("FLASK_HOST", "0.0.0.0")
FLASK_PORT  = int(os.getenv("FLASK_PORT", 5000))
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "false").lower() == "true"
