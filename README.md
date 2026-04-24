# 🔍 Hệ Thống Tìm Kiếm Tài Liệu Tiếng Việt

> Mini search engine cho tài liệu tiếng Việt sử dụng **BM25 + TF-IDF + Query Expansion**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-green)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 📖 Giới thiệu

Dự án xây dựng một hệ thống tìm kiếm tài liệu tiếng Việt hoàn chỉnh, áp dụng các kỹ thuật cốt lõi của môn **Truy Xuất Thông Tin (Information Retrieval)**:

| Thành phần | Chi tiết |
|---|---|
| Tokenization | `underthesea` – word segmentation tiếng Việt |
| Inverted Index | Tự xây dựng bằng Python |
| Ranking | TF-IDF (scikit-learn) và BM25 (tự cài đặt) |
| Query Expansion | Synonym expansion dựa trên từ điển đồng nghĩa tiếng Việt |
| Evaluation | Precision@K, Recall, MAP, NDCG |
| Web UI | Flask + Bootstrap 5 |

---

## 🗂️ Cấu trúc project

```
TRUY-XUAT-THONG-TIN_BTL/
│
├── README.md               # Tài liệu này
├── PLAN.md                 # Kế hoạch triển khai chi tiết
├── requirements.txt        # Danh sách thư viện
│
├── src/                    # Source code chính
│   ├── __init__.py
│   ├── config.py           # Cấu hình toàn cục (đường dẫn, tham số)
│   │
│   ├── processing/         # Phase 1 – Tiền xử lý văn bản
│   │   ├── __init__.py
│   │   ├── cleaner.py      # Làm sạch text (lowercase, bỏ ký tự đặc biệt)
│   │   └── tokenizer.py    # Tokenization tiếng Việt (underthesea)
│   │
│   ├── indexing/           # Phase 2 – Xây dựng chỉ mục
│   │   ├── __init__.py
│   │   └── index_builder.py  # Xây dựng & lưu Inverted Index
│   │
│   ├── searching/          # Phase 3 – Tìm kiếm & xếp hạng
│   │   ├── __init__.py
│   │   ├── bm25.py         # BM25 ranking
│   │   ├── tfidf.py        # TF-IDF ranking
│   │   └── query_expansion.py  # Mở rộng truy vấn
│   │
│   ├── evaluation/         # Phase 4 – Đánh giá
│   │   ├── __init__.py
│   │   └── metrics.py      # Precision, Recall, MAP, NDCG
│   │
│   └── web/                # Phase 3 – Giao diện web
│       ├── __init__.py
│       ├── app.py          # Flask application
│       ├── routes.py       # API routes
│       └── templates/
│           └── index.html  # Giao diện tìm kiếm
│
├── data/
│   ├── raw/                # Dữ liệu gốc (CSV từ Kaggle, v.v.)
│   ├── processed/          # Dữ liệu sau khi tiền xử lý
│   └── index/              # Inverted index đã build
│
├── notebooks/              # Jupyter notebooks hướng dẫn
│   └── 01_data_exploration.ipynb
│
├── scripts/                # Script chạy từng bước / toàn pipeline
│   ├── preprocess.py
│   ├── build_index.py
│   ├── run_search.py
│   ├── evaluate.py
│   └── run_pipeline.py     # Chạy toàn bộ pipeline
│
└── tests/                  # Unit tests
    ├── test_processing.py
    ├── test_indexing.py
    ├── test_searching.py
    └── test_evaluation.py
```

---

## 🚀 Cài đặt & Chạy thử

### 1. Clone repository

```bash
git clone https://github.com/hoangvuu1/TRUY-XUAT-THONG-TIN_BTL.git
cd TRUY-XUAT-THONG-TIN_BTL
```

### 2. Tạo môi trường ảo (khuyến nghị)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### 4. Chuẩn bị dữ liệu

Tải dataset **Vietnamese Online News** từ Kaggle:
<https://www.kaggle.com/datasets/sarahhimeko/vietnamese-online-news-csv-dataset>

Đặt file CSV vào thư mục `data/raw/`:

```
data/raw/vietnamese_news.csv
```

### 5. Chạy toàn bộ pipeline

```bash
python scripts/run_pipeline.py
```

Hoặc chạy từng bước:

```bash
# Bước 1: Tiền xử lý
python scripts/preprocess.py

# Bước 2: Xây dựng index
python scripts/build_index.py

# Bước 3: Khởi động web
python -m flask --app src/web/app.py run --debug
```

### 6. Truy cập giao diện web

Mở trình duyệt tại: <http://localhost:5000>

---

## 🧪 Chạy tests

```bash
pytest tests/ -v --cov=src
```

---

## 📊 Demo

```
Query: "học máy ứng dụng y tế"

Kết quả (Top 5):
1. [Score: 0.94] Ứng dụng trí tuệ nhân tạo trong chẩn đoán bệnh...
2. [Score: 0.87] Học máy giúp phát hiện ung thư sớm...
3. [Score: 0.82] AI trong ngành y tế Việt Nam năm 2024...
4. [Score: 0.76] Công nghệ deep learning phân tích hình ảnh y khoa...
5. [Score: 0.71] Bệnh viện Bạch Mai ứng dụng AI hỗ trợ bác sĩ...
```

---

## ⚙️ Tham số cấu hình

Mở `src/config.py` để điều chỉnh:

```python
BM25_K1 = 1.5          # Tham số BM25 k1
BM25_B  = 0.75         # Tham số BM25 b
TOP_K   = 10           # Số kết quả trả về
MAX_DOCS = 10_000      # Giới hạn số tài liệu xử lý
```

---

## 📚 Tài liệu tham khảo

- Manning et al., *Introduction to Information Retrieval*, Cambridge University Press, 2008. [Online](https://nlp.stanford.edu/IR-book/)
- Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond*, 2009.
- `underthesea` – Vietnamese NLP Toolkit: <https://github.com/undertheseanlp/underthesea>
- Kaggle dataset: <https://www.kaggle.com/datasets/sarahhimeko/vietnamese-online-news-csv-dataset>

---

## 👥 Nhóm thực hiện

| Tên | MSSV | Vai trò |
|---|---|---|
| Nguyễn Hoàng Vũ | ... | Project Lead |

---

## 📄 License

MIT License – xem file [LICENSE](LICENSE) để biết thêm chi tiết.
