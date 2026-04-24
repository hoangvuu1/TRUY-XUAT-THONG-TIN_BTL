# 📋 PLAN.md – Kế hoạch triển khai hệ thống tìm kiếm tài liệu tiếng Việt

> **Đề tài:** Xây dựng hệ thống tìm kiếm tài liệu tiếng Việt sử dụng BM25 + TF-IDF + Query Expansion  
> **Môn học:** Truy Xuất Thông Tin (Information Retrieval)

---

## ✅ Checklist tiến độ tổng thể

- [ ] **Phase 1** – Thu thập & xử lý dữ liệu
  - [ ] 1.1 Download & khảo sát dataset
  - [ ] 1.2 Làm sạch văn bản
  - [ ] 1.3 Tokenization tiếng Việt
  - [ ] 1.4 Lưu dữ liệu đã xử lý
- [ ] **Phase 2** – Xây dựng Index & Model
  - [ ] 2.1 Xây dựng Inverted Index
  - [ ] 2.2 Cài đặt TF-IDF
  - [ ] 2.3 Cài đặt BM25
  - [ ] 2.4 Lưu index xuống đĩa
- [ ] **Phase 3** – Giao diện tìm kiếm
  - [ ] 3.1 Query Expansion
  - [ ] 3.2 Search pipeline (nhận query → trả kết quả)
  - [ ] 3.3 Flask web app
  - [ ] 3.4 Giao diện HTML/Bootstrap
- [ ] **Phase 4** – Đánh giá & báo cáo
  - [ ] 4.1 Xây dựng tập test (queries + relevance judgments)
  - [ ] 4.2 Tính Precision@K, Recall, MAP, NDCG
  - [ ] 4.3 So sánh TF-IDF vs BM25 vs BM25+Expansion
  - [ ] 4.4 Viết báo cáo & slide

---

## 🗓️ Timeline gợi ý

| Tuần | Phase | Mục tiêu |
|------|-------|-----------|
| 1    | Phase 1 | Dataset sẵn sàng, tokenization chạy được |
| 2    | Phase 2 | Inverted index + TF-IDF hoạt động |
| 3    | Phase 3 | BM25 + Query Expansion + Web UI |
| 4    | Phase 4 | Evaluation, báo cáo, demo |

---

## 🔵 Phase 1 – Thu thập & Xử lý Dữ liệu

### 1.1 Download & Khảo sát Dataset

**Nguồn dữ liệu:** [Vietnamese Online News CSV Dataset](https://www.kaggle.com/datasets/sarahhimeko/vietnamese-online-news-csv-dataset)

- ~150 000 bài báo tiếng Việt (VnExpress, Tuổi Trẻ, Dân Trí, ...)
- Cột: `title`, `content`, `category`, `date_published`, `source`

**Công việc:**
1. Download file CSV về `data/raw/`
2. Dùng notebook `notebooks/01_data_exploration.ipynb` để khám phá:
   - Phân phối category
   - Độ dài trung bình của bài viết
   - Kiểm tra missing values

```python
import pandas as pd

df = pd.read_csv("data/raw/vietnamese_news.csv")
print(df.shape)
print(df.isnull().sum())
print(df["category"].value_counts())
```

**Output mong đợi:** `data/raw/vietnamese_news.csv` đã tải về, đã khảo sát xong.

---

### 1.2 Làm sạch văn bản (`src/processing/cleaner.py`)

Các bước làm sạch:
1. Chuyển lowercase
2. Loại bỏ HTML tags (nếu có)
3. Loại bỏ ký tự đặc biệt, giữ lại chữ, số, dấu tiếng Việt
4. Chuẩn hoá khoảng trắng

```python
import re

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)          # bỏ HTML
    text = re.sub(r"[^\w\s\u00C0-\u024F]", " ", text)  # giữ Unicode
    text = re.sub(r"\s+", " ", text).strip()
    return text
```

**Output mong đợi:** Văn bản sạch, không có HTML/ký tự rác.

---

### 1.3 Tokenization tiếng Việt (`src/processing/tokenizer.py`)

Dùng thư viện `underthesea`:

```python
from underthesea import word_tokenize

def tokenize(text: str) -> list[str]:
    """Tách từ tiếng Việt, trả về list token."""
    tokens = word_tokenize(text, format="text").split()
    return tokens
```

**Lưu ý:**
- `underthesea.word_tokenize` ghép các từ ghép bằng dấu `_` (VD: `học_máy`)
- Loại bỏ stopwords nếu cần (file `data/processed/stopwords.txt`)
- Có thể dùng VnCoreNLP thay thế nếu cần POS tagging

**Output mong đợi:**
```
"học máy là gì" → ["học_máy", "là", "gì"]
```

---

### 1.4 Lưu dữ liệu đã xử lý

Lưu dưới dạng JSON Lines để dễ load:

```python
# data/processed/documents.jsonl
{"doc_id": 0, "title": "...", "tokens": ["học_máy", "là", ...]}
{"doc_id": 1, "title": "...", "tokens": [...]}
```

Script: `scripts/preprocess.py`

---

## 🟡 Phase 2 – Xây dựng Index & Model

### 2.1 Xây dựng Inverted Index (`src/indexing/index_builder.py`)

Cấu trúc Inverted Index:

```
index = {
    "học_máy": {0: 5, 1: 2, 9: 1},   # term → {doc_id: term_freq}
    "hà_nội":  {2: 3, 4: 1},
    ...
}
```

Thuật toán:
```
FOR mỗi tài liệu d:
    FOR mỗi token t trong d:
        index[t][doc_id] += 1
```

Lưu ý: Lưu thêm `doc_lengths` (độ dài mỗi tài liệu) và `avg_doc_length` cho BM25.

---

### 2.2 TF-IDF Ranking (`src/searching/tfidf.py`)

Dùng `scikit-learn.TfidfVectorizer`:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(tokenizer=my_tokenizer, max_features=50_000)
tfidf_matrix = vectorizer.fit_transform(corpus)
```

Tìm kiếm bằng cosine similarity:
```python
from sklearn.metrics.pairwise import cosine_similarity

query_vec = vectorizer.transform([query])
scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
top_k = scores.argsort()[::-1][:10]
```

---

### 2.3 BM25 Ranking (`src/searching/bm25.py`)

Công thức BM25:

```
score(q, d) = Σ IDF(t) * (tf(t,d) * (k1+1)) / (tf(t,d) + k1*(1 - b + b*|d|/avgdl))
```

Tham số mặc định: `k1 = 1.5`, `b = 0.75`

```python
class BM25:
    def __init__(self, k1=1.5, b=0.75): ...
    def fit(self, corpus): ...           # Xây dựng index từ corpus
    def get_scores(self, query): ...     # Trả về mảng score cho mỗi doc
    def get_top_k(self, query, k=10): ...
```

---

### 2.4 Lưu Index xuống đĩa

```python
import joblib

# Lưu
joblib.dump(bm25_model, "data/index/bm25.pkl")
joblib.dump(vectorizer, "data/index/tfidf_vectorizer.pkl")
joblib.dump(tfidf_matrix, "data/index/tfidf_matrix.pkl")
```

Script: `scripts/build_index.py`

---

## 🟢 Phase 3 – Giao diện Tìm kiếm

### 3.1 Query Expansion (`src/searching/query_expansion.py`)

Chiến lược mở rộng truy vấn:

**Cách 1: Synonym Dictionary (đơn giản, không cần model)**
```python
SYNONYMS = {
    "AI": ["trí_tuệ_nhân_tạo", "machine_learning", "học_máy"],
    "du_lịch": ["tham_quan", "khám_phá", "nghỉ_dưỡng"],
    ...
}

def expand_query(query_tokens: list[str]) -> list[str]:
    expanded = list(query_tokens)
    for token in query_tokens:
        expanded.extend(SYNONYMS.get(token, []))
    return list(set(expanded))
```

**Cách 2: Word co-occurrence (nâng cao)**
- Tìm các từ xuất hiện cùng nhau trong corpus
- Thêm các từ có PMI cao nhất vào query

---

### 3.2 Search Pipeline (`src/searching/`)

```
User Query
    │
    ▼
[1] Làm sạch & Tokenize
    │
    ▼
[2] Query Expansion
    │
    ▼
[3] Ranking (TF-IDF hoặc BM25)
    │
    ▼
[4] Trả về Top-K documents
```

---

### 3.3 Flask Web App (`src/web/app.py`, `src/web/routes.py`)

Endpoint chính:
- `GET /` – Trang chủ
- `POST /search` – Nhận `query`, trả về JSON kết quả
- `GET /health` – Health check

```python
@app.route("/search", methods=["POST"])
def search():
    data = request.get_json()
    query = data.get("query", "")
    method = data.get("method", "bm25")  # "tfidf" | "bm25" | "bm25_expand"
    results = search_engine.search(query, method=method, top_k=10)
    return jsonify(results)
```

---

### 3.4 Giao diện HTML (`src/web/templates/index.html`)

Tính năng UI:
- 🔍 Search box nhập truy vấn
- 📋 Dropdown chọn phương pháp (TF-IDF / BM25 / BM25+Expansion)
- 📄 Hiển thị Top-K kết quả với: tiêu đề, đoạn trích, score, category
- ⚡ Thời gian xử lý query

---

## 🔴 Phase 4 – Đánh giá & Báo cáo

### 4.1 Tập test

Tạo `data/processed/test_queries.json`:
```json
[
  {
    "query": "học máy ứng dụng y tế",
    "relevant_doc_ids": [12, 45, 102, 234]
  },
  ...
]
```

**Quy trình tạo ground truth:**
1. Chọn 50–100 queries đại diện
2. Với mỗi query, đánh dấu tay top-20 tài liệu liên quan

---

### 4.2 Metrics (`src/evaluation/metrics.py`)

```python
def precision_at_k(retrieved, relevant, k): ...
def recall_at_k(retrieved, relevant, k): ...
def average_precision(retrieved, relevant): ...
def mean_average_precision(results_list, relevant_list): ...
def ndcg_at_k(retrieved, relevant, k): ...
```

---

### 4.3 So sánh kết quả

| Method | MAP | P@10 | NDCG@10 |
|--------|-----|------|---------|
| TF-IDF | ... | ... | ... |
| BM25 | ... | ... | ... |
| BM25 + Expansion | ... | ... | ... |

Script: `scripts/evaluate.py`

---

### 4.4 Báo cáo & Slide

Cấu trúc báo cáo gợi ý:
1. **Giới thiệu** – Bài toán, mục tiêu, dataset
2. **Cơ sở lý thuyết** – TF-IDF, BM25, Query Expansion, Inverted Index
3. **Thiết kế hệ thống** – Kiến trúc, các module
4. **Thực nghiệm** – Thiết lập, kết quả, phân tích
5. **Kết luận** – Nhận xét, hướng phát triển
6. **Tài liệu tham khảo**

---

## 📌 Ghi chú kỹ thuật

### Hiệu năng
- Nếu corpus > 50k tài liệu, dùng sparse matrix (`scipy.sparse`) để tiết kiệm RAM
- Cache inverted index bằng `joblib`
- Giới hạn `MAX_DOCS` trong `config.py` khi phát triển

### Mở rộng
- Thêm **Neural Reranking** (BERT/PhoBERT) cho Phase nâng cao
- Tích hợp **Elasticsearch** để scale lên production
- Thêm **Feedback** – người dùng đánh dấu kết quả liên quan → cải thiện model

---

*Cập nhật lần cuối: 2026-04-24*
