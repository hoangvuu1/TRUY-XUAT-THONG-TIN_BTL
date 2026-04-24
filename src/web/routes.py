"""
routes.py – Định nghĩa các URL route cho Flask app.

Routes:
  GET  /           – Trang chủ (search UI)
  POST /api/search – API tìm kiếm, trả về JSON
  GET  /health     – Health check
"""
from __future__ import annotations

import logging

from flask import Blueprint, current_app, jsonify, render_template, request

bp = Blueprint("main", __name__)
logger = logging.getLogger(__name__)


@bp.route("/")
def index():
    """Trang chủ – hiển thị giao diện tìm kiếm."""
    return render_template("index.html")


@bp.route("/api/search", methods=["POST"])
def search():
    """API tìm kiếm.

    Request JSON:
        {
            "query" : "học máy ứng dụng y tế",
            "method": "bm25",     // "tfidf" | "bm25" | "bm25_expand"
            "top_k" : 10
        }

    Response JSON:
        {
            "query"         : "...",
            "expanded_query": "..." | null,
            "method"        : "bm25",
            "results"       : [...],
            "elapsed_ms"    : 12.3
        }
    """
    data = request.get_json(force=True, silent=True) or {}
    query = str(data.get("query", "")).strip()
    method = str(data.get("method", "bm25")).strip()
    top_k = int(data.get("top_k", 10))

    if not query:
        return jsonify({"error": "Vui lòng nhập từ khoá tìm kiếm."}), 400

    if method not in ("tfidf", "bm25", "bm25_expand"):
        return jsonify({"error": f"Phương pháp không hợp lệ: {method}"}), 400

    engine = current_app.config.get("SEARCH_ENGINE")
    if engine is None:
        return jsonify({"error": "Search engine chưa sẵn sàng."}), 503

    result = engine.search(query, method=method, top_k=top_k)
    return jsonify(result)


@bp.route("/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok"})
