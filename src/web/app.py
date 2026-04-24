"""
app.py – Flask application factory.

Khởi tạo Flask app, load search engine, đăng ký blueprints/routes.

Chạy development server:
    flask --app src/web/app.py run --debug

Hoặc:
    python -m flask --app src/web/app.py run
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from flask import Flask

# Đảm bảo import được module src
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src import config  # noqa: E402 – sau sys.path adjustment

logger = logging.getLogger(__name__)


def create_app() -> Flask:
    """Application factory – tạo và cấu hình Flask app.

    Returns:
        Flask app đã cấu hình xong.
    """
    app = Flask(__name__, template_folder="templates")
    app.config["SECRET_KEY"] = "change-me-in-production"

    # ── Khởi tạo Search Engine ────────────────────────────────────────────────
    from src.web.search_engine import SearchEngine  # lazy import

    search_engine = SearchEngine()
    search_engine.load()
    app.config["SEARCH_ENGINE"] = search_engine

    # ── Đăng ký routes ───────────────────────────────────────────────────────
    from src.web.routes import bp  # noqa: E402

    app.register_blueprint(bp)

    logger.info("Flask app đã sẵn sàng. Truy cập http://localhost:5000")
    return app


# Cho phép chạy trực tiếp: python src/web/app.py
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = create_app()
    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=config.FLASK_DEBUG,
    )
