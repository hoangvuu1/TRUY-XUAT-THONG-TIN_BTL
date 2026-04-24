"""
run_pipeline.py – Chạy toàn bộ pipeline từ đầu đến cuối.

Bước:
  1. Preprocess
  2. Build index (Inverted Index + BM25 + TF-IDF)
  3. Khởi động Flask web server

Chạy:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --no-web    # Chỉ preprocess + build index
    python scripts/run_pipeline.py --max-docs 5000
"""
import argparse
import logging
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_step(script: str, extra_args: list[str] | None = None) -> None:
    """Chạy một script Python và kiểm tra lỗi."""
    cmd = [sys.executable, str(ROOT / "scripts" / script)] + (extra_args or [])
    logger.info("▶ Chạy: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        logger.error("Script %s thất bại (code %d). Dừng pipeline.", script, result.returncode)
        sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Chạy toàn bộ pipeline")
    parser.add_argument("--max-docs", type=int, default=config.MAX_DOCS,
                        help=f"Số tài liệu tối đa (mặc định: {config.MAX_DOCS})")
    parser.add_argument("--no-web", action="store_true",
                        help="Bỏ qua bước khởi động web server")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  🔍 Vietnamese Document Retrieval – Pipeline")
    print("="*60 + "\n")

    # Bước 1: Preprocess
    logger.info("📄 Bước 1/3: Tiền xử lý dữ liệu ...")
    run_step("preprocess.py", ["--max-docs", str(args.max_docs)])

    # Bước 2: Build index
    logger.info("🗄️  Bước 2/3: Xây dựng index ...")
    run_step("build_index.py")

    if args.no_web:
        logger.info("✅ Pipeline hoàn tất (không khởi động web).")
        return

    # Bước 3: Khởi động Flask
    logger.info("🌐 Bước 3/3: Khởi động web server ...")
    logger.info("   Truy cập: http://localhost:%d", config.FLASK_PORT)
    web_cmd = [
        sys.executable, "-m", "flask",
        "--app", "src/web/app.py",
        "run",
        "--host", config.FLASK_HOST,
        "--port", str(config.FLASK_PORT),
    ]
    if config.FLASK_DEBUG:
        web_cmd.append("--debug")
    subprocess.run(web_cmd, cwd=str(ROOT), check=False)


if __name__ == "__main__":
    main()
