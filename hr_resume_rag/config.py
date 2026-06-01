from __future__ import annotations

import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT_DIR / ".env", override=True)
except Exception:
    pass

DATA_DIR = ROOT_DIR / "data"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
ALLOWED_DOMAINS = ("BANKING", "INFORMATION-TECHNOLOGY")
LOW_TEXT_THRESHOLD = 800
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def ensure_artifacts_dir(path: Path = ARTIFACTS_DIR) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def llm_config() -> dict[str, str | None]:
    return {
        "api_key": os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY"),
        "base_url": os.getenv("LLM_BASE_URL", "https://api.deepseek.com"),
        "model": os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro"),
    }
