import os
import json
import time
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

LABEL_MAP = {
    "intern": 0,
    "junior": 1,
    "senior": 2,
    "lead": 3,
    "template": 4,
    "low_value": 5,
}
LABEL_NAMES = {v: k for k, v in LABEL_MAP.items()}
CATEGORIES = list(LABEL_MAP.keys())

CATEGORY_DESCRIPTIONS = {
    "intern": "Simple project, typically a first/second programming exercise. Minimal structure, no tests, basic README.",
    "junior": "Shows some engineering practices but limited experience. Basic tests, simple architecture, limited documentation.",
    "senior": "Well-structured, follows best practices, good test coverage, CI/CD, proper documentation, clean architecture.",
    "lead": "Complex system design, advanced patterns, active community, significant releases, multiple contributors.",
    "template": "Repository that is a template, boilerplate, starter kit, or near-identical copy of another project.",
    "low_value": "Essentially empty or abandoned — no meaningful code, just configuration files, or clearly inactive.",
}


def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data, path: str):
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_env(key: str, required: bool = True) -> str:
    val = os.getenv(key)
    if required and not val:
        raise EnvironmentError(f"Missing required environment variable: {key}")
    return val


def rate_limited_request(func, *args, max_retries=5, **kwargs):
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if "rate limit" in str(e).lower() or "429" in str(e):
                wait = 2 ** attempt * 5
                logger.warning(f"Rate limited. Waiting {wait}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Max retries exceeded for {func.__name__}")
