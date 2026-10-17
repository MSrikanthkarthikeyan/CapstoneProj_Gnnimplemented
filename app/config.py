"""
Centralized configuration for the Learning Path Builder.
All secrets/config are pulled from environment variables (see .env.example).
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    YOUTUBE_API_KEY: str = os.getenv("YOUTUBE_API_KEY", "")

    BACKEND_URL: str = os.getenv("BACKEND_URL", "http://localhost:8000")

    MAX_GRAPH_NODES: int = int(os.getenv("MAX_GRAPH_NODES", "25"))
    REQUEST_TIMEOUT_SECONDS: int = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "15"))

    WIKIPEDIA_API_URL: str = "https://en.wikipedia.org/w/api.php"
    MIT_OCW_RSS_SEARCH_URL: str = "https://ocw.mit.edu/rss/all/rss_ocw.xml"

    # Mastery / adaptive tutoring thresholds
    MASTERY_PASS_THRESHOLD: float = 0.7  # score >= this counts node as mastered
    QUIZ_QUESTIONS_PER_NODE: int = 3


settings = Settings()


def require_gemini_key():
    if not settings.GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and fill it in."
        )
