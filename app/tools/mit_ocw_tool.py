"""
Phase A - Tool 3: MIT OpenCourseWare ingestion via public RSS feed metadata.
TOS-compliant: MIT OCW publishes public RSS feeds explicitly for syndication;
we only read titles/descriptions/links, never scrape gated content.
"""
from typing import List, Dict
import feedparser
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=6))
def _parse_feed(url: str):
    feed = feedparser.parse(url)
    if feed.bozo and not feed.entries:
        raise RuntimeError(f"Failed to parse MIT OCW feed: {feed.bozo_exception}")
    return feed


def fetch_mit_ocw_materials(topic: str, max_results: int = 5) -> List[Dict]:
    """
    Queries the MIT OCW public RSS feed and filters entries relevant to `topic`
    by simple keyword matching over title/summary (no scraping of gated pages).

    Returns a list of:
        {
            "course_title": str,
            "description": str,
            "url": str,
            "published": str,
            "matched_keywords": list[str]
        }
    """
    feed = _parse_feed(settings.MIT_OCW_RSS_SEARCH_URL)
    topic_terms = [t.lower() for t in topic.split() if len(t) > 2]

    matches = []
    for entry in feed.entries:
        title = getattr(entry, "title", "") or ""
        summary = getattr(entry, "summary", "") or ""
        haystack = f"{title} {summary}".lower()

        matched = [t for t in topic_terms if t in haystack]
        if matched:
            matches.append({
                "course_title": title,
                "description": summary[:600],
                "url": getattr(entry, "link", ""),
                "published": getattr(entry, "published", ""),
                "matched_keywords": matched,
            })

    matches.sort(key=lambda m: len(m["matched_keywords"]), reverse=True)
    return matches[:max_results]
