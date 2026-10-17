"""
Phase A - Tool 1: Wikipedia ingestion via the official MediaWiki API.
TOS-compliant: uses public REST endpoints, no scraping, includes a descriptive User-Agent.
"""
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings

USER_AGENT = "LearningPathBuilder/1.0 (educational MVP; contact: example@example.com)"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=6))
def _get(params: dict) -> dict:
    resp = requests.get(
        settings.WIKIPEDIA_API_URL,
        params=params,
        headers={"User-Agent": USER_AGENT},
        timeout=settings.REQUEST_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_wikipedia_summary(topic: str) -> dict:
    """
    Queries the MediaWiki API for an introductory summary, key section titles,
    and outgoing wikilinks for a given topic.

    Returns:
        {
            "topic": str,
            "title": str,
            "summary": str,
            "sections": list[str],
            "links": list[str],
            "source_url": str,
            "found": bool
        }
    """
    search_data = _get({
        "action": "query",
        "list": "search",
        "srsearch": topic,
        "srlimit": 1,
        "format": "json",
    })

    hits = search_data.get("query", {}).get("search", [])
    if not hits:
        return {
            "topic": topic, "title": None, "summary": "", "sections": [],
            "links": [], "source_url": None, "found": False,
        }

    page_title = hits[0]["title"]

    extract_data = _get({
        "action": "query",
        "prop": "extracts|links|info",
        "exintro": True,
        "explaintext": True,
        "titles": page_title,
        "pllimit": "max",
        "inprop": "url",
        "format": "json",
    })

    pages = extract_data.get("query", {}).get("pages", {})
    page = next(iter(pages.values()), {})

    summary = page.get("extract", "") or ""
    links = [l["title"] for l in page.get("links", []) if "title" in l][:40]
    source_url = page.get("fullurl", f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}")

    sections_data = _get({
        "action": "parse",
        "page": page_title,
        "prop": "sections",
        "format": "json",
    })
    sections = [s["line"] for s in sections_data.get("parse", {}).get("sections", [])][:20]

    return {
        "topic": topic,
        "title": page_title,
        "summary": summary,
        "sections": sections,
        "links": links,
        "source_url": source_url,
        "found": True,
    }
