"""
Phase A - Tool 2: YouTube ingestion via the official YouTube Data API v3
(googleapis client) plus youtube-transcript-api for closed-caption snippets.
TOS-compliant: uses the official Data API for search/metadata (requires an API key),
and the public timedtext/caption-track access pattern for transcripts (no video download).
"""
from typing import List, Dict
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings

try:
    from googleapiclient.discovery import build
except ImportError:  # pragma: no cover
    build = None

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
except ImportError:  # pragma: no cover
    YouTubeTranscriptApi = None
    TranscriptsDisabled = NoTranscriptFound = Exception


def _get_youtube_client():
    if build is None:
        raise RuntimeError("google-api-python-client is not installed.")
    if not settings.YOUTUBE_API_KEY:
        raise RuntimeError("YOUTUBE_API_KEY is not set. See .env.example.")
    return build("youtube", "v3", developerKey=settings.YOUTUBE_API_KEY)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=6))
def _search_videos(query: str, max_results: int) -> list:
    youtube = _get_youtube_client()
    request = youtube.search().list(
        q=f"{query} tutorial",
        part="snippet",
        type="video",
        maxResults=max_results,
        order="relevance",
        relevanceLanguage="en",
        safeSearch="strict",
        videoEmbeddable="true",
    )
    response = request.execute()
    return response.get("items", [])


def _fetch_video_stats(youtube, video_ids: List[str]) -> Dict[str, dict]:
    if not video_ids:
        return {}
    request = youtube.videos().list(part="statistics,contentDetails", id=",".join(video_ids))
    response = request.execute()
    return {item["id"]: item for item in response.get("items", [])}


def _fetch_transcript_snippet(video_id: str, max_chars: int = 800) -> str:
    if YouTubeTranscriptApi is None:
        return ""
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["en", "en-US"])
        text = " ".join(seg["text"] for seg in transcript)
        return text[:max_chars]
    except (TranscriptsDisabled, NoTranscriptFound, Exception):
        return ""


def fetch_youtube_tutorials(query: str, max_results: int = 3) -> List[Dict]:
    """
    Uses the YouTube Data API v3 to find educational videos for `query`,
    then pulls a short transcript snippet (closed captions) for each via
    youtube-transcript-api.

    Returns a list of:
        {
            "video_id": str,
            "title": str,
            "channel": str,
            "description": str,
            "url": str,
            "thumbnail": str,
            "view_count": int | None,
            "transcript_snippet": str
        }
    """
    items = _search_videos(query, max_results)
    if not items:
        return []

    youtube = _get_youtube_client()
    video_ids = [it["id"]["videoId"] for it in items if "videoId" in it.get("id", {})]
    stats_map = _fetch_video_stats(youtube, video_ids)

    results = []
    for it in items:
        vid = it.get("id", {}).get("videoId")
        if not vid:
            continue
        snippet = it.get("snippet", {})
        stats = stats_map.get(vid, {}).get("statistics", {})
        results.append({
            "video_id": vid,
            "title": snippet.get("title", ""),
            "channel": snippet.get("channelTitle", ""),
            "description": snippet.get("description", "")[:500],
            "url": f"https://www.youtube.com/watch?v={vid}",
            "thumbnail": snippet.get("thumbnails", {}).get("high", {}).get("url", ""),
            "view_count": int(stats.get("viewCount", 0)) if stats.get("viewCount") else None,
            "transcript_snippet": _fetch_transcript_snippet(vid),
        })

    results.sort(key=lambda r: r["view_count"] or 0, reverse=True)
    return results
