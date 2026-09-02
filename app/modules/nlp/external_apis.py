"""
External enrichment APIs used by the Study Material pipeline. Both degrade
gracefully to an empty list/None if the corresponding key isn't set in
.env, rather than failing the whole pipeline — study material is still
useful without videos/articles attached.
"""
import logging
import requests
from flask import current_app

logger = logging.getLogger("eduai")


def get_youtube_videos(query: str, max_results: int = 3) -> list[dict]:
    api_key = current_app.config.get("YOUTUBE_API_KEY")
    if not api_key:
        return []
    try:
        resp = requests.get(
            "https://www.googleapis.com/youtube/v3/search",
            params={
                "part": "snippet",
                "q": query,
                "type": "video",
                "maxResults": max_results,
                "key": api_key,
                "relevanceLanguage": "en",
                "safeSearch": "strict",
            },
            timeout=8,
        )
        resp.raise_for_status()
        items = resp.json().get("items", [])
        return [
            {
                "video_id": it["id"]["videoId"],
                "title": it["snippet"]["title"],
                "thumbnail": it["snippet"]["thumbnails"]["medium"]["url"],
                "channel": it["snippet"]["channelTitle"],
                "url": f"https://www.youtube.com/watch?v={it['id']['videoId']}",
            }
            for it in items
        ]
    except requests.RequestException as exc:
        logger.warning("YouTube API request failed for query '%s': %s", query, exc)
        return []


def get_google_article(query: str) -> dict | None:
    api_key = current_app.config.get("GOOGLE_API_KEY")
    cx = current_app.config.get("GOOGLE_SEARCH_ENGINE_ID")
    if not api_key or not cx:
        return None
    try:
        resp = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params={"key": api_key, "cx": cx, "q": query, "num": 1},
            timeout=8,
        )
        resp.raise_for_status()
        items = resp.json().get("items", [])
        if not items:
            return None
        item = items[0]
        return {
            "title": item.get("title"),
            "snippet": item.get("snippet"),
            "url": item.get("link"),
        }
    except requests.RequestException as exc:
        logger.warning("Google Custom Search request failed for query '%s': %s", query, exc)
        return None
