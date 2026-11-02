"""
TubePulse - YouTube Data API v3 Ingestion Module
Handles video metadata retrieval and paginated comment ingestion.
Uses Python's standard library urllib for zero-dependency live API calls,
ensuring it works out of the box without requiring external client packages.
"""

import os
import re
import json
from urllib.parse import urlparse, parse_qs, quote
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import pandas as pd

def _load_env_file():
    """Automatically loads .env file into os.environ if present."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(base_dir, ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, val = line.split("=", 1)
                        key = key.strip()
                        val = val.strip().strip('"').strip("'").strip()
                        if key:
                            os.environ[key] = val
        except Exception:
            pass

_load_env_file()

def get_api_key() -> str:
    """Returns the YouTube API key from environment, ensuring .env is re-checked."""
    _load_env_file()
    return os.getenv("YOUTUBE_API_KEY", "").strip()

def extract_video_id(url_or_id: str) -> str:
    """Extract YouTube video ID from any URL format or raw ID."""
    clean = url_or_id.strip()
    if not clean:
        return "dQw4w9WgXcQ"
    
    if re.match(r'^[0-9A-Za-z_-]{11}$', clean):
        return clean

    try:
        parsed = urlparse(clean)
        if "youtube.com" in parsed.netloc:
            qs = parse_qs(parsed.query)
            if "v" in qs and qs["v"]:
                return qs["v"][0][:11]
            parts = [p for p in parsed.path.split("/") if p]
            if len(parts) >= 2 and parts[0] in ["shorts", "embed", "v"]:
                return parts[1][:11]

        if "youtu.be" in parsed.netloc:
            parts = [p for p in parsed.path.split("/") if p]
            if parts:
                return parts[0][:11]
    except Exception:
        pass

    v_param = re.search(r'[?&]v=([0-9A-Za-z_-]{11})', clean)
    if v_param:
        return v_param.group(1)

    token = re.search(r'([0-9A-Za-z_-]{11})', clean)
    if token:
        return token.group(1)

    return clean[:11]

def get_fallback_data():
    """Load local pre-staged CSV files for reliable fallback if offline or no key."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sample_dir = os.path.join(base_dir, "sample_data")
    filepath = os.path.join(sample_dir, "tutorial_loved.csv")
    
    meta = {
        "title": "PySpark Masterclass: Distributed Data Processing from Scratch",
        "channel": "Data Engineering Academy",
        "views": 1520000,
        "likes": 135200,
        "comment_count": 8920
    }
    
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
    else:
        df = pd.DataFrame([
            {"author": "User1", "text": "Clear explanation of distributed cluster concepts.", "like_count": 420, "published_at": "2026-10-01"},
            {"author": "User2", "text": "Disappointed by the unexpected price increase and removed features.", "like_count": 310, "published_at": "2026-10-02"},
            {"author": "BotUser", "text": "Message me on WhatsApp +18005550199 for guaranteed crypto returns", "like_count": 2, "published_at": "2026-10-03"}
        ])
    return meta, df

def _http_get_json(url: str):
    """Executes an HTTPS GET request using Python standard library urllib."""
    req = Request(url, headers={"User-Agent": "TubePulse/1.0 (Windows NT 10.0)"})
    with urlopen(req, timeout=12) as response:
        return json.loads(response.read().decode("utf-8"))

def fetch_youtube_comments(video_id: str, max_comments: int = 500, api_key: str = None, on_progress=None, **kwargs):
    """
    Fetches real-time comments from YouTube Data API v3 using standard library urllib.
    Zero external client library required.
    Returns: (video_meta, comments_df, error_message)
    """
    api_key = (api_key or get_api_key()).strip()
    
    if not api_key:
        if on_progress:
            on_progress(30, "No API key detected in .env. Serving fallback benchmark...")
        meta, df = get_fallback_data()
        return meta, df, "No API key found in .env. Showing benchmark dataset."
            
    try:
        if on_progress:
            on_progress(10, f"Querying YouTube Data API v3 for ID: {video_id}...")

        # 1. Fetch Video Metadata via REST endpoint
        meta_url = (
            f"https://www.googleapis.com/youtube/v3/videos"
            f"?part=snippet,statistics"
            f"&id={video_id}"
            f"&key={api_key}"
        )
        
        try:
            meta_json = _http_get_json(meta_url)
        except HTTPError as e:
            err_body = e.read().decode('utf-8', errors='ignore')
            if "quotaExceeded" in err_body:
                err_msg = "Google YouTube Data API daily quota limit exceeded."
            elif "API_KEY_INVALID" in err_body or "keyInvalid" in err_body:
                err_msg = "YouTube API key is invalid. Please check your .env file."
            elif "accessNotConfigured" in err_body:
                err_msg = "YouTube Data API v3 is not enabled in your Google Cloud project."
            else:
                err_msg = f"YouTube API Error ({e.code}): {e.reason}"
            meta, df = get_fallback_data()
            return meta, df, err_msg

        items = meta_json.get("items", [])
        if not items:
            err_msg = f"No video found on YouTube with ID '{video_id}'."
            meta, df = get_fallback_data()
            return meta, df, err_msg

        video_item = items[0]
        meta = {
            "title": video_item["snippet"]["title"],
            "channel": video_item["snippet"]["channelTitle"],
            "views": int(video_item["statistics"].get("viewCount", 0)),
            "likes": int(video_item["statistics"].get("likeCount", 0)),
            "comment_count": int(video_item["statistics"].get("commentCount", 0))
        }

        if on_progress:
            on_progress(25, f"Retrieved: {meta['title'][:40]}... Ingesting comment stream...")

        # 2. Fetch Comments (Paginated)
        comments = []
        next_page_token = None
        error_notice = None

        while len(comments) < max_comments:
            batch_size = min(100, max_comments - len(comments))
            cmnt_url = (
                f"https://www.googleapis.com/youtube/v3/commentThreads"
                f"?part=snippet"
                f"&videoId={video_id}"
                f"&maxResults={batch_size}"
                f"&textFormat=plainText"
                f"&key={api_key}"
            )
            if next_page_token:
                cmnt_url += f"&pageToken={quote(next_page_token)}"

            try:
                cmnt_json = _http_get_json(cmnt_url)
            except HTTPError as he:
                err_body = he.read().decode('utf-8', errors='ignore')
                if "commentsDisabled" in err_body:
                    error_notice = "Comments are disabled by the creator on this video (common on YouTube Music tracks)."
                elif "quotaExceeded" in err_body:
                    error_notice = "Daily quota reached on YouTube Data API."
                else:
                    error_notice = f"Comment Fetch Notice ({he.code}): {he.reason}"
                break
            except Exception as ex:
                error_notice = f"Comment Ingestion Notice: {str(ex)[:40]}"
                break

            cmnt_items = cmnt_json.get("items", [])
            if not cmnt_items:
                break

            for item in cmnt_items:
                snippet = item["snippet"]["topLevelComment"]["snippet"]
                comments.append({
                    "author": snippet.get("authorDisplayName", "Anonymous"),
                    "text": snippet.get("textDisplay", ""),
                    "like_count": int(snippet.get("likeCount", 0)),
                    "published_at": snippet.get("publishedAt", "")
                })

            if on_progress:
                pct = 25 + int((len(comments) / max_comments) * 28)
                on_progress(min(52, pct), f"Harvested {len(comments)} comments from YouTube...")

            next_page_token = cmnt_json.get("nextPageToken")
            if not next_page_token:
                break

        # If zero comments were fetched
        if not comments:
            if not error_notice:
                error_notice = "This video has zero public comments."
            df = pd.DataFrame([{
                "author": meta["channel"],
                "text": f"Video {meta['title']} currently has no public comments.",
                "like_count": meta["likes"],
                "published_at": ""
            }])
            return meta, df, error_notice

        df = pd.DataFrame(comments)
        return meta, df, error_notice

    except Exception as e:
        error_msg = f"API Connection Error: {str(e)}"
        meta, df = get_fallback_data()
        return meta, df, error_msg
