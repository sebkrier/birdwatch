"""Twitter archive importer service.

Handles parsing and importing Twitter archive files into the database.
Designed to be idempotent - re-running the same archive won't create duplicates.
"""

import hashlib
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator, Optional

import orjson
from pydantic import BaseModel, Field

from app.db.pool import get_connection
from app.db import queries


class ParsedTweet(BaseModel):
    """Parsed tweet ready for database insertion."""

    id: str
    created_at: datetime
    full_text: str
    is_retweet: bool = False
    is_reply: bool = False
    is_thread: bool = False
    reply_to_tweet_id: Optional[str] = None
    reply_to_user_id: Optional[str] = None
    reply_to_username: Optional[str] = None
    conversation_id: Optional[str] = None
    retweet_count: int = 0
    favorite_count: int = 0
    hashtags: list[str] = Field(default_factory=list)
    mentions: list[str] = Field(default_factory=list)
    urls: list[str] = Field(default_factory=list)


class ParsedMedia(BaseModel):
    """Parsed media ready for database insertion."""

    id: str
    tweet_id: str
    media_type: str
    media_url: Optional[str] = None
    local_path: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    duration_ms: Optional[int] = None
    alt_text: Optional[str] = None


# Regex patterns for JS wrappers
JS_WRAPPER_PATTERNS = [
    r"^[\s\ufeff]*window\.YTD\.[a-z_]+\.part\d+\s*=\s*",  # window.YTD.tweets.part0 =, window.YTD.deleted_tweets.part0 =
    r"^[\s\ufeff]*Grailbird\.data\.tweets[_\w]*\s*=\s*",  # Grailbird.data.tweets_2024_01 =
]


def strip_js_wrapper(content: str) -> str:
    """Strip JavaScript wrapper from archive file content.

    Twitter archives wrap JSON data in JavaScript variable assignments:
    - window.YTD.tweets.part0 = [...]
    - Grailbird.data.tweets_2024_01 = [...]

    Returns the raw JSON array.
    """
    for pattern in JS_WRAPPER_PATTERNS:
        content = re.sub(pattern, "", content, count=1)

    return content.strip()


def parse_twitter_datetime(date_str: str) -> datetime:
    """Parse Twitter's datetime format.

    Format: "Mon Jan 15 10:15:00 +0000 2024"
    """
    # Twitter uses this format: %a %b %d %H:%M:%S %z %Y
    return datetime.strptime(date_str, "%a %b %d %H:%M:%S %z %Y")


def detect_retweet(full_text: str, *, retweeted: bool = False) -> bool:
    """Detect if a tweet is a retweet.

    Retweets are detected by:
    - The 'retweeted' flag being True
    - Text starting with 'RT @'
    """
    if retweeted:
        return True

    return full_text.startswith("RT @")


def detect_reply(
    *,
    full_text: str = "",
    in_reply_to: Optional[str] = None,
) -> bool:
    """Detect if a tweet is a reply.

    Replies are detected by:
    - Having an in_reply_to_status_id
    - Text starting with @mention
    """
    if in_reply_to:
        return True

    return full_text.startswith("@")


def extract_entities(entities: dict) -> dict:
    """Extract hashtags, URLs, and mentions from entities."""
    return {
        "hashtags": [h.get("text", "") for h in entities.get("hashtags", [])],
        "urls": [u.get("expanded_url", u.get("url", "")) for u in entities.get("urls", [])],
        "mentions": [m.get("screen_name", "") for m in entities.get("user_mentions", [])],
    }


def extract_media(extended_entities: Optional[dict], tweet_id: str) -> list[ParsedMedia]:
    """Extract media items from extended_entities."""
    if not extended_entities:
        return []

    media_items = extended_entities.get("media", [])
    result = []

    for item in media_items:
        width = None
        height = None
        if "sizes" in item and "large" in item["sizes"]:
            width = item["sizes"]["large"].get("w")
            height = item["sizes"]["large"].get("h")

        duration_ms = None
        if "video_info" in item:
            duration_ms = item["video_info"].get("duration_millis")

        result.append(
            ParsedMedia(
                id=item.get("id_str", ""),
                tweet_id=tweet_id,
                media_type=item.get("type", "photo"),
                media_url=item.get("media_url_https"),
                width=width,
                height=height,
                duration_ms=duration_ms,
                alt_text=item.get("ext_alt_text"),
            )
        )

    return result


def parse_tweet(raw: dict) -> ParsedTweet:
    """Parse a raw tweet object from the archive.

    Raw format: {"tweet": {...tweet data...}}
    """
    tweet_data = raw.get("tweet", raw)

    # Parse entities
    entities = tweet_data.get("entities", {})
    extracted = extract_entities(entities)

    # Parse counts (they come as strings)
    retweet_count = int(tweet_data.get("retweet_count", 0) or 0)
    favorite_count = int(tweet_data.get("favorite_count", 0) or 0)

    full_text = tweet_data.get("full_text", "")
    in_reply_to = tweet_data.get("in_reply_to_status_id_str")

    return ParsedTweet(
        id=tweet_data["id_str"],
        created_at=parse_twitter_datetime(tweet_data["created_at"]),
        full_text=full_text,
        is_retweet=detect_retweet(
            full_text,
            retweeted=tweet_data.get("retweeted", False),
        ),
        is_reply=detect_reply(
            full_text=full_text,
            in_reply_to=in_reply_to,
        ),
        reply_to_tweet_id=in_reply_to,
        reply_to_user_id=tweet_data.get("in_reply_to_user_id_str"),
        reply_to_username=tweet_data.get("in_reply_to_screen_name"),
        retweet_count=retweet_count,
        favorite_count=favorite_count,
        hashtags=extracted["hashtags"],
        mentions=extracted["mentions"],
        urls=extracted["urls"],
    )


def parse_tweets_js(content: str) -> list[ParsedTweet]:
    """Parse tweets from JavaScript file content.

    Yields parsed tweets, skipping any that fail to parse.
    """
    json_content = strip_js_wrapper(content)

    try:
        data = orjson.loads(json_content)
    except orjson.JSONDecodeError:
        return []

    tweets = []
    for item in data:
        try:
            tweets.append(parse_tweet(item))
        except (KeyError, ValueError):
            # Skip malformed entries
            continue

    return tweets


def compute_archive_hash(content: bytes) -> str:
    """Compute SHA256 hash of archive content."""
    return hashlib.sha256(content).hexdigest()


async def import_from_js_content(
    content: str,
    source_filename: str,
) -> dict:
    """Import tweets from JS file content into the database.

    Returns import statistics.
    """
    tweets = parse_tweets_js(content)

    stats = {
        "imported": 0,
        "updated": 0,
        "skipped": 0,
    }

    async with get_connection() as conn:
        async with conn.cursor() as cur:
            for tweet in tweets:
                try:
                    result = await queries.insert_tweet(
                        cur,
                        id=tweet.id,
                        created_at=tweet.created_at,
                        full_text=tweet.full_text,
                        is_retweet=tweet.is_retweet,
                        is_reply=tweet.is_reply,
                        is_thread=tweet.is_thread,
                        reply_to_tweet_id=tweet.reply_to_tweet_id,
                        reply_to_user_id=tweet.reply_to_user_id,
                        reply_to_username=tweet.reply_to_username,
                        conversation_id=tweet.conversation_id,
                        retweet_count=tweet.retweet_count,
                        favorite_count=tweet.favorite_count,
                        hashtags=tweet.hashtags,
                        mentions=tweet.mentions,
                        urls=tweet.urls,
                        archive_source=source_filename,
                    )
                    # Check if it was an insert or update based on xmax
                    stats["imported"] += 1
                except Exception:
                    stats["skipped"] += 1

            await conn.commit()

    return stats


async def import_archive(
    archive_path: Path,
    original_filename: str,
) -> AsyncGenerator[dict, None]:
    """Import a Twitter archive ZIP file.

    Yields progress updates for SSE streaming.
    """
    # Compute hash for deduplication
    archive_content = archive_path.read_bytes()
    archive_hash = compute_archive_hash(archive_content)

    # Check if already imported
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            existing = await queries.get_import_by_hash(cur, archive_hash)
            if existing:
                yield {
                    "status": "skipped",
                    "message": "Archive already imported",
                    "previous_import": existing["id"],
                }
                return

            # Create import record
            import_record = await queries.create_import_record(
                cur,
                archive_filename=original_filename,
                archive_hash=archive_hash,
            )
            import_id = import_record["id"]
            await conn.commit()

    total_tweets = 0
    total_media = 0
    processed_tweets = 0
    processed_media = 0

    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            # Find all tweet JS files
            tweet_files = [
                name
                for name in zf.namelist()
                if name.endswith(".js")
                and ("tweets" in name.lower() or "tweet" in name.lower())
                and "data/" in name
            ]

            # First pass: count total
            for filename in tweet_files:
                content = zf.read(filename).decode("utf-8")
                tweets = parse_tweets_js(content)
                total_tweets += len(tweets)

            yield {
                "status": "in_progress",
                "tweets_total": total_tweets,
                "tweets_processed": 0,
                "media_processed": 0,
                "current_file": "Starting...",
            }

            # Second pass: import
            for filename in tweet_files:
                yield {
                    "status": "in_progress",
                    "tweets_total": total_tweets,
                    "tweets_processed": processed_tweets,
                    "media_processed": processed_media,
                    "current_file": filename,
                }

                content = zf.read(filename).decode("utf-8")
                stats = await import_from_js_content(content, filename)
                processed_tweets += stats["imported"] + stats["updated"]

                yield {
                    "status": "in_progress",
                    "tweets_total": total_tweets,
                    "tweets_processed": processed_tweets,
                    "media_processed": processed_media,
                    "current_file": filename,
                }

        # Update import record
        async with get_connection() as conn:
            async with conn.cursor() as cur:
                await queries.update_import_record(
                    cur,
                    import_id,
                    tweets_imported=processed_tweets,
                    media_imported=processed_media,
                    status="completed",
                )
                await conn.commit()

        yield {
            "status": "completed",
            "tweets_total": total_tweets,
            "tweets_processed": processed_tweets,
            "media_processed": processed_media,
        }

    except Exception as e:
        # Update import record with failure
        async with get_connection() as conn:
            async with conn.cursor() as cur:
                await queries.update_import_record(
                    cur,
                    import_id,
                    status="failed",
                )
                await conn.commit()

        yield {
            "status": "failed",
            "error": str(e),
            "tweets_processed": processed_tweets,
        }
        raise
