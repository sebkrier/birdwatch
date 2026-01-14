"""Tests for the Twitter archive importer.

These tests are written BEFORE the implementation (TDD).
"""

import pytest
from datetime import datetime, timezone


class TestJSWrapper:
    """Tests for stripping JavaScript wrappers from archive files."""

    def test_strips_tweets_wrapper(self):
        """Should strip window.YTD.tweets.part0 wrapper."""
        from app.services.importer import strip_js_wrapper

        js = 'window.YTD.tweets.part0 = [{"tweet": {}}]'
        assert strip_js_wrapper(js) == '[{"tweet": {}}]'

    def test_strips_various_wrappers(self):
        """Should handle different archive wrapper formats."""
        from app.services.importer import strip_js_wrapper

        # Different archive versions use different wrappers
        assert strip_js_wrapper('window.YTD.tweet.part0 = []') == '[]'
        assert strip_js_wrapper('window.YTD.tweets.part1 = []') == '[]'
        assert strip_js_wrapper('window.YTD.tweets.part99 = []') == '[]'

    def test_strips_grailbird_wrapper(self):
        """Should handle old Grailbird format."""
        from app.services.importer import strip_js_wrapper

        assert strip_js_wrapper('Grailbird.data.tweets_2024_01 = []') == '[]'
        assert strip_js_wrapper('Grailbird.data.tweets = []') == '[]'

    def test_handles_already_valid_json(self):
        """Should pass through valid JSON unchanged."""
        from app.services.importer import strip_js_wrapper

        assert strip_js_wrapper('[{"tweet": {}}]') == '[{"tweet": {}}]'
        assert strip_js_wrapper('[]') == '[]'

    def test_handles_whitespace(self):
        """Should handle leading/trailing whitespace."""
        from app.services.importer import strip_js_wrapper

        assert strip_js_wrapper('  window.YTD.tweets.part0 = []  ') == '[]'
        assert strip_js_wrapper('\nwindow.YTD.tweets.part0 = []\n') == '[]'

    def test_handles_bom(self):
        """Should handle UTF-8 BOM."""
        from app.services.importer import strip_js_wrapper

        # UTF-8 BOM is \ufeff
        js_with_bom = '\ufeffwindow.YTD.tweets.part0 = []'
        assert strip_js_wrapper(js_with_bom) == '[]'


class TestTweetParsing:
    """Tests for parsing tweet objects from archive."""

    def test_parses_basic_tweet(self, sample_tweet_raw):
        """Should parse a basic tweet with all fields."""
        from app.services.importer import parse_tweet

        tweet = parse_tweet(sample_tweet_raw)
        assert tweet.id == "1234567890123456789"
        assert tweet.full_text == "This is a test tweet #testing @someone https://t.co/abc123"
        assert tweet.retweet_count == 5
        assert tweet.favorite_count == 42
        assert tweet.is_retweet is False
        assert tweet.is_reply is False

    def test_parses_datetime(self, sample_tweet_raw):
        """Should correctly parse Twitter's date format."""
        from app.services.importer import parse_tweet

        tweet = parse_tweet(sample_tweet_raw)
        assert tweet.created_at.year == 2024
        assert tweet.created_at.month == 1
        assert tweet.created_at.day == 15
        assert tweet.created_at.hour == 10
        assert tweet.created_at.minute == 15

    def test_handles_missing_optional_fields(self):
        """Should handle tweets with minimal fields."""
        from app.services.importer import parse_tweet

        raw = {
            "tweet": {
                "id_str": "123",
                "created_at": "Mon Jan 15 10:15:00 +0000 2024",
                "full_text": "Hi",
            }
        }
        tweet = parse_tweet(raw)
        assert tweet.id == "123"
        assert tweet.full_text == "Hi"
        assert tweet.retweet_count == 0
        assert tweet.favorite_count == 0
        assert tweet.hashtags == []
        assert tweet.mentions == []
        assert tweet.urls == []

    def test_handles_string_counts(self):
        """Should convert string counts to integers."""
        from app.services.importer import parse_tweet

        raw = {
            "tweet": {
                "id_str": "123",
                "created_at": "Mon Jan 15 10:15:00 +0000 2024",
                "full_text": "Test",
                "retweet_count": "999",
                "favorite_count": "1234",
            }
        }
        tweet = parse_tweet(raw)
        assert tweet.retweet_count == 999
        assert tweet.favorite_count == 1234


class TestRetweetDetection:
    """Tests for detecting retweets."""

    def test_detects_rt_prefix(self):
        """Should detect RT prefix in text."""
        from app.services.importer import detect_retweet

        assert detect_retweet("RT @user: some tweet") is True
        assert detect_retweet("RT @user some tweet") is True

    def test_detects_retweeted_flag(self):
        """Should detect retweeted flag."""
        from app.services.importer import detect_retweet

        assert detect_retweet("some tweet", retweeted=True) is True

    def test_normal_tweet_not_retweet(self):
        """Should not flag normal tweets as retweets."""
        from app.services.importer import detect_retweet

        assert detect_retweet("Hello world") is False
        assert detect_retweet("Hello world", retweeted=False) is False

    def test_rt_in_middle_not_retweet(self):
        """Should not flag tweets with RT in middle."""
        from app.services.importer import detect_retweet

        assert detect_retweet("I saw the RT earlier") is False
        assert detect_retweet("Great RT from @user") is False


class TestReplyDetection:
    """Tests for detecting replies."""

    def test_detects_reply_by_field(self):
        """Should detect reply by in_reply_to field."""
        from app.services.importer import detect_reply

        assert detect_reply(in_reply_to="123") is True

    def test_detects_reply_by_mention_start(self):
        """Should detect reply by @ at start."""
        from app.services.importer import detect_reply

        assert detect_reply(full_text="@user hello") is True
        assert detect_reply(full_text="@user @other reply") is True

    def test_normal_tweet_not_reply(self):
        """Should not flag normal tweets as replies."""
        from app.services.importer import detect_reply

        assert detect_reply(full_text="Hello @user") is False  # Mention not at start
        assert detect_reply(full_text="Hello world") is False

    def test_empty_reply_to_not_reply(self):
        """Should handle empty/None in_reply_to."""
        from app.services.importer import detect_reply

        assert detect_reply(in_reply_to=None, full_text="Hello") is False
        assert detect_reply(in_reply_to="", full_text="Hello") is False


class TestEntityExtraction:
    """Tests for extracting entities from tweets."""

    def test_extracts_hashtags(self):
        """Should extract hashtags."""
        from app.services.importer import extract_entities

        entities = {
            "hashtags": [{"text": "ai"}, {"text": "ml"}],
            "urls": [],
            "user_mentions": [],
        }
        result = extract_entities(entities)
        assert result["hashtags"] == ["ai", "ml"]

    def test_extracts_expanded_urls(self):
        """Should extract expanded URLs, not t.co URLs."""
        from app.services.importer import extract_entities

        entities = {
            "hashtags": [],
            "urls": [
                {"url": "https://t.co/abc", "expanded_url": "https://example.com"},
                {"url": "https://t.co/xyz", "expanded_url": "https://other.com/page"},
            ],
            "user_mentions": [],
        }
        result = extract_entities(entities)
        assert result["urls"] == ["https://example.com", "https://other.com/page"]

    def test_extracts_mentions(self):
        """Should extract mentioned usernames."""
        from app.services.importer import extract_entities

        entities = {
            "hashtags": [],
            "urls": [],
            "user_mentions": [
                {"screen_name": "jack", "id_str": "123"},
                {"screen_name": "elonmusk", "id_str": "456"},
            ],
        }
        result = extract_entities(entities)
        assert result["mentions"] == ["jack", "elonmusk"]

    def test_handles_empty_entities(self):
        """Should handle empty entities."""
        from app.services.importer import extract_entities

        result = extract_entities({})
        assert result["hashtags"] == []
        assert result["urls"] == []
        assert result["mentions"] == []

    def test_handles_missing_entity_keys(self):
        """Should handle missing entity keys."""
        from app.services.importer import extract_entities

        result = extract_entities({"hashtags": [{"text": "test"}]})
        assert result["hashtags"] == ["test"]
        assert result["urls"] == []
        assert result["mentions"] == []


class TestMediaExtraction:
    """Tests for extracting media from tweets."""

    def test_extracts_photo_media(self):
        """Should extract photo media."""
        from app.services.importer import extract_media

        extended_entities = {
            "media": [
                {
                    "id_str": "12345",
                    "type": "photo",
                    "media_url_https": "https://pbs.twimg.com/media/xxx.jpg",
                    "sizes": {"large": {"w": 2048, "h": 1536}},
                }
            ]
        }
        media = extract_media(extended_entities, "tweet123")
        assert len(media) == 1
        assert media[0].id == "12345"
        assert media[0].tweet_id == "tweet123"
        assert media[0].media_type == "photo"
        assert media[0].width == 2048
        assert media[0].height == 1536

    def test_extracts_video_media(self):
        """Should extract video media with duration."""
        from app.services.importer import extract_media

        extended_entities = {
            "media": [
                {
                    "id_str": "67890",
                    "type": "video",
                    "media_url_https": "https://pbs.twimg.com/media/thumb.jpg",
                    "sizes": {"large": {"w": 1280, "h": 720}},
                    "video_info": {
                        "duration_millis": 30000,
                        "variants": [
                            {"url": "https://video.twimg.com/xxx.mp4", "bitrate": 832000}
                        ],
                    },
                }
            ]
        }
        media = extract_media(extended_entities, "tweet456")
        assert len(media) == 1
        assert media[0].media_type == "video"
        assert media[0].duration_ms == 30000

    def test_extracts_multiple_media(self):
        """Should extract multiple media items."""
        from app.services.importer import extract_media

        extended_entities = {
            "media": [
                {"id_str": "1", "type": "photo", "media_url_https": "https://a.jpg"},
                {"id_str": "2", "type": "photo", "media_url_https": "https://b.jpg"},
                {"id_str": "3", "type": "photo", "media_url_https": "https://c.jpg"},
            ]
        }
        media = extract_media(extended_entities, "tweet789")
        assert len(media) == 3

    def test_handles_no_media(self):
        """Should handle tweets without media."""
        from app.services.importer import extract_media

        media = extract_media({}, "tweet000")
        assert media == []

        media = extract_media(None, "tweet000")
        assert media == []


class TestJSONParsing:
    """Tests for JSON parsing from archive files."""

    def test_parses_tweets_from_js(self, sample_js_content):
        """Should parse tweets from JS file content."""
        from app.services.importer import parse_tweets_js

        tweets = list(parse_tweets_js(sample_js_content))
        assert len(tweets) == 2
        assert tweets[0].id == "123"
        assert tweets[0].full_text == "First tweet"
        assert tweets[1].id == "456"
        assert tweets[1].full_text == "Second tweet"

    def test_handles_empty_array(self):
        """Should handle empty tweet arrays."""
        from app.services.importer import parse_tweets_js

        tweets = list(parse_tweets_js("window.YTD.tweets.part0 = []"))
        assert tweets == []

    def test_handles_malformed_json_gracefully(self):
        """Should skip malformed entries without crashing."""
        from app.services.importer import parse_tweets_js

        # One good tweet, one bad
        js = '''window.YTD.tweets.part0 = [
            {"tweet": {"id_str": "good", "created_at": "Mon Jan 15 10:15:00 +0000 2024", "full_text": "OK"}},
            {"tweet": {"id_str": "bad"}},
            {"tweet": {"id_str": "good2", "created_at": "Mon Jan 15 10:15:00 +0000 2024", "full_text": "Also OK"}}
        ]'''
        tweets = list(parse_tweets_js(js))
        # Should get the valid tweets, skip invalid ones
        assert len(tweets) >= 1


class TestArchiveHashingAndDeduplication:
    """Tests for archive hashing and deduplication."""

    def test_computes_consistent_hash(self):
        """Should compute same hash for same content."""
        from app.services.importer import compute_archive_hash

        content = b"test archive content"
        hash1 = compute_archive_hash(content)
        hash2 = compute_archive_hash(content)
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex

    def test_different_content_different_hash(self):
        """Should compute different hash for different content."""
        from app.services.importer import compute_archive_hash

        hash1 = compute_archive_hash(b"content 1")
        hash2 = compute_archive_hash(b"content 2")
        assert hash1 != hash2


class TestDateTimeParsing:
    """Tests for Twitter's datetime format parsing."""

    def test_parses_standard_format(self):
        """Should parse standard Twitter datetime format."""
        from app.services.importer import parse_twitter_datetime

        dt = parse_twitter_datetime("Mon Jan 15 10:15:00 +0000 2024")
        assert dt.year == 2024
        assert dt.month == 1
        assert dt.day == 15
        assert dt.hour == 10
        assert dt.minute == 15
        assert dt.second == 0

    def test_parses_various_weekdays(self):
        """Should parse various weekday names."""
        from app.services.importer import parse_twitter_datetime

        # Test different weekdays and months
        dt = parse_twitter_datetime("Wed Dec 25 23:59:59 +0000 2023")
        assert dt.month == 12
        assert dt.day == 25

    def test_preserves_timezone(self):
        """Should preserve UTC timezone."""
        from app.services.importer import parse_twitter_datetime

        dt = parse_twitter_datetime("Mon Jan 15 10:15:00 +0000 2024")
        assert dt.tzinfo is not None


# Integration tests that require database
@pytest.mark.asyncio
class TestImporterIntegration:
    """Integration tests for the importer (require test database)."""

    async def test_import_creates_tweets(self, test_db, sample_js_content, tmp_path):
        """Should create tweets in database from JS content."""
        from app.services.importer import import_from_js_content
        from app.db.pool import get_cursor
        from app.db import queries

        await import_from_js_content(sample_js_content, "test.js")

        async with get_cursor() as cur:
            count = await queries.get_tweet_count(cur)
            assert count == 2

    async def test_reimport_is_idempotent(self, test_db, sample_js_content, tmp_path):
        """Should not create duplicates on re-import."""
        from app.services.importer import import_from_js_content
        from app.db.pool import get_cursor
        from app.db import queries

        # Import twice
        await import_from_js_content(sample_js_content, "test.js")
        await import_from_js_content(sample_js_content, "test.js")

        async with get_cursor() as cur:
            count = await queries.get_tweet_count(cur)
            assert count == 2  # Still only 2 tweets

    async def test_reimport_updates_counts(self, test_db):
        """Should update engagement counts on re-import."""
        from app.services.importer import import_from_js_content
        from app.db.pool import get_cursor
        from app.db import queries

        # Import with low counts
        js1 = '''window.YTD.tweets.part0 = [{
            "tweet": {
                "id_str": "same_id",
                "created_at": "Mon Jan 15 10:15:00 +0000 2024",
                "full_text": "Test tweet",
                "retweet_count": "5",
                "favorite_count": "10"
            }
        }]'''
        await import_from_js_content(js1, "test1.js")

        # Re-import with higher counts
        js2 = '''window.YTD.tweets.part0 = [{
            "tweet": {
                "id_str": "same_id",
                "created_at": "Mon Jan 15 10:15:00 +0000 2024",
                "full_text": "Test tweet",
                "retweet_count": "50",
                "favorite_count": "100"
            }
        }]'''
        await import_from_js_content(js2, "test2.js")

        async with get_cursor() as cur:
            tweet = await queries.get_tweet_by_id(cur, "same_id")
            assert tweet["retweet_count"] == 50
            assert tweet["favorite_count"] == 100
