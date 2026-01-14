"""Pytest configuration and fixtures."""

import asyncio
import os
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

# Set test environment (port 5434 to avoid conflict with system PostgreSQL)
os.environ["DATABASE_URL"] = "postgresql://localhost:5434/birdwatch_test"
os.environ["DEBUG"] = "true"


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def test_db():
    """Create and clean up test database."""
    from app.db.pool import init_pool, close_pool, get_connection

    await init_pool()

    # Clean tables before each test
    async with get_connection() as conn:
        await conn.execute("TRUNCATE tweets, media, clusters, tweet_coordinates, import_history CASCADE")

    yield

    await close_pool()


@pytest_asyncio.fixture
async def client(test_db) -> AsyncGenerator[AsyncClient, None]:
    """Create test client."""
    from app.main import app

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as ac:
        yield ac


@pytest.fixture
def sample_tweet_raw():
    """Sample raw tweet data from Twitter archive."""
    return {
        "tweet": {
            "id_str": "1234567890123456789",
            "created_at": "Mon Jan 15 10:15:00 +0000 2024",
            "full_text": "This is a test tweet #testing @someone https://t.co/abc123",
            "retweet_count": "5",
            "favorite_count": "42",
            "in_reply_to_status_id_str": None,
            "in_reply_to_user_id_str": None,
            "in_reply_to_screen_name": None,
            "retweeted": False,
            "entities": {
                "hashtags": [{"text": "testing", "indices": [23, 31]}],
                "urls": [
                    {
                        "url": "https://t.co/abc123",
                        "expanded_url": "https://example.com/page",
                        "display_url": "example.com/page",
                    }
                ],
                "user_mentions": [{"screen_name": "someone", "id_str": "111222333"}],
            },
        }
    }


@pytest.fixture
def sample_retweet_raw():
    """Sample retweet data."""
    return {
        "tweet": {
            "id_str": "9876543210987654321",
            "created_at": "Tue Jan 16 11:30:00 +0000 2024",
            "full_text": "RT @original: This is the original tweet content",
            "retweet_count": "100",
            "favorite_count": "0",
            "retweeted": True,
            "entities": {"hashtags": [], "urls": [], "user_mentions": []},
        }
    }


@pytest.fixture
def sample_reply_raw():
    """Sample reply data."""
    return {
        "tweet": {
            "id_str": "5555555555555555555",
            "created_at": "Wed Jan 17 09:00:00 +0000 2024",
            "full_text": "@user Yes, I agree with that point!",
            "retweet_count": "0",
            "favorite_count": "3",
            "in_reply_to_status_id_str": "4444444444444444444",
            "in_reply_to_user_id_str": "333333333",
            "in_reply_to_screen_name": "user",
            "retweeted": False,
            "entities": {
                "hashtags": [],
                "urls": [],
                "user_mentions": [{"screen_name": "user", "id_str": "333333333"}],
            },
        }
    }


@pytest.fixture
def sample_js_content():
    """Sample JS file content from Twitter archive."""
    return '''window.YTD.tweets.part0 = [
  {
    "tweet": {
      "id_str": "123",
      "created_at": "Mon Jan 15 10:15:00 +0000 2024",
      "full_text": "First tweet",
      "retweet_count": "1",
      "favorite_count": "2",
      "entities": {"hashtags": [], "urls": [], "user_mentions": []}
    }
  },
  {
    "tweet": {
      "id_str": "456",
      "created_at": "Tue Jan 16 11:00:00 +0000 2024",
      "full_text": "Second tweet",
      "retweet_count": "3",
      "favorite_count": "4",
      "entities": {"hashtags": [], "urls": [], "user_mentions": []}
    }
  }
]'''
