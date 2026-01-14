"""Tweet detail routes."""

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

from app.db.pool import get_cursor
from app.db import queries

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))


@router.get("/{tweet_id}", response_class=HTMLResponse)
async def tweet_detail(request: Request, tweet_id: str):
    """View a single tweet with context and similar tweets."""
    async with get_cursor() as cur:
        tweet = await queries.get_tweet_by_id(cur, tweet_id)
        if not tweet:
            raise HTTPException(status_code=404, detail="Tweet not found")

        # Get media
        media = await queries.get_tweet_media(cur, tweet_id)

        # Get thread context if it's a reply or part of a thread
        thread = []
        if tweet.get("conversation_id") or tweet.get("reply_to_tweet_id"):
            thread = await queries.get_thread_context(cur, tweet_id)

        # Get similar tweets
        similar = await queries.get_similar_tweets(cur, tweet_id, limit=5)

    return templates.TemplateResponse(
        "tweet.html",
        {
            "request": request,
            "tweet": tweet,
            "media": media,
            "thread": thread,
            "similar": similar,
        },
    )
