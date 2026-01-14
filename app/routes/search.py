"""Search routes for full-text and semantic search."""

from typing import Optional
from datetime import datetime

from fastapi import APIRouter, Request, Query
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

from app.db.pool import get_cursor
from app.db import queries

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))


@router.get("/search", response_class=HTMLResponse)
async def search(
    request: Request,
    q: str = Query(default="", description="Search query"),
    similar_to: Optional[str] = Query(default=None, description="Tweet ID to find similar"),
    mode: str = Query(default="text", description="Search mode: text, semantic, hybrid"),
    sort: str = Query(default="relevance", description="Sort by: relevance, date_desc, date_asc, likes, retweets"),
    limit: int = Query(default=50, le=500),
    offset: int = Query(default=0),
    include_retweets: Optional[str] = Query(default=None),
    include_replies: Optional[str] = Query(default=None),
    date_from: Optional[str] = Query(default=None),
    date_to: Optional[str] = Query(default=None),
    cluster_id: Optional[str] = Query(default=None),
    super_cluster_id: Optional[str] = Query(default=None),
    category_id: Optional[str] = Query(default=None),
):
    """Search tweets and return HTML partial."""
    # Convert string params to proper types
    # Checkboxes: if param is missing or empty, it means unchecked (false)
    include_retweets_bool = include_retweets.lower() == "true" if include_retweets else False
    include_replies_bool = include_replies.lower() == "true" if include_replies else False

    # Debug logging
    print(f"[search] q={q!r}, include_retweets={include_retweets!r} -> {include_retweets_bool}, include_replies={include_replies!r} -> {include_replies_bool}")
    cluster_id_int = int(cluster_id) if cluster_id and cluster_id.strip() else None
    super_cluster_id_int = int(super_cluster_id) if super_cluster_id and super_cluster_id.strip() else None
    category_id_int = int(category_id) if category_id and category_id.strip() else None

    # Handle "Find similar" queries
    if similar_to:
        async with get_cursor() as cur:
            # Get the source tweet
            source_tweet = await queries.get_tweet_by_id(cur, similar_to)
            tweets = await queries.get_similar_tweets_with_metadata(cur, similar_to, limit=limit)

        return templates.TemplateResponse(
            "partials/search_results.html",
            {
                "request": request,
                "tweets": tweets,
                "query": f"Similar to: {source_tweet['full_text'][:50]}..." if source_tweet else "",
                "similar_to": similar_to,
                "source_tweet": source_tweet,
            },
        )

    if not q.strip():
        return templates.TemplateResponse(
            "partials/search_results.html",
            {"request": request, "tweets": [], "query": q},
        )

    # Parse date strings to datetime
    parsed_date_from = None
    parsed_date_to = None
    if date_from and date_from.strip():
        try:
            parsed_date_from = datetime.fromisoformat(date_from)
        except ValueError:
            pass
    if date_to and date_to.strip():
        try:
            parsed_date_to = datetime.fromisoformat(date_to)
        except ValueError:
            pass

    async with get_cursor() as cur:
        tweets, total_count = await queries.search_tweets_fulltext_with_filters(
            cur,
            q,
            limit=limit,
            offset=offset,
            sort=sort,
            include_retweets=include_retweets_bool,
            include_replies=include_replies_bool,
            date_from=parsed_date_from,
            date_to=parsed_date_to,
            cluster_id=cluster_id_int,
            super_cluster_id=super_cluster_id_int,
            category_id=category_id_int,
        )

    return templates.TemplateResponse(
        "partials/search_results.html",
        {"request": request, "tweets": tweets, "query": q, "total_count": total_count, "showing": len(tweets)},
    )


@router.get("/search/more", response_class=HTMLResponse)
async def search_more(
    request: Request,
    q: str = Query(default="", description="Search query"),
    sort: str = Query(default="relevance"),
    limit: int = Query(default=50, le=100),
    offset: int = Query(default=0),
    include_retweets: Optional[str] = Query(default=None),
    include_replies: Optional[str] = Query(default=None),
    date_from: Optional[str] = Query(default=None),
    date_to: Optional[str] = Query(default=None),
    cluster_id: Optional[str] = Query(default=None),
    super_cluster_id: Optional[str] = Query(default=None),
    category_id: Optional[str] = Query(default=None),
):
    """Load more search results (returns just tweet cards)."""
    include_retweets_bool = include_retweets.lower() == "true" if include_retweets else False
    include_replies_bool = include_replies.lower() == "true" if include_replies else False
    cluster_id_int = int(cluster_id) if cluster_id and cluster_id.strip() else None
    super_cluster_id_int = int(super_cluster_id) if super_cluster_id and super_cluster_id.strip() else None
    category_id_int = int(category_id) if category_id and category_id.strip() else None

    parsed_date_from = None
    parsed_date_to = None
    if date_from and date_from.strip():
        try:
            parsed_date_from = datetime.fromisoformat(date_from)
        except ValueError:
            pass
    if date_to and date_to.strip():
        try:
            parsed_date_to = datetime.fromisoformat(date_to)
        except ValueError:
            pass

    async with get_cursor() as cur:
        tweets, _ = await queries.search_tweets_fulltext_with_filters(
            cur,
            q,
            limit=limit,
            offset=offset,
            sort=sort,
            include_retweets=include_retweets_bool,
            include_replies=include_replies_bool,
            date_from=parsed_date_from,
            date_to=parsed_date_to,
            cluster_id=cluster_id_int,
            super_cluster_id=super_cluster_id_int,
            category_id=category_id_int,
        )

    return templates.TemplateResponse(
        "partials/tweet_cards_only.html",
        {"request": request, "tweets": tweets, "query": q},
    )
