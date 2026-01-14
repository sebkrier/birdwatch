"""Explore routes for category/subcluster visualization."""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
from psycopg.rows import dict_row

from app.db.pool import get_connection

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))


@router.get("", response_class=HTMLResponse)
async def explore_page(request: Request):
    """Explore page with category visualization."""
    categories = []
    subclusters = []

    try:
        async with get_connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                # Get categories with tweet counts
                await cur.execute('''
                    SELECT c.id, c.name, c.color, COUNT(tc.tweet_id) as tweet_count
                    FROM categories c
                    LEFT JOIN tweet_categories tc ON c.id = tc.category_id
                    GROUP BY c.id, c.name, c.color
                    ORDER BY tweet_count DESC
                ''')
                categories = await cur.fetchall()
                print(f"[explore] Loaded {len(categories)} categories")

                # Get subclusters with counts
                await cur.execute('''
                    SELECT s.id, s.category_id, s.label, s.tweet_count,
                           c.name as category_name, c.color as category_color
                    FROM subclusters s
                    JOIN categories c ON s.category_id = c.id
                    ORDER BY s.tweet_count DESC
                ''')
                subclusters = await cur.fetchall()
                print(f"[explore] Loaded {len(subclusters)} subclusters")
    except Exception as e:
        print(f"[explore] Error loading data: {e}")

    return templates.TemplateResponse(
        "explore.html",
        {"request": request, "categories": categories, "subclusters": subclusters},
    )


@router.get("/data", response_class=JSONResponse)
async def explore_data():
    """Return visualization data as JSON for Deck.gl."""
    async with get_connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            # Get tweet coordinates with category info and metadata for filtering
            await cur.execute('''
                SELECT DISTINCT ON (tc.tweet_id)
                       tc.tweet_id as id, tc.x, tc.y,
                       cat.id as category_id,
                       cat.name as category_name,
                       cat.color as category_color,
                       LEFT(t.full_text, 100) as text_preview,
                       tcat.subcluster_id,
                       t.created_at,
                       t.is_retweet,
                       t.is_reply,
                       t.favorite_count
                FROM tweet_coordinates tc
                JOIN tweets t ON t.id = tc.tweet_id
                LEFT JOIN tweet_categories tcat ON t.id = tcat.tweet_id
                LEFT JOIN categories cat ON tcat.category_id = cat.id
                ORDER BY tc.tweet_id, cat.id
            ''')
            points = await cur.fetchall()

            # Get categories
            await cur.execute('''
                SELECT c.id, c.name, c.color, COUNT(tc.tweet_id) as tweet_count
                FROM categories c
                LEFT JOIN tweet_categories tc ON c.id = tc.category_id
                GROUP BY c.id, c.name, c.color
                ORDER BY tweet_count DESC
            ''')
            categories = await cur.fetchall()

            # Get subclusters
            await cur.execute('''
                SELECT s.id, s.category_id, s.label, s.tweet_count
                FROM subclusters s
                ORDER BY s.tweet_count DESC
            ''')
            subclusters = await cur.fetchall()

    return {
        "points": [dict(p) for p in points],
        "categories": [dict(c) for c in categories],
        "subclusters": [dict(s) for s in subclusters],
    }


@router.get("/tweet/{tweet_id}", response_class=JSONResponse)
async def get_tweet_detail(tweet_id: str):
    """Get full tweet details for the preview panel."""
    async with get_connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute('''
                SELECT t.id, t.full_text, t.created_at, t.retweet_count, t.favorite_count,
                       t.is_retweet, t.is_reply, t.quality_tier
                FROM tweets t
                WHERE t.id = %s
            ''', (tweet_id,))
            tweet = await cur.fetchone()

            if tweet:
                # Get categories for this tweet
                await cur.execute('''
                    SELECT c.name, c.color
                    FROM categories c
                    JOIN tweet_categories tc ON c.id = tc.category_id
                    WHERE tc.tweet_id = %s
                ''', (tweet_id,))
                categories = await cur.fetchall()
                tweet = dict(tweet)
                tweet['categories'] = [dict(c) for c in categories]

    if not tweet:
        return JSONResponse({"error": "Tweet not found"}, status_code=404)

    return tweet


@router.get("/categories", response_class=HTMLResponse)
async def categories_list(request: Request):
    """List all categories with subclusters and sample tweets."""
    async with get_connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            # Get categories with counts
            await cur.execute('''
                SELECT c.id, c.name, c.description, c.color,
                       COUNT(tc.tweet_id) as tweet_count
                FROM categories c
                LEFT JOIN tweet_categories tc ON c.id = tc.category_id
                GROUP BY c.id, c.name, c.description, c.color
                ORDER BY tweet_count DESC
            ''')
            categories = await cur.fetchall()

            # Get subclusters grouped by category
            await cur.execute('''
                SELECT s.id, s.category_id, s.label, s.tweet_count, s.keywords
                FROM subclusters s
                ORDER BY s.category_id, s.tweet_count DESC
            ''')
            all_subclusters = await cur.fetchall()

            # Group subclusters by category
            subclusters_by_cat = {}
            for sc in all_subclusters:
                cat_id = sc['category_id']
                if cat_id not in subclusters_by_cat:
                    subclusters_by_cat[cat_id] = []
                subclusters_by_cat[cat_id].append(dict(sc))

            # Get sample tweets for each category
            category_samples = {}
            for cat in categories:
                await cur.execute('''
                    SELECT t.id, t.full_text, t.created_at, t.favorite_count
                    FROM tweets t
                    JOIN tweet_categories tc ON t.id = tc.tweet_id
                    WHERE tc.category_id = %s
                    ORDER BY t.favorite_count DESC
                    LIMIT 5
                ''', (cat['id'],))
                category_samples[cat['id']] = await cur.fetchall()

    return templates.TemplateResponse(
        "categories.html",
        {
            "request": request,
            "categories": categories,
            "subclusters_by_cat": subclusters_by_cat,
            "samples": category_samples,
        },
    )


# Keep old cluster routes for backwards compatibility
@router.get("/clusters", response_class=HTMLResponse)
async def clusters_list(request: Request):
    """Redirect to categories view."""
    return await categories_list(request)
