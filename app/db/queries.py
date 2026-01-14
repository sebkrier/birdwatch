"""All SQL queries as functions with parameterized queries only.

IMPORTANT: Never use f-strings or string concatenation for SQL.
Always use parameterized queries (%s placeholders) to prevent SQL injection.
psycopg3 uses %s for all placeholders (not $1, $2).
"""

from datetime import datetime
from typing import Optional

import psycopg
from psycopg.rows import dict_row


async def insert_tweet(
    cur: psycopg.AsyncCursor,
    *,
    id: str,
    created_at: datetime,
    full_text: str,
    is_retweet: bool = False,
    is_reply: bool = False,
    is_thread: bool = False,
    reply_to_tweet_id: Optional[str] = None,
    reply_to_user_id: Optional[str] = None,
    reply_to_username: Optional[str] = None,
    conversation_id: Optional[str] = None,
    retweet_count: int = 0,
    favorite_count: int = 0,
    hashtags: list[str] | None = None,
    mentions: list[str] | None = None,
    urls: list[str] | None = None,
    archive_source: Optional[str] = None,
) -> dict:
    """Insert a tweet, updating on conflict."""
    await cur.execute(
        """
        INSERT INTO tweets (
            id, created_at, full_text, is_retweet, is_reply, is_thread,
            reply_to_tweet_id, reply_to_user_id, reply_to_username,
            conversation_id, retweet_count, favorite_count,
            hashtags, mentions, urls, archive_source
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        ON CONFLICT (id) DO UPDATE SET
            retweet_count = EXCLUDED.retweet_count,
            favorite_count = EXCLUDED.favorite_count,
            updated_at = NOW()
        RETURNING *
        """,
        (
            id,
            created_at,
            full_text,
            is_retweet,
            is_reply,
            is_thread,
            reply_to_tweet_id,
            reply_to_user_id,
            reply_to_username,
            conversation_id,
            retweet_count,
            favorite_count,
            hashtags or [],
            mentions or [],
            urls or [],
            archive_source,
        ),
    )
    return await cur.fetchone()


async def insert_media(
    cur: psycopg.AsyncCursor,
    *,
    id: str,
    tweet_id: str,
    media_type: str,
    media_url: Optional[str] = None,
    local_path: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    duration_ms: Optional[int] = None,
    alt_text: Optional[str] = None,
) -> dict:
    """Insert media, skipping on conflict."""
    await cur.execute(
        """
        INSERT INTO media (
            id, tweet_id, media_type, media_url, local_path,
            width, height, duration_ms, alt_text
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO NOTHING
        RETURNING *
        """,
        (id, tweet_id, media_type, media_url, local_path, width, height, duration_ms, alt_text),
    )
    return await cur.fetchone()


async def get_tweet_by_id(cur: psycopg.AsyncCursor, tweet_id: str) -> Optional[dict]:
    """Get a single tweet by ID."""
    await cur.execute("SELECT * FROM tweets WHERE id = %s", (tweet_id,))
    return await cur.fetchone()


async def get_tweets_by_ids(cur: psycopg.AsyncCursor, tweet_ids: list[str]) -> list[dict]:
    """Get multiple tweets by IDs."""
    await cur.execute("SELECT * FROM tweets WHERE id = ANY(%s)", (tweet_ids,))
    return await cur.fetchall()


async def search_tweets_fulltext(
    cur: psycopg.AsyncCursor,
    query: str,
    *,
    limit: int = 50,
    offset: int = 0,
    include_retweets: bool = True,
    include_replies: bool = True,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
) -> list[dict]:
    """Full-text search using PostgreSQL ts_vector."""
    await cur.execute(
        """
        SELECT *,
               ts_rank(to_tsvector('english', full_text), plainto_tsquery('english', %s)) as rank
        FROM tweets
        WHERE to_tsvector('english', full_text) @@ plainto_tsquery('english', %s)
          AND (%s OR is_retweet = FALSE)
          AND (%s OR is_reply = FALSE)
          AND (%s::timestamptz IS NULL OR created_at >= %s)
          AND (%s::timestamptz IS NULL OR created_at <= %s)
        ORDER BY rank DESC, created_at DESC
        LIMIT %s OFFSET %s
        """,
        (query, query, include_retweets, include_replies, date_from, date_from, date_to, date_to, limit, offset),
    )
    return await cur.fetchall()


async def search_tweets_semantic(
    cur: psycopg.AsyncCursor,
    embedding: list[float],
    *,
    limit: int = 50,
    offset: int = 0,
    include_retweets: bool = True,
    include_replies: bool = True,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
) -> list[dict]:
    """Semantic search using vector similarity."""
    await cur.execute(
        """
        SELECT *,
               1 - (embedding <=> %s::vector) as similarity
        FROM tweets
        WHERE embedding IS NOT NULL
          AND (%s OR is_retweet = FALSE)
          AND (%s OR is_reply = FALSE)
          AND (%s::timestamptz IS NULL OR created_at >= %s)
          AND (%s::timestamptz IS NULL OR created_at <= %s)
        ORDER BY embedding <=> %s::vector
        LIMIT %s OFFSET %s
        """,
        (embedding, include_retweets, include_replies, date_from, date_from, date_to, date_to, embedding, limit, offset),
    )
    return await cur.fetchall()


async def search_tweets_hybrid(
    cur: psycopg.AsyncCursor,
    query: str,
    embedding: list[float],
    *,
    limit: int = 50,
    offset: int = 0,
    text_weight: float = 0.3,
    semantic_weight: float = 0.7,
    include_retweets: bool = True,
    include_replies: bool = True,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
) -> list[dict]:
    """Hybrid search combining full-text and semantic."""
    await cur.execute(
        """
        WITH text_matches AS (
            SELECT id,
                   ts_rank(to_tsvector('english', full_text), plainto_tsquery('english', %s)) as text_rank
            FROM tweets
            WHERE to_tsvector('english', full_text) @@ plainto_tsquery('english', %s)
        ),
        semantic_matches AS (
            SELECT id,
                   1 - (embedding <=> %s::vector) as semantic_rank
            FROM tweets
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT 500
        )
        SELECT t.*,
               COALESCE(tm.text_rank, 0) * %s + COALESCE(sm.semantic_rank, 0) * %s as combined_score
        FROM tweets t
        LEFT JOIN text_matches tm ON t.id = tm.id
        LEFT JOIN semantic_matches sm ON t.id = sm.id
        WHERE (tm.id IS NOT NULL OR sm.id IS NOT NULL)
          AND (%s OR t.is_retweet = FALSE)
          AND (%s OR t.is_reply = FALSE)
          AND (%s::timestamptz IS NULL OR t.created_at >= %s)
          AND (%s::timestamptz IS NULL OR t.created_at <= %s)
        ORDER BY combined_score DESC
        LIMIT %s OFFSET %s
        """,
        (
            query,
            query,
            embedding,
            embedding,
            text_weight,
            semantic_weight,
            include_retweets,
            include_replies,
            date_from,
            date_from,
            date_to,
            date_to,
            limit,
            offset,
        ),
    )
    return await cur.fetchall()


async def get_tweet_media(cur: psycopg.AsyncCursor, tweet_id: str) -> list[dict]:
    """Get all media for a tweet."""
    await cur.execute("SELECT * FROM media WHERE tweet_id = %s", (tweet_id,))
    return await cur.fetchall()


async def get_thread_context(cur: psycopg.AsyncCursor, tweet_id: str) -> list[dict]:
    """Get tweets in the same thread/conversation."""
    # First get the conversation_id of this tweet
    await cur.execute("SELECT conversation_id FROM tweets WHERE id = %s", (tweet_id,))
    result = await cur.fetchone()
    if not result or not result.get("conversation_id"):
        return []

    conversation_id = result["conversation_id"]
    await cur.execute(
        """
        SELECT * FROM tweets
        WHERE conversation_id = %s
        ORDER BY created_at
        """,
        (conversation_id,),
    )
    return await cur.fetchall()


async def get_similar_tweets(
    cur: psycopg.AsyncCursor,
    tweet_id: str,
    *,
    limit: int = 10,
) -> list[dict]:
    """Get semantically similar tweets."""
    await cur.execute(
        """
        SELECT t2.*,
               1 - (t1.embedding <=> t2.embedding) as similarity
        FROM tweets t1, tweets t2
        WHERE t1.id = %s
          AND t2.id != %s
          AND t1.embedding IS NOT NULL
          AND t2.embedding IS NOT NULL
        ORDER BY t1.embedding <=> t2.embedding
        LIMIT %s
        """,
        (tweet_id, tweet_id, limit),
    )
    return await cur.fetchall()


async def update_tweet_embedding(
    cur: psycopg.AsyncCursor,
    tweet_id: str,
    embedding: list[float],
) -> None:
    """Update a tweet's embedding."""
    await cur.execute(
        "UPDATE tweets SET embedding = %s::vector WHERE id = %s",
        (embedding, tweet_id),
    )


async def update_tweet_cluster(
    cur: psycopg.AsyncCursor,
    tweet_id: str,
    cluster_id: int,
) -> None:
    """Update a tweet's cluster assignment."""
    await cur.execute(
        "UPDATE tweets SET cluster_id = %s WHERE id = %s",
        (cluster_id, tweet_id),
    )


async def get_tweets_without_embeddings(
    cur: psycopg.AsyncCursor,
    *,
    limit: int = 1000,
) -> list[dict]:
    """Get tweets that need embeddings generated."""
    await cur.execute(
        """
        SELECT id, full_text, is_retweet, is_reply
        FROM tweets
        WHERE embedding IS NULL
        ORDER BY created_at DESC
        LIMIT %s
        """,
        (limit,),
    )
    return await cur.fetchall()


async def get_all_embeddings(cur: psycopg.AsyncCursor) -> list[dict]:
    """Get all tweets with embeddings for clustering."""
    await cur.execute(
        """
        SELECT id, embedding
        FROM tweets
        WHERE embedding IS NOT NULL
        """
    )
    return await cur.fetchall()


async def get_cluster_stats(cur: psycopg.AsyncCursor) -> list[dict]:
    """Get cluster statistics."""
    await cur.execute(
        """
        SELECT c.*, COUNT(t.id) as actual_count
        FROM clusters c
        LEFT JOIN tweets t ON t.cluster_id = c.id
        GROUP BY c.id
        ORDER BY actual_count DESC
        """
    )
    return await cur.fetchall()


async def create_import_record(
    cur: psycopg.AsyncCursor,
    *,
    archive_filename: str,
    archive_hash: Optional[str] = None,
) -> dict:
    """Create an import history record."""
    await cur.execute(
        """
        INSERT INTO import_history (archive_filename, archive_hash)
        VALUES (%s, %s)
        RETURNING *
        """,
        (archive_filename, archive_hash),
    )
    return await cur.fetchone()


async def update_import_record(
    cur: psycopg.AsyncCursor,
    import_id: int,
    *,
    tweets_imported: Optional[int] = None,
    tweets_updated: Optional[int] = None,
    tweets_skipped: Optional[int] = None,
    media_imported: Optional[int] = None,
    status: Optional[str] = None,
) -> None:
    """Update an import history record."""
    await cur.execute(
        """
        UPDATE import_history SET
            tweets_imported = COALESCE(%s, tweets_imported),
            tweets_updated = COALESCE(%s, tweets_updated),
            tweets_skipped = COALESCE(%s, tweets_skipped),
            media_imported = COALESCE(%s, media_imported),
            status = COALESCE(%s, status),
            completed_at = CASE WHEN %s IN ('completed', 'failed') THEN NOW() ELSE completed_at END
        WHERE id = %s
        """,
        (tweets_imported, tweets_updated, tweets_skipped, media_imported, status, status, import_id),
    )


async def get_import_by_hash(cur: psycopg.AsyncCursor, archive_hash: str) -> Optional[dict]:
    """Check if an archive was already imported."""
    await cur.execute(
        "SELECT * FROM import_history WHERE archive_hash = %s AND status = 'completed'",
        (archive_hash,),
    )
    return await cur.fetchone()


async def get_tweet_count(cur: psycopg.AsyncCursor) -> int:
    """Get total tweet count."""
    await cur.execute("SELECT COUNT(*) as count FROM tweets")
    result = await cur.fetchone()
    return result["count"] if result else 0


async def search_tweets_fulltext_with_filters(
    cur: psycopg.AsyncCursor,
    query: str,
    *,
    limit: int = 50,
    offset: int = 0,
    sort: str = "relevance",
    include_retweets: bool = True,
    include_replies: bool = True,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    cluster_id: Optional[int] = None,
    super_cluster_id: Optional[int] = None,
    category_id: Optional[int] = None,
) -> tuple[list[dict], int]:
    """Full-text search with cluster/category filtering and metadata."""
    # Validate and map sort parameter to prevent SQL injection
    sort_clauses = {
        "relevance": "rank DESC, t.created_at DESC",
        "date_desc": "t.created_at DESC",
        "date_asc": "t.created_at ASC",
        "likes": "t.favorite_count DESC NULLS LAST, t.created_at DESC",
        "retweets": "t.retweet_count DESC NULLS LAST, t.created_at DESC",
        "longest": "LENGTH(t.full_text) DESC, t.created_at DESC",
    }
    order_by = sort_clauses.get(sort, sort_clauses["relevance"])

    await cur.execute(
        f"""
        SELECT t.*,
               ts_rank(to_tsvector('english', t.full_text), plainto_tsquery('english', %s)) as rank,
               c.label as cluster_label,
               sc.label as super_cluster_label,
               sc.color as super_cluster_color,
               EXISTS(SELECT 1 FROM media m WHERE m.tweet_id = t.id) as has_media,
               (SELECT array_agg(cat.name) FROM tweet_categories tc
                JOIN categories cat ON tc.category_id = cat.id
                WHERE tc.tweet_id = t.id) as category_names
        FROM tweets t
        LEFT JOIN clusters c ON c.id = t.cluster_id AND c.is_super_cluster = false
        LEFT JOIN clusters sc ON sc.id = c.parent_cluster_id
        WHERE to_tsvector('english', t.full_text) @@ plainto_tsquery('english', %s)
          AND (%s OR t.is_retweet = FALSE)
          AND (%s OR t.is_reply = FALSE)
          AND (%s::timestamptz IS NULL OR t.created_at >= %s)
          AND (%s::timestamptz IS NULL OR t.created_at <= %s)
          AND (%s::int IS NULL OR t.cluster_id = %s)
          AND (%s::int IS NULL OR c.parent_cluster_id = %s)
          AND (%s::int IS NULL OR EXISTS (
              SELECT 1 FROM tweet_categories tc WHERE tc.tweet_id = t.id AND tc.category_id = %s
          ))
        ORDER BY {order_by}
        LIMIT %s OFFSET %s
        """,
        (
            query, query,
            include_retweets, include_replies,
            date_from, date_from, date_to, date_to,
            cluster_id, cluster_id,
            super_cluster_id, super_cluster_id,
            category_id, category_id,
            limit, offset
        ),
    )
    results = await cur.fetchall()

    # Get total count (without limit)
    await cur.execute(
        """
        SELECT COUNT(*) as total
        FROM tweets t
        LEFT JOIN clusters c ON c.id = t.cluster_id AND c.is_super_cluster = false
        WHERE to_tsvector('english', t.full_text) @@ plainto_tsquery('english', %s)
          AND (%s OR t.is_retweet = FALSE)
          AND (%s OR t.is_reply = FALSE)
          AND (%s::timestamptz IS NULL OR t.created_at >= %s)
          AND (%s::timestamptz IS NULL OR t.created_at <= %s)
          AND (%s::int IS NULL OR t.cluster_id = %s)
          AND (%s::int IS NULL OR c.parent_cluster_id = %s)
          AND (%s::int IS NULL OR EXISTS (
              SELECT 1 FROM tweet_categories tc WHERE tc.tweet_id = t.id AND tc.category_id = %s
          ))
        """,
        (
            query,
            include_retweets, include_replies,
            date_from, date_from, date_to, date_to,
            cluster_id, cluster_id,
            super_cluster_id, super_cluster_id,
            category_id, category_id,
        ),
    )
    row = await cur.fetchone()
    total = row['total'] if row else 0

    return results, total


async def get_similar_tweets_with_metadata(
    cur: psycopg.AsyncCursor,
    tweet_id: str,
    *,
    limit: int = 20,
) -> list[dict]:
    """Get semantically similar tweets with metadata."""
    await cur.execute(
        """
        SELECT t2.*,
               1 - (t1.embedding <=> t2.embedding) as similarity,
               c.label as cluster_label,
               sc.label as super_cluster_label,
               sc.color as super_cluster_color,
               EXISTS(SELECT 1 FROM media m WHERE m.tweet_id = t2.id) as has_media
        FROM tweets t1, tweets t2
        LEFT JOIN clusters c ON c.id = t2.cluster_id AND c.is_super_cluster = false
        LEFT JOIN clusters sc ON sc.id = c.parent_cluster_id
        WHERE t1.id = %s
          AND t2.id != %s
          AND t1.embedding IS NOT NULL
          AND t2.embedding IS NOT NULL
        ORDER BY t1.embedding <=> t2.embedding
        LIMIT %s
        """,
        (tweet_id, tweet_id, limit),
    )
    return await cur.fetchall()


async def get_all_clusters_for_filter(cur: psycopg.AsyncCursor) -> list[dict]:
    """Get all clusters for filter dropdown."""
    await cur.execute(
        """
        SELECT id, label, tweet_count, parent_cluster_id, is_super_cluster, color
        FROM clusters
        ORDER BY is_super_cluster DESC, tweet_count DESC
        """
    )
    return await cur.fetchall()


async def get_tweets_by_filters(
    cur: psycopg.AsyncCursor,
    *,
    limit: int = 100,
    offset: int = 0,
    sort_by: str = "date_desc",
    include_retweets: bool = True,
    include_replies: bool = True,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> list[dict]:
    """Get tweets by filters without requiring search terms."""
    # Map sort parameter to ORDER BY clause
    sort_clauses = {
        "relevance": "t.created_at DESC",  # Default to date for no search terms
        "date_desc": "t.created_at DESC",
        "date_asc": "t.created_at ASC",
        "likes": "t.favorite_count DESC NULLS LAST, t.created_at DESC",
        "retweets": "t.retweet_count DESC NULLS LAST, t.created_at DESC",
        "longest": "LENGTH(t.full_text) DESC, t.created_at DESC",
    }
    order_by = sort_clauses.get(sort_by, sort_clauses["date_desc"])

    await cur.execute(
        f"""
        SELECT t.*,
               c.label as cluster_label,
               sc.label as super_cluster_label,
               sc.color as super_cluster_color,
               EXISTS(SELECT 1 FROM media m WHERE m.tweet_id = t.id) as has_media,
               (SELECT array_agg(cat.name) FROM tweet_categories tc
                JOIN categories cat ON tc.category_id = cat.id
                WHERE tc.tweet_id = t.id) as category_names
        FROM tweets t
        LEFT JOIN clusters c ON c.id = t.cluster_id AND c.is_super_cluster = false
        LEFT JOIN clusters sc ON sc.id = c.parent_cluster_id
        WHERE (%s OR t.is_retweet = FALSE)
          AND (%s OR t.is_reply = FALSE)
          AND (%s::text IS NULL OR t.created_at >= %s::timestamptz)
          AND (%s::text IS NULL OR t.created_at <= %s::timestamptz)
        ORDER BY {order_by}
        LIMIT %s OFFSET %s
        """,
        (
            include_retweets, include_replies,
            date_from, date_from,
            date_to, date_to,
            limit, offset
        ),
    )
    return await cur.fetchall()
