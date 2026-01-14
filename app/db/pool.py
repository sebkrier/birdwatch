"""PostgreSQL async connection pool using psycopg3."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import psycopg
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row

from app.config import get_settings

# Global pool instance
_pool: AsyncConnectionPool | None = None


async def init_pool() -> AsyncConnectionPool:
    """Initialize the connection pool."""
    global _pool
    if _pool is not None:
        return _pool

    settings = get_settings()
    _pool = AsyncConnectionPool(
        conninfo=settings.database_url,
        min_size=settings.db_pool_min_size,
        max_size=settings.db_pool_max_size,
        open=False,
    )
    await _pool.open()
    return _pool


async def close_pool() -> None:
    """Close the connection pool."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def get_pool() -> AsyncConnectionPool:
    """Get the connection pool instance."""
    if _pool is None:
        raise RuntimeError("Database pool not initialized. Call init_pool() first.")
    return _pool


@asynccontextmanager
async def get_connection() -> AsyncGenerator[psycopg.AsyncConnection, None]:
    """Get a connection from the pool."""
    pool = get_pool()
    async with pool.connection() as conn:
        yield conn


@asynccontextmanager
async def get_cursor(
    *, row_factory=dict_row
) -> AsyncGenerator[psycopg.AsyncCursor, None]:
    """Get a cursor with dict rows by default."""
    async with get_connection() as conn:
        async with conn.cursor(row_factory=row_factory) as cur:
            yield cur
