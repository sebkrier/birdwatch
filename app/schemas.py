"""Pydantic v2 models for the application."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class TweetBase(BaseModel):
    """Base tweet model with common fields."""

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


class TweetCreate(TweetBase):
    """Model for creating a new tweet."""

    archive_source: Optional[str] = None


class Tweet(TweetBase):
    """Full tweet model with all fields."""

    cluster_id: Optional[int] = None
    archive_source: Optional[str] = None
    imported_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class TweetWithSimilarity(Tweet):
    """Tweet with similarity score for search results."""

    similarity: float = 0.0


class MediaBase(BaseModel):
    """Base media model."""

    id: str
    tweet_id: str
    media_type: str  # 'photo', 'video', 'animated_gif'
    media_url: Optional[str] = None
    local_path: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    duration_ms: Optional[int] = None
    alt_text: Optional[str] = None


class Media(MediaBase):
    """Full media model."""

    imported_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class ClusterBase(BaseModel):
    """Base cluster model."""

    label: Optional[str] = None
    description: Optional[str] = None
    keywords: list[str] = Field(default_factory=list)
    tweet_count: int = 0


class Cluster(ClusterBase):
    """Full cluster model."""

    id: int
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class ImportProgress(BaseModel):
    """Import progress for SSE updates."""

    job_id: str
    status: str  # 'in_progress', 'completed', 'failed'
    tweets_processed: int = 0
    tweets_total: int = 0
    media_processed: int = 0
    current_file: Optional[str] = None
    error: Optional[str] = None


class SearchQuery(BaseModel):
    """Search query parameters."""

    q: str
    mode: str = "hybrid"  # 'text', 'semantic', 'hybrid'
    limit: int = Field(default=50, le=200)
    offset: int = 0
    cluster_id: Optional[int] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    include_retweets: bool = True
    include_replies: bool = True
