-- Extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- For fuzzy text search

-- Tweets table
CREATE TABLE IF NOT EXISTS tweets (
    id TEXT PRIMARY KEY,                           -- Twitter's tweet ID (id_str)
    created_at TIMESTAMPTZ NOT NULL,
    full_text TEXT NOT NULL,

    -- Tweet metadata
    is_retweet BOOLEAN DEFAULT FALSE,
    is_reply BOOLEAN DEFAULT FALSE,
    is_thread BOOLEAN DEFAULT FALSE,              -- Part of a self-thread
    reply_to_tweet_id TEXT,
    reply_to_user_id TEXT,
    reply_to_username TEXT,
    conversation_id TEXT,                          -- For threading

    -- Engagement (at time of archive)
    retweet_count INTEGER DEFAULT 0,
    favorite_count INTEGER DEFAULT 0,

    -- Extracted entities (as arrays for easy querying)
    hashtags TEXT[] DEFAULT '{}',
    mentions TEXT[] DEFAULT '{}',
    urls TEXT[] DEFAULT '{}',

    -- AI-generated
    embedding vector(1536),                        -- text-embedding-3-small
    cluster_id INTEGER,

    -- Import tracking
    archive_source TEXT,                           -- Which archive file this came from
    imported_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Media table
CREATE TABLE IF NOT EXISTS media (
    id TEXT PRIMARY KEY,                           -- Twitter's media ID
    tweet_id TEXT NOT NULL REFERENCES tweets(id) ON DELETE CASCADE,
    media_type TEXT NOT NULL,                      -- 'photo', 'video', 'animated_gif'
    media_url TEXT,                                -- Original Twitter URL
    local_path TEXT,                               -- Path in media/ directory
    width INTEGER,
    height INTEGER,
    duration_ms INTEGER,                           -- For videos
    alt_text TEXT,                                 -- If available
    imported_at TIMESTAMPTZ DEFAULT NOW()
);

-- Clusters table (populated after clustering)
CREATE TABLE IF NOT EXISTS clusters (
    id SERIAL PRIMARY KEY,
    label TEXT,                                    -- AI-generated or manual
    description TEXT,
    keywords TEXT[] DEFAULT '{}',                  -- Representative terms
    tweet_count INTEGER DEFAULT 0,
    centroid vector(1536),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- UMAP coordinates for visualization (separate for flexibility)
CREATE TABLE IF NOT EXISTS tweet_coordinates (
    tweet_id TEXT PRIMARY KEY REFERENCES tweets(id) ON DELETE CASCADE,
    x FLOAT NOT NULL,
    y FLOAT NOT NULL,
    computed_at TIMESTAMPTZ DEFAULT NOW()
);

-- Import history (for tracking re-imports)
CREATE TABLE IF NOT EXISTS import_history (
    id SERIAL PRIMARY KEY,
    archive_filename TEXT NOT NULL,
    archive_hash TEXT,                             -- SHA256 of archive for deduplication
    tweets_imported INTEGER DEFAULT 0,
    tweets_updated INTEGER DEFAULT 0,
    tweets_skipped INTEGER DEFAULT 0,
    media_imported INTEGER DEFAULT 0,
    started_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    status TEXT DEFAULT 'in_progress'              -- 'in_progress', 'completed', 'failed'
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_tweets_created_at ON tweets(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_tweets_cluster ON tweets(cluster_id);
CREATE INDEX IF NOT EXISTS idx_tweets_fulltext ON tweets USING gin(to_tsvector('english', full_text));
CREATE INDEX IF NOT EXISTS idx_tweets_hashtags ON tweets USING gin(hashtags);
CREATE INDEX IF NOT EXISTS idx_tweets_mentions ON tweets USING gin(mentions);
CREATE INDEX IF NOT EXISTS idx_tweets_reply_to ON tweets(reply_to_tweet_id) WHERE reply_to_tweet_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_tweets_conversation ON tweets(conversation_id) WHERE conversation_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_media_tweet ON media(tweet_id);
CREATE INDEX IF NOT EXISTS idx_coordinates_cluster ON tweet_coordinates(tweet_id);

-- Note: The vector index is created separately after data is loaded
-- to allow for better optimization based on actual data size:
-- CREATE INDEX IF NOT EXISTS idx_tweets_embedding ON tweets USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
