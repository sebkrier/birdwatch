"""Tests for clustering service.

Tests cluster quality, count ranges, and label uniqueness.
"""

import pytest
import numpy as np
from unittest.mock import AsyncMock, patch, MagicMock

from app.services.clustering import (
    compute_tfidf_keywords,
    compute_silhouette_scores,
    create_ward_hierarchy,
    run_hdbscan_clustering,
)


class TestTfidfKeywords:
    """Tests for TF-IDF keyword extraction."""

    def test_extracts_keywords_for_each_cluster(self):
        """Keywords are extracted for each non-noise cluster."""
        texts = [
            "Machine learning is amazing",
            "Deep learning neural networks",
            "AI artificial intelligence models",
            "Climate change policy",
            "Environmental regulations law",
            "Carbon emissions reduction",
        ]
        labels = np.array([0, 0, 0, 1, 1, 1])

        keywords = compute_tfidf_keywords(texts, labels, top_n=5)

        assert 0 in keywords
        assert 1 in keywords
        assert len(keywords[0]) > 0
        assert len(keywords[1]) > 0

    def test_skips_noise_points(self):
        """Noise points (label -1) are not included in keyword extraction."""
        texts = ["Noise tweet", "AI machine learning", "Deep learning"]
        labels = np.array([-1, 0, 0])

        keywords = compute_tfidf_keywords(texts, labels, top_n=5)

        assert -1 not in keywords
        assert 0 in keywords

    def test_returns_empty_for_all_noise(self):
        """Returns empty dict when all points are noise."""
        texts = ["Noise1", "Noise2", "Noise3"]
        labels = np.array([-1, -1, -1])

        keywords = compute_tfidf_keywords(texts, labels, top_n=5)

        assert keywords == {}


class TestSilhouetteScores:
    """Tests for silhouette score computation."""

    def test_computes_overall_and_per_cluster_scores(self):
        """Both overall and per-cluster scores are computed."""
        # Create well-separated clusters
        np.random.seed(42)
        cluster1 = np.random.randn(20, 2) + [0, 0]
        cluster2 = np.random.randn(20, 2) + [10, 10]
        embeddings = np.vstack([cluster1, cluster2])
        labels = np.array([0] * 20 + [1] * 20)

        overall, per_cluster = compute_silhouette_scores(embeddings, labels)

        assert overall > 0.5  # Well-separated clusters should have high score
        assert 0 in per_cluster
        assert 1 in per_cluster
        assert per_cluster[0] > 0
        assert per_cluster[1] > 0

    def test_handles_noise_points(self):
        """Noise points are excluded from silhouette calculation."""
        np.random.seed(42)
        cluster1 = np.random.randn(20, 2) + [0, 0]
        cluster2 = np.random.randn(20, 2) + [10, 10]
        noise = np.random.randn(5, 2) + [5, 5]
        embeddings = np.vstack([cluster1, cluster2, noise])
        labels = np.array([0] * 20 + [1] * 20 + [-1] * 5)

        overall, per_cluster = compute_silhouette_scores(embeddings, labels)

        assert -1 not in per_cluster
        assert overall > 0

    def test_returns_zero_for_single_cluster(self):
        """Returns 0 when there's only one cluster."""
        embeddings = np.random.randn(20, 2)
        labels = np.array([0] * 20)

        overall, per_cluster = compute_silhouette_scores(embeddings, labels)

        assert overall == 0.0
        assert per_cluster == {}


class TestWardHierarchy:
    """Tests for Ward linkage hierarchy creation."""

    def test_creates_hierarchy_with_n_super_clusters(self):
        """Creates specified number of super-clusters."""
        np.random.seed(42)
        # Create 10 clusters
        embeddings = []
        labels = []
        for i in range(10):
            cluster = np.random.randn(20, 2) + [i * 5, i * 5]
            embeddings.extend(cluster)
            labels.extend([i] * 20)

        embeddings = np.array(embeddings)
        labels = np.array(labels)

        hierarchy = create_ward_hierarchy(embeddings, labels, n_super_clusters=3)

        assert len(set(hierarchy.values())) == 3  # 3 super-clusters
        assert len(hierarchy) == 10  # All 10 clusters mapped

    def test_handles_fewer_clusters_than_target(self):
        """Returns identity mapping when fewer clusters than target."""
        np.random.seed(42)
        embeddings = np.random.randn(40, 2)
        labels = np.array([0] * 20 + [1] * 20)

        hierarchy = create_ward_hierarchy(embeddings, labels, n_super_clusters=5)

        # With only 2 clusters, can't create 5 super-clusters
        assert len(hierarchy) == 2


class TestHdbscanClustering:
    """Tests for HDBSCAN clustering."""

    def test_min_cluster_size_enforced(self):
        """No cluster should be smaller than min_cluster_size."""
        np.random.seed(42)
        # Create data with clear clusters
        cluster1 = np.random.randn(150, 10) + np.array([0] * 10)
        cluster2 = np.random.randn(150, 10) + np.array([10] * 10)
        embeddings = np.vstack([cluster1, cluster2])

        labels = run_hdbscan_clustering(
            embeddings,
            min_cluster_size=100,
            min_samples=50,
            cluster_selection_epsilon=0.5,
        )

        # Check cluster sizes
        unique_labels = set(labels)
        for label in unique_labels:
            if label >= 0:  # Skip noise
                cluster_size = (labels == label).sum()
                assert cluster_size >= 100, f"Cluster {label} has only {cluster_size} points"

    def test_returns_noise_for_scattered_points(self):
        """Scattered points should be labeled as noise (-1)."""
        np.random.seed(42)
        embeddings = np.random.randn(100, 10) * 10  # Very scattered

        labels = run_hdbscan_clustering(
            embeddings,
            min_cluster_size=50,
            min_samples=25,
            cluster_selection_epsilon=0.3,
        )

        # Most points should be noise for scattered data
        noise_count = (labels == -1).sum()
        assert noise_count > 0


class TestClusterIntegration:
    """Integration tests for full clustering pipeline."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample embedding data with clear clusters."""
        np.random.seed(42)
        clusters = []
        texts = []
        for i in range(5):
            cluster_embeddings = np.random.randn(50, 10) + np.array([i * 5] * 10)
            clusters.append(cluster_embeddings)
            texts.extend([f"Topic {i} tweet {j}" for j in range(50)])

        return np.vstack(clusters), texts

    def test_cluster_count_in_expected_range(self, sample_data):
        """Cluster count should be reasonable given data structure."""
        embeddings, texts = sample_data

        labels = run_hdbscan_clustering(
            embeddings,
            min_cluster_size=20,
            min_samples=10,
            cluster_selection_epsilon=0.3,
        )

        n_clusters = len(set(l for l in labels if l >= 0))
        # With 5 clear clusters of 50 points each, we expect ~3-7 clusters
        assert 1 <= n_clusters <= 10, f"Got {n_clusters} clusters"

    def test_keywords_are_non_empty_strings(self, sample_data):
        """All keywords should be non-empty strings."""
        embeddings, texts = sample_data

        labels = run_hdbscan_clustering(
            embeddings,
            min_cluster_size=20,
            min_samples=10,
            cluster_selection_epsilon=0.3,
        )

        keywords = compute_tfidf_keywords(texts, labels, top_n=5)

        for cluster_id, kw_list in keywords.items():
            assert len(kw_list) > 0, f"Cluster {cluster_id} has no keywords"
            for kw in kw_list:
                assert isinstance(kw, str), f"Keyword is not string: {kw}"
                assert len(kw) > 0, f"Empty keyword in cluster {cluster_id}"


@pytest.mark.asyncio
class TestDatabaseIntegration:
    """Tests that verify database state after clustering.

    These tests run against the main database (not test db) to verify
    clustering results. Skip if schema doesn't exist.
    """

    async def test_cluster_labels_are_unique(self):
        """All cluster labels should be unique."""
        from app.db.pool import init_pool, close_pool
        from app.config import get_settings
        import psycopg

        # Connect directly to main db
        settings = get_settings()
        try:
            async with await psycopg.AsyncConnection.connect(settings.database_url) as conn:
                async with conn.cursor() as cur:
                    # Check if is_super_cluster column exists
                    await cur.execute("""
                        SELECT column_name FROM information_schema.columns
                        WHERE table_name = 'clusters' AND column_name = 'is_super_cluster'
                    """)
                    if not await cur.fetchone():
                        pytest.skip("is_super_cluster column not in schema")

                    await cur.execute("""
                        SELECT label, COUNT(*) as cnt
                        FROM clusters
                        WHERE is_super_cluster = false
                        GROUP BY label
                        HAVING COUNT(*) > 1
                    """)
                    duplicates = await cur.fetchall()

                    assert len(duplicates) == 0, f"Duplicate labels found: {duplicates}"
        except psycopg.OperationalError:
            pytest.skip("Database not available")

    async def test_all_clusters_have_labels(self):
        """All clusters should have non-empty labels."""
        from app.config import get_settings
        import psycopg

        settings = get_settings()
        try:
            async with await psycopg.AsyncConnection.connect(settings.database_url) as conn:
                async with conn.cursor() as cur:
                    await cur.execute("""
                        SELECT id, label
                        FROM clusters
                        WHERE label IS NULL OR label = ''
                    """)
                    empty_labels = await cur.fetchall()

                    assert len(empty_labels) == 0, f"Clusters without labels: {empty_labels}"
        except psycopg.OperationalError:
            pytest.skip("Database not available")

    async def test_low_quality_clusters_are_flagged(self):
        """Clusters with silhouette < 0.1 should be flagged."""
        from app.config import get_settings
        import psycopg

        settings = get_settings()
        try:
            async with await psycopg.AsyncConnection.connect(settings.database_url) as conn:
                async with conn.cursor() as cur:
                    # Check if required columns exist
                    await cur.execute("""
                        SELECT column_name FROM information_schema.columns
                        WHERE table_name = 'clusters' AND column_name = 'silhouette_score'
                    """)
                    if not await cur.fetchone():
                        pytest.skip("silhouette_score column not in schema")

                    await cur.execute("""
                        SELECT id, label, silhouette_score, quality_flag
                        FROM clusters
                        WHERE silhouette_score < 0.1
                          AND is_super_cluster = false
                          AND quality_flag IS NULL
                    """)
                    unflagged = await cur.fetchall()

                    assert len(unflagged) == 0, f"Low-quality clusters not flagged: {unflagged}"
        except psycopg.OperationalError:
            pytest.skip("Database not available")


@pytest.mark.asyncio
class TestCategoryClassification:
    """Tests for category-based tweet classification."""

    async def test_all_categories_have_tweets(self):
        """Each category should have at least some tweets assigned."""
        from app.config import get_settings
        import psycopg

        settings = get_settings()
        try:
            async with await psycopg.AsyncConnection.connect(settings.database_url) as conn:
                async with conn.cursor() as cur:
                    # Check if categories table exists
                    await cur.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables
                            WHERE table_name = 'categories'
                        )
                    """)
                    if not (await cur.fetchone())[0]:
                        pytest.skip("categories table not in schema")

                    await cur.execute("""
                        SELECT c.name, COUNT(tc.tweet_id) as cnt
                        FROM categories c
                        LEFT JOIN tweet_categories tc ON c.id = tc.category_id
                        GROUP BY c.id, c.name
                        HAVING COUNT(tc.tweet_id) = 0
                    """)
                    empty_cats = await cur.fetchall()

                    assert len(empty_cats) == 0, f"Categories with no tweets: {[c[0] for c in empty_cats]}"
        except psycopg.OperationalError:
            pytest.skip("Database not available")

    async def test_no_orphan_tweet_categories(self):
        """All tweet_categories entries should reference valid tweets."""
        from app.config import get_settings
        import psycopg

        settings = get_settings()
        try:
            async with await psycopg.AsyncConnection.connect(settings.database_url) as conn:
                async with conn.cursor() as cur:
                    # Check if tweet_categories table exists
                    await cur.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables
                            WHERE table_name = 'tweet_categories'
                        )
                    """)
                    if not (await cur.fetchone())[0]:
                        pytest.skip("tweet_categories table not in schema")

                    await cur.execute("""
                        SELECT tc.tweet_id
                        FROM tweet_categories tc
                        LEFT JOIN tweets t ON tc.tweet_id = t.id
                        WHERE t.id IS NULL
                        LIMIT 10
                    """)
                    orphans = await cur.fetchall()

                    assert len(orphans) == 0, f"Orphan tweet_categories found: {orphans}"
        except psycopg.OperationalError:
            pytest.skip("Database not available")

    async def test_subclusters_have_tweets(self):
        """Each subcluster should have tweets assigned."""
        from app.config import get_settings
        import psycopg

        settings = get_settings()
        try:
            async with await psycopg.AsyncConnection.connect(settings.database_url) as conn:
                async with conn.cursor() as cur:
                    # Check if subclusters table exists
                    await cur.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables
                            WHERE table_name = 'subclusters'
                        )
                    """)
                    if not (await cur.fetchone())[0]:
                        pytest.skip("subclusters table not in schema")

                    await cur.execute("""
                        SELECT s.label, s.tweet_count
                        FROM subclusters s
                        WHERE s.tweet_count = 0 OR s.tweet_count IS NULL
                    """)
                    empty_subclusters = await cur.fetchall()

                    assert len(empty_subclusters) == 0, f"Empty subclusters: {empty_subclusters}"
        except psycopg.OperationalError:
            pytest.skip("Database not available")

    async def test_subcluster_labels_unique_per_category(self):
        """Subcluster labels should be unique within each category."""
        from app.config import get_settings
        import psycopg

        settings = get_settings()
        try:
            async with await psycopg.AsyncConnection.connect(settings.database_url) as conn:
                async with conn.cursor() as cur:
                    # Check if subclusters table exists
                    await cur.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables
                            WHERE table_name = 'subclusters'
                        )
                    """)
                    if not (await cur.fetchone())[0]:
                        pytest.skip("subclusters table not in schema")

                    await cur.execute("""
                        SELECT category_id, label, COUNT(*) as cnt
                        FROM subclusters
                        GROUP BY category_id, label
                        HAVING COUNT(*) > 1
                    """)
                    duplicates = await cur.fetchall()

                    assert len(duplicates) == 0, f"Duplicate subcluster labels: {duplicates}"
        except psycopg.OperationalError:
            pytest.skip("Database not available")

    async def test_quality_tiers_are_valid(self):
        """All quality_tier values should be from expected set."""
        from app.config import get_settings
        import psycopg

        settings = get_settings()
        valid_tiers = {'ORIGINAL', 'REPLY_THREAD', 'LOW_VALUE', 'SUBSTANTIVE_RT', None}
        try:
            async with await psycopg.AsyncConnection.connect(settings.database_url) as conn:
                async with conn.cursor() as cur:
                    # Check if quality_tier column exists
                    await cur.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.columns
                            WHERE table_name = 'tweets' AND column_name = 'quality_tier'
                        )
                    """)
                    if not (await cur.fetchone())[0]:
                        pytest.skip("quality_tier column not in schema")

                    await cur.execute("""
                        SELECT DISTINCT quality_tier FROM tweets
                    """)
                    tiers = {row[0] for row in await cur.fetchall()}

                    invalid = tiers - valid_tiers
                    assert len(invalid) == 0, f"Invalid quality tiers: {invalid}"
        except psycopg.OperationalError:
            pytest.skip("Database not available")
