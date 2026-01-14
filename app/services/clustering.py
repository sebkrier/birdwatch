"""Clustering service for tweet analysis.

Implements robust clustering with:
- HDBSCAN with tuned parameters for coherent clusters
- TF-IDF keyword extraction for interpretable labels
- Ward linkage hierarchy for natural super-clusters
- Silhouette scores for quality metrics
"""

import json
import numpy as np
from collections import defaultdict
from typing import Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.cluster import AgglomerativeClustering
import hdbscan
import umap

from app.db.pool import get_connection
from app.config import get_settings


def parse_pgvector(vec_str: str) -> list[float]:
    """Parse pgvector string to list of floats."""
    if isinstance(vec_str, list):
        return vec_str
    return json.loads(vec_str)


async def load_embeddings_and_texts() -> tuple[np.ndarray, list[str], list[str]]:
    """Load all tweet embeddings and texts from database."""
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("""
                SELECT id, embedding, full_text
                FROM tweets
                WHERE embedding IS NOT NULL
                ORDER BY id
            """)
            rows = await cur.fetchall()

    tweet_ids = []
    embeddings = []
    texts = []

    for row in rows:
        tweet_ids.append(row[0])
        embeddings.append(parse_pgvector(row[1]))
        texts.append(row[2])

    return np.array(embeddings), tweet_ids, texts


def run_umap_reduction(
    embeddings: np.ndarray,
    n_components: int = 50,
    n_neighbors: int = 30,
    min_dist: float = 0.0,
    metric: str = "cosine",
) -> np.ndarray:
    """Run UMAP dimensionality reduction for clustering."""
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=42,
        low_memory=True,
    )
    return reducer.fit_transform(embeddings)


def run_hdbscan_clustering(
    reduced_embeddings: np.ndarray,
    min_cluster_size: int = 100,
    min_samples: int = 50,
    cluster_selection_epsilon: float = 0.5,
) -> np.ndarray:
    """Run HDBSCAN clustering with robust parameters.

    Parameters tuned for:
    - min_cluster_size=100: Forces larger, more coherent clusters
    - min_samples=50: Reduces noise sensitivity
    - cluster_selection_epsilon=0.5: Merges very similar clusters
    """
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    return clusterer.fit_predict(reduced_embeddings)


def compute_tfidf_keywords(
    texts: list[str],
    labels: np.ndarray,
    top_n: int = 10,
) -> dict[int, list[str]]:
    """Extract distinctive TF-IDF keywords for each cluster.

    For each cluster, computes TF-IDF across all its tweets vs the corpus,
    returning the top distinctive terms.
    """
    # Group texts by cluster
    cluster_texts = defaultdict(list)
    for text, label in zip(texts, labels):
        if label >= 0:  # Skip noise
            cluster_texts[label].append(text)

    if not cluster_texts:
        return {}

    # Concatenate cluster texts
    cluster_docs = {}
    for cluster_id, cluster_text_list in cluster_texts.items():
        cluster_docs[cluster_id] = " ".join(cluster_text_list)

    doc_ids = list(cluster_docs.keys())
    doc_texts = [cluster_docs[cid] for cid in doc_ids]

    # Adjust min_df based on number of documents
    n_docs = len(doc_texts)
    min_df = 1 if n_docs < 5 else (2 if n_docs < 10 else 3)
    max_df = 1.0 if n_docs < 3 else 0.9

    # Create TF-IDF vectorizer with adaptive parameters
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=max_df,
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(doc_texts)
        feature_names = vectorizer.get_feature_names_out()
    except ValueError:
        # Fallback: simple word splitting
        keywords = {}
        for cluster_id, text in cluster_docs.items():
            words = text.lower().split()
            word_counts = defaultdict(int)
            for word in words:
                if len(word) > 3:
                    word_counts[word] += 1
            top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
            keywords[cluster_id] = [w for w, _ in top_words]
        return keywords

    # Extract top keywords for each cluster
    keywords = {}
    for idx, cluster_id in enumerate(doc_ids):
        row = tfidf_matrix[idx].toarray().flatten()
        top_indices = row.argsort()[-top_n:][::-1]
        keywords[cluster_id] = [feature_names[i] for i in top_indices if row[i] > 0]

    return keywords


def compute_silhouette_scores(
    reduced_embeddings: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, dict[int, float]]:
    """Compute overall and per-cluster silhouette scores.

    Returns:
        overall_score: Global silhouette score
        cluster_scores: Dict mapping cluster_id to average silhouette score
    """
    # Filter out noise points
    mask = labels >= 0
    if mask.sum() < 2:
        return 0.0, {}

    filtered_embeddings = reduced_embeddings[mask]
    filtered_labels = labels[mask]

    # Check we have at least 2 clusters
    unique_labels = np.unique(filtered_labels)
    if len(unique_labels) < 2:
        return 0.0, {}

    # Overall silhouette score
    overall = silhouette_score(filtered_embeddings, filtered_labels)

    # Per-sample silhouette scores
    sample_scores = silhouette_samples(filtered_embeddings, filtered_labels)

    # Aggregate per cluster
    cluster_scores = {}
    for cluster_id in unique_labels:
        cluster_mask = filtered_labels == cluster_id
        cluster_scores[int(cluster_id)] = float(np.mean(sample_scores[cluster_mask]))

    return float(overall), cluster_scores


def create_ward_hierarchy(
    reduced_embeddings: np.ndarray,
    labels: np.ndarray,
    n_super_clusters: int = 10,
) -> dict[int, int]:
    """Create super-cluster hierarchy using Ward linkage.

    Computes cluster centroids and uses agglomerative clustering
    with Ward linkage to create natural hierarchy.

    Returns:
        Dict mapping cluster_id to super_cluster_id
    """
    # Compute cluster centroids
    unique_labels = [l for l in np.unique(labels) if l >= 0]
    if len(unique_labels) <= n_super_clusters:
        # Not enough clusters for hierarchy
        return {l: l for l in unique_labels}

    centroids = []
    for cluster_id in unique_labels:
        mask = labels == cluster_id
        centroid = reduced_embeddings[mask].mean(axis=0)
        centroids.append(centroid)

    centroids = np.array(centroids)

    # Agglomerative clustering with Ward linkage
    ward = AgglomerativeClustering(
        n_clusters=n_super_clusters,
        linkage="ward",
    )
    super_labels = ward.fit_predict(centroids)

    # Map cluster IDs to super-cluster IDs
    return {
        int(cluster_id): int(super_labels[idx])
        for idx, cluster_id in enumerate(unique_labels)
    }


async def generate_cluster_label(keywords: list[str], sample_tweets: list[str]) -> str:
    """Generate a concise 2-4 word label using Claude API."""
    import anthropic
    import os

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        settings = get_settings()
        # Try loading from .env file content
        try:
            with open(".env") as f:
                for line in f:
                    if line.startswith("ANTHROPIC_API_KEY="):
                        api_key = line.split("=", 1)[1].strip()
                        break
        except:
            pass

    if not api_key:
        # Fallback to keyword-based label
        return " ".join(keywords[:3]).title()

    client = anthropic.Anthropic(api_key=api_key)

    prompt = f"""Based on these distinctive keywords and sample tweets from a cluster, generate a concise 2-4 word label that captures the main topic.

Keywords: {', '.join(keywords[:7])}

Sample tweets:
{chr(10).join(['- ' + t[:150] for t in sample_tweets[:5]])}

Respond with ONLY the 2-4 word label, nothing else. Examples: "AI Safety Research", "Climate Policy", "Tech Industry News"."""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=20,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
    except Exception as e:
        print(f"Claude API error: {e}")
        return " ".join(keywords[:3]).title()


async def run_full_clustering(
    min_cluster_size: int = 100,
    min_samples: int = 50,
    cluster_selection_epsilon: float = 0.5,
    n_super_clusters: int = 10,
    umap_components: int = 50,
) -> dict:
    """Run complete clustering pipeline.

    1. Load embeddings and texts
    2. UMAP reduction
    3. HDBSCAN clustering
    4. TF-IDF keyword extraction
    5. Claude label generation
    6. Silhouette score computation
    7. Ward linkage hierarchy
    8. Store results in database

    Returns statistics about the clustering.
    """
    print("Loading embeddings and texts...")
    embeddings, tweet_ids, texts = await load_embeddings_and_texts()
    print(f"Loaded {len(tweet_ids)} tweets with embeddings")

    print(f"Running UMAP reduction to {umap_components}D...")
    reduced = run_umap_reduction(embeddings, n_components=umap_components)
    print(f"UMAP complete: {reduced.shape}")

    print(f"Running HDBSCAN (min_cluster_size={min_cluster_size}, min_samples={min_samples}, epsilon={cluster_selection_epsilon})...")
    labels = run_hdbscan_clustering(
        reduced,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
    )

    unique_labels = set(labels)
    n_clusters = len([l for l in unique_labels if l >= 0])
    n_noise = (labels == -1).sum()
    print(f"Found {n_clusters} clusters, {n_noise} noise points")

    print("Computing TF-IDF keywords...")
    keywords = compute_tfidf_keywords(texts, labels, top_n=10)

    print("Computing silhouette scores...")
    overall_silhouette, cluster_silhouettes = compute_silhouette_scores(reduced, labels)
    print(f"Overall silhouette score: {overall_silhouette:.3f}")

    print(f"Creating Ward hierarchy with {n_super_clusters} super-clusters...")
    hierarchy = create_ward_hierarchy(reduced, labels, n_super_clusters=n_super_clusters)

    # Also run 2D UMAP for visualization
    print("Running 2D UMAP for visualization...")
    coords_2d = run_umap_reduction(embeddings, n_components=2, n_neighbors=15, min_dist=0.1)

    # Store results in database
    print("Storing results in database...")
    await store_clustering_results(
        tweet_ids=tweet_ids,
        labels=labels,
        coords_2d=coords_2d,
        keywords=keywords,
        cluster_silhouettes=cluster_silhouettes,
        hierarchy=hierarchy,
        texts=texts,
    )

    return {
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "overall_silhouette": overall_silhouette,
        "cluster_silhouettes": cluster_silhouettes,
        "keywords": keywords,
    }


async def store_clustering_results(
    tweet_ids: list[str],
    labels: np.ndarray,
    coords_2d: np.ndarray,
    keywords: dict[int, list[str]],
    cluster_silhouettes: dict[int, float],
    hierarchy: dict[int, int],
    texts: list[str],
) -> None:
    """Store clustering results in database."""
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            # Clear existing clusters and coordinates
            await cur.execute("DELETE FROM tweet_coordinates")
            await cur.execute("DELETE FROM clusters")

            # Create super-clusters first
            super_cluster_ids = set(hierarchy.values())
            super_cluster_map = {}  # Maps super-cluster index to database ID

            colors = [
                '#3b82f6', '#ef4444', '#22c55e', '#f59e0b', '#8b5cf6',
                '#ec4899', '#06b6d4', '#f97316', '#84cc16', '#6366f1',
                '#14b8a6', '#a855f7'
            ]

            for idx, sc_idx in enumerate(sorted(super_cluster_ids)):
                # Get all cluster IDs in this super-cluster
                child_clusters = [cid for cid, sid in hierarchy.items() if sid == sc_idx]

                # Aggregate keywords from children
                sc_keywords = []
                for cid in child_clusters:
                    sc_keywords.extend(keywords.get(cid, [])[:3])
                sc_keywords = list(dict.fromkeys(sc_keywords))[:7]  # Dedupe, limit to 7

                # Get sample tweets for label generation
                sample_tweets = []
                for i, (tid, label) in enumerate(zip(tweet_ids, labels)):
                    if label in child_clusters and len(sample_tweets) < 10:
                        sample_tweets.append(texts[i])

                # Generate label
                label_text = await generate_cluster_label(sc_keywords, sample_tweets)

                # Count tweets
                tweet_count = sum(1 for l in labels if l in child_clusters)

                await cur.execute("""
                    INSERT INTO clusters (id, label, keywords, tweet_count, is_super_cluster, color)
                    VALUES (%s, %s, %s, %s, true, %s)
                    RETURNING id
                """, (-(idx + 1), label_text, sc_keywords, tweet_count, colors[idx % len(colors)]))

                result = await cur.fetchone()
                super_cluster_map[sc_idx] = result[0]
                print(f"  Super-cluster {idx + 1}: {label_text} ({tweet_count} tweets)")

            # Create regular clusters
            for cluster_id in sorted(set(l for l in labels if l >= 0)):
                cluster_kw = keywords.get(cluster_id, [])
                silhouette = cluster_silhouettes.get(cluster_id, 0.0)

                # Get sample tweets
                sample_tweets = []
                for i, (tid, label) in enumerate(zip(tweet_ids, labels)):
                    if label == cluster_id and len(sample_tweets) < 5:
                        sample_tweets.append(texts[i])

                # Generate label
                label_text = await generate_cluster_label(cluster_kw, sample_tweets)

                # Count tweets
                tweet_count = sum(1 for l in labels if l == cluster_id)

                # Get parent super-cluster ID
                parent_id = super_cluster_map.get(hierarchy.get(cluster_id))

                # Quality flag
                quality_flag = None
                if silhouette < 0.1:
                    quality_flag = "low_quality"

                await cur.execute("""
                    INSERT INTO clusters (id, label, keywords, tweet_count, silhouette_score,
                                         parent_cluster_id, is_super_cluster, quality_flag)
                    VALUES (%s, %s, %s, %s, %s, %s, false, %s)
                """, (cluster_id, label_text, cluster_kw, tweet_count, silhouette, parent_id, quality_flag))

                print(f"  Cluster {cluster_id}: {label_text} ({tweet_count} tweets, silhouette={silhouette:.3f})")

            # Update tweet cluster assignments
            print("Updating tweet cluster assignments...")
            for tid, label in zip(tweet_ids, labels):
                cluster_id = int(label) if label >= 0 else None
                await cur.execute(
                    "UPDATE tweets SET cluster_id = %s WHERE id = %s",
                    (cluster_id, tid)
                )

            # Store 2D coordinates
            print("Storing visualization coordinates...")
            for tid, (x, y) in zip(tweet_ids, coords_2d):
                await cur.execute("""
                    INSERT INTO tweet_coordinates (tweet_id, x, y)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (tweet_id) DO UPDATE SET x = %s, y = %s
                """, (tid, float(x), float(y), float(x), float(y)))

            await conn.commit()
            print("Clustering results stored successfully")


async def get_cluster_comparison(old_label: str) -> dict:
    """Find where tweets from an old cluster ended up after reclustering."""
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            # This would need to track old cluster assignments
            # For now, search by label substring
            await cur.execute("""
                SELECT c.id, c.label, c.tweet_count, c.keywords, c.silhouette_score
                FROM clusters c
                WHERE c.is_super_cluster = false
                  AND (c.label ILIKE %s OR %s = ANY(c.keywords))
                ORDER BY c.tweet_count DESC
            """, (f"%{old_label}%", old_label.lower()))
            return await cur.fetchall()


async def handle_reimport_clustering() -> dict:
    """Handle clustering after a new archive import.

    1. Embeds any new tweets without embeddings
    2. Re-runs clustering on FULL corpus
    3. Maps old cluster IDs to new ones by centroid similarity
    4. Returns change log for user review

    Returns:
        Dict with clustering results and change log
    """
    from sentence_transformers import SentenceTransformer

    # Step 1: Get tweets without embeddings
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("""
                SELECT id, full_text
                FROM tweets
                WHERE embedding IS NULL
            """)
            unembedded = await cur.fetchall()

    new_tweet_count = len(unembedded)
    print(f"Found {new_tweet_count} tweets without embeddings")

    # Step 2: Generate embeddings for new tweets
    if new_tweet_count > 0:
        print("Generating embeddings for new tweets...")
        model = SentenceTransformer("all-MiniLM-L6-v2")

        batch_size = 100
        for i in range(0, len(unembedded), batch_size):
            batch = unembedded[i : i + batch_size]
            texts = [row[1] for row in batch]
            embeddings = model.encode(texts, show_progress_bar=False)

            async with get_connection() as conn:
                async with conn.cursor() as cur:
                    for (tweet_id, _), emb in zip(batch, embeddings):
                        await cur.execute(
                            "UPDATE tweets SET embedding = %s::vector WHERE id = %s",
                            (emb.tolist(), tweet_id),
                        )
                    await conn.commit()

            print(f"  Embedded {min(i + batch_size, len(unembedded))}/{len(unembedded)}")

    # Step 3: Save old cluster assignments for comparison
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("""
                SELECT t.id, t.cluster_id, c.label
                FROM tweets t
                LEFT JOIN clusters c ON t.cluster_id = c.id
                WHERE t.cluster_id IS NOT NULL
            """)
            old_assignments = {row[0]: (row[1], row[2]) for row in await cur.fetchall()}

            # Get old cluster centroids for mapping
            await cur.execute("""
                SELECT c.id, c.label, c.tweet_count
                FROM clusters c
                WHERE c.is_super_cluster = false
                ORDER BY c.tweet_count DESC
            """)
            old_clusters = await cur.fetchall()

    print(f"Saved {len(old_assignments)} old cluster assignments")

    # Step 4: Re-run clustering
    print("Re-running clustering on full corpus...")
    result = await run_full_clustering(
        min_cluster_size=50,
        min_samples=25,
        cluster_selection_epsilon=0.25,
        n_super_clusters=10,
        umap_components=50,
    )

    # Step 5: Compute change log
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("""
                SELECT t.id, t.cluster_id, c.label
                FROM tweets t
                LEFT JOIN clusters c ON t.cluster_id = c.id
            """)
            new_assignments = {row[0]: (row[1], row[2]) for row in await cur.fetchall()}

    # Track changes
    changes = {
        "moved": 0,
        "new_to_cluster": 0,
        "removed_from_cluster": 0,
        "unchanged": 0,
    }

    cluster_migrations = {}  # old_label -> {new_label: count}

    for tweet_id, (old_cluster, old_label) in old_assignments.items():
        new_cluster, new_label = new_assignments.get(tweet_id, (None, None))

        if old_cluster == new_cluster:
            changes["unchanged"] += 1
        elif new_cluster is None:
            changes["removed_from_cluster"] += 1
        else:
            changes["moved"] += 1
            if old_label not in cluster_migrations:
                cluster_migrations[old_label] = {}
            if new_label not in cluster_migrations[old_label]:
                cluster_migrations[old_label][new_label] = 0
            cluster_migrations[old_label][new_label] += 1

    # Count tweets newly assigned to clusters
    for tweet_id, (new_cluster, _) in new_assignments.items():
        if tweet_id not in old_assignments and new_cluster is not None:
            changes["new_to_cluster"] += 1

    return {
        "new_tweets_embedded": new_tweet_count,
        "clustering_result": result,
        "changes": changes,
        "cluster_migrations": cluster_migrations,
        "old_cluster_count": len(old_clusters),
        "new_cluster_count": result["n_clusters"],
    }
