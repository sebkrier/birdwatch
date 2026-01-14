"""Sub-cluster tweets within categories using UMAP + HDBSCAN."""

import asyncio
import json
import numpy as np
from collections import defaultdict

import anthropic
import hdbscan
import umap
from sklearn.feature_extraction.text import TfidfVectorizer

from app.db.pool import init_pool, close_pool, get_connection


MIN_CLUSTER_SIZE = 30  # Minimum tweets per sub-cluster
MIN_SAMPLES = 15
CLUSTER_SELECTION_EPSILON = 0.3
UMAP_NEIGHBORS = 15


def parse_pgvector(vec_str):
    """Parse pgvector string to list of floats."""
    if isinstance(vec_str, list):
        return vec_str
    return json.loads(vec_str)


async def get_category_tweets(category_id: int) -> list[tuple]:
    """Get tweets and embeddings for a category."""
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("""
                SELECT t.id, t.full_text, t.embedding
                FROM tweets t
                JOIN tweet_categories tc ON t.id = tc.tweet_id
                WHERE tc.category_id = %s
                  AND t.embedding IS NOT NULL
                ORDER BY t.id
            """, (category_id,))
            return await cur.fetchall()


def compute_tfidf_keywords(texts: list[str], labels: np.ndarray, top_n: int = 5) -> dict:
    """Extract distinctive TF-IDF keywords for each sub-cluster."""
    cluster_texts = defaultdict(list)
    for text, label in zip(texts, labels):
        if label >= 0:
            cluster_texts[label].append(text)

    if not cluster_texts:
        return {}

    cluster_docs = {cid: " ".join(texts) for cid, texts in cluster_texts.items()}
    doc_ids = list(cluster_docs.keys())
    doc_texts = [cluster_docs[cid] for cid in doc_ids]

    n_docs = len(doc_texts)
    min_df = 1 if n_docs < 5 else 2
    max_df = 1.0 if n_docs < 3 else 0.9

    vectorizer = TfidfVectorizer(
        max_features=2000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=max_df,
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(doc_texts)
        feature_names = vectorizer.get_feature_names_out()
    except ValueError:
        return {cid: [] for cid in doc_ids}

    keywords = {}
    for idx, cluster_id in enumerate(doc_ids):
        row = tfidf_matrix[idx].toarray().flatten()
        top_indices = row.argsort()[-top_n:][::-1]
        keywords[cluster_id] = [feature_names[i] for i in top_indices if row[i] > 0]

    return keywords


def generate_label_from_keywords(keywords: list[str]) -> str:
    """Generate a simple label from keywords."""
    if not keywords:
        return "Miscellaneous"
    # Take first 2-3 keywords and title case
    label_words = keywords[:3]
    return " ".join(w.title() for w in label_words)


async def generate_label_with_claude(
    client: anthropic.Anthropic,
    keywords: list[str],
    sample_tweets: list[str],
    category_name: str,
) -> str:
    """Generate a concise sub-cluster label using Claude."""
    if not keywords or not sample_tweets:
        return "Miscellaneous"

    prompt = f"""This is a sub-cluster within the "{category_name}" category. Generate a concise 2-4 word label for this specific sub-topic.

Keywords: {', '.join(keywords[:5])}

Sample tweets:
{chr(10).join(['- ' + t[:150] for t in sample_tweets[:3]])}

Respond with ONLY the 2-4 word label. Be specific to this sub-topic, not the general category."""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=20,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
    except Exception as e:
        print(f"    Claude API error: {e}")
        return generate_label_from_keywords(keywords)


async def subcluster_category(
    category_id: int,
    category_name: str,
    client: anthropic.Anthropic,
) -> dict:
    """Run sub-clustering for a single category."""
    print(f"\n  Processing: {category_name}")

    # Get tweets for this category
    rows = await get_category_tweets(category_id)

    if len(rows) < MIN_CLUSTER_SIZE * 2:
        print(f"    Too few tweets ({len(rows)}) for sub-clustering, skipping")
        return {"category": category_name, "n_subclusters": 0, "skipped": True}

    tweet_ids = [r[0] for r in rows]
    texts = [r[1] for r in rows]
    embeddings = np.array([parse_pgvector(r[2]) for r in rows])

    print(f"    {len(rows)} tweets, running UMAP...", end=" ", flush=True)

    # UMAP reduction
    n_components = min(20, len(rows) // 10)  # Adaptive components
    n_neighbors = min(UMAP_NEIGHBORS, len(rows) // 5)

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
        low_memory=True,
    )
    reduced = reducer.fit_transform(embeddings)
    print("HDBSCAN...", end=" ", flush=True)

    # HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
        cluster_selection_epsilon=CLUSTER_SELECTION_EPSILON,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(reduced)

    unique_labels = set(labels)
    n_clusters = len([l for l in unique_labels if l >= 0])
    n_noise = (labels == -1).sum()
    print(f"found {n_clusters} sub-clusters ({n_noise} noise)")

    if n_clusters == 0:
        return {"category": category_name, "n_subclusters": 0, "n_noise": n_noise}

    # Extract keywords
    keywords = compute_tfidf_keywords(texts, labels, top_n=5)

    # Generate labels and store sub-clusters
    subclusters = []
    for cluster_idx in sorted(set(l for l in labels if l >= 0)):
        mask = labels == cluster_idx
        cluster_tweet_ids = [tid for tid, m in zip(tweet_ids, mask) if m]
        cluster_texts = [t for t, m in zip(texts, mask) if m]
        cluster_keywords = keywords.get(cluster_idx, [])

        # Generate label with Claude
        label = await generate_label_with_claude(
            client, cluster_keywords, cluster_texts[:5], category_name
        )

        subclusters.append({
            "label": label,
            "tweet_ids": cluster_tweet_ids,
            "keywords": cluster_keywords,
            "count": len(cluster_tweet_ids),
        })
        print(f"    - {label}: {len(cluster_tweet_ids)} tweets")

    # Store in database
    await store_subclusters(category_id, subclusters)

    return {
        "category": category_name,
        "n_subclusters": n_clusters,
        "n_noise": n_noise,
        "subclusters": [(s["label"], s["count"]) for s in subclusters],
    }


async def store_subclusters(category_id: int, subclusters: list[dict]) -> None:
    """Store sub-clusters in the database."""
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            # Delete existing sub-clusters for this category
            await cur.execute("""
                DELETE FROM subclusters WHERE category_id = %s
            """, (category_id,))

            for subcluster in subclusters:
                # Insert sub-cluster
                await cur.execute("""
                    INSERT INTO subclusters (category_id, label, keywords, tweet_count)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id
                """, (category_id, subcluster["label"], subcluster["keywords"], subcluster["count"]))

                subcluster_id = (await cur.fetchone())[0]

                # Link tweets to sub-cluster
                for tweet_id in subcluster["tweet_ids"]:
                    await cur.execute("""
                        UPDATE tweet_categories
                        SET subcluster_id = %s
                        WHERE tweet_id = %s AND category_id = %s
                    """, (subcluster_id, tweet_id, category_id))

            await conn.commit()


async def create_subclusters_table() -> None:
    """Create subclusters table if it doesn't exist."""
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            # Create subclusters table
            await cur.execute("""
                CREATE TABLE IF NOT EXISTS subclusters (
                    id SERIAL PRIMARY KEY,
                    category_id INTEGER REFERENCES categories(id) ON DELETE CASCADE,
                    label TEXT NOT NULL,
                    keywords TEXT[],
                    tweet_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)

            # Add subcluster_id to tweet_categories if not exists
            await cur.execute("""
                ALTER TABLE tweet_categories
                ADD COLUMN IF NOT EXISTS subcluster_id INTEGER REFERENCES subclusters(id) ON DELETE SET NULL
            """)

            await conn.commit()
            print("Database schema updated for subclusters")


async def main():
    print("Initializing database connection...", flush=True)
    await init_pool()

    try:
        # Load API key
        api_key = None
        with open(".env") as f:
            for line in f:
                if line.startswith("ANTHROPIC_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
                    break

        client = anthropic.Anthropic(api_key=api_key)

        # Create subclusters table
        await create_subclusters_table()

        # Get all categories
        async with get_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    SELECT c.id, c.name, COUNT(tc.tweet_id) as tweet_count
                    FROM categories c
                    LEFT JOIN tweet_categories tc ON c.id = tc.category_id
                    GROUP BY c.id, c.name
                    ORDER BY tweet_count DESC
                """)
                categories = await cur.fetchall()

        print(f"\nFound {len(categories)} categories")
        print("=" * 60)

        results = []
        for cat_id, cat_name, tweet_count in categories:
            result = await subcluster_category(cat_id, cat_name, client)
            results.append(result)

        # Summary
        print("\n" + "=" * 60)
        print("SUB-CLUSTERING COMPLETE")
        print("=" * 60)

        total_subclusters = sum(r.get("n_subclusters", 0) for r in results)
        categories_with_subclusters = sum(1 for r in results if r.get("n_subclusters", 0) > 0)

        print(f"  Categories processed: {len(categories)}")
        print(f"  Categories with sub-clusters: {categories_with_subclusters}")
        print(f"  Total sub-clusters created: {total_subclusters}")
        print("=" * 60)

    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
