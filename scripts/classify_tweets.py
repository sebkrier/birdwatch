"""Classify tweets into categories using Claude API."""

import asyncio
import json
import time
from typing import Optional

import anthropic

from app.db.pool import init_pool, close_pool, get_connection


BATCH_SIZE = 20
RATE_LIMIT_DELAY = 0.5  # seconds between API calls


async def get_categories() -> dict[str, int]:
    """Fetch all categories and return name -> id mapping."""
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT id, name FROM categories ORDER BY id")
            rows = await cur.fetchall()
    return {row[1]: row[0] for row in rows}


async def get_tweets_to_classify() -> list[tuple[str, str]]:
    """Get all tweets that need classification."""
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("""
                SELECT t.id, t.full_text
                FROM tweets t
                WHERE t.quality_tier IN ('ORIGINAL', 'REPLY_THREAD')
                  AND NOT EXISTS (
                      SELECT 1 FROM tweet_categories tc WHERE tc.tweet_id = t.id
                  )
                ORDER BY t.created_at DESC
            """)
            return await cur.fetchall()


async def store_classifications(classifications: list[tuple[str, int]]) -> None:
    """Store tweet-category assignments in database."""
    if not classifications:
        return

    async with get_connection() as conn:
        async with conn.cursor() as cur:
            for tweet_id, category_id in classifications:
                await cur.execute("""
                    INSERT INTO tweet_categories (tweet_id, category_id)
                    VALUES (%s, %s)
                    ON CONFLICT (tweet_id, category_id) DO NOTHING
                """, (tweet_id, category_id))
            await conn.commit()


def classify_batch(
    client: anthropic.Anthropic,
    tweets: list[tuple[str, str]],
    category_names: list[str],
) -> dict[str, list[str]]:
    """Classify a batch of tweets using Claude."""

    # Format tweets for the prompt
    tweet_list = "\n".join([
        f"{i+1}. [{tweet_id}] {text[:300]}"
        for i, (tweet_id, text) in enumerate(tweets)
    ])

    category_list = "\n".join([f"- {name}" for name in category_names])

    prompt = f"""Classify each tweet into 1-2 of the following categories. Choose the most relevant categories for each tweet.

CATEGORIES:
{category_list}

TWEETS:
{tweet_list}

Respond with a JSON object mapping tweet IDs to their category names (1-2 categories per tweet).
Example: {{"1234567890": ["AI Safety & Alignment"], "0987654321": ["Economic Policy", "European Politics"]}}

Only use category names from the list above. Be precise and consistent."""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )

        # Parse JSON from response
        text = response.content[0].text

        # Extract JSON from markdown code blocks if present
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        return json.loads(text.strip())

    except Exception as e:
        print(f"  Error classifying batch: {e}")
        return {}


async def main():
    print("Initializing database connection...", flush=True)
    await init_pool()

    try:
        # Load API key
        with open(".env") as f:
            for line in f:
                if line.startswith("ANTHROPIC_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
                    break

        client = anthropic.Anthropic(api_key=api_key)

        # Get categories
        categories = await get_categories()
        category_names = list(categories.keys())
        print(f"Loaded {len(categories)} categories", flush=True)

        # Get tweets to classify
        tweets = await get_tweets_to_classify()
        print(f"Found {len(tweets)} tweets to classify", flush=True)

        if not tweets:
            print("No tweets to classify!")
            return

        # Process in batches
        total_batches = (len(tweets) + BATCH_SIZE - 1) // BATCH_SIZE
        classified_count = 0
        start_time = time.time()

        for i in range(0, len(tweets), BATCH_SIZE):
            batch = tweets[i:i + BATCH_SIZE]
            batch_num = i // BATCH_SIZE + 1

            print(f"Batch {batch_num}/{total_batches} ({len(batch)} tweets)...", end=" ", flush=True)

            # Classify batch
            results = classify_batch(client, batch, category_names)

            # Store results - validate tweet IDs against batch
            valid_tweet_ids = {t[0] for t in batch}
            classifications = []
            skipped = 0
            for tweet_id, cats in results.items():
                if tweet_id not in valid_tweet_ids:
                    skipped += 1
                    continue
                for cat_name in cats:
                    if cat_name in categories:
                        classifications.append((tweet_id, categories[cat_name]))

            await store_classifications(classifications)
            classified_count += len(results) - skipped

            if skipped:
                print(f"classified {len(results) - skipped} tweets ({skipped} invalid IDs skipped)")
            else:
                print(f"classified {len(results)} tweets")

            # Rate limiting
            if i + BATCH_SIZE < len(tweets):
                time.sleep(RATE_LIMIT_DELAY)

            # Progress update every 50 batches
            if batch_num % 50 == 0:
                elapsed = time.time() - start_time
                rate = classified_count / elapsed
                remaining = (len(tweets) - classified_count) / rate if rate > 0 else 0
                print(f"  Progress: {classified_count}/{len(tweets)} ({classified_count/len(tweets)*100:.1f}%) - ETA: {remaining/60:.1f} min")

        elapsed = time.time() - start_time
        print()
        print("=" * 60)
        print(f"CLASSIFICATION COMPLETE")
        print(f"  Tweets classified: {classified_count}")
        print(f"  Time elapsed: {elapsed/60:.1f} minutes")
        print(f"  Rate: {classified_count/elapsed:.1f} tweets/second")
        print("=" * 60)

    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
