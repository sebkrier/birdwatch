"""Natural language Q&A service for tweets using LLM."""

import os
from typing import Optional
import json

from app.db.pool import get_cursor
from app.db import queries
from app.services.embeddings import get_embedding


# Will be set via environment variable or passed in
ANTHROPIC_API_KEY: Optional[str] = None


def set_api_key(key: str):
    """Set the Anthropic API key."""
    global ANTHROPIC_API_KEY
    ANTHROPIC_API_KEY = key
    os.environ["ANTHROPIC_API_KEY"] = key


async def ask_about_tweets(
    question: str,
    *,
    max_tweets: int = 50,
    include_retweets: bool = True,
    include_replies: bool = True,
) -> dict:
    """
    Answer a natural language question about the user's tweets.

    Returns:
        {
            "answer": str,  # The LLM-generated answer
            "tweets": list,  # Source tweets used to answer
            "tweet_count": int,  # Number of tweets found
        }
    """
    # Step 1: Generate embedding for the question to find relevant tweets
    question_embedding = await get_embedding(question)

    if not question_embedding:
        return {
            "answer": "Sorry, I couldn't process your question. Please try again.",
            "tweets": [],
            "tweet_count": 0,
        }

    # Step 2: Find relevant tweets using semantic search
    async with get_cursor() as cur:
        tweets = await queries.search_tweets_semantic(
            cur,
            question_embedding,
            limit=max_tweets,
            include_retweets=include_retweets,
            include_replies=include_replies,
        )

    if not tweets:
        return {
            "answer": "I couldn't find any tweets relevant to your question.",
            "tweets": [],
            "tweet_count": 0,
        }

    # Step 3: Format tweets for the LLM
    tweets_context = format_tweets_for_llm(tweets)

    # Step 4: Generate answer using Gemini
    answer = await generate_answer(question, tweets_context, len(tweets))

    return {
        "answer": answer,
        "tweets": [dict(t) for t in tweets[:10]],  # Return top 10 as sources
        "tweet_count": len(tweets),
    }


def format_tweets_for_llm(tweets: list) -> str:
    """Format tweets into a string context for the LLM."""
    lines = []
    for i, tweet in enumerate(tweets, 1):
        date = tweet['created_at'].strftime('%Y-%m-%d') if tweet.get('created_at') else 'Unknown date'
        likes = tweet.get('favorite_count', 0) or 0
        rts = tweet.get('retweet_count', 0) or 0
        text = tweet.get('full_text', '')

        is_rt = "RT" if tweet.get('is_retweet') else ""
        is_reply = "Reply" if tweet.get('is_reply') else ""
        tweet_type = f" [{is_rt}{is_reply}]" if is_rt or is_reply else ""

        lines.append(f"Tweet {i} ({date}, {likes} likes, {rts} RTs){tweet_type}:\n{text}\n")

    return "\n".join(lines)


async def generate_answer(question: str, tweets_context: str, tweet_count: int) -> str:
    """Generate an answer using Claude."""
    import litellm

    api_key = ANTHROPIC_API_KEY or os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        return "Anthropic API key not configured. Please set your API key first."

    prompt = f"""You are a helpful assistant analyzing a user's Twitter/X archive.
The user has asked a question about their tweets. Below are the most relevant tweets from their archive.

USER'S QUESTION: {question}

RELEVANT TWEETS ({tweet_count} found):
{tweets_context}

INSTRUCTIONS:
- Answer the user's question based on the tweets provided
- IMPORTANT: When referencing specific tweets, use numbered citations like [1], [2], [3] etc. matching the tweet numbers above
- If the question asks for "most popular" or "best", consider likes and retweets
- If the question asks about whether something was tweeted, answer yes/no and provide examples with citations
- If you can't find relevant information in the tweets, say so
- Keep your answer concise but informative
- Use a friendly, conversational tone

YOUR ANSWER:"""

    try:
        response = await litellm.acompletion(
            model="claude-haiku-4-5-20251001",
            messages=[{"role": "user", "content": prompt}],
            api_key=api_key,
            max_tokens=1000,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating answer: {str(e)}"


async def smart_search(
    question: str,
    *,
    max_tweets: int = 100,
) -> dict:
    """
    Enhanced search that tries to understand the question intent.

    Handles queries like:
    - "most popular tweets about X" → semantic search + sort by likes
    - "oldest tweets about X" → semantic search + sort by date
    - "did I tweet about X" → semantic search + yes/no answer
    """
    import litellm

    api_key = ANTHROPIC_API_KEY or os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        # Fall back to basic semantic search
        return await ask_about_tweets(question, max_tweets=max_tweets)

    # First, use LLM to parse the question intent
    parse_prompt = f"""Analyze this question about a Twitter archive and extract search parameters.

Question: {question}

Return a JSON object with:
- "search_terms": keywords to search for (string, can be empty if question is about date/popularity/length without specific topic)
- "sort_by": one of "relevance", "likes", "retweets", "date_desc", "date_asc", "longest"
- "include_retweets": true/false
- "include_replies": true/false
- "limit": number of results needed (default 50)
- "question_type": one of "find_examples", "yes_no", "count", "analyze", "summarize", "top_tweets"
- "year": year filter if mentioned (e.g., 2019), or null
- "date_from": start date if mentioned (YYYY-MM-DD format), or null
- "date_to": end date if mentioned (YYYY-MM-DD format), or null

Return ONLY valid JSON, no explanation."""

    try:
        parse_response = await litellm.acompletion(
            model="claude-haiku-4-5-20251001",
            messages=[{"role": "user", "content": parse_prompt}],
            api_key=api_key,
            max_tokens=200,
        )

        # Parse the JSON response
        intent_text = parse_response.choices[0].message.content
        # Clean up potential markdown formatting
        intent_text = intent_text.strip()
        if intent_text.startswith("```"):
            intent_text = intent_text.split("```")[1]
            if intent_text.startswith("json"):
                intent_text = intent_text[4:]
        intent = json.loads(intent_text)

    except Exception as e:
        # Fall back to basic search if parsing fails
        print(f"Intent parsing failed: {e}")
        return await ask_about_tweets(question, max_tweets=max_tweets)

    # Execute search based on parsed intent
    search_terms = intent.get("search_terms", "") or ""
    search_terms = search_terms.strip()
    sort_by = intent.get("sort_by", "relevance")
    include_retweets = intent.get("include_retweets", True)
    include_replies = intent.get("include_replies", True)
    question_type = intent.get("question_type", "find_examples")

    # For "longest" queries, exclude replies (they have @mentions that inflate length)
    if sort_by == "longest":
        include_replies = False
        include_retweets = False
        print("[ask] Longest query - excluding replies and retweets for meaningful results")

    # Handle date filters
    date_from = None
    date_to = None
    year = intent.get("year")
    if year:
        date_from = f"{year}-01-01"
        date_to = f"{year}-12-31"
    if intent.get("date_from"):
        date_from = intent.get("date_from")
    if intent.get("date_to"):
        date_to = intent.get("date_to")

    print(f"[ask] Parsed intent: search_terms='{search_terms}', sort_by={sort_by}, year={year}, date_from={date_from}, date_to={date_to}")

    async with get_cursor() as cur:
        all_tweets = {}  # Use dict to dedupe by tweet ID

        # If no search terms, query tweets directly by date/filters
        if not search_terms:
            print(f"[ask] No search terms - querying tweets by date range and filters")
            tweets = await queries.get_tweets_by_filters(
                cur,
                limit=500,
                include_retweets=include_retweets,
                include_replies=include_replies,
                date_from=date_from,
                date_to=date_to,
                sort_by=sort_by,
            )
            print(f"[ask] Got {len(tweets)} tweets by filters")
        else:
            # Use BOTH full-text and semantic search to get comprehensive results
            # Full-text finds exact keyword matches across ALL tweets
            # Semantic finds related content even without exact words

            # 1. Full-text search - finds exact keyword matches across ALL tweets
            try:
                fulltext_tweets, fulltext_total = await queries.search_tweets_fulltext_with_filters(
                    cur,
                    search_terms,
                    limit=5000,  # High limit to catch all keyword matches
                    include_retweets=include_retweets,
                    include_replies=include_replies,
                    date_from=date_from,
                    date_to=date_to,
                )
                print(f"[ask] Full-text search for '{search_terms}': {fulltext_total} total matches")
                for tweet in fulltext_tweets:
                    all_tweets[tweet['id']] = tweet
            except Exception as e:
                print(f"[ask] Full-text search failed: {e}")

            # 2. Semantic search - finds related content even without exact words
            embedding = await get_embedding(search_terms)
            if embedding:
                semantic_tweets = await queries.search_tweets_semantic(
                    cur,
                    embedding,
                    limit=200,  # Get more semantic matches
                    include_retweets=include_retweets,
                    include_replies=include_replies,
                    date_from=date_from,
                    date_to=date_to,
                )
                print(f"[ask] Semantic search for '{search_terms}': {len(semantic_tweets)} results")
                for tweet in semantic_tweets:
                    if tweet['id'] not in all_tweets:
                        all_tweets[tweet['id']] = tweet

            # Combine results
            tweets = list(all_tweets.values())
            print(f"[ask] Combined unique tweets: {len(tweets)}")

    # Apply sorting based on intent
    if sort_by == "likes":
        tweets = sorted(tweets, key=lambda t: t.get('favorite_count', 0) or 0, reverse=True)
    elif sort_by == "retweets":
        tweets = sorted(tweets, key=lambda t: t.get('retweet_count', 0) or 0, reverse=True)
    elif sort_by == "date_desc":
        tweets = sorted(tweets, key=lambda t: t.get('created_at'), reverse=True)
    elif sort_by == "date_asc":
        tweets = sorted(tweets, key=lambda t: t.get('created_at'))

    # Generate answer
    if not tweets:
        if search_terms:
            msg = f"I couldn't find any tweets about '{search_terms}' in your archive."
        elif year:
            msg = f"I couldn't find any tweets from {year} in your archive."
        else:
            msg = "I couldn't find any tweets matching your criteria."
        return {
            "answer": msg,
            "tweets": [],
            "tweet_count": 0,
            "intent": intent,
        }

    tweets_context = format_tweets_for_llm(tweets[:30])  # Limit context size
    answer = await generate_answer(question, tweets_context, len(tweets))

    return {
        "answer": answer,
        "tweets": [dict(t) for t in tweets[:10]],
        "tweet_count": len(tweets),
        "intent": intent,
    }
