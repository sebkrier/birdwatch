"""Embedding generation service using SentenceTransformers."""

from typing import Optional
from functools import lru_cache

# Global model instance (lazy loaded)
_model = None


def _get_model():
    """Get or initialize the embedding model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


async def get_embedding(text: str) -> Optional[list[float]]:
    """
    Generate an embedding for the given text.

    Returns a list of floats (384 dimensions for all-MiniLM-L6-v2).
    """
    if not text or not text.strip():
        return None

    try:
        model = _get_model()
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None


async def get_embeddings_batch(texts: list[str]) -> list[Optional[list[float]]]:
    """
    Generate embeddings for a batch of texts.

    More efficient than calling get_embedding multiple times.
    """
    if not texts:
        return []

    try:
        model = _get_model()
        # Filter out empty texts but keep track of indices
        valid_texts = [(i, t) for i, t in enumerate(texts) if t and t.strip()]

        if not valid_texts:
            return [None] * len(texts)

        indices, filtered_texts = zip(*valid_texts)
        embeddings = model.encode(list(filtered_texts), convert_to_numpy=True)

        # Build result list with None for empty texts
        result = [None] * len(texts)
        for i, emb in zip(indices, embeddings):
            result[i] = emb.tolist()

        return result
    except Exception as e:
        print(f"Error generating batch embeddings: {e}")
        return [None] * len(texts)
