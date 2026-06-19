"""
ResearchMind AI – Embedding Service
Uses sentence-transformers locally — completely free, no API key required.
Model: all-MiniLM-L6-v2 (~90 MB download on first use, then cached)
"""

from typing import List, Union
import numpy as np

from core.config import EMBEDDING_MODEL


# Cache the model at module level (loaded once per process)
_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


class EmbeddingService:
    """Wraps sentence-transformers for batch and single-query embedding."""

    def __init__(self):
        self.model = _get_model()

    # ── Public API ────────────────────────────────────────────────────────────

    def embed_texts(
        self, texts: List[str], batch_size: int = 64, show_progress: bool = False
    ) -> List[List[float]]:
        """Generate normalized embeddings for a batch of texts."""
        if not texts:
            return []
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """Generate a single query embedding (normalized)."""
        emb = self.model.encode(query, normalize_embeddings=True, convert_to_numpy=True)
        return emb.tolist()

    def cosine_similarity(
        self, a: List[float], b: List[float]
    ) -> float:
        """Compute cosine similarity (works on normalized vectors: just dot product)."""
        va, vb = np.array(a), np.array(b)
        return float(np.dot(va, vb))

    def batch_cosine_similarity(
        self, query_emb: List[float], corpus_embs: List[List[float]]
    ) -> List[float]:
        """Compute similarity of query against all corpus embeddings."""
        q = np.array(query_emb)
        C = np.array(corpus_embs)
        return (C @ q).tolist()
