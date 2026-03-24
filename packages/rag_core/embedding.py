"""
Sentence embedding using sentence-transformers.

Wraps the SentenceTransformer class for batch embedding with L2 normalization.
Output: float32 vectors of dimension 384 (for all-MiniLM-L6-v2).
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from packages.rag_core.device import detect_device


class EmbeddingModel:
    """
    Bi-encoder sentence embedder backed by a SentenceTransformer model.

    Outputs L2-normalized float32 vectors.

    Args:
        model_name: Any SentenceTransformer model ID.
        device:     "cuda", "mps", "cpu", or None for auto-detection.
    """

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
    ) -> None:
        self.device = device or detect_device()
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=self.device)
        self._dim: int = self.model.get_sentence_embedding_dimension()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_texts(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Embed a list of strings in batches.

        Returns:
            np.ndarray of shape (N, EMBEDDING_DIM), float32, L2-normalized.
        """
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string.

        Returns:
            np.ndarray of shape (EMBEDDING_DIM,), float32, L2-normalized.
        """
        return self.embed_texts([query])[0]

    @property
    def dimension(self) -> int:
        return self._dim

    def __repr__(self) -> str:
        return f"EmbeddingModel(model={self.model_name!r}, device={self.device!r})"
