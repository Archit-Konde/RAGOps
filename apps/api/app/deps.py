"""
FastAPI dependency providers.

Centralises all Depends()-injectable singletons: DB session, settings,
embedding model, and cross-encoder reranker.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends
from slowapi import Limiter
from slowapi.util import get_remote_address

from apps.api.app.db.session import get_session as get_db  # noqa: F401 (re-exported)
from apps.api.app.settings import Settings, get_settings
from packages.rag_core.embedding import EmbeddingModel
from packages.rag_core.rerank import CrossEncoderReranker

# Rate limiter singleton (shared across routers)
limiter = Limiter(key_func=get_remote_address)

# Module-level singletons (initialised on first call)
_embedder: EmbeddingModel | None = None
_reranker: CrossEncoderReranker | None = None


def get_embedder(
    settings: Annotated[Settings, Depends(get_settings)],
) -> EmbeddingModel:
    """Return the cached embedding model singleton."""
    global _embedder
    if _embedder is None:
        _embedder = EmbeddingModel(model_name=settings.EMBEDDING_MODEL)
    return _embedder


def get_reranker(
    settings: Annotated[Settings, Depends(get_settings)],
) -> CrossEncoderReranker:
    """Return the cached cross-encoder reranker singleton."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker(model_name=settings.RERANK_MODEL)
    return _reranker
