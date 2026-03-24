"""
rag_core — self-contained RAG building blocks.

Chunking, embedding, retrieval, and reranking utilities that are
importable independently of the FastAPI application.
"""

from packages.rag_core.chunking import RecursiveTextChunker
from packages.rag_core.embedding import EmbeddingModel
from packages.rag_core.rerank import CrossEncoderReranker

__all__ = [
    "RecursiveTextChunker",
    "EmbeddingModel",
    "CrossEncoderReranker",
]
