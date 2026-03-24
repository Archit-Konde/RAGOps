"""
RAG query service.

Orchestrates: embed question → pgvector search → rerank → build prompt → call LLM → return.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx
import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.app.db.models import Chunk
from apps.api.app.observability.metrics import log_query
from apps.api.app.settings import Settings
from packages.rag_core.embedding import EmbeddingModel
from packages.rag_core.rerank import CrossEncoderReranker

logger = logging.getLogger("ragops")

# ---------------------------------------------------------------------------
# System prompt — ported from the original RAG project
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a precise question-answering assistant. "
    "Answer questions using ONLY the provided context. "
    "If the context does not contain enough information to answer, "
    "say \"I don't have enough information in the provided documents "
    'to answer this question." '
    "Do not speculate or use knowledge outside the provided context. "
    "Cite sources using [Source N] notation where N matches the context "
    "block number."
)

# ---------------------------------------------------------------------------
# Shared httpx client — created once, reused across requests
# ---------------------------------------------------------------------------

_http_client: httpx.AsyncClient | None = None


async def get_http_client() -> httpx.AsyncClient:
    """Return the shared httpx client, creating it on first call."""
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(timeout=30.0)
    return _http_client


async def close_http_client() -> None:
    """Close the shared httpx client (call during shutdown)."""
    global _http_client
    if _http_client is not None:
        await _http_client.aclose()
        _http_client = None


# ---------------------------------------------------------------------------
# pgvector search (lives here because it's coupled to the DB schema)
# ---------------------------------------------------------------------------


async def pgvector_search(
    query_embedding: np.ndarray,
    top_k: int,
    session: AsyncSession,
) -> list[dict[str, Any]]:
    """
    Run a cosine-similarity KNN search against the chunks table.

    Returns:
        List of dicts ordered by ascending distance (most similar first).
    """
    embedding_list = query_embedding.tolist()
    distance_expr = Chunk.embedding.cosine_distance(embedding_list)

    stmt = (
        select(
            Chunk.id,
            Chunk.document_id,
            Chunk.content,
            Chunk.chunk_index,
            distance_expr.label("distance"),
        )
        .order_by(distance_expr)
        .limit(top_k)
    )

    result = await session.execute(stmt)
    rows = result.all()

    return [
        {
            "id": str(row.id),
            "document_id": str(row.document_id),
            "text": row.content,
            "chunk_index": row.chunk_index,
            "distance": float(row.distance),
            "score": 1.0 - float(row.distance),
        }
        for row in rows
    ]


# ---------------------------------------------------------------------------
# Main query function
# ---------------------------------------------------------------------------


async def query_rag(
    query: str,
    db: AsyncSession,
    embedder: EmbeddingModel,
    reranker: CrossEncoderReranker,
    settings: Settings,
    top_k: int = 5,
) -> dict[str, Any]:
    """
    End-to-end RAG: retrieve context, rerank, generate answer.

    Returns:
        {
            "answer": str,
            "sources": list[dict],
            "model": str,
            "usage": {"prompt_tokens": int, "completion_tokens": int},
        }
    """
    t0 = time.perf_counter()

    # 1. Embed question (CPU-bound → thread)
    query_vec = await asyncio.to_thread(embedder.embed_query, query)

    # 2. pgvector KNN search
    candidates = await pgvector_search(query_vec, settings.TOP_K_DENSE, db)

    if not candidates:
        return {
            "answer": "No relevant documents found.",
            "sources": [],
            "model": settings.OPENAI_MODEL,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }

    # 3. Cross-encoder rerank (CPU-bound → thread)
    reranked = await asyncio.to_thread(
        reranker.rerank, query, candidates, top_k=settings.TOP_K_RERANK
    )

    # 4. Build LLM prompt
    messages = _build_prompt(query, reranked)

    # 5. Call LLM
    llm_response = await _call_llm(messages, settings)

    answer = llm_response["choices"][0]["message"]["content"]
    usage = llm_response.get("usage", {})

    # 6. Extract source attribution
    sources = _extract_sources(reranked)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    log_query(query, elapsed_ms, len(candidates), len(reranked))

    return {
        "answer": answer,
        "sources": sources,
        "model": llm_response.get("model", settings.OPENAI_MODEL),
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
        },
    }


# ---------------------------------------------------------------------------
# Prompt building — ported from LLMGenerator.build_prompt
# ---------------------------------------------------------------------------


def _build_prompt(query: str, chunks: list[dict]) -> list[dict]:
    """Build chat messages from a query and retrieved chunks."""
    context_blocks: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        doc_id = chunk.get("document_id", "unknown")
        chunk_idx = chunk.get("chunk_index", 0)
        header = f"[Source {i}] (doc {doc_id}, chunk {chunk_idx}):"
        context_blocks.append(f"{header}\n{chunk['text']}")

    context_str = (
        "\n\n".join(context_blocks) if context_blocks else "(no context provided)"
    )
    user_content = f"Context:\n\n{context_str}\n\nQuestion: {query}"

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Async LLM call via httpx (shared client)
# ---------------------------------------------------------------------------


async def _call_llm(messages: list[dict], settings: Settings) -> dict:
    """POST to the OpenAI-compatible /chat/completions endpoint."""
    if not settings.OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY is not configured. Set it in .env or environment variables."
        )

    url = f"{settings.OPENAI_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.OPENAI_MODEL,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 1024,
    }

    client = await get_http_client()
    try:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
    except httpx.TimeoutException:
        raise RuntimeError(
            "LLM request timed out — try again or check your API endpoint."
        )
    except httpx.HTTPStatusError as e:
        raise RuntimeError(
            f"LLM API error: {e.response.status_code} — {e.response.text[:200]}"
        )

    if not data.get("choices"):
        raise RuntimeError(f"LLM returned no choices: {str(data)[:200]}")

    return data


# ---------------------------------------------------------------------------
# Source attribution
# ---------------------------------------------------------------------------


def _extract_sources(chunks: list[dict]) -> list[dict]:
    """Build a source attribution list from chunk metadata."""
    sources = []
    for i, chunk in enumerate(chunks, start=1):
        score = chunk.get("rerank_score", chunk.get("score", 0.0))
        sources.append(
            {
                "source_num": i,
                "chunk_id": chunk.get("id", "unknown"),
                "document_id": chunk.get("document_id", "unknown"),
                "chunk_index": chunk.get("chunk_index", 0),
                "score": round(float(score), 6),
            }
        )
    return sources
