"""
RAG query router.

POST /v1/query — ask a question against the ingested documents.
"""
from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.app.deps import get_db, get_embedder, get_reranker, get_settings
from apps.api.app.services import rag_service
from apps.api.app.settings import Settings
from packages.rag_core.embedding import EmbeddingModel
from packages.rag_core.rerank import CrossEncoderReranker

router = APIRouter(prefix="/v1", tags=["query"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class SourceInfo(BaseModel):
    source_num: int
    chunk_id: str
    document_id: str
    chunk_index: int
    score: float


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceInfo]
    model: str
    usage: UsageInfo


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post("/query", response_model=QueryResponse)
async def query_documents(
    body: QueryRequest,
    db: AsyncSession = Depends(get_db),
    embedder: EmbeddingModel = Depends(get_embedder),
    reranker: CrossEncoderReranker = Depends(get_reranker),
    settings: Settings = Depends(get_settings),
) -> QueryResponse:
    """Ask a question and get a grounded answer with source attribution."""
    result = await rag_service.query_rag(
        query=body.query,
        db=db,
        embedder=embedder,
        reranker=reranker,
        settings=settings,
        top_k=body.top_k,
    )
    return QueryResponse(**result)
