"""
Document ingestion router.

POST /v1/ingest — upload a file to be chunked, embedded, and stored.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, File, UploadFile
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.app.deps import get_db, get_embedder, get_settings
from apps.api.app.services import ingest_service
from apps.api.app.settings import Settings
from packages.rag_core.embedding import EmbeddingModel

router = APIRouter(prefix="/v1", tags=["ingest"])


class IngestResponse(BaseModel):
    document_id: str
    num_chunks: int
    status: str


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    embedder: EmbeddingModel = Depends(get_embedder),
    settings: Settings = Depends(get_settings),
) -> IngestResponse:
    """Upload a document for ingestion into the RAG pipeline."""
    result = await ingest_service.ingest_file(file, db, embedder, settings)
    return IngestResponse(**result)
