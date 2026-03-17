"""
Document ingestion service.

Orchestrates: read file → hash → dedup check → extract text → chunk → embed → store.
"""
from __future__ import annotations

import asyncio
import hashlib
import time
from typing import Any, Dict

from fastapi import UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.app.db.models import Chunk, Document
from apps.api.app.observability.metrics import log_ingest
from apps.api.app.settings import Settings
from packages.rag_core.chunking import RecursiveTextChunker
from packages.rag_core.embedding import EmbeddingModel

# Module-level singleton — config is static (chunk_size=512, chunk_overlap=64)
_chunker = RecursiveTextChunker()


async def ingest_file(
    file: UploadFile,
    db: AsyncSession,
    embedder: EmbeddingModel,
    settings: Settings,
) -> Dict[str, Any]:
    """
    Ingest an uploaded file into the database.

    Returns:
        {"document_id": str, "num_chunks": int, "status": "ingested"|"duplicate"}
    """
    t0 = time.perf_counter()

    # 1. Read file bytes and compute content hash
    file_bytes = await file.read()
    content_hash = hashlib.sha256(file_bytes).hexdigest()

    # 2. Dedup check
    existing = await db.execute(
        select(Document).where(Document.content_hash == content_hash)
    )
    existing_doc = existing.scalar_one_or_none()
    if existing_doc is not None:
        return {
            "document_id": str(existing_doc.id),
            "num_chunks": 0,
            "status": "duplicate",
        }

    # 3. Extract text
    filename = file.filename or "unknown"
    text = _extract_text(file_bytes, filename)

    # 4. Chunk (uses class defaults: chunk_size=512, chunk_overlap=64)
    chunks = _chunker.split_text(text)

    if not chunks:
        raise ValueError(f"No chunks produced from file: {filename}")

    # 5. Embed (CPU-bound → run in thread)
    chunk_texts = [c["text"] for c in chunks]
    embeddings = await asyncio.to_thread(embedder.embed_texts, chunk_texts)

    # 6. Store document + chunks
    doc = Document(filename=filename, content_hash=content_hash)
    db.add(doc)
    await db.flush()  # get doc.id

    chunk_rows = [
        Chunk(
            document_id=doc.id,
            content=chunk_meta["text"],
            embedding=embedding_vec.tolist(),
            chunk_index=chunk_meta["chunk_index"],
            start_char=chunk_meta["start_char"],
            end_char=chunk_meta["end_char"],
        )
        for chunk_meta, embedding_vec in zip(chunks, embeddings)
    ]
    db.add_all(chunk_rows)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    log_ingest(filename, len(chunks), elapsed_ms)

    return {
        "document_id": str(doc.id),
        "num_chunks": len(chunks),
        "status": "ingested",
    }


def _extract_text(file_bytes: bytes, filename: str) -> str:
    """Decode file bytes to text. Supports .txt, .md, and .pdf."""
    lower = filename.lower()

    if lower.endswith(".pdf"):
        return _extract_pdf_text(file_bytes)

    # Default: treat as UTF-8 text
    return file_bytes.decode("utf-8", errors="replace")


def _extract_pdf_text(file_bytes: bytes) -> str:
    """Extract text from a PDF using PyPDF2."""
    import io

    from PyPDF2 import PdfReader

    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n\n".join(pages)
