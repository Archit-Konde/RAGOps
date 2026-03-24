"""
Tests for the /v1/ingest endpoint.

These tests mock the database and embedding model to run without
external dependencies (no PostgreSQL, no GPU).
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from sqlalchemy.exc import IntegrityError

from apps.api.app.deps import get_db, get_embedder, get_settings
from apps.api.app.settings import Settings


def _mock_embedder():
    """Return a mock EmbeddingModel that produces 384-dim vectors."""
    embedder = MagicMock()
    embedder.embed_texts.return_value = np.random.randn(3, 384).astype(np.float32)
    return embedder


def _mock_session(existing_doc=None):
    """Return a mock async DB session."""
    session = AsyncMock()
    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = existing_doc
    session.execute.return_value = result_mock
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    return session


@pytest.mark.asyncio
async def test_ingest_txt_file(app, client):
    """Uploading a .txt file should return ingested status with chunks."""
    mock_session = _mock_session()
    mock_embedder = _mock_embedder()

    app.dependency_overrides[get_db] = lambda: mock_session
    app.dependency_overrides[get_embedder] = lambda: mock_embedder
    app.dependency_overrides[get_settings] = lambda: Settings()

    try:
        content = b"This is a test document with enough text to produce chunks. " * 20
        response = await client.post(
            "/v1/ingest",
            files={"file": ("test.txt", content, "text/plain")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ingested"
        assert data["num_chunks"] > 0
        assert "document_id" in data
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_ingest_duplicate_file(app, client):
    """Re-uploading the same file should return duplicate status."""
    existing_doc = MagicMock()
    existing_doc.id = uuid.uuid4()
    mock_session = _mock_session(existing_doc=existing_doc)
    mock_embedder = _mock_embedder()

    app.dependency_overrides[get_db] = lambda: mock_session
    app.dependency_overrides[get_embedder] = lambda: mock_embedder
    app.dependency_overrides[get_settings] = lambda: Settings()

    try:
        content = b"Duplicate test content."
        response = await client.post(
            "/v1/ingest",
            files={"file": ("test.txt", content, "text/plain")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "duplicate"
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_ingest_empty_file_returns_422(app, client):
    """Uploading an empty file should return 422 (no chunks produced)."""
    mock_session = _mock_session()
    mock_embedder = _mock_embedder()

    app.dependency_overrides[get_db] = lambda: mock_session
    app.dependency_overrides[get_embedder] = lambda: mock_embedder
    app.dependency_overrides[get_settings] = lambda: Settings()

    try:
        response = await client.post(
            "/v1/ingest",
            files={"file": ("empty.txt", b"", "text/plain")},
        )

        assert response.status_code == 422
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_ingest_concurrent_duplicate_returns_duplicate(app, client):
    """Concurrent upload hitting IntegrityError should return duplicate status."""
    mock_session = _mock_session()

    # First flush raises IntegrityError (concurrent insert won the race)
    mock_session.flush = AsyncMock(
        side_effect=IntegrityError("duplicate key", params=None, orig=Exception())
    )
    mock_session.rollback = AsyncMock()

    # After rollback, the re-query finds the existing doc
    existing_doc = MagicMock()
    existing_doc.id = uuid.uuid4()
    result_after_rollback = MagicMock()
    result_after_rollback.scalar_one_or_none.return_value = existing_doc

    # First call returns no existing doc, second call (after rollback) finds it
    first_result = MagicMock()
    first_result.scalar_one_or_none.return_value = None
    mock_session.execute = AsyncMock(side_effect=[first_result, result_after_rollback])

    mock_embedder = _mock_embedder()

    app.dependency_overrides[get_db] = lambda: mock_session
    app.dependency_overrides[get_embedder] = lambda: mock_embedder
    app.dependency_overrides[get_settings] = lambda: Settings()

    try:
        content = b"This is a test document with enough text to produce chunks. " * 20
        response = await client.post(
            "/v1/ingest",
            files={"file": ("test.txt", content, "text/plain")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "duplicate"
        assert data["document_id"] == str(existing_doc.id)
    finally:
        app.dependency_overrides.clear()
