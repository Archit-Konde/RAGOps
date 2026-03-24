"""
Tests for the /health endpoint.
"""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_health_returns_200(client):
    response = await client.get("/health")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_health_body(client):
    response = await client.get("/health")
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


@pytest.mark.asyncio
async def test_query_rejects_invalid_top_k(app, client):
    """top_k outside 1-20 should return 422 (validated before deps resolve)."""
    from unittest.mock import AsyncMock, MagicMock

    from apps.api.app.deps import get_db, get_embedder, get_reranker, get_settings
    from apps.api.app.settings import Settings

    app.dependency_overrides[get_db] = lambda: AsyncMock()
    app.dependency_overrides[get_embedder] = lambda: MagicMock()
    app.dependency_overrides[get_reranker] = lambda: MagicMock()
    app.dependency_overrides[get_settings] = lambda: Settings()

    try:
        response = await client.post(
            "/v1/query",
            json={"query": "test", "top_k": 0},
        )
        assert response.status_code == 422

        response = await client.post(
            "/v1/query",
            json={"query": "test", "top_k": 100},
        )
        assert response.status_code == 422
    finally:
        app.dependency_overrides.clear()
