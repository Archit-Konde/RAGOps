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
