"""
Shared test fixtures for RAGOps.
"""
from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from apps.api.app.main import create_app


@pytest.fixture
def app():
    """Create a fresh FastAPI app for testing."""
    return create_app()


@pytest.fixture
async def client(app):
    """Async HTTP client wired to the test app (no real server needed)."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
