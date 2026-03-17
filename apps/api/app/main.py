"""
FastAPI application factory.

Entry point: uvicorn apps.api.app.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from apps.api.app.db.session import close_db, init_db
from apps.api.app.routers import health, ingest, query
from apps.api.app import VERSION
from apps.api.app.services.rag_service import close_http_client
from apps.api.app.settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup / shutdown lifecycle events."""
    settings = get_settings()

    # Startup: connect to database
    await init_db(settings.DATABASE_URL)

    # Pre-load ML models so first request isn't penalised
    from apps.api.app.deps import get_embedder, get_reranker

    get_embedder(settings)
    get_reranker(settings)

    yield

    # Shutdown: release resources
    await close_http_client()
    await close_db()


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    app = FastAPI(
        title="RAGOps API",
        description="Production-style RAG service with pgvector and FastAPI.",
        version=VERSION,
        lifespan=lifespan,
    )

    app.include_router(health.router)
    app.include_router(ingest.router)
    app.include_router(query.router)

    return app


app = create_app()
