"""
FastAPI application factory.

Entry point: uvicorn apps.api.app.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text

from apps.api.app import VERSION
from apps.api.app.db import session as db_session
from apps.api.app.db.session import close_db, init_db
from apps.api.app.routers import health, ingest, landing, query
from apps.api.app.services.rag_service import close_http_client
from apps.api.app.settings import get_settings

logger = logging.getLogger(__name__)

_MIGRATIONS_DIR = Path(__file__).resolve().parent / "db" / "migrations"


async def _run_migrations() -> None:
    """Execute idempotent SQL migrations on startup."""
    migration_file = _MIGRATIONS_DIR / "001_init.sql"
    if not migration_file.exists():
        return
    sql = migration_file.read_text()
    async with db_session._engine.begin() as conn:
        await conn.execute(text(sql))
    logger.info("Database migrations applied successfully")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup / shutdown lifecycle events."""
    settings = get_settings()

    # Startup: connect to database and run migrations
    await init_db(settings.DATABASE_URL)
    await _run_migrations()

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

    app.include_router(landing.router)
    app.include_router(health.router)
    app.include_router(ingest.router)
    app.include_router(query.router)

    # Serve static assets for the landing page
    static_dir = Path(__file__).resolve().parent / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    return app


app = create_app()
