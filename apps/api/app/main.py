"""
FastAPI application factory.

Entry point: uvicorn apps.api.app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import re
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from sqlalchemy import text

from apps.api.app import VERSION
from apps.api.app.db.session import close_db, get_engine, init_db
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
    raw_sql = migration_file.read_text()
    # Strip SQL comments (-- ...) before splitting, since comments
    # may contain semicolons that break naive splitting.
    sql = re.sub(r"--[^\n]*", "", raw_sql)
    # asyncpg does not support multiple statements in one execute(),
    # so split on semicolons and run each statement individually.
    statements = [s.strip() for s in sql.split(";") if s.strip()]
    async with get_engine().begin() as conn:
        for stmt in statements:
            await conn.execute(text(stmt))
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

    return app


app = create_app()
