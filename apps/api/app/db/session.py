"""
Async database session management.

Provides init/close lifecycle hooks for the FastAPI lifespan and an async
session generator for dependency injection.
"""

from __future__ import annotations

import ssl as _ssl
from collections.abc import AsyncGenerator
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


async def init_db(database_url: str) -> None:
    """Create the async engine and session factory."""
    global _engine, _session_factory

    parsed = urlparse(database_url)

    # Detect if SSL is required (Neon, Supabase, etc.)
    query_params = parse_qs(parsed.query)
    needs_ssl = "sslmode" in query_params or ".neon.tech" in (parsed.hostname or "")

    # Remove sslmode from query — asyncpg uses connect_args instead
    query_params.pop("sslmode", None)
    clean_query = urlencode(query_params, doseq=True)

    # Rebuild URL with asyncpg driver
    url = urlunparse(
        (
            "postgresql+asyncpg",
            parsed.netloc,
            parsed.path,
            parsed.params,
            clean_query,
            parsed.fragment,
        )
    )

    connect_args: dict = {}
    if needs_ssl:
        connect_args["ssl"] = _ssl.create_default_context()

    _engine = create_async_engine(
        url,
        echo=False,
        pool_size=5,
        max_overflow=10,
        connect_args=connect_args,
    )
    _session_factory = async_sessionmaker(
        bind=_engine, class_=AsyncSession, expire_on_commit=False
    )


async def close_db() -> None:
    """Dispose the engine and release all connections."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _session_factory = None


def get_engine() -> AsyncEngine:
    """Return the current engine (raises if init_db() hasn't been called)."""
    if _engine is None:
        raise RuntimeError("Database not initialised — call init_db() first")
    return _engine


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async session; rolls back on exception, always closes."""
    if _session_factory is None:
        raise RuntimeError("Database not initialised — call init_db() first")

    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
