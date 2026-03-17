"""
Async database session management.

Provides init/close lifecycle hooks for the FastAPI lifespan and an async
session generator for dependency injection.
"""
from __future__ import annotations

from collections.abc import AsyncGenerator

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

    # Ensure we use the asyncpg driver
    url = database_url.replace("postgresql://", "postgresql+asyncpg://")

    _engine = create_async_engine(url, echo=False, pool_size=10, max_overflow=20)
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
