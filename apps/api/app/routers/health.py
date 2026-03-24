"""
Health check router.
"""

from __future__ import annotations

from fastapi import APIRouter

from apps.api.app import VERSION

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check() -> dict:
    """Liveness probe — returns 200 if the service is up."""
    return {"status": "ok", "version": VERSION}
