"""
Landing page redirect — sends visitors to Swagger UI (the demo IS the API).
"""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import RedirectResponse

router = APIRouter(tags=["landing"])


@router.get("/", include_in_schema=False)
async def landing():
    """Redirect root to interactive API docs."""
    return RedirectResponse(url="/docs")
