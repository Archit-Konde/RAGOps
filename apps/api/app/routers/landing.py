"""
Landing page router.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from apps.api.app import VERSION

router = APIRouter(tags=["landing"])

_templates_dir = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(_templates_dir))


@router.get("/", response_class=HTMLResponse)
async def landing(request: Request):
    """Project landing page."""
    return templates.TemplateResponse("index.html", {"request": request, "version": VERSION})
