"""
RAGOps — HuggingFace Spaces demo app.

Lightweight FastAPI service that serves a project landing page and
interactive API docs. The full pipeline requires PostgreSQL + pgvector;
this demo showcases the API contract and architecture.
"""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

VERSION = "0.1.0"

app = FastAPI(
    title="RAGOps API",
    description="Production-style RAG service with pgvector and FastAPI.",
    version=VERSION,
)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Health ───────────────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    return {"status": "ok", "version": VERSION}


# ── Landing page ─────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def landing(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "version": VERSION})


# ── Demo endpoints (mirrors the real API contract) ───────────────────
class IngestResponse(BaseModel):
    document_id: str
    num_chunks: int
    status: str


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class SourceInfo(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    chunk_index: int
    score: float
    text_preview: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]
    model: str
    usage: dict


@app.post("/v1/ingest", response_model=IngestResponse)
async def ingest_demo():
    """Upload a document for ingestion. **Requires PostgreSQL** — see /docs for schema."""
    return IngestResponse(
        document_id="demo-mode",
        num_chunks=0,
        status="demo — deploy with docker-compose for full functionality",
    )


@app.post("/v1/query", response_model=QueryResponse)
async def query_demo(req: QueryRequest):
    """Query the RAG pipeline. **Requires PostgreSQL + LLM** — see /docs for schema."""
    return QueryResponse(
        answer=f"Demo mode — deploy with docker-compose to query over your documents. Your query: '{req.query}'",
        sources=[],
        model="demo",
        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    )
