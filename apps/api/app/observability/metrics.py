"""
Structured logging helpers for observability.

Logs key events (ingestion, queries) as structured messages. Can be extended
with prometheus_client counters/histograms for production monitoring.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("ragops")


def log_ingest(filename: str, num_chunks: int, latency_ms: float) -> None:
    logger.info(
        "ingest completed",
        extra={
            "event": "ingest",
            "filename": filename,
            "num_chunks": num_chunks,
            "latency_ms": round(latency_ms, 1),
        },
    )


def log_query(
    query: str,
    latency_ms: float,
    num_retrieved: int,
    num_reranked: int,
) -> None:
    logger.info(
        "query completed",
        extra={
            "event": "query",
            "query": query[:200],
            "latency_ms": round(latency_ms, 1),
            "num_retrieved": num_retrieved,
            "num_reranked": num_reranked,
        },
    )
