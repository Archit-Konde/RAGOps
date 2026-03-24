"""
Benchmark script for the RAGOps pipeline.

Usage:
    python scripts/run_benchmark.py

Runs sample queries against an inline test corpus, computes retrieval metrics
(Precision@5, Recall@5, MRR), and prints a formatted results table.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

# Allow imports from project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from packages.rag_core.chunking import RecursiveTextChunker  # noqa: E402
from packages.rag_core.embedding import EmbeddingModel  # noqa: E402
from packages.rag_core.rerank import CrossEncoderReranker  # noqa: E402


# ---------------------------------------------------------------------------
# Inline test corpus — short paragraphs on distinct topics
# ---------------------------------------------------------------------------

CORPUS = """HTTP (HyperText Transfer Protocol) is an application-layer protocol for transmitting hypermedia documents. It was designed for communication between web browsers and web servers. HTTP follows a request-response model where the client sends a request and the server returns a response.

TCP (Transmission Control Protocol) provides reliable, ordered delivery of data between applications. It uses a three-way handshake to establish connections: SYN, SYN-ACK, ACK. TCP handles flow control and congestion control to prevent network overload.

DNS (Domain Name System) translates human-readable domain names into IP addresses. When you type a URL into your browser, DNS resolvers query a hierarchy of nameservers to find the corresponding IP address. DNS uses both UDP and TCP on port 53.

TLS (Transport Layer Security) encrypts communication between client and server. The TLS handshake negotiates cipher suites and exchanges certificates for authentication. Modern web traffic uses TLS 1.3, which reduced the handshake to a single round trip.

REST (Representational State Transfer) is an architectural style for designing networked applications. RESTful APIs use standard HTTP methods: GET for reading, POST for creating, PUT for updating, and DELETE for removing resources. Each resource is identified by a unique URI.

WebSocket provides full-duplex communication channels over a single TCP connection. Unlike HTTP, WebSocket allows the server to push data to the client without polling. It is commonly used for real-time applications like chat, gaming, and live data feeds.

GraphQL is a query language for APIs developed by Facebook. Unlike REST, GraphQL allows clients to request exactly the data they need in a single request. It uses a strongly-typed schema to define available data and operations.

OAuth 2.0 is an authorization framework that enables third-party applications to obtain limited access to a web service. It uses access tokens rather than credentials. Common grant types include authorization code, client credentials, and refresh tokens.

CORS (Cross-Origin Resource Sharing) is a security mechanism that allows web pages to make requests to a different domain. The server indicates which origins are permitted via Access-Control-Allow-Origin headers. Preflight OPTIONS requests are sent for non-simple cross-origin requests.

CDN (Content Delivery Network) distributes content across geographically dispersed servers to reduce latency. CDNs cache static assets like images, CSS, and JavaScript at edge locations close to users. Popular CDN providers include Cloudflare, AWS CloudFront, and Akamai."""


# ---------------------------------------------------------------------------
# Test cases: query → ground-truth relevant chunk indices
# ---------------------------------------------------------------------------

TEST_CASES = [
    {
        "query": "How does TCP establish a connection?",
        "relevant_ids": [1],
    },
    {
        "query": "What is the purpose of DNS?",
        "relevant_ids": [2],
    },
    {
        "query": "How does TLS encryption work?",
        "relevant_ids": [3],
    },
    {
        "query": "What HTTP methods are used in REST APIs?",
        "relevant_ids": [4, 0],
    },
    {
        "query": "How does OAuth handle authorization?",
        "relevant_ids": [7],
    },
]


# ---------------------------------------------------------------------------
# Metrics — ported from evaluation.py
# ---------------------------------------------------------------------------


def compute_retrieval_metrics(
    retrieved_ids: list[int | str],
    relevant_ids: list[int | str],
    k: int | None = None,
) -> dict[str, float]:
    """Compute precision, recall, F1, and MRR for a single query."""
    if k is None:
        k = len(retrieved_ids)

    retrieved_at_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)

    hits = sum(1 for doc_id in retrieved_at_k if doc_id in relevant_set)

    precision = hits / k if k > 0 else 0.0
    recall = hits / len(relevant_set) if relevant_set else 0.0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    # MRR
    mrr = 0.0
    for rank, doc_id in enumerate(retrieved_at_k):
        if doc_id in relevant_set:
            mrr = 1.0 / (rank + 1)
            break

    return {
        "precision_at_k": precision,
        "recall_at_k": recall,
        "f1_at_k": f1,
        "mrr": mrr,
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    query: str,
    corpus_embeddings: np.ndarray,
    chunks: list[dict],
    embedder: EmbeddingModel,
    reranker: CrossEncoderReranker,
    top_k: int = 5,
    rerank_top_k: int = 5,
) -> list[int]:
    """Embed → cosine search → rerank → return chunk indices."""
    query_vec = embedder.embed_query(query)

    # Dense search via NumPy (no DB needed for benchmark)
    similarities = corpus_embeddings @ query_vec
    top_indices = np.argsort(similarities)[::-1][: top_k * 3]

    candidates = [
        {
            **chunks[i],
            "score": float(similarities[i]),
            "index": chunks[i]["chunk_index"],
        }
        for i in top_indices
    ]

    # Rerank
    reranked = reranker.rerank(query, candidates, top_k=rerank_top_k)

    return [r["chunk_index"] for r in reranked]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    top_k = 5

    # Chunk corpus
    chunker = RecursiveTextChunker(chunk_size=512, chunk_overlap=64)
    chunks = chunker.split_text(CORPUS)
    print(f"Corpus chunked into {len(chunks)} chunks\n")

    # Load models
    print("[1/3] Loading embedding model...")
    embedder = EmbeddingModel()

    print("[2/3] Embedding corpus...")
    texts = [c["text"] for c in chunks]
    corpus_embeddings = embedder.embed_texts(texts, show_progress=True)

    print("[3/3] Loading cross-encoder reranker...")
    reranker = CrossEncoderReranker()

    # Run benchmark
    print(f"\nRunning {len(TEST_CASES)} queries...\n")

    all_metrics: list[dict[str, float]] = []
    for case in TEST_CASES:
        retrieved_ids = run_pipeline(
            case["query"],
            corpus_embeddings,
            chunks,
            embedder,
            reranker,
            top_k=top_k,
            rerank_top_k=top_k,
        )
        metrics = compute_retrieval_metrics(
            retrieved_ids, case["relevant_ids"], k=top_k
        )
        all_metrics.append(metrics)

        print(
            f"  Q: {case['query'][:60]:<60s}  "
            f"P@{top_k}={metrics['precision_at_k']:.4f}  "
            f"R@{top_k}={metrics['recall_at_k']:.4f}  "
            f"MRR={metrics['mrr']:.4f}"
        )

    # Aggregate
    n = len(all_metrics)
    mean_p = sum(m["precision_at_k"] for m in all_metrics) / n
    mean_r = sum(m["recall_at_k"] for m in all_metrics) / n
    mean_f1 = sum(m["f1_at_k"] for m in all_metrics) / n
    mean_mrr = sum(m["mrr"] for m in all_metrics) / n

    # Print formatted table
    print(f"\n{'=' * 60}")
    print(f"  BENCHMARK RESULTS  (top_k={top_k}, n={n} queries)")
    print(f"{'=' * 60}")
    print(f"  {'Metric':<20s}  {'Score':>10s}")
    print(f"  {'-' * 32}")
    print(f"  {'Precision@' + str(top_k):<20s}  {mean_p:>10.4f}")
    print(f"  {'Recall@' + str(top_k):<20s}  {mean_r:>10.4f}")
    print(f"  {'F1@' + str(top_k):<20s}  {mean_f1:>10.4f}")
    print(f"  {'MRR':<20s}  {mean_mrr:>10.4f}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
