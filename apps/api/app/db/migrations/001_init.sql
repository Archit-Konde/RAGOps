-- RAGOps: initial schema
-- Run with: psql -U ragops -d ragops -f 001_init.sql

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Documents table: tracks ingested files with content-hash deduplication
CREATE TABLE IF NOT EXISTS documents (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename    VARCHAR(512) NOT NULL,
    content_hash CHAR(64) NOT NULL UNIQUE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_documents_content_hash
    ON documents (content_hash);

-- Chunks table: text segments with vector embeddings
CREATE TABLE IF NOT EXISTS chunks (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id  UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    content      TEXT NOT NULL,
    embedding    vector(384) NOT NULL,
    chunk_index  INTEGER NOT NULL,
    start_char   INTEGER NOT NULL DEFAULT 0,
    end_char     INTEGER NOT NULL DEFAULT 0,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id
    ON chunks (document_id);

-- HNSW index for fast cosine-similarity search.
-- Optimal for corpora > 10K chunks; exact search is fine for smaller datasets.
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
