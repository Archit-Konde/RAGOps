"""
Unit tests for rag_core: chunking and embedding.

These tests run without a database — they validate the core algorithms directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from packages.rag_core.chunking import RecursiveTextChunker


# ---------------------------------------------------------------------------
# Chunking tests
# ---------------------------------------------------------------------------


class TestRecursiveTextChunker:
    def test_empty_input(self):
        chunker = RecursiveTextChunker()
        assert chunker.split_text("") == []

    def test_short_text_single_chunk(self):
        chunker = RecursiveTextChunker(chunk_size=100, chunk_overlap=10)
        result = chunker.split_text("Hello world.")
        assert len(result) == 1
        assert result[0]["text"] == "Hello world."
        assert result[0]["chunk_index"] == 0

    def test_chunk_size_respected(self):
        """No chunk should exceed chunk_size."""
        text = "Word " * 500  # ~2500 chars
        chunker = RecursiveTextChunker(chunk_size=200, chunk_overlap=20)
        chunks = chunker.split_text(text)
        for chunk in chunks:
            assert len(chunk["text"]) <= 200, (
                f"Chunk {chunk['chunk_index']} has {len(chunk['text'])} chars"
            )

    def test_overlap_exists(self):
        """Consecutive chunks should share overlapping text."""
        text = "Sentence one. Sentence two. Sentence three. Sentence four. " * 10
        chunker = RecursiveTextChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.split_text(text)

        if len(chunks) < 2:
            pytest.skip("Not enough chunks to test overlap")

        # At least some consecutive pairs should have overlapping characters
        overlap_found = False
        for i in range(len(chunks) - 1):
            suffix = chunks[i]["text"][-20:]
            if suffix in chunks[i + 1]["text"]:
                overlap_found = True
                break

        assert overlap_found, "No overlap detected between consecutive chunks"

    def test_chunk_indices_sequential(self):
        text = "Hello world. " * 200
        chunker = RecursiveTextChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.split_text(text)
        for i, chunk in enumerate(chunks):
            assert chunk["chunk_index"] == i

    def test_start_end_char_positions(self):
        text = "Alpha. Bravo. Charlie. Delta. Echo."
        chunker = RecursiveTextChunker(chunk_size=20, chunk_overlap=5)
        chunks = chunker.split_text(text)
        for chunk in chunks:
            assert chunk["start_char"] >= 0
            assert chunk["end_char"] > chunk["start_char"]
            assert text[chunk["start_char"] : chunk["end_char"]] == chunk["text"]

    def test_overlap_must_be_less_than_size(self):
        with pytest.raises(
            ValueError, match="chunk_overlap must be less than chunk_size"
        ):
            RecursiveTextChunker(chunk_size=100, chunk_overlap=100)

    def test_separator_priority(self):
        """Double newlines should be preferred over single newlines."""
        text = "Block A content\n\nBlock B content"
        chunker = RecursiveTextChunker(chunk_size=50, chunk_overlap=5)
        chunks = chunker.split_text(text)
        texts = [c["text"] for c in chunks]
        assert any("Block A" in t for t in texts)
        assert any("Block B" in t for t in texts)


# ---------------------------------------------------------------------------
# Embedding tests (requires sentence-transformers model download on first run)
# ---------------------------------------------------------------------------


class TestEmbeddingModel:
    @pytest.fixture(scope="class")
    def embedder(self):
        """Load the embedding model once for all tests in this class."""
        from packages.rag_core.embedding import EmbeddingModel

        return EmbeddingModel()

    def test_embed_shape(self, embedder):
        texts = ["hello", "world"]
        result = embedder.embed_texts(texts)
        assert result.shape == (2, 384)
        assert result.dtype == np.float32

    def test_embed_query_shape(self, embedder):
        result = embedder.embed_query("test query")
        assert result.shape == (384,)

    def test_l2_normalized(self, embedder):
        result = embedder.embed_texts(["normalize me"])
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_identical_strings_high_similarity(self, embedder):
        """Embedding the same string twice should yield similarity ≈ 1.0."""
        a = embedder.embed_query("the cat sat on the mat")
        b = embedder.embed_query("the cat sat on the mat")
        similarity = float(np.dot(a, b))
        assert similarity > 0.99

    def test_semantic_similarity_ordering(self, embedder):
        """cat/kitten should be more similar than cat/blockchain."""
        cat = embedder.embed_query("cat")
        kitten = embedder.embed_query("kitten")
        blockchain = embedder.embed_query("blockchain")

        sim_cat_kitten = float(np.dot(cat, kitten))
        sim_cat_blockchain = float(np.dot(cat, blockchain))

        assert sim_cat_kitten > sim_cat_blockchain

    def test_empty_input(self, embedder):
        result = embedder.embed_texts([])
        assert result.shape == (0, 384)
