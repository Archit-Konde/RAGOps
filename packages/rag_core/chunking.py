"""
Recursive text chunker — no external dependencies.

Splits raw text into overlapping chunks by trying separators in priority order,
recursively splitting oversized pieces, and merging short pieces into full-sized
chunks while maintaining a configurable overlap window.
"""
from __future__ import annotations

from collections import deque
from typing import List


class RecursiveTextChunker:
    """
    Splits text recursively using a priority-ordered list of separators.

    Args:
        chunk_size:    Maximum character length of a single chunk.
        chunk_overlap: Number of characters to repeat at the start of each
                       successive chunk (must be < chunk_size).
        separators:    Ordered list of split strings. Tried left-to-right;
                       the empty string "" triggers hard character slicing.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: List[str] | None = None,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators: List[str] = (
            separators if separators is not None
            else ["\n\n", "\n", ". ", " ", ""]
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def split_text(self, text: str) -> List[dict]:
        """
        Split *text* into chunks and return a list of chunk dicts.

        Each dict contains:
            text        – the chunk string
            chunk_index – 0-based sequential position
            start_char  – character offset in the original text
            end_char    – character offset (exclusive) in the original text
        """
        if not text:
            return []

        raw_chunks = self._recursive_split(text, self.separators)

        result: List[dict] = []
        cursor = 0  # forward-scan position in original text

        for idx, chunk in enumerate(raw_chunks):
            # Locate the chunk in the original text starting from cursor
            pos = text.find(chunk, cursor)
            if pos == -1:
                # Fallback: scan from the beginning (should never happen in
                # practice, but keeps the method robust)
                pos = text.find(chunk)
            start = pos
            end = pos + len(chunk)
            result.append(
                {
                    "text": chunk,
                    "chunk_index": idx,
                    "start_char": start,
                    "end_char": end,
                }
            )
            # Advance cursor so next find() starts after the *non-overlapping*
            # portion of this chunk (allowing the overlap region to be found
            # again for the next chunk).
            cursor = max(cursor, end - self.chunk_overlap)

        return result

    # ------------------------------------------------------------------
    # Core recursive algorithm
    # ------------------------------------------------------------------

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """
        Return a flat list of string pieces, each <= chunk_size characters.
        """
        # Hard-slice fallback: no more separators to try
        if not separators:
            return self._hard_slice(text)

        separator = separators[0]
        remaining_separators = separators[1:]

        if separator == "":
            return self._hard_slice(text)

        splits = text.split(separator)

        # Re-attach the separator to all pieces except the last so content
        # is not lost.
        pieces: List[str] = []
        for i, piece in enumerate(splits):
            if not piece:
                continue
            if i < len(splits) - 1:
                pieces.append(piece + separator)
            else:
                pieces.append(piece)

        if not pieces:
            return []

        # Recursively break any oversized piece, then merge into chunks
        good_splits: List[str] = []
        final_chunks: List[str] = []

        for piece in pieces:
            if len(piece) <= self.chunk_size:
                good_splits.append(piece)
            else:
                if good_splits:
                    final_chunks.extend(
                        self._merge_splits(good_splits, separator)
                    )
                    good_splits = []
                final_chunks.extend(
                    self._recursive_split(piece, remaining_separators)
                )

        if good_splits:
            final_chunks.extend(self._merge_splits(good_splits, separator))

        return final_chunks

    # ------------------------------------------------------------------
    # Merge short splits into full-sized chunks with overlap
    # ------------------------------------------------------------------

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """
        Combine short pieces into chunks that respect chunk_size, with an
        overlap window of chunk_overlap characters at the start of each new
        chunk.
        """
        chunks: List[str] = []
        window: deque[str] = deque()
        current_len = 0

        for piece in splits:
            piece_len = len(piece)

            if current_len + piece_len > self.chunk_size and window:
                merged = "".join(window)
                if merged.strip():
                    chunks.append(merged)

                while window and current_len > self.chunk_overlap:
                    removed = window.popleft()
                    current_len -= len(removed)

            window.append(piece)
            current_len += piece_len

        if window:
            merged = "".join(window)
            if merged.strip():
                chunks.append(merged)

        return chunks

    # ------------------------------------------------------------------
    # Hard character slicing (last-resort fallback)
    # ------------------------------------------------------------------

    def _hard_slice(self, text: str) -> List[str]:
        """
        Slice text into pieces of exactly chunk_size characters with
        chunk_overlap overlap. Used when no separator matches.
        """
        if not text:
            return []
        step = self.chunk_size - self.chunk_overlap
        if step <= 0:
            step = self.chunk_size
        pieces: List[str] = []
        start = 0
        while start < len(text):
            pieces.append(text[start: start + self.chunk_size])
            start += step
        return pieces
