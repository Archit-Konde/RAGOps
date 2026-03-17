"""
Retrieval utilities: Reciprocal Rank Fusion.

Merges multiple ranked lists into a single ranking without requiring
score calibration between retrieval methods.

RRF reference: Cormack, Clarke & Buettcher (2009).
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence


def reciprocal_rank_fusion(
    ranked_lists: Sequence[List[Dict[str, Any]]],
    rrf_k: int = 60,
) -> List[Dict[str, Any]]:
    """
    Fuse multiple ranked lists using Reciprocal Rank Fusion.

    Each list must contain dicts with an "id" key (document identifier).

    RRF_score(d) = Σ_{list L}  1 / (k + rank_L(d))

    Args:
        ranked_lists: Sequence of ranked result lists.
        rrf_k:        Smoothing constant (standard default: 60).

    Returns:
        List sorted by descending RRF score, each dict containing
        "id" and "rrf_score" keys.
    """
    scores: Dict[str, float] = {}

    for ranked_list in ranked_lists:
        for rank_0based, item in enumerate(ranked_list):
            doc_id = str(item["id"])
            rank_1based = rank_0based + 1
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank_1based)

    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return [{"id": doc_id, "rrf_score": score} for doc_id, score in sorted_items]
