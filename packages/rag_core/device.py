"""
Shared device detection for ML model loading.
"""

from __future__ import annotations

import torch


def detect_device() -> str:
    """Auto-detect the best available torch device."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
