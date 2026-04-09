"""Adaptive overlapping binning helpers.

Direct port of the R helpers ``MakeBreaks``, ``SplitWithOverlap``,
``SplitWithOverlapMids``, and ``FindEventsPerBin``.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def split_with_overlap(length: int, seg_length: int, overlap: int) -> list[np.ndarray]:
    """Return a list of integer index arrays covering ``range(length)``.

    Each segment is ``seg_length`` long, with ``overlap`` indices shared
    with the next segment. The last segment is truncated to ``length``.

    Mirrors ``SplitWithOverlap`` from the R package, which uses 1-based
    indexing; this helper returns 0-based indices.
    """
    if seg_length <= 0:
        raise ValueError("seg_length must be positive")
    if overlap < 0 or overlap >= seg_length:
        raise ValueError("overlap must satisfy 0 <= overlap < seg_length")

    stride = seg_length - overlap
    starts = list(range(0, length, stride))
    bins: list[np.ndarray] = []
    for s in starts:
        e = min(s + seg_length, length)
        bins.append(np.arange(s, e, dtype=np.int64))
        if e == length:
            break
    return bins


def split_with_overlap_mids(length: int, seg_length: int, overlap: int) -> list[int]:
    """Return the mid-point indices for each bin (used for plot x-axis)."""
    stride = seg_length - overlap
    starts = list(range(0, length, stride))
    mids: list[int] = []
    for s in starts:
        mids.append(s + math.ceil(overlap / 2))
    # Drop mids past the last valid index.
    mids = [m for m in mids if m < length]
    return mids


def make_breaks(events_per_bin: int, n_events: int) -> tuple[list[np.ndarray], int]:
    """Create overlapping bins spanning ``n_events``.

    Returns ``(breaks, events_per_bin)``. Each entry of ``breaks`` is an
    integer index array.
    """
    overlap = math.ceil(events_per_bin / 2)
    breaks = split_with_overlap(n_events, events_per_bin, overlap)
    return breaks, events_per_bin


def find_events_per_bin(
    n_events: int,
    *,
    values: np.ndarray | None = None,
    remove_zeros: bool = False,
    min_cells: int = 150,
    max_bins: int = 500,
    step: int = 500,
) -> int:
    """Adaptive bin-size calculation matching ``FindEventsPerBin``.

    ``values`` is only consulted when ``remove_zeros=True``; it should be an
    ``(n_events, n_channels)`` array so the per-channel non-zero count can
    tighten ``max_bins`` (the mass-cytometry path).
    """
    if remove_zeros:
        if values is None:
            raise ValueError("values must be provided when remove_zeros=True")
        nonzero_counts = np.sum(values != 0, axis=0)
        max_bins_mass = int(nonzero_counts.min() // min_cells)
        if max_bins_mass < max_bins:
            max_bins = max(1, max_bins_mass)

    max_cells = math.ceil((n_events / max_bins) * 2)
    max_cells = (max_cells // step) * step + step
    return max(min_cells, max_cells)
