"""Run-length-encoding helpers for consecutive-bin filtering.

Port of R's ``RemoveShortRegions`` logic. Given a boolean mask where
``True`` = good bin, runs of ``True`` shorter than ``min_run`` surrounded
by ``False`` are flipped to ``False``.
"""

from __future__ import annotations

import numpy as np


def rle(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(lengths, values)`` run-length encoding of a 1-D array."""
    values = np.asarray(values)
    if values.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=values.dtype)
    # Boundary indices between runs.
    diffs = np.concatenate(([True], values[1:] != values[:-1], [True]))
    boundaries = np.flatnonzero(diffs)
    lengths = np.diff(boundaries)
    run_values = values[boundaries[:-1]]
    return lengths, run_values


def inverse_rle(lengths: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Inverse of :func:`rle`."""
    return np.repeat(values, lengths)


def remove_short_true_runs(mask: np.ndarray, min_run: int) -> np.ndarray:
    """Flip runs of ``True`` shorter than ``min_run`` to ``False``.

    This matches R's:

        inverse.rle(within.list(rle(x), values[lengths<min_run] <- FALSE))

    Note: R's version flips *any* run shorter than ``min_run`` (regardless
    of its value). That's a no-op for False runs since they'd stay False,
    so for our boolean input we only change True runs in practice.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return mask.copy()
    lengths, values = rle(mask)
    values = values.copy()
    short = lengths < int(min_run)
    values[short] = False
    return inverse_rle(lengths, values)
