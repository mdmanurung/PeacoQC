"""Detect monotonic (increasing/decreasing) trends in channel medians.

Port of R's ``FindIncreasingDecreasingChannels``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _nadaraya_watson(y: np.ndarray, bandwidth: float = 50.0) -> np.ndarray:
    """A simple Nadaraya-Watson estimator using a box kernel.

    R's ``ksmooth(..., kernel='box', bandwidth=50)`` uses a box kernel
    that is nonzero for ``|x - x0| < bandwidth / 2``, i.e. a half-width
    of ``bandwidth / 2``.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n == 0:
        return y.copy()
    x = np.arange(n, dtype=float)
    half = 0.5 * bandwidth  # R ksmooth box kernel: nonzero for |x-x0| < bw/2
    out = np.empty(n, dtype=float)
    for i in range(n):
        mask = np.abs(x - i) < half
        if not np.any(mask):
            out[i] = y[i]
        else:
            out[i] = float(np.mean(y[mask]))
    return out


def find_increasing_decreasing_channels(
    X: np.ndarray,
    channel_names: Sequence[str],
    channel_indices: Sequence[int],
    breaks: Sequence[np.ndarray],
    *,
    bandwidth: float = 50.0,
    monotonic_fraction: float = 0.75,
) -> dict:
    """Classify channels as monotonically increasing/decreasing.

    Parameters
    ----------
    X
        Dense expression matrix.
    channel_names, channel_indices
        Matched sequences of channel names and column indices into ``X``.
    breaks
        Overlapping bins, each as an integer index array.
    bandwidth
        Kernel smoothing bandwidth (matches R's ``ksmooth(..., bandwidth=50)``).
    monotonic_fraction
        Fraction of bins that must be at the cumulative max/min for the
        channel to be flagged (R uses 3/4).

    Returns
    -------
    dict with keys ``increasing``, ``decreasing``, ``label``.
    ``label`` is one of:
    ``"No increasing or decreasing effect"``, ``"Increasing channel"``,
    ``"Decreasing channel"``, or ``"Increasing and decreasing channel"``.
    """
    increasing: list[str] = []
    decreasing: list[str] = []

    for name, idx in zip(channel_names, channel_indices):
        values = X[:, idx]
        medians = np.array([float(np.median(values[b])) for b in breaks])
        if len(medians) == 0:
            continue
        smoothed = _nadaraya_watson(medians, bandwidth=bandwidth)
        cum_max = np.maximum.accumulate(smoothed)
        cum_min = np.minimum.accumulate(smoothed)
        inc_frac = float(np.mean(cum_max == smoothed))
        dec_frac = float(np.mean(cum_min == smoothed))
        if inc_frac > monotonic_fraction:
            increasing.append(name)
        elif dec_frac > monotonic_fraction:
            decreasing.append(name)

    if increasing and decreasing:
        label = "Increasing and decreasing channel"
    elif increasing:
        label = "Increasing channel"
    elif decreasing:
        label = "Decreasing channel"
    else:
        label = "No increasing or decreasing effect"

    return {
        "increasing": increasing,
        "decreasing": decreasing,
        "label": label,
    }
