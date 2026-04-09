"""Outlier detection on the per-bin peak matrix.

- :func:`isolation_tree_outliers` uses :class:`sklearn.ensemble.IsolationForest`
  as a pragmatic replacement for the R package's custom SD-based isolation
  tree. Results are broadly equivalent but not bit-identical (documented in
  ``README.md``).
- :func:`mad_outlier_method` smooths each peak trajectory with a Savitzky-
  Golay filter (≈ R's ``smooth.spline(spar=0.5)``) and flags bins outside a
  ``median ± MAD * mad`` window.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import median_abs_deviation
from sklearn.ensemble import IsolationForest


def isolation_tree_outliers(
    peak_matrix: pd.DataFrame,
    *,
    it_limit: float = 0.6,
    random_state: int = 0,
    n_estimators: int = 100,
) -> tuple[np.ndarray, dict]:
    """Return a per-bin boolean "good" mask using :class:`IsolationForest`.

    ``it_limit`` is the R parameter. In the original package it is the
    "gain" threshold of the custom SD-based tree. Here we map it to
    isolation-forest semantics via

    .. math::
        contamination = 1 - it\\_limit

    which keeps the "higher value → less strict" sense intact (``it_limit=0.6``
    flags roughly the worst 40% of bins, matching the default behaviour on
    typical flow data).

    Parameters
    ----------
    peak_matrix
        ``(n_bins, n_features)`` frame of per-bin peak positions.
    it_limit
        R-style gain threshold; clamped to ``[0.0, 1.0]``.
    random_state
        RNG seed for reproducibility.
    n_estimators
        Number of trees in the isolation forest.

    Returns
    -------
    good_mask : np.ndarray[bool]
        Length ``n_bins``. ``True`` means the bin is kept.
    info : dict
        ``{"split_columns": [...], "anomaly_scores": np.ndarray}``. The
        ``split_columns`` list contains the unique feature names used by
        the forest's decision splits (used by the plotting code to print
        per-channel IT contributions).
    """
    if peak_matrix.shape[1] == 0:
        return np.ones(peak_matrix.shape[0], dtype=bool), {
            "split_columns": [],
            "anomaly_scores": np.zeros(peak_matrix.shape[0], dtype=float),
        }

    contamination = max(min(1.0 - float(it_limit), 0.5), 1e-4)
    forest = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
    )
    values = peak_matrix.to_numpy()
    if values.shape[0] < 4:
        # Not enough samples for a meaningful fit — keep everything.
        return np.ones(values.shape[0], dtype=bool), {
            "split_columns": [],
            "anomaly_scores": np.zeros(values.shape[0], dtype=float),
        }

    forest.fit(values)
    preds = forest.predict(values)  # +1 inlier, -1 outlier
    good_mask = preds == 1

    # Collect the feature columns that the forest actually split on.
    split_feature_set: set[int] = set()
    for est in forest.estimators_:
        tree = est.tree_
        features = tree.feature[tree.feature >= 0]
        split_feature_set.update(features.tolist())
    cols = list(peak_matrix.columns)
    split_columns = [cols[i] for i in sorted(split_feature_set)]

    return good_mask, {
        "split_columns": split_columns,
        "anomaly_scores": -forest.score_samples(values),
    }


def _smooth_trajectory(y: np.ndarray) -> np.ndarray:
    """Savitzky-Golay smoothing with R-like behaviour for small inputs."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 5:
        return y.copy()
    # Pick an odd window that is at most ~n/4 but >= 5.
    win = max(5, (n // 4) | 1)
    if win % 2 == 0:
        win += 1
    if win >= n:
        win = n - 1 if (n - 1) % 2 == 1 else n - 2
        if win < 5:
            return y.copy()
    polyorder = 3 if win > 3 else 2
    try:
        return savgol_filter(y, window_length=win, polyorder=polyorder, mode="interp")
    except ValueError:
        return y.copy()


def _mad_outliers_column(peak: np.ndarray, mad_thresh: float) -> np.ndarray:
    """Port of R's ``MADOutliers``.

    Returns a boolean mask where ``True`` means the bin is flagged.
    """
    smoothed = _smooth_trajectory(peak)
    median_peak = float(np.nanmedian(smoothed))
    mad_peak = float(median_abs_deviation(smoothed, scale="normal", nan_policy="omit"))
    # MAD can be effectively zero when the smoothed trajectory is mostly
    # constant (only a few bins deviate). In that case the R version still
    # flags anything above/below the median because ``median + 0 == median``.
    # We mirror that, using a tiny numerical tolerance to absorb smoothing
    # noise (~1e-15 from savgol on a flat input).
    numeric_eps = 1e-9 * max(1.0, abs(median_peak))
    if mad_peak <= numeric_eps:
        deviation = np.abs(smoothed - median_peak)
        return deviation > numeric_eps
    upper = median_peak + mad_thresh * mad_peak
    lower = median_peak - mad_thresh * mad_peak
    return (smoothed > upper) | (smoothed < lower)


def mad_outlier_method(
    peak_matrix: pd.DataFrame,
    good_mask_in: np.ndarray,
    *,
    mad_thresh: float,
    breaks: Iterable[np.ndarray],
    n_events: int,
    channel_columns: dict[str, list[str]],
) -> dict:
    """Apply the MAD method on the bins still good after the IT step.

    Parameters
    ----------
    peak_matrix
        Full ``(n_bins, n_features)`` peak frame.
    good_mask_in
        Boolean mask of bins surviving IT (or all-True if IT was skipped).
    mad_thresh
        The R ``MAD`` parameter — larger values are less strict.
    breaks
        The overlapping bin index arrays (one per row of ``peak_matrix``).
    n_events
        Total number of events (for computing per-channel contributions).
    channel_columns
        Mapping ``channel_name -> list of peak_matrix columns`` to compute
        the per-channel % contribution.

    Returns
    -------
    dict with keys:

    - ``mad_bins``: bool mask over the **subset** of bins that passed IT —
      ``True`` means flagged by MAD.
    - ``contribution``: ``{channel_name: percentage_removed}``.
    """
    sub = peak_matrix.iloc[good_mask_in].reset_index(drop=True)
    if sub.empty:
        return {
            "mad_bins": np.zeros(0, dtype=bool),
            "contribution": {ch: 0.0 for ch in channel_columns},
        }

    # Per column mask (True -> flagged).
    column_masks: dict[str, np.ndarray] = {}
    for col in sub.columns:
        column_masks[col] = _mad_outliers_column(sub[col].to_numpy(), mad_thresh)

    per_bin_mask = np.zeros(len(sub), dtype=bool)
    for col_mask in column_masks.values():
        per_bin_mask |= col_mask

    breaks = list(breaks)
    passing_bin_indices = np.flatnonzero(good_mask_in)

    def _removed_percentage(col_mask: np.ndarray) -> float:
        if not np.any(col_mask):
            return 0.0
        # Translate mask over the sub-frame back to original bin indices
        abs_bin_idx = passing_bin_indices[col_mask]
        removed_event_idx: set[int] = set()
        for bi in abs_bin_idx:
            removed_event_idx.update(breaks[bi].tolist())
        return 100.0 * (len(removed_event_idx) / n_events)

    contribution: dict[str, float] = {}
    for channel, cols in channel_columns.items():
        if not cols:
            contribution[channel] = 0.0
            continue
        channel_mask = np.zeros(len(sub), dtype=bool)
        for col in cols:
            if col in column_masks:
                channel_mask |= column_masks[col]
        contribution[channel] = round(_removed_percentage(channel_mask), 2)

    return {"mad_bins": per_bin_mask, "contribution": contribution}


def removed_bins_to_cells(
    breaks: Iterable[np.ndarray], bad_bin_mask: np.ndarray, n_events: int
) -> tuple[np.ndarray, np.ndarray]:
    """Translate a per-bin bad mask into a per-event mask.

    Mirrors R's ``RemovedBins``. Returns ``(good_cells, removed_event_idx)``
    where ``good_cells`` is a length-``n_events`` bool array (True = kept).
    """
    breaks = list(breaks)
    removed: set[int] = set()
    for bi in np.flatnonzero(bad_bin_mask):
        removed.update(breaks[bi].tolist())
    good_cells = np.ones(n_events, dtype=bool)
    if removed:
        good_cells[np.fromiter(removed, dtype=np.int64, count=len(removed))] = False
    removed_arr = np.fromiter(sorted(removed), dtype=np.int64, count=len(removed))
    return good_cells, removed_arr
