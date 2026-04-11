"""Outlier detection on the per-bin peak matrix.

- :func:`isolation_tree_outliers` uses a custom SD-based isolation tree
  (ported from the R package's ``isolationTreeSD``) by default. The tree
  uses variance-reduction ("gain") splitting on each column to separate
  anomalous bins.  Set ``method="sklearn"`` to use
  :class:`sklearn.ensemble.IsolationForest` instead (the pre-v0.2 default).
- :func:`mad_outlier_method` smooths each peak trajectory with a cubic
  smoothing spline matching R's ``stats::smooth.spline(spar=0.5)`` and flags
  bins outside a ``median +/- MAD * mad`` window.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.interpolate import make_smoothing_spline
from scipy.stats import median_abs_deviation


# ---------------------------------------------------------------------------
# SD-based isolation tree — faithful port of R's isolationTreeSD
# ---------------------------------------------------------------------------


def _avg_path_length(n: int) -> float:
    """Average path length for *n* data points (R's ``avgPL``)."""
    if n <= 1:
        return 0.0
    return 2.0 * (math.log(n - 1) + 0.5772156649) - 2.0 * (n - 1) / n


def _best_gain_for_column(x_sorted: np.ndarray) -> tuple[float, float | None]:
    """Return ``(best_gain, split_value)`` for one sorted column.

    The gain at each split point is

    .. math::
        \\text{gain} = \\frac{\\text{base\\_sd} - \\frac{sd_1 + sd_2}{2}}
                             {\\text{base\\_sd}}

    following R's convention that ``sd_1 = 0`` when the left partition has
    one element, and ``sd_2 = 0`` when the right partition has one element
    (``else if``).

    Among split points with equal gain within a column, the *last* one wins
    (R uses ``>=``).
    """
    n = len(x_sorted)
    if n < 2:
        return 0.0, None
    base_sd = float(np.std(x_sorted, ddof=1))
    if base_sd == 0.0:
        return 0.0, None

    # Cumulative sums for efficient running-SD computation.
    cs = np.cumsum(x_sorted)
    cs2 = np.cumsum(x_sorted ** 2)

    counts_l = np.arange(1, n, dtype=float)
    sum_l = cs[: n - 1]
    ssq_l = cs2[: n - 1]
    counts_r = float(n) - counts_l
    sum_r = cs[-1] - sum_l
    ssq_r = cs2[-1] - ssq_l

    # Bessel-corrected SD — avoid division-by-zero for single-element
    # partitions by clamping the denominator to 1.
    denom_l = np.maximum(counts_l - 1, 1)
    sd_l = np.sqrt(np.maximum(ssq_l - sum_l ** 2 / counts_l, 0.0) / denom_l)
    denom_r = np.maximum(counts_r - 1, 1)
    sd_r = np.sqrt(np.maximum(ssq_r - sum_r ** 2 / counts_r, 0.0) / denom_r)

    # R overrides: ``if (i == 1) sd_1 <- 0 else if (i == n-1) sd_2 <- 0``
    sd_l[0] = 0.0
    sd_r[-1] = 0.0

    gains = (base_sd - (sd_l + sd_r) / 2.0) / base_sd

    valid = np.isfinite(gains)
    if not valid.any():
        return 0.0, None

    gains_clean = np.where(valid, gains, -np.inf)
    max_gain = float(gains_clean.max())
    # Last index with the maximum gain (R uses ``>=`` so later splits win).
    idx = int(np.where(gains_clean == max_gain)[0][-1])
    return max_gain, float(x_sorted[idx])


def _isolation_tree_sd(
    x: np.ndarray,
    column_names: list[str],
    gain_limit: float,
) -> tuple[np.ndarray, dict]:
    """Port of R's ``isolationTreeSD``.

    Builds a single deterministic tree using SD-gain splitting, then returns
    the selection mask of the leaf with the most data points (the "inlier"
    cluster).
    """
    n_rows, n_cols = x.shape
    if n_rows == 0:
        return np.ones(0, dtype=bool), {
            "split_columns": [],
            "anomaly_scores": np.zeros(0, dtype=float),
        }
    max_depth = int(math.ceil(math.log2(max(n_rows, 2))))

    # Node storage as parallel lists.
    depth = [0]
    to_split = [True]
    path_length: list[float | None] = [None]
    split_col: list[int | None] = [None]
    split_val: list[float | None] = [None]

    # selection[i] is a length-n_rows boolean mask for node i.
    selection: list[np.ndarray] = [np.ones(n_rows, dtype=bool)]

    while True:
        pending = [i for i, ts in enumerate(to_split) if ts]
        if not pending:
            break
        nid = pending[0]
        rows = np.flatnonzero(selection[nid])
        n_pts = len(rows)

        if n_pts > 3 and depth[nid] < max_depth:
            best_gain = gain_limit
            best_col: int | None = None
            best_val: float | None = None

            for col in range(n_cols):
                x_col = np.sort(x[rows, col])
                col_gain, col_val = _best_gain_for_column(x_col)
                # R: ``if(gain_max_col > gain_max)`` — strict ``>``
                if col_gain > best_gain:
                    best_gain = col_gain
                    best_val = col_val
                    best_col = col

            if best_val is not None:
                left_mask = selection[nid] & (x[:, best_col] <= best_val)
                right_mask = selection[nid] & (x[:, best_col] > best_val)

                # Degenerate split — all points go to one side.
                if int(left_mask.sum()) == n_pts or int(right_mask.sum()) == n_pts:
                    to_split[nid] = False
                    path_length[nid] = _avg_path_length(n_pts) + depth[nid]
                    continue

                split_col[nid] = best_col
                split_val[nid] = best_val
                to_split[nid] = False

                # R: ``gain_limit <- gain_max`` — global update.
                gain_limit = best_gain

                new_depth = depth[nid] + 1
                for mask in (left_mask, right_mask):
                    depth.append(new_depth)
                    to_split.append(True)
                    path_length.append(None)
                    split_col.append(None)
                    split_val.append(None)
                    selection.append(mask)
            else:
                to_split[nid] = False
                path_length[nid] = _avg_path_length(n_pts) + depth[nid]
        else:
            to_split[nid] = False
            path_length[nid] = _avg_path_length(n_pts) + depth[nid]

    # Find the leaf (path_length is set) with the most data points.
    best_leaf = 0
    best_count = 0
    for nid in range(len(depth)):
        if path_length[nid] is not None:
            count = int(selection[nid].sum())
            if count > best_count:
                best_count = count
                best_leaf = nid
    good_mask = selection[best_leaf].copy()

    # Collect the column names actually used for splits.
    col_set: set[int] = set()
    for c in split_col:
        if c is not None:
            col_set.add(c)
    split_columns = [column_names[c] for c in sorted(col_set)]

    # Per-data-point anomaly scores (R: 2^(-path_length / avgPL(total))).
    total_dp = sum(int(s.sum()) for s in selection)
    avg_pl_total = _avg_path_length(total_dp)
    anomaly_scores = np.zeros(n_rows, dtype=float)
    for nid in range(len(depth)):
        if path_length[nid] is not None:
            pl = path_length[nid]
            score = (
                2.0 ** (-(pl) / avg_pl_total) if avg_pl_total > 0 else 0.0
            )
            anomaly_scores[selection[nid]] = score

    return good_mask, {
        "split_columns": split_columns,
        "anomaly_scores": anomaly_scores,
    }


# ---------------------------------------------------------------------------
# sklearn IsolationForest path (retained as method="sklearn" fallback)
# ---------------------------------------------------------------------------


def _isolation_forest_sklearn(
    peak_matrix: pd.DataFrame,
    *,
    it_limit: float = 0.6,
    random_state: int = 0,
    n_estimators: int = 100,
) -> tuple[np.ndarray, dict]:
    """IsolationForest-based outlier detection (pre-v0.2 default)."""
    from sklearn.ensemble import IsolationForest

    values = peak_matrix.to_numpy()
    contamination = max(min(1.0 - float(it_limit), 0.5), 1e-4)
    forest = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
    )
    forest.fit(values)
    preds = forest.predict(values)
    good_mask = preds == 1

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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def isolation_tree_outliers(
    peak_matrix: pd.DataFrame,
    *,
    it_limit: float = 0.6,
    random_state: int = 0,
    n_estimators: int = 100,
    method: str = "sd_tree",
) -> tuple[np.ndarray, dict]:
    """Return a per-bin boolean "good" mask using an isolation tree.

    ``it_limit`` is the R parameter. In the original R package it is the
    "gain" threshold of the custom SD-based tree. When ``method="sd_tree"``
    (the default) the tree uses this directly. When ``method="sklearn"`` it
    is mapped to ``contamination = 1 - it_limit``.

    Parameters
    ----------
    peak_matrix
        ``(n_bins, n_features)`` frame of per-bin peak positions.
    it_limit
        R-style gain threshold; clamped to ``[0.0, 1.0]``.
    random_state
        RNG seed for reproducibility (``method="sklearn"`` only).
    n_estimators
        Number of trees (``method="sklearn"`` only).
    method
        ``"sd_tree"`` (default, matches R) or ``"sklearn"``
        (uses :class:`sklearn.ensemble.IsolationForest`).

    Returns
    -------
    good_mask : np.ndarray[bool]
        Length ``n_bins``. ``True`` means the bin is kept.
    info : dict
        ``{"split_columns": [...], "anomaly_scores": np.ndarray}``.
    """
    if peak_matrix.shape[1] == 0:
        return np.ones(peak_matrix.shape[0], dtype=bool), {
            "split_columns": [],
            "anomaly_scores": np.zeros(peak_matrix.shape[0], dtype=float),
        }

    if peak_matrix.shape[0] < 4:
        return np.ones(peak_matrix.shape[0], dtype=bool), {
            "split_columns": [],
            "anomaly_scores": np.zeros(peak_matrix.shape[0], dtype=float),
        }

    if method == "sd_tree":
        return _isolation_tree_sd(
            peak_matrix.to_numpy(dtype=float),
            list(peak_matrix.columns),
            gain_limit=float(it_limit),
        )
    elif method == "sklearn":
        return _isolation_forest_sklearn(
            peak_matrix,
            it_limit=it_limit,
            random_state=random_state,
            n_estimators=n_estimators,
        )
    else:
        raise ValueError(
            f"Unknown method {method!r}. Use 'sd_tree' or 'sklearn'."
        )


# ---------------------------------------------------------------------------
# MAD smoothing & outlier detection
# ---------------------------------------------------------------------------


def _smooth_trajectory(y: np.ndarray) -> np.ndarray:
    """Cubic smoothing spline matching R's ``smooth.spline(spar=0.5)``.

    R's ``spar``-to-``lambda`` conversion is
    ``lambda = r * 256^(3 * spar - 1)`` where ``r`` is the penalty-matrix
    trace ratio. For equally-spaced knots with spacing ``h = 1/(n-1)``
    (after R's normalisation to [0, 1]), ``r = h^3 = 1/(n-1)^3``.
    With ``spar=0.5`` the multiplier is ``256^0.5 = 16``, so
    ``lambda_R = h^3 * 16 = 16 / (n-1)^3``.  Converting from R's
    normalised-x scale to scipy's original scale multiplies by
    ``(n-1)^3``, giving ``lam_scipy = 16`` — constant for all ``n``.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 5:
        return y.copy()
    finite = np.isfinite(y)
    if not finite.all():
        if finite.sum() < 5:
            return y.copy()
        x_all = np.arange(n, dtype=float)
        x_fit = x_all[finite]
        y_fit = y[finite]
    else:
        x_fit = np.arange(n, dtype=float)
        y_fit = y
    # GCV is ill-defined on a flat trajectory.
    if np.ptp(y_fit) <= 0.0:
        return y.copy()

    # R's spar=0.5 ≈ scipy lam=16 for equally-spaced integer x values.
    # Derivation: r = h^3 with h=1/(n-1), lambda_R = r*16 = 16/(n-1)^3,
    # lam_scipy = lambda_R * (n-1)^3 = 16.  The x_range^3 factors cancel.
    lam = 16.0

    try:
        spline = make_smoothing_spline(x_fit, y_fit, lam=lam)
        return np.asarray(spline(np.arange(n, dtype=float)), dtype=float)
    except (ValueError, np.linalg.LinAlgError):
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
    # noise (~1e-12 from the smoothing spline on a near-flat input).
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
