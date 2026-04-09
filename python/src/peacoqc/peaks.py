"""KDE-based peak detection and per-channel clustering.

Ports ``FindThemPeaks``, ``DetermineAllPeaks``, ``DuplicatePeaks``,
``TooSmallClusters``, and ``ExtractPeakValues`` from the R package.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


def _r_nrd0_bandwidth(x: np.ndarray) -> float:
    """Approximate R's ``stats::bw.nrd0``.

    R's ``density()`` default bandwidth is
    ``0.9 * min(sd, iqr/1.34) * n^(-1/5)`` (Silverman's rule of thumb).
    SciPy's ``gaussian_kde(bw_method='silverman')`` uses a *different*
    formula (``n^(-1/(d+4))``) for 1-D data, so we compute R's value
    ourselves and pass it as a scalar to ``gaussian_kde``.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return 1.0
    sd = float(np.std(x, ddof=1))
    q75, q25 = np.percentile(x, [75, 25])
    iqr = float(q75 - q25)
    hi = sd
    if iqr > 0:
        hi = min(sd, iqr / 1.34)
    if hi <= 0:
        hi = abs(x[0]) if x[0] != 0 else 1.0
    return 0.9 * hi * n ** (-0.2)


def find_peaks_kde(
    channel_data: np.ndarray,
    *,
    remove_zeros: bool = False,
    peak_removal: float = 1 / 3,
) -> np.ndarray | None:
    """Return the positions of detected peaks on a 512-point KDE grid.

    Mirrors :func:`PeacoQC::FindThemPeaks`. A point is a peak if it is a
    local maximum whose density exceeds ``peak_removal * max(density)``.
    Falls back to the global max if no local maxima are found. Returns
    ``None`` if there are fewer than 3 valid data points.
    """
    x = np.asarray(channel_data, dtype=float)
    x = x[~np.isnan(x)]
    if remove_zeros:
        x = x[x != 0]
    if len(x) < 3:
        return None

    try:
        bw = _r_nrd0_bandwidth(x)
        std = float(np.std(x, ddof=1))
        if std <= 0:
            return np.asarray([float(x[0])])
        kde = gaussian_kde(x, bw_method=bw / std)
    except np.linalg.LinAlgError:
        return None

    # R's density() default cuts by 3 * bandwidth outside the data range
    # and uses 512 grid points.
    cut = 3 * bw
    lo = float(x.min()) - cut
    hi = float(x.max()) + cut
    grid = np.linspace(lo, hi, 512)
    dens = kde(grid)

    if np.all(np.isnan(dens)):
        return None

    threshold = peak_removal * float(np.nanmax(dens))
    n = len(dens)
    # Local maxima comparison (indices 1..n-2), matching R's:
    #   dens[2:(n-1)] > dens[1:(n-2)] & dens[2:(n-1)] > dens[3:n]
    #   & dens[2:(n-1)] > peak_removal * max(dens)
    mid = dens[1:-1]
    left = dens[:-2]
    right = dens[2:]
    selection = (mid > left) & (mid > right) & (mid > threshold)
    peak_positions = grid[1:-1][selection]

    if peak_positions.size == 0:
        peak_positions = np.asarray([float(grid[int(np.nanargmax(dens))])])

    return peak_positions


def _dedupe_peaks_for_bin(
    peak_frame_for_bin: pd.DataFrame, medians: dict[int, float]
) -> pd.DataFrame:
    """Port of R's ``DuplicatePeaks``: drop duplicate cluster assignments.

    For each cluster with more than one peak in this bin, drop the peak
    whose distance to the cluster median is largest.
    """
    df = peak_frame_for_bin.copy()
    dups = df["Cluster"][df["Cluster"].duplicated()].unique()
    for cluster in dups:
        mask = df["Cluster"] == cluster
        sub = df[mask]
        dist = np.abs(sub["Peak"].to_numpy() - medians[cluster])
        worst = int(np.argmax(dist))
        drop_label = sub.index[worst]
        df = df.drop(index=drop_label)
    df = df.sort_values("Cluster", kind="stable")
    return df


def _remove_small_clusters(peak_frame: pd.DataFrame) -> pd.DataFrame:
    """Port of R's ``TooSmallClusters``.

    Clusters whose peak count is less than ``nr_bins / 2`` are dropped.
    ``nr_bins`` here is the maximum Bin number (matching R's ``max(Bin)``).
    """
    if peak_frame.empty:
        return peak_frame
    nr_bins = int(peak_frame["Bin"].astype(int).max())
    counts = peak_frame["Cluster"].value_counts()
    to_remove = counts[counts < nr_bins / 2].index
    return peak_frame[~peak_frame["Cluster"].isin(to_remove)].reset_index(drop=True)


def determine_all_peaks(
    channel_data: np.ndarray,
    breaks: Sequence[np.ndarray],
    *,
    remove_zeros: bool = False,
    peak_removal: float = 1 / 3,
    min_nr_bins_peakdetection: float = 10,
) -> pd.DataFrame | None:
    """Compute the per-bin, per-cluster peak frame for a single channel.

    Mirrors :func:`PeacoQC::DetermineAllPeaks`. Returns a DataFrame with
    columns ``Bin`` (1-based integer as string), ``Peak`` (float), and
    ``Cluster`` (integer, 1-based). Returns ``None`` if no peaks were
    found at all.
    """
    channel_data = np.asarray(channel_data, dtype=float)
    full = find_peaks_kde(channel_data, remove_zeros=remove_zeros, peak_removal=peak_removal)
    if full is None or len(full) == 0:
        return None

    per_bin_peaks: dict[int, np.ndarray] = {}
    for i, idx in enumerate(breaks, start=1):
        sub = channel_data[idx]
        peaks = find_peaks_kde(sub, remove_zeros=remove_zeros, peak_removal=peak_removal)
        if peaks is None:
            continue
        per_bin_peaks[i] = np.asarray(peaks, dtype=float)

    if not per_bin_peaks:
        return None

    # Build an initial (Bin, Peak) frame.
    rows = []
    for bin_id, peaks in per_bin_peaks.items():
        for p in peaks:
            rows.append({"Bin": bin_id, "Peak": float(p)})
    peak_frame = pd.DataFrame(rows)

    # Determine the most common peak-count across bins and use those bins
    # to compute per-cluster medians.
    lengths = np.array([len(v) for v in per_bin_peaks.values()], dtype=int)
    unique_counts = np.unique(lengths)
    # R: most_occurring = max(nr_peaks[ count_bins > (min_nr_bins_peakdetection/100)*total_bins ])
    threshold = (min_nr_bins_peakdetection / 100.0) * len(lengths)
    counts_per_value = np.array([int(np.sum(lengths == v)) for v in unique_counts])
    candidates = unique_counts[counts_per_value > threshold]
    if len(candidates) == 0:
        # Fall back to the max peak count if threshold excludes everything.
        most_occurring = int(unique_counts.max())
    else:
        most_occurring = int(candidates.max())

    qualifying_bins = [b for b, p in per_bin_peaks.items() if len(p) == most_occurring]
    if qualifying_bins:
        limited = np.vstack([np.sort(per_bin_peaks[b]) for b in qualifying_bins])
        medians_to_use = np.median(limited, axis=0)
    else:  # pragma: no cover - defensive
        medians_to_use = np.asarray([np.median(peak_frame["Peak"].to_numpy())])

    if len(medians_to_use) > 1:
        medians_dict = {i + 1: float(m) for i, m in enumerate(medians_to_use)}
        # Assign each peak to the nearest cluster median.
        cluster_ids = np.array(
            [
                int(np.argmin(np.abs(p - medians_to_use))) + 1
                for p in peak_frame["Peak"].to_numpy()
            ],
            dtype=int,
        )
        peak_frame["Cluster"] = cluster_ids
        final_medians = medians_dict
    else:
        peak_frame["Cluster"] = 1
        final_medians = {1: float(np.median(peak_frame["Peak"].to_numpy()))}

    # Deduplicate within each bin -> one peak per cluster.
    deduped_parts = []
    for bin_id, sub in peak_frame.groupby("Bin", sort=False):
        deduped_parts.append(_dedupe_peaks_for_bin(sub, final_medians))
    peak_frame = pd.concat(deduped_parts, ignore_index=True)
    peak_frame = _remove_small_clusters(peak_frame)
    if peak_frame.empty:
        return None
    # Cast Bin to string for parity with R's factor, but keep int type for easy
    # arithmetic downstream; store both columns.
    peak_frame["Bin"] = peak_frame["Bin"].astype(int)
    peak_frame["Cluster"] = peak_frame["Cluster"].astype(int)
    return peak_frame.reset_index(drop=True)


def extract_peak_values(peak_frame: pd.DataFrame, n_bins: int) -> dict[int, np.ndarray]:
    """Port of R's ``ExtractPeakValues``.

    Returns a dict ``{cluster_id: values}`` where ``values`` is a
    length-``n_bins`` float vector containing the per-bin peak position or
    the cluster median when no peak exists in that bin.
    """
    out: dict[int, np.ndarray] = {}
    for cluster in sorted(peak_frame["Cluster"].unique()):
        sub = peak_frame[peak_frame["Cluster"] == cluster]
        median_val = float(np.median(sub["Peak"].to_numpy()))
        vec = np.full(n_bins, median_val, dtype=float)
        for _, row in sub.iterrows():
            vec[int(row["Bin"]) - 1] = float(row["Peak"])
        out[int(cluster)] = vec
    return out


def determine_peaks_all_channels(
    X: np.ndarray,
    channel_names: Sequence[str],
    channel_indices: Sequence[int],
    breaks: Sequence[np.ndarray],
    *,
    remove_zeros: bool = False,
    peak_removal: float = 1 / 3,
    min_nr_bins_peakdetection: float = 10,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, list[str]]]:
    """Run :func:`determine_all_peaks` for every requested channel.

    Returns
    -------
    peak_matrix : pandas.DataFrame
        Shape ``(n_bins, total_peak_columns)`` — one column per
        ``(channel, cluster)``. Column names are ``f"{channel}__{cluster}"``.
    per_channel_peaks : dict[str, DataFrame]
        Per-channel frames identical to R's ``results[[channel]]``.
    channel_columns : dict[str, list[str]]
        Mapping from each channel to its columns in ``peak_matrix``.
    """
    n_bins = len(breaks)
    per_channel_peaks: dict[str, pd.DataFrame] = {}
    channel_columns: dict[str, list[str]] = {}
    collected: dict[str, np.ndarray] = {}

    for ch_name, ch_idx in zip(channel_names, channel_indices):
        channel_data = X[:, ch_idx]
        frame = determine_all_peaks(
            channel_data,
            breaks,
            remove_zeros=remove_zeros,
            peak_removal=peak_removal,
            min_nr_bins_peakdetection=min_nr_bins_peakdetection,
        )
        if frame is None:
            channel_columns[ch_name] = []
            continue

        per_channel_peaks[ch_name] = frame
        cluster_values = extract_peak_values(frame, n_bins)
        cols = []
        for cluster_id, values in cluster_values.items():
            col_name = f"{ch_name}__{cluster_id}"
            collected[col_name] = values
            cols.append(col_name)
        channel_columns[ch_name] = cols

    if not collected:
        peak_matrix = pd.DataFrame(index=range(n_bins))
    else:
        peak_matrix = pd.DataFrame(collected, index=range(n_bins))

    return peak_matrix, per_channel_peaks, channel_columns
