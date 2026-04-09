"""Top-level :func:`peaco_qc` orchestration.

Equivalent of R's ``PeacoQC()`` entry point: adaptive binning, per-channel
peak detection, isolation-forest outlier detection, MAD filtering, and
consecutive-bin cleanup, producing a :class:`PeacoQCResult`.
"""

from __future__ import annotations

import os
import warnings
from typing import Literal, Sequence

import anndata as ad
import numpy as np

from ._utils import append_original_id, as_dense, filename_of, resolve_channels
from .binning import find_events_per_bin, make_breaks
from .consecutive import remove_short_true_runs
from .outliers import (
    isolation_tree_outliers,
    mad_outlier_method,
    removed_bins_to_cells,
)
from .peaks import determine_peaks_all_channels
from .report import append_row
from .results import PeacoQCResult
from .signal_stability import find_increasing_decreasing_channels


def peaco_qc(
    adata: ad.AnnData,
    channels: Sequence[int | str],
    *,
    determine_good_cells: Literal["all", "IT", "MAD"] | bool = "all",
    min_cells: int = 150,
    max_bins: int = 500,
    step: int = 500,
    events_per_bin: int | None = None,
    mad: float = 6,
    it_limit: float = 0.6,
    consecutive_bins: int = 5,
    remove_zeros: bool = False,
    force_it: int = 150,
    peak_removal: float = 1 / 3,
    min_nr_bins_peakdetection: float = 10,
    time_channel: str | None = "Time",
    report_path: str | os.PathLike[str] | None = None,
    random_state: int = 0,
) -> PeacoQCResult:
    """Run the full PeacoQC pipeline on an :class:`anndata.AnnData`.

    Parameters mirror the R function; see the upstream
    `documentation <https://www.bioconductor.org/packages/release/bioc/manuals/PeacoQC/man/PeacoQC.pdf>`_
    for details. Differences from the R port:

    - ``output_directory``/``save_fcs``/``plot``/``name_directory``/``suffix_fcs``
      are omitted — use :func:`plot_peaco_qc` and :func:`write_fcs` on the
      returned :class:`PeacoQCResult` instead.
    - ``determine_good_cells`` accepts the Python-style values ``"all"``,
      ``"IT"``, ``"MAD"``, or ``False``.
    - The isolation tree is replaced by
      :class:`sklearn.ensemble.IsolationForest` (see README).
    """
    if not isinstance(adata, ad.AnnData):
        raise TypeError("adata should be an AnnData object.")
    channel_names = resolve_channels(adata, channels)
    if determine_good_cells not in ("all", "IT", "MAD", False):
        raise ValueError(
            "determine_good_cells must be one of 'all', 'IT', 'MAD', or False."
        )

    all_var_names = list(adata.var_names)
    channel_indices = [all_var_names.index(n) for n in channel_names]
    X = as_dense(adata.X).astype(float, copy=False)
    n_events = X.shape[0]

    # ---- time channel sanity check (warn only) -----------------------------
    if time_channel is not None:
        needle = time_channel.lower()
        time_idx = next(
            (i for i, n in enumerate(all_var_names) if needle in str(n).lower()),
            None,
        )
        if time_idx is not None:
            t = X[:, time_idx]
            if np.any(np.diff(t) < 0):
                warnings.warn(
                    "There is an inconsistency in the time channel. "
                    "It seems the events are not ordered by time.",
                    stacklevel=2,
                )

    if n_events < 500:
        warnings.warn(
            "There are less than 500 cells available. This may be insufficient "
            "for robust IT/MAD analysis.",
            stacklevel=2,
        )

    # ---- adaptive bin size -------------------------------------------------
    if events_per_bin is None:
        events_per_bin = find_events_per_bin(
            n_events,
            values=X[:, channel_indices] if remove_zeros else None,
            remove_zeros=remove_zeros,
            min_cells=min_cells,
            max_bins=max_bins,
            step=step,
        )
    breaks, events_per_bin = make_breaks(events_per_bin, n_events)
    nr_bins = len(breaks)

    # ---- signal stability check --------------------------------------------
    weird = find_increasing_decreasing_channels(
        X, channel_names, channel_indices, breaks
    )

    # ---- peak detection ----------------------------------------------------
    peak_matrix, per_channel_peaks, channel_columns = determine_peaks_all_channels(
        X,
        channel_names,
        channel_indices,
        breaks,
        remove_zeros=remove_zeros,
        peak_removal=peak_removal,
        min_nr_bins_peakdetection=min_nr_bins_peakdetection,
    )

    # ---- IT step -----------------------------------------------------------
    good_mask = np.ones(nr_bins, dtype=bool)
    it_percentage: float | None = None
    outlier_it_cells = np.zeros(n_events, dtype=bool)
    it_info: dict | None = None

    run_it = determine_good_cells in ("all", "IT")
    if run_it:
        if nr_bins >= force_it and peak_matrix.shape[1] > 0:
            it_good, it_info = isolation_tree_outliers(
                peak_matrix, it_limit=it_limit, random_state=random_state
            )
            good_mask = it_good
            bad_bin_mask = ~it_good
            it_good_cells, _ = removed_bins_to_cells(breaks, bad_bin_mask, n_events)
            outlier_it_cells = ~it_good_cells
            it_percentage = float(outlier_it_cells.mean() * 100.0)
        else:
            warnings.warn(
                "There are not enough bins for a robust isolation-tree analysis.",
                stacklevel=2,
            )
            it_percentage = None

    # ---- MAD step ----------------------------------------------------------
    mad_percentage: float | None = None
    outlier_mad_cells = np.zeros(n_events, dtype=bool)
    mad_contribution: dict[str, float] = {ch: 0.0 for ch in channel_names}

    run_mad = determine_good_cells in ("all", "MAD")
    if run_mad and peak_matrix.shape[1] > 0:
        mad_out = mad_outlier_method(
            peak_matrix,
            good_mask_in=good_mask,
            mad_thresh=float(mad),
            breaks=breaks,
            n_events=n_events,
            channel_columns=channel_columns,
        )
        mad_bins_subset = mad_out["mad_bins"]
        mad_contribution = mad_out["contribution"]

        # Translate subset mask back to a full-length bin mask.
        full_mad_bins = np.zeros(nr_bins, dtype=bool)
        full_mad_bins[np.flatnonzero(good_mask)] = mad_bins_subset
        mad_cells_good, _ = removed_bins_to_cells(
            breaks, full_mad_bins, n_events
        )
        outlier_mad_cells = ~mad_cells_good
        mad_percentage = float(outlier_mad_cells.mean() * 100.0)
        # Update good_mask: flagged-by-MAD bins become bad.
        good_mask[good_mask] = ~mad_bins_subset

    # ---- consecutive-bin cleanup ------------------------------------------
    consecutive_cells_mask = np.zeros(n_events, dtype=bool)
    consecutive_percentage = 0.0
    if determine_good_cells in ("all", "IT", "MAD") and peak_matrix.shape[1] > 0:
        new_good_mask = remove_short_true_runs(good_mask, consecutive_bins)
        consecutive_bin_flip = good_mask & ~new_good_mask
        if np.any(consecutive_bin_flip):
            cons_good_cells, _ = removed_bins_to_cells(
                breaks, consecutive_bin_flip, n_events
            )
            consecutive_cells_mask = ~cons_good_cells
            consecutive_percentage = float(consecutive_cells_mask.mean() * 100.0)
        good_mask = new_good_mask

    # ---- finalise ----------------------------------------------------------
    if determine_good_cells in ("all", "IT", "MAD") and peak_matrix.shape[1] > 0:
        good_cells_full, _ = removed_bins_to_cells(breaks, ~good_mask, n_events)
    else:
        good_cells_full = np.ones(n_events, dtype=bool)

    percentage_removed = float((1.0 - good_cells_full.mean()) * 100.0)
    if percentage_removed > 70:
        warnings.warn(
            f"More than 70% was removed from file "
            f"{filename_of(adata) or '?'}.",
            stacklevel=2,
        )

    kept_idx = np.where(good_cells_full)[0]
    filtered = adata[kept_idx].copy()
    if "Original_ID" in adata.obs.columns:
        filtered.obs["Original_ID"] = np.asarray(
            adata.obs["Original_ID"].values[kept_idx]
        )
    else:
        append_original_id(filtered, kept_idx)

    analysis_label = (
        "all"
        if determine_good_cells == "all"
        else ("IT" if determine_good_cells == "IT"
              else ("MAD" if determine_good_cells == "MAD" else "none"))
    )

    parameters = {
        "MAD": mad,
        "IT_limit": it_limit,
        "consecutive_bins": consecutive_bins,
        "events_per_bin": events_per_bin,
        "min_cells": min_cells,
        "max_bins": max_bins,
        "step": step,
        "peak_removal": peak_removal,
        "min_nr_bins_peakdetection": min_nr_bins_peakdetection,
        "remove_zeros": remove_zeros,
        "force_it": force_it,
        "random_state": random_state,
    }

    result = PeacoQCResult(
        adata=filtered,
        good_cells=good_cells_full,
        outlier_it=outlier_it_cells,
        outlier_mad=outlier_mad_cells,
        consecutive_cells=consecutive_cells_mask,
        percentage_removed=percentage_removed,
        it_percentage=it_percentage,
        mad_percentage=mad_percentage,
        consecutive_percentage=consecutive_percentage,
        peaks=per_channel_peaks,
        peak_matrix=peak_matrix,
        breaks=breaks,
        weird_channels=weird,
        events_per_bin=events_per_bin,
        nr_bins=nr_bins,
        analysis=analysis_label,
        parameters=parameters,
        filename=filename_of(adata),
        it_info=it_info,
        mad_contribution=mad_contribution,
    )

    # ---- optional CSV report ----------------------------------------------
    if report_path is not None and determine_good_cells in ("all", "IT", "MAD"):
        append_row(
            report_path,
            {
                "Filename": result.filename or "",
                "Nr. Measurements before cleaning": n_events,
                "Nr. Measurements after cleaning": int(good_cells_full.sum()),
                "% Full analysis": percentage_removed,
                "Analysis by": analysis_label,
                "% IT analysis": it_percentage,
                "% MAD analysis": mad_percentage,
                "% Consecutive cells": consecutive_percentage,
                "MAD": mad,
                "IT limit": it_limit,
                "Consecutive bins": consecutive_bins,
                "Events per bin": events_per_bin,
                "Increasing/Decreasing channel": weird["label"],
            },
        )

    return result
