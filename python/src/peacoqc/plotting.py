"""Multi-panel QC overview plot.

Port of R's :func:`PlotPeacoQC`. Produces a matplotlib :class:`~matplotlib.figure.Figure`
with an event-rate histogram (if a time channel is present) and one
scatter plot per requested channel. Background rectangles mark the
different QC regions; peak trajectories are overlaid on the scatter.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Optional, Sequence

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from ._utils import as_dense, resolve_channels, time_channel_name
from .binning import split_with_overlap_mids
from .results import PeacoQCResult

_COLORS = {
    "good": (1.0, 1.0, 1.0),
    "IT": "indianred",
    "MAD": "mediumpurple",
    "consecutive": "plum",
}


def _make_overview_blocks(result: PeacoQCResult) -> list[tuple[int, int, str]]:
    """Compute ``(start, end, label)`` tuples over event indices.

    The label is one of ``'good'``, ``'IT'``, ``'MAD'``, or ``'consecutive'``.
    """
    n = len(result.good_cells)
    category = np.empty(n, dtype=object)
    category[:] = "good"
    # Order matches R: IT first, then consecutive, then MAD.
    if result.outlier_it is not None:
        category[result.outlier_it] = "IT"
    if result.consecutive_cells is not None:
        category[result.consecutive_cells] = "consecutive"
    if result.outlier_mad is not None:
        category[result.outlier_mad] = "MAD"

    blocks: list[tuple[int, int, str]] = []
    if n == 0:
        return blocks
    start = 0
    for i in range(1, n):
        if category[i] != category[start]:
            blocks.append((start, i, category[start]))
            start = i
    blocks.append((start, n, category[start]))
    return blocks


def _fill_background(ax, blocks, x_for_index=None) -> None:
    """Shade QC blocks across the full y-range of ``ax``.

    If ``x_for_index`` is provided, it is used to translate event indices
    to x-axis coordinates (e.g. the time channel values).
    """
    for start, end, label in blocks:
        if label == "good":
            continue
        color = _COLORS.get(label, "lightgray")
        if x_for_index is None:
            x0, x1 = start, end
        else:
            x0 = x_for_index[min(start, len(x_for_index) - 1)]
            x1 = x_for_index[min(end - 1, len(x_for_index) - 1)]
        ax.axvspan(x0, x1, color=color, alpha=0.4, linewidth=0)


def _contribution_label(
    result: PeacoQCResult, channel: str, marker: str | None
) -> str:
    base = marker or channel
    contributions: list[str] = []
    if result.it_info is not None:
        split_cols = result.it_info.get("split_columns", [])
        if any(c.startswith(f"{channel}__") for c in split_cols):
            contributions.append("IT: +")
    mad_pct = result.mad_contribution.get(channel, 0.0) if result.mad_contribution else 0.0
    if mad_pct > 0:
        contributions.append(f"MAD: {mad_pct:g}%")
    text = base
    if contributions:
        text = f"{base}\n{' '.join(contributions)}"
    if channel in result.weird_channels.get("increasing", []):
        text += "\nWARNING: Increasing channel."
    elif channel in result.weird_channels.get("decreasing", []):
        text += "\nWARNING: Decreasing channel."
    return text


def plot_peaco_qc(
    adata: ad.AnnData,
    result: PeacoQCResult,
    channels: Optional[Sequence[int | str]] = None,
    *,
    output_path: str | os.PathLike[str] | None = None,
    display_cells: int = 2000,
    time_channel: str | None = "Time",
    title: str | None = None,
    random_state: int = 0,
) -> Figure:
    """Render the QC overview figure.

    Parameters
    ----------
    adata
        The :class:`anndata.AnnData` that was passed to :func:`peaco_qc`.
    result
        The :class:`PeacoQCResult` returned by :func:`peaco_qc`.
    channels
        Optional list of channels to display. Defaults to the channels
        present in ``result.peaks``.
    output_path
        If given, the figure is saved to this path (``.png`` suggested).
    display_cells
        Number of randomly-sampled events plotted per channel.
    time_channel
        Name (or substring) of the time channel for the top event-rate plot.
    title
        Optional title for the event-rate panel. Defaults to the QC summary.
    random_state
        Seed for subsampling event indices.
    """
    if channels is None:
        channels = list(result.peaks.keys()) or list(adata.var_names)
    channel_names = resolve_channels(adata, channels)

    X = as_dense(adata.X)
    n_events = X.shape[0]
    if n_events == 0:
        raise ValueError("adata has zero events.")

    rng = np.random.default_rng(random_state)
    display_n = min(display_cells, n_events)
    sample_idx = np.sort(rng.choice(n_events, size=display_n, replace=False))

    blocks = _make_overview_blocks(result)
    time_col_name = time_channel_name(adata, time_channel) if time_channel else None
    time_values = None
    if time_col_name is not None:
        time_values = X[:, list(adata.var_names).index(time_col_name)]

    n_channels = len(channel_names)
    n_panels = n_channels + (1 if time_values is not None else 0)
    if n_panels == 0:
        raise ValueError("Nothing to plot.")
    n_row = max(1, int(math.floor(math.sqrt(n_panels + 1))))
    n_col = int(math.ceil(n_panels / n_row))

    fig, axes = plt.subplots(
        n_row, n_col, figsize=(5 * n_col, 3 * n_row), squeeze=False
    )
    flat_axes = [ax for row in axes for ax in row]

    axis_iter = iter(flat_axes)

    # ---- event-rate panel --------------------------------------------------
    if time_values is not None:
        ax = next(axis_iter)
        bin_width = max((time_values.max() - time_values.min()) / 200.0, 1.0)
        hist_edges = np.arange(
            time_values.min(), time_values.max() + bin_width, bin_width
        )
        counts, edges = np.histogram(time_values, bins=hist_edges)
        centers = 0.5 * (edges[:-1] + edges[1:])
        _fill_background(ax, blocks, x_for_index=time_values)
        ax.plot(centers, counts, "o", markersize=2, color="black")
        ax.set_title(title if title is not None else _default_title(result), fontsize=10)
        ax.set_xlabel("Time")
        ax.set_ylabel("Nr of cells")

    # ---- per-channel panels ------------------------------------------------
    mid_breaks = split_with_overlap_mids(
        n_events, result.events_per_bin, math.ceil(result.events_per_bin / 2)
    )
    # Ensure we have at least as many mid_breaks as bins by padding with the
    # last event if needed (mirrors ``MakeMidBreaks``).
    if n_events % result.events_per_bin != 0:
        mid_breaks = list(mid_breaks) + [n_events - 1]
    mid_breaks = np.asarray(mid_breaks[: result.nr_bins], dtype=int)

    for ch_name in channel_names:
        ax = next(axis_iter)
        idx = list(adata.var_names).index(ch_name)
        values = X[sample_idx, idx]
        _fill_background(ax, blocks)
        ax.scatter(sample_idx, values, s=1, color="dimgray", alpha=0.6)

        # Peak trajectories
        peak_frame = result.peaks.get(ch_name)
        if peak_frame is not None and not peak_frame.empty:
            for cluster_id, sub in peak_frame.groupby("Cluster"):
                bin_idx = sub["Bin"].to_numpy(dtype=int) - 1
                safe_idx = np.clip(bin_idx, 0, len(mid_breaks) - 1)
                xs = mid_breaks[safe_idx]
                ys = sub["Peak"].to_numpy()
                order = np.argsort(xs)
                ax.plot(xs[order], ys[order], color="gray", linewidth=1)

        marker = None
        if "marker" in adata.var.columns:
            m = adata.var.loc[ch_name, "marker"]
            marker = None if (m is None or str(m) in ("", "nan")) else str(m)
        ax.set_title(_contribution_label(result, ch_name, marker), fontsize=9)
        ax.set_xlabel("Cells")
        ax.set_ylabel("Value")

    # ---- blank any leftover axes ------------------------------------------
    for ax in axis_iter:
        ax.axis("off")

    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def _default_title(result: PeacoQCResult) -> str:
    name = result.filename or "PeacoQC"
    return f"{name}: {result.percentage_removed:.2f}% removed"
