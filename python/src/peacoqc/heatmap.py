"""Multi-file QC heatmap.

Port of R's :func:`PeacoQCHeatmap`. Reads a PeacoQC report (CSV written by
:func:`append_row` *or* the R package's tab-separated ``PeacoQC_report.txt``)
and renders a matplotlib heatmap of the removal percentages with parameter
annotations on the left and a directional trend column on the right.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

_REMOVAL_COLS = [
    "% Full analysis",
    "% IT analysis",
    "% MAD analysis",
    "% Consecutive cells",
]

_PARAM_COLS = ["MAD", "IT limit", "Consecutive bins", "Events per bin"]

_TREND_COLORS = {
    "No increasing or decreasing effect": "#26C485",
    "Increasing channel": "#AF3800",
    "Decreasing channel": "#721817",
    "Increasing and decreasing channel": "#A50104",
}


def _read_report(path: Path) -> pd.DataFrame:
    """Read either our CSV or the R-style TSV report."""
    # pandas sniffer handles both with sep=None + engine='python'.
    df = pd.read_csv(path, sep=None, engine="python")
    # R stores "Not_used" as a string, our CSV does too.
    return df.replace("Not_used", np.nan)


def _removal_cmap() -> LinearSegmentedColormap:
    # 0% pale -> 20% yellow -> 100% red (matches R).
    return LinearSegmentedColormap.from_list(
        "peacoqc_removal",
        [(0.0, "#EBEBD3"), (0.2, "#FFD151"), (1.0, "red")],
    )


def peaco_qc_heatmap(
    report_path: str | os.PathLike[str],
    *,
    show_values: bool = True,
    show_row_names: bool = True,
    latest_tests: bool = False,
    title: str = "PeacoQC report",
    output_path: str | os.PathLike[str] | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Render the QC heatmap for one or more files.

    Parameters
    ----------
    report_path
        Path to a PeacoQC report (CSV or the R TSV format).
    show_values
        If True, overlay the percentage value on each cell.
    show_row_names
        If True, label rows with the filename.
    latest_tests
        If True, keep only the latest row per filename.
    title
        Heatmap title.
    output_path
        Optional path to save the figure to.
    figsize
        Optional explicit figure size.
    """
    path = Path(report_path)
    if not path.exists():
        raise FileNotFoundError(f"Report not found: {path}")

    df = _read_report(path)

    missing = [c for c in _REMOVAL_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Report is missing required columns: {missing}. "
            f"Columns present: {list(df.columns)}"
        )

    if latest_tests:
        df = df.drop_duplicates(subset="Filename", keep="last").reset_index(drop=True)

    # Numeric coercion for the removal columns
    removal = df[_REMOVAL_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    n_rows = len(df)
    if figsize is None:
        fig_w = max(6.5, 2.0 + 0.8 * len(_REMOVAL_COLS) + 2.5)
        fig_h = max(2.0, 0.35 * max(n_rows, 4) + 1.5)
        figsize = (fig_w, fig_h)

    fig = plt.figure(figsize=figsize)
    # Grid: [left annotations | heatmap | right trend column]
    gs = fig.add_gridspec(
        nrows=1,
        ncols=3,
        width_ratios=[len(_PARAM_COLS), len(_REMOVAL_COLS), 1],
        wspace=0.1,
    )

    # ---- left annotation (parameters) -------------------------------------
    ax_left = fig.add_subplot(gs[0, 0])
    param_data = (
        df.reindex(columns=_PARAM_COLS)
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )
    # Normalise per-column to 0..1 for coloring.
    param_norm = np.zeros_like(param_data)
    for j in range(param_data.shape[1]):
        col = param_data[:, j]
        lo = np.nanmin(col)
        hi = np.nanmax(col)
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            param_norm[:, j] = (col - lo) / (hi - lo)
    ax_left.imshow(param_norm, aspect="auto", cmap="Purples", vmin=0, vmax=1)
    ax_left.set_xticks(range(len(_PARAM_COLS)))
    ax_left.set_xticklabels(_PARAM_COLS, rotation=45, ha="right", fontsize=8)
    ax_left.set_yticks([])
    if show_values:
        for i in range(param_data.shape[0]):
            for j in range(param_data.shape[1]):
                v = param_data[i, j]
                if np.isfinite(v):
                    ax_left.text(j, i, f"{v:g}", ha="center", va="center", fontsize=7)

    # ---- main heatmap ------------------------------------------------------
    ax_main = fig.add_subplot(gs[0, 1], sharey=ax_left)
    cmap = _removal_cmap()
    img = ax_main.imshow(removal, aspect="auto", cmap=cmap, vmin=0, vmax=100)
    ax_main.set_xticks(range(len(_REMOVAL_COLS)))
    ax_main.set_xticklabels(_REMOVAL_COLS, rotation=45, ha="right", fontsize=8)
    if show_row_names:
        ax_main.set_yticks(range(n_rows))
        ax_main.set_yticklabels(df["Filename"].astype(str).tolist(), fontsize=8)
        ax_main.tick_params(axis="y", which="both", left=False, labelleft=True)
    else:
        ax_main.set_yticks([])

    if show_values:
        for i in range(removal.shape[0]):
            for j in range(removal.shape[1]):
                v = removal[i, j]
                if np.isfinite(v):
                    ax_main.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7)

    ax_main.set_title(title, fontweight="bold")

    # ---- right trend annotation -------------------------------------------
    ax_right = fig.add_subplot(gs[0, 2], sharey=ax_main)
    trends = df.get("Increasing/Decreasing channel", pd.Series([""] * n_rows))
    trend_rgb = np.ones((n_rows, 1, 3), dtype=float)
    for i, label in enumerate(trends.astype(str).tolist()):
        color = _TREND_COLORS.get(label, "#cccccc")
        rgb = plt.matplotlib.colors.to_rgb(color)
        trend_rgb[i, 0, :] = rgb
    ax_right.imshow(trend_rgb, aspect="auto")
    ax_right.set_xticks([0])
    ax_right.set_xticklabels(["Incr/Decr"], rotation=45, ha="right", fontsize=8)
    ax_right.set_yticks([])

    # ---- colorbar ----------------------------------------------------------
    cbar = fig.colorbar(
        img, ax=[ax_left, ax_main, ax_right], orientation="horizontal",
        fraction=0.04, pad=0.12,
    )
    cbar.set_label("Removed percentage")

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig
