"""Typed result object returned by :func:`peacoqc.peaco_qc`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import anndata as ad
import numpy as np
import pandas as pd


@dataclass
class PeacoQCResult:
    """Container for the output of :func:`peacoqc.peaco_qc`."""

    adata: ad.AnnData
    """The filtered :class:`anndata.AnnData` (slice of the input)."""

    good_cells: np.ndarray
    """Bool mask of length ``n_events_in``. ``True`` = kept."""

    outlier_it: np.ndarray
    """Bool mask: ``True`` = removed by the IT step."""

    outlier_mad: np.ndarray
    """Bool mask: ``True`` = removed by the MAD step."""

    consecutive_cells: np.ndarray
    """Bool mask: ``True`` = removed by the consecutive-bin filter."""

    percentage_removed: float
    it_percentage: Optional[float]
    mad_percentage: Optional[float]
    consecutive_percentage: float

    peaks: dict[str, pd.DataFrame]
    """Per-channel peak frames (``Bin``, ``Peak``, ``Cluster``)."""

    peak_matrix: pd.DataFrame
    """``(n_bins, total_peak_columns)`` frame."""

    breaks: list[np.ndarray]
    """Overlapping bin index arrays."""

    weird_channels: dict[str, Any]
    """``{'increasing': [...], 'decreasing': [...], 'label': str}``."""

    events_per_bin: int
    nr_bins: int
    analysis: str
    """Which method(s) were applied: ``'all'``, ``'IT'``, ``'MAD'``, or ``'none'``."""

    parameters: dict[str, Any]
    filename: Optional[str]
    it_info: Optional[dict[str, Any]]
    mad_contribution: dict[str, float] = field(default_factory=dict)

    def annotate(self, original: ad.AnnData) -> ad.AnnData:
        """Attach per-cell QC masks to ``original.obs`` in-place.

        This is the sibling of R's ``ff[results$GoodCells, ]`` style
        workflow: for users who want to keep their full AnnData around
        and just tag cells instead of slicing.
        """
        original.obs["peacoqc_good"] = self.good_cells
        original.obs["peacoqc_outlier_it"] = self.outlier_it
        original.obs["peacoqc_outlier_mad"] = self.outlier_mad
        original.obs["peacoqc_consecutive"] = self.consecutive_cells
        uns = original.uns.setdefault("peacoqc", {})
        uns["percentage_removed"] = self.percentage_removed
        uns["it_percentage"] = self.it_percentage
        uns["mad_percentage"] = self.mad_percentage
        uns["consecutive_percentage"] = self.consecutive_percentage
        uns["weird_channels"] = self.weird_channels
        uns["mad_contribution"] = self.mad_contribution
        uns["parameters"] = self.parameters
        return original
