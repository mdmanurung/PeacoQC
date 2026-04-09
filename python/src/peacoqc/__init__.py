"""Python port of the R/Bioconductor PeacoQC package."""

from __future__ import annotations

from .doublets import remove_doublets
from .heatmap import peaco_qc_heatmap
from .io import read_fcs, write_fcs
from .margins import remove_margins
from .plotting import plot_peaco_qc
from .qc import peaco_qc
from .results import PeacoQCResult

__all__ = [
    "read_fcs",
    "write_fcs",
    "remove_margins",
    "remove_doublets",
    "peaco_qc",
    "plot_peaco_qc",
    "peaco_qc_heatmap",
    "PeacoQCResult",
]

__version__ = "0.1.0"
