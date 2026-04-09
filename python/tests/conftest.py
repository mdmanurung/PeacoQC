"""Pytest fixtures shared by the peacoqc test suite."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

FCS_SAMPLE = Path(__file__).parent.parent.parent / "inst" / "extdata" / "111_Comp_Trans.fcs"
FCS_RAW = Path(__file__).parent.parent.parent / "inst" / "extdata" / "111.fcs"
R_REPORT_TSV = Path(__file__).parent.parent.parent / "inst" / "extdata" / "PeacoQC_report.txt"

# R vignette channel indices (1-based): c(1, 3, 5:14, 18, 21)
# -> zero-based: 0, 2, 4..13, 17, 20
VIGNETTE_CHANNEL_IDX = [0, 2] + list(range(4, 14)) + [17, 20]


@pytest.fixture(scope="session")
def sample_adata():
    pytest.importorskip("anndata")
    import peacoqc

    if not FCS_SAMPLE.exists():
        pytest.skip(f"example FCS not found: {FCS_SAMPLE}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return peacoqc.read_fcs(str(FCS_SAMPLE))


@pytest.fixture(scope="session")
def sample_channel_names(sample_adata):
    return [sample_adata.var_names[i] for i in VIGNETTE_CHANNEL_IDX]
