"""Smoke tests for :func:`peacoqc.read_fcs`."""

from __future__ import annotations

from conftest import FCS_RAW, FCS_SAMPLE


def test_read_transformed_fcs(sample_adata):
    # Known fixture dimensions from the R vignette data.
    assert sample_adata.shape == (9617, 22)
    assert "min_range" in sample_adata.var.columns
    assert "max_range" in sample_adata.var.columns
    assert sample_adata.uns["peacoqc"]["filename"] == "111_Comp_Trans.fcs"


def test_read_raw_fcs():
    """111.fcs has no PnS so it takes the readfcs fallback path."""
    import peacoqc
    if not FCS_RAW.exists():  # pragma: no cover - data missing
        import pytest
        pytest.skip("raw FCS fixture missing")
    adata = peacoqc.read_fcs(str(FCS_RAW))
    assert adata.n_obs > 0
    assert "min_range" in adata.var.columns
    assert "max_range" in adata.var.columns
