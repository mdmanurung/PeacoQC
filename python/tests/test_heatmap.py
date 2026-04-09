"""Smoke tests for :func:`peacoqc.peaco_qc_heatmap`."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import peacoqc

from conftest import R_REPORT_TSV


def test_heatmap_reads_r_tsv(tmp_path):
    if not R_REPORT_TSV.exists():  # pragma: no cover
        import pytest

        pytest.skip("R report fixture missing")
    out = tmp_path / "heatmap.png"
    fig = peacoqc.peaco_qc_heatmap(str(R_REPORT_TSV), output_path=str(out))
    assert out.exists()
    # Should produce a figure with multiple axes (params, main, trend, cbar).
    assert len(fig.axes) >= 3


def test_heatmap_reads_our_csv(sample_adata, sample_channel_names, tmp_path):
    report = tmp_path / "report.csv"
    peacoqc.peaco_qc(
        sample_adata,
        channels=sample_channel_names,
        determine_good_cells="all",
        report_path=str(report),
    )
    out = tmp_path / "heatmap.png"
    fig = peacoqc.peaco_qc_heatmap(str(report), output_path=str(out))
    assert out.exists()
    assert len(fig.axes) >= 3


def test_latest_tests_deduplicates(tmp_path):
    if not R_REPORT_TSV.exists():
        import pytest

        pytest.skip("R report fixture missing")
    fig = peacoqc.peaco_qc_heatmap(str(R_REPORT_TSV), latest_tests=True)
    assert len(fig.axes) >= 3
