"""Smoke tests for :func:`peacoqc.plot_peaco_qc`."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import peacoqc


def test_plot_peaco_qc_produces_figure(sample_adata, sample_channel_names, tmp_path):
    result = peacoqc.peaco_qc(
        sample_adata,
        channels=sample_channel_names,
        determine_good_cells="all",
    )
    out = tmp_path / "qc.png"
    fig = peacoqc.plot_peaco_qc(
        sample_adata, result, channels=sample_channel_names, output_path=str(out)
    )
    assert out.exists()
    assert out.stat().st_size > 1000
    # Time channel present -> one extra panel on top.
    assert len(fig.axes) >= len(sample_channel_names)
