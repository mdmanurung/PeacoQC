"""End-to-end smoke tests for :func:`peacoqc.peaco_qc`."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

import peacoqc


@pytest.fixture(scope="module")
def qc_result(sample_adata, sample_channel_names, tmp_path_factory):
    report = tmp_path_factory.mktemp("peacoqc") / "report.csv"
    result = peacoqc.peaco_qc(
        sample_adata,
        channels=sample_channel_names,
        determine_good_cells="all",
        report_path=str(report),
    )
    return result, report


def test_peaco_qc_returns_sensible_result(qc_result, sample_adata):
    result, _ = qc_result
    assert 0.0 <= result.percentage_removed < 100.0
    assert result.good_cells.sum() == result.adata.n_obs
    assert result.adata.n_obs <= sample_adata.n_obs
    assert len(result.good_cells) == sample_adata.n_obs
    assert result.analysis == "all"
    assert result.parameters["events_per_bin"] == 500


def test_peaco_qc_report_written(qc_result):
    _, report = qc_result
    assert report.exists()
    with report.open() as fh:
        reader = csv.reader(fh)
        header = next(reader)
        row = next(reader)
    assert "Filename" in header
    assert "% Full analysis" in header
    assert row  # at least one row written


def test_peaco_qc_it_only(sample_adata, sample_channel_names):
    # Force a larger bin count so IT actually runs
    result = peacoqc.peaco_qc(
        sample_adata,
        channels=sample_channel_names,
        determine_good_cells="IT",
        events_per_bin=150,
        force_it=30,
    )
    assert result.analysis == "IT"
    # mad_percentage should be None when MAD not run
    assert result.mad_percentage is None


def test_peaco_qc_mad_only(sample_adata, sample_channel_names):
    result = peacoqc.peaco_qc(
        sample_adata,
        channels=sample_channel_names,
        determine_good_cells="MAD",
    )
    assert result.analysis == "MAD"
    assert result.it_percentage is None
    assert result.mad_percentage is not None
