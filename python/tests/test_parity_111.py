"""Parity tests against R reference fixtures.

These tests are skipped unless the R reference JSON exists. Generate it
with:

    Rscript python/tests/fixtures/gen_r_reference.R

from the repository root (requires the upstream R package installed).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import peacoqc
from conftest import FCS_SAMPLE, VIGNETTE_CHANNEL_IDX

REFERENCE_JSON = Path(__file__).parent / "data" / "r_reference_111.json"

pytestmark = pytest.mark.parity


@pytest.fixture(scope="module")
def r_reference():
    if not REFERENCE_JSON.exists():
        pytest.skip(
            "R reference fixtures missing. Run "
            "`Rscript python/tests/fixtures/gen_r_reference.R` to generate."
        )
    with REFERENCE_JSON.open() as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def py_result():
    adata = peacoqc.read_fcs(str(FCS_SAMPLE))
    return peacoqc.peaco_qc(
        adata,
        channels=VIGNETTE_CHANNEL_IDX,
        determine_good_cells="all",
    )


def test_percentage_removed_within_tolerance(py_result, r_reference):
    assert abs(py_result.percentage_removed - r_reference["percentage_removed"]) < 3.0


def test_good_cells_jaccard(py_result, r_reference):
    py_good = py_result.good_cells.astype(bool)
    # R indices are 1-based and only lists the GOOD indices.
    r_good_idx = np.asarray(r_reference["good_cells_idx_1based"], dtype=int) - 1
    r_good = np.zeros(len(py_good), dtype=bool)
    r_good[r_good_idx] = True
    inter = np.logical_and(py_good, r_good).sum()
    union = np.logical_or(py_good, r_good).sum()
    jaccard = inter / union if union else 1.0
    assert jaccard >= 0.90, f"Jaccard too low: {jaccard:.3f}"


def test_weird_channel_label(py_result, r_reference):
    assert py_result.weird_channels["label"] == r_reference["weird_channel_label"]


def test_events_per_bin(py_result, r_reference):
    assert py_result.events_per_bin == int(r_reference["events_per_bin"])
