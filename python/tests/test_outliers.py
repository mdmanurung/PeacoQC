"""Tests for :mod:`peacoqc.outliers`."""

from __future__ import annotations

import numpy as np
import pandas as pd

from peacoqc.outliers import (
    _mad_outliers_column,
    isolation_tree_outliers,
    mad_outlier_method,
    removed_bins_to_cells,
)


def test_isolation_tree_marks_outlier_bin():
    rng = np.random.default_rng(0)
    n_bins = 60
    X = rng.normal(0.0, 0.1, size=(n_bins, 3))
    # Add a couple of extreme bins
    X[10] += 5
    X[40] -= 5
    df = pd.DataFrame(X, columns=["ch1__1", "ch2__1", "ch3__1"])
    good, info = isolation_tree_outliers(df, it_limit=0.6, random_state=0)
    assert not good[10]
    assert not good[40]
    # Plenty of inliers should remain.
    assert good.sum() > 0.5 * n_bins
    # IsolationForest builds its splits on some subset of the columns.
    assert isinstance(info["split_columns"], list)


def test_mad_outliers_column_flags_sustained_drift():
    """A multi-bin sustained elevation should be flagged.

    PeacoQC's MAD filter is designed to catch sustained drift or plateaus
    in the per-bin peak trajectory — not single-bin glitches, which get
    smoothed out. The test uses a 5-bin plateau at value 100 inside a
    mostly-constant baseline.
    """
    y = np.full(40, 1.0)
    y[18:23] = 100.0
    flags = _mad_outliers_column(y, mad_thresh=6.0)
    # At least some of the plateau bins should be flagged.
    assert flags[18:23].any()


def test_mad_outlier_method_contribution():
    peak_matrix = pd.DataFrame(
        {
            "chA__1": np.full(30, 1.0),
            "chB__1": np.full(30, 5.0),
        }
    )
    peak_matrix.loc[13:17, "chA__1"] = 100.0
    breaks = [np.arange(i * 10, (i + 1) * 10) for i in range(30)]
    out = mad_outlier_method(
        peak_matrix,
        good_mask_in=np.ones(30, dtype=bool),
        mad_thresh=6.0,
        breaks=breaks,
        n_events=300,
        channel_columns={"chA": ["chA__1"], "chB": ["chB__1"]},
    )
    # Plateau bins flagged, others fine.
    assert out["mad_bins"][13:18].any()
    assert out["contribution"]["chA"] > 0
    assert out["contribution"]["chB"] == 0


def test_removed_bins_to_cells():
    breaks = [np.arange(0, 5), np.arange(5, 10), np.arange(10, 15)]
    bad_bins = np.array([False, True, False])
    good, removed = removed_bins_to_cells(breaks, bad_bins, n_events=15)
    assert good.sum() == 10
    assert removed.tolist() == list(range(5, 10))
