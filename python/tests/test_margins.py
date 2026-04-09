"""Tests for :func:`peacoqc.remove_margins`."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import peacoqc


def _make_synthetic() -> ad.AnnData:
    """Build a tiny synthetic AnnData with known out-of-range events.

    The R logic clamps the min/max threshold to the channel's actual
    min/max, so the test data is constructed such that every channel has
    *both* an injected "below minRange" event and an injected "above
    maxRange" event. This way the clamping falls back to min_range /
    max_range and only the injected boundary events are flagged.
    """
    rng = np.random.default_rng(42)
    X = rng.uniform(20.0, 80.0, size=(100, 3)).astype(np.float32)
    # For every channel, inject one extreme-low and one extreme-high event
    # at known rows so the natural data range doesn't bound the threshold.
    below = -20.0  # below min_range=-10
    above = 120.0  # above max_range=110
    X[0, :] = below
    X[1, :] = above
    X[2, 0] = below
    X[2, 1] = above
    var = pd.DataFrame(
        {
            "channel": ["ch0", "ch1", "ch2"],
            "marker": ["m0", "m1", "m2"],
            "min_range": [-10.0, -10.0, -10.0],
            "max_range": [110.0, 110.0, 110.0],
        },
        index=["ch0", "ch1", "ch2"],
    )
    return ad.AnnData(X=X, var=var)


def test_remove_margins_synthetic():
    adata = _make_synthetic()
    filtered = peacoqc.remove_margins(adata, channels=["ch0", "ch1", "ch2"])
    # Rows 0, 1, 2 should all be flagged on at least one channel.
    assert filtered.n_obs == 97
    assert "Original_ID" in filtered.obs.columns
    assert set(filtered.obs["Original_ID"].tolist()) == set(range(3, 100))


def test_remove_margins_channel_override_tightens():
    """Overriding ``maxRange`` to a lower value should remove additional
    rows that would have been kept under the default specifications.
    """
    adata = _make_synthetic()
    # A very low max_range override should flag anything above 50.
    filtered = peacoqc.remove_margins(
        adata,
        channels=["ch0"],
        channel_specifications={"ch0": (-10.0, 50.0)},
    )
    # Significantly more rows removed than the default ch0-only case.
    default_only_ch0 = peacoqc.remove_margins(adata, channels=["ch0"])
    assert filtered.n_obs < default_only_ch0.n_obs


def test_remove_margins_return_indices():
    adata = _make_synthetic()
    filtered, margin_idx = peacoqc.remove_margins(
        adata, channels=["ch0", "ch1", "ch2"], return_indices=True
    )
    assert sorted(margin_idx.tolist()) == [0, 1, 2]


def test_remove_margins_on_real_fcs(sample_adata, sample_channel_names):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        filtered = peacoqc.remove_margins(sample_adata, channels=sample_channel_names)
    assert "Original_ID" in filtered.obs.columns
    assert filtered.n_obs <= sample_adata.n_obs
