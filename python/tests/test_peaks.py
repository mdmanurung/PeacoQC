"""Tests for :mod:`peacoqc.peaks`."""

from __future__ import annotations

import numpy as np
import pandas as pd

from peacoqc.binning import make_breaks
from peacoqc.peaks import (
    determine_all_peaks,
    determine_peaks_all_channels,
    extract_peak_values,
    find_peaks_kde,
)


def test_find_peaks_kde_bimodal():
    rng = np.random.default_rng(0)
    left = rng.normal(0.0, 0.2, 500)
    right = rng.normal(5.0, 0.2, 500)
    peaks = find_peaks_kde(np.concatenate([left, right]))
    assert peaks is not None
    assert len(peaks) == 2
    assert abs(peaks[0] - 0) < 0.5
    assert abs(peaks[1] - 5) < 0.5


def test_find_peaks_kde_unimodal_fallback():
    rng = np.random.default_rng(1)
    x = rng.normal(3.0, 0.5, 200)
    peaks = find_peaks_kde(x)
    assert peaks is not None
    # Should have exactly one peak near 3.0
    assert len(peaks) >= 1
    assert abs(peaks[0] - 3.0) < 0.5


def test_determine_all_peaks_two_clusters():
    rng = np.random.default_rng(2)
    n = 4000
    data = np.concatenate(
        [rng.normal(0.0, 0.3, n // 2), rng.normal(3.0, 0.3, n // 2)]
    )
    rng.shuffle(data)
    breaks, _ = make_breaks(400, len(data))
    frame = determine_all_peaks(data, breaks)
    assert frame is not None
    assert set(frame["Cluster"].unique()) == {1, 2}


def test_extract_peak_values_shape():
    df = pd.DataFrame(
        {"Bin": [1, 2, 3], "Peak": [1.0, 1.1, 1.2], "Cluster": [1, 1, 1]}
    )
    result = extract_peak_values(df, n_bins=5)
    assert 1 in result
    values = result[1]
    assert len(values) == 5
    # bins not present in the frame should equal the cluster median (1.1)
    assert values[3] == 1.1
    assert values[0] == 1.0
    assert values[1] == 1.1
    assert values[2] == 1.2


def test_determine_peaks_all_channels_runs_on_real_fcs(sample_adata, sample_channel_names):
    from peacoqc._utils import as_dense

    X = as_dense(sample_adata.X).astype(float)
    idx = [list(sample_adata.var_names).index(n) for n in sample_channel_names]
    breaks, _ = make_breaks(500, sample_adata.n_obs)
    peak_matrix, per_channel, cols = determine_peaks_all_channels(
        X, sample_channel_names, idx, breaks
    )
    assert peak_matrix.shape[0] == len(breaks)
    assert peak_matrix.shape[1] > 0
    assert len(per_channel) > 0
