"""Tests for :func:`peacoqc.remove_doublets`."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

import peacoqc


def _synth_singlets_plus_doublets() -> ad.AnnData:
    rng = np.random.default_rng(0)
    n = 1000
    fsc_h = rng.uniform(1e4, 1e5, size=n)
    fsc_a = fsc_h * rng.normal(1.0, 0.02, size=n)  # singlet ratio ~1
    # Inject 20 doublets with ratio ~2
    fsc_a[:20] = fsc_h[:20] * 2.0
    X = np.column_stack([fsc_a, fsc_h]).astype(np.float32)
    var = pd.DataFrame(
        {"channel": ["FSC-A", "FSC-H"], "marker": ["", ""]},
        index=["FSC-A", "FSC-H"],
    )
    return ad.AnnData(X=X, var=var)


def test_remove_doublets_synthetic():
    adata = _synth_singlets_plus_doublets()
    filtered = peacoqc.remove_doublets(adata, channel1="FSC-A", channel2="FSC-H")
    # All 20 injected doublets should be flagged.
    assert filtered.n_obs <= 980
    assert "Original_ID" in filtered.obs.columns
    # Ensure none of the retained cells were injected doublets.
    assert not set(filtered.obs["Original_ID"].tolist()) & set(range(20))


def test_remove_doublets_preserves_original_id_chain():
    adata = _synth_singlets_plus_doublets()
    adata.obs["Original_ID"] = np.arange(adata.n_obs, dtype=np.int64) + 5000
    filtered = peacoqc.remove_doublets(adata)
    # Original_ID should be a slice of the incoming IDs, not a fresh range.
    assert filtered.obs["Original_ID"].min() >= 5000
