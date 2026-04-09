"""Round-trip smoke test for the optional :func:`peacoqc.write_fcs`."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

flowio = pytest.importorskip("flowio")

import peacoqc


def test_write_fcs_round_trip(sample_adata, tmp_path):
    out = tmp_path / "roundtrip.fcs"
    peacoqc.write_fcs(sample_adata, str(out))
    assert out.exists()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        re = peacoqc.read_fcs(str(out))
    assert re.shape == sample_adata.shape
    # Values should survive a round-trip (float32 precision).
    a = np.asarray(sample_adata.X)
    b = np.asarray(re.X)
    assert np.allclose(a, b, rtol=1e-4, atol=1e-3)
