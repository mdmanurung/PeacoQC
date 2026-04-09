"""Tests for :mod:`peacoqc.consecutive`."""

from __future__ import annotations

import numpy as np

from peacoqc.consecutive import inverse_rle, remove_short_true_runs, rle


def test_rle_roundtrip():
    arr = np.array([True, True, False, True, True, True, False, False])
    lengths, values = rle(arr)
    assert lengths.tolist() == [2, 1, 3, 2]
    assert values.tolist() == [True, False, True, False]
    assert inverse_rle(lengths, values).tolist() == arr.tolist()


def test_remove_short_true_runs_min_len_3():
    arr = np.array([True, True, False, True, True, True, False, False])
    # The first run of True has length 2 -> flipped to False.
    # The second run of True has length 3 -> kept.
    flipped = remove_short_true_runs(arr, min_run=3)
    assert flipped.tolist() == [False, False, False, True, True, True, False, False]


def test_remove_short_keeps_all_when_min_run_is_one():
    arr = np.array([True, False, True, False])
    assert remove_short_true_runs(arr, min_run=1).tolist() == arr.tolist()


def test_remove_short_handles_empty():
    assert remove_short_true_runs(np.zeros(0, dtype=bool), min_run=5).shape == (0,)
