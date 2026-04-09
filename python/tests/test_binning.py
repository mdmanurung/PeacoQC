"""Tests for :mod:`peacoqc.binning`."""

from __future__ import annotations

import math

import numpy as np

from peacoqc.binning import (
    find_events_per_bin,
    make_breaks,
    split_with_overlap,
    split_with_overlap_mids,
)


def test_split_with_overlap_basic():
    bins = split_with_overlap(10, seg_length=4, overlap=2)
    # stride=2: [0..3], [2..5], [4..7], [6..9]
    assert [b.tolist() for b in bins] == [
        [0, 1, 2, 3],
        [2, 3, 4, 5],
        [4, 5, 6, 7],
        [6, 7, 8, 9],
    ]


def test_split_with_overlap_truncates_last():
    bins = split_with_overlap(11, seg_length=4, overlap=2)
    assert bins[-1].tolist() == [8, 9, 10]


def test_split_with_overlap_mids():
    mids = split_with_overlap_mids(10, seg_length=4, overlap=2)
    assert all(m < 10 for m in mids)
    # halfway offset = ceil(2/2) = 1, starts = 0,2,4,6,8 -> mids 1,3,5,7,9
    assert mids == [1, 3, 5, 7, 9]


def test_make_breaks_covers_all_events():
    breaks, eps = make_breaks(500, 2000)
    assert eps == 500
    covered = set()
    for b in breaks:
        covered.update(b.tolist())
    assert covered == set(range(2000))


def test_find_events_per_bin_default():
    # n=9617 (our example fixture), min_cells=150, max_bins=500, step=500
    # max_cells = ceil(9617/500 * 2) = 39 -> (39//500)*500 + 500 = 500
    # max(150, 500) = 500
    assert find_events_per_bin(9617) == 500


def test_find_events_per_bin_floor_on_min_cells():
    # Very few events -> should fall back to min_cells
    assert find_events_per_bin(100, min_cells=150) == 500  # step adds 500
