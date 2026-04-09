"""Port of ``PeacoQC::RemoveMargins``."""

from __future__ import annotations

import warnings
from typing import Mapping, Sequence

import anndata as ad
import numpy as np
import pandas as pd

from ._utils import append_original_id, channel_values, filename_of, resolve_channels


def remove_margins(
    adata: ad.AnnData,
    channels: Sequence[int | str],
    *,
    channel_specifications: Mapping[str, tuple[float, float]] | None = None,
    remove_min: Sequence[int | str] | None = None,
    remove_max: Sequence[int | str] | None = None,
    return_indices: bool = False,
) -> ad.AnnData | tuple[ad.AnnData, np.ndarray]:
    """Remove margin events from flow cytometry data.

    This is a direct port of :func:`PeacoQC::RemoveMargins`. For each
    requested channel, events whose value is at or below the channel's
    ``min_range`` (clipped to the per-channel minimum) are considered
    "min margin" events, and symmetrically for ``max_range``. Any event
    flagged on any channel is removed.

    Parameters
    ----------
    adata
        Input :class:`anndata.AnnData`.
    channels
        Indices or names of channels to check for margin events.
    channel_specifications
        Optional ``{channel_name: (min_range, max_range)}`` overrides for
        channels whose stored FCS ranges are incorrect.
    remove_min
        Channels to check for min-margin events (defaults to ``channels``).
    remove_max
        Channels to check for max-margin events (defaults to ``channels``).
    return_indices
        If True, return ``(filtered_adata, margin_indices)`` where
        ``margin_indices`` are the integer positions in ``adata`` that were
        removed.

    Returns
    -------
    AnnData (or tuple)
        Filtered :class:`anndata.AnnData` with an ``Original_ID`` column
        added to ``.obs``. If ``return_indices`` is True, a tuple is
        returned instead.
    """
    if not isinstance(adata, ad.AnnData):
        raise TypeError("adata should be an AnnData object.")

    channel_names = resolve_channels(adata, channels)
    remove_min_names = resolve_channels(adata, remove_min) if remove_min is not None else channel_names
    remove_max_names = resolve_channels(adata, remove_max) if remove_max is not None else channel_names

    if "min_range" not in adata.var.columns or "max_range" not in adata.var.columns:
        raise ValueError(
            "adata.var is missing min_range/max_range columns. "
            "Use peacoqc.read_fcs to load your file, or populate these "
            "columns manually before calling remove_margins."
        )

    specs: dict[str, tuple[float, float]] = {}
    if channel_specifications is not None:
        for name, pair in channel_specifications.items():
            if name not in adata.var_names:
                raise ValueError(
                    f"channel_specifications key {name!r} is not a channel in adata."
                )
            if len(pair) != 2:
                raise ValueError(
                    "Each channel_specifications entry must be a (minRange, maxRange) pair."
                )
            specs[name] = (float(pair[0]), float(pair[1]))

    n_events = adata.n_obs
    selection = np.ones(n_events, dtype=bool)
    margin_rows = []

    for ch in channel_names:
        values = channel_values(adata, ch)
        if ch in specs:
            min_range, max_range = specs[ch]
        else:
            var_row = adata.var.loc[ch]
            min_range = float(var_row["min_range"])
            max_range = float(var_row["max_range"])

        n_min = 0
        n_max = 0

        if ch in remove_min_names:
            # R: e[, d] <= max(min(meta[d,"minRange"], 0), min(e[, d]))
            threshold_min = max(min(min_range, 0.0), float(values.min()))
            min_margin = values <= threshold_min
            n_min = int(min_margin.sum())
            selection &= ~min_margin

        if ch in remove_max_names:
            # R: e[, d] >= min(meta[d,"maxRange"], max(e[, d]))
            threshold_max = min(max_range, float(values.max())) if np.isfinite(max_range) else float(values.max())
            max_margin = values >= threshold_max
            n_max = int(max_margin.sum())
            selection &= ~max_margin

        margin_rows.append((ch, n_min, n_max))

    margin_matrix = pd.DataFrame(
        margin_rows, columns=["channel", "min", "max"]
    ).set_index("channel")

    removed_frac = 1.0 - selection.mean()
    if removed_frac > 0.1:
        warnings.warn(
            f"More than {removed_frac * 100:.2f}% of events are considered "
            f"margin events in file {filename_of(adata) or '?'}. "
            "This should be verified.",
            stacklevel=2,
        )

    kept_idx = np.where(selection)[0]
    filtered = adata[kept_idx].copy()
    append_original_id(filtered, kept_idx)

    peacoqc_uns = filtered.uns.setdefault("peacoqc", {})
    peacoqc_uns["margin_matrix"] = margin_matrix

    if return_indices:
        return filtered, np.where(~selection)[0]
    return filtered
