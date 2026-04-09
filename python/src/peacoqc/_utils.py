"""Shared internal helpers."""

from __future__ import annotations

from typing import Iterable, Sequence

import anndata as ad
import numpy as np


def resolve_channels(adata: ad.AnnData, channels: Sequence[int | str] | None) -> list[str]:
    """Coerce a mix of indices and names into a list of channel names."""
    if channels is None:
        raise ValueError("channels must be provided (indices or names).")
    names: list[str] = []
    all_names = list(adata.var_names)
    for c in channels:
        if isinstance(c, (int, np.integer)):
            names.append(all_names[int(c)])
        else:
            c_str = str(c)
            if c_str not in all_names:
                raise ValueError(f"Channel {c_str!r} not found in adata.var_names.")
            names.append(c_str)
    return names


def ensure_original_id(adata: ad.AnnData) -> ad.AnnData:
    """Attach an ``Original_ID`` column to ``adata.obs`` if missing."""
    if "Original_ID" not in adata.obs.columns:
        adata.obs["Original_ID"] = np.arange(adata.n_obs, dtype=np.int64)
    return adata


def append_original_id(filtered: ad.AnnData, selection_idx: np.ndarray) -> ad.AnnData:
    """Set ``Original_ID`` on a filtered copy to the original cell indices.

    ``selection_idx`` is the integer index array into the *pre-filter* frame
    corresponding to the rows kept in ``filtered``.
    """
    filtered.obs["Original_ID"] = np.asarray(selection_idx, dtype=np.int64)
    return filtered


def filename_of(adata: ad.AnnData) -> str | None:
    peacoqc_uns = adata.uns.get("peacoqc") if hasattr(adata, "uns") else None
    if isinstance(peacoqc_uns, dict):
        fn = peacoqc_uns.get("filename")
        if fn:
            return str(fn)
    meta = adata.uns.get("meta") if hasattr(adata, "uns") else None
    if isinstance(meta, dict):
        for key in ("filename", "fil", "$FIL", "FILENAME"):
            v = meta.get(key)
            if v is None:
                continue
            v = str(v).strip()
            if v:
                return v
    return None


def time_channel_name(adata: ad.AnnData, time_channel_parameter: str | None = "Time") -> str | None:
    """Find the time channel column name (case-insensitive substring match)."""
    if time_channel_parameter is None:
        return None
    needle = time_channel_parameter.lower()
    for name in adata.var_names:
        if needle in str(name).lower():
            return str(name)
    return None


def channel_values(adata: ad.AnnData, channel: str) -> np.ndarray:
    """Return a 1D numpy array of values for a single channel."""
    idx = list(adata.var_names).index(channel)
    X = adata.X
    if hasattr(X, "toarray"):
        col = X[:, idx].toarray().ravel()
    else:
        col = np.asarray(X[:, idx]).ravel()
    return np.asarray(col, dtype=float)


def as_dense(X) -> np.ndarray:
    if hasattr(X, "toarray"):
        return X.toarray()
    return np.asarray(X)
