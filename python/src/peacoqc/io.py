"""FCS I/O helpers.

``read_fcs`` wraps :func:`pytometry.io.read_fcs` and falls back to building
an :class:`anndata.AnnData` directly from :class:`readfcs.ReadFCS` when the
FCS file has no ``$PnS`` (marker) field, which is common for raw flow files.
Either way we normalize the min/max range keywords
(``flowcore_p{n}rmin``/``rmax`` or ``$PnR``) into ``adata.var``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd


def _ranges_from_meta(meta: dict, n_channels: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (min_range, max_range) arrays of length ``n_channels``."""
    min_range = np.zeros(n_channels, dtype=float)
    max_range = np.full(n_channels, np.nan, dtype=float)

    # normalize keys to lowercase for comparison
    lower = {str(k).lower(): v for k, v in meta.items()}

    for idx in range(n_channels):
        pn = idx + 1
        # Preferred: flowcore_p{n}r{min,max} (set by flowCore when a transform
        # has been applied) — these are the post-transform ranges.
        k_min = f"flowcore_p{pn}rmin"
        k_max = f"flowcore_p{pn}rmax"
        if k_min in lower and k_max in lower:
            try:
                min_range[idx] = float(lower[k_min])
                max_range[idx] = float(lower[k_max])
                continue
            except (TypeError, ValueError):
                pass
        # Fallback: $PnR (max only; min defaults to 0)
        k_r = f"$p{pn}r"
        if k_r in lower:
            try:
                max_range[idx] = float(lower[k_r])
            except (TypeError, ValueError):
                pass
    return min_range, max_range


def _adata_from_readfcs(fcs: Any) -> ad.AnnData:
    """Build an AnnData directly from a :class:`readfcs.ReadFCS` instance.

    This is the fallback path used when the file lacks ``$PnS``. It mimics
    the layout ``pytometry.io.read_fcs`` produces: ``adata.X`` is the
    expression matrix, ``adata.var`` has ``channel`` / ``marker`` columns,
    and ``adata.uns['meta']`` carries the raw FCS keyword dictionary.
    """
    data = fcs.data  # DataFrame (n_events, n_channels)
    channels = fcs.channels  # DataFrame with PnN, PnB, PnE, PnG, PnR, PnV

    pnn = channels["PnN"].to_numpy()
    if "PnS" in channels.columns:
        pns = channels["PnS"].to_numpy()
    else:
        pns = np.array([""] * len(pnn))

    var = pd.DataFrame(
        {
            "channel": pnn,
            "marker": pns,
        },
        index=pd.Index(pnn, name=None),
    )
    # Preserve PnR etc. on the var table for downstream use.
    for col in ("PnB", "PnE", "PnG", "PnR", "PnV"):
        if col in channels.columns:
            var[col] = channels[col].to_numpy()

    X = data.to_numpy().astype(np.float32, copy=False)
    adata = ad.AnnData(X=X, var=var)
    adata.uns["meta"] = dict(fcs.meta)
    return adata


def read_fcs(path: str | os.PathLike[str]) -> ad.AnnData:
    """Read an FCS file and return an :class:`anndata.AnnData`.

    Parameters
    ----------
    path
        Path to an ``.fcs`` file.

    Returns
    -------
    AnnData
        Matrix with ``n_events`` rows and ``n_channels`` columns.
        ``adata.var`` carries the channel metadata plus ``min_range`` and
        ``max_range`` columns parsed from the FCS keywords. The filename is
        stored in ``adata.uns['peacoqc']['filename']``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"FCS file not found: {path}")

    try:
        import pytometry as pm  # noqa: F401

        adata = pm.io.read_fcs(str(path))
    except Exception:
        # Fallback: some FCS files have no $PnS field. pytometry/readfcs
        # raises in that case; build the AnnData ourselves.
        import readfcs

        fcs = readfcs.ReadFCS(str(path))
        adata = _adata_from_readfcs(fcs)

    meta: dict = adata.uns.get("meta", {}) or {}
    min_range, max_range = _ranges_from_meta(meta, adata.n_vars)
    adata.var["min_range"] = min_range
    adata.var["max_range"] = max_range

    # Stash provenance info
    peacoqc_uns = adata.uns.setdefault("peacoqc", {})
    peacoqc_uns["filename"] = path.name

    return adata


def write_fcs(adata: ad.AnnData, path: str | os.PathLike[str]) -> None:
    """Write an :class:`anndata.AnnData` back to an FCS file.

    Requires the ``flowio`` optional extra (``pip install peacoqc[fcs]``).

    The expression matrix is taken from ``adata.X``. Channel names are taken
    from ``adata.var.index``; marker names from ``adata.var['marker']`` if
    present.
    """
    try:
        import flowio
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "write_fcs requires the 'flowio' optional dependency. "
            "Install with: pip install peacoqc[fcs]"
        ) from exc

    path = Path(path)
    X = np.asarray(adata.X, dtype=np.float32)
    channel_names = list(adata.var.index.astype(str))
    if "marker" in adata.var.columns:
        marker_names = [str(m) if pd.notna(m) and str(m) else "" for m in adata.var["marker"]]
    else:
        marker_names = [""] * len(channel_names)

    # flowio.create_fcs expects the matrix flattened row-major.
    flat = X.reshape(-1).astype(np.float32)
    with open(path, "wb") as fh:
        flowio.create_fcs(
            fh,
            flat.tolist(),
            channel_names=channel_names,
            opt_channel_names=marker_names,
        )
