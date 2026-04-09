"""Port of ``PeacoQC::RemoveDoublets``."""

from __future__ import annotations

import anndata as ad
import numpy as np
from scipy.stats import median_abs_deviation

from ._utils import append_original_id, channel_values


def remove_doublets(
    adata: ad.AnnData,
    *,
    channel1: str = "FSC-A",
    channel2: str = "FSC-H",
    nmad: float = 4.0,
    b: float = 0.0,
    verbose: bool = False,
    return_indices: bool = False,
) -> ad.AnnData | tuple[ad.AnnData, np.ndarray]:
    """Remove doublet events based on the ratio of two channels.

    Cells are kept if ``channel1 / (1e-10 + channel2 + b) <
    median(ratio) + nmad * mad(ratio)``. This is the same rule used by
    :func:`PeacoQC::RemoveDoublets`.

    Parameters
    ----------
    adata
        Input :class:`anndata.AnnData`.
    channel1, channel2
        Channel names used for the ratio. Defaults match the R package
        (``FSC-A`` / ``FSC-H``).
    nmad
        Bandwidth above the median ratio that is still accepted. Defaults
        to 4.
    b
        Offset added to ``channel2`` before computing the ratio.
    verbose
        If True, prints the median ratio and MAD width.
    return_indices
        If True, also return the integer indices that were flagged as
        doublets.
    """
    if not isinstance(adata, ad.AnnData):
        raise TypeError("adata should be an AnnData object.")
    if channel1 not in adata.var_names or channel2 not in adata.var_names:
        raise ValueError(
            f"Both channel1={channel1!r} and channel2={channel2!r} must be in adata.var_names."
        )

    c1 = channel_values(adata, channel1)
    c2 = channel_values(adata, channel2)
    ratio = c1 / (1e-10 + c2 + b)

    median_ratio = float(np.median(ratio))
    # R's stats::mad uses constant 1.4826 by default -> scale='normal' in scipy
    mad_ratio = float(median_abs_deviation(ratio, scale="normal"))

    if verbose:
        print(f"Median ratio: {median_ratio}, width: {nmad * mad_ratio}")

    selection = ratio < median_ratio + nmad * mad_ratio
    kept_idx = np.where(selection)[0]

    filtered = adata[kept_idx].copy()
    if "Original_ID" in adata.obs.columns:
        # chain through the existing IDs so Original_ID stays aligned with the
        # very first AnnData loaded from the FCS.
        filtered.obs["Original_ID"] = np.asarray(
            adata.obs["Original_ID"].values[kept_idx]
        )
    else:
        append_original_id(filtered, kept_idx)

    if return_indices:
        return filtered, np.where(~selection)[0]
    return filtered
