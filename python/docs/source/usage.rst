Usage guide
===========

The executable tour of the package lives in the vignette notebook
below — a Python port of the canonical
`R vignette <https://bioconductor.org/packages/release/bioc/vignettes/PeacoQC/inst/doc/PeacoQC.html>`_.
It runs end-to-end against the example FCS files shipped in
``inst/extdata``.

.. toctree::
   :maxdepth: 2

   vignette

Selecting analysis modes
------------------------

:func:`peacoqc.peaco_qc` accepts a ``determine_good_cells`` argument
that controls which outlier steps run:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Value
     - Behaviour
   * - ``"all"``
     - Run both the IsolationForest (IT) and MAD steps. This is the
       default and mirrors the R package.
   * - ``"IT"``
     - Only the IsolationForest step runs.
   * - ``"MAD"``
     - Only the MAD step runs.
   * - ``False``
     - Neither outlier step runs. ``result.peaks`` is still populated
       so you can inspect peak trajectories; this is the Python
       equivalent of R's ``PlotPeacoQC(display_peaks=TRUE)``.

The IsolationForest step is only performed when the number of bins
is at least ``force_it`` (150 by default). For small files lower
``force_it`` or shrink ``events_per_bin``.

Annotating instead of slicing
-----------------------------

By default :func:`peacoqc.peaco_qc` returns a new
:class:`anndata.AnnData` that contains only the good cells. If you
prefer to keep the full object and annotate which events are good,
call :meth:`PeacoQCResult.annotate <peacoqc.PeacoQCResult.annotate>`:

.. code-block:: python

    result = peacoqc.peaco_qc(adata, channels=channel_names)
    result.annotate(adata)  # in-place

    adata.obs["peacoqc_good"].sum()
    adata.obs[["peacoqc_outlier_it",
               "peacoqc_outlier_mad",
               "peacoqc_consecutive"]].head()

This writes boolean columns (``peacoqc_good``, ``peacoqc_outlier_it``,
``peacoqc_outlier_mad``, ``peacoqc_consecutive``) onto ``adata.obs``
and copies the QC summary onto ``adata.uns['peacoqc']``.

Multi-sample reports
--------------------

When ``peaco_qc(..., report_path=...)`` is called for several files
with the same report path, the rows are appended to a single CSV.
Pass that path to :func:`peacoqc.peaco_qc_heatmap` to get the
cross-sample overview — see the vignette for a worked example on
both a freshly generated CSV and the R package's legacy TSV report.
