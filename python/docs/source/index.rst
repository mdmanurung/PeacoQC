peacoqc
=======

**peacoqc** is a Python port of the R/Bioconductor
`PeacoQC <https://github.com/saeyslab/PeacoQC>`_ package — peak-based
quality control for flow and mass cytometry data. It identifies and
removes events originating from clogs, speed changes, and other
measurement artefacts by binning a file in time, detecting density
peaks per channel, and flagging bins whose peak structure deviates
from the rest of the run.

FCS I/O is provided by
`pytometry <https://github.com/scverse/pytometry>`_ / ``readfcs``, so
results live in :class:`anndata.AnnData` and plug directly into the
`scverse <https://scverse.org>`_ ecosystem.

.. code-block:: python

    import peacoqc

    adata = peacoqc.read_fcs("sample.fcs")
    channels = ["FSC-A", "SSC-A", "B710-A", "B515-A", "R780-A"]

    adata = peacoqc.remove_margins(adata, channels=channels)
    adata = peacoqc.remove_doublets(adata, channel1="FSC-A", channel2="FSC-H")

    result = peacoqc.peaco_qc(
        adata,
        channels=channels,
        determine_good_cells="all",
        report_path="peacoqc_report.csv",
    )

    peacoqc.plot_peaco_qc(adata, result, output_path="sample_qc.png")
    peacoqc.write_fcs(result.adata, "sample_qc.fcs")

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   usage
   api

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
