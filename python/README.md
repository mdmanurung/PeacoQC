# peacoqc (Python)

Python port of the R/Bioconductor
[PeacoQC](https://github.com/saeyslab/PeacoQC) package — peak-based quality
control for flow and mass cytometry data. FCS I/O is provided by
[pytometry](https://github.com/scverse/pytometry) / `readfcs`, so results live
in `AnnData` and plug into the scverse ecosystem.

## Installation

```bash
pip install peacoqc                 # core
pip install "peacoqc[fcs]"          # + flowio, needed for write_fcs
pip install "peacoqc[plotting]"     # + seaborn
pip install "peacoqc[test]"         # + pytest for running the test suite
```

## Quickstart

```python
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
peacoqc.write_fcs(result.adata, "sample_qc.fcs")            # requires [fcs] extra
peacoqc.peaco_qc_heatmap("peacoqc_report.csv", output_path="heatmap.png")
```

See `docs/quickstart.md` for more detail, and the R package's
[vignette](https://bioconductor.org/packages/release/bioc/vignettes/PeacoQC/inst/doc/PeacoQC.html)
for the underlying methodology.

## Design notes

- **FCS reader**: `peacoqc.read_fcs` is a thin wrapper over
  `pytometry.io.read_fcs` that normalizes min/max range metadata from
  `flowcore_p{n}rmin`/`rmax` into `adata.var['min_range']` / `['max_range']`.
- **Isolation Tree**: the R package ships a custom SD-based isolation tree.
  This port uses `sklearn.ensemble.IsolationForest` — the results are
  broadly equivalent but not bit-identical.
- **Smoothing**: per-channel peak trajectories are smoothed with
  `scipy.interpolate.make_smoothing_spline` (GCV-chosen lambda) — a cubic
  smoothing spline, the same family as R's `smooth.spline`. R's explicit
  `spar=0.5` parameter is not reproduced exactly.
- **Output layout**: the port returns a typed `PeacoQCResult` dataclass and
  writes a CSV report by default. The original R-style report TSV is still
  readable by `peaco_qc_heatmap`.

## License

GPL-3.0-or-later (inherited from the upstream R package).
