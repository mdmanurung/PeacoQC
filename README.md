# PeacoQC

Peak-based selection of high quality cytometry data.

PeacoQC provides quality-control functions that remove outliers and
unstable events introduced by e.g. clogs or speed changes during
acquisition, and visualises QC results for single samples or whole
experiments.

## Status

 This repository now hosts the **Python** port of PeacoQC, which is the
 actively maintained implementation. The original R package (previously
 on Bioconductor as `saeyslab/PeacoQC`) is being retired; its source has
 been removed from this repo. The only R file kept is
 `python/tests/fixtures/gen_r_reference.R`, a one-off script used to
 generate parity fixtures for the Python test suite.

GitHub Actions is configured to validate the Python port with test,
documentation, and packaging checks, and GitHub releases can publish
the Python distribution to PyPI.

## Installation and usage

See [`python/README.md`](python/README.md) for installation and a short
overview, and [`python/docs/quickstart.md`](python/docs/quickstart.md)
for a worked example.

```python
import peacoqc

adata = peacoqc.read_fcs("sample.fcs")
result = peacoqc.peaco_qc(adata, channels=[...])
```

## Methodology

The underlying method is described in the original PeacoQC publication
and the legacy Bioconductor
[vignette](https://bioconductor.org/packages/release/bioc/vignettes/PeacoQC/inst/doc/PeacoQC.html).
Differences between the R and Python implementations are listed under
"Design notes" in `python/README.md`.

## License

GPL-3.0-or-later.
