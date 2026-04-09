# Quickstart

This is a short walk-through of the Python port of PeacoQC on the example
FCS file that ships with the upstream R package.

```python
import warnings
warnings.filterwarnings("ignore")  # anndata + pytometry produce some deprecation noise

import peacoqc

# 1. Load a compensated + transformed FCS file.
adata = peacoqc.read_fcs("inst/extdata/111_Comp_Trans.fcs")
print(adata.shape)  # (9617, 22)

# The R vignette uses channels c(1, 3, 5:14, 18, 21) -> zero-based:
channel_indices = [0, 2] + list(range(4, 14)) + [17, 20]
channel_names   = [adata.var_names[i] for i in channel_indices]

# 2. Pre-processing (optional; can be done before read_fcs on raw files too).
adata = peacoqc.remove_margins(adata, channels=channel_names)
adata = peacoqc.remove_doublets(adata, channel1="FSC-A", channel2="FSC-H")

# 3. Run the full PeacoQC pipeline.
result = peacoqc.peaco_qc(
    adata,
    channels=channel_names,
    determine_good_cells="all",
    report_path="peacoqc_report.csv",
)
print(f"{result.percentage_removed:.2f}% removed")
print(f"  IT: {result.it_percentage}")
print(f"  MAD: {result.mad_percentage}")
print(f"  Consecutive: {result.consecutive_percentage}")

# 4. Cleaned AnnData — ready for scanpy downstream.
clean = result.adata
clean.obs["Original_ID"].head()

# 5. Diagnostic plots and heatmap.
peacoqc.plot_peaco_qc(adata, result, output_path="sample_qc.png")
peacoqc.peaco_qc_heatmap("peacoqc_report.csv", output_path="heatmap.png")

# 6. (Optional) write the cleaned AnnData back to FCS with flowio.
# Requires the extra: pip install "peacoqc[fcs]"
peacoqc.write_fcs(clean, "sample_qc.fcs")
```

## Selecting analysis modes

- `determine_good_cells="all"` runs both the IsolationForest and MAD steps.
- `"IT"` runs only the IsolationForest step.
- `"MAD"` runs only the MAD step.
- `False` returns the peak frames only, without any event removal (useful
  for just inspecting peaks).

Note that the IsolationForest step is only performed when the number of
bins is at least `force_it` (150 by default). For small files you will
either need to lower `force_it` or shrink `events_per_bin`.

## Annotating the original AnnData instead of slicing

```python
result = peacoqc.peaco_qc(adata, channels=channel_names)
result.annotate(adata)  # in-place
adata.obs["peacoqc_good"].sum()
```

This writes boolean columns (`peacoqc_good`, `peacoqc_outlier_it`,
`peacoqc_outlier_mad`, `peacoqc_consecutive`) onto `adata.obs` and copies
the QC summary onto `adata.uns['peacoqc']`.
