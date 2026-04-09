# One-off R script to generate parity fixtures for the Python port.
# Run once, from the root of the PeacoQC repository, to produce
# ``tests/data/r_reference_111.json``. Requires the upstream R package
# PeacoQC to be installed.
#
#   Rscript python/tests/fixtures/gen_r_reference.R

suppressPackageStartupMessages({
  library(PeacoQC)
  library(flowCore)
  library(jsonlite)
})

fcs_path <- file.path("inst", "extdata", "111_Comp_Trans.fcs")
ff       <- flowCore::read.FCS(fcs_path)
channels <- c(1, 3, 5:14, 18, 21)

set.seed(0)
res <- PeacoQC(
  ff,
  channels,
  determine_good_cells = "all",
  plot                 = FALSE,
  save_fcs             = FALSE,
  report               = FALSE,
  output_directory     = NULL
)

cluster_medians <- lapply(names(res)[names(res) %in% colnames(flowCore::exprs(ff))],
  function(ch) {
    f <- res[[ch]]
    if (!is.data.frame(f)) return(NULL)
    sapply(split(f$Peak, f$Cluster), median)
  })
names(cluster_medians) <- names(res)[names(res) %in% colnames(flowCore::exprs(ff))]

out <- list(
  percentage_removed     = res$PercentageRemoved,
  it_percentage          = res$ITPercentage,
  mad_percentage         = res$MADPercentage,
  consecutive_percentage = res$ConsecutiveCellsPercentage,
  good_cells_idx_1based  = which(res$GoodCells),
  weird_channel_label    = res$WeirdChannels$Changing_channel,
  events_per_bin         = res$EventsPerBin,
  nr_bins                = res$nr_bins,
  cluster_medians        = cluster_medians
)

out_dir <- file.path("python", "tests", "data")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
writeLines(jsonlite::toJSON(out, auto_unbox = TRUE, pretty = TRUE),
           file.path(out_dir, "r_reference_111.json"))
cat("Wrote", file.path(out_dir, "r_reference_111.json"), "\n")
