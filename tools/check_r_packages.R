packages <- c(
  "data.table", "digest", "dplyr", "ggalluvial", "ggplot2",
  "glmmTMB", "gridExtra", "jsonlite", "MASS", "Matrix",
  "mgcv", "patchwork", "R.utils"
)
missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing) > 0L) {
  cat("Missing R packages:\n")
  cat(paste0("  - ", missing, collapse = "\n"), "\n")
  quit(status = 1L)
}
cat("All required R packages are available.\n")
