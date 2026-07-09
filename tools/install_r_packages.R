options(repos = c(CRAN = "https://cloud.r-project.org"))
packages <- c(
  "data.table", "digest", "dplyr", "ggalluvial", "ggplot2",
  "glmmTMB", "gridExtra", "jsonlite", "MASS", "Matrix",
  "mgcv", "patchwork", "R.utils"
)
missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing) == 0L) {
  cat("All required R packages are already installed.\n")
} else {
  cat("Installing:", paste(missing, collapse = ", "), "\n")
  install.packages(missing)
}
