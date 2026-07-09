# =============================================================================
# Step 03A: Build equal-contribution FAI sensitivity datasets
#
# Recommended location
# --------------------
# <REPOSITORY_ROOT>/11_SEM_mechanism_analysis/00_Script/
# 03A_build_equal_contribution_FAI_sensitivity.R
#
# Purpose
# -------
# This script does not modify the sealed Step 03 outputs. It reads the locked
# Step 03 FAI/PCA master and adds one sensitivity response:
#
#   FAI_EqualContribution_Sensitivity =
#       0.5 * Fire_Count / pooled_mean(Fire_Count)
#     + 0.5 * Burned_Pixel_Count / pooled_mean(Burned_Pixel_Count)
#
# Because the means include all zero and positive rows, the pooled aggregate
# contribution of Fire Count and Burned Pixel Count is exactly 50% / 50%.
#
# ForestPixelCount and Log_ForestPixelCount are preserved unchanged for the
# future exposure offset. No PCA is recomputed and no SEM model is fitted.
# =============================================================================

options(stringsAsFactors = FALSE, warn = 1, scipen = 999)

required_packages <- c("data.table", "jsonlite", "digest", "R.utils")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]

if (length(missing_packages) > 0L) {
  stop(
    paste0(
      "Missing packages: ",
      paste(missing_packages, collapse = ", "),
      "\nRun: install.packages(c(",
      paste(sprintf('"%s"', missing_packages), collapse = ", "),
      "), repos='https://cloud.r-project.org')"
    ),
    call. = FALSE
  )
}

suppressPackageStartupMessages(library(data.table))

# -----------------------------------------------------------------------------
# Locked paths and expectations
# -----------------------------------------------------------------------------

CODE_VERSION <- paste0(
  "2026-07-01_SEM_FAI_SENSITIVITY_V1_",
  "POOLED_EQUAL_AGGREGATE_CONTRIBUTION"
)

EXPECTED_STEP03_VERSION <- paste0(
  "2026-07-01_SEM_FAI_COMPOSITE_DATASET_V1_",
  "POOLED_Q95_FAI_COMMON_OUTCOME_NEUTRAL_PCA"
)

EXPECTED_STEP03_SCRIPT_SHA256 <- paste0(
  "a18d7aafd65958d2177d881ef8a65405",
  "ddb5e85a42ccefa5e5e06181fa803102"
)

script_args <- commandArgs(trailingOnly = FALSE)
script_file_arg <- grep("^--file=", script_args, value = TRUE)
if (length(script_file_arg) > 0L) {
  script_path <- normalizePath(
    sub("^--file=", "", script_file_arg[1L]),
    winslash = "/",
    mustWork = TRUE
  )
  default_code_root <- normalizePath(
    file.path(dirname(script_path), "..", ".."),
    winslash = "/",
    mustWork = TRUE
  )
} else {
  default_code_root <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
}
CODE_ROOT <- Sys.getenv("WILDFIRE_CODE_ROOT", unset = default_code_root)
SEM_ROOT <- file.path(CODE_ROOT, "11_SEM_mechanism_analysis")
SCRIPT_DIR <- file.path(SEM_ROOT, "00_Script")
STEP03_ROOT <- file.path(SEM_ROOT, "03_SEM_FAI_and_Composite_Scores")

STEP03_METHOD_FILE <- file.path(STEP03_ROOT, "00_Method_Definition.json")
STEP03_MASTER_FILE <- file.path(
  STEP03_ROOT,
  "09_Locked_SEM_FAI_Composite_Master.csv.gz"
)
STEP03_SCRIPT_FILE <- file.path(
  SCRIPT_DIR,
  "03_build_FAI_and_common_PCA_scores.R"
)
STEP03_MANIFEST_FILE <- file.path(STEP03_ROOT, "14_Output_Manifest.csv")

OUTPUT_ROOT <- file.path(SEM_ROOT, "03A_SEM_FAI_Sensitivity")
METHOD_FILE <- file.path(OUTPUT_ROOT, "00_Method_Definition.json")
AUDIT_FILE <- file.path(OUTPUT_ROOT, "01_Input_Integrity_Audit.csv")
SCALING_FILE <- file.path(
  OUTPUT_ROOT,
  "02_Equal_Contribution_FAI_Scaling_Constants.csv"
)
DISTRIBUTION_FILE <- file.path(
  OUTPUT_ROOT,
  "03_Equal_Contribution_FAI_Distribution_QA.csv"
)
CONTRIBUTION_FILE <- file.path(
  OUTPUT_ROOT,
  "04_Equal_Contribution_FAI_Component_QA.csv"
)
COMPARISON_FILE <- file.path(
  OUTPUT_ROOT,
  "05_Primary_vs_Equal_Contribution_FAI_Comparison.csv"
)
MASTER_OUT <- file.path(
  OUTPUT_ROOT,
  "06_Locked_SEM_FAI_Sensitivity_Master.csv.gz"
)
SPRING_OUT <- file.path(
  OUTPUT_ROOT,
  "07_Spring_FAI_Sensitivity_Data.csv.gz"
)
SUMMER_OUT <- file.path(
  OUTPUT_ROOT,
  "08_Summer_FAI_Sensitivity_Data.csv.gz"
)
AUTUMN_OUT <- file.path(
  OUTPUT_ROOT,
  "09_Autumn_FAI_Sensitivity_Data.csv.gz"
)
MANIFEST_FILE <- file.path(OUTPUT_ROOT, "10_Output_Manifest.csv")
SOFTWARE_FILE <- file.path(OUTPUT_ROOT, "Software_Environment.json")
LOG_FILE <- file.path(OUTPUT_ROOT, "sem_step03A_fai_sensitivity.log")

EXPECTED_ROWS <- 371187L
EXPECTED_ROWS_PER_SEASON <- 123729L
EXPECTED_GRIDS <- 4982L

REQUIRED_FIELDS <- c(
  "SEM_Row_ID",
  "GRID_UID",
  "Country",
  "Year",
  "Year_Z",
  "Season",
  "Season_Label",
  "ForestPixelCount",
  "Log_ForestPixelCount",
  "Fire_Count",
  "Burned_Pixel_Count",
  "FAI_Q95",
  "FAI_Max_Sensitivity",
  "Meteorology_Score",
  "Vegetation_Score",
  "Topography_Score",
  "Anthropogenic_Score"
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

timestamp_text <- function() format(Sys.time(), "%Y-%m-%d %H:%M:%S")

log_message <- function(message) {
  line <- sprintf("[%s] %s", timestamp_text(), message)
  cat(line, "\n")
  flush.console()
  cat(line, "\n", file = LOG_FILE, append = TRUE, sep = "")
}

sha256_file <- function(path) {
  digest::digest(file = path, algo = "sha256", serialize = FALSE)
}

read_json <- function(path) {
  jsonlite::read_json(path, simplifyVector = TRUE)
}

write_json <- function(x, path) {
  jsonlite::write_json(
    x,
    path = path,
    auto_unbox = TRUE,
    pretty = TRUE,
    null = "null",
    na = "null",
    digits = 16
  )
}

parse_bool <- function(x) {
  if (is.logical(x)) return(x)
  tolower(trimws(as.character(x))) %in% c("true", "1", "yes", "y")
}

safe_spearman <- function(x, y) {
  as.numeric(
    suppressWarnings(
      stats::cor(x, y, method = "spearman", use = "complete.obs")
    )
  )
}

safe_pearson <- function(x, y) {
  as.numeric(
    suppressWarnings(
      stats::cor(x, y, method = "pearson", use = "complete.obs")
    )
  )
}

script_path <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)

  if (length(file_arg) == 1L) {
    return(
      normalizePath(
        sub("^--file=", "", file_arg),
        winslash = "/",
        mustWork = FALSE
      )
    )
  }

  normalizePath(
    file.path(
      SCRIPT_DIR,
      "03A_build_equal_contribution_FAI_sensitivity.R"
    ),
    winslash = "/",
    mustWork = FALSE
  )
}

add_check <- function(
  rows,
  category,
  check,
  passed,
  observed,
  expected,
  severity = "ERROR",
  detail = ""
) {
  rows[[length(rows) + 1L]] <- data.table(
    Category = category,
    Check = check,
    Severity = severity,
    Passed = isTRUE(passed),
    Observed = paste(observed, collapse = ";"),
    Expected = paste(expected, collapse = ";"),
    Detail = detail
  )
  rows
}

# -----------------------------------------------------------------------------
# Initialize
# -----------------------------------------------------------------------------

if (dir.exists(OUTPUT_ROOT)) {
  stop(
    paste0(
      "Output directory already exists:\n",
      OUTPUT_ROOT,
      "\nDelete or rename it before an intentional full rerun."
    ),
    call. = FALSE
  )
}

dir.create(OUTPUT_ROOT, recursive = TRUE, showWarnings = FALSE)
writeLines(character(), LOG_FILE)

log_message("Starting SEM Step 03A equal-contribution FAI sensitivity.")
log_message(paste("Code version:", CODE_VERSION))

required_files <- c(
  STEP03_METHOD_FILE,
  STEP03_MASTER_FILE,
  STEP03_SCRIPT_FILE,
  STEP03_MANIFEST_FILE
)

missing_files <- required_files[!file.exists(required_files)]

if (length(missing_files) > 0L) {
  stop(
    paste(
      "Missing required Step 03 files:",
      paste(missing_files, collapse = "\n"),
      sep = "\n"
    ),
    call. = FALSE
  )
}

# -----------------------------------------------------------------------------
# Verify the sealed Step 03 source
# -----------------------------------------------------------------------------

step03_method <- read_json(STEP03_METHOD_FILE)
step03_manifest <- fread(STEP03_MANIFEST_FILE)

audit <- list()

audit <- add_check(
  audit,
  "Version",
  "Step03_code_version",
  identical(step03_method$Code_Version, EXPECTED_STEP03_VERSION),
  step03_method$Code_Version,
  EXPECTED_STEP03_VERSION
)

actual_step03_script_hash <- sha256_file(STEP03_SCRIPT_FILE)

audit <- add_check(
  audit,
  "Hash",
  "Step03_script_hash_actual",
  identical(actual_step03_script_hash, EXPECTED_STEP03_SCRIPT_SHA256),
  actual_step03_script_hash,
  EXPECTED_STEP03_SCRIPT_SHA256
)

audit <- add_check(
  audit,
  "Hash",
  "Step03_script_hash_recorded",
  identical(
    step03_method$Script_SHA256,
    EXPECTED_STEP03_SCRIPT_SHA256
  ),
  step03_method$Script_SHA256,
  EXPECTED_STEP03_SCRIPT_SHA256
)

master_manifest_row <- step03_manifest[
  gsub("\\\\", "/", Relative_Path) ==
    "09_Locked_SEM_FAI_Composite_Master.csv.gz"
]

audit <- add_check(
  audit,
  "Manifest",
  "Step03_master_manifest_row_unique",
  nrow(master_manifest_row) == 1L,
  nrow(master_manifest_row),
  1L
)

if (nrow(master_manifest_row) != 1L) {
  stop(
    "Step 03 master manifest entry is absent or duplicated.",
    call. = FALSE
  )
}

actual_master_hash <- sha256_file(STEP03_MASTER_FILE)
actual_master_size <- file.info(STEP03_MASTER_FILE)$size

audit <- add_check(
  audit,
  "Manifest",
  "Step03_master_hash_matches_manifest",
  identical(actual_master_hash, master_manifest_row$SHA256[[1L]]),
  actual_master_hash,
  master_manifest_row$SHA256[[1L]]
)

audit <- add_check(
  audit,
  "Manifest",
  "Step03_master_size_matches_manifest",
  identical(
    as.numeric(actual_master_size),
    as.numeric(master_manifest_row$File_Size_Bytes[[1L]])
  ),
  actual_master_size,
  master_manifest_row$File_Size_Bytes[[1L]]
)

# -----------------------------------------------------------------------------
# Read the locked Step 03 master
# -----------------------------------------------------------------------------

log_message("Reading the locked Step 03 FAI/PCA master.")

data <- fread(STEP03_MASTER_FILE, showProgress = TRUE)

missing_fields <- setdiff(REQUIRED_FIELDS, names(data))

audit <- add_check(
  audit,
  "Input",
  "All_required_fields_present",
  length(missing_fields) == 0L,
  paste(missing_fields, collapse = ";"),
  "No missing required field"
)

if (length(missing_fields) > 0L) {
  fwrite(rbindlist(audit, fill = TRUE), AUDIT_FILE)
  stop(
    paste(
      "Missing required fields:",
      paste(missing_fields, collapse = ", ")
    ),
    call. = FALSE
  )
}

data[, GRID_UID := as.character(GRID_UID)]
data[, Year := as.integer(Year)]
data[, Season := as.integer(Season)]
data[, Fire_Count := as.numeric(Fire_Count)]
data[, Burned_Pixel_Count := as.numeric(Burned_Pixel_Count)]
data[, ForestPixelCount := as.numeric(ForestPixelCount)]
data[, Log_ForestPixelCount := as.numeric(Log_ForestPixelCount)]
data[, FAI_Q95 := as.numeric(FAI_Q95)]

audit <- add_check(
  audit,
  "Universe",
  "Panel_rows",
  nrow(data) == EXPECTED_ROWS,
  nrow(data),
  EXPECTED_ROWS
)

audit <- add_check(
  audit,
  "Universe",
  "Unique_GRID_UIDs",
  uniqueN(data$GRID_UID) == EXPECTED_GRIDS,
  uniqueN(data$GRID_UID),
  EXPECTED_GRIDS
)

season_counts <- data[, .N, by = Season][order(Season)]

audit <- add_check(
  audit,
  "Universe",
  "Rows_per_season",
  nrow(season_counts) == 3L &&
    all(season_counts$N == EXPECTED_ROWS_PER_SEASON),
  paste(season_counts$N, collapse = ";"),
  paste(rep(EXPECTED_ROWS_PER_SEASON, 3L), collapse = ";")
)

audit <- add_check(
  audit,
  "Universe",
  "Unique_panel_keys",
  !anyDuplicated(data[, .(GRID_UID, Year, Season)]),
  anyDuplicated(data[, .(GRID_UID, Year, Season)]),
  0L
)

audit <- add_check(
  audit,
  "Response",
  "Counts_nonnegative_and_finite",
  all(is.finite(data$Fire_Count)) &&
    all(is.finite(data$Burned_Pixel_Count)) &&
    all(data$Fire_Count >= 0) &&
    all(data$Burned_Pixel_Count >= 0),
  paste(
    min(data$Fire_Count),
    min(data$Burned_Pixel_Count),
    sep = ";"
  ),
  "Both minima >=0"
)

audit <- add_check(
  audit,
  "Exposure",
  "Exposure_preserved_and_valid",
  all(is.finite(data$ForestPixelCount)) &&
    all(data$ForestPixelCount > 0) &&
    max(
      abs(
        data$Log_ForestPixelCount -
          log(data$ForestPixelCount)
      )
    ) < 1e-10,
  max(
    abs(
      data$Log_ForestPixelCount -
        log(data$ForestPixelCount)
    )
  ),
  "<1e-10"
)

# -----------------------------------------------------------------------------
# Build equal-contribution sensitivity FAI
# -----------------------------------------------------------------------------

log_message("Constructing pooled equal-contribution FAI sensitivity.")

pooled_mean_fc <- mean(data$Fire_Count)
pooled_mean_ba <- mean(data$Burned_Pixel_Count)

if (
  !is.finite(pooled_mean_fc) ||
    !is.finite(pooled_mean_ba) ||
    pooled_mean_fc <= 0 ||
    pooled_mean_ba <= 0
) {
  stop(
    "Pooled FC or BA mean is not positive and finite.",
    call. = FALSE
  )
}

data[
  ,
  `:=`(
    FAI_Equal_FC_Component = (
      0.5 * Fire_Count / pooled_mean_fc
    ),
    FAI_Equal_BA_Component = (
      0.5 * Burned_Pixel_Count / pooled_mean_ba
    )
  )
]

data[
  ,
  FAI_EqualContribution_Sensitivity := (
    FAI_Equal_FC_Component +
      FAI_Equal_BA_Component
  )
]

data[
  ,
  FAI_Equal_per_ForestPixel := (
    FAI_EqualContribution_Sensitivity /
      ForestPixelCount
  )
]

data[
  ,
  FAI_Equal_FC_Share := fifelse(
    FAI_EqualContribution_Sensitivity > 0,
    FAI_Equal_FC_Component /
      FAI_EqualContribution_Sensitivity,
    NA_real_
  )
]

data[
  ,
  FAI_Equal_BA_Share := fifelse(
    FAI_EqualContribution_Sensitivity > 0,
    FAI_Equal_BA_Component /
      FAI_EqualContribution_Sensitivity,
    NA_real_
  )
]

total_equal_fai <- sum(
  data$FAI_EqualContribution_Sensitivity
)

overall_fc_share <- sum(
  data$FAI_Equal_FC_Component
) / total_equal_fai

overall_ba_share <- sum(
  data$FAI_Equal_BA_Component
) / total_equal_fai

audit <- add_check(
  audit,
  "Sensitivity_FAI",
  "Equal_FAI_nonnegative_and_finite",
  all(
    is.finite(
      data$FAI_EqualContribution_Sensitivity
    )
  ) &&
    all(
      data$FAI_EqualContribution_Sensitivity >= 0
    ),
  paste(
    min(data$FAI_EqualContribution_Sensitivity),
    max(data$FAI_EqualContribution_Sensitivity),
    sep = ";"
  ),
  "Finite and minimum >=0"
)

audit <- add_check(
  audit,
  "Sensitivity_FAI",
  "Equal_FAI_zero_equivalence",
  all(
    (
      data$FAI_EqualContribution_Sensitivity == 0
    ) ==
      (
        data$Fire_Count == 0 &
          data$Burned_Pixel_Count == 0
      )
  ),
  sum(
    (
      data$FAI_EqualContribution_Sensitivity == 0
    ) !=
      (
        data$Fire_Count == 0 &
          data$Burned_Pixel_Count == 0
      )
  ),
  0L
)

audit <- add_check(
  audit,
  "Sensitivity_FAI",
  "Equal_components_sum_exactly",
  max(
    abs(
      data$FAI_EqualContribution_Sensitivity -
        (
          data$FAI_Equal_FC_Component +
            data$FAI_Equal_BA_Component
        )
    )
  ) < 1e-12,
  max(
    abs(
      data$FAI_EqualContribution_Sensitivity -
        (
          data$FAI_Equal_FC_Component +
            data$FAI_Equal_BA_Component
        )
    )
  ),
  "<1e-12"
)

audit <- add_check(
  audit,
  "Sensitivity_FAI",
  "Overall_FC_contribution_exactly_half",
  abs(overall_fc_share - 0.5) < 1e-12,
  overall_fc_share,
  0.5
)

audit <- add_check(
  audit,
  "Sensitivity_FAI",
  "Overall_BA_contribution_exactly_half",
  abs(overall_ba_share - 0.5) < 1e-12,
  overall_ba_share,
  0.5
)

audit <- add_check(
  audit,
  "Sensitivity_FAI",
  "Positive_row_shares_sum_one",
  max(
    abs(
      data[
        FAI_EqualContribution_Sensitivity > 0,
        FAI_Equal_FC_Share +
          FAI_Equal_BA_Share
      ] -
        1
    )
  ) < 1e-12,
  max(
    abs(
      data[
        FAI_EqualContribution_Sensitivity > 0,
        FAI_Equal_FC_Share +
          FAI_Equal_BA_Share
      ] -
        1
    )
  ),
  "<1e-12"
)

# -----------------------------------------------------------------------------
# QA summaries
# -----------------------------------------------------------------------------

scaling <- data.table(
  Sensitivity_Field = "FAI_EqualContribution_Sensitivity",
  Component = c("Fire_Count", "Burned_Pixel_Count"),
  Scaling_Statistic = "Pooled arithmetic mean including zero rows",
  Scaling_Value = c(pooled_mean_fc, pooled_mean_ba),
  Nominal_Weight = 0.5,
  Pooled_Across_Seasons = TRUE,
  Pooled_Across_Years = TRUE,
  Pooled_Across_Countries = TRUE,
  Aggregate_Contribution_Share = c(
    overall_fc_share,
    overall_ba_share
  ),
  Exposure_Inside_Index = FALSE,
  Future_Exposure_Treatment = (
    "Use offset(Log_ForestPixelCount) in the SEM outcome model"
  )
)

fwrite(scaling, SCALING_FILE)

distribution <- data[
  ,
  .(
    N = .N,
    Zero_N = sum(
      FAI_EqualContribution_Sensitivity == 0
    ),
    Zero_Percent = 100 * mean(
      FAI_EqualContribution_Sensitivity == 0
    ),
    Positive_N = sum(
      FAI_EqualContribution_Sensitivity > 0
    ),
    Mean = mean(
      FAI_EqualContribution_Sensitivity
    ),
    SD = stats::sd(
      FAI_EqualContribution_Sensitivity
    ),
    Minimum = min(
      FAI_EqualContribution_Sensitivity
    ),
    Median = stats::median(
      FAI_EqualContribution_Sensitivity
    ),
    Q90 = as.numeric(
      stats::quantile(
        FAI_EqualContribution_Sensitivity,
        0.90,
        names = FALSE,
        type = 7
      )
    ),
    Q95 = as.numeric(
      stats::quantile(
        FAI_EqualContribution_Sensitivity,
        0.95,
        names = FALSE,
        type = 7
      )
    ),
    Q99 = as.numeric(
      stats::quantile(
        FAI_EqualContribution_Sensitivity,
        0.99,
        names = FALSE,
        type = 7
      )
    ),
    Maximum = max(
      FAI_EqualContribution_Sensitivity
    ),
    Positive_Mean = if (
      any(
        FAI_EqualContribution_Sensitivity > 0
      )
    ) {
      mean(
        FAI_EqualContribution_Sensitivity[
          FAI_EqualContribution_Sensitivity > 0
        ]
      )
    } else {
      NA_real_
    },
    Positive_Median = if (
      any(
        FAI_EqualContribution_Sensitivity > 0
      )
    ) {
      stats::median(
        FAI_EqualContribution_Sensitivity[
          FAI_EqualContribution_Sensitivity > 0
        ]
      )
    } else {
      NA_real_
    }
  ),
  by = .(
    Season,
    Season_Label,
    Country
  )
]

overall_distribution <- data[
  ,
  .(
    Season = 0L,
    Season_Label = "All_Seasons",
    Country = "All_Countries",
    N = .N,
    Zero_N = sum(
      FAI_EqualContribution_Sensitivity == 0
    ),
    Zero_Percent = 100 * mean(
      FAI_EqualContribution_Sensitivity == 0
    ),
    Positive_N = sum(
      FAI_EqualContribution_Sensitivity > 0
    ),
    Mean = mean(
      FAI_EqualContribution_Sensitivity
    ),
    SD = stats::sd(
      FAI_EqualContribution_Sensitivity
    ),
    Minimum = min(
      FAI_EqualContribution_Sensitivity
    ),
    Median = stats::median(
      FAI_EqualContribution_Sensitivity
    ),
    Q90 = as.numeric(
      stats::quantile(
        FAI_EqualContribution_Sensitivity,
        0.90,
        names = FALSE,
        type = 7
      )
    ),
    Q95 = as.numeric(
      stats::quantile(
        FAI_EqualContribution_Sensitivity,
        0.95,
        names = FALSE,
        type = 7
      )
    ),
    Q99 = as.numeric(
      stats::quantile(
        FAI_EqualContribution_Sensitivity,
        0.99,
        names = FALSE,
        type = 7
      )
    ),
    Maximum = max(
      FAI_EqualContribution_Sensitivity
    ),
    Positive_Mean = mean(
      FAI_EqualContribution_Sensitivity[
        FAI_EqualContribution_Sensitivity > 0
      ]
    ),
    Positive_Median = stats::median(
      FAI_EqualContribution_Sensitivity[
        FAI_EqualContribution_Sensitivity > 0
      ]
    )
  )
]

distribution <- rbindlist(
  list(overall_distribution, distribution),
  fill = TRUE
)

setorder(distribution, Season, Country)
fwrite(distribution, DISTRIBUTION_FILE)

contribution <- data[
  ,
  .(
    N = .N,
    Positive_FAI_N = sum(
      FAI_EqualContribution_Sensitivity > 0
    ),
    Total_Equal_FAI = sum(
      FAI_EqualContribution_Sensitivity
    ),
    Total_FC_Component = sum(
      FAI_Equal_FC_Component
    ),
    Total_BA_Component = sum(
      FAI_Equal_BA_Component
    ),
    Aggregate_FC_Contribution_Share = (
      sum(FAI_Equal_FC_Component) /
        sum(FAI_EqualContribution_Sensitivity)
    ),
    Aggregate_BA_Contribution_Share = (
      sum(FAI_Equal_BA_Component) /
        sum(FAI_EqualContribution_Sensitivity)
    ),
    Mean_Positive_Row_FC_Share = mean(
      FAI_Equal_FC_Share,
      na.rm = TRUE
    ),
    Mean_Positive_Row_BA_Share = mean(
      FAI_Equal_BA_Share,
      na.rm = TRUE
    ),
    FC_Only_Positive_N = sum(
      Fire_Count > 0 &
        Burned_Pixel_Count == 0
    ),
    BA_Only_Positive_N = sum(
      Fire_Count == 0 &
        Burned_Pixel_Count > 0
    ),
    Both_Positive_N = sum(
      Fire_Count > 0 &
        Burned_Pixel_Count > 0
    )
  ),
  by = .(
    Season,
    Season_Label,
    Country
  )
]

overall_contribution <- data[
  ,
  .(
    Season = 0L,
    Season_Label = "All_Seasons",
    Country = "All_Countries",
    N = .N,
    Positive_FAI_N = sum(
      FAI_EqualContribution_Sensitivity > 0
    ),
    Total_Equal_FAI = sum(
      FAI_EqualContribution_Sensitivity
    ),
    Total_FC_Component = sum(
      FAI_Equal_FC_Component
    ),
    Total_BA_Component = sum(
      FAI_Equal_BA_Component
    ),
    Aggregate_FC_Contribution_Share = (
      sum(FAI_Equal_FC_Component) /
        sum(FAI_EqualContribution_Sensitivity)
    ),
    Aggregate_BA_Contribution_Share = (
      sum(FAI_Equal_BA_Component) /
        sum(FAI_EqualContribution_Sensitivity)
    ),
    Mean_Positive_Row_FC_Share = mean(
      FAI_Equal_FC_Share,
      na.rm = TRUE
    ),
    Mean_Positive_Row_BA_Share = mean(
      FAI_Equal_BA_Share,
      na.rm = TRUE
    ),
    FC_Only_Positive_N = sum(
      Fire_Count > 0 &
        Burned_Pixel_Count == 0
    ),
    BA_Only_Positive_N = sum(
      Fire_Count == 0 &
        Burned_Pixel_Count > 0
    ),
    Both_Positive_N = sum(
      Fire_Count > 0 &
        Burned_Pixel_Count > 0
    )
  )
]

contribution <- rbindlist(
  list(overall_contribution, contribution),
  fill = TRUE
)

setorder(contribution, Season, Country)
fwrite(contribution, CONTRIBUTION_FILE)

comparison <- data[
  ,
  .(
    N = .N,
    Spearman_Primary_vs_Equal = safe_spearman(
      FAI_Q95,
      FAI_EqualContribution_Sensitivity
    ),
    Pearson_Primary_vs_Equal = safe_pearson(
      FAI_Q95,
      FAI_EqualContribution_Sensitivity
    ),
    Spearman_Equal_vs_FC = safe_spearman(
      FAI_EqualContribution_Sensitivity,
      Fire_Count
    ),
    Spearman_Equal_vs_BA = safe_spearman(
      FAI_EqualContribution_Sensitivity,
      Burned_Pixel_Count
    ),
    Spearman_Primary_vs_FC = safe_spearman(
      FAI_Q95,
      Fire_Count
    ),
    Spearman_Primary_vs_BA = safe_spearman(
      FAI_Q95,
      Burned_Pixel_Count
    ),
    Mean_Primary_FAI = mean(FAI_Q95),
    Mean_Equal_FAI = mean(
      FAI_EqualContribution_Sensitivity
    ),
    Zero_Percent = 100 * mean(
      FAI_EqualContribution_Sensitivity == 0
    )
  ),
  by = .(
    Season,
    Season_Label,
    Country
  )
]

overall_comparison <- data[
  ,
  .(
    Season = 0L,
    Season_Label = "All_Seasons",
    Country = "All_Countries",
    N = .N,
    Spearman_Primary_vs_Equal = safe_spearman(
      FAI_Q95,
      FAI_EqualContribution_Sensitivity
    ),
    Pearson_Primary_vs_Equal = safe_pearson(
      FAI_Q95,
      FAI_EqualContribution_Sensitivity
    ),
    Spearman_Equal_vs_FC = safe_spearman(
      FAI_EqualContribution_Sensitivity,
      Fire_Count
    ),
    Spearman_Equal_vs_BA = safe_spearman(
      FAI_EqualContribution_Sensitivity,
      Burned_Pixel_Count
    ),
    Spearman_Primary_vs_FC = safe_spearman(
      FAI_Q95,
      Fire_Count
    ),
    Spearman_Primary_vs_BA = safe_spearman(
      FAI_Q95,
      Burned_Pixel_Count
    ),
    Mean_Primary_FAI = mean(FAI_Q95),
    Mean_Equal_FAI = mean(
      FAI_EqualContribution_Sensitivity
    ),
    Zero_Percent = 100 * mean(
      FAI_EqualContribution_Sensitivity == 0
    )
  )
]

comparison <- rbindlist(
  list(overall_comparison, comparison),
  fill = TRUE
)

setorder(comparison, Season, Country)
fwrite(comparison, COMPARISON_FILE)

# -----------------------------------------------------------------------------
# Write augmented locked datasets
# -----------------------------------------------------------------------------

log_message("Writing pooled and season-specific sensitivity datasets.")

setorder(data, Season, GRID_UID, Year)

fwrite(data, MASTER_OUT, compress = "gzip")
fwrite(data[Season == 1L], SPRING_OUT, compress = "gzip")
fwrite(data[Season == 2L], SUMMER_OUT, compress = "gzip")
fwrite(data[Season == 3L], AUTUMN_OUT, compress = "gzip")

audit <- add_check(
  audit,
  "Output",
  "Sensitivity_master_rows",
  nrow(data) == EXPECTED_ROWS,
  nrow(data),
  EXPECTED_ROWS
)

audit <- add_check(
  audit,
  "Output",
  "Sensitivity_season_rows",
  all(
    c(
      nrow(data[Season == 1L]),
      nrow(data[Season == 2L]),
      nrow(data[Season == 3L])
    ) == EXPECTED_ROWS_PER_SEASON
  ),
  paste(
    c(
      nrow(data[Season == 1L]),
      nrow(data[Season == 2L]),
      nrow(data[Season == 3L])
    ),
    collapse = ";"
  ),
  paste(
    rep(EXPECTED_ROWS_PER_SEASON, 3L),
    collapse = ";"
  )
)

new_fields <- c(
  "FAI_Equal_FC_Component",
  "FAI_Equal_BA_Component",
  "FAI_EqualContribution_Sensitivity",
  "FAI_Equal_per_ForestPixel",
  "FAI_Equal_FC_Share",
  "FAI_Equal_BA_Share"
)

audit <- add_check(
  audit,
  "Output",
  "All_new_fields_present",
  all(new_fields %in% names(data)),
  sum(new_fields %in% names(data)),
  length(new_fields)
)

audit <- add_check(
  audit,
  "Output",
  "No_missing_required_future_model_fields",
  all(
    complete.cases(
      data[
        ,
        c(
          "GRID_UID",
          "Country",
          "Year",
          "Year_Z",
          "Season",
          "ForestPixelCount",
          "Log_ForestPixelCount",
          "FAI_Q95",
          "FAI_EqualContribution_Sensitivity",
          "Meteorology_Score",
          "Vegetation_Score",
          "Topography_Score",
          "Anthropogenic_Score"
        ),
        with = FALSE
      ]
    )
  ),
  sum(
    !complete.cases(
      data[
        ,
        c(
          "GRID_UID",
          "Country",
          "Year",
          "Year_Z",
          "Season",
          "ForestPixelCount",
          "Log_ForestPixelCount",
          "FAI_Q95",
          "FAI_EqualContribution_Sensitivity",
          "Meteorology_Score",
          "Vegetation_Score",
          "Topography_Score",
          "Anthropogenic_Score"
        ),
        with = FALSE
      ]
    )
  ),
  0L
)

final_audit <- rbindlist(audit, fill = TRUE)
fwrite(final_audit, AUDIT_FILE)

error_failures <- final_audit[
  Severity == "ERROR" &
    !Passed
]
warning_failures <- final_audit[
  Severity == "WARNING" &
    !Passed
]

if (nrow(error_failures) > 0L) {
  stop(
    "Step 03A audit failed. See 01_Input_Integrity_Audit.csv.",
    call. = FALSE
  )
}

# -----------------------------------------------------------------------------
# Method and software metadata
# -----------------------------------------------------------------------------

method_definition <- list(
  Code_Version = CODE_VERSION,
  Created = as.character(Sys.time()),
  Script_Path = script_path(),
  Script_SHA256 = if (file.exists(script_path())) {
    sha256_file(script_path())
  } else {
    NA_character_
  },
  Step03_Code_Version = step03_method$Code_Version,
  Step03_Script_SHA256 = step03_method$Script_SHA256,
  Step03_Master = normalizePath(
    STEP03_MASTER_FILE,
    winslash = "/",
    mustWork = TRUE
  ),
  Step03_Master_SHA256 = actual_master_hash,
  Primary_FAI_Preserved = "FAI_Q95",
  Sensitivity_Field = "FAI_EqualContribution_Sensitivity",
  Formula = paste0(
    "0.5 * Fire_Count / pooled mean(Fire_Count) + ",
    "0.5 * Burned_Pixel_Count / pooled mean(Burned_Pixel_Count)"
  ),
  Means_Include_Zero_Rows = TRUE,
  Pooled_Across_Seasons = TRUE,
  Pooled_Across_Years = TRUE,
  Pooled_Across_Countries = TRUE,
  Pooled_Aggregate_FC_Contribution = overall_fc_share,
  Pooled_Aggregate_BA_Contribution = overall_ba_share,
  PCA_Recomputed = FALSE,
  Exposure_Field = "ForestPixelCount",
  Offset_Field = "Log_ForestPixelCount",
  Exposure_Inside_Sensitivity_FAI = FALSE,
  No_SEM_Model_Fitting = TRUE,
  No_Path_Selection = TRUE,
  No_Final_PiecewiseSEM_Assembly = TRUE,
  No_Direct_Indirect_Effect_Calculation = TRUE
)

write_json(method_definition, METHOD_FILE)

software_environment <- list(
  R_Version = R.version.string,
  Platform = R.version$platform,
  data_table = as.character(packageVersion("data.table")),
  jsonlite = as.character(packageVersion("jsonlite")),
  digest = as.character(packageVersion("digest")),
  R_utils = as.character(packageVersion("R.utils"))
)

write_json(software_environment, SOFTWARE_FILE)

# Completion log is written before the output manifest so its hash stays fixed.
log_message(
  paste(
    "Step 03A completed: the pooled equal-contribution FAI sensitivity",
    "was added without changing the primary FAI or PCA scores."
  )
)

# -----------------------------------------------------------------------------
# Output manifest
# -----------------------------------------------------------------------------

manifest_files <- list.files(
  OUTPUT_ROOT,
  recursive = TRUE,
  full.names = TRUE,
  include.dirs = FALSE
)

manifest_files <- manifest_files[
  normalizePath(
    manifest_files,
    winslash = "/",
    mustWork = FALSE
  ) !=
    normalizePath(
      MANIFEST_FILE,
      winslash = "/",
      mustWork = FALSE
    )
]

manifest <- rbindlist(
  lapply(
    sort(manifest_files),
    function(path) {
      data.table(
        Relative_Path = substring(
          normalizePath(path, winslash = "/", mustWork = TRUE),
          nchar(
            normalizePath(
              OUTPUT_ROOT,
              winslash = "/",
              mustWork = TRUE
            )
          ) + 2L
        ),
        File_Size_Bytes = file.info(path)$size,
        SHA256 = sha256_file(path)
      )
    }
  ),
  fill = TRUE
)

fwrite(manifest, MANIFEST_FILE)

# -----------------------------------------------------------------------------
# Console completion summary
# -----------------------------------------------------------------------------

cat("\n")
cat(
  "Output directory:",
  normalizePath(OUTPUT_ROOT, winslash = "/", mustWork = FALSE),
  "\n"
)
cat("Audit ERROR failures:", nrow(error_failures), "\n")
cat("Audit WARNING failures:", nrow(warning_failures), "\n")
cat("Locked sensitivity master rows:", nrow(data), "/", EXPECTED_ROWS, "\n")
cat("Unique GRID_UIDs:", uniqueN(data$GRID_UID), "/", EXPECTED_GRIDS, "\n")
cat("Pooled mean Fire Count:", format(pooled_mean_fc, digits = 12), "\n")
cat(
  "Pooled mean Burned Pixel Count:",
  format(pooled_mean_ba, digits = 12),
  "\n"
)
cat(
  "Overall equal-FAI FC contribution:",
  format(overall_fc_share, digits = 12),
  "\n"
)
cat(
  "Overall equal-FAI BA contribution:",
  format(overall_ba_share, digits = 12),
  "\n"
)
cat(
  "Spearman primary vs equal-contribution FAI:",
  format(
    safe_spearman(
      data$FAI_Q95,
      data$FAI_EqualContribution_Sensitivity
    ),
    digits = 12
  ),
  "\n"
)
cat("PCA scores recomputed: False\n")
cat("SEM models fitted: 0\n")
cat("Final piecewiseSEM assembled: False\n")
cat("Direct/indirect effects calculated: False\n")
