# =============================================================================
# Step 03: Build grid-year-season FAI and four common PCA composite scores
# Save as:
# <REPOSITORY_ROOT>/11_SEM_mechanism_analysis/00_Script/
# 03_build_FAI_and_common_PCA_scores.R
# =============================================================================

options(stringsAsFactors = FALSE, warn = 1, scipen = 999)

required_packages <- c("data.table", "jsonlite", "digest", "R.utils")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages) > 0L) {
  stop(
    paste0(
      "Missing packages: ", paste(missing_packages, collapse = ", "),
      "\nRun: install.packages(c(",
      paste(sprintf('"%s"', missing_packages), collapse = ", "),
      "), repos='https://cloud.r-project.org')"
    ),
    call. = FALSE
  )
}

suppressPackageStartupMessages(library(data.table))

# -----------------------------------------------------------------------------
# Locked paths and settings
# -----------------------------------------------------------------------------

CODE_VERSION <- paste0(
  "2026-07-01_SEM_FAI_COMPOSITE_DATASET_V1_",
  "POOLED_Q95_FAI_COMMON_OUTCOME_NEUTRAL_PCA"
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
STEP01_ROOT <- file.path(SEM_ROOT, "01_SEM_Analysis_Dataset")
STEP02_ROOT <- file.path(SEM_ROOT, "02_SEM_Indicator_Screening_Model_Lock")
STEP01_MASTER <- file.path(
  STEP01_ROOT,
  "04_Locked_SEM_Master_Standardized.csv.gz"
)

OUTPUT_ROOT <- file.path(SEM_ROOT, "03_SEM_FAI_and_Composite_Scores")
METHOD_FILE <- file.path(OUTPUT_ROOT, "00_Method_Definition.json")
AUDIT_FILE <- file.path(OUTPUT_ROOT, "01_Input_Integrity_Audit.csv")
INDICATOR_FILE <- file.path(
  OUTPUT_ROOT,
  "02_Locked_Indicator_Domain_Definition.csv"
)
FAI_SCALE_FILE <- file.path(OUTPUT_ROOT, "03_FAI_Scaling_Constants.csv")
PCA_LOADINGS_FILE <- file.path(OUTPUT_ROOT, "04_PCA_PC1_Loadings.csv")
PCA_VARIANCE_FILE <- file.path(OUTPUT_ROOT, "05_PCA_Variance_Explained.csv")
PCA_QA_FILE <- file.path(OUTPUT_ROOT, "06_PCA_Score_Season_QA.csv")
FAI_QA_FILE <- file.path(OUTPUT_ROOT, "07_FAI_Distribution_QA.csv")
FAI_COMPONENT_FILE <- file.path(
  OUTPUT_ROOT,
  "08_FAI_Component_Contribution_QA.csv"
)
MASTER_OUT <- file.path(
  OUTPUT_ROOT,
  "09_Locked_SEM_FAI_Composite_Master.csv.gz"
)
SPRING_OUT <- file.path(OUTPUT_ROOT, "10_Spring_FAI_SEM_Data.csv.gz")
SUMMER_OUT <- file.path(OUTPUT_ROOT, "11_Summer_FAI_SEM_Data.csv.gz")
AUTUMN_OUT <- file.path(OUTPUT_ROOT, "12_Autumn_FAI_SEM_Data.csv.gz")
DESIGN_FILE <- file.path(
  OUTPUT_ROOT,
  "13_Prospective_Three_Equation_SEM_Design.csv"
)
MANIFEST_FILE <- file.path(OUTPUT_ROOT, "14_Output_Manifest.csv")
SOFTWARE_FILE <- file.path(OUTPUT_ROOT, "Software_Environment.json")
LOG_FILE <- file.path(OUTPUT_ROOT, "sem_step03_fai_composite_dataset.log")

EXPECTED_ROWS <- 371187L
EXPECTED_ROWS_PER_SEASON <- 123729L
EXPECTED_GRIDS <- 4982L
EXPECTED_YEARS <- 2001:2025

SEASON_LABELS <- c(`1` = "Spring", `2` = "Summer", `3` = "Autumn")
COUNTRY_LEVELS <- c("China", "NK", "Russia")

DOMAINS <- list(
  Meteorology = c(
    "Temp", "Rhum", "Wind", "LtgProxy", "SPEI3",
    "SPEI12", "Pre", "SSRD", "SPEI1"
  ),
  Vegetation = c("EVI", "PTC", "NE"),
  Topography = c("DEM", "Slope", "Aspect"),
  Anthropogenic = c("POP", "Road_dens", "Dis_Farm")
)

ORIENTATION <- list(
  Meteorology = c(
    Temp = 1, Rhum = -1, Wind = 1, LtgProxy = 1,
    SPEI3 = -1, SPEI12 = -1, Pre = -1, SSRD = 1, SPEI1 = -1
  ),
  Vegetation = c(EVI = 1, PTC = 1, NE = 1),
  Topography = c(DEM = 1, Slope = 1),
  Anthropogenic = c(POP = 1, Road_dens = 1, Dis_Farm = -1)
)

SCORE_NAMES <- c(
  Meteorology = "Meteorology_Score",
  Vegetation = "Vegetation_Score",
  Topography = "Topography_Score",
  Anthropogenic = "Anthropogenic_Score"
)

LOCKED_INDICATORS <- unname(unlist(DOMAINS, use.names = FALSE))

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

normalize_country <- function(x) {
  original <- trimws(as.character(x))
  key <- tolower(gsub("[^a-z]", "", original))
  result <- original
  result[key %in% c("china", "chn")] <- "China"
  result[
    key %in% c(
      "nk", "northkorea", "northkorean", "dprk",
      "democraticpeoplesrepublicofkorea"
    )
  ] <- "NK"
  result[key %in% c("russia", "russianfederation", "rus")] <- "Russia"
  result
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
    file.path(SCRIPT_DIR, "03_build_FAI_and_common_PCA_scores.R"),
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

safe_spearman <- function(x, y) {
  as.numeric(
    suppressWarnings(
      stats::cor(x, y, method = "spearman", use = "complete.obs")
    )
  )
}

# -----------------------------------------------------------------------------
# Initialize
# -----------------------------------------------------------------------------

if (dir.exists(OUTPUT_ROOT)) {
  stop(
    paste0(
      "Output directory already exists:\n", OUTPUT_ROOT,
      "\nDelete or rename it before an intentional full rerun."
    ),
    call. = FALSE
  )
}

dir.create(OUTPUT_ROOT, recursive = TRUE, showWarnings = FALSE)
writeLines(character(), LOG_FILE)

log_message("Starting SEM Step 03 FAI and composite-score construction.")
log_message(paste("Code version:", CODE_VERSION))

if (!file.exists(STEP01_MASTER)) {
  stop(paste("Step 01 master not found:", STEP01_MASTER), call. = FALSE)
}
if (!dir.exists(STEP02_ROOT)) {
  stop(paste("Step 02 folder not found:", STEP02_ROOT), call. = FALSE)
}

step02_files <- list.files(
  STEP02_ROOT,
  recursive = TRUE,
  full.names = TRUE,
  include.dirs = FALSE
)

# -----------------------------------------------------------------------------
# Read and validate Step 01 master
# -----------------------------------------------------------------------------

log_message("Reading the sealed Step 01 master.")
master <- fread(STEP01_MASTER, showProgress = TRUE)

base_fields <- c(
  "GRID_UID", "Country", "Year", "Season",
  "Fire_Count", "Burned_Pixel_Count", "ForestPixelCount"
)
missing_base <- setdiff(base_fields, names(master))
missing_indicators <- setdiff(LOCKED_INDICATORS, names(master))

if (length(missing_base) > 0L) {
  stop(
    paste(
      "Missing required Step 01 fields:",
      paste(missing_base, collapse = ", ")
    ),
    call. = FALSE
  )
}
if (length(missing_indicators) > 0L) {
  stop(
    paste(
      "Missing locked Step 02 indicators:",
      paste(missing_indicators, collapse = ", ")
    ),
    call. = FALSE
  )
}

master[, GRID_UID := as.character(GRID_UID)]
master[, Country := normalize_country(Country)]
master[, Year := as.integer(Year)]
master[, Season := as.integer(Season)]
master[, Fire_Count := as.numeric(Fire_Count)]
master[, Burned_Pixel_Count := as.numeric(Burned_Pixel_Count)]
master[, ForestPixelCount := as.numeric(ForestPixelCount)]
master[, Season_Label := unname(SEASON_LABELS[as.character(Season)])]

if (!"Year_Z" %in% names(master)) {
  master[, Year_Z := as.numeric(scale(Year))]
} else {
  master[, Year_Z := as.numeric(Year_Z)]
}

if (!"Log_ForestPixelCount" %in% names(master)) {
  master[, Log_ForestPixelCount := log(ForestPixelCount)]
} else {
  master[, Log_ForestPixelCount := as.numeric(Log_ForestPixelCount)]
}

master[
  ,
  SEM_Row_ID := paste(GRID_UID, Year, Season, sep = "|")
]

audit <- list()

audit <- add_check(
  audit, "Input", "Step01_master_exists",
  file.exists(STEP01_MASTER), STEP01_MASTER, "Existing file"
)
audit <- add_check(
  audit, "Input", "Step02_folder_nonempty",
  length(step02_files) > 0L, length(step02_files), ">0"
)
audit <- add_check(
  audit, "Universe", "Panel_rows",
  nrow(master) == EXPECTED_ROWS, nrow(master), EXPECTED_ROWS
)
audit <- add_check(
  audit, "Universe", "Unique_GRID_UIDs",
  uniqueN(master$GRID_UID) == EXPECTED_GRIDS,
  uniqueN(master$GRID_UID), EXPECTED_GRIDS
)

season_counts <- master[, .N, by = Season][order(Season)]
audit <- add_check(
  audit, "Universe", "Rows_per_season",
  nrow(season_counts) == 3L &&
    all(season_counts$N == EXPECTED_ROWS_PER_SEASON),
  paste(season_counts$N, collapse = ";"),
  paste(rep(EXPECTED_ROWS_PER_SEASON, 3L), collapse = ";")
)
audit <- add_check(
  audit, "Universe", "Years_2001_2025",
  identical(sort(unique(master$Year)), EXPECTED_YEARS),
  paste(range(master$Year), collapse = "-"), "2001-2025"
)
audit <- add_check(
  audit, "Universe", "Unique_GRID_UID_Year_Season_keys",
  !anyDuplicated(master[, .(GRID_UID, Year, Season)]),
  anyDuplicated(master[, .(GRID_UID, Year, Season)]), 0L
)
audit <- add_check(
  audit, "Universe", "Country_levels",
  setequal(unique(master$Country), COUNTRY_LEVELS),
  paste(sort(unique(master$Country)), collapse = ";"),
  paste(sort(COUNTRY_LEVELS), collapse = ";")
)
audit <- add_check(
  audit, "Response", "Counts_nonnegative_finite",
  all(is.finite(master$Fire_Count)) &&
    all(is.finite(master$Burned_Pixel_Count)) &&
    all(master$Fire_Count >= 0) &&
    all(master$Burned_Pixel_Count >= 0),
  paste(min(master$Fire_Count), min(master$Burned_Pixel_Count), sep = ";"),
  "Both minima >=0"
)
audit <- add_check(
  audit, "Response", "Counts_integer_like",
  max(abs(master$Fire_Count - round(master$Fire_Count))) < 1e-8 &&
    max(
      abs(
        master$Burned_Pixel_Count -
          round(master$Burned_Pixel_Count)
      )
    ) < 1e-8,
  paste(
    max(abs(master$Fire_Count - round(master$Fire_Count))),
    max(
      abs(
        master$Burned_Pixel_Count -
          round(master$Burned_Pixel_Count)
      )
    ),
    sep = ";"
  ),
  "<1e-8 for both"
)
audit <- add_check(
  audit, "Exposure", "ForestPixelCount_positive",
  all(is.finite(master$ForestPixelCount)) &&
    all(master$ForestPixelCount > 0),
  min(master$ForestPixelCount), ">0"
)
audit <- add_check(
  audit, "Exposure", "Log_exposure_exact",
  max(
    abs(
      master$Log_ForestPixelCount -
        log(master$ForestPixelCount)
    )
  ) < 1e-10,
  max(
    abs(
      master$Log_ForestPixelCount -
        log(master$ForestPixelCount)
    )
  ),
  "<1e-10"
)

indicator_finite <- vapply(
  master[, ..LOCKED_INDICATORS],
  function(x) all(is.finite(as.numeric(x))),
  logical(1)
)
audit <- add_check(
  audit, "Indicators", "All_18_indicator_fields_present",
  length(missing_indicators) == 0L,
  length(intersect(LOCKED_INDICATORS, names(master))),
  length(LOCKED_INDICATORS)
)
audit <- add_check(
  audit, "Indicators", "All_indicator_values_finite",
  all(indicator_finite),
  paste(names(indicator_finite)[!indicator_finite], collapse = ";"),
  "No non-finite values"
)

fwrite(rbindlist(audit, fill = TRUE), AUDIT_FILE)

if (any(!rbindlist(audit)$Passed)) {
  stop(
    "Initial input checks failed. See 01_Input_Integrity_Audit.csv.",
    call. = FALSE
  )
}

# -----------------------------------------------------------------------------
# Save locked indicator-domain definition
# -----------------------------------------------------------------------------

indicator_rows <- list()
for (domain in names(DOMAINS)) {
  weights <- ORIENTATION[[domain]]
  for (indicator in DOMAINS[[domain]]) {
    indicator_rows[[length(indicator_rows) + 1L]] <- data.table(
      Domain = domain,
      Indicator = indicator,
      Source_Field = indicator,
      Standardized_Field = paste0("Z_", indicator),
      Orientation_Anchor_Weight = if (
        indicator %in% names(weights)
      ) {
        as.numeric(weights[[indicator]])
      } else {
        0
      },
      Included_In_PC1 = TRUE,
      Selected_From_FAI = FALSE,
      Selected_From_Response = FALSE,
      Source = "Sealed Step 02 indicator-domain lock"
    )
  }
}
indicator_lock <- rbindlist(indicator_rows, fill = TRUE)
fwrite(indicator_lock, INDICATOR_FILE)

# -----------------------------------------------------------------------------
# Within-season standardization
# -----------------------------------------------------------------------------

log_message("Creating within-season z scores for the 18 indicators.")

standardization_rows <- list()
for (indicator in LOCKED_INDICATORS) {
  z_field <- paste0("Z_", indicator)

  master[
    ,
    (z_field) := {
      x <- as.numeric(get(indicator))
      s <- stats::sd(x)
      if (!is.finite(s) || s <= 0) {
        stop(
          paste(
            "Invalid within-season SD:",
            indicator,
            "season",
            unique(Season)
          ),
          call. = FALSE
        )
      }
      (x - mean(x)) / s
    },
    by = Season
  ]

  qa <- master[
    ,
    .(
      Raw_Mean = mean(as.numeric(get(indicator))),
      Raw_SD = stats::sd(as.numeric(get(indicator))),
      Z_Mean = mean(as.numeric(get(z_field))),
      Z_SD = stats::sd(as.numeric(get(z_field)))
    ),
    by = .(Season, Season_Label)
  ]
  qa[, Indicator := indicator]
  standardization_rows[[length(standardization_rows) + 1L]] <- qa
}
standardization_qa <- rbindlist(standardization_rows, fill = TRUE)

audit <- add_check(
  audit, "Standardization", "Indicator_season_means_zero",
  max(abs(standardization_qa$Z_Mean)) < 1e-10,
  max(abs(standardization_qa$Z_Mean)), "<1e-10"
)
audit <- add_check(
  audit, "Standardization", "Indicator_season_SDs_one",
  max(abs(standardization_qa$Z_SD - 1)) < 1e-10,
  max(abs(standardization_qa$Z_SD - 1)), "<1e-10"
)

# -----------------------------------------------------------------------------
# Four common pooled, outcome-neutral PC1 scores
# -----------------------------------------------------------------------------

log_message("Constructing four common pooled outcome-neutral PC1 scores.")

loading_rows <- list()
variance_rows <- list()

for (domain in names(DOMAINS)) {
  indicators <- DOMAINS[[domain]]
  z_fields <- paste0("Z_", indicators)
  x <- as.matrix(master[, ..z_fields])
  storage.mode(x) <- "double"

  pca <- stats::prcomp(
    x,
    center = FALSE,
    scale. = FALSE,
    rank. = 1L,
    retx = TRUE
  )

  loadings <- as.numeric(pca$rotation[, 1L])
  names(loadings) <- indicators
  raw_score <- as.numeric(pca$x[, 1L])

  weights <- ORIENTATION[[domain]]
  anchor_fields <- paste0("Z_", names(weights))
  anchor_matrix <- as.matrix(master[, ..anchor_fields])
  anchor <- as.numeric(anchor_matrix %*% as.numeric(weights)) /
    sum(abs(weights))

  correlation_before <- suppressWarnings(
    stats::cor(raw_score, anchor, use = "complete.obs")
  )
  multiplier <- if (
    is.finite(correlation_before) &&
      correlation_before < 0
  ) {
    -1
  } else {
    1
  }

  raw_score <- raw_score * multiplier
  loadings <- loadings * multiplier
  correlation_after <- suppressWarnings(
    stats::cor(raw_score, anchor, use = "complete.obs")
  )

  score_field <- unname(SCORE_NAMES[[domain]])
  raw_field <- paste0(score_field, "_Raw_PC1")
  master[, (raw_field) := raw_score]
  master[
    ,
    (score_field) := {
      values <- as.numeric(get(raw_field))
      (values - mean(values)) / stats::sd(values)
    },
    by = Season
  ]

  loading_rows[[length(loading_rows) + 1L]] <- data.table(
    Domain = domain,
    Score_Field = score_field,
    Indicator = indicators,
    PC1_Loading = as.numeric(loadings),
    Orientation_Anchor_Weight = vapply(
      indicators,
      function(indicator) {
        if (indicator %in% names(weights)) {
          as.numeric(weights[[indicator]])
        } else {
          0
        }
      },
      numeric(1)
    ),
    Orientation_Multiplier = multiplier,
    Anchor_Correlation_Before = correlation_before,
    Anchor_Correlation_After = correlation_after
  )

  pc1_variance <- pca$sdev[1L]^2
  total_variance <- sum(apply(x, 2L, stats::var))
  variance_rows[[length(variance_rows) + 1L]] <- data.table(
    Domain = domain,
    Score_Field = score_field,
    Indicator_N = length(indicators),
    PC1_Standard_Deviation = pca$sdev[1L],
    PC1_Variance = pc1_variance,
    Total_Input_Variance = total_variance,
    PC1_Explained_Fraction = pc1_variance / total_variance,
    PC1_Explained_Percent = 100 * pc1_variance / total_variance,
    PCA_Rows = nrow(x),
    Input_Standardization = "Within-season z scores",
    PCA_Pooling = "All seasons pooled; no response used"
  )
}

pca_loadings <- rbindlist(loading_rows, fill = TRUE)
pca_variance <- rbindlist(variance_rows, fill = TRUE)
fwrite(pca_loadings, PCA_LOADINGS_FILE)
fwrite(pca_variance, PCA_VARIANCE_FILE)

pca_qa_rows <- list()
for (domain in names(SCORE_NAMES)) {
  score_field <- unname(SCORE_NAMES[[domain]])
  qa <- master[
    ,
    .(
      N = .N,
      Mean = mean(as.numeric(get(score_field))),
      SD = stats::sd(as.numeric(get(score_field))),
      Minimum = min(as.numeric(get(score_field))),
      Q025 = as.numeric(
        stats::quantile(
          as.numeric(get(score_field)),
          0.025,
          names = FALSE,
          type = 7
        )
      ),
      Median = stats::median(as.numeric(get(score_field))),
      Q975 = as.numeric(
        stats::quantile(
          as.numeric(get(score_field)),
          0.975,
          names = FALSE,
          type = 7
        )
      ),
      Maximum = max(as.numeric(get(score_field)))
    ),
    by = .(Season, Season_Label)
  ]
  qa[, `:=`(Domain = domain, Score_Field = score_field)]
  pca_qa_rows[[length(pca_qa_rows) + 1L]] <- qa
}
pca_qa <- rbindlist(pca_qa_rows, fill = TRUE)
setcolorder(
  pca_qa,
  c(
    "Domain", "Score_Field", "Season", "Season_Label",
    setdiff(
      names(pca_qa),
      c("Domain", "Score_Field", "Season", "Season_Label")
    )
  )
)
fwrite(pca_qa, PCA_QA_FILE)

audit <- add_check(
  audit, "PCA", "Four_scores_created",
  all(unname(SCORE_NAMES) %in% names(master)),
  sum(unname(SCORE_NAMES) %in% names(master)), 4L
)
audit <- add_check(
  audit, "PCA", "Score_season_means_zero",
  max(abs(pca_qa$Mean)) < 1e-10,
  max(abs(pca_qa$Mean)), "<1e-10"
)
audit <- add_check(
  audit, "PCA", "Score_season_SDs_one",
  max(abs(pca_qa$SD - 1)) < 1e-10,
  max(abs(pca_qa$SD - 1)), "<1e-10"
)
audit <- add_check(
  audit, "PCA", "Orientation_correlations_nonnegative",
  all(pca_loadings$Anchor_Correlation_After >= -1e-12),
  min(pca_loadings$Anchor_Correlation_After), ">=0"
)

# -----------------------------------------------------------------------------
# Common-scale FAI
# -----------------------------------------------------------------------------

log_message("Constructing pooled-Q95 FAI and maximum-scaled sensitivity FAI.")

positive_fc <- master[Fire_Count > 0, Fire_Count]
positive_ba <- master[Burned_Pixel_Count > 0, Burned_Pixel_Count]

if (length(positive_fc) == 0L || length(positive_ba) == 0L) {
  stop("No positive FC or BA-pixel values available.", call. = FALSE)
}

q95_fc <- as.numeric(
  stats::quantile(
    positive_fc,
    0.95,
    names = FALSE,
    type = 7
  )
)
q95_ba <- as.numeric(
  stats::quantile(
    positive_ba,
    0.95,
    names = FALSE,
    type = 7
  )
)
max_fc <- max(master$Fire_Count)
max_ba <- max(master$Burned_Pixel_Count)

if (
  any(!is.finite(c(q95_fc, q95_ba, max_fc, max_ba))) ||
    any(c(q95_fc, q95_ba, max_fc, max_ba) <= 0)
) {
  stop("Invalid FAI scaling constants.", call. = FALSE)
}

master[
  ,
  `:=`(
    FAI_FC_Component_Q95 = 0.5 * Fire_Count / q95_fc,
    FAI_BA_Component_Q95 = 0.5 * Burned_Pixel_Count / q95_ba,
    FAI_Q95 = 0.5 * (
      Fire_Count / q95_fc +
        Burned_Pixel_Count / q95_ba
    ),
    FAI_Max_Sensitivity = 0.5 * (
      Fire_Count / max_fc +
        Burned_Pixel_Count / max_ba
    ),
    FAI_Occurrence = as.integer(
      Fire_Count > 0 |
        Burned_Pixel_Count > 0
    )
  )
]

master[, FAI_Q95_per_ForestPixel := FAI_Q95 / ForestPixelCount]
master[
  ,
  FAI_FC_Share_Q95 := fifelse(
    FAI_Q95 > 0,
    FAI_FC_Component_Q95 / FAI_Q95,
    NA_real_
  )
]
master[
  ,
  FAI_BA_Share_Q95 := fifelse(
    FAI_Q95 > 0,
    FAI_BA_Component_Q95 / FAI_Q95,
    NA_real_
  )
]

fai_scale <- data.table(
  FAI_Version = c(
    "FAI_Q95", "FAI_Q95",
    "FAI_Max_Sensitivity", "FAI_Max_Sensitivity"
  ),
  Component = c(
    "Fire_Count", "Burned_Pixel_Count",
    "Fire_Count", "Burned_Pixel_Count"
  ),
  Scaling_Statistic = c(
    "Positive-value pooled 95th percentile",
    "Positive-value pooled 95th percentile",
    "Pooled maximum",
    "Pooled maximum"
  ),
  Scaling_Value = c(q95_fc, q95_ba, max_fc, max_ba),
  Component_Weight = 0.5,
  Pooled_Across_Seasons = TRUE,
  Pooled_Across_Years = TRUE,
  Pooled_Across_Countries = TRUE,
  Exposure_Inside_Index = FALSE,
  Planned_Exposure_Treatment = (
    "Use offset(Log_ForestPixelCount) in the future FAI model"
  )
)
fwrite(fai_scale, FAI_SCALE_FILE)

audit <- add_check(
  audit, "FAI", "FAI_nonnegative_finite",
  all(is.finite(master$FAI_Q95)) &&
    all(master$FAI_Q95 >= 0),
  paste(min(master$FAI_Q95), max(master$FAI_Q95), sep = ";"),
  "Finite and minimum >=0"
)
audit <- add_check(
  audit, "FAI", "FAI_zero_equivalence",
  all(
    (master$FAI_Q95 == 0) ==
      (
        master$Fire_Count == 0 &
          master$Burned_Pixel_Count == 0
      )
  ),
  sum(
    (master$FAI_Q95 == 0) !=
      (
        master$Fire_Count == 0 &
          master$Burned_Pixel_Count == 0
      )
  ),
  0L
)
audit <- add_check(
  audit, "FAI", "FAI_components_sum",
  max(
    abs(
      master$FAI_Q95 -
        (
          master$FAI_FC_Component_Q95 +
            master$FAI_BA_Component_Q95
        )
    )
  ) < 1e-12,
  max(
    abs(
      master$FAI_Q95 -
        (
          master$FAI_FC_Component_Q95 +
            master$FAI_BA_Component_Q95
        )
    )
  ),
  "<1e-12"
)
audit <- add_check(
  audit, "FAI", "Positive_row_shares_sum_one",
  max(
    abs(
      master[
        FAI_Q95 > 0,
        FAI_FC_Share_Q95 + FAI_BA_Share_Q95
      ] -
        1
    )
  ) < 1e-12,
  max(
    abs(
      master[
        FAI_Q95 > 0,
        FAI_FC_Share_Q95 + FAI_BA_Share_Q95
      ] -
        1
    )
  ),
  "<1e-12"
)

# -----------------------------------------------------------------------------
# FAI summaries
# -----------------------------------------------------------------------------

fai_qa <- master[
  ,
  .(
    N = .N,
    Zero_N = sum(FAI_Q95 == 0),
    Zero_Percent = 100 * mean(FAI_Q95 == 0),
    Positive_N = sum(FAI_Q95 > 0),
    Mean = mean(FAI_Q95),
    SD = stats::sd(FAI_Q95),
    Minimum = min(FAI_Q95),
    Median = stats::median(FAI_Q95),
    Q90 = as.numeric(
      stats::quantile(FAI_Q95, 0.90, names = FALSE, type = 7)
    ),
    Q95 = as.numeric(
      stats::quantile(FAI_Q95, 0.95, names = FALSE, type = 7)
    ),
    Q99 = as.numeric(
      stats::quantile(FAI_Q95, 0.99, names = FALSE, type = 7)
    ),
    Maximum = max(FAI_Q95),
    Positive_Mean = if (any(FAI_Q95 > 0)) {
      mean(FAI_Q95[FAI_Q95 > 0])
    } else {
      NA_real_
    },
    Positive_Median = if (any(FAI_Q95 > 0)) {
      stats::median(FAI_Q95[FAI_Q95 > 0])
    } else {
      NA_real_
    },
    Spearman_FAI_FC = safe_spearman(FAI_Q95, Fire_Count),
    Spearman_FAI_BA = safe_spearman(
      FAI_Q95,
      Burned_Pixel_Count
    ),
    Spearman_Q95_MaxSensitivity = safe_spearman(
      FAI_Q95,
      FAI_Max_Sensitivity
    )
  ),
  by = .(Season, Season_Label, Country)
]

overall_fai_qa <- master[
  ,
  .(
    Season = 0L,
    Season_Label = "All_Seasons",
    Country = "All_Countries",
    N = .N,
    Zero_N = sum(FAI_Q95 == 0),
    Zero_Percent = 100 * mean(FAI_Q95 == 0),
    Positive_N = sum(FAI_Q95 > 0),
    Mean = mean(FAI_Q95),
    SD = stats::sd(FAI_Q95),
    Minimum = min(FAI_Q95),
    Median = stats::median(FAI_Q95),
    Q90 = as.numeric(
      stats::quantile(FAI_Q95, 0.90, names = FALSE, type = 7)
    ),
    Q95 = as.numeric(
      stats::quantile(FAI_Q95, 0.95, names = FALSE, type = 7)
    ),
    Q99 = as.numeric(
      stats::quantile(FAI_Q95, 0.99, names = FALSE, type = 7)
    ),
    Maximum = max(FAI_Q95),
    Positive_Mean = mean(FAI_Q95[FAI_Q95 > 0]),
    Positive_Median = stats::median(FAI_Q95[FAI_Q95 > 0]),
    Spearman_FAI_FC = safe_spearman(FAI_Q95, Fire_Count),
    Spearman_FAI_BA = safe_spearman(
      FAI_Q95,
      Burned_Pixel_Count
    ),
    Spearman_Q95_MaxSensitivity = safe_spearman(
      FAI_Q95,
      FAI_Max_Sensitivity
    )
  )
]

fai_qa <- rbindlist(list(overall_fai_qa, fai_qa), fill = TRUE)
setorder(fai_qa, Season, Country)
fwrite(fai_qa, FAI_QA_FILE)

fai_components <- master[
  ,
  .(
    N = .N,
    Positive_FAI_N = sum(FAI_Q95 > 0),
    Total_FAI_Q95 = sum(FAI_Q95),
    Total_FC_Component_Q95 = sum(FAI_FC_Component_Q95),
    Total_BA_Component_Q95 = sum(FAI_BA_Component_Q95),
    Aggregate_FC_Contribution_Share = (
      sum(FAI_FC_Component_Q95) /
        sum(FAI_Q95)
    ),
    Aggregate_BA_Contribution_Share = (
      sum(FAI_BA_Component_Q95) /
        sum(FAI_Q95)
    ),
    Mean_Positive_Row_FC_Share = mean(
      FAI_FC_Share_Q95,
      na.rm = TRUE
    ),
    Mean_Positive_Row_BA_Share = mean(
      FAI_BA_Share_Q95,
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
  by = .(Season, Season_Label, Country)
]
fwrite(fai_components, FAI_COMPONENT_FILE)

# -----------------------------------------------------------------------------
# Prospective 3-equation seasonal SEM design (not fitted here)
# -----------------------------------------------------------------------------

design_rows <- list()
for (season in 1:3) {
  season_label <- unname(SEASON_LABELS[as.character(season)])

  design_rows[[length(design_rows) + 1L]] <- data.table(
    Season = season,
    Season_Label = season_label,
    Order = 1L,
    Submodel = "Meteorology_mediator",
    Response = "Meteorology_Score",
    Predictors = "Topography_Score + Country + Year_Z",
    Offset = "",
    Planned_Family = "Gaussian identity",
    Random_Intercepts = "GRID_UID and Year",
    Status = "Prospective only; not fitted in Step 03"
  )
  design_rows[[length(design_rows) + 1L]] <- data.table(
    Season = season,
    Season_Label = season_label,
    Order = 2L,
    Submodel = "Vegetation_mediator",
    Response = "Vegetation_Score",
    Predictors = paste(
      "Meteorology_Score + Topography_Score +",
      "Anthropogenic_Score + Country + Year_Z"
    ),
    Offset = "",
    Planned_Family = "Gaussian identity",
    Random_Intercepts = "GRID_UID and Year",
    Status = "Prospective only; not fitted in Step 03"
  )
  design_rows[[length(design_rows) + 1L]] <- data.table(
    Season = season,
    Season_Label = season_label,
    Order = 3L,
    Submodel = "FAI_outcome",
    Response = "FAI_Q95",
    Predictors = paste(
      "Meteorology_Score + Vegetation_Score +",
      "Topography_Score + Anthropogenic_Score +",
      "Country + Year_Z"
    ),
    Offset = "offset(Log_ForestPixelCount)",
    Planned_Family = "Tweedie log",
    Random_Intercepts = "GRID_UID and Year",
    Status = paste(
      "Prospective only; family and diagnostics",
      "to be confirmed in Step 04"
    )
  )
}
prospective_design <- rbindlist(design_rows, fill = TRUE)
fwrite(prospective_design, DESIGN_FILE)

# -----------------------------------------------------------------------------
# Locked datasets
# -----------------------------------------------------------------------------

log_message("Writing locked pooled and season-specific datasets.")

optional_fields <- intersect(
  c(
    "GRID_ID", "Spatial_Block",
    "Centroid_X_UTM52_m", "Centroid_Y_UTM52_m",
    "Longitude", "Latitude"
  ),
  names(master)
)
z_fields <- paste0("Z_", LOCKED_INDICATORS)
raw_score_fields <- paste0(unname(SCORE_NAMES), "_Raw_PC1")

output_fields <- unique(
  c(
    "SEM_Row_ID", "GRID_UID", optional_fields,
    "Country", "Year", "Year_Z", "Season", "Season_Label",
    "ForestPixelCount", "Log_ForestPixelCount",
    "Fire_Count", "Burned_Pixel_Count",
    "FAI_FC_Component_Q95", "FAI_BA_Component_Q95",
    "FAI_Q95", "FAI_Max_Sensitivity",
    "FAI_Q95_per_ForestPixel", "FAI_Occurrence",
    "FAI_FC_Share_Q95", "FAI_BA_Share_Q95",
    LOCKED_INDICATORS, z_fields, raw_score_fields,
    unname(SCORE_NAMES)
  )
)

locked <- master[, ..output_fields]
setorder(locked, Season, GRID_UID, Year)

fwrite(locked, MASTER_OUT, compress = "gzip")
fwrite(locked[Season == 1L], SPRING_OUT, compress = "gzip")
fwrite(locked[Season == 2L], SUMMER_OUT, compress = "gzip")
fwrite(locked[Season == 3L], AUTUMN_OUT, compress = "gzip")

audit <- add_check(
  audit, "Output", "Locked_master_rows",
  nrow(locked) == EXPECTED_ROWS, nrow(locked), EXPECTED_ROWS
)
audit <- add_check(
  audit, "Output", "Season_output_rows",
  all(
    c(
      nrow(locked[Season == 1L]),
      nrow(locked[Season == 2L]),
      nrow(locked[Season == 3L])
    ) == EXPECTED_ROWS_PER_SEASON
  ),
  paste(
    c(
      nrow(locked[Season == 1L]),
      nrow(locked[Season == 2L]),
      nrow(locked[Season == 3L])
    ),
    collapse = ";"
  ),
  paste(rep(EXPECTED_ROWS_PER_SEASON, 3L), collapse = ";")
)

required_model_fields <- c(
  "GRID_UID", "Country", "Year", "Year_Z", "Season",
  "ForestPixelCount", "Log_ForestPixelCount", "FAI_Q95",
  unname(SCORE_NAMES)
)
missing_model_rows <- sum(
  !complete.cases(locked[, ..required_model_fields])
)
audit <- add_check(
  audit, "Output", "No_missing_future_model_fields",
  missing_model_rows == 0L, missing_model_rows, 0L
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
    "Step 03 final audit failed. See 01_Input_Integrity_Audit.csv.",
    call. = FALSE
  )
}

# -----------------------------------------------------------------------------
# Method and software metadata
# -----------------------------------------------------------------------------

step02_inventory <- lapply(
  sort(step02_files),
  function(path) {
    list(
      Path = normalizePath(path, winslash = "/", mustWork = TRUE),
      File_Size_Bytes = file.info(path)$size,
      SHA256 = sha256_file(path)
    )
  }
)

method_definition <- list(
  Code_Version = CODE_VERSION,
  Created = as.character(Sys.time()),
  Script_Path = script_path(),
  Script_SHA256 = if (file.exists(script_path())) {
    sha256_file(script_path())
  } else {
    NA_character_
  },
  Step01_Master = normalizePath(
    STEP01_MASTER,
    winslash = "/",
    mustWork = TRUE
  ),
  Step01_Master_SHA256 = sha256_file(STEP01_MASTER),
  Step02_Root = normalizePath(
    STEP02_ROOT,
    winslash = "/",
    mustWork = TRUE
  ),
  Step02_File_Inventory = step02_inventory,
  Locked_Domains = DOMAINS,
  PCA = list(
    Input = "Within-season z-standardized locked indicators",
    Pooling = "All three seasons pooled",
    Component = "PC1",
    Outcome_Neutral = TRUE,
    Response_Used = FALSE,
    Orientation_Weights = ORIENTATION,
    Final_Score_Standardization = (
      "Mean 0 and sample SD 1 within each season"
    )
  ),
  FAI = list(
    Primary_Field = "FAI_Q95",
    Formula = paste0(
      "0.5 * (Fire_Count / pooled positive FC Q95 + ",
      "Burned_Pixel_Count / pooled positive BA-pixel Q95)"
    ),
    Quantile_Type = 7L,
    Common_Scale_Across_Seasons = TRUE,
    Sensitivity_Field = "FAI_Max_Sensitivity",
    Sensitivity_Formula = paste0(
      "0.5 * (Fire_Count / pooled FC maximum + ",
      "Burned_Pixel_Count / pooled BA-pixel maximum)"
    ),
    Exposure_Field = "ForestPixelCount",
    Offset_Field = "Log_ForestPixelCount",
    Exposure_Included_Inside_FAI = FALSE,
    FAI_Standardized = FALSE
  ),
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

# -----------------------------------------------------------------------------
# Manifest
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
# Completion
# -----------------------------------------------------------------------------

log_message(
  paste(
    "Step 03 completed: common-scale FAI and four common PCA scores",
    "were created without fitting any SEM model."
  )
)

cat("\n")
cat(
  "Output directory:",
  normalizePath(OUTPUT_ROOT, winslash = "/", mustWork = FALSE),
  "\n"
)
cat("Audit ERROR failures:", nrow(error_failures), "\n")
cat("Audit WARNING failures:", nrow(warning_failures), "\n")
cat("Locked master rows:", nrow(locked), "/", EXPECTED_ROWS, "\n")
cat(
  "Unique GRID_UIDs:",
  uniqueN(locked$GRID_UID),
  "/",
  EXPECTED_GRIDS,
  "\n"
)
cat("Pooled positive FC Q95:", format(q95_fc, digits = 12), "\n")
cat(
  "Pooled positive BA-pixel Q95:",
  format(q95_ba, digits = 12),
  "\n"
)
cat(
  "Overall FAI zero percentage:",
  format(100 * mean(locked$FAI_Q95 == 0), digits = 8),
  "\n"
)
cat("Four PCA scores created: True\n")
cat("SEM models fitted: 0\n")
cat("Final piecewiseSEM assembled: False\n")
cat("Direct/indirect effects calculated: False\n")
