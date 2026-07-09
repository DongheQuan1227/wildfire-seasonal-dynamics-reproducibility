# =============================================================================
# Step 04: Fit three seasonal FAI piecewise-SEM component model sets
#
# Recommended location
# --------------------
# <REPOSITORY_ROOT>/11_SEM_mechanism_analysis/00_Script/
# 04_fit_three_seasonal_FAI_SEMs.R
#
# Scientific structure
# --------------------
# For each season:
#
#   Topography  ---> Meteorology
#   Topography  ---> Vegetation
#   Meteorology ---> Vegetation
#   Anthropogenic ---> Vegetation
#
#   Meteorology ----\
#   Vegetation ------\
#   Topography --------> FAI
#   Anthropogenic ----/
#
# Topography and Anthropogenic activity are treated as correlated exogenous
# drivers. No directional causal path is imposed between them.
#
# Primary FAI:
#   FAI_Q95
#
# Sensitivity FAI:
#   FAI_EqualContribution_Sensitivity
#
# Exposure:
#   offset(Log_ForestPixelCount)
#
# This step fits:
#   - 3 seasons x 3 primary component models = 9 primary models
#   - 3 equal-contribution FAI sensitivity outcome models
#   - 12 fitted models in total
#
# It does NOT calculate final direct, indirect, or total effects and does NOT
# create final publication path diagrams.
#
# Compatibility note
# ------------------
# This script deliberately does not require piecewiseSEM. The user's current
# R version is 4.1.3, whereas the current CRAN piecewiseSEM release requires a
# newer R version. The fitted glmmTMB component models and locked path graph
# are saved for manual effect decomposition and figures in Step 05.
# =============================================================================

options(stringsAsFactors = FALSE, warn = 1, scipen = 999)

required_packages <- c(
  "data.table",
  "glmmTMB",
  "jsonlite",
  "digest",
  "R.utils"
)

missing_packages <- required_packages[
  !vapply(
    required_packages,
    requireNamespace,
    logical(1),
    quietly = TRUE
  )
]

if (length(missing_packages) > 0L) {
  stop(
    paste0(
      "Missing packages: ",
      paste(missing_packages, collapse = ", "),
      "\nInstall them with:\n",
      "install.packages(c(",
      paste(sprintf('"%s"', missing_packages), collapse = ", "),
      "), repos='https://cloud.r-project.org')"
    ),
    call. = FALSE
  )
}

suppressPackageStartupMessages({
  library(data.table)
  library(glmmTMB)
})

# -----------------------------------------------------------------------------
# Version, paths, and expectations
# -----------------------------------------------------------------------------

CODE_VERSION <- paste0(
  "2026-07-01_SEM_THREE_SEASON_FAI_FITS_V1_",
  "NINE_PRIMARY_PLUS_THREE_SENSITIVITY_MODELS"
)

EXPECTED_STEP03A_VERSION <- paste0(
  "2026-07-01_SEM_FAI_SENSITIVITY_V1_",
  "POOLED_EQUAL_AGGREGATE_CONTRIBUTION"
)

EXPECTED_STEP03A_SCRIPT_SHA256 <- paste0(
  "dfd903f2b9fbfa9b89b4c9c37573ee09",
  "bd81666be6668cb349d6c3b77b1e827e"
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
STEP03A_ROOT <- file.path(SEM_ROOT, "03A_SEM_FAI_Sensitivity")

STEP03A_METHOD_FILE <- file.path(STEP03A_ROOT, "00_Method_Definition.json")
STEP03A_AUDIT_FILE <- file.path(
  STEP03A_ROOT,
  "01_Input_Integrity_Audit.csv"
)
STEP03A_MASTER_FILE <- file.path(
  STEP03A_ROOT,
  "06_Locked_SEM_FAI_Sensitivity_Master.csv.gz"
)
STEP03A_MANIFEST_FILE <- file.path(
  STEP03A_ROOT,
  "10_Output_Manifest.csv"
)
STEP03A_SCRIPT_FILE <- file.path(
  SCRIPT_DIR,
  "03A_build_equal_contribution_FAI_sensitivity.R"
)

OUTPUT_ROOT <- file.path(
  SEM_ROOT,
  "04_Seasonal_FAI_Piecewise_SEM"
)

MODEL_DIR <- file.path(OUTPUT_ROOT, "Models")
BUNDLE_DIR <- file.path(OUTPUT_ROOT, "Seasonal_SEM_Bundles")

METHOD_FILE <- file.path(OUTPUT_ROOT, "00_Method_Definition.json")
INPUT_AUDIT_FILE <- file.path(
  OUTPUT_ROOT,
  "01_Input_Integrity_Audit.csv"
)
PATH_LOCK_FILE <- file.path(
  OUTPUT_ROOT,
  "02_Locked_Path_Graph_and_Roles.csv"
)
FORMULA_LOCK_FILE <- file.path(
  OUTPUT_ROOT,
  "03_Locked_Model_Formulas.csv"
)
RESPONSE_SUPPORT_FILE <- file.path(
  OUTPUT_ROOT,
  "04_FAI_Response_Support.csv"
)
EXOGENOUS_CORRELATION_FILE <- file.path(
  OUTPUT_ROOT,
  "05_Topography_Anthropogenic_Correlation.csv"
)
MODEL_REGISTRY_FILE <- file.path(
  OUTPUT_ROOT,
  "06_Model_Registry.csv"
)
FIT_ATTEMPTS_FILE <- file.path(
  OUTPUT_ROOT,
  "07_Fit_Attempts.csv"
)
FIXED_EFFECT_FILE <- file.path(
  OUTPUT_ROOT,
  "08_Fixed_Effect_Estimates.csv"
)
RANDOM_EFFECT_FILE <- file.path(
  OUTPUT_ROOT,
  "09_Random_Effect_Estimates.csv"
)
FIT_STATISTICS_FILE <- file.path(
  OUTPUT_ROOT,
  "10_Model_Fit_Statistics.csv"
)
CONVERGENCE_FILE <- file.path(
  OUTPUT_ROOT,
  "11_Convergence_and_Model_QA.csv"
)
PRIMARY_PATH_FILE <- file.path(
  OUTPUT_ROOT,
  "12_Primary_SEM_Path_Coefficients.csv"
)
SENSITIVITY_COMPARISON_FILE <- file.path(
  OUTPUT_ROOT,
  "13_Primary_vs_Equal_FAI_Path_Comparison.csv"
)
SEASON_SUMMARY_FILE <- file.path(
  OUTPUT_ROOT,
  "14_Seasonal_SEM_Fit_Summary.csv"
)
MANIFEST_FILE <- file.path(
  OUTPUT_ROOT,
  "15_Output_Manifest.csv"
)
SOFTWARE_FILE <- file.path(
  OUTPUT_ROOT,
  "Software_Environment.json"
)
LOG_FILE <- file.path(
  OUTPUT_ROOT,
  "sem_step04_three_seasonal_fai_fits.log"
)

EXPECTED_ROWS <- 371187L
EXPECTED_ROWS_PER_SEASON <- 123729L
EXPECTED_GRIDS <- 4982L
EXPECTED_PRIMARY_MODEL_N <- 9L
EXPECTED_SENSITIVITY_MODEL_N <- 3L
EXPECTED_TOTAL_MODEL_N <- 12L

SEASON_LABELS <- c(
  `1` = "Spring",
  `2` = "Summer",
  `3` = "Autumn"
)

COUNTRY_LEVELS <- c(
  "China",
  "NK",
  "Russia"
)

DOMAIN_SCORE_FIELDS <- c(
  "Meteorology_Score",
  "Vegetation_Score",
  "Topography_Score",
  "Anthropogenic_Score"
)

PRIMARY_RESPONSE <- "FAI_Q95"
SENSITIVITY_RESPONSE <- "FAI_EqualContribution_Sensitivity"

FORCE_RESTART <- identical(
  Sys.getenv("SEM_STEP04_FORCE_RESTART", unset = "0"),
  "1"
)

TMB_THREADS <- suppressWarnings(
  as.integer(
    Sys.getenv("SEM_STEP04_TMB_THREADS", unset = "1")
  )
)

if (!is.finite(TMB_THREADS) || TMB_THREADS < 1L) {
  stop("SEM_STEP04_TMB_THREADS must be an integer >= 1.", call. = FALSE)
}

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

formula_text <- function(formula_object) {
  paste(deparse(formula_object), collapse = " ")
}

safe_number <- function(expression, default = NA_real_) {
  tryCatch(
    as.numeric(expression),
    error = function(e) default,
    warning = function(w) default
  )
}

finite_max_abs <- function(values) {
  values <- as.numeric(values)
  values <- values[is.finite(values)]
  if (length(values) == 0L) return(NA_real_)
  max(abs(values))
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
    file.path(SCRIPT_DIR, "04_fit_three_seasonal_FAI_SEMs.R"),
    winslash = "/",
    mustWork = FALSE
  )
}

extract_random_effect_table <- function(fit) {
  variance_components <- tryCatch(
    VarCorr(fit)$cond,
    error = function(e) NULL
  )

  if (
    is.null(variance_components) ||
      length(variance_components) == 0L
  ) {
    return(data.table())
  }

  rows <- list()

  for (group_name in names(variance_components)) {
    component <- variance_components[[group_name]]
    covariance_matrix <- as.matrix(component)
    standard_deviations <- attr(component, "stddev")

    if (is.null(standard_deviations)) {
      standard_deviations <- sqrt(diag(covariance_matrix))
    }

    term_names <- names(standard_deviations)

    if (is.null(term_names)) {
      term_names <- colnames(covariance_matrix)
    }

    if (is.null(term_names)) {
      term_names <- paste0("Term_", seq_along(standard_deviations))
    }

    for (term_index in seq_along(standard_deviations)) {
      rows[[length(rows) + 1L]] <- data.table(
        Model_Component = "conditional",
        Group = group_name,
        Term = term_names[[term_index]],
        Variance = covariance_matrix[term_index, term_index],
        Std_Dev = standard_deviations[[term_index]]
      )
    }
  }

  rbindlist(rows, fill = TRUE)
}

extract_tweedie_power <- function(fit) {
  parameters <- tryCatch(
    fit$fit$parfull,
    error = function(e) NULL
  )

  if (is.null(parameters)) return(NA_real_)

  parameter_names <- names(parameters)

  if (is.null(parameter_names)) return(NA_real_)

  psi_index <- which(parameter_names == "psi")

  if (length(psi_index) == 0L) return(NA_real_)

  1 + stats::plogis(as.numeric(parameters[[psi_index[[1L]]]]))
}

default_control <- function() {
  glmmTMBControl(
    optCtrl = list(
      iter.max = 10000L,
      eval.max = 10000L
    ),
    parallel = TMB_THREADS
  )
}

bfgs_control <- function() {
  glmmTMBControl(
    optimizer = optim,
    optArgs = list(method = "BFGS"),
    optCtrl = list(maxit = 10000L),
    parallel = TMB_THREADS
  )
}

fit_attempt <- function(
  formula_object,
  data_object,
  family_object,
  optimizer_name
) {
  warning_messages <- character()
  start_time <- Sys.time()

  fit <- tryCatch(
    withCallingHandlers(
      glmmTMB(
        formula = formula_object,
        data = data_object,
        family = family_object,
        ziformula = ~0,
        dispformula = ~1,
        control = if (optimizer_name == "nlminb") {
          default_control()
        } else {
          bfgs_control()
        }
      ),
      warning = function(w) {
        warning_messages <<- c(
          warning_messages,
          conditionMessage(w)
        )
        invokeRestart("muffleWarning")
      }
    ),
    error = function(e) e
  )

  elapsed <- as.numeric(
    difftime(Sys.time(), start_time, units = "secs")
  )

  if (inherits(fit, "error")) {
    return(
      list(
        fit = NULL,
        optimizer = optimizer_name,
        valid = FALSE,
        convergence_code = NA_integer_,
        pdHess = FALSE,
        objective = NA_real_,
        logLik = NA_real_,
        sdreport_gradient = NA_real_,
        optimizer_gradient = NA_real_,
        coefficients_finite = FALSE,
        standard_errors_finite = FALSE,
        random_effects_valid = FALSE,
        warning_text = paste(unique(warning_messages), collapse = " | "),
        error_text = conditionMessage(fit),
        elapsed_seconds = elapsed
      )
    )
  }

  coefficient_matrix <- tryCatch(
    summary(fit)$coefficients$cond,
    error = function(e) NULL
  )

  coefficients_finite <- (
    !is.null(coefficient_matrix) &&
      nrow(coefficient_matrix) > 0L &&
      all(is.finite(coefficient_matrix[, "Estimate"]))
  )

  standard_errors_finite <- (
    !is.null(coefficient_matrix) &&
      nrow(coefficient_matrix) > 0L &&
      all(is.finite(coefficient_matrix[, "Std. Error"]))
  )

  random_table <- extract_random_effect_table(fit)
  required_groups <- c("GRID_UID", "Year_Factor")

  random_effects_valid <- (
    nrow(random_table) >= 2L &&
      all(required_groups %in% random_table$Group) &&
      all(is.finite(random_table$Std_Dev)) &&
      all(random_table$Std_Dev > 0)
  )

  convergence_code <- fit$fit$convergence
  pd_hessian <- isTRUE(fit$sdr$pdHess)
  objective <- safe_number(fit$fit$objective)
  loglik <- safe_number(logLik(fit))
  sdreport_gradient <- finite_max_abs(
    tryCatch(fit$sdr$gradient.fixed, error = function(e) numeric())
  )
  optimizer_gradient <- finite_max_abs(
    tryCatch(
      fit$obj$gr(fit$fit$par),
      error = function(e) numeric()
    )
  )

  valid <- (
    identical(as.integer(convergence_code), 0L) &&
      pd_hessian &&
      is.finite(objective) &&
      is.finite(loglik) &&
      coefficients_finite &&
      standard_errors_finite &&
      random_effects_valid
  )

  list(
    fit = fit,
    optimizer = optimizer_name,
    valid = valid,
    convergence_code = convergence_code,
    pdHess = pd_hessian,
    objective = objective,
    logLik = loglik,
    sdreport_gradient = sdreport_gradient,
    optimizer_gradient = optimizer_gradient,
    coefficients_finite = coefficients_finite,
    standard_errors_finite = standard_errors_finite,
    random_effects_valid = random_effects_valid,
    warning_text = paste(unique(warning_messages), collapse = " | "),
    error_text = "",
    elapsed_seconds = elapsed
  )
}

fit_locked_model <- function(
  model_id,
  model_role,
  season,
  season_label,
  response_label,
  formula_object,
  family_object,
  family_label,
  data_object,
  primary_model
) {
  model_path <- file.path(MODEL_DIR, paste0(model_id, ".rds"))

  signature <- digest::digest(
    paste(
      CODE_VERSION,
      model_id,
      formula_text(formula_object),
      family_label,
      nrow(data_object),
      sha256_file(STEP03A_MASTER_FILE),
      sep = "|"
    ),
    algo = "sha256",
    serialize = FALSE
  )

  if (file.exists(model_path) && !FORCE_RESTART) {
    existing <- tryCatch(readRDS(model_path), error = function(e) NULL)

    if (
      !is.null(existing) &&
        identical(existing$code_version, CODE_VERSION) &&
        identical(existing$input_signature, signature)
    ) {
      log_message(paste("Resuming completed model:", model_id))
      return(existing)
    }
  }

  attempt_rows <- list()

  log_message(
    paste(
      "Fitting",
      model_id,
      "| family=",
      family_label,
      "| optimizer=nlminb"
    )
  )

  first <- fit_attempt(
    formula_object = formula_object,
    data_object = data_object,
    family_object = family_object,
    optimizer_name = "nlminb"
  )

  attempt_rows[[length(attempt_rows) + 1L]] <- data.table(
    Model_ID = model_id,
    Attempt = 1L,
    Optimizer = first$optimizer,
    Family = family_label,
    Formula = formula_text(formula_object),
    N = nrow(data_object),
    Convergence_Code = first$convergence_code,
    Positive_Definite_Hessian = first$pdHess,
    Objective = first$objective,
    LogLik = first$logLik,
    SDReport_Max_Absolute_Gradient = first$sdreport_gradient,
    Optimizer_Max_Absolute_Gradient = first$optimizer_gradient,
    Coefficients_Finite = first$coefficients_finite,
    Standard_Errors_Finite = first$standard_errors_finite,
    Random_Effects_Valid = first$random_effects_valid,
    Computationally_Valid = first$valid,
    Selected = first$valid,
    Warning_Text = first$warning_text,
    Error_Text = first$error_text,
    Elapsed_Seconds = first$elapsed_seconds
  )

  selected <- first
  fallback_used <- FALSE

  if (!first$valid) {
    fallback_used <- TRUE

    log_message(
      paste(
        "nlminb did not pass validation for",
        model_id,
        "; trying BFGS."
      )
    )

    second <- fit_attempt(
      formula_object = formula_object,
      data_object = data_object,
      family_object = family_object,
      optimizer_name = "BFGS"
    )

    attempt_rows[[length(attempt_rows) + 1L]] <- data.table(
      Model_ID = model_id,
      Attempt = 2L,
      Optimizer = second$optimizer,
      Family = family_label,
      Formula = formula_text(formula_object),
      N = nrow(data_object),
      Convergence_Code = second$convergence_code,
      Positive_Definite_Hessian = second$pdHess,
      Objective = second$objective,
      LogLik = second$logLik,
      SDReport_Max_Absolute_Gradient = second$sdreport_gradient,
      Optimizer_Max_Absolute_Gradient = second$optimizer_gradient,
      Coefficients_Finite = second$coefficients_finite,
      Standard_Errors_Finite = second$standard_errors_finite,
      Random_Effects_Valid = second$random_effects_valid,
      Computationally_Valid = second$valid,
      Selected = second$valid,
      Warning_Text = second$warning_text,
      Error_Text = second$error_text,
      Elapsed_Seconds = second$elapsed_seconds
    )

    selected <- second
  }

  status <- if (selected$valid) "SUCCESS" else "FAILED"

  record <- list(
    code_version = CODE_VERSION,
    input_signature = signature,
    model_id = model_id,
    model_role = model_role,
    season = season,
    season_label = season_label,
    response_label = response_label,
    primary_model = primary_model,
    family_label = family_label,
    formula = formula_object,
    status = status,
    selected_optimizer = if (selected$valid) {
      selected$optimizer
    } else {
      ""
    },
    fallback_used = fallback_used,
    fit = if (selected$valid) selected$fit else NULL,
    attempts = rbindlist(attempt_rows, fill = TRUE),
    saved_at = as.character(Sys.time())
  )

  saveRDS(record, model_path, compress = "gzip")

  if (selected$valid) {
    log_message(
      paste(
        "Model completed:",
        model_id,
        "| optimizer=",
        selected$optimizer
      )
    )
  } else {
    log_message(paste("MODEL FAILED:", model_id))
  }

  record
}

# -----------------------------------------------------------------------------
# Output initialization
# -----------------------------------------------------------------------------

if (FORCE_RESTART && dir.exists(OUTPUT_ROOT)) {
  unlink(OUTPUT_ROOT, recursive = TRUE, force = TRUE)
}

dir.create(OUTPUT_ROOT, recursive = TRUE, showWarnings = FALSE)
dir.create(MODEL_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(BUNDLE_DIR, recursive = TRUE, showWarnings = FALSE)

if (!file.exists(LOG_FILE) || FORCE_RESTART) {
  writeLines(character(), LOG_FILE)
}

log_message("Starting Step 04 three-season FAI component-model fitting.")
log_message(paste("Code version:", CODE_VERSION))
log_message(
  paste(
    "Planned models:",
    EXPECTED_PRIMARY_MODEL_N,
    "primary +",
    EXPECTED_SENSITIVITY_MODEL_N,
    "sensitivity"
  )
)

# -----------------------------------------------------------------------------
# Input integrity
# -----------------------------------------------------------------------------

required_files <- c(
  STEP03A_METHOD_FILE,
  STEP03A_AUDIT_FILE,
  STEP03A_MASTER_FILE,
  STEP03A_MANIFEST_FILE,
  STEP03A_SCRIPT_FILE
)

missing_files <- required_files[!file.exists(required_files)]

if (length(missing_files) > 0L) {
  stop(
    paste(
      "Missing required Step 03A files:",
      paste(missing_files, collapse = "\n"),
      sep = "\n"
    ),
    call. = FALSE
  )
}

audit <- list()
step03a_method <- read_json(STEP03A_METHOD_FILE)

audit <- add_check(
  audit,
  "Version",
  "Step03A_code_version",
  identical(step03a_method$Code_Version, EXPECTED_STEP03A_VERSION),
  step03a_method$Code_Version,
  EXPECTED_STEP03A_VERSION
)

actual_step03a_script_hash <- sha256_file(STEP03A_SCRIPT_FILE)

audit <- add_check(
  audit,
  "Hash",
  "Step03A_script_hash",
  identical(
    actual_step03a_script_hash,
    EXPECTED_STEP03A_SCRIPT_SHA256
  ),
  actual_step03a_script_hash,
  EXPECTED_STEP03A_SCRIPT_SHA256
)

formal_step03a_audit <- fread(STEP03A_AUDIT_FILE)

audit <- add_check(
  audit,
  "Step03A_Audit",
  "All_Step03A_ERROR_checks_pass",
  all(
    vapply(
      formal_step03a_audit[Severity == "ERROR", Passed],
      parse_bool,
      logical(1)
    )
  ),
  sum(
    vapply(
      formal_step03a_audit[Severity == "ERROR", Passed],
      parse_bool,
      logical(1)
    )
  ),
  nrow(formal_step03a_audit[Severity == "ERROR"])
)

step03a_manifest <- fread(STEP03A_MANIFEST_FILE)
master_manifest_row <- step03a_manifest[
  gsub("\\\\", "/", Relative_Path) ==
    "06_Locked_SEM_FAI_Sensitivity_Master.csv.gz"
]

audit <- add_check(
  audit,
  "Manifest",
  "Step03A_master_manifest_entry_unique",
  nrow(master_manifest_row) == 1L,
  nrow(master_manifest_row),
  1L
)

if (nrow(master_manifest_row) != 1L) {
  fwrite(rbindlist(audit, fill = TRUE), INPUT_AUDIT_FILE)
  stop("Step 03A master manifest entry invalid.", call. = FALSE)
}

actual_master_hash <- sha256_file(STEP03A_MASTER_FILE)
actual_master_size <- file.info(STEP03A_MASTER_FILE)$size

audit <- add_check(
  audit,
  "Manifest",
  "Step03A_master_hash_matches",
  identical(actual_master_hash, master_manifest_row$SHA256[[1L]]),
  actual_master_hash,
  master_manifest_row$SHA256[[1L]]
)

audit <- add_check(
  audit,
  "Manifest",
  "Step03A_master_size_matches",
  identical(
    as.numeric(actual_master_size),
    as.numeric(master_manifest_row$File_Size_Bytes[[1L]])
  ),
  actual_master_size,
  master_manifest_row$File_Size_Bytes[[1L]]
)

log_message("Reading locked Step 03A sensitivity master.")

data <- fread(STEP03A_MASTER_FILE, showProgress = TRUE)

required_fields <- c(
  "SEM_Row_ID",
  "GRID_UID",
  "Country",
  "Year",
  "Year_Z",
  "Season",
  "Season_Label",
  "ForestPixelCount",
  "Log_ForestPixelCount",
  PRIMARY_RESPONSE,
  SENSITIVITY_RESPONSE,
  DOMAIN_SCORE_FIELDS
)

missing_fields <- setdiff(required_fields, names(data))

audit <- add_check(
  audit,
  "Input",
  "All_required_fields_present",
  length(missing_fields) == 0L,
  paste(missing_fields, collapse = ";"),
  "No missing required field"
)

if (length(missing_fields) > 0L) {
  fwrite(rbindlist(audit, fill = TRUE), INPUT_AUDIT_FILE)
  stop(
    paste(
      "Required fields missing:",
      paste(missing_fields, collapse = ", ")
    ),
    call. = FALSE
  )
}

data[, GRID_UID := factor(as.character(GRID_UID))]
data[, Country := factor(as.character(Country), levels = COUNTRY_LEVELS)]
data[, Year := as.integer(Year)]
data[, Year_Z := as.numeric(Year_Z)]
data[, Year_Factor := factor(Year)]
data[, Season := as.integer(Season)]
data[, ForestPixelCount := as.numeric(ForestPixelCount)]
data[, Log_ForestPixelCount := as.numeric(Log_ForestPixelCount)]
data[, (PRIMARY_RESPONSE) := as.numeric(get(PRIMARY_RESPONSE))]
data[, (SENSITIVITY_RESPONSE) := as.numeric(get(SENSITIVITY_RESPONSE))]

for (field in DOMAIN_SCORE_FIELDS) {
  data[, (field) := as.numeric(get(field))]
}

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

future_model_fields <- c(
  "GRID_UID",
  "Country",
  "Year_Factor",
  "Year_Z",
  "ForestPixelCount",
  "Log_ForestPixelCount",
  PRIMARY_RESPONSE,
  SENSITIVITY_RESPONSE,
  DOMAIN_SCORE_FIELDS
)

missing_model_rows <- sum(
  !complete.cases(data[, ..future_model_fields])
)

audit <- add_check(
  audit,
  "Input",
  "No_missing_model_values",
  missing_model_rows == 0L,
  missing_model_rows,
  0L
)

audit <- add_check(
  audit,
  "Response",
  "FAI_responses_nonnegative_and_finite",
  all(is.finite(data[[PRIMARY_RESPONSE]])) &&
    all(is.finite(data[[SENSITIVITY_RESPONSE]])) &&
    all(data[[PRIMARY_RESPONSE]] >= 0) &&
    all(data[[SENSITIVITY_RESPONSE]] >= 0),
  paste(
    min(data[[PRIMARY_RESPONSE]]),
    min(data[[SENSITIVITY_RESPONSE]]),
    sep = ";"
  ),
  "Both finite with minimum >=0"
)

audit <- add_check(
  audit,
  "Exposure",
  "Exposure_valid",
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

initial_audit <- rbindlist(audit, fill = TRUE)
fwrite(initial_audit, INPUT_AUDIT_FILE)

if (
  any(
    initial_audit$Severity == "ERROR" &
      !initial_audit$Passed
  )
) {
  stop(
    "Step 04 input integrity failed. See 01_Input_Integrity_Audit.csv.",
    call. = FALSE
  )
}

# -----------------------------------------------------------------------------
# Lock path graph and formulas
# -----------------------------------------------------------------------------

path_lock <- data.table(
  From = c(
    "Topography",
    "Topography",
    "Meteorology",
    "Anthropogenic",
    "Meteorology",
    "Vegetation",
    "Topography",
    "Anthropogenic",
    "Topography",
    "Country",
    "Year",
    "GRID_UID",
    "Year_random"
  ),
  To = c(
    "Meteorology",
    "Vegetation",
    "Vegetation",
    "Vegetation",
    "FAI",
    "FAI",
    "FAI",
    "FAI",
    "Anthropogenic",
    "All endogenous variables",
    "All endogenous variables",
    "All component models",
    "All component models"
  ),
  Relation = c(
    rep("Directed hypothesized pathway", 8L),
    "Undirected exogenous covariance",
    "Nuisance adjustment",
    "Nuisance adjustment",
    "Random intercept",
    "Random intercept"
  ),
  Included_In_Final_Mechanism_Diagram = c(
    rep(TRUE, 9L),
    rep(FALSE, 4L)
  ),
  Causal_Interpretation = c(
    "Plausible upstream physical influence",
    "Plausible physical influence",
    "Hypothesized ecological mediation",
    "Hypothesized land-use/access influence on vegetation",
    "Conditional direct association with fire activity",
    "Conditional direct association with fire activity",
    "Conditional direct association with fire activity",
    "Conditional direct association with fire activity",
    "No causal direction imposed between exogenous domains",
    "Control only",
    "Control only",
    "Repeated-grid heterogeneity control",
    "Shared-year heterogeneity control"
  )
)

fwrite(path_lock, PATH_LOCK_FILE)

meteorology_formula <- as.formula(
  paste(
    "Meteorology_Score ~",
    "Topography_Score + Country + Year_Z +",
    "(1 | GRID_UID) + (1 | Year_Factor)"
  )
)

vegetation_formula <- as.formula(
  paste(
    "Vegetation_Score ~",
    "Meteorology_Score + Topography_Score +",
    "Anthropogenic_Score + Country + Year_Z +",
    "(1 | GRID_UID) + (1 | Year_Factor)"
  )
)

primary_fai_formula <- as.formula(
  paste(
    "FAI_Q95 ~",
    "Meteorology_Score + Vegetation_Score +",
    "Topography_Score + Anthropogenic_Score +",
    "Country + Year_Z +",
    "offset(Log_ForestPixelCount) +",
    "(1 | GRID_UID) + (1 | Year_Factor)"
  )
)

sensitivity_fai_formula <- as.formula(
  paste(
    "FAI_EqualContribution_Sensitivity ~",
    "Meteorology_Score + Vegetation_Score +",
    "Topography_Score + Anthropogenic_Score +",
    "Country + Year_Z +",
    "offset(Log_ForestPixelCount) +",
    "(1 | GRID_UID) + (1 | Year_Factor)"
  )
)

formula_lock_rows <- list()

for (season in 1:3) {
  season_label <- unname(SEASON_LABELS[as.character(season)])

  formula_lock_rows[[length(formula_lock_rows) + 1L]] <- data.table(
    Season = season,
    Season_Label = season_label,
    Model_Role = "Meteorology_mediator",
    Primary_SEM_Model = TRUE,
    Response = "Meteorology_Score",
    Formula = formula_text(meteorology_formula),
    Family = "Gaussian identity",
    Exposure_Offset = FALSE
  )

  formula_lock_rows[[length(formula_lock_rows) + 1L]] <- data.table(
    Season = season,
    Season_Label = season_label,
    Model_Role = "Vegetation_mediator",
    Primary_SEM_Model = TRUE,
    Response = "Vegetation_Score",
    Formula = formula_text(vegetation_formula),
    Family = "Gaussian identity",
    Exposure_Offset = FALSE
  )

  formula_lock_rows[[length(formula_lock_rows) + 1L]] <- data.table(
    Season = season,
    Season_Label = season_label,
    Model_Role = "FAI_primary_outcome",
    Primary_SEM_Model = TRUE,
    Response = PRIMARY_RESPONSE,
    Formula = formula_text(primary_fai_formula),
    Family = "Tweedie log",
    Exposure_Offset = TRUE
  )

  formula_lock_rows[[length(formula_lock_rows) + 1L]] <- data.table(
    Season = season,
    Season_Label = season_label,
    Model_Role = "FAI_equal_sensitivity_outcome",
    Primary_SEM_Model = FALSE,
    Response = SENSITIVITY_RESPONSE,
    Formula = formula_text(sensitivity_fai_formula),
    Family = "Tweedie log",
    Exposure_Offset = TRUE
  )
}

formula_lock <- rbindlist(formula_lock_rows, fill = TRUE)
fwrite(formula_lock, FORMULA_LOCK_FILE)

# -----------------------------------------------------------------------------
# Response support and exogenous covariance
# -----------------------------------------------------------------------------

response_support <- data[
  ,
  .(
    N = .N,
    Grid_N = uniqueN(GRID_UID),
    Year_N = uniqueN(Year),
    Primary_Zero_N = sum(get(PRIMARY_RESPONSE) == 0),
    Primary_Zero_Percent = 100 * mean(get(PRIMARY_RESPONSE) == 0),
    Primary_Positive_N = sum(get(PRIMARY_RESPONSE) > 0),
    Primary_Mean = mean(get(PRIMARY_RESPONSE)),
    Primary_Positive_Mean = mean(
      get(PRIMARY_RESPONSE)[get(PRIMARY_RESPONSE) > 0]
    ),
    Equal_Zero_N = sum(get(SENSITIVITY_RESPONSE) == 0),
    Equal_Zero_Percent = 100 * mean(
      get(SENSITIVITY_RESPONSE) == 0
    ),
    Equal_Positive_N = sum(get(SENSITIVITY_RESPONSE) > 0),
    Equal_Mean = mean(get(SENSITIVITY_RESPONSE)),
    Equal_Positive_Mean = mean(
      get(SENSITIVITY_RESPONSE)[get(SENSITIVITY_RESPONSE) > 0]
    )
  ),
  by = .(
    Season,
    Season_Label,
    Country
  )
]

fwrite(response_support, RESPONSE_SUPPORT_FILE)

exogenous_correlation <- data[
  ,
  .(
    N = .N,
    Pearson_R = as.numeric(
      stats::cor(
        Topography_Score,
        Anthropogenic_Score,
        method = "pearson",
        use = "complete.obs"
      )
    ),
    Spearman_Rho = as.numeric(
      stats::cor(
        Topography_Score,
        Anthropogenic_Score,
        method = "spearman",
        use = "complete.obs"
      )
    ),
    Relation_Lock = (
      "Correlated exogenous variables; no directional path imposed"
    )
  ),
  by = .(
    Season,
    Season_Label
  )
]

fwrite(exogenous_correlation, EXOGENOUS_CORRELATION_FILE)

# -----------------------------------------------------------------------------
# Fit all seasonal component models
# -----------------------------------------------------------------------------

records <- list()

for (season in 1:3) {
  season_label <- unname(SEASON_LABELS[as.character(season)])
  season_prefix <- paste0("S", season, "_", season_label)
  season_data <- data[Season == season]

  log_message(
    paste(
      "Beginning seasonal model set:",
      season_label,
      "| rows=",
      nrow(season_data)
    )
  )

  meteorology_id <- paste0(
    season_prefix,
    "_Meteorology_Mediator"
  )

  vegetation_id <- paste0(
    season_prefix,
    "_Vegetation_Mediator"
  )

  primary_fai_id <- paste0(
    season_prefix,
    "_FAI_Primary"
  )

  sensitivity_fai_id <- paste0(
    season_prefix,
    "_FAI_Equal_Sensitivity"
  )

  records[[meteorology_id]] <- fit_locked_model(
    model_id = meteorology_id,
    model_role = "Meteorology_mediator",
    season = season,
    season_label = season_label,
    response_label = "Meteorology_Score",
    formula_object = meteorology_formula,
    family_object = gaussian(link = "identity"),
    family_label = "gaussian_identity",
    data_object = season_data,
    primary_model = TRUE
  )

  records[[vegetation_id]] <- fit_locked_model(
    model_id = vegetation_id,
    model_role = "Vegetation_mediator",
    season = season,
    season_label = season_label,
    response_label = "Vegetation_Score",
    formula_object = vegetation_formula,
    family_object = gaussian(link = "identity"),
    family_label = "gaussian_identity",
    data_object = season_data,
    primary_model = TRUE
  )

  records[[primary_fai_id]] <- fit_locked_model(
    model_id = primary_fai_id,
    model_role = "FAI_primary_outcome",
    season = season,
    season_label = season_label,
    response_label = PRIMARY_RESPONSE,
    formula_object = primary_fai_formula,
    family_object = glmmTMB::tweedie(link = "log"),
    family_label = "tweedie_log",
    data_object = season_data,
    primary_model = TRUE
  )

  records[[sensitivity_fai_id]] <- fit_locked_model(
    model_id = sensitivity_fai_id,
    model_role = "FAI_equal_sensitivity_outcome",
    season = season,
    season_label = season_label,
    response_label = SENSITIVITY_RESPONSE,
    formula_object = sensitivity_fai_formula,
    family_object = glmmTMB::tweedie(link = "log"),
    family_label = "tweedie_log",
    data_object = season_data,
    primary_model = FALSE
  )

  bundle <- list(
    code_version = CODE_VERSION,
    season = season,
    season_label = season_label,
    path_graph = path_lock,
    topography_anthropogenic_covariance = (
      exogenous_correlation[Season == season]
    ),
    meteorology_model_id = meteorology_id,
    vegetation_model_id = vegetation_id,
    primary_fai_model_id = primary_fai_id,
    sensitivity_fai_model_id = sensitivity_fai_id,
    meteorology_model = records[[meteorology_id]]$fit,
    vegetation_model = records[[vegetation_id]]$fit,
    primary_fai_model = records[[primary_fai_id]]$fit,
    sensitivity_fai_model = records[[sensitivity_fai_id]]$fit,
    final_effects_calculated = FALSE,
    final_path_diagram_created = FALSE
  )

  saveRDS(
    bundle,
    file.path(
      BUNDLE_DIR,
      paste0(season_prefix, "_SEM_Bundle.rds")
    ),
    compress = "gzip"
  )

  rm(season_data, bundle)
  invisible(gc())
}

# -----------------------------------------------------------------------------
# Extract model tables
# -----------------------------------------------------------------------------

registry_rows <- list()
attempt_rows <- list()
fixed_rows <- list()
random_rows <- list()
fit_stat_rows <- list()
qa_rows <- list()

for (model_id in names(records)) {
  record <- records[[model_id]]
  model_path <- file.path(MODEL_DIR, paste0(model_id, ".rds"))

  registry_rows[[length(registry_rows) + 1L]] <- data.table(
    Model_ID = model_id,
    Status = record$status,
    Season = record$season,
    Season_Label = record$season_label,
    Model_Role = record$model_role,
    Response = record$response_label,
    Primary_SEM_Model = record$primary_model,
    Family = record$family_label,
    Formula = formula_text(record$formula),
    Selected_Optimizer = record$selected_optimizer,
    Optimizer_Fallback_Used = record$fallback_used,
    Model_Relative_Path = file.path("Models", basename(model_path)),
    Model_File_Size_Bytes = file.info(model_path)$size,
    Model_SHA256 = sha256_file(model_path)
  )

  attempt_rows[[length(attempt_rows) + 1L]] <- copy(record$attempts)

  if (record$status != "SUCCESS") next

  fit <- record$fit
  coefficient_matrix <- summary(fit)$coefficients$cond
  coefficient_table <- as.data.table(
    coefficient_matrix,
    keep.rownames = "Term"
  )

  setnames(
    coefficient_table,
    names(coefficient_table),
    c(
      "Term",
      "Estimate",
      "Std_Error",
      "Test_Statistic",
      "P_Value"
    )
  )

  coefficient_table[
    ,
    `:=`(
      Model_ID = model_id,
      Season = record$season,
      Season_Label = record$season_label,
      Model_Role = record$model_role,
      Response = record$response_label,
      Primary_SEM_Model = record$primary_model,
      Family = record$family_label,
      Coefficient_Scale = if (
        record$family_label == "tweedie_log"
      ) {
        "Log expected FAI per one-SD predictor change"
      } else {
        "Standardized linear coefficient"
      }
    )
  ]

  fixed_rows[[length(fixed_rows) + 1L]] <- coefficient_table

  random_table <- extract_random_effect_table(fit)

  random_table[
    ,
    `:=`(
      Model_ID = model_id,
      Season = record$season,
      Season_Label = record$season_label,
      Model_Role = record$model_role,
      Response = record$response_label
    )
  ]

  random_rows[[length(random_rows) + 1L]] <- random_table

  fit_stat_rows[[length(fit_stat_rows) + 1L]] <- data.table(
    Model_ID = model_id,
    Season = record$season,
    Season_Label = record$season_label,
    Model_Role = record$model_role,
    Response = record$response_label,
    Primary_SEM_Model = record$primary_model,
    N_Used = nobs(fit),
    AIC = safe_number(AIC(fit)),
    BIC = safe_number(BIC(fit)),
    LogLik = safe_number(logLik(fit)),
    Residual_DF = safe_number(df.residual(fit)),
    Dispersion_Parameter = safe_number(sigma(fit)),
    Tweedie_Power = if (
      record$family_label == "tweedie_log"
    ) {
      extract_tweedie_power(fit)
    } else {
      NA_real_
    }
  )

  selected_attempt <- record$attempts[Selected == TRUE][1L]
  random_sd <- random_table$Std_Dev

  qa_rows[[length(qa_rows) + 1L]] <- data.table(
    Model_ID = model_id,
    Season = record$season,
    Season_Label = record$season_label,
    Model_Role = record$model_role,
    Response = record$response_label,
    Primary_SEM_Model = record$primary_model,
    Convergence_Code = selected_attempt$Convergence_Code,
    Positive_Definite_Hessian = (
      selected_attempt$Positive_Definite_Hessian
    ),
    Objective = selected_attempt$Objective,
    SDReport_Max_Absolute_Gradient = (
      selected_attempt$SDReport_Max_Absolute_Gradient
    ),
    Optimizer_Max_Absolute_Gradient = (
      selected_attempt$Optimizer_Max_Absolute_Gradient
    ),
    Coefficients_Finite = selected_attempt$Coefficients_Finite,
    Standard_Errors_Finite = (
      selected_attempt$Standard_Errors_Finite
    ),
    Random_Effects_Valid = selected_attempt$Random_Effects_Valid,
    Random_Effect_N = nrow(random_table),
    Minimum_Random_Effect_SD = min(random_sd, na.rm = TRUE),
    Near_Zero_Random_Effect = any(random_sd < 1e-4),
    Warning_Text = selected_attempt$Warning_Text,
    QA_Passed = selected_attempt$Computationally_Valid
  )
}

model_registry <- rbindlist(registry_rows, fill = TRUE)
fit_attempts <- rbindlist(attempt_rows, fill = TRUE)
fixed_effects <- rbindlist(fixed_rows, fill = TRUE)
random_effects <- rbindlist(random_rows, fill = TRUE)
fit_statistics <- rbindlist(fit_stat_rows, fill = TRUE)
convergence_qa <- rbindlist(qa_rows, fill = TRUE)

fwrite(model_registry, MODEL_REGISTRY_FILE)
fwrite(fit_attempts, FIT_ATTEMPTS_FILE)
fwrite(fixed_effects, FIXED_EFFECT_FILE)
fwrite(random_effects, RANDOM_EFFECT_FILE)
fwrite(fit_statistics, FIT_STATISTICS_FILE)
fwrite(convergence_qa, CONVERGENCE_FILE)

# -----------------------------------------------------------------------------
# Primary path table and sensitivity comparison
# -----------------------------------------------------------------------------

mechanism_terms <- c(
  "Meteorology_Score",
  "Vegetation_Score",
  "Topography_Score",
  "Anthropogenic_Score"
)

primary_path_coefficients <- fixed_effects[
  Primary_SEM_Model == TRUE &
    Term %in% mechanism_terms
]

primary_path_coefficients[
  ,
  Path := paste(Term, "->", Response)
]

setcolorder(
  primary_path_coefficients,
  c(
    "Season",
    "Season_Label",
    "Path",
    "Model_ID",
    "Model_Role",
    "Response",
    "Term",
    "Estimate",
    "Std_Error",
    "Test_Statistic",
    "P_Value",
    "Family",
    "Coefficient_Scale",
    "Primary_SEM_Model"
  )
)

fwrite(primary_path_coefficients, PRIMARY_PATH_FILE)

primary_outcome_paths <- fixed_effects[
  Model_Role == "FAI_primary_outcome" &
    Term %in% mechanism_terms,
  .(
    Season,
    Season_Label,
    Term,
    Primary_Estimate = Estimate,
    Primary_Std_Error = Std_Error,
    Primary_P_Value = P_Value
  )
]

sensitivity_outcome_paths <- fixed_effects[
  Model_Role == "FAI_equal_sensitivity_outcome" &
    Term %in% mechanism_terms,
  .(
    Season,
    Season_Label,
    Term,
    Sensitivity_Estimate = Estimate,
    Sensitivity_Std_Error = Std_Error,
    Sensitivity_P_Value = P_Value
  )
]

sensitivity_comparison <- merge(
  primary_outcome_paths,
  sensitivity_outcome_paths,
  by = c("Season", "Season_Label", "Term"),
  all = TRUE
)

sensitivity_comparison[
  ,
  `:=`(
    Sign_Consistent = (
      sign(Primary_Estimate) ==
        sign(Sensitivity_Estimate)
    ),
    Absolute_Estimate_Change = abs(
      Sensitivity_Estimate -
        Primary_Estimate
    ),
    Relative_Estimate_Change = abs(
      Sensitivity_Estimate -
        Primary_Estimate
    ) /
      pmax(abs(Primary_Estimate), 1e-12),
    Primary_Significant_0p05 = Primary_P_Value < 0.05,
    Sensitivity_Significant_0p05 = Sensitivity_P_Value < 0.05,
    Significance_Class_Consistent = (
      (Primary_P_Value < 0.05) ==
        (Sensitivity_P_Value < 0.05)
    )
  )
]

fwrite(sensitivity_comparison, SENSITIVITY_COMPARISON_FILE)

# -----------------------------------------------------------------------------
# Seasonal summary and final audit
# -----------------------------------------------------------------------------

season_summary <- model_registry[
  ,
  .(
    Registered_Models = .N,
    Successful_Models = sum(Status == "SUCCESS"),
    Failed_Models = sum(Status != "SUCCESS"),
    Primary_Models = sum(Primary_SEM_Model),
    Sensitivity_Models = sum(!Primary_SEM_Model),
    Optimizer_Fallbacks = sum(Optimizer_Fallback_Used),
    All_Models_Successful = all(Status == "SUCCESS")
  ),
  by = .(
    Season,
    Season_Label
  )
]

season_summary <- merge(
  season_summary,
  exogenous_correlation[
    ,
    .(
      Season,
      Pearson_Topography_Anthropogenic = Pearson_R,
      Spearman_Topography_Anthropogenic = Spearman_Rho
    )
  ],
  by = "Season",
  all.x = TRUE
)

season_summary <- merge(
  season_summary,
  response_support[
    ,
    .(
      Primary_FAI_Zero_Percent = weighted.mean(
        Primary_Zero_Percent,
        w = N
      ),
      Equal_FAI_Zero_Percent = weighted.mean(
        Equal_Zero_Percent,
        w = N
      )
    ),
    by = Season
  ],
  by = "Season",
  all.x = TRUE
)

season_summary[
  ,
  `:=`(
    Final_Direct_Indirect_Effects_Calculated = FALSE,
    Final_Path_Diagram_Created = FALSE
  )
]

fwrite(season_summary, SEASON_SUMMARY_FILE)

successful_model_n <- sum(model_registry$Status == "SUCCESS")
successful_primary_n <- sum(
  model_registry$Status == "SUCCESS" &
    model_registry$Primary_SEM_Model
)
successful_sensitivity_n <- sum(
  model_registry$Status == "SUCCESS" &
    !model_registry$Primary_SEM_Model
)

audit <- add_check(
  audit,
  "Models",
  "All_12_models_registered",
  nrow(model_registry) == EXPECTED_TOTAL_MODEL_N,
  nrow(model_registry),
  EXPECTED_TOTAL_MODEL_N
)

audit <- add_check(
  audit,
  "Models",
  "All_9_primary_models_successful",
  successful_primary_n == EXPECTED_PRIMARY_MODEL_N,
  successful_primary_n,
  EXPECTED_PRIMARY_MODEL_N
)

audit <- add_check(
  audit,
  "Models",
  "All_3_sensitivity_models_successful",
  successful_sensitivity_n == EXPECTED_SENSITIVITY_MODEL_N,
  successful_sensitivity_n,
  EXPECTED_SENSITIVITY_MODEL_N
)

audit <- add_check(
  audit,
  "Models",
  "All_successful_models_pass_QA",
  nrow(convergence_qa) == successful_model_n &&
    all(convergence_qa$QA_Passed),
  sum(convergence_qa$QA_Passed),
  successful_model_n
)

audit <- add_check(
  audit,
  "Path_Graph",
  "Topography_anthropogenic_direction_not_imposed",
  nrow(
    path_lock[
      From == "Topography" &
        To == "Anthropogenic" &
        Relation == "Undirected exogenous covariance"
    ]
  ) == 1L,
  nrow(
    path_lock[
      From == "Topography" &
        To == "Anthropogenic" &
        Relation == "Undirected exogenous covariance"
    ]
  ),
  1L
)

final_audit <- rbindlist(audit, fill = TRUE)
fwrite(final_audit, INPUT_AUDIT_FILE)

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
    "Step 04 final audit failed. See 01_Input_Integrity_Audit.csv.",
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
  Step03A_Code_Version = step03a_method$Code_Version,
  Step03A_Script_SHA256 = step03a_method$Script_SHA256,
  Step03A_Master = normalizePath(
    STEP03A_MASTER_FILE,
    winslash = "/",
    mustWork = TRUE
  ),
  Step03A_Master_SHA256 = actual_master_hash,
  Primary_Response = PRIMARY_RESPONSE,
  Sensitivity_Response = SENSITIVITY_RESPONSE,
  Exposure_Field = "ForestPixelCount",
  Offset_Field = "Log_ForestPixelCount",
  Seasons = as.list(SEASON_LABELS),
  Primary_Component_Models_Per_Season = 3L,
  Total_Primary_Models = EXPECTED_PRIMARY_MODEL_N,
  Total_Sensitivity_Outcome_Models = EXPECTED_SENSITIVITY_MODEL_N,
  Total_Fitted_Models = EXPECTED_TOTAL_MODEL_N,
  Path_Graph = list(
    Topography_to_Meteorology = TRUE,
    Topography_to_Vegetation = TRUE,
    Meteorology_to_Vegetation = TRUE,
    Anthropogenic_to_Vegetation = TRUE,
    Meteorology_to_FAI = TRUE,
    Vegetation_to_FAI = TRUE,
    Topography_to_FAI = TRUE,
    Anthropogenic_to_FAI = TRUE,
    Topography_correlated_with_Anthropogenic = TRUE,
    Direction_between_Topography_and_Anthropogenic = "Not imposed"
  ),
  Interpretation = list(
    Topography = (
      "Plausible upstream physical driver; causal hypothesis"
    ),
    Anthropogenic = paste(
      "Hypothesized driver represented by observational proxies;",
      "interpret as conditional pathway, not definitive causal proof"
    ),
    Topography_Anthropogenic = paste(
      "Correlated exogenous domains; no directional causal arrow",
      "between them"
    )
  ),
  Nuisance_Adjustments = c("Country", "Year_Z"),
  Random_Intercepts = c("GRID_UID", "Year"),
  Mediator_Family = "Gaussian identity",
  FAI_Family = "Tweedie log",
  Optimizer_Strategy = (
    "nlminb first; BFGS only if nlminb fails computational validation"
  ),
  piecewiseSEM_Package_Required = FALSE,
  Reason_piecewiseSEM_Not_Required = paste(
    "Component models and path graph are saved directly because the",
    "current CRAN package requires a newer R version than the user's",
    "R 4.1.3 environment."
  ),
  Final_Effect_Decomposition_Calculated = FALSE,
  Final_Path_Diagrams_Created = FALSE
)

write_json(method_definition, METHOD_FILE)

software_environment <- list(
  R_Version = R.version.string,
  Platform = R.version$platform,
  data_table = as.character(packageVersion("data.table")),
  glmmTMB = as.character(packageVersion("glmmTMB")),
  TMB = as.character(packageVersion("TMB")),
  Matrix = as.character(packageVersion("Matrix")),
  jsonlite = as.character(packageVersion("jsonlite")),
  digest = as.character(packageVersion("digest")),
  R_utils = as.character(packageVersion("R.utils"))
)

write_json(software_environment, SOFTWARE_FILE)

# Write the completion record before creating the manifest so the log hash
# remains stable.
log_message(
  paste(
    "Step 04 completed: three seasonal primary FAI component-model sets",
    "and three equal-contribution FAI sensitivity outcome models were fitted."
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
# Completion summary
# -----------------------------------------------------------------------------

cat("\n")
cat(
  "Output directory:",
  normalizePath(OUTPUT_ROOT, winslash = "/", mustWork = FALSE),
  "\n"
)
cat("Audit ERROR failures:", nrow(error_failures), "\n")
cat("Audit WARNING failures:", nrow(warning_failures), "\n")
cat(
  "Primary models successful:",
  successful_primary_n,
  "/",
  EXPECTED_PRIMARY_MODEL_N,
  "\n"
)
cat(
  "Sensitivity models successful:",
  successful_sensitivity_n,
  "/",
  EXPECTED_SENSITIVITY_MODEL_N,
  "\n"
)
cat(
  "Total models successful:",
  successful_model_n,
  "/",
  EXPECTED_TOTAL_MODEL_N,
  "\n"
)
cat(
  "Optimizer fallbacks used:",
  sum(model_registry$Optimizer_Fallback_Used),
  "\n"
)
cat(
  "Primary FAI path signs stable in sensitivity analysis:",
  sum(sensitivity_comparison$Sign_Consistent),
  "/",
  nrow(sensitivity_comparison),
  "\n"
)
cat(
  "Topography-Anthropogenic directional path imposed: False\n"
)
cat(
  "Topography-Anthropogenic exogenous covariance retained: True\n"
)
cat("Final direct/indirect effects calculated: False\n")
cat("Final path diagrams created: False\n")
