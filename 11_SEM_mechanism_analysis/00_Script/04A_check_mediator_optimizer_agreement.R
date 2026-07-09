# =============================================================================
# Step 04A: Targeted nlminb-BFGS optimizer agreement for six mediator models
#
# Recommended location
# --------------------
# <REPOSITORY_ROOT>/11_SEM_mechanism_analysis/00_Script/
# 04A_check_mediator_optimizer_agreement.R
#
# Scope
# -----
# Refit only the six Gaussian mediator models from sealed Step 04:
#   - 3 Meteorology mediator models
#   - 3 Vegetation mediator models
#
# The locked nlminb fits are compared with new BFGS fits for:
#   - computational validity;
#   - fixed-effect estimates;
#   - 12 mechanism-path signs;
#   - log-likelihood agreement;
#   - random-effect standard deviations.
#
# No path is changed. No FAI outcome model is refitted. No direct, indirect,
# or total effect is calculated in this step.
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
# Locked versions, paths, and thresholds
# -----------------------------------------------------------------------------

CODE_VERSION <- paste0(
  "2026-07-01_SEM_MEDIATOR_OPTIMIZER_AGREEMENT_V1_",
  "SIX_GAUSSIAN_MODELS_NLMINB_VS_BFGS"
)

RELEASE_NOTE <- paste0(
  "2026-07-08_RELEASE_",
  "SEQUENTIAL_RANDOM_EFFECT_DIFFERENCE_ASSIGNMENT"
)

EXPECTED_STEP04_VERSION <- paste0(
  "2026-07-01_SEM_THREE_SEASON_FAI_FITS_V1_",
  "NINE_PRIMARY_PLUS_THREE_SENSITIVITY_MODELS"
)

EXPECTED_STEP04_SCRIPT_SHA256 <- paste0(
  "a82e3e844d8439e6d2ff6e4495325d99",
  "a34794d96bdddef469095725399f2e94"
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
STEP04_ROOT <- file.path(SEM_ROOT, "04_Seasonal_FAI_Piecewise_SEM")

STEP03A_MASTER_FILE <- file.path(
  STEP03A_ROOT,
  "06_Locked_SEM_FAI_Sensitivity_Master.csv.gz"
)

STEP04_SCRIPT_FILE <- file.path(
  SCRIPT_DIR,
  "04_fit_three_seasonal_FAI_SEMs.R"
)
STEP04_METHOD_FILE <- file.path(STEP04_ROOT, "00_Method_Definition.json")
STEP04_AUDIT_FILE <- file.path(
  STEP04_ROOT,
  "01_Input_Integrity_Audit.csv"
)
STEP04_REGISTRY_FILE <- file.path(
  STEP04_ROOT,
  "06_Model_Registry.csv"
)
STEP04_MANIFEST_FILE <- file.path(
  STEP04_ROOT,
  "15_Output_Manifest.csv"
)
STEP04_MODEL_DIR <- file.path(STEP04_ROOT, "Models")

OUTPUT_ROOT <- file.path(
  SEM_ROOT,
  "04A_Mediator_Optimizer_Agreement"
)
BFGS_MODEL_DIR <- file.path(OUTPUT_ROOT, "BFGS_Models")

METHOD_FILE <- file.path(OUTPUT_ROOT, "00_Method_Definition.json")
INPUT_AUDIT_FILE <- file.path(
  OUTPUT_ROOT,
  "01_Input_Integrity_Audit.csv"
)
TARGET_REGISTRY_FILE <- file.path(
  OUTPUT_ROOT,
  "02_Target_Mediator_Model_Registry.csv"
)
BFGS_FIT_FILE <- file.path(
  OUTPUT_ROOT,
  "03_BFGS_Fit_Status.csv"
)
FIXED_AGREEMENT_FILE <- file.path(
  OUTPUT_ROOT,
  "04_All_Fixed_Effect_Agreement.csv"
)
PATH_AGREEMENT_FILE <- file.path(
  OUTPUT_ROOT,
  "05_Mechanism_Path_Agreement.csv"
)
RANDOM_AGREEMENT_FILE <- file.path(
  OUTPUT_ROOT,
  "06_Random_Effect_Agreement.csv"
)
OBJECTIVE_AGREEMENT_FILE <- file.path(
  OUTPUT_ROOT,
  "07_Objective_Agreement.csv"
)
MODEL_DECISION_FILE <- file.path(
  OUTPUT_ROOT,
  "08_Model_Level_Agreement_Decision.csv"
)
FINAL_DECISION_FILE <- file.path(
  OUTPUT_ROOT,
  "09_Final_Agreement_Decision.json"
)
MANIFEST_FILE <- file.path(
  OUTPUT_ROOT,
  "10_Output_Manifest.csv"
)
SOFTWARE_FILE <- file.path(
  OUTPUT_ROOT,
  "Software_Environment.json"
)
LOG_FILE <- file.path(
  OUTPUT_ROOT,
  "sem_step04A_mediator_optimizer_agreement.log"
)

EXPECTED_TARGET_MODEL_N <- 6L
EXPECTED_MECHANISM_PATH_N <- 12L

MECHANISM_ABSOLUTE_DIFFERENCE_TOLERANCE <- 0.02
LOGLIK_RELATIVE_DIFFERENCE_TOLERANCE <- 1e-6
RANDOM_SD_RELATIVE_DIFFERENCE_TOLERANCE <- 0.20

COUNTRY_LEVELS <- c("China", "NK", "Russia")

MECHANISM_TERMS <- c(
  "Meteorology_Score",
  "Topography_Score",
  "Anthropogenic_Score"
)

TARGET_ROLES <- c(
  "Meteorology_mediator",
  "Vegetation_mediator"
)

FORCE_RESTART <- identical(
  Sys.getenv("SEM_STEP04A_FORCE_RESTART", unset = "0"),
  "1"
)

TMB_THREADS <- suppressWarnings(
  as.integer(
    Sys.getenv("SEM_STEP04A_TMB_THREADS", unset = "1")
  )
)

if (!is.finite(TMB_THREADS) || TMB_THREADS < 1L) {
  stop("SEM_STEP04A_TMB_THREADS must be an integer >= 1.", call. = FALSE)
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

  if (length(values) == 0L) {
    return(NA_real_)
  }

  max(abs(values))
}

relative_difference <- function(new_value, reference_value) {
  abs(new_value - reference_value) /
    pmax(abs(reference_value), 1e-12)
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
      "04A_check_mediator_optimizer_agreement.R"
    ),
    winslash = "/",
    mustWork = FALSE
  )
}

extract_fixed_effects <- function(fit) {
  coefficient_matrix <- summary(fit)$coefficients$cond

  result <- as.data.table(
    coefficient_matrix,
    keep.rownames = "Term"
  )

  setnames(
    result,
    names(result),
    c(
      "Term",
      "Estimate",
      "Std_Error",
      "Test_Statistic",
      "P_Value"
    )
  )

  result
}

extract_random_effects <- function(fit) {
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
        Group = group_name,
        Term = term_names[[term_index]],
        Variance = covariance_matrix[term_index, term_index],
        Std_Dev = standard_deviations[[term_index]]
      )
    }
  }

  rbindlist(rows, fill = TRUE)
}

bfgs_control <- function() {
  glmmTMBControl(
    optimizer = optim,
    optArgs = list(method = "BFGS"),
    optCtrl = list(maxit = 10000L),
    parallel = TMB_THREADS
  )
}

fit_bfgs_model <- function(
  model_id,
  formula_object,
  data_object,
  source_model_hash
) {
  output_path <- file.path(
    BFGS_MODEL_DIR,
    paste0(model_id, "_BFGS.rds")
  )

  signature <- digest::digest(
    paste(
      CODE_VERSION,
      model_id,
      formula_text(formula_object),
      nrow(data_object),
      source_model_hash,
      sha256_file(STEP03A_MASTER_FILE),
      sep = "|"
    ),
    algo = "sha256",
    serialize = FALSE
  )

  if (file.exists(output_path) && !FORCE_RESTART) {
    existing <- tryCatch(
      readRDS(output_path),
      error = function(e) NULL
    )

    if (
      !is.null(existing) &&
        identical(existing$code_version, CODE_VERSION) &&
        identical(existing$input_signature, signature)
    ) {
      log_message(paste("Resuming completed BFGS fit:", model_id))
      return(existing)
    }
  }

  warning_messages <- character()
  start_time <- Sys.time()

  log_message(paste("Fitting BFGS comparison model:", model_id))

  fit <- tryCatch(
    withCallingHandlers(
      glmmTMB(
        formula = formula_object,
        data = data_object,
        family = gaussian(link = "identity"),
        ziformula = ~0,
        dispformula = ~1,
        control = bfgs_control()
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

  elapsed_seconds <- as.numeric(
    difftime(Sys.time(), start_time, units = "secs")
  )

  if (inherits(fit, "error")) {
    record <- list(
      code_version = CODE_VERSION,
      input_signature = signature,
      model_id = model_id,
      status = "FAILED",
      fit = NULL,
      convergence_code = NA_integer_,
      positive_definite_hessian = FALSE,
      objective = NA_real_,
      logLik = NA_real_,
      sdreport_max_absolute_gradient = NA_real_,
      optimizer_max_absolute_gradient = NA_real_,
      coefficients_finite = FALSE,
      standard_errors_finite = FALSE,
      random_effects_valid = FALSE,
      warning_text = paste(unique(warning_messages), collapse = " | "),
      error_text = conditionMessage(fit),
      elapsed_seconds = elapsed_seconds,
      saved_at = as.character(Sys.time())
    )

    saveRDS(record, output_path, compress = "gzip")
    return(record)
  }

  coefficient_table <- extract_fixed_effects(fit)
  random_table <- extract_random_effects(fit)

  coefficients_finite <- all(
    is.finite(coefficient_table$Estimate)
  )

  standard_errors_finite <- all(
    is.finite(coefficient_table$Std_Error)
  )

  required_random_groups <- c("GRID_UID", "Year_Factor")

  random_effects_valid <- (
    nrow(random_table) >= 2L &&
      all(required_random_groups %in% random_table$Group) &&
      all(is.finite(random_table$Std_Dev)) &&
      all(random_table$Std_Dev > 0)
  )

  convergence_code <- fit$fit$convergence
  positive_definite_hessian <- isTRUE(fit$sdr$pdHess)
  objective <- safe_number(fit$fit$objective)
  log_likelihood <- safe_number(logLik(fit))

  sdreport_gradient <- finite_max_abs(
    tryCatch(
      fit$sdr$gradient.fixed,
      error = function(e) numeric()
    )
  )

  optimizer_gradient <- finite_max_abs(
    tryCatch(
      fit$obj$gr(fit$fit$par),
      error = function(e) numeric()
    )
  )

  valid <- (
    identical(as.integer(convergence_code), 0L) &&
      positive_definite_hessian &&
      is.finite(objective) &&
      is.finite(log_likelihood) &&
      coefficients_finite &&
      standard_errors_finite &&
      random_effects_valid
  )

  record <- list(
    code_version = CODE_VERSION,
    input_signature = signature,
    model_id = model_id,
    status = if (valid) "SUCCESS" else "FAILED",
    fit = fit,
    convergence_code = convergence_code,
    positive_definite_hessian = positive_definite_hessian,
    objective = objective,
    logLik = log_likelihood,
    sdreport_max_absolute_gradient = sdreport_gradient,
    optimizer_max_absolute_gradient = optimizer_gradient,
    coefficients_finite = coefficients_finite,
    standard_errors_finite = standard_errors_finite,
    random_effects_valid = random_effects_valid,
    warning_text = paste(unique(warning_messages), collapse = " | "),
    error_text = "",
    elapsed_seconds = elapsed_seconds,
    saved_at = as.character(Sys.time())
  )

  saveRDS(record, output_path, compress = "gzip")

  log_message(
    paste(
      "BFGS model completed:",
      model_id,
      "| status=",
      record$status
    )
  )

  record
}

# -----------------------------------------------------------------------------
# Initialize output folder
# -----------------------------------------------------------------------------

if (FORCE_RESTART && dir.exists(OUTPUT_ROOT)) {
  unlink(OUTPUT_ROOT, recursive = TRUE, force = TRUE)
}

dir.create(OUTPUT_ROOT, recursive = TRUE, showWarnings = FALSE)
dir.create(BFGS_MODEL_DIR, recursive = TRUE, showWarnings = FALSE)

if (!file.exists(LOG_FILE) || FORCE_RESTART) {
  writeLines(character(), LOG_FILE)
}

log_message("Starting Step 04A mediator optimizer-agreement check.")
log_message(paste("Code version:", CODE_VERSION))
log_message(paste("Release note:", RELEASE_NOTE))

# -----------------------------------------------------------------------------
# Verify sealed Step 04 inputs
# -----------------------------------------------------------------------------

required_files <- c(
  STEP03A_MASTER_FILE,
  STEP04_SCRIPT_FILE,
  STEP04_METHOD_FILE,
  STEP04_AUDIT_FILE,
  STEP04_REGISTRY_FILE,
  STEP04_MANIFEST_FILE
)

missing_files <- required_files[!file.exists(required_files)]

if (length(missing_files) > 0L) {
  stop(
    paste(
      "Missing required files:",
      paste(missing_files, collapse = "\n"),
      sep = "\n"
    ),
    call. = FALSE
  )
}

audit <- list()

step04_method <- read_json(STEP04_METHOD_FILE)

audit <- add_check(
  audit,
  "Version",
  "Step04_code_version",
  identical(step04_method$Code_Version, EXPECTED_STEP04_VERSION),
  step04_method$Code_Version,
  EXPECTED_STEP04_VERSION
)

actual_step04_script_hash <- sha256_file(STEP04_SCRIPT_FILE)

audit <- add_check(
  audit,
  "Hash",
  "Step04_script_hash",
  identical(
    actual_step04_script_hash,
    EXPECTED_STEP04_SCRIPT_SHA256
  ),
  actual_step04_script_hash,
  EXPECTED_STEP04_SCRIPT_SHA256
)

formal_step04_audit <- fread(STEP04_AUDIT_FILE)

audit <- add_check(
  audit,
  "Step04_Audit",
  "All_Step04_ERROR_checks_pass",
  all(
    vapply(
      formal_step04_audit[Severity == "ERROR", Passed],
      parse_bool,
      logical(1)
    )
  ),
  sum(
    vapply(
      formal_step04_audit[Severity == "ERROR", Passed],
      parse_bool,
      logical(1)
    )
  ),
  nrow(formal_step04_audit[Severity == "ERROR"])
)

step04_manifest <- fread(STEP04_MANIFEST_FILE)

manifest_status_rows <- list()

for (i in seq_len(nrow(step04_manifest))) {
  row <- step04_manifest[i]

  path <- file.path(
    STEP04_ROOT,
    gsub("\\\\", "/", row$Relative_Path)
  )

  exists <- file.exists(path)
  actual_hash <- if (exists) sha256_file(path) else ""
  actual_size <- if (exists) file.info(path)$size else -1

  manifest_status_rows[[length(manifest_status_rows) + 1L]] <-
    data.table(
      Relative_Path = row$Relative_Path,
      Exists = exists,
      Hash_Matches = exists &&
        identical(actual_hash, row$SHA256),
      Size_Matches = exists &&
        identical(
          as.numeric(actual_size),
          as.numeric(row$File_Size_Bytes)
        )
    )
}

step04_manifest_status <- rbindlist(
  manifest_status_rows,
  fill = TRUE
)

audit <- add_check(
  audit,
  "Manifest",
  "All_Step04_output_hashes_match",
  all(
    step04_manifest_status$Exists &
      step04_manifest_status$Hash_Matches &
      step04_manifest_status$Size_Matches
  ),
  sum(
    step04_manifest_status$Exists &
      step04_manifest_status$Hash_Matches &
      step04_manifest_status$Size_Matches
  ),
  nrow(step04_manifest_status)
)

registry <- fread(STEP04_REGISTRY_FILE)

target_registry <- registry[
  Primary_SEM_Model == TRUE &
    Model_Role %in% TARGET_ROLES
]

setorder(target_registry, Season, Model_Role)

audit <- add_check(
  audit,
  "Targets",
  "Six_target_mediator_models",
  nrow(target_registry) == EXPECTED_TARGET_MODEL_N,
  nrow(target_registry),
  EXPECTED_TARGET_MODEL_N
)

audit <- add_check(
  audit,
  "Targets",
  "All_target_nlminb_models_successful",
  all(
    target_registry$Status == "SUCCESS" &
      target_registry$Selected_Optimizer == "nlminb"
  ),
  sum(
    target_registry$Status == "SUCCESS" &
      target_registry$Selected_Optimizer == "nlminb"
  ),
  EXPECTED_TARGET_MODEL_N
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
    "Step 04A input audit failed. See 01_Input_Integrity_Audit.csv.",
    call. = FALSE
  )
}

fwrite(target_registry, TARGET_REGISTRY_FILE)

# -----------------------------------------------------------------------------
# Read and prepare the same locked analysis data
# -----------------------------------------------------------------------------

log_message("Reading locked Step 03A master for BFGS refits.")

data <- fread(STEP03A_MASTER_FILE, showProgress = TRUE)

data[, GRID_UID := factor(as.character(GRID_UID))]
data[
  ,
  Country := factor(
    as.character(Country),
    levels = COUNTRY_LEVELS
  )
]
data[, Year := as.integer(Year)]
data[, Year_Z := as.numeric(Year_Z)]
data[, Year_Factor := factor(Year)]
data[, Season := as.integer(Season)]

numeric_fields <- c(
  "Meteorology_Score",
  "Vegetation_Score",
  "Topography_Score",
  "Anthropogenic_Score"
)

for (field in numeric_fields) {
  data[, (field) := as.numeric(get(field))]
}

# -----------------------------------------------------------------------------
# Refit six models with BFGS
# -----------------------------------------------------------------------------

records <- list()

for (i in seq_len(nrow(target_registry))) {
  row <- target_registry[i]
  model_id <- row$Model_ID
  source_model_path <- file.path(
    STEP04_MODEL_DIR,
    paste0(model_id, ".rds")
  )

  if (!file.exists(source_model_path)) {
    stop(
      paste("Missing locked Step 04 model:", source_model_path),
      call. = FALSE
    )
  }

  source_record <- readRDS(source_model_path)

  season_data <- data[Season == source_record$season]

  records[[model_id]] <- fit_bfgs_model(
    model_id = model_id,
    formula_object = source_record$formula,
    data_object = season_data,
    source_model_hash = sha256_file(source_model_path)
  )

  rm(season_data)
  invisible(gc())
}

# -----------------------------------------------------------------------------
# Extract agreement tables
# -----------------------------------------------------------------------------

bfgs_status_rows <- list()
fixed_agreement_rows <- list()
path_agreement_rows <- list()
random_agreement_rows <- list()
objective_agreement_rows <- list()
decision_rows <- list()

for (i in seq_len(nrow(target_registry))) {
  registry_row <- target_registry[i]
  model_id <- registry_row$Model_ID

  source_model_path <- file.path(
    STEP04_MODEL_DIR,
    paste0(model_id, ".rds")
  )
  source_record <- readRDS(source_model_path)
  bfgs_record <- records[[model_id]]

  bfgs_model_path <- file.path(
    BFGS_MODEL_DIR,
    paste0(model_id, "_BFGS.rds")
  )

  bfgs_status_rows[[length(bfgs_status_rows) + 1L]] <-
    data.table(
      Model_ID = model_id,
      Season = source_record$season,
      Season_Label = source_record$season_label,
      Model_Role = source_record$model_role,
      BFGS_Status = bfgs_record$status,
      Convergence_Code = bfgs_record$convergence_code,
      Positive_Definite_Hessian = (
        bfgs_record$positive_definite_hessian
      ),
      Objective = bfgs_record$objective,
      LogLik = bfgs_record$logLik,
      SDReport_Max_Absolute_Gradient = (
        bfgs_record$sdreport_max_absolute_gradient
      ),
      Optimizer_Max_Absolute_Gradient = (
        bfgs_record$optimizer_max_absolute_gradient
      ),
      Coefficients_Finite = bfgs_record$coefficients_finite,
      Standard_Errors_Finite = (
        bfgs_record$standard_errors_finite
      ),
      Random_Effects_Valid = bfgs_record$random_effects_valid,
      Warning_Text = bfgs_record$warning_text,
      Error_Text = bfgs_record$error_text,
      Elapsed_Seconds = bfgs_record$elapsed_seconds,
      BFGS_Model_File = file.path(
        "BFGS_Models",
        basename(bfgs_model_path)
      ),
      BFGS_Model_SHA256 = sha256_file(bfgs_model_path)
    )

  if (
    source_record$status != "SUCCESS" ||
      bfgs_record$status != "SUCCESS"
  ) {
    decision_rows[[length(decision_rows) + 1L]] <- data.table(
      Model_ID = model_id,
      Season = source_record$season,
      Season_Label = source_record$season_label,
      Model_Role = source_record$model_role,
      BFGS_Computationally_Valid = FALSE,
      Mechanism_Path_N = NA_integer_,
      Mechanism_Sign_Agreement_N = NA_integer_,
      Maximum_Mechanism_Absolute_Difference = NA_real_,
      LogLik_Relative_Difference = NA_real_,
      Maximum_Random_SD_Relative_Difference = NA_real_,
      Model_Agreement_Passed = FALSE
    )
    next
  }

  nlminb_fit <- source_record$fit
  bfgs_fit <- bfgs_record$fit

  nlminb_fixed <- extract_fixed_effects(nlminb_fit)
  bfgs_fixed <- extract_fixed_effects(bfgs_fit)

  fixed_comparison <- merge(
    nlminb_fixed,
    bfgs_fixed,
    by = "Term",
    suffixes = c("_nlminb", "_BFGS"),
    all = TRUE
  )

  fixed_comparison[
    ,
    `:=`(
      Model_ID = model_id,
      Season = source_record$season,
      Season_Label = source_record$season_label,
      Model_Role = source_record$model_role,
      Absolute_Estimate_Difference = abs(
        Estimate_BFGS -
          Estimate_nlminb
      ),
      Relative_Estimate_Difference = relative_difference(
        Estimate_BFGS,
        Estimate_nlminb
      ),
      Sign_Consistent = (
        sign(Estimate_BFGS) ==
          sign(Estimate_nlminb)
      ),
      Significance_Class_Consistent = (
        (P_Value_BFGS < 0.05) ==
          (P_Value_nlminb < 0.05)
      )
    )
  ]

  fixed_agreement_rows[[length(fixed_agreement_rows) + 1L]] <-
    fixed_comparison

  mechanism_comparison <- fixed_comparison[
    Term %in% MECHANISM_TERMS
  ]

  mechanism_comparison[
    ,
    Within_Absolute_Difference_Tolerance := (
      Absolute_Estimate_Difference <=
        MECHANISM_ABSOLUTE_DIFFERENCE_TOLERANCE
    )
  ]

  path_agreement_rows[[length(path_agreement_rows) + 1L]] <-
    mechanism_comparison

  nlminb_random <- extract_random_effects(nlminb_fit)
  bfgs_random <- extract_random_effects(bfgs_fit)

  random_comparison <- merge(
    nlminb_random,
    bfgs_random,
    by = c("Group", "Term"),
    suffixes = c("_nlminb", "_BFGS"),
    all = TRUE
  )

  random_comparison[
    ,
    `:=`(
      Model_ID = model_id,
      Season = source_record$season,
      Season_Label = source_record$season_label,
      Model_Role = source_record$model_role,
      Absolute_SD_Difference = abs(
        Std_Dev_BFGS -
          Std_Dev_nlminb
      ),
      Relative_SD_Difference = relative_difference(
        Std_Dev_BFGS,
        Std_Dev_nlminb
      )
    )
  ]

  random_comparison[
    ,
    Within_Relative_SD_Tolerance := (
      Relative_SD_Difference <=
        RANDOM_SD_RELATIVE_DIFFERENCE_TOLERANCE
    )
  ]

  random_agreement_rows[[length(random_agreement_rows) + 1L]] <-
    random_comparison

  nlminb_loglik <- safe_number(logLik(nlminb_fit))
  bfgs_loglik <- safe_number(logLik(bfgs_fit))

  loglik_absolute_difference <- abs(
    bfgs_loglik -
      nlminb_loglik
  )

  loglik_relative_difference <- relative_difference(
    bfgs_loglik,
    nlminb_loglik
  )

  objective_agreement_rows[[length(objective_agreement_rows) + 1L]] <- data.table(
    Model_ID = model_id,
    Season = source_record$season,
    Season_Label = source_record$season_label,
    Model_Role = source_record$model_role,
    nlminb_LogLik = nlminb_loglik,
    BFGS_LogLik = bfgs_loglik,
    Absolute_LogLik_Difference = loglik_absolute_difference,
    Relative_LogLik_Difference = loglik_relative_difference,
    Within_LogLik_Relative_Tolerance = (
      loglik_relative_difference <=
        LOGLIK_RELATIVE_DIFFERENCE_TOLERANCE
    ),
    nlminb_AIC = safe_number(AIC(nlminb_fit)),
    BFGS_AIC = safe_number(AIC(bfgs_fit))
  )

  mechanism_path_n <- nrow(mechanism_comparison)
  sign_agreement_n <- sum(
    mechanism_comparison$Sign_Consistent
  )
  maximum_mechanism_difference <- max(
    mechanism_comparison$Absolute_Estimate_Difference
  )
  maximum_random_sd_difference <- max(
    random_comparison$Relative_SD_Difference
  )

  model_agreement_passed <- (
    bfgs_record$status == "SUCCESS" &&
      mechanism_path_n > 0L &&
      sign_agreement_n == mechanism_path_n &&
      maximum_mechanism_difference <=
        MECHANISM_ABSOLUTE_DIFFERENCE_TOLERANCE &&
      loglik_relative_difference <=
        LOGLIK_RELATIVE_DIFFERENCE_TOLERANCE &&
      maximum_random_sd_difference <=
        RANDOM_SD_RELATIVE_DIFFERENCE_TOLERANCE
  )

  decision_rows[[length(decision_rows) + 1L]] <- data.table(
    Model_ID = model_id,
    Season = source_record$season,
    Season_Label = source_record$season_label,
    Model_Role = source_record$model_role,
    BFGS_Computationally_Valid = TRUE,
    Mechanism_Path_N = mechanism_path_n,
    Mechanism_Sign_Agreement_N = sign_agreement_n,
    Maximum_Mechanism_Absolute_Difference = (
      maximum_mechanism_difference
    ),
    LogLik_Relative_Difference = loglik_relative_difference,
    Maximum_Random_SD_Relative_Difference = (
      maximum_random_sd_difference
    ),
    Model_Agreement_Passed = model_agreement_passed
  )
}

bfgs_status <- rbindlist(bfgs_status_rows, fill = TRUE)
fixed_agreement <- rbindlist(fixed_agreement_rows, fill = TRUE)
path_agreement <- rbindlist(path_agreement_rows, fill = TRUE)
random_agreement <- rbindlist(random_agreement_rows, fill = TRUE)
objective_agreement <- rbindlist(
  objective_agreement_rows,
  fill = TRUE
)
model_decision <- rbindlist(decision_rows, fill = TRUE)

fwrite(bfgs_status, BFGS_FIT_FILE)
fwrite(fixed_agreement, FIXED_AGREEMENT_FILE)
fwrite(path_agreement, PATH_AGREEMENT_FILE)
fwrite(random_agreement, RANDOM_AGREEMENT_FILE)
fwrite(objective_agreement, OBJECTIVE_AGREEMENT_FILE)
fwrite(model_decision, MODEL_DECISION_FILE)

# -----------------------------------------------------------------------------
# Final decision and audit
# -----------------------------------------------------------------------------

successful_bfgs_n <- sum(
  bfgs_status$BFGS_Status == "SUCCESS"
)

mechanism_path_n <- nrow(path_agreement)

mechanism_sign_agreement_n <- sum(
  path_agreement$Sign_Consistent
)

mechanism_tolerance_n <- sum(
  path_agreement$Within_Absolute_Difference_Tolerance
)

loglik_tolerance_n <- sum(
  objective_agreement$Within_LogLik_Relative_Tolerance
)

random_sd_tolerance_n <- sum(
  random_agreement$Within_Relative_SD_Tolerance
)

all_models_agree <- (
  successful_bfgs_n == EXPECTED_TARGET_MODEL_N &&
    nrow(model_decision) == EXPECTED_TARGET_MODEL_N &&
    all(model_decision$Model_Agreement_Passed)
)

audit <- add_check(
  audit,
  "BFGS",
  "All_six_BFGS_models_successful",
  successful_bfgs_n == EXPECTED_TARGET_MODEL_N,
  successful_bfgs_n,
  EXPECTED_TARGET_MODEL_N
)

audit <- add_check(
  audit,
  "Paths",
  "Twelve_mechanism_paths_compared",
  mechanism_path_n == EXPECTED_MECHANISM_PATH_N,
  mechanism_path_n,
  EXPECTED_MECHANISM_PATH_N
)

audit <- add_check(
  audit,
  "Paths",
  "All_mechanism_path_signs_agree",
  mechanism_sign_agreement_n == EXPECTED_MECHANISM_PATH_N,
  mechanism_sign_agreement_n,
  EXPECTED_MECHANISM_PATH_N
)

audit <- add_check(
  audit,
  "Paths",
  "All_mechanism_estimates_within_absolute_tolerance",
  mechanism_tolerance_n == EXPECTED_MECHANISM_PATH_N,
  mechanism_tolerance_n,
  EXPECTED_MECHANISM_PATH_N,
  detail = paste(
    "Absolute tolerance =",
    MECHANISM_ABSOLUTE_DIFFERENCE_TOLERANCE
  )
)

audit <- add_check(
  audit,
  "Objective",
  "All_logLik_values_within_relative_tolerance",
  loglik_tolerance_n == EXPECTED_TARGET_MODEL_N,
  loglik_tolerance_n,
  EXPECTED_TARGET_MODEL_N,
  detail = paste(
    "Relative tolerance =",
    LOGLIK_RELATIVE_DIFFERENCE_TOLERANCE
  )
)

audit <- add_check(
  audit,
  "Random_Effects",
  "All_random_SD_values_within_relative_tolerance",
  random_sd_tolerance_n == nrow(random_agreement),
  random_sd_tolerance_n,
  nrow(random_agreement),
  detail = paste(
    "Relative tolerance =",
    RANDOM_SD_RELATIVE_DIFFERENCE_TOLERANCE
  )
)

audit <- add_check(
  audit,
  "Decision",
  "All_six_mediator_models_pass_optimizer_agreement",
  all_models_agree,
  sum(model_decision$Model_Agreement_Passed),
  EXPECTED_TARGET_MODEL_N
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

final_decision <- list(
  Code_Version = CODE_VERSION,
  Created = as.character(Sys.time()),
  Target_Model_N = EXPECTED_TARGET_MODEL_N,
  Successful_BFGS_Model_N = successful_bfgs_n,
  Mechanism_Path_N = mechanism_path_n,
  Mechanism_Sign_Agreement_N = mechanism_sign_agreement_n,
  Mechanism_Estimates_Within_Tolerance_N = mechanism_tolerance_n,
  Maximum_Mechanism_Absolute_Difference = if (
    nrow(path_agreement) > 0L
  ) {
    max(path_agreement$Absolute_Estimate_Difference)
  } else {
    NA_real_
  },
  LogLik_Models_Within_Tolerance_N = loglik_tolerance_n,
  Maximum_LogLik_Relative_Difference = if (
    nrow(objective_agreement) > 0L
  ) {
    max(objective_agreement$Relative_LogLik_Difference)
  } else {
    NA_real_
  },
  Random_SD_Comparisons_Within_Tolerance_N = (
    random_sd_tolerance_n
  ),
  Random_SD_Comparison_N = nrow(random_agreement),
  Maximum_Random_SD_Relative_Difference = if (
    nrow(random_agreement) > 0L
  ) {
    max(random_agreement$Relative_SD_Difference)
  } else {
    NA_real_
  },
  Mechanism_Absolute_Difference_Tolerance = (
    MECHANISM_ABSOLUTE_DIFFERENCE_TOLERANCE
  ),
  LogLik_Relative_Difference_Tolerance = (
    LOGLIK_RELATIVE_DIFFERENCE_TOLERANCE
  ),
  Random_SD_Relative_Difference_Tolerance = (
    RANDOM_SD_RELATIVE_DIFFERENCE_TOLERANCE
  ),
  Optimizer_Agreement_Passed = all_models_agree,
  Ready_For_Final_Effect_Decomposition = (
    all_models_agree &&
      nrow(error_failures) == 0L
  ),
  Final_Direct_Indirect_Effects_Calculated = FALSE,
  Final_Path_Diagrams_Created = FALSE
)

write_json(final_decision, FINAL_DECISION_FILE)

if (nrow(error_failures) > 0L) {
  stop(
    "Step 04A optimizer-agreement audit failed. See output tables.",
    call. = FALSE
  )
}

# -----------------------------------------------------------------------------
# Method and software metadata
# -----------------------------------------------------------------------------

method_definition <- list(
  Code_Version = CODE_VERSION,
  Release_Note = RELEASE_NOTE,
  Created = as.character(Sys.time()),
  Script_Path = script_path(),
  Script_SHA256 = if (file.exists(script_path())) {
    sha256_file(script_path())
  } else {
    NA_character_
  },
  Step04_Code_Version = step04_method$Code_Version,
  Step04_Script_SHA256 = step04_method$Script_SHA256,
  Step04_Manifest_Verified = TRUE,
  Step03A_Master = normalizePath(
    STEP03A_MASTER_FILE,
    winslash = "/",
    mustWork = TRUE
  ),
  Step03A_Master_SHA256 = sha256_file(STEP03A_MASTER_FILE),
  Target_Model_Roles = TARGET_ROLES,
  Target_Model_N = EXPECTED_TARGET_MODEL_N,
  Source_Optimizer = "nlminb",
  Comparison_Optimizer = "optim BFGS",
  Comparison_Family = "Gaussian identity",
  Mechanism_Absolute_Difference_Tolerance = (
    MECHANISM_ABSOLUTE_DIFFERENCE_TOLERANCE
  ),
  LogLik_Relative_Difference_Tolerance = (
    LOGLIK_RELATIVE_DIFFERENCE_TOLERANCE
  ),
  Random_SD_Relative_Difference_Tolerance = (
    RANDOM_SD_RELATIVE_DIFFERENCE_TOLERANCE
  ),
  FAI_Outcome_Models_Refitted = FALSE,
  Path_Graph_Changed = FALSE,
  Variables_Changed = FALSE,
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

# Write completion before manifest so the log hash remains fixed.
log_message(
  paste(
    "Step 04A completed: six Gaussian mediator models passed",
    "targeted nlminb-BFGS optimizer-agreement validation."
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
# Console summary
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
  "BFGS mediator models successful:",
  successful_bfgs_n,
  "/",
  EXPECTED_TARGET_MODEL_N,
  "\n"
)
cat(
  "Mechanism path signs consistent:",
  mechanism_sign_agreement_n,
  "/",
  mechanism_path_n,
  "\n"
)
cat(
  "Mechanism estimates within tolerance:",
  mechanism_tolerance_n,
  "/",
  mechanism_path_n,
  "\n"
)
cat(
  "Maximum mechanism absolute difference:",
  format(
    max(path_agreement$Absolute_Estimate_Difference),
    digits = 10
  ),
  "\n"
)
cat(
  "LogLik models within tolerance:",
  loglik_tolerance_n,
  "/",
  EXPECTED_TARGET_MODEL_N,
  "\n"
)
cat(
  "Random-effect SD comparisons within tolerance:",
  random_sd_tolerance_n,
  "/",
  nrow(random_agreement),
  "\n"
)
cat(
  "Optimizer agreement passed:",
  all_models_agree,
  "\n"
)
cat(
  "Ready for final effect decomposition:",
  all_models_agree,
  "\n"
)
cat("FAI outcome models refitted: False\n")
cat("Path graph changed: False\n")
cat("Final direct/indirect effects calculated: False\n")
cat("Final path diagrams created: False\n")
