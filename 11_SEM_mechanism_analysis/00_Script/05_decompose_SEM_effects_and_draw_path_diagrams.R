# =============================================================================
# Step 05: Seasonal SEM effect decomposition and final path diagrams
#
# Recommended location
# --------------------
# <REPOSITORY_ROOT>/11_SEM_mechanism_analysis/00_Script/
# 05_decompose_SEM_effects_and_draw_path_diagrams.R
#
# Purpose
# -------
# Using the sealed Step 04 nlminb component models and the sealed Step 04A
# optimizer-agreement decision, this script:
#
#   1. extracts the eight directed mechanism paths for each season;
#   2. calculates direct, specific indirect, total indirect, and total effects;
#   3. propagates fixed-effect uncertainty with reproducible multivariate
#      normal Monte Carlo simulation;
#   4. repeats the outcome decomposition for the equal-contribution FAI
#      sensitivity response;
#   5. creates three final seasonal path diagrams for the primary FAI.
#
# Effect scale
# ------------
# Mediator equations use standardized Gaussian coefficients.
# FAI equations use a Tweedie log link. Therefore all effects ending in FAI,
# including products of coefficients, are reported on the conditional
# log-expected-FAI scale. exp(effect) is also reported as a response ratio.
#
# Important interpretation
# ------------------------
# These are model-implied pathway effects under the locked observational
# causal hypothesis. They do not by themselves prove causality.
#
# No model is refitted in this step.
# =============================================================================

options(stringsAsFactors = FALSE, warn = 1, scipen = 999)

required_packages <- c(
  "data.table",
  "glmmTMB",
  "MASS",
  "Matrix",
  "ggplot2",
  "grid",
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
  library(ggplot2)
})

# -----------------------------------------------------------------------------
# Locked versions, paths, and settings
# -----------------------------------------------------------------------------

CODE_VERSION <- paste0(
  "2026-07-01_SEM_EFFECT_DECOMPOSITION_V1_",
  "THREE_SEASONS_PRIMARY_AND_EQUAL_FAI"
)

RELEASE_NOTE <- paste0(
  "2026-07-08_RELEASE_",
  "MC_NAMES_LIST_INDEX_AND_INDIRECT_COUNT"
)

EXPECTED_STEP04_VERSION <- paste0(
  "2026-07-01_SEM_THREE_SEASON_FAI_FITS_V1_",
  "NINE_PRIMARY_PLUS_THREE_SENSITIVITY_MODELS"
)

EXPECTED_STEP04_SCRIPT_SHA256 <- paste0(
  "a82e3e844d8439e6d2ff6e4495325d99",
  "a34794d96bdddef469095725399f2e94"
)

EXPECTED_STEP04A_VERSION <- paste0(
  "2026-07-01_SEM_MEDIATOR_OPTIMIZER_AGREEMENT_V1_",
  "SIX_GAUSSIAN_MODELS_NLMINB_VS_BFGS"
)

EXPECTED_STEP04A_SCRIPT_SHA256 <- paste0(
  "b6d15ed4f5b3ed015df8657263a2f4fb",
  "0c23a7423ac7ee26bd84f856b895046e"
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
STEP04_ROOT <- file.path(SEM_ROOT, "04_Seasonal_FAI_Piecewise_SEM")
STEP04A_ROOT <- file.path(SEM_ROOT, "04A_Mediator_Optimizer_Agreement")

STEP04_SCRIPT_FILE <- file.path(
  SCRIPT_DIR,
  "04_fit_three_seasonal_FAI_SEMs.R"
)
STEP04A_SCRIPT_FILE <- file.path(
  SCRIPT_DIR,
  "04A_check_mediator_optimizer_agreement.R"
)

STEP04_METHOD_FILE <- file.path(STEP04_ROOT, "00_Method_Definition.json")
STEP04_AUDIT_FILE <- file.path(
  STEP04_ROOT,
  "01_Input_Integrity_Audit.csv"
)
STEP04_EXOGENOUS_FILE <- file.path(
  STEP04_ROOT,
  "05_Topography_Anthropogenic_Correlation.csv"
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

STEP04A_METHOD_FILE <- file.path(
  STEP04A_ROOT,
  "00_Method_Definition.json"
)
STEP04A_AUDIT_FILE <- file.path(
  STEP04A_ROOT,
  "01_Input_Integrity_Audit.csv"
)
STEP04A_DECISION_FILE <- file.path(
  STEP04A_ROOT,
  "09_Final_Agreement_Decision.json"
)
STEP04A_MANIFEST_FILE <- file.path(
  STEP04A_ROOT,
  "10_Output_Manifest.csv"
)

OUTPUT_ROOT <- file.path(
  SEM_ROOT,
  "05_SEM_Effect_Decomposition_and_Path_Diagrams"
)
FIGURE_DIR <- file.path(OUTPUT_ROOT, "Figures")

METHOD_FILE <- file.path(OUTPUT_ROOT, "00_Method_Definition.json")
INPUT_AUDIT_FILE <- file.path(
  OUTPUT_ROOT,
  "01_Input_Integrity_Audit.csv"
)
EFFECT_DEFINITION_FILE <- file.path(
  OUTPUT_ROOT,
  "02_Locked_Effect_Definitions.csv"
)
DIRECT_EFFECT_FILE <- file.path(
  OUTPUT_ROOT,
  "03_Primary_Direct_Path_Effects.csv"
)
INDIRECT_EFFECT_FILE <- file.path(
  OUTPUT_ROOT,
  "04_Primary_Indirect_Effects_to_FAI.csv"
)
TOTAL_EFFECT_FILE <- file.path(
  OUTPUT_ROOT,
  "05_Primary_Total_Effects_to_FAI.csv"
)
ALL_PRIMARY_EFFECT_FILE <- file.path(
  OUTPUT_ROOT,
  "06_Primary_All_Effects_to_FAI.csv"
)
SENSITIVITY_EFFECT_FILE <- file.path(
  OUTPUT_ROOT,
  "07_Equal_FAI_All_Effects_to_FAI.csv"
)
SENSITIVITY_COMPARISON_FILE <- file.path(
  OUTPUT_ROOT,
  "08_Primary_vs_Equal_FAI_Total_Effect_Comparison.csv"
)
EDGE_TABLE_FILE <- file.path(
  OUTPUT_ROOT,
  "09_Path_Diagram_Edge_Table.csv"
)
FIGURE_METADATA_FILE <- file.path(
  OUTPUT_ROOT,
  "10_Figure_Metadata.csv"
)
MONTE_CARLO_QA_FILE <- file.path(
  OUTPUT_ROOT,
  "11_Monte_Carlo_QA.csv"
)
SEASONAL_SUMMARY_FILE <- file.path(
  OUTPUT_ROOT,
  "12_Seasonal_Effect_Summary.csv"
)
CAPTION_FILE <- file.path(
  OUTPUT_ROOT,
  "13_Figure_Caption.txt"
)
MANIFEST_FILE <- file.path(
  OUTPUT_ROOT,
  "14_Output_Manifest.csv"
)
SOFTWARE_FILE <- file.path(
  OUTPUT_ROOT,
  "Software_Environment.json"
)
LOG_FILE <- file.path(
  OUTPUT_ROOT,
  "sem_step05_effect_decomposition.log"
)

EXPECTED_PRIMARY_MODEL_N <- 9L
EXPECTED_SENSITIVITY_MODEL_N <- 3L
EXPECTED_TOTAL_MODEL_N <- 12L
EXPECTED_SEASON_N <- 3L
EXPECTED_DIRECT_PATH_N <- 24L
EXPECTED_SPECIFIC_INDIRECT_N <- 15L
EXPECTED_TOTAL_EFFECT_N <- 12L

N_SIM <- suppressWarnings(
  as.integer(
    Sys.getenv("SEM_STEP05_N_SIM", unset = "50000")
  )
)

if (!is.finite(N_SIM) || N_SIM < 10000L) {
  stop("SEM_STEP05_N_SIM must be an integer >= 10000.", call. = FALSE)
}

BASE_SEED <- 2026070100L

SEASON_LABELS <- c(
  `1` = "Spring",
  `2` = "Summer",
  `3` = "Autumn"
)

PRIMARY_RESPONSE <- "FAI_Q95"
SENSITIVITY_RESPONSE <- "FAI_EqualContribution_Sensitivity"

FONT_FAMILY <- "Times New Roman"

FORCE_RESTART <- identical(
  Sys.getenv("SEM_STEP05_FORCE_RESTART", unset = "0"),
  "1"
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
      "05_decompose_SEM_effects_and_draw_path_diagrams.R"
    ),
    winslash = "/",
    mustWork = FALSE
  )
}

formula_text <- function(formula_object) {
  paste(deparse(formula_object), collapse = " ")
}

significance_symbol <- function(p_value) {
  if (is.na(p_value)) return("")
  if (p_value < 0.001) return("***")
  if (p_value < 0.01) return("**")
  if (p_value < 0.05) return("*")
  "ns"
}

simulation_p_value <- function(draws) {
  draws <- as.numeric(draws)
  draws <- draws[is.finite(draws)]

  if (length(draws) == 0L) {
    return(NA_real_)
  }

  lower_probability <- (
    sum(draws <= 0) + 1
  ) / (
    length(draws) + 1
  )

  upper_probability <- (
    sum(draws >= 0) + 1
  ) / (
    length(draws) + 1
  )

  min(1, 2 * min(lower_probability, upper_probability))
}

summarize_draws <- function(
  draws,
  point_estimate,
  season,
  season_label,
  response,
  predictor,
  effect_type,
  path_name,
  path_formula,
  scale_label
) {
  draws <- as.numeric(draws)

  lower <- as.numeric(
    stats::quantile(draws, 0.025, names = FALSE, type = 7)
  )
  upper <- as.numeric(
    stats::quantile(draws, 0.975, names = FALSE, type = 7)
  )
  p_mc <- simulation_p_value(draws)

  ends_in_fai <- identical(response, "FAI")

  data.table(
    Season = season,
    Season_Label = season_label,
    Response = response,
    Predictor = predictor,
    Effect_Type = effect_type,
    Path_Name = path_name,
    Path_Formula = path_formula,
    Effect_Scale = scale_label,
    Estimate = point_estimate,
    MC_Median = stats::median(draws),
    MC_SD = stats::sd(draws),
    Lower_95 = lower,
    Upper_95 = upper,
    MC_P_Value = p_mc,
    CI_Excludes_Zero = (
      lower > 0 ||
        upper < 0
    ),
    Response_Ratio = if (ends_in_fai) {
      exp(point_estimate)
    } else {
      NA_real_
    },
    Percent_Change = if (ends_in_fai) {
      100 * (exp(point_estimate) - 1)
    } else {
      NA_real_
    },
    Lower_Response_Ratio = if (ends_in_fai) {
      exp(lower)
    } else {
      NA_real_
    },
    Upper_Response_Ratio = if (ends_in_fai) {
      exp(upper)
    } else {
      NA_real_
    }
  )
}

safe_mvrnorm <- function(
  n,
  mu,
  covariance_matrix,
  seed
) {
  mu_names <- names(mu)
  mu <- as.numeric(mu)
  names(mu) <- mu_names

  covariance_matrix <- as.matrix(covariance_matrix)
  covariance_matrix <- (
    covariance_matrix +
      t(covariance_matrix)
  ) / 2

  if (
    any(!is.finite(mu)) ||
      any(!is.finite(covariance_matrix))
  ) {
    stop("Non-finite coefficient mean or covariance matrix.", call. = FALSE)
  }

  set.seed(seed)

  draws <- tryCatch(
    MASS::mvrnorm(
      n = n,
      mu = mu,
      Sigma = covariance_matrix,
      tol = 1e-10,
      empirical = FALSE
    ),
    error = function(e) NULL
  )

  if (is.null(draws)) {
    adjusted <- as.matrix(
      Matrix::nearPD(
        covariance_matrix,
        corr = FALSE,
        keepDiag = TRUE
      )$mat
    )

    set.seed(seed)

    draws <- MASS::mvrnorm(
      n = n,
      mu = mu,
      Sigma = adjusted,
      tol = 1e-10,
      empirical = FALSE
    )
  }

  if (is.null(dim(draws))) {
    draws <- matrix(draws, ncol = length(mu))
  }

  colnames(draws) <- names(mu)
  draws
}

draw_fixed_effects <- function(fit, n, seed) {
  coefficient_vector <- glmmTMB::fixef(fit)$cond
  covariance_matrix <- as.matrix(vcov(fit)$cond)

  common_names <- intersect(
    names(coefficient_vector),
    colnames(covariance_matrix)
  )

  coefficient_vector <- coefficient_vector[common_names]
  covariance_matrix <- covariance_matrix[
    common_names,
    common_names,
    drop = FALSE
  ]

  safe_mvrnorm(
    n = n,
    mu = coefficient_vector,
    covariance_matrix = covariance_matrix,
    seed = seed
  )
}

extract_coefficient <- function(fit, term) {
  coefficient_vector <- glmmTMB::fixef(fit)$cond

  if (!term %in% names(coefficient_vector)) {
    stop(
      paste(
        "Required coefficient not found:",
        term
      ),
      call. = FALSE
    )
  }

  as.numeric(coefficient_vector[[term]])
}

extract_direct_table <- function(
  fit,
  season,
  season_label,
  model_role,
  response_label,
  term_map
) {
  coefficient_matrix <- summary(fit)$coefficients$cond

  rows <- list()

  for (term in names(term_map)) {
    if (!term %in% rownames(coefficient_matrix)) {
      stop(
        paste(
          "Missing direct-effect term:",
          term,
          "in",
          model_role,
          season_label
        ),
        call. = FALSE
      )
    }

    rows[[length(rows) + 1L]] <- data.table(
      Season = season,
      Season_Label = season_label,
      Model_Role = model_role,
      From = term_map[[term]],
      To = response_label,
      Term = term,
      Estimate = coefficient_matrix[term, "Estimate"],
      Std_Error = coefficient_matrix[term, "Std. Error"],
      Test_Statistic = coefficient_matrix[term, "z value"],
      P_Value = coefficient_matrix[term, "Pr(>|z|)"],
      Significance = significance_symbol(
        coefficient_matrix[term, "Pr(>|z|)"]
      ),
      Effect_Scale = if (response_label == "FAI") {
        "Conditional log expected FAI"
      } else {
        "Standardized Gaussian coefficient"
      },
      Response_Ratio = if (response_label == "FAI") {
        exp(coefficient_matrix[term, "Estimate"])
      } else {
        NA_real_
      },
      Percent_Change = if (response_label == "FAI") {
        100 * (
          exp(coefficient_matrix[term, "Estimate"]) -
            1
        )
      } else {
        NA_real_
      }
    )
  }

  rbindlist(rows, fill = TRUE)
}

verify_manifest <- function(root, manifest_path) {
  manifest <- fread(manifest_path)
  rows <- list()

  for (i in seq_len(nrow(manifest))) {
    row <- manifest[i]

    path <- file.path(
      root,
      gsub("\\\\", "/", row$Relative_Path)
    )

    exists <- file.exists(path)
    actual_hash <- if (exists) sha256_file(path) else ""
    actual_size <- if (exists) file.info(path)$size else -1

    rows[[length(rows) + 1L]] <- data.table(
      Relative_Path = row$Relative_Path,
      Exists = exists,
      Hash_Matches = (
        exists &&
          identical(actual_hash, row$SHA256)
      ),
      Size_Matches = (
        exists &&
          identical(
            as.numeric(actual_size),
            as.numeric(row$File_Size_Bytes)
          )
      )
    )
  }

  rbindlist(rows, fill = TRUE)
}

# -----------------------------------------------------------------------------
# Figure helpers
# -----------------------------------------------------------------------------

node_table <- data.table(
  Node = c(
    "Topography",
    "Anthropogenic",
    "Meteorology",
    "Vegetation",
    "FAI"
  ),
  Label = c(
    "Topography",
    "Anthropogenic\nactivity",
    "Meteorology",
    "Vegetation",
    "Forest fire\nactivity (FAI)"
  ),
  x = c(0.10, 0.10, 0.50, 0.50, 0.90),
  y = c(0.78, 0.24, 0.82, 0.28, 0.55)
)

edge_geometry <- data.table(
  From = c(
    "Topography",
    "Topography",
    "Meteorology",
    "Anthropogenic",
    "Meteorology",
    "Vegetation",
    "Topography",
    "Anthropogenic"
  ),
  To = c(
    "Meteorology",
    "Vegetation",
    "Vegetation",
    "Vegetation",
    "FAI",
    "FAI",
    "FAI",
    "FAI"
  ),
  label_x = c(
    0.30,
    0.31,
    0.54,
    0.30,
    0.70,
    0.70,
    0.48,
    0.49
  ),
  label_y = c(
    0.83,
    0.49,
    0.55,
    0.24,
    0.76,
    0.36,
    0.69,
    0.39
  )
)

make_path_figure <- function(
  edge_data,
  exogenous_rho,
  season_label,
  output_png,
  output_pdf
) {
  plot_edges <- merge(
    edge_data,
    edge_geometry,
    by = c("From", "To"),
    all.x = TRUE
  )

  plot_edges <- merge(
    plot_edges,
    node_table[
      ,
      .(
        From = Node,
        x_from = x,
        y_from = y
      )
    ],
    by = "From",
    all.x = TRUE
  )

  plot_edges <- merge(
    plot_edges,
    node_table[
      ,
      .(
        To = Node,
        x_to = x,
        y_to = y
      )
    ],
    by = "To",
    all.x = TRUE
  )

  plot_edges[
    ,
    `:=`(
      Edge_Label = paste0(
        sprintf("%.3f", Estimate),
        Significance
      ),
      Edge_Colour = ifelse(
        Estimate >= 0,
        "Positive",
        "Negative"
      ),
      Edge_Linetype = ifelse(
        P_Value < 0.05,
        "Significant",
        "Not significant"
      )
    )
  ]

  plot_object <- ggplot() +
    geom_segment(
      data = plot_edges,
      aes(
        x = x_from,
        y = y_from,
        xend = x_to,
        yend = y_to,
        colour = Edge_Colour,
        linetype = Edge_Linetype
      ),
      linewidth = 0.75,
      arrow = grid::arrow(
        length = grid::unit(0.18, "cm"),
        type = "closed"
      ),
      lineend = "round"
    ) +
    geom_curve(
      aes(
        x = 0.12,
        y = 0.70,
        xend = 0.12,
        yend = 0.32
      ),
      curvature = 0.65,
      linewidth = 0.65,
      linetype = "dotted",
      colour = "black",
      arrow = grid::arrow(
        ends = "both",
        length = grid::unit(0.15, "cm"),
        type = "open"
      )
    ) +
    annotate(
      "label",
      x = 0.055,
      y = 0.51,
      label = paste0(
        "rho = ",
        sprintf("%.3f", exogenous_rho)
      ),
      family = FONT_FAMILY,
      size = 3.2,
      label.size = 0.20,
      fill = "white"
    ) +
    geom_label(
      data = plot_edges,
      aes(
        x = label_x,
        y = label_y,
        label = Edge_Label
      ),
      family = FONT_FAMILY,
      size = 3.35,
      label.size = 0.18,
      fill = "white",
      label.padding = grid::unit(0.10, "lines")
    ) +
    geom_label(
      data = node_table,
      aes(
        x = x,
        y = y,
        label = Label
      ),
      family = FONT_FAMILY,
      size = 4.0,
      fontface = "bold",
      label.size = 0.45,
      label.padding = grid::unit(0.28, "lines"),
      fill = "white"
    ) +
    annotate(
      "text",
      x = 0.50,
      y = 0.04,
      label = paste(
        "Solid: p < 0.05; dashed: p >= 0.05.",
        "FAI paths are log-link coefficients."
      ),
      family = FONT_FAMILY,
      size = 3.0
    ) +
    scale_colour_manual(
      values = c(
        Positive = "black",
        Negative = "grey40"
      ),
      guide = "none"
    ) +
    scale_linetype_manual(
      values = c(
        Significant = "solid",
        `Not significant` = "dashed"
      ),
      guide = "none"
    ) +
    coord_cartesian(
      xlim = c(0, 1),
      ylim = c(0, 1),
      clip = "off"
    ) +
    labs(
      title = paste0(
        season_label,
        " seasonal SEM"
      )
    ) +
    theme_void(base_family = FONT_FAMILY) +
    theme(
      plot.title = element_text(
        family = FONT_FAMILY,
        face = "bold",
        size = 15,
        hjust = 0.5,
        margin = margin(b = 8)
      ),
      plot.margin = margin(8, 12, 8, 12)
    )

  ggsave(
    filename = output_png,
    plot = plot_object,
    width = 9.0,
    height = 6.2,
    units = "in",
    dpi = 600,
    bg = "white"
  )

  if (capabilities("cairo")) {
    ggsave(
      filename = output_pdf,
      plot = plot_object,
      width = 9.0,
      height = 6.2,
      units = "in",
      device = grDevices::cairo_pdf,
      family = FONT_FAMILY,
      bg = "white"
    )
  } else {
    ggsave(
      filename = output_pdf,
      plot = plot_object,
      width = 9.0,
      height = 6.2,
      units = "in",
      device = "pdf",
      bg = "white"
    )
  }

  plot_object
}

# -----------------------------------------------------------------------------
# Initialize output directory
# -----------------------------------------------------------------------------

if (FORCE_RESTART && dir.exists(OUTPUT_ROOT)) {
  unlink(OUTPUT_ROOT, recursive = TRUE, force = TRUE)
}

if (dir.exists(OUTPUT_ROOT)) {
  stop(
    paste0(
      "Output directory already exists:\n",
      OUTPUT_ROOT,
      "\nDelete or rename it before an intentional full rerun, or set ",
      "SEM_STEP05_FORCE_RESTART=1."
    ),
    call. = FALSE
  )
}

dir.create(OUTPUT_ROOT, recursive = TRUE, showWarnings = FALSE)
dir.create(FIGURE_DIR, recursive = TRUE, showWarnings = FALSE)
writeLines(character(), LOG_FILE)

if (.Platform$OS.type == "windows") {
  try(
    grDevices::windowsFonts(
      TimesNewRoman = grDevices::windowsFont(FONT_FAMILY)
    ),
    silent = TRUE
  )
}

log_message("Starting Step 05 seasonal SEM effect decomposition.")
log_message(paste("Code version:", CODE_VERSION))
log_message(paste("Release note:", RELEASE_NOTE))
log_message(paste("Monte Carlo draws per model:", N_SIM))

# -----------------------------------------------------------------------------
# Verify sealed Step 04 and Step 04A
# -----------------------------------------------------------------------------

required_files <- c(
  STEP04_SCRIPT_FILE,
  STEP04A_SCRIPT_FILE,
  STEP04_METHOD_FILE,
  STEP04_AUDIT_FILE,
  STEP04_EXOGENOUS_FILE,
  STEP04_REGISTRY_FILE,
  STEP04_MANIFEST_FILE,
  STEP04A_METHOD_FILE,
  STEP04A_AUDIT_FILE,
  STEP04A_DECISION_FILE,
  STEP04A_MANIFEST_FILE
)

missing_files <- required_files[!file.exists(required_files)]

if (length(missing_files) > 0L) {
  stop(
    paste(
      "Missing required sealed inputs:",
      paste(missing_files, collapse = "\n"),
      sep = "\n"
    ),
    call. = FALSE
  )
}

audit <- list()

step04_method <- read_json(STEP04_METHOD_FILE)
step04a_method <- read_json(STEP04A_METHOD_FILE)
step04a_decision <- read_json(STEP04A_DECISION_FILE)

audit <- add_check(
  audit,
  "Version",
  "Step04_code_version",
  identical(step04_method$Code_Version, EXPECTED_STEP04_VERSION),
  step04_method$Code_Version,
  EXPECTED_STEP04_VERSION
)

audit <- add_check(
  audit,
  "Hash",
  "Step04_script_hash",
  identical(
    sha256_file(STEP04_SCRIPT_FILE),
    EXPECTED_STEP04_SCRIPT_SHA256
  ),
  sha256_file(STEP04_SCRIPT_FILE),
  EXPECTED_STEP04_SCRIPT_SHA256
)

audit <- add_check(
  audit,
  "Version",
  "Step04A_code_version",
  identical(step04a_method$Code_Version, EXPECTED_STEP04A_VERSION),
  step04a_method$Code_Version,
  EXPECTED_STEP04A_VERSION
)

audit <- add_check(
  audit,
  "Hash",
  "Step04A_script_hash",
  identical(
    sha256_file(STEP04A_SCRIPT_FILE),
    EXPECTED_STEP04A_SCRIPT_SHA256
  ),
  sha256_file(STEP04A_SCRIPT_FILE),
  EXPECTED_STEP04A_SCRIPT_SHA256
)

audit <- add_check(
  audit,
  "Decision",
  "Step04A_ready_for_effect_decomposition",
  isTRUE(step04a_decision$Optimizer_Agreement_Passed) &&
    isTRUE(step04a_decision$Ready_For_Final_Effect_Decomposition),
  paste(
    step04a_decision$Optimizer_Agreement_Passed,
    step04a_decision$Ready_For_Final_Effect_Decomposition,
    sep = ";"
  ),
  "TRUE;TRUE"
)

step04_audit <- fread(STEP04_AUDIT_FILE)
step04a_audit <- fread(STEP04A_AUDIT_FILE)

audit <- add_check(
  audit,
  "Audit",
  "Step04_all_ERROR_checks_pass",
  all(
    vapply(
      step04_audit[Severity == "ERROR", Passed],
      parse_bool,
      logical(1)
    )
  ),
  sum(
    vapply(
      step04_audit[Severity == "ERROR", Passed],
      parse_bool,
      logical(1)
    )
  ),
  nrow(step04_audit[Severity == "ERROR"])
)

audit <- add_check(
  audit,
  "Audit",
  "Step04A_all_ERROR_checks_pass",
  all(
    vapply(
      step04a_audit[Severity == "ERROR", Passed],
      parse_bool,
      logical(1)
    )
  ),
  sum(
    vapply(
      step04a_audit[Severity == "ERROR", Passed],
      parse_bool,
      logical(1)
    )
  ),
  nrow(step04a_audit[Severity == "ERROR"])
)

step04_manifest_status <- verify_manifest(
  STEP04_ROOT,
  STEP04_MANIFEST_FILE
)

step04a_manifest_status <- verify_manifest(
  STEP04A_ROOT,
  STEP04A_MANIFEST_FILE
)

audit <- add_check(
  audit,
  "Manifest",
  "Step04_manifest_verified",
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

audit <- add_check(
  audit,
  "Manifest",
  "Step04A_manifest_verified",
  all(
    step04a_manifest_status$Exists &
      step04a_manifest_status$Hash_Matches &
      step04a_manifest_status$Size_Matches
  ),
  sum(
    step04a_manifest_status$Exists &
      step04a_manifest_status$Hash_Matches &
      step04a_manifest_status$Size_Matches
  ),
  nrow(step04a_manifest_status)
)

registry <- fread(STEP04_REGISTRY_FILE)

audit <- add_check(
  audit,
  "Models",
  "Twelve_models_successful",
  nrow(registry) == EXPECTED_TOTAL_MODEL_N &&
    all(registry$Status == "SUCCESS"),
  paste(
    nrow(registry),
    sum(registry$Status == "SUCCESS"),
    sep = ";"
  ),
  paste(
    EXPECTED_TOTAL_MODEL_N,
    EXPECTED_TOTAL_MODEL_N,
    sep = ";"
  )
)

audit <- add_check(
  audit,
  "Models",
  "Nine_primary_three_sensitivity",
  sum(vapply(
    registry$Primary_SEM_Model,
    parse_bool,
    logical(1)
  )) == EXPECTED_PRIMARY_MODEL_N &&
    sum(!vapply(
      registry$Primary_SEM_Model,
      parse_bool,
      logical(1)
    )) == EXPECTED_SENSITIVITY_MODEL_N,
  paste(
    sum(vapply(
      registry$Primary_SEM_Model,
      parse_bool,
      logical(1)
    )),
    sum(!vapply(
      registry$Primary_SEM_Model,
      parse_bool,
      logical(1)
    )),
    sep = ";"
  ),
  paste(
    EXPECTED_PRIMARY_MODEL_N,
    EXPECTED_SENSITIVITY_MODEL_N,
    sep = ";"
  )
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
    "Step 05 input audit failed. See 01_Input_Integrity_Audit.csv.",
    call. = FALSE
  )
}

# -----------------------------------------------------------------------------
# Lock effect definitions
# -----------------------------------------------------------------------------

effect_definitions <- data.table(
  Predictor = c(
    "Topography",
    "Topography",
    "Topography",
    "Topography",
    "Topography",
    "Anthropogenic",
    "Anthropogenic",
    "Anthropogenic",
    "Meteorology",
    "Meteorology",
    "Meteorology",
    "Vegetation"
  ),
  Effect_Type = c(
    "Direct",
    "Specific indirect",
    "Specific indirect",
    "Specific indirect",
    "Total",
    "Direct",
    "Specific indirect",
    "Total",
    "Direct",
    "Specific indirect",
    "Total",
    "Total"
  ),
  Path_Name = c(
    "Topography_direct_to_FAI",
    "Topography_via_Meteorology",
    "Topography_via_Vegetation",
    "Topography_via_Meteorology_and_Vegetation",
    "Topography_total",
    "Anthropogenic_direct_to_FAI",
    "Anthropogenic_via_Vegetation",
    "Anthropogenic_total",
    "Meteorology_direct_to_FAI",
    "Meteorology_via_Vegetation",
    "Meteorology_total",
    "Vegetation_total"
  ),
  Path_Formula = c(
    "TF",
    "TM * MF",
    "TV * VF",
    "TM * MV * VF",
    "TF + TM*MF + TV*VF + TM*MV*VF",
    "AF",
    "AV * VF",
    "AF + AV*VF",
    "MF",
    "MV * VF",
    "MF + MV*VF",
    "VF"
  ),
  Scale = "Conditional log expected FAI",
  Included_In_Total_Indirect = c(
    FALSE,
    TRUE,
    TRUE,
    TRUE,
    FALSE,
    FALSE,
    TRUE,
    FALSE,
    FALSE,
    TRUE,
    FALSE,
    FALSE
  )
)

fwrite(effect_definitions, EFFECT_DEFINITION_FILE)

# -----------------------------------------------------------------------------
# Load seasonal models and calculate effects
# -----------------------------------------------------------------------------

exogenous_correlation <- fread(STEP04_EXOGENOUS_FILE)

direct_rows <- list()
primary_indirect_rows <- list()
primary_total_rows <- list()
primary_all_rows <- list()
sensitivity_all_rows <- list()
sensitivity_comparison_rows <- list()
edge_rows <- list()
mc_qa_rows <- list()
season_summary_rows <- list()
figure_metadata_rows <- list()

for (season in 1:3) {
  season_label <- unname(SEASON_LABELS[as.character(season)])
  season_prefix <- paste0("S", season, "_", season_label)

  log_message(
    paste(
      "Calculating effects for",
      season_label
    )
  )

  model_ids <- list(
    meteorology = paste0(
      season_prefix,
      "_Meteorology_Mediator"
    ),
    vegetation = paste0(
      season_prefix,
      "_Vegetation_Mediator"
    ),
    primary_fai = paste0(
      season_prefix,
      "_FAI_Primary"
    ),
    sensitivity_fai = paste0(
      season_prefix,
      "_FAI_Equal_Sensitivity"
    )
  )

  model_paths <- lapply(
    model_ids,
    function(model_id) {
      file.path(
        STEP04_MODEL_DIR,
        paste0(model_id, ".rds")
      )
    }
  )

  missing_model_paths <- unlist(model_paths)[
    !file.exists(unlist(model_paths))
  ]

  if (length(missing_model_paths) > 0L) {
    stop(
      paste(
        "Missing seasonal model files:",
        paste(missing_model_paths, collapse = "\n"),
        sep = "\n"
      ),
      call. = FALSE
    )
  }

  meteorology_record <- readRDS(model_paths$meteorology)
  vegetation_record <- readRDS(model_paths$vegetation)
  primary_fai_record <- readRDS(model_paths$primary_fai)
  sensitivity_fai_record <- readRDS(model_paths$sensitivity_fai)

  meteorology_fit <- meteorology_record$fit
  vegetation_fit <- vegetation_record$fit
  primary_fai_fit <- primary_fai_record$fit
  sensitivity_fai_fit <- sensitivity_fai_record$fit

  direct_rows[[length(direct_rows) + 1L]] <- extract_direct_table(
    fit = meteorology_fit,
    season = season,
    season_label = season_label,
    model_role = "Meteorology_mediator",
    response_label = "Meteorology",
    term_map = c(
      Topography_Score = "Topography"
    )
  )

  direct_rows[[length(direct_rows) + 1L]] <- extract_direct_table(
    fit = vegetation_fit,
    season = season,
    season_label = season_label,
    model_role = "Vegetation_mediator",
    response_label = "Vegetation",
    term_map = c(
      Meteorology_Score = "Meteorology",
      Topography_Score = "Topography",
      Anthropogenic_Score = "Anthropogenic"
    )
  )

  direct_rows[[length(direct_rows) + 1L]] <- extract_direct_table(
    fit = primary_fai_fit,
    season = season,
    season_label = season_label,
    model_role = "FAI_primary_outcome",
    response_label = "FAI",
    term_map = c(
      Meteorology_Score = "Meteorology",
      Vegetation_Score = "Vegetation",
      Topography_Score = "Topography",
      Anthropogenic_Score = "Anthropogenic"
    )
  )

  # Locked point coefficients
  TM <- extract_coefficient(
    meteorology_fit,
    "Topography_Score"
  )

  MV <- extract_coefficient(
    vegetation_fit,
    "Meteorology_Score"
  )

  TV <- extract_coefficient(
    vegetation_fit,
    "Topography_Score"
  )

  AV <- extract_coefficient(
    vegetation_fit,
    "Anthropogenic_Score"
  )

  MF_primary <- extract_coefficient(
    primary_fai_fit,
    "Meteorology_Score"
  )

  VF_primary <- extract_coefficient(
    primary_fai_fit,
    "Vegetation_Score"
  )

  TF_primary <- extract_coefficient(
    primary_fai_fit,
    "Topography_Score"
  )

  AF_primary <- extract_coefficient(
    primary_fai_fit,
    "Anthropogenic_Score"
  )

  MF_sensitivity <- extract_coefficient(
    sensitivity_fai_fit,
    "Meteorology_Score"
  )

  VF_sensitivity <- extract_coefficient(
    sensitivity_fai_fit,
    "Vegetation_Score"
  )

  TF_sensitivity <- extract_coefficient(
    sensitivity_fai_fit,
    "Topography_Score"
  )

  AF_sensitivity <- extract_coefficient(
    sensitivity_fai_fit,
    "Anthropogenic_Score"
  )

  # Monte Carlo draws. Mediator draws are common to primary and sensitivity.
  meteorology_draws <- draw_fixed_effects(
    meteorology_fit,
    n = N_SIM,
    seed = BASE_SEED + season * 10L + 1L
  )

  vegetation_draws <- draw_fixed_effects(
    vegetation_fit,
    n = N_SIM,
    seed = BASE_SEED + season * 10L + 2L
  )

  primary_fai_draws <- draw_fixed_effects(
    primary_fai_fit,
    n = N_SIM,
    seed = BASE_SEED + season * 10L + 3L
  )

  sensitivity_fai_draws <- draw_fixed_effects(
    sensitivity_fai_fit,
    n = N_SIM,
    seed = BASE_SEED + season * 10L + 4L
  )

  required_draw_columns <- list(
    meteorology = c("Topography_Score"),
    vegetation = c(
      "Meteorology_Score",
      "Topography_Score",
      "Anthropogenic_Score"
    ),
    primary_fai = c(
      "Meteorology_Score",
      "Vegetation_Score",
      "Topography_Score",
      "Anthropogenic_Score"
    ),
    sensitivity_fai = c(
      "Meteorology_Score",
      "Vegetation_Score",
      "Topography_Score",
      "Anthropogenic_Score"
    )
  )

  draw_matrices <- list(
    meteorology = meteorology_draws,
    vegetation = vegetation_draws,
    primary_fai = primary_fai_draws,
    sensitivity_fai = sensitivity_fai_draws
  )

  for (draw_name in names(draw_matrices)) {
    missing_draw_columns <- setdiff(
      required_draw_columns[[draw_name]],
      colnames(draw_matrices[[draw_name]])
    )

    if (length(missing_draw_columns) > 0L) {
      stop(
        paste(
          "Missing Monte Carlo coefficient columns:",
          draw_name,
          paste(missing_draw_columns, collapse = ", ")
        ),
        call. = FALSE
      )
    }
  }

  TM_d <- meteorology_draws[, "Topography_Score"]

  MV_d <- vegetation_draws[, "Meteorology_Score"]
  TV_d <- vegetation_draws[, "Topography_Score"]
  AV_d <- vegetation_draws[, "Anthropogenic_Score"]

  MFp_d <- primary_fai_draws[, "Meteorology_Score"]
  VFp_d <- primary_fai_draws[, "Vegetation_Score"]
  TFp_d <- primary_fai_draws[, "Topography_Score"]
  AFp_d <- primary_fai_draws[, "Anthropogenic_Score"]

  MFs_d <- sensitivity_fai_draws[, "Meteorology_Score"]
  VFs_d <- sensitivity_fai_draws[, "Vegetation_Score"]
  TFs_d <- sensitivity_fai_draws[, "Topography_Score"]
  AFs_d <- sensitivity_fai_draws[, "Anthropogenic_Score"]

  # Primary specific indirect effects
  primary_specific <- list(
    Topography_via_Meteorology = list(
      predictor = "Topography",
      formula = "TM * MF",
      estimate = TM * MF_primary,
      draws = TM_d * MFp_d
    ),
    Topography_via_Vegetation = list(
      predictor = "Topography",
      formula = "TV * VF",
      estimate = TV * VF_primary,
      draws = TV_d * VFp_d
    ),
    Topography_via_Meteorology_and_Vegetation = list(
      predictor = "Topography",
      formula = "TM * MV * VF",
      estimate = TM * MV * VF_primary,
      draws = TM_d * MV_d * VFp_d
    ),
    Anthropogenic_via_Vegetation = list(
      predictor = "Anthropogenic",
      formula = "AV * VF",
      estimate = AV * VF_primary,
      draws = AV_d * VFp_d
    ),
    Meteorology_via_Vegetation = list(
      predictor = "Meteorology",
      formula = "MV * VF",
      estimate = MV * VF_primary,
      draws = MV_d * VFp_d
    ),
    Vegetation_direct_to_FAI = list(
      predictor = "Vegetation",
      formula = "VF",
      estimate = VF_primary,
      draws = VFp_d
    )
  )

  for (effect_name in names(primary_specific)) {
    effect <- primary_specific[[effect_name]]

    effect_type <- if (
      effect_name == "Vegetation_direct_to_FAI"
    ) {
      "Direct"
    } else {
      "Specific indirect"
    }

    summarized <- summarize_draws(
      draws = effect$draws,
      point_estimate = effect$estimate,
      season = season,
      season_label = season_label,
      response = "FAI",
      predictor = effect$predictor,
      effect_type = effect_type,
      path_name = effect_name,
      path_formula = effect$formula,
      scale_label = "Conditional log expected FAI"
    )

    if (effect_type == "Specific indirect") {
      primary_indirect_rows[[length(primary_indirect_rows) + 1L]] <- summarized
    }

    primary_all_rows[[length(primary_all_rows) + 1L]] <- summarized
  }

  # Primary direct-to-FAI effects
  primary_direct_effects <- list(
    Topography_direct_to_FAI = list(
      predictor = "Topography",
      formula = "TF",
      estimate = TF_primary,
      draws = TFp_d
    ),
    Anthropogenic_direct_to_FAI = list(
      predictor = "Anthropogenic",
      formula = "AF",
      estimate = AF_primary,
      draws = AFp_d
    ),
    Meteorology_direct_to_FAI = list(
      predictor = "Meteorology",
      formula = "MF",
      estimate = MF_primary,
      draws = MFp_d
    )
  )

  for (effect_name in names(primary_direct_effects)) {
    effect <- primary_direct_effects[[effect_name]]

    summarized <- summarize_draws(
      draws = effect$draws,
      point_estimate = effect$estimate,
      season = season,
      season_label = season_label,
      response = "FAI",
      predictor = effect$predictor,
      effect_type = "Direct",
      path_name = effect_name,
      path_formula = effect$formula,
      scale_label = "Conditional log expected FAI"
    )

    primary_all_rows[[length(primary_all_rows) + 1L]] <- summarized
  }

  # Primary total indirect and total effects
  T_indirect_estimate <- (
    TM * MF_primary +
      TV * VF_primary +
      TM * MV * VF_primary
  )

  T_indirect_draws <- (
    TM_d * MFp_d +
      TV_d * VFp_d +
      TM_d * MV_d * VFp_d
  )

  A_indirect_estimate <- AV * VF_primary
  A_indirect_draws <- AV_d * VFp_d

  M_indirect_estimate <- MV * VF_primary
  M_indirect_draws <- MV_d * VFp_d

  primary_totals <- list(
    Topography_total_indirect = list(
      predictor = "Topography",
      type = "Total indirect",
      formula = "TM*MF + TV*VF + TM*MV*VF",
      estimate = T_indirect_estimate,
      draws = T_indirect_draws
    ),
    Topography_total = list(
      predictor = "Topography",
      type = "Total",
      formula = "TF + TM*MF + TV*VF + TM*MV*VF",
      estimate = TF_primary + T_indirect_estimate,
      draws = TFp_d + T_indirect_draws
    ),
    Anthropogenic_total_indirect = list(
      predictor = "Anthropogenic",
      type = "Total indirect",
      formula = "AV*VF",
      estimate = A_indirect_estimate,
      draws = A_indirect_draws
    ),
    Anthropogenic_total = list(
      predictor = "Anthropogenic",
      type = "Total",
      formula = "AF + AV*VF",
      estimate = AF_primary + A_indirect_estimate,
      draws = AFp_d + A_indirect_draws
    ),
    Meteorology_total_indirect = list(
      predictor = "Meteorology",
      type = "Total indirect",
      formula = "MV*VF",
      estimate = M_indirect_estimate,
      draws = M_indirect_draws
    ),
    Meteorology_total = list(
      predictor = "Meteorology",
      type = "Total",
      formula = "MF + MV*VF",
      estimate = MF_primary + M_indirect_estimate,
      draws = MFp_d + M_indirect_draws
    ),
    Vegetation_total = list(
      predictor = "Vegetation",
      type = "Total",
      formula = "VF",
      estimate = VF_primary,
      draws = VFp_d
    )
  )

  for (effect_name in names(primary_totals)) {
    effect <- primary_totals[[effect_name]]

    summarized <- summarize_draws(
      draws = effect$draws,
      point_estimate = effect$estimate,
      season = season,
      season_label = season_label,
      response = "FAI",
      predictor = effect$predictor,
      effect_type = effect$type,
      path_name = effect_name,
      path_formula = effect$formula,
      scale_label = "Conditional log expected FAI"
    )

    primary_total_rows[[length(primary_total_rows) + 1L]] <- summarized

    primary_all_rows[[length(primary_all_rows) + 1L]] <- summarized
  }

  # Sensitivity response: direct, indirect, and total effects
  sensitivity_specific <- list(
    Topography_direct_to_FAI = list(
      predictor = "Topography",
      type = "Direct",
      formula = "TF",
      estimate = TF_sensitivity,
      draws = TFs_d
    ),
    Topography_via_Meteorology = list(
      predictor = "Topography",
      type = "Specific indirect",
      formula = "TM * MF",
      estimate = TM * MF_sensitivity,
      draws = TM_d * MFs_d
    ),
    Topography_via_Vegetation = list(
      predictor = "Topography",
      type = "Specific indirect",
      formula = "TV * VF",
      estimate = TV * VF_sensitivity,
      draws = TV_d * VFs_d
    ),
    Topography_via_Meteorology_and_Vegetation = list(
      predictor = "Topography",
      type = "Specific indirect",
      formula = "TM * MV * VF",
      estimate = TM * MV * VF_sensitivity,
      draws = TM_d * MV_d * VFs_d
    ),
    Anthropogenic_direct_to_FAI = list(
      predictor = "Anthropogenic",
      type = "Direct",
      formula = "AF",
      estimate = AF_sensitivity,
      draws = AFs_d
    ),
    Anthropogenic_via_Vegetation = list(
      predictor = "Anthropogenic",
      type = "Specific indirect",
      formula = "AV * VF",
      estimate = AV * VF_sensitivity,
      draws = AV_d * VFs_d
    ),
    Meteorology_direct_to_FAI = list(
      predictor = "Meteorology",
      type = "Direct",
      formula = "MF",
      estimate = MF_sensitivity,
      draws = MFs_d
    ),
    Meteorology_via_Vegetation = list(
      predictor = "Meteorology",
      type = "Specific indirect",
      formula = "MV * VF",
      estimate = MV * VF_sensitivity,
      draws = MV_d * VFs_d
    ),
    Vegetation_total = list(
      predictor = "Vegetation",
      type = "Total",
      formula = "VF",
      estimate = VF_sensitivity,
      draws = VFs_d
    )
  )

  T_indirect_s_estimate <- (
    TM * MF_sensitivity +
      TV * VF_sensitivity +
      TM * MV * VF_sensitivity
  )

  T_indirect_s_draws <- (
    TM_d * MFs_d +
      TV_d * VFs_d +
      TM_d * MV_d * VFs_d
  )

  A_indirect_s_estimate <- AV * VF_sensitivity
  A_indirect_s_draws <- AV_d * VFs_d

  M_indirect_s_estimate <- MV * VF_sensitivity
  M_indirect_s_draws <- MV_d * VFs_d

  sensitivity_totals <- list(
    Topography_total_indirect = list(
      predictor = "Topography",
      type = "Total indirect",
      formula = "TM*MF + TV*VF + TM*MV*VF",
      estimate = T_indirect_s_estimate,
      draws = T_indirect_s_draws
    ),
    Topography_total = list(
      predictor = "Topography",
      type = "Total",
      formula = "TF + TM*MF + TV*VF + TM*MV*VF",
      estimate = TF_sensitivity + T_indirect_s_estimate,
      draws = TFs_d + T_indirect_s_draws
    ),
    Anthropogenic_total_indirect = list(
      predictor = "Anthropogenic",
      type = "Total indirect",
      formula = "AV*VF",
      estimate = A_indirect_s_estimate,
      draws = A_indirect_s_draws
    ),
    Anthropogenic_total = list(
      predictor = "Anthropogenic",
      type = "Total",
      formula = "AF + AV*VF",
      estimate = AF_sensitivity + A_indirect_s_estimate,
      draws = AFs_d + A_indirect_s_draws
    ),
    Meteorology_total_indirect = list(
      predictor = "Meteorology",
      type = "Total indirect",
      formula = "MV*VF",
      estimate = M_indirect_s_estimate,
      draws = M_indirect_s_draws
    ),
    Meteorology_total = list(
      predictor = "Meteorology",
      type = "Total",
      formula = "MF + MV*VF",
      estimate = MF_sensitivity + M_indirect_s_estimate,
      draws = MFs_d + M_indirect_s_draws
    )
  )

  sensitivity_effects_this_season <- list()

  for (effect_name in names(sensitivity_specific)) {
    effect <- sensitivity_specific[[effect_name]]

    sensitivity_effects_this_season[[length(sensitivity_effects_this_season) + 1L]] <- summarize_draws(
      draws = effect$draws,
      point_estimate = effect$estimate,
      season = season,
      season_label = season_label,
      response = "FAI",
      predictor = effect$predictor,
      effect_type = effect$type,
      path_name = effect_name,
      path_formula = effect$formula,
      scale_label = "Conditional log expected equal-contribution FAI"
    )
  }

  for (effect_name in names(sensitivity_totals)) {
    effect <- sensitivity_totals[[effect_name]]

    sensitivity_effects_this_season[[length(sensitivity_effects_this_season) + 1L]] <- summarize_draws(
      draws = effect$draws,
      point_estimate = effect$estimate,
      season = season,
      season_label = season_label,
      response = "FAI",
      predictor = effect$predictor,
      effect_type = effect$type,
      path_name = effect_name,
      path_formula = effect$formula,
      scale_label = "Conditional log expected equal-contribution FAI"
    )
  }

  sensitivity_season_table <- rbindlist(
    sensitivity_effects_this_season,
    fill = TRUE
  )

  sensitivity_all_rows[[length(sensitivity_all_rows) + 1L]] <- sensitivity_season_table

  # Primary vs sensitivity total-effect comparison
  primary_total_this_season <- rbindlist(
    primary_total_rows[
      vapply(
        primary_total_rows,
        function(x) {
          is.data.table(x) &&
            nrow(x) == 1L &&
            x$Season[[1L]] == season &&
            x$Effect_Type[[1L]] == "Total"
        },
        logical(1)
      )
    ],
    fill = TRUE
  )

  sensitivity_total_this_season <- sensitivity_season_table[
    Effect_Type == "Total"
  ]

  total_comparison <- merge(
    primary_total_this_season[
      ,
      .(
        Season,
        Season_Label,
        Predictor,
        Primary_Estimate = Estimate,
        Primary_Lower_95 = Lower_95,
        Primary_Upper_95 = Upper_95,
        Primary_CI_Excludes_Zero = CI_Excludes_Zero
      )
    ],
    sensitivity_total_this_season[
      ,
      .(
        Season,
        Season_Label,
        Predictor,
        Sensitivity_Estimate = Estimate,
        Sensitivity_Lower_95 = Lower_95,
        Sensitivity_Upper_95 = Upper_95,
        Sensitivity_CI_Excludes_Zero = CI_Excludes_Zero
      )
    ],
    by = c(
      "Season",
      "Season_Label",
      "Predictor"
    ),
    all = TRUE
  )

  total_comparison[
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
      CI_Exclusion_Class_Consistent = (
        Primary_CI_Excludes_Zero ==
          Sensitivity_CI_Excludes_Zero
      )
    )
  ]

  sensitivity_comparison_rows[[length(sensitivity_comparison_rows) + 1L]] <- total_comparison

  # Diagram edges from primary direct paths
  direct_season <- rbindlist(
    direct_rows[
      vapply(
        direct_rows,
        function(x) {
          is.data.table(x) &&
            nrow(x) > 0L &&
            all(x$Season == season)
        },
        logical(1)
      )
    ],
    fill = TRUE
  )

  edge_rows[[length(edge_rows) + 1L]] <- direct_season[
    ,
    .(
      Season,
      Season_Label,
      From,
      To,
      Estimate,
      Std_Error,
      P_Value,
      Significance,
      Effect_Scale
    )
  ]

  # Monte Carlo QA
  coefficient_point_table <- data.table(
    Coefficient = c(
      "TM",
      "MV",
      "TV",
      "AV",
      "MF_primary",
      "VF_primary",
      "TF_primary",
      "AF_primary",
      "MF_sensitivity",
      "VF_sensitivity",
      "TF_sensitivity",
      "AF_sensitivity"
    ),
    Point_Estimate = c(
      TM,
      MV,
      TV,
      AV,
      MF_primary,
      VF_primary,
      TF_primary,
      AF_primary,
      MF_sensitivity,
      VF_sensitivity,
      TF_sensitivity,
      AF_sensitivity
    ),
    MC_Mean = c(
      mean(TM_d),
      mean(MV_d),
      mean(TV_d),
      mean(AV_d),
      mean(MFp_d),
      mean(VFp_d),
      mean(TFp_d),
      mean(AFp_d),
      mean(MFs_d),
      mean(VFs_d),
      mean(TFs_d),
      mean(AFs_d)
    )
  )

  coefficient_point_table[
    ,
    `:=`(
      Season = season,
      Season_Label = season_label,
      Absolute_MC_Mean_Difference = abs(
        MC_Mean -
          Point_Estimate
      )
    )
  ]

  mc_qa_rows[[length(mc_qa_rows) + 1L]] <- coefficient_point_table

  # Seasonal summary
  primary_total_summary <- primary_total_this_season[
    ,
    .(
      Predictor,
      Estimate,
      Lower_95,
      Upper_95,
      Percent_Change,
      CI_Excludes_Zero
    )
  ]

  strongest_predictor <- primary_total_summary[
    which.max(abs(Estimate))
  ]

  season_summary_rows[[length(season_summary_rows) + 1L]] <- data.table(
    Season = season,
    Season_Label = season_label,
    Strongest_Total_Effect_Predictor = (
      strongest_predictor$Predictor
    ),
    Strongest_Total_Effect = strongest_predictor$Estimate,
    Strongest_Total_Effect_Lower_95 = (
      strongest_predictor$Lower_95
    ),
    Strongest_Total_Effect_Upper_95 = (
      strongest_predictor$Upper_95
    ),
    Strongest_Total_Effect_Percent_Change = (
      strongest_predictor$Percent_Change
    ),
    Primary_Total_Effects_Excluding_Zero = sum(
      primary_total_summary$CI_Excludes_Zero
    ),
    Sensitivity_Total_Effect_Signs_Consistent = sum(
      total_comparison$Sign_Consistent
    ),
    Sensitivity_Total_Effect_N = nrow(total_comparison)
  )

  # Draw final figure
  rho_row <- exogenous_correlation[Season == season]

  if (nrow(rho_row) != 1L) {
    stop(
      paste(
        "Exogenous correlation row not unique for",
        season_label
      ),
      call. = FALSE
    )
  }

  figure_png <- file.path(
    FIGURE_DIR,
    paste0(
      sprintf("%02d", season),
      "_",
      season_label,
      "_SEM_Path_Diagram.png"
    )
  )

  figure_pdf <- file.path(
    FIGURE_DIR,
    paste0(
      sprintf("%02d", season),
      "_",
      season_label,
      "_SEM_Path_Diagram.pdf"
    )
  )

  invisible(
    make_path_figure(
      edge_data = direct_season,
      exogenous_rho = rho_row$Spearman_Rho[[1L]],
      season_label = season_label,
      output_png = figure_png,
      output_pdf = figure_pdf
    )
  )

  figure_metadata_rows[[length(figure_metadata_rows) + 1L]] <-
    data.table(
      Season = season,
      Season_Label = season_label,
      PNG_Relative_Path = file.path(
        "Figures",
        basename(figure_png)
      ),
      PNG_Size_Bytes = file.info(figure_png)$size,
      PNG_SHA256 = sha256_file(figure_png),
      PDF_Relative_Path = file.path(
        "Figures",
        basename(figure_pdf)
      ),
      PDF_Size_Bytes = file.info(figure_pdf)$size,
      PDF_SHA256 = sha256_file(figure_pdf),
      Font_Family = FONT_FAMILY,
      PNG_DPI = 600L,
      Width_Inches = 9.0,
      Height_Inches = 6.2
    )

  rm(
    meteorology_draws,
    vegetation_draws,
    primary_fai_draws,
    sensitivity_fai_draws
  )
  invisible(gc())
}

# -----------------------------------------------------------------------------
# Assemble and write tables
# -----------------------------------------------------------------------------

direct_effects <- rbindlist(direct_rows, fill = TRUE)
primary_indirect_effects <- rbindlist(
  primary_indirect_rows,
  fill = TRUE
)
primary_total_effects <- rbindlist(
  primary_total_rows,
  fill = TRUE
)
primary_all_effects <- rbindlist(
  primary_all_rows,
  fill = TRUE
)
sensitivity_all_effects <- rbindlist(
  sensitivity_all_rows,
  fill = TRUE
)
sensitivity_comparison <- rbindlist(
  sensitivity_comparison_rows,
  fill = TRUE
)
edge_table <- rbindlist(edge_rows, fill = TRUE)
mc_qa <- rbindlist(mc_qa_rows, fill = TRUE)
seasonal_summary <- rbindlist(
  season_summary_rows,
  fill = TRUE
)
figure_metadata <- rbindlist(
  figure_metadata_rows,
  fill = TRUE
)

setorder(direct_effects, Season, To, From)
setorder(
  primary_indirect_effects,
  Season,
  Predictor,
  Path_Name
)
setorder(
  primary_total_effects,
  Season,
  Predictor,
  Effect_Type
)
setorder(
  primary_all_effects,
  Season,
  Predictor,
  Effect_Type,
  Path_Name
)
setorder(
  sensitivity_all_effects,
  Season,
  Predictor,
  Effect_Type,
  Path_Name
)
setorder(
  sensitivity_comparison,
  Season,
  Predictor
)
setorder(edge_table, Season, To, From)

fwrite(direct_effects, DIRECT_EFFECT_FILE)
fwrite(primary_indirect_effects, INDIRECT_EFFECT_FILE)
fwrite(primary_total_effects, TOTAL_EFFECT_FILE)
fwrite(primary_all_effects, ALL_PRIMARY_EFFECT_FILE)
fwrite(sensitivity_all_effects, SENSITIVITY_EFFECT_FILE)
fwrite(sensitivity_comparison, SENSITIVITY_COMPARISON_FILE)
fwrite(edge_table, EDGE_TABLE_FILE)
fwrite(figure_metadata, FIGURE_METADATA_FILE)
fwrite(mc_qa, MONTE_CARLO_QA_FILE)
fwrite(seasonal_summary, SEASONAL_SUMMARY_FILE)

caption_text <- paste(
  "Seasonal piecewise structural equation models for forest fire activity.",
  "Directed arrows show the locked primary-model coefficients.",
  "Coefficients for paths ending in Meteorology or Vegetation are",
  "standardized Gaussian coefficients, whereas coefficients for paths",
  "ending in forest fire activity (FAI) are conditional Tweedie log-link",
  "coefficients. Solid arrows indicate p < 0.05 and dashed arrows indicate",
  "p >= 0.05. Black and grey arrows denote positive and negative",
  "coefficients, respectively. The dotted double-headed connection shows",
  "the Spearman correlation between the exogenous Topography and",
  "Anthropogenic domains. Country, linear year, grid random intercepts,",
  "year random intercepts, and the forest-exposure offset were included",
  "as controls but are omitted from the diagrams for clarity."
)

writeLines(caption_text, CAPTION_FILE)

# -----------------------------------------------------------------------------
# Final audit
# -----------------------------------------------------------------------------

audit <- add_check(
  audit,
  "Outputs",
  "Twenty_four_direct_paths",
  nrow(direct_effects) == EXPECTED_DIRECT_PATH_N,
  nrow(direct_effects),
  EXPECTED_DIRECT_PATH_N
)

audit <- add_check(
  audit,
  "Outputs",
  "Fifteen_primary_specific_indirect_effects",
  nrow(primary_indirect_effects) == EXPECTED_SPECIFIC_INDIRECT_N,
  nrow(primary_indirect_effects),
  EXPECTED_SPECIFIC_INDIRECT_N
)

audit <- add_check(
  audit,
  "Outputs",
  "Twelve_primary_total_effects",
  nrow(primary_total_effects[Effect_Type == "Total"]) ==
    EXPECTED_TOTAL_EFFECT_N,
  nrow(primary_total_effects[Effect_Type == "Total"]),
  EXPECTED_TOTAL_EFFECT_N
)

audit <- add_check(
  audit,
  "Outputs",
  "Three_seasonal_path_figures",
  nrow(figure_metadata) == EXPECTED_SEASON_N &&
    all(figure_metadata$PNG_Size_Bytes > 0) &&
    all(figure_metadata$PDF_Size_Bytes > 0),
  nrow(figure_metadata),
  EXPECTED_SEASON_N
)

audit <- add_check(
  audit,
  "Monte_Carlo",
  "All_MC_means_close_to_point_estimates",
  all(mc_qa$Absolute_MC_Mean_Difference < 0.01),
  max(mc_qa$Absolute_MC_Mean_Difference),
  "<0.01"
)

audit <- add_check(
  audit,
  "Sensitivity",
  "All_total_effect_signs_stable",
  all(sensitivity_comparison$Sign_Consistent),
  sum(sensitivity_comparison$Sign_Consistent),
  nrow(sensitivity_comparison)
)

audit <- add_check(
  audit,
  "Method",
  "No_models_refitted",
  TRUE,
  FALSE,
  FALSE,
  detail = (
    "Step 05 only reads sealed Step 04 model objects."
  )
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
    "Step 05 final audit failed. See 01_Input_Integrity_Audit.csv.",
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
  Step04A_Code_Version = step04a_method$Code_Version,
  Step04A_Script_SHA256 = step04a_method$Script_SHA256,
  Step04A_Optimizer_Agreement_Passed = (
    step04a_decision$Optimizer_Agreement_Passed
  ),
  Primary_Response = PRIMARY_RESPONSE,
  Sensitivity_Response = SENSITIVITY_RESPONSE,
  Monte_Carlo_Draws_Per_Model = N_SIM,
  Base_Seed = BASE_SEED,
  Monte_Carlo_Distribution = (
    "Multivariate normal using each model's fixed-effect covariance matrix"
  ),
  Cross_Model_Covariance_Assumption = paste(
    "Coefficient draws from separate component models are treated as",
    "independent; within-model fixed-effect covariance is retained."
  ),
  Direct_Effect_Scale = list(
    Mediator_Equations = "Standardized Gaussian coefficient",
    FAI_Equation = "Conditional Tweedie log-link coefficient"
  ),
  Indirect_and_Total_Effect_Scale = (
    "Conditional log expected FAI"
  ),
  Response_Ratio_Transformation = "exp(effect)",
  Percent_Change_Transformation = "100 * (exp(effect) - 1)",
  Topography_Anthropogenic_Relation = (
    "Observed exogenous correlation; no directional causal path"
  ),
  Model_Refitting_Performed = FALSE,
  Final_Effect_Decomposition_Calculated = TRUE,
  Final_Path_Diagrams_Created = TRUE,
  Causal_Interpretation_Limit = paste(
    "Model-implied effects under the locked observational causal",
    "hypothesis; not definitive causal proof."
  )
)

write_json(method_definition, METHOD_FILE)

software_environment <- list(
  R_Version = R.version.string,
  Platform = R.version$platform,
  data_table = as.character(packageVersion("data.table")),
  glmmTMB = as.character(packageVersion("glmmTMB")),
  MASS = as.character(packageVersion("MASS")),
  Matrix = as.character(packageVersion("Matrix")),
  ggplot2 = as.character(packageVersion("ggplot2")),
  jsonlite = as.character(packageVersion("jsonlite")),
  digest = as.character(packageVersion("digest")),
  R_utils = as.character(packageVersion("R.utils")),
  Cairo_Available = capabilities("cairo")
)

write_json(software_environment, SOFTWARE_FILE)

# Completion log before manifest so the log hash remains stable.
log_message(
  paste(
    "Step 05 completed: seasonal direct, indirect, and total effects",
    "and three final primary-FAI path diagrams were created."
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
  "Primary direct paths:",
  nrow(direct_effects),
  "/",
  EXPECTED_DIRECT_PATH_N,
  "\n"
)
cat(
  "Primary specific indirect effects:",
  nrow(primary_indirect_effects),
  "/",
  EXPECTED_SPECIFIC_INDIRECT_N,
  "\n"
)
cat(
  "Primary total effects:",
  nrow(primary_total_effects[Effect_Type == "Total"]),
  "/",
  EXPECTED_TOTAL_EFFECT_N,
  "\n"
)
cat(
  "Sensitivity total-effect signs stable:",
  sum(sensitivity_comparison$Sign_Consistent),
  "/",
  nrow(sensitivity_comparison),
  "\n"
)
cat(
  "Final seasonal path diagrams created:",
  nrow(figure_metadata),
  "/",
  EXPECTED_SEASON_N,
  "\n"
)
cat(
  "Maximum MC mean-point difference:",
  max(mc_qa$Absolute_MC_Mean_Difference),
  "\n"
)
cat("Models refitted: 0\n")
cat("Final direct/indirect effects calculated: True\n")
cat("Final path diagrams created: True\n")
