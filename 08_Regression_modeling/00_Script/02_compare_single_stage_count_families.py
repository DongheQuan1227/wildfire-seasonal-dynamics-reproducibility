# -*- coding: utf-8 -*-
"""
Predictor-adjusted comparison of single-stage Poisson, NB1, and NB2 models.

Recommended public location
---------------------------
<CODE_ROOT>/08_Regression_modeling/00_Script/
    02_compare_single_stage_count_families.py

Input
-----
<CODE_ROOT>/00_Input_data/Base_Table_2001_2025.csv.bz2

Output
------
<CODE_ROOT>/08_Regression_modeling/
    02_Single_Stage_Count_Family_Comparison/

Purpose
-------
This script identifies the most suitable non-zero-inflated, single-stage
count family before predictor optimization.

The compared models are:

1. Poisson regression;
2. negative binomial type 1 (NB1):
       Var(Y | X) = mu + alpha * mu
3. negative binomial type 2 (NB2):
       Var(Y | X) = mu + alpha * mu^2

All models use:

    exposure = ForestPixelCount

which is equivalent to adding:

    offset = log(ForestPixelCount)

The comparison uses one prespecified, low-collinearity reference predictor
set for every family, season, and response. This isolates distribution-family
performance from later predictor-selection decisions. Random-forest variable
importance is not used.

Model-family comparison is based primarily on five-fold temporal out-of-fold
predictive negative log-likelihood, with zero-probability Brier score and
rate-scale RMSE as secondary criteria.

This is a family-screening step, not the final model comparison.
"""

from __future__ import annotations

import json
import math
import platform
import shutil
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy
import sklearn
import statsmodels
from scipy import sparse
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from statsmodels.discrete.discrete_model import (
    NegativeBinomial,
    Poisson as DiscretePoisson,
)
from statsmodels.genmod.families import (
    Poisson as PoissonFamily,
)
from statsmodels.genmod.generalized_linear_model import (
    GLM,
)


# =============================================================================
# Repository-relative paths
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REGRESSION_ROOT = SCRIPT_DIR.parent
CODE_ROOT = REGRESSION_ROOT.parent

INPUT_CSV = (
    CODE_ROOT
    / "00_Input_data"
    / "Base_Table_2001_2025.csv.bz2"
)

OUTPUT_ROOT = (
    REGRESSION_ROOT
    / "02_Single_Stage_Count_Family_Comparison"
)

LOG_FILE = OUTPUT_ROOT / "single_stage_count_family_comparison.log"


# =============================================================================
# Analysis settings
# =============================================================================

RANDOM_SEED = 2026

FOREST_PIXEL_AREA_KM2 = 0.25
FOREST_AREA_THRESHOLD_KM2 = 2.5

SEASONS = {
    1: "Spring",
    2: "Summer",
    3: "Autumn",
}

COUNT_RESPONSES = {
    "Fire_Count": "FCD",
    "Burned_Pixel_Count": "BAD",
}

MODEL_FAMILIES = [
    "Poisson",
    "NB1",
    "NB2",
]

TEMPORAL_FOLDS = {
    1: [2024, 2002, 2006, 2004, 2009],
    2: [2016, 2015, 2007, 2018, 2014],
    3: [2025, 2021, 2013, 2019, 2003],
    4: [2010, 2023, 2017, 2011, 2005],
    5: [2020, 2022, 2001, 2012, 2008],
}

COUNTRY_LEVELS = [
    "China",
    "NK",
    "Russia",
]
COUNTRY_REFERENCE = "China"
COUNTRY_DUMMY_COLUMNS = [
    "Country_NK",
    "Country_Russia",
]

# -------------------------------------------------------------------------
# Prespecified family-screening reference set
# -------------------------------------------------------------------------
#
# The reference set contains interpretable representatives of:
# - forest structure,
# - fire weather / ignition,
# - drought,
# - terrain,
# - anthropogenic influence.
#
# It was chosen from the count-response audit before examining family
# comparison results. The full audit showed no pair with |Spearman rho| >=
# 0.80 inside this set, and its approximate audit-stage maximum VIF was below
# 5. This set is not claimed to be the final predictor set.
#
REFERENCE_CONTINUOUS_PREDICTORS = [
    "BD",
    "PTC",
    "FWI",
    "LtgProxy",
    "SPEI3",
    "DEM",
    "Slope",
    "Aspect",
    "POP",
    "Dis_Farm",
    "Road_dens",
]

REFERENCE_PREDICTOR_DOMAINS = {
    "BD": "Forest_structure",
    "PTC": "Forest_structure",
    "FWI": "Fire_weather",
    "LtgProxy": "Meteorology",
    "SPEI3": "Drought",
    "DEM": "Terrain",
    "Slope": "Terrain",
    "Aspect": "Terrain",
    "POP": "Anthropogenic",
    "Dis_Farm": "Anthropogenic",
    "Road_dens": "Anthropogenic",
}

# Audit-guided transformations are fixed before family comparison.
LOG1P_PREDICTORS = {
    "LtgProxy",
    "POP",
    "Dis_Farm",
    "Road_dens",
}

VIF_SAMPLE_MAX = 100_000
FIT_MAX_ITERATIONS = 500

# Multiple starts and optimizers improve robustness of NB maximum likelihood.
NB_ALPHA_STARTS = [
    0.1,
    0.5,
    1.0,
    5.0,
]
NB_OPTIMIZERS = [
    "bfgs",
    "lbfgs",
    "newton",
]


# =============================================================================
# Logging
# =============================================================================

def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


# =============================================================================
# Basic utilities
# =============================================================================

def temporal_fold_mapping() -> Dict[int, int]:
    mapping: Dict[int, int] = {}

    for fold, years in TEMPORAL_FOLDS.items():
        for year in years:
            if int(year) in mapping:
                raise ValueError(
                    f"Year {year} appears in more than one temporal fold."
                )
            mapping[int(year)] = int(fold)

    return mapping


def safe_r2(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> float:
    if len(observed) < 2:
        return np.nan

    if float(np.var(observed)) <= 0:
        return np.nan

    return float(r2_score(observed, predicted))


def safe_roc_auc(
    observed_binary: np.ndarray,
    predicted_probability: np.ndarray,
) -> float:
    if len(np.unique(observed_binary)) < 2:
        return np.nan

    return float(
        roc_auc_score(
            observed_binary,
            predicted_probability,
        )
    )


def safe_pr_auc(
    observed_binary: np.ndarray,
    predicted_probability: np.ndarray,
) -> float:
    if len(np.unique(observed_binary)) < 2:
        return np.nan

    return float(
        average_precision_score(
            observed_binary,
            predicted_probability,
        )
    )


def calibration_parameters(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> Tuple[float, float]:
    if (
        len(observed) < 3
        or float(np.std(predicted)) <= 0
    ):
        return np.nan, np.nan

    model = LinearRegression()
    model.fit(
        predicted.reshape(-1, 1),
        observed,
    )

    return (
        float(model.intercept_),
        float(model.coef_[0]),
    )


def finite_or_nan(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan

    return result if np.isfinite(result) else np.nan


# =============================================================================
# Data preparation
# =============================================================================

def required_columns() -> List[str]:
    return [
        "GRID_UID",
        "GRID_ID",
        "Country",
        "Year",
        "Season",
        "Fire_Count",
        "Burned_Pixel_Count",
        "ForestPixelCount",
        "ForestArea_km2",
        *REFERENCE_CONTINUOUS_PREDICTORS,
    ]


def prepare_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Base table not found:\n{INPUT_CSV}"
        )

    log(f"Reading input: {INPUT_CSV}")
    data = pd.read_csv(INPUT_CSV)

    missing = sorted(
        set(required_columns()) - set(data.columns)
    )
    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(missing)
        )

    duplicate_mask = data.duplicated(
        ["GRID_UID", "Year", "Season"],
        keep=False,
    )
    if duplicate_mask.any():
        data.loc[duplicate_mask].to_csv(
            OUTPUT_ROOT
            / "ERROR_Duplicate_Panel_Records.csv",
            index=False,
            encoding="utf-8-sig",
        )
        raise ValueError(
            "Duplicate GRID_UID-Year-Season records were found."
        )

    expected_area = (
        pd.to_numeric(
            data["ForestPixelCount"],
            errors="coerce",
        )
        * FOREST_PIXEL_AREA_KM2
    )
    observed_area = pd.to_numeric(
        data["ForestArea_km2"],
        errors="coerce",
    )
    area_error = np.abs(observed_area - expected_area)

    if area_error.isna().any():
        raise ValueError(
            "ForestPixelCount or ForestArea_km2 contains "
            "nonnumeric values."
        )

    if float(area_error.max()) > 1e-9:
        raise ValueError(
            "ForestArea_km2 is inconsistent with "
            "ForestPixelCount * 0.25."
        )

    data = data.loc[
        observed_area > FOREST_AREA_THRESHOLD_KM2
    ].copy()

    if data.empty:
        raise ValueError(
            "No rows remain after the forest-area threshold."
        )

    if (
        pd.to_numeric(
            data["ForestPixelCount"],
            errors="coerce",
        )
        <= 0
    ).any():
        raise ValueError(
            "ForestPixelCount must be positive after filtering."
        )

    unexpected_countries = sorted(
        set(data["Country"].dropna().astype(str))
        - set(COUNTRY_LEVELS)
    )
    if unexpected_countries:
        raise ValueError(
            "Unexpected Country values: "
            + ", ".join(unexpected_countries)
        )

    unexpected_seasons = sorted(
        set(data["Season"].dropna().astype(int))
        - set(SEASONS)
    )
    if unexpected_seasons:
        raise ValueError(
            "Unexpected Season values: "
            + ", ".join(map(str, unexpected_seasons))
        )

    for response in COUNT_RESPONSES:
        values = pd.to_numeric(
            data[response],
            errors="coerce",
        )

        invalid = (
            values.isna()
            | ~np.isfinite(values)
            | (values < 0)
            | (
                np.abs(values - np.round(values))
                > 1e-9
            )
        )

        if invalid.any():
            data.loc[
                invalid,
                [
                    "GRID_UID",
                    "GRID_ID",
                    "Country",
                    "Year",
                    "Season",
                    response,
                    "ForestPixelCount",
                ],
            ].to_csv(
                OUTPUT_ROOT
                / f"ERROR_Invalid_{response}.csv",
                index=False,
                encoding="utf-8-sig",
            )
            raise ValueError(
                f"{response} contains invalid count values."
            )

        data[response] = values.astype(int)

    mapping = temporal_fold_mapping()
    data["Temporal_Fold"] = (
        data["Year"]
        .astype(int)
        .map(mapping)
    )

    if data["Temporal_Fold"].isna().any():
        missing_years = sorted(
            data.loc[
                data["Temporal_Fold"].isna(),
                "Year",
            ]
            .dropna()
            .astype(int)
            .unique()
        )
        raise ValueError(
            "Years missing from temporal folds: "
            + ", ".join(map(str, missing_years))
        )

    data["Temporal_Fold"] = (
        data["Temporal_Fold"].astype(int)
    )

    model_columns = [
        "Country",
        "Fire_Count",
        "Burned_Pixel_Count",
        "ForestPixelCount",
        *REFERENCE_CONTINUOUS_PREDICTORS,
    ]

    finite_model_data = (
        data[model_columns]
        .replace([np.inf, -np.inf], np.nan)
        .notna()
        .all(axis=1)
    )

    excluded = data.loc[
        ~finite_model_data,
        [
            "GRID_UID",
            "GRID_ID",
            "Country",
            "Year",
            "Season",
            *model_columns,
        ],
    ].copy()

    excluded.to_csv(
        OUTPUT_ROOT
        / "Excluded_Incomplete_Reference_Set_Records.csv",
        index=False,
        encoding="utf-8-sig",
    )

    data = (
        data.loc[finite_model_data]
        .copy()
        .reset_index(drop=True)
    )

    data["FCD"] = (
        data["Fire_Count"].astype(float)
        / data["ForestPixelCount"].astype(float)
    )
    data["BAD"] = (
        data["Burned_Pixel_Count"].astype(float)
        / data["ForestPixelCount"].astype(float)
    )

    log(
        f"Prepared common complete-case sample: "
        f"{len(data):,} rows; "
        f"{data['GRID_UID'].nunique():,} grids; "
        f"{len(excluded):,} incomplete rows excluded."
    )

    return data, excluded


# =============================================================================
# Predictor transformation and design matrices
# =============================================================================

def transform_continuous_predictors(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    transformed = frame[
        REFERENCE_CONTINUOUS_PREDICTORS
    ].astype(float).copy()

    for variable in LOG1P_PREDICTORS:
        if variable not in transformed.columns:
            continue

        minimum = float(transformed[variable].min())
        if minimum < 0:
            raise ValueError(
                f"{variable} has a negative value "
                f"({minimum}) and cannot use log1p."
            )

        transformed[variable] = np.log1p(
            transformed[variable].to_numpy(dtype=float)
        )

    return transformed


def country_dummies(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    country = frame["Country"].astype(str)

    unexpected = sorted(
        set(country) - set(COUNTRY_LEVELS)
    )
    if unexpected:
        raise ValueError(
            "Unexpected Country values: "
            + ", ".join(unexpected)
        )

    return pd.DataFrame(
        {
            "Country_NK":
                (country == "NK").astype(float).to_numpy(),
            "Country_Russia":
                (country == "Russia").astype(float).to_numpy(),
        },
        index=frame.index,
    )


def make_design_matrices(
    training: pd.DataFrame,
    validation: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    training_continuous = (
        transform_continuous_predictors(training)
    )
    validation_continuous = (
        transform_continuous_predictors(validation)
    )

    scaler = StandardScaler()
    x_training_continuous = scaler.fit_transform(
        training_continuous.to_numpy(dtype=float)
    )
    x_validation_continuous = scaler.transform(
        validation_continuous.to_numpy(dtype=float)
    )

    x_training_country = country_dummies(
        training
    ).to_numpy(dtype=float)
    x_validation_country = country_dummies(
        validation
    ).to_numpy(dtype=float)

    x_training = np.column_stack(
        [
            np.ones(len(training), dtype=float),
            x_training_continuous,
            x_training_country,
        ]
    )
    x_validation = np.column_stack(
        [
            np.ones(len(validation), dtype=float),
            x_validation_continuous,
            x_validation_country,
        ]
    )

    return x_training, x_validation, scaler


def full_data_design_matrix(
    frame: pd.DataFrame,
) -> Tuple[np.ndarray, StandardScaler]:
    continuous = transform_continuous_predictors(frame)

    scaler = StandardScaler()
    continuous_scaled = scaler.fit_transform(
        continuous.to_numpy(dtype=float)
    )

    country = country_dummies(frame).to_numpy(dtype=float)

    design = np.column_stack(
        [
            np.ones(len(frame), dtype=float),
            continuous_scaled,
            country,
        ]
    )

    return design, scaler


def design_term_names() -> List[str]:
    return [
        "Intercept",
        *REFERENCE_CONTINUOUS_PREDICTORS,
        *COUNTRY_DUMMY_COLUMNS,
    ]


# =============================================================================
# Reference-set VIF
# =============================================================================

def reference_set_vif(
    data: pd.DataFrame,
) -> pd.DataFrame:
    transformed = transform_continuous_predictors(data)

    if len(transformed) > VIF_SAMPLE_MAX:
        transformed = transformed.sample(
            n=VIF_SAMPLE_MAX,
            random_state=RANDOM_SEED,
        )

    standardized = (
        transformed - transformed.mean(axis=0)
    ) / transformed.std(axis=0, ddof=0)

    correlation = standardized.corr().to_numpy(dtype=float)
    inverse = np.linalg.pinv(
        correlation,
        hermitian=True,
    )
    vif_values = np.diag(inverse)
    condition_number = float(
        np.linalg.cond(correlation)
    )

    rows: List[Dict[str, object]] = []

    for variable, vif_value in zip(
        REFERENCE_CONTINUOUS_PREDICTORS,
        vif_values,
    ):
        rows.append(
            {
                "Variable": variable,
                "Domain": REFERENCE_PREDICTOR_DOMAINS[variable],
                "Transformation": (
                    "log1p_then_z_standardized"
                    if variable in LOG1P_PREDICTORS
                    else "z_standardized"
                ),
                "VIF": float(vif_value),
                "VIF_Above_5": bool(vif_value >= 5.0),
                "VIF_Above_10": bool(vif_value >= 10.0),
                "Correlation_Matrix_Condition_Number": (
                    condition_number
                ),
                "Sample_N": int(len(standardized)),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["VIF", "Variable"],
        ascending=[False, True],
    ).reset_index(drop=True)


# =============================================================================
# Model fitting
# =============================================================================

def convergence_status(
    result: object,
) -> bool:
    mle_retvals = getattr(
        result,
        "mle_retvals",
        {},
    )

    if isinstance(mle_retvals, dict):
        if "converged" in mle_retvals:
            return bool(mle_retvals["converged"])

        if "warnflag" in mle_retvals:
            return int(mle_retvals["warnflag"]) == 0

    return True


def warning_text(
    captured_warnings: Sequence[warnings.WarningMessage],
) -> str:
    unique: List[str] = []

    for item in captured_warnings:
        message = (
            f"{item.category.__name__}: "
            f"{str(item.message)}"
        )
        if message not in unique:
            unique.append(message)

    return " | ".join(unique)


def fit_poisson(
    observed: np.ndarray,
    design: np.ndarray,
    exposure: np.ndarray,
) -> Tuple[object, Dict[str, object]]:
    """Fit Poisson MLE with a numerically stable fallback.

    The primary implementation is the discrete Poisson Newton MLE used by
    the original run. In extremely sparse combinations, coefficient
    optimization can finish but Hessian inversion for covariance estimation
    can fail with ``LinAlgError: Singular matrix``. In that case, the function
    falls back to an equivalent Poisson log-link GLM fitted by IRLS with a
    pseudoinverse weighted-least-squares step.
    """
    primary_error = ""

    try:
        discrete_model = DiscretePoisson(
            observed,
            design,
            exposure=exposure,
            missing="raise",
            check_rank=False,
        )

        with warnings.catch_warnings(record=True) as caught_primary:
            warnings.simplefilter("always")
            result = discrete_model.fit(
                method="newton",
                maxiter=FIT_MAX_ITERATIONS,
                disp=0,
                full_output=True,
            )

        converged = convergence_status(result)
        if converged and np.isfinite(result.params).all():
            status = {
                "Family": "Poisson",
                "Optimizer": "discrete_newton",
                "Alpha_Start": np.nan,
                "Converged": True,
                "Warning_Text": warning_text(caught_primary),
                "Alpha_Estimate": np.nan,
                "Attempt_Count": 1,
                "Failed_Attempt_Count": 0,
            }
            return result, status

        primary_error = (
            "Discrete Poisson Newton returned nonconverged or "
            "nonfinite coefficients."
        )

    except Exception as error:
        primary_error = f"{type(error).__name__}: {error}"

    glm_model = GLM(
        observed,
        design,
        family=PoissonFamily(),
        exposure=exposure,
        missing="raise",
    )

    with warnings.catch_warnings(record=True) as caught_fallback:
        warnings.simplefilter("always")
        result = glm_model.fit(
            method="IRLS",
            maxiter=FIT_MAX_ITERATIONS,
            tol=1e-8,
            full_output=True,
            disp=False,
            wls_method="pinv",
        )

    converged = bool(getattr(result, "converged", True))
    if not converged or not np.isfinite(result.params).all():
        raise RuntimeError(
            "Both discrete Poisson Newton and GLM IRLS fallback failed. "
            f"Primary failure: {primary_error}"
        )

    fallback_warning = warning_text(caught_fallback)
    combined_warning = (
        f"Primary discrete fit failed: {primary_error}"
        + (f" | Fallback warnings: {fallback_warning}" if fallback_warning else "")
    )

    status = {
        "Family": "Poisson",
        "Optimizer": "GLM_IRLS_pinv_fallback",
        "Alpha_Start": np.nan,
        "Converged": converged,
        "Warning_Text": combined_warning,
        "Alpha_Estimate": np.nan,
        "Attempt_Count": 2,
        "Failed_Attempt_Count": 1,
    }

    return result, status


def stable_count_log_probability(
    observed: np.ndarray,
    predicted_mean: np.ndarray,
    family: str,
    alpha: float | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return numerically stable log PMF and zero probability.

    Statsmodels' discrete NB likelihood can return NaN for extremely sparse
    data when the fitted mean approaches zero. The formulas below evaluate
    the same Poisson, NB1, and NB2 probability models using stable log-space
    expressions.
    """
    observed = np.asarray(observed, dtype=float)
    mu = np.asarray(predicted_mean, dtype=float)
    mu = np.clip(mu, np.finfo(float).tiny, 1e300)

    if family == "Poisson":
        log_probability = (
            scipy.special.xlogy(observed, mu)
            - mu
            - scipy.special.gammaln(observed + 1.0)
        )
        log_zero_probability = -mu

    elif family == "NB2":
        if alpha is None or not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("NB2 requires a finite positive alpha.")

        size = 1.0 / float(alpha)
        log_denominator = np.log1p(float(alpha) * mu)

        log_probability = (
            scipy.special.gammaln(observed + size)
            - scipy.special.gammaln(size)
            - scipy.special.gammaln(observed + 1.0)
            - size * log_denominator
            + scipy.special.xlogy(observed, float(alpha) * mu)
            - observed * log_denominator
        )
        log_zero_probability = -size * log_denominator

    elif family == "NB1":
        if alpha is None or not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("NB1 requires a finite positive alpha.")

        alpha_value = float(alpha)
        size = np.clip(mu / alpha_value, np.finfo(float).tiny, 1e300)
        log_success_probability = -math.log1p(alpha_value)
        log_failure_probability = (
            math.log(alpha_value) - math.log1p(alpha_value)
        )

        log_probability = np.empty_like(mu, dtype=float)
        zero_mask = observed == 0
        positive_mask = ~zero_mask

        log_probability[zero_mask] = (
            size[zero_mask] * log_success_probability
        )

        if positive_mask.any():
            y_positive = observed[positive_mask]
            size_positive = size[positive_mask]
            log_probability[positive_mask] = (
                scipy.special.gammaln(y_positive + size_positive)
                - scipy.special.gammaln(size_positive)
                - scipy.special.gammaln(y_positive + 1.0)
                + size_positive * log_success_probability
                + y_positive * log_failure_probability
            )

        log_zero_probability = size * log_success_probability

    else:
        raise ValueError(f"Unsupported family: {family}")

    zero_probability = np.exp(
        np.clip(log_zero_probability, -745.0, 0.0)
    )

    return log_probability, zero_probability


def mean_from_fitted_parameters(
    result: object,
    design: np.ndarray,
    exposure: np.ndarray,
) -> np.ndarray:
    coefficient_count = int(design.shape[1])
    parameters = np.asarray(result.params, dtype=float)

    if len(parameters) < coefficient_count:
        raise ValueError(
            "Fitted parameter vector is shorter than the design matrix."
        )

    beta = parameters[:coefficient_count]
    linear_predictor = (
        design @ beta
        + np.log(np.asarray(exposure, dtype=float))
    )
    linear_predictor = np.clip(
        linear_predictor,
        -745.0,
        700.0,
    )

    predicted_mean = np.exp(linear_predictor)

    if not np.isfinite(predicted_mean).all():
        raise FloatingPointError(
            "Predicted count mean contains non-finite values."
        )

    return predicted_mean


def fit_negative_binomial(
    observed: np.ndarray,
    design: np.ndarray,
    exposure: np.ndarray,
    family: str,
    poisson_parameters: np.ndarray,
) -> Tuple[object, Dict[str, object]]:
    if family not in {"NB1", "NB2"}:
        raise ValueError(
            f"Unsupported negative-binomial family: {family}"
        )

    loglike_method = family.lower()

    model = NegativeBinomial(
        observed,
        design,
        loglike_method=loglike_method,
        exposure=exposure,
        missing="raise",
        check_rank=False,
    )

    attempts: List[Dict[str, object]] = []
    best_result = None
    best_status = None
    best_stable_log_likelihood = -np.inf

    for alpha_start in NB_ALPHA_STARTS:
        start_params = np.concatenate(
            [
                np.asarray(
                    poisson_parameters,
                    dtype=float,
                )[: design.shape[1]],
                np.array(
                    [float(alpha_start)],
                    dtype=float,
                ),
            ]
        )

        for optimizer in NB_OPTIMIZERS:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")

                try:
                    result = model.fit(
                        start_params=start_params,
                        method=optimizer,
                        maxiter=FIT_MAX_ITERATIONS,
                        disp=0,
                        full_output=True,
                    )

                    converged = convergence_status(result)
                    parameters = np.asarray(
                        result.params,
                        dtype=float,
                    )
                    alpha_estimate = finite_or_nan(
                        parameters[-1]
                    )
                    internal_log_likelihood = finite_or_nan(
                        getattr(result, "llf", np.nan)
                    )

                    stable_log_likelihood = np.nan
                    stable_error = ""

                    if (
                        np.isfinite(parameters).all()
                        and np.isfinite(alpha_estimate)
                        and alpha_estimate > 0
                    ):
                        try:
                            predicted_mean = mean_from_fitted_parameters(
                                result=result,
                                design=design,
                                exposure=exposure,
                            )
                            log_probability, _ = (
                                stable_count_log_probability(
                                    observed=observed,
                                    predicted_mean=predicted_mean,
                                    family=family,
                                    alpha=alpha_estimate,
                                )
                            )
                            stable_log_likelihood = float(
                                np.sum(log_probability)
                            )
                        except Exception as error:
                            stable_error = (
                                f"StableLikelihoodError: "
                                f"{type(error).__name__}: {error}"
                            )

                    valid = (
                        converged
                        and np.isfinite(alpha_estimate)
                        and alpha_estimate > 0
                        and np.isfinite(stable_log_likelihood)
                        and np.isfinite(parameters).all()
                    )

                    messages = [warning_text(caught)]
                    if not np.isfinite(internal_log_likelihood):
                        messages.append(
                            "Statsmodels internal log-likelihood was "
                            "non-finite; stable log-space likelihood used."
                        )
                    if stable_error:
                        messages.append(stable_error)

                    status = {
                        "Family": family,
                        "Optimizer": optimizer,
                        "Alpha_Start": float(alpha_start),
                        "Converged": bool(converged),
                        "Warning_Text": " | ".join(
                            message for message in messages if message
                        ),
                        "Alpha_Estimate": alpha_estimate,
                        "Internal_LogLikelihood": internal_log_likelihood,
                        "Stable_LogLikelihood": stable_log_likelihood,
                        "Valid_Fit": bool(valid),
                    }

                    attempts.append(status)

                    if (
                        valid
                        and stable_log_likelihood
                        > best_stable_log_likelihood
                    ):
                        best_result = result
                        best_status = status
                        best_stable_log_likelihood = (
                            stable_log_likelihood
                        )
                        break

                except Exception as error:
                    attempts.append(
                        {
                            "Family": family,
                            "Optimizer": optimizer,
                            "Alpha_Start": float(alpha_start),
                            "Converged": False,
                            "Warning_Text": (
                                f"{type(error).__name__}: {error}"
                            ),
                            "Alpha_Estimate": np.nan,
                            "Internal_LogLikelihood": np.nan,
                            "Stable_LogLikelihood": np.nan,
                            "Valid_Fit": False,
                        }
                    )

        if best_result is not None:
            break

    if best_result is None or best_status is None:
        attempt_summary = "\n".join(
            json.dumps(
                attempt,
                ensure_ascii=False,
                default=str,
            )
            for attempt in attempts
        )
        raise RuntimeError(
            f"All fitting attempts failed for {family}.\n"
            f"{attempt_summary}"
        )

    best_status = dict(best_status)
    best_status["Attempt_Count"] = len(attempts)
    best_status["Failed_Attempt_Count"] = int(
        sum(
            not bool(attempt.get("Valid_Fit", False))
            for attempt in attempts
        )
    )
    best_status["Likelihood_Evaluation"] = (
        "Stable_log_space_formula"
    )

    return best_result, best_status


def fitted_distribution_outputs(
    result: object,
    design: np.ndarray,
    exposure: np.ndarray,
    observed: np.ndarray,
    family: str,
    alpha: float | None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    predicted_mean = mean_from_fitted_parameters(
        result=result,
        design=design,
        exposure=exposure,
    )

    log_probability, predicted_zero_probability = (
        stable_count_log_probability(
            observed=observed,
            predicted_mean=predicted_mean,
            family=family,
            alpha=alpha,
        )
    )

    if not np.isfinite(log_probability).all():
        raise FloatingPointError(
            "Predictive log probability contains non-finite values."
        )

    if not np.isfinite(predicted_zero_probability).all():
        raise FloatingPointError(
            "Predicted zero probability contains non-finite values."
        )

    predicted_zero_probability = np.clip(
        predicted_zero_probability,
        0.0,
        1.0,
    )

    return (
        predicted_mean,
        predicted_zero_probability,
        log_probability,
    )


# =============================================================================
# Metrics
# =============================================================================

def prediction_metrics(
    observed_count: np.ndarray,
    predicted_count: np.ndarray,
    exposure: np.ndarray,
    predicted_zero_probability: np.ndarray,
    log_probability: np.ndarray,
) -> Dict[str, float]:
    observed_count = np.asarray(
        observed_count,
        dtype=float,
    )
    predicted_count = np.asarray(
        predicted_count,
        dtype=float,
    )
    exposure = np.asarray(
        exposure,
        dtype=float,
    )
    predicted_zero_probability = np.asarray(
        predicted_zero_probability,
        dtype=float,
    )
    log_probability = np.asarray(
        log_probability,
        dtype=float,
    )

    observed_rate = observed_count / exposure
    predicted_rate = predicted_count / exposure

    observed_zero = (
        observed_count == 0
    ).astype(int)
    observed_positive = (
        observed_count > 0
    ).astype(int)
    predicted_positive_probability = (
        1.0 - predicted_zero_probability
    )

    count_calibration_intercept, count_calibration_slope = (
        calibration_parameters(
            observed_count,
            predicted_count,
        )
    )
    rate_calibration_intercept, rate_calibration_slope = (
        calibration_parameters(
            observed_rate,
            predicted_rate,
        )
    )

    total_observed = float(observed_count.sum())
    total_predicted = float(predicted_count.sum())

    result: Dict[str, float] = {
        "N": int(len(observed_count)),
        "Observed_Total_Count": total_observed,
        "Predicted_Total_Count": total_predicted,
        "Predicted_to_Observed_Total_Count_Ratio": (
            total_predicted / total_observed
            if total_observed > 0
            else np.nan
        ),
        "Observed_Mean_Count": float(
            observed_count.mean()
        ),
        "Predicted_Mean_Count": float(
            predicted_count.mean()
        ),
        "Count_RMSE": float(
            math.sqrt(
                mean_squared_error(
                    observed_count,
                    predicted_count,
                )
            )
        ),
        "Count_MAE": float(
            mean_absolute_error(
                observed_count,
                predicted_count,
            )
        ),
        "Count_R2": safe_r2(
            observed_count,
            predicted_count,
        ),
        "Count_Calibration_Intercept": (
            count_calibration_intercept
        ),
        "Count_Calibration_Slope": (
            count_calibration_slope
        ),
        "Observed_Mean_Rate": float(
            observed_rate.mean()
        ),
        "Predicted_Mean_Rate": float(
            predicted_rate.mean()
        ),
        "Rate_RMSE": float(
            math.sqrt(
                mean_squared_error(
                    observed_rate,
                    predicted_rate,
                )
            )
        ),
        "Rate_MAE": float(
            mean_absolute_error(
                observed_rate,
                predicted_rate,
            )
        ),
        "Rate_R2": safe_r2(
            observed_rate,
            predicted_rate,
        ),
        "Rate_Calibration_Intercept": (
            rate_calibration_intercept
        ),
        "Rate_Calibration_Slope": (
            rate_calibration_slope
        ),
        "Observed_Zero_Proportion": float(
            observed_zero.mean()
        ),
        "Predicted_Zero_Proportion": float(
            predicted_zero_probability.mean()
        ),
        "Predicted_minus_Observed_Zero_Proportion": float(
            predicted_zero_probability.mean()
            - observed_zero.mean()
        ),
        "Zero_Probability_Brier": float(
            np.mean(
                (
                    predicted_zero_probability
                    - observed_zero
                ) ** 2
            )
        ),
        "Occurrence_ROC_AUC": safe_roc_auc(
            observed_positive,
            predicted_positive_probability,
        ),
        "Occurrence_PR_AUC": safe_pr_auc(
            observed_positive,
            predicted_positive_probability,
        ),
        "Total_Predictive_LogLikelihood": float(
            log_probability.sum()
        ),
        "Mean_Predictive_LogLikelihood": float(
            log_probability.mean()
        ),
        "Mean_Negative_LogLikelihood": float(
            -log_probability.mean()
        ),
    }

    positive = observed_count > 0
    zero = ~positive

    result["Positive_N"] = int(positive.sum())
    result["Zero_N"] = int(zero.sum())

    if positive.any():
        result["Positive_Count_MAE"] = float(
            mean_absolute_error(
                observed_count[positive],
                predicted_count[positive],
            )
        )
        result["Positive_Rate_MAE"] = float(
            mean_absolute_error(
                observed_rate[positive],
                predicted_rate[positive],
            )
        )
    else:
        result["Positive_Count_MAE"] = np.nan
        result["Positive_Rate_MAE"] = np.nan

    if zero.any():
        result["Zero_Row_Predicted_Count_Mean"] = float(
            predicted_count[zero].mean()
        )
        result["Zero_Row_Predicted_Rate_Mean"] = float(
            predicted_rate[zero].mean()
        )
    else:
        result["Zero_Row_Predicted_Count_Mean"] = np.nan
        result["Zero_Row_Predicted_Rate_Mean"] = np.nan

    return result


# =============================================================================
# Temporal cross-validation
# =============================================================================

def prediction_frame(
    validation: pd.DataFrame,
    response: str,
    rate_name: str,
    family: str,
    fold: int,
    predicted_count: np.ndarray,
    predicted_zero_probability: np.ndarray,
    log_probability: np.ndarray,
) -> pd.DataFrame:
    result = validation[
        [
            "GRID_UID",
            "GRID_ID",
            "Country",
            "Year",
            "Season",
            "Temporal_Fold",
            "ForestPixelCount",
        ]
    ].copy()

    observed_count = (
        validation[response]
        .to_numpy(dtype=float)
    )
    exposure = (
        validation["ForestPixelCount"]
        .to_numpy(dtype=float)
    )

    result["Count_Response"] = response
    result["Rate_Scale_Name"] = rate_name
    result["Family"] = family
    result["Fold"] = int(fold)
    result["Observed_Count"] = observed_count
    result["Predicted_Count"] = predicted_count
    result["Observed_Rate"] = (
        observed_count / exposure
    )
    result["Predicted_Rate"] = (
        predicted_count / exposure
    )
    result["Observed_Zero"] = (
        observed_count == 0
    ).astype(int)
    result["Predicted_Zero_Probability"] = (
        predicted_zero_probability
    )
    result["Predictive_LogProbability"] = (
        log_probability
    )

    return result


def run_one_season_response(
    data: pd.DataFrame,
    season_code: int,
    response: str,
    rate_name: str,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    season_label = SEASONS[season_code]

    season_data = (
        data.loc[data["Season"] == season_code]
        .copy()
        .reset_index(drop=True)
    )

    fold_metric_rows: List[Dict[str, object]] = []
    fit_status_rows: List[Dict[str, object]] = []
    prediction_frames: List[pd.DataFrame] = []

    for fold in range(1, 6):
        validation_mask = (
            season_data["Temporal_Fold"] == fold
        )
        training = (
            season_data.loc[~validation_mask]
            .copy()
            .reset_index(drop=True)
        )
        validation = (
            season_data.loc[validation_mask]
            .copy()
            .reset_index(drop=True)
        )

        if training.empty or validation.empty:
            raise ValueError(
                f"Empty temporal fold for "
                f"{season_label} {response}, fold {fold}."
            )

        x_training, x_validation, _ = (
            make_design_matrices(
                training,
                validation,
            )
        )

        y_training = (
            training[response]
            .to_numpy(dtype=float)
        )
        y_validation = (
            validation[response]
            .to_numpy(dtype=float)
        )
        exposure_training = (
            training["ForestPixelCount"]
            .to_numpy(dtype=float)
        )
        exposure_validation = (
            validation["ForestPixelCount"]
            .to_numpy(dtype=float)
        )

        poisson_result, poisson_status = fit_poisson(
            observed=y_training,
            design=x_training,
            exposure=exposure_training,
        )

        family_results: Dict[str, Tuple[object, Dict[str, object]]] = {
            "Poisson": (
                poisson_result,
                poisson_status,
            )
        }

        for family in ["NB1", "NB2"]:
            family_results[family] = (
                fit_negative_binomial(
                    observed=y_training,
                    design=x_training,
                    exposure=exposure_training,
                    family=family,
                    poisson_parameters=poisson_result.params,
                )
            )

        for family in MODEL_FAMILIES:
            result, status = family_results[family]

            (
                predicted_count,
                predicted_zero_probability,
                log_probability,
            ) = fitted_distribution_outputs(
                result=result,
                design=x_validation,
                exposure=exposure_validation,
                observed=y_validation,
                family=family,
                alpha=(
                    finite_or_nan(status.get("Alpha_Estimate", np.nan))
                    if family in {"NB1", "NB2"}
                    else None
                ),
            )

            metrics = prediction_metrics(
                observed_count=y_validation,
                predicted_count=predicted_count,
                exposure=exposure_validation,
                predicted_zero_probability=(
                    predicted_zero_probability
                ),
                log_probability=log_probability,
            )
            metrics.update(
                {
                    "Season_Code": season_code,
                    "Season_Label": season_label,
                    "Count_Response": response,
                    "Rate_Scale_Name": rate_name,
                    "Family": family,
                    "Fold": fold,
                    "Train_N": int(len(training)),
                    "Validation_N": int(len(validation)),
                    "Alpha_Estimate": (
                        finite_or_nan(
                            status.get("Alpha_Estimate", np.nan)
                        )
                        if family in {"NB1", "NB2"}
                        else np.nan
                    ),
                    "Converged": bool(
                        status.get("Converged", True)
                    ),
                    "Optimizer": status.get(
                        "Optimizer",
                        "",
                    ),
                }
            )
            fold_metric_rows.append(metrics)

            status_row = {
                "Season_Code": season_code,
                "Season_Label": season_label,
                "Count_Response": response,
                "Rate_Scale_Name": rate_name,
                "Family": family,
                "Fold": fold,
                "Train_N": int(len(training)),
            }
            status_row.update(status)
            fit_status_rows.append(status_row)

            prediction_frames.append(
                prediction_frame(
                    validation=validation,
                    response=response,
                    rate_name=rate_name,
                    family=family,
                    fold=fold,
                    predicted_count=predicted_count,
                    predicted_zero_probability=(
                        predicted_zero_probability
                    ),
                    log_probability=log_probability,
                )
            )

            log(
                f"Completed temporal fold {fold}: "
                f"{season_label} {response} {family}; "
                f"NLL={metrics['Mean_Negative_LogLikelihood']:.6f}; "
                f"rate RMSE={metrics['Rate_RMSE']:.8f}; "
                f"zero Brier={metrics['Zero_Probability_Brier']:.6f}."
            )

    return (
        pd.DataFrame(fold_metric_rows),
        pd.DataFrame(fit_status_rows),
        pd.concat(
            prediction_frames,
            ignore_index=True,
        ),
    )


def recover_existing_oof_checkpoint(
    prediction_path: Path,
    data: pd.DataFrame,
    season_code: int,
    response: str,
    rate_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Validate and summarize a previously completed OOF prediction file.

    This permits safe continuation after an interrupted run. Existing OOF
    files are reused only when all three families, all five folds, and the
    expected number of records are present. Fold-level alpha estimates are
    unavailable from old prediction-only checkpoints and are recorded as NA.
    """
    predictions = pd.read_csv(
        prediction_path,
        compression="gzip",
    )

    required = {
        "Season",
        "Count_Response",
        "Rate_Scale_Name",
        "Family",
        "Fold",
        "Observed_Count",
        "Predicted_Count",
        "ForestPixelCount",
        "Observed_Zero",
        "Predicted_Zero_Probability",
        "Predictive_LogProbability",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(
            "Existing OOF checkpoint is missing columns: "
            + ", ".join(missing)
        )

    expected_n = int((data["Season"] == season_code).sum())
    expected_rows = expected_n * len(MODEL_FAMILIES)

    if len(predictions) != expected_rows:
        raise ValueError(
            f"Existing OOF checkpoint has {len(predictions):,} rows; "
            f"expected {expected_rows:,}."
        )

    if set(predictions["Family"].astype(str)) != set(MODEL_FAMILIES):
        raise ValueError("Existing OOF checkpoint lacks one or more families.")

    if set(predictions["Fold"].astype(int)) != {1, 2, 3, 4, 5}:
        raise ValueError("Existing OOF checkpoint lacks one or more folds.")

    if set(predictions["Season"].astype(int)) != {season_code}:
        raise ValueError("Existing OOF checkpoint has the wrong season.")

    if set(predictions["Count_Response"].astype(str)) != {response}:
        raise ValueError("Existing OOF checkpoint has the wrong response.")

    if set(predictions["Rate_Scale_Name"].astype(str)) != {rate_name}:
        raise ValueError("Existing OOF checkpoint has the wrong rate scale.")

    fold_rows: List[Dict[str, object]] = []
    status_rows: List[Dict[str, object]] = []

    for (family, fold), group in predictions.groupby(
        ["Family", "Fold"],
        sort=True,
    ):
        metrics = prediction_metrics(
            observed_count=group["Observed_Count"].to_numpy(dtype=float),
            predicted_count=group["Predicted_Count"].to_numpy(dtype=float),
            exposure=group["ForestPixelCount"].to_numpy(dtype=float),
            predicted_zero_probability=group[
                "Predicted_Zero_Probability"
            ].to_numpy(dtype=float),
            log_probability=group[
                "Predictive_LogProbability"
            ].to_numpy(dtype=float),
        )
        metrics.update(
            {
                "Season_Code": season_code,
                "Season_Label": SEASONS[season_code],
                "Count_Response": response,
                "Rate_Scale_Name": rate_name,
                "Family": str(family),
                "Fold": int(fold),
                "Train_N": int(expected_n - len(group)),
                "Validation_N": int(len(group)),
                "Alpha_Estimate": np.nan,
                "Converged": True,
                "Optimizer": "Recovered_existing_OOF_checkpoint",
            }
        )
        fold_rows.append(metrics)

        status_rows.append(
            {
                "Season_Code": season_code,
                "Season_Label": SEASONS[season_code],
                "Count_Response": response,
                "Rate_Scale_Name": rate_name,
                "Family": str(family),
                "Fold": int(fold),
                "Train_N": int(expected_n - len(group)),
                "Optimizer": "Recovered_existing_OOF_checkpoint",
                "Alpha_Start": np.nan,
                "Converged": True,
                "Warning_Text": (
                    "Predictions recovered from a completed checkpoint; "
                    "fold-level fit metadata were not stored by the earlier run."
                ),
                "Alpha_Estimate": np.nan,
                "Attempt_Count": np.nan,
                "Failed_Attempt_Count": np.nan,
            }
        )

    return (
        pd.DataFrame(fold_rows),
        pd.DataFrame(status_rows),
        predictions,
    )


def pooled_oof_metrics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    group_columns = [
        "Season",
        "Count_Response",
        "Rate_Scale_Name",
        "Family",
    ]

    for (
        season_code,
        response,
        rate_name,
        family,
    ), group in predictions.groupby(
        group_columns,
        sort=True,
    ):
        metrics = prediction_metrics(
            observed_count=group[
                "Observed_Count"
            ].to_numpy(dtype=float),
            predicted_count=group[
                "Predicted_Count"
            ].to_numpy(dtype=float),
            exposure=group[
                "ForestPixelCount"
            ].to_numpy(dtype=float),
            predicted_zero_probability=group[
                "Predicted_Zero_Probability"
            ].to_numpy(dtype=float),
            log_probability=group[
                "Predictive_LogProbability"
            ].to_numpy(dtype=float),
        )

        fold_nll = (
            group.groupby("Fold")[
                "Predictive_LogProbability"
            ]
            .mean()
            .mul(-1.0)
        )
        fold_rate_rmse = (
            group.groupby("Fold")
            .apply(
                lambda subset: math.sqrt(
                    mean_squared_error(
                        subset["Observed_Rate"],
                        subset["Predicted_Rate"],
                    )
                ),
                include_groups=False,
            )
        )
        fold_zero_brier = (
            group.groupby("Fold")
            .apply(
                lambda subset: float(
                    np.mean(
                        (
                            subset[
                                "Predicted_Zero_Probability"
                            ]
                            - subset["Observed_Zero"]
                        ) ** 2
                    )
                ),
                include_groups=False,
            )
        )

        metrics.update(
            {
                "Season_Code": int(season_code),
                "Season_Label": SEASONS[
                    int(season_code)
                ],
                "Count_Response": response,
                "Rate_Scale_Name": rate_name,
                "Family": family,
                "Fold_NLL_Mean": float(
                    fold_nll.mean()
                ),
                "Fold_NLL_SD": float(
                    fold_nll.std(ddof=1)
                ),
                "Fold_NLL_SE": float(
                    fold_nll.std(ddof=1)
                    / math.sqrt(len(fold_nll))
                ),
                "Fold_Rate_RMSE_Mean": float(
                    fold_rate_rmse.mean()
                ),
                "Fold_Rate_RMSE_SD": float(
                    fold_rate_rmse.std(ddof=1)
                ),
                "Fold_Zero_Brier_Mean": float(
                    fold_zero_brier.mean()
                ),
                "Fold_Zero_Brier_SD": float(
                    fold_zero_brier.std(ddof=1)
                ),
            }
        )
        rows.append(metrics)

    return pd.DataFrame(rows)


# =============================================================================
# Full-data model fits
# =============================================================================

def full_data_fits(
    data: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fit_rows: List[Dict[str, object]] = []
    coefficient_rows: List[Dict[str, object]] = []
    status_rows: List[Dict[str, object]] = []

    terms = design_term_names()

    for season_code, season_label in SEASONS.items():
        season_data = (
            data.loc[data["Season"] == season_code]
            .copy()
            .reset_index(drop=True)
        )

        design, _ = full_data_design_matrix(
            season_data
        )
        exposure = (
            season_data["ForestPixelCount"]
            .to_numpy(dtype=float)
        )

        for response, rate_name in COUNT_RESPONSES.items():
            observed = (
                season_data[response]
                .to_numpy(dtype=float)
            )

            poisson_result, poisson_status = fit_poisson(
                observed=observed,
                design=design,
                exposure=exposure,
            )

            family_results: Dict[
                str,
                Tuple[object, Dict[str, object]],
            ] = {
                "Poisson": (
                    poisson_result,
                    poisson_status,
                )
            }

            for family in ["NB1", "NB2"]:
                family_results[family] = (
                    fit_negative_binomial(
                        observed=observed,
                        design=design,
                        exposure=exposure,
                        family=family,
                        poisson_parameters=(
                            poisson_result.params
                        ),
                    )
                )

            for family in MODEL_FAMILIES:
                result, status = family_results[family]

                (
                    predicted_count,
                    predicted_zero_probability,
                    log_probability,
                ) = fitted_distribution_outputs(
                    result=result,
                    design=design,
                    exposure=exposure,
                    observed=observed,
                    family=family,
                    alpha=(
                        finite_or_nan(status.get("Alpha_Estimate", np.nan))
                        if family in {"NB1", "NB2"}
                        else None
                    ),
                )

                metrics = prediction_metrics(
                    observed_count=observed,
                    predicted_count=predicted_count,
                    exposure=exposure,
                    predicted_zero_probability=(
                        predicted_zero_probability
                    ),
                    log_probability=log_probability,
                )

                metrics.update(
                    {
                        "Season_Code": season_code,
                        "Season_Label": season_label,
                        "Count_Response": response,
                        "Rate_Scale_Name": rate_name,
                        "Family": family,
                        "N_Parameters": int(
                            len(result.params)
                        ),
                        "LogLikelihood": float(
                            np.sum(log_probability)
                        ),
                        "AIC": float(
                            2.0 * len(result.params)
                            - 2.0 * np.sum(log_probability)
                        ),
                        "BIC": float(
                            math.log(len(observed))
                            * len(result.params)
                            - 2.0 * np.sum(log_probability)
                        ),
                        "Alpha_Estimate": (
                            finite_or_nan(
                                status.get("Alpha_Estimate", np.nan)
                            )
                            if family in {"NB1", "NB2"}
                            else np.nan
                        ),
                        "Converged": bool(
                            status.get(
                                "Converged",
                                True,
                            )
                        ),
                        "Optimizer": status.get(
                            "Optimizer",
                            "",
                        ),
                    }
                )
                fit_rows.append(metrics)

                parameter_values = np.asarray(
                    result.params,
                    dtype=float,
                )

                expected_parameter_count = len(terms)
                coefficient_parameter_values = (
                    parameter_values[
                        :expected_parameter_count
                    ]
                )

                try:
                    standard_errors = np.asarray(
                        result.bse,
                        dtype=float,
                    )[:expected_parameter_count]
                except Exception:
                    standard_errors = np.full(
                        expected_parameter_count,
                        np.nan,
                        dtype=float,
                    )

                if len(standard_errors) < expected_parameter_count:
                    standard_errors = np.pad(
                        standard_errors,
                        (
                            0,
                            expected_parameter_count
                            - len(standard_errors),
                        ),
                        constant_values=np.nan,
                    )

                for term, estimate, standard_error in zip(
                    terms,
                    coefficient_parameter_values,
                    standard_errors,
                ):
                    coefficient_rows.append(
                        {
                            "Season_Code": season_code,
                            "Season_Label": season_label,
                            "Count_Response": response,
                            "Rate_Scale_Name": rate_name,
                            "Family": family,
                            "Term": term,
                            "Coefficient": float(estimate),
                            "Standard_Error": float(
                                standard_error
                            ),
                            "CI95_Lower": float(
                                estimate
                                - 1.959963984540054
                                * standard_error
                            ),
                            "CI95_Upper": float(
                                estimate
                                + 1.959963984540054
                                * standard_error
                            ),
                            "Coefficient_Scale": (
                                "Log rate; per 1 SD of modeled predictor"
                                if term
                                in REFERENCE_CONTINUOUS_PREDICTORS
                                else (
                                    "Log rate; versus China"
                                    if term
                                    in COUNTRY_DUMMY_COLUMNS
                                    else "Log rate intercept"
                                )
                            ),
                        }
                    )

                status_row = {
                    "Season_Code": season_code,
                    "Season_Label": season_label,
                    "Count_Response": response,
                    "Rate_Scale_Name": rate_name,
                    "Family": family,
                    "Fold": "Full_Data",
                    "Train_N": int(len(season_data)),
                }
                status_row.update(status)
                status_rows.append(status_row)

                log(
                    f"Completed full-data fit: "
                    f"{season_label} {response} {family}; "
                    f"AIC={metrics['AIC']:.3f}; "
                    f"alpha={metrics['Alpha_Estimate']}."
                )

    return (
        pd.DataFrame(fit_rows),
        pd.DataFrame(coefficient_rows),
        pd.DataFrame(status_rows),
    )


# =============================================================================
# Ranking summaries
# =============================================================================

def family_ranking(
    pooled_metrics: pd.DataFrame,
) -> pd.DataFrame:
    ranking = pooled_metrics.copy()

    ranking["NLL_Rank"] = (
        ranking.groupby(
            ["Season_Code", "Count_Response"]
        )[
            "Mean_Negative_LogLikelihood"
        ]
        .rank(
            method="min",
            ascending=True,
        )
    )
    ranking["Zero_Brier_Rank"] = (
        ranking.groupby(
            ["Season_Code", "Count_Response"]
        )[
            "Zero_Probability_Brier"
        ]
        .rank(
            method="min",
            ascending=True,
        )
    )
    ranking["Rate_RMSE_Rank"] = (
        ranking.groupby(
            ["Season_Code", "Count_Response"]
        )[
            "Rate_RMSE"
        ]
        .rank(
            method="min",
            ascending=True,
        )
    )

    ranking["Family_Selection_Rank"] = (
        ranking.sort_values(
            [
                "Season_Code",
                "Count_Response",
                "Mean_Negative_LogLikelihood",
                "Zero_Probability_Brier",
                "Rate_RMSE",
                "Family",
            ]
        )
        .groupby(
            ["Season_Code", "Count_Response"]
        )
        .cumcount()
        .add(1)
        .sort_index()
    )

    return ranking.sort_values(
        [
            "Season_Code",
            "Count_Response",
            "Family_Selection_Rank",
        ]
    ).reset_index(drop=True)


def fold_win_counts(
    fold_metrics: pd.DataFrame,
) -> pd.DataFrame:
    ranked = fold_metrics.copy()

    ranked["Fold_NLL_Rank"] = (
        ranked.groupby(
            [
                "Season_Code",
                "Count_Response",
                "Fold",
            ]
        )[
            "Mean_Negative_LogLikelihood"
        ]
        .rank(
            method="min",
            ascending=True,
        )
    )

    summary = (
        ranked.assign(
            Fold_NLL_Win=(
                ranked["Fold_NLL_Rank"] == 1
            ).astype(int)
        )
        .groupby(
            [
                "Season_Code",
                "Season_Label",
                "Count_Response",
                "Rate_Scale_Name",
                "Family",
            ],
            as_index=False,
        )
        .agg(
            Temporal_Fold_NLL_Wins=(
                "Fold_NLL_Win",
                "sum",
            ),
            Mean_Fold_NLL=(
                "Mean_Negative_LogLikelihood",
                "mean",
            ),
            SD_Fold_NLL=(
                "Mean_Negative_LogLikelihood",
                "std",
            ),
            Mean_Fold_Zero_Brier=(
                "Zero_Probability_Brier",
                "mean",
            ),
            Mean_Fold_Rate_RMSE=(
                "Rate_RMSE",
                "mean",
            ),
        )
    )

    return summary.sort_values(
        [
            "Season_Code",
            "Count_Response",
            "Temporal_Fold_NLL_Wins",
            "Mean_Fold_NLL",
        ],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)


def provisional_best_families(
    ranking: pd.DataFrame,
) -> pd.DataFrame:
    best = ranking.loc[
        ranking["Family_Selection_Rank"] == 1
    ].copy()

    best["Status"] = (
        "Provisional single-stage family; "
        "review required before predictor optimization"
    )
    best["Selection_Primary_Criterion"] = (
        "Lowest pooled temporal OOF mean negative log-likelihood"
    )
    best["Selection_Secondary_Criteria"] = (
        "Zero-probability Brier score; rate-scale RMSE"
    )

    return best.sort_values(
        [
            "Season_Code",
            "Count_Response",
        ]
    ).reset_index(drop=True)


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Preserve completed OOF checkpoints so an interrupted run can resume.
    # All other files are regenerated to prevent stale summary tables.
    for existing in list(OUTPUT_ROOT.iterdir()):
        if (
            existing.is_file()
            and existing.name.startswith("Temporal_OOF_Predictions_")
            and existing.name.endswith(".csv.gz")
        ):
            continue

        if existing.is_dir():
            shutil.rmtree(existing)
        else:
            existing.unlink()

    log(
        "Starting single-stage count-family comparison."
    )
    log(f"Input: {INPUT_CSV}")
    log(f"Output: {OUTPUT_ROOT}")
    log(
        "Families: Poisson, NB1, NB2."
    )
    log(
        "Exposure: ForestPixelCount."
    )
    log(
        "Reference predictors: "
        + ", ".join(
            REFERENCE_CONTINUOUS_PREDICTORS
        )
    )

    data, excluded = prepare_data()

    vif = reference_set_vif(data)

    if float(vif["VIF"].max()) >= 10.0:
        raise ValueError(
            "Reference predictor set has maximum VIF >= 10. "
            "Family comparison should not proceed."
        )

    all_fold_metrics: List[pd.DataFrame] = []
    all_fit_status: List[pd.DataFrame] = []
    pooled_metric_tables: List[pd.DataFrame] = []

    for season_code, season_label in SEASONS.items():
        for response, rate_name in COUNT_RESPONSES.items():
            log(
                f"Starting temporal CV: "
                f"{season_label} {response}."
            )

            prediction_path = (
                OUTPUT_ROOT
                / (
                    "Temporal_OOF_Predictions_"
                    f"S{season_code}_{response}.csv.gz"
                )
            )

            reused_checkpoint = False
            if prediction_path.exists():
                try:
                    (
                        fold_metrics,
                        fit_status,
                        predictions,
                    ) = recover_existing_oof_checkpoint(
                        prediction_path=prediction_path,
                        data=data,
                        season_code=season_code,
                        response=response,
                        rate_name=rate_name,
                    )
                    reused_checkpoint = True
                    log(
                        f"Reused validated OOF checkpoint: "
                        f"{prediction_path.name}."
                    )
                except Exception as error:
                    log(
                        f"Existing checkpoint rejected and will be rebuilt: "
                        f"{prediction_path.name}; "
                        f"{type(error).__name__}: {error}"
                    )
                    prediction_path.unlink(missing_ok=True)

            if not reused_checkpoint:
                (
                    fold_metrics,
                    fit_status,
                    predictions,
                ) = run_one_season_response(
                    data=data,
                    season_code=season_code,
                    response=response,
                    rate_name=rate_name,
                )

                predictions.to_csv(
                    prediction_path,
                    index=False,
                    compression="gzip",
                )

                log(
                    f"Saved OOF predictions: "
                    f"{prediction_path.name}."
                )

            all_fold_metrics.append(fold_metrics)
            all_fit_status.append(fit_status)

            pooled = pooled_oof_metrics(
                predictions
            )
            pooled_metric_tables.append(pooled)

    fold_metrics = pd.concat(
        all_fold_metrics,
        ignore_index=True,
    )
    fit_status = pd.concat(
        all_fit_status,
        ignore_index=True,
    )
    pooled_metrics = pd.concat(
        pooled_metric_tables,
        ignore_index=True,
    )

    (
        full_fit_summary,
        full_coefficients,
        full_fit_status,
    ) = full_data_fits(data)

    fit_status = pd.concat(
        [
            fit_status,
            full_fit_status,
        ],
        ignore_index=True,
    )

    ranking = family_ranking(
        pooled_metrics
    )
    fold_wins = fold_win_counts(
        fold_metrics
    )
    provisional_best = provisional_best_families(
        ranking
    )

    reference_definition = pd.DataFrame(
        [
            {
                "Variable": variable,
                "Domain": (
                    REFERENCE_PREDICTOR_DOMAINS[
                        variable
                    ]
                ),
                "Transformation": (
                    "log1p_then_training_fold_z_standardization"
                    if variable in LOG1P_PREDICTORS
                    else "training_fold_z_standardization"
                ),
                "Role": (
                    "Prespecified family-screening reference covariate"
                ),
            }
            for variable in REFERENCE_CONTINUOUS_PREDICTORS
        ]
        + [
            {
                "Variable": "Country_NK",
                "Domain": "Country_context",
                "Transformation": "Binary indicator",
                "Role": "Fixed effect; China reference",
            },
            {
                "Variable": "Country_Russia",
                "Domain": "Country_context",
                "Transformation": "Binary indicator",
                "Role": "Fixed effect; China reference",
            },
            {
                "Variable": "ForestPixelCount",
                "Domain": "Exposure",
                "Transformation": (
                    "Automatically log-transformed by model exposure"
                ),
                "Role": (
                    "Exposure with coefficient fixed at one"
                ),
            },
        ]
    )

    qa = pd.DataFrame(
        [
            (
                "Rows_in_common_complete_case_sample",
                len(data),
            ),
            (
                "Excluded_incomplete_reference_set_rows",
                len(excluded),
            ),
            (
                "Unique_GRID_UIDs",
                data["GRID_UID"].nunique(),
            ),
            (
                "Year_min",
                int(data["Year"].min()),
            ),
            (
                "Year_max",
                int(data["Year"].max()),
            ),
            (
                "Reference_continuous_predictor_count",
                len(
                    REFERENCE_CONTINUOUS_PREDICTORS
                ),
            ),
            (
                "Reference_set_maximum_VIF",
                float(vif["VIF"].max()),
            ),
            (
                "Compared_families",
                ";".join(MODEL_FAMILIES),
            ),
            (
                "Exposure",
                "ForestPixelCount",
            ),
            (
                "Equivalent_offset",
                "log(ForestPixelCount)",
            ),
            (
                "Temporal_fold_count",
                5,
            ),
            (
                "Primary_family_selection_metric",
                "Pooled temporal OOF mean negative log-likelihood",
            ),
            (
                "Secondary_family_selection_metrics",
                "Zero-probability Brier; rate-scale RMSE",
            ),
            (
                "Random_forest_variable_selection_used",
                False,
            ),
        ],
        columns=["Metric", "Value"],
    )

    method_definition = {
        "workflow_stage":
            "Predictor-adjusted single-stage count-family comparison",
        "families": {
            "Poisson":
                "Variance equals conditional mean",
            "NB1":
                "Variance = mu + alpha * mu",
            "NB2":
                "Variance = mu + alpha * mu^2",
        },
        "responses": {
            "Fire_Count":
                "Seasonal active-fire detection count",
            "Burned_Pixel_Count":
                "Seasonal burned-pixel detection count",
        },
        "rate_scales": {
            "FCD":
                "Fire_Count / ForestPixelCount",
            "BAD":
                "Burned_Pixel_Count / ForestPixelCount",
        },
        "exposure":
            "ForestPixelCount",
        "equivalent_offset":
            "log(ForestPixelCount)",
        "reference_predictors":
            REFERENCE_CONTINUOUS_PREDICTORS,
        "log1p_predictors":
            sorted(LOG1P_PREDICTORS),
        "country_effect":
            "Fixed contextual effect with China as reference",
        "temporal_folds":
            TEMPORAL_FOLDS,
        "primary_selection_metric":
            "Pooled temporal OOF mean negative log-likelihood",
        "secondary_selection_metrics": [
            "Zero-probability Brier score",
            "Rate-scale RMSE",
        ],
        "scope_note":
            "The reference predictor set is used only to isolate "
            "distribution-family performance. Predictor representations "
            "and final predictor sets are optimized after the family is "
            "reviewed and provisionally selected.",
        "uses_random_forest_variable_selection":
            False,
        "final_inference_stage":
            "Step 06d GRID_UID-Year two-way clustered inference",
        "zero_inflation_or_hurdle_included":
            False,
        "poisson_fitting_method":
            "Discrete Poisson Newton MLE with GLM IRLS pseudoinverse fallback after Hessian singularity or nonconvergence",
        "resume_behavior":
            "Validated completed OOF prediction files are reused after an interrupted run.",
    }

    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "statsmodels": statsmodels.__version__,
        "run_timestamp":
            datetime.now().isoformat(
                timespec="seconds"
            ),
    }

    with (
        OUTPUT_ROOT / "00_Method_Definition.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(
            method_definition,
            handle,
            indent=2,
        )

    qa.to_csv(
        OUTPUT_ROOT
        / "01_Data_QA_Summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    reference_definition.to_csv(
        OUTPUT_ROOT
        / "02_Reference_Predictor_Set.csv",
        index=False,
        encoding="utf-8-sig",
    )
    vif.to_csv(
        OUTPUT_ROOT
        / "03_Reference_Set_VIF.csv",
        index=False,
        encoding="utf-8-sig",
    )
    fold_metrics.to_csv(
        OUTPUT_ROOT
        / "04_Temporal_Fold_Metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pooled_metrics.to_csv(
        OUTPUT_ROOT
        / "05_Pooled_Temporal_OOF_Metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    ranking.to_csv(
        OUTPUT_ROOT
        / "06_Family_Ranking.csv",
        index=False,
        encoding="utf-8-sig",
    )
    fold_wins.to_csv(
        OUTPUT_ROOT
        / "07_Temporal_Fold_Win_Counts.csv",
        index=False,
        encoding="utf-8-sig",
    )
    provisional_best.to_csv(
        OUTPUT_ROOT
        / "08_Provisional_Best_Single_Stage_Family.csv",
        index=False,
        encoding="utf-8-sig",
    )
    full_fit_summary.to_csv(
        OUTPUT_ROOT
        / "09_Full_Data_Fit_Summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    full_coefficients.to_csv(
        OUTPUT_ROOT
        / "10_Full_Data_Coefficients.csv",
        index=False,
        encoding="utf-8-sig",
    )
    fit_status.to_csv(
        OUTPUT_ROOT
        / "11_Model_Fit_Status.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with (
        OUTPUT_ROOT / "Software_Environment.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(
            environment,
            handle,
            indent=2,
        )

    log(
        "Single-stage count-family comparison "
        "completed successfully."
    )
    log(
        "Next step: review Poisson/NB1/NB2 family results "
        "before count-model predictor optimization."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
