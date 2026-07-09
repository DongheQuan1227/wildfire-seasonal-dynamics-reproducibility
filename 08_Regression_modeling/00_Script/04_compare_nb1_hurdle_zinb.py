# -*- coding: utf-8 -*-
"""
Matched temporal-CV comparison of single-stage NB1, hurdle NB1, and ZINB1 (stable truncated-NB1 revision).

Recommended public location
---------------------------
<REPOSITORY_ROOT>/08_Regression_modeling/00_Script/
    04_compare_nb1_hurdle_zinb.py

Inputs
------
<REPOSITORY_ROOT>/00_Input_data/Base_Table_2001_2025.csv.bz2
<REPOSITORY_ROOT>/08_Regression_modeling/
    03_Single_Stage_NB1_Configuration_Comparison/
        09_Provisional_Best_NB1_Configuration.csv

Output
------
<REPOSITORY_ROOT>/08_Regression_modeling/
    04_NB1_Hurdle_ZINB_Matched_Comparison/

All three models use the same records, predictors, transformations,
Country fixed effects, exposure, and five temporal folds within each
season-response combination.

Count responses:
    Fire_Count
    Burned_Pixel_Count

Exposure:
    ForestPixelCount

Equivalent count-model offset:
    log(ForestPixelCount)

Compared structures:
1. Single-stage NB1.
2. Hurdle NB1: binomial-logit occurrence model plus zero-truncated NB1
   positive-count model.
3. ZINB1: logit structural-zero process plus an NB1 count process that
   can also produce zeros.

The primary metric is pooled temporal out-of-fold mean negative
log-likelihood. Secondary metrics are zero-probability Brier score and
rate-scale RMSE.
"""

from __future__ import annotations

import json
import math
import platform
import shutil
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy
import sklearn
import statsmodels
from scipy import special
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
from statsmodels.discrete.discrete_model import NegativeBinomial
from statsmodels.genmod.families import Binomial, Poisson as PoissonFamily
from statsmodels.genmod.generalized_linear_model import GLM


# =============================================================================
# Paths
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REGRESSION_ROOT = SCRIPT_DIR.parent
CODE_ROOT = REGRESSION_ROOT.parent

INPUT_CSV = CODE_ROOT / "00_Input_data" / "Base_Table_2001_2025.csv.bz2"

BEST_CONFIGURATION_CSV = (
    REGRESSION_ROOT
    / "03_Single_Stage_NB1_Configuration_Comparison"
    / "09_Provisional_Best_NB1_Configuration.csv"
)

OUTPUT_ROOT = REGRESSION_ROOT / "04_NB1_Hurdle_ZINB_Matched_Comparison"
CHECKPOINT_ROOT = OUTPUT_ROOT / "_checkpoints"
LOG_FILE = OUTPUT_ROOT / "nb1_hurdle_zinb_matched_comparison.log"


# =============================================================================
# Lightweight result container for stable custom likelihood fits
# =============================================================================

@dataclass
class StableOptimizationResult:
    params: np.ndarray
    bse: np.ndarray
    converged: bool
    mle_retvals: Dict[str, object]


# =============================================================================
# Settings
# =============================================================================

FOREST_PIXEL_AREA_KM2 = 0.25
FOREST_AREA_THRESHOLD_KM2 = 2.5
PROBABILITY_EPSILON = 1e-12
FIT_MAX_ITERATIONS = 1000

SEASONS = {1: "Spring", 2: "Summer", 3: "Autumn"}
COUNT_RESPONSES = {
    "Fire_Count": "FCD",
    "Burned_Pixel_Count": "BAD",
}
MODEL_STRUCTURES = ["Single_Stage_NB1", "Hurdle_NB1", "ZINB1"]

TEMPORAL_FOLDS = {
    1: [2024, 2002, 2006, 2004, 2009],
    2: [2016, 2015, 2007, 2018, 2014],
    3: [2025, 2021, 2013, 2019, 2003],
    4: [2010, 2023, 2017, 2011, 2005],
    5: [2020, 2022, 2001, 2012, 2008],
}

COUNTRY_LEVELS = ["China", "NK", "Russia"]
COUNTRY_DUMMY_COLUMNS = ["Country_NK", "Country_Russia"]

AUDIT_GUIDED_LOG1P_VARIABLES = {
    "ND", "NE", "LtgProxy", "POP", "Dis_Farm", "Road_dens"
}

NB_ALPHA_STARTS = [0.1, 1.0, 10.0, 50.0, 100.0]
COUNT_OPTIMIZERS = ["bfgs", "lbfgs"]
ZINB_OPTIMIZERS = ["bfgs", "lbfgs"]


# =============================================================================
# Logging and general helpers
# =============================================================================

def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def parse_predictors(value: str) -> List[str]:
    result = [item.strip() for item in str(value).split(";") if item.strip()]
    if not result:
        raise ValueError("No predictors were defined.")
    return result


def temporal_fold_mapping() -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    for fold, years in TEMPORAL_FOLDS.items():
        for year in years:
            year = int(year)
            if year in mapping:
                raise ValueError(f"Year {year} appears in multiple folds.")
            mapping[year] = int(fold)
    return mapping


def warning_text(captured: Sequence[warnings.WarningMessage]) -> str:
    messages: List[str] = []
    for item in captured:
        message = f"{item.category.__name__}: {str(item.message)}"
        if message not in messages:
            messages.append(message)
    return " | ".join(messages)


def convergence_status(result: object) -> bool:
    values = getattr(result, "mle_retvals", {})
    if isinstance(values, dict):
        if "converged" in values:
            return bool(values["converged"])
        if "warnflag" in values:
            return int(values["warnflag"]) == 0
    return bool(getattr(result, "converged", True))


def finite_or_nan(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return np.nan
    return numeric if np.isfinite(numeric) else np.nan


def safe_r2(observed: np.ndarray, predicted: np.ndarray) -> float:
    if len(observed) < 2 or float(np.var(observed)) <= 0:
        return np.nan
    return float(r2_score(observed, predicted))


def safe_roc_auc(observed_binary: np.ndarray, probability: np.ndarray) -> float:
    if len(np.unique(observed_binary)) < 2:
        return np.nan
    return float(roc_auc_score(observed_binary, probability))


def safe_pr_auc(observed_binary: np.ndarray, probability: np.ndarray) -> float:
    if len(np.unique(observed_binary)) < 2:
        return np.nan
    return float(average_precision_score(observed_binary, probability))


def calibration_parameters(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> Tuple[float, float]:
    if len(observed) < 3 or float(np.std(predicted)) <= 0:
        return np.nan, np.nan
    model = LinearRegression()
    model.fit(predicted.reshape(-1, 1), observed)
    return float(model.intercept_), float(model.coef_[0])


def stable_expit(linear_predictor: np.ndarray) -> np.ndarray:
    probability = special.expit(np.asarray(linear_predictor, dtype=float))
    return np.clip(
        probability,
        PROBABILITY_EPSILON,
        1.0 - PROBABILITY_EPSILON,
    )


def stable_log_one_minus_exp(log_probability: np.ndarray) -> np.ndarray:
    """Compute log(1-exp(x)) for x <= 0."""
    x = np.asarray(log_probability, dtype=float)
    result = np.empty_like(x)
    low = x < -math.log(2.0)
    result[low] = np.log1p(-np.exp(x[low]))
    result[~low] = np.log(-np.expm1(x[~low]))
    return result


# =============================================================================
# Input validation
# =============================================================================

def validate_best_configurations() -> pd.DataFrame:
    if not BEST_CONFIGURATION_CSV.exists():
        raise FileNotFoundError(
            "Step-03 best-configuration table not found:\n"
            f"{BEST_CONFIGURATION_CSV}"
        )

    table = pd.read_csv(BEST_CONFIGURATION_CSV)
    required = {
        "Season_Code", "Season_Label", "Count_Response", "Rate_Scale_Name",
        "Config_ID", "Transformation_Scheme", "Continuous_Predictors",
        "Maximum_VIF", "All_Folds_Converged", "Eligible_for_Selection",
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(
            "Step-03 table is missing columns:\n" + "\n".join(missing)
        )

    expected = {
        (season, response)
        for season in SEASONS
        for response in COUNT_RESPONSES
    }
    observed = {
        (int(row.Season_Code), str(row.Count_Response))
        for row in table.itertuples(index=False)
    }
    if observed != expected:
        raise ValueError(
            "Step-03 table does not contain exactly the six expected models."
        )

    if not table["All_Folds_Converged"].astype(bool).all():
        raise ValueError("A selected Step-03 configuration did not converge.")
    if not table["Eligible_for_Selection"].astype(bool).all():
        raise ValueError("A selected Step-03 configuration was ineligible.")
    if (table["Maximum_VIF"].astype(float) >= 10.0).any():
        raise ValueError("A selected Step-03 configuration has VIF >= 10.")

    return table.sort_values(
        ["Season_Code", "Count_Response"]
    ).reset_index(drop=True)


def prepare_data(
    best_configurations: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Base table not found:\n{INPUT_CSV}")

    data = pd.read_csv(INPUT_CSV)
    all_predictors = sorted({
        predictor
        for value in best_configurations["Continuous_Predictors"]
        for predictor in parse_predictors(value)
    })

    required = {
        "GRID_UID", "GRID_ID", "Country", "Year", "Season",
        "Fire_Count", "Burned_Pixel_Count", "ForestPixelCount",
        "ForestArea_km2", *all_predictors,
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(
            "Base table is missing required columns:\n" + "\n".join(missing)
        )

    duplicate = data.duplicated(["GRID_UID", "Year", "Season"], keep=False)
    if duplicate.any():
        data.loc[duplicate].to_csv(
            OUTPUT_ROOT / "ERROR_Duplicate_Panel_Records.csv",
            index=False,
            encoding="utf-8-sig",
        )
        raise ValueError("Duplicate GRID_UID-Year-Season records found.")

    exposure = pd.to_numeric(data["ForestPixelCount"], errors="coerce")
    area = pd.to_numeric(data["ForestArea_km2"], errors="coerce")
    if exposure.isna().any() or area.isna().any():
        raise ValueError("Exposure or forest area contains nonnumeric values.")
    if float(np.abs(area - exposure * FOREST_PIXEL_AREA_KM2).max()) > 1e-9:
        raise ValueError("ForestArea_km2 != ForestPixelCount * 0.25.")

    data = data.loc[area > FOREST_AREA_THRESHOLD_KM2].copy()
    if (data["ForestPixelCount"].astype(float) <= 0).any():
        raise ValueError("ForestPixelCount must be positive.")

    unexpected_country = sorted(
        set(data["Country"].dropna().astype(str)) - set(COUNTRY_LEVELS)
    )
    if unexpected_country:
        raise ValueError(
            "Unexpected Country values: " + ", ".join(unexpected_country)
        )

    for response in COUNT_RESPONSES:
        values = pd.to_numeric(data[response], errors="coerce")
        invalid = (
            values.isna()
            | ~np.isfinite(values)
            | (values < 0)
            | (np.abs(values - np.round(values)) > 1e-9)
        )
        if invalid.any():
            raise ValueError(f"{response} contains invalid count values.")
        data[response] = values.astype(int)

    data["Temporal_Fold"] = (
        data["Year"].astype(int).map(temporal_fold_mapping())
    )
    if data["Temporal_Fold"].isna().any():
        raise ValueError("Some years are missing from temporal folds.")

    complete_columns = [
        "Country", "Fire_Count", "Burned_Pixel_Count",
        "ForestPixelCount", *all_predictors,
    ]
    complete = (
        data[complete_columns]
        .replace([np.inf, -np.inf], np.nan)
        .notna()
        .all(axis=1)
    )

    excluded = data.loc[
        ~complete,
        ["GRID_UID", "GRID_ID", "Country", "Year", "Season", *all_predictors],
    ].copy()
    excluded.to_csv(
        OUTPUT_ROOT / "Excluded_Incomplete_Matched_Comparison_Records.csv",
        index=False,
        encoding="utf-8-sig",
    )

    data = data.loc[complete].copy().reset_index(drop=True)
    data["Temporal_Fold"] = data["Temporal_Fold"].astype(int)
    return data, excluded


# =============================================================================
# Predictor matrices
# =============================================================================

def transform_predictors(
    frame: pd.DataFrame,
    predictors: Sequence[str],
    scheme: str,
) -> pd.DataFrame:
    transformed = frame.loc[:, predictors].astype(float).copy()
    if scheme == "Raw":
        return transformed
    if scheme != "Audit_Guided_Log1p":
        raise ValueError(f"Unknown transformation scheme: {scheme}")

    for variable in predictors:
        if variable not in AUDIT_GUIDED_LOG1P_VARIABLES:
            continue
        minimum = float(transformed[variable].min())
        if minimum < 0:
            raise ValueError(f"{variable} cannot use log1p; minimum={minimum}.")
        transformed[variable] = np.log1p(
            transformed[variable].to_numpy(dtype=float)
        )
    return transformed


def country_dummies(frame: pd.DataFrame) -> np.ndarray:
    country = frame["Country"].astype(str)
    return np.column_stack([
        (country == "NK").astype(float).to_numpy(),
        (country == "Russia").astype(float).to_numpy(),
    ])


def design_matrices(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    predictors: Sequence[str],
    scheme: str,
) -> Tuple[np.ndarray, np.ndarray]:
    train_continuous = transform_predictors(training, predictors, scheme)
    valid_continuous = transform_predictors(validation, predictors, scheme)

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_continuous.to_numpy(dtype=float))
    valid_scaled = scaler.transform(valid_continuous.to_numpy(dtype=float))

    x_train = np.column_stack([
        np.ones(len(training), dtype=float),
        train_scaled,
        country_dummies(training),
    ])
    x_valid = np.column_stack([
        np.ones(len(validation), dtype=float),
        valid_scaled,
        country_dummies(validation),
    ])
    return x_train, x_valid


# =============================================================================
# Stable count distribution calculations
# =============================================================================

def count_mean(
    parameters: np.ndarray,
    design: np.ndarray,
    exposure: np.ndarray,
) -> np.ndarray:
    beta = np.asarray(parameters, dtype=float)[: design.shape[1]]
    eta = design @ beta + np.log(np.asarray(exposure, dtype=float))
    mean = np.exp(np.clip(eta, -745.0, 700.0))
    if not np.isfinite(mean).all():
        raise FloatingPointError("Predicted count mean is non-finite.")
    return mean


def nb1_logpmf_and_logzero(
    observed: np.ndarray,
    mean: np.ndarray,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray]:
    observed = np.asarray(observed, dtype=float)
    mean = np.clip(np.asarray(mean, dtype=float), np.finfo(float).tiny, 1e300)
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("NB1 alpha must be finite and positive.")

    size = np.clip(mean / alpha, np.finfo(float).tiny, 1e300)
    log_success = -math.log1p(alpha)
    log_failure = math.log(alpha) - math.log1p(alpha)

    log_probability = np.empty_like(mean)
    zero = observed == 0
    positive = ~zero
    log_probability[zero] = size[zero] * log_success

    if positive.any():
        y = observed[positive]
        r = size[positive]
        log_probability[positive] = (
            special.gammaln(y + r)
            - special.gammaln(r)
            - special.gammaln(y + 1.0)
            + r * log_success
            + y * log_failure
        )

    log_zero = size * log_success
    if not np.isfinite(log_probability).all():
        raise FloatingPointError("NB1 log probability is non-finite.")
    return log_probability, log_zero


# =============================================================================
# Model fitting
# =============================================================================

def fit_poisson_start(
    observed: np.ndarray,
    design: np.ndarray,
    exposure: np.ndarray,
) -> np.ndarray:
    model = GLM(
        observed,
        design,
        family=PoissonFamily(),
        offset=np.log(exposure),
        missing="raise",
    )
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("ignore")
        result = model.fit(
            method="irls",
            maxiter=300,
            tol=1e-8,
            wls_method="pinv",
            disp=False,
        )
    parameters = np.asarray(result.params, dtype=float)
    if not np.isfinite(parameters).all():
        raise RuntimeError("Poisson starting coefficients are non-finite.")
    return parameters


def fit_binary_glm(
    target: np.ndarray,
    design: np.ndarray,
) -> Tuple[object, Dict[str, object]]:
    model = GLM(target, design, family=Binomial(), missing="raise")
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        result = model.fit(
            method="irls",
            maxiter=500,
            tol=1e-8,
            wls_method="pinv",
            disp=False,
        )
    parameters = np.asarray(result.params, dtype=float)
    if not np.isfinite(parameters).all():
        raise RuntimeError("Binary GLM parameters are non-finite.")
    return result, {
        "Optimizer": "GLM_IRLS_pinv",
        "Converged": bool(getattr(result, "converged", True)),
        "Warning_Text": warning_text(captured),
    }


def fit_nb1(
    observed: np.ndarray,
    design: np.ndarray,
    exposure: np.ndarray,
) -> Tuple[object, Dict[str, object]]:
    poisson_parameters = fit_poisson_start(observed, design, exposure)
    model = NegativeBinomial(
        observed,
        design,
        loglike_method="nb1",
        exposure=exposure,
        missing="raise",
        check_rank=False,
    )

    attempts: List[Dict[str, object]] = []
    best_result = None
    best_status = None
    best_ll = -np.inf

    for alpha_start in NB_ALPHA_STARTS:
        start = np.concatenate([poisson_parameters, [alpha_start]])
        for optimizer in COUNT_OPTIMIZERS:
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                try:
                    result = model.fit(
                        start_params=start,
                        method=optimizer,
                        maxiter=FIT_MAX_ITERATIONS,
                        disp=0,
                        full_output=True,
                    )
                    params = np.asarray(result.params, dtype=float)
                    converged = convergence_status(result)
                    alpha = finite_or_nan(params[-1])
                    stable_ll = np.nan
                    if (
                        converged
                        and np.isfinite(params).all()
                        and np.isfinite(alpha)
                        and alpha > 0
                    ):
                        mean = count_mean(params, design, exposure)
                        logp, _ = nb1_logpmf_and_logzero(
                            observed, mean, alpha
                        )
                        stable_ll = float(logp.sum())
                    valid = bool(np.isfinite(stable_ll))
                    status = {
                        "Optimizer": optimizer,
                        "Alpha_Start": alpha_start,
                        "Converged": converged,
                        "Alpha_Estimate": alpha,
                        "Stable_LogLikelihood": stable_ll,
                        "Warning_Text": warning_text(captured),
                        "Valid_Fit": valid,
                    }
                    attempts.append(status)
                    if valid and stable_ll > best_ll:
                        best_result = result
                        best_status = status
                        best_ll = stable_ll
                        break
                except Exception as error:
                    attempts.append({
                        "Optimizer": optimizer,
                        "Alpha_Start": alpha_start,
                        "Converged": False,
                        "Alpha_Estimate": np.nan,
                        "Stable_LogLikelihood": np.nan,
                        "Warning_Text": f"{type(error).__name__}: {error}",
                        "Valid_Fit": False,
                    })
        if best_result is not None:
            break

    if best_result is None or best_status is None:
        raise RuntimeError(
            "All NB1 fitting attempts failed:\n"
            + "\n".join(json.dumps(item, default=str) for item in attempts)
        )

    best_status = dict(best_status)
    best_status["Attempt_Count"] = len(attempts)
    best_status["Failed_Attempt_Count"] = sum(
        not bool(item["Valid_Fit"]) for item in attempts
    )
    return best_result, best_status


def single_stage_outputs(
    result: object,
    observed: np.ndarray,
    design: np.ndarray,
    exposure: np.ndarray,
) -> Dict[str, np.ndarray | float]:
    params = np.asarray(result.params, dtype=float)
    alpha = float(params[-1])
    mean = count_mean(params, design, exposure)
    logp, logzero = nb1_logpmf_and_logzero(observed, mean, alpha)
    zero_probability = np.exp(np.clip(logzero, -745.0, 0.0))
    return {
        "Predicted_Count": mean,
        "Predicted_Zero_Probability": zero_probability,
        "Predictive_LogProbability": logp,
        "Alpha": alpha,
    }


def truncated_nb1_objective_and_gradient(
    transformed_parameters: np.ndarray,
    observed_positive: np.ndarray,
    design_positive: np.ndarray,
    exposure_positive: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """Stable zero-truncated NB1 mean negative log-likelihood and gradient."""
    y = np.asarray(observed_positive, dtype=float)
    x = np.asarray(design_positive, dtype=float)
    exposure = np.asarray(exposure_positive, dtype=float)

    coefficient_count = x.shape[1]
    beta = np.asarray(
        transformed_parameters[:coefficient_count],
        dtype=float,
    )
    alpha = float(
        np.exp(transformed_parameters[coefficient_count])
    )

    linear_predictor = (
        x @ beta + np.log(exposure)
    )
    linear_predictor = np.clip(
        linear_predictor,
        -745.0,
        700.0,
    )
    mean = np.clip(
        np.exp(linear_predictor),
        np.finfo(float).tiny,
        1e300,
    )

    size = np.clip(
        mean / alpha,
        np.finfo(float).tiny,
        1e300,
    )
    log_success = -math.log1p(alpha)
    log_failure = (
        math.log(alpha) - math.log1p(alpha)
    )
    success_probability = 1.0 / (1.0 + alpha)
    failure_probability = alpha / (1.0 + alpha)

    digamma_difference = (
        special.digamma(y + size)
        - special.digamma(size)
        + log_success
    )

    base_log_probability = (
        special.gammaln(y + size)
        - special.gammaln(size)
        - special.gammaln(y + 1.0)
        + size * log_success
        + y * log_failure
    )
    log_zero_probability = size * log_success
    log_nonzero_probability = stable_log_one_minus_exp(
        log_zero_probability
    )
    truncated_log_probability = (
        base_log_probability
        - log_nonzero_probability
    )

    if not np.isfinite(truncated_log_probability).all():
        return 1e100, np.zeros_like(transformed_parameters)

    zero_to_nonzero_odds = np.exp(
        np.clip(
            log_zero_probability
            - log_nonzero_probability,
            -745.0,
            700.0,
        )
    )

    beta_score_multiplier = size * (
        digamma_difference
        + zero_to_nonzero_odds * log_success
    )
    beta_score = x.T @ beta_score_multiplier

    log_alpha_score = (
        -size * digamma_difference
        - size * failure_probability
        + y * success_probability
        + zero_to_nonzero_odds
        * (
            -size
            * (log_success + failure_probability)
        )
    )

    score = np.concatenate(
        [
            beta_score,
            np.array(
                [float(log_alpha_score.sum())],
                dtype=float,
            ),
        ]
    )

    sample_size = float(len(y))
    objective = float(
        -truncated_log_probability.sum()
        / sample_size
    )
    gradient = -score / sample_size

    if not np.isfinite(objective):
        return 1e100, np.zeros_like(transformed_parameters)

    if not np.isfinite(gradient).all():
        return 1e100, np.zeros_like(transformed_parameters)

    return objective, gradient


def approximate_bse_from_lbfgs(
    optimization_result: object,
    sample_size: int,
    alpha: float,
) -> np.ndarray:
    """Approximate coefficient SEs from the L-BFGS inverse Hessian."""
    parameter_count = len(
        np.asarray(optimization_result.x, dtype=float)
    )

    try:
        inverse_hessian = optimization_result.hess_inv
        if hasattr(inverse_hessian, "todense"):
            inverse_hessian = inverse_hessian.todense()
        inverse_hessian = np.asarray(
            inverse_hessian,
            dtype=float,
        )

        if inverse_hessian.shape != (
            parameter_count,
            parameter_count,
        ):
            raise ValueError(
                "Unexpected inverse-Hessian shape."
            )

        covariance = inverse_hessian / float(sample_size)
        standard_errors = np.sqrt(
            np.clip(
                np.diag(covariance),
                0.0,
                np.inf,
            )
        )
        standard_errors[-1] = (
            alpha * standard_errors[-1]
        )

        if not np.isfinite(standard_errors).all():
            raise ValueError(
                "Approximate standard errors are non-finite."
            )

        return standard_errors

    except Exception:
        return np.full(
            parameter_count,
            np.nan,
            dtype=float,
        )


def fit_truncated_nb1(
    observed_positive: np.ndarray,
    design_positive: np.ndarray,
    exposure_positive: np.ndarray,
) -> Tuple[object, Dict[str, object]]:
    """
    Fit zero-truncated NB1 by stable custom maximum likelihood.

    The experimental statsmodels truncated-count implementation failed to
    converge for these sparse positive-count samples. The likelihood here is
    the same zero-truncated NB1 likelihood, optimized with SciPy L-BFGS-B and
    an analytic gradient. Alpha is optimized on the log scale.
    """
    y = np.asarray(observed_positive, dtype=float)
    x = np.asarray(design_positive, dtype=float)
    exposure = np.asarray(exposure_positive, dtype=float)

    if (y <= 0).any():
        raise ValueError(
            "Truncated NB1 received nonpositive counts."
        )

    if (exposure <= 0).any():
        raise ValueError(
            "Truncated NB1 received nonpositive exposure."
        )

    poisson_beta = fit_poisson_start(
        y,
        x,
        exposure,
    )

    starts: List[np.ndarray] = []
    for intercept_shift, alpha_start in [
        (-1.0, 30.0),
        (-0.5, 10.0),
        (-1.5, 50.0),
        (0.0, 100.0),
    ]:
        beta_start = poisson_beta.copy()
        beta_start[0] += intercept_shift
        starts.append(
            np.concatenate(
                [
                    beta_start,
                    np.array(
                        [math.log(alpha_start)],
                        dtype=float,
                    ),
                ]
            )
        )

    bounds = (
        [(-30.0, 30.0)] * x.shape[1]
        + [(math.log(1e-6), math.log(1e6))]
    )

    attempts: List[Dict[str, object]] = []
    best_optimization = None
    best_objective = np.inf
    best_start_index = None

    for start_index, start in enumerate(starts, start=1):
        try:
            optimization = minimize(
                fun=lambda parameters: (
                    truncated_nb1_objective_and_gradient(
                        parameters,
                        y,
                        x,
                        exposure,
                    )
                ),
                x0=start,
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
                options={
                    "maxiter": 2500,
                    "maxls": 80,
                    "ftol": 1e-12,
                    "gtol": 1e-6,
                },
            )

            objective, gradient = (
                truncated_nb1_objective_and_gradient(
                    optimization.x,
                    y,
                    x,
                    exposure,
                )
            )
            maximum_gradient = float(
                np.max(np.abs(gradient))
            )
            transformed_parameters = np.asarray(
                optimization.x,
                dtype=float,
            )
            alpha = float(
                np.exp(transformed_parameters[-1])
            )

            converged = bool(
                optimization.success
                or maximum_gradient <= 1e-4
            )
            valid = bool(
                converged
                and np.isfinite(objective)
                and np.isfinite(
                    transformed_parameters
                ).all()
                and np.isfinite(alpha)
                and alpha > 0
            )

            attempts.append(
                {
                    "Optimizer": (
                        "Custom_L-BFGS-B_analytic_gradient"
                    ),
                    "Start_Index": start_index,
                    "Converged": converged,
                    "Alpha_Estimate": alpha,
                    "Stable_LogLikelihood": (
                        -objective * len(y)
                        if np.isfinite(objective)
                        else np.nan
                    ),
                    "Maximum_Absolute_Gradient": (
                        maximum_gradient
                    ),
                    "Warning_Text": str(
                        optimization.message
                    ),
                    "Valid_Fit": valid,
                }
            )

            if valid and objective < best_objective:
                best_optimization = optimization
                best_objective = objective
                best_start_index = start_index

            if (
                valid
                and optimization.success
                and maximum_gradient <= 1e-5
            ):
                break

        except Exception as error:
            attempts.append(
                {
                    "Optimizer": (
                        "Custom_L-BFGS-B_analytic_gradient"
                    ),
                    "Start_Index": start_index,
                    "Converged": False,
                    "Alpha_Estimate": np.nan,
                    "Stable_LogLikelihood": np.nan,
                    "Maximum_Absolute_Gradient": np.nan,
                    "Warning_Text": (
                        f"{type(error).__name__}: {error}"
                    ),
                    "Valid_Fit": False,
                }
            )

    if best_optimization is None:
        raise RuntimeError(
            "All custom zero-truncated NB1 fitting attempts failed:\n"
            + "\n".join(
                json.dumps(item, default=str)
                for item in attempts
            )
        )

    transformed_parameters = np.asarray(
        best_optimization.x,
        dtype=float,
    )
    alpha = float(
        np.exp(transformed_parameters[-1])
    )
    parameters = np.concatenate(
        [
            transformed_parameters[:-1],
            np.array([alpha], dtype=float),
        ]
    )
    standard_errors = approximate_bse_from_lbfgs(
        best_optimization,
        sample_size=len(y),
        alpha=alpha,
    )

    _, final_gradient = (
        truncated_nb1_objective_and_gradient(
            transformed_parameters,
            y,
            x,
            exposure,
        )
    )
    final_maximum_gradient = float(
        np.max(np.abs(final_gradient))
    )

    result = StableOptimizationResult(
        params=parameters,
        bse=standard_errors,
        converged=True,
        mle_retvals={
            "converged": True,
            "optimizer": (
                "Custom_L-BFGS-B_analytic_gradient"
            ),
            "message": str(
                best_optimization.message
            ),
            "maximum_absolute_gradient": (
                final_maximum_gradient
            ),
        },
    )

    status = {
        "Optimizer": (
            "Custom_L-BFGS-B_analytic_gradient"
        ),
        "Start_Index": int(best_start_index),
        "Converged": True,
        "Alpha_Estimate": alpha,
        "Stable_LogLikelihood": float(
            -best_objective * len(y)
        ),
        "Maximum_Absolute_Gradient": (
            final_maximum_gradient
        ),
        "Warning_Text": str(
            best_optimization.message
        ),
        "Valid_Fit": True,
        "Attempt_Count": len(attempts),
        "Failed_Attempt_Count": int(
            sum(
                not bool(item["Valid_Fit"])
                for item in attempts
            )
        ),
        "SE_Method": (
            "L-BFGS inverse-Hessian approximation; "
            "delta method for alpha"
        ),
    }

    return result, status

def fit_hurdle_nb1(
    observed: np.ndarray,
    design: np.ndarray,
    exposure: np.ndarray,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    occurrence_target = (np.asarray(observed) > 0).astype(float)
    occurrence_result, occurrence_status = fit_binary_glm(
        occurrence_target, design
    )
    positive = occurrence_target == 1
    positive_result, positive_status = fit_truncated_nb1(
        np.asarray(observed)[positive],
        design[positive],
        np.asarray(exposure)[positive],
    )
    return {
        "Occurrence_Result": occurrence_result,
        "Positive_Result": positive_result,
    }, {
        "Occurrence_Optimizer": occurrence_status["Optimizer"],
        "Occurrence_Converged": occurrence_status["Converged"],
        "Occurrence_Warning_Text": occurrence_status["Warning_Text"],
        "Positive_Optimizer": positive_status["Optimizer"],
        "Positive_Converged": positive_status["Converged"],
        "Positive_Warning_Text": positive_status["Warning_Text"],
        "Alpha_Estimate": positive_status["Alpha_Estimate"],
        "Converged": bool(
            occurrence_status["Converged"] and positive_status["Converged"]
        ),
        "Valid_Fit": bool(
            occurrence_status["Converged"] and positive_status["Valid_Fit"]
        ),
    }


def hurdle_outputs(
    fitted: Dict[str, object],
    observed: np.ndarray,
    design: np.ndarray,
    exposure: np.ndarray,
) -> Dict[str, np.ndarray | float]:
    occurrence_params = np.asarray(
        fitted["Occurrence_Result"].params, dtype=float
    )
    positive_params = np.asarray(
        fitted["Positive_Result"].params, dtype=float
    )

    p_occurrence = stable_expit(design @ occurrence_params)
    alpha = float(positive_params[-1])
    base_mean = count_mean(positive_params, design, exposure)
    base_logp, base_logzero = nb1_logpmf_and_logzero(
        observed, base_mean, alpha
    )
    base_log_nonzero = stable_log_one_minus_exp(base_logzero)
    base_zero = np.exp(np.clip(base_logzero, -745.0, 0.0))
    conditional_positive_mean = base_mean / np.clip(
        1.0 - base_zero,
        PROBABILITY_EPSILON,
        None,
    )

    predicted_count = p_occurrence * conditional_positive_mean
    predicted_zero = 1.0 - p_occurrence

    observed = np.asarray(observed, dtype=float)
    log_probability = np.empty_like(observed)
    zero = observed == 0
    positive = ~zero
    log_probability[zero] = np.log(
        np.clip(predicted_zero[zero], PROBABILITY_EPSILON, 1.0)
    )
    log_probability[positive] = (
        np.log(np.clip(p_occurrence[positive], PROBABILITY_EPSILON, 1.0))
        + base_logp[positive]
        - base_log_nonzero[positive]
    )

    if not np.isfinite(log_probability).all():
        raise FloatingPointError("Hurdle log probability is non-finite.")

    return {
        "Predicted_Count": predicted_count,
        "Predicted_Zero_Probability": predicted_zero,
        "Predictive_LogProbability": log_probability,
        "Alpha": alpha,
    }


def zinb1_outputs(
    parameters: np.ndarray,
    observed: np.ndarray,
    count_design: np.ndarray,
    inflation_design: np.ndarray,
    exposure: np.ndarray,
) -> Dict[str, np.ndarray | float]:
    params = np.asarray(parameters, dtype=float)
    k_infl = inflation_design.shape[1]
    k_count = count_design.shape[1]
    expected = k_infl + k_count + 1
    if len(params) != expected:
        raise ValueError(
            f"Unexpected ZINB1 parameter length: {len(params)} != {expected}."
        )

    gamma = params[:k_infl]
    beta = params[k_infl:k_infl + k_count]
    alpha = float(params[-1])

    structural_zero = stable_expit(inflation_design @ gamma)
    main_params = np.concatenate([beta, [alpha]])
    main_mean = count_mean(main_params, count_design, exposure)
    base_logp, base_logzero = nb1_logpmf_and_logzero(
        observed, main_mean, alpha
    )

    log_pi = np.log(structural_zero)
    log_main = np.log1p(-structural_zero)
    observed = np.asarray(observed, dtype=float)
    log_probability = np.empty_like(observed)
    zero = observed == 0
    positive = ~zero

    log_probability[zero] = np.logaddexp(
        log_pi[zero],
        log_main[zero] + base_logzero[zero],
    )
    log_probability[positive] = (
        log_main[positive] + base_logp[positive]
    )

    base_zero = np.exp(np.clip(base_logzero, -745.0, 0.0))
    predicted_zero = structural_zero + (1.0 - structural_zero) * base_zero
    predicted_count = (1.0 - structural_zero) * main_mean

    if not np.isfinite(log_probability).all():
        raise FloatingPointError("ZINB1 log probability is non-finite.")

    return {
        "Predicted_Count": predicted_count,
        "Predicted_Zero_Probability": predicted_zero,
        "Predictive_LogProbability": log_probability,
        "Alpha": alpha,
    }


def zinb1_objective_and_gradient(
    transformed_parameters: np.ndarray,
    observed: np.ndarray,
    design: np.ndarray,
    exposure: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """Stable ZINB1 mean negative log-likelihood and analytic gradient."""
    y = np.asarray(observed, dtype=float)
    x = np.asarray(design, dtype=float)
    exposure = np.asarray(exposure, dtype=float)

    component_count = x.shape[1]
    gamma = transformed_parameters[:component_count]
    beta = transformed_parameters[
        component_count:2 * component_count
    ]
    alpha = float(np.exp(transformed_parameters[-1]))

    structural_zero = stable_expit(x @ gamma)
    log_pi = np.log(structural_zero)
    log_main = np.log1p(-structural_zero)

    linear_predictor = x @ beta + np.log(exposure)
    linear_predictor = np.clip(
        linear_predictor, -745.0, 700.0
    )
    mean = np.clip(
        np.exp(linear_predictor),
        np.finfo(float).tiny,
        1e300,
    )

    size = np.clip(
        mean / alpha,
        np.finfo(float).tiny,
        1e300,
    )
    log_success = -math.log1p(alpha)
    log_failure = math.log(alpha) - math.log1p(alpha)
    success_probability = 1.0 / (1.0 + alpha)
    failure_probability = alpha / (1.0 + alpha)

    digamma_difference = (
        special.digamma(y + size)
        - special.digamma(size)
        + log_success
    )
    base_log_probability = (
        special.gammaln(y + size)
        - special.gammaln(size)
        - special.gammaln(y + 1.0)
        + size * log_success
        + y * log_failure
    )
    base_log_zero = size * log_success

    zero = y == 0
    positive = ~zero
    log_probability = np.empty_like(y)
    log_probability[positive] = (
        log_main[positive]
        + base_log_probability[positive]
    )
    log_probability[zero] = np.logaddexp(
        log_pi[zero],
        log_main[zero] + base_log_zero[zero],
    )

    if not np.isfinite(log_probability).all():
        return 1e100, np.zeros_like(transformed_parameters)

    inflation_multiplier = np.empty_like(y)
    inflation_multiplier[positive] = (
        -structural_zero[positive]
    )
    if zero.any():
        log_one_minus_base_zero = stable_log_one_minus_exp(
            base_log_zero[zero]
        )
        inflation_multiplier[zero] = np.exp(
            np.clip(
                log_pi[zero]
                + log_main[zero]
                + log_one_minus_base_zero
                - log_probability[zero],
                -745.0,
                700.0,
            )
        )
    gamma_score = x.T @ inflation_multiplier

    count_multiplier = np.empty_like(y)
    eta_score = np.empty_like(y)
    count_multiplier[positive] = (
        size[positive]
        * digamma_difference[positive]
    )
    eta_score[positive] = (
        -size[positive]
        * digamma_difference[positive]
        - size[positive] * failure_probability
        + y[positive] * success_probability
    )
    if zero.any():
        count_zero_posterior = np.exp(
            np.clip(
                log_main[zero]
                + base_log_zero[zero]
                - log_probability[zero],
                -745.0,
                0.0,
            )
        )
        count_multiplier[zero] = (
            count_zero_posterior
            * size[zero]
            * log_success
        )
        eta_score[zero] = (
            count_zero_posterior
            * (-size[zero])
            * (log_success + failure_probability)
        )

    beta_score = x.T @ count_multiplier
    score = np.concatenate(
        [
            gamma_score,
            beta_score,
            np.array([eta_score.sum()], dtype=float),
        ]
    )

    sample_size = float(len(y))
    objective = float(-log_probability.sum() / sample_size)
    gradient = -score / sample_size

    if not np.isfinite(objective):
        return 1e100, np.zeros_like(transformed_parameters)
    if not np.isfinite(gradient).all():
        return 1e100, np.zeros_like(transformed_parameters)

    return objective, gradient


def fit_zinb1_custom_fallback(
    observed: np.ndarray,
    design: np.ndarray,
    exposure: np.ndarray,
    nb1_result: object,
) -> Tuple[object, Dict[str, object]]:
    """Custom stable ZINB1 fallback used only if statsmodels fits fail."""
    y = np.asarray(observed, dtype=float)
    x = np.asarray(design, dtype=float)
    exposure = np.asarray(exposure, dtype=float)

    zero_target = (y == 0).astype(float)
    zero_start_result, zero_start_status = fit_binary_glm(
        zero_target, x
    )
    zero_start = np.asarray(
        zero_start_result.params, dtype=float
    )
    count_parameters = np.asarray(
        nb1_result.params, dtype=float
    )
    count_beta = count_parameters[:-1]
    count_alpha = float(
        np.clip(count_parameters[-1], 1e-6, 1e6)
    )

    low_zero = np.zeros_like(zero_start)
    low_zero[0] = -3.0

    starts = [
        np.concatenate(
            [low_zero, count_beta, [math.log(count_alpha)]]
        ),
        np.concatenate(
            [zero_start * 0.25, count_beta, [math.log(count_alpha)]]
        ),
        np.concatenate(
            [zero_start * 0.50, count_beta, [math.log(count_alpha)]]
        ),
    ]
    bounds = (
        [(-25.0, 25.0)] * x.shape[1]
        + [(-30.0, 30.0)] * x.shape[1]
        + [(math.log(1e-6), math.log(1e6))]
    )

    attempts = []
    best = None
    best_objective = np.inf
    best_start = None

    for start_index, start in enumerate(starts, start=1):
        try:
            optimization = minimize(
                fun=lambda parameters: (
                    zinb1_objective_and_gradient(
                        parameters, y, x, exposure
                    )
                ),
                x0=start,
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
                options={
                    "maxiter": 2500,
                    "maxls": 80,
                    "ftol": 1e-12,
                    "gtol": 1e-6,
                },
            )
            objective, gradient = zinb1_objective_and_gradient(
                optimization.x, y, x, exposure
            )
            maximum_gradient = float(
                np.max(np.abs(gradient))
            )
            alpha = float(np.exp(optimization.x[-1]))
            converged = bool(
                optimization.success
                or maximum_gradient <= 1e-4
            )
            valid = bool(
                converged
                and np.isfinite(objective)
                and np.isfinite(optimization.x).all()
                and np.isfinite(alpha)
                and alpha > 0
            )
            attempts.append({
                "Optimizer": "Custom_L-BFGS-B_analytic_gradient",
                "Start_Index": start_index,
                "Converged": converged,
                "Alpha_Estimate": alpha,
                "Stable_LogLikelihood": (
                    -objective * len(y)
                    if np.isfinite(objective)
                    else np.nan
                ),
                "Maximum_Absolute_Gradient": maximum_gradient,
                "Warning_Text": str(optimization.message),
                "Valid_Fit": valid,
            })
            if valid and objective < best_objective:
                best = optimization
                best_objective = objective
                best_start = start_index
            if (
                start_index >= 2
                and valid
                and optimization.success
                and maximum_gradient <= 1e-5
            ):
                break
        except Exception as error:
            attempts.append({
                "Optimizer": "Custom_L-BFGS-B_analytic_gradient",
                "Start_Index": start_index,
                "Converged": False,
                "Alpha_Estimate": np.nan,
                "Stable_LogLikelihood": np.nan,
                "Maximum_Absolute_Gradient": np.nan,
                "Warning_Text": f"{type(error).__name__}: {error}",
                "Valid_Fit": False,
            })

    if best is None:
        raise RuntimeError(
            "All custom ZINB1 fallback attempts failed:\n"
            + "\n".join(
                json.dumps(item, default=str)
                for item in attempts
            )
        )

    transformed = np.asarray(best.x, dtype=float)
    alpha = float(np.exp(transformed[-1]))
    parameters = np.concatenate(
        [transformed[:-1], [alpha]]
    )
    standard_errors = approximate_bse_from_lbfgs(
        best, len(y), alpha
    )
    _, gradient = zinb1_objective_and_gradient(
        transformed, y, x, exposure
    )
    maximum_gradient = float(np.max(np.abs(gradient)))

    result = StableOptimizationResult(
        params=parameters,
        bse=standard_errors,
        converged=True,
        mle_retvals={
            "converged": True,
            "optimizer": "Custom_L-BFGS-B_analytic_gradient",
            "message": str(best.message),
            "maximum_absolute_gradient": maximum_gradient,
        },
    )
    status = {
        "Optimizer": "Custom_L-BFGS-B_analytic_gradient_fallback",
        "Start_Index": int(best_start),
        "Converged": True,
        "Alpha_Estimate": alpha,
        "Stable_LogLikelihood": float(
            -best_objective * len(y)
        ),
        "Maximum_Absolute_Gradient": maximum_gradient,
        "Warning_Text": str(best.message),
        "Valid_Fit": True,
        "Attempt_Count": len(attempts),
        "Failed_Attempt_Count": int(
            sum(not bool(item["Valid_Fit"]) for item in attempts)
        ),
        "Zero_Start_Optimizer": zero_start_status["Optimizer"],
        "SE_Method": (
            "L-BFGS inverse-Hessian approximation; "
            "delta method for alpha"
        ),
    }
    return result, status


def fit_zinb1(
    observed: np.ndarray,
    design: np.ndarray,
    exposure: np.ndarray,
    nb1_result: object,
) -> Tuple[object, Dict[str, object]]:
    zero_target = (np.asarray(observed) == 0).astype(float)
    zero_start_result, zero_start_status = fit_binary_glm(
        zero_target, design
    )
    inflation_start = np.asarray(zero_start_result.params, dtype=float)
    count_start = np.asarray(nb1_result.params, dtype=float)

    starts = [
        np.concatenate([inflation_start, count_start]),
        np.concatenate([inflation_start * 0.5, count_start]),
        np.concatenate([inflation_start * 0.25, count_start]),
    ]

    model = ZeroInflatedNegativeBinomialP(
        observed,
        design,
        exog_infl=design,
        exposure=exposure,
        inflation="logit",
        p=1,
        missing="raise",
    )

    attempts: List[Dict[str, object]] = []
    best_result = None
    best_status = None
    best_ll = -np.inf

    for start_index, start in enumerate(starts, start=1):
        for optimizer in ZINB_OPTIMIZERS:
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                try:
                    result = model.fit(
                        start_params=start,
                        method=optimizer,
                        maxiter=FIT_MAX_ITERATIONS,
                        disp=0,
                        full_output=True,
                    )
                    params = np.asarray(result.params, dtype=float)
                    converged = convergence_status(result)
                    alpha = finite_or_nan(params[-1])
                    stable_ll = np.nan
                    if (
                        converged
                        and np.isfinite(params).all()
                        and np.isfinite(alpha)
                        and alpha > 0
                    ):
                        outputs = zinb1_outputs(
                            params, observed, design, design, exposure
                        )
                        stable_ll = float(
                            np.asarray(
                                outputs["Predictive_LogProbability"],
                                dtype=float,
                            ).sum()
                        )
                    valid = bool(np.isfinite(stable_ll))
                    status = {
                        "Optimizer": optimizer,
                        "Start_Index": start_index,
                        "Converged": converged,
                        "Alpha_Estimate": alpha,
                        "Stable_LogLikelihood": stable_ll,
                        "Warning_Text": warning_text(captured),
                        "Valid_Fit": valid,
                    }
                    attempts.append(status)
                    if valid and stable_ll > best_ll:
                        best_result = result
                        best_status = status
                        best_ll = stable_ll
                        break
                except Exception as error:
                    attempts.append({
                        "Optimizer": optimizer,
                        "Start_Index": start_index,
                        "Converged": False,
                        "Alpha_Estimate": np.nan,
                        "Stable_LogLikelihood": np.nan,
                        "Warning_Text": f"{type(error).__name__}: {error}",
                        "Valid_Fit": False,
                    })
        if best_result is not None:
            break

    if best_result is None or best_status is None:
        fallback_result, fallback_status = fit_zinb1_custom_fallback(
            observed=observed,
            design=design,
            exposure=exposure,
            nb1_result=nb1_result,
        )
        fallback_status = dict(fallback_status)
        fallback_status[
            "Statsmodels_Failed_Attempts"
        ] = len(attempts)
        fallback_status[
            "Statsmodels_Failure_Summary"
        ] = " | ".join(
            str(item.get("Warning_Text", ""))
            for item in attempts
        )
        return fallback_result, fallback_status

    best_status = dict(best_status)
    best_status["Attempt_Count"] = len(attempts)
    best_status["Failed_Attempt_Count"] = sum(
        not bool(item["Valid_Fit"]) for item in attempts
    )
    best_status["Zero_Start_Optimizer"] = zero_start_status["Optimizer"]
    return best_result, best_status


# =============================================================================
# Metrics and predictions
# =============================================================================

def prediction_metrics(
    observed_count: np.ndarray,
    predicted_count: np.ndarray,
    exposure: np.ndarray,
    predicted_zero_probability: np.ndarray,
    log_probability: np.ndarray,
) -> Dict[str, float]:
    observed_count = np.asarray(observed_count, dtype=float)
    predicted_count = np.asarray(predicted_count, dtype=float)
    exposure = np.asarray(exposure, dtype=float)
    predicted_zero_probability = np.asarray(
        predicted_zero_probability, dtype=float
    )
    log_probability = np.asarray(log_probability, dtype=float)

    observed_rate = observed_count / exposure
    predicted_rate = predicted_count / exposure
    observed_zero = (observed_count == 0).astype(int)
    observed_positive = (observed_count > 0).astype(int)
    predicted_positive = 1.0 - predicted_zero_probability

    calibration_intercept, calibration_slope = calibration_parameters(
        observed_rate, predicted_rate
    )

    total_observed = float(observed_count.sum())
    total_predicted = float(predicted_count.sum())
    positive = observed_positive == 1
    zero = observed_zero == 1

    return {
        "N": int(len(observed_count)),
        "Observed_Total_Count": total_observed,
        "Predicted_Total_Count": total_predicted,
        "Predicted_to_Observed_Total_Count_Ratio": (
            total_predicted / total_observed if total_observed > 0 else np.nan
        ),
        "Mean_Negative_LogLikelihood": float(-log_probability.mean()),
        "Total_Predictive_LogLikelihood": float(log_probability.sum()),
        "Zero_Probability_Brier": float(
            np.mean((predicted_zero_probability - observed_zero) ** 2)
        ),
        "Observed_Zero_Proportion": float(observed_zero.mean()),
        "Predicted_Zero_Proportion": float(
            predicted_zero_probability.mean()
        ),
        "Predicted_minus_Observed_Zero_Proportion": float(
            predicted_zero_probability.mean() - observed_zero.mean()
        ),
        "Occurrence_ROC_AUC": safe_roc_auc(
            observed_positive, predicted_positive
        ),
        "Occurrence_PR_AUC": safe_pr_auc(
            observed_positive, predicted_positive
        ),
        "Rate_RMSE": float(
            math.sqrt(mean_squared_error(observed_rate, predicted_rate))
        ),
        "Rate_MAE": float(
            mean_absolute_error(observed_rate, predicted_rate)
        ),
        "Rate_R2": safe_r2(observed_rate, predicted_rate),
        "Rate_Calibration_Intercept": calibration_intercept,
        "Rate_Calibration_Slope": calibration_slope,
        "Positive_Rate_MAE": (
            float(mean_absolute_error(
                observed_rate[positive], predicted_rate[positive]
            ))
            if positive.any()
            else np.nan
        ),
        "Zero_Row_Predicted_Rate_Mean": (
            float(predicted_rate[zero].mean()) if zero.any() else np.nan
        ),
    }


def make_prediction_frame(
    validation: pd.DataFrame,
    response: str,
    rate_name: str,
    structure: str,
    fold: int,
    outputs: Dict[str, np.ndarray | float],
) -> pd.DataFrame:
    result = validation[[
        "GRID_UID", "GRID_ID", "Country", "Year", "Season",
        "Temporal_Fold", "ForestPixelCount",
    ]].copy()

    observed = validation[response].to_numpy(dtype=float)
    exposure = validation["ForestPixelCount"].to_numpy(dtype=float)
    predicted = np.asarray(outputs["Predicted_Count"], dtype=float)

    result["Count_Response"] = response
    result["Rate_Scale_Name"] = rate_name
    result["Model_Structure"] = structure
    result["Fold"] = int(fold)
    result["Observed_Count"] = observed
    result["Predicted_Count"] = predicted
    result["Observed_Rate"] = observed / exposure
    result["Predicted_Rate"] = predicted / exposure
    result["Observed_Zero"] = (observed == 0).astype(int)
    result["Predicted_Zero_Probability"] = np.asarray(
        outputs["Predicted_Zero_Probability"], dtype=float
    )
    result["Predictive_LogProbability"] = np.asarray(
        outputs["Predictive_LogProbability"], dtype=float
    )
    return result


def fit_one_fold(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    response: str,
    rate_name: str,
    predictors: Sequence[str],
    scheme: str,
    fold: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x_train, x_valid = design_matrices(
        training, validation, predictors, scheme
    )
    y_train = training[response].to_numpy(dtype=float)
    y_valid = validation[response].to_numpy(dtype=float)
    exposure_train = training["ForestPixelCount"].to_numpy(dtype=float)
    exposure_valid = validation["ForestPixelCount"].to_numpy(dtype=float)

    metric_rows: List[Dict[str, object]] = []
    status_rows: List[Dict[str, object]] = []
    prediction_frames: List[pd.DataFrame] = []

    nb1_result, nb1_status = fit_nb1(
        y_train, x_train, exposure_train
    )
    nb1_output = single_stage_outputs(
        nb1_result, y_valid, x_valid, exposure_valid
    )

    hurdle_result, hurdle_status = fit_hurdle_nb1(
        y_train, x_train, exposure_train
    )
    hurdle_output = hurdle_outputs(
        hurdle_result, y_valid, x_valid, exposure_valid
    )

    zinb_result, zinb_status = fit_zinb1(
        y_train, x_train, exposure_train, nb1_result
    )
    zinb_output = zinb1_outputs(
        np.asarray(zinb_result.params, dtype=float),
        y_valid,
        x_valid,
        x_valid,
        exposure_valid,
    )

    fitted_items = [
        ("Single_Stage_NB1", nb1_output, nb1_status),
        ("Hurdle_NB1", hurdle_output, hurdle_status),
        ("ZINB1", zinb_output, zinb_status),
    ]

    for structure, outputs, status in fitted_items:
        metrics = prediction_metrics(
            y_valid,
            np.asarray(outputs["Predicted_Count"], dtype=float),
            exposure_valid,
            np.asarray(outputs["Predicted_Zero_Probability"], dtype=float),
            np.asarray(outputs["Predictive_LogProbability"], dtype=float),
        )
        metrics.update({
            "Model_Structure": structure,
            "Fold": fold,
            "Train_N": int(len(training)),
            "Validation_N": int(len(validation)),
            "Alpha_Estimate": outputs["Alpha"],
        })
        metric_rows.append(metrics)

        status_rows.append({
            "Model_Structure": structure,
            "Fold": fold,
            **status,
        })
        prediction_frames.append(
            make_prediction_frame(
                validation,
                response,
                rate_name,
                structure,
                fold,
                outputs,
            )
        )

    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(status_rows),
        pd.concat(prediction_frames, ignore_index=True),
    )


# =============================================================================
# Checkpoints
# =============================================================================

def checkpoint_paths(season_code: int, response: str) -> Tuple[Path, Path, Path]:
    prefix = f"S{season_code}_{response}"
    return (
        CHECKPOINT_ROOT / f"{prefix}_metrics.csv",
        CHECKPOINT_ROOT / f"{prefix}_status.csv",
        CHECKPOINT_ROOT / f"{prefix}_predictions.csv.gz",
    )


def valid_checkpoint(
    season_code: int,
    response: str,
    expected_rows_per_model: int,
) -> bool:
    metric_path, status_path, prediction_path = checkpoint_paths(
        season_code, response
    )
    if not (
        metric_path.exists() and status_path.exists() and prediction_path.exists()
    ):
        return False

    try:
        metrics = pd.read_csv(metric_path)
        status = pd.read_csv(status_path)
        predictions = pd.read_csv(prediction_path, compression="gzip")
    except Exception:
        return False

    if len(metrics) != 15 or len(status) != 15:
        return False
    if len(predictions) != expected_rows_per_model * 3:
        return False
    if set(metrics["Model_Structure"]) != set(MODEL_STRUCTURES):
        return False
    if set(metrics["Fold"].astype(int)) != {1, 2, 3, 4, 5}:
        return False
    return bool(
        np.isfinite(predictions["Predictive_LogProbability"]).all()
    )


def run_one_season_response(
    season_data: pd.DataFrame,
    response: str,
    rate_name: str,
    predictors: Sequence[str],
    scheme: str,
    season_code: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    metric_path, status_path, prediction_path = checkpoint_paths(
        season_code, response
    )

    if valid_checkpoint(season_code, response, len(season_data)):
        log(f"Reused checkpoint: {SEASONS[season_code]} {response}.")
        return (
            pd.read_csv(metric_path),
            pd.read_csv(status_path),
            pd.read_csv(prediction_path, compression="gzip"),
        )

    metric_tables: List[pd.DataFrame] = []
    status_tables: List[pd.DataFrame] = []
    prediction_tables: List[pd.DataFrame] = []

    for fold in range(1, 6):
        validation_mask = season_data["Temporal_Fold"] == fold
        training = season_data.loc[~validation_mask].copy().reset_index(drop=True)
        validation = season_data.loc[validation_mask].copy().reset_index(drop=True)

        metrics, status, predictions = fit_one_fold(
            training,
            validation,
            response,
            rate_name,
            predictors,
            scheme,
            fold,
        )
        metric_tables.append(metrics)
        status_tables.append(status)
        prediction_tables.append(predictions)

        for row in metrics.itertuples(index=False):
            log(
                f"Completed fold {fold}: {SEASONS[season_code]} {response} "
                f"{row.Model_Structure}; "
                f"NLL={row.Mean_Negative_LogLikelihood:.6f}; "
                f"Brier={row.Zero_Probability_Brier:.6f}; "
                f"RMSE={row.Rate_RMSE:.8f}."
            )

    metrics = pd.concat(metric_tables, ignore_index=True)
    status = pd.concat(status_tables, ignore_index=True)
    predictions = pd.concat(prediction_tables, ignore_index=True)

    metrics.to_csv(metric_path, index=False, encoding="utf-8-sig")
    status.to_csv(status_path, index=False, encoding="utf-8-sig")
    predictions.to_csv(prediction_path, index=False, compression="gzip")
    return metrics, status, predictions


# =============================================================================
# Aggregation and ranking
# =============================================================================

def pooled_oof_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    grouping = [
        "Season", "Count_Response", "Rate_Scale_Name", "Model_Structure"
    ]
    for (season_code, response, rate_name, structure), group in predictions.groupby(
        grouping,
        sort=True,
    ):
        metrics = prediction_metrics(
            group["Observed_Count"].to_numpy(dtype=float),
            group["Predicted_Count"].to_numpy(dtype=float),
            group["ForestPixelCount"].to_numpy(dtype=float),
            group["Predicted_Zero_Probability"].to_numpy(dtype=float),
            group["Predictive_LogProbability"].to_numpy(dtype=float),
        )
        fold_nll = (
            group.groupby("Fold")["Predictive_LogProbability"].mean() * -1.0
        )
        metrics.update({
            "Season_Code": int(season_code),
            "Season_Label": SEASONS[int(season_code)],
            "Count_Response": response,
            "Rate_Scale_Name": rate_name,
            "Model_Structure": structure,
            "Fold_NLL_Mean": float(fold_nll.mean()),
            "Fold_NLL_SD": float(fold_nll.std(ddof=1)),
            "Fold_NLL_SE": float(fold_nll.std(ddof=1) / math.sqrt(5)),
        })
        rows.append(metrics)
    return pd.DataFrame(rows)


def model_ranking(pooled: pd.DataFrame) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    for (_, _), subset in pooled.groupby(
        ["Season_Code", "Count_Response"], sort=True
    ):
        subset = subset.copy().sort_values([
            "Mean_Negative_LogLikelihood",
            "Zero_Probability_Brier",
            "Rate_RMSE",
            "Model_Structure",
        ])
        subset["Model_Rank"] = np.arange(1, len(subset) + 1)
        baseline = subset.loc[
            subset["Model_Structure"] == "Single_Stage_NB1"
        ].iloc[0]
        subset["Delta_NLL_vs_Single_Stage_NB1"] = (
            subset["Mean_Negative_LogLikelihood"]
            - baseline["Mean_Negative_LogLikelihood"]
        )
        subset["Relative_NLL_Change_vs_Single_Stage_NB1_pct"] = (
            subset["Mean_Negative_LogLikelihood"]
            / baseline["Mean_Negative_LogLikelihood"]
            - 1.0
        ) * 100.0
        subset["Delta_Zero_Brier_vs_Single_Stage_NB1"] = (
            subset["Zero_Probability_Brier"]
            - baseline["Zero_Probability_Brier"]
        )
        subset["Delta_Rate_RMSE_vs_Single_Stage_NB1"] = (
            subset["Rate_RMSE"] - baseline["Rate_RMSE"]
        )
        parts.append(subset)
    return pd.concat(parts, ignore_index=True).sort_values([
        "Season_Code", "Count_Response", "Model_Rank"
    ]).reset_index(drop=True)


def fold_win_counts(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    ranked = fold_metrics.copy()
    ranked["Fold_NLL_Rank"] = ranked.groupby([
        "Season_Code", "Count_Response", "Fold"
    ])["Mean_Negative_LogLikelihood"].rank(method="min", ascending=True)
    ranked["Fold_NLL_Win"] = (ranked["Fold_NLL_Rank"] == 1).astype(int)
    return (
        ranked.groupby([
            "Season_Code", "Season_Label", "Count_Response",
            "Rate_Scale_Name", "Model_Structure",
        ], as_index=False)
        .agg(
            Temporal_Fold_NLL_Wins=("Fold_NLL_Win", "sum"),
            Mean_Fold_NLL=("Mean_Negative_LogLikelihood", "mean"),
            SD_Fold_NLL=("Mean_Negative_LogLikelihood", "std"),
            Mean_Fold_Zero_Brier=("Zero_Probability_Brier", "mean"),
            Mean_Fold_Rate_RMSE=("Rate_RMSE", "mean"),
        )
        .sort_values([
            "Season_Code", "Count_Response", "Temporal_Fold_NLL_Wins",
            "Mean_Fold_NLL",
        ], ascending=[True, True, False, True])
        .reset_index(drop=True)
    )


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    # Preserve valid checkpoints across restarts while rebuilding summaries.
    checkpoint_backup = OUTPUT_ROOT.parent / "_step04_checkpoint_backup"
    if checkpoint_backup.exists():
        shutil.rmtree(checkpoint_backup)

    if OUTPUT_ROOT.exists():
        if CHECKPOINT_ROOT.exists():
            shutil.move(str(CHECKPOINT_ROOT), str(checkpoint_backup))
        shutil.rmtree(OUTPUT_ROOT)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    if checkpoint_backup.exists():
        shutil.move(str(checkpoint_backup), str(CHECKPOINT_ROOT))

    log("Starting matched NB1, hurdle NB1, and ZINB1 comparison.")
    log(f"Input: {INPUT_CSV}")
    log(f"Best configurations: {BEST_CONFIGURATION_CSV}")
    log(f"Output: {OUTPUT_ROOT}")
    log(
        "Truncated NB1 compatibility mode: using "
        "offset=log(ForestPixelCount) instead of an exposure array."
    )

    best_configurations = validate_best_configurations()
    data, excluded = prepare_data(best_configurations)
    log(
        f"Prepared {len(data):,} complete rows and "
        f"{data['GRID_UID'].nunique():,} grids; "
        f"excluded {len(excluded):,} rows."
    )

    fold_tables: List[pd.DataFrame] = []
    status_tables: List[pd.DataFrame] = []
    prediction_tables: List[pd.DataFrame] = []

    for configuration in best_configurations.itertuples(index=False):
        season_code = int(configuration.Season_Code)
        response = str(configuration.Count_Response)
        rate_name = str(configuration.Rate_Scale_Name)
        predictors = parse_predictors(configuration.Continuous_Predictors)
        scheme = str(configuration.Transformation_Scheme)

        season_data = (
            data.loc[data["Season"] == season_code]
            .copy()
            .reset_index(drop=True)
        )
        log(
            f"Starting {SEASONS[season_code]} {response}; "
            f"config={configuration.Config_ID}; "
            f"predictors={len(predictors)}; scheme={scheme}."
        )

        metrics, status, predictions = run_one_season_response(
            season_data,
            response,
            rate_name,
            predictors,
            scheme,
            season_code,
        )

        for table in [metrics, status]:
            table.insert(0, "Season_Code", season_code)
            table.insert(1, "Season_Label", SEASONS[season_code])
            table.insert(2, "Count_Response", response)
            table.insert(3, "Rate_Scale_Name", rate_name)
            table.insert(4, "Config_ID", configuration.Config_ID)

        predictions["Config_ID"] = configuration.Config_ID
        fold_tables.append(metrics)
        status_tables.append(status)
        prediction_tables.append(predictions)

        output_prediction = OUTPUT_ROOT / (
            f"Matched_OOF_Predictions_S{season_code}_{response}.csv.gz"
        )
        predictions.to_csv(output_prediction, index=False, compression="gzip")
        log(f"Saved {output_prediction.name}.")

    fold_metrics = pd.concat(fold_tables, ignore_index=True)
    fit_status = pd.concat(status_tables, ignore_index=True)
    predictions = pd.concat(prediction_tables, ignore_index=True)

    pooled = pooled_oof_metrics(predictions)
    ranking = model_ranking(pooled)
    fold_wins = fold_win_counts(fold_metrics)
    best = (
        ranking.loc[ranking["Model_Rank"] == 1]
        .copy()
        .sort_values(["Season_Code", "Count_Response"])
        .reset_index(drop=True)
    )
    best["Status"] = (
        "Preferred count structure before component-specific predictor "
        "refinement and final clustered inference"
    )

    qa = pd.DataFrame([
        ("Rows_in_common_complete_case_sample", len(data)),
        ("Excluded_incomplete_rows", len(excluded)),
        ("Unique_GRID_UIDs", data["GRID_UID"].nunique()),
        ("Year_min", int(data["Year"].min())),
        ("Year_max", int(data["Year"].max())),
        ("Compared_model_structures", ";".join(MODEL_STRUCTURES)),
        ("Temporal_fold_count", 5),
        ("Exposure", "ForestPixelCount"),
        ("Equivalent_count_offset", "log(ForestPixelCount)"),
        (
            "Primary_selection_metric",
            "Pooled temporal OOF mean negative log-likelihood",
        ),
        (
            "Secondary_selection_metrics",
            "Zero-probability Brier; rate-scale RMSE",
        ),
        ("Final_inference_stage", "Step 06d GRID_UID-Year two-way clustered inference"),
        ("Matched_predictor_design", True),
    ], columns=["Metric", "Value"])

    method = {
        "workflow_stage": "Matched excess-zero structure comparison",
        "models": {
            "Single_Stage_NB1": "One NB1 count process",
            "Hurdle_NB1": (
                "Binomial-logit occurrence process plus zero-truncated "
                "NB1 positive-count process"
            ),
            "ZINB1": (
                "Logit structural-zero process plus an NB1 count process "
                "that can also generate zeros"
            ),
        },
        "responses": list(COUNT_RESPONSES.keys()),
        "rate_scales": COUNT_RESPONSES,
        "exposure": "ForestPixelCount",
        "equivalent_offset": "log(ForestPixelCount)",
        "country_effect": "Fixed contextual effect; China reference",
        "matched_design": (
            "Same records, predictors, transformations, Country contrasts, "
            "exposure, and temporal folds within each season-response model"
        ),
        "predictor_source": str(BEST_CONFIGURATION_CSV),
        "primary_metric": (
            "Pooled temporal OOF mean negative log-likelihood"
        ),
        "secondary_metrics": [
            "Zero-probability Brier score",
            "Rate-scale RMSE",
        ],
        "final_inference_stage": "Step 06d GRID_UID-Year two-way clustered inference",
        "scope_note": (
            "The preferred structure remains provisional until component-specific "
            "predictor refinement and final clustered inference are completed."
        ),
    }

    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "statsmodels": statsmodels.__version__,
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    with (OUTPUT_ROOT / "00_Method_Definition.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(method, handle, indent=2)

    qa.to_csv(
        OUTPUT_ROOT / "01_Data_QA_Summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    best_configurations.to_csv(
        OUTPUT_ROOT / "02_Matched_Predictor_Configurations.csv",
        index=False,
        encoding="utf-8-sig",
    )
    fold_metrics.to_csv(
        OUTPUT_ROOT / "03_Temporal_Fold_Metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pooled.to_csv(
        OUTPUT_ROOT / "04_Pooled_Temporal_OOF_Metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    ranking.to_csv(
        OUTPUT_ROOT / "05_Model_Structure_Ranking.csv",
        index=False,
        encoding="utf-8-sig",
    )
    fold_wins.to_csv(
        OUTPUT_ROOT / "06_Temporal_Fold_Win_Counts.csv",
        index=False,
        encoding="utf-8-sig",
    )
    best.to_csv(
        OUTPUT_ROOT / "07_Provisional_Best_Excess_Zero_Structure.csv",
        index=False,
        encoding="utf-8-sig",
    )
    fit_status.to_csv(
        OUTPUT_ROOT / "08_Temporal_Model_Fit_Status.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with (OUTPUT_ROOT / "Software_Environment.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(environment, handle, indent=2)

    if CHECKPOINT_ROOT.exists():
        shutil.rmtree(CHECKPOINT_ROOT)

    log("Matched NB1, hurdle NB1, and ZINB1 comparison completed successfully.")
    log(
        "Next step: review the preferred excess-zero structure before "
        "component-specific predictor optimization."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
