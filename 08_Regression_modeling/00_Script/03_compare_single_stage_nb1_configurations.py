# -*- coding: utf-8 -*-
"""
Development of predictor representations for the single-stage NB1 model.

Recommended public location
---------------------------
<CODE_ROOT>/08_Regression_modeling/00_Script/
    03_compare_single_stage_nb1_configurations.py

Inputs
------
<CODE_ROOT>/00_Input_data/Base_Table_2001_2025.csv.bz2
<CODE_ROOT>/08_Regression_modeling/
    02_Single_Stage_Count_Family_Comparison/
        08_Provisional_Best_Single_Stage_Family.csv

Output
------
<CODE_ROOT>/08_Regression_modeling/
    03_Single_Stage_NB1_Configuration_Comparison/

Purpose
-------
The preceding family comparison selected negative binomial type 1 (NB1)
for all six season-response combinations. This script therefore keeps the
distribution family fixed and compares prespecified predictor
representations.

All models use:

    response = Fire_Count or Burned_Pixel_Count
    exposure = ForestPixelCount
    equivalent offset = log(ForestPixelCount)
    Country fixed effects, with China as the reference

The comparison varies:

1. climate representation:
   - raw meteorological variables;
   - integrated FWI;
   - selected FWI components;
2. drought representation:
   - no SPEI or one SPEI timescale at a time;
3. predictor scale:
   - original scale;
   - audit-guided log1p transformations.

A common complete-case sample and identical five-fold temporal splits are
used for every configuration. Candidate configurations with maximum VIF
greater than or equal to 10 remain documented but cannot be provisionally
selected.

This is a model-development step. Its cross-validation metrics are not the
final unbiased comparison with hurdle/ZINB mixed models or random forest.
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
from scipy import special
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from statsmodels.discrete.discrete_model import NegativeBinomial
from statsmodels.genmod.families import Poisson as PoissonFamily
from statsmodels.genmod.generalized_linear_model import GLM


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

FAMILY_SELECTION_CSV = (
    REGRESSION_ROOT
    / "02_Single_Stage_Count_Family_Comparison"
    / "08_Provisional_Best_Single_Stage_Family.csv"
)

OUTPUT_ROOT = (
    REGRESSION_ROOT
    / "03_Single_Stage_NB1_Configuration_Comparison"
)

LOG_FILE = OUTPUT_ROOT / "single_stage_nb1_configuration_comparison.log"
CHECKPOINT_ROOT = OUTPUT_ROOT / "_checkpoints"


# =============================================================================
# Settings
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

TEMPORAL_FOLDS = {
    1: [2024, 2002, 2006, 2004, 2009],
    2: [2016, 2015, 2007, 2018, 2014],
    3: [2025, 2021, 2013, 2019, 2003],
    4: [2010, 2023, 2017, 2011, 2005],
    5: [2020, 2022, 2001, 2012, 2008],
}

COUNTRY_LEVELS = ["China", "NK", "Russia"]
COUNTRY_REFERENCE = "China"
COUNTRY_DUMMY_COLUMNS = ["Country_NK", "Country_Russia"]

# Common low-collinearity adjustment set. These variables are not asserted to
# be the final predictors. The next workflow step performs predictor
# refinement after the best representation is selected.
COMMON_PREDICTORS = [
    "BD",
    "PTC",
    "DEM",
    "Slope",
    "Aspect",
    "POP",
    "Dis_Farm",
    "Road_dens",
]

CLIMATE_REPRESENTATIONS = {
    "Raw_Meteorology": [
        "Temp",
        "Pre",
        "Rhum",
        "Wind",
        "SSRD",
        "LtgProxy",
    ],
    "FWI_Integrated": [
        "FWI",
        "LtgProxy",
    ],
    # FFMC and ISI are strongly correlated in the audit. ISI is therefore
    # not combined with FFMC in this family-screening component set.
    "FWI_Components": [
        "FFMC",
        "DMC",
        "DC",
        "LtgProxy",
    ],
}

DROUGHT_REPRESENTATIONS = {
    "No_SPEI": [],
    "SPEI1": ["SPEI1"],
    "SPEI3": ["SPEI3"],
    "SPEI6": ["SPEI6"],
    "SPEI12": ["SPEI12"],
    "SPEI24": ["SPEI24"],
}

TRANSFORMATION_SCHEMES = [
    "Raw",
    "Audit_Guided_Log1p",
]

AUDIT_GUIDED_LOG1P_VARIABLES = {
    "ND",
    "NE",
    "LtgProxy",
    "POP",
    "Dis_Farm",
    "Road_dens",
}

MAXIMUM_VIF_FOR_SELECTION = 10.0
VIF_SAMPLE_MAX = 100_000

FIT_MAX_ITERATIONS = 500
NB_ALPHA_STARTS = [0.1, 1.0, 10.0, 50.0]
NB_OPTIMIZERS = ["bfgs", "lbfgs"]

BEST_PREDICTION_SAMPLE_FILE_PREFIX = "Best_NB1_OOF_Predictions"


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
# Utility functions
# =============================================================================

def temporal_fold_mapping() -> Dict[int, int]:
    mapping: Dict[int, int] = {}

    for fold, years in TEMPORAL_FOLDS.items():
        for year in years:
            year = int(year)
            if year in mapping:
                raise ValueError(
                    f"Year {year} appears in multiple temporal folds."
                )
            mapping[year] = int(fold)

    return mapping


def safe_r2(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> float:
    if len(observed) < 2 or float(np.var(observed)) <= 0:
        return np.nan

    return float(r2_score(observed, predicted))


def safe_roc_auc(
    observed_binary: np.ndarray,
    probability: np.ndarray,
) -> float:
    if len(np.unique(observed_binary)) < 2:
        return np.nan

    return float(
        roc_auc_score(observed_binary, probability)
    )


def safe_pr_auc(
    observed_binary: np.ndarray,
    probability: np.ndarray,
) -> float:
    if len(np.unique(observed_binary)) < 2:
        return np.nan

    return float(
        average_precision_score(
            observed_binary,
            probability,
        )
    )


def calibration_parameters(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> Tuple[float, float]:
    if len(observed) < 3 or float(np.std(predicted)) <= 0:
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


def warning_text(
    captured: Sequence[warnings.WarningMessage],
) -> str:
    messages: List[str] = []

    for item in captured:
        message = (
            f"{item.category.__name__}: "
            f"{str(item.message)}"
        )
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


def json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if pd.isna(value):
        return None
    raise TypeError(
        f"Object of type {type(value).__name__} "
        "is not JSON serializable."
    )


def checkpoint_path(
    season_code: int,
    response: str,
    config_id: str,
) -> Path:
    safe_response = str(response).replace(" ", "_")
    return (
        CHECKPOINT_ROOT
        / f"S{season_code}_{safe_response}_{config_id}.json"
    )


def save_checkpoint(
    path: Path,
    fold_metrics: pd.DataFrame,
    pooled: Dict[str, object],
    fit_status: pd.DataFrame,
) -> None:
    payload = {
        "fold_metrics": fold_metrics.to_dict(
            orient="records"
        ),
        "pooled": pooled,
        "fit_status": fit_status.to_dict(
            orient="records"
        ),
    }

    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            default=json_default,
        )
    temporary.replace(path)


def load_checkpoint(
    path: Path,
) -> Tuple[pd.DataFrame, Dict[str, object], pd.DataFrame]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    return (
        pd.DataFrame(payload["fold_metrics"]),
        dict(payload["pooled"]),
        pd.DataFrame(payload["fit_status"]),
    )


# =============================================================================
# Candidate configuration definitions
# =============================================================================

def build_configuration_table() -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    config_number = 0

    for climate_name, climate_variables in (
        CLIMATE_REPRESENTATIONS.items()
    ):
        for drought_name, drought_variables in (
            DROUGHT_REPRESENTATIONS.items()
        ):
            for transformation in TRANSFORMATION_SCHEMES:
                config_number += 1

                predictors = list(
                    dict.fromkeys(
                        [
                            *COMMON_PREDICTORS,
                            *climate_variables,
                            *drought_variables,
                        ]
                    )
                )

                rows.append(
                    {
                        "Config_ID": f"NB1_CFG_{config_number:02d}",
                        "Climate_Representation": climate_name,
                        "Drought_Representation": drought_name,
                        "Transformation_Scheme": transformation,
                        "Continuous_Predictor_Count": len(predictors),
                        "Continuous_Predictors": ";".join(predictors),
                        "Common_Predictors": ";".join(COMMON_PREDICTORS),
                        "Climate_Predictors": ";".join(
                            climate_variables
                        ),
                        "Drought_Predictors": ";".join(
                            drought_variables
                        ),
                        "Country_Fixed_Effect": True,
                        "Exposure": "ForestPixelCount",
                        "Distribution_Family": "NB1",
                    }
                )

    return pd.DataFrame(rows)


def parse_predictors(value: str) -> List[str]:
    predictors = [
        item.strip()
        for item in str(value).split(";")
        if item.strip()
    ]

    if not predictors:
        raise ValueError("A configuration has no predictors.")

    return predictors


# =============================================================================
# Data preparation
# =============================================================================

def prepare_data(
    configurations: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Base table not found:\n{INPUT_CSV}"
        )

    data = pd.read_csv(INPUT_CSV)

    all_predictors = sorted(
        {
            predictor
            for value in configurations[
                "Continuous_Predictors"
            ]
            for predictor in parse_predictors(value)
        }
    )

    required = {
        "GRID_UID",
        "GRID_ID",
        "Country",
        "Year",
        "Season",
        "Fire_Count",
        "Burned_Pixel_Count",
        "ForestPixelCount",
        "ForestArea_km2",
        *all_predictors,
    }

    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(
            "Base table is missing required columns:\n"
            + "\n".join(missing)
        )

    duplicate_mask = data.duplicated(
        ["GRID_UID", "Year", "Season"],
        keep=False,
    )
    if duplicate_mask.any():
        data.loc[duplicate_mask].to_csv(
            OUTPUT_ROOT / "ERROR_Duplicate_Panel_Records.csv",
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

    if expected_area.isna().any() or observed_area.isna().any():
        raise ValueError(
            "ForestPixelCount or ForestArea_km2 contains "
            "nonnumeric values."
        )

    maximum_area_error = float(
        np.abs(observed_area - expected_area).max()
    )
    if maximum_area_error > 1e-9:
        raise ValueError(
            "ForestArea_km2 is inconsistent with "
            "ForestPixelCount * 0.25."
        )

    data = data.loc[
        observed_area > FOREST_AREA_THRESHOLD_KM2
    ].copy()

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
            raise ValueError(
                f"{response} contains invalid count values."
            )

        data[response] = values.astype(int)

    if (
        pd.to_numeric(
            data["ForestPixelCount"],
            errors="coerce",
        )
        <= 0
    ).any():
        raise ValueError(
            "ForestPixelCount must be positive."
        )

    mapping = temporal_fold_mapping()
    data["Temporal_Fold"] = (
        data["Year"].astype(int).map(mapping)
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
            "Years missing from temporal fold definitions: "
            + ", ".join(map(str, missing_years))
        )

    common_complete_columns = [
        "Country",
        "Fire_Count",
        "Burned_Pixel_Count",
        "ForestPixelCount",
        *all_predictors,
    ]

    complete = (
        data[common_complete_columns]
        .replace([np.inf, -np.inf], np.nan)
        .notna()
        .all(axis=1)
    )

    excluded = data.loc[
        ~complete,
        [
            "GRID_UID",
            "GRID_ID",
            "Country",
            "Year",
            "Season",
            *all_predictors,
        ],
    ].copy()

    excluded.to_csv(
        OUTPUT_ROOT
        / "Excluded_Incomplete_Common_Candidate_Records.csv",
        index=False,
        encoding="utf-8-sig",
    )

    data = (
        data.loc[complete]
        .copy()
        .reset_index(drop=True)
    )
    data["Temporal_Fold"] = (
        data["Temporal_Fold"].astype(int)
    )

    data["FCD"] = (
        data["Fire_Count"].astype(float)
        / data["ForestPixelCount"].astype(float)
    )
    data["BAD"] = (
        data["Burned_Pixel_Count"].astype(float)
        / data["ForestPixelCount"].astype(float)
    )

    return data, excluded


def validate_family_selection() -> pd.DataFrame:
    if not FAMILY_SELECTION_CSV.exists():
        raise FileNotFoundError(
            "Step-02 family-selection result not found:\n"
            f"{FAMILY_SELECTION_CSV}"
        )

    selection = pd.read_csv(FAMILY_SELECTION_CSV)

    required = {
        "Season_Code",
        "Count_Response",
        "Family",
    }
    missing = sorted(required - set(selection.columns))
    if missing:
        raise ValueError(
            "Family-selection table is missing columns:\n"
            + "\n".join(missing)
        )

    expected_pairs = {
        (season, response)
        for season in SEASONS
        for response in COUNT_RESPONSES
    }
    observed_pairs = {
        (int(row.Season_Code), str(row.Count_Response))
        for row in selection.itertuples(index=False)
    }

    if observed_pairs != expected_pairs:
        raise ValueError(
            "Step-02 family-selection table does not contain "
            "exactly the six expected season-response combinations."
        )

    if set(selection["Family"].astype(str)) != {"NB1"}:
        raise ValueError(
            "Step 03 requires NB1 to have been selected for all "
            "six season-response combinations."
        )

    return selection


# =============================================================================
# Transformations and matrices
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
        raise ValueError(
            f"Unknown transformation scheme: {scheme}"
        )

    for variable in predictors:
        if variable not in AUDIT_GUIDED_LOG1P_VARIABLES:
            continue

        minimum = float(transformed[variable].min())
        if minimum < 0:
            raise ValueError(
                f"{variable} contains negative values and "
                "cannot use log1p."
            )

        transformed[variable] = np.log1p(
            transformed[variable].to_numpy(dtype=float)
        )

    return transformed


def country_dummies(frame: pd.DataFrame) -> np.ndarray:
    country = frame["Country"].astype(str)

    return np.column_stack(
        [
            (country == "NK").astype(float).to_numpy(),
            (country == "Russia").astype(float).to_numpy(),
        ]
    )


def design_matrices(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    predictors: Sequence[str],
    scheme: str,
) -> Tuple[np.ndarray, np.ndarray]:
    training_continuous = transform_predictors(
        training,
        predictors,
        scheme,
    )
    validation_continuous = transform_predictors(
        validation,
        predictors,
        scheme,
    )

    scaler = StandardScaler()
    training_scaled = scaler.fit_transform(
        training_continuous.to_numpy(dtype=float)
    )
    validation_scaled = scaler.transform(
        validation_continuous.to_numpy(dtype=float)
    )

    x_training = np.column_stack(
        [
            np.ones(len(training), dtype=float),
            training_scaled,
            country_dummies(training),
        ]
    )
    x_validation = np.column_stack(
        [
            np.ones(len(validation), dtype=float),
            validation_scaled,
            country_dummies(validation),
        ]
    )

    return x_training, x_validation


def full_design_matrix(
    frame: pd.DataFrame,
    predictors: Sequence[str],
    scheme: str,
) -> np.ndarray:
    continuous = transform_predictors(
        frame,
        predictors,
        scheme,
    )

    scaler = StandardScaler()
    scaled = scaler.fit_transform(
        continuous.to_numpy(dtype=float)
    )

    return np.column_stack(
        [
            np.ones(len(frame), dtype=float),
            scaled,
            country_dummies(frame),
        ]
    )


# =============================================================================
# VIF
# =============================================================================

def configuration_vif(
    data: pd.DataFrame,
    configuration: pd.Series,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    predictors = parse_predictors(
        configuration["Continuous_Predictors"]
    )
    scheme = str(
        configuration["Transformation_Scheme"]
    )

    transformed = transform_predictors(
        data,
        predictors,
        scheme,
    )

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
    values = np.diag(inverse)
    condition_number = float(
        np.linalg.cond(correlation)
    )

    rows = []

    for variable, value in zip(predictors, values):
        rows.append(
            {
                "Config_ID": configuration["Config_ID"],
                "Climate_Representation": (
                    configuration[
                        "Climate_Representation"
                    ]
                ),
                "Drought_Representation": (
                    configuration[
                        "Drought_Representation"
                    ]
                ),
                "Transformation_Scheme": scheme,
                "Variable": variable,
                "VIF": float(value),
                "VIF_Above_5": bool(value >= 5.0),
                "VIF_Above_10": bool(value >= 10.0),
                "Condition_Number": condition_number,
                "Sample_N": int(len(standardized)),
            }
        )

    summary = {
        "Maximum_VIF": float(np.max(values)),
        "Mean_VIF": float(np.mean(values)),
        "VIF_Above_5_Count": int(
            np.sum(values >= 5.0)
        ),
        "VIF_Above_10_Count": int(
            np.sum(values >= 10.0)
        ),
        "Condition_Number": condition_number,
    }

    return pd.DataFrame(rows), summary


# =============================================================================
# Stable NB1 fitting
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
            maxiter=200,
            tol=1e-8,
            wls_method="pinv",
            disp=False,
        )

    parameters = np.asarray(
        result.params,
        dtype=float,
    )

    if not np.isfinite(parameters).all():
        raise RuntimeError(
            "Poisson GLM starting coefficients are non-finite."
        )

    return parameters


def predicted_mean(
    parameters: np.ndarray,
    design: np.ndarray,
    exposure: np.ndarray,
) -> np.ndarray:
    beta = np.asarray(parameters, dtype=float)[
        : design.shape[1]
    ]

    linear_predictor = (
        design @ beta
        + np.log(np.asarray(exposure, dtype=float))
    )
    linear_predictor = np.clip(
        linear_predictor,
        -745.0,
        700.0,
    )

    result = np.exp(linear_predictor)

    if not np.isfinite(result).all():
        raise FloatingPointError(
            "Predicted mean contains non-finite values."
        )

    return result


def stable_nb1_outputs(
    observed: np.ndarray,
    mean: np.ndarray,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray]:
    observed = np.asarray(observed, dtype=float)
    mean = np.clip(
        np.asarray(mean, dtype=float),
        np.finfo(float).tiny,
        1e300,
    )

    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError(
            "NB1 alpha must be finite and positive."
        )

    size = np.clip(
        mean / float(alpha),
        np.finfo(float).tiny,
        1e300,
    )
    log_success = -math.log1p(float(alpha))
    log_failure = (
        math.log(float(alpha))
        - math.log1p(float(alpha))
    )

    log_probability = np.empty_like(mean)
    zero = observed == 0
    positive = ~zero

    log_probability[zero] = (
        size[zero] * log_success
    )

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

    log_zero_probability = size * log_success
    zero_probability = np.exp(
        np.clip(log_zero_probability, -745.0, 0.0)
    )

    if not np.isfinite(log_probability).all():
        raise FloatingPointError(
            "NB1 log probability contains non-finite values."
        )

    return log_probability, zero_probability


def fit_nb1(
    observed: np.ndarray,
    design: np.ndarray,
    exposure: np.ndarray,
) -> Tuple[object, Dict[str, object]]:
    poisson_parameters = fit_poisson_start(
        observed,
        design,
        exposure,
    )

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
    best_stable_log_likelihood = -np.inf

    for alpha_start in NB_ALPHA_STARTS:
        start_parameters = np.concatenate(
            [
                poisson_parameters,
                np.array([alpha_start], dtype=float),
            ]
        )

        for optimizer in NB_OPTIMIZERS:
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")

                try:
                    result = model.fit(
                        start_params=start_parameters,
                        method=optimizer,
                        maxiter=FIT_MAX_ITERATIONS,
                        disp=0,
                        full_output=True,
                    )

                    parameters = np.asarray(
                        result.params,
                        dtype=float,
                    )
                    alpha = finite_or_nan(parameters[-1])
                    converged = convergence_status(result)

                    stable_log_likelihood = np.nan

                    if (
                        converged
                        and np.isfinite(parameters).all()
                        and np.isfinite(alpha)
                        and alpha > 0
                    ):
                        mean = predicted_mean(
                            parameters,
                            design,
                            exposure,
                        )
                        log_probability, _ = (
                            stable_nb1_outputs(
                                observed,
                                mean,
                                alpha,
                            )
                        )
                        stable_log_likelihood = float(
                            log_probability.sum()
                        )

                    valid = bool(
                        converged
                        and np.isfinite(parameters).all()
                        and np.isfinite(alpha)
                        and alpha > 0
                        and np.isfinite(
                            stable_log_likelihood
                        )
                    )

                    status = {
                        "Optimizer": optimizer,
                        "Alpha_Start": float(alpha_start),
                        "Converged": bool(converged),
                        "Alpha_Estimate": alpha,
                        "Stable_LogLikelihood": (
                            stable_log_likelihood
                        ),
                        "Warning_Text": warning_text(captured),
                        "Valid_Fit": valid,
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
                            "Optimizer": optimizer,
                            "Alpha_Start": float(alpha_start),
                            "Converged": False,
                            "Alpha_Estimate": np.nan,
                            "Stable_LogLikelihood": np.nan,
                            "Warning_Text": (
                                f"{type(error).__name__}: {error}"
                            ),
                            "Valid_Fit": False,
                        }
                    )

        if best_result is not None:
            break

    if best_result is None or best_status is None:
        raise RuntimeError(
            "All NB1 fitting attempts failed:\n"
            + "\n".join(
                json.dumps(
                    attempt,
                    ensure_ascii=False,
                    default=str,
                )
                for attempt in attempts
            )
        )

    best_status = dict(best_status)
    best_status["Attempt_Count"] = len(attempts)
    best_status["Failed_Attempt_Count"] = int(
        sum(
            not bool(item["Valid_Fit"])
            for item in attempts
        )
    )

    return best_result, best_status


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

    rate_intercept, rate_slope = calibration_parameters(
        observed_rate,
        predicted_rate,
    )

    total_observed = float(observed_count.sum())
    total_predicted = float(predicted_count.sum())

    return {
        "N": int(len(observed_count)),
        "Observed_Total_Count": total_observed,
        "Predicted_Total_Count": total_predicted,
        "Predicted_to_Observed_Total_Count_Ratio": (
            total_predicted / total_observed
            if total_observed > 0
            else np.nan
        ),
        "Mean_Negative_LogLikelihood": float(
            -log_probability.mean()
        ),
        "Total_Predictive_LogLikelihood": float(
            log_probability.sum()
        ),
        "Zero_Probability_Brier": float(
            np.mean(
                (
                    predicted_zero_probability
                    - observed_zero
                ) ** 2
            )
        ),
        "Observed_Zero_Proportion": float(
            observed_zero.mean()
        ),
        "Predicted_Zero_Proportion": float(
            predicted_zero_probability.mean()
        ),
        "Occurrence_ROC_AUC": safe_roc_auc(
            observed_positive,
            predicted_positive_probability,
        ),
        "Occurrence_PR_AUC": safe_pr_auc(
            observed_positive,
            predicted_positive_probability,
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
        "Rate_Calibration_Intercept": rate_intercept,
        "Rate_Calibration_Slope": rate_slope,
        "Positive_Rate_MAE": (
            float(
                mean_absolute_error(
                    observed_rate[observed_positive == 1],
                    predicted_rate[observed_positive == 1],
                )
            )
            if observed_positive.any()
            else np.nan
        ),
        "Zero_Row_Predicted_Rate_Mean": (
            float(
                predicted_rate[observed_zero == 1].mean()
            )
            if observed_zero.any()
            else np.nan
        ),
    }


# =============================================================================
# Configuration evaluation
# =============================================================================

def evaluate_configuration(
    season_data: pd.DataFrame,
    response: str,
    rate_name: str,
    configuration: pd.Series,
    save_predictions: bool = False,
) -> Tuple[
    pd.DataFrame,
    Dict[str, object],
    pd.DataFrame,
    pd.DataFrame,
]:
    predictors = parse_predictors(
        configuration["Continuous_Predictors"]
    )
    scheme = str(
        configuration["Transformation_Scheme"]
    )

    fold_rows: List[Dict[str, object]] = []
    status_rows: List[Dict[str, object]] = []
    prediction_frames: List[pd.DataFrame] = []

    observed_pooled: List[np.ndarray] = []
    predicted_pooled: List[np.ndarray] = []
    exposure_pooled: List[np.ndarray] = []
    zero_probability_pooled: List[np.ndarray] = []
    log_probability_pooled: List[np.ndarray] = []

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

        x_training, x_validation = design_matrices(
            training,
            validation,
            predictors,
            scheme,
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

        result, status = fit_nb1(
            y_training,
            x_training,
            exposure_training,
        )

        alpha = float(result.params[-1])
        predicted = predicted_mean(
            result.params,
            x_validation,
            exposure_validation,
        )
        log_probability, zero_probability = (
            stable_nb1_outputs(
                y_validation,
                predicted,
                alpha,
            )
        )

        metrics = prediction_metrics(
            y_validation,
            predicted,
            exposure_validation,
            zero_probability,
            log_probability,
        )
        metrics.update(
            {
                "Config_ID": configuration["Config_ID"],
                "Climate_Representation": (
                    configuration[
                        "Climate_Representation"
                    ]
                ),
                "Drought_Representation": (
                    configuration[
                        "Drought_Representation"
                    ]
                ),
                "Transformation_Scheme": scheme,
                "Continuous_Predictor_Count": len(
                    predictors
                ),
                "Fold": fold,
                "Alpha_Estimate": alpha,
                "Train_N": int(len(training)),
                "Validation_N": int(len(validation)),
            }
        )
        fold_rows.append(metrics)

        status_rows.append(
            {
                "Config_ID": configuration["Config_ID"],
                "Fold": fold,
                "Response": response,
                **status,
            }
        )

        observed_pooled.append(y_validation)
        predicted_pooled.append(predicted)
        exposure_pooled.append(exposure_validation)
        zero_probability_pooled.append(
            zero_probability
        )
        log_probability_pooled.append(
            log_probability
        )

        if save_predictions:
            prediction = validation[
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

            prediction["Count_Response"] = response
            prediction["Rate_Scale_Name"] = rate_name
            prediction["Config_ID"] = (
                configuration["Config_ID"]
            )
            prediction["Observed_Count"] = y_validation
            prediction["Predicted_Count"] = predicted
            prediction["Observed_Rate"] = (
                y_validation / exposure_validation
            )
            prediction["Predicted_Rate"] = (
                predicted / exposure_validation
            )
            prediction[
                "Predicted_Zero_Probability"
            ] = zero_probability
            prediction[
                "Predictive_LogProbability"
            ] = log_probability
            prediction_frames.append(prediction)

    observed = np.concatenate(observed_pooled)
    predicted = np.concatenate(predicted_pooled)
    exposure = np.concatenate(exposure_pooled)
    zero_probability = np.concatenate(
        zero_probability_pooled
    )
    log_probability = np.concatenate(
        log_probability_pooled
    )

    pooled = prediction_metrics(
        observed,
        predicted,
        exposure,
        zero_probability,
        log_probability,
    )
    pooled.update(
        {
            "Config_ID": configuration["Config_ID"],
            "Climate_Representation": (
                configuration[
                    "Climate_Representation"
                ]
            ),
            "Drought_Representation": (
                configuration[
                    "Drought_Representation"
                ]
            ),
            "Transformation_Scheme": scheme,
            "Continuous_Predictor_Count": len(predictors),
            "Continuous_Predictors": ";".join(predictors),
            "Fold_NLL_Mean": float(
                np.mean(
                    [
                        row[
                            "Mean_Negative_LogLikelihood"
                        ]
                        for row in fold_rows
                    ]
                )
            ),
            "Fold_NLL_SD": float(
                np.std(
                    [
                        row[
                            "Mean_Negative_LogLikelihood"
                        ]
                        for row in fold_rows
                    ],
                    ddof=1,
                )
            ),
        }
    )

    predictions = (
        pd.concat(
            prediction_frames,
            ignore_index=True,
        )
        if prediction_frames
        else pd.DataFrame()
    )

    return (
        pd.DataFrame(fold_rows),
        pooled,
        pd.DataFrame(status_rows),
        predictions,
    )


# =============================================================================
# Full-data fit for selected configurations
# =============================================================================

def fit_selected_full_data(
    season_data: pd.DataFrame,
    response: str,
    rate_name: str,
    configuration: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    predictors = parse_predictors(
        configuration["Continuous_Predictors"]
    )
    scheme = str(
        configuration["Transformation_Scheme"]
    )

    design = full_design_matrix(
        season_data,
        predictors,
        scheme,
    )
    observed = (
        season_data[response]
        .to_numpy(dtype=float)
    )
    exposure = (
        season_data["ForestPixelCount"]
        .to_numpy(dtype=float)
    )

    result, status = fit_nb1(
        observed,
        design,
        exposure,
    )

    alpha = float(result.params[-1])
    predicted = predicted_mean(
        result.params,
        design,
        exposure,
    )
    log_probability, zero_probability = (
        stable_nb1_outputs(
            observed,
            predicted,
            alpha,
        )
    )

    metrics = prediction_metrics(
        observed,
        predicted,
        exposure,
        zero_probability,
        log_probability,
    )

    parameter_count = int(len(result.params))
    log_likelihood = float(
        log_probability.sum()
    )

    metrics.update(
        {
            "Config_ID": configuration["Config_ID"],
            "Climate_Representation": (
                configuration[
                    "Climate_Representation"
                ]
            ),
            "Drought_Representation": (
                configuration[
                    "Drought_Representation"
                ]
            ),
            "Transformation_Scheme": scheme,
            "Continuous_Predictor_Count": len(predictors),
            "Continuous_Predictors": ";".join(predictors),
            "Alpha_Estimate": alpha,
            "Parameter_Count": parameter_count,
            "Stable_LogLikelihood": log_likelihood,
            "AIC": float(
                -2.0 * log_likelihood
                + 2.0 * parameter_count
            ),
            "BIC": float(
                -2.0 * log_likelihood
                + math.log(len(observed))
                * parameter_count
            ),
            "Optimizer": status["Optimizer"],
            "Converged": status["Converged"],
            "Warning_Text": status["Warning_Text"],
        }
    )

    terms = [
        "Intercept",
        *predictors,
        *COUNTRY_DUMMY_COLUMNS,
        "Alpha",
    ]
    estimates = np.asarray(
        result.params,
        dtype=float,
    )

    try:
        standard_errors = np.asarray(
            result.bse,
            dtype=float,
        )
    except Exception:
        standard_errors = np.full(
            len(estimates),
            np.nan,
        )

    if len(standard_errors) != len(estimates):
        standard_errors = np.full(
            len(estimates),
            np.nan,
        )

    coefficient_rows = []

    for term, estimate, standard_error in zip(
        terms,
        estimates,
        standard_errors,
    ):
        coefficient_rows.append(
            {
                "Config_ID": configuration["Config_ID"],
                "Term": term,
                "Coefficient": float(estimate),
                "Standard_Error": finite_or_nan(
                    standard_error
                ),
                "CI95_Lower": (
                    float(
                        estimate
                        - 1.959963984540054
                        * standard_error
                    )
                    if np.isfinite(standard_error)
                    else np.nan
                ),
                "CI95_Upper": (
                    float(
                        estimate
                        + 1.959963984540054
                        * standard_error
                    )
                    if np.isfinite(standard_error)
                    else np.nan
                ),
                "Coefficient_Scale": (
                    "NB1 dispersion parameter"
                    if term == "Alpha"
                    else (
                        "Log rate; Country contrast versus China"
                        if term in COUNTRY_DUMMY_COLUMNS
                        else (
                            "Log rate; per 1 SD of modeled predictor"
                            if term != "Intercept"
                            else "Log rate intercept"
                        )
                    )
                ),
                "Transformation": (
                    "log1p_then_z_standardized"
                    if (
                        term in predictors
                        and scheme
                        == "Audit_Guided_Log1p"
                        and term
                        in AUDIT_GUIDED_LOG1P_VARIABLES
                    )
                    else (
                        "z_standardized"
                        if term in predictors
                        else "Not_applicable"
                    )
                ),
            }
        )

    return (
        pd.DataFrame([metrics]),
        pd.DataFrame(coefficient_rows),
    )


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    resume_mode = (
        OUTPUT_ROOT.exists()
        and CHECKPOINT_ROOT.exists()
    )

    if not resume_mode:
        if OUTPUT_ROOT.exists():
            shutil.rmtree(OUTPUT_ROOT)

        OUTPUT_ROOT.mkdir(
            parents=True,
            exist_ok=False,
        )
        CHECKPOINT_ROOT.mkdir(
            parents=False,
            exist_ok=False,
        )
    else:
        CHECKPOINT_ROOT.mkdir(
            parents=True,
            exist_ok=True,
        )

    log(
        "Starting single-stage NB1 configuration comparison."
    )
    log(
        "Resume mode: "
        + ("enabled" if resume_mode else "disabled")
        + "."
    )
    log(f"Input: {INPUT_CSV}")
    log(f"Output: {OUTPUT_ROOT}")

    family_selection = validate_family_selection()
    configurations = build_configuration_table()
    data, excluded = prepare_data(configurations)

    log(
        f"Validated NB1 selection for all six models."
    )
    log(
        f"Configurations: {len(configurations)}."
    )
    log(
        f"Common complete-case sample: {len(data):,} rows; "
        f"{data['GRID_UID'].nunique():,} grids."
    )

    all_vif_rows: List[pd.DataFrame] = []
    vif_summaries: Dict[str, Dict[str, float]] = {}

    for _, configuration in configurations.iterrows():
        vif_rows, vif_summary = configuration_vif(
            data,
            configuration,
        )
        all_vif_rows.append(vif_rows)
        vif_summaries[
            str(configuration["Config_ID"])
        ] = vif_summary

    vif_table = pd.concat(
        all_vif_rows,
        ignore_index=True,
    )

    fold_tables: List[pd.DataFrame] = []
    pooled_rows: List[Dict[str, object]] = []
    status_tables: List[pd.DataFrame] = []

    for season_code, season_label in SEASONS.items():
        season_data = (
            data.loc[data["Season"] == season_code]
            .copy()
            .reset_index(drop=True)
        )

        for response, rate_name in COUNT_RESPONSES.items():
            log(
                f"Starting {season_label} {response} "
                f"configuration comparison."
            )

            for index, configuration in configurations.iterrows():
                config_id = str(
                    configuration["Config_ID"]
                )
                saved_checkpoint = checkpoint_path(
                    season_code,
                    response,
                    config_id,
                )

                if saved_checkpoint.exists():
                    (
                        fold_metrics,
                        pooled,
                        fit_status,
                    ) = load_checkpoint(
                        saved_checkpoint
                    )
                    log(
                        f"Reused checkpoint: "
                        f"{season_label} {response} "
                        f"{config_id}."
                    )
                else:
                    (
                        fold_metrics,
                        pooled,
                        fit_status,
                        _,
                    ) = evaluate_configuration(
                        season_data=season_data,
                        response=response,
                        rate_name=rate_name,
                        configuration=configuration,
                        save_predictions=False,
                    )
                    save_checkpoint(
                        saved_checkpoint,
                        fold_metrics,
                        pooled,
                        fit_status,
                    )

                vif_summary = vif_summaries[config_id]

                fold_metrics.insert(
                    0,
                    "Season_Code",
                    season_code,
                )
                fold_metrics.insert(
                    1,
                    "Season_Label",
                    season_label,
                )
                fold_metrics.insert(
                    2,
                    "Count_Response",
                    response,
                )
                fold_metrics.insert(
                    3,
                    "Rate_Scale_Name",
                    rate_name,
                )

                fit_status.insert(
                    0,
                    "Season_Code",
                    season_code,
                )
                fit_status.insert(
                    1,
                    "Season_Label",
                    season_label,
                )
                fit_status.insert(
                    2,
                    "Count_Response",
                    response,
                )
                fit_status.insert(
                    3,
                    "Rate_Scale_Name",
                    rate_name,
                )

                pooled.update(
                    {
                        "Season_Code": season_code,
                        "Season_Label": season_label,
                        "Count_Response": response,
                        "Rate_Scale_Name": rate_name,
                        **vif_summary,
                    }
                )

                fold_tables.append(fold_metrics)
                status_tables.append(fit_status)
                pooled_rows.append(pooled)

                log(
                    f"Completed {season_label} {response} "
                    f"{config_id} "
                    f"({index + 1}/{len(configurations)}): "
                    f"NLL={pooled['Mean_Negative_LogLikelihood']:.6f}; "
                    f"zero Brier={pooled['Zero_Probability_Brier']:.6f}; "
                    f"rate RMSE={pooled['Rate_RMSE']:.8f}; "
                    f"max VIF={pooled['Maximum_VIF']:.3f}."
                )

    fold_metrics = pd.concat(
        fold_tables,
        ignore_index=True,
    )
    pooled_metrics = pd.DataFrame(pooled_rows)
    fit_status = pd.concat(
        status_tables,
        ignore_index=True,
    )

    pooled_metrics["VIF_Admissible"] = (
        pooled_metrics["Maximum_VIF"]
        < MAXIMUM_VIF_FOR_SELECTION
    )

    convergence_summary = (
        fit_status.groupby(
            [
                "Season_Code",
                "Count_Response",
                "Config_ID",
            ],
            as_index=False,
        )["Converged"]
        .all()
        .rename(
            columns={
                "Converged": "All_Folds_Converged"
            }
        )
    )

    pooled_metrics = pooled_metrics.merge(
        convergence_summary,
        on=[
            "Season_Code",
            "Count_Response",
            "Config_ID",
        ],
        how="left",
        validate="one_to_one",
    )

    if pooled_metrics["All_Folds_Converged"].isna().any():
        raise RuntimeError(
            "Missing convergence summaries for one or more "
            "season-response-configuration combinations."
        )

    pooled_metrics["Eligible_for_Selection"] = (
        pooled_metrics["VIF_Admissible"]
        & pooled_metrics["All_Folds_Converged"]
    )

    ranking_parts = []

    for (
        season_code,
        response,
    ), subset in pooled_metrics.groupby(
        ["Season_Code", "Count_Response"],
        sort=True,
    ):
        subset = subset.copy()

        eligible = subset.loc[
            subset["Eligible_for_Selection"]
        ].copy()

        if eligible.empty:
            raise RuntimeError(
                f"No eligible NB1 configuration for "
                f"season {season_code}, response {response}."
            )

        eligible = eligible.sort_values(
            [
                "Mean_Negative_LogLikelihood",
                "Zero_Probability_Brier",
                "Rate_RMSE",
                "Continuous_Predictor_Count",
                "Maximum_VIF",
                "Config_ID",
            ]
        )
        eligible["Configuration_Rank"] = np.arange(
            1,
            len(eligible) + 1,
        )

        ineligible = subset.loc[
            ~subset["Eligible_for_Selection"]
        ].copy()
        ineligible["Configuration_Rank"] = np.nan

        ranking_parts.extend([eligible, ineligible])

    ranking = pd.concat(
        ranking_parts,
        ignore_index=True,
    ).sort_values(
        [
            "Season_Code",
            "Count_Response",
            "Configuration_Rank",
            "Config_ID",
        ],
        na_position="last",
    ).reset_index(drop=True)

    top5 = (
        ranking.loc[
            ranking["Configuration_Rank"].notna()
            & (ranking["Configuration_Rank"] <= 5)
        ]
        .copy()
        .sort_values(
            [
                "Season_Code",
                "Count_Response",
                "Configuration_Rank",
            ]
        )
        .reset_index(drop=True)
    )

    best = (
        ranking.loc[
            ranking["Configuration_Rank"] == 1
        ]
        .copy()
        .sort_values(
            ["Season_Code", "Count_Response"]
        )
        .reset_index(drop=True)
    )
    best["Status"] = (
        "Provisional optimized NB1 representation; "
        "predictor refinement remains to be performed"
    )

    best_prediction_tables: List[pd.DataFrame] = []
    best_full_fit_tables: List[pd.DataFrame] = []
    best_coefficient_tables: List[pd.DataFrame] = []

    for _, best_row in best.iterrows():
        season_code = int(best_row["Season_Code"])
        response = str(best_row["Count_Response"])
        rate_name = str(best_row["Rate_Scale_Name"])
        config_id = str(best_row["Config_ID"])

        configuration = configurations.loc[
            configurations["Config_ID"] == config_id
        ].iloc[0]

        season_data = (
            data.loc[data["Season"] == season_code]
            .copy()
            .reset_index(drop=True)
        )

        (
            _,
            _,
            _,
            predictions,
        ) = evaluate_configuration(
            season_data=season_data,
            response=response,
            rate_name=rate_name,
            configuration=configuration,
            save_predictions=True,
        )

        prediction_path = (
            OUTPUT_ROOT
            / (
                f"{BEST_PREDICTION_SAMPLE_FILE_PREFIX}_"
                f"S{season_code}_{response}.csv.gz"
            )
        )
        predictions.to_csv(
            prediction_path,
            index=False,
            compression="gzip",
        )
        best_prediction_tables.append(
            pd.DataFrame(
                [
                    {
                        "Season_Code": season_code,
                        "Season_Label": SEASONS[
                            season_code
                        ],
                        "Count_Response": response,
                        "Rate_Scale_Name": rate_name,
                        "Config_ID": config_id,
                        "Prediction_File": (
                            prediction_path.name
                        ),
                        "Rows": len(predictions),
                    }
                ]
            )
        )

        full_fit, coefficients = (
            fit_selected_full_data(
                season_data=season_data,
                response=response,
                rate_name=rate_name,
                configuration=configuration,
            )
        )
        full_fit.insert(
            0,
            "Season_Code",
            season_code,
        )
        full_fit.insert(
            1,
            "Season_Label",
            SEASONS[season_code],
        )
        full_fit.insert(
            2,
            "Count_Response",
            response,
        )
        full_fit.insert(
            3,
            "Rate_Scale_Name",
            rate_name,
        )

        coefficients.insert(
            0,
            "Season_Code",
            season_code,
        )
        coefficients.insert(
            1,
            "Season_Label",
            SEASONS[season_code],
        )
        coefficients.insert(
            2,
            "Count_Response",
            response,
        )
        coefficients.insert(
            3,
            "Rate_Scale_Name",
            rate_name,
        )

        best_full_fit_tables.append(full_fit)
        best_coefficient_tables.append(
            coefficients
        )

    best_prediction_manifest = pd.concat(
        best_prediction_tables,
        ignore_index=True,
    )
    best_full_fit = pd.concat(
        best_full_fit_tables,
        ignore_index=True,
    )
    best_coefficients = pd.concat(
        best_coefficient_tables,
        ignore_index=True,
    )

    qa = pd.DataFrame(
        [
            (
                "Rows_in_common_complete_case_sample",
                len(data),
            ),
            (
                "Excluded_incomplete_rows",
                len(excluded),
            ),
            (
                "Unique_GRID_UIDs",
                data["GRID_UID"].nunique(),
            ),
            ("Year_min", int(data["Year"].min())),
            ("Year_max", int(data["Year"].max())),
            (
                "Compared_configuration_count",
                len(configurations),
            ),
            (
                "Climate_representation_count",
                len(CLIMATE_REPRESENTATIONS),
            ),
            (
                "Drought_representation_count",
                len(DROUGHT_REPRESENTATIONS),
            ),
            (
                "Transformation_scheme_count",
                len(TRANSFORMATION_SCHEMES),
            ),
            (
                "Distribution_family",
                "NB1",
            ),
            (
                "Exposure",
                "ForestPixelCount",
            ),
            (
                "Maximum_VIF_for_selection",
                f"< {MAXIMUM_VIF_FOR_SELECTION}",
            ),
            (
                "Primary_selection_metric",
                "Pooled temporal OOF mean negative log-likelihood",
            ),
            (
                "Random_forest_variable_selection_used",
                False,
            ),
        ],
        columns=["Metric", "Value"],
    )

    method = {
        "workflow_stage":
            "Single-stage NB1 predictor-representation comparison",
        "distribution_family":
            "Negative binomial type 1",
        "variance_function":
            "Var(Y | X) = mu + alpha * mu",
        "responses":
            list(COUNT_RESPONSES.keys()),
        "exposure":
            "ForestPixelCount",
        "equivalent_offset":
            "log(ForestPixelCount)",
        "common_predictors":
            COMMON_PREDICTORS,
        "climate_representations":
            CLIMATE_REPRESENTATIONS,
        "drought_representations":
            DROUGHT_REPRESENTATIONS,
        "transformation_schemes":
            TRANSFORMATION_SCHEMES,
        "log1p_variables":
            sorted(
                AUDIT_GUIDED_LOG1P_VARIABLES
            ),
        "country_effect":
            "Fixed contextual effect; China reference",
        "temporal_folds":
            TEMPORAL_FOLDS,
        "selection_rule":
            "Among converged configurations with maximum VIF < 10, "
            "rank first by pooled temporal OOF mean negative "
            "log-likelihood, then zero-probability Brier score, "
            "rate RMSE, predictor count, and maximum VIF.",
        "scope_note":
            "This step selects predictor representations, not the "
            "final predictor subset. Predictor refinement follows in "
            "the next workflow step.",
        "uses_random_forest_variable_selection":
            False,
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
        json.dump(method, handle, indent=2)

    qa.to_csv(
        OUTPUT_ROOT / "01_Data_QA_Summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    family_selection.to_csv(
        OUTPUT_ROOT
        / "02_Confirmed_NB1_Family_Selection.csv",
        index=False,
        encoding="utf-8-sig",
    )
    configurations.to_csv(
        OUTPUT_ROOT
        / "03_Candidate_Configuration_Definitions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    vif_table.to_csv(
        OUTPUT_ROOT
        / "04_Configuration_VIF_Details.csv",
        index=False,
        encoding="utf-8-sig",
    )
    fold_metrics.to_csv(
        OUTPUT_ROOT
        / "05_Temporal_Fold_Metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pooled_metrics.to_csv(
        OUTPUT_ROOT
        / "06_Pooled_Temporal_OOF_Metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    ranking.to_csv(
        OUTPUT_ROOT
        / "07_Configuration_Ranking.csv",
        index=False,
        encoding="utf-8-sig",
    )
    top5.to_csv(
        OUTPUT_ROOT
        / "08_Top5_Configurations.csv",
        index=False,
        encoding="utf-8-sig",
    )
    best.to_csv(
        OUTPUT_ROOT
        / "09_Provisional_Best_NB1_Configuration.csv",
        index=False,
        encoding="utf-8-sig",
    )
    fit_status.to_csv(
        OUTPUT_ROOT
        / "10_Model_Fit_Status.csv",
        index=False,
        encoding="utf-8-sig",
    )
    best_full_fit.to_csv(
        OUTPUT_ROOT
        / "11_Best_Configuration_Full_Data_Fit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    best_coefficients.to_csv(
        OUTPUT_ROOT
        / "12_Best_Configuration_Full_Data_Coefficients.csv",
        index=False,
        encoding="utf-8-sig",
    )
    best_prediction_manifest.to_csv(
        OUTPUT_ROOT
        / "13_Best_OOF_Prediction_Manifest.csv",
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

    if CHECKPOINT_ROOT.exists():
        shutil.rmtree(CHECKPOINT_ROOT)

    log(
        "Single-stage NB1 configuration comparison "
        "completed successfully."
    )
    log(
        "Next step: review the top configurations before "
        "NB1-specific predictor refinement."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
