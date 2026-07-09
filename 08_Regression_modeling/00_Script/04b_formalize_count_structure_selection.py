# -*- coding: utf-8 -*-
"""
Formal stability audit and selection of wildfire count-model structure.

Recommended public location
---------------------------
<CODE_ROOT>/08_Regression_modeling/00_Script/
    04b_formalize_count_structure_selection.py

Inputs
------
<CODE_ROOT>/08_Regression_modeling/
    04_NB1_Hurdle_ZINB_Matched_Comparison/
        03_Temporal_Fold_Metrics.csv
        04_Pooled_Temporal_OOF_Metrics.csv
        08_Temporal_Model_Fit_Status.csv
        Matched_OOF_Predictions_*.csv.gz

Output
------
<CODE_ROOT>/08_Regression_modeling/
    04b_Formal_Count_Structure_Selection/

Purpose
-------
This script performs a formal prediction-stability audit after the matched
comparison of:

1. single-stage NB1;
2. hurdle NB1;
3. ZINB1.

It does not refit any model.

The script independently recalculates temporal out-of-fold metrics from the
full prediction files, checks them against the Step-04 summary tables, applies
generic stability rules, identifies practically equivalent negative
log-likelihood values, and produces the formal count-structure selection.

The stability rules are intentionally generous and are designed to identify
catastrophic numerical extrapolation rather than ordinary prediction error.

Selection sequence
------------------
1. Exclude models that fail convergence, numerical-validity, calibration,
   or relative-RMSE stability rules.
2. Among stable models, identify the minimum pooled temporal OOF negative
   log-likelihood.
3. Treat another stable model as NLL-equivalent only when both:
       absolute delta NLL <= 0.0001
       relative delta NLL <= 0.01 percent
4. Within an NLL-equivalent set, select by:
       a. lower zero-probability Brier score;
       b. lower rate-scale RMSE;
       c. smaller absolute total-count calibration error;
       d. lower model-complexity tier;
       e. model name for deterministic ordering.

The complete decision path is saved for reproducibility.
"""

from __future__ import annotations

import json
import math
import platform
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# Repository-relative paths
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REGRESSION_ROOT = SCRIPT_DIR.parent

SOURCE_ROOT = (
    REGRESSION_ROOT
    / "04_NB1_Hurdle_ZINB_Matched_Comparison"
)

SOURCE_FOLD_METRICS = (
    SOURCE_ROOT / "03_Temporal_Fold_Metrics.csv"
)
SOURCE_POOLED_METRICS = (
    SOURCE_ROOT / "04_Pooled_Temporal_OOF_Metrics.csv"
)
SOURCE_FIT_STATUS = (
    SOURCE_ROOT / "08_Temporal_Model_Fit_Status.csv"
)

OUTPUT_ROOT = (
    REGRESSION_ROOT
    / "04b_Formal_Count_Structure_Selection"
)

LOG_FILE = (
    OUTPUT_ROOT
    / "formal_count_structure_selection.log"
)


# =============================================================================
# Expected analysis structure
# =============================================================================

SEASONS = {
    1: "Spring",
    2: "Summer",
    3: "Autumn",
}

COUNT_RESPONSES = {
    "Fire_Count": "FCD",
    "Burned_Pixel_Count": "BAD",
}

MODEL_STRUCTURES = [
    "Single_Stage_NB1",
    "Hurdle_NB1",
    "ZINB1",
]

EXPECTED_FOLDS = {
    1, 2, 3, 4, 5,
}

MODEL_COMPLEXITY_TIER = {
    "Single_Stage_NB1": 1,
    "Hurdle_NB1": 2,
    "ZINB1": 2,
}


# =============================================================================
# Formal stability thresholds
# =============================================================================

# All prediction columns must be finite and internally valid.
NUMERICAL_TOLERANCE = 1e-10

# A pooled or fold-specific predicted/observed total-count ratio outside
# [0.1, 10] is considered catastrophic calibration failure.
TOTAL_COUNT_RATIO_LOWER = 0.10
TOTAL_COUNT_RATIO_UPPER = 10.0

# A model's pooled and fold-specific rate RMSE may not exceed ten times the
# matched single-stage NB1 rate RMSE.
MAX_RMSE_MULTIPLE_VS_BASELINE = 10.0

# A model may not predict a maximum rate more than 100 times the maximum
# observed rate in the same pooled or fold-specific sample.
MAX_PREDICTED_RATE_MULTIPLE_OF_OBSERVED_MAX = 100.0

# NLL equivalence requires both criteria.
NLL_EQUIVALENCE_ABSOLUTE = 0.0001
NLL_EQUIVALENCE_RELATIVE_PERCENT = 0.01

# Recalculated and source metrics must agree under a combined
# absolute-relative tolerance. A relative tolerance is essential because
# an unstable candidate model can produce extremely large finite values;
# harmless CSV round-trip differences can then be large in absolute terms
# while remaining negligible relative to the value itself.
SOURCE_RECONCILIATION_ABSOLUTE_TOLERANCE = 1e-7
SOURCE_RECONCILIATION_RELATIVE_TOLERANCE = 1e-9


# =============================================================================
# Logging
# =============================================================================

def log(message: str) -> None:
    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    line = f"[{timestamp}] {message}"
    print(line, flush=True)

    with LOG_FILE.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(line + "\n")


# =============================================================================
# General helpers
# =============================================================================

def robust_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if value is None:
        return False

    if isinstance(value, (int, np.integer)):
        return bool(value)

    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return False
        return bool(value)

    text = str(value).strip().lower()

    if text in {
        "true", "t", "1", "yes", "y",
    }:
        return True

    if text in {
        "false", "f", "0", "no", "n",
        "", "nan", "none",
    }:
        return False

    raise ValueError(
        f"Cannot interpret boolean value: {value!r}"
    )


def finite_or_nan(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan

    return number if np.isfinite(number) else np.nan


def safe_ratio(
    numerator: float,
    denominator: float,
) -> float:
    if (
        not np.isfinite(numerator)
        or not np.isfinite(denominator)
        or denominator == 0
    ):
        return np.nan

    return float(numerator / denominator)


def safe_rmse(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> float:
    difference = (
        np.asarray(predicted, dtype=float)
        - np.asarray(observed, dtype=float)
    )

    return float(
        np.sqrt(
            np.mean(
                np.square(difference)
            )
        )
    )


def safe_r2(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> float:
    observed = np.asarray(
        observed,
        dtype=float,
    )
    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    denominator = float(
        np.sum(
            np.square(
                observed - observed.mean()
            )
        )
    )

    if denominator <= 0:
        return np.nan

    numerator = float(
        np.sum(
            np.square(
                observed - predicted
            )
        )
    )

    return float(
        1.0 - numerator / denominator
    )


def maximum_rate_multiple(
    observed_rate: np.ndarray,
    predicted_rate: np.ndarray,
) -> float:
    observed_rate = np.asarray(
        observed_rate,
        dtype=float,
    )
    predicted_rate = np.asarray(
        predicted_rate,
        dtype=float,
    )

    observed_maximum = float(
        np.max(observed_rate)
    )
    predicted_maximum = float(
        np.max(predicted_rate)
    )

    if observed_maximum > 0:
        return float(
            predicted_maximum / observed_maximum
        )

    if predicted_maximum <= 0:
        return 1.0

    return np.inf


def required_columns_present(
    table: pd.DataFrame,
    columns: Iterable[str],
    table_name: str,
) -> None:
    missing = sorted(
        set(columns) - set(table.columns)
    )

    if missing:
        raise ValueError(
            f"{table_name} is missing required columns:\n"
            + "\n".join(missing)
        )


# =============================================================================
# Input validation
# =============================================================================

def prediction_file_mapping() -> Dict[
    Tuple[int, str],
    Path,
]:
    files = sorted(
        SOURCE_ROOT.glob(
            "Matched_OOF_Predictions_"
            "S*_*.csv.gz"
        )
    )

    mapping: Dict[
        Tuple[int, str],
        Path,
    ] = {}

    for path in files:
        name = path.name

        matched = False

        for season_code in SEASONS:
            for response in COUNT_RESPONSES:
                expected = (
                    "Matched_OOF_Predictions_"
                    f"S{season_code}_{response}.csv.gz"
                )

                if name == expected:
                    key = (
                        season_code,
                        response,
                    )

                    if key in mapping:
                        raise RuntimeError(
                            f"Duplicate prediction file for {key}."
                        )

                    mapping[key] = path
                    matched = True
                    break

            if matched:
                break

    expected_keys = {
        (season_code, response)
        for season_code in SEASONS
        for response in COUNT_RESPONSES
    }

    if set(mapping) != expected_keys:
        missing = sorted(
            expected_keys - set(mapping)
        )
        unexpected = sorted(
            set(mapping) - expected_keys
        )

        raise FileNotFoundError(
            "The complete set of six matched OOF prediction "
            "files was not found.\n"
            f"Missing: {missing}\n"
            f"Unexpected: {unexpected}"
        )

    return mapping


def read_source_tables() -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    required_files = [
        SOURCE_FOLD_METRICS,
        SOURCE_POOLED_METRICS,
        SOURCE_FIT_STATUS,
    ]

    missing = [
        path
        for path in required_files
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Required Step-04 source files are missing:\n"
            + "\n".join(
                str(path)
                for path in missing
            )
        )

    fold_metrics = pd.read_csv(
        SOURCE_FOLD_METRICS
    )
    pooled_metrics = pd.read_csv(
        SOURCE_POOLED_METRICS
    )
    fit_status = pd.read_csv(
        SOURCE_FIT_STATUS
    )

    common_required = {
        "Season_Code",
        "Count_Response",
        "Model_Structure",
    }

    required_columns_present(
        fold_metrics,
        {
            *common_required,
            "Fold",
            "Mean_Negative_LogLikelihood",
            "Zero_Probability_Brier",
            "Rate_RMSE",
            "Predicted_to_Observed_Total_Count_Ratio",
        },
        "Step-04 fold metrics",
    )
    required_columns_present(
        pooled_metrics,
        {
            *common_required,
            "Mean_Negative_LogLikelihood",
            "Zero_Probability_Brier",
            "Rate_RMSE",
            "Predicted_to_Observed_Total_Count_Ratio",
        },
        "Step-04 pooled metrics",
    )
    required_columns_present(
        fit_status,
        {
            *common_required,
            "Fold",
            "Converged",
            "Valid_Fit",
        },
        "Step-04 fit status",
    )

    expected_pairs = {
        (
            season_code,
            response,
            model,
        )
        for season_code in SEASONS
        for response in COUNT_RESPONSES
        for model in MODEL_STRUCTURES
    }

    observed_pooled_pairs = {
        (
            int(row.Season_Code),
            str(row.Count_Response),
            str(row.Model_Structure),
        )
        for row in pooled_metrics.itertuples(
            index=False
        )
    }

    if observed_pooled_pairs != expected_pairs:
        raise ValueError(
            "Step-04 pooled metrics do not contain exactly "
            "the expected 18 season-response-model combinations."
        )

    return (
        fold_metrics,
        pooled_metrics,
        fit_status,
    )


# =============================================================================
# Prediction-file validation and metric calculation
# =============================================================================

PREDICTION_REQUIRED_COLUMNS = {
    "GRID_UID",
    "GRID_ID",
    "Country",
    "Year",
    "Season",
    "Temporal_Fold",
    "ForestPixelCount",
    "Count_Response",
    "Rate_Scale_Name",
    "Model_Structure",
    "Fold",
    "Observed_Count",
    "Predicted_Count",
    "Observed_Rate",
    "Predicted_Rate",
    "Observed_Zero",
    "Predicted_Zero_Probability",
    "Predictive_LogProbability",
    "Config_ID",
}


def validate_prediction_file(
    data: pd.DataFrame,
    season_code: int,
    response: str,
    source_path: Path,
) -> None:
    required_columns_present(
        data,
        PREDICTION_REQUIRED_COLUMNS,
        source_path.name,
    )

    if data.empty:
        raise ValueError(
            f"Prediction file is empty: {source_path}"
        )

    if set(
        data["Season"].astype(int).unique()
    ) != {season_code}:
        raise ValueError(
            f"Unexpected Season values in {source_path.name}."
        )

    if set(
        data["Count_Response"].astype(str).unique()
    ) != {response}:
        raise ValueError(
            f"Unexpected Count_Response in {source_path.name}."
        )

    if set(
        data["Rate_Scale_Name"].astype(str).unique()
    ) != {
        COUNT_RESPONSES[response]
    }:
        raise ValueError(
            f"Unexpected Rate_Scale_Name in {source_path.name}."
        )

    if set(
        data["Model_Structure"].astype(str).unique()
    ) != set(MODEL_STRUCTURES):
        raise ValueError(
            f"Unexpected model structures in {source_path.name}."
        )

    if set(
        data["Fold"].astype(int).unique()
    ) != EXPECTED_FOLDS:
        raise ValueError(
            f"Unexpected fold values in {source_path.name}."
        )

    if not (
        data["Fold"].astype(int)
        == data["Temporal_Fold"].astype(int)
    ).all():
        raise ValueError(
            f"Fold and Temporal_Fold disagree in {source_path.name}."
        )

    if (
        pd.to_numeric(
            data["ForestPixelCount"],
            errors="coerce",
        )
        <= 0
    ).any():
        raise ValueError(
            f"Nonpositive ForestPixelCount in {source_path.name}."
        )

    # Each validation record must occur once for each model.
    record_columns = [
        "GRID_UID",
        "Year",
        "Season",
        "Fold",
    ]

    counts = (
        data.groupby(
            record_columns,
            dropna=False,
        )["Model_Structure"]
        .nunique()
    )

    if (
        counts != len(MODEL_STRUCTURES)
    ).any():
        raise ValueError(
            f"Not every validation record has all three model "
            f"predictions in {source_path.name}."
        )

    duplicate_model_record = data.duplicated(
        [
            *record_columns,
            "Model_Structure",
        ],
        keep=False,
    )

    if duplicate_model_record.any():
        raise ValueError(
            f"Duplicate record-model predictions in "
            f"{source_path.name}."
        )

    # Observed outcomes and exposures must be identical among models.
    consistency = (
        data.groupby(
            record_columns,
            dropna=False,
        )
        .agg(
            Observed_Count_Nunique=(
                "Observed_Count",
                "nunique",
            ),
            Observed_Rate_Nunique=(
                "Observed_Rate",
                "nunique",
            ),
            Exposure_Nunique=(
                "ForestPixelCount",
                "nunique",
            ),
            Observed_Zero_Nunique=(
                "Observed_Zero",
                "nunique",
            ),
        )
    )

    if (
        consistency.to_numpy() != 1
    ).any():
        raise ValueError(
            f"Observed values differ among models in "
            f"{source_path.name}."
        )


def metric_row(
    subset: pd.DataFrame,
    season_code: int,
    response: str,
    model_structure: str,
    fold: int | str,
) -> Dict[str, object]:
    observed_count = pd.to_numeric(
        subset["Observed_Count"],
        errors="coerce",
    ).to_numpy(dtype=float)

    predicted_count = pd.to_numeric(
        subset["Predicted_Count"],
        errors="coerce",
    ).to_numpy(dtype=float)

    observed_rate = pd.to_numeric(
        subset["Observed_Rate"],
        errors="coerce",
    ).to_numpy(dtype=float)

    predicted_rate = pd.to_numeric(
        subset["Predicted_Rate"],
        errors="coerce",
    ).to_numpy(dtype=float)

    zero_probability = pd.to_numeric(
        subset["Predicted_Zero_Probability"],
        errors="coerce",
    ).to_numpy(dtype=float)

    log_probability = pd.to_numeric(
        subset["Predictive_LogProbability"],
        errors="coerce",
    ).to_numpy(dtype=float)

    observed_zero = (
        observed_count == 0
    ).astype(float)

    finite_columns = np.column_stack(
        [
            observed_count,
            predicted_count,
            observed_rate,
            predicted_rate,
            zero_probability,
            log_probability,
        ]
    )
    nonfinite_count = int(
        (~np.isfinite(finite_columns)).sum()
    )

    negative_prediction_count = int(
        np.sum(
            predicted_count
            < -NUMERICAL_TOLERANCE
        )
    )
    negative_rate_count = int(
        np.sum(
            predicted_rate
            < -NUMERICAL_TOLERANCE
        )
    )
    probability_outside_unit_interval_count = int(
        np.sum(
            (
                zero_probability
                < -NUMERICAL_TOLERANCE
            )
            | (
                zero_probability
                > 1.0 + NUMERICAL_TOLERANCE
            )
        )
    )
    positive_log_probability_count = int(
        np.sum(
            log_probability
            > NUMERICAL_TOLERANCE
        )
    )

    observed_total = float(
        np.sum(observed_count)
    )
    predicted_total = float(
        np.sum(predicted_count)
    )

    mean_nll = float(
        -np.mean(log_probability)
    )
    zero_brier = float(
        np.mean(
            np.square(
                zero_probability
                - observed_zero
            )
        )
    )
    rate_rmse = safe_rmse(
        observed_rate,
        predicted_rate,
    )

    observed_rate_maximum = float(
        np.max(observed_rate)
    )
    predicted_rate_maximum = float(
        np.max(predicted_rate)
    )

    observed_count_maximum = float(
        np.max(observed_count)
    )
    predicted_count_maximum = float(
        np.max(predicted_count)
    )

    return {
        "Season_Code": season_code,
        "Season_Label": SEASONS[
            season_code
        ],
        "Count_Response": response,
        "Rate_Scale_Name": (
            COUNT_RESPONSES[response]
        ),
        "Model_Structure": model_structure,
        "Fold": fold,
        "N": int(len(subset)),
        "Observed_Total_Count": observed_total,
        "Predicted_Total_Count": predicted_total,
        "Predicted_to_Observed_Total_Count_Ratio": (
            safe_ratio(
                predicted_total,
                observed_total,
            )
        ),
        "Mean_Negative_LogLikelihood": mean_nll,
        "Total_Predictive_LogLikelihood": float(
            np.sum(log_probability)
        ),
        "Zero_Probability_Brier": zero_brier,
        "Observed_Zero_Proportion": float(
            np.mean(observed_zero)
        ),
        "Predicted_Zero_Proportion": float(
            np.mean(zero_probability)
        ),
        "Predicted_minus_Observed_Zero_Proportion": float(
            np.mean(zero_probability)
            - np.mean(observed_zero)
        ),
        "Rate_RMSE": rate_rmse,
        "Rate_MAE": float(
            np.mean(
                np.abs(
                    predicted_rate
                    - observed_rate
                )
            )
        ),
        "Rate_R2": safe_r2(
            observed_rate,
            predicted_rate,
        ),
        "Observed_Count_Maximum": observed_count_maximum,
        "Predicted_Count_Maximum": predicted_count_maximum,
        "Predicted_Count_Maximum_to_Observed_Maximum_Ratio": (
            safe_ratio(
                predicted_count_maximum,
                observed_count_maximum,
            )
        ),
        "Observed_Rate_Maximum": observed_rate_maximum,
        "Predicted_Rate_Maximum": predicted_rate_maximum,
        "Predicted_Rate_Maximum_to_Observed_Maximum_Ratio": (
            maximum_rate_multiple(
                observed_rate,
                predicted_rate,
            )
        ),
        "Nonfinite_Value_Count": nonfinite_count,
        "Negative_Predicted_Count_Count": (
            negative_prediction_count
        ),
        "Negative_Predicted_Rate_Count": (
            negative_rate_count
        ),
        "Zero_Probability_Outside_0_1_Count": (
            probability_outside_unit_interval_count
        ),
        "Positive_LogProbability_Count": (
            positive_log_probability_count
        ),
    }


def calculate_prediction_audits(
    data: pd.DataFrame,
    season_code: int,
    response: str,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    pooled_rows: List[Dict[str, object]] = []
    fold_rows: List[Dict[str, object]] = []

    for model_structure in MODEL_STRUCTURES:
        model_data = data.loc[
            data["Model_Structure"]
            == model_structure
        ].copy()

        pooled_rows.append(
            metric_row(
                subset=model_data,
                season_code=season_code,
                response=response,
                model_structure=model_structure,
                fold="Pooled",
            )
        )

        for fold in sorted(EXPECTED_FOLDS):
            fold_data = model_data.loc[
                model_data["Fold"].astype(int)
                == fold
            ].copy()

            fold_rows.append(
                metric_row(
                    subset=fold_data,
                    season_code=season_code,
                    response=response,
                    model_structure=model_structure,
                    fold=fold,
                )
            )

    return (
        pd.DataFrame(pooled_rows),
        pd.DataFrame(fold_rows),
    )


# =============================================================================
# Source reconciliation
# =============================================================================

RECONCILIATION_METRICS = [
    "Mean_Negative_LogLikelihood",
    "Zero_Probability_Brier",
    "Rate_RMSE",
    "Predicted_to_Observed_Total_Count_Ratio",
]


def reconcile_source_metrics(
    recalculated_pooled: pd.DataFrame,
    recalculated_fold: pd.DataFrame,
    source_pooled: pd.DataFrame,
    source_fold: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    datasets = [
        (
            "Pooled",
            recalculated_pooled,
            source_pooled,
            [
                "Season_Code",
                "Count_Response",
                "Model_Structure",
            ],
        ),
        (
            "Fold",
            recalculated_fold,
            source_fold,
            [
                "Season_Code",
                "Count_Response",
                "Model_Structure",
                "Fold",
            ],
        ),
    ]

    for level, recalculated, source, keys in datasets:
        recalculated_subset = (
            recalculated[
                [
                    *keys,
                    *RECONCILIATION_METRICS,
                ]
            ]
            .copy()
        )
        source_subset = (
            source[
                [
                    *keys,
                    *RECONCILIATION_METRICS,
                ]
            ]
            .copy()
        )

        merged = recalculated_subset.merge(
            source_subset,
            on=keys,
            how="outer",
            suffixes=(
                "_Recalculated",
                "_Source",
            ),
            indicator=True,
        )

        for _, row in merged.iterrows():
            base = {
                key: row[key]
                for key in keys
            }

            for metric in RECONCILIATION_METRICS:
                recalculated_value = finite_or_nan(
                    row[
                        f"{metric}_Recalculated"
                    ]
                )
                source_value = finite_or_nan(
                    row[
                        f"{metric}_Source"
                    ]
                )
                difference = (
                    abs(
                        recalculated_value
                        - source_value
                    )
                    if (
                        np.isfinite(
                            recalculated_value
                        )
                        and np.isfinite(
                            source_value
                        )
                    )
                    else np.nan
                )

                comparison_scale = (
                    max(
                        abs(recalculated_value),
                        abs(source_value),
                        np.finfo(float).tiny,
                    )
                    if (
                        np.isfinite(
                            recalculated_value
                        )
                        and np.isfinite(
                            source_value
                        )
                    )
                    else np.nan
                )

                relative_difference = (
                    difference / comparison_scale
                    if (
                        np.isfinite(difference)
                        and np.isfinite(
                            comparison_scale
                        )
                        and comparison_scale > 0
                    )
                    else np.nan
                )

                within_tolerance = bool(
                    row["_merge"] == "both"
                    and np.isfinite(
                        recalculated_value
                    )
                    and np.isfinite(
                        source_value
                    )
                    and np.isclose(
                        recalculated_value,
                        source_value,
                        rtol=(
                            SOURCE_RECONCILIATION_RELATIVE_TOLERANCE
                        ),
                        atol=(
                            SOURCE_RECONCILIATION_ABSOLUTE_TOLERANCE
                        ),
                        equal_nan=False,
                    )
                )

                rows.append(
                    {
                        "Level": level,
                        **base,
                        "Metric": metric,
                        "Recalculated_Value": (
                            recalculated_value
                        ),
                        "Source_Value": (
                            source_value
                        ),
                        "Absolute_Difference": (
                            difference
                        ),
                        "Relative_Difference": (
                            relative_difference
                        ),
                        "Absolute_Tolerance": (
                            SOURCE_RECONCILIATION_ABSOLUTE_TOLERANCE
                        ),
                        "Relative_Tolerance": (
                            SOURCE_RECONCILIATION_RELATIVE_TOLERANCE
                        ),
                        "Within_Tolerance": (
                            within_tolerance
                        ),
                        "Merge_Status": row[
                            "_merge"
                        ],
                    }
                )

    result = pd.DataFrame(rows)

    if not result["Within_Tolerance"].all():
        failed = result.loc[
            ~result["Within_Tolerance"]
        ]

        failed.to_csv(
            OUTPUT_ROOT
            / "ERROR_Source_Metric_Reconciliation.csv",
            index=False,
            encoding="utf-8-sig",
        )

        raise RuntimeError(
            "Recalculated OOF metrics do not reconcile with "
            "the Step-04 source summaries. See "
            "ERROR_Source_Metric_Reconciliation.csv."
        )

    return result


# =============================================================================
# Fit-status audit
# =============================================================================

def fit_status_summary(
    fit_status: pd.DataFrame,
) -> pd.DataFrame:
    status = fit_status.copy()

    status["Converged_Boolean"] = status[
        "Converged"
    ].map(robust_bool)
    status["Valid_Fit_Boolean"] = status[
        "Valid_Fit"
    ].map(robust_bool)

    rows: List[Dict[str, object]] = []

    for (
        season_code,
        response,
        model_structure,
    ), group in status.groupby(
        [
            "Season_Code",
            "Count_Response",
            "Model_Structure",
        ],
        sort=True,
    ):
        folds = set(
            group["Fold"].astype(int)
        )

        rows.append(
            {
                "Season_Code": int(
                    season_code
                ),
                "Season_Label": SEASONS[
                    int(season_code)
                ],
                "Count_Response": response,
                "Rate_Scale_Name": (
                    COUNT_RESPONSES[
                        response
                    ]
                ),
                "Model_Structure": (
                    model_structure
                ),
                "Expected_Fold_Count": len(
                    EXPECTED_FOLDS
                ),
                "Observed_Fold_Count": int(
                    group["Fold"].nunique()
                ),
                "Expected_Folds_Present": (
                    folds == EXPECTED_FOLDS
                ),
                "All_Folds_Converged": bool(
                    group[
                        "Converged_Boolean"
                    ].all()
                ),
                "All_Folds_Valid": bool(
                    group[
                        "Valid_Fit_Boolean"
                    ].all()
                ),
                "Warning_Row_Count": int(
                    group[
                        "Warning_Text"
                    ]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .ne("")
                    .sum()
                ),
                "Hessian_Warning_Row_Count": int(
                    group[
                        "Warning_Text"
                    ]
                    .fillna("")
                    .astype(str)
                    .str.contains(
                        "Hessian",
                        case=False,
                        regex=False,
                    )
                    .sum()
                ),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# Stability decisions
# =============================================================================

def add_baseline_relative_metrics(
    pooled: pd.DataFrame,
    fold: pd.DataFrame,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    pooled = pooled.copy()
    fold = fold.copy()

    pooled_baseline = (
        pooled.loc[
            pooled["Model_Structure"]
            == "Single_Stage_NB1",
            [
                "Season_Code",
                "Count_Response",
                "Rate_RMSE",
            ],
        ]
        .rename(
            columns={
                "Rate_RMSE":
                    "Baseline_NB1_Rate_RMSE",
            }
        )
    )

    fold_baseline = (
        fold.loc[
            fold["Model_Structure"]
            == "Single_Stage_NB1",
            [
                "Season_Code",
                "Count_Response",
                "Fold",
                "Rate_RMSE",
            ],
        ]
        .rename(
            columns={
                "Rate_RMSE":
                    "Baseline_NB1_Rate_RMSE",
            }
        )
    )

    pooled = pooled.merge(
        pooled_baseline,
        on=[
            "Season_Code",
            "Count_Response",
        ],
        how="left",
        validate="many_to_one",
    )

    fold = fold.merge(
        fold_baseline,
        on=[
            "Season_Code",
            "Count_Response",
            "Fold",
        ],
        how="left",
        validate="many_to_one",
    )

    pooled[
        "Rate_RMSE_Multiple_vs_Single_Stage_NB1"
    ] = (
        pooled["Rate_RMSE"]
        / pooled["Baseline_NB1_Rate_RMSE"]
    )

    fold[
        "Rate_RMSE_Multiple_vs_Single_Stage_NB1"
    ] = (
        fold["Rate_RMSE"]
        / fold["Baseline_NB1_Rate_RMSE"]
    )

    return pooled, fold


def stability_decisions(
    pooled: pd.DataFrame,
    fold: pd.DataFrame,
    status_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for pooled_row in pooled.itertuples(
        index=False
    ):
        season_code = int(
            pooled_row.Season_Code
        )
        response = str(
            pooled_row.Count_Response
        )
        model_structure = str(
            pooled_row.Model_Structure
        )

        fold_subset = fold.loc[
            (
                fold["Season_Code"]
                == season_code
            )
            & (
                fold["Count_Response"]
                == response
            )
            & (
                fold["Model_Structure"]
                == model_structure
            )
        ].copy()

        status_row = status_summary.loc[
            (
                status_summary[
                    "Season_Code"
                ]
                == season_code
            )
            & (
                status_summary[
                    "Count_Response"
                ]
                == response
            )
            & (
                status_summary[
                    "Model_Structure"
                ]
                == model_structure
            )
        ]

        if len(status_row) != 1:
            raise RuntimeError(
                "Could not identify one fit-status summary for "
                f"{season_code}, {response}, {model_structure}."
            )

        status_record = status_row.iloc[0]

        numerical_valid = bool(
            int(
                pooled_row.Nonfinite_Value_Count
            ) == 0
            and int(
                pooled_row.Negative_Predicted_Count_Count
            ) == 0
            and int(
                pooled_row.Negative_Predicted_Rate_Count
            ) == 0
            and int(
                pooled_row.Zero_Probability_Outside_0_1_Count
            ) == 0
            and int(
                pooled_row.Positive_LogProbability_Count
            ) == 0
            and (
                fold_subset[
                    [
                        "Nonfinite_Value_Count",
                        "Negative_Predicted_Count_Count",
                        "Negative_Predicted_Rate_Count",
                        "Zero_Probability_Outside_0_1_Count",
                        "Positive_LogProbability_Count",
                    ]
                ]
                .to_numpy(dtype=float)
                == 0
            ).all()
        )

        pooled_ratio = float(
            pooled_row.Predicted_to_Observed_Total_Count_Ratio
        )

        pooled_total_ratio_valid = bool(
            np.isfinite(pooled_ratio)
            and TOTAL_COUNT_RATIO_LOWER
            <= pooled_ratio
            <= TOTAL_COUNT_RATIO_UPPER
        )

        fold_total_ratio_valid = bool(
            fold_subset[
                "Predicted_to_Observed_Total_Count_Ratio"
            ]
            .between(
                TOTAL_COUNT_RATIO_LOWER,
                TOTAL_COUNT_RATIO_UPPER,
                inclusive="both",
            )
            .all()
        )

        pooled_rmse_multiple = float(
            pooled_row.Rate_RMSE_Multiple_vs_Single_Stage_NB1
        )

        pooled_rmse_valid = bool(
            np.isfinite(
                pooled_rmse_multiple
            )
            and pooled_rmse_multiple
            <= MAX_RMSE_MULTIPLE_VS_BASELINE
        )

        fold_rmse_valid = bool(
            fold_subset[
                "Rate_RMSE_Multiple_vs_Single_Stage_NB1"
            ]
            .le(
                MAX_RMSE_MULTIPLE_VS_BASELINE
            )
            .all()
        )

        pooled_maximum_rate_multiple = float(
            pooled_row.Predicted_Rate_Maximum_to_Observed_Maximum_Ratio
        )

        pooled_maximum_rate_valid = bool(
            np.isfinite(
                pooled_maximum_rate_multiple
            )
            and pooled_maximum_rate_multiple
            <= MAX_PREDICTED_RATE_MULTIPLE_OF_OBSERVED_MAX
        )

        fold_maximum_rate_valid = bool(
            fold_subset[
                "Predicted_Rate_Maximum_to_Observed_Maximum_Ratio"
            ]
            .le(
                MAX_PREDICTED_RATE_MULTIPLE_OF_OBSERVED_MAX
            )
            .all()
        )

        expected_folds_present = bool(
            status_record[
                "Expected_Folds_Present"
            ]
        )
        all_folds_converged = bool(
            status_record[
                "All_Folds_Converged"
            ]
        )
        all_folds_valid = bool(
            status_record[
                "All_Folds_Valid"
            ]
        )

        checks = {
            "Expected_Folds_Present":
                expected_folds_present,
            "All_Folds_Converged":
                all_folds_converged,
            "All_Folds_Valid":
                all_folds_valid,
            "Numerical_Predictions_Valid":
                numerical_valid,
            "Pooled_Total_Count_Ratio_Valid":
                pooled_total_ratio_valid,
            "All_Fold_Total_Count_Ratios_Valid":
                fold_total_ratio_valid,
            "Pooled_RMSE_Multiple_Valid":
                pooled_rmse_valid,
            "All_Fold_RMSE_Multiples_Valid":
                fold_rmse_valid,
            "Pooled_Maximum_Rate_Valid":
                pooled_maximum_rate_valid,
            "All_Fold_Maximum_Rates_Valid":
                fold_maximum_rate_valid,
        }

        failed_checks = [
            name
            for name, passed in checks.items()
            if not passed
        ]

        stable = len(failed_checks) == 0

        rows.append(
            {
                "Season_Code": season_code,
                "Season_Label": SEASONS[
                    season_code
                ],
                "Count_Response": response,
                "Rate_Scale_Name": (
                    COUNT_RESPONSES[
                        response
                    ]
                ),
                "Model_Structure": model_structure,
                **checks,
                "Pooled_Total_Count_Ratio": (
                    pooled_ratio
                ),
                "Minimum_Fold_Total_Count_Ratio": float(
                    fold_subset[
                        "Predicted_to_Observed_Total_Count_Ratio"
                    ].min()
                ),
                "Maximum_Fold_Total_Count_Ratio": float(
                    fold_subset[
                        "Predicted_to_Observed_Total_Count_Ratio"
                    ].max()
                ),
                "Pooled_Rate_RMSE_Multiple_vs_Baseline": (
                    pooled_rmse_multiple
                ),
                "Maximum_Fold_Rate_RMSE_Multiple_vs_Baseline": float(
                    fold_subset[
                        "Rate_RMSE_Multiple_vs_Single_Stage_NB1"
                    ].max()
                ),
                "Pooled_Predicted_Rate_Maximum_to_Observed_Maximum_Ratio": (
                    pooled_maximum_rate_multiple
                ),
                "Maximum_Fold_Predicted_Rate_Maximum_to_Observed_Maximum_Ratio": float(
                    fold_subset[
                        "Predicted_Rate_Maximum_to_Observed_Maximum_Ratio"
                    ].max()
                ),
                "Hessian_Warning_Row_Count": int(
                    status_record[
                        "Hessian_Warning_Row_Count"
                    ]
                ),
                "Stable_for_Formal_Selection": (
                    stable
                ),
                "Failed_Stability_Checks": (
                    ";".join(failed_checks)
                ),
                "Decision_Note": (
                    "Hessian warnings do not by themselves exclude "
                    "a model from predictive-structure comparison; "
                    "they prohibit coefficient-inference use until "
                    "a stable final fit is obtained."
                ),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# Formal selection
# =============================================================================

def formal_selection(
    pooled: pd.DataFrame,
    stability: pd.DataFrame,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    merged = pooled.merge(
        stability,
        on=[
            "Season_Code",
            "Season_Label",
            "Count_Response",
            "Rate_Scale_Name",
            "Model_Structure",
        ],
        how="left",
        validate="one_to_one",
    )

    tie_break_tables: List[pd.DataFrame] = []
    selected_rows: List[pd.Series] = []
    decision_rows: List[Dict[str, object]] = []

    for (
        season_code,
        response,
    ), subset in merged.groupby(
        [
            "Season_Code",
            "Count_Response",
        ],
        sort=True,
    ):
        subset = subset.copy()

        stable = subset.loc[
            subset[
                "Stable_for_Formal_Selection"
            ]
        ].copy()

        if stable.empty:
            raise RuntimeError(
                f"No stable model remains for season "
                f"{season_code}, response {response}."
            )

        minimum_nll = float(
            stable[
                "Mean_Negative_LogLikelihood"
            ].min()
        )

        stable[
            "Delta_NLL_from_Stable_Minimum"
        ] = (
            stable[
                "Mean_Negative_LogLikelihood"
            ]
            - minimum_nll
        )

        stable[
            "Relative_Delta_NLL_from_Stable_Minimum_pct"
        ] = (
            stable[
                "Delta_NLL_from_Stable_Minimum"
            ]
            / minimum_nll
            * 100.0
        )

        stable[
            "NLL_Equivalent_to_Stable_Minimum"
        ] = (
            stable[
                "Delta_NLL_from_Stable_Minimum"
            ]
            .le(
                NLL_EQUIVALENCE_ABSOLUTE
            )
            & stable[
                "Relative_Delta_NLL_from_Stable_Minimum_pct"
            ]
            .le(
                NLL_EQUIVALENCE_RELATIVE_PERCENT
            )
        )

        stable[
            "Absolute_Total_Count_Calibration_Error"
        ] = np.abs(
            stable[
                "Predicted_to_Observed_Total_Count_Ratio"
            ]
            - 1.0
        )

        stable[
            "Model_Complexity_Tier"
        ] = (
            stable["Model_Structure"]
            .map(MODEL_COMPLEXITY_TIER)
        )

        equivalent = stable.loc[
            stable[
                "NLL_Equivalent_to_Stable_Minimum"
            ]
        ].copy()

        equivalent = equivalent.sort_values(
            [
                "Zero_Probability_Brier",
                "Rate_RMSE",
                "Absolute_Total_Count_Calibration_Error",
                "Model_Complexity_Tier",
                "Model_Structure",
            ]
        )

        equivalent[
            "Equivalent_Set_TieBreak_Rank"
        ] = np.arange(
            1,
            len(equivalent) + 1,
        )

        stable = stable.merge(
            equivalent[
                [
                    "Model_Structure",
                    "Equivalent_Set_TieBreak_Rank",
                ]
            ],
            on="Model_Structure",
            how="left",
            validate="one_to_one",
        )

        stable = stable.sort_values(
            [
                "NLL_Equivalent_to_Stable_Minimum",
                "Equivalent_Set_TieBreak_Rank",
                "Mean_Negative_LogLikelihood",
                "Model_Structure",
            ],
            ascending=[
                False,
                True,
                True,
                True,
            ],
            na_position="last",
        )

        stable[
            "Formal_Selection_Rank"
        ] = np.arange(
            1,
            len(stable) + 1,
        )

        selected = stable.loc[
            stable[
                "Formal_Selection_Rank"
            ] == 1
        ].iloc[0]

        selected_rows.append(selected)
        tie_break_tables.append(stable)

        equivalent_models = (
            equivalent[
                "Model_Structure"
            ]
            .astype(str)
            .tolist()
        )

        if len(equivalent_models) == 1:
            selection_basis = (
                "Unique lowest stable pooled temporal "
                "OOF negative log-likelihood."
            )
        else:
            selection_basis = (
                "NLL-equivalent stable models were compared "
                "using zero-probability Brier score, rate RMSE, "
                "absolute total-count calibration error, and "
                "model-complexity tier."
            )

        decision_rows.append(
            {
                "Season_Code": int(
                    season_code
                ),
                "Season_Label": SEASONS[
                    int(season_code)
                ],
                "Count_Response": response,
                "Rate_Scale_Name": (
                    COUNT_RESPONSES[
                        response
                    ]
                ),
                "Stable_Model_Count": int(
                    len(stable)
                ),
                "Unstable_Model_Count": int(
                    len(subset) - len(stable)
                ),
                "Stable_Minimum_NLL": minimum_nll,
                "NLL_Equivalent_Model_Count": int(
                    len(equivalent)
                ),
                "NLL_Equivalent_Models": (
                    ";".join(
                        equivalent_models
                    )
                ),
                "Selected_Model_Structure": (
                    selected[
                        "Model_Structure"
                    ]
                ),
                "Selection_Basis": (
                    selection_basis
                ),
            }
        )

    tie_break = pd.concat(
        tie_break_tables,
        ignore_index=True,
    )

    selected = pd.DataFrame(
        selected_rows
    ).reset_index(drop=True)

    decision_log = pd.DataFrame(
        decision_rows
    )

    selected[
        "Selection_Status"
    ] = (
        "Formal fixed-effect count-structure selection "
        "after OOF prediction-stability audit"
    )

    selected[
        "Next_Step"
    ] = (
        "Refine component-specific predictors, then audit panel dependence "
        "and finalize GRID_UID-Year two-way clustered inference."
    )

    return (
        tie_break,
        selected,
        decision_log,
    )


# =============================================================================
# Selected prediction export
# =============================================================================

def export_selected_predictions(
    prediction_paths: Dict[
        Tuple[int, str],
        Path,
    ],
    selected: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for selection in selected.itertuples(
        index=False
    ):
        season_code = int(
            selection.Season_Code
        )
        response = str(
            selection.Count_Response
        )
        model_structure = str(
            selection.Model_Structure
        )

        source_path = prediction_paths[
            (
                season_code,
                response,
            )
        ]

        source = pd.read_csv(
            source_path,
            compression="gzip",
        )

        output = source.loc[
            source["Model_Structure"]
            == model_structure
        ].copy()

        expected_rows = int(
            source["GRID_UID"].nunique()
            * source["Year"].nunique()
        )

        # The panel can be unbalanced, so validate against one model's row count
        # rather than requiring the grid-year product.
        source_model_counts = (
            source.groupby(
                "Model_Structure"
            ).size()
        )
        expected_model_rows = int(
            source_model_counts.iloc[0]
        )

        if not (
            source_model_counts
            == expected_model_rows
        ).all():
            raise RuntimeError(
                "Model-specific prediction row counts differ "
                f"for S{season_code} {response}."
            )

        if len(output) != expected_model_rows:
            raise RuntimeError(
                "Selected prediction row count is inconsistent "
                f"for S{season_code} {response}."
            )

        output[
            "Formal_Selected_Model"
        ] = model_structure

        destination = (
            OUTPUT_ROOT
            / (
                "Selected_OOF_Predictions_"
                f"S{season_code}_{response}.csv.gz"
            )
        )

        output.to_csv(
            destination,
            index=False,
            compression="gzip",
        )

        rows.append(
            {
                "Season_Code": season_code,
                "Season_Label": SEASONS[
                    season_code
                ],
                "Count_Response": response,
                "Rate_Scale_Name": (
                    COUNT_RESPONSES[
                        response
                    ]
                ),
                "Selected_Model_Structure": (
                    model_structure
                ),
                "Prediction_File": (
                    destination.name
                ),
                "Rows": int(
                    len(output)
                ),
                "Source_Model_Row_Count": (
                    expected_model_rows
                ),
                "Panel_Grid_Year_Product_for_Reference": (
                    expected_rows
                ),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    if OUTPUT_ROOT.exists():
        shutil.rmtree(
            OUTPUT_ROOT
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=False,
    )

    log(
        "Starting formal count-structure stability audit."
    )
    log(f"Source: {SOURCE_ROOT}")
    log(f"Output: {OUTPUT_ROOT}")

    (
        source_fold,
        source_pooled,
        source_status,
    ) = read_source_tables()

    prediction_paths = prediction_file_mapping()

    pooled_tables: List[pd.DataFrame] = []
    fold_tables: List[pd.DataFrame] = []

    source_manifest_rows: List[
        Dict[str, object]
    ] = []

    for (
        season_code,
        response,
    ), path in sorted(
        prediction_paths.items()
    ):
        log(
            f"Reading full OOF predictions: "
            f"{path.name}."
        )

        data = pd.read_csv(
            path,
            compression="gzip",
        )

        validate_prediction_file(
            data=data,
            season_code=season_code,
            response=response,
            source_path=path,
        )

        pooled_audit, fold_audit = (
            calculate_prediction_audits(
                data=data,
                season_code=season_code,
                response=response,
            )
        )

        pooled_tables.append(
            pooled_audit
        )
        fold_tables.append(
            fold_audit
        )
        source_manifest_rows.append(
            {
                "Season_Code": (
                    season_code
                ),
                "Season_Label": SEASONS[
                    season_code
                ],
                "Count_Response": (
                    response
                ),
                "Rate_Scale_Name": (
                    COUNT_RESPONSES[
                        response
                    ]
                ),
                "Source_Prediction_File": (
                    path.name
                ),
                "Rows": int(
                    len(data)
                ),
                "Model_Structures": (
                    ";".join(
                        sorted(
                            data[
                                "Model_Structure"
                            ]
                            .astype(str)
                            .unique()
                        )
                    )
                ),
                "Temporal_Folds": (
                    ";".join(
                        map(
                            str,
                            sorted(
                                data[
                                    "Fold"
                                ]
                                .astype(int)
                                .unique()
                            ),
                        )
                    )
                ),
            }
        )

        log(
            f"Validated and audited: "
            f"{SEASONS[season_code]} {response}; "
            f"{len(data):,} prediction rows."
        )

    pooled_audit = pd.concat(
        pooled_tables,
        ignore_index=True,
    )
    fold_audit = pd.concat(
        fold_tables,
        ignore_index=True,
    )

    (
        pooled_audit,
        fold_audit,
    ) = add_baseline_relative_metrics(
        pooled=pooled_audit,
        fold=fold_audit,
    )

    reconciliation = reconcile_source_metrics(
        recalculated_pooled=pooled_audit,
        recalculated_fold=fold_audit,
        source_pooled=source_pooled,
        source_fold=source_fold,
    )

    log(
        "Recalculated OOF metrics reconciled with "
        "the Step-04 source summaries."
    )

    status_summary = fit_status_summary(
        source_status
    )

    stability = stability_decisions(
        pooled=pooled_audit,
        fold=fold_audit,
        status_summary=status_summary,
    )

    (
        tie_break,
        selected,
        decision_log,
    ) = formal_selection(
        pooled=pooled_audit,
        stability=stability,
    )

    selected_prediction_manifest = (
        export_selected_predictions(
            prediction_paths=prediction_paths,
            selected=selected,
        )
    )

    unstable = stability.loc[
        ~stability[
            "Stable_for_Formal_Selection"
        ]
    ].copy()

    stability_rule_table = pd.DataFrame(
        [
            {
                "Rule_ID": "S01",
                "Rule": (
                    "All expected temporal folds are present."
                ),
                "Threshold_or_Requirement": (
                    "Folds 1-5"
                ),
                "Purpose": (
                    "Ensure complete temporal validation."
                ),
            },
            {
                "Rule_ID": "S02",
                "Rule": (
                    "All source fits converged and were "
                    "marked valid."
                ),
                "Threshold_or_Requirement": (
                    "All Converged=True and Valid_Fit=True"
                ),
                "Purpose": (
                    "Exclude failed optimization."
                ),
            },
            {
                "Rule_ID": "S03",
                "Rule": (
                    "Predicted counts, rates, probabilities, "
                    "and log probabilities are numerically valid."
                ),
                "Threshold_or_Requirement": (
                    "Finite; nonnegative counts/rates; "
                    "zero probabilities in [0,1]; "
                    "log probabilities <= 0"
                ),
                "Purpose": (
                    "Exclude invalid predictions."
                ),
            },
            {
                "Rule_ID": "S04",
                "Rule": (
                    "Pooled and fold-specific total-count "
                    "calibration ratios are bounded."
                ),
                "Threshold_or_Requirement": (
                    f"{TOTAL_COUNT_RATIO_LOWER} <= "
                    "predicted/observed total <= "
                    f"{TOTAL_COUNT_RATIO_UPPER}"
                ),
                "Purpose": (
                    "Identify catastrophic aggregate "
                    "over- or underprediction."
                ),
            },
            {
                "Rule_ID": "S05",
                "Rule": (
                    "Pooled and fold-specific rate RMSE "
                    "remain within a generous multiple of "
                    "the matched single-stage NB1 baseline."
                ),
                "Threshold_or_Requirement": (
                    "<= "
                    f"{MAX_RMSE_MULTIPLE_VS_BASELINE}"
                    " times baseline NB1 RMSE"
                ),
                "Purpose": (
                    "Identify catastrophic predictive "
                    "extrapolation hidden by many zero rows."
                ),
            },
            {
                "Rule_ID": "S06",
                "Rule": (
                    "Maximum predicted rate remains within "
                    "a generous multiple of the maximum "
                    "observed rate."
                ),
                "Threshold_or_Requirement": (
                    "<= "
                    f"{MAX_PREDICTED_RATE_MULTIPLE_OF_OBSERVED_MAX}"
                    " times observed maximum"
                ),
                "Purpose": (
                    "Identify isolated extreme predictions."
                ),
            },
            {
                "Rule_ID": "E01",
                "Rule": (
                    "NLL equivalence requires both absolute "
                    "and relative closeness to the lowest "
                    "stable pooled temporal OOF NLL."
                ),
                "Threshold_or_Requirement": (
                    "Absolute delta <= "
                    f"{NLL_EQUIVALENCE_ABSOLUTE}; "
                    "relative delta <= "
                    f"{NLL_EQUIVALENCE_RELATIVE_PERCENT}%"
                ),
                "Purpose": (
                    "Avoid selecting on negligible NLL "
                    "differences."
                ),
            },
            {
                "Rule_ID": "T01",
                "Rule": (
                    "Tie-break NLL-equivalent stable models."
                ),
                "Threshold_or_Requirement": (
                    "Lower zero Brier; lower rate RMSE; "
                    "smaller total-count calibration error; "
                    "lower complexity tier"
                ),
                "Purpose": (
                    "Prefer stable and better-calibrated "
                    "prediction when NLL is practically equal."
                ),
            },
        ]
    )

    qa = pd.DataFrame(
        [
            (
                "Source_prediction_file_count",
                len(prediction_paths),
            ),
            (
                "Expected_season_response_count",
                6,
            ),
            (
                "Compared_model_structure_count",
                len(MODEL_STRUCTURES),
            ),
            (
                "Recalculated_pooled_model_count",
                len(pooled_audit),
            ),
            (
                "Recalculated_fold_model_count",
                len(fold_audit),
            ),
            (
                "Source_reconciliation_row_count",
                len(reconciliation),
            ),
            (
                "Source_reconciliation_all_within_tolerance",
                reconciliation[
                    "Within_Tolerance"
                ].all(),
            ),
            (
                "Stable_model_count",
                int(
                    stability[
                        "Stable_for_Formal_Selection"
                    ].sum()
                ),
            ),
            (
                "Unstable_model_count",
                int(
                    (
                        ~stability[
                            "Stable_for_Formal_Selection"
                        ]
                    ).sum()
                ),
            ),
            (
                "Formal_selected_model_count",
                len(selected),
            ),
            (
                "Selected_prediction_file_count",
                len(
                    selected_prediction_manifest
                ),
            ),
            (
                "Model_refitting_performed",
                False,
            ),
            (
                "Random_forest_selection_used",
                False,
            ),
        ],
        columns=[
            "Metric",
            "Value",
        ],
    )

    method = {
        "workflow_stage":
            "Formal post-hoc prediction-stability audit and "
            "count-structure selection",
        "model_refitting_performed":
            False,
        "source_directory":
            str(SOURCE_ROOT),
        "compared_models":
            MODEL_STRUCTURES,
        "stability_thresholds": {
            "total_count_ratio_lower":
                TOTAL_COUNT_RATIO_LOWER,
            "total_count_ratio_upper":
                TOTAL_COUNT_RATIO_UPPER,
            "maximum_rmse_multiple_vs_single_stage_nb1":
                MAX_RMSE_MULTIPLE_VS_BASELINE,
            "maximum_predicted_rate_multiple_of_observed_maximum":
                MAX_PREDICTED_RATE_MULTIPLE_OF_OBSERVED_MAX,
        },
        "nll_equivalence": {
            "absolute_delta_maximum":
                NLL_EQUIVALENCE_ABSOLUTE,
            "relative_delta_percent_maximum":
                NLL_EQUIVALENCE_RELATIVE_PERCENT,
            "both_criteria_required":
                True,
        },
        "tie_break_order": [
            "Lower zero-probability Brier score",
            "Lower rate-scale RMSE",
            "Smaller absolute total-count calibration error",
            "Lower model-complexity tier",
            "Model name for deterministic ordering",
        ],
        "source_metric_reconciliation": {
            "absolute_tolerance":
                SOURCE_RECONCILIATION_ABSOLUTE_TOLERANCE,
            "relative_tolerance":
                SOURCE_RECONCILIATION_RELATIVE_TOLERANCE,
            "rule":
                "Recalculated and source metrics must satisfy "
                "numpy.isclose using both tolerances.",
        },
        "hessian_warning_policy":
            "Hessian warnings do not alone invalidate OOF "
            "prediction comparison, but coefficient inference "
            "requires a stable final model fit.",
        "selection_scope":
            "Count-structure selection only. Component-specific predictor "
            "refinement and clustered inference remain for subsequent steps.",
    }

    environment = {
        "python":
            sys.version,
        "platform":
            platform.platform(),
        "numpy":
            np.__version__,
        "pandas":
            pd.__version__,
        "run_timestamp":
            datetime.now().isoformat(
                timespec="seconds"
            ),
    }

    with (
        OUTPUT_ROOT
        / "00_Method_Definition.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            method,
            handle,
            indent=2,
        )

    qa.to_csv(
        OUTPUT_ROOT
        / "01_Data_QA_Summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    stability_rule_table.to_csv(
        OUTPUT_ROOT
        / "02_Formal_Stability_and_Selection_Rules.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        source_manifest_rows
    ).to_csv(
        OUTPUT_ROOT
        / "03_Source_Prediction_File_Manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pooled_audit.to_csv(
        OUTPUT_ROOT
        / "04_Recalculated_Pooled_Prediction_Audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    fold_audit.to_csv(
        OUTPUT_ROOT
        / "05_Recalculated_Fold_Prediction_Audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    reconciliation.to_csv(
        OUTPUT_ROOT
        / "06_Source_Metric_Reconciliation.csv",
        index=False,
        encoding="utf-8-sig",
    )
    status_summary.to_csv(
        OUTPUT_ROOT
        / "07_Fit_Status_Summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    stability.to_csv(
        OUTPUT_ROOT
        / "08_Model_Stability_Decisions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    tie_break.to_csv(
        OUTPUT_ROOT
        / "09_NLL_Equivalence_and_TieBreak.csv",
        index=False,
        encoding="utf-8-sig",
    )
    decision_log.to_csv(
        OUTPUT_ROOT
        / "10_Selection_Decision_Log.csv",
        index=False,
        encoding="utf-8-sig",
    )
    selected.to_csv(
        OUTPUT_ROOT
        / "11_Final_Selected_Count_Structure.csv",
        index=False,
        encoding="utf-8-sig",
    )
    unstable.to_csv(
        OUTPUT_ROOT
        / "12_Excluded_Unstable_Models.csv",
        index=False,
        encoding="utf-8-sig",
    )
    selected_prediction_manifest.to_csv(
        OUTPUT_ROOT
        / "13_Selected_OOF_Prediction_Manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with (
        OUTPUT_ROOT
        / "Software_Environment.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            environment,
            handle,
            indent=2,
        )

    log(
        "Formal count-structure selection completed successfully."
    )

    for row in selected.itertuples(
        index=False
    ):
        log(
            f"Selected: {row.Season_Label} "
            f"{row.Count_Response} -> "
            f"{row.Model_Structure}."
        )

    log(
        "Next step: refine component-specific predictors, then audit panel "
        "dependence and finalize two-way clustered inference."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
