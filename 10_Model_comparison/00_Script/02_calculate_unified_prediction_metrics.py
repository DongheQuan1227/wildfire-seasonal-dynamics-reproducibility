# -*- coding: utf-8 -*-
"""
Step 02: calculate unified predictive-performance metrics for aligned OOF data.

Recommended public location
---------------------------
<CODE_ROOT>/10_Model_comparison/00_Script/02_calculate_unified_prediction_metrics.py

Purpose
-------
Read the two strictly aligned wide-format files produced by model-comparison
Step 01 and calculate directly comparable predictive-performance metrics for:

1. Baseline NB1;
2. Final Regression;
3. Final Hurdle RF.

Temporal and spatial OOF validation are analysed separately. This step does not
fit, tune, select, bootstrap, or statistically rank any model. It calculates:

* unconditional rate and count accuracy;
* occurrence-probability discrimination and calibration;
* conditional-positive severity accuracy on observed-positive records;
* decile calibration tables;
* performance for the upper 10% and upper 5% of observed positive rates;
* the same unified metrics within each held-out fold/block.

Predictive log probability is deliberately excluded from the three-model
comparison because Final Hurdle RF does not define a complete count probability
mass function.

Outputs
-------
<CODE_ROOT>/10_Model_comparison/02_Unified_Prediction_Metrics/

    00_Method_Definition.json
    01_Input_Integrity_Audit.csv
    02_Overall_Unified_Prediction_Metrics.csv
    03_Fold_Level_Unified_Prediction_Metrics.csv
    04_Occurrence_Calibration_Bins.csv
    05_Rate_Calibration_Bins.csv
    06_Extreme_Event_Performance.csv
    07_Metric_Definitions.csv
    08_Computation_QA.csv
    09_Output_Manifest.csv
    Software_Environment.json
    unified_prediction_metrics.log
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


# =============================================================================
# Repository-relative paths
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_COMPARISON_ROOT = SCRIPT_DIR.parent

STEP01_ROOT = MODEL_COMPARISON_ROOT / "01_Aligned_Model_OOF"
STEP01_METHOD_FILE = STEP01_ROOT / "00_Method_Definition.json"
STEP01_MANIFEST_FILE = STEP01_ROOT / "07_Aligned_OOF_Manifest.csv"
TEMPORAL_INPUT_FILE = STEP01_ROOT / "05_Aligned_Temporal_OOF.csv.gz"
SPATIAL_INPUT_FILE = STEP01_ROOT / "06_Aligned_Spatial_OOF.csv.gz"

OUTPUT_ROOT = MODEL_COMPARISON_ROOT / "02_Unified_Prediction_Metrics"
LOG_FILE = OUTPUT_ROOT / "unified_prediction_metrics.log"
METHOD_FILE = OUTPUT_ROOT / "00_Method_Definition.json"
INPUT_AUDIT_FILE = OUTPUT_ROOT / "01_Input_Integrity_Audit.csv"
OVERALL_METRICS_FILE = OUTPUT_ROOT / "02_Overall_Unified_Prediction_Metrics.csv"
FOLD_METRICS_FILE = OUTPUT_ROOT / "03_Fold_Level_Unified_Prediction_Metrics.csv"
OCCURRENCE_CALIBRATION_FILE = OUTPUT_ROOT / "04_Occurrence_Calibration_Bins.csv"
RATE_CALIBRATION_FILE = OUTPUT_ROOT / "05_Rate_Calibration_Bins.csv"
EXTREME_FILE = OUTPUT_ROOT / "06_Extreme_Event_Performance.csv"
METRIC_DEFINITION_FILE = OUTPUT_ROOT / "07_Metric_Definitions.csv"
QA_FILE = OUTPUT_ROOT / "08_Computation_QA.csv"
OUTPUT_MANIFEST_FILE = OUTPUT_ROOT / "09_Output_Manifest.csv"
SOFTWARE_FILE = OUTPUT_ROOT / "Software_Environment.json"


# =============================================================================
# Locked settings
# =============================================================================

STEP02_CODE_VERSION = "2026-06-30_MODEL_COMPARISON_UNIFIED_METRICS_V1A_ZERO_POSITIVE_FOLD_COMPATIBILITY"
EXPECTED_STEP01_CODE_VERSION = "2026-06-30_MODEL_COMPARISON_ALIGNED_OOF_V1"

VALIDATION_SCHEMES = ["Temporal", "Spatial"]
MODELS = ["Baseline_NB1", "Final_Regression", "Final_Hurdle_RF"]
SEASONS = {1: "Spring", 2: "Summer", 3: "Autumn"}
RESPONSES = {
    "Fire_Count": "FCD",
    "Burned_Pixel_Count": "BAD",
}

EXPECTED_ROWS_PER_SCHEME = 742_374
EXPECTED_ROWS_PER_SEASON_RESPONSE = 123_729
EXPECTED_GRID_COUNT = 4_982
EXPECTED_FOLDS = {1, 2, 3, 4, 5}

CALIBRATION_BIN_COUNT = 10
CALIBRATION_PROBABILITY_CLIP = 1e-8
EXTREME_QUANTILES = [0.90, 0.95]
FLOAT_TOLERANCE = 1e-10

SHARED_COLUMNS = [
    "GRID_UID",
    "Country",
    "GRID_ID",
    "Year",
    "Season",
    "Season_Label",
    "Temporal_Fold",
    "Spatial_Block",
    "Validation_Scheme",
    "Validation_Fold",
    "Count_Response",
    "Rate_Scale_Name",
    "ForestPixelCount",
    "Observed_Count",
    "Observed_Rate",
    "Observed_Zero",
]

MODEL_SUFFIXES = [
    "Predicted_Count",
    "Predicted_Rate",
    "Predicted_Positive_Probability",
    "Predicted_Conditional_Positive_Count",
    "Predicted_Conditional_Positive_Rate",
]

INPUT_USECOLS = SHARED_COLUMNS + [
    f"{model}_{suffix}"
    for model in MODELS
    for suffix in MODEL_SUFFIXES
]


# =============================================================================
# Logging and file helpers
# =============================================================================


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as file:
        file.write(line + "\n")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def finite_array(values: pd.Series | np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return np.nan
    if abs(denominator) <= np.finfo(float).tiny:
        return np.nan
    return float(numerator / denominator)


# =============================================================================
# Input provenance and structural checks
# =============================================================================


def verify_step01_inputs() -> pd.DataFrame:
    required = [
        STEP01_METHOD_FILE,
        STEP01_MANIFEST_FILE,
        TEMPORAL_INPUT_FILE,
        SPATIAL_INPUT_FILE,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required Step-01 files are missing:\n" + "\n".join(missing)
        )

    with STEP01_METHOD_FILE.open("r", encoding="utf-8") as file:
        method = json.load(file)

    observed_version = str(method.get("Code_Version", ""))
    if observed_version != EXPECTED_STEP01_CODE_VERSION:
        raise ValueError(
            "Unexpected Step-01 code version.\n"
            f"Expected: {EXPECTED_STEP01_CODE_VERSION}\n"
            f"Observed: {observed_version}"
        )

    manifest = pd.read_csv(STEP01_MANIFEST_FILE)
    required_manifest_columns = {
        "Validation_Scheme",
        "Output_File",
        "Rows",
        "GRID_UID_Count",
        "Season_Response_Combinations",
        "Model_Count",
        "Models",
        "SHA256",
        "Step01_Code_Version",
    }
    missing_manifest = sorted(required_manifest_columns - set(manifest.columns))
    if missing_manifest:
        raise ValueError(
            "Step-01 manifest is missing columns:\n" + "\n".join(missing_manifest)
        )

    rows: List[Dict[str, object]] = []
    file_by_scheme = {
        "Temporal": TEMPORAL_INPUT_FILE,
        "Spatial": SPATIAL_INPUT_FILE,
    }

    for scheme in VALIDATION_SCHEMES:
        subset = manifest.loc[manifest["Validation_Scheme"] == scheme]
        if len(subset) != 1:
            raise ValueError(
                f"Step-01 manifest must contain exactly one {scheme} row."
            )

        record = subset.iloc[0]
        path = file_by_scheme[scheme]
        actual_hash = sha256_file(path)
        expected_hash = str(record["SHA256"])
        actual_size = int(path.stat().st_size)

        checks = {
            "File_Exists": path.exists(),
            "SHA256_Matches_Manifest": actual_hash == expected_hash,
            "Rows_Match_Locked_Value": int(record["Rows"]) == EXPECTED_ROWS_PER_SCHEME,
            "GRID_UID_Count_Matches": int(record["GRID_UID_Count"]) == EXPECTED_GRID_COUNT,
            "Combination_Count_Matches": int(record["Season_Response_Combinations"]) == 6,
            "Model_Count_Matches": int(record["Model_Count"]) == len(MODELS),
            "Model_List_Matches": set(str(record["Models"]).split(";")) == set(MODELS),
            "Step01_Version_Matches": str(record["Step01_Code_Version"]) == EXPECTED_STEP01_CODE_VERSION,
        }

        for check_name, passed in checks.items():
            rows.append(
                {
                    "Validation_Scheme": scheme,
                    "Check": check_name,
                    "Passed": bool(passed),
                    "Observed": (
                        actual_hash
                        if check_name == "SHA256_Matches_Manifest"
                        else ""
                    ),
                    "Expected": (
                        expected_hash
                        if check_name == "SHA256_Matches_Manifest"
                        else ""
                    ),
                    "Input_File": str(path),
                    "Input_File_Size_Bytes": actual_size,
                }
            )

        if not all(checks.values()):
            failed = [name for name, passed in checks.items() if not passed]
            raise ValueError(
                f"Step-01 {scheme} input integrity check failed: {failed}"
            )

    audit = pd.DataFrame(rows)
    audit.to_csv(INPUT_AUDIT_FILE, index=False, encoding="utf-8-sig")
    return audit


def read_aligned_file(path: Path, expected_scheme: str) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    missing = sorted(set(INPUT_USECOLS) - set(header.columns))
    if missing:
        raise ValueError(
            f"Aligned {expected_scheme} file is missing required columns:\n"
            + "\n".join(missing)
        )

    data = pd.read_csv(path, usecols=INPUT_USECOLS)

    if len(data) != EXPECTED_ROWS_PER_SCHEME:
        raise ValueError(
            f"{expected_scheme} aligned OOF has {len(data):,} rows; "
            f"expected {EXPECTED_ROWS_PER_SCHEME:,}."
        )

    if set(data["Validation_Scheme"].astype(str).unique()) != {expected_scheme}:
        raise ValueError(
            f"Validation_Scheme values do not match {expected_scheme}."
        )

    if data["GRID_UID"].nunique() != EXPECTED_GRID_COUNT:
        raise ValueError(
            f"{expected_scheme} GRID_UID count differs from {EXPECTED_GRID_COUNT:,}."
        )

    if set(pd.to_numeric(data["Validation_Fold"], errors="raise").astype(int).unique()) != EXPECTED_FOLDS:
        raise ValueError(f"{expected_scheme} validation folds are not 1-5.")

    duplicate_key = data.duplicated(
        ["GRID_UID", "Year", "Season", "Count_Response"],
        keep=False,
    )
    if duplicate_key.any():
        raise ValueError(f"Duplicate OOF keys found in {expected_scheme} input.")

    group_counts = (
        data.groupby(["Season", "Count_Response"], observed=True)
        .size()
        .reset_index(name="Rows")
    )
    if len(group_counts) != 6 or not (
        group_counts["Rows"].to_numpy(dtype=int) == EXPECTED_ROWS_PER_SEASON_RESPONSE
    ).all():
        raise ValueError(
            f"{expected_scheme} season-response row counts do not match locked values."
        )

    numeric_shared = [
        "Year",
        "Season",
        "Temporal_Fold",
        "Spatial_Block",
        "Validation_Fold",
        "ForestPixelCount",
        "Observed_Count",
        "Observed_Rate",
        "Observed_Zero",
    ]
    numeric_model = [
        f"{model}_{suffix}"
        for model in MODELS
        for suffix in MODEL_SUFFIXES
    ]

    for column in numeric_shared + numeric_model:
        data[column] = pd.to_numeric(data[column], errors="raise")

    if not np.isfinite(data[numeric_shared + numeric_model].to_numpy(dtype=float)).all():
        raise ValueError(f"Non-finite required values found in {expected_scheme} input.")

    if (data["ForestPixelCount"] <= 0).any():
        raise ValueError("ForestPixelCount must be positive.")

    observed_rate_check = data["Observed_Count"] / data["ForestPixelCount"]
    if float(np.max(np.abs(observed_rate_check - data["Observed_Rate"]))) > FLOAT_TOLERANCE:
        raise ValueError("Observed_Rate identity failed.")

    observed_zero_check = (data["Observed_Count"] == 0).astype(int)
    if not np.array_equal(observed_zero_check.to_numpy(), data["Observed_Zero"].astype(int).to_numpy()):
        raise ValueError("Observed_Zero identity failed.")

    for model in MODELS:
        pred_count = data[f"{model}_Predicted_Count"].to_numpy(dtype=float)
        pred_rate = data[f"{model}_Predicted_Rate"].to_numpy(dtype=float)
        pos_prob = data[f"{model}_Predicted_Positive_Probability"].to_numpy(dtype=float)
        cond_count = data[f"{model}_Predicted_Conditional_Positive_Count"].to_numpy(dtype=float)
        cond_rate = data[f"{model}_Predicted_Conditional_Positive_Rate"].to_numpy(dtype=float)

        if (pred_count < -FLOAT_TOLERANCE).any() or (pred_rate < -FLOAT_TOLERANCE).any():
            raise ValueError(f"Negative unconditional predictions found for {model}.")
        if (pos_prob < -FLOAT_TOLERANCE).any() or (pos_prob > 1 + FLOAT_TOLERANCE).any():
            raise ValueError(f"Positive probabilities outside [0, 1] for {model}.")
        if (cond_count < -FLOAT_TOLERANCE).any() or (cond_rate < -FLOAT_TOLERANCE).any():
            raise ValueError(f"Negative conditional-positive predictions found for {model}.")

        rate_identity = pred_count / data["ForestPixelCount"].to_numpy(dtype=float)
        if float(np.max(np.abs(rate_identity - pred_rate))) > 1e-9:
            raise ValueError(f"Predicted rate/count identity failed for {model}.")

        hurdle_count_identity = pos_prob * cond_count
        hurdle_rate_identity = pos_prob * cond_rate
        if float(np.max(np.abs(hurdle_count_identity - pred_count))) > 1e-8:
            raise ValueError(f"Positive-probability count identity failed for {model}.")
        if float(np.max(np.abs(hurdle_rate_identity - pred_rate))) > 1e-10:
            raise ValueError(f"Positive-probability rate identity failed for {model}.")

    return data


# =============================================================================
# Statistical helpers
# =============================================================================


def safe_rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(predicted - observed))))


def safe_mae(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(predicted - observed)))


def safe_median_ae(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.median(np.abs(predicted - observed)))


def safe_r2(observed: np.ndarray, predicted: np.ndarray) -> float:
    centered = observed - np.mean(observed)
    denominator = float(np.sum(np.square(centered)))
    if denominator <= np.finfo(float).tiny:
        return np.nan
    numerator = float(np.sum(np.square(predicted - observed)))
    return float(1.0 - numerator / denominator)


def safe_spearman(observed: np.ndarray, predicted: np.ndarray) -> float:
    if len(observed) < 3:
        return np.nan
    if np.std(observed) <= np.finfo(float).tiny:
        return np.nan
    if np.std(predicted) <= np.finfo(float).tiny:
        return np.nan
    result = spearmanr(observed, predicted, nan_policy="omit")
    return float(result.statistic)


def linear_calibration(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> Tuple[float, float]:
    x_mean = float(np.mean(predicted))
    y_mean = float(np.mean(observed))
    centered_x = predicted - x_mean
    denominator = float(np.sum(np.square(centered_x)))
    if denominator <= np.finfo(float).tiny:
        return np.nan, np.nan
    slope = float(np.sum(centered_x * (observed - y_mean)) / denominator)
    intercept = float(y_mean - slope * x_mean)
    return intercept, slope


def occurrence_calibration(
    observed_binary: np.ndarray,
    predicted_probability: np.ndarray,
) -> Tuple[float, float, int, bool]:
    if len(np.unique(observed_binary)) < 2:
        return np.nan, np.nan, 0, False

    clipped = np.clip(
        predicted_probability,
        CALIBRATION_PROBABILITY_CLIP,
        1.0 - CALIBRATION_PROBABILITY_CLIP,
    )
    clipped_count = int(np.sum(clipped != predicted_probability))
    logit_prediction = np.log(clipped) - np.log1p(-clipped)

    if np.std(logit_prediction) <= np.finfo(float).tiny:
        return np.nan, np.nan, clipped_count, False

    y = observed_binary.astype(np.float64)

    def objective(parameters: np.ndarray) -> Tuple[float, np.ndarray]:
        intercept = float(parameters[0])
        slope = float(parameters[1])
        eta = intercept + slope * logit_prediction
        probability = expit(eta)
        loss = float(np.sum(np.logaddexp(0.0, eta) - y * eta))
        residual = probability - y
        gradient = np.array(
            [
                float(np.sum(residual)),
                float(np.sum(residual * logit_prediction)),
            ],
            dtype=np.float64,
        )
        return loss, gradient

    result = minimize(
        fun=lambda pars: objective(pars)[0],
        x0=np.array([0.0, 1.0], dtype=np.float64),
        jac=lambda pars: objective(pars)[1],
        method="L-BFGS-B",
        options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8},
    )

    if not result.success or not np.isfinite(result.x).all():
        return np.nan, np.nan, clipped_count, False

    return float(result.x[0]), float(result.x[1]), clipped_count, True


def make_quantile_bins(values: np.ndarray, requested_bins: int) -> np.ndarray:
    series = pd.Series(values)
    if series.nunique(dropna=True) <= 1:
        return np.ones(len(series), dtype=int)

    try:
        categories = pd.qcut(
            series,
            q=requested_bins,
            labels=False,
            duplicates="drop",
        )
        result = categories.to_numpy(dtype=float)
        if np.isnan(result).any():
            raise ValueError("NaN quantile bins")
        return result.astype(int) + 1
    except (ValueError, TypeError):
        ranks = series.rank(method="average", pct=True).to_numpy(dtype=float)
        result = np.ceil(ranks * requested_bins).astype(int)
        return np.clip(result, 1, requested_bins)


def calibration_bin_table(
    validation_scheme: str,
    season: int,
    response: str,
    model: str,
    observed: np.ndarray,
    predicted: np.ndarray,
    calibration_type: str,
) -> Tuple[pd.DataFrame, float]:
    bin_index = make_quantile_bins(predicted, CALIBRATION_BIN_COUNT)
    temporary = pd.DataFrame(
        {
            "Observed": observed,
            "Predicted": predicted,
            "Bin_Order": bin_index,
        }
    )

    rows: List[Dict[str, object]] = []
    weighted_absolute_gap = 0.0
    total_n = len(temporary)

    for order, group in temporary.groupby("Bin_Order", sort=True, observed=True):
        mean_predicted = float(group["Predicted"].mean())
        mean_observed = float(group["Observed"].mean())
        gap = mean_predicted - mean_observed
        n = int(len(group))
        weighted_absolute_gap += (n / total_n) * abs(gap)

        rows.append(
            {
                "Validation_Scheme": validation_scheme,
                "Season": int(season),
                "Season_Label": SEASONS[int(season)],
                "Count_Response": response,
                "Rate_Scale_Name": RESPONSES[response],
                "Model": model,
                "Calibration_Type": calibration_type,
                "Requested_Bin_Count": CALIBRATION_BIN_COUNT,
                "Actual_Bin_Count": int(len(np.unique(bin_index))),
                "Bin_Order": int(order),
                "N": n,
                "Predicted_Min": float(group["Predicted"].min()),
                "Predicted_Q25": float(group["Predicted"].quantile(0.25)),
                "Predicted_Median": float(group["Predicted"].median()),
                "Predicted_Mean": mean_predicted,
                "Predicted_Q75": float(group["Predicted"].quantile(0.75)),
                "Predicted_Max": float(group["Predicted"].max()),
                "Observed_Mean": mean_observed,
                "Observed_Median": float(group["Observed"].median()),
                "Prediction_Minus_Observation": gap,
                "Absolute_Calibration_Gap": abs(gap),
            }
        )

    return pd.DataFrame(rows), float(weighted_absolute_gap)


def continuous_metric_block(
    observed: np.ndarray,
    predicted: np.ndarray,
    prefix: str,
) -> Dict[str, float]:
    intercept, slope = linear_calibration(observed, predicted)
    return {
        f"{prefix}_RMSE": safe_rmse(observed, predicted),
        f"{prefix}_MAE": safe_mae(observed, predicted),
        f"{prefix}_Median_Absolute_Error": safe_median_ae(observed, predicted),
        f"{prefix}_R2": safe_r2(observed, predicted),
        f"{prefix}_Spearman": safe_spearman(observed, predicted),
        f"{prefix}_Mean_Bias": float(np.mean(predicted - observed)),
        f"{prefix}_Mean_Observed": float(np.mean(observed)),
        f"{prefix}_Mean_Predicted": float(np.mean(predicted)),
        f"{prefix}_Predicted_to_Observed_Mean_Ratio": safe_ratio(
            float(np.mean(predicted)), float(np.mean(observed))
        ),
        f"{prefix}_Total_Observed": float(np.sum(observed)),
        f"{prefix}_Total_Predicted": float(np.sum(predicted)),
        f"{prefix}_Predicted_to_Observed_Total_Ratio": safe_ratio(
            float(np.sum(predicted)), float(np.sum(observed))
        ),
        f"{prefix}_Calibration_Intercept": intercept,
        f"{prefix}_Calibration_Slope": slope,
    }


def unavailable_continuous_metric_block(prefix: str) -> Dict[str, float]:
    """Return the complete continuous-metric schema as NA when no observations exist.

    This is used only for statistically undefined fold-level conditional-positive
    metrics (for example, a held-out spatial block containing no observed positive
    burned-area records). It is not a computational failure.
    """
    suffixes = [
        "RMSE",
        "MAE",
        "Median_Absolute_Error",
        "R2",
        "Spearman",
        "Mean_Bias",
        "Mean_Observed",
        "Mean_Predicted",
        "Predicted_to_Observed_Mean_Ratio",
        "Total_Observed",
        "Total_Predicted",
        "Predicted_to_Observed_Total_Ratio",
        "Calibration_Intercept",
        "Calibration_Slope",
    ]
    return {f"{prefix}_{suffix}": np.nan for suffix in suffixes}


# =============================================================================
# Core metric calculation
# =============================================================================


def calculate_metric_row(
    group: pd.DataFrame,
    validation_scheme: str,
    season: int,
    response: str,
    model: str,
    fold: Optional[int],
    include_calibration_bins: bool,
) -> Tuple[Dict[str, object], pd.DataFrame, pd.DataFrame]:
    observed_count = group["Observed_Count"].to_numpy(dtype=np.float64)
    observed_rate = group["Observed_Rate"].to_numpy(dtype=np.float64)
    observed_positive = (observed_count > 0).astype(np.int8)

    predicted_count = group[f"{model}_Predicted_Count"].to_numpy(dtype=np.float64)
    predicted_rate = group[f"{model}_Predicted_Rate"].to_numpy(dtype=np.float64)
    predicted_probability = group[
        f"{model}_Predicted_Positive_Probability"
    ].to_numpy(dtype=np.float64)
    conditional_count = group[
        f"{model}_Predicted_Conditional_Positive_Count"
    ].to_numpy(dtype=np.float64)
    conditional_rate = group[
        f"{model}_Predicted_Conditional_Positive_Rate"
    ].to_numpy(dtype=np.float64)

    positive_mask = observed_positive == 1
    positive_n = int(np.sum(positive_mask))
    negative_n = int(len(observed_positive) - positive_n)
    occurrence_has_two_classes = positive_n > 0 and negative_n > 0
    positive_metrics_available = positive_n > 0

    occurrence_intercept, occurrence_slope, clipped_n, calibration_converged = (
        occurrence_calibration(observed_positive, predicted_probability)
    )

    prevalence = float(np.mean(observed_positive))
    brier = float(brier_score_loss(observed_positive, predicted_probability))
    null_brier = float(np.mean(np.square(observed_positive - prevalence)))

    if occurrence_has_two_classes:
        occurrence_pr_auc = float(
            average_precision_score(observed_positive, predicted_probability)
        )
        occurrence_roc_auc = float(
            roc_auc_score(observed_positive, predicted_probability)
        )
    else:
        occurrence_pr_auc = np.nan
        occurrence_roc_auc = np.nan

    unavailable_reasons: List[str] = []
    if not positive_metrics_available:
        unavailable_reasons.append(
            "No observed positive response in this held-out subset; "
            "conditional-positive metrics are undefined"
        )
    if not occurrence_has_two_classes:
        unavailable_reasons.append(
            "Observed occurrence has one class only; PR-AUC, ROC-AUC, and "
            "occurrence calibration intercept/slope are undefined"
        )

    row: Dict[str, object] = {
        "Validation_Scheme": validation_scheme,
        "Validation_Fold": np.nan if fold is None else int(fold),
        "Season": int(season),
        "Season_Label": SEASONS[int(season)],
        "Count_Response": response,
        "Rate_Scale_Name": RESPONSES[response],
        "Model": model,
        "N": int(len(group)),
        "GRID_UID_Count": int(group["GRID_UID"].nunique()),
        "Year_Count": int(group["Year"].nunique()),
        "Observed_Positive_N": positive_n,
        "Observed_Negative_N": negative_n,
        "Observed_Positive_Fraction": prevalence,
        "Positive_Conditional_Metrics_Available": bool(
            positive_metrics_available
        ),
        "Occurrence_Discrimination_Metrics_Available": bool(
            occurrence_has_two_classes
        ),
        "Occurrence_Calibration_Metrics_Available": bool(
            occurrence_has_two_classes and calibration_converged
        ),
        "Unavailable_Metric_Reason": "; ".join(unavailable_reasons),
    }

    row.update(continuous_metric_block(observed_count, predicted_count, "Count"))
    row.update(continuous_metric_block(observed_rate, predicted_rate, "Rate"))

    if positive_metrics_available:
        row.update(
            continuous_metric_block(
                observed_count[positive_mask],
                conditional_count[positive_mask],
                "Positive_Conditional_Count",
            )
        )
        row.update(
            continuous_metric_block(
                observed_rate[positive_mask],
                conditional_rate[positive_mask],
                "Positive_Conditional_Rate",
            )
        )
    else:
        row.update(
            unavailable_continuous_metric_block(
                "Positive_Conditional_Count"
            )
        )
        row.update(
            unavailable_continuous_metric_block(
                "Positive_Conditional_Rate"
            )
        )

    row.update(
        {
            "Occurrence_Brier_Score": brier,
            "Occurrence_Null_Brier_Score": null_brier,
            "Occurrence_Brier_Skill_Score": (
                np.nan if null_brier <= np.finfo(float).tiny else 1.0 - brier / null_brier
            ),
            "Occurrence_PR_AUC": occurrence_pr_auc,
            "Occurrence_ROC_AUC": occurrence_roc_auc,
            "Occurrence_Mean_Observed": prevalence,
            "Occurrence_Mean_Predicted": float(np.mean(predicted_probability)),
            "Occurrence_Mean_Bias": float(
                np.mean(predicted_probability) - prevalence
            ),
            "Occurrence_Predicted_to_Observed_Ratio": safe_ratio(
                float(np.mean(predicted_probability)), prevalence
            ),
            "Occurrence_Calibration_Intercept": occurrence_intercept,
            "Occurrence_Calibration_Slope": occurrence_slope,
            "Occurrence_Calibration_Probability_Clip": CALIBRATION_PROBABILITY_CLIP,
            "Occurrence_Calibration_Clipped_N": clipped_n,
            "Occurrence_Calibration_Converged": bool(calibration_converged),
            "Occurrence_Exact_Zero_Probability_N": int(
                np.sum(predicted_probability == 0.0)
            ),
            "Occurrence_Exact_One_Probability_N": int(
                np.sum(predicted_probability == 1.0)
            ),
        }
    )

    if include_calibration_bins:
        occurrence_bins, occurrence_ece = calibration_bin_table(
            validation_scheme=validation_scheme,
            season=season,
            response=response,
            model=model,
            observed=observed_positive.astype(float),
            predicted=predicted_probability,
            calibration_type="Occurrence_Probability",
        )
        rate_bins, rate_calibration_mae = calibration_bin_table(
            validation_scheme=validation_scheme,
            season=season,
            response=response,
            model=model,
            observed=observed_rate,
            predicted=predicted_rate,
            calibration_type="Unconditional_Rate",
        )
        row["Occurrence_Expected_Calibration_Error"] = occurrence_ece
        row["Rate_Binned_Calibration_MAE"] = rate_calibration_mae
    else:
        occurrence_bins = pd.DataFrame()
        rate_bins = pd.DataFrame()
        row["Occurrence_Expected_Calibration_Error"] = np.nan
        row["Rate_Binned_Calibration_MAE"] = np.nan

    return row, occurrence_bins, rate_bins


def calculate_extreme_rows(
    group: pd.DataFrame,
    validation_scheme: str,
    season: int,
    response: str,
    model: str,
) -> List[Dict[str, object]]:
    observed_count = group["Observed_Count"].to_numpy(dtype=np.float64)
    positive_mask = observed_count > 0
    positive = group.loc[positive_mask].copy()

    observed_rate = positive["Observed_Rate"].to_numpy(dtype=np.float64)
    observed_positive_count = positive["Observed_Count"].to_numpy(dtype=np.float64)
    predicted_rate = positive[f"{model}_Predicted_Rate"].to_numpy(dtype=np.float64)
    predicted_count = positive[f"{model}_Predicted_Count"].to_numpy(dtype=np.float64)

    rows: List[Dict[str, object]] = []

    for quantile in EXTREME_QUANTILES:
        threshold = float(np.quantile(observed_rate, quantile))
        true_extreme = observed_rate >= threshold
        true_n = int(np.sum(true_extreme))

        if true_n <= 0:
            raise ValueError("Extreme subset unexpectedly empty.")

        order = np.argsort(-predicted_rate, kind="mergesort")
        predicted_top = np.zeros(len(positive), dtype=bool)
        predicted_top[order[:true_n]] = True
        overlap = int(np.sum(true_extreme & predicted_top))

        binary_extreme = true_extreme.astype(np.int8)
        extreme_pr_auc = (
            float(average_precision_score(binary_extreme, predicted_rate))
            if len(np.unique(binary_extreme)) == 2
            else np.nan
        )
        extreme_roc_auc = (
            float(roc_auc_score(binary_extreme, predicted_rate))
            if len(np.unique(binary_extreme)) == 2
            else np.nan
        )

        rate_obs_extreme = observed_rate[true_extreme]
        rate_pred_extreme = predicted_rate[true_extreme]
        count_obs_extreme = observed_positive_count[true_extreme]
        count_pred_extreme = predicted_count[true_extreme]

        rows.append(
            {
                "Validation_Scheme": validation_scheme,
                "Season": int(season),
                "Season_Label": SEASONS[int(season)],
                "Count_Response": response,
                "Rate_Scale_Name": RESPONSES[response],
                "Model": model,
                "Extreme_Definition": (
                    f"Observed positive rate >= positive-only Q{int(quantile * 100)}"
                ),
                "Extreme_Quantile": float(quantile),
                "Observed_Positive_N": int(len(positive)),
                "Extreme_Threshold_Observed_Rate": threshold,
                "True_Extreme_N": true_n,
                "Predicted_Top_N": true_n,
                "Top_N_Overlap": overlap,
                "Top_N_Overlap_Fraction": safe_ratio(overlap, true_n),
                "Extreme_Classification_PR_AUC": extreme_pr_auc,
                "Extreme_Classification_ROC_AUC": extreme_roc_auc,
                "Extreme_Rate_RMSE": safe_rmse(rate_obs_extreme, rate_pred_extreme),
                "Extreme_Rate_MAE": safe_mae(rate_obs_extreme, rate_pred_extreme),
                "Extreme_Rate_Mean_Bias": float(
                    np.mean(rate_pred_extreme - rate_obs_extreme)
                ),
                "Extreme_Rate_Mean_Observed": float(np.mean(rate_obs_extreme)),
                "Extreme_Rate_Mean_Predicted": float(np.mean(rate_pred_extreme)),
                "Extreme_Rate_Predicted_to_Observed_Ratio": safe_ratio(
                    float(np.mean(rate_pred_extreme)),
                    float(np.mean(rate_obs_extreme)),
                ),
                "Extreme_Count_RMSE": safe_rmse(
                    count_obs_extreme, count_pred_extreme
                ),
                "Extreme_Count_MAE": safe_mae(
                    count_obs_extreme, count_pred_extreme
                ),
                "Extreme_Count_Mean_Bias": float(
                    np.mean(count_pred_extreme - count_obs_extreme)
                ),
                "Extreme_Count_Mean_Observed": float(
                    np.mean(count_obs_extreme)
                ),
                "Extreme_Count_Mean_Predicted": float(
                    np.mean(count_pred_extreme)
                ),
                "Extreme_Count_Predicted_to_Observed_Ratio": safe_ratio(
                    float(np.mean(count_pred_extreme)),
                    float(np.mean(count_obs_extreme)),
                ),
            }
        )

    return rows


# =============================================================================
# Definitions and QA
# =============================================================================


def metric_definitions() -> pd.DataFrame:
    rows = [
        ("Rate_RMSE", "Root mean squared error of unconditional predicted rate.", "Lower"),
        ("Rate_MAE", "Mean absolute error of unconditional predicted rate.", "Lower"),
        ("Rate_Median_Absolute_Error", "Median absolute error of unconditional predicted rate.", "Lower"),
        ("Rate_R2", "1 minus SSE/SST for unconditional predicted rate; may be negative.", "Higher"),
        ("Rate_Spearman", "Spearman rank correlation between observed and predicted rate.", "Higher"),
        ("Rate_Mean_Bias", "Mean(predicted rate minus observed rate).", "Closer_to_zero"),
        ("Rate_Predicted_to_Observed_Mean_Ratio", "Mean predicted rate divided by mean observed rate.", "Closer_to_one"),
        ("Count_RMSE", "Root mean squared error of unconditional predicted count.", "Lower"),
        ("Count_MAE", "Mean absolute error of unconditional predicted count.", "Lower"),
        ("Count_R2", "1 minus SSE/SST for unconditional predicted count; may be negative.", "Higher"),
        ("Count_Spearman", "Spearman rank correlation between observed and predicted count.", "Higher"),
        ("Occurrence_Brier_Score", "Mean squared error of the predicted positive-response probability.", "Lower"),
        ("Occurrence_Brier_Skill_Score", "1 minus model Brier divided by prevalence-only null Brier.", "Higher"),
        ("Occurrence_PR_AUC", "Average precision for observed count > 0; prevalence is the no-skill reference.", "Higher"),
        ("Occurrence_ROC_AUC", "ROC area under the curve for observed count > 0.", "Higher"),
        ("Occurrence_Calibration_Intercept", "Intercept from logit(observed occurrence) = a + b*logit(predicted probability).", "Closer_to_zero"),
        ("Occurrence_Calibration_Slope", "Slope from logit calibration model; probability is clipped only for fitting this diagnostic.", "Closer_to_one"),
        ("Occurrence_Expected_Calibration_Error", "Frequency-weighted absolute observed-predicted gap across probability quantile bins.", "Lower"),
        ("Positive_Conditional_Rate_RMSE", "RMSE of conditional-positive predicted rate, evaluated only where observed count > 0.", "Lower"),
        ("Positive_Conditional_Rate_MAE", "MAE of conditional-positive predicted rate, evaluated only where observed count > 0.", "Lower"),
        ("Positive_Conditional_Count_RMSE", "RMSE of conditional-positive predicted count, evaluated only where observed count > 0.", "Lower"),
        ("Positive_Conditional_Count_MAE", "MAE of conditional-positive predicted count, evaluated only where observed count > 0.", "Lower"),
        ("Rate_Binned_Calibration_MAE", "Frequency-weighted absolute observed-predicted mean-rate gap across prediction quantile bins.", "Lower"),
        ("Top_N_Overlap_Fraction", "Fraction of observed positive-rate extremes recovered among an equal number of highest predicted rates.", "Higher"),
        ("Extreme_Classification_PR_AUC", "Average precision for identifying positive-only upper-tail observed rates using predicted rate.", "Higher"),
        ("Positive_Conditional_Metrics_Available", "True only when the evaluated subset contains at least one observed positive response; otherwise conditional-positive metrics are NA by definition.", "Diagnostic"),
        ("Occurrence_Discrimination_Metrics_Available", "True only when the evaluated subset contains both occurrence classes; otherwise PR-AUC and ROC-AUC are NA by definition.", "Diagnostic"),
        ("Unavailable_Metric_Reason", "Reason that a fold-level metric is statistically undefined rather than computationally failed.", "Diagnostic"),
    ]

    return pd.DataFrame(
        rows,
        columns=["Metric", "Definition", "Preferred_Direction"],
    )


def computation_qa(
    overall: pd.DataFrame,
    fold: pd.DataFrame,
    occurrence_bins: pd.DataFrame,
    rate_bins: pd.DataFrame,
    extreme: pd.DataFrame,
) -> pd.DataFrame:
    expected_overall_rows = 2 * 3 * 2 * 3
    expected_fold_rows = expected_overall_rows * 5
    expected_extreme_rows = expected_overall_rows * len(EXTREME_QUANTILES)

    rows: List[Dict[str, object]] = []

    def add(check: str, passed: bool, observed: object, expected: object) -> None:
        rows.append(
            {
                "Check": check,
                "Passed": bool(passed),
                "Observed": observed,
                "Expected": expected,
            }
        )

    add("Overall_metric_row_count", len(overall) == expected_overall_rows, len(overall), expected_overall_rows)
    add("Fold_metric_row_count", len(fold) == expected_fold_rows, len(fold), expected_fold_rows)
    add("Extreme_metric_row_count", len(extreme) == expected_extreme_rows, len(extreme), expected_extreme_rows)
    add("Occurrence_calibration_nonempty", not occurrence_bins.empty, len(occurrence_bins), "> 0")
    add("Rate_calibration_nonempty", not rate_bins.empty, len(rate_bins), "> 0")
    add(
        "Overall_unique_key_count",
        not overall.duplicated(["Validation_Scheme", "Season", "Count_Response", "Model"]).any(),
        int(overall.duplicated(["Validation_Scheme", "Season", "Count_Response", "Model"]).sum()),
        0,
    )
    add(
        "Fold_unique_key_count",
        not fold.duplicated(["Validation_Scheme", "Validation_Fold", "Season", "Count_Response", "Model"]).any(),
        int(fold.duplicated(["Validation_Scheme", "Validation_Fold", "Season", "Count_Response", "Model"]).sum()),
        0,
    )
    add(
        "All_overall_N_locked",
        bool((overall["N"] == EXPECTED_ROWS_PER_SEASON_RESPONSE).all()),
        sorted(overall["N"].astype(int).unique().tolist()),
        EXPECTED_ROWS_PER_SEASON_RESPONSE,
    )
    add(
        "All_overall_grid_counts_locked",
        bool((overall["GRID_UID_Count"] == EXPECTED_GRID_COUNT).all()),
        sorted(overall["GRID_UID_Count"].astype(int).unique().tolist()),
        EXPECTED_GRID_COUNT,
    )
    add(
        "All_probabilistic_metrics_finite",
        bool(
            np.isfinite(
                overall[
                    [
                        "Occurrence_Brier_Score",
                        "Occurrence_PR_AUC",
                        "Occurrence_ROC_AUC",
                        "Occurrence_Expected_Calibration_Error",
                    ]
                ].to_numpy(dtype=float)
            ).all()
        ),
        "finite" if np.isfinite(overall[["Occurrence_Brier_Score", "Occurrence_PR_AUC", "Occurrence_ROC_AUC", "Occurrence_Expected_Calibration_Error"]].to_numpy(dtype=float)).all() else "nonfinite",
        "finite",
    )
    add(
        "All_primary_error_metrics_finite",
        bool(
            np.isfinite(
                overall[
                    [
                        "Rate_RMSE",
                        "Rate_MAE",
                        "Count_RMSE",
                        "Count_MAE",
                        "Positive_Conditional_Rate_RMSE",
                        "Positive_Conditional_Rate_MAE",
                    ]
                ].to_numpy(dtype=float)
            ).all()
        ),
        "finite" if np.isfinite(overall[["Rate_RMSE", "Rate_MAE", "Count_RMSE", "Count_MAE", "Positive_Conditional_Rate_RMSE", "Positive_Conditional_Rate_MAE"]].to_numpy(dtype=float)).all() else "nonfinite",
        "finite",
    )
    add(
        "Brier_range_valid",
        bool(overall["Occurrence_Brier_Score"].between(0.0, 1.0).all()),
        f"{overall['Occurrence_Brier_Score'].min():.6g} to {overall['Occurrence_Brier_Score'].max():.6g}",
        "0 to 1",
    )
    add(
        "AUC_ranges_valid",
        bool(
            overall["Occurrence_PR_AUC"].between(0.0, 1.0).all()
            and overall["Occurrence_ROC_AUC"].between(0.0, 1.0).all()
        ),
        "within [0,1]",
        "within [0,1]",
    )
    add(
        "Overall_occurrence_calibration_converged",
        bool(overall["Occurrence_Calibration_Converged"].astype(bool).all()),
        int(overall["Occurrence_Calibration_Converged"].astype(bool).sum()),
        expected_overall_rows,
    )
    add(
        "Extreme_overlap_range_valid",
        bool(extreme["Top_N_Overlap_Fraction"].between(0.0, 1.0).all()),
        f"{extreme['Top_N_Overlap_Fraction'].min():.6g} to {extreme['Top_N_Overlap_Fraction'].max():.6g}",
        "0 to 1",
    )
    conditional_columns = [
        "Positive_Conditional_Rate_RMSE",
        "Positive_Conditional_Rate_MAE",
        "Positive_Conditional_Count_RMSE",
        "Positive_Conditional_Count_MAE",
    ]
    zero_positive_fold_mask = fold["Observed_Positive_N"].astype(int) == 0
    positive_fold_mask = ~zero_positive_fold_mask
    single_class_fold_mask = ~fold[
        "Occurrence_Discrimination_Metrics_Available"
    ].astype(bool)
    two_class_fold_mask = ~single_class_fold_mask

    add(
        "Zero_positive_fold_rows_recorded_without_failure",
        bool(zero_positive_fold_mask.any()),
        int(zero_positive_fold_mask.sum()),
        ">= 1 because at least one held-out spatial subset has no BAD positives",
    )
    add(
        "Zero_positive_fold_conditional_metrics_are_NA",
        bool(
            fold.loc[zero_positive_fold_mask, conditional_columns]
            .isna()
            .all()
            .all()
        ),
        int(
            fold.loc[zero_positive_fold_mask, conditional_columns]
            .isna()
            .sum()
            .sum()
        ),
        int(zero_positive_fold_mask.sum() * len(conditional_columns)),
    )
    add(
        "Positive_fold_primary_conditional_metrics_finite",
        bool(
            np.isfinite(
                fold.loc[positive_fold_mask, conditional_columns]
                .to_numpy(dtype=float)
            ).all()
        ),
        "finite",
        "finite",
    )
    add(
        "Single_class_fold_AUCs_are_NA",
        bool(
            fold.loc[
                single_class_fold_mask,
                ["Occurrence_PR_AUC", "Occurrence_ROC_AUC"],
            ]
            .isna()
            .all()
            .all()
        ),
        int(single_class_fold_mask.sum()),
        "all single-class fold rows",
    )
    add(
        "Two_class_fold_AUCs_finite",
        bool(
            np.isfinite(
                fold.loc[
                    two_class_fold_mask,
                    ["Occurrence_PR_AUC", "Occurrence_ROC_AUC"],
                ].to_numpy(dtype=float)
            ).all()
        ),
        "finite",
        "finite",
    )
    add(
        "All_fold_unconditional_primary_metrics_finite",
        bool(
            np.isfinite(
                fold[["Rate_RMSE", "Rate_MAE", "Count_RMSE", "Count_MAE"]]
                .to_numpy(dtype=float)
            ).all()
        ),
        "finite",
        "finite",
    )

    add(
        "Predictive_log_probability_excluded",
        not any("LogProbability" in column for column in overall.columns),
        "excluded",
        "excluded from three-model metrics",
    )

    qa = pd.DataFrame(rows)
    if not qa["Passed"].all():
        failed = qa.loc[~qa["Passed"]]
        raise ValueError(
            "Unified metric QA failed:\n" + failed.to_string(index=False)
        )
    return qa


def build_output_manifest() -> pd.DataFrame:
    output_files = [
        METHOD_FILE,
        INPUT_AUDIT_FILE,
        OVERALL_METRICS_FILE,
        FOLD_METRICS_FILE,
        OCCURRENCE_CALIBRATION_FILE,
        RATE_CALIBRATION_FILE,
        EXTREME_FILE,
        METRIC_DEFINITION_FILE,
        QA_FILE,
        SOFTWARE_FILE,
        LOG_FILE,
    ]

    rows: List[Dict[str, object]] = []
    for path in output_files:
        if not path.exists():
            raise FileNotFoundError(f"Expected output missing: {path}")
        rows.append(
            {
                "File": path.name,
                "Relative_Path": str(path.relative_to(OUTPUT_ROOT)),
                "File_Size_Bytes": int(path.stat().st_size),
                "SHA256": sha256_file(path),
                "Step02_Code_Version": STEP02_CODE_VERSION,
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text("", encoding="utf-8")

    log("Starting unified model-comparison predictive metrics.")
    log(f"Step 02 code version: {STEP02_CODE_VERSION}.")
    log("Temporal and spatial OOF are analysed separately.")
    log("No model is fitted, tuned, selected, or predicted by this script.")
    log("Predictive log probability is excluded from the three-model comparison.")

    input_audit = verify_step01_inputs()
    log("Verified Step-01 method version, manifest, and aligned-file hashes.")

    overall_rows: List[Dict[str, object]] = []
    fold_rows: List[Dict[str, object]] = []
    occurrence_bin_frames: List[pd.DataFrame] = []
    rate_bin_frames: List[pd.DataFrame] = []
    extreme_rows: List[Dict[str, object]] = []

    input_files = {
        "Temporal": TEMPORAL_INPUT_FILE,
        "Spatial": SPATIAL_INPUT_FILE,
    }

    for validation_scheme in VALIDATION_SCHEMES:
        log(f"Reading and validating {validation_scheme} aligned OOF.")
        data = read_aligned_file(
            input_files[validation_scheme],
            expected_scheme=validation_scheme,
        )
        log(
            f"Verified {validation_scheme}: {len(data):,} rows, "
            f"{data['GRID_UID'].nunique():,} grids."
        )

        for season in sorted(SEASONS):
            for response in RESPONSES:
                group = data.loc[
                    (data["Season"] == season)
                    & (data["Count_Response"] == response)
                ].copy()

                if len(group) != EXPECTED_ROWS_PER_SEASON_RESPONSE:
                    raise ValueError(
                        f"Unexpected rows for {validation_scheme}, S{season}, {response}."
                    )

                for model in MODELS:
                    log(
                        f"Metrics: {validation_scheme}, {SEASONS[season]}, "
                        f"{response}, {model}."
                    )
                    row, occurrence_bins, rate_bins = calculate_metric_row(
                        group=group,
                        validation_scheme=validation_scheme,
                        season=season,
                        response=response,
                        model=model,
                        fold=None,
                        include_calibration_bins=True,
                    )
                    overall_rows.append(row)
                    occurrence_bin_frames.append(occurrence_bins)
                    rate_bin_frames.append(rate_bins)
                    extreme_rows.extend(
                        calculate_extreme_rows(
                            group=group,
                            validation_scheme=validation_scheme,
                            season=season,
                            response=response,
                            model=model,
                        )
                    )

                    for fold in sorted(EXPECTED_FOLDS):
                        fold_group = group.loc[
                            group["Validation_Fold"].astype(int) == fold
                        ]
                        if fold_group.empty:
                            raise ValueError(
                                f"Empty fold for {validation_scheme}, S{season}, "
                                f"{response}, {model}, fold {fold}."
                            )
                        fold_row, _, _ = calculate_metric_row(
                            group=fold_group,
                            validation_scheme=validation_scheme,
                            season=season,
                            response=response,
                            model=model,
                            fold=fold,
                            include_calibration_bins=False,
                        )
                        fold_rows.append(fold_row)
                        if not bool(
                            fold_row[
                                "Positive_Conditional_Metrics_Available"
                            ]
                        ) or not bool(
                            fold_row[
                                "Occurrence_Discrimination_Metrics_Available"
                            ]
                        ):
                            log(
                                "Undefined fold-level metrics recorded as NA: "
                                f"{validation_scheme}, {SEASONS[season]}, "
                                f"{response}, {model}, fold={fold}; "
                                f"{fold_row['Unavailable_Metric_Reason']}."
                            )

        del data

    overall = pd.DataFrame(overall_rows)
    fold_metrics = pd.DataFrame(fold_rows)
    occurrence_bins = pd.concat(occurrence_bin_frames, ignore_index=True)
    rate_bins = pd.concat(rate_bin_frames, ignore_index=True)
    extreme = pd.DataFrame(extreme_rows)

    overall = overall.sort_values(
        ["Validation_Scheme", "Season", "Count_Response", "Model"]
    ).reset_index(drop=True)
    fold_metrics = fold_metrics.sort_values(
        [
            "Validation_Scheme",
            "Validation_Fold",
            "Season",
            "Count_Response",
            "Model",
        ]
    ).reset_index(drop=True)
    occurrence_bins = occurrence_bins.sort_values(
        ["Validation_Scheme", "Season", "Count_Response", "Model", "Bin_Order"]
    ).reset_index(drop=True)
    rate_bins = rate_bins.sort_values(
        ["Validation_Scheme", "Season", "Count_Response", "Model", "Bin_Order"]
    ).reset_index(drop=True)
    extreme = extreme.sort_values(
        [
            "Validation_Scheme",
            "Season",
            "Count_Response",
            "Model",
            "Extreme_Quantile",
        ]
    ).reset_index(drop=True)

    definitions = metric_definitions()
    qa = computation_qa(
        overall=overall,
        fold=fold_metrics,
        occurrence_bins=occurrence_bins,
        rate_bins=rate_bins,
        extreme=extreme,
    )

    overall.to_csv(OVERALL_METRICS_FILE, index=False, encoding="utf-8-sig")
    fold_metrics.to_csv(FOLD_METRICS_FILE, index=False, encoding="utf-8-sig")
    occurrence_bins.to_csv(
        OCCURRENCE_CALIBRATION_FILE, index=False, encoding="utf-8-sig"
    )
    rate_bins.to_csv(RATE_CALIBRATION_FILE, index=False, encoding="utf-8-sig")
    extreme.to_csv(EXTREME_FILE, index=False, encoding="utf-8-sig")
    definitions.to_csv(METRIC_DEFINITION_FILE, index=False, encoding="utf-8-sig")
    qa.to_csv(QA_FILE, index=False, encoding="utf-8-sig")

    method = {
        "Step": "02_Calculate_Unified_Prediction_Metrics",
        "Code_Version": STEP02_CODE_VERSION,
        "Created": datetime.now().isoformat(timespec="seconds"),
        "Purpose": (
            "Calculate directly comparable temporal- and spatial-OOF predictive "
            "metrics for Baseline NB1, Final Regression, and Final Hurdle RF."
        ),
        "No_Model_Fitting": True,
        "No_Model_Tuning": True,
        "No_Model_Selection": True,
        "No_Bootstrap": True,
        "Models": MODELS,
        "Validation_Schemes": VALIDATION_SCHEMES,
        "Input": {
            "Step01_Code_Version": EXPECTED_STEP01_CODE_VERSION,
            "Temporal_File": str(TEMPORAL_INPUT_FILE),
            "Spatial_File": str(SPATIAL_INPUT_FILE),
            "Rows_Per_Scheme": EXPECTED_ROWS_PER_SCHEME,
            "Rows_Per_Season_Response": EXPECTED_ROWS_PER_SEASON_RESPONSE,
            "GRID_UID_Count": EXPECTED_GRID_COUNT,
        },
        "Primary_Accuracy_Metrics": [
            "Rate_RMSE",
            "Rate_MAE",
            "Count_RMSE",
            "Count_MAE",
        ],
        "Occurrence_Metrics": [
            "Brier score",
            "Brier skill score",
            "PR-AUC",
            "ROC-AUC",
            "calibration intercept",
            "calibration slope",
            "expected calibration error",
        ],
        "Undefined_Fold_Metric_Policy": {
            "No_observed_positive_response": (
                "Conditional-positive severity metrics are recorded as NA; "
                "unconditional rate/count metrics and Brier score remain valid."
            ),
            "Single_occurrence_class": (
                "PR-AUC, ROC-AUC, and occurrence calibration intercept/slope "
                "are recorded as NA because they are statistically undefined."
            ),
            "Interpretation": (
                "NA denotes an unavailable estimand in that held-out subset, "
                "not a model-computation failure and not a score of zero."
            ),
        },
        "Conditional_Positive_Evaluation": (
            "Conditional-positive count and rate predictions are evaluated only "
            "on records where Observed_Count > 0."
        ),
        "Calibration": {
            "Bin_Count": CALIBRATION_BIN_COUNT,
            "Occurrence_Logit_Clip": CALIBRATION_PROBABILITY_CLIP,
            "Clip_Use": (
                "Only for fitting occurrence calibration intercept and slope; "
                "Brier, PR-AUC, ROC-AUC, and calibration bins use original probabilities."
            ),
        },
        "Extreme_Event_Definition": (
            "Upper 10% and upper 5% of observed rates among observed-positive "
            "records within each validation-season-response combination."
        ),
        "Predictive_LogProbability": (
            "Excluded because the final Hurdle RF supplies occurrence probability "
            "and conditional mean severity but not a complete count probability mass function."
        ),
        "Interpretation_Limits": [
            "Temporal and spatial OOF represent different generalization tasks and are not pooled.",
            "This step provides descriptive performance metrics; paired uncertainty is added in Step 03.",
            "Negative R2 is valid and indicates worse squared-error performance than the observed mean.",
            "Occurrence PR-AUC should be interpreted relative to the observed positive fraction.",
            "Conditional-positive metrics do not evaluate occurrence prediction.",
        ],
        "Output_Row_Counts": {
            "Overall_Metrics": int(len(overall)),
            "Fold_Metrics": int(len(fold_metrics)),
            "Occurrence_Calibration_Bins": int(len(occurrence_bins)),
            "Rate_Calibration_Bins": int(len(rate_bins)),
            "Extreme_Event_Rows": int(len(extreme)),
        },
    }
    write_json(METHOD_FILE, method)

    software = {
        "Created": datetime.now().isoformat(timespec="seconds"),
        "Python": sys.version,
        "Platform": platform.platform(),
        "NumPy": np.__version__,
        "pandas": pd.__version__,
        "SciPy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "Step02_Code_Version": STEP02_CODE_VERSION,
    }
    write_json(SOFTWARE_FILE, software)

    log(
        "Computed 36 overall model-performance rows, 180 fold-level rows, "
        f"{len(occurrence_bins):,} occurrence calibration-bin rows, "
        f"{len(rate_bins):,} rate calibration-bin rows, and "
        f"{len(extreme):,} extreme-event rows."
    )
    log("All computation QA checks passed.")
    log(
        "Step 02 completed: unified temporal and spatial OOF predictive metrics "
        "were calculated for Baseline NB1, Final Regression, and Final Hurdle RF "
        "without fitting or selecting any model."
    )

    # The final log line is written before hashing the log, so the manifest hash
    # remains stable after this point.
    manifest = build_output_manifest()
    manifest.to_csv(OUTPUT_MANIFEST_FILE, index=False, encoding="utf-8-sig")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        error_text = traceback.format_exc()
        try:
            log("=" * 78)
            log("ERROR")
            log(error_text)
            log("=" * 78)
        except Exception:
            print(error_text, file=sys.stderr)
        raise
