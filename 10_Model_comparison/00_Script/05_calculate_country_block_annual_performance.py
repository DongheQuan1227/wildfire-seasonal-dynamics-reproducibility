# -*- coding: utf-8 -*-
"""
Model Comparison Step 05: country, spatial-block, and annual robustness.

This script reads the two strictly aligned OOF panels created by Step 01 and
calculates descriptive performance for Baseline NB1, Final Regression, and
Final Hurdle RF at country, spatial-block, year, country-year, and
spatial-block-year scales.

No model is fitted, tuned, selected, or used to regenerate predictions.
Temporal and spatial OOF results are always analysed separately.
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
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


# =============================================================================
# Paths and locked settings
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_COMPARISON_ROOT = SCRIPT_DIR.parent

STEP01_ROOT = MODEL_COMPARISON_ROOT / "01_Aligned_Model_OOF"
STEP02_ROOT = MODEL_COMPARISON_ROOT / "02_Unified_Prediction_Metrics"
OUTPUT_ROOT = MODEL_COMPARISON_ROOT / "05_Country_Block_Annual_Performance"

STEP01_METHOD_FILE = STEP01_ROOT / "00_Method_Definition.json"
STEP01_MANIFEST_FILE = STEP01_ROOT / "07_Aligned_OOF_Manifest.csv"
TEMPORAL_OOF_FILE = STEP01_ROOT / "05_Aligned_Temporal_OOF.csv.gz"
SPATIAL_OOF_FILE = STEP01_ROOT / "06_Aligned_Spatial_OOF.csv.gz"

STEP02_METHOD_FILE = STEP02_ROOT / "00_Method_Definition.json"
STEP02_OVERALL_FILE = STEP02_ROOT / "02_Overall_Unified_Prediction_Metrics.csv"
STEP02_MANIFEST_FILE = STEP02_ROOT / "09_Output_Manifest.csv"

LOG_FILE = OUTPUT_ROOT / "country_block_annual_performance.log"
METHOD_FILE = OUTPUT_ROOT / "00_Method_Definition.json"
INPUT_AUDIT_FILE = OUTPUT_ROOT / "01_Input_Integrity_Audit.csv"
GLOBAL_REPRODUCTION_FILE = OUTPUT_ROOT / "01b_Step02_Global_Metric_Reproduction_Audit.csv"
COUNTRY_PERFORMANCE_FILE = OUTPUT_ROOT / "02_Country_Level_Performance.csv"
BLOCK_PERFORMANCE_FILE = OUTPUT_ROOT / "03_Spatial_Block_Level_Performance.csv"
ANNUAL_GRID_PERFORMANCE_FILE = OUTPUT_ROOT / "04_Annual_Grid_Level_Performance.csv"
ANNUAL_AGGREGATE_FILE = OUTPUT_ROOT / "05_Annual_Aggregated_Predictions.csv"
ANNUAL_SERIES_FILE = OUTPUT_ROOT / "06_Annual_Series_Performance.csv"
COUNTRY_ANNUAL_AGGREGATE_FILE = OUTPUT_ROOT / "07_Country_Annual_Aggregated_Predictions.csv"
COUNTRY_ANNUAL_SERIES_FILE = OUTPUT_ROOT / "08_Country_Annual_Series_Performance.csv"
BLOCK_ANNUAL_AGGREGATE_FILE = OUTPUT_ROOT / "09_Spatial_Block_Annual_Aggregated_Predictions.csv"
BLOCK_ANNUAL_SERIES_FILE = OUTPUT_ROOT / "10_Spatial_Block_Annual_Series_Performance.csv"
SUBGROUP_RANK_FILE = OUTPUT_ROOT / "11_Subgroup_Model_Ranks.csv"
SUBGROUP_WIN_FILE = OUTPUT_ROOT / "12_Subgroup_Model_Win_Summary.csv"
ANNUAL_RANK_FILE = OUTPUT_ROOT / "13_Annual_Series_Model_Ranks.csv"
ANNUAL_WIN_FILE = OUTPUT_ROOT / "14_Annual_Series_Model_Win_Summary.csv"
COMPUTATION_QA_FILE = OUTPUT_ROOT / "15_Computation_QA.csv"
METRIC_DEFINITION_FILE = OUTPUT_ROOT / "16_Metric_Definitions.csv"
OUTPUT_MANIFEST_FILE = OUTPUT_ROOT / "17_Output_Manifest.csv"
SOFTWARE_FILE = OUTPUT_ROOT / "Software_Environment.json"

STEP05_CODE_VERSION = "2026-06-30_MODEL_COMPARISON_COUNTRY_BLOCK_ANNUAL_V1"
EXPECTED_STEP01_VERSION = "2026-06-30_MODEL_COMPARISON_ALIGNED_OOF_V1"
EXPECTED_STEP02_VERSION = (
    "2026-06-30_MODEL_COMPARISON_UNIFIED_METRICS_"
    "V1A_ZERO_POSITIVE_FOLD_COMPATIBILITY"
)

MODELS = ["Baseline_NB1", "Final_Regression", "Final_Hurdle_RF"]
VALIDATION_SCHEMES = ["Temporal", "Spatial"]
SEASONS = {1: "Spring", 2: "Summer", 3: "Autumn"}
RESPONSES = {
    "Fire_Count": "FCD",
    "Burned_Pixel_Count": "BAD",
}
EXPECTED_YEARS = list(range(2001, 2026))
EXPECTED_COUNTRIES = {"China", "NK", "Russia"}
EXPECTED_SPATIAL_BLOCKS = {1, 2, 3, 4, 5}
EXPECTED_ROWS_PER_SCHEME = 742_374
EXPECTED_GRID_COUNT = 4_982
EXPECTED_ROWS_PER_SEASON_RESPONSE = 123_729
EXPECTED_BLOCK_GRID_COUNTS = {1: 912, 2: 1042, 3: 735, 4: 1222, 5: 1071}

FLOAT_TOLERANCE = 1e-10
HASH_CHUNK_SIZE = 1024 * 1024

BASE_COLUMNS = [
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
]
MODEL_SUFFIXES = [
    "Predicted_Count",
    "Predicted_Rate",
    "Predicted_Positive_Probability",
    "Predicted_Conditional_Positive_Count",
    "Predicted_Conditional_Positive_Rate",
]
MODEL_COLUMNS = [f"{model}_{suffix}" for model in MODELS for suffix in MODEL_SUFFIXES]
INPUT_COLUMNS = BASE_COLUMNS + MODEL_COLUMNS

CORE_REPRODUCTION_METRICS = [
    "N",
    "GRID_UID_Count",
    "Year_Count",
    "Observed_Positive_N",
    "Observed_Negative_N",
    "Observed_Positive_Fraction",
    "Count_RMSE",
    "Count_MAE",
    "Count_Median_Absolute_Error",
    "Count_R2",
    "Count_Spearman",
    "Count_Mean_Bias",
    "Count_Mean_Observed",
    "Count_Mean_Predicted",
    "Count_Predicted_to_Observed_Mean_Ratio",
    "Count_Total_Observed",
    "Count_Total_Predicted",
    "Count_Predicted_to_Observed_Total_Ratio",
    "Rate_RMSE",
    "Rate_MAE",
    "Rate_Median_Absolute_Error",
    "Rate_R2",
    "Rate_Spearman",
    "Rate_Mean_Bias",
    "Rate_Mean_Observed",
    "Rate_Mean_Predicted",
    "Rate_Predicted_to_Observed_Mean_Ratio",
    "Rate_Total_Observed",
    "Rate_Total_Predicted",
    "Rate_Predicted_to_Observed_Total_Ratio",
    "Positive_Conditional_Count_RMSE",
    "Positive_Conditional_Count_MAE",
    "Positive_Conditional_Count_R2",
    "Positive_Conditional_Count_Spearman",
    "Positive_Conditional_Count_Mean_Bias",
    "Positive_Conditional_Rate_RMSE",
    "Positive_Conditional_Rate_MAE",
    "Positive_Conditional_Rate_R2",
    "Positive_Conditional_Rate_Spearman",
    "Positive_Conditional_Rate_Mean_Bias",
    "Occurrence_Brier_Score",
    "Occurrence_PR_AUC",
    "Occurrence_ROC_AUC",
    "Occurrence_Mean_Observed",
    "Occurrence_Mean_Predicted",
    "Occurrence_Mean_Bias",
    "Occurrence_Predicted_to_Observed_Ratio",
]


# =============================================================================
# Generic helpers
# =============================================================================

def log(message: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}"
    print(line, flush=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(HASH_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(denominator) or abs(denominator) <= np.finfo(float).tiny:
        return np.nan
    return float(numerator / denominator)


def safe_rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(predicted - observed))))


def safe_mae(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(predicted - observed)))


def safe_median_ae(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.median(np.abs(predicted - observed)))


def safe_r2(observed: np.ndarray, predicted: np.ndarray) -> float:
    denominator = float(np.sum(np.square(observed - np.mean(observed))))
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
    return float(spearmanr(observed, predicted, nan_policy="omit").statistic)


def linear_slope(years: np.ndarray, values: np.ndarray) -> float:
    years = years.astype(float)
    values = values.astype(float)
    if len(years) < 2 or np.std(years) <= np.finfo(float).tiny:
        return np.nan
    centered_years = years - np.mean(years)
    denominator = float(np.sum(np.square(centered_years)))
    if denominator <= np.finfo(float).tiny:
        return np.nan
    return float(np.sum(centered_years * (values - np.mean(values))) / denominator)


def years_at_extreme(years: np.ndarray, values: np.ndarray, mode: str) -> str:
    if mode == "max":
        target = np.nanmax(values)
    elif mode == "min":
        target = np.nanmin(values)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    selected = years[np.isclose(values, target, rtol=0.0, atol=1e-15)]
    return ";".join(str(int(year)) for year in selected)


def top_year_overlap(years: np.ndarray, observed: np.ndarray, predicted: np.ndarray, n: int) -> float:
    n = min(int(n), len(years))
    observed_order = np.argsort(-observed, kind="mergesort")[:n]
    predicted_order = np.argsort(-predicted, kind="mergesort")[:n]
    observed_years = set(years[observed_order].astype(int).tolist())
    predicted_years = set(years[predicted_order].astype(int).tolist())
    return safe_ratio(len(observed_years & predicted_years), n)


def output_manifest_row(path: Path) -> Dict[str, object]:
    return {
        "File": path.name,
        "Relative_Path": str(path.relative_to(OUTPUT_ROOT)),
        "File_Size_Bytes": int(path.stat().st_size),
        "SHA256": sha256_file(path),
        "Step05_Code_Version": STEP05_CODE_VERSION,
    }


# =============================================================================
# Input verification
# =============================================================================

def manifest_hash_for_file(manifest: pd.DataFrame, file_name: str) -> str:
    file_column = "Output_File" if "Output_File" in manifest.columns else "File"
    relative_column = "Relative_Path" if "Relative_Path" in manifest.columns else file_column
    match = manifest.loc[
        (manifest[file_column].astype(str) == file_name)
        | (manifest[relative_column].astype(str) == file_name)
    ]
    if len(match) != 1:
        raise ValueError(f"Manifest must contain exactly one row for {file_name}; found {len(match)}.")
    return str(match.iloc[0]["SHA256"])


def verify_upstream() -> pd.DataFrame:
    required = [
        STEP01_METHOD_FILE,
        STEP01_MANIFEST_FILE,
        TEMPORAL_OOF_FILE,
        SPATIAL_OOF_FILE,
        STEP02_METHOD_FILE,
        STEP02_OVERALL_FILE,
        STEP02_MANIFEST_FILE,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Required upstream files are missing:\n" + "\n".join(missing))

    step01_method = read_json(STEP01_METHOD_FILE)
    step02_method = read_json(STEP02_METHOD_FILE)
    step01_manifest = pd.read_csv(STEP01_MANIFEST_FILE)
    step02_manifest = pd.read_csv(STEP02_MANIFEST_FILE)

    if step01_method.get("Code_Version") != EXPECTED_STEP01_VERSION:
        raise ValueError(
            f"Unexpected Step 01 version: {step01_method.get('Code_Version')}"
        )
    if step02_method.get("Code_Version") != EXPECTED_STEP02_VERSION:
        raise ValueError(
            f"Unexpected Step 02 version: {step02_method.get('Code_Version')}"
        )

    rows: List[Dict[str, object]] = []
    for scheme, path in [("Temporal", TEMPORAL_OOF_FILE), ("Spatial", SPATIAL_OOF_FILE)]:
        expected_hash = manifest_hash_for_file(step01_manifest, path.name)
        actual_hash = sha256_file(path)
        passed = expected_hash == actual_hash
        rows.append(
            {
                "Source": "Step01",
                "Validation_Scheme": scheme,
                "File": path.name,
                "Expected_SHA256": expected_hash,
                "Actual_SHA256": actual_hash,
                "Passed": passed,
            }
        )
        if not passed:
            raise ValueError(f"Step 01 aligned OOF hash mismatch: {path}")

    expected_hash = manifest_hash_for_file(step02_manifest, STEP02_OVERALL_FILE.name)
    actual_hash = sha256_file(STEP02_OVERALL_FILE)
    passed = expected_hash == actual_hash
    rows.append(
        {
            "Source": "Step02",
            "Validation_Scheme": "Both",
            "File": STEP02_OVERALL_FILE.name,
            "Expected_SHA256": expected_hash,
            "Actual_SHA256": actual_hash,
            "Passed": passed,
        }
    )
    if not passed:
        raise ValueError("Step 02 overall metrics hash mismatch.")

    return pd.DataFrame(rows)


def read_and_validate_aligned(path: Path, expected_scheme: str) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    missing = sorted(set(INPUT_COLUMNS) - set(header.columns))
    if missing:
        raise ValueError(f"{path.name} is missing required columns:\n" + "\n".join(missing))

    data = pd.read_csv(path, usecols=INPUT_COLUMNS)
    if len(data) != EXPECTED_ROWS_PER_SCHEME:
        raise ValueError(
            f"Unexpected {expected_scheme} row count: {len(data):,}; "
            f"expected {EXPECTED_ROWS_PER_SCHEME:,}."
        )

    if data["Validation_Scheme"].astype(str).nunique() != 1:
        raise ValueError(f"Multiple Validation_Scheme values found in {path.name}.")
    if str(data["Validation_Scheme"].iloc[0]) != expected_scheme:
        raise ValueError(
            f"Validation scheme mismatch in {path.name}: "
            f"{data['Validation_Scheme'].iloc[0]} vs {expected_scheme}."
        )

    numeric_columns = [
        "Year",
        "Season",
        "Temporal_Fold",
        "Spatial_Block",
        "Validation_Fold",
        "ForestPixelCount",
        "Observed_Count",
        "Observed_Rate",
        *MODEL_COLUMNS,
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="raise")

    if not np.isfinite(data[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError(f"Non-finite required values found in {path.name}.")
    if data["GRID_UID"].nunique() != EXPECTED_GRID_COUNT:
        raise ValueError(f"Unexpected GRID_UID count in {path.name}.")
    if set(data["Year"].astype(int).unique()) != set(EXPECTED_YEARS):
        raise ValueError(f"Unexpected year universe in {path.name}.")
    if set(data["Season"].astype(int).unique()) != set(SEASONS):
        raise ValueError(f"Unexpected season universe in {path.name}.")
    if set(data["Count_Response"].astype(str).unique()) != set(RESPONSES):
        raise ValueError(f"Unexpected response universe in {path.name}.")
    if set(data["Country"].astype(str).unique()) != EXPECTED_COUNTRIES:
        raise ValueError(
            f"Unexpected countries in {path.name}: "
            f"{sorted(data['Country'].astype(str).unique())}"
        )
    if set(data["Spatial_Block"].astype(int).unique()) != EXPECTED_SPATIAL_BLOCKS:
        raise ValueError(f"Unexpected spatial blocks in {path.name}.")

    duplicate = data.duplicated(["GRID_UID", "Year", "Season", "Count_Response"], keep=False)
    if duplicate.any():
        raise ValueError(f"Duplicate aligned OOF keys found in {path.name}.")

    combination_counts = data.groupby(["Season", "Count_Response"]).size()
    if len(combination_counts) != 6 or not (
        combination_counts == EXPECTED_ROWS_PER_SEASON_RESPONSE
    ).all():
        raise ValueError(f"Unexpected season-response counts in {path.name}.")

    block_grid_counts = (
        data[["GRID_UID", "Spatial_Block"]]
        .drop_duplicates()["Spatial_Block"]
        .astype(int)
        .value_counts()
        .sort_index()
        .to_dict()
    )
    if block_grid_counts != EXPECTED_BLOCK_GRID_COUNTS:
        raise ValueError(
            f"Spatial block grid counts differ in {path.name}: {block_grid_counts}"
        )

    observed_rate = data["Observed_Count"] / data["ForestPixelCount"]
    if float(np.max(np.abs(observed_rate - data["Observed_Rate"]))) > FLOAT_TOLERANCE:
        raise ValueError(f"Observed rate identity failed in {path.name}.")

    exposure = data["ForestPixelCount"].to_numpy(dtype=float)
    for model in MODELS:
        predicted_count = data[f"{model}_Predicted_Count"].to_numpy(dtype=float)
        predicted_rate = data[f"{model}_Predicted_Rate"].to_numpy(dtype=float)
        probability = data[f"{model}_Predicted_Positive_Probability"].to_numpy(dtype=float)
        conditional_count = data[
            f"{model}_Predicted_Conditional_Positive_Count"
        ].to_numpy(dtype=float)
        conditional_rate = data[
            f"{model}_Predicted_Conditional_Positive_Rate"
        ].to_numpy(dtype=float)

        if (probability < -FLOAT_TOLERANCE).any() or (probability > 1 + FLOAT_TOLERANCE).any():
            raise ValueError(f"Probability outside [0,1] for {model} in {path.name}.")
        if float(np.max(np.abs(predicted_count / exposure - predicted_rate))) > 1e-9:
            raise ValueError(f"Count-rate identity failed for {model} in {path.name}.")
        if float(np.max(np.abs(probability * conditional_count - predicted_count))) > 1e-8:
            raise ValueError(f"Hurdle count identity failed for {model} in {path.name}.")
        if float(np.max(np.abs(probability * conditional_rate - predicted_rate))) > 1e-10:
            raise ValueError(f"Hurdle rate identity failed for {model} in {path.name}.")

    return data


# =============================================================================
# Performance metrics
# =============================================================================

def continuous_metric_block(
    observed: np.ndarray,
    predicted: np.ndarray,
    prefix: str,
) -> Dict[str, float]:
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
    }


def unavailable_continuous_metric_block(prefix: str) -> Dict[str, float]:
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
    ]
    return {f"{prefix}_{suffix}": np.nan for suffix in suffixes}


def calculate_metric_row(
    group: pd.DataFrame,
    validation_scheme: str,
    season: int,
    response: str,
    model: str,
    extra_metadata: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    observed_count = group["Observed_Count"].to_numpy(dtype=float)
    observed_rate = group["Observed_Rate"].to_numpy(dtype=float)
    observed_positive = (observed_count > 0).astype(np.int8)

    predicted_count = group[f"{model}_Predicted_Count"].to_numpy(dtype=float)
    predicted_rate = group[f"{model}_Predicted_Rate"].to_numpy(dtype=float)
    probability = group[
        f"{model}_Predicted_Positive_Probability"
    ].to_numpy(dtype=float)
    conditional_count = group[
        f"{model}_Predicted_Conditional_Positive_Count"
    ].to_numpy(dtype=float)
    conditional_rate = group[
        f"{model}_Predicted_Conditional_Positive_Rate"
    ].to_numpy(dtype=float)

    positive_mask = observed_positive == 1
    positive_n = int(positive_mask.sum())
    negative_n = int(len(group) - positive_n)
    two_classes = positive_n > 0 and negative_n > 0
    prevalence = float(np.mean(observed_positive))
    brier = float(brier_score_loss(observed_positive, probability))
    null_brier = float(np.mean(np.square(observed_positive - prevalence)))

    row: Dict[str, object] = {
        "Validation_Scheme": validation_scheme,
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
        "Positive_Conditional_Metrics_Available": positive_n > 0,
        "Occurrence_Discrimination_Metrics_Available": two_classes,
    }
    if extra_metadata:
        row.update(extra_metadata)

    row.update(continuous_metric_block(observed_count, predicted_count, "Count"))
    row.update(continuous_metric_block(observed_rate, predicted_rate, "Rate"))

    if positive_n > 0:
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
        row.update(unavailable_continuous_metric_block("Positive_Conditional_Count"))
        row.update(unavailable_continuous_metric_block("Positive_Conditional_Rate"))

    row.update(
        {
            "Occurrence_Brier_Score": brier,
            "Occurrence_Null_Brier_Score": null_brier,
            "Occurrence_Brier_Skill_Score": (
                np.nan
                if null_brier <= np.finfo(float).tiny
                else float(1.0 - brier / null_brier)
            ),
            "Occurrence_PR_AUC": (
                float(average_precision_score(observed_positive, probability))
                if two_classes
                else np.nan
            ),
            "Occurrence_ROC_AUC": (
                float(roc_auc_score(observed_positive, probability))
                if two_classes
                else np.nan
            ),
            "Occurrence_Mean_Observed": prevalence,
            "Occurrence_Mean_Predicted": float(np.mean(probability)),
            "Occurrence_Mean_Bias": float(np.mean(probability) - prevalence),
            "Occurrence_Predicted_to_Observed_Ratio": safe_ratio(
                float(np.mean(probability)), prevalence
            ),
            "Occurrence_Exact_Zero_Probability_N": int(np.sum(probability == 0.0)),
            "Occurrence_Exact_One_Probability_N": int(np.sum(probability == 1.0)),
        }
    )
    return row


def calculate_group_performance(
    data: pd.DataFrame,
    validation_scheme: str,
    group_columns: Sequence[str],
    group_label: str,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    full_group_columns = [*group_columns, "Season", "Count_Response"]

    for key, group in data.groupby(full_group_columns, sort=True, observed=True):
        if not isinstance(key, tuple):
            key = (key,)
        metadata = dict(zip(full_group_columns, key))
        season = int(metadata.pop("Season"))
        response = str(metadata.pop("Count_Response"))
        metadata["Grouping_Level"] = group_label

        for model in MODELS:
            rows.append(
                calculate_metric_row(
                    group=group,
                    validation_scheme=validation_scheme,
                    season=season,
                    response=response,
                    model=model,
                    extra_metadata=metadata,
                )
            )

    return pd.DataFrame(rows)


# =============================================================================
# Annual aggregation and annual-series metrics
# =============================================================================

def aggregate_predictions(
    data: pd.DataFrame,
    validation_scheme: str,
    group_columns: Sequence[str],
    grouping_level: str,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    full_group_columns = [*group_columns, "Year", "Season", "Count_Response"]

    for key, group in data.groupby(full_group_columns, sort=True, observed=True):
        if not isinstance(key, tuple):
            key = (key,)
        metadata = dict(zip(full_group_columns, key))
        year = int(metadata.pop("Year"))
        season = int(metadata.pop("Season"))
        response = str(metadata.pop("Count_Response"))

        observed_count = group["Observed_Count"].to_numpy(dtype=float)
        observed_positive = (observed_count > 0).astype(np.int8)
        exposure_total = float(group["ForestPixelCount"].sum())
        observed_total = float(observed_count.sum())
        observed_weighted_rate = safe_ratio(observed_total, exposure_total)

        for model in MODELS:
            predicted_count = group[f"{model}_Predicted_Count"].to_numpy(dtype=float)
            probability = group[
                f"{model}_Predicted_Positive_Probability"
            ].to_numpy(dtype=float)
            predicted_total = float(predicted_count.sum())
            predicted_weighted_rate = safe_ratio(predicted_total, exposure_total)

            row: Dict[str, object] = {
                "Grouping_Level": grouping_level,
                "Validation_Scheme": validation_scheme,
                **metadata,
                "Year": year,
                "Season": season,
                "Season_Label": SEASONS[season],
                "Count_Response": response,
                "Rate_Scale_Name": RESPONSES[response],
                "Model": model,
                "Record_N": int(len(group)),
                "GRID_UID_Count": int(group["GRID_UID"].nunique()),
                "ForestPixelCount_Total": exposure_total,
                "Observed_Total_Count": observed_total,
                "Predicted_Total_Count": predicted_total,
                "Total_Count_Bias": predicted_total - observed_total,
                "Predicted_to_Observed_Total_Count_Ratio": safe_ratio(
                    predicted_total, observed_total
                ),
                "Observed_Exposure_Weighted_Rate": observed_weighted_rate,
                "Predicted_Exposure_Weighted_Rate": predicted_weighted_rate,
                "Exposure_Weighted_Rate_Bias": (
                    predicted_weighted_rate - observed_weighted_rate
                ),
                "Predicted_to_Observed_Exposure_Weighted_Rate_Ratio": safe_ratio(
                    predicted_weighted_rate, observed_weighted_rate
                ),
                "Observed_Positive_Grid_N": int(observed_positive.sum()),
                "Observed_Positive_Grid_Fraction": float(observed_positive.mean()),
                "Expected_Positive_Grid_N": float(probability.sum()),
                "Mean_Predicted_Positive_Probability": float(probability.mean()),
                "Expected_minus_Observed_Positive_Grid_N": float(
                    probability.sum() - observed_positive.sum()
                ),
            }
            rows.append(row)

    return pd.DataFrame(rows)


def annual_series_metric_block(
    years: np.ndarray,
    observed: np.ndarray,
    predicted: np.ndarray,
    prefix: str,
) -> Dict[str, object]:
    observed_peak = years_at_extreme(years, observed, "max")
    predicted_peak = years_at_extreme(years, predicted, "max")
    observed_low = years_at_extreme(years, observed, "min")
    predicted_low = years_at_extreme(years, predicted, "min")

    return {
        f"{prefix}_RMSE": safe_rmse(observed, predicted),
        f"{prefix}_MAE": safe_mae(observed, predicted),
        f"{prefix}_R2": safe_r2(observed, predicted),
        f"{prefix}_Spearman": safe_spearman(observed, predicted),
        f"{prefix}_Mean_Bias": float(np.mean(predicted - observed)),
        f"{prefix}_Predicted_to_Observed_Total_Ratio": safe_ratio(
            float(np.sum(predicted)), float(np.sum(observed))
        ),
        f"{prefix}_Observed_Linear_Slope_per_Year": linear_slope(years, observed),
        f"{prefix}_Predicted_Linear_Slope_per_Year": linear_slope(years, predicted),
        f"{prefix}_Predicted_minus_Observed_Slope": (
            linear_slope(years, predicted) - linear_slope(years, observed)
        ),
        f"{prefix}_Observed_Peak_Year": observed_peak,
        f"{prefix}_Predicted_Peak_Year": predicted_peak,
        f"{prefix}_Peak_Year_Exact_Match": observed_peak == predicted_peak,
        f"{prefix}_Observed_Low_Year": observed_low,
        f"{prefix}_Predicted_Low_Year": predicted_low,
        f"{prefix}_Low_Year_Exact_Match": observed_low == predicted_low,
        f"{prefix}_Top3_Year_Overlap_Fraction": top_year_overlap(
            years, observed, predicted, 3
        ),
        f"{prefix}_Top5_Year_Overlap_Fraction": top_year_overlap(
            years, observed, predicted, 5
        ),
    }


def calculate_annual_series_performance(
    annual: pd.DataFrame,
    group_columns: Sequence[str],
    grouping_level: str,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    full_group_columns = [
        "Validation_Scheme",
        *group_columns,
        "Season",
        "Count_Response",
        "Model",
    ]

    for key, group in annual.groupby(full_group_columns, sort=True, observed=True):
        if not isinstance(key, tuple):
            key = (key,)
        metadata = dict(zip(full_group_columns, key))
        group = group.sort_values("Year").reset_index(drop=True)

        years = group["Year"].to_numpy(dtype=int)
        if set(years.tolist()) != set(EXPECTED_YEARS) or len(years) != len(EXPECTED_YEARS):
            raise ValueError(
                f"Annual series is not a complete 25-year panel for {metadata}."
            )

        observed_count = group["Observed_Total_Count"].to_numpy(dtype=float)
        predicted_count = group["Predicted_Total_Count"].to_numpy(dtype=float)
        observed_rate = group[
            "Observed_Exposure_Weighted_Rate"
        ].to_numpy(dtype=float)
        predicted_rate = group[
            "Predicted_Exposure_Weighted_Rate"
        ].to_numpy(dtype=float)

        row: Dict[str, object] = {
            "Grouping_Level": grouping_level,
            **metadata,
            "Season_Label": SEASONS[int(metadata["Season"])],
            "Rate_Scale_Name": RESPONSES[str(metadata["Count_Response"])],
            "Year_N": int(len(years)),
            "First_Year": int(years.min()),
            "Last_Year": int(years.max()),
        }
        row.update(
            annual_series_metric_block(
                years, observed_count, predicted_count, "Annual_Total_Count"
            )
        )
        row.update(
            annual_series_metric_block(
                years,
                observed_rate,
                predicted_rate,
                "Annual_Exposure_Weighted_Rate",
            )
        )
        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# Ranking and QA
# =============================================================================

def add_subgroup_ranks(
    country: pd.DataFrame,
    block: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    country_rank = country.copy()
    country_rank["Subgroup_Type"] = "Country"
    country_rank["Subgroup_ID"] = country_rank["Country"].astype(str)

    block_rank = block.copy()
    block_rank["Subgroup_Type"] = "Spatial_Block"
    block_rank["Subgroup_ID"] = block_rank["Spatial_Block"].astype(int).astype(str)

    selected_columns = [
        "Subgroup_Type",
        "Subgroup_ID",
        "Validation_Scheme",
        "Season",
        "Season_Label",
        "Count_Response",
        "Rate_Scale_Name",
        "Model",
        "Rate_RMSE",
        "Rate_MAE",
        "Occurrence_Brier_Score",
        "Positive_Conditional_Rate_RMSE",
        "Positive_Conditional_Rate_MAE",
    ]
    combined = pd.concat(
        [country_rank[selected_columns], block_rank[selected_columns]],
        ignore_index=True,
    )

    group_columns = [
        "Subgroup_Type",
        "Subgroup_ID",
        "Validation_Scheme",
        "Season",
        "Count_Response",
    ]
    rank_metrics = [
        "Rate_RMSE",
        "Rate_MAE",
        "Occurrence_Brier_Score",
        "Positive_Conditional_Rate_RMSE",
        "Positive_Conditional_Rate_MAE",
    ]
    for metric in rank_metrics:
        combined[f"{metric}_Rank_Lower_Is_Better"] = (
            combined.groupby(group_columns, dropna=False)[metric]
            .rank(ascending=True, method="min")
            .astype("Int64")
        )

    win_rows: List[Dict[str, object]] = []
    for metric in rank_metrics:
        rank_column = f"{metric}_Rank_Lower_Is_Better"
        temporary = combined.loc[combined[rank_column].notna()].copy()
        temporary["Rank1"] = temporary[rank_column] == 1
        summary = (
            temporary.groupby(
                ["Subgroup_Type", "Validation_Scheme", "Model"],
                as_index=False,
            )
            .agg(
                Comparison_N=(rank_column, "size"),
                Rank1_N=("Rank1", "sum"),
                Mean_Rank=(rank_column, "mean"),
                Median_Rank=(rank_column, "median"),
            )
        )
        summary.insert(2, "Metric", metric)
        win_rows.extend(summary.to_dict("records"))

    return combined, pd.DataFrame(win_rows)


def add_annual_ranks(
    overall: pd.DataFrame,
    country: pd.DataFrame,
    block: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    frames: List[pd.DataFrame] = []

    overall_copy = overall.copy()
    overall_copy["Annual_Group_Type"] = "Overall"
    overall_copy["Annual_Group_ID"] = "All"
    frames.append(overall_copy)

    country_copy = country.copy()
    country_copy["Annual_Group_Type"] = "Country"
    country_copy["Annual_Group_ID"] = country_copy["Country"].astype(str)
    frames.append(country_copy)

    block_copy = block.copy()
    block_copy["Annual_Group_Type"] = "Spatial_Block"
    block_copy["Annual_Group_ID"] = block_copy["Spatial_Block"].astype(int).astype(str)
    frames.append(block_copy)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    metrics = [
        "Annual_Total_Count_RMSE",
        "Annual_Total_Count_MAE",
        "Annual_Exposure_Weighted_Rate_RMSE",
        "Annual_Exposure_Weighted_Rate_MAE",
    ]
    group_columns = [
        "Annual_Group_Type",
        "Annual_Group_ID",
        "Validation_Scheme",
        "Season",
        "Count_Response",
    ]
    for metric in metrics:
        combined[f"{metric}_Rank_Lower_Is_Better"] = (
            combined.groupby(group_columns, dropna=False)[metric]
            .rank(ascending=True, method="min")
            .astype("Int64")
        )

    win_rows: List[Dict[str, object]] = []
    for metric in metrics:
        rank_column = f"{metric}_Rank_Lower_Is_Better"
        temporary = combined.loc[combined[rank_column].notna()].copy()
        temporary["Rank1"] = temporary[rank_column] == 1
        summary = (
            temporary.groupby(
                ["Annual_Group_Type", "Validation_Scheme", "Model"],
                as_index=False,
            )
            .agg(
                Comparison_N=(rank_column, "size"),
                Rank1_N=("Rank1", "sum"),
                Mean_Rank=(rank_column, "mean"),
                Median_Rank=(rank_column, "median"),
            )
        )
        summary.insert(2, "Metric", metric)
        win_rows.extend(summary.to_dict("records"))

    return combined, pd.DataFrame(win_rows)


def reproduce_step02_global_metrics(
    scheme_data: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    step02 = pd.read_csv(STEP02_OVERALL_FILE)
    recalculated_rows: List[Dict[str, object]] = []

    for scheme, data in scheme_data.items():
        for season in sorted(SEASONS):
            for response in RESPONSES:
                group = data.loc[
                    (data["Season"].astype(int) == season)
                    & (data["Count_Response"].astype(str) == response)
                ]
                for model in MODELS:
                    recalculated_rows.append(
                        calculate_metric_row(
                            group,
                            scheme,
                            season,
                            response,
                            model,
                        )
                    )

    recalculated = pd.DataFrame(recalculated_rows)
    keys = ["Validation_Scheme", "Season", "Count_Response", "Model"]
    merged = step02.merge(
        recalculated,
        on=keys,
        suffixes=("_Step02", "_Step05_Recalculated"),
        how="outer",
        indicator=True,
    )

    rows: List[Dict[str, object]] = []
    for _, record in merged.iterrows():
        for metric in CORE_REPRODUCTION_METRICS:
            left_name = f"{metric}_Step02"
            right_name = f"{metric}_Step05_Recalculated"
            if left_name not in merged.columns or right_name not in merged.columns:
                rows.append(
                    {
                        **{key: record.get(key) for key in keys},
                        "Metric": metric,
                        "Step02_Value": np.nan,
                        "Step05_Recalculated_Value": np.nan,
                        "Absolute_Difference": np.nan,
                        "Passed": False,
                        "Detail": "Metric column missing",
                    }
                )
                continue

            left = record[left_name]
            right = record[right_name]
            if pd.isna(left) and pd.isna(right):
                passed = True
                difference = 0.0
            elif pd.isna(left) != pd.isna(right):
                passed = False
                difference = np.nan
            else:
                difference = abs(float(left) - float(right))
                passed = bool(
                    np.isclose(
                        float(left),
                        float(right),
                        atol=5e-12,
                        rtol=5e-10,
                        equal_nan=True,
                    )
                )

            rows.append(
                {
                    **{key: record.get(key) for key in keys},
                    "Metric": metric,
                    "Step02_Value": left,
                    "Step05_Recalculated_Value": right,
                    "Absolute_Difference": difference,
                    "Passed": passed and record["_merge"] == "both",
                    "Detail": "" if record["_merge"] == "both" else str(record["_merge"]),
                }
            )

    result = pd.DataFrame(rows)
    if not result["Passed"].all():
        failed = result.loc[~result["Passed"]]
        failed.to_csv(
            OUTPUT_ROOT / "ERROR_Step02_Global_Metric_Reproduction.csv",
            index=False,
            encoding="utf-8-sig",
        )
        raise ValueError(
            f"Step 02 global metric reproduction failed for {len(failed)} comparisons."
        )
    return result


def build_computation_qa(
    country: pd.DataFrame,
    block: pd.DataFrame,
    annual_grid: pd.DataFrame,
    annual: pd.DataFrame,
    annual_series: pd.DataFrame,
    country_annual: pd.DataFrame,
    country_series: pd.DataFrame,
    block_annual: pd.DataFrame,
    block_series: pd.DataFrame,
    scheme_data: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
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

    add("Country_level_row_count", len(country) == 108, len(country), 108)
    add("Spatial_block_level_row_count", len(block) == 180, len(block), 180)
    add("Annual_grid_level_row_count", len(annual_grid) == 900, len(annual_grid), 900)
    add("Annual_aggregate_row_count", len(annual) == 900, len(annual), 900)
    add("Annual_series_row_count", len(annual_series) == 36, len(annual_series), 36)
    add("Country_annual_aggregate_row_count", len(country_annual) == 2700, len(country_annual), 2700)
    add("Country_annual_series_row_count", len(country_series) == 108, len(country_series), 108)
    add("Block_annual_aggregate_row_count", len(block_annual) == 4500, len(block_annual), 4500)
    add("Block_annual_series_row_count", len(block_series) == 180, len(block_series), 180)

    for scheme, data in scheme_data.items():
        observed_annual = annual.loc[annual["Validation_Scheme"] == scheme]
        country_sum = (
            country_annual.loc[country_annual["Validation_Scheme"] == scheme]
            .groupby(["Year", "Season", "Count_Response", "Model"], as_index=False)
            .agg(
                Observed_Total_Count=("Observed_Total_Count", "sum"),
                Predicted_Total_Count=("Predicted_Total_Count", "sum"),
                ForestPixelCount_Total=("ForestPixelCount_Total", "sum"),
            )
        )
        block_sum = (
            block_annual.loc[block_annual["Validation_Scheme"] == scheme]
            .groupby(["Year", "Season", "Count_Response", "Model"], as_index=False)
            .agg(
                Observed_Total_Count=("Observed_Total_Count", "sum"),
                Predicted_Total_Count=("Predicted_Total_Count", "sum"),
                ForestPixelCount_Total=("ForestPixelCount_Total", "sum"),
            )
        )
        annual_selected = observed_annual[
            [
                "Year",
                "Season",
                "Count_Response",
                "Model",
                "Observed_Total_Count",
                "Predicted_Total_Count",
                "ForestPixelCount_Total",
            ]
        ].sort_values(["Year", "Season", "Count_Response", "Model"]).reset_index(drop=True)
        country_selected = country_sum.sort_values(
            ["Year", "Season", "Count_Response", "Model"]
        ).reset_index(drop=True)
        block_selected = block_sum.sort_values(
            ["Year", "Season", "Count_Response", "Model"]
        ).reset_index(drop=True)

        country_difference = float(
            np.nanmax(
                np.abs(
                    annual_selected[
                        ["Observed_Total_Count", "Predicted_Total_Count", "ForestPixelCount_Total"]
                    ].to_numpy(dtype=float)
                    - country_selected[
                        ["Observed_Total_Count", "Predicted_Total_Count", "ForestPixelCount_Total"]
                    ].to_numpy(dtype=float)
                )
            )
        )
        block_difference = float(
            np.nanmax(
                np.abs(
                    annual_selected[
                        ["Observed_Total_Count", "Predicted_Total_Count", "ForestPixelCount_Total"]
                    ].to_numpy(dtype=float)
                    - block_selected[
                        ["Observed_Total_Count", "Predicted_Total_Count", "ForestPixelCount_Total"]
                    ].to_numpy(dtype=float)
                )
            )
        )
        add(
            f"{scheme}_country_annual_sums_equal_overall",
            country_difference <= 1e-8,
            country_difference,
            "<=1e-8",
        )
        add(
            f"{scheme}_block_annual_sums_equal_overall",
            block_difference <= 1e-8,
            block_difference,
            "<=1e-8",
        )

        country_n = country.loc[country["Validation_Scheme"] == scheme].groupby(
            ["Season", "Count_Response", "Model"]
        )["N"].sum()
        block_n = block.loc[block["Validation_Scheme"] == scheme].groupby(
            ["Season", "Count_Response", "Model"]
        )["N"].sum()
        expected_n = EXPECTED_ROWS_PER_SEASON_RESPONSE
        add(
            f"{scheme}_country_partition_row_counts",
            bool((country_n == expected_n).all()),
            sorted(country_n.unique().tolist()),
            [expected_n],
        )
        add(
            f"{scheme}_block_partition_row_counts",
            bool((block_n == expected_n).all()),
            sorted(block_n.unique().tolist()),
            [expected_n],
        )

    temporal_observed = annual.loc[
        annual["Validation_Scheme"] == "Temporal",
        [
            "Year",
            "Season",
            "Count_Response",
            "Model",
            "Observed_Total_Count",
            "Observed_Exposure_Weighted_Rate",
            "ForestPixelCount_Total",
        ],
    ].sort_values(["Year", "Season", "Count_Response", "Model"]).reset_index(drop=True)
    spatial_observed = annual.loc[
        annual["Validation_Scheme"] == "Spatial",
        [
            "Year",
            "Season",
            "Count_Response",
            "Model",
            "Observed_Total_Count",
            "Observed_Exposure_Weighted_Rate",
            "ForestPixelCount_Total",
        ],
    ].sort_values(["Year", "Season", "Count_Response", "Model"]).reset_index(drop=True)

    observed_difference = float(
        np.nanmax(
            np.abs(
                temporal_observed[
                    [
                        "Observed_Total_Count",
                        "Observed_Exposure_Weighted_Rate",
                        "ForestPixelCount_Total",
                    ]
                ].to_numpy(dtype=float)
                - spatial_observed[
                    [
                        "Observed_Total_Count",
                        "Observed_Exposure_Weighted_Rate",
                        "ForestPixelCount_Total",
                    ]
                ].to_numpy(dtype=float)
            )
        )
    )
    add(
        "Temporal_and_spatial_observed_annual_aggregates_identical",
        observed_difference <= 1e-12,
        observed_difference,
        "<=1e-12",
    )

    all_series_complete = all(
        frame["Year_N"].eq(25).all()
        for frame in [annual_series, country_series, block_series]
    )
    add(
        "All_annual_series_have_25_years",
        all_series_complete,
        "all 25" if all_series_complete else "incomplete",
        "all 25",
    )

    return pd.DataFrame(rows)


def metric_definitions() -> pd.DataFrame:
    rows = [
        ("Country-level performance", "Record-level OOF metrics calculated within each country, season, response, validation scheme, and model."),
        ("Spatial-block-level performance", "Record-level OOF metrics calculated within each locked spatial block. Temporal and spatial OOF remain separate."),
        ("Annual grid-level performance", "Within-year metrics across all grid records for each season-response-model combination."),
        ("Annual total count", "Sum of observed or predicted response counts over all records in the aggregation unit and year."),
        ("Annual exposure-weighted rate", "Annual total count divided by the summed ForestPixelCount exposure."),
        ("Annual-series RMSE/MAE", "Error across the 25 annual aggregated values, not across individual grid-year records."),
        ("Annual-series Spearman", "Rank agreement between observed and predicted annual aggregates across 2001-2025."),
        ("Annual linear slope", "Ordinary least-squares slope of the annual aggregate against calendar year; descriptive, not a trend-significance test."),
        ("Top-3/Top-5 year overlap", "Fraction of observed highest-activity years also appearing among the same number of highest predicted years."),
        ("Occurrence Brier score", "Mean squared error of the positive-response probability; lower is better."),
        ("Conditional-positive metrics", "Evaluated only for records where the observed response count is positive; recorded as NA when no positive records exist."),
        ("Subgroup ranks", "Descriptive within-subgroup ranks. No new bootstrap or multiplicity-adjusted inference is assigned."),
    ]
    return pd.DataFrame(rows, columns=["Metric_or_Table", "Definition"])


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text("", encoding="utf-8")

    log("Starting country, spatial-block, and annual model-performance analysis.")
    log(f"Step 05 code version: {STEP05_CODE_VERSION}.")
    log("Temporal and spatial OOF are analysed separately.")
    log("No model is fitted, tuned, selected, or used to regenerate predictions.")

    input_audit = verify_upstream()
    log("Verified Step 01 and Step 02 versions and locked input hashes.")

    scheme_data: Dict[str, pd.DataFrame] = {}
    country_frames: List[pd.DataFrame] = []
    block_frames: List[pd.DataFrame] = []
    annual_grid_frames: List[pd.DataFrame] = []
    annual_aggregate_frames: List[pd.DataFrame] = []
    country_annual_frames: List[pd.DataFrame] = []
    block_annual_frames: List[pd.DataFrame] = []

    for scheme, path in [("Temporal", TEMPORAL_OOF_FILE), ("Spatial", SPATIAL_OOF_FILE)]:
        log(f"Reading and validating {scheme} aligned OOF.")
        data = read_and_validate_aligned(path, scheme)
        scheme_data[scheme] = data
        log(f"Verified {scheme}: {len(data):,} rows, {data['GRID_UID'].nunique():,} grids.")

        log(f"Calculating {scheme} country-level metrics.")
        country_frames.append(
            calculate_group_performance(data, scheme, ["Country"], "Country")
        )

        log(f"Calculating {scheme} spatial-block-level metrics.")
        block_frames.append(
            calculate_group_performance(
                data, scheme, ["Spatial_Block"], "Spatial_Block"
            )
        )

        log(f"Calculating {scheme} annual grid-level metrics.")
        annual_grid_frames.append(
            calculate_group_performance(data, scheme, ["Year"], "Annual_Grid_Level")
        )

        log(f"Aggregating {scheme} annual predictions.")
        annual_aggregate_frames.append(
            aggregate_predictions(data, scheme, [], "Overall_Annual")
        )
        country_annual_frames.append(
            aggregate_predictions(data, scheme, ["Country"], "Country_Annual")
        )
        block_annual_frames.append(
            aggregate_predictions(
                data, scheme, ["Spatial_Block"], "Spatial_Block_Annual"
            )
        )

    country_performance = pd.concat(country_frames, ignore_index=True)
    block_performance = pd.concat(block_frames, ignore_index=True)
    annual_grid_performance = pd.concat(annual_grid_frames, ignore_index=True)
    annual_aggregate = pd.concat(annual_aggregate_frames, ignore_index=True)
    country_annual = pd.concat(country_annual_frames, ignore_index=True)
    block_annual = pd.concat(block_annual_frames, ignore_index=True)

    log("Calculating 25-year annual-series performance summaries.")
    annual_series = calculate_annual_series_performance(
        annual_aggregate, [], "Overall_Annual_Series"
    )
    country_annual_series = calculate_annual_series_performance(
        country_annual, ["Country"], "Country_Annual_Series"
    )
    block_annual_series = calculate_annual_series_performance(
        block_annual, ["Spatial_Block"], "Spatial_Block_Annual_Series"
    )

    subgroup_ranks, subgroup_wins = add_subgroup_ranks(
        country_performance, block_performance
    )
    annual_ranks, annual_wins = add_annual_ranks(
        annual_series, country_annual_series, block_annual_series
    )

    log("Reproducing Step 02 global metrics under the Step 05 implementation.")
    global_reproduction = reproduce_step02_global_metrics(scheme_data)

    computation_qa = build_computation_qa(
        country_performance,
        block_performance,
        annual_grid_performance,
        annual_aggregate,
        annual_series,
        country_annual,
        country_annual_series,
        block_annual,
        block_annual_series,
        scheme_data,
    )
    if not computation_qa["Passed"].all():
        computation_qa.to_csv(
            COMPUTATION_QA_FILE, index=False, encoding="utf-8-sig"
        )
        raise ValueError("Step 05 computation QA failed.")

    input_audit.to_csv(INPUT_AUDIT_FILE, index=False, encoding="utf-8-sig")
    global_reproduction.to_csv(
        GLOBAL_REPRODUCTION_FILE, index=False, encoding="utf-8-sig"
    )
    country_performance.to_csv(
        COUNTRY_PERFORMANCE_FILE, index=False, encoding="utf-8-sig"
    )
    block_performance.to_csv(
        BLOCK_PERFORMANCE_FILE, index=False, encoding="utf-8-sig"
    )
    annual_grid_performance.to_csv(
        ANNUAL_GRID_PERFORMANCE_FILE, index=False, encoding="utf-8-sig"
    )
    annual_aggregate.to_csv(
        ANNUAL_AGGREGATE_FILE, index=False, encoding="utf-8-sig"
    )
    annual_series.to_csv(
        ANNUAL_SERIES_FILE, index=False, encoding="utf-8-sig"
    )
    country_annual.to_csv(
        COUNTRY_ANNUAL_AGGREGATE_FILE, index=False, encoding="utf-8-sig"
    )
    country_annual_series.to_csv(
        COUNTRY_ANNUAL_SERIES_FILE, index=False, encoding="utf-8-sig"
    )
    block_annual.to_csv(
        BLOCK_ANNUAL_AGGREGATE_FILE, index=False, encoding="utf-8-sig"
    )
    block_annual_series.to_csv(
        BLOCK_ANNUAL_SERIES_FILE, index=False, encoding="utf-8-sig"
    )
    subgroup_ranks.to_csv(
        SUBGROUP_RANK_FILE, index=False, encoding="utf-8-sig"
    )
    subgroup_wins.to_csv(
        SUBGROUP_WIN_FILE, index=False, encoding="utf-8-sig"
    )
    annual_ranks.to_csv(
        ANNUAL_RANK_FILE, index=False, encoding="utf-8-sig"
    )
    annual_wins.to_csv(
        ANNUAL_WIN_FILE, index=False, encoding="utf-8-sig"
    )
    computation_qa.to_csv(
        COMPUTATION_QA_FILE, index=False, encoding="utf-8-sig"
    )
    metric_definitions().to_csv(
        METRIC_DEFINITION_FILE, index=False, encoding="utf-8-sig"
    )

    method = {
        "Code_Version": STEP05_CODE_VERSION,
        "Created": datetime.now().isoformat(timespec="seconds"),
        "Purpose": (
            "Descriptive robustness comparison of three locked models at "
            "country, spatial-block, annual, country-annual, and "
            "spatial-block-annual scales."
        ),
        "Models": MODELS,
        "Validation_Schemes": VALIDATION_SCHEMES,
        "Years": [2001, 2025],
        "Countries": sorted(EXPECTED_COUNTRIES),
        "Spatial_Blocks": sorted(EXPECTED_SPATIAL_BLOCKS),
        "No_Model_Fitting": True,
        "No_Model_Tuning": True,
        "No_Model_Selection": True,
        "No_OOF_Prediction_Recomputation": True,
        "No_Moran_Recomputation": True,
        "No_New_Bootstrap_Inference": True,
        "Primary_Interpretation": (
            "Country and block metrics describe regional heterogeneity. "
            "Annual-series metrics assess reproduction of interannual totals "
            "and exposure-weighted rates. Ranks are descriptive and do not "
            "replace Step 03 paired bootstrap inference."
        ),
        "Input_Files": {
            "Temporal": str(TEMPORAL_OOF_FILE),
            "Spatial": str(SPATIAL_OOF_FILE),
            "Step02_Overall": str(STEP02_OVERALL_FILE),
        },
    }
    write_json(METHOD_FILE, method)

    environment = {
        "Created": datetime.now().isoformat(timespec="seconds"),
        "Python": sys.version,
        "Platform": platform.platform(),
        "NumPy": np.__version__,
        "pandas": pd.__version__,
        "SciPy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
    }
    write_json(SOFTWARE_FILE, environment)

    # Write the completion line before hashing the log so the manifest records
    # the final, immutable log content. No log message is written after this.
    log(
        "Step 05 completed: country, spatial-block, annual, country-annual, "
        "and spatial-block-annual performance was calculated for three locked "
        "models under temporal and spatial OOF without fitting, selecting, or "
        "predicting any model."
    )

    manifest_targets = [
        METHOD_FILE,
        INPUT_AUDIT_FILE,
        GLOBAL_REPRODUCTION_FILE,
        COUNTRY_PERFORMANCE_FILE,
        BLOCK_PERFORMANCE_FILE,
        ANNUAL_GRID_PERFORMANCE_FILE,
        ANNUAL_AGGREGATE_FILE,
        ANNUAL_SERIES_FILE,
        COUNTRY_ANNUAL_AGGREGATE_FILE,
        COUNTRY_ANNUAL_SERIES_FILE,
        BLOCK_ANNUAL_AGGREGATE_FILE,
        BLOCK_ANNUAL_SERIES_FILE,
        SUBGROUP_RANK_FILE,
        SUBGROUP_WIN_FILE,
        ANNUAL_RANK_FILE,
        ANNUAL_WIN_FILE,
        COMPUTATION_QA_FILE,
        METRIC_DEFINITION_FILE,
        SOFTWARE_FILE,
        LOG_FILE,
    ]
    manifest = pd.DataFrame([output_manifest_row(path) for path in manifest_targets])
    manifest.to_csv(
        OUTPUT_MANIFEST_FILE, index=False, encoding="utf-8-sig"
    )
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
