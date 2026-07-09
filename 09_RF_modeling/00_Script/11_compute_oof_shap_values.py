# -*- coding: utf-8 -*-

# Step 11: temporal out-of-fold SHAP analysis.
# Public reproducibility version.
# Input paths are resolved relative to the code repository.
# ForestPixelCount stores the original pixel count; ForestArea_km2 stores physical area.


from __future__ import annotations

import json
import math
import platform
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

try:
    import shap
except ImportError as error:
    raise SystemExit(
        "\nThe 'shap' package is required.\n"
        "Install it with:\n"
        'python -m pip install shap\n'
    ) from error

import matplotlib.pyplot as plt


# =============================================================================
# Paths and settings
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_ROOT = SCRIPT_DIR.parent
CODE_ROOT = MODEL_ROOT.parent
INPUT_CSV = CODE_ROOT / "00_Input_data" / "Base_Table_2001_2025.csv.bz2"

OUTPUT_ROOT = MODEL_ROOT / "11_Final_OOF_SHAP_Direction"
LOG_FILE = OUTPUT_ROOT / "11_oof_SHAP_direction.log"
STEP_ID = "11_OOF_SHAP_Direction"
STEP_CODE_VERSION = "2026-06-30_STEP11_RENUMBERED_AFTER_RF_RESIDUAL_MORAN"

# ForestPixelCount is the count of 500 m x 500 m forest pixels.
# ForestArea_km2 is ForestPixelCount * 0.25; models use ForestArea_km2 > 2.5.
FOREST_AREA_THRESHOLD_KM2 = 2.5
RANDOM_SEED = 2026
N_ESTIMATORS = 80

# Occurrence SHAP uses a representative random sample from each validation fold.
MAX_OCCURRENCE_SHAP_ROWS_PER_FOLD = 300

# Severity SHAP uses actual positive validation records only.
MAX_SEVERITY_SHAP_ROWS_PER_FOLD = 100

BIN_COUNT = 10
TOP_FEATURES_IN_BAR_FIGURE = 12
TOP_CONTINUOUS_FEATURES_IN_DEPENDENCE_FIGURE = 4

SEASONS = {
    1: "Spring",
    2: "Summer",
    3: "Autumn",
}

RESPONSES = ["FCD", "BAD"]

TEMPORAL_FOLDS = {
    1: [2024, 2002, 2006, 2004, 2009],
    2: [2016, 2015, 2007, 2018, 2014],
    3: [2025, 2021, 2013, 2019, 2003],
    4: [2010, 2023, 2017, 2011, 2005],
    5: [2020, 2022, 2001, 2012, 2008],
}

CONTINUOUS_PREDICTORS = [
    "BD",
    "ND",
    "NE",
    "EVI",
    "PTC",
    "Temp",
    "Pre",
    "Rhum",
    "Wind",
    "SSRD",
    "LtgProxy",
    "SPEI3",
    "SPEI12",
    "DEM",
    "Slope",
    "Aspect",
    "POP",
    "Dis_Farm",
    "Road_dens",
    "Dis_Railway",
]

COUNTRY_DUMMY_COLUMNS = [
    "Country_China",
    "Country_NK",
    "Country_Russia",
]

MODEL_PREDICTORS = (
    CONTINUOUS_PREDICTORS
    + COUNTRY_DUMMY_COLUMNS
)

PREDICTOR_GROUPS: Dict[str, List[str]] = {
    "Forest_Structure": [
        "BD",
        "ND",
        "NE",
        "EVI",
        "PTC",
    ],
    "Meteorology": [
        "Temp",
        "Pre",
        "Rhum",
        "Wind",
        "SSRD",
    ],
    "Lightning": [
        "LtgProxy",
    ],
    "Drought": [
        "SPEI3",
        "SPEI12",
    ],
    "Terrain": [
        "DEM",
        "Slope",
        "Aspect",
    ],
    "Anthropogenic": [
        "POP",
        "Dis_Farm",
        "Road_dens",
        "Dis_Railway",
    ],
    "Country_Context": COUNTRY_DUMMY_COLUMNS,
}

LOCKED_CONFIGS: Dict[Tuple[int, str], Dict[str, object]] = {
    (1, "FCD"): {
        "occurrence_name": "O1_Baseline",
        "occurrence_max_features": "sqrt",
        "occurrence_min_samples_leaf": 5,
        "occurrence_max_samples": 0.70,
        "severity_name": "S3_Leaf10",
        "severity_max_features": "sqrt",
        "severity_min_samples_leaf": 10,
        "severity_max_samples": 0.80,
    },
    (1, "BAD"): {
        "occurrence_name": "O4_Features50_Leaf20",
        "occurrence_max_features": 0.50,
        "occurrence_min_samples_leaf": 20,
        "occurrence_max_samples": 0.70,
        "severity_name": "S3_Leaf10",
        "severity_max_features": "sqrt",
        "severity_min_samples_leaf": 10,
        "severity_max_samples": 0.80,
    },
    (2, "FCD"): {
        "occurrence_name": "O2_Leaf20",
        "occurrence_max_features": "sqrt",
        "occurrence_min_samples_leaf": 20,
        "occurrence_max_samples": 0.70,
        "severity_name": "S3_Leaf10",
        "severity_max_features": "sqrt",
        "severity_min_samples_leaf": 10,
        "severity_max_samples": 0.80,
    },
    (2, "BAD"): {
        "occurrence_name": "O4_Features50_Leaf20",
        "occurrence_max_features": 0.50,
        "occurrence_min_samples_leaf": 20,
        "occurrence_max_samples": 0.70,
        "severity_name": "S3_Leaf10",
        "severity_max_features": "sqrt",
        "severity_min_samples_leaf": 10,
        "severity_max_samples": 0.80,
    },
    (3, "FCD"): {
        "occurrence_name": "O2_Leaf20",
        "occurrence_max_features": "sqrt",
        "occurrence_min_samples_leaf": 20,
        "occurrence_max_samples": 0.70,
        "severity_name": "S3_Leaf10",
        "severity_max_features": "sqrt",
        "severity_min_samples_leaf": 10,
        "severity_max_samples": 0.80,
    },
    (3, "BAD"): {
        "occurrence_name": "O1_Baseline",
        "occurrence_max_features": "sqrt",
        "occurrence_min_samples_leaf": 5,
        "occurrence_max_samples": 0.70,
        "severity_name": "S5_Features50_Leaf10",
        "severity_max_features": 0.50,
        "severity_min_samples_leaf": 10,
        "severity_max_samples": 0.80,
    },
}


# =============================================================================
# Logging
# =============================================================================

def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as file:
        file.write(line + "\n")


# =============================================================================
# Basic helpers
# =============================================================================

def safe_spearman(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    valid = np.isfinite(x) & np.isfinite(y)

    if valid.sum() < 3:
        return np.nan

    x_valid = x[valid]
    y_valid = y[valid]

    if np.std(x_valid) == 0 or np.std(y_valid) == 0:
        return np.nan

    return float(
        spearmanr(
            x_valid,
            y_valid,
            nan_policy="omit",
        ).statistic
    )


def predictor_family(predictor: str) -> str:
    for group_name, columns in PREDICTOR_GROUPS.items():
        if predictor in columns:
            return group_name

    raise KeyError(f"No group defined for predictor: {predictor}")


def classify_direction(
    predictor: str,
    spearman_value: float,
    high_minus_low_shap: float,
) -> str:
    if predictor in COUNTRY_DUMMY_COLUMNS:
        if high_minus_low_shap > 0:
            return "Presence_associated_with_higher_prediction"
        if high_minus_low_shap < 0:
            return "Presence_associated_with_lower_prediction"
        return "No_clear_difference"

    if not np.isfinite(spearman_value):
        return "Uncertain"

    same_positive = (
        spearman_value >= 0.20
        and high_minus_low_shap > 0
    )

    same_negative = (
        spearman_value <= -0.20
        and high_minus_low_shap < 0
    )

    if same_positive:
        return "Generally_increasing"

    if same_negative:
        return "Generally_decreasing"

    if abs(spearman_value) < 0.10:
        return "Weak_or_non_monotonic"

    return "Mixed_or_non_monotonic"


# =============================================================================
# Data preparation
# =============================================================================

def prepare_data() -> pd.DataFrame:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Base table not found:\n{INPUT_CSV}"
        )

    data = pd.read_csv(INPUT_CSV)

    required = {
        "GRID_UID",
        "Country",
        "GRID_ID",
        "Year",
        "Season",
        "Fire_Count",
        "Burned_Pixel_Count",
        "ForestPixelCount",
        *CONTINUOUS_PREDICTORS,
    }

    missing_columns = sorted(required - set(data.columns))

    if missing_columns:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(missing_columns)
        )

    duplicate_panel = data.duplicated(
        ["GRID_UID", "Year", "Season"],
        keep=False,
    )

    if duplicate_panel.any():
        data.loc[duplicate_panel].to_csv(
            OUTPUT_ROOT / "ERROR_Duplicate_Panel_Records.csv",
            index=False,
            encoding="utf-8-sig",
        )
        raise ValueError(
            "Duplicate GRID_UID-Year-Season records were found."
        )

    data["ForestArea_km2"] = data["ForestPixelCount"] * 0.25

    data["FCD"] = np.where(
        data["ForestPixelCount"] > 0,
        data["Fire_Count"] / data["ForestPixelCount"],
        np.nan,
    )

    data["BAD"] = np.where(
        data["ForestPixelCount"] > 0,
        data["Burned_Pixel_Count"] / data["ForestPixelCount"],
        np.nan,
    )

    year_to_fold: Dict[int, int] = {}

    for fold, years in TEMPORAL_FOLDS.items():
        for year in years:
            year_to_fold[int(year)] = int(fold)

    data["Temporal_Fold"] = (
        data["Year"].map(year_to_fold).astype(int)
    )

    data = data.loc[
        data["ForestArea_km2"] > FOREST_AREA_THRESHOLD_KM2
    ].copy()

    country_dummies = pd.get_dummies(
        data["Country"],
        prefix="Country",
        dtype=float,
    )

    for column in COUNTRY_DUMMY_COLUMNS:
        if column not in country_dummies.columns:
            country_dummies[column] = 0.0

    country_dummies = country_dummies[
        COUNTRY_DUMMY_COLUMNS
    ]

    data = pd.concat(
        [
            data.reset_index(drop=True),
            country_dummies.reset_index(drop=True),
        ],
        axis=1,
    )

    incomplete = data[
        MODEL_PREDICTORS + RESPONSES
    ].isna().any(axis=1)

    excluded = data.loc[
        incomplete,
        [
            "GRID_UID",
            "Country",
            "GRID_ID",
            "Year",
            "Season",
            "ForestPixelCount",
            "ForestArea_km2",
            *CONTINUOUS_PREDICTORS,
        ],
    ].copy()

    excluded.to_csv(
        OUTPUT_ROOT / "Excluded_Missing_Records.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return data.loc[~incomplete].copy()


def sample_occurrence_validation(
    validation: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    if len(validation) <= MAX_OCCURRENCE_SHAP_ROWS_PER_FOLD:
        return validation.copy()

    return validation.sample(
        n=MAX_OCCURRENCE_SHAP_ROWS_PER_FOLD,
        random_state=seed,
    ).copy()


def sample_severity_validation(
    validation: pd.DataFrame,
    response: str,
    seed: int,
) -> pd.DataFrame:
    positive = validation.loc[
        validation[response] > 0
    ].copy()

    if len(positive) <= MAX_SEVERITY_SHAP_ROWS_PER_FOLD:
        return positive

    return positive.sample(
        n=MAX_SEVERITY_SHAP_ROWS_PER_FOLD,
        random_state=seed,
    ).copy()


# =============================================================================
# Model fitting
# =============================================================================

def fit_stage_models(
    training: pd.DataFrame,
    response: str,
    config: Dict[str, object],
    seed: int,
) -> Tuple[
    RandomForestClassifier,
    RandomForestRegressor,
]:
    X_train = training[
        MODEL_PREDICTORS
    ].to_numpy(dtype=np.float32)

    y_train = training[
        response
    ].to_numpy(dtype=np.float64)

    occurrence_train = (y_train > 0).astype(np.int8)
    positive_train = y_train > 0

    occurrence_model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_features=config[
            "occurrence_max_features"
        ],
        min_samples_leaf=int(
            config[
                "occurrence_min_samples_leaf"
            ]
        ),
        max_samples=float(
            config[
                "occurrence_max_samples"
            ]
        ),
        bootstrap=True,
        class_weight=None,
        n_jobs=-1,
        random_state=seed + 1,
        criterion="gini",
    )

    occurrence_model.fit(
        X_train,
        occurrence_train,
    )

    severity_model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_features=config[
            "severity_max_features"
        ],
        min_samples_leaf=int(
            config[
                "severity_min_samples_leaf"
            ]
        ),
        max_samples=float(
            config[
                "severity_max_samples"
            ]
        ),
        bootstrap=True,
        n_jobs=-1,
        random_state=seed + 2,
        criterion="squared_error",
    )

    severity_model.fit(
        X_train[positive_train],
        y_train[positive_train],
    )

    return occurrence_model, severity_model


# =============================================================================
# SHAP extraction compatible with multiple SHAP versions
# =============================================================================

def extract_classifier_positive_class_shap(
    model: RandomForestClassifier,
    X: np.ndarray,
) -> Tuple[np.ndarray, float, float]:
    explainer = shap.TreeExplainer(model)

    raw_values = explainer.shap_values(
        X,
        approximate=True,
        check_additivity=False,
    )

    if isinstance(raw_values, list):
        values = np.asarray(raw_values[1], dtype=float)
    else:
        array = np.asarray(raw_values, dtype=float)

        if array.ndim == 3 and array.shape[-1] >= 2:
            values = array[:, :, 1]
        elif array.ndim == 2:
            values = array
        else:
            raise ValueError(
                f"Unexpected classifier SHAP shape: {array.shape}"
            )

    expected_raw = np.asarray(
        explainer.expected_value,
        dtype=float,
    ).reshape(-1)

    expected_value = (
        float(expected_raw[1])
        if len(expected_raw) >= 2
        else float(expected_raw[0])
    )

    predicted_probability = model.predict_proba(X)[:, 1]
    reconstructed = expected_value + values.sum(axis=1)

    additivity_mae = float(
        np.mean(
            np.abs(
                predicted_probability - reconstructed
            )
        )
    )

    return values, expected_value, additivity_mae


def extract_regression_shap(
    model: RandomForestRegressor,
    X: np.ndarray,
) -> Tuple[np.ndarray, float, float]:
    explainer = shap.TreeExplainer(model)

    raw_values = explainer.shap_values(
        X,
        approximate=True,
        check_additivity=False,
    )

    values = np.asarray(raw_values, dtype=float)

    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[:, :, 0]

    if values.ndim != 2:
        raise ValueError(
            f"Unexpected regression SHAP shape: {values.shape}"
        )

    expected_raw = np.asarray(
        explainer.expected_value,
        dtype=float,
    ).reshape(-1)

    expected_value = float(expected_raw[0])

    predicted = model.predict(X)
    reconstructed = expected_value + values.sum(axis=1)

    additivity_mae = float(
        np.mean(
            np.abs(
                predicted - reconstructed
            )
        )
    )

    return values, expected_value, additivity_mae


# =============================================================================
# OOF SHAP calculation
# =============================================================================

def shap_frame(
    sample: pd.DataFrame,
    stage: str,
    response: str,
    season: int,
    fold: int,
    values: np.ndarray,
    expected_value: float,
) -> pd.DataFrame:
    result = sample[
        [
            "GRID_UID",
            "Country",
            "GRID_ID",
            "Year",
            "Season",
            "Temporal_Fold",
            response,
            *MODEL_PREDICTORS,
        ]
    ].copy()

    result = result.rename(
        columns={response: "Observed_Response"}
    )

    result["Stage"] = stage
    result["Response"] = response
    result["Fold"] = fold
    result["SHAP_Expected_Value"] = expected_value

    for index, predictor in enumerate(MODEL_PREDICTORS):
        result[f"SHAP_{predictor}"] = values[:, index]

    return result


def run_oof_shap(
    data: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    occurrence_frames: List[pd.DataFrame] = []
    severity_frames: List[pd.DataFrame] = []
    qa_rows: List[Dict[str, object]] = []

    for season, season_label in SEASONS.items():
        season_data = data.loc[
            data["Season"] == season
        ].copy()

        for response in RESPONSES:
            config = LOCKED_CONFIGS[(season, response)]

            for fold in range(1, 6):
                training = season_data.loc[
                    season_data["Temporal_Fold"] != fold
                ].copy()

                validation = season_data.loc[
                    season_data["Temporal_Fold"] == fold
                ].copy()

                seed = (
                    RANDOM_SEED
                    + season * 100_000
                    + (10_000 if response == "BAD" else 0)
                    + fold * 100
                )

                occurrence_model, severity_model = fit_stage_models(
                    training=training,
                    response=response,
                    config=config,
                    seed=seed,
                )

                occurrence_sample = sample_occurrence_validation(
                    validation=validation,
                    seed=seed + 10,
                )

                severity_sample = sample_severity_validation(
                    validation=validation,
                    response=response,
                    seed=seed + 20,
                )

                X_occurrence = occurrence_sample[
                    MODEL_PREDICTORS
                ].to_numpy(dtype=np.float32)

                log(
                    f"{season_label} {response}, fold {fold}: "
                    f"calculating occurrence approximate SHAP "
                    f"(n={len(occurrence_sample):,})..."
                )

                occurrence_values, occurrence_expected, occurrence_error = (
                    extract_classifier_positive_class_shap(
                        model=occurrence_model,
                        X=X_occurrence,
                    )
                )

                occurrence_frames.append(
                    shap_frame(
                        sample=occurrence_sample,
                        stage="Occurrence",
                        response=response,
                        season=season,
                        fold=fold,
                        values=occurrence_values,
                        expected_value=occurrence_expected,
                    )
                )

                if severity_sample.empty:
                    raise ValueError(
                        f"No positive validation records for "
                        f"{season_label} {response}, fold {fold}."
                    )

                X_severity = severity_sample[
                    MODEL_PREDICTORS
                ].to_numpy(dtype=np.float32)

                log(
                    f"{season_label} {response}, fold {fold}: "
                    f"calculating severity approximate SHAP "
                    f"(n={len(severity_sample):,})..."
                )

                severity_values, severity_expected, severity_error = (
                    extract_regression_shap(
                        model=severity_model,
                        X=X_severity,
                    )
                )

                severity_frames.append(
                    shap_frame(
                        sample=severity_sample,
                        stage="Positive_Severity",
                        response=response,
                        season=season,
                        fold=fold,
                        values=severity_values,
                        expected_value=severity_expected,
                    )
                )

                qa_rows.append(
                    {
                        "Season": season,
                        "Season_Label": season_label,
                        "Response": response,
                        "Fold": fold,
                        "Training_Rows": len(training),
                        "Validation_Rows": len(validation),
                        "Validation_Positive_N": int(
                            (validation[response] > 0).sum()
                        ),
                        "Occurrence_SHAP_Rows": len(
                            occurrence_sample
                        ),
                        "Severity_SHAP_Rows": len(
                            severity_sample
                        ),
                        "Occurrence_Expected_Value":
                            occurrence_expected,
                        "Severity_Expected_Value":
                            severity_expected,
                        "Occurrence_Additivity_MAE":
                            occurrence_error,
                        "Severity_Additivity_MAE":
                            severity_error,
                    }
                )

                log(
                    f"{season_label} {response}, fold {fold}: "
                    f"occurrence SHAP n={len(occurrence_sample):,}; "
                    f"severity SHAP n={len(severity_sample):,}."
                )

    return (
        pd.concat(occurrence_frames, ignore_index=True),
        pd.concat(severity_frames, ignore_index=True),
        pd.DataFrame(qa_rows),
    )


# =============================================================================
# SHAP summaries
# =============================================================================

def make_individual_summary(
    shap_data: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    grouping = [
        "Stage",
        "Season",
        "Response",
    ]

    for group_values, group in shap_data.groupby(
        grouping,
        sort=False,
    ):
        stage, season, response = group_values

        for predictor in MODEL_PREDICTORS:
            feature = group[predictor].to_numpy(dtype=float)
            shap_value = group[
                f"SHAP_{predictor}"
            ].to_numpy(dtype=float)

            if predictor in COUNTRY_DUMMY_COLUMNS:
                low_mask = feature == 0
                high_mask = feature == 1
            else:
                q20 = np.nanquantile(feature, 0.20)
                q80 = np.nanquantile(feature, 0.80)
                low_mask = feature <= q20
                high_mask = feature >= q80

            low_mean = (
                float(np.nanmean(shap_value[low_mask]))
                if low_mask.any()
                else np.nan
            )

            high_mean = (
                float(np.nanmean(shap_value[high_mask]))
                if high_mask.any()
                else np.nan
            )

            difference = high_mean - low_mean
            rho = safe_spearman(feature, shap_value)

            rows.append(
                {
                    "Stage": stage,
                    "Season": season,
                    "Season_Label": SEASONS[int(season)],
                    "Response": response,
                    "Predictor": predictor,
                    "Predictor_Group": predictor_family(predictor),
                    "N": len(group),
                    "Mean_Absolute_SHAP": float(
                        np.mean(np.abs(shap_value))
                    ),
                    "Median_Absolute_SHAP": float(
                        np.median(np.abs(shap_value))
                    ),
                    "Mean_SHAP": float(
                        np.mean(shap_value)
                    ),
                    "SHAP_SD": float(
                        np.std(shap_value, ddof=1)
                    ),
                    "Feature_SHAP_Spearman": rho,
                    "Low_Value_Mean_SHAP": low_mean,
                    "High_Value_Mean_SHAP": high_mean,
                    "High_minus_Low_Mean_SHAP": difference,
                    "Positive_SHAP_Fraction": float(
                        np.mean(shap_value > 0)
                    ),
                    "Direction_Classification": classify_direction(
                        predictor=predictor,
                        spearman_value=rho,
                        high_minus_low_shap=difference,
                    ),
                }
            )

    summary = pd.DataFrame(rows)

    summary["Mean_Absolute_SHAP_Rank"] = (
        summary.groupby(
            ["Stage", "Season", "Response"]
        )["Mean_Absolute_SHAP"]
        .rank(
            ascending=False,
            method="min",
        )
        .astype(int)
    )

    return summary.sort_values(
        [
            "Stage",
            "Season",
            "Response",
            "Mean_Absolute_SHAP_Rank",
        ]
    )


def make_group_summary(
    shap_data: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for (
        stage,
        season,
        response,
    ), group in shap_data.groupby(
        ["Stage", "Season", "Response"],
        sort=False,
    ):
        for group_name, predictors in PREDICTOR_GROUPS.items():
            group_shap = np.zeros(len(group), dtype=float)

            for predictor in predictors:
                group_shap += group[
                    f"SHAP_{predictor}"
                ].to_numpy(dtype=float)

            rows.append(
                {
                    "Stage": stage,
                    "Season": season,
                    "Season_Label": SEASONS[int(season)],
                    "Response": response,
                    "Predictor_Group": group_name,
                    "Group_Columns": ";".join(predictors),
                    "N": len(group),
                    "Mean_Absolute_Group_SHAP": float(
                        np.mean(np.abs(group_shap))
                    ),
                    "Median_Absolute_Group_SHAP": float(
                        np.median(np.abs(group_shap))
                    ),
                    "Mean_Group_SHAP": float(
                        np.mean(group_shap)
                    ),
                    "Group_SHAP_SD": float(
                        np.std(group_shap, ddof=1)
                    ),
                    "Positive_Group_SHAP_Fraction": float(
                        np.mean(group_shap > 0)
                    ),
                }
            )

    summary = pd.DataFrame(rows)

    summary["Mean_Absolute_Group_SHAP_Rank"] = (
        summary.groupby(
            ["Stage", "Season", "Response"]
        )["Mean_Absolute_Group_SHAP"]
        .rank(
            ascending=False,
            method="min",
        )
        .astype(int)
    )

    return summary.sort_values(
        [
            "Stage",
            "Season",
            "Response",
            "Mean_Absolute_Group_SHAP_Rank",
        ]
    )


def make_binned_response_table(
    shap_data: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for (
        stage,
        season,
        response,
    ), group in shap_data.groupby(
        ["Stage", "Season", "Response"],
        sort=False,
    ):
        for predictor in CONTINUOUS_PREDICTORS:
            feature = group[predictor]
            shap_values = group[f"SHAP_{predictor}"]

            try:
                bins = pd.qcut(
                    feature,
                    q=BIN_COUNT,
                    duplicates="drop",
                )
            except ValueError:
                continue

            temporary = pd.DataFrame(
                {
                    "Feature": feature.to_numpy(),
                    "SHAP": shap_values.to_numpy(),
                    "Bin": bins,
                }
            )

            for bin_index, (_, bin_data) in enumerate(
                temporary.groupby(
                    "Bin",
                    observed=True,
                    sort=True,
                ),
                start=1,
            ):
                rows.append(
                    {
                        "Stage": stage,
                        "Season": season,
                        "Season_Label": SEASONS[int(season)],
                        "Response": response,
                        "Predictor": predictor,
                        "Bin_Order": bin_index,
                        "N": len(bin_data),
                        "Feature_Min": float(
                            bin_data["Feature"].min()
                        ),
                        "Feature_Median": float(
                            bin_data["Feature"].median()
                        ),
                        "Feature_Mean": float(
                            bin_data["Feature"].mean()
                        ),
                        "Feature_Max": float(
                            bin_data["Feature"].max()
                        ),
                        "Mean_SHAP": float(
                            bin_data["SHAP"].mean()
                        ),
                        "Median_SHAP": float(
                            bin_data["SHAP"].median()
                        ),
                        "SHAP_Q25": float(
                            bin_data["SHAP"].quantile(0.25)
                        ),
                        "SHAP_Q75": float(
                            bin_data["SHAP"].quantile(0.75)
                        ),
                    }
                )

    return pd.DataFrame(rows)


def make_country_context_summary(
    shap_data: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for (
        stage,
        season,
        response,
        country,
    ), group in shap_data.groupby(
        ["Stage", "Season", "Response", "Country"],
        sort=False,
    ):
        country_group_shap = np.zeros(len(group), dtype=float)

        for predictor in COUNTRY_DUMMY_COLUMNS:
            country_group_shap += group[
                f"SHAP_{predictor}"
            ].to_numpy(dtype=float)

        row = {
            "Stage": stage,
            "Season": season,
            "Season_Label": SEASONS[int(season)],
            "Response": response,
            "Country": country,
            "N": len(group),
            "Mean_Country_Context_SHAP": float(
                country_group_shap.mean()
            ),
            "Mean_Absolute_Country_Context_SHAP": float(
                np.mean(np.abs(country_group_shap))
            ),
        }

        for predictor in COUNTRY_DUMMY_COLUMNS:
            row[f"Mean_SHAP_{predictor}"] = float(
                group[f"SHAP_{predictor}"].mean()
            )

        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# Figures
# =============================================================================

def make_bar_figures(
    individual_summary: pd.DataFrame,
) -> None:
    figure_dir = OUTPUT_ROOT / "Figures" / "Importance_Bars"
    figure_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 10

    for (
        stage,
        season,
        response,
    ), group in individual_summary.groupby(
        ["Stage", "Season", "Response"],
        sort=False,
    ):
        top = (
            group.sort_values(
                "Mean_Absolute_SHAP",
                ascending=False,
            )
            .head(TOP_FEATURES_IN_BAR_FIGURE)
            .sort_values(
                "Mean_Absolute_SHAP",
                ascending=True,
            )
        )

        fig, ax = plt.subplots(
            figsize=(7.2, 5.4)
        )

        ax.barh(
            top["Predictor"],
            top["Mean_Absolute_SHAP"],
        )

        ax.set_xlabel("Mean absolute SHAP value")
        ax.set_ylabel("")
        ax.set_title(
            f"{SEASONS[int(season)]} {response} — "
            f"{stage.replace('_', ' ')}"
        )

        fig.tight_layout()

        stem = (
            f"SHAP_Bar_S{int(season)}_{response}_{stage}"
        )

        fig.savefig(
            figure_dir / f"{stem}.png",
            dpi=300,
            bbox_inches="tight",
        )

        fig.savefig(
            figure_dir / f"{stem}.pdf",
            bbox_inches="tight",
        )

        plt.close(fig)


def make_dependence_figures(
    shap_data: pd.DataFrame,
    individual_summary: pd.DataFrame,
    binned_response: pd.DataFrame,
) -> None:
    """Create SHAP dependence figures for the leading continuous predictors.

    Each panel shows held-out sample-level SHAP values together with decile-bin
    mean SHAP values and the interquartile range. These figures visualize the
    direction and nonlinearity of model associations. They are not interpreted
    as causal response curves or exact ecological thresholds.
    """
    figure_dir = OUTPUT_ROOT / "Figures" / "Nonlinear_Response"
    figure_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 10

    for (
        stage,
        season,
        response,
    ), summary_group in individual_summary.groupby(
        ["Stage", "Season", "Response"],
        sort=False,
    ):
        top_predictors = (
            summary_group[
                summary_group["Predictor"].isin(CONTINUOUS_PREDICTORS)
            ]
            .sort_values("Mean_Absolute_SHAP", ascending=False)
            .head(TOP_CONTINUOUS_FEATURES_IN_DEPENDENCE_FIGURE)["Predictor"]
            .tolist()
        )

        if not top_predictors:
            continue

        sample_group = shap_data[
            (shap_data["Stage"] == stage)
            & (shap_data["Season"] == season)
            & (shap_data["Response"] == response)
        ]

        bin_group = binned_response[
            (binned_response["Stage"] == stage)
            & (binned_response["Season"] == season)
            & (binned_response["Response"] == response)
        ]

        fig, axes = plt.subplots(
            2,
            2,
            figsize=(10.0, 7.6),
        )
        axes = np.asarray(axes).ravel()

        for ax, predictor in zip(axes, top_predictors):
            x = sample_group[predictor].to_numpy(dtype=float)
            y = sample_group[f"SHAP_{predictor}"].to_numpy(dtype=float)

            finite = np.isfinite(x) & np.isfinite(y)
            ax.scatter(
                x[finite],
                y[finite],
                s=9,
                alpha=0.16,
                linewidths=0,
                rasterized=True,
            )

            curve = (
                bin_group[bin_group["Predictor"] == predictor]
                .sort_values("Bin_Order")
            )

            if not curve.empty:
                curve_x = curve["Feature_Median"].to_numpy(dtype=float)
                curve_y = curve["Mean_SHAP"].to_numpy(dtype=float)
                curve_q25 = curve["SHAP_Q25"].to_numpy(dtype=float)
                curve_q75 = curve["SHAP_Q75"].to_numpy(dtype=float)

                ax.fill_between(
                    curve_x,
                    curve_q25,
                    curve_q75,
                    alpha=0.18,
                    linewidth=0,
                )
                ax.plot(
                    curve_x,
                    curve_y,
                    marker="o",
                    markersize=3.5,
                    linewidth=1.5,
                )

            ax.axhline(0.0, linewidth=0.8, linestyle="--")
            ax.set_title(predictor)
            ax.set_xlabel(predictor)
            ax.set_ylabel("SHAP value")

        for ax in axes[len(top_predictors):]:
            ax.set_visible(False)

        fig.suptitle(
            f"{SEASONS[int(season)]} {response} — "
            f"{stage.replace('_', ' ')}",
            y=0.995,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.97))

        stem = (
            f"SHAP_Dependence_S{int(season)}_{response}_{stage}"
        )

        fig.savefig(
            figure_dir / f"{stem}.png",
            dpi=300,
            bbox_inches="tight",
        )
        fig.savefig(
            figure_dir / f"{stem}.pdf",
            bbox_inches="tight",
        )
        plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOG_FILE.write_text(
        "",
        encoding="utf-8",
    )

    log("=" * 78)
    log("Starting Step 11 temporal-OOF SHAP direction analysis")
    log(f"Step 11 code version: {STEP_CODE_VERSION}")
    log(f"Input: {INPUT_CSV}")
    log(
        "Forest threshold: ForestArea_km2 > 2.5 km2 "
        "(equivalent to ForestPixelCount > 10)"
    )
    log("Main model: Drivers_Only with Country context")
    log(f"Trees per RF: {N_ESTIMATORS}")
    log(f"SHAP version: {shap.__version__}")
    log("=" * 78)

    data = prepare_data()

    occurrence_shap, severity_shap, shap_qa = run_oof_shap(
        data
    )

    all_shap = pd.concat(
        [
            occurrence_shap,
            severity_shap,
        ],
        ignore_index=True,
    )

    individual_summary = make_individual_summary(
        all_shap
    )

    group_summary = make_group_summary(
        all_shap
    )

    binned_response = make_binned_response_table(
        all_shap
    )

    country_summary = make_country_context_summary(
        all_shap
    )

    target_rows: List[Dict[str, object]] = []

    for predictor in MODEL_PREDICTORS:
        target_rows.append(
            {
                "Target_Type": "Individual",
                "Target_Name": predictor,
                "Columns": predictor,
                "Interpret_as_direct_driver": (
                    predictor not in COUNTRY_DUMMY_COLUMNS
                ),
            }
        )

    for group_name, columns in PREDICTOR_GROUPS.items():
        target_rows.append(
            {
                "Target_Type": "Group",
                "Target_Name": group_name,
                "Columns": ";".join(columns),
                "Interpret_as_direct_driver": (
                    group_name != "Country_Context"
                ),
            }
        )

    pd.DataFrame(target_rows).to_csv(
        OUTPUT_ROOT / "00_SHAP_Target_Definitions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    shap_qa.to_csv(
        OUTPUT_ROOT / "01_SHAP_Fold_QA.csv",
        index=False,
        encoding="utf-8-sig",
    )

    individual_summary.to_csv(
        OUTPUT_ROOT / "02_Individual_SHAP_Summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    group_summary.to_csv(
        OUTPUT_ROOT / "03_Group_SHAP_Summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    binned_response.to_csv(
        OUTPUT_ROOT / "04_Binned_SHAP_Response.csv",
        index=False,
        encoding="utf-8-sig",
    )

    country_summary.to_csv(
        OUTPUT_ROOT / "05_Country_Context_SHAP_Summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Detailed sample-level SHAP values are retained locally but omitted
    # from the upload ZIP because they may be large.
    occurrence_shap.to_csv(
        OUTPUT_ROOT / "Occurrence_OOF_SHAP_Values.csv.gz",
        index=False,
        compression="gzip",
    )

    severity_shap.to_csv(
        OUTPUT_ROOT / "Positive_Severity_OOF_SHAP_Values.csv.gz",
        index=False,
        compression="gzip",
    )

    make_bar_figures(
        individual_summary
    )

    make_dependence_figures(
        all_shap,
        individual_summary,
        binned_response,
    )

    qa = pd.DataFrame(
        [
            [
                "ForestArea_threshold_rule",
                "ForestArea_km2 > 2.5",
            ],
            [
                "Equivalent_forest_area_rule",
                "ForestArea_km2 > 2.5",
            ],
            ["Rows_used", len(data)],
            [
                "Unique_GRID_UIDs_used",
                data["GRID_UID"].nunique(),
            ],
            [
                "Continuous_driver_count",
                len(CONTINUOUS_PREDICTORS),
            ],
            [
                "Country_context_dummy_count",
                len(COUNTRY_DUMMY_COLUMNS),
            ],
            ["Trees_per_RF", N_ESTIMATORS],
            [
                "Occurrence_max_SHAP_rows_per_fold",
                MAX_OCCURRENCE_SHAP_ROWS_PER_FOLD,
            ],
            [
                "Severity_max_SHAP_rows_per_fold",
                MAX_SEVERITY_SHAP_ROWS_PER_FOLD,
            ],
            [
                "Dependence_figure_predictor_count",
                TOP_CONTINUOUS_FEATURES_IN_DEPENDENCE_FIGURE,
            ],
            [
                "Dependence_figure_content",
                "Held-out SHAP scatter plus decile mean and interquartile range",
            ],
            [
                "SHAP_interpretation",
                (
                    "Occurrence SHAP explains the positive-class "
                    "probability contribution. Severity SHAP explains "
                    "conditional positive response. Country is a "
                    "context control, not a causal driver."
                ),
            ],
        ],
        columns=["Check", "Value"],
    )

    qa.to_csv(
        OUTPUT_ROOT / "06_Data_QA_Summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    environment = {
        "Step": STEP_ID,
        "Step_code_version": STEP_CODE_VERSION,
        "Output_directory": str(OUTPUT_ROOT),
        "Created": datetime.now().isoformat(
            timespec="seconds"
        ),
        "Python": sys.version,
        "Platform": platform.platform(),
        "NumPy": np.__version__,
        "pandas": pd.__version__,
        "SciPy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "SHAP": shap.__version__,
        "Matplotlib": plt.matplotlib.__version__,
        "Input_file": str(INPUT_CSV),
        "Forest_threshold": {
            "rule": "ForestArea_km2 > 2.5",
            "unit": "km2",
            "equivalent_area": "> 2.5 km2",
        },
        "Model": (
            "Locked two-stage hurdle random forest with Country "
            "dummy variables retained as contextual controls."
        ),
        "SHAP_design": {
            "calculation_mode": "Approximate TreeSHAP for rapid directional sensitivity analysis",
            "validation": (
                "Five-fold grouped temporal out-of-fold explanation"
            ),
            "occurrence_sample": (
                f"Representative random sample up to "
                f"{MAX_OCCURRENCE_SHAP_ROWS_PER_FOLD:,} rows per fold"
            ),
            "severity_sample": (
                f"Actual positive validation records, up to "
                f"{MAX_SEVERITY_SHAP_ROWS_PER_FOLD:,} rows per fold"
            ),
            "classifier_output": (
                "Positive-class model output as returned by "
                "TreeExplainer for the fitted sklearn RF classifier"
            ),
            "severity_output": (
                "Conditional positive FCD or BAD prediction"
            ),
        },
        "Interpretation_limits": [
            (
                "Approximate SHAP describes model associations and does not establish "
                "causal effects."
            ),
            (
                "Attribution among correlated predictors can be shared or "
                "redistributed; grouped permutation importance remains primary."
            ),
            (
                "Country SHAP values represent contextual differences and "
                "must not be interpreted as environmental mechanisms."
            ),
            (
                "The expected hurdle response is a product of occurrence "
                "probability and conditional severity, so stage-specific SHAP "
                "is reported rather than an invalid simple sum across stages."
            ),
        ],
    }

    with (
        OUTPUT_ROOT / "Software_Environment_and_Method.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            environment,
            file,
            ensure_ascii=False,
            indent=2,
        )




    log("=" * 78)
    log("Step 11 OOF SHAP direction analysis completed successfully.")
    log("=" * 78)

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
