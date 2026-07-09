# -*- coding: utf-8 -*-

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
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


# =============================================================================
# Paths and settings
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_ROOT = SCRIPT_DIR.parent
CODE_ROOT = MODEL_ROOT.parent
INPUT_CSV = CODE_ROOT / "00_Input_data" / "Base_Table_2001_2025.csv.bz2"

OUTPUT_ROOT = MODEL_ROOT / "08_Final_Permutation_Importance"
LOG_FILE = OUTPUT_ROOT / "final_permutation_importance.log"

# ForestPixelCount is the number of 500 m x 500 m forest pixels.
# ForestArea_km2 is ForestPixelCount * 0.25; models use ForestArea_km2 > 2.5.
FOREST_AREA_THRESHOLD_KM2 = 2.5
RANDOM_SEED = 2026
N_ESTIMATORS = 300
PERMUTATION_REPEATS = 3
MAX_VALIDATION_ROWS_FOR_PERMUTATION = 15_000

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

COUNTRY_DUMMY_COLUMNS = [
    "Country_China",
    "Country_NK",
    "Country_Russia",
]

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
    "Country": COUNTRY_DUMMY_COLUMNS,
}

# Locked after the previous tuning step.
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
# Metric helpers
# =============================================================================

def safe_spearman(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> float:
    if len(observed) < 3:
        return np.nan

    if np.std(observed) == 0 or np.std(predicted) == 0:
        return np.nan

    return float(
        spearmanr(
            observed,
            predicted,
            nan_policy="omit",
        ).statistic
    )


def safe_roc_auc(
    labels: np.ndarray,
    scores: np.ndarray,
) -> float:
    if len(np.unique(labels)) < 2:
        return np.nan

    return float(
        roc_auc_score(labels, scores)
    )


def safe_pr_auc(
    labels: np.ndarray,
    scores: np.ndarray,
) -> float:
    if len(np.unique(labels)) < 2:
        return np.nan

    return float(
        average_precision_score(
            labels,
            scores,
        )
    )


def occurrence_metric_set(
    observed_occurrence: np.ndarray,
    probability: np.ndarray,
) -> Dict[str, float]:
    probability = np.clip(
        probability,
        1e-6,
        1 - 1e-6,
    )

    return {
        "ROC_AUC": safe_roc_auc(
            observed_occurrence,
            probability,
        ),
        "PR_AUC": safe_pr_auc(
            observed_occurrence,
            probability,
        ),
        "Brier": float(
            brier_score_loss(
                observed_occurrence,
                probability,
            )
        ),
    }


def severity_metric_set(
    observed_positive: np.ndarray,
    conditional_prediction: np.ndarray,
) -> Dict[str, float]:
    conditional_prediction = np.clip(
        conditional_prediction,
        0.0,
        None,
    )

    return {
        "RMSE": float(
            math.sqrt(
                mean_squared_error(
                    observed_positive,
                    conditional_prediction,
                )
            )
        ),
        "MAE": float(
            mean_absolute_error(
                observed_positive,
                conditional_prediction,
            )
        ),
        "Spearman": safe_spearman(
            observed_positive,
            conditional_prediction,
        ),
    }


def combined_metric_set(
    observed: np.ndarray,
    expected_prediction: np.ndarray,
) -> Dict[str, float]:
    expected_prediction = np.clip(
        expected_prediction,
        0.0,
        None,
    )

    return {
        "RMSE": float(
            math.sqrt(
                mean_squared_error(
                    observed,
                    expected_prediction,
                )
            )
        ),
        "MAE": float(
            mean_absolute_error(
                observed,
                expected_prediction,
            )
        ),
        "R2": float(
            r2_score(
                observed,
                expected_prediction,
            )
        ),
        "Spearman": safe_spearman(
            observed,
            expected_prediction,
        ),
    }


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

    missing_columns = sorted(
        required - set(data.columns)
    )

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
        )
        raise ValueError(
            "Duplicate GRID_UID-Year-Season records were found."
        )

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
            *CONTINUOUS_PREDICTORS,
        ],
    ].copy()

    excluded.to_csv(
        OUTPUT_ROOT / "Excluded_Missing_Records.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return data.loc[
        ~incomplete
    ].copy()


def select_permutation_validation_sample(
    validation: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    if len(validation) <= MAX_VALIDATION_ROWS_FOR_PERMUTATION:
        return validation.copy()

    return validation.sample(
        n=MAX_VALIDATION_ROWS_FOR_PERMUTATION,
        random_state=seed,
    ).copy()


# =============================================================================
# Model fitting
# =============================================================================

def fit_models(
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

    occurrence_train = (
        y_train > 0
    ).astype(np.int8)

    positive_train = y_train > 0

    if positive_train.sum() < 20:
        raise ValueError(
            "Too few positive training observations."
        )

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
# Permutation importance
# =============================================================================

def permute_columns(
    matrix: np.ndarray,
    column_indices: Sequence[int],
    rng: np.random.Generator,
) -> np.ndarray:
    permuted = matrix.copy()
    row_order = rng.permutation(
        len(permuted)
    )

    # The same row permutation is applied to every column in a group,
    # preserving the group's internal covariance structure.
    permuted[
        :,
        list(column_indices),
    ] = permuted[
        row_order,
        :,
    ][
        :,
        list(column_indices),
    ]

    return permuted


def importance_rows_for_target(
    *,
    season: int,
    response: str,
    fold: int,
    target_type: str,
    target_name: str,
    target_columns: Sequence[str],
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    occurrence_model: RandomForestClassifier,
    severity_model: RandomForestRegressor,
    baseline_occurrence: Dict[str, float],
    baseline_severity: Dict[str, float],
    baseline_combined: Dict[str, float],
    repeat_count: int,
    seed: int,
) -> List[Dict[str, object]]:
    column_indices = [
        MODEL_PREDICTORS.index(column)
        for column in target_columns
    ]

    positive = y_validation > 0
    observed_occurrence = (
        y_validation > 0
    ).astype(np.int8)

    rows: List[Dict[str, object]] = []

    for repeat in range(1, repeat_count + 1):
        rng = np.random.default_rng(
            seed + repeat
        )

        X_permuted = permute_columns(
            matrix=X_validation,
            column_indices=column_indices,
            rng=rng,
        )

        probability = (
            occurrence_model.predict_proba(
                X_permuted
            )[:, 1]
        )

        conditional = np.clip(
            severity_model.predict(
                X_permuted
            ),
            0.0,
            None,
        )

        expected = probability * conditional

        occurrence_permuted = occurrence_metric_set(
            observed_occurrence,
            probability,
        )

        combined_permuted = combined_metric_set(
            y_validation,
            expected,
        )

        if positive.any():
            severity_permuted = severity_metric_set(
                y_validation[positive],
                conditional[positive],
            )
        else:
            severity_permuted = {
                "RMSE": np.nan,
                "MAE": np.nan,
                "Spearman": np.nan,
            }

        rows.append(
            {
                "Season": season,
                "Season_Label": SEASONS[season],
                "Response": response,
                "Fold": fold,
                "Target_Type": target_type,
                "Target_Name": target_name,
                "Target_Columns": ";".join(
                    target_columns
                ),
                "Repeat": repeat,

                # Positive values mean performance worsened after permutation.
                "Occurrence_Delta_PR_AUC": (
                    baseline_occurrence["PR_AUC"]
                    - occurrence_permuted["PR_AUC"]
                ),
                "Occurrence_Delta_ROC_AUC": (
                    baseline_occurrence["ROC_AUC"]
                    - occurrence_permuted["ROC_AUC"]
                ),
                "Occurrence_Increase_Brier": (
                    occurrence_permuted["Brier"]
                    - baseline_occurrence["Brier"]
                ),

                "Severity_Increase_RMSE": (
                    severity_permuted["RMSE"]
                    - baseline_severity["RMSE"]
                ),
                "Severity_Increase_MAE": (
                    severity_permuted["MAE"]
                    - baseline_severity["MAE"]
                ),
                "Severity_Decrease_Spearman": (
                    baseline_severity["Spearman"]
                    - severity_permuted["Spearman"]
                ),

                "Combined_Increase_RMSE": (
                    combined_permuted["RMSE"]
                    - baseline_combined["RMSE"]
                ),
                "Combined_Increase_MAE": (
                    combined_permuted["MAE"]
                    - baseline_combined["MAE"]
                ),
                "Combined_Decrease_R2": (
                    baseline_combined["R2"]
                    - combined_permuted["R2"]
                ),
                "Combined_Decrease_Spearman": (
                    baseline_combined["Spearman"]
                    - combined_permuted["Spearman"]
                ),
            }
        )

    return rows


def run_permutation_importance(
    data: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    baseline_rows: List[Dict[str, object]] = []
    importance_rows: List[Dict[str, object]] = []

    for season, season_label in SEASONS.items():
        season_data = data.loc[
            data["Season"] == season
        ].copy()

        for response in RESPONSES:
            config = LOCKED_CONFIGS[
                (season, response)
            ]

            for fold in range(1, 6):
                training = season_data.loc[
                    season_data[
                        "Temporal_Fold"
                    ] != fold
                ].copy()

                validation_full = season_data.loc[
                    season_data[
                        "Temporal_Fold"
                    ] == fold
                ].copy()

                sample_seed = (
                    RANDOM_SEED
                    + season * 100_000
                    + (
                        10_000
                        if response == "BAD"
                        else 0
                    )
                    + fold * 100
                )

                validation = (
                    select_permutation_validation_sample(
                        validation=validation_full,
                        seed=sample_seed,
                    )
                )

                (
                    occurrence_model,
                    severity_model,
                ) = fit_models(
                    training=training,
                    response=response,
                    config=config,
                    seed=sample_seed,
                )

                X_validation = validation[
                    MODEL_PREDICTORS
                ].to_numpy(dtype=np.float32)

                y_validation = validation[
                    response
                ].to_numpy(dtype=np.float64)

                occurrence_probability = (
                    occurrence_model.predict_proba(
                        X_validation
                    )[:, 1]
                )

                conditional_severity = np.clip(
                    severity_model.predict(
                        X_validation
                    ),
                    0.0,
                    None,
                )

                expected_prediction = (
                    occurrence_probability
                    * conditional_severity
                )

                observed_occurrence = (
                    y_validation > 0
                ).astype(np.int8)

                positive = y_validation > 0

                baseline_occurrence = (
                    occurrence_metric_set(
                        observed_occurrence,
                        occurrence_probability,
                    )
                )

                if positive.any():
                    baseline_severity = (
                        severity_metric_set(
                            y_validation[positive],
                            conditional_severity[
                                positive
                            ],
                        )
                    )
                else:
                    baseline_severity = {
                        "RMSE": np.nan,
                        "MAE": np.nan,
                        "Spearman": np.nan,
                    }

                baseline_combined = (
                    combined_metric_set(
                        y_validation,
                        expected_prediction,
                    )
                )

                baseline_rows.append(
                    {
                        "Season": season,
                        "Season_Label":
                            season_label,
                        "Response": response,
                        "Fold": fold,
                        "Validation_Rows_Full":
                            len(validation_full),
                        "Validation_Rows_Used":
                            len(validation),
                        "Validation_Positive_N":
                            int(positive.sum()),
                        "Occurrence_Config":
                            config[
                                "occurrence_name"
                            ],
                        "Severity_Config":
                            config[
                                "severity_name"
                            ],
                        **{
                            f"Occurrence_{key}":
                                value
                            for key, value
                            in baseline_occurrence.items()
                        },
                        **{
                            f"Severity_{key}":
                                value
                            for key, value
                            in baseline_severity.items()
                        },
                        **{
                            f"Combined_{key}":
                                value
                            for key, value
                            in baseline_combined.items()
                        },
                    }
                )

                for predictor_index, predictor in enumerate(
                    CONTINUOUS_PREDICTORS
                ):
                    importance_rows.extend(
                        importance_rows_for_target(
                            season=season,
                            response=response,
                            fold=fold,
                            target_type="Individual",
                            target_name=predictor,
                            target_columns=[predictor],
                            X_validation=X_validation,
                            y_validation=y_validation,
                            occurrence_model=
                                occurrence_model,
                            severity_model=
                                severity_model,
                            baseline_occurrence=
                                baseline_occurrence,
                            baseline_severity=
                                baseline_severity,
                            baseline_combined=
                                baseline_combined,
                            repeat_count=
                                PERMUTATION_REPEATS,
                            seed=(
                                sample_seed
                                + 1_000
                                + predictor_index
                                * 100
                            ),
                        )
                    )

                for group_index, (
                    group_name,
                    group_columns,
                ) in enumerate(
                    PREDICTOR_GROUPS.items()
                ):
                    importance_rows.extend(
                        importance_rows_for_target(
                            season=season,
                            response=response,
                            fold=fold,
                            target_type="Group",
                            target_name=group_name,
                            target_columns=group_columns,
                            X_validation=X_validation,
                            y_validation=y_validation,
                            occurrence_model=
                                occurrence_model,
                            severity_model=
                                severity_model,
                            baseline_occurrence=
                                baseline_occurrence,
                            baseline_severity=
                                baseline_severity,
                            baseline_combined=
                                baseline_combined,
                            repeat_count=
                                PERMUTATION_REPEATS,
                            seed=(
                                sample_seed
                                + 100_000
                                + group_index
                                * 1_000
                            ),
                        )
                    )

                log(
                    f"{season_label} {response}, "
                    f"fold {fold} completed."
                )

    return (
        pd.DataFrame(baseline_rows),
        pd.DataFrame(importance_rows),
    )


# =============================================================================
# Summaries and ranking
# =============================================================================

def summarize_importance(
    importance: pd.DataFrame,
) -> pd.DataFrame:
    metric_columns = [
        column
        for column in importance.columns
        if column.startswith(
            (
                "Occurrence_",
                "Severity_",
                "Combined_",
            )
        )
    ]

    grouping = [
        "Season",
        "Season_Label",
        "Response",
        "Target_Type",
        "Target_Name",
        "Target_Columns",
    ]

    rows: List[Dict[str, object]] = []

    for group_values, group in importance.groupby(
        grouping,
        sort=False,
    ):
        row = dict(
            zip(
                grouping,
                group_values,
            )
        )

        row["Fold_Repeat_N"] = len(group)

        for metric in metric_columns:
            row[f"{metric}_Mean"] = float(
                group[metric].mean()
            )
            row[f"{metric}_SD"] = float(
                group[metric].std(ddof=1)
            )
            row[f"{metric}_Positive_Fraction"] = float(
                (group[metric] > 0).mean()
            )

        rows.append(row)

    summary = pd.DataFrame(rows)

    summary["Occurrence_PR_AUC_Rank"] = (
        summary.groupby(
            [
                "Season",
                "Response",
                "Target_Type",
            ]
        )[
            "Occurrence_Delta_PR_AUC_Mean"
        ]
        .rank(
            ascending=False,
            method="min",
        )
        .astype(int)
    )

    summary["Severity_RMSE_Rank"] = (
        summary.groupby(
            [
                "Season",
                "Response",
                "Target_Type",
            ]
        )[
            "Severity_Increase_RMSE_Mean"
        ]
        .rank(
            ascending=False,
            method="min",
        )
        .astype(int)
    )

    summary["Combined_RMSE_Rank"] = (
        summary.groupby(
            [
                "Season",
                "Response",
                "Target_Type",
            ]
        )[
            "Combined_Increase_RMSE_Mean"
        ]
        .rank(
            ascending=False,
            method="min",
        )
        .astype(int)
    )

    return summary.sort_values(
        [
            "Target_Type",
            "Season",
            "Response",
            "Combined_RMSE_Rank",
        ]
    )


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
    log("Starting final temporal-OOF permutation importance")
    log(f"Input: {INPUT_CSV}")
    log(
        "Forest threshold: ForestArea_km2 > 2.5 km2 "
        "(equivalent to > 2.5 km2)"
    )
    log(f"Trees per RF: {N_ESTIMATORS}")
    log(
        f"Permutation repeats: {PERMUTATION_REPEATS}"
    )
    log(
        "Main model: Drivers_Only; spatial controls are "
        "not included in driver importance."
    )
    log("=" * 78)

    data = prepare_data()

    baseline, importance = (
        run_permutation_importance(data)
    )

    baseline.to_csv(
        OUTPUT_ROOT
        / "01_Baseline_Fold_Metrics_on_Importance_Sample.csv",
        index=False,
        encoding="utf-8-sig",
    )

    importance.to_csv(
        OUTPUT_ROOT
        / "02_Permutation_Importance_By_Fold_and_Repeat.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = summarize_importance(
        importance
    )

    summary.to_csv(
        OUTPUT_ROOT
        / "03_Permutation_Importance_Summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary.loc[
        summary["Target_Type"] == "Individual"
    ].to_csv(
        OUTPUT_ROOT
        / "04_Individual_Permutation_Importance_Summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary.loc[
        summary["Target_Type"] == "Group"
    ].to_csv(
        OUTPUT_ROOT
        / "05_Grouped_Permutation_Importance_Summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    predictor_definition_rows: List[Dict[str, object]] = []

    for predictor in CONTINUOUS_PREDICTORS:
        predictor_definition_rows.append(
            {
                "Type": "Individual",
                "Name": predictor,
                "Columns": predictor,
                "Included_in_driver_interpretation": True,
            }
        )

    for group_name, columns in PREDICTOR_GROUPS.items():
        predictor_definition_rows.append(
            {
                "Type": "Group",
                "Name": group_name,
                "Columns": ";".join(columns),
                "Included_in_driver_interpretation": True,
            }
        )

    pd.DataFrame(
        predictor_definition_rows
    ).to_csv(
        OUTPUT_ROOT
        / "00_Permutation_Target_Definitions.csv",
        index=False,
        encoding="utf-8-sig",
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
            [
                "Rows_used",
                len(data),
            ],
            [
                "Unique_GRID_UIDs_used",
                data["GRID_UID"].nunique(),
            ],
            [
                "Continuous_predictor_count",
                len(CONTINUOUS_PREDICTORS),
            ],
            [
                "Country_dummy_count",
                len(COUNTRY_DUMMY_COLUMNS),
            ],
            [
                "Individual_targets",
                len(CONTINUOUS_PREDICTORS),
            ],
            [
                "Grouped_targets",
                len(PREDICTOR_GROUPS),
            ],
            [
                "Permutation_repeats",
                PERMUTATION_REPEATS,
            ],
            [
                "Maximum_validation_rows_per_fold",
                MAX_VALIDATION_ROWS_FOR_PERMUTATION,
            ],
            [
                "Trees_per_RF",
                N_ESTIMATORS,
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
        "Created": datetime.now().isoformat(
            timespec="seconds"
        ),
        "Python": sys.version,
        "Platform": platform.platform(),
        "NumPy": np.__version__,
        "pandas": pd.__version__,
        "SciPy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "Input_file": str(INPUT_CSV),
        "Forest_threshold": {
            "rule": "ForestArea_km2 > 2.5",
            "unit": "km2",
            "equivalent_area": "> 2.5 km2",
        },
        "Model": (
            "Locked Drivers_Only two-stage hurdle random forest"
        ),
        "Validation": (
            "Five-fold grouped temporal cross-validation"
        ),
        "Trees_per_RF": N_ESTIMATORS,
        "Permutation_repeats": PERMUTATION_REPEATS,
        "Permutation_sample": (
            "A reproducible random sample of at most "
            f"{MAX_VALIDATION_ROWS_FOR_PERMUTATION:,} validation "
            "records per fold. The same sample is used for baseline "
            "and all permutations within a fold."
        ),
        "Group_permutation": (
            "All columns in a predictor group use the same row "
            "permutation, preserving within-group relationships."
        ),
        "Interpretation": (
            "Positive importance values indicate deterioration after "
            "permutation. Group importance is primary for correlated "
            "families; individual importance is supplementary."
        ),
        "Spatial_controls": (
            "Excluded from driver importance. Drivers_Only remains "
            "the main interpretable model; XY/RBF results are reported "
            "as spatial-correction sensitivity only."
        ),
    }

    with (
        OUTPUT_ROOT
        / "Software_Environment_and_Method.json"
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
    log("Final permutation importance completed successfully.")
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
