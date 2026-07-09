# -*- coding: utf-8 -*-

# Public reproducibility version.
# Input paths are resolved relative to the code repository.
# ForestPixelCount stores the original pixel count; ForestArea_km2 stores physical area.


from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime
import math
import platform
import sys
import traceback
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import spearmanr
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
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

OUTPUT_ROOT = MODEL_ROOT / "01_RF_Hurdle_Comparison"

FOREST_AREA_THRESHOLD_KM2 = 2.5
RANDOM_SEED = 2026

# Same exploratory predictor set used in the sampling comparison.
# Coordinates are retained in the base table for later spatial validation,
# but they are deliberately excluded from model predictors.
PREDICTORS = [
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
    "FFMC",
    "DMC",
    "DC",
    "SPEI1",
    "SPEI3",
    "SPEI6",
    "SPEI12",
    "SPEI24",
    "DEM",
    "Slope",
    "Aspect",
    "POP",
    "Dis_Farm",
    "Dis_Build",
    "Road_dens",
    "Dis_Railway",
    "Dis_Power",
]

COUNTRY_DUMMY_COLUMNS = [
    "Country_China",
    "Country_NK",
    "Country_Russia",
]

COORDINATE_COLUMNS = [
    "Centroid_X_UTM52_m",
    "Centroid_Y_UTM52_m",
    "Longitude",
    "Latitude",
]

# Fixed temporal folds already used in the sampling exploration.
TEMPORAL_FOLDS = {
    1: [2024, 2002, 2006, 2004, 2009],
    2: [2016, 2015, 2007, 2018, 2014],
    3: [2025, 2021, 2013, 2019, 2003],
    4: [2010, 2023, 2017, 2011, 2005],
    5: [2020, 2022, 2001, 2012, 2008],
}


# =============================================================================
# Logging
# =============================================================================

def make_logger(log_path: Path):
    def log(message: str) -> None:
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as file:
            file.write(message + "\n")
    return log


# =============================================================================
# Utility functions
# =============================================================================

def safe_spearman(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> float:
    if len(observed) < 3:
        return np.nan

    if (
        np.nanstd(observed) == 0
        or np.nanstd(predicted) == 0
    ):
        return np.nan

    return float(
        spearmanr(
            observed,
            predicted,
            nan_policy="omit",
        ).statistic
    )


def safe_auc(
    labels: np.ndarray,
    scores: np.ndarray,
) -> float:
    if len(np.unique(labels)) < 2:
        return np.nan

    return float(
        roc_auc_score(labels, scores)
    )


def safe_average_precision(
    labels: np.ndarray,
    scores: np.ndarray,
) -> float:
    if len(np.unique(labels)) < 2:
        return np.nan

    return float(
        average_precision_score(labels, scores)
    )


def calculate_regression_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> Dict[str, float]:
    predicted = np.clip(
        predicted,
        a_min=0.0,
        a_max=None,
    )

    positive_mask = observed > 0
    zero_mask = observed == 0
    occurrence = positive_mask.astype(int)

    metrics: Dict[str, float] = {
        "N": int(len(observed)),
        "Positive_N": int(positive_mask.sum()),
        "Zero_N": int(zero_mask.sum()),
        "Observed_Mean": float(np.mean(observed)),
        "Predicted_Mean": float(np.mean(predicted)),
        "RMSE_All": float(
            math.sqrt(
                mean_squared_error(
                    observed,
                    predicted,
                )
            )
        ),
        "MAE_All": float(
            mean_absolute_error(
                observed,
                predicted,
            )
        ),
        "R2_All": float(
            r2_score(
                observed,
                predicted,
            )
        ),
        "Spearman_All": safe_spearman(
            observed,
            predicted,
        ),
        "Occurrence_ROC_AUC_from_Response_Score":
            safe_auc(occurrence, predicted),
        "Occurrence_PR_AUC_from_Response_Score":
            safe_average_precision(
                occurrence,
                predicted,
            ),
    }

    if positive_mask.any():
        positive_observed = observed[
            positive_mask
        ]
        positive_predicted = predicted[
            positive_mask
        ]

        metrics.update(
            {
                "RMSE_Positive": float(
                    math.sqrt(
                        mean_squared_error(
                            positive_observed,
                            positive_predicted,
                        )
                    )
                ),
                "MAE_Positive": float(
                    mean_absolute_error(
                        positive_observed,
                        positive_predicted,
                    )
                ),
                "Spearman_Positive": safe_spearman(
                    positive_observed,
                    positive_predicted,
                ),
                "Positive_Observed_Mean": float(
                    np.mean(positive_observed)
                ),
                "Positive_Predicted_Mean": float(
                    np.mean(positive_predicted)
                ),
            }
        )
    else:
        metrics.update(
            {
                "RMSE_Positive": np.nan,
                "MAE_Positive": np.nan,
                "Spearman_Positive": np.nan,
                "Positive_Observed_Mean": np.nan,
                "Positive_Predicted_Mean": np.nan,
            }
        )

    if zero_mask.any():
        zero_predictions = predicted[
            zero_mask
        ]

        metrics.update(
            {
                "Zero_Predicted_Mean": float(
                    np.mean(zero_predictions)
                ),
                "Zero_Predicted_Median": float(
                    np.median(zero_predictions)
                ),
                "Zero_Predicted_P95": float(
                    np.quantile(
                        zero_predictions,
                        0.95,
                    )
                ),
            }
        )
    else:
        metrics.update(
            {
                "Zero_Predicted_Mean": np.nan,
                "Zero_Predicted_Median": np.nan,
                "Zero_Predicted_P95": np.nan,
            }
        )

    return metrics


def calculate_occurrence_metrics(
    observed_occurrence: np.ndarray,
    probability: np.ndarray,
) -> Dict[str, float]:
    probability = np.clip(
        probability,
        1e-6,
        1 - 1e-6,
    )

    return {
        "Occurrence_Prevalence": float(
            observed_occurrence.mean()
        ),
        "Mean_Predicted_Probability": float(
            probability.mean()
        ),
        "Occurrence_ROC_AUC": safe_auc(
            observed_occurrence,
            probability,
        ),
        "Occurrence_PR_AUC": safe_average_precision(
            observed_occurrence,
            probability,
        ),
        "Occurrence_Brier": float(
            brier_score_loss(
                observed_occurrence,
                probability,
            )
        ),
        "Occurrence_LogLoss": float(
            log_loss(
                observed_occurrence,
                probability,
                labels=[0, 1],
            )
        ),
    }


# =============================================================================
# Data preparation
# =============================================================================

def prepare_data(
    season: int,
    response: str,
    output_dir: Path,
    log,
) -> tuple[pd.DataFrame, List[str]]:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Input table not found:\n{INPUT_CSV}"
        )

    log(f"Reading: {INPUT_CSV}")
    raw = pd.read_csv(INPUT_CSV)

    required = {
        "GRID_UID",
        "Country",
        "GRID_ID",
        "Year",
        "Season",
        "Fire_Count",
        "Burned_Pixel_Count",
        "ForestPixelCount",
        *PREDICTORS,
        *COORDINATE_COLUMNS,
    }

    missing_columns = sorted(
        required - set(raw.columns)
    )
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    duplicate_panel = raw.duplicated(
        ["GRID_UID", "Year", "Season"],
        keep=False,
    )
    if duplicate_panel.any():
        duplicate_file = (
            output_dir / "ERROR_duplicate_panel_records.csv"
        )
        raw.loc[duplicate_panel].to_csv(
            duplicate_file,
            index=False,
        )
        raise ValueError(
            "Duplicate GRID_UID-Year-Season records found."
        )

    raw["FCD"] = np.where(
        raw["ForestPixelCount"] > 0,
        raw["Fire_Count"] / raw["ForestPixelCount"],
        np.nan,
    )
    raw["BAD"] = np.where(
        raw["ForestPixelCount"] > 0,
        raw["Burned_Pixel_Count"] / raw["ForestPixelCount"],
        np.nan,
    )

    fold_rows = []
    year_to_fold: Dict[int, int] = {}

    for fold, years in TEMPORAL_FOLDS.items():
        for year in years:
            year_to_fold[int(year)] = int(fold)
            fold_rows.append(
                {
                    "Year": int(year),
                    "Temporal_Fold": int(fold),
                }
            )

    pd.DataFrame(fold_rows).sort_values(
        "Year"
    ).to_csv(
        output_dir / "Temporal_Fold_Assignment.csv",
        index=False,
    )

    selected = raw.loc[
        (raw["Season"] == season)
        & (
            raw["ForestArea_km2"]
            > FOREST_AREA_THRESHOLD_KM2
        )
    ].copy()

    selected["Temporal_Fold"] = (
        selected["Year"]
        .map(year_to_fold)
        .astype(int)
    )

    country_dummies = pd.get_dummies(
        selected["Country"],
        prefix="Country",
        dtype=float,
    )

    for column in COUNTRY_DUMMY_COLUMNS:
        if column not in country_dummies.columns:
            country_dummies[column] = 0.0

    country_dummies = country_dummies[
        COUNTRY_DUMMY_COLUMNS
    ]

    selected = pd.concat(
        [
            selected.reset_index(drop=True),
            country_dummies.reset_index(drop=True),
        ],
        axis=1,
    )

    model_predictors = (
        PREDICTORS + COUNTRY_DUMMY_COLUMNS
    )

    missing_matrix = selected[
        model_predictors
    ].isna()

    incomplete_mask = (
        missing_matrix.any(axis=1)
        | selected[response].isna()
    )

    excluded = selected.loc[
        incomplete_mask,
        [
            "GRID_UID",
            "Country",
            "GRID_ID",
            "Year",
            "Season",
            "ForestPixelCount",
            response,
            *PREDICTORS,
        ],
    ].copy()

    excluded["Missing_Fields"] = (
        missing_matrix.loc[incomplete_mask]
        .apply(
            lambda row: ";".join(
                row.index[row].tolist()
            ),
            axis=1,
        )
        .to_numpy()
    )

    excluded.to_csv(
        output_dir / "Excluded_Missing_Records.csv",
        index=False,
    )

    excluded_grid_summary = (
        excluded.groupby(
            ["GRID_UID", "Country", "GRID_ID"],
            as_index=False,
        )
        .agg(
            Excluded_Rows=("Year", "size"),
            First_Year=("Year", "min"),
            Last_Year=("Year", "max"),
            Missing_Fields=(
                "Missing_Fields",
                lambda values: ";".join(
                    sorted(
                        {
                            field
                            for value in values
                            for field in str(value).split(";")
                            if field
                        }
                    )
                ),
            ),
        )
    )

    excluded_grid_summary.to_csv(
        output_dir / "Excluded_Missing_GRID_UIDs.csv",
        index=False,
    )

    model_data = selected.loc[
        ~incomplete_mask
    ].copy()

    overview = pd.DataFrame(
        [
            ["Season", season],
            ["Response", response],
            [
                "Rows_before_forest_threshold",
                len(raw.loc[raw["Season"] == season]),
            ],
            [
                "Rows_after_ForestArea_km2_gt2_5",
                len(selected),
            ],
            [
                "Rows_excluded_for_missing_values",
                int(incomplete_mask.sum()),
            ],
            [
                "Rows_used_for_modeling",
                len(model_data),
            ],
            [
                "Unique_GRID_UIDs_used",
                model_data["GRID_UID"].nunique(),
            ],
            [
                "Positive_rows",
                int((model_data[response] > 0).sum()),
            ],
            [
                "Zero_rows",
                int((model_data[response] == 0).sum()),
            ],
            [
                "Zero_percentage",
                float(
                    (
                        model_data[response]
                        == 0
                    ).mean()
                    * 100
                ),
            ],
            [
                "Predictor_count",
                len(model_predictors),
            ],
        ],
        columns=["Check", "Value"],
    )

    overview.to_csv(
        output_dir / "Data_QA_Summary.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "Predictor": model_predictors,
            "Role": (
                ["Numeric predictor"] * len(PREDICTORS)
                + ["Country dummy"]
                * len(COUNTRY_DUMMY_COLUMNS)
            ),
        }
    ).to_csv(
        output_dir / "Predictor_List.csv",
        index=False,
    )

    log(
        f"Prepared model data: {len(model_data):,} rows; "
        f"excluded {int(incomplete_mask.sum()):,} incomplete rows."
    )

    return model_data, model_predictors


# =============================================================================
# Model comparison
# =============================================================================

def run_comparison(
    data: pd.DataFrame,
    predictors: List[str],
    response: str,
    n_estimators: int,
    output_dir: Path,
    log,
) -> None:
    fold_metric_rows: List[Dict[str, object]] = []
    occurrence_metric_rows: List[Dict[str, object]] = []
    severity_metric_rows: List[Dict[str, object]] = []
    importance_rows: List[Dict[str, object]] = []
    prediction_frames: List[pd.DataFrame] = []

    for fold in range(1, 6):
        validation = data.loc[
            data["Temporal_Fold"] == fold
        ].copy()

        training = data.loc[
            data["Temporal_Fold"] != fold
        ].copy()

        validation_years = sorted(
            validation["Year"].unique()
        )

        log(
            f"Fold {fold}: validation years = "
            f"{validation_years}"
        )

        X_train = training[
            predictors
        ].to_numpy(dtype=np.float32)

        y_train = training[
            response
        ].to_numpy(dtype=np.float64)

        X_valid = validation[
            predictors
        ].to_numpy(dtype=np.float32)

        y_valid = validation[
            response
        ].to_numpy(dtype=np.float64)

        occurrence_train = (
            y_train > 0
        ).astype(np.int8)

        occurrence_valid = (
            y_valid > 0
        ).astype(np.int8)

        fold_seed = RANDOM_SEED + fold * 1000

        # ---------------------------------------------------------------------
        # Model 1: full single-stage RF regression
        # ---------------------------------------------------------------------
        full_rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_features="sqrt",
            min_samples_leaf=5,
            max_samples=0.70,
            bootstrap=True,
            n_jobs=-1,
            random_state=fold_seed + 1,
            criterion="squared_error",
        )

        full_rf.fit(
            X_train,
            y_train,
        )

        full_prediction = np.clip(
            full_rf.predict(X_valid),
            0.0,
            None,
        )

        full_metrics = calculate_regression_metrics(
            observed=y_valid,
            predicted=full_prediction,
        )

        fold_metric_rows.append(
            {
                "Fold": fold,
                "Model": "Full_RF",
                "Validation_Years": ",".join(
                    map(str, validation_years)
                ),
                **full_metrics,
            }
        )

        for predictor, importance in zip(
            predictors,
            full_rf.feature_importances_,
        ):
            importance_rows.append(
                {
                    "Fold": fold,
                    "Component": "Full_RF",
                    "Predictor": predictor,
                    "Importance": float(importance),
                }
            )

        # ---------------------------------------------------------------------
        # Model 2: two-part hurdle RF
        # Stage 1 uses all training records without zero downsampling.
        # Stage 2 uses only positive training records.
        # ---------------------------------------------------------------------
        occurrence_rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_features="sqrt",
            min_samples_leaf=5,
            max_samples=0.70,
            bootstrap=True,
            class_weight=None,
            n_jobs=-1,
            random_state=fold_seed + 2,
            criterion="gini",
        )

        occurrence_rf.fit(
            X_train,
            occurrence_train,
        )

        occurrence_probability = (
            occurrence_rf.predict_proba(
                X_valid
            )[:, 1]
        )

        occurrence_metrics = (
            calculate_occurrence_metrics(
                observed_occurrence=occurrence_valid,
                probability=occurrence_probability,
            )
        )

        occurrence_metric_rows.append(
            {
                "Fold": fold,
                "Validation_Years": ",".join(
                    map(str, validation_years)
                ),
                "Training_Prevalence": float(
                    occurrence_train.mean()
                ),
                **occurrence_metrics,
            }
        )

        positive_train_mask = y_train > 0
        positive_valid_mask = y_valid > 0

        positive_train_n = int(
            positive_train_mask.sum()
        )

        if positive_train_n < 20:
            raise ValueError(
                "Too few positive training records for "
                f"severity RF in fold {fold}: "
                f"{positive_train_n}"
            )

        severity_rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_features="sqrt",
            min_samples_leaf=3,
            max_samples=0.80,
            bootstrap=True,
            n_jobs=-1,
            random_state=fold_seed + 3,
            criterion="squared_error",
        )

        severity_rf.fit(
            X_train[positive_train_mask],
            y_train[positive_train_mask],
        )

        severity_prediction_all = np.clip(
            severity_rf.predict(X_valid),
            0.0,
            None,
        )

        hurdle_expected = (
            occurrence_probability
            * severity_prediction_all
        )

        hurdle_metrics = calculate_regression_metrics(
            observed=y_valid,
            predicted=hurdle_expected,
        )

        fold_metric_rows.append(
            {
                "Fold": fold,
                "Model": "Hurdle_RF",
                "Validation_Years": ",".join(
                    map(str, validation_years)
                ),
                **hurdle_metrics,
            }
        )

        if positive_valid_mask.any():
            severity_observed = y_valid[
                positive_valid_mask
            ]
            severity_predicted = (
                severity_prediction_all[
                    positive_valid_mask
                ]
            )

            severity_metrics = {
                "Fold": fold,
                "Validation_Years": ",".join(
                    map(str, validation_years)
                ),
                "Training_Positive_N":
                    positive_train_n,
                "Validation_Positive_N": int(
                    positive_valid_mask.sum()
                ),
                "Severity_Observed_Mean": float(
                    severity_observed.mean()
                ),
                "Severity_Predicted_Mean": float(
                    severity_predicted.mean()
                ),
                "Severity_RMSE": float(
                    math.sqrt(
                        mean_squared_error(
                            severity_observed,
                            severity_predicted,
                        )
                    )
                ),
                "Severity_MAE": float(
                    mean_absolute_error(
                        severity_observed,
                        severity_predicted,
                    )
                ),
                "Severity_Spearman":
                    safe_spearman(
                        severity_observed,
                        severity_predicted,
                    ),
            }
        else:
            severity_metrics = {
                "Fold": fold,
                "Validation_Years": ",".join(
                    map(str, validation_years)
                ),
                "Training_Positive_N":
                    positive_train_n,
                "Validation_Positive_N": 0,
                "Severity_Observed_Mean": np.nan,
                "Severity_Predicted_Mean": np.nan,
                "Severity_RMSE": np.nan,
                "Severity_MAE": np.nan,
                "Severity_Spearman": np.nan,
            }

        severity_metric_rows.append(
            severity_metrics
        )

        for predictor, importance in zip(
            predictors,
            occurrence_rf.feature_importances_,
        ):
            importance_rows.append(
                {
                    "Fold": fold,
                    "Component": "Hurdle_Occurrence_RF",
                    "Predictor": predictor,
                    "Importance": float(importance),
                }
            )

        for predictor, importance in zip(
            predictors,
            severity_rf.feature_importances_,
        ):
            importance_rows.append(
                {
                    "Fold": fold,
                    "Component": "Hurdle_Severity_RF",
                    "Predictor": predictor,
                    "Importance": float(importance),
                }
            )

        prediction_frame = validation[
            [
                "GRID_UID",
                "Country",
                "GRID_ID",
                "Year",
                "Season",
                "Temporal_Fold",
                response,
                *COORDINATE_COLUMNS,
            ]
        ].copy()

        prediction_frame = prediction_frame.rename(
            columns={response: "Observed"}
        )

        prediction_frame[
            "Observed_Occurrence"
        ] = occurrence_valid

        prediction_frame[
            "Full_RF_Prediction"
        ] = full_prediction

        prediction_frame[
            "Hurdle_Occurrence_Probability"
        ] = occurrence_probability

        prediction_frame[
            "Hurdle_Conditional_Severity"
        ] = severity_prediction_all

        prediction_frame[
            "Hurdle_Expected_Prediction"
        ] = hurdle_expected

        prediction_frames.append(
            prediction_frame
        )

    # -------------------------------------------------------------------------
    # Save fold-level outputs
    # -------------------------------------------------------------------------
    fold_metrics = pd.DataFrame(
        fold_metric_rows
    )

    fold_metrics.to_csv(
        output_dir / "Model_Comparison_Fold_Metrics.csv",
        index=False,
    )

    occurrence_metrics_df = pd.DataFrame(
        occurrence_metric_rows
    )
    occurrence_metrics_df.to_csv(
        output_dir / "Hurdle_Occurrence_Fold_Metrics.csv",
        index=False,
    )

    severity_metrics_df = pd.DataFrame(
        severity_metric_rows
    )
    severity_metrics_df.to_csv(
        output_dir / "Hurdle_Severity_Fold_Metrics.csv",
        index=False,
    )

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    predictions.to_csv(
        output_dir / "Cross_Validated_Predictions.csv.gz",
        index=False,
        compression="gzip",
    )

    # -------------------------------------------------------------------------
    # Model summary
    # -------------------------------------------------------------------------
    id_columns = {
        "Fold",
        "Model",
        "Validation_Years",
    }

    metric_columns = [
        column
        for column in fold_metrics.columns
        if column not in id_columns
    ]

    summary_rows: List[Dict[str, object]] = []

    for model_name, group in fold_metrics.groupby(
        "Model",
        sort=False,
    ):
        row: Dict[str, object] = {
            "Model": model_name,
            "Folds": len(group),
        }

        for metric in metric_columns:
            row[f"{metric}_Mean"] = float(
                group[metric].mean()
            )
            row[f"{metric}_SD"] = float(
                group[metric].std(ddof=1)
            )

        summary_rows.append(row)

    model_summary = pd.DataFrame(
        summary_rows
    )

    model_order = {
        "Full_RF": 0,
        "Hurdle_RF": 1,
    }

    model_summary["_order"] = (
        model_summary["Model"]
        .map(model_order)
    )

    model_summary = (
        model_summary.sort_values("_order")
        .drop(columns="_order")
    )

    model_summary.to_csv(
        output_dir / "Model_Comparison_Summary.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Paired fold differences: Hurdle minus Full
    # Negative RMSE/MAE means Hurdle is better.
    # Positive R2/Spearman/AUC means Hurdle is better.
    # -------------------------------------------------------------------------
    full_by_fold = (
        fold_metrics.loc[
            fold_metrics["Model"] == "Full_RF"
        ]
        .set_index("Fold")
    )

    hurdle_by_fold = (
        fold_metrics.loc[
            fold_metrics["Model"] == "Hurdle_RF"
        ]
        .set_index("Fold")
    )

    paired_rows = []

    for fold in sorted(full_by_fold.index):
        row = {
            "Fold": fold,
            "Validation_Years":
                full_by_fold.at[
                    fold,
                    "Validation_Years",
                ],
        }

        for metric in metric_columns:
            row[
                f"Hurdle_minus_Full_{metric}"
            ] = (
                hurdle_by_fold.at[fold, metric]
                - full_by_fold.at[fold, metric]
            )

        paired_rows.append(row)

    pd.DataFrame(paired_rows).to_csv(
        output_dir / "Paired_Fold_Differences.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Feature-importance summaries
    # -------------------------------------------------------------------------
    importance_df = pd.DataFrame(
        importance_rows
    )

    importance_df.to_csv(
        output_dir / "Feature_Importance_By_Fold.csv",
        index=False,
    )

    importance_summary = (
        importance_df.groupby(
            ["Component", "Predictor"],
            as_index=False,
        )
        .agg(
            Importance_Mean=("Importance", "mean"),
            Importance_SD=("Importance", "std"),
            Importance_Min=("Importance", "min"),
            Importance_Max=("Importance", "max"),
        )
    )

    importance_summary[
        "Rank_within_Component"
    ] = (
        importance_summary.groupby(
            "Component"
        )["Importance_Mean"]
        .rank(
            ascending=False,
            method="min",
        )
        .astype(int)
    )

    importance_summary = (
        importance_summary.sort_values(
            [
                "Component",
                "Rank_within_Component",
            ]
        )
    )

    importance_summary.to_csv(
        output_dir / "Feature_Importance_Summary.csv",
        index=False,
    )

    log(
        "Full RF versus hurdle RF comparison "
        "completed successfully."
    )

# =============================================================================
# All-model settings
# =============================================================================

ROOT_LOG = OUTPUT_ROOT / "RF_hurdle_all_models.log"
N_ESTIMATORS = 100

MODEL_COMBINATIONS = [
    (1, "FCD"),
    (1, "BAD"),
    (2, "FCD"),
    (2, "BAD"),
    (3, "FCD"),
    (3, "BAD"),
]



def root_log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    with ROOT_LOG.open("a", encoding="utf-8") as file:
        file.write(line + "\n")


def run_one_combination(
    season: int,
    response: str,
) -> None:
    label = f"S{season}_{response}"
    temporary_dir = OUTPUT_ROOT / f"_temporary_{label}"

    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)
    temporary_dir.mkdir(parents=True, exist_ok=True)

    log_path = temporary_dir / "RF_hurdle_comparison.log"
    log_path.write_text("", encoding="utf-8")
    log = make_logger(log_path)

    root_log("=" * 78)
    root_log(f"Starting {label}")
    root_log("=" * 78)

    log("=" * 78)
    log("Starting full RF versus hurdle RF comparison")
    log(f"Season: {season}")
    log(f"Response: {response}")
    log(f"Trees per RF: {N_ESTIMATORS}")
    log(f"Input: {INPUT_CSV}")
    log(f"Output root: {OUTPUT_ROOT}")
    log("=" * 78)

    environment = {
        "Python": sys.version,
        "Platform": platform.platform(),
        "NumPy": np.__version__,
        "pandas": pd.__version__,
        "SciPy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
    }

    with (temporary_dir / "Software_Environment.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(environment, file, ensure_ascii=False, indent=2)

    data, predictors = prepare_data(
        season=season,
        response=response,
        output_dir=temporary_dir,
        log=log,
    )

    run_comparison(
        data=data,
        predictors=predictors,
        response=response,
        n_estimators=N_ESTIMATORS,
        output_dir=temporary_dir,
        log=log,
    )

    log("=" * 78)
    log("Completed successfully.")
    log("=" * 78)

    # Keep one output directory per script. Files for each model are
    # distinguished by the S1_FCD, S1_BAD, ..., S3_BAD prefix.
    for source in sorted(temporary_dir.iterdir()):
        destination = OUTPUT_ROOT / f"{label}_{source.name}"
        if destination.exists():
            if destination.is_dir():
                shutil.rmtree(destination)
            else:
                destination.unlink()
        shutil.move(str(source), str(destination))

    temporary_dir.rmdir()
    root_log(f"Completed {label} successfully.")


def read_summary(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def build_combined_summary() -> Path:
    rows_out: list[dict[str, object]] = []
    for season, response in MODEL_COMBINATIONS:
        label = f"S{season}_{response}"
        summary_path = OUTPUT_ROOT / f"{label}_Model_Comparison_Summary.csv"
        for row in read_summary(summary_path):
            rows_out.append({"Season": season, "Response": response, **row})

    if not rows_out:
        raise RuntimeError("No model comparison summaries were generated.")

    output_path = OUTPUT_ROOT / "00_Combined_Model_Comparison_Summary.csv"
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)
    return output_path




# =============================================================================
# Main
# =============================================================================

def main() -> int:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input table not found:\n{INPUT_CSV}")

    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True)
    ROOT_LOG.write_text("", encoding="utf-8")

    root_log("=" * 78)
    root_log("Starting all-season full RF versus hurdle RF comparison")
    root_log(f"Python executable: {sys.executable}")
    root_log(f"Trees per RF: {N_ESTIMATORS}")
    root_log(f"Input: {INPUT_CSV}")
    root_log(f"Output: {OUTPUT_ROOT}")
    root_log("=" * 78)

    for season, response in MODEL_COMBINATIONS:
        run_one_combination(season=season, response=response)

    combined_summary = build_combined_summary()

    root_log("=" * 78)
    root_log("All six model combinations completed successfully.")
    root_log(f"Combined summary: {combined_summary}")
    root_log("=" * 78)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        error_text = traceback.format_exc()
        try:
            OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
            root_log("=" * 78)
            root_log("ERROR")
            root_log(error_text)
            root_log("=" * 78)
        except Exception:
            print(error_text, file=sys.stderr)
        raise
