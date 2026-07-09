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

OUTPUT_ROOT = MODEL_ROOT / "05_Hurdle_RF_Hyperparameter_Tuning"
LOG_FILE = OUTPUT_ROOT / "hurdle_RF_hyperparameter_tuning.log"

FOREST_AREA_THRESHOLD_KM2 = 2.5
RANDOM_SEED = 2026
N_ESTIMATORS = 150

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

# RF main-model predictor set selected after the collinearity and
# forest-structure comparisons.
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

# Occurrence-stage candidate configurations.
OCCURRENCE_CONFIGS: Dict[str, Dict[str, object]] = {
    "O1_Baseline": {
        "max_features": "sqrt",
        "min_samples_leaf": 5,
        "max_samples": 0.70,
        "class_weight": None,
    },
    "O2_Leaf20": {
        "max_features": "sqrt",
        "min_samples_leaf": 20,
        "max_samples": 0.70,
        "class_weight": None,
    },
    "O3_Leaf50": {
        "max_features": "sqrt",
        "min_samples_leaf": 50,
        "max_samples": 0.70,
        "class_weight": None,
    },
    "O4_Features50_Leaf20": {
        "max_features": 0.50,
        "min_samples_leaf": 20,
        "max_samples": 0.70,
        "class_weight": None,
    },
    "O5_Balanced_Leaf20": {
        "max_features": "sqrt",
        "min_samples_leaf": 20,
        "max_samples": 0.70,
        "class_weight": "balanced_subsample",
    },
    "O6_Features50_Balanced": {
        "max_features": 0.50,
        "min_samples_leaf": 20,
        "max_samples": 0.70,
        "class_weight": "balanced_subsample",
    },
}

# Positive-severity-stage candidate configurations.
SEVERITY_CONFIGS: Dict[str, Dict[str, object]] = {
    "S1_Baseline": {
        "max_features": "sqrt",
        "min_samples_leaf": 3,
        "max_samples": 0.80,
    },
    "S2_Leaf1": {
        "max_features": "sqrt",
        "min_samples_leaf": 1,
        "max_samples": 0.80,
    },
    "S3_Leaf10": {
        "max_features": "sqrt",
        "min_samples_leaf": 10,
        "max_samples": 0.80,
    },
    "S4_Features50_Leaf3": {
        "max_features": 0.50,
        "min_samples_leaf": 3,
        "max_samples": 0.80,
    },
    "S5_Features50_Leaf10": {
        "max_features": 0.50,
        "min_samples_leaf": 10,
        "max_samples": 0.80,
    },
    "S6_AllFeatures_Leaf3": {
        "max_features": 1.00,
        "min_samples_leaf": 3,
        "max_samples": 0.80,
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

    return float(roc_auc_score(labels, scores))


def safe_pr_auc(
    labels: np.ndarray,
    scores: np.ndarray,
) -> float:
    if len(np.unique(labels)) < 2:
        return np.nan

    return float(average_precision_score(labels, scores))


def occurrence_metrics(
    observed: np.ndarray,
    probability: np.ndarray,
) -> Dict[str, float]:
    probability = np.clip(probability, 1e-6, 1 - 1e-6)

    return {
        "Occurrence_Prevalence": float(observed.mean()),
        "Mean_Predicted_Probability": float(probability.mean()),
        "Probability_Mean_Bias": float(
            probability.mean() - observed.mean()
        ),
        "Absolute_Probability_Mean_Bias": float(
            abs(probability.mean() - observed.mean())
        ),
        "Occurrence_ROC_AUC": safe_roc_auc(
            observed,
            probability,
        ),
        "Occurrence_PR_AUC": safe_pr_auc(
            observed,
            probability,
        ),
        "Occurrence_Brier": float(
            brier_score_loss(observed, probability)
        ),
        "Occurrence_LogLoss": float(
            log_loss(
                observed,
                probability,
                labels=[0, 1],
            )
        ),
    }


def severity_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> Dict[str, float]:
    predicted = np.clip(predicted, 0.0, None)

    return {
        "Severity_Observed_Mean": float(observed.mean()),
        "Severity_Predicted_Mean": float(predicted.mean()),
        "Severity_Mean_Bias": float(
            predicted.mean() - observed.mean()
        ),
        "Absolute_Severity_Mean_Bias": float(
            abs(predicted.mean() - observed.mean())
        ),
        "Severity_RMSE": float(
            math.sqrt(
                mean_squared_error(observed, predicted)
            )
        ),
        "Severity_MAE": float(
            mean_absolute_error(observed, predicted)
        ),
        "Severity_Spearman": safe_spearman(
            observed,
            predicted,
        ),
    }


def combined_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> Dict[str, float]:
    predicted = np.clip(predicted, 0.0, None)

    positive_mask = observed > 0
    zero_mask = observed == 0
    occurrence = positive_mask.astype(int)

    result: Dict[str, float] = {
        "N": int(len(observed)),
        "Positive_N": int(positive_mask.sum()),
        "Zero_N": int(zero_mask.sum()),
        "Observed_Mean": float(observed.mean()),
        "Predicted_Mean": float(predicted.mean()),
        "Mean_Bias": float(
            predicted.mean() - observed.mean()
        ),
        "Absolute_Mean_Bias": float(
            abs(predicted.mean() - observed.mean())
        ),
        "RMSE_All": float(
            math.sqrt(
                mean_squared_error(observed, predicted)
            )
        ),
        "MAE_All": float(
            mean_absolute_error(observed, predicted)
        ),
        "R2_All": float(
            r2_score(observed, predicted)
        ),
        "Spearman_All": safe_spearman(
            observed,
            predicted,
        ),
        "Occurrence_ROC_AUC_from_Expected_Response":
            safe_roc_auc(occurrence, predicted),
        "Occurrence_PR_AUC_from_Expected_Response":
            safe_pr_auc(occurrence, predicted),
    }

    if positive_mask.any():
        positive_observed = observed[positive_mask]
        positive_predicted = predicted[positive_mask]

        result.update(
            {
                "RMSE_Positive_Expected_Response": float(
                    math.sqrt(
                        mean_squared_error(
                            positive_observed,
                            positive_predicted,
                        )
                    )
                ),
                "MAE_Positive_Expected_Response": float(
                    mean_absolute_error(
                        positive_observed,
                        positive_predicted,
                    )
                ),
                "Spearman_Positive_Expected_Response":
                    safe_spearman(
                        positive_observed,
                        positive_predicted,
                    ),
            }
        )
    else:
        result.update(
            {
                "RMSE_Positive_Expected_Response": np.nan,
                "MAE_Positive_Expected_Response": np.nan,
                "Spearman_Positive_Expected_Response": np.nan,
            }
        )

    if zero_mask.any():
        zero_predictions = predicted[zero_mask]

        result.update(
            {
                "Zero_Predicted_Mean": float(
                    zero_predictions.mean()
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
        result.update(
            {
                "Zero_Predicted_Mean": np.nan,
                "Zero_Predicted_P95": np.nan,
            }
        )

    return result


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

    incomplete_mask = data[
        MODEL_PREDICTORS + RESPONSES
    ].isna().any(axis=1)

    excluded = data.loc[
        incomplete_mask,
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

    model_data = data.loc[
        ~incomplete_mask
    ].copy()

    qa = pd.DataFrame(
        [
            ["Input_rows", len(data)],
            [
                "Rows_excluded_for_any_model_predictor_or_response_missing",
                int(incomplete_mask.sum()),
            ],
            ["Rows_used", len(model_data)],
            [
                "Unique_GRID_UIDs_used",
                model_data["GRID_UID"].nunique(),
            ],
            [
                "Continuous_predictor_count",
                len(CONTINUOUS_PREDICTORS),
            ],
            [
                "Total_model_column_count",
                len(MODEL_PREDICTORS),
            ],
            ["Trees_per_RF", N_ESTIMATORS],
        ],
        columns=["Check", "Value"],
    )

    qa.to_csv(
        OUTPUT_ROOT / "Data_QA_Summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return model_data


# =============================================================================
# Summary and ranking helpers
# =============================================================================

def summarize_numeric(
    data: pd.DataFrame,
    group_columns: Sequence[str],
    excluded_numeric_columns: Sequence[str],
) -> pd.DataFrame:
    numeric_columns = [
        column
        for column in data.select_dtypes(
            include=[np.number]
        ).columns
        if column not in set(excluded_numeric_columns)
    ]

    rows: List[Dict[str, object]] = []

    for group_values, group in data.groupby(
        list(group_columns),
        sort=False,
    ):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)

        row = dict(zip(group_columns, group_values))
        row["Evaluation_Folds"] = len(group)

        for column in numeric_columns:
            row[f"{column}_Mean"] = float(
                group[column].mean()
            )
            row[f"{column}_SD"] = float(
                group[column].std(ddof=1)
            )

        rows.append(row)

    return pd.DataFrame(rows)


def add_rank_sum(
    summary: pd.DataFrame,
    group_columns: Sequence[str],
    higher_is_better: Sequence[str],
    lower_is_better: Sequence[str],
    rank_name: str,
) -> pd.DataFrame:
    output_frames: List[pd.DataFrame] = []

    for _, group in summary.groupby(
        list(group_columns),
        sort=False,
    ):
        ranked = group.copy()
        rank_columns: List[str] = []

        for column in higher_is_better:
            rank_column = f"Rank_{column}"
            ranked[rank_column] = ranked[column].rank(
                ascending=False,
                method="average",
            )
            rank_columns.append(rank_column)

        for column in lower_is_better:
            rank_column = f"Rank_{column}"
            ranked[rank_column] = ranked[column].rank(
                ascending=True,
                method="average",
            )
            rank_columns.append(rank_column)

        ranked[rank_name] = ranked[
            rank_columns
        ].sum(axis=1)

        ranked[
            f"{rank_name}_Position"
        ] = ranked[rank_name].rank(
            ascending=True,
            method="min",
        ).astype(int)

        output_frames.append(ranked)

    return pd.concat(
        output_frames,
        ignore_index=True,
    )


# =============================================================================
# Tuning workflow
# =============================================================================

def run_tuning(
    data: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    occurrence_rows: List[Dict[str, object]] = []
    severity_rows: List[Dict[str, object]] = []
    combined_rows: List[Dict[str, object]] = []

    for season, season_label in SEASONS.items():
        for response in RESPONSES:
            subset = data.loc[
                data["Season"] == season
            ].copy()

            log(
                f"Starting {season_label} {response}"
            )

            for fold in range(1, 6):
                training = subset.loc[
                    subset["Temporal_Fold"] != fold
                ]

                validation = subset.loc[
                    subset["Temporal_Fold"] == fold
                ]

                validation_years = ",".join(
                    map(
                        str,
                        sorted(
                            validation["Year"].unique()
                        ),
                    )
                )

                X_train = training[
                    MODEL_PREDICTORS
                ].to_numpy(dtype=np.float32)

                X_valid = validation[
                    MODEL_PREDICTORS
                ].to_numpy(dtype=np.float32)

                y_train = training[
                    response
                ].to_numpy(dtype=np.float64)

                y_valid = validation[
                    response
                ].to_numpy(dtype=np.float64)

                occurrence_train = (
                    y_train > 0
                ).astype(np.int8)

                occurrence_valid = (
                    y_valid > 0
                ).astype(np.int8)

                positive_train = y_train > 0
                positive_valid = y_valid > 0

                occurrence_predictions: Dict[str, np.ndarray] = {}
                severity_predictions: Dict[str, np.ndarray] = {}

                # Fit each occurrence candidate once for this fold.
                for config_index, (
                    config_name,
                    config,
                ) in enumerate(
                    OCCURRENCE_CONFIGS.items()
                ):
                    seed = (
                        RANDOM_SEED
                        + season * 100_000
                        + (
                            10_000
                            if response == "BAD"
                            else 0
                        )
                        + fold * 1_000
                        + config_index * 10
                        + 1
                    )

                    model = RandomForestClassifier(
                        n_estimators=N_ESTIMATORS,
                        max_features=config[
                            "max_features"
                        ],
                        min_samples_leaf=int(
                            config[
                                "min_samples_leaf"
                            ]
                        ),
                        max_samples=float(
                            config["max_samples"]
                        ),
                        bootstrap=True,
                        class_weight=config[
                            "class_weight"
                        ],
                        n_jobs=-1,
                        random_state=seed,
                        criterion="gini",
                    )

                    model.fit(
                        X_train,
                        occurrence_train,
                    )

                    probability = (
                        model.predict_proba(
                            X_valid
                        )[:, 1]
                    )

                    occurrence_predictions[
                        config_name
                    ] = probability

                    occurrence_rows.append(
                        {
                            "Season": season,
                            "Season_Label":
                                season_label,
                            "Response": response,
                            "Fold": fold,
                            "Validation_Years":
                                validation_years,
                            "Occurrence_Config":
                                config_name,
                            **occurrence_metrics(
                                occurrence_valid,
                                probability,
                            ),
                        }
                    )

                # Fit each positive-severity candidate once for this fold.
                for config_index, (
                    config_name,
                    config,
                ) in enumerate(
                    SEVERITY_CONFIGS.items()
                ):
                    seed = (
                        RANDOM_SEED
                        + season * 100_000
                        + (
                            10_000
                            if response == "BAD"
                            else 0
                        )
                        + fold * 1_000
                        + config_index * 10
                        + 2
                    )

                    model = RandomForestRegressor(
                        n_estimators=N_ESTIMATORS,
                        max_features=config[
                            "max_features"
                        ],
                        min_samples_leaf=int(
                            config[
                                "min_samples_leaf"
                            ]
                        ),
                        max_samples=float(
                            config["max_samples"]
                        ),
                        bootstrap=True,
                        n_jobs=-1,
                        random_state=seed,
                        criterion="squared_error",
                    )

                    model.fit(
                        X_train[positive_train],
                        y_train[positive_train],
                    )

                    conditional_prediction = np.clip(
                        model.predict(X_valid),
                        0.0,
                        None,
                    )

                    severity_predictions[
                        config_name
                    ] = conditional_prediction

                    if positive_valid.any():
                        severity_rows.append(
                            {
                                "Season": season,
                                "Season_Label":
                                    season_label,
                                "Response":
                                    response,
                                "Fold": fold,
                                "Validation_Years":
                                    validation_years,
                                "Severity_Config":
                                    config_name,
                                "Training_Positive_N":
                                    int(
                                        positive_train.sum()
                                    ),
                                "Validation_Positive_N":
                                    int(
                                        positive_valid.sum()
                                    ),
                                **severity_metrics(
                                    y_valid[
                                        positive_valid
                                    ],
                                    conditional_prediction[
                                        positive_valid
                                    ],
                                ),
                            }
                        )

                # Combine every occurrence and severity candidate without
                # refitting, because expected response is their product.
                for occurrence_name, probability in (
                    occurrence_predictions.items()
                ):
                    for severity_name, conditional in (
                        severity_predictions.items()
                    ):
                        expected_response = (
                            probability * conditional
                        )

                        combined_rows.append(
                            {
                                "Season": season,
                                "Season_Label":
                                    season_label,
                                "Response": response,
                                "Fold": fold,
                                "Validation_Years":
                                    validation_years,
                                "Occurrence_Config":
                                    occurrence_name,
                                "Severity_Config":
                                    severity_name,
                                "Combined_Config": (
                                    f"{occurrence_name}"
                                    f"__{severity_name}"
                                ),
                                **combined_metrics(
                                    y_valid,
                                    expected_response,
                                ),
                            }
                        )

                log(
                    f"Completed {season_label} "
                    f"{response}, fold {fold}"
                )

    return (
        pd.DataFrame(occurrence_rows),
        pd.DataFrame(severity_rows),
        pd.DataFrame(combined_rows),
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
    log("Starting hurdle-RF hyperparameter tuning")
    log(f"Input: {INPUT_CSV}")
    log(f"Trees per RF: {N_ESTIMATORS}")
    log(
        f"Occurrence candidates: "
        f"{len(OCCURRENCE_CONFIGS)}"
    )
    log(
        f"Severity candidates: "
        f"{len(SEVERITY_CONFIGS)}"
    )
    log("=" * 78)

    data = prepare_data()

    # Save candidate definitions.
    definition_rows: List[Dict[str, object]] = []

    for name, config in OCCURRENCE_CONFIGS.items():
        definition_rows.append(
            {
                "Stage": "Occurrence",
                "Config": name,
                **config,
            }
        )

    for name, config in SEVERITY_CONFIGS.items():
        definition_rows.append(
            {
                "Stage": "Severity",
                "Config": name,
                "max_features":
                    config["max_features"],
                "min_samples_leaf":
                    config["min_samples_leaf"],
                "max_samples":
                    config["max_samples"],
                "class_weight": None,
            }
        )

    pd.DataFrame(definition_rows).to_csv(
        OUTPUT_ROOT
        / "00_Hyperparameter_Candidate_Definitions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        {
            "Order": range(
                1,
                len(CONTINUOUS_PREDICTORS) + 1,
            ),
            "Predictor": CONTINUOUS_PREDICTORS,
        }
    ).to_csv(
        OUTPUT_ROOT
        / "01_Selected_Continuous_Predictors.csv",
        index=False,
        encoding="utf-8-sig",
    )

    occurrence_fold, severity_fold, combined_fold = (
        run_tuning(data)
    )

    occurrence_fold.to_csv(
        OUTPUT_ROOT
        / "02_Occurrence_Candidate_Fold_Metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    severity_fold.to_csv(
        OUTPUT_ROOT
        / "03_Severity_Candidate_Fold_Metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    combined_fold.to_csv(
        OUTPUT_ROOT
        / "04_Combined_Candidate_Fold_Metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    occurrence_summary = summarize_numeric(
        occurrence_fold,
        group_columns=[
            "Season",
            "Season_Label",
            "Response",
            "Occurrence_Config",
        ],
        excluded_numeric_columns=[
            "Season",
            "Fold",
        ],
    )

    severity_summary = summarize_numeric(
        severity_fold,
        group_columns=[
            "Season",
            "Season_Label",
            "Response",
            "Severity_Config",
        ],
        excluded_numeric_columns=[
            "Season",
            "Fold",
            "Training_Positive_N",
            "Validation_Positive_N",
        ],
    )

    combined_summary = summarize_numeric(
        combined_fold,
        group_columns=[
            "Season",
            "Season_Label",
            "Response",
            "Occurrence_Config",
            "Severity_Config",
            "Combined_Config",
        ],
        excluded_numeric_columns=[
            "Season",
            "Fold",
        ],
    )

    occurrence_ranked = add_rank_sum(
        occurrence_summary,
        group_columns=[
            "Season",
            "Response",
        ],
        higher_is_better=[
            "Occurrence_ROC_AUC_Mean",
            "Occurrence_PR_AUC_Mean",
        ],
        lower_is_better=[
            "Occurrence_Brier_Mean",
            "Occurrence_LogLoss_Mean",
            "Absolute_Probability_Mean_Bias_Mean",
        ],
        rank_name="Occurrence_Rank_Sum",
    )

    severity_ranked = add_rank_sum(
        severity_summary,
        group_columns=[
            "Season",
            "Response",
        ],
        higher_is_better=[
            "Severity_Spearman_Mean",
        ],
        lower_is_better=[
            "Severity_RMSE_Mean",
            "Severity_MAE_Mean",
            "Absolute_Severity_Mean_Bias_Mean",
        ],
        rank_name="Severity_Rank_Sum",
    )

    combined_ranked = add_rank_sum(
        combined_summary,
        group_columns=[
            "Season",
            "Response",
        ],
        higher_is_better=[
            "R2_All_Mean",
            "Spearman_All_Mean",
            "Occurrence_ROC_AUC_from_Expected_Response_Mean",
            "Occurrence_PR_AUC_from_Expected_Response_Mean",
        ],
        lower_is_better=[
            "RMSE_All_Mean",
            "MAE_All_Mean",
            "Absolute_Mean_Bias_Mean",
        ],
        rank_name="Combined_Rank_Sum",
    )

    occurrence_ranked.to_csv(
        OUTPUT_ROOT
        / "05_Occurrence_Candidate_Summary_and_Ranking.csv",
        index=False,
        encoding="utf-8-sig",
    )

    severity_ranked.to_csv(
        OUTPUT_ROOT
        / "06_Severity_Candidate_Summary_and_Ranking.csv",
        index=False,
        encoding="utf-8-sig",
    )

    combined_ranked.to_csv(
        OUTPUT_ROOT
        / "07_Combined_Candidate_Summary_and_Ranking.csv",
        index=False,
        encoding="utf-8-sig",
    )

    best_occurrence = (
        occurrence_ranked.sort_values(
            [
                "Season",
                "Response",
                "Occurrence_Rank_Sum_Position",
            ]
        )
        .groupby(
            ["Season", "Response"],
            as_index=False,
        )
        .first()
    )

    best_severity = (
        severity_ranked.sort_values(
            [
                "Season",
                "Response",
                "Severity_Rank_Sum_Position",
            ]
        )
        .groupby(
            ["Season", "Response"],
            as_index=False,
        )
        .first()
    )

    best_combined = (
        combined_ranked.sort_values(
            [
                "Season",
                "Response",
                "Combined_Rank_Sum_Position",
            ]
        )
        .groupby(
            ["Season", "Response"],
            as_index=False,
        )
        .first()
    )

    best_occurrence.to_csv(
        OUTPUT_ROOT
        / "08_Best_Occurrence_Config_by_Model.csv",
        index=False,
        encoding="utf-8-sig",
    )

    best_severity.to_csv(
        OUTPUT_ROOT
        / "09_Best_Severity_Config_by_Model.csv",
        index=False,
        encoding="utf-8-sig",
    )

    best_combined.to_csv(
        OUTPUT_ROOT
        / "10_Best_Combined_Config_by_Model.csv",
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
        "ForestArea_threshold_km2":
            FOREST_AREA_THRESHOLD_KM2,
        "Trees_per_RF": N_ESTIMATORS,
        "Temporal_folds": TEMPORAL_FOLDS,
        "Continuous_predictors":
            CONTINUOUS_PREDICTORS,
        "Country_handling":
            "One-hot dummy variables",
        "Validation":
            "Five-fold grouped temporal cross-validation",
        "Important_note": (
            "This is an exploratory hyperparameter comparison. "
            "The same temporal folds are used for ranking candidate "
            "configurations, so the reported results are not treated "
            "as an independent final performance estimate."
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
    log("Hyperparameter tuning completed successfully.")
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
            print(
                error_text,
                file=sys.stderr,
            )

        raise
