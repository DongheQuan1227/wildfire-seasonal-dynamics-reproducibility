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
from typing import Dict, List, Sequence

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
# Paths and global settings
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_ROOT = SCRIPT_DIR.parent
CODE_ROOT = MODEL_ROOT.parent
INPUT_CSV = CODE_ROOT / "00_Input_data" / "Base_Table_2001_2025.csv.bz2"

OUTPUT_ROOT = MODEL_ROOT / "03_RF_Predictor_Set_Comparison"
LOG_FILE = OUTPUT_ROOT / "predictor_set_comparison.log"

FOREST_AREA_THRESHOLD_KM2 = 2.5
RANDOM_SEED = 2026
N_ESTIMATORS = 50

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

LANDSCAPE = ["BD", "ND", "NE", "EVI", "PTC"]
METEOROLOGY = ["Temp", "Pre", "Rhum", "Wind", "SSRD"]
LIGHTNING = ["LtgProxy"]
FIRE_COMPONENTS = ["FFMC", "DMC", "DC"]
DROUGHT_ALL = ["SPEI1", "SPEI3", "SPEI6", "SPEI12", "SPEI24"]
DROUGHT_REDUCED = ["SPEI3", "SPEI12"]
TERRAIN = ["DEM", "Slope", "Aspect"]
ANTHRO_ALL = [
    "POP",
    "Dis_Farm",
    "Dis_Build",
    "Road_dens",
    "Dis_Railway",
    "Dis_Power",
]
ANTHRO_REDUCED = [
    "POP",
    "Dis_Farm",
    "Road_dens",
    "Dis_Railway",
]

# Five scientifically interpretable candidate sets.
# Country dummy variables are added to every model but are not included in VIF.
PREDICTOR_SETS: Dict[str, List[str]] = {
    # Same 28 continuous predictors used in the previous hurdle-RF comparison.
    "Baseline_28": (
        LANDSCAPE
        + METEOROLOGY
        + LIGHTNING
        + FIRE_COMPONENTS
        + DROUGHT_ALL
        + TERRAIN
        + ANTHRO_ALL
    ),

    # Keeps a seasonal drought metric and a longer-term drought metric.
    "Reduced_Drought_25": (
        LANDSCAPE
        + METEOROLOGY
        + LIGHTNING
        + FIRE_COMPONENTS
        + DROUGHT_REDUCED
        + TERRAIN
        + ANTHRO_ALL
    ),

    # Also removes building and power-distance variables that are strongly
    # redundant with population and other access variables.
    "Reduced_Drought_Access_23": (
        LANDSCAPE
        + METEOROLOGY
        + LIGHTNING
        + FIRE_COMPONENTS
        + DROUGHT_REDUCED
        + TERRAIN
        + ANTHRO_REDUCED
    ),

    # Replaces FFMC, DMC, and DC with the integrated FWI index.
    "Integrated_FWI_21": (
        LANDSCAPE
        + METEOROLOGY
        + LIGHTNING
        + ["FWI"]
        + DROUGHT_REDUCED
        + TERRAIN
        + ANTHRO_REDUCED
    ),

    # Uses raw meteorology and drought without any FWI-family predictor.
    "Raw_Climate_20": (
        LANDSCAPE
        + METEOROLOGY
        + LIGHTNING
        + DROUGHT_REDUCED
        + TERRAIN
        + ANTHRO_REDUCED
    ),
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


def regression_metrics(
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
        "Mean_Bias": float(predicted.mean() - observed.mean()),
        "Relative_Mean_Bias_Percent": (
            float(
                100.0
                * (predicted.mean() - observed.mean())
                / observed.mean()
            )
            if observed.mean() != 0
            else np.nan
        ),
        "RMSE_All": float(
            math.sqrt(mean_squared_error(observed, predicted))
        ),
        "MAE_All": float(
            mean_absolute_error(observed, predicted)
        ),
        "R2_All": float(r2_score(observed, predicted)),
        "Spearman_All": safe_spearman(observed, predicted),
        "Occurrence_ROC_AUC_from_Expected_Response": safe_roc_auc(
            occurrence,
            predicted,
        ),
        "Occurrence_PR_AUC_from_Expected_Response": safe_pr_auc(
            occurrence,
            predicted,
        ),
    }

    if positive_mask.any():
        obs_positive = observed[positive_mask]
        pred_positive = predicted[positive_mask]

        result.update(
            {
                "Positive_Observed_Mean": float(obs_positive.mean()),
                "Positive_Predicted_Mean": float(pred_positive.mean()),
                "RMSE_Positive_Expected_Response": float(
                    math.sqrt(
                        mean_squared_error(
                            obs_positive,
                            pred_positive,
                        )
                    )
                ),
                "MAE_Positive_Expected_Response": float(
                    mean_absolute_error(
                        obs_positive,
                        pred_positive,
                    )
                ),
                "Spearman_Positive_Expected_Response": safe_spearman(
                    obs_positive,
                    pred_positive,
                ),
            }
        )
    else:
        result.update(
            {
                "Positive_Observed_Mean": np.nan,
                "Positive_Predicted_Mean": np.nan,
                "RMSE_Positive_Expected_Response": np.nan,
                "MAE_Positive_Expected_Response": np.nan,
                "Spearman_Positive_Expected_Response": np.nan,
            }
        )

    if zero_mask.any():
        zero_predictions = predicted[zero_mask]

        result.update(
            {
                "Zero_Predicted_Mean": float(zero_predictions.mean()),
                "Zero_Predicted_Median": float(
                    np.median(zero_predictions)
                ),
                "Zero_Predicted_P95": float(
                    np.quantile(zero_predictions, 0.95)
                ),
            }
        )
    else:
        result.update(
            {
                "Zero_Predicted_Mean": np.nan,
                "Zero_Predicted_Median": np.nan,
                "Zero_Predicted_P95": np.nan,
            }
        )

    return result


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
        "Occurrence_ROC_AUC": safe_roc_auc(observed, probability),
        "Occurrence_PR_AUC": safe_pr_auc(observed, probability),
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


# =============================================================================
# Collinearity helpers
# =============================================================================

def calculate_vif_table(
    data: pd.DataFrame,
    predictors: Sequence[str],
) -> pd.DataFrame:
    complete = data[list(predictors)].dropna()

    if complete.empty:
        raise ValueError("No complete rows are available for VIF.")

    if len(complete) > 100_000:
        complete = complete.sample(
            n=100_000,
            random_state=RANDOM_SEED,
        )

    standard_deviation = complete.std(ddof=0)
    zero_variance = standard_deviation[
        standard_deviation <= 0
    ].index.tolist()

    usable = [
        predictor
        for predictor in predictors
        if predictor not in zero_variance
    ]

    standardized = (
        complete[usable] - complete[usable].mean()
    ) / complete[usable].std(ddof=0)

    correlation = np.corrcoef(
        standardized.to_numpy(dtype=float),
        rowvar=False,
    )
    correlation = (correlation + correlation.T) / 2.0
    np.fill_diagonal(correlation, 1.0)

    rows: List[Dict[str, object]] = []

    for index, predictor in enumerate(usable):
        other_indices = [
            i for i in range(len(usable)) if i != index
        ]

        r_yx = correlation[index, other_indices]
        r_xx = correlation[np.ix_(other_indices, other_indices)]

        try:
            inverse = np.linalg.inv(r_xx)
            inverse_method = "inverse"
        except np.linalg.LinAlgError:
            inverse = np.linalg.pinv(r_xx, rcond=1e-12)
            inverse_method = "pseudoinverse"

        r_squared = float(r_yx @ inverse @ r_yx.T)
        r_squared = max(0.0, min(1.0, r_squared))
        tolerance = 1.0 - r_squared

        vif = (
            np.inf
            if tolerance <= 1e-12
            else 1.0 / tolerance
        )

        rows.append(
            {
                "Predictor": predictor,
                "VIF": vif,
                "Tolerance": tolerance,
                "R_squared_against_other_predictors": r_squared,
                "Inverse_Method": inverse_method,
            }
        )

    for predictor in zero_variance:
        rows.append(
            {
                "Predictor": predictor,
                "VIF": np.inf,
                "Tolerance": 0.0,
                "R_squared_against_other_predictors": 1.0,
                "Inverse_Method": "zero_variance",
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["VIF", "Predictor"],
        ascending=[False, True],
    )


def maximum_absolute_spearman(
    data: pd.DataFrame,
    predictors: Sequence[str],
) -> Dict[str, object]:
    matrix = data[list(predictors)].corr(
        method="spearman",
        min_periods=100,
    )

    best_value = -1.0
    best_pair = ""

    for i, variable_1 in enumerate(predictors):
        for j in range(i + 1, len(predictors)):
            variable_2 = predictors[j]
            value = matrix.loc[variable_1, variable_2]

            if pd.notna(value) and abs(value) > best_value:
                best_value = abs(value)
                best_pair = f"{variable_1}~{variable_2}"

    return {
        "Maximum_Absolute_Spearman_Rho": best_value,
        "Maximum_Correlation_Pair": best_pair,
    }


# =============================================================================
# Data preparation
# =============================================================================

def validate_predictor_sets() -> None:
    for set_name, predictors in PREDICTOR_SETS.items():
        duplicates = sorted(
            {
                predictor
                for predictor in predictors
                if predictors.count(predictor) > 1
            }
        )

        if duplicates:
            raise ValueError(
                f"Duplicate predictors in {set_name}: {duplicates}"
            )


def prepare_base_data() -> pd.DataFrame:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Base table not found:\n{INPUT_CSV}"
        )

    data = pd.read_csv(INPUT_CSV)

    all_predictors = sorted(
        {
            predictor
            for predictors in PREDICTOR_SETS.values()
            for predictor in predictors
        }
    )

    required = {
        "GRID_UID",
        "Country",
        "GRID_ID",
        "Year",
        "Season",
        "Fire_Count",
        "Burned_Pixel_Count",
        "ForestPixelCount",
        *all_predictors,
    }

    missing = sorted(required - set(data.columns))

    if missing:
        raise ValueError(
            "The base table is missing required columns:\n"
            + "\n".join(missing)
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

    return data.loc[
        data["ForestArea_km2"] > FOREST_AREA_THRESHOLD_KM2
    ].copy()


def add_country_dummies(data: pd.DataFrame) -> pd.DataFrame:
    dummies = pd.get_dummies(
        data["Country"],
        prefix="Country",
        dtype=float,
    )

    for column in COUNTRY_DUMMY_COLUMNS:
        if column not in dummies.columns:
            dummies[column] = 0.0

    dummies = dummies[COUNTRY_DUMMY_COLUMNS]

    return pd.concat(
        [
            data.reset_index(drop=True),
            dummies.reset_index(drop=True),
        ],
        axis=1,
    )


# =============================================================================
# Hurdle-RF comparison
# =============================================================================

def fit_one_candidate(
    data: pd.DataFrame,
    season: int,
    response: str,
    set_name: str,
    continuous_predictors: Sequence[str],
) -> Dict[str, List[Dict[str, object]]]:
    model_predictors = (
        list(continuous_predictors)
        + COUNTRY_DUMMY_COLUMNS
    )

    season_data = data.loc[
        data["Season"] == season
    ].copy()

    missing_mask = season_data[
        model_predictors + [response]
    ].isna().any(axis=1)

    model_data = season_data.loc[
        ~missing_mask
    ].copy()

    fold_rows: List[Dict[str, object]] = []
    occurrence_rows: List[Dict[str, object]] = []
    severity_rows: List[Dict[str, object]] = []

    for fold in range(1, 6):
        training = model_data.loc[
            model_data["Temporal_Fold"] != fold
        ]
        validation = model_data.loc[
            model_data["Temporal_Fold"] == fold
        ]

        X_train = training[
            model_predictors
        ].to_numpy(dtype=np.float32)

        X_valid = validation[
            model_predictors
        ].to_numpy(dtype=np.float32)

        y_train = training[
            response
        ].to_numpy(dtype=np.float64)

        y_valid = validation[
            response
        ].to_numpy(dtype=np.float64)

        occurrence_train = (y_train > 0).astype(np.int8)
        occurrence_valid = (y_valid > 0).astype(np.int8)

        seed = (
            RANDOM_SEED
            + season * 100_000
            + (1 if response == "FCD" else 2) * 10_000
            + list(PREDICTOR_SETS).index(set_name) * 1_000
            + fold * 10
        )

        occurrence_model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_features="sqrt",
            min_samples_leaf=5,
            max_samples=0.70,
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

        probability = occurrence_model.predict_proba(
            X_valid
        )[:, 1]

        positive_train = y_train > 0
        positive_valid = y_valid > 0

        severity_model = RandomForestRegressor(
            n_estimators=N_ESTIMATORS,
            max_features="sqrt",
            min_samples_leaf=3,
            max_samples=0.80,
            bootstrap=True,
            n_jobs=-1,
            random_state=seed + 2,
            criterion="squared_error",
        )

        severity_model.fit(
            X_train[positive_train],
            y_train[positive_train],
        )

        conditional_severity = np.clip(
            severity_model.predict(X_valid),
            0.0,
            None,
        )

        expected_response = (
            probability * conditional_severity
        )

        years = ",".join(
            map(
                str,
                sorted(validation["Year"].unique()),
            )
        )

        fold_rows.append(
            {
                "Season": season,
                "Season_Label": SEASONS[season],
                "Response": response,
                "Predictor_Set": set_name,
                "Continuous_Predictor_Count": len(
                    continuous_predictors
                ),
                "Total_Model_Column_Count": len(
                    model_predictors
                ),
                "Fold": fold,
                "Validation_Years": years,
                "Rows_Used": len(model_data),
                "Rows_Excluded_for_Missing": int(
                    missing_mask.sum()
                ),
                **regression_metrics(
                    y_valid,
                    expected_response,
                ),
            }
        )

        occurrence_rows.append(
            {
                "Season": season,
                "Season_Label": SEASONS[season],
                "Response": response,
                "Predictor_Set": set_name,
                "Fold": fold,
                "Validation_Years": years,
                **occurrence_metrics(
                    occurrence_valid,
                    probability,
                ),
            }
        )

        if positive_valid.any():
            severity_observed = y_valid[positive_valid]
            severity_predicted = conditional_severity[
                positive_valid
            ]

            severity_rows.append(
                {
                    "Season": season,
                    "Season_Label": SEASONS[season],
                    "Response": response,
                    "Predictor_Set": set_name,
                    "Fold": fold,
                    "Validation_Years": years,
                    "Training_Positive_N": int(
                        positive_train.sum()
                    ),
                    "Validation_Positive_N": int(
                        positive_valid.sum()
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
                    "Severity_Spearman": safe_spearman(
                        severity_observed,
                        severity_predicted,
                    ),
                }
            )

    return {
        "fold": fold_rows,
        "occurrence": occurrence_rows,
        "severity": severity_rows,
    }


def summarize_numeric(
    data: pd.DataFrame,
    group_columns: Sequence[str],
    excluded_columns: Sequence[str],
) -> pd.DataFrame:
    numeric_columns = [
        column
        for column in data.select_dtypes(
            include=[np.number]
        ).columns
        if column not in set(excluded_columns)
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


# =============================================================================
# Main workflow
# =============================================================================

def main() -> int:
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text("", encoding="utf-8")

    log("=" * 78)
    log("Starting hurdle-RF predictor-set comparison")
    log(f"Input: {INPUT_CSV}")
    log(f"Trees per RF: {N_ESTIMATORS}")
    log("=" * 78)

    validate_predictor_sets()
    data = add_country_dummies(prepare_base_data())

    # Save predictor-set definitions.
    definition_rows = []

    for set_name, predictors in PREDICTOR_SETS.items():
        for order, predictor in enumerate(predictors, start=1):
            definition_rows.append(
                {
                    "Predictor_Set": set_name,
                    "Continuous_Predictor_Count": len(predictors),
                    "Order": order,
                    "Predictor": predictor,
                }
            )

    pd.DataFrame(definition_rows).to_csv(
        OUTPUT_ROOT / "00_Predictor_Set_Definitions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    all_fold_rows: List[Dict[str, object]] = []
    all_occurrence_rows: List[Dict[str, object]] = []
    all_severity_rows: List[Dict[str, object]] = []
    collinearity_rows: List[Dict[str, object]] = []
    vif_rows: List[Dict[str, object]] = []

    for season, season_label in SEASONS.items():
        season_data = data.loc[
            data["Season"] == season
        ]

        for set_name, predictors in PREDICTOR_SETS.items():
            # Collinearity diagnostics are response-independent.
            vif_table = calculate_vif_table(
                season_data,
                predictors,
            )

            max_corr = maximum_absolute_spearman(
                season_data,
                predictors,
            )

            collinearity_rows.append(
                {
                    "Season": season,
                    "Season_Label": season_label,
                    "Predictor_Set": set_name,
                    "Predictor_Count": len(predictors),
                    "Maximum_VIF": float(
                        vif_table["VIF"].max()
                    ),
                    "Median_VIF": float(
                        vif_table["VIF"].median()
                    ),
                    "VIF_ge_5_Count": int(
                        (vif_table["VIF"] >= 5).sum()
                    ),
                    "VIF_ge_10_Count": int(
                        (vif_table["VIF"] >= 10).sum()
                    ),
                    **max_corr,
                }
            )

            vif_table.insert(0, "Season", season)
            vif_table.insert(1, "Season_Label", season_label)
            vif_table.insert(2, "Predictor_Set", set_name)
            vif_rows.extend(
                vif_table.to_dict(orient="records")
            )

            for response in RESPONSES:
                log(
                    f"Running {season_label} {response} "
                    f"with {set_name} "
                    f"({len(predictors)} continuous predictors)"
                )

                result = fit_one_candidate(
                    data=data,
                    season=season,
                    response=response,
                    set_name=set_name,
                    continuous_predictors=predictors,
                )

                all_fold_rows.extend(result["fold"])
                all_occurrence_rows.extend(result["occurrence"])
                all_severity_rows.extend(result["severity"])

    fold_metrics = pd.DataFrame(all_fold_rows)
    occurrence_metrics_df = pd.DataFrame(all_occurrence_rows)
    severity_metrics_df = pd.DataFrame(all_severity_rows)

    fold_metrics.to_csv(
        OUTPUT_ROOT / "01_Predictor_Set_Fold_Metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    occurrence_metrics_df.to_csv(
        OUTPUT_ROOT / "02_Occurrence_Fold_Metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    severity_metrics_df.to_csv(
        OUTPUT_ROOT / "03_Severity_Fold_Metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    model_summary = summarize_numeric(
        fold_metrics,
        group_columns=[
            "Season",
            "Season_Label",
            "Response",
            "Predictor_Set",
        ],
        excluded_columns=[
            "Season",
            "Fold",
            "Continuous_Predictor_Count",
            "Total_Model_Column_Count",
            "Rows_Used",
            "Rows_Excluded_for_Missing",
        ],
    )

    occurrence_summary = summarize_numeric(
        occurrence_metrics_df,
        group_columns=[
            "Season",
            "Season_Label",
            "Response",
            "Predictor_Set",
        ],
        excluded_columns=["Season", "Fold"],
    )

    severity_summary = summarize_numeric(
        severity_metrics_df,
        group_columns=[
            "Season",
            "Season_Label",
            "Response",
            "Predictor_Set",
        ],
        excluded_columns=[
            "Season",
            "Fold",
            "Training_Positive_N",
            "Validation_Positive_N",
        ],
    )

    model_summary.to_csv(
        OUTPUT_ROOT / "04_Predictor_Set_Model_Summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    occurrence_summary.to_csv(
        OUTPUT_ROOT / "05_Occurrence_Summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    severity_summary.to_csv(
        OUTPUT_ROOT / "06_Severity_Summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(collinearity_rows).to_csv(
        OUTPUT_ROOT / "07_Residual_Collinearity_by_Set.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(vif_rows).to_csv(
        OUTPUT_ROOT / "08_VIF_by_Set_and_Season.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Difference from the 28-variable baseline.
    baseline = model_summary.loc[
        model_summary["Predictor_Set"] == "Baseline_28"
    ].copy()

    comparison_rows: List[Dict[str, object]] = []

    key_columns = [
        "Season",
        "Season_Label",
        "Response",
    ]

    numeric_summary_columns = [
        column
        for column in model_summary.select_dtypes(
            include=[np.number]
        ).columns
        if column not in {"Season", "Evaluation_Folds"}
    ]

    for _, candidate_row in model_summary.iterrows():
        key_mask = np.ones(len(baseline), dtype=bool)

        for key in key_columns:
            key_mask &= (
                baseline[key].to_numpy()
                == candidate_row[key]
            )

        baseline_match = baseline.loc[key_mask]

        if len(baseline_match) != 1:
            raise RuntimeError(
                "Could not uniquely match the baseline summary."
            )

        baseline_row = baseline_match.iloc[0]

        output_row = {
            key: candidate_row[key]
            for key in key_columns
        }

        output_row["Predictor_Set"] = candidate_row[
            "Predictor_Set"
        ]

        for column in numeric_summary_columns:
            output_row[
                f"Candidate_minus_Baseline_{column}"
            ] = (
                candidate_row[column]
                - baseline_row[column]
            )

        comparison_rows.append(output_row)

    pd.DataFrame(comparison_rows).to_csv(
        OUTPUT_ROOT / "09_Differences_from_Baseline_28.csv",
        index=False,
        encoding="utf-8-sig",
    )

    environment = {
        "Created": datetime.now().isoformat(timespec="seconds"),
        "Python": sys.version,
        "Platform": platform.platform(),
        "NumPy": np.__version__,
        "pandas": pd.__version__,
        "SciPy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "Input_file": str(INPUT_CSV),
        "Trees_per_RF": N_ESTIMATORS,
        "ForestArea_threshold_km2": FOREST_AREA_THRESHOLD_KM2,
        "Temporal_folds": TEMPORAL_FOLDS,
        "Country_handling": "One-hot dummy variables in every model",
        "Validation": (
            "Five-fold grouped temporal cross-validation; "
            "validation prevalence unchanged"
        ),
        "Purpose": (
            "Exploratory predictor-set comparison only. "
            "Final RF tuning will use more trees after a set is selected."
        ),
    }

    with (
        OUTPUT_ROOT / "Software_Environment_and_Method.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            environment,
            file,
            ensure_ascii=False,
            indent=2,
        )



    log("=" * 78)
    log("Predictor-set comparison completed successfully.")
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
