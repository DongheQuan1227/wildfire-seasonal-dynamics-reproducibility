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
# Paths and settings
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_ROOT = SCRIPT_DIR.parent
CODE_ROOT = MODEL_ROOT.parent
INPUT_CSV = CODE_ROOT / "00_Input_data" / "Base_Table_2001_2025.csv.bz2"

OUTPUT_ROOT = MODEL_ROOT / "04_Forest_Structure_Predictor_Comparison"
LOG_FILE = OUTPUT_ROOT / "forest_structure_comparison.log"

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

FOREST_STRUCTURE = ["BD", "ND", "NE", "EVI", "PTC"]

OTHER_RAW_CLIMATE_PREDICTORS = [
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

# Final targeted comparison around the current Raw_Climate_20 candidate.
PREDICTOR_SETS: Dict[str, List[str]] = {
    "All_Forest_Variables_20": (
        ["BD", "ND", "NE", "EVI", "PTC"]
        + OTHER_RAW_CLIMATE_PREDICTORS
    ),
    "Drop_BD_19": (
        ["ND", "NE", "EVI", "PTC"]
        + OTHER_RAW_CLIMATE_PREDICTORS
    ),
    "Drop_ND_19": (
        ["BD", "NE", "EVI", "PTC"]
        + OTHER_RAW_CLIMATE_PREDICTORS
    ),
    "Drop_NE_19": (
        ["BD", "ND", "EVI", "PTC"]
        + OTHER_RAW_CLIMATE_PREDICTORS
    ),
    "Drop_PTC_19": (
        ["BD", "ND", "NE", "EVI"]
        + OTHER_RAW_CLIMATE_PREDICTORS
    ),
    "EVI_PTC_Only_17": (
        ["EVI", "PTC"]
        + OTHER_RAW_CLIMATE_PREDICTORS
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
                "Zero_Predicted_P95": float(
                    np.quantile(zero_predictions, 0.95)
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
# VIF helpers
# =============================================================================

def calculate_vif_table(
    data: pd.DataFrame,
    predictors: Sequence[str],
    sample_seed: int,
) -> pd.DataFrame:
    complete = data[list(predictors)].dropna()

    if complete.empty:
        raise ValueError("No complete rows are available for VIF.")

    if len(complete) > 100_000:
        complete = complete.sample(
            n=100_000,
            random_state=sample_seed,
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
        vif = np.inf if tolerance <= 1e-12 else 1.0 / tolerance

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


# =============================================================================
# Input preparation and forest-variable audit
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


def prepare_data() -> pd.DataFrame:
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

    data = data.loc[
        data["ForestArea_km2"] > FOREST_AREA_THRESHOLD_KM2
    ].copy()

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


def write_forest_variable_audit(data: pd.DataFrame) -> None:
    audit = data[
        ["GRID_UID", "Year", "Season", *FOREST_STRUCTURE]
    ].copy()

    audit["BD_ND_NE_Sum"] = (
        audit["BD"] + audit["ND"] + audit["NE"]
    )

    descriptive_rows: List[Dict[str, object]] = []

    for season_code, season_label in {
        0: "All_Seasons",
        **SEASONS,
    }.items():
        subset = (
            audit
            if season_code == 0
            else audit.loc[audit["Season"] == season_code]
        )

        for variable in [
            "BD",
            "ND",
            "NE",
            "EVI",
            "PTC",
            "BD_ND_NE_Sum",
        ]:
            values = subset[variable]

            descriptive_rows.append(
                {
                    "Season": season_code,
                    "Season_Label": season_label,
                    "Variable": variable,
                    "N": int(values.notna().sum()),
                    "Missing_N": int(values.isna().sum()),
                    "Zero_N": int((values == 0).sum()),
                    "Minimum": float(values.min()),
                    "Q05": float(values.quantile(0.05)),
                    "Median": float(values.median()),
                    "Mean": float(values.mean()),
                    "Q95": float(values.quantile(0.95)),
                    "Maximum": float(values.max()),
                    "Unique_N": int(values.nunique(dropna=True)),
                }
            )

    pd.DataFrame(descriptive_rows).to_csv(
        OUTPUT_ROOT / "01_Forest_Variable_Descriptive_Audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    correlation_rows: List[Dict[str, object]] = []

    variables = [
        "BD",
        "ND",
        "NE",
        "EVI",
        "PTC",
        "BD_ND_NE_Sum",
    ]

    for season_code, season_label in {
        0: "All_Seasons",
        **SEASONS,
    }.items():
        subset = (
            audit
            if season_code == 0
            else audit.loc[audit["Season"] == season_code]
        )

        matrix = subset[variables].corr(
            method="spearman",
            min_periods=100,
        )

        for i, variable_1 in enumerate(variables):
            for j in range(i + 1, len(variables)):
                variable_2 = variables[j]

                correlation_rows.append(
                    {
                        "Season": season_code,
                        "Season_Label": season_label,
                        "Variable_1": variable_1,
                        "Variable_2": variable_2,
                        "Spearman_Rho": matrix.loc[
                            variable_1,
                            variable_2,
                        ],
                    }
                )

    pd.DataFrame(correlation_rows).to_csv(
        OUTPUT_ROOT / "02_Forest_Variable_Spearman_Audit.csv",
        index=False,
        encoding="utf-8-sig",
    )


# =============================================================================
# Hurdle RF
# =============================================================================

def fit_candidate(
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

    set_index = list(PREDICTOR_SETS).index(set_name)

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
            + set_index * 1_000
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

        expected_response = probability * conditional_severity

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
            observed_severity = y_valid[positive_valid]
            predicted_severity = conditional_severity[
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
                        observed_severity.mean()
                    ),
                    "Severity_Predicted_Mean": float(
                        predicted_severity.mean()
                    ),
                    "Severity_RMSE": float(
                        math.sqrt(
                            mean_squared_error(
                                observed_severity,
                                predicted_severity,
                            )
                        )
                    ),
                    "Severity_MAE": float(
                        mean_absolute_error(
                            observed_severity,
                            predicted_severity,
                        )
                    ),
                    "Severity_Spearman": safe_spearman(
                        observed_severity,
                        predicted_severity,
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
    excluded_numeric: Sequence[str],
) -> pd.DataFrame:
    numeric_columns = [
        column
        for column in data.select_dtypes(
            include=[np.number]
        ).columns
        if column not in set(excluded_numeric)
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
# Main
# =============================================================================

def main() -> int:
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text("", encoding="utf-8")

    log("=" * 78)
    log("Starting forest-structure predictor comparison")
    log(f"Input: {INPUT_CSV}")
    log(f"Trees per RF: {N_ESTIMATORS}")
    log("=" * 78)

    validate_predictor_sets()
    data = prepare_data()
    write_forest_variable_audit(data)

    definition_rows: List[Dict[str, object]] = []

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

    fold_rows: List[Dict[str, object]] = []
    occurrence_rows: List[Dict[str, object]] = []
    severity_rows: List[Dict[str, object]] = []
    vif_rows: List[Dict[str, object]] = []
    vif_summary_rows: List[Dict[str, object]] = []

    for season, season_label in SEASONS.items():
        season_data = data.loc[data["Season"] == season]

        for set_name, predictors in PREDICTOR_SETS.items():
            vif = calculate_vif_table(
                season_data,
                predictors,
                sample_seed=RANDOM_SEED + season,
            )

            vif.insert(0, "Season", season)
            vif.insert(1, "Season_Label", season_label)
            vif.insert(2, "Predictor_Set", set_name)

            vif_rows.extend(vif.to_dict(orient="records"))

            vif_summary_rows.append(
                {
                    "Season": season,
                    "Season_Label": season_label,
                    "Predictor_Set": set_name,
                    "Predictor_Count": len(predictors),
                    "Maximum_VIF": float(vif["VIF"].max()),
                    "Median_VIF": float(vif["VIF"].median()),
                    "VIF_ge_5_Count": int(
                        (vif["VIF"] >= 5).sum()
                    ),
                    "VIF_ge_10_Count": int(
                        (vif["VIF"] >= 10).sum()
                    ),
                    "Maximum_VIF_Predictor": str(
                        vif.iloc[0]["Predictor"]
                    ),
                }
            )

            for response in RESPONSES:
                log(
                    f"Running {season_label} {response}: "
                    f"{set_name}"
                )

                result = fit_candidate(
                    data=data,
                    season=season,
                    response=response,
                    set_name=set_name,
                    continuous_predictors=predictors,
                )

                fold_rows.extend(result["fold"])
                occurrence_rows.extend(result["occurrence"])
                severity_rows.extend(result["severity"])

    fold_df = pd.DataFrame(fold_rows)
    occurrence_df = pd.DataFrame(occurrence_rows)
    severity_df = pd.DataFrame(severity_rows)

    fold_df.to_csv(
        OUTPUT_ROOT / "03_Model_Fold_Metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    occurrence_df.to_csv(
        OUTPUT_ROOT / "04_Occurrence_Fold_Metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    severity_df.to_csv(
        OUTPUT_ROOT / "05_Severity_Fold_Metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    model_summary = summarize_numeric(
        fold_df,
        group_columns=[
            "Season",
            "Season_Label",
            "Response",
            "Predictor_Set",
        ],
        excluded_numeric=[
            "Season",
            "Fold",
            "Continuous_Predictor_Count",
            "Rows_Used",
            "Rows_Excluded_for_Missing",
        ],
    )

    occurrence_summary = summarize_numeric(
        occurrence_df,
        group_columns=[
            "Season",
            "Season_Label",
            "Response",
            "Predictor_Set",
        ],
        excluded_numeric=["Season", "Fold"],
    )

    severity_summary = summarize_numeric(
        severity_df,
        group_columns=[
            "Season",
            "Season_Label",
            "Response",
            "Predictor_Set",
        ],
        excluded_numeric=[
            "Season",
            "Fold",
            "Training_Positive_N",
            "Validation_Positive_N",
        ],
    )

    model_summary.to_csv(
        OUTPUT_ROOT / "06_Model_Summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    occurrence_summary.to_csv(
        OUTPUT_ROOT / "07_Occurrence_Summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    severity_summary.to_csv(
        OUTPUT_ROOT / "08_Severity_Summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(vif_summary_rows).to_csv(
        OUTPUT_ROOT / "09_VIF_Summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(vif_rows).to_csv(
        OUTPUT_ROOT / "10_VIF_Detail.csv",
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
        "Country_handling": "One-hot dummy variables",
        "Validation": "Five-fold grouped temporal cross-validation",
        "Purpose": (
            "Final targeted predictor-screening step before RF "
            "hyperparameter tuning."
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
    log("Forest-structure comparison completed successfully.")
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
