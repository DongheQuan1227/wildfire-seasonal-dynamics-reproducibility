# -*- coding: utf-8 -*-
"""
Generate final model-comparison tables, figure data, manuscript figures,
and a locked interpretation summary from Model Comparison Steps 02-05.

Recommended location
--------------------
<REPOSITORY_ROOT>/10_Model_comparison/00_Script/
    06_generate_model_comparison_tables_figures.py

This script only reads previously locked model-comparison outputs. It does not
fit, tune, select, or predict any model; it does not rerun bootstrap or Moran's I.
All figure text and output labels are in English for journal use.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


# =============================================================================
# Locked configuration
# =============================================================================

CODE_VERSION = "2026-06-30_MODEL_COMPARISON_TABLES_FIGURES_V1"
CODE_ROOT = Path(
    os.environ.get(
        "MODEL_COMPARISON_CODE_ROOT",
        str(Path(__file__).resolve().parents[2]),
    )
).resolve()
ROOT = CODE_ROOT / "10_Model_comparison"

STEP02 = ROOT / "02_Unified_Prediction_Metrics"
STEP03 = ROOT / "03_Paired_Cluster_Bootstrap"
STEP04 = ROOT / "04_Residual_Spatial_Autocorrelation_Comparison"
STEP05 = ROOT / "05_Country_Block_Annual_Performance"
OUTPUT = ROOT / "06_Model_Comparison_Tables_Figures"
FIGURE_DIR = OUTPUT / "Figures"
FIGURE_DATA_DIR = OUTPUT / "Figure_Data"
SUPPLEMENT_DIR = OUTPUT / "Supplementary_Tables"
LOG_FILE = OUTPUT / "model_comparison_tables_figures.log"

EXPECTED_VERSIONS = {
    "Step02": "2026-06-30_MODEL_COMPARISON_UNIFIED_METRICS_V1A_ZERO_POSITIVE_FOLD_COMPATIBILITY",
    "Step03": "2026-06-30_MODEL_COMPARISON_PAIRED_GRID_CLUSTER_BOOTSTRAP_V1",
    "Step04": "2026-06-30_MODEL_COMPARISON_RESIDUAL_MORAN_V1",
    "Step05": "2026-06-30_MODEL_COMPARISON_COUNTRY_BLOCK_ANNUAL_V1",
}

INPUTS = {
    "Step02_Method": STEP02 / "00_Method_Definition.json",
    "Step02_Overall": STEP02 / "02_Overall_Unified_Prediction_Metrics.csv",
    "Step02_Fold": STEP02 / "03_Fold_Level_Unified_Prediction_Metrics.csv",
    "Step02_Extreme": STEP02 / "06_Extreme_Event_Performance.csv",
    "Step02_Manifest": STEP02 / "09_Output_Manifest.csv",
    "Step03_Method": STEP03 / "00_Method_Definition.json",
    "Step03_Bootstrap": STEP03 / "03_Paired_Bootstrap_Difference_Summary.csv",
    "Step03_Nonadditive": STEP03 / "06_Descriptive_Nonadditive_Metric_Differences.csv",
    "Step03_Stable_Wins": STEP03 / "07_Stable_Win_Count_Summary.csv",
    "Step03_Manifest": STEP03 / "12_Output_Manifest.csv",
    "Step04_Method": STEP04 / "00_Method_Definition.json",
    "Step04_Primary_Moran": STEP04 / "02_Combined_Primary_Main_Sample_Morans_I.csv",
    "Step04_KNN_Stability": STEP04 / "07_KNN_Stability_Summary.csv",
    "Step04_Tradeoff": STEP04 / "11_Paired_Prediction_Moran_Tradeoff.csv",
    "Step04_Moran_Wins": STEP04 / "12b_Primary_Moran_Lowest_Count_Summary.csv",
    "Step04_Manifest": STEP04 / "13_Output_Manifest.csv",
    "Step05_Method": STEP05 / "00_Method_Definition.json",
    "Step05_Country": STEP05 / "02_Country_Level_Performance.csv",
    "Step05_Block": STEP05 / "03_Spatial_Block_Level_Performance.csv",
    "Step05_Annual_Aggregate": STEP05 / "05_Annual_Aggregated_Predictions.csv",
    "Step05_Annual_Series": STEP05 / "06_Annual_Series_Performance.csv",
    "Step05_Country_Annual_Series": STEP05 / "08_Country_Annual_Series_Performance.csv",
    "Step05_Block_Annual_Series": STEP05 / "10_Spatial_Block_Annual_Series_Performance.csv",
    "Step05_Subgroup_Wins": STEP05 / "12_Subgroup_Model_Win_Summary.csv",
    "Step05_Annual_Wins": STEP05 / "14_Annual_Series_Model_Win_Summary.csv",
    "Step05_Manifest": STEP05 / "17_Output_Manifest.csv",
}

MODELS = ["Baseline_NB1", "Final_Regression", "Final_Hurdle_RF"]
MODEL_LABELS = {
    "Baseline_NB1": "Baseline NB1",
    "Final_Regression": "Final Regression",
    "Final_Hurdle_RF": "Hurdle RF",
}
MODEL_COLORS = {
    "Baseline_NB1": "#777777",
    "Final_Regression": "#2F6B9A",
    "Final_Hurdle_RF": "#D47A28",
}
MODEL_MARKERS = {
    "Baseline_NB1": "o",
    "Final_Regression": "s",
    "Final_Hurdle_RF": "^",
}
VALIDATIONS = ["Temporal", "Spatial"]
SEASONS = [1, 2, 3]
SEASON_LABELS = {1: "Spring", 2: "Summer", 3: "Autumn"}
RESPONSES = ["Fire_Count", "Burned_Pixel_Count"]
RESPONSE_LABELS = {"Fire_Count": "FCD", "Burned_Pixel_Count": "BAD"}
COMBINATION_ORDER = [
    (season, response)
    for season in SEASONS
    for response in RESPONSES
]
COMBINATION_LABELS = [
    f"{SEASON_LABELS[s]} {RESPONSE_LABELS[r]}"
    for s, r in COMBINATION_ORDER
]

PRIMARY_METRICS = [
    "Rate_RMSE",
    "Rate_MAE",
    "Occurrence_Brier_Score",
    "Positive_Conditional_Rate_RMSE",
    "Positive_Conditional_Rate_MAE",
]

HASH_CHUNK_SIZE = 1024 * 1024
ATOL = 1e-12


# =============================================================================
# Utilities
# =============================================================================

def log(message: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as file:
        file.write(line + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(HASH_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, data: Mapping[str, object]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def manifest_expected_hash(manifest_path: Path, filename: str) -> str:
    manifest = pd.read_csv(manifest_path)
    file_col = "File" if "File" in manifest.columns else "Output_File"
    relative_col = "Relative_Path" if "Relative_Path" in manifest.columns else file_col
    match = manifest.loc[
        (manifest[file_col].astype(str) == filename)
        | (manifest[relative_col].astype(str) == filename)
    ]
    if len(match) != 1:
        return ""
    return str(match.iloc[0]["SHA256"])


def save_csv(data: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False, encoding="utf-8-sig")


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 120,
            "savefig.dpi": 400,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> List[Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    paths = [
        FIGURE_DIR / f"{stem}.png",
        FIGURE_DIR / f"{stem}.pdf",
        FIGURE_DIR / f"{stem}.tif",
    ]
    fig.savefig(paths[0], bbox_inches="tight", dpi=400)
    fig.savefig(paths[1], bbox_inches="tight")
    fig.savefig(
        paths[2],
        bbox_inches="tight",
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)
    return paths


def rank_lower(data: pd.DataFrame, metric: str, groups: Sequence[str]) -> pd.Series:
    return data.groupby(list(groups), dropna=False)[metric].rank(
        ascending=True, method="min"
    ).astype("Int64")


def rank_higher(data: pd.DataFrame, metric: str, groups: Sequence[str]) -> pd.Series:
    return data.groupby(list(groups), dropna=False)[metric].rank(
        ascending=False, method="min"
    ).astype("Int64")


def safe_winner(values: Mapping[str, float], lower_is_better: bool = True) -> str:
    finite = {key: value for key, value in values.items() if np.isfinite(value)}
    if not finite:
        return "Unavailable"
    target = min(finite.values()) if lower_is_better else max(finite.values())
    winners = sorted(
        key for key, value in finite.items() if np.isclose(value, target, atol=ATOL, rtol=0)
    )
    return ";".join(winners)


# =============================================================================
# Table construction
# =============================================================================

def build_overall_main_table(
    overall: pd.DataFrame,
    moran: pd.DataFrame,
    annual_series: pd.DataFrame,
) -> pd.DataFrame:
    selected = overall[
        [
            "Validation_Scheme",
            "Season",
            "Season_Label",
            "Count_Response",
            "Rate_Scale_Name",
            "Model",
            "Observed_Positive_Fraction",
            "Rate_RMSE",
            "Rate_MAE",
            "Rate_R2",
            "Rate_Spearman",
            "Rate_Mean_Bias",
            "Occurrence_Brier_Score",
            "Occurrence_PR_AUC",
            "Occurrence_ROC_AUC",
            "Positive_Conditional_Rate_RMSE",
            "Positive_Conditional_Rate_MAE",
            "Rate_Binned_Calibration_MAE",
        ]
    ].copy()

    moran_selected = moran[
        [
            "Validation_Scheme",
            "Season_Code",
            "Count_Response",
            "Model_Role",
            "Moran_I",
            "Combined_Primary_Global_BH_FDR",
        ]
    ].rename(columns={"Season_Code": "Season", "Model_Role": "Model"})

    annual_selected = annual_series[
        [
            "Validation_Scheme",
            "Season",
            "Count_Response",
            "Model",
            "Annual_Exposure_Weighted_Rate_RMSE",
            "Annual_Exposure_Weighted_Rate_MAE",
            "Annual_Exposure_Weighted_Rate_R2",
            "Annual_Exposure_Weighted_Rate_Spearman",
            "Annual_Exposure_Weighted_Rate_Predicted_minus_Observed_Slope",
            "Annual_Exposure_Weighted_Rate_Top3_Year_Overlap_Fraction",
            "Annual_Exposure_Weighted_Rate_Top5_Year_Overlap_Fraction",
        ]
    ]

    merged = selected.merge(
        moran_selected,
        on=["Validation_Scheme", "Season", "Count_Response", "Model"],
        how="inner",
        validate="one_to_one",
    ).merge(
        annual_selected,
        on=["Validation_Scheme", "Season", "Count_Response", "Model"],
        how="inner",
        validate="one_to_one",
    )

    groups = ["Validation_Scheme", "Season", "Count_Response"]
    for metric in [
        "Rate_RMSE",
        "Rate_MAE",
        "Occurrence_Brier_Score",
        "Positive_Conditional_Rate_RMSE",
        "Positive_Conditional_Rate_MAE",
        "Moran_I",
        "Annual_Exposure_Weighted_Rate_RMSE",
        "Annual_Exposure_Weighted_Rate_MAE",
    ]:
        merged[f"{metric}_Rank_Lower_Is_Better"] = rank_lower(merged, metric, groups)
    for metric in ["Rate_R2", "Rate_Spearman", "Occurrence_PR_AUC"]:
        merged[f"{metric}_Rank_Higher_Is_Better"] = rank_higher(merged, metric, groups)

    merged["Model_Label"] = merged["Model"].map(MODEL_LABELS)
    merged["_Validation_Order"] = merged["Validation_Scheme"].map({"Temporal": 0, "Spatial": 1})
    merged["_Response_Order"] = merged["Count_Response"].map({"Fire_Count": 0, "Burned_Pixel_Count": 1})
    merged["_Model_Order"] = merged["Model"].map({model: index for index, model in enumerate(MODELS)})
    merged = merged.sort_values(["_Validation_Order", "Season", "_Response_Order", "_Model_Order"]).drop(
        columns=["_Validation_Order", "_Response_Order", "_Model_Order"]
    )
    return merged.reset_index(drop=True)


def build_rf_regression_paired_table(
    bootstrap: pd.DataFrame,
    tradeoff: pd.DataFrame,
    overall: pd.DataFrame,
    annual_series: pd.DataFrame,
) -> pd.DataFrame:
    pair_name = "Final_Hurdle_RF_vs_Final_Regression"
    metrics = [
        "Rate_RMSE",
        "Rate_MAE",
        "Occurrence_Brier_Score",
        "Positive_Conditional_Rate_RMSE",
        "Positive_Conditional_Rate_MAE",
    ]
    selected = bootstrap.loc[
        (bootstrap["Model_Pair"] == pair_name)
        & (bootstrap["Metric"].isin(metrics))
    ].copy()

    pivot = selected.pivot_table(
        index=[
            "Validation_Scheme",
            "Season",
            "Season_Label",
            "Count_Response",
        ],
        columns="Metric",
        values=[
            "Candidate_Point_Estimate",
            "Reference_Point_Estimate",
            "Candidate_Minus_Reference",
            "Percent_Improvement_Positive_Is_Better",
            "Percent_Improvement_CI_Lower",
            "Percent_Improvement_CI_Upper",
            "Difference_CI_Lower",
            "Difference_CI_Upper",
            "Probability_Candidate_Better",
            "Stable_Conclusion",
        ],
        aggfunc="first",
    )
    pivot.columns = [f"{metric}_{field}" for field, metric in pivot.columns]
    pivot = pivot.reset_index()

    tradeoff_selected = tradeoff.loc[
        tradeoff["Model_Pair"] == pair_name,
        [
            "Validation_Scheme",
            "Season_Code",
            "Count_Response",
            "Candidate_Moran_I",
            "Reference_Moran_I",
            "Candidate_Minus_Reference_Moran_I",
            "Moran_I_Percent_Reduction_Positive_Favors_Candidate",
            "Primary_RMSE_Moran_Tradeoff_Category",
        ],
    ].rename(columns={"Season_Code": "Season"})
    pivot = pivot.merge(
        tradeoff_selected,
        on=["Validation_Scheme", "Season", "Count_Response"],
        how="left",
        validate="one_to_one",
    )

    annual = annual_series.loc[
        annual_series["Model"].isin(["Final_Regression", "Final_Hurdle_RF"]),
        [
            "Validation_Scheme",
            "Season",
            "Count_Response",
            "Model",
            "Annual_Exposure_Weighted_Rate_RMSE",
            "Annual_Exposure_Weighted_Rate_MAE",
        ],
    ]
    annual_wide = annual.pivot(
        index=["Validation_Scheme", "Season", "Count_Response"],
        columns="Model",
        values=[
            "Annual_Exposure_Weighted_Rate_RMSE",
            "Annual_Exposure_Weighted_Rate_MAE",
        ],
    )
    annual_wide.columns = [f"{metric}_{model}" for metric, model in annual_wide.columns]
    annual_wide = annual_wide.reset_index()
    annual_wide["Annual_Rate_RMSE_Winner"] = annual_wide.apply(
        lambda row: safe_winner(
            {
                "Final_Regression": row[
                    "Annual_Exposure_Weighted_Rate_RMSE_Final_Regression"
                ],
                "Final_Hurdle_RF": row[
                    "Annual_Exposure_Weighted_Rate_RMSE_Final_Hurdle_RF"
                ],
            }
        ),
        axis=1,
    )
    annual_wide["Annual_Rate_MAE_Winner"] = annual_wide.apply(
        lambda row: safe_winner(
            {
                "Final_Regression": row[
                    "Annual_Exposure_Weighted_Rate_MAE_Final_Regression"
                ],
                "Final_Hurdle_RF": row[
                    "Annual_Exposure_Weighted_Rate_MAE_Final_Hurdle_RF"
                ],
            }
        ),
        axis=1,
    )
    pivot = pivot.merge(
        annual_wide,
        on=["Validation_Scheme", "Season", "Count_Response"],
        how="left",
        validate="one_to_one",
    )

    overall_pair = overall.loc[
        overall["Model"].isin(["Final_Regression", "Final_Hurdle_RF"]),
        [
            "Validation_Scheme",
            "Season",
            "Count_Response",
            "Model",
            "Rate_Binned_Calibration_MAE",
            "Occurrence_PR_AUC",
        ],
    ]
    overall_wide = overall_pair.pivot(
        index=["Validation_Scheme", "Season", "Count_Response"],
        columns="Model",
        values=["Rate_Binned_Calibration_MAE", "Occurrence_PR_AUC"],
    )
    overall_wide.columns = [f"{metric}_{model}" for metric, model in overall_wide.columns]
    overall_wide = overall_wide.reset_index()
    overall_wide["Rate_Calibration_Winner"] = overall_wide.apply(
        lambda row: safe_winner(
            {
                "Final_Regression": row[
                    "Rate_Binned_Calibration_MAE_Final_Regression"
                ],
                "Final_Hurdle_RF": row[
                    "Rate_Binned_Calibration_MAE_Final_Hurdle_RF"
                ],
            }
        ),
        axis=1,
    )
    overall_wide["Occurrence_PR_AUC_Winner"] = overall_wide.apply(
        lambda row: safe_winner(
            {
                "Final_Regression": row["Occurrence_PR_AUC_Final_Regression"],
                "Final_Hurdle_RF": row["Occurrence_PR_AUC_Final_Hurdle_RF"],
            },
            lower_is_better=False,
        ),
        axis=1,
    )
    pivot = pivot.merge(
        overall_wide,
        on=["Validation_Scheme", "Season", "Count_Response"],
        how="left",
        validate="one_to_one",
    )

    def interpretation(row: pd.Series) -> str:
        rmse = row.get("Rate_RMSE_Stable_Conclusion", "")
        moran_delta = float(row["Candidate_Minus_Reference_Moran_I"])
        annual_winner = row["Annual_Rate_RMSE_Winner"]
        if rmse == "Candidate_stably_better" and moran_delta < 0:
            return "RF has stably lower grid-level RMSE and lower residual Moran's I."
        if rmse == "Candidate_stably_better" and moran_delta > 0:
            if annual_winner == "Final_Regression":
                return (
                    "RF has stably lower grid-level RMSE, whereas regression has lower "
                    "residual Moran's I and lower annual-series RMSE."
                )
            return (
                "RF has stably lower grid-level RMSE but higher residual Moran's I; "
                "the models are complementary."
            )
        if rmse == "Reference_stably_better":
            return "Regression has stably lower grid-level RMSE and lower residual Moran's I."
        if moran_delta > 0:
            return "Grid-level RMSE does not differ stably; regression has lower residual Moran's I."
        if moran_delta < 0:
            return "Grid-level RMSE does not differ stably; RF has lower residual Moran's I."
        return "No clear integrated distinction."

    pivot["Integrated_Interpretation"] = pivot.apply(interpretation, axis=1)
    pivot["Rate_Scale_Name"] = pivot["Count_Response"].map(RESPONSE_LABELS)
    pivot["_Validation_Order"] = pivot["Validation_Scheme"].map({"Temporal": 0, "Spatial": 1})
    pivot["_Response_Order"] = pivot["Count_Response"].map({"Fire_Count": 0, "Burned_Pixel_Count": 1})
    pivot = pivot.sort_values(["_Validation_Order", "Season", "_Response_Order"]).drop(
        columns=["_Validation_Order", "_Response_Order"]
    )
    return pivot.reset_index(drop=True)


def build_integrated_recommendation(paired: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for _, row in paired.iterrows():
        rmse_conclusion = str(row["Rate_RMSE_Stable_Conclusion"])
        positive_conclusion = str(row["Positive_Conditional_Rate_RMSE_Stable_Conclusion"])
        moran_delta = float(row["Candidate_Minus_Reference_Moran_I"])
        if rmse_conclusion == "Candidate_stably_better":
            prediction_priority = "Final_Hurdle_RF"
        elif rmse_conclusion == "Reference_stably_better":
            prediction_priority = "Final_Regression"
        else:
            prediction_priority = "No_stable_difference"
        residual_priority = "Final_Hurdle_RF" if moran_delta < 0 else "Final_Regression"
        severity_priority = (
            "Final_Hurdle_RF"
            if positive_conclusion == "Candidate_stably_better"
            else (
                "Final_Regression"
                if positive_conclusion == "Reference_stably_better"
                else "No_stable_difference"
            )
        )
        rows.append(
            {
                "Validation_Scheme": row["Validation_Scheme"],
                "Season": int(row["Season"]),
                "Season_Label": row["Season_Label"],
                "Count_Response": row["Count_Response"],
                "Rate_Scale_Name": row["Rate_Scale_Name"],
                "Grid_Level_RMSE_Priority": prediction_priority,
                "Conditional_Positive_Severity_Priority": severity_priority,
                "Residual_Spatial_Structure_Priority": residual_priority,
                "Annual_Rate_RMSE_Winner": row["Annual_Rate_RMSE_Winner"],
                "Annual_Rate_MAE_Winner": row["Annual_Rate_MAE_Winner"],
                "Rate_Calibration_Winner": row["Rate_Calibration_Winner"],
                "Occurrence_PR_AUC_Winner": row["Occurrence_PR_AUC_Winner"],
                "Integrated_Interpretation": row["Integrated_Interpretation"],
                "Model_Selection_Status": (
                    "Descriptive_complementarity_not_single_model_selection"
                ),
            }
        )
    return pd.DataFrame(rows)


def build_regional_win_summary(
    subgroup_wins: pd.DataFrame,
    annual_wins: pd.DataFrame,
) -> pd.DataFrame:
    subgroup = (
        subgroup_wins.groupby(
            ["Subgroup_Type", "Validation_Scheme", "Model"], as_index=False
        )
        .agg(
            Metric_Comparison_N=("Comparison_N", "sum"),
            Metric_Rank1_N=("Rank1_N", "sum"),
            Mean_of_Metric_Mean_Ranks=("Mean_Rank", "mean"),
        )
        .assign(Result_Family="Grid_Level_Subgroup")
        .rename(columns={"Subgroup_Type": "Aggregation_Type"})
    )
    annual = (
        annual_wins.groupby(
            ["Annual_Group_Type", "Validation_Scheme", "Model"], as_index=False
        )
        .agg(
            Metric_Comparison_N=("Comparison_N", "sum"),
            Metric_Rank1_N=("Rank1_N", "sum"),
            Mean_of_Metric_Mean_Ranks=("Mean_Rank", "mean"),
        )
        .assign(Result_Family="Annual_Series")
        .rename(columns={"Annual_Group_Type": "Aggregation_Type"})
    )
    return pd.concat([subgroup, annual], ignore_index=True).sort_values(
        ["Result_Family", "Aggregation_Type", "Validation_Scheme", "Model"]
    ).reset_index(drop=True)


def build_key_statistics(
    overall_main: pd.DataFrame,
    bootstrap: pd.DataFrame,
    moran_wins: pd.DataFrame,
    subgroup_wins: pd.DataFrame,
    annual_wins: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    rf_pair = bootstrap.loc[
        bootstrap["Model_Pair"] == "Final_Hurdle_RF_vs_Final_Regression"
    ]
    for validation in VALIDATIONS:
        for metric in [
            "Rate_RMSE",
            "Rate_MAE",
            "Positive_Conditional_Rate_RMSE",
            "Positive_Conditional_Rate_MAE",
            "Occurrence_Brier_Score",
        ]:
            part = rf_pair.loc[
                (rf_pair["Validation_Scheme"] == validation)
                & (rf_pair["Metric"] == metric)
            ]
            rows.append(
                {
                    "Result_Family": "Paired_RF_vs_Regression",
                    "Validation_Scheme": validation,
                    "Metric": metric,
                    "RF_Stably_Better_N": int(
                        (part["Stable_Conclusion"] == "Candidate_stably_better").sum()
                    ),
                    "Regression_Stably_Better_N": int(
                        (part["Stable_Conclusion"] == "Reference_stably_better").sum()
                    ),
                    "No_Stable_Difference_N": int(
                        (part["Stable_Conclusion"] == "No_stable_difference").sum()
                    ),
                    "Comparison_N": int(len(part)),
                }
            )
    for _, row in moran_wins.iterrows():
        rows.append(
            {
                "Result_Family": "Lowest_Residual_Moran_I",
                "Validation_Scheme": row["Validation_Scheme"],
                "Metric": row["Model_Role"],
                "RF_Stably_Better_N": np.nan,
                "Regression_Stably_Better_N": np.nan,
                "No_Stable_Difference_N": np.nan,
                "Comparison_N": int(row["Lowest_Moran_I_N"]),
            }
        )
    return pd.DataFrame(rows)


# =============================================================================
# Figures
# =============================================================================

def figure_relative_performance(overall: pd.DataFrame) -> pd.DataFrame:
    metrics = ["Rate_RMSE", "Positive_Conditional_Rate_RMSE"]
    rows: List[Dict[str, object]] = []
    for validation in VALIDATIONS:
        for season, response in COMBINATION_ORDER:
            subset = overall.loc[
                (overall["Validation_Scheme"] == validation)
                & (overall["Season"] == season)
                & (overall["Count_Response"] == response)
            ]
            for metric in metrics:
                values = subset.set_index("Model")[metric]
                baseline = float(values.loc["Baseline_NB1"])
                for model in MODELS:
                    rows.append(
                        {
                            "Validation_Scheme": validation,
                            "Season": season,
                            "Season_Label": SEASON_LABELS[season],
                            "Count_Response": response,
                            "Rate_Scale_Name": RESPONSE_LABELS[response],
                            "Combination_Label": f"{SEASON_LABELS[season]} {RESPONSE_LABELS[response]}",
                            "Metric": metric,
                            "Model": model,
                            "Metric_Value": float(values.loc[model]),
                            "Relative_to_Baseline_Percent": 100.0 * float(values.loc[model]) / baseline,
                        }
                    )
    data = pd.DataFrame(rows)
    save_csv(data, FIGURE_DATA_DIR / "Fig_MC1_Relative_Performance_Data.csv")

    configure_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 7.2), sharex=True)
    x = np.arange(len(COMBINATION_LABELS))
    metric_titles = {
        "Rate_RMSE": "Grid-level rate RMSE",
        "Positive_Conditional_Rate_RMSE": "Conditional-positive rate RMSE",
    }
    for row_index, validation in enumerate(VALIDATIONS):
        for col_index, metric in enumerate(metrics):
            ax = axes[row_index, col_index]
            part = data.loc[
                (data["Validation_Scheme"] == validation)
                & (data["Metric"] == metric)
            ]
            for model in MODELS:
                values = (
                    part.loc[part["Model"] == model]
                    .set_index(["Season", "Count_Response"])
                    .reindex(COMBINATION_ORDER)["Relative_to_Baseline_Percent"]
                    .to_numpy(dtype=float)
                )
                ax.plot(
                    x,
                    values,
                    marker=MODEL_MARKERS[model],
                    linewidth=1.5,
                    markersize=5.5,
                    label=MODEL_LABELS[model],
                    color=MODEL_COLORS[model],
                )
            ax.axhline(100.0, color="#A0A0A0", linewidth=0.8, linestyle="--")
            ax.set_title(f"{validation} OOF: {metric_titles[metric]}")
            ax.set_ylabel("Relative error (% of baseline)")
            ax.grid(axis="y", alpha=0.25, linewidth=0.6)
            ax.set_xticks(x)
            ax.set_xticklabels(COMBINATION_LABELS, rotation=35, ha="right")
            ax.text(
                -0.10,
                1.04,
                chr(ord("a") + row_index * 2 + col_index),
                transform=ax.transAxes,
                fontsize=12,
                fontweight="bold",
            )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, "Fig_MC1_Overall_Relative_Prediction_Error")
    return data


def figure_bootstrap_forest(bootstrap: pd.DataFrame) -> pd.DataFrame:
    pair = bootstrap.loc[
        (bootstrap["Model_Pair"] == "Final_Hurdle_RF_vs_Final_Regression")
        & bootstrap["Metric"].isin(
            ["Rate_RMSE", "Positive_Conditional_Rate_RMSE"]
        )
    ].copy()
    pair["Combination_Label"] = pair.apply(
        lambda row: f"{row['Season_Label']} {RESPONSE_LABELS[row['Count_Response']]}",
        axis=1,
    )
    save_csv(pair, FIGURE_DATA_DIR / "Fig_MC2_RF_vs_Regression_Bootstrap_Data.csv")

    configure_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.3), sharex=False)
    metrics = ["Rate_RMSE", "Positive_Conditional_Rate_RMSE"]
    metric_titles = {
        "Rate_RMSE": "Grid-level rate RMSE",
        "Positive_Conditional_Rate_RMSE": "Conditional-positive rate RMSE",
    }
    y = np.arange(len(COMBINATION_LABELS))
    for row_index, validation in enumerate(VALIDATIONS):
        for col_index, metric in enumerate(metrics):
            ax = axes[row_index, col_index]
            part = (
                pair.loc[
                    (pair["Validation_Scheme"] == validation)
                    & (pair["Metric"] == metric)
                ]
                .set_index(["Season", "Count_Response"])
                .reindex(COMBINATION_ORDER)
                .reset_index()
            )
            point = part["Percent_Improvement_Positive_Is_Better"].to_numpy(dtype=float)
            lower = part["Percent_Improvement_CI_Lower"].to_numpy(dtype=float)
            upper = part["Percent_Improvement_CI_Upper"].to_numpy(dtype=float)
            xerr = np.vstack([point - lower, upper - point])
            stable = part["Stable_Conclusion"].astype(str).to_numpy()
            colors = [
                MODEL_COLORS["Final_Hurdle_RF"]
                if value == "Candidate_stably_better"
                else (
                    MODEL_COLORS["Final_Regression"]
                    if value == "Reference_stably_better"
                    else "#888888"
                )
                for value in stable
            ]
            for index in range(len(point)):
                ax.errorbar(
                    point[index],
                    y[index],
                    xerr=xerr[:, index : index + 1],
                    fmt="o",
                    color=colors[index],
                    ecolor=colors[index],
                    elinewidth=1.3,
                    capsize=3,
                    markersize=5.5,
                )
            ax.axvline(0.0, color="#333333", linewidth=0.8, linestyle="--")
            ax.set_yticks(y)
            ax.set_yticklabels(COMBINATION_LABELS)
            ax.invert_yaxis()
            ax.set_xlabel("Improvement of Hurdle RF over regression (%)")
            ax.set_title(f"{validation} OOF: {metric_titles[metric]}")
            ax.grid(axis="x", alpha=0.25, linewidth=0.6)
            ax.text(
                -0.10,
                1.04,
                chr(ord("a") + row_index * 2 + col_index),
                transform=ax.transAxes,
                fontsize=12,
                fontweight="bold",
            )
    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=MODEL_COLORS["Final_Hurdle_RF"], markeredgecolor=MODEL_COLORS["Final_Hurdle_RF"], label="RF stably better"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=MODEL_COLORS["Final_Regression"], markeredgecolor=MODEL_COLORS["Final_Regression"], label="Regression stably better"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#888888", markeredgecolor="#888888", label="No stable difference"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.015))
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    save_figure(fig, "Fig_MC2_Paired_Bootstrap_RF_vs_Regression")
    return pair


def figure_moran(moran: pd.DataFrame) -> pd.DataFrame:
    data = moran[
        [
            "Validation_Scheme",
            "Season_Code",
            "Season_Label",
            "Count_Response",
            "Rate_Scale_Name",
            "Model_Role",
            "Moran_I",
            "Combined_Primary_Global_BH_FDR",
        ]
    ].copy()
    data["Combination_Label"] = data.apply(
        lambda row: f"{row['Season_Label']} {RESPONSE_LABELS[row['Count_Response']]}",
        axis=1,
    )
    save_csv(data, FIGURE_DATA_DIR / "Fig_MC3_Residual_Moran_Data.csv")

    configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.9), sharey=True)
    x = np.arange(len(COMBINATION_LABELS))
    for index, validation in enumerate(VALIDATIONS):
        ax = axes[index]
        part = data.loc[data["Validation_Scheme"] == validation]
        for model in MODELS:
            values = (
                part.loc[part["Model_Role"] == model]
                .set_index(["Season_Code", "Count_Response"])
                .reindex(COMBINATION_ORDER)["Moran_I"]
                .to_numpy(dtype=float)
            )
            ax.plot(
                x,
                values,
                marker=MODEL_MARKERS[model],
                linewidth=1.5,
                markersize=5.5,
                color=MODEL_COLORS[model],
                label=MODEL_LABELS[model],
            )
        ax.set_title(f"{validation} OOF")
        ax.set_xticks(x)
        ax.set_xticklabels(COMBINATION_LABELS, rotation=35, ha="right")
        ax.set_ylabel("Residual Moran's I")
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax.text(-0.08, 1.04, chr(ord("a") + index), transform=ax.transAxes, fontsize=12, fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(fig, "Fig_MC3_Residual_Spatial_Autocorrelation")
    return data


def figure_regional_wins(subgroup_wins: pd.DataFrame) -> pd.DataFrame:
    data = (
        subgroup_wins.groupby(
            ["Subgroup_Type", "Validation_Scheme", "Model"], as_index=False
        )
        .agg(Rank1_N=("Rank1_N", "sum"), Comparison_N=("Comparison_N", "sum"))
    )
    save_csv(data, FIGURE_DATA_DIR / "Fig_MC4_Regional_Win_Count_Data.csv")

    configure_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4), sharey=False)
    for row_index, subgroup_type in enumerate(["Country", "Spatial_Block"]):
        for col_index, validation in enumerate(VALIDATIONS):
            ax = axes[row_index, col_index]
            part = data.loc[
                (data["Subgroup_Type"] == subgroup_type)
                & (data["Validation_Scheme"] == validation)
            ].set_index("Model").reindex(MODELS)
            x = np.arange(len(MODELS))
            values = part["Rank1_N"].to_numpy(dtype=float)
            bars = ax.bar(
                x,
                values,
                color=[MODEL_COLORS[model] for model in MODELS],
                width=0.65,
            )
            denominator = int(part["Comparison_N"].iloc[0]) if len(part) else 0
            ymax = max(1.0, float(np.nanmax(values)) * 1.18)
            ax.set_ylim(0, ymax)
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{int(value)}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
            label = "Countries" if subgroup_type == "Country" else "Spatial blocks"
            ax.set_title(f"{validation} OOF: {label}")
            ax.set_xticks(x)
            ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], rotation=20, ha="right")
            ax.set_ylabel("Number of metric-wise first ranks")
            ax.grid(axis="y", alpha=0.25, linewidth=0.6)
            ax.text(
                0.98,
                0.88,
                f"Total comparisons/model = {denominator}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
            )
            ax.text(-0.10, 1.04, chr(ord("a") + row_index * 2 + col_index), transform=ax.transAxes, fontsize=12, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "Fig_MC4_Regional_Model_Win_Counts")
    return data


def figure_annual_series(annual: pd.DataFrame, response: str, stem: str) -> pd.DataFrame:
    data = annual.loc[annual["Count_Response"] == response].copy()
    save_csv(data, FIGURE_DATA_DIR / f"{stem}_Data.csv")

    configure_matplotlib()
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 7.2), sharex=True, sharey=False)
    for row_index, validation in enumerate(VALIDATIONS):
        for col_index, season in enumerate(SEASONS):
            ax = axes[row_index, col_index]
            part = data.loc[
                (data["Validation_Scheme"] == validation)
                & (data["Season"] == season)
            ].sort_values("Year")
            observed = (
                part.drop_duplicates("Year")
                .set_index("Year")
                .reindex(range(2001, 2026))["Observed_Total_Count"]
            )
            ax.plot(
                observed.index,
                observed.values,
                color="#111111",
                linewidth=1.8,
                label="Observed",
            )
            for model in MODELS:
                predicted = (
                    part.loc[part["Model"] == model]
                    .set_index("Year")
                    .reindex(range(2001, 2026))["Predicted_Total_Count"]
                )
                ax.plot(
                    predicted.index,
                    predicted.values,
                    color=MODEL_COLORS[model],
                    linewidth=1.25,
                    label=MODEL_LABELS[model],
                )
            ax.set_title(f"{validation} OOF: {SEASON_LABELS[season]}")
            ax.set_xlabel("Year")
            ax.set_ylabel("Annual total fire count" if response == "Fire_Count" else "Annual total burned-pixel count")
            ax.grid(alpha=0.20, linewidth=0.5)
            ax.tick_params(axis="x", rotation=35)
            ax.text(-0.10, 1.04, chr(ord("a") + row_index * 3 + col_index), transform=ax.transAxes, fontsize=12, fontweight="bold")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, stem)
    return data


# =============================================================================
# Captions and text summary
# =============================================================================

def write_captions(path: Path) -> None:
    text = """Fig. MC1. Prediction errors of the three models relative to the Baseline NB1 model under temporal and spatial out-of-fold validation. Values below 100% indicate lower error than the baseline. Grid-level rate RMSE and conditional-positive rate RMSE are shown for Fire Count Density (FCD) and Burned Area Density (BAD) across spring, summer, and autumn.\n\nFig. MC2. Paired GRID_UID-cluster bootstrap comparison of Final Hurdle RF versus Final Regression. Points show the percentage improvement of Hurdle RF, and horizontal error bars show percentile-bootstrap 95% confidence intervals based on 2,000 stratified grid-cluster resamples. Positive values favor Hurdle RF. Orange points indicate a stable RF advantage, blue points indicate a stable regression advantage, and grey points indicate no stable difference.\n\nFig. MC3. Residual spatial autocorrelation of Baseline NB1, Final Regression, and Final Hurdle RF under temporal and spatial out-of-fold validation. Moran's I was calculated from grid-level mean residual rates using row-standardized 8-nearest-neighbor weights and 9,999 permutations. All primary tests remained significant after the joint three-model Benjamini-Hochberg correction.\n\nFig. MC4. Descriptive counts of metric-wise first ranks across country and spatial-block subgroup comparisons. Counts aggregate five metrics: rate RMSE, rate MAE, occurrence Brier score, conditional-positive rate RMSE, and conditional-positive rate MAE. These counts describe regional consistency and do not replace paired bootstrap inference.\n\nFig. S-MC1. Observed and predicted annual total Fire Count under temporal and spatial out-of-fold validation during 2001-2025.\n\nFig. S-MC2. Observed and predicted annual total burned-pixel count under temporal and spatial out-of-fold validation during 2001-2025.\n"""
    path.write_text(text, encoding="utf-8")


def write_key_findings(
    path: Path,
    paired: pd.DataFrame,
    moran_wins: pd.DataFrame,
    subgroup_wins: pd.DataFrame,
    annual_wins: pd.DataFrame,
) -> None:
    lines: List[str] = []
    lines.append("Locked model-comparison findings")
    lines.append("================================")
    lines.append("")
    for validation in VALIDATIONS:
        part = paired.loc[paired["Validation_Scheme"] == validation]
        for metric in ["Rate_RMSE", "Positive_Conditional_Rate_RMSE"]:
            conclusion_col = f"{metric}_Stable_Conclusion"
            rf = int((part[conclusion_col] == "Candidate_stably_better").sum())
            reg = int((part[conclusion_col] == "Reference_stably_better").sum())
            none = int((part[conclusion_col] == "No_stable_difference").sum())
            lines.append(
                f"{validation} OOF, {metric}: RF stably better {rf}/6; "
                f"regression stably better {reg}/6; no stable difference {none}/6."
            )
    lines.append("")
    lines.append("Primary residual Moran's I lowest-count summary:")
    for _, row in moran_wins.sort_values(["Validation_Scheme", "Model_Role"]).iterrows():
        lines.append(
            f"- {row['Validation_Scheme']} OOF, {MODEL_LABELS.get(row['Model_Role'], row['Model_Role'])}: "
            f"lowest Moran's I in {int(row['Lowest_Moran_I_N'])}/{int(row['Comparison_N'])} combinations."
        )
    lines.append("")
    lines.append(
        "Interpretation: Hurdle RF provides the strongest temporal grid-level and "
        "conditional-positive severity prediction, whereas Final Regression is more "
        "stable for several spatial-extrapolation, residual-spatial-structure, and "
        "aggregated annual-series tasks. The outputs support model complementarity "
        "rather than a single universally superior model."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    SUPPLEMENT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text("", encoding="utf-8")

    log("Starting Model Comparison Step 06: final tables and figures.")

    missing = [str(path) for path in INPUTS.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Required locked inputs are missing:\n" + "\n".join(missing))

    audit_rows: List[Dict[str, object]] = []

    # Version and manifest checks.
    method_files = {
        "Step02": INPUTS["Step02_Method"],
        "Step03": INPUTS["Step03_Method"],
        "Step04": INPUTS["Step04_Method"],
        "Step05": INPUTS["Step05_Method"],
    }
    for step, method_path in method_files.items():
        observed = read_json(method_path).get("Code_Version")
        audit_rows.append(
            {
                "Category": "Version",
                "Check": f"{step}_Code_Version",
                "Passed": observed == EXPECTED_VERSIONS[step],
                "Observed": observed,
                "Expected": EXPECTED_VERSIONS[step],
            }
        )

    manifest_map = {
        "Step02": INPUTS["Step02_Manifest"],
        "Step03": INPUTS["Step03_Manifest"],
        "Step04": INPUTS["Step04_Manifest"],
        "Step05": INPUTS["Step05_Manifest"],
    }
    hash_inputs = {
        "Step02": [INPUTS["Step02_Overall"], INPUTS["Step02_Fold"], INPUTS["Step02_Extreme"]],
        "Step03": [INPUTS["Step03_Bootstrap"], INPUTS["Step03_Nonadditive"], INPUTS["Step03_Stable_Wins"]],
        "Step04": [INPUTS["Step04_Primary_Moran"], INPUTS["Step04_KNN_Stability"], INPUTS["Step04_Tradeoff"], INPUTS["Step04_Moran_Wins"]],
        "Step05": [INPUTS["Step05_Country"], INPUTS["Step05_Block"], INPUTS["Step05_Annual_Aggregate"], INPUTS["Step05_Annual_Series"], INPUTS["Step05_Country_Annual_Series"], INPUTS["Step05_Block_Annual_Series"], INPUTS["Step05_Subgroup_Wins"], INPUTS["Step05_Annual_Wins"]],
    }
    hash_audit_rows: List[Dict[str, object]] = []
    for step, paths in hash_inputs.items():
        for path in paths:
            expected = manifest_expected_hash(manifest_map[step], path.name)
            actual = sha256_file(path)
            passed = bool(expected and actual == expected)
            hash_audit_rows.append(
                {
                    "Step": step,
                    "File": path.name,
                    "Expected_SHA256": expected,
                    "Actual_SHA256": actual,
                    "Passed": passed,
                }
            )
    hash_audit = pd.DataFrame(hash_audit_rows)
    audit_rows.append(
        {
            "Category": "Hash",
            "Check": "All_locked_input_hashes_match",
            "Passed": bool(hash_audit["Passed"].all()),
            "Observed": int(hash_audit["Passed"].sum()),
            "Expected": int(len(hash_audit)),
        }
    )

    # Read locked data.
    log("Reading locked Step 02-05 outputs.")
    overall = pd.read_csv(INPUTS["Step02_Overall"])
    fold = pd.read_csv(INPUTS["Step02_Fold"])
    extreme = pd.read_csv(INPUTS["Step02_Extreme"])
    bootstrap = pd.read_csv(INPUTS["Step03_Bootstrap"])
    nonadditive = pd.read_csv(INPUTS["Step03_Nonadditive"])
    stable_wins = pd.read_csv(INPUTS["Step03_Stable_Wins"])
    moran = pd.read_csv(INPUTS["Step04_Primary_Moran"])
    knn_stability = pd.read_csv(INPUTS["Step04_KNN_Stability"])
    tradeoff = pd.read_csv(INPUTS["Step04_Tradeoff"])
    moran_wins = pd.read_csv(INPUTS["Step04_Moran_Wins"])
    country = pd.read_csv(INPUTS["Step05_Country"])
    block = pd.read_csv(INPUTS["Step05_Block"])
    annual_aggregate = pd.read_csv(INPUTS["Step05_Annual_Aggregate"])
    annual_series = pd.read_csv(INPUTS["Step05_Annual_Series"])
    country_annual_series = pd.read_csv(INPUTS["Step05_Country_Annual_Series"])
    block_annual_series = pd.read_csv(INPUTS["Step05_Block_Annual_Series"])
    subgroup_wins = pd.read_csv(INPUTS["Step05_Subgroup_Wins"])
    annual_wins = pd.read_csv(INPUTS["Step05_Annual_Wins"])

    expected_rows = {
        "Step02_Overall": (len(overall), 36),
        "Step03_Bootstrap": (len(bootstrap), 432),
        "Step04_Primary_Moran": (len(moran), 36),
        "Step05_Country": (len(country), 108),
        "Step05_Block": (len(block), 180),
        "Step05_Annual_Aggregate": (len(annual_aggregate), 900),
        "Step05_Annual_Series": (len(annual_series), 36),
    }
    for name, (observed, expected) in expected_rows.items():
        audit_rows.append(
            {
                "Category": "Row_Count",
                "Check": name,
                "Passed": observed == expected,
                "Observed": observed,
                "Expected": expected,
            }
        )

    # Build tables.
    log("Building manuscript and supplementary tables.")
    overall_main = build_overall_main_table(overall, moran, annual_series)
    paired_main = build_rf_regression_paired_table(bootstrap, tradeoff, overall, annual_series)
    recommendation = build_integrated_recommendation(paired_main)
    regional_wins = build_regional_win_summary(subgroup_wins, annual_wins)
    key_statistics = build_key_statistics(overall_main, bootstrap, moran_wins, subgroup_wins, annual_wins)

    save_csv(overall_main, OUTPUT / "02_Main_Table_Overall_Model_Performance.csv")
    save_csv(paired_main, OUTPUT / "03_Main_Table_RF_vs_Regression_Paired_Comparison.csv")
    save_csv(
        moran[
            [
                "Validation_Scheme",
                "Season_Code",
                "Season_Label",
                "Count_Response",
                "Rate_Scale_Name",
                "Model_Role",
                "Moran_I",
                "Expected_I",
                "Permutation_P_TwoSided",
                "Combined_Primary_Global_BH_FDR",
                "Residual_Mean_Absolute_Value",
                "Residual_Root_Mean_Square",
            ]
        ].sort_values(["Validation_Scheme", "Season_Code", "Count_Response", "Model_Role"]),
        OUTPUT / "04_Main_Table_Residual_Moran_I.csv",
    )
    save_csv(regional_wins, OUTPUT / "05_Main_Table_Regional_and_Annual_Win_Summary.csv")
    save_csv(recommendation, OUTPUT / "06_Integrated_Season_Response_Interpretation.csv")
    save_csv(key_statistics, OUTPUT / "07_Key_Comparison_Statistics.csv")

    # Preserve full locked tables needed for supplement/reproducibility.
    supplementary = {
        "S01_All_Overall_Unified_Metrics.csv": overall,
        "S02_All_Fold_Level_Metrics.csv": fold,
        "S03_All_Extreme_Event_Performance.csv": extreme,
        "S04_All_Paired_Bootstrap_Differences.csv": bootstrap,
        "S05_Nonadditive_Metric_Differences.csv": nonadditive,
        "S06_Stable_Win_Counts.csv": stable_wins,
        "S07_All_Primary_Moran_Results.csv": moran,
        "S08_KNN_Stability_Summary.csv": knn_stability,
        "S09_Prediction_Moran_Tradeoff.csv": tradeoff,
        "S10_Country_Level_Performance.csv": country,
        "S11_Spatial_Block_Level_Performance.csv": block,
        "S12_Annual_Aggregated_Predictions.csv": annual_aggregate,
        "S13_Overall_Annual_Series_Performance.csv": annual_series,
        "S14_Country_Annual_Series_Performance.csv": country_annual_series,
        "S15_Block_Annual_Series_Performance.csv": block_annual_series,
        "S16_Subgroup_Model_Win_Summary.csv": subgroup_wins,
        "S17_Annual_Series_Model_Win_Summary.csv": annual_wins,
    }
    for filename, data in supplementary.items():
        save_csv(data, SUPPLEMENT_DIR / filename)

    # Figures.
    log("Generating manuscript figures in PNG, vector PDF, and 600-dpi TIFF formats.")
    figure_relative_performance(overall)
    figure_bootstrap_forest(bootstrap)
    figure_moran(moran)
    figure_regional_wins(subgroup_wins)
    figure_annual_series(annual_aggregate, "Fire_Count", "Fig_SMC1_Annual_Fire_Count_Observed_vs_Predicted")
    figure_annual_series(annual_aggregate, "Burned_Pixel_Count", "Fig_SMC2_Annual_Burned_Pixel_Count_Observed_vs_Predicted")

    write_captions(OUTPUT / "08_Figure_Captions.txt")
    write_key_findings(
        OUTPUT / "09_Locked_Key_Findings.txt",
        paired_main,
        moran_wins,
        subgroup_wins,
        annual_wins,
    )

    method = {
        "Code_Version": CODE_VERSION,
        "Created": datetime.now().isoformat(timespec="seconds"),
        "Purpose": "Generate final manuscript tables, figure data, figures, and locked interpretation summaries from Model Comparison Steps 02-05.",
        "Input_Steps": ["Step02", "Step03", "Step04", "Step05"],
        "No_Model_Fitting": True,
        "No_Model_Tuning": True,
        "No_Model_Selection": True,
        "No_OOF_Prediction_Recomputation": True,
        "No_Bootstrap_Recomputation": True,
        "No_Moran_Recomputation": True,
        "Main_Interpretive_Principle": "Prediction accuracy, residual spatial autocorrelation, calibration, regional robustness, and annual aggregation are reported jointly; no universally superior model is declared.",
        "Figure_Formats": ["PNG_400dpi", "PDF_vector", "TIFF_600dpi_LZW"],
        "Figure_Font": "Times New Roman with serif fallback",
    }
    write_json(OUTPUT / "00_Method_Definition.json", method)

    # Final QA.
    qa_rows: List[Dict[str, object]] = []
    expected_outputs = {
        "Overall_Main": (len(overall_main), 36),
        "Paired_Main": (len(paired_main), 12),
        "Recommendation": (len(recommendation), 12),
        "Regional_Wins": (len(regional_wins), 30),
    }
    for name, (observed, expected) in expected_outputs.items():
        qa_rows.append(
            {
                "Check": f"{name}_row_count",
                "Passed": observed == expected,
                "Observed": observed,
                "Expected": expected,
            }
        )
    figure_files = sorted(FIGURE_DIR.glob("*"))
    qa_rows.append(
        {
            "Check": "Figure_file_count",
            "Passed": len(figure_files) == 18,
            "Observed": len(figure_files),
            "Expected": 18,
        }
    )
    qa_rows.append(
        {
            "Check": "All_main_table_key_fields_nonmissing",
            "Passed": bool(
                overall_main[
                    ["Validation_Scheme", "Season", "Count_Response", "Model"]
                ].notna().all().all()
                and paired_main[
                    ["Validation_Scheme", "Season", "Count_Response"]
                ].notna().all().all()
            ),
            "Observed": "checked",
            "Expected": "no missing key fields",
        }
    )
    qa = pd.DataFrame(qa_rows)
    save_csv(pd.DataFrame(audit_rows), OUTPUT / "01_Input_Integrity_Audit.csv")
    save_csv(hash_audit, OUTPUT / "01b_Upstream_Hash_Audit.csv")
    save_csv(qa, OUTPUT / "10_Computation_QA.csv")

    if not pd.DataFrame(audit_rows)["Passed"].all():
        failed = pd.DataFrame(audit_rows).loc[~pd.DataFrame(audit_rows)["Passed"]]
        raise RuntimeError("Input integrity audit failed:\n" + failed.to_string(index=False))
    if not qa["Passed"].all():
        failed = qa.loc[~qa["Passed"]]
        raise RuntimeError("Computation QA failed:\n" + failed.to_string(index=False))

    environment = {
        "Python": sys.version,
        "Platform": platform.platform(),
        "NumPy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
    }
    write_json(OUTPUT / "Software_Environment.json", environment)

    log(
        "Step 06 completed: final model-comparison manuscript tables, supplementary tables, figure data, and publication figures were generated from locked Steps 02-05 without fitting, selecting, or predicting any model."
    )

    # Manifest excludes itself; the completed log is now stable and can be hashed.
    manifest_rows: List[Dict[str, object]] = []
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.name not in {"11_Output_Manifest.csv"}:
            manifest_rows.append(
                {
                    "File": path.name,
                    "Relative_Path": str(path.relative_to(OUTPUT)),
                    "File_Size_Bytes": int(path.stat().st_size),
                    "SHA256": sha256_file(path),
                    "Step06_Code_Version": CODE_VERSION,
                }
            )
    save_csv(pd.DataFrame(manifest_rows), OUTPUT / "11_Output_Manifest.csv")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        error_text = traceback.format_exc()
        try:
            log("ERROR")
            log(error_text)
        except Exception:
            print(error_text, file=sys.stderr)
        raise
