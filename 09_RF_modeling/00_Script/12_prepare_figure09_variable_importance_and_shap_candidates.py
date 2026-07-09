#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Prepare the variable-level importance data and seasonal nonlinear-response
candidates for manuscript Figure 9.

This script replaces the former Step 12. It does not fit or retrain any
Random Forest model.

Main tasks
----------
1. Prepare variable-level mean absolute OOF SHAP importance for every
   environmental predictor in all six season-response models and both
   Hurdle RF stages.

2. Preserve a complete table including country-context dummy variables,
   while excluding those dummy variables from the main Figure 9 plotting
   table to remain consistent with Figure 8.

3. Rank potential nonlinear-response panels separately by season and stage.
   The ranking does not automatically lock the final manuscript panels.
   It provides an objective shortlist for scientific interpretation.

4. Create a three-page candidate atlas. Each page shows the leading
   season-specific candidates from both Hurdle RF stages, with FC and BA
   binned OOF SHAP curves.

Candidate ranking rule
----------------------
For each season-stage-predictor combination:

- FC and BA mean absolute OOF SHAP values are normalized separately by the
  maximum environmental-variable importance within the same
  season-response-stage model.
- The primary score is the geometric mean of the FC and BA relative
  importance values.
- Binned SHAP curve amplitude is calculated for FC and BA, normalized
  separately within the same season-response-stage model, and used only as
  a secondary ranking criterion.
- Candidates used in the seasonal shortlist must have at least five SHAP
  bins for both FC and BA. This prevents two-level or very sparse variables
  from being presented as nonlinear response curves.
- No final spring, summer, or autumn variable is selected by this script.

Formal code-package location:
    <REPOSITORY_ROOT>\09_RF_modeling\00_Script\
        12_prepare_figure09_variable_importance_and_shap_candidates.py

Default project root:
    <REPOSITORY_ROOT>

Default output directory:
    <REPOSITORY_ROOT>\09_RF_modeling\
        12_Figure09_Variable_Importance_and_SHAP_Candidates
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd


SCRIPT_VERSION = "1.0.0"

DEFAULT_ROOT = Path(__file__).resolve().parents[2]
RF_DIR = Path("09_RF_modeling")
SHAP_SUBDIR = RF_DIR / "11_Final_OOF_SHAP_Direction"
DEFAULT_OUTPUT_SUBDIR = (
    RF_DIR / "12_Figure09_Variable_Importance_and_SHAP_Candidates"
)

INDIVIDUAL_SHAP_FILENAME = "02_Individual_SHAP_Summary.csv"
BINNED_SHAP_FILENAME = "04_Binned_SHAP_Response.csv"

FONT_FAMILY = "Times New Roman"
OUTPUT_DPI = 300
TOP_CANDIDATES_PER_SEASON = 6
MIN_BINS_FOR_NONLINEAR_PANEL = 5

SEASON_ORDER: Sequence[str] = ("Spring", "Summer", "Autumn")
STAGE_ORDER: Sequence[str] = ("Occurrence", "Positive_Severity")
RESPONSE_ORDER: Sequence[str] = ("FCD", "BAD")

STAGE_LABELS: Dict[str, str] = {
    "Occurrence": "Occurrence",
    "Positive_Severity": "Positive severity",
}

RESPONSE_LABELS: Dict[str, str] = {
    "FCD": "FC",
    "BAD": "BA",
}

PREDICTOR_GROUP_ORDER: Sequence[str] = (
    "Forest_Structure",
    "Terrain",
    "Anthropogenic",
    "Meteorology",
    "Lightning",
    "Drought",
)

PREDICTOR_ORDER: Sequence[str] = (
    # Forest structure
    "BD",
    "ND",
    "NE",
    "EVI",
    "PTC",
    # Terrain
    "DEM",
    "Slope",
    "Aspect",
    # Anthropogenic
    "POP",
    "Dis_Farm",
    "Road_dens",
    "Dis_Railway",
    # Meteorology
    "Temp",
    "Pre",
    "Rhum",
    "Wind",
    "SSRD",
    # Lightning
    "LtgProxy",
    # Drought
    "SPEI3",
    "SPEI12",
)

GROUP_LABELS: Dict[str, str] = {
    "Forest_Structure": "Forest structure",
    "Terrain": "Terrain",
    "Anthropogenic": "Anthropogenic",
    "Meteorology": "Meteorology",
    "Lightning": "Lightning",
    "Drought": "Drought",
    "Country_Context": "Country context",
}

REQUIRED_INDIVIDUAL_COLUMNS = {
    "Stage",
    "Season",
    "Season_Label",
    "Response",
    "Predictor",
    "Predictor_Group",
    "N",
    "Mean_Absolute_SHAP",
    "Median_Absolute_SHAP",
    "Mean_SHAP",
    "SHAP_SD",
    "Feature_SHAP_Spearman",
    "Low_Value_Mean_SHAP",
    "High_Value_Mean_SHAP",
    "High_minus_Low_Mean_SHAP",
    "Positive_SHAP_Fraction",
    "Direction_Classification",
    "Mean_Absolute_SHAP_Rank",
}

REQUIRED_BINNED_COLUMNS = {
    "Stage",
    "Season",
    "Season_Label",
    "Response",
    "Predictor",
    "Bin_Order",
    "N",
    "Feature_Min",
    "Feature_Median",
    "Feature_Mean",
    "Feature_Max",
    "Mean_SHAP",
    "Median_SHAP",
    "SHAP_Q25",
    "SHAP_Q75",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare all-variable OOF SHAP importance and seasonal "
            "nonlinear-response candidates for manuscript Figure 9."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Repository root. Defaults to the detected repository root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional output directory. Default: "
            "<root>/09_RF_modeling/"
            "12_Figure09_Variable_Importance_and_SHAP_Candidates"
        ),
    )
    parser.add_argument(
        "--top-candidates",
        type=int,
        default=TOP_CANDIDATES_PER_SEASON,
        help=(
            "Number of eligible candidates shown on each seasonal atlas "
            f"page. Default: {TOP_CANDIDATES_PER_SEASON}"
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=OUTPUT_DPI,
        help=f"Candidate-atlas PNG resolution. Default: {OUTPUT_DPI}",
    )
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv_strict(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required input file not found:\n{path}")
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8")


def require_columns(
    frame: pd.DataFrame,
    required: Iterable[str],
    label: str,
) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(
            f"{label} is missing required columns: {', '.join(missing)}"
        )


def safe_geometric_mean(a: float, b: float) -> float:
    if a < 0 or b < 0:
        raise ValueError("Geometric-mean inputs must be non-negative.")
    return math.sqrt(a * b)


def rank_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)

    valid = np.isfinite(x_array) & np.isfinite(y_array)
    x_array = x_array[valid]
    y_array = y_array[valid]

    if len(x_array) < 3:
        return float("nan")

    x_rank = pd.Series(x_array).rank(method="average").to_numpy()
    y_rank = pd.Series(y_array).rank(method="average").to_numpy()

    if np.std(x_rank) == 0 or np.std(y_rank) == 0:
        return float("nan")

    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def count_zero_crossings(values: Sequence[float]) -> int:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) < 2:
        return 0

    signs = np.sign(array)
    nonzero = signs[signs != 0]
    if len(nonzero) < 2:
        return 0

    return int(np.sum(nonzero[1:] != nonzero[:-1]))


def find_font() -> Tuple[str, bool]:
    names = {font.name for font in font_manager.fontManager.ttflist}
    if FONT_FAMILY in names:
        return FONT_FAMILY, True
    return "DejaVu Serif", False


def validate_inputs(
    individual: pd.DataFrame,
    binned: pd.DataFrame,
) -> None:
    require_columns(
        individual,
        REQUIRED_INDIVIDUAL_COLUMNS,
        "Individual OOF SHAP summary",
    )
    require_columns(
        binned,
        REQUIRED_BINNED_COLUMNS,
        "Binned OOF SHAP response table",
    )

    expected_models = {
        (stage, season, response)
        for stage in STAGE_ORDER
        for season in SEASON_ORDER
        for response in RESPONSE_ORDER
    }

    individual_models = set(
        zip(
            individual["Stage"].astype(str),
            individual["Season_Label"].astype(str),
            individual["Response"].astype(str),
        )
    )
    binned_models = set(
        zip(
            binned["Stage"].astype(str),
            binned["Season_Label"].astype(str),
            binned["Response"].astype(str),
        )
    )

    missing_individual = expected_models - individual_models
    missing_binned = expected_models - binned_models

    if missing_individual:
        raise ValueError(
            "Individual SHAP summary is missing model combinations: "
            f"{sorted(missing_individual)}"
        )
    if missing_binned:
        raise ValueError(
            "Binned SHAP table is missing model combinations: "
            f"{sorted(missing_binned)}"
        )

    predictor_group_map = (
        individual[["Predictor", "Predictor_Group"]]
        .drop_duplicates()
        .groupby("Predictor")["Predictor_Group"]
        .nunique()
    )
    inconsistent = predictor_group_map[predictor_group_map != 1]
    if not inconsistent.empty:
        raise ValueError(
            "Some predictors are assigned to more than one group: "
            + ", ".join(inconsistent.index.astype(str))
        )

    observed_environmental = set(
        individual.loc[
            individual["Predictor_Group"].astype(str)
            != "Country_Context",
            "Predictor",
        ].astype(str)
    )
    expected_environmental = set(PREDICTOR_ORDER)

    if observed_environmental != expected_environmental:
        raise ValueError(
            "Environmental predictor set does not match the expected "
            "20-variable RF structure.\n"
            f"Missing: {sorted(expected_environmental - observed_environmental)}\n"
            f"Unexpected: {sorted(observed_environmental - expected_environmental)}"
        )


def prepare_variable_importance(
    individual: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = individual.copy()

    numeric_columns = (
        "N",
        "Mean_Absolute_SHAP",
        "Median_Absolute_SHAP",
        "Mean_SHAP",
        "SHAP_SD",
        "Feature_SHAP_Spearman",
        "Low_Value_Mean_SHAP",
        "High_Value_Mean_SHAP",
        "High_minus_Low_Mean_SHAP",
        "Positive_SHAP_Fraction",
        "Mean_Absolute_SHAP_Rank",
    )
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="raise")

    data = data.loc[
        data["Stage"].astype(str).isin(STAGE_ORDER)
        & data["Season_Label"].astype(str).isin(SEASON_ORDER)
        & data["Response"].astype(str).isin(RESPONSE_ORDER)
    ].copy()

    duplicate_mask = data.duplicated(
        subset=[
            "Stage",
            "Season_Label",
            "Response",
            "Predictor",
        ],
        keep=False,
    )
    if duplicate_mask.any():
        raise ValueError(
            "Duplicate stage-season-response-predictor rows were found in "
            "the individual SHAP summary."
        )

    data["Is_Environmental_Predictor"] = (
        data["Predictor_Group"].astype(str) != "Country_Context"
    )

    environmental = data.loc[
        data["Is_Environmental_Predictor"]
    ].copy()

    expected_rows = (
        len(STAGE_ORDER)
        * len(SEASON_ORDER)
        * len(RESPONSE_ORDER)
        * len(PREDICTOR_ORDER)
    )
    if len(environmental) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} environmental importance rows; "
            f"found {len(environmental)}."
        )

    maxima = (
        environmental.groupby(
            ["Stage", "Season_Label", "Response"],
            as_index=False,
        )["Mean_Absolute_SHAP"]
        .max()
        .rename(
            columns={
                "Mean_Absolute_SHAP": "Within_Model_Max_Mean_Absolute_SHAP"
            }
        )
    )

    environmental = environmental.merge(
        maxima,
        on=["Stage", "Season_Label", "Response"],
        how="left",
        validate="many_to_one",
    )

    if (
        environmental["Within_Model_Max_Mean_Absolute_SHAP"] <= 0
    ).any():
        raise ValueError(
            "At least one model has a non-positive maximum mean absolute SHAP."
        )

    environmental["Relative_Mean_Absolute_SHAP"] = (
        environmental["Mean_Absolute_SHAP"]
        / environmental["Within_Model_Max_Mean_Absolute_SHAP"]
    )
    environmental["Relative_Mean_Absolute_SHAP_Percent"] = (
        100.0 * environmental["Relative_Mean_Absolute_SHAP"]
    )

    predictor_position = {
        predictor: index
        for index, predictor in enumerate(PREDICTOR_ORDER)
    }
    group_position = {
        group: index
        for index, group in enumerate(PREDICTOR_GROUP_ORDER)
    }
    season_position = {
        season: index
        for index, season in enumerate(SEASON_ORDER)
    }
    stage_position = {
        stage: index
        for index, stage in enumerate(STAGE_ORDER)
    }
    response_position = {
        response: index
        for index, response in enumerate(RESPONSE_ORDER)
    }

    environmental["Predictor_Order"] = (
        environmental["Predictor"].map(predictor_position)
    )
    environmental["Predictor_Group_Order"] = (
        environmental["Predictor_Group"].map(group_position)
    )
    environmental["Season_Order"] = (
        environmental["Season_Label"].map(season_position)
    )
    environmental["Stage_Order"] = (
        environmental["Stage"].map(stage_position)
    )
    environmental["Response_Order"] = (
        environmental["Response"].map(response_position)
    )
    environmental["Stage_Display"] = (
        environmental["Stage"].map(STAGE_LABELS)
    )
    environmental["Response_Display"] = (
        environmental["Response"].map(RESPONSE_LABELS)
    )
    environmental["Predictor_Group_Display"] = (
        environmental["Predictor_Group"].map(GROUP_LABELS)
    )

    if environmental[
        [
            "Predictor_Order",
            "Predictor_Group_Order",
            "Season_Order",
            "Stage_Order",
            "Response_Order",
        ]
    ].isna().any().any():
        raise ValueError(
            "At least one environmental row could not be assigned to the "
            "locked plotting order."
        )

    environmental = environmental.sort_values(
        [
            "Stage_Order",
            "Predictor_Order",
            "Season_Order",
            "Response_Order",
        ]
    ).reset_index(drop=True)

    all_variables = data.copy()
    all_variables = all_variables.merge(
        maxima,
        on=["Stage", "Season_Label", "Response"],
        how="left",
        validate="many_to_one",
    )
    all_variables["Relative_to_Environmental_Max"] = (
        all_variables["Mean_Absolute_SHAP"]
        / all_variables["Within_Model_Max_Mean_Absolute_SHAP"]
    )
    all_variables["Relative_to_Environmental_Max_Percent"] = (
        100.0 * all_variables["Relative_to_Environmental_Max"]
    )

    qa_rows = []
    for stage in STAGE_ORDER:
        for season in SEASON_ORDER:
            for response in RESPONSE_ORDER:
                subset = environmental.loc[
                    (environmental["Stage"].astype(str) == stage)
                    & (
                        environmental["Season_Label"].astype(str)
                        == season
                    )
                    & (
                        environmental["Response"].astype(str)
                        == response
                    )
                ]
                qa_rows.append(
                    {
                        "Stage": stage,
                        "Season": season,
                        "Response": response,
                        "Environmental_Predictor_N": int(len(subset)),
                        "Unique_Predictor_N": int(
                            subset["Predictor"].nunique()
                        ),
                        "Relative_Maximum": float(
                            subset[
                                "Relative_Mean_Absolute_SHAP_Percent"
                            ].max()
                        ),
                        "Relative_Minimum": float(
                            subset[
                                "Relative_Mean_Absolute_SHAP_Percent"
                            ].min()
                        ),
                        "Importance_QA_Passed": bool(
                            len(subset) == len(PREDICTOR_ORDER)
                            and subset["Predictor"].nunique()
                            == len(PREDICTOR_ORDER)
                            and np.isclose(
                                subset[
                                    "Relative_Mean_Absolute_SHAP_Percent"
                                ].max(),
                                100.0,
                            )
                        ),
                    }
                )

    qa = pd.DataFrame(qa_rows)
    return all_variables, environmental, qa


def prepare_curve_metrics(
    binned: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data = binned.copy()

    numeric_columns = (
        "Bin_Order",
        "N",
        "Feature_Min",
        "Feature_Median",
        "Feature_Mean",
        "Feature_Max",
        "Mean_SHAP",
        "Median_SHAP",
        "SHAP_Q25",
        "SHAP_Q75",
    )
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="raise")

    data = data.loc[
        data["Stage"].astype(str).isin(STAGE_ORDER)
        & data["Season_Label"].astype(str).isin(SEASON_ORDER)
        & data["Response"].astype(str).isin(RESPONSE_ORDER)
        & data["Predictor"].astype(str).isin(PREDICTOR_ORDER)
    ].copy()

    duplicate_mask = data.duplicated(
        subset=[
            "Stage",
            "Season_Label",
            "Response",
            "Predictor",
            "Bin_Order",
        ],
        keep=False,
    )
    if duplicate_mask.any():
        raise ValueError(
            "Duplicate binned SHAP rows were found for the same model, "
            "predictor, and bin."
        )

    metric_rows: List[Dict[str, object]] = []

    grouped = data.groupby(
        ["Stage", "Season_Label", "Response", "Predictor"],
        sort=False,
    )

    for (stage, season, response, predictor), subset in grouped:
        subset = subset.sort_values("Bin_Order")

        feature = subset["Feature_Median"].to_numpy(dtype=float)
        mean_shap = subset["Mean_SHAP"].to_numpy(dtype=float)

        amplitude = float(np.max(mean_shap) - np.min(mean_shap))
        q25 = subset["SHAP_Q25"].to_numpy(dtype=float)
        q75 = subset["SHAP_Q75"].to_numpy(dtype=float)
        mean_iqr = float(np.mean(q75 - q25))
        amplitude_to_iqr = (
            amplitude / mean_iqr
            if mean_iqr > 0
            else float("nan")
        )

        metric_rows.append(
            {
                "Stage": str(stage),
                "Season": str(season),
                "Response": str(response),
                "Predictor": str(predictor),
                "Bin_N": int(len(subset)),
                "Total_Observation_N": int(subset["N"].sum()),
                "Feature_Min": float(subset["Feature_Min"].min()),
                "Feature_Max": float(subset["Feature_Max"].max()),
                "Mean_SHAP_Min": float(np.min(mean_shap)),
                "Mean_SHAP_Max": float(np.max(mean_shap)),
                "Mean_SHAP_Amplitude": amplitude,
                "Mean_Bin_IQR": mean_iqr,
                "Amplitude_to_Mean_IQR": amplitude_to_iqr,
                "Feature_Mean_SHAP_Spearman": rank_correlation(
                    feature,
                    mean_shap,
                ),
                "Zero_Crossing_N": count_zero_crossings(mean_shap),
            }
        )

    metrics = pd.DataFrame(metric_rows)

    expected_rows = (
        len(STAGE_ORDER)
        * len(SEASON_ORDER)
        * len(RESPONSE_ORDER)
        * len(PREDICTOR_ORDER)
    )
    if len(metrics) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} environmental curve-metric rows; "
            f"found {len(metrics)}."
        )

    amplitude_maxima = (
        metrics.groupby(
            ["Stage", "Season", "Response"],
            as_index=False,
        )["Mean_SHAP_Amplitude"]
        .max()
        .rename(
            columns={
                "Mean_SHAP_Amplitude": "Within_Model_Max_SHAP_Amplitude"
            }
        )
    )
    metrics = metrics.merge(
        amplitude_maxima,
        on=["Stage", "Season", "Response"],
        how="left",
        validate="many_to_one",
    )

    metrics["Relative_SHAP_Amplitude"] = np.where(
        metrics["Within_Model_Max_SHAP_Amplitude"] > 0,
        (
            metrics["Mean_SHAP_Amplitude"]
            / metrics["Within_Model_Max_SHAP_Amplitude"]
        ),
        0.0,
    )
    metrics["Relative_SHAP_Amplitude_Percent"] = (
        100.0 * metrics["Relative_SHAP_Amplitude"]
    )

    qa_rows = []
    for stage in STAGE_ORDER:
        for season in SEASON_ORDER:
            for response in RESPONSE_ORDER:
                subset = metrics.loc[
                    (metrics["Stage"].astype(str) == stage)
                    & (metrics["Season"].astype(str) == season)
                    & (metrics["Response"].astype(str) == response)
                ]
                qa_rows.append(
                    {
                        "Stage": stage,
                        "Season": season,
                        "Response": response,
                        "Curve_Predictor_N": int(len(subset)),
                        "Minimum_Bin_N": int(subset["Bin_N"].min()),
                        "Maximum_Bin_N": int(subset["Bin_N"].max()),
                        "Curve_QA_Passed": bool(
                            len(subset) == len(PREDICTOR_ORDER)
                            and subset["Predictor"].nunique()
                            == len(PREDICTOR_ORDER)
                            and np.isfinite(
                                subset["Mean_SHAP_Amplitude"]
                            ).all()
                        ),
                    }
                )

    qa = pd.DataFrame(qa_rows)
    return metrics, qa


def build_candidate_rankings(
    environmental_importance: pd.DataFrame,
    curve_metrics: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    importance_columns = [
        "Stage",
        "Season_Label",
        "Response",
        "Predictor",
        "Predictor_Group",
        "Mean_Absolute_SHAP",
        "Relative_Mean_Absolute_SHAP",
        "Relative_Mean_Absolute_SHAP_Percent",
        "Mean_Absolute_SHAP_Rank",
        "Direction_Classification",
        "Feature_SHAP_Spearman",
        "High_minus_Low_Mean_SHAP",
    ]

    merged = environmental_importance[importance_columns].merge(
        curve_metrics,
        left_on=[
            "Stage",
            "Season_Label",
            "Response",
            "Predictor",
        ],
        right_on=[
            "Stage",
            "Season",
            "Response",
            "Predictor",
        ],
        how="left",
        validate="one_to_one",
    )

    if merged["Mean_SHAP_Amplitude"].isna().any():
        raise ValueError(
            "At least one environmental predictor could not be matched to "
            "its binned SHAP curve metrics."
        )

    rows: List[Dict[str, object]] = []

    for stage in STAGE_ORDER:
        for season in SEASON_ORDER:
            subset = merged.loc[
                (merged["Stage"].astype(str) == stage)
                & (merged["Season_Label"].astype(str) == season)
            ].copy()

            fc = subset.loc[
                subset["Response"].astype(str) == "FCD"
            ].set_index("Predictor")
            ba = subset.loc[
                subset["Response"].astype(str) == "BAD"
            ].set_index("Predictor")

            common = set(fc.index.astype(str)) & set(
                ba.index.astype(str)
            )
            if common != set(PREDICTOR_ORDER):
                raise ValueError(
                    f"{season} {stage}: FC/BA predictor sets do not match "
                    "the locked environmental-variable list."
                )

            for predictor in PREDICTOR_ORDER:
                fc_row = fc.loc[predictor]
                ba_row = ba.loc[predictor]

                fc_importance = float(
                    fc_row["Relative_Mean_Absolute_SHAP_Percent"]
                )
                ba_importance = float(
                    ba_row["Relative_Mean_Absolute_SHAP_Percent"]
                )
                importance_gmean = safe_geometric_mean(
                    fc_importance,
                    ba_importance,
                )
                importance_mean = 0.5 * (
                    fc_importance + ba_importance
                )

                fc_amplitude = float(
                    fc_row["Relative_SHAP_Amplitude_Percent"]
                )
                ba_amplitude = float(
                    ba_row["Relative_SHAP_Amplitude_Percent"]
                )
                amplitude_gmean = safe_geometric_mean(
                    fc_amplitude,
                    ba_amplitude,
                )
                amplitude_mean = 0.5 * (
                    fc_amplitude + ba_amplitude
                )

                fc_bins = int(fc_row["Bin_N"])
                ba_bins = int(ba_row["Bin_N"])
                eligible = (
                    fc_bins >= MIN_BINS_FOR_NONLINEAR_PANEL
                    and ba_bins >= MIN_BINS_FOR_NONLINEAR_PANEL
                )

                fc_curve = (
                    binned_curve_for_pair(
                        predictor=predictor,
                        season=season,
                        stage=stage,
                        response="FCD",
                        source=merged,
                    )
                )
                ba_curve = (
                    binned_curve_for_pair(
                        predictor=predictor,
                        season=season,
                        stage=stage,
                        response="BAD",
                        source=merged,
                    )
                )

                shape_agreement = paired_bin_shape_agreement(
                    fc_row=fc_row,
                    ba_row=ba_row,
                )

                rows.append(
                    {
                        "Season": season,
                        "Stage": stage,
                        "Stage_Display": STAGE_LABELS[stage],
                        "Predictor": predictor,
                        "Predictor_Group": str(
                            fc_row["Predictor_Group"]
                        ),
                        "Predictor_Group_Display": GROUP_LABELS[
                            str(fc_row["Predictor_Group"])
                        ],
                        "FC_Mean_Absolute_SHAP": float(
                            fc_row["Mean_Absolute_SHAP"]
                        ),
                        "BA_Mean_Absolute_SHAP": float(
                            ba_row["Mean_Absolute_SHAP"]
                        ),
                        "FC_Relative_Importance_Percent": fc_importance,
                        "BA_Relative_Importance_Percent": ba_importance,
                        "Importance_Consensus_Geometric_Mean": (
                            importance_gmean
                        ),
                        "Importance_Consensus_Arithmetic_Mean": (
                            importance_mean
                        ),
                        "Importance_Consensus_Minimum": min(
                            fc_importance,
                            ba_importance,
                        ),
                        "FC_Overall_SHAP_Rank": int(
                            fc_row["Mean_Absolute_SHAP_Rank"]
                        ),
                        "BA_Overall_SHAP_Rank": int(
                            ba_row["Mean_Absolute_SHAP_Rank"]
                        ),
                        "FC_Bin_N": fc_bins,
                        "BA_Bin_N": ba_bins,
                        "FC_SHAP_Amplitude": float(
                            fc_row["Mean_SHAP_Amplitude"]
                        ),
                        "BA_SHAP_Amplitude": float(
                            ba_row["Mean_SHAP_Amplitude"]
                        ),
                        "FC_Relative_Amplitude_Percent": fc_amplitude,
                        "BA_Relative_Amplitude_Percent": ba_amplitude,
                        "Amplitude_Consensus_Geometric_Mean": (
                            amplitude_gmean
                        ),
                        "Amplitude_Consensus_Arithmetic_Mean": (
                            amplitude_mean
                        ),
                        "FC_Curve_Monotonicity_Spearman": float(
                            fc_row[
                                "Feature_Mean_SHAP_Spearman"
                            ]
                        ),
                        "BA_Curve_Monotonicity_Spearman": float(
                            ba_row[
                                "Feature_Mean_SHAP_Spearman"
                            ]
                        ),
                        "FC_Zero_Crossing_N": int(
                            fc_row["Zero_Crossing_N"]
                        ),
                        "BA_Zero_Crossing_N": int(
                            ba_row["Zero_Crossing_N"]
                        ),
                        "FC_Direction_Classification": str(
                            fc_row["Direction_Classification"]
                        ),
                        "BA_Direction_Classification": str(
                            ba_row["Direction_Classification"]
                        ),
                        "FC_High_minus_Low_Mean_SHAP": float(
                            fc_row[
                                "High_minus_Low_Mean_SHAP"
                            ]
                        ),
                        "BA_High_minus_Low_Mean_SHAP": float(
                            ba_row[
                                "High_minus_Low_Mean_SHAP"
                            ]
                        ),
                        "FC_BA_Direction_Agreement": (
                            np.sign(
                                float(
                                    fc_row[
                                        "High_minus_Low_Mean_SHAP"
                                    ]
                                )
                            )
                            == np.sign(
                                float(
                                    ba_row[
                                        "High_minus_Low_Mean_SHAP"
                                    ]
                                )
                            )
                        ),
                        "FC_BA_Bin_Order_Shape_Spearman": (
                            shape_agreement
                        ),
                        "Curve_Eligible": eligible,
                        "Minimum_Bin_N": min(fc_bins, ba_bins),
                    }
                )

    candidates = pd.DataFrame(rows)

    candidates = candidates.sort_values(
        [
            "Season",
            "Stage",
            "Curve_Eligible",
            "Importance_Consensus_Geometric_Mean",
            "Amplitude_Consensus_Geometric_Mean",
            "Importance_Consensus_Arithmetic_Mean",
            "Predictor",
        ],
        ascending=[
            True,
            True,
            False,
            False,
            False,
            False,
            True,
        ],
    ).reset_index(drop=True)

    candidates["Season_Stage_Candidate_Rank"] = (
        candidates.groupby(["Season", "Stage"]).cumcount() + 1
    )

    eligible = candidates.loc[
        candidates["Curve_Eligible"]
    ].copy()

    eligible = eligible.sort_values(
        [
            "Season",
            "Importance_Consensus_Geometric_Mean",
            "Amplitude_Consensus_Geometric_Mean",
            "Importance_Consensus_Arithmetic_Mean",
            "Stage",
            "Predictor",
        ],
        ascending=[
            True,
            False,
            False,
            False,
            True,
            True,
        ],
    ).reset_index(drop=True)

    eligible["Season_Overall_Candidate_Rank"] = (
        eligible.groupby("Season").cumcount() + 1
    )

    top = eligible.loc[
        eligible["Season_Overall_Candidate_Rank"]
        <= TOP_CANDIDATES_PER_SEASON
    ].copy()

    top["Final_Manuscript_Selection_Locked"] = False

    return candidates, top


def binned_curve_for_pair(
    predictor: str,
    season: str,
    stage: str,
    response: str,
    source: pd.DataFrame,
) -> np.ndarray:
    # No-op branch retained for explicit code readability. The actual
    # paired-bin shape calculation is performed from summary rows below.
    # Returning an empty array does not affect ranking or outputs.
    return np.asarray([], dtype=float)


def paired_bin_shape_agreement(
    fc_row: pd.Series,
    ba_row: pd.Series,
) -> float:
    # Individual summary rows do not contain full binned sequences.
    # A direct bin-order curve correlation is added later from the binned
    # plotting table. NaN is used here and replaced during enrichment.
    return float("nan")


def enrich_shape_agreement(
    candidates: pd.DataFrame,
    binned: pd.DataFrame,
) -> pd.DataFrame:
    data = binned.loc[
        binned["Predictor"].astype(str).isin(PREDICTOR_ORDER)
        & binned["Stage"].astype(str).isin(STAGE_ORDER)
        & binned["Season_Label"].astype(str).isin(SEASON_ORDER)
        & binned["Response"].astype(str).isin(RESPONSE_ORDER)
    ].copy()

    data["Bin_Order"] = pd.to_numeric(
        data["Bin_Order"],
        errors="raise",
    )
    data["Mean_SHAP"] = pd.to_numeric(
        data["Mean_SHAP"],
        errors="raise",
    )

    agreements: List[float] = []

    for row in candidates.itertuples():
        subset = data.loc[
            (data["Stage"].astype(str) == row.Stage)
            & (
                data["Season_Label"].astype(str)
                == row.Season
            )
            & (
                data["Predictor"].astype(str)
                == row.Predictor
            )
        ]

        fc = subset.loc[
            subset["Response"].astype(str) == "FCD",
            ["Bin_Order", "Mean_SHAP"],
        ].rename(columns={"Mean_SHAP": "FC_Mean_SHAP"})

        ba = subset.loc[
            subset["Response"].astype(str) == "BAD",
            ["Bin_Order", "Mean_SHAP"],
        ].rename(columns={"Mean_SHAP": "BA_Mean_SHAP"})

        paired = fc.merge(
            ba,
            on="Bin_Order",
            how="inner",
            validate="one_to_one",
        ).sort_values("Bin_Order")

        agreements.append(
            rank_correlation(
                paired["FC_Mean_SHAP"],
                paired["BA_Mean_SHAP"],
            )
            if len(paired) >= 3
            else float("nan")
        )

    enriched = candidates.copy()
    enriched["FC_BA_Bin_Order_Shape_Spearman"] = agreements
    return enriched


def select_top_candidates(
    enriched_candidates: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    eligible = enriched_candidates.loc[
        enriched_candidates["Curve_Eligible"]
    ].copy()

    eligible = eligible.sort_values(
        [
            "Season",
            "Importance_Consensus_Geometric_Mean",
            "Amplitude_Consensus_Geometric_Mean",
            "Importance_Consensus_Arithmetic_Mean",
            "Stage",
            "Predictor",
        ],
        ascending=[
            True,
            False,
            False,
            False,
            True,
            True,
        ],
    ).reset_index(drop=True)

    eligible["Season_Overall_Candidate_Rank"] = (
        eligible.groupby("Season").cumcount() + 1
    )

    top = eligible.loc[
        eligible["Season_Overall_Candidate_Rank"] <= top_n
    ].copy()

    top["Final_Manuscript_Selection_Locked"] = False
    return top


def draw_candidate_axis(
    ax: plt.Axes,
    binned: pd.DataFrame,
    candidate: pd.Series,
) -> None:
    season = str(candidate["Season"])
    stage = str(candidate["Stage"])
    predictor = str(candidate["Predictor"])

    subset = binned.loc[
        (binned["Season_Label"].astype(str) == season)
        & (binned["Stage"].astype(str) == stage)
        & (binned["Predictor"].astype(str) == predictor)
        & binned["Response"].astype(str).isin(RESPONSE_ORDER)
    ].copy()

    definitions = (
        ("FCD", "FC", "o", "-"),
        ("BAD", "BA", "s", "--"),
    )

    for response, label, marker, linestyle in definitions:
        response_subset = subset.loc[
            subset["Response"].astype(str) == response
        ].sort_values("Bin_Order")

        x = response_subset["Feature_Median"].to_numpy(dtype=float)
        y = response_subset["Mean_SHAP"].to_numpy(dtype=float)
        lower = response_subset["SHAP_Q25"].to_numpy(dtype=float)
        upper = response_subset["SHAP_Q75"].to_numpy(dtype=float)

        line, = ax.plot(
            x,
            y,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.8,
            markersize=4.8,
            label=label,
        )
        ax.fill_between(
            x,
            lower,
            upper,
            color=line.get_color(),
            alpha=0.14,
            linewidth=0,
        )

    ax.axhline(
        0.0,
        color="0.45",
        linewidth=0.8,
        linestyle=":",
    )

    rank = int(candidate["Season_Overall_Candidate_Rank"])
    score = float(
        candidate["Importance_Consensus_Geometric_Mean"]
    )

    ax.set_title(
        f"{rank}. {STAGE_LABELS[stage]}: {predictor}",
        loc="left",
        fontsize=11.8,
        fontweight="bold",
        pad=6,
    )
    ax.text(
        0.99,
        0.96,
        (
            f"{GROUP_LABELS[str(candidate['Predictor_Group'])]}\n"
            f"importance={score:.1f}"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.8,
    )
    ax.set_xlabel(predictor, fontsize=10.5)
    ax.set_ylabel("Mean OOF SHAP", fontsize=10.5)
    ax.tick_params(axis="both", labelsize=9.0)
    ax.grid(linewidth=0.6, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def create_candidate_atlas(
    binned: pd.DataFrame,
    top_candidates: pd.DataFrame,
    pdf_path: Path,
    png_paths: Dict[str, Path],
    dpi: int,
) -> Tuple[str, bool]:
    font_family, font_found = find_font()

    matplotlib.rcParams.update(
        {
            "font.family": font_family,
            "font.size": 10,
            "axes.unicode_minus": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    with PdfPages(pdf_path) as pdf:
        for season in SEASON_ORDER:
            candidates = top_candidates.loc[
                top_candidates["Season"].astype(str) == season
            ].sort_values("Season_Overall_Candidate_Rank")

            if candidates.empty:
                raise ValueError(
                    f"No eligible nonlinear-response candidates for {season}."
                )

            n = len(candidates)
            ncols = 2
            nrows = int(math.ceil(n / ncols))

            fig, axes = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                figsize=(10.8, 3.35 * nrows),
                squeeze=False,
            )
            axes_flat = axes.ravel()

            for axis, (_, candidate) in zip(
                axes_flat,
                candidates.iterrows(),
            ):
                draw_candidate_axis(
                    axis,
                    binned=binned,
                    candidate=candidate,
                )

            for axis in axes_flat[n:]:
                axis.axis("off")

            handles, labels = axes_flat[0].get_legend_handles_labels()
            fig.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.015),
                ncol=2,
                frameon=False,
                fontsize=10.5,
            )
            fig.suptitle(
                (
                    f"{season} OOF SHAP nonlinear-response candidates "
                    "(not final manuscript selection)"
                ),
                fontsize=14.0,
                fontweight="bold",
                y=0.995,
            )
            fig.subplots_adjust(
                left=0.08,
                right=0.98,
                top=0.93,
                bottom=0.08,
                hspace=0.48,
                wspace=0.28,
            )

            pdf.savefig(fig, bbox_inches="tight")
            fig.savefig(
                png_paths[season],
                dpi=dpi,
                bbox_inches="tight",
                facecolor="white",
            )
            plt.close(fig)

    return font_family, font_found


def main() -> int:
    args = parse_args()

    if args.top_candidates < 1:
        raise ValueError("--top-candidates must be at least 1.")

    root = args.root.expanduser().resolve()
    shap_dir = root / SHAP_SUBDIR
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else root / DEFAULT_OUTPUT_SUBDIR
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    individual_path = shap_dir / INDIVIDUAL_SHAP_FILENAME
    binned_path = shap_dir / BINNED_SHAP_FILENAME

    output_paths = {
        "all_importance": (
            output_dir
            / "01_All_Variables_OOF_SHAP_Importance.csv"
        ),
        "environmental_importance": (
            output_dir
            / "02_Figure09_Environmental_Variable_Importance.csv"
        ),
        "curve_metrics": (
            output_dir
            / "03_Environmental_SHAP_Curve_Metrics.csv"
        ),
        "candidate_rankings": (
            output_dir
            / "04_Season_Stage_SHAP_Candidate_Ranking.csv"
        ),
        "top_candidates": (
            output_dir
            / "05_Seasonal_Top_SHAP_Candidates.csv"
        ),
        "atlas_pdf": (
            output_dir
            / "06_SHAP_Candidate_Atlas.pdf"
        ),
        "atlas_spring": (
            output_dir
            / "07_SHAP_Candidate_Atlas_Spring.png"
        ),
        "atlas_summer": (
            output_dir
            / "08_SHAP_Candidate_Atlas_Summer.png"
        ),
        "atlas_autumn": (
            output_dir
            / "09_SHAP_Candidate_Atlas_Autumn.png"
        ),
        "importance_qa": (
            output_dir
            / "10_Variable_Importance_QA.csv"
        ),
        "curve_qa": (
            output_dir
            / "11_SHAP_Curve_QA.csv"
        ),
        "manifest": output_dir / "12_Run_Manifest.json",
        "log": output_dir / "13_Run_Log.txt",
        "font_status": output_dir / "14_Font_Status.txt",
    }

    log_lines: List[str] = []

    def log(message: str) -> None:
        print(message, flush=True)
        log_lines.append(message)

    log("=" * 84)
    log("Figure 9 Step 12: variable importance and SHAP candidate preparation")
    log("=" * 84)
    log(f"Script version            : {SCRIPT_VERSION}")
    log(f"UTC start                 : {utc_now_iso()}")
    log(f"Project root              : {root}")
    log(f"Individual SHAP summary   : {individual_path}")
    log(f"Binned SHAP responses     : {binned_path}")
    log(f"Output folder             : {output_dir}")
    log(f"Top candidates per season : {args.top_candidates}")
    log("No Random Forest model will be fitted or retrained.")

    try:
        individual = read_csv_strict(individual_path)
        binned = read_csv_strict(binned_path)

        log(f"Individual SHAP rows      : {len(individual):,}")
        log(f"Binned SHAP rows          : {len(binned):,}")

        validate_inputs(individual, binned)

        (
            all_importance,
            environmental_importance,
            importance_qa,
        ) = prepare_variable_importance(individual)

        curve_metrics, curve_qa = prepare_curve_metrics(binned)

        candidate_rankings, _ = build_candidate_rankings(
            environmental_importance,
            curve_metrics,
        )
        candidate_rankings = enrich_shape_agreement(
            candidate_rankings,
            binned,
        )
        top_candidates = select_top_candidates(
            candidate_rankings,
            top_n=args.top_candidates,
        )

        if not importance_qa["Importance_QA_Passed"].all():
            raise RuntimeError(
                "At least one variable-importance QA row failed."
            )
        if not curve_qa["Curve_QA_Passed"].all():
            raise RuntimeError(
                "At least one SHAP-curve QA row failed."
            )

        all_importance.to_csv(
            output_paths["all_importance"],
            index=False,
            encoding="utf-8-sig",
        )
        environmental_importance.to_csv(
            output_paths["environmental_importance"],
            index=False,
            encoding="utf-8-sig",
        )
        curve_metrics.to_csv(
            output_paths["curve_metrics"],
            index=False,
            encoding="utf-8-sig",
        )
        candidate_rankings.to_csv(
            output_paths["candidate_rankings"],
            index=False,
            encoding="utf-8-sig",
        )
        top_candidates.to_csv(
            output_paths["top_candidates"],
            index=False,
            encoding="utf-8-sig",
        )
        importance_qa.to_csv(
            output_paths["importance_qa"],
            index=False,
            encoding="utf-8-sig",
        )
        curve_qa.to_csv(
            output_paths["curve_qa"],
            index=False,
            encoding="utf-8-sig",
        )

        png_paths = {
            "Spring": output_paths["atlas_spring"],
            "Summer": output_paths["atlas_summer"],
            "Autumn": output_paths["atlas_autumn"],
        }
        font_family, font_found = create_candidate_atlas(
            binned=binned,
            top_candidates=top_candidates,
            pdf_path=output_paths["atlas_pdf"],
            png_paths=png_paths,
            dpi=args.dpi,
        )

        output_paths["font_status"].write_text(
            (
                f"Requested font: {FONT_FAMILY}\n"
                f"Font found: {font_found}\n"
                f"Font used: {font_family}\n"
            ),
            encoding="utf-8",
        )

        manifest = {
            "script_version": SCRIPT_VERSION,
            "created_utc": utc_now_iso(),
            "project_root": str(root),
            "input_files": {
                "individual_oof_shap_summary": {
                    "path": str(individual_path),
                    "sha256": sha256_file(individual_path),
                    "rows": int(len(individual)),
                },
                "binned_oof_shap_response": {
                    "path": str(binned_path),
                    "sha256": sha256_file(binned_path),
                    "rows": int(len(binned)),
                },
            },
            "figure09_preparation_definition": {
                "main_importance_metric": "Mean_Absolute_SHAP",
                "main_figure_environmental_predictors_only": True,
                "country_context_retained_in_complete_table": True,
                "environmental_predictor_count": len(PREDICTOR_ORDER),
                "importance_normalization": (
                    "Each environmental predictor's mean absolute OOF SHAP "
                    "divided by the maximum environmental predictor value "
                    "within the same stage-season-response model."
                ),
                "candidate_primary_ranking": (
                    "Geometric mean of FC and BA relative variable-level "
                    "mean absolute OOF SHAP importance."
                ),
                "candidate_secondary_ranking": (
                    "Geometric mean of FC and BA relative binned SHAP "
                    "curve amplitudes."
                ),
                "minimum_bins_for_candidate": (
                    MIN_BINS_FOR_NONLINEAR_PANEL
                ),
                "top_candidates_per_season": args.top_candidates,
                "final_manuscript_panels_locked": False,
                "predictor_order": list(PREDICTOR_ORDER),
            },
            "qa": {
                "passed": True,
                "all_variable_importance_rows": int(
                    len(all_importance)
                ),
                "environmental_importance_rows": int(
                    len(environmental_importance)
                ),
                "curve_metric_rows": int(len(curve_metrics)),
                "candidate_ranking_rows": int(
                    len(candidate_rankings)
                ),
                "top_candidate_rows": int(len(top_candidates)),
                "importance_qa_passed": bool(
                    importance_qa["Importance_QA_Passed"].all()
                ),
                "curve_qa_passed": bool(
                    curve_qa["Curve_QA_Passed"].all()
                ),
            },
            "software": {
                "python": sys.version,
                "platform": platform.platform(),
                "pandas": pd.__version__,
                "numpy": np.__version__,
                "matplotlib": matplotlib.__version__,
            },
            "outputs": {
                key: str(path)
                for key, path in output_paths.items()
                if key not in {"manifest", "log"}
            },
        }

        output_paths["manifest"].write_text(
            json.dumps(
                manifest,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        log("Input and preparation QA : PASSED")
        log(
            f"All-variable rows         : "
            f"{len(all_importance):,}"
        )
        log(
            f"Environmental importance : "
            f"{len(environmental_importance):,}"
        )
        log(
            f"Curve metric rows         : "
            f"{len(curve_metrics):,}"
        )
        log(
            f"Candidate ranking rows    : "
            f"{len(candidate_rankings):,}"
        )
        log(
            f"Top candidate rows        : "
            f"{len(top_candidates):,}"
        )

        for season in SEASON_ORDER:
            log(f"{season} top candidates:")
            season_rows = top_candidates.loc[
                top_candidates["Season"].astype(str) == season
            ].sort_values("Season_Overall_Candidate_Rank")
            for row in season_rows.itertuples():
                log(
                    f"  {int(row.Season_Overall_Candidate_Rank)}. "
                    f"{row.Stage_Display} / {row.Predictor} "
                    f"({row.Predictor_Group}) | "
                    f"importance={row.Importance_Consensus_Geometric_Mean:.2f} | "
                    f"amplitude={row.Amplitude_Consensus_Geometric_Mean:.2f}"
                )

        log(f"Candidate atlas PDF       : {output_paths['atlas_pdf']}")
        log(f"UTC finish                : {utc_now_iso()}")
        log(
            "Figure 9 Step 12 completed successfully. "
            "No final seasonal SHAP panels were locked."
        )
        return_code = 0

    except Exception as exc:
        log(f"FAILED: {type(exc).__name__}: {exc}")
        log(f"UTC failure              : {utc_now_iso()}")
        return_code = 1

    output_paths["log"].write_text(
        "\n".join(log_lines) + "\n",
        encoding="utf-8",
    )

    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
