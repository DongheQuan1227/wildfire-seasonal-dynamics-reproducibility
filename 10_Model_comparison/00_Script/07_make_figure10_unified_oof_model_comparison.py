#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Create manuscript Figure 10: unified temporal/spatial OOF model comparison
and residual spatial autocorrelation.

The script reads only locked outputs from Model Comparison Step 06. It does
not fit, tune, select, or predict any model. It does not recompute bootstrap
confidence intervals or Moran's I.

Figure structure
----------------
(a) Temporal OOF: overall forest-area-standardized rate RMSE.
(b) Spatial OOF: overall forest-area-standardized rate RMSE.
(c) Temporal OOF: conditional-positive rate RMSE.
(d) Spatial OOF: conditional-positive rate RMSE.
(e) Temporal OOF: residual Moran's I.
(f) Spatial OOF: residual Moran's I.

For panels (a)-(d), error is expressed relative to the corresponding
Baseline NB1 value within each season-response-validation combination.
A black outer ring marks the optimized ZINB1/NB1 or Hurdle RF model when the
paired grid-cluster bootstrap 95% confidence interval for the RF-minus-
ZINB1/NB1 difference excludes zero in favor of that model.

Formal code-package location:
    <REPOSITORY_ROOT>\10_Model_comparison\00_Script\
        07_make_figure10_unified_oof_model_comparison.py

Default project root:
    <REPOSITORY_ROOT>

Default output directory:
    <REPOSITORY_ROOT>\10_Model_comparison\07_Manuscript_Figure_10
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


SCRIPT_VERSION = "1.0.6"

DEFAULT_ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = Path("10_Model_comparison")
LOCKED_RESULT_SUBDIR = MODULE_DIR / "06_Model_Comparison_Tables_Figures"
DEFAULT_OUTPUT_SUBDIR = MODULE_DIR / "07_Manuscript_Figure_10"

OVERALL_FILENAME = "02_Main_Table_Overall_Model_Performance.csv"
PAIRED_FILENAME = "03_Main_Table_RF_vs_Regression_Paired_Comparison.csv"
MORAN_FILENAME = "04_Main_Table_Residual_Moran_I.csv"

FONT_FAMILY = "Times New Roman"
OUTPUT_DPI = 600
FIGURE_WIDTH_IN = 12.2
FIGURE_HEIGHT_IN = 11.2

VALIDATION_ORDER: Sequence[str] = ("Temporal", "Spatial")
SEASON_ORDER: Sequence[Tuple[int, str]] = (
    (1, "Spring"),
    (2, "Summer"),
    (3, "Autumn"),
)
RESPONSE_ORDER: Sequence[str] = (
    "Fire_Count",
    "Burned_Pixel_Count",
)
MODEL_ORDER: Sequence[str] = (
    "Baseline_NB1",
    "Final_Regression",
    "Final_Hurdle_RF",
)

MODEL_LABELS: Dict[str, str] = {
    "Baseline_NB1": "Baseline NB1",
    "Final_Regression": "Optimized ZINB1/NB1",
    "Final_Hurdle_RF": "Hurdle RF",
}
MODEL_MARKERS: Dict[str, str] = {
    "Baseline_NB1": "o",
    "Final_Regression": "s",
    "Final_Hurdle_RF": "^",
}
MODEL_LINESTYLES: Dict[str, object] = {
    "Baseline_NB1": (0, (2, 2)),
    "Final_Regression": "-",
    "Final_Hurdle_RF": "-",
}
RESPONSE_LABELS: Dict[str, str] = {
    "Fire_Count": "FC",
    "Burned_Pixel_Count": "BA",
}
STABLE_CONCLUSIONS = {
    "Candidate_stably_better",
    "Reference_stably_better",
    "No_stable_difference",
}

COMBINATION_ORDER: Sequence[Tuple[int, str]] = tuple(
    (season_code, response)
    for season_code, _ in SEASON_ORDER
    for response in RESPONSE_ORDER
)
COMBINATION_POSITION = {
    combination: index
    for index, combination in enumerate(COMBINATION_ORDER)
}

METRIC_DEFINITIONS = (
    {
        "metric": "Rate_RMSE",
        "paired_prefix": "Rate_RMSE",
        "row": 0,
        "title": "Overall rate RMSE",
    },
    {
        "metric": "Positive_Conditional_Rate_RMSE",
        "paired_prefix": "Positive_Conditional_Rate_RMSE",
        "row": 1,
        "title": "Conditional-positive rate RMSE",
    },
)

REQUIRED_OVERALL_COLUMNS = {
    "Validation_Scheme",
    "Season",
    "Season_Label",
    "Count_Response",
    "Model",
    "Rate_RMSE",
    "Positive_Conditional_Rate_RMSE",
}

REQUIRED_PAIRED_COLUMNS = {
    "Validation_Scheme",
    "Season",
    "Season_Label",
    "Count_Response",
    "Rate_RMSE_Stable_Conclusion",
    "Rate_RMSE_Difference_CI_Lower",
    "Rate_RMSE_Difference_CI_Upper",
    "Positive_Conditional_Rate_RMSE_Stable_Conclusion",
    "Positive_Conditional_Rate_RMSE_Difference_CI_Lower",
    "Positive_Conditional_Rate_RMSE_Difference_CI_Upper",
}

REQUIRED_MORAN_COLUMNS = {
    "Validation_Scheme",
    "Season_Code",
    "Season_Label",
    "Count_Response",
    "Model_Role",
    "Moran_I",
    "Combined_Primary_Global_BH_FDR",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create manuscript Figure 10 from locked unified model-"
            "comparison, paired-bootstrap, and Moran's I outputs."
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
            "<root>/10_Model_comparison/07_Manuscript_Figure_10"
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=OUTPUT_DPI,
        help=f"Raster output resolution. Default: {OUTPUT_DPI}",
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


def find_font() -> Tuple[str, bool]:
    available = {font.name for font in font_manager.fontManager.ttflist}
    if FONT_FAMILY in available:
        return FONT_FAMILY, True
    return "DejaVu Serif", False


def validate_and_prepare_relative_error(
    overall: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    require_columns(
        overall,
        REQUIRED_OVERALL_COLUMNS,
        "Overall model-performance table",
    )

    data = overall.loc[
        overall["Validation_Scheme"].astype(str).isin(VALIDATION_ORDER)
        & overall["Season"].isin([item[0] for item in SEASON_ORDER])
        & overall["Count_Response"].astype(str).isin(RESPONSE_ORDER)
        & overall["Model"].astype(str).isin(MODEL_ORDER)
    ].copy()

    duplicate_mask = data.duplicated(
        subset=[
            "Validation_Scheme",
            "Season",
            "Count_Response",
            "Model",
        ],
        keep=False,
    )
    if duplicate_mask.any():
        raise ValueError(
            "Duplicate validation-season-response-model rows were found "
            "in the overall performance table."
        )

    expected_rows = (
        len(VALIDATION_ORDER)
        * len(COMBINATION_ORDER)
        * len(MODEL_ORDER)
    )
    if len(data) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} overall-performance rows; "
            f"found {len(data)}."
        )

    rows: List[Dict[str, object]] = []
    qa_rows: List[Dict[str, object]] = []

    for validation in VALIDATION_ORDER:
        for season_code, season_label in SEASON_ORDER:
            for response in RESPONSE_ORDER:
                subset = data.loc[
                    (data["Validation_Scheme"].astype(str) == validation)
                    & (data["Season"] == season_code)
                    & (
                        data["Count_Response"].astype(str)
                        == response
                    )
                ].copy()

                if set(subset["Model"].astype(str)) != set(MODEL_ORDER):
                    raise ValueError(
                        f"{validation} {season_label} {response}: "
                        "the three expected models are not all present."
                    )

                baseline_row = subset.loc[
                    subset["Model"].astype(str) == "Baseline_NB1"
                ]
                if len(baseline_row) != 1:
                    raise ValueError(
                        f"{validation} {season_label} {response}: "
                        "expected exactly one Baseline NB1 row."
                    )

                for metric_definition in METRIC_DEFINITIONS:
                    metric = metric_definition["metric"]
                    subset[metric] = pd.to_numeric(
                        subset[metric],
                        errors="raise",
                    )
                    baseline = float(baseline_row.iloc[0][metric])
                    if not np.isfinite(baseline) or baseline <= 0:
                        raise ValueError(
                            f"{validation} {season_label} {response} "
                            f"{metric}: invalid baseline value {baseline}."
                        )

                    for model in MODEL_ORDER:
                        model_row = subset.loc[
                            subset["Model"].astype(str) == model
                        ]
                        value = float(model_row.iloc[0][metric])
                        relative = 100.0 * value / baseline
                        rows.append(
                            {
                                "Validation_Scheme": validation,
                                "Season": season_code,
                                "Season_Label": season_label,
                                "Count_Response": response,
                                "Response_Label": RESPONSE_LABELS[response],
                                "Combination_Position": (
                                    COMBINATION_POSITION[
                                        (season_code, response)
                                    ]
                                ),
                                "Metric": metric,
                                "Metric_Display": metric_definition["title"],
                                "Model": model,
                                "Model_Label": MODEL_LABELS[model],
                                "Metric_Value": value,
                                "Baseline_Metric_Value": baseline,
                                "Relative_to_Baseline_Percent": relative,
                            }
                        )

                qa_rows.append(
                    {
                        "Validation_Scheme": validation,
                        "Season": season_code,
                        "Season_Label": season_label,
                        "Count_Response": response,
                        "Model_N": int(len(subset)),
                        "Relative_Baseline_Rate_RMSE": 100.0,
                        "Relative_Baseline_Positive_Conditional_Rate_RMSE": 100.0,
                        "Relative_Error_QA_Passed": bool(
                            len(subset) == len(MODEL_ORDER)
                        ),
                    }
                )

    plotting = pd.DataFrame(rows)
    qa = pd.DataFrame(qa_rows)
    return plotting, qa


def validate_and_prepare_paired(
    paired: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    require_columns(
        paired,
        REQUIRED_PAIRED_COLUMNS,
        "RF-versus-ZINB1/NB1 paired-comparison table",
    )

    data = paired.loc[
        paired["Validation_Scheme"].astype(str).isin(VALIDATION_ORDER)
        & paired["Season"].isin([item[0] for item in SEASON_ORDER])
        & paired["Count_Response"].astype(str).isin(RESPONSE_ORDER)
    ].copy()

    duplicate_mask = data.duplicated(
        subset=[
            "Validation_Scheme",
            "Season",
            "Count_Response",
        ],
        keep=False,
    )
    if duplicate_mask.any():
        raise ValueError(
            "Duplicate validation-season-response rows were found in the "
            "paired RF-versus-ZINB1/NB1 comparison table."
        )

    expected_rows = len(VALIDATION_ORDER) * len(COMBINATION_ORDER)
    if len(data) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} paired-comparison rows; "
            f"found {len(data)}."
        )

    rows: List[Dict[str, object]] = []
    qa_rows: List[Dict[str, object]] = []

    for row in data.itertuples():
        validation = str(row.Validation_Scheme)
        season = int(row.Season)
        response = str(row.Count_Response)

        for definition in METRIC_DEFINITIONS:
            prefix = definition["paired_prefix"]
            conclusion = str(
                getattr(row, f"{prefix}_Stable_Conclusion")
            )
            ci_lower = float(
                getattr(row, f"{prefix}_Difference_CI_Lower")
            )
            ci_upper = float(
                getattr(row, f"{prefix}_Difference_CI_Upper")
            )

            if conclusion not in STABLE_CONCLUSIONS:
                raise ValueError(
                    f"Unexpected stable conclusion: {conclusion}"
                )

            if conclusion == "Candidate_stably_better":
                supported_model = "Final_Hurdle_RF"
            elif conclusion == "Reference_stably_better":
                supported_model = "Final_Regression"
            else:
                supported_model = ""

            ci_excludes_zero = bool(
                ci_upper < 0 or ci_lower > 0
            )
            expected_exclusion = (
                conclusion != "No_stable_difference"
            )
            if ci_excludes_zero != expected_exclusion:
                raise ValueError(
                    f"{validation} {season} {response} {prefix}: "
                    "stable conclusion is inconsistent with the "
                    "reported confidence interval."
                )

            rows.append(
                {
                    "Validation_Scheme": validation,
                    "Season": season,
                    "Season_Label": next(
                        label
                        for code, label in SEASON_ORDER
                        if code == season
                    ),
                    "Count_Response": response,
                    "Response_Label": RESPONSE_LABELS[response],
                    "Combination_Position": (
                        COMBINATION_POSITION[(season, response)]
                    ),
                    "Metric": definition["metric"],
                    "Stable_Conclusion": conclusion,
                    "Supported_Model": supported_model,
                    "Difference_CI_Lower": ci_lower,
                    "Difference_CI_Upper": ci_upper,
                    "CI_Excludes_Zero": ci_excludes_zero,
                }
            )

        qa_rows.append(
            {
                "Validation_Scheme": validation,
                "Season": season,
                "Count_Response": response,
                "Paired_Row_QA_Passed": True,
            }
        )

    plotting = pd.DataFrame(rows)
    qa = pd.DataFrame(qa_rows)
    return plotting, qa


def validate_and_prepare_moran(
    moran: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    require_columns(
        moran,
        REQUIRED_MORAN_COLUMNS,
        "Residual Moran's I table",
    )

    data = moran.loc[
        moran["Validation_Scheme"].astype(str).isin(VALIDATION_ORDER)
        & moran["Season_Code"].isin([item[0] for item in SEASON_ORDER])
        & moran["Count_Response"].astype(str).isin(RESPONSE_ORDER)
        & moran["Model_Role"].astype(str).isin(MODEL_ORDER)
    ].copy()

    duplicate_mask = data.duplicated(
        subset=[
            "Validation_Scheme",
            "Season_Code",
            "Count_Response",
            "Model_Role",
        ],
        keep=False,
    )
    if duplicate_mask.any():
        raise ValueError(
            "Duplicate validation-season-response-model rows were found "
            "in the residual Moran's I table."
        )

    expected_rows = (
        len(VALIDATION_ORDER)
        * len(COMBINATION_ORDER)
        * len(MODEL_ORDER)
    )
    if len(data) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} Moran rows; found {len(data)}."
        )

    data["Moran_I"] = pd.to_numeric(
        data["Moran_I"],
        errors="raise",
    )
    data["Combined_Primary_Global_BH_FDR"] = pd.to_numeric(
        data["Combined_Primary_Global_BH_FDR"],
        errors="raise",
    )

    data["Season"] = data["Season_Code"].astype(int)
    data["Response_Label"] = data["Count_Response"].map(
        RESPONSE_LABELS
    )
    data["Combination_Position"] = [
        COMBINATION_POSITION[(int(season), str(response))]
        for season, response in zip(
            data["Season_Code"],
            data["Count_Response"],
        )
    ]
    data["Model"] = data["Model_Role"].astype(str)
    data["Model_Label"] = data["Model"].map(MODEL_LABELS)

    minimums = (
        data.groupby(
            [
                "Validation_Scheme",
                "Season_Code",
                "Count_Response",
            ],
            as_index=False,
        )["Moran_I"]
        .min()
        .rename(columns={"Moran_I": "Minimum_Moran_I"})
    )
    data = data.merge(
        minimums,
        on=[
            "Validation_Scheme",
            "Season_Code",
            "Count_Response",
        ],
        how="left",
        validate="many_to_one",
    )
    data["Lowest_Moran_Within_Combination"] = np.isclose(
        data["Moran_I"],
        data["Minimum_Moran_I"],
        rtol=0,
        atol=1e-12,
    )

    qa_rows = []
    for validation in VALIDATION_ORDER:
        for season_code, season_label in SEASON_ORDER:
            for response in RESPONSE_ORDER:
                subset = data.loc[
                    (data["Validation_Scheme"].astype(str) == validation)
                    & (data["Season_Code"] == season_code)
                    & (
                        data["Count_Response"].astype(str)
                        == response
                    )
                ]
                qa_rows.append(
                    {
                        "Validation_Scheme": validation,
                        "Season": season_code,
                        "Season_Label": season_label,
                        "Count_Response": response,
                        "Model_N": int(len(subset)),
                        "Lowest_Moran_Model_N": int(
                            subset[
                                "Lowest_Moran_Within_Combination"
                            ].sum()
                        ),
                        "All_Primary_Moran_FDR_Significant": bool(
                            (
                                subset[
                                    "Combined_Primary_Global_BH_FDR"
                                ]
                                < 0.05
                            ).all()
                        ),
                        "Moran_QA_Passed": bool(
                            len(subset) == len(MODEL_ORDER)
                            and subset[
                                "Lowest_Moran_Within_Combination"
                            ].sum()
                            >= 1
                        ),
                    }
                )

    qa = pd.DataFrame(qa_rows)
    return data.reset_index(drop=True), qa


def add_combination_axis(
    ax: plt.Axes,
    show_labels: bool,
) -> None:
    ax.set_xlim(-0.5, len(COMBINATION_ORDER) - 0.5)
    ax.set_xticks(range(len(COMBINATION_ORDER)))

    if show_labels:
        ax.set_xticklabels(
            ["FC", "BA", "FC", "BA", "FC", "BA"],
            fontsize=18.0,
        )
        ax.tick_params(axis="x", length=0, pad=4)

        for center, (_, season_label) in zip(
            (0.5, 2.5, 4.5),
            SEASON_ORDER,
        ):
            ax.text(
                center,
                -0.15,
                season_label,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=18.0,
                clip_on=False,
            )
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)

    for boundary in (1.5, 3.5):
        ax.axvline(
            boundary,
            linewidth=0.8,
            color="0.72",
            zorder=1,
        )


def draw_relative_error_panel(
    ax: plt.Axes,
    relative_data: pd.DataFrame,
    paired_data: pd.DataFrame,
    validation: str,
    metric: str,
    panel_label: str,
    title: str,
    show_x_labels: bool,
    show_y_label: bool,
) -> List[Line2D]:
    subset = relative_data.loc[
        (relative_data["Validation_Scheme"].astype(str) == validation)
        & (relative_data["Metric"].astype(str) == metric)
    ].copy()

    handles: List[Line2D] = []

    for model in MODEL_ORDER:
        model_data = (
            subset.loc[
                subset["Model"].astype(str) == model
            ]
            .sort_values("Combination_Position")
        )

        line, = ax.plot(
            model_data["Combination_Position"],
            model_data["Relative_to_Baseline_Percent"],
            marker=MODEL_MARKERS[model],
            linestyle=MODEL_LINESTYLES[model],
            linewidth=2.0,
            markersize=7.0,
            label=MODEL_LABELS[model],
        )
        handles.append(line)

    stable = paired_data.loc[
        (paired_data["Validation_Scheme"].astype(str) == validation)
        & (paired_data["Metric"].astype(str) == metric)
        & paired_data["CI_Excludes_Zero"].astype(bool)
    ].copy()

    for row in stable.itertuples():
        supported_model = str(row.Supported_Model)
        value_row = subset.loc[
            (
                subset["Combination_Position"]
                == int(row.Combination_Position)
            )
            & (
                subset["Model"].astype(str)
                == supported_model
            )
        ]
        if len(value_row) != 1:
            raise ValueError(
                "Could not match a bootstrap-supported model to its "
                "relative-error plotting row."
            )

        ax.scatter(
            [int(row.Combination_Position)],
            [
                float(
                    value_row.iloc[0][
                        "Relative_to_Baseline_Percent"
                    ]
                )
            ],
            s=150,
            facecolors="none",
            edgecolors="black",
            linewidths=1.5,
            zorder=5,
        )

    ax.axhline(
        100.0,
        linewidth=0.9,
        linestyle=":",
        color="0.45",
        zorder=1,
    )
    ax.set_ylim(50.0, 132.0)
    ax.set_title(
        f"{panel_label} {validation} OOF: {title}",
        loc="left",
        fontsize=20.0,
        fontweight="bold",
        pad=8,
    )
    if show_y_label:
        ax.set_ylabel(
            "Relative RMSE\n(% of baseline)",
            fontsize=18.75,
        )

    add_combination_axis(ax, show_labels=show_x_labels)
    ax.tick_params(axis="y", labelsize=16.5)
    ax.grid(
        axis="y",
        linewidth=0.7,
        alpha=0.35,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return handles


def draw_moran_panel(
    ax: plt.Axes,
    moran_data: pd.DataFrame,
    validation: str,
    panel_label: str,
    show_x_labels: bool,
    show_y_label: bool,
) -> None:
    subset = moran_data.loc[
        moran_data["Validation_Scheme"].astype(str) == validation
    ].copy()

    for model in MODEL_ORDER:
        model_data = (
            subset.loc[
                subset["Model"].astype(str) == model
            ]
            .sort_values("Combination_Position")
        )
        ax.plot(
            model_data["Combination_Position"],
            model_data["Moran_I"],
            marker=MODEL_MARKERS[model],
            linestyle=MODEL_LINESTYLES[model],
            linewidth=2.0,
            markersize=7.0,
            label=MODEL_LABELS[model],
        )

    lowest = subset.loc[
        subset["Lowest_Moran_Within_Combination"].astype(bool)
    ]
    ax.scatter(
        lowest["Combination_Position"],
        lowest["Moran_I"],
        marker="D",
        s=105,
        facecolors="none",
        edgecolors="black",
        linewidths=1.35,
        zorder=5,
    )

    ax.set_ylim(0.15, 0.55)
    ax.set_title(
        f"{panel_label} {validation} OOF: residual Moran's I",
        loc="left",
        fontsize=20.0,
        fontweight="bold",
        pad=8,
    )
    if show_y_label:
        ax.set_ylabel(
            "Residual Moran's I",
            fontsize=18.75,
        )

    add_combination_axis(ax, show_labels=show_x_labels)
    ax.tick_params(axis="y", labelsize=16.5)
    ax.grid(
        axis="y",
        linewidth=0.7,
        alpha=0.35,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def make_figure(
    relative_data: pd.DataFrame,
    paired_data: pd.DataFrame,
    moran_data: pd.DataFrame,
    output_paths: Dict[str, Path],
    dpi: int,
) -> None:
    font_family, font_found = find_font()

    matplotlib.rcParams.update(
        {
            "font.family": font_family,
            "font.size": 18,
            "axes.unicode_minus": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN),
        sharex=False,
        constrained_layout=False,
    )

    handles = draw_relative_error_panel(
        ax=axes[0, 0],
        relative_data=relative_data,
        paired_data=paired_data,
        validation="Temporal",
        metric="Rate_RMSE",
        panel_label="(a)",
        title="overall rate error",
        show_x_labels=False,
        show_y_label=True,
    )
    draw_relative_error_panel(
        ax=axes[0, 1],
        relative_data=relative_data,
        paired_data=paired_data,
        validation="Spatial",
        metric="Rate_RMSE",
        panel_label="(b)",
        title="overall rate error",
        show_x_labels=False,
        show_y_label=False,
    )
    draw_relative_error_panel(
        ax=axes[1, 0],
        relative_data=relative_data,
        paired_data=paired_data,
        validation="Temporal",
        metric="Positive_Conditional_Rate_RMSE",
        panel_label="(c)",
        title="positive-conditional error",
        show_x_labels=False,
        show_y_label=True,
    )
    draw_relative_error_panel(
        ax=axes[1, 1],
        relative_data=relative_data,
        paired_data=paired_data,
        validation="Spatial",
        metric="Positive_Conditional_Rate_RMSE",
        panel_label="(d)",
        title="positive-conditional error",
        show_x_labels=False,
        show_y_label=False,
    )
    draw_moran_panel(
        ax=axes[2, 0],
        moran_data=moran_data,
        validation="Temporal",
        panel_label="(e)",
        show_x_labels=True,
        show_y_label=True,
    )
    draw_moran_panel(
        ax=axes[2, 1],
        moran_data=moran_data,
        validation="Spatial",
        panel_label="(f)",
        show_x_labels=True,
        show_y_label=False,
    )

    model_legend = fig.legend(
        handles=handles,
        labels=[MODEL_LABELS[model] for model in MODEL_ORDER],
        loc="upper center",
        bbox_to_anchor=(0.50, 0.095),
        ncol=3,
        frameon=False,
        fontsize=18.0,
        handlelength=2.3,
        columnspacing=1.35,
    )

    bootstrap_ring = Line2D(
        [0],
        [0],
        marker="o",
        markersize=10.0,
        markerfacecolor="none",
        markeredgecolor="black",
        markeredgewidth=1.5,
        linestyle="none",
        label=(
            "Paired bootstrap-supported RF–ZINB1/NB1 winner "
            "(95% CI excludes zero)"
        ),
    )
    lowest_moran = Line2D(
        [0],
        [0],
        marker="D",
        markersize=8.0,
        markerfacecolor="none",
        markeredgecolor="black",
        markeredgewidth=1.35,
        linestyle="none",
        label="Lowest residual Moran's I within combination",
    )

    fig.legend(
        handles=[bootstrap_ring, lowest_moran],
        loc="lower center",
        bbox_to_anchor=(0.50, 0.012),
        ncol=2,
        frameon=False,
        fontsize=15.75,
        handletextpad=0.55,
        columnspacing=1.6,
    )

    fig.subplots_adjust(
        left=0.095,
        right=0.985,
        top=0.935,
        bottom=0.15,
        hspace=0.34,
        wspace=0.22,
    )

    fig.savefig(
        output_paths["pdf"],
        bbox_inches="tight",
    )
    fig.savefig(
        output_paths["png"],
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    try:
        fig.savefig(
            output_paths["tif"],
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
            pil_kwargs={"compression": "tiff_lzw"},
        )
    except (TypeError, ValueError):
        fig.savefig(
            output_paths["tif"],
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)

    output_paths["font_status"].write_text(
        (
            f"Requested font: {FONT_FAMILY}\n"
            f"Font found: {font_found}\n"
            f"Font used: {font_family}\n"
        ),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()

    root = args.root.expanduser().resolve()
    locked_dir = root / LOCKED_RESULT_SUBDIR
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else root / DEFAULT_OUTPUT_SUBDIR
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    overall_path = locked_dir / OVERALL_FILENAME
    paired_path = locked_dir / PAIRED_FILENAME
    moran_path = locked_dir / MORAN_FILENAME

    output_paths = {
        "pdf": (
            output_dir
            / "Figure_10_Unified_OOF_Model_Comparison.pdf"
        ),
        "png": (
            output_dir
            / "Figure_10_Unified_OOF_Model_Comparison.png"
        ),
        "tif": (
            output_dir
            / "Figure_10_Unified_OOF_Model_Comparison.tif"
        ),
        "relative_plotting": (
            output_dir
            / "Figure_10_Relative_Error_Plotting_Data.csv"
        ),
        "paired_plotting": (
            output_dir
            / "Figure_10_Paired_Bootstrap_Plotting_Data.csv"
        ),
        "moran_plotting": (
            output_dir
            / "Figure_10_Residual_Moran_Plotting_Data.csv"
        ),
        "qa": (
            output_dir
            / "Figure_10_QA_Summary.csv"
        ),
        "manifest": (
            output_dir
            / "Figure_10_Run_Manifest.json"
        ),
        "log": (
            output_dir
            / "Figure_10_Run_Log.txt"
        ),
        "font_status": (
            output_dir
            / "Figure_10_Font_Status.txt"
        ),
    }

    log_lines: List[str] = []

    def log(message: str) -> None:
        print(message, flush=True)
        log_lines.append(message)

    log("=" * 84)
    log("Manuscript Figure 10: unified OOF model comparison")
    log("=" * 84)
    log(f"Script version          : {SCRIPT_VERSION}")
    log(f"UTC start               : {utc_now_iso()}")
    log(f"Project root            : {root}")
    log(f"Overall performance     : {overall_path}")
    log(f"Paired comparison       : {paired_path}")
    log(f"Residual Moran's I      : {moran_path}")
    log(f"Output folder           : {output_dir}")
    log(
        "No model fitting, prediction, bootstrap, or Moran's I "
        "recomputation will be performed."
    )

    try:
        overall = read_csv_strict(overall_path)
        paired = read_csv_strict(paired_path)
        moran = read_csv_strict(moran_path)

        log(f"Overall rows            : {len(overall):,}")
        log(f"Paired rows             : {len(paired):,}")
        log(f"Moran rows              : {len(moran):,}")

        relative_plotting, relative_qa = (
            validate_and_prepare_relative_error(overall)
        )
        paired_plotting, paired_qa = (
            validate_and_prepare_paired(paired)
        )
        moran_plotting, moran_qa = (
            validate_and_prepare_moran(moran)
        )

        if not relative_qa["Relative_Error_QA_Passed"].all():
            raise RuntimeError(
                "At least one relative-error QA row failed."
            )
        if not paired_qa["Paired_Row_QA_Passed"].all():
            raise RuntimeError(
                "At least one paired-comparison QA row failed."
            )
        if not moran_qa["Moran_QA_Passed"].all():
            raise RuntimeError(
                "At least one Moran QA row failed."
            )

        relative_plotting.to_csv(
            output_paths["relative_plotting"],
            index=False,
            encoding="utf-8-sig",
        )
        paired_plotting.to_csv(
            output_paths["paired_plotting"],
            index=False,
            encoding="utf-8-sig",
        )
        moran_plotting.to_csv(
            output_paths["moran_plotting"],
            index=False,
            encoding="utf-8-sig",
        )

        relative_qa["QA_Type"] = "Relative error"
        paired_qa["QA_Type"] = "Paired bootstrap"
        moran_qa["QA_Type"] = "Residual Moran"
        qa = pd.concat(
            [relative_qa, paired_qa, moran_qa],
            ignore_index=True,
            sort=False,
        )
        qa.to_csv(
            output_paths["qa"],
            index=False,
            encoding="utf-8-sig",
        )

        log("Input and figure QA    : PASSED")
        log(
            f"Relative plotting rows  : "
            f"{len(relative_plotting):,}"
        )
        log(
            f"Paired plotting rows    : "
            f"{len(paired_plotting):,}"
        )
        log(
            f"Moran plotting rows     : "
            f"{len(moran_plotting):,}"
        )

        for validation in VALIDATION_ORDER:
            for metric_definition in METRIC_DEFINITIONS:
                subset = paired_plotting.loc[
                    (
                        paired_plotting[
                            "Validation_Scheme"
                        ].astype(str)
                        == validation
                    )
                    & (
                        paired_plotting["Metric"].astype(str)
                        == metric_definition["metric"]
                    )
                ]
                counts = (
                    subset["Stable_Conclusion"]
                    .value_counts()
                    .to_dict()
                )
                log(
                    f"{validation} / "
                    f"{metric_definition['metric']}: "
                    f"RF better="
                    f"{counts.get('Candidate_stably_better', 0)}, "
                    f"ZINB1/NB1 better="
                    f"{counts.get('Reference_stably_better', 0)}, "
                    f"no stable difference="
                    f"{counts.get('No_stable_difference', 0)}"
                )

        for validation in VALIDATION_ORDER:
            subset = moran_plotting.loc[
                moran_plotting[
                    "Validation_Scheme"
                ].astype(str)
                == validation
            ]
            winners = (
                subset.loc[
                    subset[
                        "Lowest_Moran_Within_Combination"
                    ].astype(bool)
                ]
                .groupby("Model")
                .size()
                .to_dict()
            )
            log(
                f"{validation} lowest Moran counts: "
                + ", ".join(
                    f"{MODEL_LABELS[model]}="
                    f"{int(winners.get(model, 0))}"
                    for model in MODEL_ORDER
                )
            )

        make_figure(
            relative_data=relative_plotting,
            paired_data=paired_plotting,
            moran_data=moran_plotting,
            output_paths=output_paths,
            dpi=args.dpi,
        )

        manifest = {
            "script_version": SCRIPT_VERSION,
            "created_utc": utc_now_iso(),
            "project_root": str(root),
            "input_files": {
                "overall_model_performance": {
                    "path": str(overall_path),
                    "sha256": sha256_file(overall_path),
                    "rows": int(len(overall)),
                },
                "rf_vs_regression_paired_comparison": {
                    "path": str(paired_path),
                    "sha256": sha256_file(paired_path),
                    "rows": int(len(paired)),
                },
                "residual_moran_i": {
                    "path": str(moran_path),
                    "sha256": sha256_file(moran_path),
                    "rows": int(len(moran)),
                },
            },
            "figure_definition": {
                "panels": {
                    "a": "Temporal OOF overall rate RMSE",
                    "b": "Spatial OOF overall rate RMSE",
                    "c": (
                        "Temporal OOF conditional-positive rate RMSE"
                    ),
                    "d": (
                        "Spatial OOF conditional-positive rate RMSE"
                    ),
                    "e": "Temporal OOF residual Moran's I",
                    "f": "Spatial OOF residual Moran's I",
                },
                "error_normalization": (
                    "Metric divided by the Baseline NB1 value within "
                    "the same validation-season-response combination "
                    "and multiplied by 100."
                ),
                "bootstrap_ring": (
                    "Black outer ring marks the Hurdle RF or optimized "
                    "ZINB1/NB1 model favored by the paired grid-cluster "
                    "bootstrap when the 95% CI excludes zero."
                ),
                "moran_diamond": (
                    "Open black diamond marks the model with the lowest "
                    "residual Moran's I within each validation-season-"
                    "response combination."
                ),
                "display_response_labels": ["FC", "BA"],
                "underlying_rate_scales": [
                    "Fire_Count/ForestPixelCount",
                    "Burned_Pixel_Count/ForestPixelCount",
                ],
            },
            "qa": {
                "passed": True,
                "relative_plotting_rows": int(
                    len(relative_plotting)
                ),
                "paired_plotting_rows": int(
                    len(paired_plotting)
                ),
                "moran_plotting_rows": int(
                    len(moran_plotting)
                ),
                "relative_qa_passed": bool(
                    relative_qa[
                        "Relative_Error_QA_Passed"
                    ].all()
                ),
                "paired_qa_passed": bool(
                    paired_qa[
                        "Paired_Row_QA_Passed"
                    ].all()
                ),
                "moran_qa_passed": bool(
                    moran_qa[
                        "Moran_QA_Passed"
                    ].all()
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

        log("Figure creation        : PASSED")
        log(f"PDF                    : {output_paths['pdf']}")
        log(f"TIFF                   : {output_paths['tif']}")
        log(f"PNG                    : {output_paths['png']}")
        log(f"QA                     : {output_paths['qa']}")
        log(f"Manifest               : {output_paths['manifest']}")
        log(f"UTC finish             : {utc_now_iso()}")
        log("Figure 10 completed successfully.")
        return_code = 0

    except Exception as exc:
        log(f"FAILED: {type(exc).__name__}: {exc}")
        log(f"UTC failure            : {utc_now_iso()}")
        return_code = 1

    output_paths["log"].write_text(
        "\n".join(log_lines) + "\n",
        encoding="utf-8",
    )

    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
