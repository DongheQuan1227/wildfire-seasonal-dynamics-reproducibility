#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Create Supplementary Figure S7: annual observed and OOF-predicted FC and BA.

Figure layout
-------------
Six rows × two columns.

Rows:
(a) Spring temporal OOF
(b) Spring spatial OOF
(c) Summer temporal OOF
(d) Summer spatial OOF
(e) Autumn temporal OOF
(f) Autumn spatial OOF

Columns:
- FC
- BA

Each panel compares annual observed totals with OOF predictions from:
- Baseline NB1
- Final Regression
- Hurdle Random Forest

No model is fitted, refitted, tuned, or selected.

Formal code-package location
----------------------------
<REPOSITORY_ROOT>\10_Model_comparison\00_Script\
    08_make_figure_s7_annual_oof_predictions.py

Default input
-------------
<REPOSITORY_ROOT>\10_Model_comparison\
    05_Country_Block_Annual_Performance\
    05_Annual_Aggregated_Predictions.csv

Default output directory
------------------------
<REPOSITORY_ROOT>\10_Model_comparison\
    08_Supplementary_Figure_S7_Annual_OOF_Predictions
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
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd


SCRIPT_VERSION = "1.2.0"

DEFAULT_ROOT = Path(__file__).resolve().parents[2]
MODEL_COMPARISON_DIR = Path("10_Model_comparison")

INPUT_SUBDIR = (
    MODEL_COMPARISON_DIR
    / "05_Country_Block_Annual_Performance"
)
INPUT_FILENAME = "05_Annual_Aggregated_Predictions.csv"

DEFAULT_OUTPUT_SUBDIR = (
    MODEL_COMPARISON_DIR
    / "08_Supplementary_Figure_S7_Annual_OOF_Predictions"
)

FONT_FAMILY = "Times New Roman"
OUTPUT_DPI = 600

FIGURE_WIDTH_IN = 13.6
FIGURE_HEIGHT_IN = 18.8

START_YEAR = 2001
END_YEAR = 2025
EXPECTED_YEAR_N = END_YEAR - START_YEAR + 1

SEASON_ORDER: Sequence[str] = (
    "Spring",
    "Summer",
    "Autumn",
)

VALIDATION_ORDER: Sequence[str] = (
    "Temporal",
    "Spatial",
)

RESPONSE_ORDER: Sequence[str] = (
    "FCD",
    "BAD",
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

RESPONSE_LABELS: Dict[str, str] = {
    "FCD": "FC",
    "BAD": "BA",
}

ROW_DEFINITIONS: Sequence[Tuple[str, str]] = (
    ("Spring", "Temporal"),
    ("Spring", "Spatial"),
    ("Summer", "Temporal"),
    ("Summer", "Spatial"),
    ("Autumn", "Temporal"),
    ("Autumn", "Spatial"),
)

ROW_PANEL_LABELS: Sequence[str] = (
    "(a)",
    "(b)",
    "(c)",
    "(d)",
    "(e)",
    "(f)",
)

MODEL_COLORS: Dict[str, str] = {
    "Baseline_NB1": "#7F7F7F",
    "Final_Regression": "#4472C4",
    "Final_Hurdle_RF": "#ED7D31",
}

MODEL_LINESTYLES: Dict[str, object] = {
    "Baseline_NB1": (0, (3, 2)),
    "Final_Regression": (0, (5, 2)),
    "Final_Hurdle_RF": "-",
}

MODEL_MARKERS: Dict[str, str] = {
    "Baseline_NB1": "^",
    "Final_Regression": "s",
    "Final_Hurdle_RF": "o",
}

OBSERVED_COLOR = "black"
OBSERVED_LINEWIDTH = 1.65
OBSERVED_MARKERSIZE = 3.5

MODEL_LINEWIDTH = 1.15
MODEL_MARKERSIZE = 2.7

ROW_TITLE_FONT_SIZE = 22.2
COLUMN_TITLE_FONT_SIZE = 23.7
AXIS_LABEL_FONT_SIZE = 19.2
TICK_FONT_SIZE = 17.1
LEGEND_FONT_SIZE = 18.6

YEAR_TICKS = (2001, 2005, 2010, 2015, 2020, 2025)

REQUIRED_COLUMNS = {
    "Grouping_Level",
    "Validation_Scheme",
    "Year",
    "Season",
    "Season_Label",
    "Count_Response",
    "Rate_Scale_Name",
    "Model",
    "Observed_Total_Count",
    "Predicted_Total_Count",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create Supplementary Figure S7 from sealed annual "
            "observed and OOF-predicted FC/BA totals."
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
            "<root>/10_Model_comparison/"
            "08_Supplementary_Figure_S7_Annual_OOF_Predictions"
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
    names = {font.name for font in font_manager.fontManager.ttflist}
    if FONT_FAMILY in names:
        return FONT_FAMILY, True
    return "DejaVu Serif", False


def validate_input(
    source: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    require_columns(
        source,
        REQUIRED_COLUMNS,
        "Annual aggregated OOF predictions",
    )

    data = source.copy()

    text_columns = (
        "Grouping_Level",
        "Validation_Scheme",
        "Season_Label",
        "Count_Response",
        "Rate_Scale_Name",
        "Model",
    )
    for column in text_columns:
        data[column] = data[column].astype(str).str.strip()

    numeric_columns = (
        "Year",
        "Season",
        "Observed_Total_Count",
        "Predicted_Total_Count",
    )
    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="raise",
        )

    data["Year"] = data["Year"].astype(int)
    data["Season"] = data["Season"].astype(int)

    data = data.loc[
        (data["Grouping_Level"] == "Overall_Annual")
        & data["Validation_Scheme"].isin(VALIDATION_ORDER)
        & data["Season_Label"].isin(SEASON_ORDER)
        & data["Rate_Scale_Name"].isin(RESPONSE_ORDER)
        & data["Model"].isin(MODEL_ORDER)
        & data["Year"].between(START_YEAR, END_YEAR)
    ].copy()

    expected_rows = (
        len(VALIDATION_ORDER)
        * len(SEASON_ORDER)
        * len(RESPONSE_ORDER)
        * len(MODEL_ORDER)
        * EXPECTED_YEAR_N
    )

    if len(data) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} selected annual rows; "
            f"found {len(data)}."
        )

    duplicate_mask = data.duplicated(
        subset=[
            "Validation_Scheme",
            "Year",
            "Season_Label",
            "Rate_Scale_Name",
            "Model",
        ],
        keep=False,
    )
    if duplicate_mask.any():
        examples = data.loc[
            duplicate_mask,
            [
                "Validation_Scheme",
                "Year",
                "Season_Label",
                "Rate_Scale_Name",
                "Model",
            ],
        ].head(20)
        raise ValueError(
            "Duplicate annual model rows were found:\n"
            + examples.to_string(index=False)
        )

    if not np.isfinite(
        data[
            [
                "Observed_Total_Count",
                "Predicted_Total_Count",
            ]
        ].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Observed or predicted annual totals contain non-finite values."
        )

    if (
        data["Observed_Total_Count"] < 0
    ).any() or (
        data["Predicted_Total_Count"] < 0
    ).any():
        raise ValueError(
            "Observed or predicted annual totals contain negative values."
        )

    # Observed annual totals must be identical across models and validation
    # schemes for each year-season-response combination.
    observed_nunique = (
        data.groupby(
            [
                "Year",
                "Season_Label",
                "Rate_Scale_Name",
            ],
            as_index=False,
        )["Observed_Total_Count"]
        .nunique()
    )

    if (observed_nunique["Observed_Total_Count"] != 1).any():
        failed = observed_nunique.loc[
            observed_nunique["Observed_Total_Count"] != 1
        ]
        raise ValueError(
            "Observed annual totals are not identical across models and "
            "validation schemes:\n"
            + failed.head(20).to_string(index=False)
        )

    qa_rows: List[Dict[str, object]] = []

    for validation in VALIDATION_ORDER:
        for season in SEASON_ORDER:
            for response in RESPONSE_ORDER:
                subset = data.loc[
                    (data["Validation_Scheme"] == validation)
                    & (data["Season_Label"] == season)
                    & (data["Rate_Scale_Name"] == response)
                ]

                model_counts = (
                    subset.groupby("Model")["Year"].nunique()
                    .reindex(MODEL_ORDER)
                )

                qa_rows.append(
                    {
                        "Validation_Scheme": validation,
                        "Season": season,
                        "Response": response,
                        "Row_N": int(len(subset)),
                        "Year_N": int(subset["Year"].nunique()),
                        "First_Year": int(subset["Year"].min()),
                        "Last_Year": int(subset["Year"].max()),
                        "Model_N": int(subset["Model"].nunique()),
                        "Baseline_NB1_Year_N": int(
                            model_counts["Baseline_NB1"]
                        ),
                        "Final_Regression_Year_N": int(
                            model_counts["Final_Regression"]
                        ),
                        "Final_Hurdle_RF_Year_N": int(
                            model_counts["Final_Hurdle_RF"]
                        ),
                        "QA_Passed": bool(
                            len(subset)
                            == EXPECTED_YEAR_N * len(MODEL_ORDER)
                            and subset["Year"].nunique()
                            == EXPECTED_YEAR_N
                            and subset["Year"].min()
                            == START_YEAR
                            and subset["Year"].max()
                            == END_YEAR
                            and subset["Model"].nunique()
                            == len(MODEL_ORDER)
                            and (model_counts == EXPECTED_YEAR_N).all()
                        ),
                    }
                )

    qa = pd.DataFrame(qa_rows)

    if not qa["QA_Passed"].all():
        failed = qa.loc[~qa["QA_Passed"]]
        raise RuntimeError(
            "At least one annual season-response-validation panel failed QA:\n"
            + failed.to_string(index=False)
        )

    return (
        data.sort_values(
            [
                "Season_Label",
                "Validation_Scheme",
                "Rate_Scale_Name",
                "Model",
                "Year",
            ]
        ).reset_index(drop=True),
        qa,
    )


def calculate_panel_limits(
    data: pd.DataFrame,
) -> Dict[Tuple[str, str], Tuple[float, float]]:
    """
    Use the same y-axis limits for temporal and spatial validation within
    each season-response pair.
    """
    limits: Dict[Tuple[str, str], Tuple[float, float]] = {}

    for season in SEASON_ORDER:
        for response in RESPONSE_ORDER:
            subset = data.loc[
                (data["Season_Label"] == season)
                & (data["Rate_Scale_Name"] == response)
            ]

            values = np.concatenate(
                [
                    subset["Observed_Total_Count"].to_numpy(dtype=float),
                    subset["Predicted_Total_Count"].to_numpy(dtype=float),
                ]
            )

            data_min = float(np.min(values))
            data_max = float(np.max(values))
            span = max(data_max - data_min, 1.0)

            lower = max(0.0, data_min - span * 0.07)
            upper = data_max + span * 0.09

            limits[(season, response)] = (lower, upper)

    return limits


def count_formatter(value: float, _position: int) -> str:
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        scaled = value / 1_000
        if abs(scaled - round(scaled)) < 0.05:
            return f"{scaled:.0f}k"
        return f"{scaled:.1f}k"
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.0f}"
    return f"{value:.1f}"


def draw_panel(
    axis: plt.Axes,
    data: pd.DataFrame,
    season: str,
    validation: str,
    response: str,
    y_limits: Tuple[float, float],
    show_x_label: bool,
    show_y_label: bool,
) -> None:
    subset = data.loc[
        (data["Season_Label"] == season)
        & (data["Validation_Scheme"] == validation)
        & (data["Rate_Scale_Name"] == response)
    ].copy()

    observed = (
        subset[
            [
                "Year",
                "Observed_Total_Count",
            ]
        ]
        .drop_duplicates()
        .sort_values("Year")
    )

    axis.plot(
        observed["Year"],
        observed["Observed_Total_Count"],
        color=OBSERVED_COLOR,
        linewidth=OBSERVED_LINEWIDTH,
        marker="D",
        markersize=OBSERVED_MARKERSIZE,
        markeredgewidth=0.45,
        label="Observed",
        zorder=5,
    )

    for model in MODEL_ORDER:
        model_data = subset.loc[
            subset["Model"] == model
        ].sort_values("Year")

        axis.plot(
            model_data["Year"],
            model_data["Predicted_Total_Count"],
            color=MODEL_COLORS[model],
            linestyle=MODEL_LINESTYLES[model],
            linewidth=MODEL_LINEWIDTH,
            marker=MODEL_MARKERS[model],
            markersize=MODEL_MARKERSIZE,
            markeredgewidth=0.35,
            label=MODEL_LABELS[model],
            zorder=3,
        )

    axis.set_xlim(START_YEAR - 0.5, END_YEAR + 0.5)
    axis.set_ylim(*y_limits)

    axis.set_xticks(YEAR_TICKS)
    if show_x_label:
        axis.set_xlabel(
            "Year",
            fontsize=AXIS_LABEL_FONT_SIZE,
            labelpad=3,
        )
    else:
        axis.tick_params(axis="x", labelbottom=False)

    if show_y_label:
        axis.set_ylabel(
            "Annual total",
            fontsize=AXIS_LABEL_FONT_SIZE,
            labelpad=4,
        )

    axis.tick_params(
        axis="both",
        labelsize=TICK_FONT_SIZE,
        length=3,
        width=0.7,
    )

    axis.yaxis.set_major_locator(
        MaxNLocator(nbins=5, min_n_ticks=3)
    )
    axis.yaxis.set_major_formatter(
        FuncFormatter(count_formatter)
    )

    axis.grid(
        axis="y",
        color="0.90",
        linewidth=0.55,
        alpha=0.75,
        zorder=0,
    )

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("0.35")
    axis.spines["bottom"].set_color("0.35")
    axis.spines["left"].set_linewidth(0.75)
    axis.spines["bottom"].set_linewidth(0.75)


def add_row_labels(
    axes: np.ndarray,
) -> None:
    for row_index, ((season, validation), panel_label) in enumerate(
        zip(ROW_DEFINITIONS, ROW_PANEL_LABELS)
    ):
        axis = axes[row_index, 0]
        validation_text = (
            "temporal OOF"
            if validation == "Temporal"
            else "spatial OOF"
        )

        axis.text(
            0.00,
            1.055,
            f"{panel_label} {season} {validation_text}",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=ROW_TITLE_FONT_SIZE,
            fontweight="bold",
            clip_on=False,
        )


def make_figure(
    data: pd.DataFrame,
    output_paths: Dict[str, Path],
    dpi: int,
) -> Tuple[str, bool]:
    font_family, font_found = find_font()

    matplotlib.rcParams.update(
        {
            "font.family": font_family,
            "font.size": TICK_FONT_SIZE,
            "axes.unicode_minus": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure, axes = plt.subplots(
        nrows=6,
        ncols=2,
        figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN),
        squeeze=False,
    )

    limits = calculate_panel_limits(data)

    for row_index, (season, validation) in enumerate(ROW_DEFINITIONS):
        for column_index, response in enumerate(RESPONSE_ORDER):
            draw_panel(
                axis=axes[row_index, column_index],
                data=data,
                season=season,
                validation=validation,
                response=response,
                y_limits=limits[(season, response)],
                show_x_label=(row_index == len(ROW_DEFINITIONS) - 1),
                show_y_label=(column_index == 0),
            )

    axes[0, 0].set_title(
        "FC",
        fontsize=COLUMN_TITLE_FONT_SIZE,
        fontweight="bold",
        pad=35,
    )
    axes[0, 1].set_title(
        "BA",
        fontsize=COLUMN_TITLE_FONT_SIZE,
        fontweight="bold",
        pad=35,
    )

    add_row_labels(axes)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=OBSERVED_COLOR,
            linewidth=OBSERVED_LINEWIDTH,
            marker="D",
            markersize=4.2,
            label="Observed",
        )
    ]

    for model in MODEL_ORDER:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=MODEL_COLORS[model],
                linestyle=MODEL_LINESTYLES[model],
                linewidth=MODEL_LINEWIDTH,
                marker=MODEL_MARKERS[model],
                markersize=3.7,
                label=MODEL_LABELS[model],
            )
        )

    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=4,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        handlelength=2.6,
        columnspacing=1.5,
    )

    figure.subplots_adjust(
        left=0.105,
        right=0.980,
        top=0.972,
        bottom=0.090,
        hspace=0.75,
        wspace=0.28,
    )

    figure.savefig(
        output_paths["pdf"],
        bbox_inches="tight",
    )
    figure.savefig(
        output_paths["png"],
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )

    try:
        figure.savefig(
            output_paths["tif"],
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
            pil_kwargs={"compression": "tiff_lzw"},
        )
    except (TypeError, ValueError):
        figure.savefig(
            output_paths["tif"],
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
        )

    plt.close(figure)

    return font_family, font_found


def write_caption(
    path: Path,
) -> None:
    caption = (
        "Fig. S7. Annual observed and out-of-fold (OOF) predicted FC and BA "
        "totals for the Baseline NB1, Final Regression, and Hurdle Random "
        "Forest models under temporal and spatial validation. Rows show "
        "spring (a, b), summer (c, d), and autumn (e, f), with temporal and "
        "spatial OOF results presented separately. FC denotes Fire Count, "
        "whereas BA was modelled as the number of burned 500-m pixels."
    )

    path.write_text(caption + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()

    root = args.root.expanduser().resolve()
    input_path = root / INPUT_SUBDIR / INPUT_FILENAME

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else root / DEFAULT_OUTPUT_SUBDIR
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {
        "pdf": (
            output_dir
            / "Figure_S7_Annual_Observed_and_OOF_Predicted_FC_BA.pdf"
        ),
        "png": (
            output_dir
            / "Figure_S7_Annual_Observed_and_OOF_Predicted_FC_BA.png"
        ),
        "tif": (
            output_dir
            / "Figure_S7_Annual_Observed_and_OOF_Predicted_FC_BA.tif"
        ),
        "plotting_data": (
            output_dir
            / "Figure_S7_Plotting_Data.csv"
        ),
        "qa": (
            output_dir
            / "Figure_S7_QA.csv"
        ),
        "caption": (
            output_dir
            / "Figure_S7_Caption.txt"
        ),
        "font_status": (
            output_dir
            / "Figure_S7_Font_Status.txt"
        ),
        "manifest": (
            output_dir
            / "Figure_S7_Run_Manifest.json"
        ),
        "log": (
            output_dir
            / "Figure_S7_Run_Log.txt"
        ),
    }

    log_lines: List[str] = []

    def log(message: str) -> None:
        print(message, flush=True)
        log_lines.append(message)

    log("=" * 88)
    log("Supplementary Figure S7: annual observed and OOF-predicted FC/BA")
    log("=" * 88)
    log(f"Script version          : {SCRIPT_VERSION}")
    log(f"UTC start               : {utc_now_iso()}")
    log(f"Project root            : {root}")
    log(f"Annual prediction input : {input_path}")
    log(f"Output directory        : {output_dir}")
    log("No model will be fitted, refitted, tuned, or selected.")

    try:
        source = read_csv_strict(input_path)
        data, qa = validate_input(source)

        data.to_csv(
            output_paths["plotting_data"],
            index=False,
            encoding="utf-8-sig",
        )
        qa.to_csv(
            output_paths["qa"],
            index=False,
            encoding="utf-8-sig",
        )

        font_family, font_found = make_figure(
            data=data,
            output_paths=output_paths,
            dpi=args.dpi,
        )

        write_caption(output_paths["caption"])

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
            "input_file": {
                "path": str(input_path),
                "sha256": sha256_file(input_path),
                "rows": int(len(source)),
            },
            "figure_definition": {
                "layout": "6 rows × 2 columns",
                "rows": [
                    {
                        "panel": panel,
                        "season": season,
                        "validation": validation,
                    }
                    for panel, (season, validation) in zip(
                        ROW_PANEL_LABELS,
                        ROW_DEFINITIONS,
                    )
                ],
                "columns": ["FC", "BA"],
                "models": list(MODEL_ORDER),
                "observed_metric": "Observed_Total_Count",
                "predicted_metric": "Predicted_Total_Count",
                "years": [START_YEAR, END_YEAR],
                "same_y_limits_for_temporal_and_spatial_within_season_response": True,
            },
            "qa": {
                "passed": True,
                "selected_row_count": int(len(data)),
                "panel_count": int(len(qa)),
                "all_panel_qa_passed": bool(
                    qa["QA_Passed"].all()
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
                key: str(value)
                for key, value in output_paths.items()
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

        log("Input and panel QA      : PASSED")
        log(f"Selected plotting rows  : {len(data):,}")
        log(f"Panel QA rows           : {len(qa):,}")
        log(f"PDF                     : {output_paths['pdf']}")
        log(f"TIFF                    : {output_paths['tif']}")
        log(f"PNG                     : {output_paths['png']}")
        log(f"UTC finish              : {utc_now_iso()}")
        log("Supplementary Figure S7 completed successfully.")
        return_code = 0

    except Exception as exc:
        log(f"FAILED: {type(exc).__name__}: {exc}")
        log(f"UTC failure             : {utc_now_iso()}")
        return_code = 1

    output_paths["log"].write_text(
        "\n".join(log_lines) + "\n",
        encoding="utf-8",
    )

    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
