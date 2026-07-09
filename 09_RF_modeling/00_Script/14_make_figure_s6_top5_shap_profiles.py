#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Supplementary Figure S6: 6 × 5 matrix of top-ranked OOF SHAP response profiles.

Layout
------
Rows:
1. Spring occurrence
2. Spring positive magnitude
3. Summer occurrence
4. Summer positive magnitude
5. Autumn occurrence
6. Autumn positive magnitude

Columns:
Within each row, the top five curve-eligible predictors ranked by the locked
Step-12 season-stage candidate ranking.

Data policy
-----------
- Reads sealed outputs only.
- Does not retrain any Random Forest model.
- Each panel overlays FC and BA binned OOF SHAP response profiles.

Repository location
-------------------------
<REPOSITORY_ROOT>/09_RF_modeling/00_Script/
    14_make_figure_s6_top5_shap_profiles.py

Default output directory
--------------------------
<REPOSITORY_ROOT>/09_RF_modeling/
    14_Figure_S6_Top5_SHAP_Profiles
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
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


SCRIPT_VERSION = "1.7.0"

DEFAULT_ROOT = Path(__file__).resolve().parents[2]
RF_DIR = Path("09_RF_modeling")

CANDIDATE_SUBDIR = (
    RF_DIR / "12_Figure09_Variable_Importance_and_SHAP_Candidates"
)
SHAP_SOURCE_SUBDIR = RF_DIR / "11_Final_OOF_SHAP_Direction"
DEFAULT_OUTPUT_SUBDIR = (
    RF_DIR / "14_Figure_S6_Top5_SHAP_Profiles"
)

CANDIDATE_FILENAME = "04_Season_Stage_SHAP_Candidate_Ranking.csv"
BINNED_SHAP_FILENAME = "04_Binned_SHAP_Response.csv"

FONT_FAMILY = "Times New Roman"
OUTPUT_DPI = 600

ROW_TITLE_FONT_SIZE = 20.0
AXIS_LABEL_FONT_SIZE = 16.5
Y_LABEL_FONT_SIZE = 16.5
TICK_LABEL_FONT_SIZE = 13.5
LEGEND_FONT_SIZE = 17.0
BASE_FONT_SIZE = 13.0
FIGURE_WIDTH_IN = 11.8
FIGURE_HEIGHT_IN = 14.2

SEASON_ORDER: Sequence[str] = ("Spring", "Summer", "Autumn")
STAGE_ORDER: Sequence[str] = ("Occurrence", "Positive_Severity")
RESPONSE_ORDER: Sequence[str] = ("FCD", "BAD")

ROW_DEFINITIONS = [
    ("Spring", "Occurrence"),
    ("Spring", "Positive_Severity"),
    ("Summer", "Occurrence"),
    ("Summer", "Positive_Severity"),
    ("Autumn", "Occurrence"),
    ("Autumn", "Positive_Severity"),
]

ROW_PANEL_LABELS = {
    1: "(a)",
    2: "(b)",
    3: "(c)",
    4: "(d)",
    5: "(e)",
    6: "(f)",
}

STAGE_DISPLAY: Dict[str, str] = {
    "Occurrence": "occurrence",
    "Positive_Severity": "positive magnitude",
}

RESPONSE_DISPLAY: Dict[str, str] = {
    "FCD": "FC",
    "BAD": "BA",
}

REQUIRED_CANDIDATE_COLUMNS = {
    "Season",
    "Stage",
    "Predictor",
    "Predictor_Group",
    "Curve_Eligible",
    "Season_Stage_Candidate_Rank",
    "Importance_Consensus_Geometric_Mean",
    "Amplitude_Consensus_Geometric_Mean",
    "FC_Bin_N",
    "BA_Bin_N",
}

REQUIRED_BINNED_COLUMNS = {
    "Stage",
    "Season_Label",
    "Response",
    "Predictor",
    "Bin_Order",
    "N",
    "Feature_Median",
    "Mean_SHAP",
    "SHAP_Q25",
    "SHAP_Q75",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create Supplementary Figure S6 as a 6×5 matrix of top-ranked binned OOF SHAP response "
            "profiles from sealed Hurdle RF outputs."
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
            "<root>/09_RF_modeling/14_Figure_S6_Top5_SHAP_Profiles"
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


def parse_boolean_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    normalized = series.astype(str).str.strip().str.lower()
    true_values = {"true", "t", "1", "yes", "y"}
    false_values = {"false", "f", "0", "no", "n", "", "nan", "none"}

    unknown = sorted(
        value
        for value in normalized.dropna().unique()
        if value not in true_values and value not in false_values
    )
    if unknown:
        raise ValueError(
            "Curve_Eligible contains unrecognized Boolean values: "
            + ", ".join(unknown)
        )

    return normalized.isin(true_values)


def find_font() -> Tuple[str, bool]:
    names = {font.name for font in font_manager.fontManager.ttflist}
    if FONT_FAMILY in names:
        return FONT_FAMILY, True
    return "DejaVu Serif", False


def validate_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        candidates,
        REQUIRED_CANDIDATE_COLUMNS,
        "Step-12 SHAP candidate ranking",
    )

    data = candidates.copy()
    data["Season"] = data["Season"].astype(str)
    data["Stage"] = data["Stage"].astype(str)
    data["Predictor"] = data["Predictor"].astype(str)
    data["Curve_Eligible"] = parse_boolean_series(data["Curve_Eligible"])

    numeric_columns = (
        "Season_Stage_Candidate_Rank",
        "Importance_Consensus_Geometric_Mean",
        "Amplitude_Consensus_Geometric_Mean",
        "FC_Bin_N",
        "BA_Bin_N",
    )
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="raise")

    data = data.loc[
        data["Season"].isin(SEASON_ORDER)
        & data["Stage"].isin(STAGE_ORDER)
    ].copy()

    duplicate_mask = data.duplicated(
        subset=["Season", "Stage", "Predictor"],
        keep=False,
    )
    if duplicate_mask.any():
        examples = (
            data.loc[
                duplicate_mask,
                ["Season", "Stage", "Predictor"],
            ]
            .drop_duplicates()
            .head(10)
        )
        raise ValueError(
            "Duplicate season-stage-predictor rows were found in the "
            "candidate ranking:\n"
            + examples.to_string(index=False)
        )

    return data


def validate_binned(binned: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        binned,
        REQUIRED_BINNED_COLUMNS,
        "Binned OOF SHAP response table",
    )

    data = binned.copy()
    for column in ("Stage", "Season_Label", "Response", "Predictor"):
        data[column] = data[column].astype(str)

    numeric_columns = (
        "Bin_Order",
        "N",
        "Feature_Median",
        "Mean_SHAP",
        "SHAP_Q25",
        "SHAP_Q75",
    )
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="raise")

    data = data.loc[
        data["Stage"].isin(STAGE_ORDER)
        & data["Season_Label"].isin(SEASON_ORDER)
        & data["Response"].isin(RESPONSE_ORDER)
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
            "Duplicate binned SHAP rows were found for the same "
            "stage-season-response-predictor-bin combination."
        )

    return data


def select_top5_per_row(candidates: pd.DataFrame) -> pd.DataFrame:
    selected_rows: List[Dict[str, object]] = []

    for row_index, (season, stage) in enumerate(ROW_DEFINITIONS, start=1):
        subset = candidates.loc[
            (candidates["Season"] == season)
            & (candidates["Stage"] == stage)
            & candidates["Curve_Eligible"]
            & (candidates["FC_Bin_N"] >= 5)
            & (candidates["BA_Bin_N"] >= 5)
        ].copy()

        subset = subset.sort_values(
            [
                "Season_Stage_Candidate_Rank",
                "Importance_Consensus_Geometric_Mean",
                "Amplitude_Consensus_Geometric_Mean",
                "Predictor",
            ],
            ascending=[True, False, False, True],
        )

        top5 = subset.head(5).copy()
        if len(top5) < 5:
            raise ValueError(
                f"{season}/{stage} has only {len(top5)} eligible predictors; "
                "at least five are required for the 6×5 supplementary figure."
            )

        for col_index, row in enumerate(top5.itertuples(index=False), start=1):
            selected_rows.append(
                {
                    "Row_Index": row_index,
                    "Column_Index": col_index,
                    "Season": season,
                    "Stage": stage,
                    "Stage_Display": STAGE_DISPLAY[stage],
                    "Predictor": str(row.Predictor),
                    "Predictor_Group": str(row.Predictor_Group),
                    "Season_Stage_Candidate_Rank": int(
                        row.Season_Stage_Candidate_Rank
                    ),
                    "Importance_Consensus_Geometric_Mean": float(
                        row.Importance_Consensus_Geometric_Mean
                    ),
                    "Amplitude_Consensus_Geometric_Mean": float(
                        row.Amplitude_Consensus_Geometric_Mean
                    ),
                    "FC_Bin_N": int(row.FC_Bin_N),
                    "BA_Bin_N": int(row.BA_Bin_N),
                }
            )

    selection = pd.DataFrame(selected_rows)
    if len(selection) != 30:
        raise RuntimeError(
            f"Expected 30 selected panels; obtained {len(selection)}."
        )
    return selection


def prepare_plotting_data(
    selection: pd.DataFrame,
    binned: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    plot_frames: List[pd.DataFrame] = []
    qa_rows: List[Dict[str, object]] = []

    for row in selection.itertuples(index=False):
        subset = binned.loc[
            (binned["Season_Label"] == row.Season)
            & (binned["Stage"] == row.Stage)
            & (binned["Predictor"] == row.Predictor)
            & binned["Response"].isin(RESPONSE_ORDER)
        ].copy()

        panel_passed = True
        details: List[str] = []

        for response in RESPONSE_ORDER:
            response_subset = subset.loc[
                subset["Response"] == response
            ].sort_values("Bin_Order")

            if len(response_subset) < 5:
                panel_passed = False
                details.append(f"{response}: fewer than five bins")

            if response_subset["Bin_Order"].duplicated().any():
                panel_passed = False
                details.append(f"{response}: duplicate bin order")

            if not response_subset["Feature_Median"].is_monotonic_increasing:
                panel_passed = False
                details.append(
                    f"{response}: feature medians are not increasing"
                )

            if (
                response_subset["SHAP_Q25"]
                > response_subset["SHAP_Q75"]
            ).any():
                panel_passed = False
                details.append(f"{response}: Q25 exceeds Q75")

            if not np.isfinite(
                response_subset[
                    [
                        "Feature_Median",
                        "Mean_SHAP",
                        "SHAP_Q25",
                        "SHAP_Q75",
                    ]
                ].to_numpy(dtype=float)
            ).all():
                panel_passed = False
                details.append(f"{response}: non-finite plotting value")

        subset["Row_Index"] = row.Row_Index
        subset["Column_Index"] = row.Column_Index
        subset["Selected_Season"] = row.Season
        subset["Selected_Stage"] = row.Stage
        subset["Selected_Predictor"] = row.Predictor
        subset["Response_Display"] = subset["Response"].map(RESPONSE_DISPLAY)
        plot_frames.append(subset)

        qa_rows.append(
            {
                "Row_Index": row.Row_Index,
                "Column_Index": row.Column_Index,
                "Season": row.Season,
                "Stage": row.Stage,
                "Predictor": row.Predictor,
                "FC_Bin_Count": int(
                    subset.loc[subset["Response"] == "FCD", "Bin_Order"]
                    .nunique()
                ),
                "BA_Bin_Count": int(
                    subset.loc[subset["Response"] == "BAD", "Bin_Order"]
                    .nunique()
                ),
                "Panel_QA_Passed": panel_passed,
                "QA_Details": "; ".join(details),
            }
        )

    plotting = pd.concat(plot_frames, ignore_index=True)
    qa = pd.DataFrame(qa_rows)

    if not qa["Panel_QA_Passed"].all():
        failed = qa.loc[~qa["Panel_QA_Passed"]]
        raise RuntimeError(
            "At least one Supplementary Figure S6 panel failed QA:\n"
            + failed.to_string(index=False)
        )

    return plotting, qa


def panel_title(
    row_number: int,
    col_number: int,
    season: str,
    stage_display: str,
    predictor: str,
) -> str:
    return ""


def draw_panel(
    axis: plt.Axes,
    plotting: pd.DataFrame,
    selection_row,
    show_y_label: bool,
    show_x_label: bool,
) -> List[Line2D]:
    subset = plotting.loc[
        (plotting["Row_Index"] == selection_row.Row_Index)
        & (plotting["Column_Index"] == selection_row.Column_Index)
    ].copy()

    handles: List[Line2D] = []
    styles = (
        ("FCD", "FC", "o", "-"),
        ("BAD", "BA", "s", (0, (4, 2))),
    )

    for response, label, marker, linestyle in styles:
        response_subset = subset.loc[
            subset["Response"] == response
        ].sort_values("Bin_Order")

        x = response_subset["Feature_Median"].to_numpy(dtype=float)
        y = response_subset["Mean_SHAP"].to_numpy(dtype=float)
        lower = response_subset["SHAP_Q25"].to_numpy(dtype=float)
        upper = response_subset["SHAP_Q75"].to_numpy(dtype=float)

        line, = axis.plot(
            x,
            y,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.05,
            markersize=2.9,
            markeredgewidth=0.45,
            solid_capstyle="round",
            dash_capstyle="round",
            label=label,
            zorder=3,
        )
        axis.fill_between(
            x,
            lower,
            upper,
            color=line.get_color(),
            alpha=0.11,
            linewidth=0,
            zorder=1,
        )
        handles.append(line)

    # Keep zero inside the plotting range and use it as the only y-axis tick.
    ymin, ymax = axis.get_ylim()
    span = max(ymax - ymin, 1e-12)
    if ymin > 0:
        ymin = -0.04 * span
    if ymax < 0:
        ymax = 0.04 * span
    axis.set_ylim(ymin, ymax)

    axis.axhline(
        0.0,
        color="0.66",
        linestyle=":",
        linewidth=0.55,
        zorder=2,
    )

    predictor_label = str(selection_row.Predictor)
    unit_lookup = {
        "DEM": "m",
        "BD": "%",
        "PTC": "%",
        "POP": "",
        "Dis_Farm": "km",
        "Dis_Railway": "km",
        "Road_dens": "km km⁻²",
        "Slope": "°",
        "Aspect": "°",
        "Temp": "°C",
        "Pre": "mm",
        "Rhum": "",
        "Wind": "m s⁻¹",
        "SSRD": "J m⁻²",
        "FFMC": "",
        "DMC": "",
        "DC": "",
        "LtgProxy": "",
        "SPEI3": "",
        "SPEI6": "",
        "SPEI12": "",
        "SPEI24": "",
        "EVI": "",
        "ND": "",
        "NE": "",
    }
    unit = unit_lookup.get(predictor_label, "")
    x_label = predictor_label if not unit else f"{predictor_label} ({unit})"

    axis.set_title("")
    axis.set_xlabel(
        x_label,
        fontsize=AXIS_LABEL_FONT_SIZE,
        labelpad=4.0,
    )
    axis.set_ylabel("")

    # Only the zero label is retained on the y-axis.
    axis.set_yticks([0.0])
    axis.set_yticklabels(["0"])
    axis.yaxis.get_offset_text().set_visible(False)

    axis.tick_params(
        axis="x",
        labelsize=TICK_LABEL_FONT_SIZE,
        length=2.6,
        width=0.65,
        pad=2.5,
    )
    axis.tick_params(
        axis="y",
        labelsize=TICK_LABEL_FONT_SIZE,
        length=2.2,
        width=0.65,
        pad=2.0,
    )
    axis.xaxis.get_offset_text().set_fontsize(TICK_LABEL_FONT_SIZE)

    # Keep only very light vertical guides.
    axis.grid(
        axis="x",
        color="0.92",
        linewidth=0.40,
        alpha=0.60,
        zorder=0,
    )

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(0.65)
    axis.spines["bottom"].set_linewidth(0.65)
    axis.spines["left"].set_color("0.30")
    axis.spines["bottom"].set_color("0.30")

    return handles

def add_row_labels(
    axes: np.ndarray,
) -> None:
    """
    Add exactly one panel label to the upper-left of each five-panel row.
    The labels are attached to the first axis in each row so they remain
    correctly positioned after subplot spacing is adjusted.
    """
    row_labels = [
        "Spring occurrence",
        "Spring positive magnitude",
        "Summer occurrence",
        "Summer positive magnitude",
        "Autumn occurrence",
        "Autumn positive magnitude",
    ]

    for row_i, label in enumerate(row_labels):
        axis = axes[row_i, 0]
        row_panel = ROW_PANEL_LABELS[row_i + 1]
        axis.text(
            -0.02,
            1.14,
            f"{row_panel} {label}",
            transform=axis.transAxes,
            va="bottom",
            ha="left",
            fontsize=ROW_TITLE_FONT_SIZE,
            fontweight="bold",
            clip_on=False,
            zorder=100,
        )


def make_figure(
    selection: pd.DataFrame,
    plotting: pd.DataFrame,
    output_paths: Dict[str, Path],
    dpi: int,
) -> Tuple[str, bool]:
    font_family, font_found = find_font()

    matplotlib.rcParams.update(
        {
            "font.family": font_family,
            "font.size": BASE_FONT_SIZE,
            "axes.unicode_minus": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure, axes = plt.subplots(
        nrows=6,
        ncols=5,
        figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN),
        squeeze=False,
    )

    shared_handles: List[Line2D] = []

    selection_sorted = selection.sort_values(
        ["Row_Index", "Column_Index"]
    ).reset_index(drop=True)

    for row_i in range(6):
        for col_i in range(5):
            axis = axes[row_i, col_i]
            selection_row = selection_sorted.loc[
                (selection_sorted["Row_Index"] == row_i + 1)
                & (selection_sorted["Column_Index"] == col_i + 1)
            ].iloc[0]

            handles = draw_panel(
                axis,
                plotting,
                selection_row,
                show_y_label=False,
                show_x_label=True,
            )
            if not shared_handles:
                shared_handles = handles

    figure.legend(
        handles=shared_handles,
        labels=["FC", "BA"],
        loc="lower center",
        bbox_to_anchor=(0.5, 0),
        ncol=2,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        handlelength=2.4,
        columnspacing=1.8,
    )

    figure.subplots_adjust(
        left=0.090,
        right=0.995,
        top=0.970,
        bottom=0.085,
        hspace=1.08,
        wspace=0.30,
    )

    # Add row-level panel labels only after final axis positions are fixed.
    add_row_labels(axes)

    # One shared y-axis title keeps the 6 × 5 matrix readable.
    figure.text(
        0.05,
        0.515,
        "Mean OOF SHAP value",
        rotation=90,
        va="center",
        ha="center",
        fontsize=Y_LABEL_FONT_SIZE,
        fontweight="normal",
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
    caption_path: Path,
) -> None:
    caption = (
        "Fig. S6. Binned out-of-fold SHAP response profiles for the "
        "top five curve-eligible predictors in each season-stage combination "
        "of the Hurdle Random Forest models. Rows correspond to spring, "
        "summer, and autumn, each split into occurrence and positive-magnitude "
        "stages. Within each row, predictors are ordered by the locked "
        "season-stage candidate ranking from Step 12. Lines show mean "
        "out-of-fold SHAP values within feature-value bins, and shaded bands "
        "show the corresponding interquartile range. FC denotes Fire Count, "
        "whereas BA was modelled as the number of burned 500-m pixels. This "
        "supplementary figure uses a dense 6×5 layout while retaining "
        "readable enough for supplementary presentation."
    )
    caption_path.write_text(caption + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()

    root = args.root.expanduser().resolve()
    candidate_dir = root / CANDIDATE_SUBDIR
    shap_source_dir = root / SHAP_SOURCE_SUBDIR

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else root / DEFAULT_OUTPUT_SUBDIR
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_path = candidate_dir / CANDIDATE_FILENAME
    binned_path = shap_source_dir / BINNED_SHAP_FILENAME

    output_paths = {
        "pdf": output_dir / "Trial_Figure_S6_6x5_Top5_SHAP_Profiles.pdf",
        "png": output_dir / "Trial_Figure_S6_6x5_Top5_SHAP_Profiles.png",
        "tif": output_dir / "Trial_Figure_S6_6x5_Top5_SHAP_Profiles.tif",
        "selection": output_dir / "Trial_Figure_S6_Selected_Relationships.csv",
        "plotting": output_dir / "Trial_Figure_S6_Plotting_Data.csv",
        "qa": output_dir / "Trial_Figure_S6_QA.csv",
        "caption": output_dir / "Trial_Figure_S6_Caption.txt",
        "font_status": output_dir / "Trial_Figure_S6_Font_Status.txt",
        "manifest": output_dir / "Trial_Figure_S6_Run_Manifest.json",
        "log": output_dir / "Trial_Figure_S6_Run_Log.txt",
    }

    log_lines: List[str] = []

    def log(message: str) -> None:
        print(message, flush=True)
        log_lines.append(message)

    log("=" * 88)
    log("Supplementary Figure S6: 6×5 top-ranked OOF SHAP profiles")
    log("=" * 88)
    log(f"Script version          : {SCRIPT_VERSION}")
    log(f"UTC start               : {utc_now_iso()}")
    log(f"Project root            : {root}")
    log(f"Candidate ranking       : {candidate_path}")
    log(f"Binned OOF SHAP         : {binned_path}")
    log(f"Output directory        : {output_dir}")
    log("No Random Forest model will be fitted or retrained.")
    log("Style update            : y-axis shows zero only; line=1.05 pt; marker=2.9 pt; slightly stronger styling than v7")

    try:
        candidate_source = read_csv_strict(candidate_path)
        binned_source = read_csv_strict(binned_path)

        candidates = validate_candidates(candidate_source)
        binned = validate_binned(binned_source)

        selection = select_top5_per_row(candidates)
        plotting, qa = prepare_plotting_data(selection, binned)

        selection.to_csv(
            output_paths["selection"],
            index=False,
            encoding="utf-8-sig",
        )
        plotting.to_csv(
            output_paths["plotting"],
            index=False,
            encoding="utf-8-sig",
        )
        qa.to_csv(
            output_paths["qa"],
            index=False,
            encoding="utf-8-sig",
        )

        font_family, font_found = make_figure(
            selection=selection,
            plotting=plotting,
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
            "input_files": {
                "step12_candidate_ranking": {
                    "path": str(candidate_path),
                    "sha256": sha256_file(candidate_path),
                    "rows": int(len(candidate_source)),
                },
                "final_binned_oof_shap": {
                    "path": str(binned_path),
                    "sha256": sha256_file(binned_path),
                    "rows": int(len(binned_source)),
                },
            },
            "selection_definition": {
                "panel_count": 30,
                "layout": "6 rows × 5 columns",
                "row_order": [
                    {
                        "row_index": index + 1,
                        "season": season,
                        "stage": stage,
                    }
                    for index, (season, stage) in enumerate(ROW_DEFINITIONS)
                ],
                "column_definition": (
                    "Within each season-stage row, predictors are ordered by "
                    "the locked Step-12 season-stage candidate ranking."
                ),
                "curve_eligible_required": True,
                "minimum_bins_per_response": 5,
                "selected_relationships": selection.to_dict("records"),
            },
            "qa": {
                "passed": True,
                "selected_panel_count": int(len(selection)),
                "plotting_row_count": int(len(plotting)),
                "all_panel_qa_passed": bool(
                    qa["Panel_QA_Passed"].all()
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

        log("Input and figure QA     : PASSED")
        log("Selected top-5 predictors by row:")
        for season, stage in ROW_DEFINITIONS:
            row_subset = selection.loc[
                (selection["Season"] == season)
                & (selection["Stage"] == stage)
            ].sort_values("Column_Index")
            names = ", ".join(row_subset["Predictor"].tolist())
            log(f"  {season} / {STAGE_DISPLAY[stage]}: {names}")

        log(f"Plotting rows           : {len(plotting):,}")
        log(f"PDF                     : {output_paths['pdf']}")
        log(f"TIFF                    : {output_paths['tif']}")
        log(f"PNG                     : {output_paths['png']}")
        log(f"UTC finish              : {utc_now_iso()}")
        log("Supplementary Figure S6 completed successfully.")
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
