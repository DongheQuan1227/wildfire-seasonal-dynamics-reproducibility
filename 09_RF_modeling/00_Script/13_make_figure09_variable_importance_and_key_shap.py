#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Create manuscript Figure 9 from sealed Hurdle Random Forest OOF SHAP outputs.

The figure contains:
(a) all-variable occurrence-stage OOF SHAP importance;
(b) all-variable positive-magnitude-stage OOF SHAP importance;
(c) spring occurrence response to DEM;
(d) summer positive-magnitude response to EVI;
(e) autumn occurrence response to SPEI3.

No Random Forest model is fitted or retrained.

Formal code-package location:
    <REPOSITORY_ROOT>\09_RF_modeling\00_Script\
        13_make_figure09_variable_importance_and_key_shap.py

Default output directory:
    <REPOSITORY_ROOT>\09_RF_modeling\13_Manuscript_Figure_09
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
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


SCRIPT_VERSION = "1.0.4"

DEFAULT_ROOT = Path(__file__).resolve().parents[2]
RF_DIR = Path("09_RF_modeling")
STEP12_SUBDIR = RF_DIR / "12_Figure09_Variable_Importance_and_SHAP_Candidates"
SHAP_SOURCE_SUBDIR = RF_DIR / "11_Final_OOF_SHAP_Direction"
DEFAULT_OUTPUT_SUBDIR = RF_DIR / "13_Manuscript_Figure_09"

IMPORTANCE_FILENAME = "02_Figure09_Environmental_Variable_Importance.csv"
CANDIDATE_FILENAME = "04_Season_Stage_SHAP_Candidate_Ranking.csv"
BINNED_SHAP_FILENAME = "04_Binned_SHAP_Response.csv"

FONT_FAMILY = "Times New Roman"
OUTPUT_DPI = 600
FIGURE_WIDTH_IN = 13.4
FIGURE_HEIGHT_IN = 10.8

SEASON_ORDER: Sequence[str] = ("Spring", "Summer", "Autumn")
STAGE_ORDER: Sequence[str] = ("Occurrence", "Positive_Severity")
RESPONSE_ORDER: Sequence[str] = ("FCD", "BAD")

MODEL_ORDER: Sequence[Tuple[str, str]] = (
    ("Spring", "FCD"),
    ("Spring", "BAD"),
    ("Summer", "FCD"),
    ("Summer", "BAD"),
    ("Autumn", "FCD"),
    ("Autumn", "BAD"),
)

PREDICTOR_ORDER: Sequence[str] = (
    "BD", "ND", "NE", "EVI", "PTC",
    "DEM", "Slope", "Aspect",
    "POP", "Dis_Farm", "Road_dens", "Dis_Railway",
    "Temp", "Pre", "Rhum", "Wind", "SSRD",
    "LtgProxy",
    "SPEI3", "SPEI12",
)

GROUP_BOUNDARIES: Sequence[int] = (5, 8, 12, 17, 18)

STAGE_LABELS: Dict[str, str] = {
    "Occurrence": "Occurrence stage",
    "Positive_Severity": "Positive-magnitude stage",
}

SELECTED_RELATIONSHIPS: Sequence[Dict[str, str]] = (
    {
        "panel": "(c)",
        "season": "Spring",
        "stage": "Occurrence",
        "predictor": "DEM",
        "xlabel": "DEM (m)",
        "rationale": (
            "Highest FC-BA consensus importance and curve amplitude in "
            "spring occurrence; both responses show a shared low-elevation peak."
        ),
    },
    {
        "panel": "(d)",
        "season": "Summer",
        "stage": "Positive_Severity",
        "predictor": "EVI",
        "xlabel": "EVI",
        "rationale": (
            "Highest FC-BA consensus importance and curve amplitude in "
            "summer positive magnitude; both responses show a strong decline."
        ),
    },
    {
        "panel": "(e)",
        "season": "Autumn",
        "stage": "Occurrence",
        "predictor": "SPEI3",
        "xlabel": "SPEI3",
        "rationale": (
            "Second-ranked autumn occurrence predictor but highest shared "
            "curve amplitude, strong FC-BA agreement, and direct relevance "
            "to seasonal moisture limitation."
        ),
    },
)

REQUIRED_IMPORTANCE_COLUMNS = {
    "Stage", "Season_Label", "Response", "Predictor", "Predictor_Group",
    "Mean_Absolute_SHAP", "Relative_Mean_Absolute_SHAP_Percent",
    "Predictor_Order", "Season_Order", "Stage_Order", "Response_Order",
}

REQUIRED_CANDIDATE_COLUMNS = {
    "Season", "Stage", "Predictor", "Predictor_Group",
    "Importance_Consensus_Geometric_Mean",
    "Amplitude_Consensus_Geometric_Mean",
    "FC_Overall_SHAP_Rank", "BA_Overall_SHAP_Rank",
    "FC_BA_Direction_Agreement", "FC_BA_Bin_Order_Shape_Spearman",
    "FC_Bin_N", "BA_Bin_N", "Curve_Eligible",
    "Season_Stage_Candidate_Rank",
}

REQUIRED_BINNED_COLUMNS = {
    "Stage", "Season_Label", "Response", "Predictor", "Bin_Order", "N",
    "Feature_Median", "Mean_SHAP", "SHAP_Q25", "SHAP_Q75",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create manuscript Figure 9 from the revised Step-12 "
            "all-variable OOF SHAP importance and key nonlinear responses."
        )
    )
    parser.add_argument(
        "--root", type=Path, default=DEFAULT_ROOT,
        help="Repository root. Defaults to the detected repository root.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help=(
            "Optional output directory. Default: "
            "<root>/09_RF_modeling/13_Manuscript_Figure_09"
        ),
    )
    parser.add_argument(
        "--dpi", type=int, default=OUTPUT_DPI,
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


def require_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def find_font() -> Tuple[str, bool]:
    names = {font.name for font in font_manager.fontManager.ttflist}
    if FONT_FAMILY in names:
        return FONT_FAMILY, True
    return "DejaVu Serif", False


def validate_importance(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    require_columns(data, REQUIRED_IMPORTANCE_COLUMNS, "Step-12 variable importance")

    subset = data.loc[
        data["Stage"].astype(str).isin(STAGE_ORDER)
        & data["Season_Label"].astype(str).isin(SEASON_ORDER)
        & data["Response"].astype(str).isin(RESPONSE_ORDER)
        & data["Predictor"].astype(str).isin(PREDICTOR_ORDER)
    ].copy()

    expected_rows = len(STAGE_ORDER) * len(MODEL_ORDER) * len(PREDICTOR_ORDER)
    if len(subset) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} environmental importance rows; found {len(subset)}."
        )

    duplicate_mask = subset.duplicated(
        subset=["Stage", "Season_Label", "Response", "Predictor"],
        keep=False,
    )
    if duplicate_mask.any():
        raise ValueError("Duplicate stage-season-response-predictor importance rows found.")

    subset["Relative_Importance_Percent"] = pd.to_numeric(
        subset["Relative_Mean_Absolute_SHAP_Percent"], errors="raise"
    )
    if ((subset["Relative_Importance_Percent"] < 0)
            | (subset["Relative_Importance_Percent"] > 100.000001)).any():
        raise ValueError("Relative SHAP importance contains values outside [0, 100].")

    predictor_position = {name: index for index, name in enumerate(PREDICTOR_ORDER)}
    model_position = {key: index for index, key in enumerate(MODEL_ORDER)}

    subset["Predictor_Position"] = subset["Predictor"].map(predictor_position)
    subset["Model_Position"] = [
        model_position[(season, response)]
        for season, response in zip(
            subset["Season_Label"].astype(str),
            subset["Response"].astype(str),
        )
    ]

    qa_rows: List[Dict[str, object]] = []
    for stage in STAGE_ORDER:
        for season, response in MODEL_ORDER:
            model = subset.loc[
                (subset["Stage"].astype(str) == stage)
                & (subset["Season_Label"].astype(str) == season)
                & (subset["Response"].astype(str) == response)
            ]
            qa_rows.append({
                "Stage": stage,
                "Season": season,
                "Response": response,
                "Predictor_N": int(len(model)),
                "Unique_Predictor_N": int(model["Predictor"].nunique()),
                "Relative_Maximum": float(model["Relative_Importance_Percent"].max()),
                "QA_Passed": bool(
                    len(model) == len(PREDICTOR_ORDER)
                    and model["Predictor"].nunique() == len(PREDICTOR_ORDER)
                    and np.isclose(model["Relative_Importance_Percent"].max(), 100.0)
                ),
            })

    qa = pd.DataFrame(qa_rows)
    return subset.sort_values(
        ["Stage", "Predictor_Position", "Model_Position"]
    ).reset_index(drop=True), qa


def validate_selected_relationships(
    candidates: pd.DataFrame,
    binned: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    require_columns(candidates, REQUIRED_CANDIDATE_COLUMNS, "Step-12 candidate ranking")
    require_columns(binned, REQUIRED_BINNED_COLUMNS, "Final binned OOF SHAP table")

    numeric_binned = ["Bin_Order", "N", "Feature_Median", "Mean_SHAP", "SHAP_Q25", "SHAP_Q75"]
    for column in numeric_binned:
        binned[column] = pd.to_numeric(binned[column], errors="raise")

    selection_rows: List[Dict[str, object]] = []
    plot_frames: List[pd.DataFrame] = []
    qa_rows: List[Dict[str, object]] = []

    for definition in SELECTED_RELATIONSHIPS:
        season = definition["season"]
        stage = definition["stage"]
        predictor = definition["predictor"]

        candidate = candidates.loc[
            (candidates["Season"].astype(str) == season)
            & (candidates["Stage"].astype(str) == stage)
            & (candidates["Predictor"].astype(str) == predictor)
        ]
        if len(candidate) != 1:
            raise ValueError(
                f"Expected exactly one candidate row for {season}/{stage}/{predictor}; "
                f"found {len(candidate)}."
            )
        candidate = candidate.iloc[0]
        if not bool(candidate["Curve_Eligible"]):
            raise ValueError(f"Selected relationship is not curve-eligible: {season}/{stage}/{predictor}")

        subset = binned.loc[
            (binned["Season_Label"].astype(str) == season)
            & (binned["Stage"].astype(str) == stage)
            & (binned["Predictor"].astype(str) == predictor)
            & binned["Response"].astype(str).isin(RESPONSE_ORDER)
        ].copy()

        response_counts = subset.groupby("Response")["Bin_Order"].nunique()
        panel_passed = True
        details: List[str] = []
        for response in RESPONSE_ORDER:
            response_subset = subset.loc[
                subset["Response"].astype(str) == response
            ].sort_values("Bin_Order")
            if len(response_subset) < 5:
                panel_passed = False
                details.append(f"{response}: fewer than five bins")
            if response_subset["Bin_Order"].duplicated().any():
                panel_passed = False
                details.append(f"{response}: duplicate bins")
            if not response_subset["Feature_Median"].is_monotonic_increasing:
                panel_passed = False
                details.append(f"{response}: non-monotonic feature medians")
            if (response_subset["SHAP_Q25"] > response_subset["SHAP_Q75"]).any():
                panel_passed = False
                details.append(f"{response}: Q25 exceeds Q75")

        subset["Figure_Panel"] = definition["panel"]
        subset["Selected_Season"] = season
        subset["Selected_Stage"] = stage
        subset["Selected_Predictor"] = predictor
        subset["Response_Display"] = subset["Response"].map({"FCD": "FC", "BAD": "BA"})
        plot_frames.append(subset)

        selection_rows.append({
            "Figure_Panel": definition["panel"],
            "Season": season,
            "Stage": stage,
            "Stage_Display": STAGE_LABELS[stage],
            "Predictor": predictor,
            "Predictor_Group": str(candidate["Predictor_Group"]),
            "Season_Stage_Candidate_Rank": int(candidate["Season_Stage_Candidate_Rank"]),
            "FC_Overall_SHAP_Rank": int(candidate["FC_Overall_SHAP_Rank"]),
            "BA_Overall_SHAP_Rank": int(candidate["BA_Overall_SHAP_Rank"]),
            "Importance_Consensus_Geometric_Mean": float(
                candidate["Importance_Consensus_Geometric_Mean"]
            ),
            "Amplitude_Consensus_Geometric_Mean": float(
                candidate["Amplitude_Consensus_Geometric_Mean"]
            ),
            "FC_BA_Direction_Agreement": bool(candidate["FC_BA_Direction_Agreement"]),
            "FC_BA_Bin_Order_Shape_Spearman": float(
                candidate["FC_BA_Bin_Order_Shape_Spearman"]
            ),
            "Scientific_Rationale": definition["rationale"],
        })

        qa_rows.append({
            "Figure_Panel": definition["panel"],
            "Season": season,
            "Stage": stage,
            "Predictor": predictor,
            "FC_Bin_N": int(response_counts.get("FCD", 0)),
            "BA_Bin_N": int(response_counts.get("BAD", 0)),
            "Panel_QA_Passed": panel_passed,
            "QA_Details": "; ".join(details),
        })

    selection = pd.DataFrame(selection_rows)
    plotting = pd.concat(plot_frames, ignore_index=True)
    qa = pd.DataFrame(qa_rows)
    return selection, plotting, qa


def bubble_size(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 0.0, 100.0)
    return 24.0 + 195.0 * np.power(clipped / 100.0, 0.72)


def add_two_tier_model_axis(ax: plt.Axes) -> None:
    ax.set_xticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels(["FC", "BA", "FC", "BA", "FC", "BA"], fontsize=17.7)
    ax.tick_params(axis="x", length=0, pad=4)
    for center, season in zip((0.5, 2.5, 4.5), SEASON_ORDER):
        ax.text(
            center, -0.075, season,
            transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=17.7, clip_on=False,
        )


def draw_importance_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    stage: str,
    panel: str,
    cmap: matplotlib.colors.Colormap,
    norm: Normalize,
    show_y_labels: bool,
) -> None:
    subset = data.loc[data["Stage"].astype(str) == stage]
    ax.scatter(
        subset["Model_Position"],
        subset["Predictor_Position"],
        s=bubble_size(subset["Relative_Importance_Percent"].to_numpy(dtype=float)),
        c=subset["Relative_Importance_Percent"],
        cmap=cmap,
        norm=norm,
        edgecolors="black",
        linewidths=0.35,
        zorder=3,
    )

    ax.set_xlim(-0.5, len(MODEL_ORDER) - 0.5)
    ax.set_ylim(len(PREDICTOR_ORDER) - 0.5, -0.5)
    ax.set_yticks(range(len(PREDICTOR_ORDER)))
    if show_y_labels:
        ax.set_yticklabels(PREDICTOR_ORDER, fontsize=16.5)
    else:
        ax.tick_params(axis="y", labelleft=False)
    ax.tick_params(axis="y", length=0, pad=4)
    add_two_tier_model_axis(ax)

    for boundary in GROUP_BOUNDARIES:
        ax.axhline(boundary - 0.5, color="0.62", linewidth=0.85, zorder=1)
    for row in range(1, len(PREDICTOR_ORDER)):
        if row not in GROUP_BOUNDARIES:
            ax.axhline(row - 0.5, color="0.90", linewidth=0.45, zorder=1)
    for column in range(len(MODEL_ORDER)):
        ax.axvline(column, color="0.92", linewidth=0.45, zorder=1)

    ax.set_title(
        f"{panel} {STAGE_LABELS[stage]} OOF SHAP importance",
        loc="left", fontsize=18.75, fontweight="bold", pad=9,
    )
    for spine in ("top", "right", "bottom"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("0.55")
    ax.spines["left"].set_linewidth(0.7)


def draw_shap_panel(
    ax: plt.Axes,
    plotting: pd.DataFrame,
    definition: Dict[str, str],
    show_y_label: bool,
) -> List[Line2D]:
    subset = plotting.loc[
        (plotting["Selected_Season"].astype(str) == definition["season"])
        & (plotting["Selected_Stage"].astype(str) == definition["stage"])
        & (plotting["Selected_Predictor"].astype(str) == definition["predictor"])
    ]

    handles: List[Line2D] = []
    styles = (
        ("FCD", "FC", "o", "-"),
        ("BAD", "BA", "s", (0, (4, 2))),
    )
    for response, label, marker, linestyle in styles:
        response_subset = subset.loc[
            subset["Response"].astype(str) == response
        ].sort_values("Bin_Order")
        x = response_subset["Feature_Median"].to_numpy(dtype=float)
        y = response_subset["Mean_SHAP"].to_numpy(dtype=float)
        lower = response_subset["SHAP_Q25"].to_numpy(dtype=float)
        upper = response_subset["SHAP_Q75"].to_numpy(dtype=float)
        line, = ax.plot(
            x, y, marker=marker, linestyle=linestyle,
            linewidth=2.0, markersize=5.2, label=label,
        )
        ax.fill_between(x, lower, upper, color=line.get_color(), alpha=0.15, linewidth=0)
        handles.append(line)

    ax.axhline(0.0, color="0.40", linestyle=":", linewidth=0.9)
    stage_short = (
        "occurrence" if definition["stage"] == "Occurrence" else "positive magnitude"
    )
    ax.set_title(
        f"{definition['panel']} {definition['season']} {stage_short}: "
        f"{definition['predictor']}",
        loc="left", fontsize=16.75, fontweight="bold", pad=7,
    )
    ax.set_xlabel(definition["xlabel"], fontsize=17.7)
    if show_y_label:
        ax.set_ylabel("Mean OOF SHAP value", fontsize=17.7)
    ax.tick_params(axis="both", labelsize=15.3)
    ax.grid(color="0.88", linewidth=0.6, alpha=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return handles


def make_figure(
    importance: pd.DataFrame,
    shap_plotting: pd.DataFrame,
    output_paths: Dict[str, Path],
    dpi: int,
) -> None:
    font_family, font_found = find_font()
    matplotlib.rcParams.update({
        "font.family": font_family,
        "font.size": 16.5,
        "axes.unicode_minus": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig = plt.figure(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN), constrained_layout=False)
    grid = fig.add_gridspec(
        nrows=3, ncols=6,
        height_ratios=[1.90, 0.10, 0.82],
        hspace=0.34, wspace=0.34,
    )

    ax_a = fig.add_subplot(grid[0, 0:3])
    ax_b = fig.add_subplot(grid[0, 3:6])
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=0.0, vmax=100.0)
    draw_importance_panel(ax_a, importance, "Occurrence", "(a)", cmap, norm, True)
    draw_importance_panel(ax_b, importance, "Positive_Severity", "(b)", cmap, norm, False)

    shap_axes = (
        fig.add_subplot(grid[2, 0:2]),
        fig.add_subplot(grid[2, 2:4]),
        fig.add_subplot(grid[2, 4:6]),
    )
    shared_handles: List[Line2D] = []
    for index, (axis, definition) in enumerate(zip(shap_axes, SELECTED_RELATIONSHIPS)):
        handles = draw_shap_panel(
            axis, shap_plotting, definition, show_y_label=(index == 0)
        )
        if not shared_handles:
            shared_handles = handles

    colorbar_ax = fig.add_subplot(grid[1, 0:4])
    scalar = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    colorbar = fig.colorbar(scalar, cax=colorbar_ax, orientation="horizontal")
    colorbar.set_label(
        "Relative mean absolute OOF SHAP importance (% of within-model maximum)",
        fontsize=15.9, labelpad=3,
    )
    colorbar.ax.tick_params(labelsize=14.25, length=3)

    size_axis = fig.add_subplot(grid[1, 4:6])
    size_axis.axis("off")
    size_handles = [
        Line2D(
            [0], [0], marker="o", color="none",
            markerfacecolor="0.75", markeredgecolor="black",
            markeredgewidth=0.35,
            markersize=np.sqrt(bubble_size(np.array([value]))[0]) / 1.55,
            label=str(value),
        )
        for value in (25, 50, 75, 100)
    ]
    size_axis.legend(
        handles=size_handles,
        title="Relative importance",
        loc="center",
        ncol=4,
        frameon=False,
        fontsize=17.25,
        title_fontsize=14.7,
        handletextpad=0.2,
        columnspacing=0.7,
    )

    fig.legend(
        handles=shared_handles,
        labels=["FC", "BA"],
        loc="lower center",
        bbox_to_anchor=(0.50, -0.025),
        ncol=2,
        frameon=False,
        fontsize=17.25,
        handlelength=2.4,
        columnspacing=1.8,
    )

    fig.subplots_adjust(left=0.095, right=0.985, top=0.965, bottom=0.075)

    fig.savefig(output_paths["pdf"], bbox_inches="tight")
    fig.savefig(output_paths["png"], dpi=dpi, bbox_inches="tight", facecolor="white")
    try:
        fig.savefig(
            output_paths["tif"], dpi=dpi, bbox_inches="tight", facecolor="white",
            pil_kwargs={"compression": "tiff_lzw"},
        )
    except (TypeError, ValueError):
        fig.savefig(output_paths["tif"], dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    output_paths["font_status"].write_text(
        f"Requested font: {FONT_FAMILY}\nFont found: {font_found}\nFont used: {font_family}\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    step12_dir = root / STEP12_SUBDIR
    shap_source_dir = root / SHAP_SOURCE_SUBDIR
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else root / DEFAULT_OUTPUT_SUBDIR
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    importance_path = step12_dir / IMPORTANCE_FILENAME
    candidate_path = step12_dir / CANDIDATE_FILENAME
    binned_path = shap_source_dir / BINNED_SHAP_FILENAME

    output_paths = {
        "pdf": output_dir / "Figure_09_RF_Variable_Importance_and_Key_SHAP.pdf",
        "png": output_dir / "Figure_09_RF_Variable_Importance_and_Key_SHAP.png",
        "tif": output_dir / "Figure_09_RF_Variable_Importance_and_Key_SHAP.tif",
        "importance_data": output_dir / "Figure_09_Variable_Importance_Plotting_Data.csv",
        "selection": output_dir / "Figure_09_Selected_Nonlinear_Relationships.csv",
        "shap_data": output_dir / "Figure_09_Selected_SHAP_Plotting_Data.csv",
        "importance_qa": output_dir / "Figure_09_Variable_Importance_QA.csv",
        "shap_qa": output_dir / "Figure_09_Selected_SHAP_QA.csv",
        "manifest": output_dir / "Figure_09_Run_Manifest.json",
        "log": output_dir / "Figure_09_Run_Log.txt",
        "font_status": output_dir / "Figure_09_Font_Status.txt",
    }

    log_lines: List[str] = []
    def log(message: str) -> None:
        print(message, flush=True)
        log_lines.append(message)

    log("=" * 84)
    log("Manuscript Figure 9: RF variable importance and key nonlinear responses")
    log("=" * 84)
    log(f"Script version          : {SCRIPT_VERSION}")
    log(f"UTC start               : {utc_now_iso()}")
    log(f"Project root            : {root}")
    log(f"Variable importance     : {importance_path}")
    log(f"Candidate ranking       : {candidate_path}")
    log(f"Binned OOF SHAP         : {binned_path}")
    log(f"Output folder           : {output_dir}")
    log("No Random Forest model will be fitted or retrained.")

    try:
        importance_source = read_csv_strict(importance_path)
        candidates = read_csv_strict(candidate_path)
        binned = read_csv_strict(binned_path)

        importance, importance_qa = validate_importance(importance_source)
        selection, shap_plotting, shap_qa = validate_selected_relationships(
            candidates, binned
        )

        if not importance_qa["QA_Passed"].all():
            raise RuntimeError("At least one variable-importance QA row failed.")
        if not shap_qa["Panel_QA_Passed"].all():
            raise RuntimeError("At least one selected SHAP panel QA row failed.")

        importance.to_csv(output_paths["importance_data"], index=False, encoding="utf-8-sig")
        selection.to_csv(output_paths["selection"], index=False, encoding="utf-8-sig")
        shap_plotting.to_csv(output_paths["shap_data"], index=False, encoding="utf-8-sig")
        importance_qa.to_csv(output_paths["importance_qa"], index=False, encoding="utf-8-sig")
        shap_qa.to_csv(output_paths["shap_qa"], index=False, encoding="utf-8-sig")

        make_figure(importance, shap_plotting, output_paths, args.dpi)

        manifest = {
            "script_version": SCRIPT_VERSION,
            "created_utc": utc_now_iso(),
            "project_root": str(root),
            "input_files": {
                "step12_environmental_variable_importance": {
                    "path": str(importance_path),
                    "sha256": sha256_file(importance_path),
                    "rows": int(len(importance_source)),
                },
                "step12_candidate_ranking": {
                    "path": str(candidate_path),
                    "sha256": sha256_file(candidate_path),
                    "rows": int(len(candidates)),
                },
                "final_binned_oof_shap": {
                    "path": str(binned_path),
                    "sha256": sha256_file(binned_path),
                    "rows": int(len(binned)),
                },
            },
            "figure_definition": {
                "importance_metric": "Mean absolute OOF SHAP",
                "importance_normalization": (
                    "Within each stage-season-response model, values are expressed "
                    "as percent of the maximum environmental-variable importance."
                ),
                "environmental_predictor_count": len(PREDICTOR_ORDER),
                "country_context_displayed": False,
                "selected_nonlinear_relationships": selection.to_dict("records"),
                "shap_curve": "Mean OOF SHAP by feature-value bin",
                "shap_band": "Within-bin SHAP interquartile range",
            },
            "qa": {
                "passed": True,
                "importance_plot_rows": int(len(importance)),
                "selected_relationship_rows": int(len(selection)),
                "selected_shap_plot_rows": int(len(shap_plotting)),
                "importance_qa_passed": bool(importance_qa["QA_Passed"].all()),
                "shap_qa_passed": bool(shap_qa["Panel_QA_Passed"].all()),
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
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        log("Input and figure QA    : PASSED")
        log(f"Importance plot rows    : {len(importance):,}")
        log(f"Selected SHAP rows      : {len(shap_plotting):,}")
        log("Selected nonlinear relationships:")
        for row in selection.itertuples():
            log(
                f"  {row.Figure_Panel} {row.Season} / {row.Stage_Display}: "
                f"{row.Predictor} | stage rank={row.Season_Stage_Candidate_Rank} | "
                f"FC rank={row.FC_Overall_SHAP_Rank} | BA rank={row.BA_Overall_SHAP_Rank}"
            )
        log("Figure creation        : PASSED")
        log(f"PDF                    : {output_paths['pdf']}")
        log(f"TIFF                   : {output_paths['tif']}")
        log(f"PNG                    : {output_paths['png']}")
        log(f"UTC finish             : {utc_now_iso()}")
        log("Figure 9 completed successfully.")
        return_code = 0

    except Exception as exc:
        log(f"FAILED: {type(exc).__name__}: {exc}")
        log(f"UTC failure            : {utc_now_iso()}")
        return_code = 1

    output_paths["log"].write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
