#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create manuscript Figure 8 from the locked final NB1/ZINB1 inference outputs.

The script does not fit or retrain any model. It reads the final Step-06d
coefficient and model-structure tables, validates their internal consistency,
and creates a compact two-panel coefficient matrix. Predictor-domain labels
are intentionally omitted from the left margin to save space; the domain
order should be explained in the manuscript figure caption:

(a) Conditional count rate.
(b) Fire realization: the negative of the structural-zero coefficient,
    so positive values consistently indicate greater fire realization.

Filled circles indicate that the two-way GRID_UID-Year cluster 95% confidence
interval excludes zero. Hollow circles indicate that it includes zero.

Repository location:
    <REPOSITORY_ROOT>/08_Regression_modeling/00_Script/
        10_make_figure08_final_regression_effects.py

Default output directory:
    <REPOSITORY_ROOT>/08_Regression_modeling/10_Manuscript_Figure_08
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
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


SCRIPT_VERSION = "1.5.0"
DEFAULT_ROOT = Path(__file__).resolve().parents[2]
INPUT_SUBDIR = Path(
    "08_Regression_modeling"
) / "06d_Final_Stable_Fixed_Effect_Inference"
STRUCTURE_FILENAME = "14_Final_Six_Fixed_Effect_Inference_Structures.csv"
COEFFICIENT_FILENAME = "16_Final_Selected_Coefficients_and_Two_Way_Inference.csv"
DEFAULT_OUTPUT_SUBDIR = Path(
    "08_Regression_modeling"
) / "10_Manuscript_Figure_08"

FONT_FAMILY = "Times New Roman"
COLOR_LIMIT = 2.5
OUTPUT_DPI = 600
FIGURE_WIDTH_IN = 11.2
FIGURE_HEIGHT_IN = 8.6

MODEL_ORDER: Sequence[Tuple[str, str]] = (
    ("Spring", "FCD"),
    ("Spring", "BAD"),
    ("Summer", "FCD"),
    ("Summer", "BAD"),
    ("Autumn", "FCD"),
    ("Autumn", "BAD"),
)

MODEL_LABELS: Dict[Tuple[str, str], str] = {
    ("Spring", "FCD"): "Spring\nFC",
    ("Spring", "BAD"): "Spring\nBA",
    ("Summer", "FCD"): "Summer\nFC",
    ("Summer", "BAD"): "Summer\nBA",
    ("Autumn", "FCD"): "Autumn\nFC",
    ("Autumn", "BAD"): "Autumn\nBA",
}

PREDICTOR_GROUPS: Sequence[Tuple[str, Sequence[str]]] = (
    ("Vegetation", ("BD", "PTC")),
    ("Topography", ("DEM", "Slope", "Aspect")),
    ("Anthropogenic", ("POP", "Dis_Farm", "Road_dens")),
    ("Meteorology", ("Temp", "Pre", "Rhum", "Wind", "SSRD")),
    ("Fire weather", ("FFMC", "DMC", "DC")),
    ("Lightning", ("LtgProxy",)),
    ("Drought", ("SPEI3", "SPEI6", "SPEI12", "SPEI24")),
)

PREDICTOR_LABELS: Dict[str, str] = {
    "BD": "BD",
    "PTC": "PTC",
    "DEM": "DEM",
    "Slope": "Slope",
    "Aspect": "Aspect",
    "POP": "POP",
    "Dis_Farm": "Dis_Farm",
    "Road_dens": "Road_dens",
    "Temp": "Temp",
    "Pre": "Pre",
    "Rhum": "Rhum",
    "Wind": "Wind",
    "SSRD": "SSRD",
    "FFMC": "FFMC",
    "DMC": "DMC",
    "DC": "DC",
    "LtgProxy": "LtgProxy",
    "SPEI3": "SPEI3",
    "SPEI6": "SPEI6",
    "SPEI12": "SPEI12",
    "SPEI24": "SPEI24",
}

REQUIRED_STRUCTURE_COLUMNS = {
    "Season_Code",
    "Season_Label",
    "Count_Response",
    "Rate_Scale_Name",
    "Model_Structure",
    "Selected_Inference_Candidate_ID",
    "Structural_Zero_Predictors",
    "Structural_Zero_Country_Terms",
    "Count_Predictors",
    "Count_Country_Terms",
    "Final_Selected_Fit_Stable_after_Polish",
    "Selection_Status",
}

REQUIRED_COEFFICIENT_COLUMNS = {
    "Season_Code",
    "Season_Label",
    "Count_Response",
    "Rate_Scale_Name",
    "Model_Structure",
    "Candidate_ID",
    "Inference_Covariance_Source",
    "Component",
    "Parameter",
    "Estimate",
    "Two_Way_Cluster_SE",
    "Two_Way_Cluster_p",
    "Two_Way_Cluster_CI_Lower",
    "Two_Way_Cluster_CI_Upper",
}

ENVIRONMENTAL_COMPONENTS = {"Count", "Structural_Zero"}
EXCLUDED_PARAMETERS = {"Intercept", "Country_NK", "Country_Russia", "log_alpha"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create manuscript Figure 8 from the locked final regression "
            "coefficient outputs without refitting any model."
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
            "<root>/08_Regression_modeling/10_Manuscript_Figure_08"
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
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
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
        raise ValueError(
            f"{label} is missing required columns: {', '.join(missing)}"
        )


def parse_semicolon_terms(value: object) -> List[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    return [term.strip() for term in text.split(";") if term.strip()]


def predictor_order() -> List[str]:
    ordered: List[str] = []
    for _, variables in PREDICTOR_GROUPS:
        ordered.extend(variables)
    return ordered


def predictor_domain_map() -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for domain, variables in PREDICTOR_GROUPS:
        for variable in variables:
            mapping[variable] = domain
    return mapping


def model_key_from_row(row: pd.Series) -> Tuple[str, str]:
    return str(row["Season_Label"]), str(row["Rate_Scale_Name"])


def validate_inputs(
    structures: pd.DataFrame,
    coefficients: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str]]:
    require_columns(structures, REQUIRED_STRUCTURE_COLUMNS, "Structure table")
    require_columns(coefficients, REQUIRED_COEFFICIENT_COLUMNS, "Coefficient table")

    errors: List[str] = []
    qa_rows: List[Dict[str, object]] = []

    if len(structures) != 6:
        errors.append(f"Expected 6 structure rows; found {len(structures)}.")

    structure_keys = structures.apply(model_key_from_row, axis=1).tolist()
    if len(set(structure_keys)) != len(structure_keys):
        errors.append("Duplicate season-response keys found in the structure table.")

    expected_keys = set(MODEL_ORDER)
    observed_keys = set(structure_keys)
    if observed_keys != expected_keys:
        errors.append(
            "Structure-table model keys differ from the six expected models. "
            f"Missing={sorted(expected_keys - observed_keys)}; "
            f"Unexpected={sorted(observed_keys - expected_keys)}"
        )

    stable_values = structures["Final_Selected_Fit_Stable_after_Polish"].astype(str)
    if not stable_values.str.lower().isin({"true", "1"}).all():
        errors.append("At least one final structure is not marked stable after polish.")

    relevant = coefficients.loc[
        coefficients["Component"].isin(ENVIRONMENTAL_COMPONENTS)
        & ~coefficients["Parameter"].isin(EXCLUDED_PARAMETERS)
    ].copy()

    duplicate_mask = relevant.duplicated(
        subset=["Season_Label", "Rate_Scale_Name", "Component", "Parameter"],
        keep=False,
    )
    if duplicate_mask.any():
        duplicate_rows = relevant.loc[
            duplicate_mask,
            ["Season_Label", "Rate_Scale_Name", "Component", "Parameter"],
        ]
        errors.append(
            "Duplicate coefficient keys found: "
            + duplicate_rows.astype(str).agg("/".join, axis=1).str.cat(sep=", ")
        )

    numeric_columns = [
        "Estimate",
        "Two_Way_Cluster_SE",
        "Two_Way_Cluster_p",
        "Two_Way_Cluster_CI_Lower",
        "Two_Way_Cluster_CI_Upper",
    ]
    for column in numeric_columns:
        numeric = pd.to_numeric(relevant[column], errors="coerce")
        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            errors.append(f"Non-finite values found in {column}.")

    invalid_ci = (
        pd.to_numeric(relevant["Two_Way_Cluster_CI_Lower"], errors="coerce")
        > pd.to_numeric(relevant["Two_Way_Cluster_CI_Upper"], errors="coerce")
    )
    if invalid_ci.any():
        errors.append("At least one confidence interval has lower > upper.")

    known_predictors = set(predictor_order())
    observed_predictors = set(relevant["Parameter"].astype(str))
    unknown_predictors = sorted(observed_predictors - known_predictors)
    missing_from_coefficients = sorted(known_predictors - observed_predictors)
    if unknown_predictors:
        errors.append(
            "Coefficient table contains unmapped environmental predictors: "
            + ", ".join(unknown_predictors)
        )
    if missing_from_coefficients:
        errors.append(
            "Expected manuscript predictors are absent from all coefficients: "
            + ", ".join(missing_from_coefficients)
        )

    for _, structure in structures.iterrows():
        key = model_key_from_row(structure)
        season, rate_scale = key
        model_rows = relevant.loc[
            (relevant["Season_Label"].astype(str) == season)
            & (relevant["Rate_Scale_Name"].astype(str) == rate_scale)
        ].copy()

        expected_count = set(parse_semicolon_terms(structure["Count_Predictors"]))
        observed_count = set(
            model_rows.loc[model_rows["Component"] == "Count", "Parameter"].astype(str)
        )

        expected_zero = set(
            parse_semicolon_terms(structure["Structural_Zero_Predictors"])
        )
        observed_zero = set(
            model_rows.loc[
                model_rows["Component"] == "Structural_Zero", "Parameter"
            ].astype(str)
        )

        candidate_values = set(model_rows["Candidate_ID"].astype(str))
        structure_values = set(model_rows["Model_Structure"].astype(str))
        candidate_ok = candidate_values == {
            str(structure["Selected_Inference_Candidate_ID"])
        }
        structure_ok = structure_values == {str(structure["Model_Structure"])}
        count_ok = expected_count == observed_count
        zero_ok = expected_zero == observed_zero

        qa_rows.append(
            {
                "Season": season,
                "Rate_Scale": rate_scale,
                "Count_Response": structure["Count_Response"],
                "Model_Structure": structure["Model_Structure"],
                "Selected_Candidate": structure["Selected_Inference_Candidate_ID"],
                "Expected_Count_Predictor_N": len(expected_count),
                "Observed_Count_Predictor_N": len(observed_count),
                "Count_Predictors_Exact_Match": count_ok,
                "Expected_Structural_Zero_Predictor_N": len(expected_zero),
                "Observed_Structural_Zero_Predictor_N": len(observed_zero),
                "Structural_Zero_Predictors_Exact_Match": zero_ok,
                "Candidate_ID_Exact_Match": candidate_ok,
                "Model_Structure_Exact_Match": structure_ok,
                "Count_Missing": ";".join(sorted(expected_count - observed_count)),
                "Count_Unexpected": ";".join(sorted(observed_count - expected_count)),
                "Structural_Zero_Missing": ";".join(
                    sorted(expected_zero - observed_zero)
                ),
                "Structural_Zero_Unexpected": ";".join(
                    sorted(observed_zero - expected_zero)
                ),
                "Model_QA_Passed": bool(
                    count_ok and zero_ok and candidate_ok and structure_ok
                ),
            }
        )

        if not count_ok:
            errors.append(
                f"{season} {rate_scale}: count predictors do not match the "
                "locked structure."
            )
        if not zero_ok:
            errors.append(
                f"{season} {rate_scale}: structural-zero predictors do not "
                "match the locked structure."
            )
        if not candidate_ok:
            errors.append(
                f"{season} {rate_scale}: coefficient Candidate_ID does not "
                "match the selected candidate."
            )
        if not structure_ok:
            errors.append(
                f"{season} {rate_scale}: coefficient Model_Structure does not "
                "match the locked structure."
            )

    qa = pd.DataFrame(qa_rows)
    return qa, errors


def prepare_plotting_data(coefficients: pd.DataFrame) -> pd.DataFrame:
    data = coefficients.loc[
        coefficients["Component"].isin(ENVIRONMENTAL_COMPONENTS)
        & ~coefficients["Parameter"].isin(EXCLUDED_PARAMETERS)
    ].copy()

    numeric_columns = [
        "Estimate",
        "Two_Way_Cluster_SE",
        "Two_Way_Cluster_p",
        "Two_Way_Cluster_CI_Lower",
        "Two_Way_Cluster_CI_Upper",
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="raise")

    data["Plot_Component"] = np.where(
        data["Component"] == "Count",
        "Count process",
        "Fire realization",
    )

    is_zero = data["Component"] == "Structural_Zero"
    data["Plot_Effect"] = data["Estimate"]
    data["Plot_CI_Lower"] = data["Two_Way_Cluster_CI_Lower"]
    data["Plot_CI_Upper"] = data["Two_Way_Cluster_CI_Upper"]

    data.loc[is_zero, "Plot_Effect"] = -data.loc[is_zero, "Estimate"]
    data.loc[is_zero, "Plot_CI_Lower"] = -data.loc[
        is_zero, "Two_Way_Cluster_CI_Upper"
    ]
    data.loc[is_zero, "Plot_CI_Upper"] = -data.loc[
        is_zero, "Two_Way_Cluster_CI_Lower"
    ]

    data["CI_Excludes_Zero"] = (
        (data["Plot_CI_Lower"] > 0) | (data["Plot_CI_Upper"] < 0)
    )
    data["Significance_Display"] = np.where(
        data["CI_Excludes_Zero"],
        "95% CI excludes zero",
        "95% CI includes zero",
    )
    data["Display_Effect_Clipped"] = data["Plot_Effect"].clip(
        lower=-COLOR_LIMIT,
        upper=COLOR_LIMIT,
    )
    data["Effect_Clipped_For_Display"] = (
        data["Plot_Effect"].abs() > COLOR_LIMIT
    )

    model_position = {key: index for index, key in enumerate(MODEL_ORDER)}
    data["Model_Key"] = list(
        zip(data["Season_Label"].astype(str), data["Rate_Scale_Name"].astype(str))
    )
    data["Model_Position"] = data["Model_Key"].map(model_position)
    data["Model_Label"] = data["Model_Key"].map(MODEL_LABELS)

    predictors = predictor_order()
    predictor_position = {variable: index for index, variable in enumerate(predictors)}
    domain_map = predictor_domain_map()
    data["Predictor_Position"] = data["Parameter"].map(predictor_position)
    data["Predictor_Label"] = data["Parameter"].map(PREDICTOR_LABELS)
    data["Predictor_Domain"] = data["Parameter"].map(domain_map)

    if data[
        ["Model_Position", "Predictor_Position", "Predictor_Label", "Predictor_Domain"]
    ].isna().any().any():
        raise ValueError("Failed to map one or more model or predictor labels.")

    return data.sort_values(
        ["Plot_Component", "Predictor_Position", "Model_Position"]
    ).reset_index(drop=True)


def find_font() -> Tuple[str, bool]:
    available = {font.name for font in font_manager.fontManager.ttflist}
    if FONT_FAMILY in available:
        return FONT_FAMILY, True
    return "DejaVu Serif", False


def marker_size(effect: np.ndarray) -> np.ndarray:
    capped = np.minimum(np.abs(effect), COLOR_LIMIT)
    return 24.0 + 105.0 * np.power(capped / COLOR_LIMIT, 0.62)


def add_domain_bands(
    ax: plt.Axes,
    predictors: Sequence[str],
    show_domain_labels: bool,
) -> None:
    cursor = 0
    band_index = 0
    for _, variables in PREDICTOR_GROUPS:
        start = cursor - 0.5
        end = cursor + len(variables) - 0.5
        if band_index % 2 == 0:
            ax.axhspan(start, end, color="0.96", zorder=0)
        if cursor > 0:
            ax.axhline(cursor - 0.5, color="0.78", linewidth=0.8, zorder=1)
        cursor += len(variables)
        band_index += 1


def draw_component(
    ax: plt.Axes,
    data: pd.DataFrame,
    component: str,
    panel_label: str,
    font_family: str,
    cmap: matplotlib.colors.Colormap,
    norm: TwoSlopeNorm,
    show_y_labels: bool,
) -> None:
    subset = data.loc[data["Plot_Component"] == component].copy()

    add_domain_bands(
        ax,
        predictor_order(),
        show_domain_labels=show_y_labels,
    )

    significant = subset.loc[subset["CI_Excludes_Zero"]]
    nonsignificant = subset.loc[~subset["CI_Excludes_Zero"]]

    if not significant.empty:
        ax.scatter(
            significant["Model_Position"],
            significant["Predictor_Position"],
            s=marker_size(significant["Plot_Effect"].to_numpy(dtype=float)),
            c=significant["Display_Effect_Clipped"],
            cmap=cmap,
            norm=norm,
            marker="o",
            edgecolors="black",
            linewidths=0.45,
            zorder=4,
        )

    if not nonsignificant.empty:
        ax.scatter(
            nonsignificant["Model_Position"],
            nonsignificant["Predictor_Position"],
            s=marker_size(nonsignificant["Plot_Effect"].to_numpy(dtype=float)),
            facecolors="none",
            edgecolors="0.55",
            marker="o",
            linewidths=1.15,
            zorder=4,
        )

    ax.set_xlim(-0.5, len(MODEL_ORDER) - 0.5)
    ax.set_ylim(len(predictor_order()) - 0.5, -0.5)
    ax.set_xticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels(
        ["FC", "BA", "FC", "BA", "FC", "BA"],
        fontsize=16.5,
    )
    ax.tick_params(axis="x", length=0, pad=7)
    ax.tick_params(axis="y", length=0, pad=5)

    for center, season in zip((0.5, 2.5, 4.5), ("Spring", "Summer", "Autumn")):
        ax.text(
            center,
            -0.060,
            season,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=16.5,
            clip_on=False,
        )

    ax.set_yticks(range(len(predictor_order())))
    if show_y_labels:
        ax.set_yticklabels(
            [PREDICTOR_LABELS[var] for var in predictor_order()],
            fontsize=16.0,
        )
    else:
        ax.tick_params(axis="y", labelleft=False)

    ax.set_title(panel_label, loc="left", fontsize=18.0, fontweight="bold", pad=12)

    ax.grid(axis="x", color="0.88", linewidth=0.7, zorder=1)
    ax.set_axisbelow(True)

    for spine in ("top", "right", "bottom"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("0.65")
    ax.spines["left"].set_linewidth(0.7)

    if component == "Fire realization":
        summer_ba_x = MODEL_ORDER.index(("Summer", "BAD"))
        ax.text(
            summer_ba_x,
            10.0,
            "N/A\nSingle-stage\nNB1",
            ha="center",
            va="center",
            fontsize=14.0,
            color="0.40",
            linespacing=0.98,
            zorder=5,
        )


def make_figure(plot_data: pd.DataFrame, output_paths: Dict[str, Path], dpi: int) -> None:
    font_family, font_found = find_font()
    matplotlib.rcParams.update(
        {
            "font.family": font_family,
            "font.size": 14,
            "axes.unicode_minus": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    cmap = plt.get_cmap("coolwarm")
    norm = TwoSlopeNorm(vmin=-COLOR_LIMIT, vcenter=0.0, vmax=COLOR_LIMIT)

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN),
        sharey=True,
        gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.08},
    )

    draw_component(
        ax=axes[0],
        data=plot_data,
        component="Count process",
        panel_label="(a) Conditional count rate",
        font_family=font_family,
        cmap=cmap,
        norm=norm,
        show_y_labels=True,
    )
    draw_component(
        ax=axes[1],
        data=plot_data,
        component="Fire realization",
        panel_label="(b) Fire realization",
        font_family=font_family,
        cmap=cmap,
        norm=norm,
        show_y_labels=False,
    )

    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    significance_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="black",
            markerfacecolor="0.45",
            markeredgecolor="black",
            markersize=11.0,
            linewidth=0,
            label="95% CI excludes zero",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="0.55",
            markerfacecolor="none",
            markeredgecolor="0.55",
            markersize=11.0,
            linewidth=0,
            label="95% CI includes zero",
        ),
    ]
    size_effects = [0.25, 0.75, 1.50, 2.50]
    size_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="black",
            markerfacecolor="0.72",
            markeredgecolor="black",
            markeredgewidth=0.45,
            markersize=float(np.sqrt(marker_size(np.array([value]))[0])) * 0.86,
            linewidth=0,
            label=f"|{value:.2f}|",
        )
        for value in size_effects
    ]

    fig.legend(
        handles=significance_handles,
        loc="lower left",
        bbox_to_anchor=(0.03, 0.09),
        ncol=2,
        frameon=False,
        fontsize=16.0,
        handletextpad=0.60,
        columnspacing=1.8,
    )

    fig.text(
        0.79,
        0.14,
        "Absolute effect",
        ha="center",
        va="center",
        fontsize=16.0,
    )
    fig.legend(
        handles=size_handles,
        loc="lower center",
        bbox_to_anchor=(0.77, 0.075),
        ncol=4,
        frameon=False,
        fontsize=16.0,
        handletextpad=0.35,
        columnspacing=1.15,
    )

    colorbar_ax = fig.add_axes([0.18, 0.038, 0.73, 0.030])
    colorbar = fig.colorbar(
        sm,
        cax=colorbar_ax,
        orientation="horizontal",
    )
    colorbar.set_label(
        "Standardized coefficient",
        fontsize=16,
        labelpad=7,
    )
    colorbar.ax.tick_params(labelsize=16.0, length=4)
    fig.subplots_adjust(left=0.085, right=0.99, top=0.95, bottom=0.235)

    fig.savefig(output_paths["pdf"], bbox_inches="tight")
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
        # Compatibility fallback for older Matplotlib/Pillow versions.
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
    input_dir = root / INPUT_SUBDIR
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else root / DEFAULT_OUTPUT_SUBDIR
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    structure_path = input_dir / STRUCTURE_FILENAME
    coefficient_path = input_dir / COEFFICIENT_FILENAME

    output_paths = {
        "pdf": output_dir / "Figure_08_Season_Specific_Regression_Effects.pdf",
        "png": output_dir / "Figure_08_Season_Specific_Regression_Effects.png",
        "tif": output_dir / "Figure_08_Season_Specific_Regression_Effects.tif",
        "plot_data": output_dir / "Figure_08_Plotting_Data.csv",
        "qa": output_dir / "Figure_08_Model_Structure_QA.csv",
        "manifest": output_dir / "Figure_08_Run_Manifest.json",
        "log": output_dir / "Figure_08_Run_Log.txt",
        "font_status": output_dir / "Figure_08_Font_Status.txt",
    }

    log_lines: List[str] = []

    def log(message: str) -> None:
        print(message, flush=True)
        log_lines.append(message)

    log("=" * 78)
    log("Manuscript Figure 8: Final NB1/ZINB1 regression effects")
    log("=" * 78)
    log(f"Script version : {SCRIPT_VERSION}")
    log(f"UTC start      : {utc_now_iso()}")
    log(f"Project root   : {root}")
    log(f"Input folder   : {input_dir}")
    log(f"Output folder  : {output_dir}")
    log("No model will be fitted or retrained.")

    try:
        structures = read_csv_strict(structure_path)
        coefficients = read_csv_strict(coefficient_path)
        log(f"Structure rows : {len(structures):,}")
        log(f"Coefficient rows: {len(coefficients):,}")

        qa, errors = validate_inputs(structures, coefficients)
        qa.to_csv(output_paths["qa"], index=False, encoding="utf-8-sig")

        if errors:
            for error in errors:
                log(f"QA ERROR: {error}")
            raise RuntimeError(
                f"Input QA failed with {len(errors)} error(s). See the run log."
            )

        plot_data = prepare_plotting_data(coefficients)
        plot_data.to_csv(
            output_paths["plot_data"],
            index=False,
            encoding="utf-8-sig",
        )

        count_n = int((plot_data["Plot_Component"] == "Count process").sum())
        realization_n = int(
            (plot_data["Plot_Component"] == "Fire realization").sum()
        )
        significant_n = int(plot_data["CI_Excludes_Zero"].sum())
        clipped_n = int(plot_data["Effect_Clipped_For_Display"].sum())

        log("Input QA       : PASSED")
        log(f"Count effects  : {count_n:,}")
        log(f"Realization effects: {realization_n:,}")
        log(f"CI excludes zero: {significant_n:,} / {len(plot_data):,}")
        log(f"Display-capped effects: {clipped_n:,}")

        make_figure(plot_data, output_paths, dpi=args.dpi)

        manifest = {
            "script_version": SCRIPT_VERSION,
            "created_utc": utc_now_iso(),
            "project_root": str(root),
            "input_files": {
                "structure_table": {
                    "path": str(structure_path),
                    "sha256": sha256_file(structure_path),
                    "rows": int(len(structures)),
                },
                "coefficient_table": {
                    "path": str(coefficient_path),
                    "sha256": sha256_file(coefficient_path),
                    "rows": int(len(coefficients)),
                },
            },
            "figure_definition": {
                "count_process_effect": "Final standardized count coefficient (conditional count rate component)",
                "fire_realization_effect": (
                    "Negative of the final standardized structural-zero coefficient"
                ),
                "confidence_interval": (
                    "Two-way GRID_UID-Year cluster 95% confidence interval"
                ),
                "significance_symbol": (
                    "Filled if transformed confidence interval excludes zero; "
                    "hollow otherwise"
                ),
                "country_terms_displayed": False,
                "intercepts_displayed": False,
                "dispersion_displayed": False,
                "color_and_size_limit": COLOR_LIMIT,
            },
            "qa": {
                "passed": True,
                "models": int(len(qa)),
                "all_model_rows_passed": bool(qa["Model_QA_Passed"].all()),
                "plotting_rows": int(len(plot_data)),
                "count_effect_rows": count_n,
                "fire_realization_effect_rows": realization_n,
                "ci_excludes_zero_rows": significant_n,
                "display_capped_rows": clipped_n,
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
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        log("Figure creation : PASSED")
        log(f"PDF             : {output_paths['pdf']}")
        log(f"TIFF            : {output_paths['tif']}")
        log(f"PNG             : {output_paths['png']}")
        log(f"Plotting data   : {output_paths['plot_data']}")
        log(f"QA table        : {output_paths['qa']}")
        log(f"Manifest        : {output_paths['manifest']}")
        log(f"UTC finish      : {utc_now_iso()}")
        log("Figure 8 completed successfully.")
        return_code = 0

    except Exception as exc:
        log(f"FAILED: {type(exc).__name__}: {exc}")
        log(f"UTC failure     : {utc_now_iso()}")
        return_code = 1

    output_paths["log"].write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
