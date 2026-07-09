#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Create only Figure 11(d): seasonal total effects on FAI with 95% CI.

Repository location:
<REPOSITORY_ROOT>/11_SEM_mechanism_analysis/00_Script/
06_make_figure11d_total_effects.py

Default output folder:
<REPOSITORY_ROOT>/11_SEM_mechanism_analysis/06_Manuscript_Figure_11D

This script reads locked SEM effect-decomposition outputs only and does not
refit any SEM model.
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


SCRIPT_VERSION = "1.9.0"

DEFAULT_ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = Path("11_SEM_mechanism_analysis")
STEP05_DIR = MODULE_DIR / "05_SEM_Effect_Decomposition_and_Path_Diagrams"
OUTPUT_DIR = MODULE_DIR / "06_Manuscript_Figure_11D"

TOTAL_EFFECT_FILE = "05_Primary_Total_Effects_to_FAI.csv"
SENSITIVITY_FILE = "08_Primary_vs_Equal_FAI_Total_Effect_Comparison.csv"
MC_QA_FILE = "11_Monte_Carlo_QA.csv"

FONT_FAMILY = "Times New Roman"
OUTPUT_DPI = 600
FIG_SIZE = (9.0, 6.5)

SEASONS: Sequence[Tuple[int, str]] = (
    (1, "Spring"),
    (2, "Summer"),
    (3, "Autumn"),
)

PREDICTOR_ORDER: Sequence[str] = (
    "Topography",
    "Meteorology",
    "Vegetation",
    "Anthropogenic",
)

SEASON_MARKERS = {
    "Spring": "o",
    "Summer": "s",
    "Autumn": "^",
}

SEASON_COLORS = {
    "Spring": "#1B9E77",
    "Summer": "#D95F02",
    "Autumn": "#7570B3",
}

TOTAL_REQUIRED = {
    "Season",
    "Season_Label",
    "Predictor",
    "Effect_Type",
    "Estimate",
    "Lower_95",
    "Upper_95",
    "CI_Excludes_Zero",
    "Effect_Scale",
}
SENS_REQUIRED = {
    "Season",
    "Season_Label",
    "Predictor",
    "Sign_Consistent",
    "CI_Exclusion_Class_Consistent",
}
MC_REQUIRED = {
    "Season",
    "Season_Label",
    "Absolute_MC_Mean_Difference",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Figure 11(d): total effects on FAI with 95% CI."
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
            "<root>/11_SEM_mechanism_analysis/06_Manuscript_Figure_11D"
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=OUTPUT_DPI,
        help=f"Raster output DPI. Default: {OUTPUT_DPI}",
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


def parse_bool(series: pd.Series, label: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    mapped = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
                "yes": True,
                "no": False,
            }
        )
    )
    if mapped.isna().any():
        bad = sorted(series.loc[mapped.isna()].astype(str).unique())
        raise ValueError(
            f"Could not parse Boolean values in {label}: {', '.join(bad)}"
        )
    return mapped.astype(bool)


def find_font() -> Tuple[str, bool]:
    available = {font.name for font in font_manager.fontManager.ttflist}
    if FONT_FAMILY in available:
        return FONT_FAMILY, True
    return "DejaVu Serif", False


def validate_inputs(
    total: pd.DataFrame,
    sensitivity: pd.DataFrame,
    mc_qa: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    require_columns(total, TOTAL_REQUIRED, "Total-effect table")
    require_columns(sensitivity, SENS_REQUIRED, "Sensitivity table")
    require_columns(mc_qa, MC_REQUIRED, "Monte Carlo QA table")

    season_codes = {s for s, _ in SEASONS}
    season_labels = {lab for _, lab in SEASONS}

    total = total.loc[
        total["Season"].isin(season_codes)
        & total["Season_Label"].astype(str).isin(season_labels)
        & total["Effect_Type"].astype(str).eq("Total")
        & total["Predictor"].astype(str).isin(PREDICTOR_ORDER)
    ].copy()

    for column in ("Estimate", "Lower_95", "Upper_95"):
        total[column] = pd.to_numeric(total[column], errors="raise")

    total["CI_Excludes_Zero"] = parse_bool(
        total["CI_Excludes_Zero"],
        "CI_Excludes_Zero",
    )

    expected_rows = len(SEASONS) * len(PREDICTOR_ORDER)
    if len(total) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} total-effect rows; found {len(total)}."
        )

    if total.duplicated(["Season", "Predictor"], keep=False).any():
        raise ValueError(
            "Duplicate season-predictor rows found in total-effect table."
        )

    if (total["Lower_95"] > total["Upper_95"]).any():
        raise ValueError(
            "At least one total-effect CI has Lower_95 > Upper_95."
        )

    sensitivity = sensitivity.loc[
        sensitivity["Season"].isin(season_codes)
        & sensitivity["Season_Label"].astype(str).isin(season_labels)
        & sensitivity["Predictor"].astype(str).isin(PREDICTOR_ORDER)
    ].copy()

    sensitivity["Sign_Consistent"] = parse_bool(
        sensitivity["Sign_Consistent"],
        "Sign_Consistent",
    )
    sensitivity["CI_Exclusion_Class_Consistent"] = parse_bool(
        sensitivity["CI_Exclusion_Class_Consistent"],
        "CI_Exclusion_Class_Consistent",
    )

    if len(sensitivity) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} sensitivity rows; "
            f"found {len(sensitivity)}."
        )
    if not sensitivity["Sign_Consistent"].all():
        raise ValueError(
            "Primary vs equal-FAI sign consistency failed."
        )
    if not sensitivity["CI_Exclusion_Class_Consistent"].all():
        raise ValueError(
            "Primary vs equal-FAI CI-class consistency failed."
        )

    mc_qa = mc_qa.loc[
        mc_qa["Season"].isin(season_codes)
        & mc_qa["Season_Label"].astype(str).isin(season_labels)
    ].copy()
    mc_qa["Absolute_MC_Mean_Difference"] = pd.to_numeric(
        mc_qa["Absolute_MC_Mean_Difference"],
        errors="raise",
    )
    max_mc = float(mc_qa["Absolute_MC_Mean_Difference"].max())
    if max_mc > 0.01:
        raise ValueError(
            "Monte Carlo agreement exceeds locked tolerance: "
            f"{max_mc:.6f}"
        )

    predictor_rank = {
        predictor: index
        for index, predictor in enumerate(PREDICTOR_ORDER)
    }
    season_rank = {
        label: index
        for index, (_, label) in enumerate(SEASONS)
    }
    total["Predictor_Rank"] = total["Predictor"].map(predictor_rank)
    total["Season_Rank"] = total["Season_Label"].map(season_rank)
    total = total.sort_values(
        ["Predictor_Rank", "Season_Rank"]
    ).reset_index(drop=True)

    qa = pd.DataFrame(
        [
            {
                "Check": "Total-effect row count",
                "Observed": len(total),
                "Expected": expected_rows,
                "Passed": len(total) == expected_rows,
            },
            {
                "Check": "Sensitivity sign consistency",
                "Observed": int(sensitivity["Sign_Consistent"].sum()),
                "Expected": expected_rows,
                "Passed": bool(sensitivity["Sign_Consistent"].all()),
            },
            {
                "Check": "Sensitivity CI-class consistency",
                "Observed": int(
                    sensitivity["CI_Exclusion_Class_Consistent"].sum()
                ),
                "Expected": expected_rows,
                "Passed": bool(
                    sensitivity["CI_Exclusion_Class_Consistent"].all()
                ),
            },
            {
                "Check": "Monte Carlo mean agreement <= 0.01",
                "Observed": max_mc,
                "Expected": "<=0.01",
                "Passed": max_mc <= 0.01,
            },
        ]
    )

    return total, qa


def make_figure(
    total: pd.DataFrame,
    output_paths: Dict[str, Path],
    dpi: int,
) -> Tuple[str, bool]:
    font_family, font_found = find_font()

    matplotlib.rcParams.update(
        {
            "font.family": font_family,
            "font.size": 12,
            "axes.unicode_minus": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.facecolor": "none",
            "figure.facecolor": "none",
            "savefig.facecolor": "none",
        }
    )

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    base_y = {
        predictor: len(PREDICTOR_ORDER) - 1 - index
        for index, predictor in enumerate(PREDICTOR_ORDER)
    }
    offsets = {
        "Spring": 0.24,
        "Summer": 0.00,
        "Autumn": -0.24,
    }

    for _, row in total.iterrows():
        season = str(row["Season_Label"])
        predictor = str(row["Predictor"])
        estimate = float(row["Estimate"])
        lower = float(row["Lower_95"])
        upper = float(row["Upper_95"])
        excludes_zero = bool(row["CI_Excludes_Zero"])

        y_value = base_y[predictor] + offsets[season]
        color = SEASON_COLORS[season]
        marker = SEASON_MARKERS[season]

        ax.errorbar(
            estimate,
            y_value,
            xerr=np.array(
                [[estimate - lower], [upper - estimate]]
            ),
            fmt=marker,
            markersize=11.0,
            markerfacecolor=(color if excludes_zero else "none"),
            markeredgecolor=color,
            markeredgewidth=1.8,
            ecolor=color,
            elinewidth=2.1,
            capsize=4.2,
            capthick=1.5,
            linestyle="none",
            zorder=3,
        )

    ax.axvline(
        0.0,
        color="#808080",
        linewidth=1.3,
        linestyle=":",
        zorder=1,
    )

    ax.set_yticks(
        [base_y[predictor] for predictor in PREDICTOR_ORDER]
    )
    ax.set_yticklabels(
        PREDICTOR_ORDER,
        fontsize=23,
        color="black",
    )
    ax.tick_params(
        axis="x",
        labelsize=19,
        colors="black",
    )
    ax.tick_params(
        axis="y",
        length=0,
        colors="black",
    )

    x_min = min(-0.32, float(total["Lower_95"].min()) - 0.10)
    x_max = max(0.62, float(total["Upper_95"].max()) + 0.10)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.55, len(PREDICTOR_ORDER) - 0.45)

    ax.set_title(
        "",
        loc="left",
        fontsize=28,
        fontweight="bold",
        color="black",
        pad=2,
    )
    ax.set_xlabel(
        "Total effect on log-expected FAI",
        fontsize=28,
        color="black",
        labelpad=8,
    )

    ax.grid(
        axis="x",
        color="#A0A0A0",
        linewidth=0.8,
        alpha=0.45,
        zorder=0,
    )
    for spine_name in ("top", "right", "left"):
        ax.spines[spine_name].set_visible(False)
    ax.spines["bottom"].set_color("black")
    ax.spines["bottom"].set_linewidth(1.0)

    season_handles = [
        Line2D(
            [0],
            [0],
            marker=SEASON_MARKERS[label],
            linestyle="none",
            markersize=10.2,
            markerfacecolor=SEASON_COLORS[label],
            markeredgecolor=SEASON_COLORS[label],
            markeredgewidth=1.2,
            label=label,
        )
        for _, label in SEASONS
    ]


    fig.legend(
        handles=season_handles,
        loc="lower center",
        bbox_to_anchor=(0.50, 0.03),
        ncol=3,
        frameon=False,
        fontsize=28,
        labelcolor="black",
        handletextpad=0.55,
        columnspacing=1.8,
        borderaxespad=0.0,
    )

    fig.subplots_adjust(
        left=0.18,
        right=0.96,
        top=0.96,
        bottom=0.24,
    )

    fig.savefig(
        output_paths["pdf"],
        bbox_inches="tight",
        transparent=True,
    )
    fig.savefig(
        output_paths["png"],
        dpi=dpi,
        bbox_inches="tight",
        transparent=True,
    )
    try:
        fig.savefig(
            output_paths["tif"],
            dpi=dpi,
            bbox_inches="tight",
            transparent=True,
            pil_kwargs={"compression": "tiff_lzw"},
        )
    except (TypeError, ValueError):
        fig.savefig(
            output_paths["tif"],
            dpi=dpi,
            bbox_inches="tight",
            transparent=True,
        )

    plt.close(fig)
    return font_family, font_found


def main() -> int:
    args = parse_args()

    root = args.root.expanduser().resolve()
    step05 = root / STEP05_DIR
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else root / OUTPUT_DIR
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    input_paths = {
        "total_effects": step05 / TOTAL_EFFECT_FILE,
        "sensitivity": step05 / SENSITIVITY_FILE,
        "mc_qa": step05 / MC_QA_FILE,
    }

    output_paths = {
        "pdf": output_dir / "Figure_11d_Total_Effects_with_95CI.pdf",
        "png": output_dir / "Figure_11d_Total_Effects_with_95CI.png",
        "tif": output_dir / "Figure_11d_Total_Effects_with_95CI.tif",
        "plotting": output_dir / "Figure_11d_Total_Effects_Plotting_Data.csv",
        "qa": output_dir / "Figure_11d_QA.csv",
        "manifest": output_dir / "Figure_11d_Run_Manifest.json",
        "log": output_dir / "Figure_11d_Run_Log.txt",
        "font": output_dir / "Figure_11d_Font_Status.txt",
    }

    log_lines: List[str] = []

    def log(message: str) -> None:
        print(message, flush=True)
        log_lines.append(message)

    log("=" * 82)
    log("Figure 11(d): total effects with 95% CI")
    log("=" * 82)
    log(f"Script version          : {SCRIPT_VERSION}")
    log(f"UTC start               : {utc_now_iso()}")
    log(f"Project root            : {root}")
    log(f"Output directory        : {output_dir}")
    log("No SEM fitting or effect recalculation will be performed.")

    try:
        total_raw = read_csv_strict(input_paths["total_effects"])
        sensitivity_raw = read_csv_strict(input_paths["sensitivity"])
        mc_raw = read_csv_strict(input_paths["mc_qa"])

        total, qa = validate_inputs(
            total_raw,
            sensitivity_raw,
            mc_raw,
        )

        total.to_csv(
            output_paths["plotting"],
            index=False,
            encoding="utf-8-sig",
        )
        qa.to_csv(
            output_paths["qa"],
            index=False,
            encoding="utf-8-sig",
        )

        font_used, font_found = make_figure(
            total,
            output_paths,
            args.dpi,
        )

        output_paths["font"].write_text(
            (
                f"Requested font: {FONT_FAMILY}\n"
                f"Font found: {font_found}\n"
                f"Font used: {font_used}\n"
            ),
            encoding="utf-8",
        )

        manifest = {
            "script_version": SCRIPT_VERSION,
            "created_utc": utc_now_iso(),
            "project_root": str(root),
            "figure": {
                "panel": "Figure 11(d)",
                "title": "Total effects with 95% CI",
                "response": "FAI",
                "predictors": list(PREDICTOR_ORDER),
                "seasons": [label for _, label in SEASONS],
                "transparent_background": True,
                "intended_use": (
                    "Bottom-right panel of the user's PPT SEM layout"
                ),
            },
            "input_files": {
                key: {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
                for key, path in input_paths.items()
            },
            "qa": {
                "all_passed": bool(qa["Passed"].all()),
                "checks": qa.to_dict(orient="records"),
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
            },
        }
        output_paths["manifest"].write_text(
            json.dumps(
                manifest,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        log("Input and QA            : PASSED")
        log(f"Total-effect rows       : {len(total)}")
        log(
            "CI includes zero rows   : "
            f"{int((~total['CI_Excludes_Zero']).sum())}"
        )
        log(f"PNG                     : {output_paths['png']}")
        log(f"TIFF                    : {output_paths['tif']}")
        log(f"PDF                     : {output_paths['pdf']}")
        log(f"QA                      : {output_paths['qa']}")
        log(f"Manifest                : {output_paths['manifest']}")
        log(f"UTC finish              : {utc_now_iso()}")
        log("Figure 11(d) completed successfully.")
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
