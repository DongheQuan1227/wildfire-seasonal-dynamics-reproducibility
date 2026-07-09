# -*- coding: utf-8 -*-
"""
Step 09: forest-area threshold sensitivity analysis for the locked final
NB1/ZINB1 regression models.

Recommended public location
---------------------------
<REPOSITORY_ROOT>/08_Regression_modeling/00_Script/
    09_run_forest_threshold_sensitivity.py

Purpose
-------
The main analysis uses the strict rule:

    ForestArea_km2 > 2.5

Because one 500 m x 500 m forest pixel represents 0.25 km2, this is
equivalent to:

    ForestPixelCount > 10

Thus, a grid with exactly 10 forest pixels (2.5 km2) is excluded and at
least 11 pixels are required.

This sensitivity analysis applies the same strict operator to three
prespecified thresholds:

    ForestArea_km2 > 1.0
    ForestArea_km2 > 2.5  [main analysis]
    ForestArea_km2 > 5.0

For each threshold, the script:

1. quantifies retained rows, grids, positive observations, Fire Count, and
   Burned Pixel Count;
2. refits the six locked final regression structures without selecting a
   new family, predictor set, or transformation;
3. compares full-data maximum-likelihood coefficient directions and
   magnitudes with the main 2.5-km2 threshold;
4. compares fitted predictions on a common support set
   (ForestArea_km2 > 5.0) so all three threshold models are evaluated on
   exactly the same rows;
5. creates supplementary coefficient-stability and retention figures.

Important inferential limit
---------------------------
This is a preprocessing-sensitivity analysis. New two-way cluster-robust
standard errors are not calculated here. The formal inferential results
remain those from Step 06d at the main 2.5-km2 threshold. Threshold
robustness is evaluated using sample/event retention, coefficient direction
and magnitude, and common-support fitted-prediction agreement.

The common-support diagnostics are full-data fitted diagnostics, not
out-of-fold validation. No Random Forest model and no SEM model is trained.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import platform
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
import statsmodels
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =============================================================================
# Paths
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REGRESSION_ROOT = SCRIPT_DIR.parent
CODE_ROOT = REGRESSION_ROOT.parent

INPUT_CSV = CODE_ROOT / "00_Input_data" / "Base_Table_2001_2025.csv.bz2"

STEP02_SCRIPT = SCRIPT_DIR / "02_compare_single_stage_count_families.py"
STEP04_SCRIPT = SCRIPT_DIR / "04_compare_nb1_hurdle_zinb.py"
STEP05_SCRIPT = SCRIPT_DIR / "05_refine_component_specific_predictors.py"
STEP06C_SCRIPT = (
    SCRIPT_DIR
    / "06c_audit_parameter_identifiability_and_stabilize_inference.py"
)
STEP07_SCRIPT = SCRIPT_DIR / "07_run_regression_spatial_block_validation.py"

FINAL_REGRESSION_ROOT = (
    REGRESSION_ROOT
    / "06d_Final_Stable_Fixed_Effect_Inference"
)
FINAL_STRUCTURE_CSV = (
    FINAL_REGRESSION_ROOT
    / "14_Final_Six_Fixed_Effect_Inference_Structures.csv"
)
FINAL_COEFFICIENT_CSV = (
    FINAL_REGRESSION_ROOT
    / "16_Final_Selected_Coefficients_and_Two_Way_Inference.csv"
)

OUTPUT_ROOT = REGRESSION_ROOT / "09_Forest_Threshold_Sensitivity"
CHECKPOINT_ROOT = OUTPUT_ROOT / "Checkpoints"
FIGURE_ROOT = OUTPUT_ROOT / "Figures"

LOG_FILE = OUTPUT_ROOT / "forest_threshold_sensitivity.log"
METHOD_FILE = OUTPUT_ROOT / "00_Method_Definition.json"
MAIN_THRESHOLD_AUDIT_FILE = OUTPUT_ROOT / "01_Main_Threshold_Audit.csv"
PARAMETER_ORDER_AUDIT_FILE = (
    OUTPUT_ROOT / "01A_ZINB1_Parameter_Order_Audit.csv"
)
LOCKED_DEFINITION_FILE = OUTPUT_ROOT / "02_Locked_Final_Model_Definitions.csv"
RETENTION_FILE = OUTPUT_ROOT / "03_Threshold_Sample_and_Event_Retention.csv"
MODEL_SAMPLE_FILE = OUTPUT_ROOT / "04_Model_Sample_QA.csv"
FIT_STATUS_FILE = OUTPUT_ROOT / "05_Threshold_Model_Fit_Status.csv"
COEFFICIENT_FILE = OUTPUT_ROOT / "06_Threshold_Coefficients.csv"
COEFFICIENT_STABILITY_FILE = (
    OUTPUT_ROOT / "07_Coefficient_Stability_vs_Main_Threshold.csv"
)
COMMON_PREDICTION_FILE = (
    OUTPUT_ROOT / "08_Common_Support_Fitted_Predictions.csv.gz"
)
COMMON_METRIC_FILE = (
    OUTPUT_ROOT / "09_Common_Support_Fitted_Metrics.csv"
)
PREDICTION_AGREEMENT_FILE = (
    OUTPUT_ROOT / "10_Common_Support_Prediction_Agreement.csv"
)
MAIN_REPRODUCTION_FILE = (
    OUTPUT_ROOT / "11_Main_Threshold_Coefficient_Reproduction.csv"
)
STABILITY_SUMMARY_FILE = (
    OUTPUT_ROOT / "12_Threshold_Stability_Summary.csv"
)
OUTPUT_MANIFEST_FILE = OUTPUT_ROOT / "13_Output_Manifest.csv"
SOFTWARE_FILE = OUTPUT_ROOT / "Software_Environment.json"


# =============================================================================
# Locked design
# =============================================================================

CODE_VERSION = (
    "2026-07-01_FOREST_THRESHOLD_SENSITIVITY_V1_"
    "LOCKED_FINAL_NB1_ZINB1_POINT_ESTIMATES_AND_COMMON_SUPPORT"
)

RELEASE_NOTE = (
    "2026-07-08_RELEASE_CORRECT_ZINB1_PARAMETER_LABEL_ORDER"
)

FOREST_PIXEL_AREA_KM2 = 0.25
THRESHOLDS_KM2 = (1.0, 2.5, 5.0)
MAIN_THRESHOLD_KM2 = 2.5
COMMON_SUPPORT_THRESHOLD_KM2 = 5.0
STRICT_OPERATOR = ">"

SEASONS: Dict[int, str] = {
    1: "Spring",
    2: "Summer",
    3: "Autumn",
}

COUNT_RESPONSES = (
    "Fire_Count",
    "Burned_Pixel_Count",
)

RATE_NAMES = {
    "Fire_Count": "FCD",
    "Burned_Pixel_Count": "BAD",
}

COUNTRY_LEVELS = ("China", "NK", "Russia")
COUNTRY_DUMMY_COLUMNS = (
    "Country_China",
    "Country_NK",
    "Country_Russia",
)

EXPECTED_MAIN_ROWS_PER_MODEL = 123_729
EXPECTED_MAIN_GRID_COUNT = 4_982
EXPECTED_MODEL_COMBINATIONS = 6
EXPECTED_FIT_COUNT = (
    len(THRESHOLDS_KM2) * EXPECTED_MODEL_COMBINATIONS
)

RELATIVE_GRADIENT_MAX = 5e-5
COEFFICIENT_REFERENCE_FLOOR = 0.05

FORCE_RESTART = (
    str(
        __import__("os").environ.get(
            "FOREST_THRESHOLD_FORCE_RESTART",
            "0",
        )
    )
    == "1"
)


# =============================================================================
# Generic helpers
# =============================================================================

def log(message: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}"
    print(line, flush=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def require_files(paths: Iterable[Path]) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required files are missing:\n"
            + "\n".join(str(path) for path in missing)
        )


def load_module(path: Path, module_name: str):
    specification = importlib.util.spec_from_file_location(
        module_name,
        path,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Could not import {path.name}.")

    module = importlib.util.module_from_spec(specification)
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = module

    try:
        specification.loader.exec_module(module)
    except Exception:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
        raise

    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_list(value: object) -> List[str]:
    if pd.isna(value) or not str(value).strip():
        return []
    return list(
        dict.fromkeys(
            item.strip()
            for item in str(value).split(";")
            if item.strip()
        )
    )


def threshold_tag(threshold: float) -> str:
    return str(threshold).replace(".", "p")


def safe_filename(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def row_key_sha256(data: pd.DataFrame) -> str:
    canonical = (
        data[["GRID_UID", "Year", "Season"]]
        .assign(
            GRID_UID=lambda frame: frame["GRID_UID"].astype(str),
            Year=lambda frame: frame["Year"].astype(int),
            Season=lambda frame: frame["Season"].astype(int),
        )
        .sort_values(["GRID_UID", "Year", "Season"])
        .to_csv(
            index=False,
            header=False,
            lineterminator="\n",
        )
        .encode("utf-8")
    )
    return hashlib.sha256(canonical).hexdigest()


def model_signature(
    definition: Mapping[str, object],
    threshold: float,
    sample: pd.DataFrame,
) -> str:
    fields = [
        "Season_Code",
        "Count_Response",
        "Model_Structure",
        "Candidate_ID",
        "Structural_Zero_Predictors",
        "Structural_Zero_Country_Terms",
        "Count_Predictors",
        "Count_Country_Terms",
        "Transformation_Scheme",
    ]
    payload = {
        field: str(definition.get(field, ""))
        for field in fields
    }
    payload.update(
        {
            "Code_Version": CODE_VERSION,
            "Forest_Threshold_km2": threshold,
            "Strict_Operator": STRICT_OPERATOR,
            "Input_SHA256": sha256_file(INPUT_CSV),
            "Sample_Row_Key_SHA256": row_key_sha256(sample),
        }
    )
    text = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=True,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def valid_fit_status(status: Mapping[str, object]) -> bool:
    try:
        relative_gradient = float(
            status.get("Relative_Gradient", np.nan)
        )
        objective = float(
            status.get("Negative_LogLikelihood", np.nan)
        )
    except (TypeError, ValueError):
        return False

    return bool(
        status.get("Converged", False)
        and np.isfinite(relative_gradient)
        and relative_gradient <= RELATIVE_GRADIENT_MAX
        and not status.get("Parameter_Bound_Hit", True)
        and np.isfinite(objective)
    )


def safe_spearman(x: Sequence[float], y: Sequence[float]) -> float:
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    valid = np.isfinite(x_array) & np.isfinite(y_array)

    if valid.sum() < 3:
        return np.nan

    x_array = x_array[valid]
    y_array = y_array[valid]

    if (
        np.allclose(x_array, x_array[0])
        or np.allclose(y_array, y_array[0])
    ):
        return np.nan

    return float(stats.spearmanr(x_array, y_array)[0])


def safe_pearson(x: Sequence[float], y: Sequence[float]) -> float:
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    valid = np.isfinite(x_array) & np.isfinite(y_array)

    if valid.sum() < 3:
        return np.nan

    x_array = x_array[valid]
    y_array = y_array[valid]

    if (
        np.allclose(x_array, x_array[0])
        or np.allclose(y_array, y_array[0])
    ):
        return np.nan

    return float(np.corrcoef(x_array, y_array)[0, 1])


def safe_r2(observed: Sequence[float], predicted: Sequence[float]) -> float:
    observed_array = np.asarray(observed, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)

    if (
        len(observed_array) < 2
        or np.allclose(observed_array, observed_array[0])
    ):
        return np.nan

    return float(r2_score(observed_array, predicted_array))


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )


# =============================================================================
# Main-threshold audit
# =============================================================================

def audit_main_threshold() -> pd.DataFrame:
    source = STEP07_SCRIPT.read_text(
        encoding="utf-8",
        errors="replace",
    )

    constant_match = re.search(
        r"FOREST_AREA_THRESHOLD_KM2\s*=\s*([0-9.]+)",
        source,
    )
    strict_filter_present = bool(
        re.search(
            r"forest_area\s*>\s*FOREST_AREA_THRESHOLD_KM2",
            source,
        )
    )

    if constant_match is None:
        raise RuntimeError(
            "Could not locate FOREST_AREA_THRESHOLD_KM2 in Step 07."
        )

    observed_threshold = float(constant_match.group(1))

    checks = [
        {
            "Check": "Main_regression_threshold_constant",
            "Observed": observed_threshold,
            "Expected": MAIN_THRESHOLD_KM2,
            "Passed": observed_threshold == MAIN_THRESHOLD_KM2,
        },
        {
            "Check": "Main_regression_strict_operator",
            "Observed": strict_filter_present,
            "Expected": True,
            "Passed": strict_filter_present,
        },
        {
            "Check": "Main_rule",
            "Observed": "ForestArea_km2 > 2.5",
            "Expected": "ForestArea_km2 > 2.5",
            "Passed": (
                observed_threshold == MAIN_THRESHOLD_KM2
                and strict_filter_present
            ),
        },
        {
            "Check": "Equivalent_pixel_rule",
            "Observed": "ForestPixelCount > 10",
            "Expected": "ForestPixelCount > 10",
            "Passed": True,
        },
        {
            "Check": "Minimum_retained_integer_pixel_count",
            "Observed": 11,
            "Expected": 11,
            "Passed": True,
        },
    ]

    result = pd.DataFrame(checks)

    if not result["Passed"].all():
        raise RuntimeError(
            "The current Step 07 threshold does not match the locked "
            "main-analysis rule."
        )

    return result


# =============================================================================
# Locked model definitions and data
# =============================================================================

def final_definitions(step07: object) -> pd.DataFrame:
    final_structure = step07.read_final_structure()
    definitions = step07.build_model_definitions(final_structure)
    definitions = definitions.loc[
        definitions["Model_Role"] == "Final_Regression"
    ].copy()

    if len(definitions) != EXPECTED_MODEL_COMBINATIONS:
        raise RuntimeError(
            "Exactly six final-regression definitions were expected."
        )

    return definitions.sort_values(
        ["Season_Code", "Count_Response"]
    ).reset_index(drop=True)


def read_base_data(definitions: pd.DataFrame) -> pd.DataFrame:
    predictor_columns = sorted(
        {
            predictor
            for column in (
                "Structural_Zero_Predictors",
                "Count_Predictors",
            )
            for value in definitions[column]
            for predictor in parse_list(value)
        }
    )

    required_raw = {
        "GRID_UID",
        "GRID_ID",
        "Country",
        "Year",
        "Season",
        "Fire_Count",
        "Burned_Pixel_Count",
        "ForestPixelCount",
        "ForestArea_km2",
        *predictor_columns,
    }

    header = pd.read_csv(INPUT_CSV, nrows=0)
    missing = sorted(required_raw - set(header.columns))
    if missing:
        raise ValueError(
            "Base_Table_2001_2025.csv.bz2 is missing required columns:\n"
            + "\n".join(missing)
        )

    data = pd.read_csv(
        INPUT_CSV,
        usecols=sorted(required_raw),
        low_memory=False,
    )

    data["GRID_UID"] = data["GRID_UID"].astype(str)

    if data.duplicated(
        ["GRID_UID", "Year", "Season"],
        keep=False,
    ).any():
        raise ValueError(
            "Duplicate GRID_UID-Year-Season records were found."
        )

    data["ForestPixelCount"] = pd.to_numeric(
        data["ForestPixelCount"],
        errors="coerce",
    )
    data["ForestArea_km2"] = pd.to_numeric(
        data["ForestArea_km2"],
        errors="coerce",
    )

    if (
        data["ForestPixelCount"].isna().any()
        or data["ForestArea_km2"].isna().any()
    ):
        raise ValueError(
            "ForestPixelCount or ForestArea_km2 contains nonnumeric values."
        )

    area_error = np.abs(
        data["ForestArea_km2"]
        - data["ForestPixelCount"] * FOREST_PIXEL_AREA_KM2
    )

    if float(area_error.max()) > 1e-9:
        raise ValueError(
            "ForestArea_km2 is inconsistent with "
            "ForestPixelCount * 0.25."
        )

    for response in COUNT_RESPONSES:
        values = pd.to_numeric(data[response], errors="coerce")
        invalid = (
            values.isna()
            | ~np.isfinite(values)
            | (values < 0)
            | (np.abs(values - np.round(values)) > 1e-9)
        )
        if invalid.any():
            raise ValueError(f"{response} contains invalid count values.")
        data[response] = values.astype(int)

    unexpected_countries = sorted(
        set(data["Country"].dropna().astype(str))
        - set(COUNTRY_LEVELS)
    )
    if unexpected_countries:
        raise ValueError(
            "Unexpected Country values: "
            + ", ".join(unexpected_countries)
        )

    country_dummies = pd.get_dummies(
        data["Country"].astype(str),
        prefix="Country",
        dtype=float,
    )

    for column in COUNTRY_DUMMY_COLUMNS:
        if column not in country_dummies.columns:
            country_dummies[column] = 0.0

    country_dummies = country_dummies[
        list(COUNTRY_DUMMY_COLUMNS)
    ]

    data = pd.concat(
        [
            data.reset_index(drop=True),
            country_dummies.reset_index(drop=True),
        ],
        axis=1,
    )

    return data


def definition_components(
    definition: Mapping[str, object],
) -> Dict[str, object]:
    return {
        "count_predictors": parse_list(
            definition["Count_Predictors"]
        ),
        "count_country_terms": parse_list(
            definition["Count_Country_Terms"]
        ),
        "zero_predictors": parse_list(
            definition["Structural_Zero_Predictors"]
        ),
        "zero_country_terms": parse_list(
            definition["Structural_Zero_Country_Terms"]
        ),
        "transformation_scheme": str(
            definition["Transformation_Scheme"]
        ),
        "model_structure": str(
            definition["Model_Structure"]
        ),
    }


def model_sample(
    base_data: pd.DataFrame,
    definition: Mapping[str, object],
    threshold: float,
) -> pd.DataFrame:
    components = definition_components(definition)
    response = str(definition["Count_Response"])
    season = int(definition["Season_Code"])

    required = list(
        dict.fromkeys(
            [
                "GRID_UID",
                "GRID_ID",
                "Country",
                "Year",
                "Season",
                "ForestPixelCount",
                "ForestArea_km2",
                response,
                *components["count_predictors"],
                *components["count_country_terms"],
                *components["zero_predictors"],
                *components["zero_country_terms"],
            ]
        )
    )

    subset = base_data.loc[
        (base_data["Season"].astype(int) == season)
        & (base_data["ForestArea_km2"] > threshold),
        required,
    ].copy()

    numeric_required = [
        column
        for column in required
        if column not in {"GRID_UID", "GRID_ID", "Country"}
    ]

    for column in numeric_required:
        subset[column] = pd.to_numeric(
            subset[column],
            errors="coerce",
        )

    finite_mask = np.isfinite(
        subset[numeric_required].to_numpy(dtype=float)
    ).all(axis=1)

    subset = subset.loc[finite_mask].copy()
    subset = subset.sort_values(
        ["GRID_UID", "Year", "Season"]
    ).reset_index(drop=True)

    if subset.empty:
        raise ValueError(
            f"No complete rows for threshold {threshold}, "
            f"{definition['Season_Label']} {response}."
        )

    if (subset["ForestPixelCount"] <= 0).any():
        raise ValueError(
            "ForestPixelCount must be positive after threshold filtering."
        )

    if subset.duplicated(
        ["GRID_UID", "Year", "Season"],
        keep=False,
    ).any():
        raise ValueError(
            "Duplicate model-sample panel keys were found."
        )

    return subset


# =============================================================================
# Threshold retention summaries
# =============================================================================

def retention_tables(
    base_data: pd.DataFrame,
) -> pd.DataFrame:
    all_forest = base_data.loc[
        base_data["ForestArea_km2"] > 0
    ].copy()

    threshold_one = base_data.loc[
        base_data["ForestArea_km2"] > 1.0
    ].copy()

    rows: List[Dict[str, object]] = []

    for threshold in THRESHOLDS_KM2:
        eligible = base_data.loc[
            base_data["ForestArea_km2"] > threshold
        ].copy()

        for season, season_label in SEASONS.items():
            season_data = eligible.loc[
                eligible["Season"].astype(int) == season
            ]
            season_all = all_forest.loc[
                all_forest["Season"].astype(int) == season
            ]
            season_one = threshold_one.loc[
                threshold_one["Season"].astype(int) == season
            ]

            for country in ("All", *COUNTRY_LEVELS):
                if country == "All":
                    current = season_data
                    reference_all = season_all
                    reference_one = season_one
                else:
                    current = season_data.loc[
                        season_data["Country"] == country
                    ]
                    reference_all = season_all.loc[
                        season_all["Country"] == country
                    ]
                    reference_one = season_one.loc[
                        season_one["Country"] == country
                    ]

                for response in COUNT_RESPONSES:
                    total = float(current[response].sum())
                    reference_all_total = float(
                        reference_all[response].sum()
                    )
                    reference_one_total = float(
                        reference_one[response].sum()
                    )

                    rows.append(
                        {
                            "Forest_Threshold_km2": threshold,
                            "Strict_Rule": (
                                f"ForestArea_km2 > {threshold:g}"
                            ),
                            "Minimum_Retained_Forest_Pixels": (
                                int(
                                    math.floor(
                                        threshold
                                        / FOREST_PIXEL_AREA_KM2
                                    )
                                    + 1
                                )
                            ),
                            "Season_Code": season,
                            "Season_Label": season_label,
                            "Country": country,
                            "Count_Response": response,
                            "Rate_Scale_Name": RATE_NAMES[response],
                            "Rows": int(len(current)),
                            "GRID_UID_Count": int(
                                current["GRID_UID"].nunique()
                            ),
                            "Positive_Rows": int(
                                (current[response] > 0).sum()
                            ),
                            "Zero_Rows": int(
                                (current[response] == 0).sum()
                            ),
                            "Zero_Percent": (
                                100.0
                                * float(
                                    (current[response] == 0).mean()
                                )
                                if len(current)
                                else np.nan
                            ),
                            "Observed_Total_Count": total,
                            "Rows_Retained_vs_All_Forest_Percent": (
                                100.0
                                * len(current)
                                / len(reference_all)
                                if len(reference_all)
                                else np.nan
                            ),
                            "Event_Count_Retained_vs_All_Forest_Percent": (
                                100.0
                                * total
                                / reference_all_total
                                if reference_all_total > 0
                                else np.nan
                            ),
                            "Rows_Retained_vs_Threshold1_Percent": (
                                100.0
                                * len(current)
                                / len(reference_one)
                                if len(reference_one)
                                else np.nan
                            ),
                            "Event_Count_Retained_vs_Threshold1_Percent": (
                                100.0
                                * total
                                / reference_one_total
                                if reference_one_total > 0
                                else np.nan
                            ),
                        }
                    )

    return pd.DataFrame(rows)


# =============================================================================
# Model fitting
# =============================================================================

def parameter_rows(
    theta: np.ndarray,
    x_names: Sequence[str],
    z_names: Sequence[str],
    threshold: float,
    definition: Mapping[str, object],
) -> pd.DataFrame:
    model = str(definition["Model_Structure"])
    expected_length = (
        len(x_names)
        + (len(z_names) if model == "ZINB1" else 0)
        + 1
    )

    if len(theta) != expected_length:
        raise ValueError(
            f"Unexpected theta length for {definition['Season_Label']} "
            f"{definition['Count_Response']}: "
            f"{len(theta)} versus {expected_length}."
        )

    rows: List[Dict[str, object]] = []
    position = 0

    # The locked Step-06c ZINB1 likelihood stores parameters as:
    #   structural-zero gamma -> count beta -> log(alpha).
    # Single-stage NB1 stores:
    #   count beta -> log(alpha).
    if model == "ZINB1":
        for term in z_names:
            estimate = float(theta[position])
            rows.append(
                {
                    "Forest_Threshold_km2": threshold,
                    "Season_Code": int(
                        definition["Season_Code"]
                    ),
                    "Season_Label": str(
                        definition["Season_Label"]
                    ),
                    "Count_Response": str(
                        definition["Count_Response"]
                    ),
                    "Rate_Scale_Name": str(
                        definition["Rate_Scale_Name"]
                    ),
                    "Model_Structure": model,
                    "Component": "Structural_Zero",
                    "Term": str(term),
                    "Estimate": estimate,
                    "Exponentiated_Estimate": float(
                        np.exp(estimate)
                    ),
                    "Interpretation_Scale": (
                        "Structural-zero odds ratio per model-scale "
                        "predictor increase"
                    ),
                }
            )
            position += 1

    for term in x_names:
        estimate = float(theta[position])
        rows.append(
            {
                "Forest_Threshold_km2": threshold,
                "Season_Code": int(definition["Season_Code"]),
                "Season_Label": str(definition["Season_Label"]),
                "Count_Response": str(definition["Count_Response"]),
                "Rate_Scale_Name": str(definition["Rate_Scale_Name"]),
                "Model_Structure": model,
                "Component": "Count",
                "Term": str(term),
                "Estimate": estimate,
                "Exponentiated_Estimate": float(np.exp(estimate)),
                "Interpretation_Scale": (
                    "Conditional expected-count ratio per model-scale "
                    "predictor increase"
                ),
            }
        )
        position += 1

    log_alpha = float(theta[position])
    rows.append(
        {
            "Forest_Threshold_km2": threshold,
            "Season_Code": int(definition["Season_Code"]),
            "Season_Label": str(definition["Season_Label"]),
            "Count_Response": str(definition["Count_Response"]),
            "Rate_Scale_Name": str(definition["Rate_Scale_Name"]),
            "Model_Structure": model,
            "Component": "Dispersion",
            "Term": "Log_Alpha",
            "Estimate": log_alpha,
            "Exponentiated_Estimate": float(np.exp(log_alpha)),
            "Interpretation_Scale": "NB1 alpha dispersion parameter",
        }
    )

    return pd.DataFrame(rows)


def fit_checkpoint_paths(
    threshold: float,
    definition: Mapping[str, object],
) -> Tuple[Path, Path]:
    stem = "_".join(
        [
            f"Threshold_{threshold_tag(threshold)}km2",
            f"S{int(definition['Season_Code'])}",
            safe_filename(definition["Count_Response"]),
        ]
    )
    return (
        CHECKPOINT_ROOT / f"{stem}.npz",
        CHECKPOINT_ROOT / f"{stem}.json",
    )


def fit_one_model(
    step04: object,
    step05: object,
    step06c: object,
    step07: object,
    sample: pd.DataFrame,
    definition: Mapping[str, object],
    threshold: float,
) -> Dict[str, object]:
    components = definition_components(definition)
    response = str(definition["Count_Response"])
    model = str(components["model_structure"])
    scheme = str(components["transformation_scheme"])

    signature = model_signature(
        definition,
        threshold,
        sample,
    )

    npz_path, json_path = fit_checkpoint_paths(
        threshold,
        definition,
    )

    if (
        not FORCE_RESTART
        and npz_path.exists()
        and json_path.exists()
    ):
        metadata = json.loads(
            json_path.read_text(encoding="utf-8")
        )

        if metadata.get("Signature") == signature:
            arrays = np.load(npz_path, allow_pickle=False)
            theta = np.asarray(arrays["theta"], dtype=float)
            x_names = [
                str(value)
                for value in arrays["x_names"].tolist()
            ]
            z_names = [
                str(value)
                for value in arrays["z_names"].tolist()
            ]
            status = dict(metadata["Optimizer_Status"])

            if valid_fit_status(status):
                log(
                    "Reused checkpoint: "
                    f">{threshold:g} km2 | "
                    f"{definition['Season_Label']} {response}."
                )
                return {
                    "theta": theta,
                    "x_names": x_names,
                    "z_names": z_names,
                    "status": status,
                    "signature": signature,
                    "checkpoint_reused": True,
                    "npz_path": npz_path,
                    "json_path": json_path,
                }

    x_training, _, x_names = step06c.train_valid_design(
        sample,
        sample,
        components["count_predictors"],
        scheme,
        components["count_country_terms"],
    )

    if model == "ZINB1":
        z_training, _, z_names = step06c.train_valid_design(
            sample,
            sample,
            components["zero_predictors"],
            scheme,
            components["zero_country_terms"],
        )
    elif model == "Single_Stage_NB1":
        z_training = np.empty(
            (len(sample), 0),
            dtype=float,
        )
        z_names = []
    else:
        raise ValueError(
            f"Unsupported model structure: {model}"
        )

    if (
        np.linalg.matrix_rank(x_training)
        != x_training.shape[1]
    ):
        raise ValueError(
            f"Rank-deficient count design for >{threshold:g} km2, "
            f"{definition['Season_Label']} {response}."
        )

    if (
        model == "ZINB1"
        and np.linalg.matrix_rank(z_training)
        != z_training.shape[1]
    ):
        raise ValueError(
            f"Rank-deficient zero design for >{threshold:g} km2, "
            f"{definition['Season_Label']} {response}."
        )

    y = sample[response].to_numpy(dtype=float)
    exposure = sample["ForestPixelCount"].to_numpy(dtype=float)

    training_start = step06c.training_start(
        step04,
        step05,
        y,
        x_training,
        z_training,
        exposure,
        model,
    )
    neutral_start = step06c.neutral_start(
        y,
        x_training,
        z_training,
        exposure,
        model,
    )

    log(
        "Fitting locked final regression: "
        f">{threshold:g} km2 | "
        f"{definition['Season_Label']} {response} | {model}."
    )

    theta, status = (
        step07.optimize_spatial_fold_with_rescue(
            step06c,
            training_start,
            neutral_start,
            y,
            x_training,
            z_training,
            exposure,
            model,
        )
    )

    theta = np.asarray(theta, dtype=float)
    status = dict(status)

    if not valid_fit_status(status):
        write_json(
            json_path,
            {
                "Signature": signature,
                "Optimizer_Status": status,
                "Status": "FAILED",
            },
        )
        raise RuntimeError(
            "Threshold model did not pass the locked optimization rule: "
            f">{threshold:g} km2, "
            f"{definition['Season_Label']} {response}.\n"
            f"Status: {json.dumps(status, default=str)}"
        )

    np.savez_compressed(
        npz_path,
        theta=theta,
        x_names=np.asarray(x_names, dtype=str),
        z_names=np.asarray(z_names, dtype=str),
    )

    write_json(
        json_path,
        {
            "Signature": signature,
            "Status": "SUCCESS",
            "Optimizer_Status": status,
            "Threshold_km2": threshold,
            "Strict_Rule": (
                f"ForestArea_km2 > {threshold:g}"
            ),
            "Season_Code": int(
                definition["Season_Code"]
            ),
            "Season_Label": str(
                definition["Season_Label"]
            ),
            "Count_Response": response,
            "Model_Structure": model,
            "Sample_Rows": int(len(sample)),
            "Sample_GRID_UID_Count": int(
                sample["GRID_UID"].nunique()
            ),
            "X_Names": list(map(str, x_names)),
            "Z_Names": list(map(str, z_names)),
        },
    )

    return {
        "theta": theta,
        "x_names": list(map(str, x_names)),
        "z_names": list(map(str, z_names)),
        "status": status,
        "signature": signature,
        "checkpoint_reused": False,
        "npz_path": npz_path,
        "json_path": json_path,
    }


def prediction_frame(
    step06c: object,
    training_sample: pd.DataFrame,
    validation_sample: pd.DataFrame,
    definition: Mapping[str, object],
    fit: Mapping[str, object],
    threshold: float,
) -> pd.DataFrame:
    components = definition_components(definition)
    response = str(definition["Count_Response"])
    model = str(components["model_structure"])
    scheme = str(components["transformation_scheme"])

    _, x_validation, x_names = step06c.train_valid_design(
        training_sample,
        validation_sample,
        components["count_predictors"],
        scheme,
        components["count_country_terms"],
    )

    if list(map(str, x_names)) != list(fit["x_names"]):
        raise ValueError(
            "Count-design names changed between fitting and prediction."
        )

    if model == "ZINB1":
        _, z_validation, z_names = step06c.train_valid_design(
            training_sample,
            validation_sample,
            components["zero_predictors"],
            scheme,
            components["zero_country_terms"],
        )
    else:
        z_validation = np.empty(
            (len(validation_sample), 0),
            dtype=float,
        )
        z_names = []

    if list(map(str, z_names)) != list(fit["z_names"]):
        raise ValueError(
            "Zero-design names changed between fitting and prediction."
        )

    y = validation_sample[response].to_numpy(dtype=float)
    exposure = validation_sample[
        "ForestPixelCount"
    ].to_numpy(dtype=float)

    outputs = step06c.stable_predictions(
        np.asarray(fit["theta"], dtype=float),
        y,
        x_validation,
        z_validation,
        exposure,
        model,
    )

    result = validation_sample[
        [
            "GRID_UID",
            "GRID_ID",
            "Country",
            "Year",
            "Season",
            "ForestPixelCount",
            "ForestArea_km2",
        ]
    ].copy()

    result["Forest_Threshold_km2"] = threshold
    result["Season_Label"] = str(
        definition["Season_Label"]
    )
    result["Count_Response"] = response
    result["Rate_Scale_Name"] = str(
        definition["Rate_Scale_Name"]
    )
    result["Model_Structure"] = model
    result["Observed_Count"] = y
    result["Predicted_Count"] = np.asarray(
        outputs["Predicted_Count"],
        dtype=float,
    )
    result["Observed_Rate"] = y / exposure
    result["Predicted_Rate"] = (
        result["Predicted_Count"].to_numpy(dtype=float)
        / exposure
    )
    result["Predicted_Zero_Probability"] = np.asarray(
        outputs["Predicted_Zero_Probability"],
        dtype=float,
    )
    result["Predictive_LogProbability"] = np.asarray(
        outputs["Predictive_LogProbability"],
        dtype=float,
    )

    numeric_outputs = [
        "Predicted_Count",
        "Predicted_Rate",
        "Predicted_Zero_Probability",
        "Predictive_LogProbability",
    ]

    if not np.isfinite(
        result[numeric_outputs].to_numpy(dtype=float)
    ).all():
        raise FloatingPointError(
            "Common-support predictions contain non-finite values."
        )

    return result


# =============================================================================
# Diagnostics and comparisons
# =============================================================================

def fitted_metrics(prediction: pd.DataFrame) -> Dict[str, float]:
    observed_count = prediction[
        "Observed_Count"
    ].to_numpy(dtype=float)
    predicted_count = prediction[
        "Predicted_Count"
    ].to_numpy(dtype=float)
    observed_rate = prediction[
        "Observed_Rate"
    ].to_numpy(dtype=float)
    predicted_rate = prediction[
        "Predicted_Rate"
    ].to_numpy(dtype=float)

    observed_zero = (
        observed_count == 0
    ).astype(float)
    predicted_zero = prediction[
        "Predicted_Zero_Probability"
    ].to_numpy(dtype=float)

    return {
        "Rows": int(len(prediction)),
        "GRID_UID_Count": int(
            prediction["GRID_UID"].nunique()
        ),
        "Count_RMSE": float(
            mean_squared_error(
                observed_count,
                predicted_count,
                squared=False,
            )
        ),
        "Count_MAE": float(
            mean_absolute_error(
                observed_count,
                predicted_count,
            )
        ),
        "Count_R2": safe_r2(
            observed_count,
            predicted_count,
        ),
        "Count_Spearman": safe_spearman(
            observed_count,
            predicted_count,
        ),
        "Rate_RMSE": float(
            mean_squared_error(
                observed_rate,
                predicted_rate,
                squared=False,
            )
        ),
        "Rate_MAE": float(
            mean_absolute_error(
                observed_rate,
                predicted_rate,
            )
        ),
        "Rate_R2": safe_r2(
            observed_rate,
            predicted_rate,
        ),
        "Rate_Spearman": safe_spearman(
            observed_rate,
            predicted_rate,
        ),
        "Rate_Mean_Bias": float(
            np.mean(predicted_rate - observed_rate)
        ),
        "Predicted_to_Observed_Mean_Rate_Ratio": (
            float(
                np.mean(predicted_rate)
                / np.mean(observed_rate)
            )
            if np.mean(observed_rate) > 0
            else np.nan
        ),
        "Zero_Brier": float(
            np.mean((observed_zero - predicted_zero) ** 2)
        ),
        "Mean_Predictive_LogProbability": float(
            prediction[
                "Predictive_LogProbability"
            ].mean()
        ),
    }


def coefficient_stability(
    coefficients: pd.DataFrame,
) -> pd.DataFrame:
    key_columns = [
        "Season_Code",
        "Season_Label",
        "Count_Response",
        "Rate_Scale_Name",
        "Model_Structure",
        "Component",
        "Term",
    ]

    reference = coefficients.loc[
        coefficients["Forest_Threshold_km2"]
        == MAIN_THRESHOLD_KM2,
        key_columns + ["Estimate"],
    ].rename(
        columns={"Estimate": "Main_Threshold_Estimate"}
    )

    result = coefficients.merge(
        reference,
        on=key_columns,
        how="left",
        validate="many_to_one",
    )

    result["Is_Intercept"] = (
        result["Term"]
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "", regex=True)
        .isin({"intercept", "const"})
    )

    result["Is_Dispersion"] = (
        result["Component"] == "Dispersion"
    )

    result["Is_Driver_Coefficient"] = (
        ~result["Is_Intercept"]
        & ~result["Is_Dispersion"]
    )

    result["Sign_Consistent_vs_Main"] = (
        np.sign(result["Estimate"])
        == np.sign(result["Main_Threshold_Estimate"])
    )

    result["Absolute_Change_vs_Main"] = np.abs(
        result["Estimate"]
        - result["Main_Threshold_Estimate"]
    )

    result["Relative_Change_vs_Main"] = (
        result["Absolute_Change_vs_Main"]
        / np.maximum(
            np.abs(
                result["Main_Threshold_Estimate"]
            ),
            COEFFICIENT_REFERENCE_FLOOR,
        )
    )

    return result


def prediction_agreement(
    common_predictions: pd.DataFrame,
) -> pd.DataFrame:
    keys = [
        "GRID_UID",
        "Year",
        "Season",
        "Count_Response",
    ]

    reference = common_predictions.loc[
        common_predictions["Forest_Threshold_km2"]
        == MAIN_THRESHOLD_KM2,
        keys
        + [
            "Predicted_Count",
            "Predicted_Rate",
            "Predicted_Zero_Probability",
        ],
    ].rename(
        columns={
            "Predicted_Count": "Main_Predicted_Count",
            "Predicted_Rate": "Main_Predicted_Rate",
            "Predicted_Zero_Probability": (
                "Main_Predicted_Zero_Probability"
            ),
        }
    )

    rows: List[Dict[str, object]] = []

    grouped = common_predictions.groupby(
        [
            "Forest_Threshold_km2",
            "Season",
            "Season_Label",
            "Count_Response",
            "Rate_Scale_Name",
        ],
        sort=True,
    )

    for group_values, group in grouped:
        (
            threshold,
            season,
            season_label,
            response,
            rate_name,
        ) = group_values

        merged = group.merge(
            reference,
            on=keys,
            how="inner",
            validate="one_to_one",
        )

        if len(merged) != len(group):
            raise ValueError(
                "Common-support prediction keys did not align with "
                "the main-threshold reference."
            )

        rows.append(
            {
                "Forest_Threshold_km2": threshold,
                "Season_Code": int(season),
                "Season_Label": season_label,
                "Count_Response": response,
                "Rate_Scale_Name": rate_name,
                "Rows": int(len(merged)),
                "Predicted_Count_Pearson_vs_Main": safe_pearson(
                    merged["Predicted_Count"],
                    merged["Main_Predicted_Count"],
                ),
                "Predicted_Count_Spearman_vs_Main": safe_spearman(
                    merged["Predicted_Count"],
                    merged["Main_Predicted_Count"],
                ),
                "Predicted_Rate_Pearson_vs_Main": safe_pearson(
                    merged["Predicted_Rate"],
                    merged["Main_Predicted_Rate"],
                ),
                "Predicted_Rate_Spearman_vs_Main": safe_spearman(
                    merged["Predicted_Rate"],
                    merged["Main_Predicted_Rate"],
                ),
                "Predicted_Rate_MAE_vs_Main": float(
                    np.mean(
                        np.abs(
                            merged["Predicted_Rate"]
                            - merged["Main_Predicted_Rate"]
                        )
                    )
                ),
                "Predicted_Zero_Probability_MAE_vs_Main": float(
                    np.mean(
                        np.abs(
                            merged[
                                "Predicted_Zero_Probability"
                            ]
                            - merged[
                                "Main_Predicted_Zero_Probability"
                            ]
                        )
                    )
                ),
                "Mean_Predicted_Rate_Ratio_vs_Main": (
                    float(
                        merged["Predicted_Rate"].mean()
                        / merged["Main_Predicted_Rate"].mean()
                    )
                    if merged[
                        "Main_Predicted_Rate"
                    ].mean() > 0
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def normalize_term(value: object) -> str:
    text = re.sub(
        r"[^a-z0-9]+",
        "",
        str(value).lower(),
    )
    if text in {"const", "intercept"}:
        return "intercept"
    return text


def normalize_component(value: object) -> str:
    text = re.sub(
        r"[^a-z0-9]+",
        "",
        str(value).lower(),
    )

    if "zero" in text or text in {"zi", "structuralzero"}:
        return "structuralzero"
    if "count" in text or "conditional" in text:
        return "count"
    if "disp" in text or "alpha" in text:
        return "dispersion"

    return text


def find_column(
    columns: Sequence[str],
    candidates: Sequence[str],
) -> str | None:
    lookup = {
        re.sub(r"[^a-z0-9]+", "", column.lower()): column
        for column in columns
    }

    for candidate in candidates:
        key = re.sub(
            r"[^a-z0-9]+",
            "",
            candidate.lower(),
        )
        if key in lookup:
            return lookup[key]

    return None


def reproduce_main_coefficients(
    coefficients: pd.DataFrame,
) -> pd.DataFrame:
    if not FINAL_COEFFICIENT_CSV.exists():
        return pd.DataFrame(
            [
                {
                    "Status": "Formal coefficient file not found",
                    "Compared_Row_Count": 0,
                }
            ]
        )

    formal = pd.read_csv(FINAL_COEFFICIENT_CSV)
    columns = list(formal.columns)

    season_column = find_column(
        columns,
        ["Season_Code", "Season"],
    )
    response_column = find_column(
        columns,
        ["Count_Response", "Response"],
    )
    component_column = find_column(
        columns,
        ["Component", "Model_Component"],
    )
    term_column = find_column(
        columns,
        ["Term", "Parameter", "Variable"],
    )
    estimate_column = find_column(
        columns,
        ["Estimate", "Coefficient"],
    )

    required_mapping = {
        "Season": season_column,
        "Response": response_column,
        "Component": component_column,
        "Term": term_column,
        "Estimate": estimate_column,
    }

    if any(value is None for value in required_mapping.values()):
        return pd.DataFrame(
            [
                {
                    "Status": (
                        "Formal coefficient columns could not be "
                        "identified automatically"
                    ),
                    "Compared_Row_Count": 0,
                    "Formal_Columns": ";".join(columns),
                }
            ]
        )

    formal_copy = formal[
        [
            season_column,
            response_column,
            component_column,
            term_column,
            estimate_column,
        ]
    ].copy()

    formal_copy.columns = [
        "Season_Code",
        "Count_Response",
        "Component",
        "Term",
        "Formal_Estimate",
    ]

    formal_copy["Season_Code"] = pd.to_numeric(
        formal_copy["Season_Code"],
        errors="coerce",
    )
    formal_copy["Formal_Estimate"] = pd.to_numeric(
        formal_copy["Formal_Estimate"],
        errors="coerce",
    )
    formal_copy["Component_Normalized"] = formal_copy[
        "Component"
    ].map(normalize_component)
    formal_copy["Term_Normalized"] = formal_copy[
        "Term"
    ].map(normalize_term)

    refit = coefficients.loc[
        coefficients["Forest_Threshold_km2"]
        == MAIN_THRESHOLD_KM2
    ].copy()

    refit["Component_Normalized"] = refit[
        "Component"
    ].map(normalize_component)
    refit["Term_Normalized"] = refit[
        "Term"
    ].map(normalize_term)

    merged = refit.merge(
        formal_copy,
        on=[
            "Season_Code",
            "Count_Response",
            "Component_Normalized",
            "Term_Normalized",
        ],
        how="inner",
    )

    if merged.empty:
        return pd.DataFrame(
            [
                {
                    "Status": (
                        "No formal coefficient rows matched the "
                        "main-threshold refit"
                    ),
                    "Compared_Row_Count": 0,
                }
            ]
        )

    merged["Absolute_Difference"] = np.abs(
        merged["Estimate"]
        - merged["Formal_Estimate"]
    )
    merged["Status"] = "Compared"
    merged["Compared_Row_Count"] = len(merged)

    return merged


# =============================================================================
# Figures
# =============================================================================

def make_retention_figure(retention: pd.DataFrame) -> None:
    overall = retention.loc[
        retention["Country"] == "All"
    ].copy()

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10, 7),
        constrained_layout=True,
    )

    combinations = [
        ("Fire_Count", "GRID_UID_Count", "Retained grids"),
        (
            "Fire_Count",
            "Event_Count_Retained_vs_Threshold1_Percent",
            "Fire Count retained (%)",
        ),
        (
            "Burned_Pixel_Count",
            "GRID_UID_Count",
            "Retained grids",
        ),
        (
            "Burned_Pixel_Count",
            "Event_Count_Retained_vs_Threshold1_Percent",
            "Burned pixels retained (%)",
        ),
    ]

    for axis, (response, value, title) in zip(
        axes.ravel(),
        combinations,
    ):
        subset = overall.loc[
            overall["Count_Response"] == response
        ]

        for season, season_label in SEASONS.items():
            season_data = subset.loc[
                subset["Season_Code"] == season
            ].sort_values("Forest_Threshold_km2")

            axis.plot(
                season_data["Forest_Threshold_km2"],
                season_data[value],
                marker="o",
                label=season_label,
            )

        axis.set_title(title, fontfamily="Times New Roman")
        axis.set_xlabel(
            "Minimum forest-area threshold (km²; strict >)",
            fontfamily="Times New Roman",
        )
        axis.set_ylabel(
            title,
            fontfamily="Times New Roman",
        )
        axis.grid(alpha=0.25)

    axes[0, 0].legend(
        frameon=False,
        prop={"family": "Times New Roman"},
    )

    for axis in axes.ravel():
        for label in (
            axis.get_xticklabels()
            + axis.get_yticklabels()
        ):
            label.set_fontfamily("Times New Roman")

    path = FIGURE_ROOT / "Figure_S1_Threshold_Retention.png"
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def make_coefficient_figures(
    stability: pd.DataFrame,
) -> None:
    plot_data = stability.loc[
        stability["Is_Driver_Coefficient"]
    ].copy()

    offsets = {
        1.0: -0.18,
        2.5: 0.0,
        5.0: 0.18,
    }

    markers = {
        1.0: "o",
        2.5: "s",
        5.0: "^",
    }

    for (
        season,
        season_label,
        response,
        rate_name,
    ), group in plot_data.groupby(
        [
            "Season_Code",
            "Season_Label",
            "Count_Response",
            "Rate_Scale_Name",
        ],
        sort=True,
    ):
        group = group.copy()
        group["Display_Term"] = (
            group["Component"].astype(str)
            + ": "
            + group["Term"].astype(str)
        )

        order = (
            group.loc[
                group["Forest_Threshold_km2"]
                == MAIN_THRESHOLD_KM2
            ]
            .sort_values(
                ["Component", "Estimate"]
            )["Display_Term"]
            .tolist()
        )

        if not order:
            continue

        y_lookup = {
            term: index
            for index, term in enumerate(order)
        }

        height = max(5.5, 0.28 * len(order) + 1.8)
        fig, axis = plt.subplots(
            figsize=(8.5, height),
        )

        axis.axvline(
            0,
            linewidth=0.8,
            color="black",
        )

        for threshold in THRESHOLDS_KM2:
            subset = group.loc[
                group["Forest_Threshold_km2"] == threshold
            ].copy()

            y = np.asarray(
                [
                    y_lookup[value] + offsets[threshold]
                    for value in subset["Display_Term"]
                ],
                dtype=float,
            )

            axis.scatter(
                subset["Estimate"],
                y,
                marker=markers[threshold],
                s=28,
                label=f">{threshold:g} km²",
            )

        axis.set_yticks(range(len(order)))
        axis.set_yticklabels(
            order,
            fontfamily="Times New Roman",
        )
        axis.set_xlabel(
            "Full-data maximum-likelihood coefficient",
            fontfamily="Times New Roman",
        )
        axis.set_title(
            f"{season_label} {rate_name}: threshold sensitivity",
            fontfamily="Times New Roman",
            fontweight="bold",
        )
        axis.grid(axis="x", alpha=0.25)
        axis.legend(
            frameon=False,
            prop={"family": "Times New Roman"},
        )

        for label in axis.get_xticklabels():
            label.set_fontfamily("Times New Roman")

        path = (
            FIGURE_ROOT
            / (
                f"Coefficient_Stability_S{int(season)}_"
                f"{season_label}_{rate_name}.png"
            )
        )
        fig.savefig(
            path,
            dpi=600,
            bbox_inches="tight",
        )
        plt.close(fig)


# =============================================================================
# Manifest
# =============================================================================

def write_manifest() -> pd.DataFrame:
    files = sorted(
        path
        for path in OUTPUT_ROOT.rglob("*")
        if path.is_file()
        and path != OUTPUT_MANIFEST_FILE
    )

    rows = []

    for path in files:
        rows.append(
            {
                "Relative_Path": path.relative_to(
                    OUTPUT_ROOT
                ).as_posix(),
                "File_Size_Bytes": path.stat().st_size,
                "SHA256": sha256_file(path),
            }
        )

    manifest = pd.DataFrame(rows)
    manifest.to_csv(
        OUTPUT_MANIFEST_FILE,
        index=False,
        encoding="utf-8-sig",
    )
    return manifest


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    require_files(
        [
            INPUT_CSV,
            STEP02_SCRIPT,
            STEP04_SCRIPT,
            STEP05_SCRIPT,
            STEP06C_SCRIPT,
            STEP07_SCRIPT,
            FINAL_STRUCTURE_CSV,
        ]
    )

    if FORCE_RESTART and OUTPUT_ROOT.exists():
        import shutil
        shutil.rmtree(OUTPUT_ROOT)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)

    if FORCE_RESTART or not LOG_FILE.exists():
        LOG_FILE.write_text("", encoding="utf-8")

    log("Starting forest-area threshold sensitivity analysis.")
    log(f"Code version: {CODE_VERSION}")
    log(f"Release note: {RELEASE_NOTE}")

    threshold_audit = audit_main_threshold()
    threshold_audit.to_csv(
        MAIN_THRESHOLD_AUDIT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    parameter_order_audit = pd.DataFrame(
        [
            {
                "Model_Structure": "ZINB1",
                "Locked_Internal_Order": (
                    "Structural_Zero;Count;Dispersion"
                ),
                "Exported_Label_Order": (
                    "Structural_Zero;Count;Dispersion"
                ),
                "Passed": True,
                "Evidence": (
                    "Step-06c loglike_score_obs uses "
                    "gamma=theta[:kz], beta=theta[kz:kz+kx]"
                ),
            },
            {
                "Model_Structure": "Single_Stage_NB1",
                "Locked_Internal_Order": "Count;Dispersion",
                "Exported_Label_Order": "Count;Dispersion",
                "Passed": True,
                "Evidence": (
                    "Step-06c NB1 uses beta=theta[:kx]"
                ),
            },
        ]
    )
    parameter_order_audit.to_csv(
        PARAMETER_ORDER_AUDIT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    log(
        "Verified main threshold: ForestArea_km2 > 2.5, "
        "equivalent to ForestPixelCount > 10."
    )

    step02 = load_module(
        STEP02_SCRIPT,
        "forest_threshold_step02",
    )
    step04 = load_module(
        STEP04_SCRIPT,
        "forest_threshold_step04",
    )
    step05 = load_module(
        STEP05_SCRIPT,
        "forest_threshold_step05",
    )
    step06c = load_module(
        STEP06C_SCRIPT,
        "forest_threshold_step06c",
    )
    step07 = load_module(
        STEP07_SCRIPT,
        "forest_threshold_step07",
    )

    definitions = final_definitions(step07)
    definitions.to_csv(
        LOCKED_DEFINITION_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    data = read_base_data(definitions)

    retention = retention_tables(data)
    retention.to_csv(
        RETENTION_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    samples: Dict[
        Tuple[float, int, str],
        pd.DataFrame,
    ] = {}

    sample_rows: List[Dict[str, object]] = []

    for threshold in THRESHOLDS_KM2:
        for definition in definitions.to_dict(
            orient="records"
        ):
            sample = model_sample(
                data,
                definition,
                threshold,
            )

            season = int(definition["Season_Code"])
            response = str(definition["Count_Response"])
            samples[(threshold, season, response)] = sample

            sample_rows.append(
                {
                    "Forest_Threshold_km2": threshold,
                    "Strict_Rule": (
                        f"ForestArea_km2 > {threshold:g}"
                    ),
                    "Minimum_Retained_Forest_Pixels": int(
                        math.floor(
                            threshold
                            / FOREST_PIXEL_AREA_KM2
                        )
                        + 1
                    ),
                    "Season_Code": season,
                    "Season_Label": str(
                        definition["Season_Label"]
                    ),
                    "Count_Response": response,
                    "Rate_Scale_Name": str(
                        definition["Rate_Scale_Name"]
                    ),
                    "Model_Structure": str(
                        definition["Model_Structure"]
                    ),
                    "Rows": int(len(sample)),
                    "GRID_UID_Count": int(
                        sample["GRID_UID"].nunique()
                    ),
                    "Year_Count": int(
                        sample["Year"].nunique()
                    ),
                    "Minimum_Year": int(
                        sample["Year"].min()
                    ),
                    "Maximum_Year": int(
                        sample["Year"].max()
                    ),
                    "Positive_Rows": int(
                        (sample[response] > 0).sum()
                    ),
                    "Zero_Rows": int(
                        (sample[response] == 0).sum()
                    ),
                    "Observed_Total_Count": float(
                        sample[response].sum()
                    ),
                    "Duplicate_Panel_Keys": int(
                        sample.duplicated(
                            [
                                "GRID_UID",
                                "Year",
                                "Season",
                            ]
                        ).sum()
                    ),
                    "Sample_Row_Key_SHA256": (
                        row_key_sha256(sample)
                    ),
                }
            )

    sample_qa = pd.DataFrame(sample_rows)

    main_sample_check = sample_qa.loc[
        sample_qa["Forest_Threshold_km2"]
        == MAIN_THRESHOLD_KM2
    ]

    if not (
        (main_sample_check["Rows"]
         == EXPECTED_MAIN_ROWS_PER_MODEL).all()
        and (
            main_sample_check["GRID_UID_Count"]
            == EXPECTED_MAIN_GRID_COUNT
        ).all()
    ):
        sample_qa.to_csv(
            MODEL_SAMPLE_FILE,
            index=False,
            encoding="utf-8-sig",
        )
        raise ValueError(
            "The 2.5-km2 samples did not reproduce the locked "
            "123,729-row / 4,982-grid main-analysis universe."
        )

    sample_qa.to_csv(
        MODEL_SAMPLE_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    fits: Dict[
        Tuple[float, int, str],
        Dict[str, object],
    ] = {}

    fit_status_rows: List[Dict[str, object]] = []
    coefficient_frames: List[pd.DataFrame] = []
    common_prediction_frames: List[pd.DataFrame] = []
    common_metric_rows: List[Dict[str, object]] = []

    for threshold in THRESHOLDS_KM2:
        for definition in definitions.to_dict(
            orient="records"
        ):
            season = int(definition["Season_Code"])
            response = str(definition["Count_Response"])
            key = (threshold, season, response)
            sample = samples[key]

            fit = fit_one_model(
                step04,
                step05,
                step06c,
                step07,
                sample,
                definition,
                threshold,
            )
            fits[key] = fit

            status = dict(fit["status"])

            fit_status_rows.append(
                {
                    "Forest_Threshold_km2": threshold,
                    "Season_Code": season,
                    "Season_Label": str(
                        definition["Season_Label"]
                    ),
                    "Count_Response": response,
                    "Rate_Scale_Name": str(
                        definition["Rate_Scale_Name"]
                    ),
                    "Model_Structure": str(
                        definition["Model_Structure"]
                    ),
                    "Rows": int(len(sample)),
                    "GRID_UID_Count": int(
                        sample["GRID_UID"].nunique()
                    ),
                    "Converged": bool(
                        status.get("Converged", False)
                    ),
                    "Optimizer_Success": bool(
                        status.get(
                            "Optimizer_Success",
                            False,
                        )
                    ),
                    "Status_Code": status.get(
                        "Status_Code"
                    ),
                    "Negative_LogLikelihood": status.get(
                        "Negative_LogLikelihood"
                    ),
                    "Gradient_Infinity_Norm": status.get(
                        "Gradient_Infinity_Norm"
                    ),
                    "Relative_Gradient": status.get(
                        "Relative_Gradient"
                    ),
                    "Parameter_Bound_Hit": bool(
                        status.get(
                            "Parameter_Bound_Hit",
                            True,
                        )
                    ),
                    "Accepted_By_Locked_Rule": (
                        valid_fit_status(status)
                    ),
                    "Checkpoint_Reused": bool(
                        fit["checkpoint_reused"]
                    ),
                    "Definition_Signature": (
                        fit["signature"]
                    ),
                    "Parameter_Count": int(
                        len(fit["theta"])
                    ),
                }
            )

            coefficient_frames.append(
                parameter_rows(
                    np.asarray(
                        fit["theta"],
                        dtype=float,
                    ),
                    fit["x_names"],
                    fit["z_names"],
                    threshold,
                    definition,
                )
            )

            common_sample = samples[
                (
                    COMMON_SUPPORT_THRESHOLD_KM2,
                    season,
                    response,
                )
            ]

            prediction = prediction_frame(
                step06c,
                sample,
                common_sample,
                definition,
                fit,
                threshold,
            )
            common_prediction_frames.append(prediction)

            metric = fitted_metrics(prediction)
            common_metric_rows.append(
                {
                    "Forest_Threshold_km2": threshold,
                    "Common_Support_Rule": (
                        "ForestArea_km2 > 5"
                    ),
                    "Diagnostic_Type": (
                        "Full-data fitted diagnostic; "
                        "not out-of-fold validation"
                    ),
                    "Season_Code": season,
                    "Season_Label": str(
                        definition["Season_Label"]
                    ),
                    "Count_Response": response,
                    "Rate_Scale_Name": str(
                        definition["Rate_Scale_Name"]
                    ),
                    "Model_Structure": str(
                        definition["Model_Structure"]
                    ),
                    **metric,
                }
            )

    fit_status = pd.DataFrame(fit_status_rows)

    if len(fit_status) != EXPECTED_FIT_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_FIT_COUNT} threshold fits, "
            f"found {len(fit_status)}."
        )

    if not fit_status[
        "Accepted_By_Locked_Rule"
    ].all():
        fit_status.to_csv(
            FIT_STATUS_FILE,
            index=False,
            encoding="utf-8-sig",
        )
        raise RuntimeError(
            "At least one threshold model failed the locked "
            "optimization rule."
        )

    fit_status.to_csv(
        FIT_STATUS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    coefficients = pd.concat(
        coefficient_frames,
        ignore_index=True,
    )
    coefficients.to_csv(
        COEFFICIENT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    stability = coefficient_stability(coefficients)
    stability.to_csv(
        COEFFICIENT_STABILITY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    common_predictions = pd.concat(
        common_prediction_frames,
        ignore_index=True,
    )

    common_predictions.to_csv(
        COMMON_PREDICTION_FILE,
        index=False,
        compression="gzip",
    )

    common_metrics = pd.DataFrame(
        common_metric_rows
    )
    common_metrics.to_csv(
        COMMON_METRIC_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    agreement = prediction_agreement(
        common_predictions
    )
    agreement.to_csv(
        PREDICTION_AGREEMENT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    reproduction = reproduce_main_coefficients(
        coefficients
    )
    reproduction.to_csv(
        MAIN_REPRODUCTION_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    if (
        "Absolute_Difference" not in reproduction.columns
        or len(reproduction) == 0
        or not np.isfinite(
            pd.to_numeric(
                reproduction["Absolute_Difference"],
                errors="coerce",
            )
        ).all()
        or float(
            pd.to_numeric(
                reproduction["Absolute_Difference"],
                errors="coerce",
            ).max()
        ) > 1e-3
    ):
        raise RuntimeError(
            "Corrected 2.5-km2 coefficient labels did not reproduce "
            "the formal Step-06d point estimates within 1e-3. "
            "See 11_Main_Threshold_Coefficient_Reproduction.csv."
        )

    driver_stability = stability.loc[
        stability["Is_Driver_Coefficient"]
        & (
            stability["Forest_Threshold_km2"]
            != MAIN_THRESHOLD_KM2
        )
    ].copy()

    summary_rows: List[Dict[str, object]] = []

    for (
        season,
        season_label,
        response,
        rate_name,
    ), group in driver_stability.groupby(
        [
            "Season_Code",
            "Season_Label",
            "Count_Response",
            "Rate_Scale_Name",
        ],
        sort=True,
    ):
        agreement_group = agreement.loc[
            (agreement["Season_Code"] == season)
            & (
                agreement["Count_Response"]
                == response
            )
            & (
                agreement["Forest_Threshold_km2"]
                != MAIN_THRESHOLD_KM2
            )
        ]

        summary_rows.append(
            {
                "Season_Code": int(season),
                "Season_Label": season_label,
                "Count_Response": response,
                "Rate_Scale_Name": rate_name,
                "Driver_Coefficient_Comparisons": int(
                    len(group)
                ),
                "Driver_Coefficient_Sign_Consistency_N": int(
                    group[
                        "Sign_Consistent_vs_Main"
                    ].sum()
                ),
                "Driver_Coefficient_Sign_Consistency_Percent": (
                    100.0
                    * float(
                        group[
                            "Sign_Consistent_vs_Main"
                        ].mean()
                    )
                ),
                "Median_Absolute_Coefficient_Change": float(
                    group[
                        "Absolute_Change_vs_Main"
                    ].median()
                ),
                "Maximum_Absolute_Coefficient_Change": float(
                    group[
                        "Absolute_Change_vs_Main"
                    ].max()
                ),
                "Median_Relative_Coefficient_Change": float(
                    group[
                        "Relative_Change_vs_Main"
                    ].median()
                ),
                "Minimum_Common_Support_Predicted_Rate_Spearman_vs_Main": (
                    float(
                        agreement_group[
                            "Predicted_Rate_Spearman_vs_Main"
                        ].min()
                    )
                ),
                "Maximum_Common_Support_Predicted_Rate_MAE_vs_Main": (
                    float(
                        agreement_group[
                            "Predicted_Rate_MAE_vs_Main"
                        ].max()
                    )
                ),
                "Interpretation": (
                    "Descriptive threshold robustness; "
                    "formal two-way inference remains Step 06d "
                    "at >2.5 km2"
                ),
            }
        )

    stability_summary = pd.DataFrame(
        summary_rows
    )
    stability_summary.to_csv(
        STABILITY_SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    make_retention_figure(retention)
    make_coefficient_figures(stability)

    method_definition = {
        "Step": "09_Forest_Threshold_Sensitivity",
        "Code_Version": CODE_VERSION,
        "Release_Note": RELEASE_NOTE,
        "Created": datetime.now().isoformat(
            timespec="seconds"
        ),
        "Main_Threshold_Verified": True,
        "Main_Threshold_Rule": (
            "ForestArea_km2 > 2.5"
        ),
        "Equivalent_Main_Pixel_Rule": (
            "ForestPixelCount > 10"
        ),
        "Minimum_Main_Retained_Pixel_Count": 11,
        "Sensitivity_Thresholds_km2": list(
            THRESHOLDS_KM2
        ),
        "Strict_Operator": STRICT_OPERATOR,
        "Model_Definition_Source": str(
            FINAL_STRUCTURE_CSV
        ),
        "Model_Structures_ReSelected": False,
        "Predictors_ReSelected": False,
        "Transformations_ReSelected": False,
        "Likelihood_Implementation": (
            "Step-06c explicit stable NB1/ZINB1 likelihood"
        ),
        "Optimization_Acceptance_Rule": {
            "Converged": True,
            "Maximum_Relative_Gradient": (
                RELATIVE_GRADIENT_MAX
            ),
            "Parameter_Bound_Hit": False,
        },
        "Exposure": "ForestPixelCount",
        "Offset_Equivalent": (
            "log(ForestPixelCount)"
        ),
        "New_Two_Way_Cluster_Robust_Inference": False,
        "Formal_Inference_Remains": str(
            FINAL_COEFFICIENT_CSV
        ),
        "Common_Support_Rule": (
            "ForestArea_km2 > 5.0"
        ),
        "Common_Support_Diagnostic": (
            "Full-data fitted prediction agreement; "
            "not OOF validation"
        ),
        "Random_Forest_Retrained": False,
        "SEM_Models_Retrained": False,
        "Primary_Use": (
            "Reviewer-facing validation of the minimum "
            "forest-area preprocessing threshold"
        ),
        "AIC_BIC_Across_Thresholds_Used_For_Selection": False,
    }

    write_json(METHOD_FILE, method_definition)

    software = {
        "Python": platform.python_version(),
        "Platform": platform.platform(),
        "NumPy": np.__version__,
        "pandas": pd.__version__,
        "SciPy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "statsmodels": statsmodels.__version__,
        "Matplotlib": matplotlib.__version__,
        "Input_File": str(INPUT_CSV),
        "Input_SHA256": sha256_file(INPUT_CSV),
        "Step07_Script_SHA256": sha256_file(
            STEP07_SCRIPT
        ),
        "Final_Structure_SHA256": sha256_file(
            FINAL_STRUCTURE_CSV
        ),
    }

    write_json(SOFTWARE_FILE, software)

    log(
        "Forest-threshold sensitivity completed: "
        f"{len(fit_status)}/{EXPECTED_FIT_COUNT} "
        "locked final-regression models passed."
    )
    log(
        "No RF model and no SEM model was trained. "
        "No model selection was performed."
    )

    manifest = write_manifest()

    print()
    print(f"Output directory: {OUTPUT_ROOT}")
    print(
        "Main threshold verified: "
        "ForestArea_km2 > 2.5"
    )
    print(
        "Equivalent pixel rule: "
        "ForestPixelCount > 10"
    )
    print(
        "Minimum retained forest pixels under main rule: 11"
    )
    print(
        "Threshold models successful: "
        f"{int(fit_status['Accepted_By_Locked_Rule'].sum())}"
        f" / {EXPECTED_FIT_COUNT}"
    )
    print(
        "Main-threshold samples reproduced: "
        f"{len(main_sample_check)} / "
        f"{EXPECTED_MODEL_COMBINATIONS}"
    )
    print(
        "Common-support threshold: "
        "ForestArea_km2 > 5"
    )
    print(
        "Main-threshold coefficient labels reproduced: "
        f"{len(reproduction)} rows; maximum absolute difference = "
        f"{float(reproduction['Absolute_Difference'].max()):.12g}"
    )
    print(
        "Driver coefficient comparisons: "
        f"{len(driver_stability)}"
    )
    print(
        "Driver coefficient signs consistent: "
        f"{int(driver_stability['Sign_Consistent_vs_Main'].sum())}"
        f" / {len(driver_stability)}"
    )
    print(
        "Output manifest files: "
        f"{len(manifest)}"
    )
    print("New robust-inference models calculated: False")
    print("Random Forest models retrained: False")
    print("SEM models retrained: False")
    print("Model selection performed: False")

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
