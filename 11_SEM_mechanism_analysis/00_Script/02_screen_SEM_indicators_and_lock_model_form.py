# -*- coding: utf-8 -*-
"""
SEM mechanism analysis Step 02:
screen redundant indicators and lock the generalized SEM model form.

Recommended location
--------------------
<REPOSITORY_ROOT>/11_SEM_mechanism_analysis/00_Script/
    02_screen_SEM_indicators_and_lock_model_form.py

Purpose
-------
This script uses the sealed SEM Step 01 dataset to:

1. verify the locked SEM input universe and hashes;
2. apply transparent, outcome-neutral structural redundancy rules;
3. evaluate within-season Spearman correlations, VIF, and condition numbers;
4. produce season-specific nonredundant indicator sets;
5. lock one common cross-season indicator set for coefficient comparability;
6. generate reduced season-response datasets containing the locked indicators;
7. lock the response-family, offset, adjustment, and random-effect structure
   for the later two-part piecewise generalized SEM.

This script does NOT fit an SEM, estimate paths, select paths from p values,
calculate direct/indirect effects, or use the fire response to choose indicators.
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
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy


# =============================================================================
# Project paths and locked versions
# =============================================================================

CODE_ROOT = Path(
    os.environ.get(
        "WILDFIRE_CODE_ROOT",
        str(Path(__file__).resolve().parents[2]),
    )
).resolve()

SEM_ROOT = CODE_ROOT / "11_SEM_mechanism_analysis"
STEP01_ROOT = SEM_ROOT / "01_SEM_Analysis_Dataset"
STEP01_SCRIPT = (
    SEM_ROOT
    / "00_Script"
    / "01_prepare_SEM_analysis_dataset.py"
)
STEP01_METHOD = STEP01_ROOT / "00_Method_Definition.json"
STEP01_AUDIT = STEP01_ROOT / "01_Input_Integrity_Audit.csv"
STEP01_MASTER_RAW = STEP01_ROOT / "03_Locked_SEM_Master_Raw.csv.gz"
STEP01_MASTER_STD = STEP01_ROOT / "04_Locked_SEM_Master_Standardized.csv.gz"
STEP01_DATASET_MANIFEST = (
    STEP01_ROOT / "05_SEM_Season_Response_Dataset_Manifest.csv"
)
STEP01_VARIABLE_DICTIONARY = STEP01_ROOT / "06_Variable_Dictionary.csv"
STEP01_STANDARDIZATION = (
    STEP01_ROOT / "08_Seasonal_Standardization_Parameters.csv"
)
STEP01_PANEL_QA = STEP01_ROOT / "12_Response_and_Panel_QA.csv"
STEP01_OUTPUT_MANIFEST = STEP01_ROOT / "15_Output_Manifest.csv"
STEP01_DATASET_DIR = STEP01_ROOT / "Season_Response_Datasets"

OUTPUT_ROOT = SEM_ROOT / "02_SEM_Indicator_Screening_Model_Lock"
SELECTED_DATASET_DIR = OUTPUT_ROOT / "Locked_Selected_Datasets"

CODE_VERSION = (
    "2026-06-30_SEM_INDICATOR_SCREENING_MODEL_LOCK_V1_"
    "OUTCOME_NEUTRAL_COMMON_CROSS_SEASON_SET"
)
EXPECTED_STEP01_VERSION = (
    "2026-06-30_SEM_DATA_PREPARATION_V1B_"
    "LOCKED_MODEL_UNIVERSE_DYNAMIC_PFT_CONSTANCY"
)
EXPECTED_STEP01_SCRIPT_SHA256 = (
    "831175c517237f38cb86778c1a19c7facb2592e52879f78611c0167dcf57da48"
)
EXPECTED_GRID_N = 4_982
EXPECTED_PANEL_N = 371_187
EXPECTED_SEASON_RESPONSE_N = 123_729
EXPECTED_SEASONS = {1: "Spring", 2: "Summer", 3: "Autumn"}
EXPECTED_RESPONSES = ["Fire_Count", "Burned_Pixel_Count"]

HASH_CHUNK_SIZE = 1024 * 1024
FLOAT_ATOL = 1e-10
FLOAT_RTOL = 1e-8

# Screening thresholds.
PAIRWISE_ABS_SPEARMAN_LIMIT = 0.80
TARGET_MAX_VIF = 5.0
HARD_MAX_VIF = 7.5
TARGET_CONDITION_NUMBER = 100.0

# Minimum conceptual coverage after screening.
MIN_DOMAIN_COUNTS = {
    "Meteorology": 5,
    "Vegetation": 2,
    "Topography": 2,
    "Anthropogenic": 3,
}


# =============================================================================
# Candidate variables and transparent a priori rules
# =============================================================================

VARIABLE_GROUPS: Dict[str, List[str]] = {
    "Meteorology": [
        "Temp",
        "Pre",
        "Rhum",
        "Wind",
        "SSRD",
        "LtgProxy",
        "FFMC",
        "DMC",
        "DC",
        "ISI",
        "BUI",
        "FWI",
        "SPEI1",
        "SPEI3",
        "SPEI6",
        "SPEI12",
        "SPEI24",
    ],
    "Vegetation": [
        "BD",
        "ND",
        "NE",
        "EVI",
        "PTC",
    ],
    "Topography": [
        "DEM",
        "Slope",
        "Aspect",
    ],
    "Anthropogenic": [
        "POP",
        "Road_dens",
        "Dis_Railway",
        "Dis_Farm",
        "Dis_Build",
        "Dis_Power",
    ],
}

ALL_PREDICTORS = [
    variable
    for domain_variables in VARIABLE_GROUPS.values()
    for variable in domain_variables
]

DOMAIN_OF = {
    variable: domain
    for domain, domain_variables in VARIABLE_GROUPS.items()
    for variable in domain_variables
}

# BUI and FWI are deterministic higher-order composites of other FWI-system
# components. They are excluded from the primary SEM indicator pool to avoid
# simultaneously treating a derived composite and its components as separate
# causal indicators. They remain available for later sensitivity analysis.
STRUCTURAL_EXCLUSIONS = {
    "BUI": (
        "Excluded from the primary SEM indicator pool because BUI is a "
        "deterministic composite derived from DMC and DC. Retaining BUI "
        "together with its components would create structural redundancy."
    ),
    "FWI": (
        "Excluded from the primary SEM indicator pool because FWI is a "
        "deterministic higher-order index derived from ISI and BUI. It is "
        "reserved for sensitivity analysis rather than entered with its "
        "component indices."
    ),
}

# The common set begins with direct, interpretable fire-triangle anchors.
# Optional variables are added only when the worst-season correlation, VIF, and
# condition-number rules remain satisfied.
CORE_ANCHORS = [
    "Temp",
    "Rhum",
    "Wind",
    "LtgProxy",
    "SPEI3",
    "SPEI12",
    "EVI",
    "PTC",
    "DEM",
    "Slope",
    "Aspect",
    "POP",
    "Road_dens",
    "Dis_Farm",
]

# Higher values are retained preferentially when a violation requires a drop.
RETENTION_PRIORITY: Dict[str, int] = {
    # Core anchors
    "Temp": 100,
    "Rhum": 100,
    "Wind": 95,
    "LtgProxy": 100,
    "SPEI3": 100,
    "SPEI12": 100,
    "EVI": 100,
    "PTC": 100,
    "DEM": 100,
    "Slope": 95,
    "Aspect": 90,
    "POP": 100,
    "Road_dens": 95,
    "Dis_Farm": 100,
    # Direct optional weather variables
    "Pre": 92,
    "SSRD": 90,
    # Plant functional types
    "BD": 84,
    "ND": 82,
    "NE": 82,
    # Infrastructure proximity
    "Dis_Build": 80,
    "Dis_Power": 80,
    "Dis_Railway": 78,
    # FWI-system components retained only if nonredundant
    "DC": 72,
    "FFMC": 70,
    "DMC": 68,
    "ISI": 68,
    # Alternative SPEI windows
    "SPEI1": 62,
    "SPEI6": 60,
    "SPEI24": 58,
    # Structurally excluded composites
    "BUI": 0,
    "FWI": 0,
}

OPTIONAL_CANDIDATES = sorted(
    [
        variable
        for variable in ALL_PREDICTORS
        if variable not in CORE_ANCHORS
        and variable not in STRUCTURAL_EXCLUSIONS
    ],
    key=lambda variable: (
        -RETENTION_PRIORITY[variable],
        variable,
    ),
)


# =============================================================================
# Output files
# =============================================================================

METHOD_FILE = OUTPUT_ROOT / "00_Method_Definition.json"
INPUT_AUDIT_FILE = OUTPUT_ROOT / "01_Input_Integrity_Audit.csv"
RULES_FILE = OUTPUT_ROOT / "02_Structural_Redundancy_Rules.csv"
SEASONAL_CORRELATION_FILE = (
    OUTPUT_ROOT / "03_Seasonal_Spearman_Correlation_Long.csv.gz"
)
WORST_CORRELATION_FILE = OUTPUT_ROOT / "04_Worst_Season_Correlation_Pairs.csv"
INITIAL_VIF_FILE = OUTPUT_ROOT / "05_Initial_Eligible_Set_Seasonal_VIF.csv"
SEASONAL_HISTORY_FILE = OUTPUT_ROOT / "06_Seasonal_Screening_History.csv"
SEASONAL_SELECTION_FILE = OUTPUT_ROOT / "07_Seasonal_Selected_Indicators.csv"
COMMON_HISTORY_FILE = OUTPUT_ROOT / "08_Common_Set_Selection_History.csv"
COMMON_SET_FILE = OUTPUT_ROOT / "09_Common_Locked_Indicator_Set.csv"
COMMON_VIF_FILE = OUTPUT_ROOT / "10_Common_Set_Seasonal_VIF.csv"
INFORMATION_RETENTION_FILE = OUTPUT_ROOT / "11_Information_Retention_QA.csv"
MODEL_FORM_FILE = OUTPUT_ROOT / "12_Model_Form_Lock.csv"
SUBMODEL_FILE = OUTPUT_ROOT / "13_Submodel_Specifications.csv"
SELECTED_DATASET_MANIFEST_FILE = OUTPUT_ROOT / "14_Selected_Dataset_Manifest.csv"
OUTPUT_MANIFEST_FILE = OUTPUT_ROOT / "15_Output_Manifest.csv"
SOFTWARE_FILE = OUTPUT_ROOT / "Software_Environment.json"
LOG_FILE = OUTPUT_ROOT / "sem_indicator_screening.log"


# =============================================================================
# Utility functions
# =============================================================================

def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
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


def write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)


def parse_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def add_audit(
    rows: List[Dict[str, object]],
    category: str,
    check: str,
    passed: bool,
    observed: object,
    expected: object,
    severity: str = "ERROR",
    detail: str = "",
) -> None:
    rows.append(
        {
            "Category": category,
            "Check": check,
            "Severity": severity,
            "Passed": bool(passed),
            "Observed": observed,
            "Expected": expected,
            "Detail": detail,
        }
    )


def manifest_row(
    manifest: pd.DataFrame,
    relative_path: str,
) -> pd.Series:
    matches = manifest.loc[
        manifest["Relative_Path"].astype(str).eq(relative_path)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one Step 01 manifest row for {relative_path}, "
            f"found {len(matches)}."
        )
    return matches.iloc[0]


def domain_counts(variables: Sequence[str]) -> Dict[str, int]:
    result = {domain: 0 for domain in VARIABLE_GROUPS}
    for variable in variables:
        result[DOMAIN_OF[variable]] += 1
    return result


def can_drop(
    variables: Sequence[str],
    variable: str,
) -> bool:
    counts = domain_counts(variables)
    domain = DOMAIN_OF[variable]
    return counts[domain] - 1 >= MIN_DOMAIN_COUNTS[domain]


def calculate_vif(
    data: pd.DataFrame,
    variables: Sequence[str],
) -> Tuple[pd.DataFrame, float, int]:
    if len(variables) < 2:
        raise ValueError("At least two variables are required for VIF.")

    matrix = data[list(variables)].to_numpy(dtype=float)
    correlation = np.corrcoef(matrix, rowvar=False)
    rank = int(np.linalg.matrix_rank(correlation))
    condition_number = float(np.linalg.cond(correlation))

    rows: List[Dict[str, object]] = []
    for index, variable in enumerate(variables):
        others = [j for j in range(len(variables)) if j != index]
        r = correlation[index, others]
        r_others = correlation[np.ix_(others, others)]
        beta = np.linalg.pinv(r_others, rcond=1e-12) @ r
        r_squared = float(r @ beta)
        r_squared = min(max(r_squared, 0.0), 1.0)
        vif = (
            float("inf")
            if 1.0 - r_squared <= 1e-12
            else float(1.0 / (1.0 - r_squared))
        )
        rows.append(
            {
                "Variable": variable,
                "Domain": DOMAIN_OF[variable],
                "VIF": vif,
                "VIF_ge_5": bool(vif >= TARGET_MAX_VIF),
                "VIF_ge_7p5": bool(vif >= HARD_MAX_VIF),
                "Correlation_Matrix_Rank": rank,
                "Variable_N": len(variables),
                "Condition_Number": condition_number,
                "Complete_Case_N": len(data),
            }
        )

    return pd.DataFrame(rows), condition_number, rank


def calculate_spearman(
    data: pd.DataFrame,
    variables: Sequence[str],
) -> pd.DataFrame:
    return data[list(variables)].corr(method="spearman")


def maximum_pair(
    correlation: pd.DataFrame,
) -> Tuple[str, str, float]:
    variables = list(correlation.columns)
    best_variable_1 = ""
    best_variable_2 = ""
    best_value = -1.0
    for i, variable_1 in enumerate(variables):
        for j in range(i + 1, len(variables)):
            variable_2 = variables[j]
            value = abs(float(correlation.loc[variable_1, variable_2]))
            if value > best_value:
                best_variable_1 = variable_1
                best_variable_2 = variable_2
                best_value = value
    return best_variable_1, best_variable_2, best_value


def evaluate_one_season(
    data: pd.DataFrame,
    variables: Sequence[str],
) -> Dict[str, object]:
    vif_table, condition_number, rank = calculate_vif(data, variables)
    spearman = calculate_spearman(data, variables)
    pair_1, pair_2, max_pair_abs = maximum_pair(spearman)
    max_vif = float(vif_table["VIF"].max())
    max_vif_variable = str(
        vif_table.sort_values(
            ["VIF", "Variable"],
            ascending=[False, True],
        ).iloc[0]["Variable"]
    )
    return {
        "VIF_Table": vif_table,
        "Spearman": spearman,
        "Maximum_VIF": max_vif,
        "Maximum_VIF_Variable": max_vif_variable,
        "Condition_Number": condition_number,
        "Rank": rank,
        "Maximum_Absolute_Spearman": max_pair_abs,
        "Maximum_Correlation_Variable_1": pair_1,
        "Maximum_Correlation_Variable_2": pair_2,
    }


def set_passes(
    evaluations: Mapping[int, Dict[str, object]],
    vif_limit: float = TARGET_MAX_VIF,
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    for season, result in evaluations.items():
        if result["Maximum_VIF"] > vif_limit + FLOAT_ATOL:
            reasons.append(
                f"{EXPECTED_SEASONS[season]} max VIF "
                f"{result['Maximum_VIF']:.6f} > {vif_limit:.1f}"
            )
        if (
            result["Maximum_Absolute_Spearman"]
            > PAIRWISE_ABS_SPEARMAN_LIMIT + FLOAT_ATOL
        ):
            reasons.append(
                f"{EXPECTED_SEASONS[season]} max |Spearman| "
                f"{result['Maximum_Absolute_Spearman']:.6f} > "
                f"{PAIRWISE_ABS_SPEARMAN_LIMIT:.2f}"
            )
        if (
            result["Condition_Number"]
            > TARGET_CONDITION_NUMBER + FLOAT_ATOL
        ):
            reasons.append(
                f"{EXPECTED_SEASONS[season]} condition number "
                f"{result['Condition_Number']:.6f} > "
                f"{TARGET_CONDITION_NUMBER:.1f}"
            )
    return len(reasons) == 0, reasons


def evaluate_across_seasons(
    season_data: Mapping[int, pd.DataFrame],
    variables: Sequence[str],
) -> Dict[int, Dict[str, object]]:
    return {
        season: evaluate_one_season(data, variables)
        for season, data in season_data.items()
    }


def variable_worst_vif(
    evaluations: Mapping[int, Dict[str, object]],
    variable: str,
) -> float:
    values = []
    for result in evaluations.values():
        table = result["VIF_Table"]
        value = float(
            table.loc[table["Variable"].eq(variable), "VIF"].iloc[0]
        )
        values.append(value)
    return max(values)


def variable_mean_abs_correlation(
    season_data: Mapping[int, pd.DataFrame],
    variables: Sequence[str],
    variable: str,
) -> float:
    values: List[float] = []
    for data in season_data.values():
        corr = calculate_spearman(data, variables)
        others = [
            other for other in variables if other != variable
        ]
        if others:
            values.extend(
                np.abs(corr.loc[variable, others].to_numpy(dtype=float)).tolist()
            )
    return float(np.mean(values)) if values else 0.0


def select_drop_candidate(
    variables: Sequence[str],
    season_data: Mapping[int, pd.DataFrame],
    evaluations: Mapping[int, Dict[str, object]],
) -> Tuple[str, str]:
    # Prefer variables participating in the worst pairwise-correlation
    # violation. If no pairwise violation remains, use the worst-VIF variable.
    pair_candidates: List[str] = []
    pair_details: List[Tuple[float, int, str, str]] = []

    for season, result in evaluations.items():
        pair_abs = float(result["Maximum_Absolute_Spearman"])
        if pair_abs > PAIRWISE_ABS_SPEARMAN_LIMIT + FLOAT_ATOL:
            variable_1 = str(
                result["Maximum_Correlation_Variable_1"]
            )
            variable_2 = str(
                result["Maximum_Correlation_Variable_2"]
            )
            pair_candidates.extend([variable_1, variable_2])
            pair_details.append(
                (
                    pair_abs,
                    season,
                    variable_1,
                    variable_2,
                )
            )

    candidates = sorted(set(pair_candidates))
    reason = ""

    if candidates:
        worst_pair = sorted(
            pair_details,
            key=lambda item: (
                -item[0],
                item[1],
                item[2],
                item[3],
            ),
        )[0]
        reason = (
            f"Worst pairwise redundancy in "
            f"{EXPECTED_SEASONS[worst_pair[1]]}: "
            f"{worst_pair[2]} vs {worst_pair[3]}, "
            f"|rho|={worst_pair[0]:.6f}."
        )
    else:
        max_vif = max(
            float(result["Maximum_VIF"])
            for result in evaluations.values()
        )
        max_condition = max(
            float(result["Condition_Number"])
            for result in evaluations.values()
        )
        if max_vif > TARGET_MAX_VIF + FLOAT_ATOL:
            candidates = sorted(
                set(
                    str(result["Maximum_VIF_Variable"])
                    for result in evaluations.values()
                    if float(result["Maximum_VIF"])
                    >= max_vif - FLOAT_ATOL
                )
            )
            reason = (
                f"Worst-season maximum VIF={max_vif:.6f}."
            )
        elif max_condition > TARGET_CONDITION_NUMBER + FLOAT_ATOL:
            candidates = list(variables)
            reason = (
                f"Worst-season condition number={max_condition:.6f}."
            )
        else:
            raise RuntimeError(
                "select_drop_candidate called without a violation."
            )

    droppable = [
        variable for variable in candidates
        if can_drop(variables, variable)
    ]
    if not droppable:
        droppable = [
            variable for variable in variables
            if can_drop(variables, variable)
        ]
    if not droppable:
        raise RuntimeError(
            "No variable can be dropped without violating minimum "
            "domain coverage."
        )

    # Drop lower-priority, more redundant, higher-VIF variables first.
    scored = []
    for variable in droppable:
        scored.append(
            (
                RETENTION_PRIORITY[variable],
                -variable_worst_vif(evaluations, variable),
                -variable_mean_abs_correlation(
                    season_data,
                    variables,
                    variable,
                ),
                variable,
            )
        )
    scored.sort()
    selected = scored[0][3]
    return selected, reason


def repair_core_set(
    season_data: Mapping[int, pd.DataFrame],
    starting_variables: Sequence[str],
    scope: str,
) -> Tuple[List[str], List[Dict[str, object]]]:
    variables = list(starting_variables)
    history: List[Dict[str, object]] = []
    iteration = 0

    while True:
        iteration += 1
        evaluations = evaluate_across_seasons(
            season_data,
            variables,
        )
        passed, reasons = set_passes(evaluations)
        history.append(
            {
                "Scope": scope,
                "Iteration": iteration,
                "Action": "Evaluate",
                "Variable": "",
                "Accepted": passed,
                "Selected_Variable_N": len(variables),
                "Selected_Variables": ";".join(variables),
                "Worst_Max_VIF": max(
                    float(result["Maximum_VIF"])
                    for result in evaluations.values()
                ),
                "Worst_Max_Absolute_Spearman": max(
                    float(result["Maximum_Absolute_Spearman"])
                    for result in evaluations.values()
                ),
                "Worst_Condition_Number": max(
                    float(result["Condition_Number"])
                    for result in evaluations.values()
                ),
                "Reason": "Pass" if passed else " | ".join(reasons),
            }
        )
        if passed:
            return variables, history

        variable, reason = select_drop_candidate(
            variables,
            season_data,
            evaluations,
        )
        variables.remove(variable)
        history.append(
            {
                "Scope": scope,
                "Iteration": iteration,
                "Action": "Drop_from_core",
                "Variable": variable,
                "Accepted": True,
                "Selected_Variable_N": len(variables),
                "Selected_Variables": ";".join(variables),
                "Worst_Max_VIF": np.nan,
                "Worst_Max_Absolute_Spearman": np.nan,
                "Worst_Condition_Number": np.nan,
                "Reason": reason,
            }
        )


def forward_add_optional(
    season_data: Mapping[int, pd.DataFrame],
    core_variables: Sequence[str],
    optional_variables: Sequence[str],
    scope: str,
) -> Tuple[List[str], List[Dict[str, object]]]:
    selected = list(core_variables)
    history: List[Dict[str, object]] = []

    for order, variable in enumerate(optional_variables, start=1):
        candidate = selected + [variable]
        evaluations = evaluate_across_seasons(
            season_data,
            candidate,
        )
        passed, reasons = set_passes(evaluations)
        if passed:
            selected.append(variable)

        history.append(
            {
                "Scope": scope,
                "Iteration": order,
                "Action": "Try_optional_addition",
                "Variable": variable,
                "Accepted": passed,
                "Selected_Variable_N": len(selected),
                "Selected_Variables": ";".join(selected),
                "Worst_Max_VIF": max(
                    float(result["Maximum_VIF"])
                    for result in evaluations.values()
                ),
                "Worst_Max_Absolute_Spearman": max(
                    float(result["Maximum_Absolute_Spearman"])
                    for result in evaluations.values()
                ),
                "Worst_Condition_Number": max(
                    float(result["Condition_Number"])
                    for result in evaluations.values()
                ),
                "Reason": (
                    "Accepted: all screening thresholds remained satisfied."
                    if passed
                    else "Rejected: " + " | ".join(reasons)
                ),
            }
        )

    return selected, history


def season_only_mapping(
    season: int,
    data: pd.DataFrame,
) -> Dict[int, pd.DataFrame]:
    return {season: data}


def selected_dataset_name(
    season: int,
    response: str,
) -> str:
    return (
        f"{season:02d}_{EXPECTED_SEASONS[season]}_"
        f"{response}_Locked_Selected_SEM.csv.gz"
    )


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    SELECTED_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text("", encoding="utf-8")

    log("Starting SEM indicator screening and model-form locking.")
    log(f"Code version: {CODE_VERSION}")

    required_files = [
        STEP01_SCRIPT,
        STEP01_METHOD,
        STEP01_AUDIT,
        STEP01_MASTER_RAW,
        STEP01_MASTER_STD,
        STEP01_DATASET_MANIFEST,
        STEP01_VARIABLE_DICTIONARY,
        STEP01_STANDARDIZATION,
        STEP01_PANEL_QA,
        STEP01_OUTPUT_MANIFEST,
    ]
    missing_files = [
        str(path) for path in required_files if not path.exists()
    ]
    if missing_files:
        raise FileNotFoundError(
            "Required Step 01 files are missing:\n"
            + "\n".join(missing_files)
        )

    audits: List[Dict[str, object]] = []

    # -------------------------------------------------------------------------
    # Verify sealed Step 01
    # -------------------------------------------------------------------------
    step01_method = read_json(STEP01_METHOD)
    step01_version = step01_method.get("Code_Version")
    add_audit(
        audits,
        "Version",
        "Step01_code_version",
        step01_version == EXPECTED_STEP01_VERSION,
        step01_version,
        EXPECTED_STEP01_VERSION,
    )

    step01_script_hash = sha256_file(STEP01_SCRIPT)
    add_audit(
        audits,
        "Hash",
        "Step01_script_SHA256",
        step01_script_hash == EXPECTED_STEP01_SCRIPT_SHA256,
        step01_script_hash,
        EXPECTED_STEP01_SCRIPT_SHA256,
    )

    step01_audit = pd.read_csv(STEP01_AUDIT)
    step01_error_rows = step01_audit.loc[
        step01_audit["Severity"].astype(str).eq("ERROR")
    ]
    step01_warning_rows = step01_audit.loc[
        step01_audit["Severity"].astype(str).eq("WARNING")
    ]
    add_audit(
        audits,
        "Step01",
        "All_Step01_ERROR_checks_passed",
        bool(step01_error_rows["Passed"].map(parse_bool).all()),
        int(step01_error_rows["Passed"].map(parse_bool).sum()),
        len(step01_error_rows),
    )
    add_audit(
        audits,
        "Step01",
        "All_Step01_WARNING_checks_passed",
        bool(step01_warning_rows["Passed"].map(parse_bool).all()),
        int(step01_warning_rows["Passed"].map(parse_bool).sum()),
        len(step01_warning_rows),
    )

    step01_manifest = pd.read_csv(STEP01_OUTPUT_MANIFEST)
    for path in [
        STEP01_MASTER_RAW,
        STEP01_MASTER_STD,
        STEP01_DATASET_MANIFEST,
        STEP01_VARIABLE_DICTIONARY,
        STEP01_STANDARDIZATION,
        STEP01_PANEL_QA,
    ]:
        relative_path = str(path.relative_to(STEP01_ROOT))
        row = manifest_row(step01_manifest, relative_path)
        actual_hash = sha256_file(path)
        actual_size = path.stat().st_size
        add_audit(
            audits,
            "Hash",
            f"Step01_manifest_{relative_path}",
            (
                actual_hash == str(row["SHA256"])
                and int(actual_size) == int(row["File_Size_Bytes"])
            ),
            f"{actual_hash}; {actual_size}",
            f"{row['SHA256']}; {int(row['File_Size_Bytes'])}",
        )

    if not all(
        row["Passed"]
        for row in audits
        if row["Severity"] == "ERROR"
    ):
        pd.DataFrame(audits).to_csv(
            INPUT_AUDIT_FILE,
            index=False,
            encoding="utf-8-sig",
        )
        raise RuntimeError(
            "Step 01 integrity checks failed. "
            "See 01_Input_Integrity_Audit.csv."
        )

    # -------------------------------------------------------------------------
    # Read locked standardized master
    # -------------------------------------------------------------------------
    required_columns = [
        "GRID_UID",
        "Country",
        "GRID_ID",
        "Year",
        "Year_Z",
        "Season",
        "Season_Label",
        "Temporal_Fold",
        "Spatial_Block",
        "ForestPixelCount",
        "Log_ForestPixelCount",
        "Fire_Count",
        "FCD",
        "Fire_Occurrence",
        "Burned_Pixel_Count",
        "BAD",
        "Burned_Occurrence",
        "SEM_Complete_All_Candidates",
        *ALL_PREDICTORS,
        *[f"Z_{variable}" for variable in ALL_PREDICTORS],
    ]

    log("Reading the sealed standardized SEM master.")
    standardized = pd.read_csv(
        STEP01_MASTER_STD,
        usecols=required_columns,
        low_memory=False,
    )

    add_audit(
        audits,
        "Universe",
        "Locked_panel_rows",
        len(standardized) == EXPECTED_PANEL_N,
        len(standardized),
        EXPECTED_PANEL_N,
    )
    add_audit(
        audits,
        "Universe",
        "GRID_UID_count",
        standardized["GRID_UID"].nunique() == EXPECTED_GRID_N,
        int(standardized["GRID_UID"].nunique()),
        EXPECTED_GRID_N,
    )
    add_audit(
        audits,
        "Universe",
        "All_candidate_rows_complete",
        bool(
            (
                standardized["SEM_Complete_All_Candidates"]
                == 1
            ).all()
        ),
        int(
            standardized[
                "SEM_Complete_All_Candidates"
            ].sum()
        ),
        EXPECTED_PANEL_N,
    )
    add_audit(
        audits,
        "Schema",
        "All_predictors_available",
        all(
            variable in standardized.columns
            and f"Z_{variable}" in standardized.columns
            for variable in ALL_PREDICTORS
        ),
        len(
            [
                variable
                for variable in ALL_PREDICTORS
                if variable in standardized.columns
                and f"Z_{variable}" in standardized.columns
            ]
        ),
        len(ALL_PREDICTORS),
    )

    season_data = {
        season: standardized.loc[
            standardized["Season"].astype(int).eq(season),
            ALL_PREDICTORS,
        ].copy()
        for season in EXPECTED_SEASONS
    }
    for season, data in season_data.items():
        add_audit(
            audits,
            "Universe",
            f"{EXPECTED_SEASONS[season]}_row_count",
            len(data) == EXPECTED_SEASON_RESPONSE_N,
            len(data),
            EXPECTED_SEASON_RESPONSE_N,
        )
        add_audit(
            audits,
            "Universe",
            f"{EXPECTED_SEASONS[season]}_finite_predictors",
            bool(np.isfinite(data.to_numpy(dtype=float)).all()),
            int(
                np.isfinite(
                    data.to_numpy(dtype=float)
                ).sum()
            ),
            int(data.size),
        )

    # -------------------------------------------------------------------------
    # Structural rules
    # -------------------------------------------------------------------------
    log("Writing transparent structural redundancy rules.")
    rule_rows: List[Dict[str, object]] = []
    for variable in ALL_PREDICTORS:
        if variable in STRUCTURAL_EXCLUSIONS:
            status = "Excluded_Primary_Retained_Sensitivity"
            rationale = STRUCTURAL_EXCLUSIONS[variable]
        elif variable in CORE_ANCHORS:
            status = "Core_Anchor"
            rationale = (
                "Direct, interpretable fire-triangle indicator included "
                "before data-driven redundancy checks."
            )
        else:
            status = "Optional_Candidate"
            rationale = (
                "Eligible for outcome-neutral forward addition if the "
                "worst-season correlation, VIF, and condition-number "
                "thresholds remain satisfied."
            )

        rule_rows.append(
            {
                "Variable": variable,
                "Domain": DOMAIN_OF[variable],
                "Rule_Status": status,
                "Retention_Priority": RETENTION_PRIORITY[variable],
                "Primary_SEM_Eligible": (
                    variable not in STRUCTURAL_EXCLUSIONS
                ),
                "Sensitivity_Analysis_Available": True,
                "Rationale": rationale,
            }
        )
    rules_table = pd.DataFrame(rule_rows)
    rules_table.to_csv(
        RULES_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    eligible_variables = [
        variable
        for variable in ALL_PREDICTORS
        if variable not in STRUCTURAL_EXCLUSIONS
    ]
    add_audit(
        audits,
        "Rules",
        "Structural_exclusions_are_BUI_and_FWI_only",
        set(STRUCTURAL_EXCLUSIONS) == {"BUI", "FWI"},
        sorted(STRUCTURAL_EXCLUSIONS),
        ["BUI", "FWI"],
    )
    add_audit(
        audits,
        "Rules",
        "All_core_anchors_eligible",
        set(CORE_ANCHORS).issubset(eligible_variables),
        len(set(CORE_ANCHORS) & set(eligible_variables)),
        len(CORE_ANCHORS),
    )

    # -------------------------------------------------------------------------
    # Full eligible-set correlation and VIF diagnostics
    # -------------------------------------------------------------------------
    log("Calculating full eligible-set correlations and VIF diagnostics.")
    correlation_rows: List[Dict[str, object]] = []
    pair_worst: Dict[Tuple[str, str], Dict[str, object]] = {}
    initial_vif_frames: List[pd.DataFrame] = []

    for season, data in season_data.items():
        spearman = calculate_spearman(data, eligible_variables)
        vif_table, condition_number, rank = calculate_vif(
            data,
            eligible_variables,
        )
        vif_table.insert(0, "Season", season)
        vif_table.insert(
            1,
            "Season_Label",
            EXPECTED_SEASONS[season],
        )
        vif_table["Indicator_Set"] = "Initial_Eligible_Set"
        initial_vif_frames.append(vif_table)

        for i, variable_1 in enumerate(eligible_variables):
            for j, variable_2 in enumerate(eligible_variables):
                rho = float(
                    spearman.loc[variable_1, variable_2]
                )
                correlation_rows.append(
                    {
                        "Season": season,
                        "Season_Label": EXPECTED_SEASONS[
                            season
                        ],
                        "Variable_1": variable_1,
                        "Domain_1": DOMAIN_OF[variable_1],
                        "Variable_2": variable_2,
                        "Domain_2": DOMAIN_OF[variable_2],
                        "Spearman_Rho": rho,
                        "Absolute_Rho": abs(rho),
                        "Same_Domain": (
                            DOMAIN_OF[variable_1]
                            == DOMAIN_OF[variable_2]
                        ),
                    }
                )

                if j > i:
                    key = tuple(sorted([variable_1, variable_2]))
                    current = pair_worst.get(key)
                    if (
                        current is None
                        or abs(rho) > current["Worst_Absolute_Rho"]
                    ):
                        pair_worst[key] = {
                            "Variable_1": key[0],
                            "Domain_1": DOMAIN_OF[key[0]],
                            "Variable_2": key[1],
                            "Domain_2": DOMAIN_OF[key[1]],
                            "Worst_Season": season,
                            "Worst_Season_Label": (
                                EXPECTED_SEASONS[season]
                            ),
                            "Worst_Spearman_Rho": rho,
                            "Worst_Absolute_Rho": abs(rho),
                            "Exceeds_0p80": bool(
                                abs(rho)
                                > PAIRWISE_ABS_SPEARMAN_LIMIT
                            ),
                            "Same_Domain": (
                                DOMAIN_OF[key[0]]
                                == DOMAIN_OF[key[1]]
                            ),
                        }

    pd.DataFrame(correlation_rows).to_csv(
        SEASONAL_CORRELATION_FILE,
        index=False,
        compression="gzip",
        encoding="utf-8-sig",
    )
    worst_correlation_table = pd.DataFrame(
        pair_worst.values()
    ).sort_values(
        [
            "Worst_Absolute_Rho",
            "Variable_1",
            "Variable_2",
        ],
        ascending=[False, True, True],
    )
    worst_correlation_table.to_csv(
        WORST_CORRELATION_FILE,
        index=False,
        encoding="utf-8-sig",
    )
    initial_vif_table = pd.concat(
        initial_vif_frames,
        ignore_index=True,
    )
    initial_vif_table.to_csv(
        INITIAL_VIF_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # Season-specific screening
    # -------------------------------------------------------------------------
    log("Running outcome-neutral season-specific indicator screening.")
    seasonal_history_rows: List[Dict[str, object]] = []
    seasonal_selection_rows: List[Dict[str, object]] = []
    seasonal_selected_sets: Dict[int, List[str]] = {}

    for season, data in season_data.items():
        scope = f"Season_{season}_{EXPECTED_SEASONS[season]}"
        repaired_core, core_history = repair_core_set(
            season_only_mapping(season, data),
            CORE_ANCHORS,
            scope,
        )
        selected, addition_history = forward_add_optional(
            season_only_mapping(season, data),
            repaired_core,
            OPTIONAL_CANDIDATES,
            scope,
        )
        seasonal_history_rows.extend(core_history)
        seasonal_history_rows.extend(addition_history)
        seasonal_selected_sets[season] = selected

        final_evaluation = evaluate_one_season(
            data,
            selected,
        )
        for variable in ALL_PREDICTORS:
            seasonal_selection_rows.append(
                {
                    "Season": season,
                    "Season_Label": EXPECTED_SEASONS[
                        season
                    ],
                    "Variable": variable,
                    "Domain": DOMAIN_OF[variable],
                    "Selected": variable in selected,
                    "Structural_Exclusion": (
                        variable in STRUCTURAL_EXCLUSIONS
                    ),
                    "Core_Anchor": variable in CORE_ANCHORS,
                    "Retention_Priority": (
                        RETENTION_PRIORITY[variable]
                    ),
                    "Final_Set_Size": len(selected),
                    "Final_Max_VIF": (
                        final_evaluation["Maximum_VIF"]
                    ),
                    "Final_Max_Absolute_Spearman": (
                        final_evaluation[
                            "Maximum_Absolute_Spearman"
                        ]
                    ),
                    "Final_Condition_Number": (
                        final_evaluation[
                            "Condition_Number"
                        ]
                    ),
                }
            )

    seasonal_history_table = pd.DataFrame(
        seasonal_history_rows
    )
    seasonal_history_table.to_csv(
        SEASONAL_HISTORY_FILE,
        index=False,
        encoding="utf-8-sig",
    )
    seasonal_selection_table = pd.DataFrame(
        seasonal_selection_rows
    )
    seasonal_selection_table.to_csv(
        SEASONAL_SELECTION_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # Common cross-season indicator set
    # -------------------------------------------------------------------------
    log("Locking a common cross-season indicator set.")
    repaired_common_core, common_core_history = repair_core_set(
        season_data,
        CORE_ANCHORS,
        "Common_Cross_Season",
    )
    common_selected, common_add_history = forward_add_optional(
        season_data,
        repaired_common_core,
        OPTIONAL_CANDIDATES,
        "Common_Cross_Season",
    )
    common_history_table = pd.DataFrame(
        common_core_history + common_add_history
    )
    common_history_table.to_csv(
        COMMON_HISTORY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    common_evaluations = evaluate_across_seasons(
        season_data,
        common_selected,
    )
    common_passed, common_reasons = set_passes(
        common_evaluations,
    )

    add_audit(
        audits,
        "Common_Set",
        "Common_set_meets_target_thresholds",
        common_passed,
        " | ".join(common_reasons) if common_reasons else "Pass",
        "Max VIF <=5, max |rho| <=0.80, condition number <=100",
    )
    common_domain_counts = domain_counts(common_selected)
    for domain, minimum in MIN_DOMAIN_COUNTS.items():
        add_audit(
            audits,
            "Common_Set",
            f"{domain}_minimum_coverage",
            common_domain_counts[domain] >= minimum,
            common_domain_counts[domain],
            f">={minimum}",
        )

    common_rows: List[Dict[str, object]] = []
    selection_count = {
        variable: sum(
            variable in selected
            for selected in seasonal_selected_sets.values()
        )
        for variable in ALL_PREDICTORS
    }

    for variable in ALL_PREDICTORS:
        selected = variable in common_selected
        if variable in STRUCTURAL_EXCLUSIONS:
            final_status = "Sensitivity_Only_Structural_Composite"
            exclusion_reason = STRUCTURAL_EXCLUSIONS[variable]
        elif selected:
            final_status = "Locked_Primary_Indicator"
            exclusion_reason = ""
        else:
            final_status = "Excluded_From_Primary_By_Redundancy_Screening"
            matching_history = common_history_table.loc[
                common_history_table["Variable"].astype(str).eq(
                    variable
                )
                & common_history_table["Action"].astype(str).eq(
                    "Try_optional_addition"
                )
            ]
            if not matching_history.empty:
                exclusion_reason = str(
                    matching_history.iloc[-1]["Reason"]
                )
            else:
                exclusion_reason = (
                    "Removed while repairing the minimum conceptual "
                    "core to satisfy the common cross-season thresholds."
                )

        common_rows.append(
            {
                "Variable": variable,
                "Domain": DOMAIN_OF[variable],
                "Final_Status": final_status,
                "Selected_Common_Set": selected,
                "Selected_Season_N": selection_count[variable],
                "Selected_Spring": (
                    variable in seasonal_selected_sets[1]
                ),
                "Selected_Summer": (
                    variable in seasonal_selected_sets[2]
                ),
                "Selected_Autumn": (
                    variable in seasonal_selected_sets[3]
                ),
                "Core_Anchor": variable in CORE_ANCHORS,
                "Structural_Exclusion": (
                    variable in STRUCTURAL_EXCLUSIONS
                ),
                "Retention_Priority": (
                    RETENTION_PRIORITY[variable]
                ),
                "Exclusion_or_Sensitivity_Rationale": (
                    exclusion_reason
                ),
                "Common_Set_Size": len(common_selected),
                "Common_Domain_Count": (
                    common_domain_counts[DOMAIN_OF[variable]]
                ),
            }
        )
    common_set_table = pd.DataFrame(common_rows)
    common_set_table.to_csv(
        COMMON_SET_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    common_vif_frames: List[pd.DataFrame] = []
    for season, result in common_evaluations.items():
        table = result["VIF_Table"].copy()
        table.insert(0, "Season", season)
        table.insert(
            1,
            "Season_Label",
            EXPECTED_SEASONS[season],
        )
        table["Indicator_Set"] = "Common_Locked_Set"
        table["Maximum_Absolute_Spearman"] = (
            result["Maximum_Absolute_Spearman"]
        )
        table["Maximum_Correlation_Variable_1"] = (
            result["Maximum_Correlation_Variable_1"]
        )
        table["Maximum_Correlation_Variable_2"] = (
            result["Maximum_Correlation_Variable_2"]
        )
        common_vif_frames.append(table)
    common_vif_table = pd.concat(
        common_vif_frames,
        ignore_index=True,
    )
    common_vif_table.to_csv(
        COMMON_VIF_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # Information-retention diagnostics for excluded variables
    # -------------------------------------------------------------------------
    log("Calculating information-retention diagnostics.")
    information_rows: List[Dict[str, object]] = []
    for variable in ALL_PREDICTORS:
        for season, data in season_data.items():
            retained_same_domain = [
                retained
                for retained in common_selected
                if DOMAIN_OF[retained] == DOMAIN_OF[variable]
                and retained != variable
            ]
            if variable in common_selected:
                best_proxy = variable
                best_rho = 1.0
            elif retained_same_domain:
                corr = data[
                    [variable, *retained_same_domain]
                ].corr(method="spearman")
                proxy_values = [
                    (
                        retained,
                        float(corr.loc[variable, retained]),
                    )
                    for retained in retained_same_domain
                ]
                proxy_values.sort(
                    key=lambda item: (
                        -abs(item[1]),
                        item[0],
                    )
                )
                best_proxy, best_rho = proxy_values[0]
            else:
                best_proxy = ""
                best_rho = np.nan

            information_rows.append(
                {
                    "Season": season,
                    "Season_Label": EXPECTED_SEASONS[
                        season
                    ],
                    "Variable": variable,
                    "Domain": DOMAIN_OF[variable],
                    "Selected_Common_Set": (
                        variable in common_selected
                    ),
                    "Best_Retained_Same_Domain_Proxy": best_proxy,
                    "Spearman_Rho_With_Proxy": best_rho,
                    "Absolute_Rho_With_Proxy": (
                        abs(best_rho)
                        if np.isfinite(best_rho)
                        else np.nan
                    ),
                    "At_Least_0p70_Represented": bool(
                        variable in common_selected
                        or (
                            np.isfinite(best_rho)
                            and abs(best_rho) >= 0.70
                        )
                    ),
                    "Interpretation": (
                        "This is a redundancy/information-retention "
                        "diagnostic, not evidence that the variables "
                        "are causally interchangeable."
                    ),
                }
            )
    information_table = pd.DataFrame(information_rows)
    information_table.to_csv(
        INFORMATION_RETENTION_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # Lock model form and submodel specifications
    # -------------------------------------------------------------------------
    log("Locking the two-part piecewise generalized SEM model form.")
    model_form_rows = [
        {
            "Decision": "Overall_Framework",
            "Locked_Value": "Two-part piecewise generalized SEM",
            "Rationale": (
                "The fire responses contain 87.4%-99.3% zeros, so "
                "occurrence and positive intensity must be modelled "
                "as distinct ecological processes."
            ),
        },
        {
            "Decision": "Analysis_Strata",
            "Locked_Value": (
                "Separate Spring, Summer, and Autumn models; "
                "separate Fire_Count and Burned_Pixel_Count responses"
            ),
            "Rationale": (
                "Seasonal mechanisms and the two fire metrics are "
                "not assumed to share identical paths."
            ),
        },
        {
            "Decision": "Primary_Indicator_Set",
            "Locked_Value": ";".join(common_selected),
            "Rationale": (
                "One common cross-season set is used so seasonal path "
                "coefficients remain directly comparable. Selection "
                "used no fire-response association."
            ),
        },
        {
            "Decision": "Occurrence_Response",
            "Locked_Value": "Observed_Occurrence",
            "Rationale": (
                "Binary occurrence distinguishes the ignition/"
                "realization process from positive fire magnitude."
            ),
        },
        {
            "Decision": "Occurrence_Family",
            "Locked_Value": "Binomial(logit) GLMM",
            "Rationale": (
                "The occurrence response is binary and includes the "
                "full locked grid-year-season panel."
            ),
        },
        {
            "Decision": "Positive_Intensity_Response",
            "Locked_Value": "Observed_Count restricted to >0",
            "Rationale": (
                "Positive intensity is retained on its native count "
                "scale rather than transformed into a Gaussian rate."
            ),
        },
        {
            "Decision": "Positive_Intensity_Primary_Family",
            "Locked_Value": "Zero-truncated negative binomial 2, log link",
            "Rationale": (
                "The positive response is overdispersed count data and "
                "contains no zeros by definition."
            ),
        },
        {
            "Decision": "Exposure_Offset",
            "Locked_Value": "offset(Log_ForestPixelCount)",
            "Rationale": (
                "The offset models fire activity per available forest "
                "exposure without treating the observed rate as Gaussian."
            ),
        },
        {
            "Decision": "Fixed_Adjustments",
            "Locked_Value": "Country + Year_Z",
            "Rationale": (
                "Country has only three levels and is therefore treated "
                "as a fixed adjustment. Year_Z controls monotonic trend."
            ),
        },
        {
            "Decision": "Random_Effects",
            "Locked_Value": "(1|GRID_UID) + (1|Year)",
            "Rationale": (
                "Crossed random intercepts account for repeated grid "
                "observations and shared year-specific anomalies."
            ),
        },
        {
            "Decision": "Spatial_Block_Use",
            "Locked_Value": "Not entered as a causal or random effect",
            "Rationale": (
                "Spatial_Block is a validation partition rather than "
                "an ecological mechanism."
            ),
        },
        {
            "Decision": "Country_Use",
            "Locked_Value": "Fixed adjustment, not a latent mechanism",
            "Rationale": (
                "Country captures broad management/institutional "
                "differences but three levels are insufficient for a "
                "stable random-effect variance estimate."
            ),
        },
        {
            "Decision": "Fallback_Family_Order",
            "Locked_Value": (
                "truncated_nbinom2 -> truncated_nbinom1 -> "
                "Gamma(log) on positive Observed_Rate"
            ),
            "Rationale": (
                "A fallback is permitted only after a documented "
                "convergence, singularity, or dispersion failure under "
                "the primary positive-count family."
            ),
        },
        {
            "Decision": "Indicator_Standardization",
            "Locked_Value": "Use Step 01 within-season Z_ variables",
            "Rationale": (
                "Within-season standardization supports coefficient "
                "comparison while retaining season-specific raw scales."
            ),
        },
        {
            "Decision": "Path_Selection",
            "Locked_Value": "Not performed in Step 02",
            "Rationale": (
                "Causal paths will be prespecified from the fire-triangle "
                "framework and evaluated in the next step; no p-value "
                "stepwise path search is allowed here."
            ),
        },
        {
            "Decision": "SEM_Software_Target",
            "Locked_Value": "R: glmmTMB + piecewiseSEM",
            "Rationale": (
                "These packages support the required generalized mixed "
                "submodels and piecewise d-separation tests."
            ),
        },
    ]
    model_form_table = pd.DataFrame(model_form_rows)
    model_form_table.to_csv(
        MODEL_FORM_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    panel_qa = pd.read_csv(STEP01_PANEL_QA)
    submodel_rows: List[Dict[str, object]] = []
    for _, panel_row in panel_qa.iterrows():
        season = int(panel_row["Season"])
        response = str(panel_row["Response"])
        positive_n = int(panel_row["Positive_N"])
        zero_n = int(panel_row["Zero_N"])

        response_file = (
            STEP01_DATASET_DIR
            / (
                f"{season:02d}_{EXPECTED_SEASONS[season]}_"
                f"{response}_SEM.csv.gz"
            )
        )

        for component in ["Occurrence", "Positive_Intensity"]:
            if component == "Occurrence":
                sample_rule = "All locked rows"
                n_used = int(panel_row["Total_N"])
                response_field = "Observed_Occurrence"
                family = "binomial(link='logit')"
                offset = "None"
                formula_template = (
                    "Observed_Occurrence ~ <prespecified_paths> + "
                    "Country + Year_Z + (1|GRID_UID) + (1|Year)"
                )
                fallback = "None"
            else:
                sample_rule = "Observed_Count > 0 only"
                n_used = positive_n
                response_field = "Observed_Count"
                family = "truncated_nbinom2(link='log')"
                offset = "offset(Log_ForestPixelCount)"
                formula_template = (
                    "Observed_Count ~ <prespecified_paths> + Country + "
                    "Year_Z + offset(Log_ForestPixelCount) + "
                    "(1|GRID_UID) + (1|Year)"
                )
                fallback = (
                    "truncated_nbinom1(link='log'); if still invalid, "
                    "Gamma(link='log') for positive Observed_Rate"
                )

            submodel_rows.append(
                {
                    "Season": season,
                    "Season_Label": EXPECTED_SEASONS[
                        season
                    ],
                    "Response": response,
                    "Component": component,
                    "Source_Dataset": str(
                        response_file.relative_to(STEP01_ROOT)
                    ),
                    "Sample_Rule": sample_rule,
                    "N_Available": n_used,
                    "Positive_N": positive_n,
                    "Zero_N": zero_n,
                    "Response_Field": response_field,
                    "Primary_Family": family,
                    "Link": (
                        "logit"
                        if component == "Occurrence"
                        else "log"
                    ),
                    "Offset": offset,
                    "Fixed_Adjustments": "Country + Year_Z",
                    "Random_Effects": (
                        "(1|GRID_UID) + (1|Year)"
                    ),
                    "Formula_Template": formula_template,
                    "Permitted_Fallback": fallback,
                    "Common_Indicator_N": len(common_selected),
                    "Common_Indicators_Z": ";".join(
                        f"Z_{variable}"
                        for variable in common_selected
                    ),
                    "SEM_Fitted_in_Step02": False,
                }
            )
    submodel_table = pd.DataFrame(submodel_rows)
    submodel_table.to_csv(
        SUBMODEL_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    add_audit(
        audits,
        "Model_Form",
        "Twelve_response_component_specifications",
        len(submodel_table) == 12,
        len(submodel_table),
        12,
    )
    add_audit(
        audits,
        "Model_Form",
        "All_positive_components_have_positive_samples",
        bool(
            (
                submodel_table.loc[
                    submodel_table["Component"].eq(
                        "Positive_Intensity"
                    ),
                    "N_Available",
                ]
                > 0
            ).all()
        ),
        int(
            submodel_table.loc[
                submodel_table["Component"].eq(
                    "Positive_Intensity"
                ),
                "N_Available",
            ].min()
        ),
        ">0",
    )

    # -------------------------------------------------------------------------
    # Generate reduced locked datasets
    # -------------------------------------------------------------------------
    log("Writing six reduced datasets with the common locked indicators.")
    source_dataset_manifest = pd.read_csv(
        STEP01_DATASET_MANIFEST
    )
    selected_manifest_rows: List[Dict[str, object]] = []

    identifier_columns = [
        "GRID_UID",
        "Country",
        "GRID_ID",
        "Year",
        "Year_Centered_2013",
        "Year_Z",
        "Season",
        "Season_Label",
        "Temporal_Fold",
        "Spatial_Block",
        "Response_Name",
        "Rate_Name",
        "ForestPixelCount",
        "Log_ForestPixelCount",
        "SEM_Complete_All_Candidates",
        "Observed_Count",
        "Observed_Rate",
        "Observed_Occurrence",
    ]
    selected_columns = [
        *identifier_columns,
        *common_selected,
        *[f"Z_{variable}" for variable in common_selected],
    ]

    for _, row in source_dataset_manifest.iterrows():
        season = int(row["Season"])
        response = str(row["Response"])
        source_path = (
            STEP01_ROOT / str(row["Relative_Path"])
        )
        source_hash = sha256_file(source_path)
        add_audit(
            audits,
            "Selected_Datasets",
            f"Source_hash_{season}_{response}",
            source_hash == str(row["SHA256"]),
            source_hash,
            str(row["SHA256"]),
        )

        source_data = pd.read_csv(
            source_path,
            usecols=selected_columns,
            low_memory=False,
        )
        output_path = (
            SELECTED_DATASET_DIR
            / selected_dataset_name(season, response)
        )
        source_data.to_csv(
            output_path,
            index=False,
            compression="gzip",
            encoding="utf-8-sig",
        )

        selected_manifest_rows.append(
            {
                "Season": season,
                "Season_Label": EXPECTED_SEASONS[
                    season
                ],
                "Response": response,
                "Source_Relative_Path": str(
                    source_path.relative_to(STEP01_ROOT)
                ),
                "Source_SHA256": source_hash,
                "Output_Relative_Path": str(
                    output_path.relative_to(OUTPUT_ROOT)
                ),
                "Rows": len(source_data),
                "Columns": len(source_data.columns),
                "GRID_UID_Count": int(
                    source_data["GRID_UID"].nunique()
                ),
                "Year_Count": int(
                    source_data["Year"].nunique()
                ),
                "Positive_N": int(
                    source_data[
                        "Observed_Occurrence"
                    ].sum()
                ),
                "Zero_N": int(
                    len(source_data)
                    - source_data[
                        "Observed_Occurrence"
                    ].sum()
                ),
                "Common_Indicator_N": len(common_selected),
                "Output_File_Size_Bytes": (
                    output_path.stat().st_size
                ),
                "Output_SHA256": sha256_file(output_path),
            }
        )

    selected_manifest = pd.DataFrame(
        selected_manifest_rows
    )
    selected_manifest.to_csv(
        SELECTED_DATASET_MANIFEST_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    add_audit(
        audits,
        "Selected_Datasets",
        "Six_selected_datasets",
        len(selected_manifest) == 6,
        len(selected_manifest),
        6,
    )
    add_audit(
        audits,
        "Selected_Datasets",
        "All_selected_datasets_have_locked_rows",
        bool(
            (
                selected_manifest["Rows"]
                == EXPECTED_SEASON_RESPONSE_N
            ).all()
        ),
        sorted(
            selected_manifest["Rows"].unique().tolist()
        ),
        [EXPECTED_SEASON_RESPONSE_N],
    )
    add_audit(
        audits,
        "Selected_Datasets",
        "All_selected_datasets_have_4982_grids",
        bool(
            (
                selected_manifest["GRID_UID_Count"]
                == EXPECTED_GRID_N
            ).all()
        ),
        sorted(
            selected_manifest[
                "GRID_UID_Count"
            ].unique().tolist()
        ),
        [EXPECTED_GRID_N],
    )

    # -------------------------------------------------------------------------
    # Save method definition, software, audit, and manifest
    # -------------------------------------------------------------------------
    method_definition = {
        "Code_Version": CODE_VERSION,
        "Created": datetime.now().isoformat(
            timespec="seconds"
        ),
        "Step01_Code_Version": step01_version,
        "Step01_Script_SHA256": step01_script_hash,
        "Step01_Master_Standardized_SHA256": sha256_file(
            STEP01_MASTER_STD
        ),
        "Selection_Principle": (
            "Outcome-neutral screening. No Fire Count, Burned Pixel "
            "Count, occurrence, rate, regression coefficient, RF "
            "importance, SHAP value, or p value is used to choose "
            "indicators."
        ),
        "Structural_Exclusions": STRUCTURAL_EXCLUSIONS,
        "Core_Anchors_Initial": CORE_ANCHORS,
        "Optional_Candidate_Order": OPTIONAL_CANDIDATES,
        "Retention_Priority": RETENTION_PRIORITY,
        "Thresholds": {
            "Maximum_Absolute_Spearman": (
                PAIRWISE_ABS_SPEARMAN_LIMIT
            ),
            "Target_Maximum_VIF": TARGET_MAX_VIF,
            "Hard_Maximum_VIF": HARD_MAX_VIF,
            "Maximum_Condition_Number": (
                TARGET_CONDITION_NUMBER
            ),
        },
        "Minimum_Domain_Coverage": MIN_DOMAIN_COUNTS,
        "Common_Locked_Indicators": common_selected,
        "Common_Locked_Indicator_N": len(
            common_selected
        ),
        "Common_Domain_Counts": common_domain_counts,
        "Season_Specific_Selected_Indicators": {
            EXPECTED_SEASONS[season]: variables
            for season, variables
            in seasonal_selected_sets.items()
        },
        "Final_Analysis_Uses_Common_Set": True,
        "Reason_For_Common_Set": (
            "A common set is required for direct seasonal comparison "
            "of standardized path coefficients. Season-specific sets "
            "are retained only as a robustness diagnostic."
        ),
        "Locked_Model_Form": {
            "Framework": (
                "Two-part piecewise generalized SEM"
            ),
            "Occurrence": "Binomial(logit) GLMM",
            "Positive_Intensity": (
                "Zero-truncated negative binomial 2 "
                "with log link"
            ),
            "Exposure_Offset": (
                "Log_ForestPixelCount"
            ),
            "Fixed_Adjustments": [
                "Country",
                "Year_Z",
            ],
            "Random_Effects": [
                "GRID_UID random intercept",
                "Year random intercept",
            ],
            "Separate_By_Season": True,
            "Separate_By_Response": True,
            "Spatial_Block_In_Model": False,
            "Country_Is_Random_Effect": False,
        },
        "No_SEM_Fitting": True,
        "No_Path_Selection": True,
        "No_Response_Based_Indicator_Selection": True,
        "No_Imputation": True,
        "No_Change_To_Step01_Standardization": True,
    }
    write_json(METHOD_FILE, method_definition)

    software = {
        "Python": sys.version,
        "Platform": platform.platform(),
        "NumPy": np.__version__,
        "pandas": pd.__version__,
        "SciPy": scipy.__version__,
    }
    write_json(SOFTWARE_FILE, software)

    audit_table = pd.DataFrame(audits)
    audit_table.to_csv(
        INPUT_AUDIT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    failed_errors = audit_table.loc[
        (~audit_table["Passed"])
        & audit_table["Severity"].eq("ERROR")
    ]
    failed_warnings = audit_table.loc[
        (~audit_table["Passed"])
        & audit_table["Severity"].eq("WARNING")
    ]

    if not failed_errors.empty:
        raise RuntimeError(
            "SEM Step 02 failed one or more ERROR checks:\n"
            + failed_errors.to_string(index=False)
        )

    log(
        "Step 02 completed: redundant mechanism indicators were screened "
        "without using fire-response information, one common cross-season "
        "indicator set was locked, six reduced datasets were generated, "
        "and the two-part piecewise generalized SEM response structure was "
        "specified without fitting any SEM or selecting causal paths."
    )

    output_rows: List[Dict[str, object]] = []
    for path in sorted(OUTPUT_ROOT.rglob("*")):
        if not path.is_file():
            continue
        if path == OUTPUT_MANIFEST_FILE:
            continue
        output_rows.append(
            {
                "File": path.name,
                "Relative_Path": str(
                    path.relative_to(OUTPUT_ROOT)
                ),
                "File_Size_Bytes": path.stat().st_size,
                "SHA256": sha256_file(path),
                "Code_Version": CODE_VERSION,
                "Step01_Code_Version": step01_version,
                "Step01_Master_Standardized_SHA256": (
                    sha256_file(STEP01_MASTER_STD)
                ),
                "Common_Indicator_N": len(
                    common_selected
                ),
                "SEM_Fitted": False,
            }
        )
    pd.DataFrame(output_rows).to_csv(
        OUTPUT_MANIFEST_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"\nOutput directory: {OUTPUT_ROOT}\n"
        f"Audit ERROR failures: {len(failed_errors)}\n"
        f"Audit WARNING failures: {len(failed_warnings)}\n"
        f"Common locked indicators: {len(common_selected)}\n"
        f"Common indicator set: {'; '.join(common_selected)}\n"
        f"Selected dataset files: {len(selected_manifest)}\n"
        f"SEM fitted: False\n"
    )
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
