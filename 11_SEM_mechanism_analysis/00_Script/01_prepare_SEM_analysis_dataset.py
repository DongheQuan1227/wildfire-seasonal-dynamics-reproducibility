# -*- coding: utf-8 -*-
"""
Prepare the locked analysis dataset for the new SEM mechanism analysis.

Recommended location
--------------------
<REPOSITORY_ROOT>/11_SEM_mechanism_analysis/00_Script/
    01_prepare_SEM_analysis_dataset.py

Purpose
-------
This script prepares, audits, and locks the SEM input universe. It does NOT fit
a structural equation model, select a final SEM structure, or infer direct and
indirect effects.

The script:
1. reads the latest 2001-2025 base table;
2. uses the already sealed Model Comparison Step 01 aligned OOF panels to lock
   the exact grid-year-season sample used by the final regression and Hurdle RF;
3. verifies that Fire Count, Burned Pixel Count, forest exposure, and observed
   rates agree exactly with the aligned OOF data;
4. retains all current candidate mechanism variables, including LtgProxy and
   Dis_Power;
5. creates raw and within-season standardized predictors without imputing,
   winsorizing, or transforming the mechanism variables;
6. creates six season-response datasets for later generalized/piecewise SEM;
7. produces missingness, distribution, correlation, VIF, panel, and constancy
   diagnostics before any SEM structure is specified.
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

import numpy as np
import pandas as pd
import scipy
from scipy.stats import kurtosis, skew


# =============================================================================
# Locked project paths and versions
# =============================================================================

CODE_ROOT = Path(
    os.environ.get(
        "WILDFIRE_CODE_ROOT",
        str(Path(__file__).resolve().parents[2]),
    )
).resolve()

BASE_TABLE = (
    CODE_ROOT
    / "00_Input_data"
    / "Base_Table_2001_2025.csv.bz2"
)

MODEL_COMPARISON_STEP01 = (
    CODE_ROOT
    / "10_Model_comparison"
    / "01_Aligned_Model_OOF"
)
TEMPORAL_ALIGNED_OOF = (
    MODEL_COMPARISON_STEP01
    / "05_Aligned_Temporal_OOF.csv.gz"
)
SPATIAL_ALIGNED_OOF = (
    MODEL_COMPARISON_STEP01
    / "06_Aligned_Spatial_OOF.csv.gz"
)
ALIGNED_METHOD_FILE = (
    MODEL_COMPARISON_STEP01
    / "00_Method_Definition.json"
)
ALIGNED_MANIFEST_FILE = (
    MODEL_COMPARISON_STEP01
    / "07_Aligned_OOF_Manifest.csv"
)

SEM_ROOT = CODE_ROOT / "11_SEM_mechanism_analysis"
OUTPUT_ROOT = SEM_ROOT / "01_SEM_Analysis_Dataset"
DATASET_DIR = OUTPUT_ROOT / "Season_Response_Datasets"

CODE_VERSION = (
    "2026-06-30_SEM_DATA_PREPARATION_V1B_"
    "LOCKED_MODEL_UNIVERSE_DYNAMIC_PFT_CONSTANCY"
)
EXPECTED_ALIGNED_VERSION = (
    "2026-06-30_MODEL_COMPARISON_ALIGNED_OOF_V1"
)
# Aligned .csv.gz files are locked to the Step-01 output manifest rather than
# to fixed compressed-byte hashes. Gzip headers can differ between otherwise
# identical reruns, while the manifest and the structural panel audits below
# verify the files used in the current reproducibility run.
EXPECTED_GRID_COUNT = 4_982
EXPECTED_YEAR_SET = set(range(2001, 2026))
EXPECTED_SEASONS = {1: "Spring", 2: "Summer", 3: "Autumn"}
EXPECTED_RESPONSES = ["Fire_Count", "Burned_Pixel_Count"]
EXPECTED_COUNTRIES = {"China", "NK", "Russia"}
EXPECTED_SPATIAL_BLOCK_GRID_COUNTS = {
    1: 912,
    2: 1042,
    3: 735,
    4: 1222,
    5: 1071,
}
EXPECTED_ALIGNED_ROWS_PER_SCHEME = 742_374
EXPECTED_ROWS_PER_SEASON_RESPONSE = 123_729
EXPECTED_LOCKED_PANEL_ROWS = 371_187

HASH_CHUNK_SIZE = 1024 * 1024
NUMERIC_ATOL = 1e-12
NUMERIC_RTOL = 1e-10
HIGH_CORRELATION_THRESHOLD = 0.70
SEVERE_CORRELATION_THRESHOLD = 0.85
HIGH_VIF_THRESHOLD = 5.0
SEVERE_VIF_THRESHOLD = 10.0


# =============================================================================
# Candidate variables and domains
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

CANDIDATE_PREDICTORS = [
    variable
    for group in VARIABLE_GROUPS.values()
    for variable in group
]

# BD, ND, and NE come from an annual plant-functional-type time series and
# are therefore allowed to change between years within the same GRID_UID.
# Only terrain variables are expected to remain time invariant.
STATIC_VARIABLES_EXPECTED_WITHIN_GRID = [
    "DEM",
    "Slope",
    "Aspect",
]

VARIABLE_UNITS: Dict[str, str] = {
    "Temp": "degC",
    "Pre": "mm",
    "Rhum": "percent",
    "Wind": "m s-1",
    "SSRD": "J m-2",
    "LtgProxy": "proxy count or density",
    "FFMC": "index",
    "DMC": "index",
    "DC": "index",
    "ISI": "index",
    "BUI": "index",
    "FWI": "index",
    "SPEI1": "standardized index",
    "SPEI3": "standardized index",
    "SPEI6": "standardized index",
    "SPEI12": "standardized index",
    "SPEI24": "standardized index",
    "BD": "percent",
    "ND": "percent",
    "NE": "percent",
    "EVI": "scaled source value",
    "PTC": "percent",
    "DEM": "km or source unit",
    "Slope": "degree",
    "Aspect": "continuous transformed aspect index",
    "POP": "population count",
    "Road_dens": "source density unit",
    "Dis_Railway": "km or source distance unit",
    "Dis_Farm": "km",
    "Dis_Build": "km",
    "Dis_Power": "km or source distance unit",
}

VARIABLE_DESCRIPTIONS: Dict[str, str] = {
    "Temp": "Seasonal surface air temperature",
    "Pre": "Seasonal total precipitation",
    "Rhum": "Seasonal surface relative humidity",
    "Wind": "Seasonal surface wind speed",
    "SSRD": "Seasonal surface solar radiation downwards",
    "LtgProxy": "Seasonal lightning proxy",
    "FFMC": "Fine Fuel Moisture Code",
    "DMC": "Duff Moisture Code",
    "DC": "Drought Code",
    "ISI": "Initial Spread Index",
    "BUI": "Buildup Index",
    "FWI": "Fire Weather Index",
    "SPEI1": "1-month Standardized Precipitation Evapotranspiration Index",
    "SPEI3": "3-month Standardized Precipitation Evapotranspiration Index",
    "SPEI6": "6-month Standardized Precipitation Evapotranspiration Index",
    "SPEI12": "12-month Standardized Precipitation Evapotranspiration Index",
    "SPEI24": "24-month Standardized Precipitation Evapotranspiration Index",
    "BD": "Broadleaf deciduous tree density",
    "ND": "Needleleaf deciduous tree density",
    "NE": "Needleleaf evergreen tree density",
    "EVI": "Enhanced Vegetation Index",
    "PTC": "Percent Tree Cover",
    "DEM": "Elevation",
    "Slope": "Slope",
    "Aspect": "Transformed aspect index",
    "POP": "Population count",
    "Road_dens": "Road network density",
    "Dis_Railway": "Distance to railway",
    "Dis_Farm": "Distance to farmland",
    "Dis_Build": "Distance to buildings",
    "Dis_Power": "Distance to overhead power line",
}


# =============================================================================
# Flexible aliases for the latest base table
# =============================================================================

ALIASES: Dict[str, List[str]] = {
    "Country": ["Country", "COUNTRY", "country", "Country_STD"],
    "GRID_ID": ["GRID_ID", "Grid_ID", "grid_id", "GRIDID"],
    "Year": ["Year", "YEAR", "year"],
    "Season": ["Season", "SEASON", "season", "Quarter", "Q"],
    "Fire_Count": ["Fire_Count", "FC", "Count", "COUNT"],
    "Burned_Pixel_Count": [
        "Burned_Pixel_Count",
        "BurnedPixelCount",
        "Area",
        "BA_Pixel_Count",
    ],
    "ForestPixelCount": [
        "ForestPixelCount",
        "Forest_Pixel_Count",
        "ForestArea",
        "Forest_Area",
    ],
    "SSRD": ["SSRD", "ssrd", "SsrD"],
    "LtgProxy": ["LtgProxy", "LightningProxy", "Ltg_Proxy"],
    "Road_dens": ["Road_dens", "Road_Dens", "RoadDensity"],
    "Dis_Railway": [
        "Dis_Railway",
        "Dis_Rail",
        "Distance_Railway",
        "Distance_to_Railway",
    ],
    "Dis_Farm": [
        "Dis_Farm",
        "Distance_Farm",
        "Distance_to_Farmland",
    ],
    "Dis_Build": [
        "Dis_Build",
        "Distance_Build",
        "Distance_to_Buildings",
    ],
    "Dis_Power": [
        "Dis_Power",
        "Distance_Power",
        "Distance_to_Power",
    ],
}

for variable in CANDIDATE_PREDICTORS:
    ALIASES.setdefault(variable, [variable])


# =============================================================================
# Output files
# =============================================================================

METHOD_FILE = OUTPUT_ROOT / "00_Method_Definition.json"
INPUT_AUDIT_FILE = OUTPUT_ROOT / "01_Input_Integrity_Audit.csv"
COLUMN_MAPPING_FILE = OUTPUT_ROOT / "02_Base_Table_Column_Mapping.csv"
MASTER_RAW_FILE = OUTPUT_ROOT / "03_Locked_SEM_Master_Raw.csv.gz"
MASTER_STANDARDIZED_FILE = (
    OUTPUT_ROOT / "04_Locked_SEM_Master_Standardized.csv.gz"
)
DATASET_MANIFEST_FILE = (
    OUTPUT_ROOT / "05_SEM_Season_Response_Dataset_Manifest.csv"
)
VARIABLE_DICTIONARY_FILE = OUTPUT_ROOT / "06_Variable_Dictionary.csv"
MISSING_DISTRIBUTION_FILE = (
    OUTPUT_ROOT / "07_Missingness_and_Distribution_QA.csv"
)
STANDARDIZATION_FILE = (
    OUTPUT_ROOT / "08_Seasonal_Standardization_Parameters.csv"
)
CORRELATION_FILE = (
    OUTPUT_ROOT / "09_Within_Season_Spearman_Correlation_Long.csv.gz"
)
HIGH_CORRELATION_FILE = OUTPUT_ROOT / "10_High_Correlation_Pairs.csv"
VIF_FILE = OUTPUT_ROOT / "11_VIF_and_Condition_Diagnostics.csv"
PANEL_QA_FILE = OUTPUT_ROOT / "12_Response_and_Panel_QA.csv"
STATIC_QA_FILE = OUTPUT_ROOT / "13_Static_Variable_Constancy_QA.csv"
SAMPLE_COMPARISON_FILE = OUTPUT_ROOT / "14_SEM_Sample_Comparison.csv"
OUTPUT_MANIFEST_FILE = OUTPUT_ROOT / "15_Output_Manifest.csv"
SOFTWARE_FILE = OUTPUT_ROOT / "Software_Environment.json"
LOG_FILE = OUTPUT_ROOT / "sem_data_preparation.log"


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


def write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)


def read_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


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


def canonical_grid_id(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    try:
        numeric = float(text)
        if math.isfinite(numeric) and numeric.is_integer():
            return str(int(numeric))
    except Exception:
        pass
    return text


def canonical_country(value: object) -> str:
    text = str(value).strip()
    mapping = {
        "China": "China",
        "CHINA": "China",
        "china": "China",
        "CN": "China",
        "North Korea": "NK",
        "North_Korea": "NK",
        "NorthKorea": "NK",
        "NORTH KOREA": "NK",
        "NK": "NK",
        "DPRK": "NK",
        "Russia": "Russia",
        "RUSSIA": "Russia",
        "russia": "Russia",
        "RU": "Russia",
    }
    return mapping.get(text, text)


def locate_column(
    columns: Sequence[str],
    canonical_name: str,
) -> str:
    aliases = ALIASES.get(canonical_name, [canonical_name])
    exact_matches = [
        alias for alias in aliases if alias in columns
    ]
    if len(exact_matches) == 1:
        return exact_matches[0]
    if len(exact_matches) > 1:
        raise ValueError(
            f"Ambiguous aliases for {canonical_name}: {exact_matches}"
        )

    lower_map: Dict[str, List[str]] = {}
    for column in columns:
        lower_map.setdefault(str(column).lower(), []).append(str(column))

    case_insensitive_matches: List[str] = []
    for alias in aliases:
        case_insensitive_matches.extend(
            lower_map.get(alias.lower(), [])
        )
    case_insensitive_matches = sorted(set(case_insensitive_matches))

    if len(case_insensitive_matches) == 1:
        return case_insensitive_matches[0]
    if len(case_insensitive_matches) > 1:
        raise ValueError(
            f"Ambiguous case-insensitive aliases for {canonical_name}: "
            f"{case_insensitive_matches}"
        )
    raise KeyError(
        f"Required field {canonical_name!r} was not found. "
        f"Accepted aliases: {aliases}"
    )


def locate_all_columns(
    source_columns: Sequence[str],
    canonical_columns: Sequence[str],
) -> Tuple[Dict[str, str], pd.DataFrame]:
    mapping: Dict[str, str] = {}
    rows: List[Dict[str, object]] = []

    for canonical in canonical_columns:
        source = locate_column(source_columns, canonical)
        mapping[canonical] = source
        rows.append(
            {
                "Canonical_Field": canonical,
                "Source_Field": source,
                "Alias_Used": canonical != source,
                "Required": True,
            }
        )
    return mapping, pd.DataFrame(rows)


def manifest_hash(
    manifest: pd.DataFrame,
    file_name: str,
) -> str:
    candidate_columns = [
        column
        for column in ["Output_File", "File", "Relative_Path"]
        if column in manifest.columns
    ]
    if not candidate_columns or "SHA256" not in manifest.columns:
        return ""

    mask = np.zeros(len(manifest), dtype=bool)
    for column in candidate_columns:
        mask |= manifest[column].astype(str).eq(file_name).to_numpy()
    match = manifest.loc[mask]
    if len(match) != 1:
        return ""
    return str(match.iloc[0]["SHA256"])


def numeric_equal(
    left: pd.Series,
    right: pd.Series,
    atol: float = NUMERIC_ATOL,
    rtol: float = NUMERIC_RTOL,
) -> Tuple[bool, float]:
    left_values = pd.to_numeric(left, errors="coerce").to_numpy(dtype=float)
    right_values = pd.to_numeric(right, errors="coerce").to_numpy(dtype=float)
    equal = np.isclose(
        left_values,
        right_values,
        atol=atol,
        rtol=rtol,
        equal_nan=True,
    )
    differences = np.abs(left_values - right_values)
    finite = differences[np.isfinite(differences)]
    maximum = float(finite.max()) if len(finite) else 0.0
    return bool(equal.all()), maximum


def calculate_distribution_row(
    data: pd.DataFrame,
    season: int,
    variable: str,
) -> Dict[str, object]:
    values = pd.to_numeric(
        data.loc[data["Season"] == season, variable],
        errors="coerce",
    )
    finite = values[np.isfinite(values.to_numpy(dtype=float))]
    n = int(len(values))
    nonmissing_n = int(finite.notna().sum())
    missing_n = n - nonmissing_n

    if nonmissing_n == 0:
        return {
            "Season": season,
            "Season_Label": EXPECTED_SEASONS[season],
            "Variable": variable,
            "N": n,
            "Nonmissing_N": 0,
            "Missing_N": missing_n,
            "Missing_Percent": 100.0,
            "Unique_N": 0,
            "Mean": np.nan,
            "SD": np.nan,
            "Minimum": np.nan,
            "P01": np.nan,
            "P05": np.nan,
            "Median": np.nan,
            "P95": np.nan,
            "P99": np.nan,
            "Maximum": np.nan,
            "Skewness": np.nan,
            "Excess_Kurtosis": np.nan,
            "Zero_Fraction": np.nan,
            "Near_Zero_Variance": True,
            "Infinite_N": int(
                np.isinf(values.to_numpy(dtype=float)).sum()
            ),
        }

    array = finite.to_numpy(dtype=float)
    sd = float(np.std(array, ddof=1)) if len(array) > 1 else np.nan
    return {
        "Season": season,
        "Season_Label": EXPECTED_SEASONS[season],
        "Variable": variable,
        "N": n,
        "Nonmissing_N": nonmissing_n,
        "Missing_N": missing_n,
        "Missing_Percent": 100.0 * missing_n / n,
        "Unique_N": int(finite.nunique(dropna=True)),
        "Mean": float(np.mean(array)),
        "SD": sd,
        "Minimum": float(np.min(array)),
        "P01": float(np.quantile(array, 0.01)),
        "P05": float(np.quantile(array, 0.05)),
        "Median": float(np.median(array)),
        "P95": float(np.quantile(array, 0.95)),
        "P99": float(np.quantile(array, 0.99)),
        "Maximum": float(np.max(array)),
        "Skewness": (
            float(skew(array, bias=False))
            if len(array) >= 3 and np.std(array) > 0
            else np.nan
        ),
        "Excess_Kurtosis": (
            float(kurtosis(array, fisher=True, bias=False))
            if len(array) >= 4 and np.std(array) > 0
            else np.nan
        ),
        "Zero_Fraction": float(np.mean(array == 0.0)),
        "Near_Zero_Variance": bool(
            not np.isfinite(sd)
            or sd <= np.finfo(float).eps
            or int(finite.nunique(dropna=True)) <= 1
        ),
        "Infinite_N": int(
            np.isinf(values.to_numpy(dtype=float)).sum()
        ),
    }


def calculate_vif_from_correlation(
    complete_data: pd.DataFrame,
    variables: Sequence[str],
    season: int,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    matrix = complete_data[list(variables)].to_numpy(dtype=float)
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
                "Season": season,
                "Season_Label": EXPECTED_SEASONS[season],
                "Variable": variable,
                "Domain": next(
                    domain
                    for domain, members in VARIABLE_GROUPS.items()
                    if variable in members
                ),
                "VIF": vif,
                "VIF_ge_5": bool(vif >= HIGH_VIF_THRESHOLD),
                "VIF_ge_10": bool(vif >= SEVERE_VIF_THRESHOLD),
                "Correlation_Matrix_Rank": rank,
                "Predictor_N": len(variables),
                "Correlation_Condition_Number": condition_number,
                "Complete_Case_N": int(len(complete_data)),
                "Method": (
                    "VIF derived from the complete-case predictor "
                    "correlation matrix using a Moore-Penrose inverse; "
                    "diagnostic only, not final variable selection."
                ),
            }
        )

    summary = {
        "Season": season,
        "Season_Label": EXPECTED_SEASONS[season],
        "Complete_Case_N": int(len(complete_data)),
        "Predictor_N": len(variables),
        "Correlation_Matrix_Rank": rank,
        "Correlation_Condition_Number": condition_number,
        "Maximum_VIF": float(
            np.nanmax(
                [
                    row["VIF"]
                    for row in rows
                    if np.isfinite(row["VIF"])
                ]
            )
        )
        if any(np.isfinite(row["VIF"]) for row in rows)
        else float("inf"),
        "Infinite_VIF_N": int(
            sum(not np.isfinite(row["VIF"]) for row in rows)
        ),
    }
    return pd.DataFrame(rows), summary


def response_file_name(
    season: int,
    response: str,
) -> str:
    return (
        f"{season:02d}_{EXPECTED_SEASONS[season]}_"
        f"{response}_SEM.csv.gz"
    )


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text("", encoding="utf-8")

    log("Starting SEM analysis-dataset preparation.")
    log(f"Code version: {CODE_VERSION}")

    required_files = [
        BASE_TABLE,
        TEMPORAL_ALIGNED_OOF,
        SPATIAL_ALIGNED_OOF,
        ALIGNED_METHOD_FILE,
        ALIGNED_MANIFEST_FILE,
    ]
    missing_files = [
        str(path) for path in required_files if not path.exists()
    ]
    if missing_files:
        raise FileNotFoundError(
            "Required files are missing:\n"
            + "\n".join(missing_files)
        )

    audits: List[Dict[str, object]] = []

    # -------------------------------------------------------------------------
    # Lock and verify Model Comparison Step 01 inputs
    # -------------------------------------------------------------------------
    aligned_method = read_json(ALIGNED_METHOD_FILE)
    aligned_version = aligned_method.get("Code_Version")
    add_audit(
        audits,
        "Version",
        "Aligned_OOF_method_version",
        aligned_version == EXPECTED_ALIGNED_VERSION,
        aligned_version,
        EXPECTED_ALIGNED_VERSION,
    )

    temporal_hash = sha256_file(TEMPORAL_ALIGNED_OOF)
    spatial_hash = sha256_file(SPATIAL_ALIGNED_OOF)

    aligned_manifest = pd.read_csv(ALIGNED_MANIFEST_FILE)
    manifest_temporal_hash = manifest_hash(
        aligned_manifest,
        TEMPORAL_ALIGNED_OOF.name,
    )
    manifest_spatial_hash = manifest_hash(
        aligned_manifest,
        SPATIAL_ALIGNED_OOF.name,
    )
    add_audit(
        audits,
        "Hash",
        "Temporal_hash_matches_Step01_manifest",
        manifest_temporal_hash == temporal_hash,
        manifest_temporal_hash,
        temporal_hash,
    )
    add_audit(
        audits,
        "Hash",
        "Spatial_hash_matches_Step01_manifest",
        manifest_spatial_hash == spatial_hash,
        manifest_spatial_hash,
        spatial_hash,
    )

    aligned_columns = [
        "GRID_UID",
        "Country",
        "GRID_ID",
        "Year",
        "Season",
        "Season_Label",
        "Temporal_Fold",
        "Spatial_Block",
        "Validation_Scheme",
        "Validation_Fold",
        "Count_Response",
        "Rate_Scale_Name",
        "ForestPixelCount",
        "Observed_Count",
        "Observed_Rate",
    ]

    log("Reading the locked Temporal and Spatial aligned OOF panels.")
    temporal = pd.read_csv(
        TEMPORAL_ALIGNED_OOF,
        usecols=aligned_columns,
    )
    spatial = pd.read_csv(
        SPATIAL_ALIGNED_OOF,
        usecols=aligned_columns,
    )

    add_audit(
        audits,
        "Aligned_OOF",
        "Temporal_row_count",
        len(temporal) == EXPECTED_ALIGNED_ROWS_PER_SCHEME,
        len(temporal),
        EXPECTED_ALIGNED_ROWS_PER_SCHEME,
    )
    add_audit(
        audits,
        "Aligned_OOF",
        "Spatial_row_count",
        len(spatial) == EXPECTED_ALIGNED_ROWS_PER_SCHEME,
        len(spatial),
        EXPECTED_ALIGNED_ROWS_PER_SCHEME,
    )

    comparison_keys = [
        "GRID_UID",
        "Country",
        "GRID_ID",
        "Year",
        "Season",
        "Season_Label",
        "Temporal_Fold",
        "Spatial_Block",
        "Count_Response",
        "Rate_Scale_Name",
    ]
    comparison_values = [
        "ForestPixelCount",
        "Observed_Count",
        "Observed_Rate",
    ]

    temporal_compare = temporal[
        comparison_keys + comparison_values
    ].sort_values(comparison_keys).reset_index(drop=True)
    spatial_compare = spatial[
        comparison_keys + comparison_values
    ].sort_values(comparison_keys).reset_index(drop=True)

    same_keys = temporal_compare[comparison_keys].equals(
        spatial_compare[comparison_keys]
    )
    add_audit(
        audits,
        "Aligned_OOF",
        "Temporal_and_spatial_keys_identical",
        same_keys,
        same_keys,
        True,
    )

    for value_column in comparison_values:
        same, maximum_difference = numeric_equal(
            temporal_compare[value_column],
            spatial_compare[value_column],
        )
        add_audit(
            audits,
            "Aligned_OOF",
            f"Temporal_and_spatial_{value_column}_identical",
            same,
            maximum_difference,
            f"maximum difference <= {NUMERIC_ATOL}",
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
            "Locked aligned OOF integrity checks failed. "
            "See 01_Input_Integrity_Audit.csv."
        )

    # -------------------------------------------------------------------------
    # Construct the one-row-per-grid-year-season locked universe
    # -------------------------------------------------------------------------
    panel_key_columns = [
        "GRID_UID",
        "Country",
        "GRID_ID",
        "Year",
        "Season",
        "Season_Label",
        "Temporal_Fold",
        "Spatial_Block",
    ]
    locked_panel = (
        temporal[panel_key_columns]
        .drop_duplicates()
        .sort_values(
            ["Season", "Country", "GRID_UID", "Year"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    add_audit(
        audits,
        "Locked_Panel",
        "Locked_panel_row_count",
        len(locked_panel) == EXPECTED_LOCKED_PANEL_ROWS,
        len(locked_panel),
        EXPECTED_LOCKED_PANEL_ROWS,
    )
    add_audit(
        audits,
        "Locked_Panel",
        "GRID_UID_count",
        locked_panel["GRID_UID"].nunique()
        == EXPECTED_GRID_COUNT,
        int(locked_panel["GRID_UID"].nunique()),
        EXPECTED_GRID_COUNT,
    )
    add_audit(
        audits,
        "Locked_Panel",
        "Year_set",
        set(locked_panel["Year"].astype(int))
        == EXPECTED_YEAR_SET,
        sorted(
            locked_panel["Year"].astype(int).unique().tolist()
        ),
        sorted(EXPECTED_YEAR_SET),
    )
    add_audit(
        audits,
        "Locked_Panel",
        "Country_set",
        set(locked_panel["Country"].astype(str))
        == EXPECTED_COUNTRIES,
        sorted(
            locked_panel["Country"].astype(str).unique().tolist()
        ),
        sorted(EXPECTED_COUNTRIES),
    )
    add_audit(
        audits,
        "Locked_Panel",
        "Season_set",
        set(locked_panel["Season"].astype(int))
        == set(EXPECTED_SEASONS),
        sorted(
            locked_panel["Season"].astype(int).unique().tolist()
        ),
        sorted(EXPECTED_SEASONS),
    )

    block_counts = (
        locked_panel[
            ["GRID_UID", "Spatial_Block"]
        ]
        .drop_duplicates()["Spatial_Block"]
        .astype(int)
        .value_counts()
        .sort_index()
        .to_dict()
    )
    add_audit(
        audits,
        "Locked_Panel",
        "Spatial_block_grid_counts",
        block_counts == EXPECTED_SPATIAL_BLOCK_GRID_COUNTS,
        block_counts,
        EXPECTED_SPATIAL_BLOCK_GRID_COUNTS,
    )

    # -------------------------------------------------------------------------
    # Read and standardize the latest base-table schema
    # -------------------------------------------------------------------------
    log(f"Reading latest base table: {BASE_TABLE}")
    source_header = pd.read_csv(
        BASE_TABLE,
        nrows=0,
        encoding="utf-8-sig",
    )
    required_canonical_columns = [
        "Country",
        "GRID_ID",
        "Year",
        "Season",
        "Fire_Count",
        "Burned_Pixel_Count",
        "ForestPixelCount",
        *CANDIDATE_PREDICTORS,
    ]
    column_mapping, column_mapping_table = locate_all_columns(
        list(source_header.columns),
        required_canonical_columns,
    )
    column_mapping_table.to_csv(
        COLUMN_MAPPING_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    source_usecols = sorted(set(column_mapping.values()))
    base = pd.read_csv(
        BASE_TABLE,
        usecols=source_usecols,
        encoding="utf-8-sig",
        low_memory=False,
    )
    base = base.rename(
        columns={
            source: canonical
            for canonical, source in column_mapping.items()
        }
    )

    base["Country"] = base["Country"].map(canonical_country)
    base["_GRID_ID_KEY"] = base["GRID_ID"].map(canonical_grid_id)
    base["Year"] = pd.to_numeric(
        base["Year"], errors="raise"
    ).astype(int)
    base["Season"] = pd.to_numeric(
        base["Season"], errors="raise"
    ).astype(int)

    locked_panel["Country"] = locked_panel["Country"].map(
        canonical_country
    )
    locked_panel["_GRID_ID_KEY"] = locked_panel["GRID_ID"].map(
        canonical_grid_id
    )
    locked_panel["Year"] = locked_panel["Year"].astype(int)
    locked_panel["Season"] = locked_panel["Season"].astype(int)

    join_columns = [
        "Country",
        "_GRID_ID_KEY",
        "Year",
        "Season",
    ]

    base_duplicate_n = int(
        base.duplicated(join_columns, keep=False).sum()
    )
    add_audit(
        audits,
        "Base_Table",
        "Unique_country_grid_year_season_rows",
        base_duplicate_n == 0,
        base_duplicate_n,
        0,
    )

    invalid_country_n = int(
        (~base["Country"].isin(EXPECTED_COUNTRIES)).sum()
    )
    invalid_season_n = int(
        (~base["Season"].isin(EXPECTED_SEASONS)).sum()
    )
    add_audit(
        audits,
        "Base_Table",
        "Country_values_valid",
        invalid_country_n == 0,
        invalid_country_n,
        0,
    )
    add_audit(
        audits,
        "Base_Table",
        "Season_values_valid",
        invalid_season_n == 0,
        invalid_season_n,
        0,
    )

    merge_columns = [
        *join_columns,
        "GRID_ID",
        "Fire_Count",
        "Burned_Pixel_Count",
        "ForestPixelCount",
        *CANDIDATE_PREDICTORS,
    ]
    merged = locked_panel.merge(
        base[merge_columns],
        on=join_columns,
        how="left",
        validate="one_to_one",
        indicator=True,
        suffixes=("_Aligned", "_Base"),
    )

    unmatched_n = int((merged["_merge"] != "both").sum())
    add_audit(
        audits,
        "Base_Table",
        "All_locked_panel_rows_found_in_base_table",
        unmatched_n == 0,
        unmatched_n,
        0,
    )

    base_key_set = set(
        zip(
            base["Country"],
            base["_GRID_ID_KEY"],
            base["Year"],
            base["Season"],
        )
    )
    locked_key_set = set(
        zip(
            locked_panel["Country"],
            locked_panel["_GRID_ID_KEY"],
            locked_panel["Year"],
            locked_panel["Season"],
        )
    )
    add_audit(
        audits,
        "Base_Table",
        "Base_table_rows_outside_locked_universe",
        True,
        len(base_key_set - locked_key_set),
        "reported only; outside rows are intentionally excluded",
        severity="INFO",
    )

    # Prefer the aligned GRID_ID representation.
    merged["GRID_ID"] = merged["GRID_ID_Aligned"]
    merged = merged.drop(
        columns=[
            "GRID_ID_Aligned",
            "GRID_ID_Base",
            "_GRID_ID_KEY",
            "_merge",
        ]
    )

    for numeric_column in [
        "Fire_Count",
        "Burned_Pixel_Count",
        "ForestPixelCount",
        *CANDIDATE_PREDICTORS,
    ]:
        merged[numeric_column] = pd.to_numeric(
            merged[numeric_column],
            errors="coerce",
        )

    negative_fire_n = int((merged["Fire_Count"] < 0).sum())
    negative_burned_n = int(
        (merged["Burned_Pixel_Count"] < 0).sum()
    )
    nonpositive_exposure_n = int(
        (merged["ForestPixelCount"] <= 0).sum()
    )
    add_audit(
        audits,
        "Response",
        "Fire_Count_nonnegative",
        negative_fire_n == 0,
        negative_fire_n,
        0,
    )
    add_audit(
        audits,
        "Response",
        "Burned_Pixel_Count_nonnegative",
        negative_burned_n == 0,
        negative_burned_n,
        0,
    )
    add_audit(
        audits,
        "Response",
        "ForestPixelCount_positive",
        nonpositive_exposure_n == 0,
        nonpositive_exposure_n,
        0,
    )

    merged["FCD"] = (
        merged["Fire_Count"] / merged["ForestPixelCount"]
    )
    merged["BAD"] = (
        merged["Burned_Pixel_Count"]
        / merged["ForestPixelCount"]
    )
    merged["Fire_Occurrence"] = (
        merged["Fire_Count"] > 0
    ).astype(np.int8)
    merged["Burned_Occurrence"] = (
        merged["Burned_Pixel_Count"] > 0
    ).astype(np.int8)
    merged["Log_ForestPixelCount"] = np.log(
        merged["ForestPixelCount"]
    )
    merged["Year_Centered_2013"] = (
        merged["Year"].astype(float) - 2013.0
    )
    merged["Year_Z"] = (
        merged["Year"].astype(float)
        - merged["Year"].astype(float).mean()
    ) / merged["Year"].astype(float).std(ddof=1)

    required_complete_columns = [
        "Fire_Count",
        "Burned_Pixel_Count",
        "ForestPixelCount",
        *CANDIDATE_PREDICTORS,
    ]
    finite_matrix = np.isfinite(
        merged[required_complete_columns].to_numpy(dtype=float)
    )
    merged["SEM_Complete_All_Candidates"] = (
        finite_matrix.all(axis=1).astype(np.int8)
    )

    complete_case_n = int(
        merged["SEM_Complete_All_Candidates"].sum()
    )
    add_audit(
        audits,
        "Missingness",
        "All_candidate_complete_case_rows_available",
        complete_case_n > 0,
        complete_case_n,
        ">0",
    )

    # -------------------------------------------------------------------------
    # Verify base-table responses against locked aligned OOF observations
    # -------------------------------------------------------------------------
    log("Verifying base-table outcomes against aligned OOF observations.")
    response_map = {
        "Fire_Count": ("Fire_Count", "FCD"),
        "Burned_Pixel_Count": (
            "Burned_Pixel_Count",
            "BAD",
        ),
    }

    aligned_response = temporal[
        [
            "GRID_UID",
            "Year",
            "Season",
            "Count_Response",
            "ForestPixelCount",
            "Observed_Count",
            "Observed_Rate",
        ]
    ].copy()

    for response_name, (count_column, rate_column) in response_map.items():
        expected = aligned_response.loc[
            aligned_response["Count_Response"]
            == response_name
        ].sort_values(
            ["Season", "GRID_UID", "Year"]
        ).reset_index(drop=True)
        observed = merged[
            [
                "GRID_UID",
                "Year",
                "Season",
                "ForestPixelCount",
                count_column,
                rate_column,
            ]
        ].sort_values(
            ["Season", "GRID_UID", "Year"]
        ).reset_index(drop=True)

        add_audit(
            audits,
            "Response_Alignment",
            f"{response_name}_row_count",
            len(expected) == len(observed),
            len(observed),
            len(expected),
        )
        # DataFrame.equals() also requires identical pandas dtypes.
        # The locked OOF and merged base table can contain the same GRID_UID
        # values while one side is read as numeric and the other as text.
        # Canonicalize identifiers and integer time keys before comparison so
        # this audit tests key values rather than CSV dtype inference.
        expected_keys = pd.DataFrame(
            {
                "GRID_UID": expected["GRID_UID"].map(
                    canonical_grid_id
                ),
                "Year": pd.to_numeric(
                    expected["Year"], errors="raise"
                ).astype(int),
                "Season": pd.to_numeric(
                    expected["Season"], errors="raise"
                ).astype(int),
            }
        )
        observed_keys = pd.DataFrame(
            {
                "GRID_UID": observed["GRID_UID"].map(
                    canonical_grid_id
                ),
                "Year": pd.to_numeric(
                    observed["Year"], errors="raise"
                ).astype(int),
                "Season": pd.to_numeric(
                    observed["Season"], errors="raise"
                ).astype(int),
            }
        )
        keys_equal = expected_keys.equals(observed_keys)
        mismatch_key_n = int(
            (
                expected_keys.astype(str)
                != observed_keys.astype(str)
            ).any(axis=1).sum()
        )
        add_audit(
            audits,
            "Response_Alignment",
            f"{response_name}_keys_equal",
            keys_equal,
            mismatch_key_n,
            0,
            detail=(
                "GRID_UID values were canonicalized and Year/Season "
                "were converted to integers before comparison."
            ),
        )

        for aligned_column, base_column in [
            ("ForestPixelCount", "ForestPixelCount"),
            ("Observed_Count", count_column),
            ("Observed_Rate", rate_column),
        ]:
            same, maximum_difference = numeric_equal(
                expected[aligned_column],
                observed[base_column],
            )
            add_audit(
                audits,
                "Response_Alignment",
                f"{response_name}_{aligned_column}_matches_base",
                same,
                maximum_difference,
                f"maximum difference <= {NUMERIC_ATOL}",
            )

    # -------------------------------------------------------------------------
    # Distribution, missingness, and standardization
    # -------------------------------------------------------------------------
    log("Calculating missingness and distribution diagnostics.")
    distribution_rows: List[Dict[str, object]] = []
    variables_for_distribution = [
        "Fire_Count",
        "Burned_Pixel_Count",
        "FCD",
        "BAD",
        "ForestPixelCount",
        *CANDIDATE_PREDICTORS,
    ]
    for season in EXPECTED_SEASONS:
        for variable in variables_for_distribution:
            distribution_rows.append(
                calculate_distribution_row(
                    merged,
                    season,
                    variable,
                )
            )
    distribution_table = pd.DataFrame(distribution_rows)
    distribution_table.to_csv(
        MISSING_DISTRIBUTION_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    predictor_distribution = distribution_table.loc[
        distribution_table["Variable"].isin(
            CANDIDATE_PREDICTORS
        )
    ]
    zero_variance_rows = predictor_distribution.loc[
        predictor_distribution["Near_Zero_Variance"]
    ]
    infinite_total = int(
        predictor_distribution["Infinite_N"].sum()
    )
    add_audit(
        audits,
        "Predictors",
        "No_near_zero_variance_predictors_within_season",
        zero_variance_rows.empty,
        int(len(zero_variance_rows)),
        0,
    )
    add_audit(
        audits,
        "Predictors",
        "No_infinite_predictor_values",
        infinite_total == 0,
        infinite_total,
        0,
    )

    log("Applying within-season z-standardization to predictors.")
    standardized = merged.copy()
    standardization_rows: List[Dict[str, object]] = []

    for season in EXPECTED_SEASONS:
        season_mask = standardized["Season"] == season
        fitting_mask = (
            season_mask
            & (
                standardized[
                    "SEM_Complete_All_Candidates"
                ]
                == 1
            )
        )

        for variable in CANDIDATE_PREDICTORS:
            fitting_values = standardized.loc[
                fitting_mask, variable
            ].to_numpy(dtype=float)
            mean_value = float(np.mean(fitting_values))
            sd_value = float(np.std(fitting_values, ddof=1))

            if not np.isfinite(sd_value) or sd_value <= 0:
                raise ValueError(
                    f"Cannot standardize {variable} in season {season}: "
                    f"SD={sd_value}"
                )

            z_column = f"Z_{variable}"
            standardized.loc[
                season_mask, z_column
            ] = (
                standardized.loc[season_mask, variable]
                - mean_value
            ) / sd_value

            standardization_rows.append(
                {
                    "Season": season,
                    "Season_Label": EXPECTED_SEASONS[season],
                    "Variable": variable,
                    "Domain": next(
                        domain
                        for domain, members
                        in VARIABLE_GROUPS.items()
                        if variable in members
                    ),
                    "Center_Mean": mean_value,
                    "Scale_SD_ddof1": sd_value,
                    "Fitting_Complete_Case_N": int(
                        fitting_mask.sum()
                    ),
                    "Standardization_Scope": (
                        "Within season; parameters estimated from "
                        "the all-candidate complete-case sample."
                    ),
                    "Raw_Variable_Transformed": False,
                    "Winsorized": False,
                    "Imputed": False,
                }
            )

    standardization_table = pd.DataFrame(
        standardization_rows
    )
    standardization_table.to_csv(
        STANDARDIZATION_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # Verify z means and SDs on fitting rows.
    for season in EXPECTED_SEASONS:
        fitting = standardized.loc[
            (standardized["Season"] == season)
            & (
                standardized[
                    "SEM_Complete_All_Candidates"
                ]
                == 1
            )
        ]
        z_columns = [
            f"Z_{variable}"
            for variable in CANDIDATE_PREDICTORS
        ]
        maximum_abs_mean = float(
            fitting[z_columns].mean().abs().max()
        )
        maximum_abs_sd_error = float(
            (fitting[z_columns].std(ddof=1) - 1.0)
            .abs()
            .max()
        )
        add_audit(
            audits,
            "Standardization",
            f"{EXPECTED_SEASONS[season]}_z_means_zero",
            maximum_abs_mean <= 1e-10,
            maximum_abs_mean,
            "<=1e-10",
        )
        add_audit(
            audits,
            "Standardization",
            f"{EXPECTED_SEASONS[season]}_z_sds_one",
            maximum_abs_sd_error <= 1e-10,
            maximum_abs_sd_error,
            "<=1e-10",
        )

    # -------------------------------------------------------------------------
    # Correlation and VIF diagnostics
    # -------------------------------------------------------------------------
    log("Calculating within-season Spearman correlations.")
    correlation_rows: List[Dict[str, object]] = []
    high_correlation_rows: List[Dict[str, object]] = []
    vif_frames: List[pd.DataFrame] = []
    vif_summary_rows: List[Dict[str, object]] = []

    domain_lookup = {
        variable: domain
        for domain, variables in VARIABLE_GROUPS.items()
        for variable in variables
    }

    for season in EXPECTED_SEASONS:
        season_complete = standardized.loc[
            (standardized["Season"] == season)
            & (
                standardized[
                    "SEM_Complete_All_Candidates"
                ]
                == 1
            ),
            CANDIDATE_PREDICTORS,
        ].copy()

        correlation = season_complete.corr(
            method="spearman"
        )

        for i, variable_1 in enumerate(CANDIDATE_PREDICTORS):
            for j, variable_2 in enumerate(
                CANDIDATE_PREDICTORS
            ):
                rho = float(
                    correlation.loc[
                        variable_1, variable_2
                    ]
                )
                correlation_rows.append(
                    {
                        "Season": season,
                        "Season_Label": EXPECTED_SEASONS[
                            season
                        ],
                        "Variable_1": variable_1,
                        "Domain_1": domain_lookup[variable_1],
                        "Variable_2": variable_2,
                        "Domain_2": domain_lookup[variable_2],
                        "Spearman_Rho": rho,
                        "Absolute_Rho": abs(rho),
                        "Same_Domain": (
                            domain_lookup[variable_1]
                            == domain_lookup[variable_2]
                        ),
                        "Complete_Case_N": int(
                            len(season_complete)
                        ),
                    }
                )
                if (
                    j > i
                    and abs(rho)
                    >= HIGH_CORRELATION_THRESHOLD
                ):
                    high_correlation_rows.append(
                        {
                            "Season": season,
                            "Season_Label": EXPECTED_SEASONS[
                                season
                            ],
                            "Variable_1": variable_1,
                            "Domain_1": domain_lookup[
                                variable_1
                            ],
                            "Variable_2": variable_2,
                            "Domain_2": domain_lookup[
                                variable_2
                            ],
                            "Spearman_Rho": rho,
                            "Absolute_Rho": abs(rho),
                            "Same_Domain": (
                                domain_lookup[variable_1]
                                == domain_lookup[variable_2]
                            ),
                            "Absolute_Rho_ge_0p70": True,
                            "Absolute_Rho_ge_0p85": (
                                abs(rho)
                                >= SEVERE_CORRELATION_THRESHOLD
                            ),
                            "Interpretation": (
                                "Diagnostic only; retain until the "
                                "indicator-selection step."
                            ),
                        }
                    )

        vif_table, vif_summary = (
            calculate_vif_from_correlation(
                season_complete,
                CANDIDATE_PREDICTORS,
                season,
            )
        )
        vif_frames.append(vif_table)
        vif_summary_rows.append(vif_summary)

    correlation_table = pd.DataFrame(
        correlation_rows
    )
    correlation_table.to_csv(
        CORRELATION_FILE,
        index=False,
        compression="gzip",
        encoding="utf-8-sig",
    )
    high_correlation_table = pd.DataFrame(
        high_correlation_rows
    )
    high_correlation_table.to_csv(
        HIGH_CORRELATION_FILE,
        index=False,
        encoding="utf-8-sig",
    )
    vif_table = pd.concat(
        vif_frames,
        ignore_index=True,
    )
    vif_table.to_csv(
        VIF_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # Panel, response, constancy, and sample-size QA
    # -------------------------------------------------------------------------
    log("Calculating response and panel QA.")
    panel_rows: List[Dict[str, object]] = []
    sample_rows: List[Dict[str, object]] = []

    for season in EXPECTED_SEASONS:
        season_data = standardized.loc[
            standardized["Season"] == season
        ]
        for response_name, (
            count_column,
            rate_column,
        ) in response_map.items():
            occurrence_column = (
                "Fire_Occurrence"
                if response_name == "Fire_Count"
                else "Burned_Occurrence"
            )
            total_n = int(len(season_data))
            complete_n = int(
                season_data[
                    "SEM_Complete_All_Candidates"
                ].sum()
            )
            positive_n = int(
                (season_data[count_column] > 0).sum()
            )
            complete_positive_n = int(
                (
                    (season_data[count_column] > 0)
                    & (
                        season_data[
                            "SEM_Complete_All_Candidates"
                        ]
                        == 1
                    )
                ).sum()
            )

            year_counts = (
                season_data.groupby("GRID_UID")["Year"]
                .nunique()
            )
            panel_rows.append(
                {
                    "Season": season,
                    "Season_Label": EXPECTED_SEASONS[
                        season
                    ],
                    "Response": response_name,
                    "Rate_Name": rate_column,
                    "Total_N": total_n,
                    "Complete_All_Candidates_N": complete_n,
                    "GRID_UID_Count": int(
                        season_data["GRID_UID"].nunique()
                    ),
                    "Country_Count": int(
                        season_data["Country"].nunique()
                    ),
                    "Year_Count": int(
                        season_data["Year"].nunique()
                    ),
                    "Positive_N": positive_n,
                    "Zero_N": total_n - positive_n,
                    "Positive_Fraction": positive_n / total_n,
                    "Complete_Case_Positive_N": (
                        complete_positive_n
                    ),
                    "Complete_Case_Zero_N": (
                        complete_n - complete_positive_n
                    ),
                    "Observed_Count_Mean": float(
                        season_data[count_column].mean()
                    ),
                    "Observed_Count_Variance": float(
                        season_data[count_column].var(ddof=1)
                    ),
                    "Observed_Rate_Mean": float(
                        season_data[rate_column].mean()
                    ),
                    "Observed_Rate_Maximum": float(
                        season_data[rate_column].max()
                    ),
                    "Exposure_Minimum": float(
                        season_data[
                            "ForestPixelCount"
                        ].min()
                    ),
                    "Exposure_Median": float(
                        season_data[
                            "ForestPixelCount"
                        ].median()
                    ),
                    "Exposure_Maximum": float(
                        season_data[
                            "ForestPixelCount"
                        ].max()
                    ),
                    "Grid_Years_Minimum": int(
                        year_counts.min()
                    ),
                    "Grid_Years_Median": float(
                        year_counts.median()
                    ),
                    "Grid_Years_Maximum": int(
                        year_counts.max()
                    ),
                    "Occurrence_Field": occurrence_column,
                    "Primary_SEM_Response_Representation": (
                        "Raw count with log(ForestPixelCount) "
                        "available as an offset; final submodel "
                        "family is not selected in this step."
                    ),
                }
            )

            sample_rows.extend(
                [
                    {
                        "Season": season,
                        "Season_Label": EXPECTED_SEASONS[
                            season
                        ],
                        "Response": response_name,
                        "Sample_Definition": (
                            "Locked_model_universe"
                        ),
                        "N": total_n,
                        "GRID_UID_Count": int(
                            season_data[
                                "GRID_UID"
                            ].nunique()
                        ),
                        "Positive_N": positive_n,
                        "Zero_N": total_n - positive_n,
                        "Percent_of_Locked_Universe": 100.0,
                    },
                    {
                        "Season": season,
                        "Season_Label": EXPECTED_SEASONS[
                            season
                        ],
                        "Response": response_name,
                        "Sample_Definition": (
                            "All_candidate_complete_case"
                        ),
                        "N": complete_n,
                        "GRID_UID_Count": int(
                            season_data.loc[
                                season_data[
                                    "SEM_Complete_All_Candidates"
                                ]
                                == 1,
                                "GRID_UID",
                            ].nunique()
                        ),
                        "Positive_N": complete_positive_n,
                        "Zero_N": (
                            complete_n
                            - complete_positive_n
                        ),
                        "Percent_of_Locked_Universe": (
                            100.0 * complete_n / total_n
                        ),
                    },
                ]
            )

    panel_table = pd.DataFrame(panel_rows)
    panel_table.to_csv(
        PANEL_QA_FILE,
        index=False,
        encoding="utf-8-sig",
    )
    sample_table = pd.DataFrame(sample_rows)
    sample_table.to_csv(
        SAMPLE_COMPARISON_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    add_audit(
        audits,
        "Panel",
        "Six_season_response_combinations",
        len(panel_table) == 6,
        len(panel_table),
        6,
    )
    add_audit(
        audits,
        "Panel",
        "Each_combination_has_locked_row_count",
        bool(
            (
                panel_table["Total_N"]
                == EXPECTED_ROWS_PER_SEASON_RESPONSE
            ).all()
        ),
        sorted(panel_table["Total_N"].unique().tolist()),
        [EXPECTED_ROWS_PER_SEASON_RESPONSE],
    )
    add_audit(
        audits,
        "Panel",
        "Each_combination_has_4982_grids",
        bool(
            (
                panel_table["GRID_UID_Count"]
                == EXPECTED_GRID_COUNT
            ).all()
        ),
        sorted(
            panel_table[
                "GRID_UID_Count"
            ].unique().tolist()
        ),
        [EXPECTED_GRID_COUNT],
    )

    log("Checking within-grid constancy of expected static variables.")
    static_rows: List[Dict[str, object]] = []
    for variable in STATIC_VARIABLES_EXPECTED_WITHIN_GRID:
        group_unique = (
            merged.groupby("GRID_UID")[variable]
            .nunique(dropna=False)
        )
        violating = group_unique[group_unique > 1]
        static_rows.append(
            {
                "Variable": variable,
                "Expected_Constant_Within_GRID_UID": True,
                "GRID_UID_Count": int(len(group_unique)),
                "Violating_GRID_UID_N": int(
                    len(violating)
                ),
                "Maximum_Unique_Values_Within_GRID_UID": int(
                    group_unique.max()
                ),
                "Passed": bool(violating.empty),
                "Interpretation": (
                    "A nonzero violation count requires source-data "
                    "review before treating the field as time invariant."
                ),
            }
        )
    static_table = pd.DataFrame(static_rows)
    static_table.to_csv(
        STATIC_QA_FILE,
        index=False,
        encoding="utf-8-sig",
    )
    add_audit(
        audits,
        "Static_Variables",
        "Expected_static_variables_constant_within_grid",
        bool(static_table["Passed"].all()),
        int(static_table["Passed"].sum()),
        len(static_table),
        severity="WARNING",
    )

    # -------------------------------------------------------------------------
    # Save master and six season-response datasets
    # -------------------------------------------------------------------------
    raw_columns = [
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
        "ForestPixelCount",
        "Log_ForestPixelCount",
        "Fire_Count",
        "FCD",
        "Fire_Occurrence",
        "Burned_Pixel_Count",
        "BAD",
        "Burned_Occurrence",
        "SEM_Complete_All_Candidates",
        *CANDIDATE_PREDICTORS,
    ]
    standardized_columns = [
        *raw_columns,
        *[f"Z_{variable}" for variable in CANDIDATE_PREDICTORS],
    ]

    merged = merged.sort_values(
        ["Season", "Country", "GRID_UID", "Year"],
        kind="mergesort",
    ).reset_index(drop=True)
    standardized = standardized.sort_values(
        ["Season", "Country", "GRID_UID", "Year"],
        kind="mergesort",
    ).reset_index(drop=True)

    log("Writing locked master SEM datasets.")
    merged[raw_columns].to_csv(
        MASTER_RAW_FILE,
        index=False,
        compression="gzip",
        encoding="utf-8-sig",
    )
    standardized[standardized_columns].to_csv(
        MASTER_STANDARDIZED_FILE,
        index=False,
        compression="gzip",
        encoding="utf-8-sig",
    )

    dataset_manifest_rows: List[Dict[str, object]] = []
    for season in EXPECTED_SEASONS:
        season_mask = standardized["Season"] == season
        for response_name, (
            count_column,
            rate_column,
        ) in response_map.items():
            occurrence_column = (
                "Fire_Occurrence"
                if response_name == "Fire_Count"
                else "Burned_Occurrence"
            )
            file_path = DATASET_DIR / response_file_name(
                season,
                response_name,
            )

            response_data = standardized.loc[
                season_mask,
                [
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
                    "ForestPixelCount",
                    "Log_ForestPixelCount",
                    "SEM_Complete_All_Candidates",
                    count_column,
                    rate_column,
                    occurrence_column,
                    *CANDIDATE_PREDICTORS,
                    *[
                        f"Z_{variable}"
                        for variable in CANDIDATE_PREDICTORS
                    ],
                ],
            ].copy()

            response_data = response_data.rename(
                columns={
                    count_column: "Observed_Count",
                    rate_column: "Observed_Rate",
                    occurrence_column: "Observed_Occurrence",
                }
            )
            response_data.insert(
                10,
                "Response_Name",
                response_name,
            )
            response_data.insert(
                11,
                "Rate_Name",
                rate_column,
            )
            response_data.to_csv(
                file_path,
                index=False,
                compression="gzip",
                encoding="utf-8-sig",
            )

            dataset_manifest_rows.append(
                {
                    "Season": season,
                    "Season_Label": EXPECTED_SEASONS[
                        season
                    ],
                    "Response": response_name,
                    "Rate_Name": rate_column,
                    "Relative_Path": str(
                        file_path.relative_to(OUTPUT_ROOT)
                    ),
                    "Rows": int(len(response_data)),
                    "Complete_All_Candidates_N": int(
                        response_data[
                            "SEM_Complete_All_Candidates"
                        ].sum()
                    ),
                    "GRID_UID_Count": int(
                        response_data["GRID_UID"].nunique()
                    ),
                    "Year_Count": int(
                        response_data["Year"].nunique()
                    ),
                    "Positive_N": int(
                        response_data[
                            "Observed_Occurrence"
                        ].sum()
                    ),
                    "Zero_N": int(
                        len(response_data)
                        - response_data[
                            "Observed_Occurrence"
                        ].sum()
                    ),
                    "File_Size_Bytes": int(
                        file_path.stat().st_size
                    ),
                    "SHA256": sha256_file(file_path),
                }
            )

    dataset_manifest = pd.DataFrame(
        dataset_manifest_rows
    )
    dataset_manifest.to_csv(
        DATASET_MANIFEST_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # Variable dictionary and method definition
    # -------------------------------------------------------------------------
    variable_rows: List[Dict[str, object]] = []
    for domain, variables in VARIABLE_GROUPS.items():
        for variable in variables:
            variable_rows.append(
                {
                    "Variable": variable,
                    "Standardized_Field": f"Z_{variable}",
                    "Domain": domain,
                    "Description": VARIABLE_DESCRIPTIONS[
                        variable
                    ],
                    "Unit_or_Scale": VARIABLE_UNITS[
                        variable
                    ],
                    "Role_in_Step01": (
                        "Candidate mechanism indicator; retained "
                        "without final SEM selection."
                    ),
                    "Raw_Transformation_Applied": "None",
                    "Standardization": (
                        "Within-season z score using all-candidate "
                        "complete cases; sample SD with ddof=1."
                    ),
                    "Direction_Reversed": False,
                    "Imputed": False,
                    "Winsorized": False,
                    "Final_SEM_Indicator_Status": (
                        "Not yet decided"
                    ),
                }
            )
    variable_dictionary = pd.DataFrame(variable_rows)
    variable_dictionary.to_csv(
        VARIABLE_DICTIONARY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    base_hash = sha256_file(BASE_TABLE)
    method_definition = {
        "Code_Version": CODE_VERSION,
        "Created": datetime.now().isoformat(timespec="seconds"),
        "Project_Period": "2001-2025",
        "Base_Table": str(BASE_TABLE),
        "Base_Table_SHA256": base_hash,
        "Locked_Sample_Source": (
            "Model Comparison Step 01 Temporal and Spatial aligned "
            "OOF panels"
        ),
        "Temporal_Aligned_OOF": str(TEMPORAL_ALIGNED_OOF),
        "Temporal_Aligned_OOF_SHA256": temporal_hash,
        "Spatial_Aligned_OOF": str(SPATIAL_ALIGNED_OOF),
        "Spatial_Aligned_OOF_SHA256": spatial_hash,
        "Aligned_OOF_Code_Version": aligned_version,
        "Locked_GRID_UID_Count": int(
            locked_panel["GRID_UID"].nunique()
        ),
        "Locked_Grid_Year_Season_Rows": int(len(locked_panel)),
        "Season_Response_Combinations": 6,
        "Candidate_Predictor_N": len(CANDIDATE_PREDICTORS),
        "Variable_Groups": VARIABLE_GROUPS,
        "Response_Representations": {
            "Fire_Count": {
                "Raw_Count": "Fire_Count",
                "Rate": "FCD = Fire_Count / ForestPixelCount",
                "Occurrence": "Fire_Count > 0",
            },
            "Burned_Pixel_Count": {
                "Raw_Count": "Burned_Pixel_Count",
                "Rate": (
                    "BAD = Burned_Pixel_Count / ForestPixelCount"
                ),
                "Occurrence": "Burned_Pixel_Count > 0",
            },
        },
        "Exposure": {
            "Field": "ForestPixelCount",
            "Offset_Field": "Log_ForestPixelCount",
            "Rule": (
                "The final generalized SEM submodels may use the raw "
                "count response with log(ForestPixelCount) as an "
                "offset. No final distribution is selected in Step 01."
            ),
        },
        "Missing_Data_Rule": (
            "No imputation. The locked model universe is retained, "
            "and SEM_Complete_All_Candidates marks rows complete for "
            "all candidate mechanism variables."
        ),
        "Standardization_Rule": (
            "Predictors only; within each fire season; center and "
            "sample SD estimated from all-candidate complete cases; "
            "no response standardization."
        ),
        "No_Raw_Predictor_Transformation": True,
        "No_Winsorization": True,
        "No_Imputation": True,
        "No_SEM_Fitting": True,
        "No_Final_Indicator_Selection": True,
        "No_Final_Path_Structure_Assumption": True,
        "No_Predictive_Model_Comparison": True,
        "No_Reuse_of_Legacy_ZB_or_Legacy_SEM_Data": True,
        "Key_Comparison": (
            "GRID_UID values are canonicalized and Year/Season are "
            "converted to integers before response-alignment key checks, "
            "avoiding false failures caused only by CSV dtype inference."
        ),
        "Time_Varying_PFT_Classification": (
            "BD, ND, and NE are annual plant-functional-type variables and "
            "are not tested as time-invariant fields. Only DEM, Slope, and "
            "Aspect are expected to remain constant within GRID_UID."
        ),
        "Correlation_Thresholds": {
            "High": HIGH_CORRELATION_THRESHOLD,
            "Severe": SEVERE_CORRELATION_THRESHOLD,
        },
        "VIF_Thresholds": {
            "High": HIGH_VIF_THRESHOLD,
            "Severe": SEVERE_VIF_THRESHOLD,
        },
        "Primary_Next_Decision": (
            "Use Step 01 diagnostics to select nonredundant observed "
            "indicators and decide between generalized SEM and "
            "piecewise SEM before fitting any mechanism model."
        ),
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

    # -------------------------------------------------------------------------
    # Final QA and output manifest
    # -------------------------------------------------------------------------
    audit_table = pd.DataFrame(audits)
    audit_table.to_csv(
        INPUT_AUDIT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    failed_errors = audit_table.loc[
        (~audit_table["Passed"])
        & (audit_table["Severity"] == "ERROR")
    ]
    failed_warnings = audit_table.loc[
        (~audit_table["Passed"])
        & (audit_table["Severity"] == "WARNING")
    ]

    if not failed_errors.empty:
        raise RuntimeError(
            "SEM data preparation failed one or more ERROR checks:\n"
            + failed_errors.to_string(index=False)
        )

    log(
        "Step 01 completed: the SEM analysis universe, candidate "
        "mechanism variables, response/exposure fields, within-season "
        "standardization parameters, and six season-response datasets "
        "were locked from the latest 2001-2025 base table and the sealed "
        "model-comparison sample without fitting or preselecting any SEM."
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
                "File_Size_Bytes": int(path.stat().st_size),
                "SHA256": sha256_file(path),
                "Code_Version": CODE_VERSION,
                "Base_Table_SHA256": base_hash,
                "Temporal_Aligned_OOF_SHA256": temporal_hash,
                "Spatial_Aligned_OOF_SHA256": spatial_hash,
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
        f"Six dataset files: {len(dataset_manifest)}\n"
        f"Locked panel rows: {len(locked_panel):,}\n"
        f"All-candidate complete rows: {complete_case_n:,}\n"
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
