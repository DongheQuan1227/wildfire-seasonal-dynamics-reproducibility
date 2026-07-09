# -*- coding: utf-8 -*-
"""
Step 10: unified temporal- and spatial-OOF residual Moran analysis for the
locked final Hurdle random-forest model.

Recommended public location
---------------------------
<CODE_ROOT>/09_RF_modeling/00_Script/
    10_test_final_hurdle_RF_residual_spatial_autocorrelation.py

Inputs
------
Existing final Hurdle RF predictions (no retraining):
<CODE_ROOT>/09_RF_modeling/06_Final_Hurdle_RF_Validation/
    Temporal_Out_of_Fold_Predictions.csv.gz
    Spatial_Out_of_Fold_Predictions.csv.gz
    Temporal_OOF_Residual_Morans_I.csv  [legacy comparison only]

Shared spatial assignment generated upstream:
<CODE_ROOT>/08_Regression_modeling/
    07_Regression_Spatial_Block_Validation/
        Spatial_Block_Assignment.csv

Locked unified Moran protocol and reference outputs:
<CODE_ROOT>/08_Regression_modeling/
    00_Script/08_test_regression_residual_spatial_autocorrelation.py
    08_Regression_Residual_Spatial_Autocorrelation/
        00_Method_Definition.json
        01_Input_and_Residual_QA.csv
        02b_Grid_Time_Support_Audit.csv
        03b_Main_4982_KNN8_Neighbor_Edges.csv.gz
        03c_Complete_4924_KNN8_Neighbor_Edges.csv.gz
        05_Primary_Main_Sample_Morans_I.csv

Output
------
<CODE_ROOT>/09_RF_modeling/
    10_Final_Hurdle_RF_Residual_Spatial_Autocorrelation/

Purpose
-------
Recalculate final Hurdle RF temporal- and spatial-OOF residual Moran's I under
exactly the same protocol used by regression Step 08. This step reads
existing OOF predictions only. It does not train, tune, select, or predict any
random-forest model and does not regenerate spatial blocks.

Unified protocol
----------------
1. Primary sample: the locked common 4,982-grid universe; mean rate residual
   across each grid's available OOF years.
2. Time-support sensitivity: the same 4,924 grids observed in all 25 years as
   regression Step 08.
3. Temporal and spatial OOF are analyzed separately.
4. Primary weights: directed k=8 UTM52N-centroid nearest-neighbor,
   row-standardized; 9,999 two-sided permutations.
5. Weight sensitivities: k=4 and k=12; 999 permutations.
6. The regression Step 08 helper functions and deterministic seed rule
   are imported after version/hash verification, ensuring one implementation.

FDR note
--------
This RF stage reports BH-FDR within its 12 primary RF tests, within each OOF
scheme, and within scheme/model. The later cross-model Step 10 must recompute
combined FDR across Baseline NB1, Final Regression, and Final Hurdle RF.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import platform
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy
from scipy import sparse


# =============================================================================
# Repository-relative paths
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
RF_ROOT = SCRIPT_DIR.parent
CODE_ROOT = RF_ROOT.parent
REGRESSION_ROOT = CODE_ROOT / "08_Regression_modeling"

RF_VALIDATION_ROOT = RF_ROOT / "06_Final_Hurdle_RF_Validation"
TEMPORAL_OOF_FILE = RF_VALIDATION_ROOT / "Temporal_Out_of_Fold_Predictions.csv.gz"
SPATIAL_OOF_FILE = RF_VALIDATION_ROOT / "Spatial_Out_of_Fold_Predictions.csv.gz"
LEGACY_MORAN_FILE = RF_VALIDATION_ROOT / "Temporal_OOF_Residual_Morans_I.csv"

STEP07_ROOT = REGRESSION_ROOT / "07_Regression_Spatial_Block_Validation"
SPATIAL_BLOCK_FILE = STEP07_ROOT / "Spatial_Block_Assignment.csv"

REGRESSION_STEP08_SCRIPT = (
    REGRESSION_ROOT
    / "00_Script"
    / "08_test_regression_residual_spatial_autocorrelation.py"
)
REGRESSION_STEP08_ROOT = (
    REGRESSION_ROOT / "08_Regression_Residual_Spatial_Autocorrelation"
)
REGRESSION_STEP08_METHOD_FILE = REGRESSION_STEP08_ROOT / "00_Method_Definition.json"
REGRESSION_INPUT_QA_FILE = REGRESSION_STEP08_ROOT / "01_Input_and_Residual_QA.csv"
REGRESSION_TIME_SUPPORT_FILE = (
    REGRESSION_STEP08_ROOT / "02b_Grid_Time_Support_Audit.csv"
)
REGRESSION_MAIN_EDGE_FILE = (
    REGRESSION_STEP08_ROOT / "03b_Main_4982_KNN8_Neighbor_Edges.csv.gz"
)
REGRESSION_COMPLETE_EDGE_FILE = (
    REGRESSION_STEP08_ROOT / "03c_Complete_4924_KNN8_Neighbor_Edges.csv.gz"
)
REGRESSION_PRIMARY_MORAN_FILE = (
    REGRESSION_STEP08_ROOT / "05_Primary_Main_Sample_Morans_I.csv"
)

OUTPUT_ROOT = RF_ROOT / "10_Final_Hurdle_RF_Residual_Spatial_Autocorrelation"
LOG_FILE = OUTPUT_ROOT / "final_hurdle_RF_residual_spatial_autocorrelation.log"

INPUT_QA_FILE = OUTPUT_ROOT / "01_Input_and_Residual_QA.csv"
RESIDUAL_QA_FILE = OUTPUT_ROOT / "02_Grid_Residual_Summary_QA.csv"
TIME_SUPPORT_FILE = OUTPUT_ROOT / "02b_Grid_Time_Support_Audit.csv"
PARTIAL_TIME_SUPPORT_FILE = OUTPUT_ROOT / "02c_Partial_Time_Support_Grids.csv"
WEIGHT_QA_FILE = OUTPUT_ROOT / "03_Spatial_Weights_QA.csv"
MAIN_EDGE_FILE = OUTPUT_ROOT / "03b_Main_4982_KNN8_Neighbor_Edges.csv.gz"
COMPLETE_EDGE_FILE = OUTPUT_ROOT / "03c_Complete_4924_KNN8_Neighbor_Edges.csv.gz"
GRID_RESIDUAL_FILE = OUTPUT_ROOT / "04_All_Grid_Mean_OOF_Residuals.csv.gz"
PRIMARY_MORAN_FILE = OUTPUT_ROOT / "05_Primary_Main_Sample_Morans_I.csv"
COMPLETE_MORAN_FILE = OUTPUT_ROOT / "06_Complete_25Y_Sensitivity_Morans_I.csv"
KNN_SENSITIVITY_FILE = OUTPUT_ROOT / "07_KNN_Sensitivity_Morans_I.csv"
LEGACY_COMPARISON_FILE = OUTPUT_ROOT / "08_Legacy_RF_Moran_Reproduction_Audit.csv"
TIME_SUPPORT_COMPARISON_FILE = (
    OUTPUT_ROOT / "09_Time_Support_Sensitivity_Comparison.csv"
)
TEMPORAL_SPATIAL_COMPARISON_FILE = (
    OUTPUT_ROOT / "10_Temporal_vs_Spatial_OOF_Moran_Comparison.csv"
)
PROTOCOL_ALIGNMENT_FILE = OUTPUT_ROOT / "11_Regression_Protocol_Alignment_Audit.csv"
OUTPUT_MANIFEST_FILE = OUTPUT_ROOT / "12_Output_Manifest.csv"


# =============================================================================
# Locked settings
# =============================================================================

STEP10_CODE_VERSION = "2026-06-30_RF_UNIFIED_TEMPORAL_SPATIAL_OOF_MORAN_V1B_UINT32_SEED_AUDIT"
EXPECTED_REGRESSION_STEP08_VERSION = (
    "2026-07-08_RELEASE_OPTIONAL_LEGACY_FOLD_COMPATIBILITY"
)
EXPECTED_REGRESSION_STEP08_SCRIPT_SHA256 = (
    "3c86c19ab60480591ee8950c4307d40b3e9c55b73255305cc01f7277fa9498b7"
)
EXPECTED_SPATIAL_MAPPING_SHA256 = (
    "715f9fe342eec78d936d223a98f5a4d0378281d3a3fa420ed250ba07fe056cfe"
)
EXPECTED_SPATIAL_BLOCK_FILE_SHA256 = (
    "da752eff8f734d29e0ff7b5535c93277b4b146b0b5a84e27efe3f08d17f61a11"
)
EXPECTED_GRID_COUNT = 4_982
EXPECTED_COMPLETE_25Y_GRID_COUNT = 4_924
EXPECTED_PARTIAL_TIME_SUPPORT_GRID_COUNT = 58
EXPECTED_ROWS_PER_SEASON_RESPONSE = 123_729
EXPECTED_ROWS_PER_OOF_FILE = EXPECTED_ROWS_PER_SEASON_RESPONSE * 6
EXPECTED_BLOCK_GRID_COUNTS = {1: 912, 2: 1042, 3: 735, 4: 1222, 5: 1071}
EXPECTED_YEARS = set(range(2001, 2026))

MODEL_ROLE = "Final_Hurdle_RF"
MODEL_ID = "Final_Hurdle_RF"
MODEL_STRUCTURE = "Two_Part_Hurdle_RF"
CANDIDATE_ID = "FINAL_LOCKED_HURDLE_RF"
VALIDATION_SCHEMES = ["Temporal", "Spatial"]
SEASONS = {1: "Spring", 2: "Summer", 3: "Autumn"}
RF_RESPONSES = ["FCD", "BAD"]
COUNT_RESPONSE_MAP = {"FCD": "Fire_Count", "BAD": "Burned_Pixel_Count"}
RATE_NAMES = {"Fire_Count": "FCD", "Burned_Pixel_Count": "BAD"}
COORDINATE_COLUMNS = [
    "Centroid_X_UTM52_m",
    "Centroid_Y_UTM52_m",
    "Longitude",
    "Latitude",
]

MAIN_SAMPLE = "Main_4982_Available_Years"
COMPLETE_SAMPLE = "Complete_4924_25_Years"
SAMPLE_DEFINITIONS = [MAIN_SAMPLE, COMPLETE_SAMPLE]
PRIMARY_K_NEIGHBORS = 8
PRIMARY_PERMUTATIONS = 9_999
SENSITIVITY_K_NEIGHBORS = [4, 12]
SENSITIVITY_PERMUTATIONS = 999
RANDOM_SEED = 2026
FDR_ALPHA = 0.05
PRIMARY_RESIDUAL_FIELD = "Mean_Residual_Rate"
PRIMARY_RESIDUAL_DEFINITION = (
    "Arithmetic mean across the relevant OOF years of Observed_Rate minus "
    "Predicted_Rate by GRID_UID"
)


# =============================================================================
# Generic helpers
# =============================================================================


def log(message: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}"
    print(line, flush=True)
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def require_files(paths: Iterable[Path]) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required files are missing:\n" + "\n".join(str(path) for path in missing)
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_regression_protocol():
    observed_hash = sha256_file(REGRESSION_STEP08_SCRIPT)
    if observed_hash != EXPECTED_REGRESSION_STEP08_SCRIPT_SHA256:
        raise ValueError(
            "Regression Step 08 script differs from the locked release source.\n"
            f"Expected SHA256: {EXPECTED_REGRESSION_STEP08_SCRIPT_SHA256}\n"
            f"Observed SHA256: {observed_hash}"
        )
    specification = importlib.util.spec_from_file_location(
        "locked_regression_step08_protocol", REGRESSION_STEP08_SCRIPT
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("Could not import the locked regression Step 08 script.")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    if module.STEP08_CODE_VERSION != EXPECTED_REGRESSION_STEP08_VERSION:
        raise ValueError(
            "Unexpected regression Step 08 protocol version: "
            f"{module.STEP08_CODE_VERSION}"
        )
    settings = {
        "PRIMARY_K_NEIGHBORS": PRIMARY_K_NEIGHBORS,
        "PRIMARY_PERMUTATIONS": PRIMARY_PERMUTATIONS,
        "SENSITIVITY_K_NEIGHBORS": SENSITIVITY_K_NEIGHBORS,
        "SENSITIVITY_PERMUTATIONS": SENSITIVITY_PERMUTATIONS,
        "RANDOM_SEED": RANDOM_SEED,
        "MAIN_SAMPLE": MAIN_SAMPLE,
        "COMPLETE_SAMPLE": COMPLETE_SAMPLE,
        "PRIMARY_RESIDUAL_FIELD": PRIMARY_RESIDUAL_FIELD,
    }
    for name, expected in settings.items():
        if getattr(module, name) != expected:
            raise ValueError(
                f"Regression Step 08 setting {name} differs from the locked RF "
                f"alignment setting. Expected {expected!r}, observed "
                f"{getattr(module, name)!r}."
            )
    return module


def spatial_mapping_sha256(blocks: pd.DataFrame) -> str:
    canonical = (
        blocks[["GRID_UID", "Spatial_Block"]]
        .assign(
            GRID_UID=lambda frame: frame["GRID_UID"].astype(str),
            Spatial_Block=lambda frame: pd.to_numeric(
                frame["Spatial_Block"], errors="raise"
            ).astype(int),
        )
        .sort_values("GRID_UID")
        .reset_index(drop=True)
    )
    payload = canonical.to_csv(
        index=False, header=False, lineterminator="\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def maximum_absolute_difference(left: Sequence[float], right: Sequence[float]) -> float:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    if a.shape != b.shape:
        return np.inf
    finite_equal = np.array_equal(np.isfinite(a), np.isfinite(b))
    if not finite_equal:
        return np.inf
    finite = np.isfinite(a) & np.isfinite(b)
    if not finite.any():
        return 0.0
    return float(np.max(np.abs(a[finite] - b[finite])))


def build_output_manifest(paths: Sequence[Path]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Expected output was not created:\n{path}")
        rows.append(
            {
                "Output_File": path.name,
                "Relative_Path": str(path.relative_to(OUTPUT_ROOT)),
                "Size_Bytes": int(path.stat().st_size),
                "SHA256": sha256_file(path),
            }
        )
    return pd.DataFrame(rows)


# =============================================================================
# Shared assignment and regression protocol reference
# =============================================================================


def read_and_verify_blocks() -> Tuple[pd.DataFrame, Dict[str, object]]:
    blocks = pd.read_csv(SPATIAL_BLOCK_FILE)
    required = {
        "GRID_UID",
        "Country",
        "GRID_ID",
        "Spatial_Block",
        *COORDINATE_COLUMNS,
    }
    missing = sorted(required - set(blocks.columns))
    if missing:
        raise ValueError(
            "Shared spatial assignment is missing columns:\n" + "\n".join(missing)
        )
    blocks["GRID_UID"] = blocks["GRID_UID"].astype(str)
    blocks["Spatial_Block"] = pd.to_numeric(
        blocks["Spatial_Block"], errors="raise"
    ).astype(int)
    if len(blocks) != EXPECTED_GRID_COUNT or blocks["GRID_UID"].nunique() != EXPECTED_GRID_COUNT:
        raise ValueError("Shared spatial assignment does not contain 4,982 unique grids.")
    if blocks["GRID_UID"].duplicated().any():
        raise ValueError("Shared spatial assignment contains duplicate GRID_UID values.")
    counts = {
        int(key): int(value)
        for key, value in blocks["Spatial_Block"].value_counts().sort_index().items()
    }
    if counts != EXPECTED_BLOCK_GRID_COUNTS:
        raise ValueError(
            f"Shared spatial-block counts differ. Expected {EXPECTED_BLOCK_GRID_COUNTS}; "
            f"observed {counts}."
        )
    mapping_hash = spatial_mapping_sha256(blocks)
    file_hash = sha256_file(SPATIAL_BLOCK_FILE)
    if mapping_hash != EXPECTED_SPATIAL_MAPPING_SHA256:
        raise ValueError("Shared spatial assignment semantic hash differs from the lock.")
    if file_hash != EXPECTED_SPATIAL_BLOCK_FILE_SHA256:
        raise ValueError("Shared spatial assignment byte hash differs from the lock.")
    return (
        blocks.sort_values("GRID_UID").reset_index(drop=True),
        {
            "Spatial_Block_Source": str(SPATIAL_BLOCK_FILE),
            "Spatial_Mapping_SHA256": mapping_hash,
            "Spatial_Block_File_SHA256": file_hash,
            "GRID_UID_Count": EXPECTED_GRID_COUNT,
            "Block_Counts": counts,
        },
    )


def read_regression_reference() -> Dict[str, object]:
    method = json.loads(REGRESSION_STEP08_METHOD_FILE.read_text(encoding="utf-8"))
    if method.get("Step08_Code_Version") != EXPECTED_REGRESSION_STEP08_VERSION:
        raise ValueError("Regression Step 08 output is not the locked release version.")
    support = pd.read_csv(REGRESSION_TIME_SUPPORT_FILE)
    if len(support) != EXPECTED_GRID_COUNT:
        raise ValueError("Regression time-support reference does not contain 4,982 grids.")
    if int(support["Complete_25_Years"].astype(bool).sum()) != EXPECTED_COMPLETE_25Y_GRID_COUNT:
        raise ValueError("Regression complete-25-year reference count is not 4,924.")
    input_qa = pd.read_csv(REGRESSION_INPUT_QA_FILE)
    file_level = input_qa.loc[
        input_qa["Rows"].notna() & input_qa["OOF_Key_Signature"].notna()
    ].copy()
    expected_signatures = set(file_level["OOF_Key_Signature"].astype(str))
    if len(expected_signatures) != 1:
        raise ValueError("Regression OOF key signature is not unique across inputs.")
    primary = pd.read_csv(REGRESSION_PRIMARY_MORAN_FILE)
    return {
        "Method": method,
        "Time_Support": support.sort_values("GRID_UID").reset_index(drop=True),
        "Expected_OOF_Key_Signature": next(iter(expected_signatures)),
        "Primary_Moran": primary,
    }


# =============================================================================
# RF OOF reading and normalization
# =============================================================================


def read_rf_oof(
    path: Path,
    validation_scheme: str,
    blocks: pd.DataFrame,
    protocol,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    data = pd.read_csv(path, compression="gzip", low_memory=False)
    required = {
        "GRID_UID",
        "Country",
        "GRID_ID",
        "Year",
        "Season",
        "Temporal_Fold",
        "Spatial_Block",
        *COORDINATE_COLUMNS,
        "Validation_Scheme",
        "Validation_Fold",
        "Response",
        "Observed",
        "Expected_Prediction",
        "Residual",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"{path.name} is missing columns:\n" + "\n".join(missing))
    if len(data) != EXPECTED_ROWS_PER_OOF_FILE:
        raise ValueError(
            f"Expected {EXPECTED_ROWS_PER_OOF_FILE:,} rows in {path.name}, "
            f"found {len(data):,}."
        )
    data["GRID_UID"] = data["GRID_UID"].astype(str)
    for column in [
        "Year",
        "Season",
        "Temporal_Fold",
        "Spatial_Block",
        "Validation_Fold",
    ]:
        data[column] = pd.to_numeric(data[column], errors="raise").astype(int)
    for column in ["Observed", "Expected_Prediction", "Residual", *COORDINATE_COLUMNS]:
        data[column] = pd.to_numeric(data[column], errors="raise")
    if not np.isfinite(
        data[["Observed", "Expected_Prediction", "Residual", *COORDINATE_COLUMNS]].to_numpy(dtype=float)
    ).all():
        raise ValueError(f"{path.name} contains non-finite numeric values.")
    if set(data["Validation_Scheme"].astype(str)) != {validation_scheme}:
        raise ValueError(f"{path.name} contains an unexpected validation scheme.")
    if set(data["Season"].unique()) != set(SEASONS):
        raise ValueError(f"{path.name} does not contain seasons 1-3 exactly.")
    if set(data["Response"].astype(str)) != set(RF_RESPONSES):
        raise ValueError(f"{path.name} does not contain FCD and BAD exactly.")
    if set(data["Year"].unique()) != EXPECTED_YEARS:
        raise ValueError(f"{path.name} does not span 2001-2025.")
    if set(data["Temporal_Fold"].unique()) != {1, 2, 3, 4, 5}:
        raise ValueError(f"{path.name} does not contain temporal folds 1-5.")
    if set(data["Spatial_Block"].unique()) != {1, 2, 3, 4, 5}:
        raise ValueError(f"{path.name} does not contain spatial blocks 1-5.")
    if data.duplicated(["GRID_UID", "Year", "Season", "Response"], keep=False).any():
        raise ValueError(f"{path.name} contains duplicate OOF keys.")
    combination_counts = data.groupby(["Season", "Response"]).size()
    if len(combination_counts) != 6 or not (
        combination_counts == EXPECTED_ROWS_PER_SEASON_RESPONSE
    ).all():
        raise ValueError(
            f"{path.name} does not contain {EXPECTED_ROWS_PER_SEASON_RESPONSE:,} "
            "rows for every season-response combination."
        )
    residual_error = np.abs(
        data["Residual"].to_numpy(dtype=float)
        - (
            data["Observed"].to_numpy(dtype=float)
            - data["Expected_Prediction"].to_numpy(dtype=float)
        )
    )
    max_residual_error = float(residual_error.max())
    if max_residual_error > 1e-10:
        raise ValueError(
            f"{path.name} residual identity error is {max_residual_error:.3e}."
        )

    block_lookup = blocks.rename(
        columns={
            "Country": "Locked_Country",
            "GRID_ID": "Locked_GRID_ID",
            "Spatial_Block": "Locked_Spatial_Block",
            "Centroid_X_UTM52_m": "Locked_Centroid_X_UTM52_m",
            "Centroid_Y_UTM52_m": "Locked_Centroid_Y_UTM52_m",
            "Longitude": "Locked_Longitude",
            "Latitude": "Locked_Latitude",
        }
    )
    data = data.merge(block_lookup, on="GRID_UID", how="left", validate="many_to_one")
    if data["Locked_Spatial_Block"].isna().any():
        raise ValueError(f"{path.name} contains grids absent from the shared assignment.")
    if not (
        data["Country"].astype(str).to_numpy()
        == data["Locked_Country"].astype(str).to_numpy()
    ).all():
        raise ValueError(f"{path.name} Country differs from the shared assignment.")
    if not (
        data["GRID_ID"].astype(str).to_numpy()
        == data["Locked_GRID_ID"].astype(str).to_numpy()
    ).all():
        raise ValueError(f"{path.name} GRID_ID differs from the shared assignment.")
    if not (
        data["Spatial_Block"].to_numpy(dtype=int)
        == data["Locked_Spatial_Block"].to_numpy(dtype=int)
    ).all():
        raise ValueError(f"{path.name} Spatial_Block differs from Step 07.")
    for column in COORDINATE_COLUMNS:
        difference = np.abs(
            data[column].to_numpy(dtype=float)
            - data[f"Locked_{column}"].to_numpy(dtype=float)
        )
        if float(difference.max()) > 1e-8:
            raise ValueError(f"{path.name} {column} differs from Step 07.")

    data["Count_Response"] = data["Response"].map(COUNT_RESPONSE_MAP)
    data["Rate_Scale_Name"] = data["Response"].astype(str)
    data["Season_Label"] = data["Season"].map(SEASONS)
    data["Model_Role"] = MODEL_ROLE
    data["Model_ID"] = MODEL_ID
    data["Model_Structure"] = MODEL_STRUCTURE
    data["Candidate_ID"] = CANDIDATE_ID
    data["Observed_Rate"] = data["Observed"].astype(float)
    data["Predicted_Rate"] = data["Expected_Prediction"].astype(float)
    data["Residual_Rate"] = data["Residual"].astype(float)

    key_signature = protocol.order_independent_row_signature(
        data, ["GRID_UID", "Year", "Season", "Count_Response"]
    )
    selected_columns = [
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
        *COORDINATE_COLUMNS,
        "Count_Response",
        "Rate_Scale_Name",
        "Model_Role",
        "Model_ID",
        "Model_Structure",
        "Candidate_ID",
        "Observed_Rate",
        "Predicted_Rate",
        "Residual_Rate",
    ]
    normalized = data[selected_columns].copy()
    qa = {
        "Validation_Scheme": validation_scheme,
        "Model_Role": MODEL_ROLE,
        "Input_File": str(path),
        "Input_File_SHA256": sha256_file(path),
        "Rows": int(len(normalized)),
        "Unique_OOF_Keys": int(
            normalized[["GRID_UID", "Year", "Season", "Count_Response"]]
            .drop_duplicates()
            .shape[0]
        ),
        "GRID_UID_Count": int(normalized["GRID_UID"].nunique()),
        "Minimum_Year": int(normalized["Year"].min()),
        "Maximum_Year": int(normalized["Year"].max()),
        "Season_Count": int(normalized["Season"].nunique()),
        "Response_Count": int(normalized["Count_Response"].nunique()),
        "Temporal_Fold_Count": int(normalized["Temporal_Fold"].nunique()),
        "Spatial_Block_Count": int(normalized["Spatial_Block"].nunique()),
        "Maximum_Residual_Identity_Error": max_residual_error,
        "OOF_Key_Signature": key_signature,
    }
    return normalized, qa


# =============================================================================
# Residual aggregation and time support
# =============================================================================


def aggregate_grid_residuals(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby(
        [
            "Validation_Scheme",
            "Model_Role",
            "Model_ID",
            "Model_Structure",
            "Candidate_ID",
            "Season",
            "Season_Label",
            "Count_Response",
            "Rate_Scale_Name",
            "GRID_UID",
            "Country",
            "GRID_ID",
            "Spatial_Block",
            *COORDINATE_COLUMNS,
        ],
        as_index=False,
    ).agg(
        Row_Count=("Year", "size"),
        Years_Observed=("Year", "nunique"),
        First_Year=("Year", "min"),
        Last_Year=("Year", "max"),
        Positive_Observed_Years=("Observed_Rate", lambda values: int((values > 0).sum())),
        Mean_Observed_Rate=("Observed_Rate", "mean"),
        Mean_Predicted_Rate=("Predicted_Rate", "mean"),
        Mean_Residual_Rate=("Residual_Rate", "mean"),
        Median_Residual_Rate=("Residual_Rate", "median"),
        Residual_Rate_SD=("Residual_Rate", "std"),
        Mean_Absolute_Residual_Rate=("Residual_Rate", lambda values: float(np.mean(np.abs(values)))),
        Mean_Squared_Residual_Rate=("Residual_Rate", lambda values: float(np.mean(np.square(values)))),
    )
    grouped["Residual_Rate_RMSE_Across_Years"] = np.sqrt(
        grouped["Mean_Squared_Residual_Rate"].to_numpy(dtype=float)
    )
    expected = EXPECTED_GRID_COUNT * 6
    if len(grouped) != expected:
        raise ValueError(
            f"Expected {expected:,} grid-season-response residual rows per OOF "
            f"scheme, found {len(grouped):,}."
        )
    return grouped


def build_time_support_audit(
    temporal: pd.DataFrame,
    spatial: pd.DataFrame,
    residuals: pd.DataFrame,
    blocks: pd.DataFrame,
    regression_support: pd.DataFrame,
) -> pd.DataFrame:
    signatures: List[str] = []
    for frame in [temporal, spatial]:
        for _, subset in frame.groupby(["Season", "Count_Response"], sort=True):
            signatures.append(
                hashlib.sha256(
                    subset[["GRID_UID", "Year"]]
                    .drop_duplicates()
                    .sort_values(["GRID_UID", "Year"])
                    .to_csv(index=False, header=False, lineterminator="\n")
                    .encode("utf-8")
                ).hexdigest()
            )
    if len(set(signatures)) != 1:
        raise ValueError(
            "GRID_UID-Year availability differs between RF OOF scheme, season, "
            "or response."
        )
    reference_panel = temporal.loc[
        (temporal["Season"] == 1) & (temporal["Count_Response"] == "Fire_Count"),
        ["GRID_UID", "Year"],
    ].drop_duplicates()
    observed_years = (
        reference_panel.groupby("GRID_UID")["Year"]
        .apply(lambda values: sorted(set(int(value) for value in values)))
        .rename("Observed_Years_List")
    )
    support = blocks.set_index("GRID_UID").join(observed_years, how="left")
    if support["Observed_Years_List"].isna().any():
        raise ValueError("Some shared grids are absent from the RF OOF panel.")
    support["Years_Observed"] = support["Observed_Years_List"].map(len).astype(int)
    support["Missing_Years_List"] = support["Observed_Years_List"].map(
        lambda years: sorted(EXPECTED_YEARS - set(years))
    )
    support["Observed_Years"] = support["Observed_Years_List"].map(
        lambda years: ";".join(str(year) for year in years)
    )
    support["Missing_Years"] = support["Missing_Years_List"].map(
        lambda years: ";".join(str(year) for year in years)
    )
    support["Missing_Year_Count"] = support["Missing_Years_List"].map(len).astype(int)
    support["Complete_25_Years"] = support["Years_Observed"] == 25
    residual_years = residuals.groupby("GRID_UID")["Years_Observed"].agg(
        ["min", "max", "nunique"]
    )
    support = support.join(residual_years, how="left").rename(
        columns={
            "min": "Minimum_Years_Observed_Across_All_12_Combinations",
            "max": "Maximum_Years_Observed_Across_All_12_Combinations",
            "nunique": "Distinct_Years_Observed_Counts_Across_Combinations",
        }
    )
    if not (
        support["Minimum_Years_Observed_Across_All_12_Combinations"].astype(int)
        == support["Years_Observed"]
    ).all() or not (
        support["Maximum_Years_Observed_Across_All_12_Combinations"].astype(int)
        == support["Years_Observed"]
    ).all():
        raise ValueError("RF temporal support differs across OOF scheme or response.")
    support = support.drop(columns=["Observed_Years_List", "Missing_Years_List"])
    support = support.reset_index().sort_values("GRID_UID").reset_index(drop=True)
    complete_count = int(support["Complete_25_Years"].sum())
    partial_count = int((~support["Complete_25_Years"]).sum())
    if complete_count != EXPECTED_COMPLETE_25Y_GRID_COUNT or partial_count != EXPECTED_PARTIAL_TIME_SUPPORT_GRID_COUNT:
        raise ValueError(
            f"RF time support is {complete_count} complete and {partial_count} "
            "partial grids; expected 4,924 and 58."
        )

    compare_columns = [
        "GRID_UID",
        "Years_Observed",
        "Observed_Years",
        "Missing_Years",
        "Missing_Year_Count",
        "Complete_25_Years",
    ]
    left = support[compare_columns].sort_values("GRID_UID").reset_index(drop=True)
    right = regression_support[compare_columns].sort_values("GRID_UID").reset_index(drop=True)
    for column in ["Observed_Years", "Missing_Years"]:
        left[column] = left[column].fillna("").astype(str)
        right[column] = right[column].fillna("").astype(str)
    for column in ["Years_Observed", "Missing_Year_Count"]:
        left[column] = pd.to_numeric(left[column], errors="raise").astype(int)
        right[column] = pd.to_numeric(right[column], errors="raise").astype(int)
    left["Complete_25_Years"] = left["Complete_25_Years"].astype(bool)
    right["Complete_25_Years"] = right["Complete_25_Years"].astype(bool)
    if not left.equals(right):
        mismatch = left.merge(right, on="GRID_UID", suffixes=("_RF", "_Regression"))
        bad = np.zeros(len(mismatch), dtype=bool)
        for column in compare_columns[1:]:
            bad |= mismatch[f"{column}_RF"].astype(str).to_numpy() != mismatch[
                f"{column}_Regression"
            ].astype(str).to_numpy()
        mismatch.loc[bad].to_csv(
            OUTPUT_ROOT / "ERROR_RF_vs_Regression_Time_Support_Mismatch.csv",
            index=False,
            encoding="utf-8-sig",
        )
        raise ValueError(
            "RF time-support membership differs from regression Step 08."
        )
    return support


# =============================================================================
# Moran calculation
# =============================================================================


def residual_vector(
    residuals: pd.DataFrame,
    sample_blocks: pd.DataFrame,
    sample_definition: str,
    validation_scheme: str,
    season: int,
    response: str,
) -> Tuple[np.ndarray, Dict[str, object]]:
    subset = residuals.loc[
        (residuals["Validation_Scheme"] == validation_scheme)
        & (residuals["Season"] == season)
        & (residuals["Count_Response"] == response)
    ].copy()
    subset = sample_blocks[["GRID_UID"]].merge(
        subset, on="GRID_UID", how="left", validate="one_to_one"
    )
    if subset[PRIMARY_RESIDUAL_FIELD].isna().any():
        raise ValueError(
            f"Missing RF residuals for {sample_definition}, {validation_scheme}, "
            f"{season}, {response}."
        )
    values = subset[PRIMARY_RESIDUAL_FIELD].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("RF Moran residual vector contains non-finite values.")
    years = subset["Years_Observed"].to_numpy(dtype=int)
    metadata = {
        "GRID_UID_Count": int(len(subset)),
        "Years_Observed_Minimum": int(years.min()),
        "Years_Observed_Median": float(np.median(years)),
        "Years_Observed_Maximum": int(years.max()),
        "Complete_25_Year_GRID_UID_Count": int((years == 25).sum()),
        "Residual_Mean": float(values.mean()),
        "Residual_SD": float(values.std(ddof=1)),
        "Residual_Minimum": float(values.min()),
        "Residual_Maximum": float(values.max()),
        "Residual_Mean_Absolute_Value": float(np.mean(np.abs(values))),
        "Residual_Root_Mean_Square": float(np.sqrt(np.mean(np.square(values)))),
    }
    return values, metadata


def apply_primary_fdr(frame: pd.DataFrame, protocol) -> pd.DataFrame:
    result = frame.copy()
    result["Permutation_P_TwoSided_Global_BH_FDR"] = protocol.bh_adjust(
        result["Permutation_P_TwoSided"].to_numpy(dtype=float)
    )
    result["Permutation_P_TwoSided_Validation_Scheme_BH_FDR"] = np.nan
    for _, index in result.groupby("Validation_Scheme").groups.items():
        result.loc[index, "Permutation_P_TwoSided_Validation_Scheme_BH_FDR"] = (
            protocol.bh_adjust(
                result.loc[index, "Permutation_P_TwoSided"].to_numpy(dtype=float)
            )
        )
    result["Permutation_P_TwoSided_Scheme_Model_BH_FDR"] = np.nan
    for _, index in result.groupby(["Validation_Scheme", "Model_Role"]).groups.items():
        result.loc[index, "Permutation_P_TwoSided_Scheme_Model_BH_FDR"] = (
            protocol.bh_adjust(
                result.loc[index, "Permutation_P_TwoSided"].to_numpy(dtype=float)
            )
        )
    result["Significant_Raw_P_0p05"] = result["Permutation_P_TwoSided"] < FDR_ALPHA
    result["Significant_Global_BH_FDR_0p05"] = (
        result["Permutation_P_TwoSided_Global_BH_FDR"] < FDR_ALPHA
    )
    result["Significant_Validation_Scheme_BH_FDR_0p05"] = (
        result["Permutation_P_TwoSided_Validation_Scheme_BH_FDR"] < FDR_ALPHA
    )
    result["Significant_Scheme_Model_BH_FDR_0p05"] = (
        result["Permutation_P_TwoSided_Scheme_Model_BH_FDR"] < FDR_ALPHA
    )
    result["Autocorrelation_Direction"] = np.where(
        result["Moran_I"] > result["Expected_I"],
        "Positive",
        np.where(
            result["Moran_I"] < result["Expected_I"],
            "Negative",
            "At_Expectation",
        ),
    )
    return result


def apply_sensitivity_fdr(frame: pd.DataFrame, protocol) -> pd.DataFrame:
    result = frame.copy()
    result["Permutation_P_TwoSided_Sample_K_BH_FDR"] = np.nan
    for _, index in result.groupby(["Sample_Definition", "K_Neighbors"]).groups.items():
        result.loc[index, "Permutation_P_TwoSided_Sample_K_BH_FDR"] = (
            protocol.bh_adjust(
                result.loc[index, "Permutation_P_TwoSided"].to_numpy(dtype=float)
            )
        )
    result["Significant_Raw_P_0p05"] = result["Permutation_P_TwoSided"] < FDR_ALPHA
    result["Significant_Sample_K_BH_FDR_0p05"] = (
        result["Permutation_P_TwoSided_Sample_K_BH_FDR"] < FDR_ALPHA
    )
    result["Autocorrelation_Direction"] = np.where(
        result["Moran_I"] > result["Expected_I"],
        "Positive",
        np.where(
            result["Moran_I"] < result["Expected_I"],
            "Negative",
            "At_Expectation",
        ),
    )
    return result


def run_moran_tests(
    residuals: pd.DataFrame,
    sample_blocks: Mapping[str, pd.DataFrame],
    weight_matrices: Mapping[Tuple[str, int], sparse.csr_matrix],
    protocol,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    primary_rows: List[Dict[str, object]] = []
    complete_rows: List[Dict[str, object]] = []
    knn_rows: List[Dict[str, object]] = []
    for sample_definition in SAMPLE_DEFINITIONS:
        blocks = sample_blocks[sample_definition]
        for validation_scheme in VALIDATION_SCHEMES:
            for season, season_label in SEASONS.items():
                for response in ["Fire_Count", "Burned_Pixel_Count"]:
                    primary_seed = protocol.deterministic_seed(
                        RANDOM_SEED,
                        season,
                        response,
                        PRIMARY_RESIDUAL_FIELD,
                        PRIMARY_K_NEIGHBORS,
                        "primary",
                    )
                    values, vector_metadata = residual_vector(
                        residuals,
                        blocks,
                        sample_definition,
                        validation_scheme,
                        season,
                        response,
                    )
                    log(
                        f"Moran primary: {sample_definition}, {validation_scheme}, "
                        f"{season_label}, {response}, k=8."
                    )
                    moran = protocol.morans_i_permutation(
                        values,
                        weight_matrices[(sample_definition, PRIMARY_K_NEIGHBORS)],
                        permutations=PRIMARY_PERMUTATIONS,
                        seed=primary_seed,
                    )
                    row = {
                        "Analysis_Type": (
                            "Primary_Main_Sample"
                            if sample_definition == MAIN_SAMPLE
                            else "Complete_25Y_Time_Support_Sensitivity"
                        ),
                        "Sample_Definition": sample_definition,
                        "Validation_Scheme": validation_scheme,
                        "Model_Role": MODEL_ROLE,
                        "Season_Code": season,
                        "Season_Label": season_label,
                        "Count_Response": response,
                        "Rate_Scale_Name": RATE_NAMES[response],
                        "Residual_Field": PRIMARY_RESIDUAL_FIELD,
                        "Residual_Definition": PRIMARY_RESIDUAL_DEFINITION,
                        "K_Neighbors": PRIMARY_K_NEIGHBORS,
                        "Weights": (
                            "8-nearest-neighbor row-standardized UTM52N "
                            "centroid weights"
                        ),
                        "Permutations": PRIMARY_PERMUTATIONS,
                        "Permutation_Seed": primary_seed,
                        **vector_metadata,
                        **moran,
                    }
                    if sample_definition == MAIN_SAMPLE:
                        primary_rows.append(row)
                    else:
                        complete_rows.append(row)

                    for k in SENSITIVITY_K_NEIGHBORS:
                        seed = protocol.deterministic_seed(
                            RANDOM_SEED,
                            season,
                            response,
                            PRIMARY_RESIDUAL_FIELD,
                            k,
                            "weights_sensitivity",
                        )
                        log(
                            f"Moran k sensitivity: {sample_definition}, "
                            f"{validation_scheme}, {season_label}, {response}, k={k}."
                        )
                        sensitivity = protocol.morans_i_permutation(
                            values,
                            weight_matrices[(sample_definition, k)],
                            permutations=SENSITIVITY_PERMUTATIONS,
                            seed=seed,
                        )
                        knn_rows.append(
                            {
                                "Analysis_Type": "Weights_Sensitivity",
                                "Sample_Definition": sample_definition,
                                "Validation_Scheme": validation_scheme,
                                "Model_Role": MODEL_ROLE,
                                "Season_Code": season,
                                "Season_Label": season_label,
                                "Count_Response": response,
                                "Rate_Scale_Name": RATE_NAMES[response],
                                "Residual_Field": PRIMARY_RESIDUAL_FIELD,
                                "Residual_Definition": PRIMARY_RESIDUAL_DEFINITION,
                                "K_Neighbors": k,
                                "Weights": (
                                    f"{k}-nearest-neighbor row-standardized "
                                    "UTM52N centroid weights"
                                ),
                                "Permutations": SENSITIVITY_PERMUTATIONS,
                                "Permutation_Seed": seed,
                                **vector_metadata,
                                **sensitivity,
                            }
                        )
    primary = pd.DataFrame(primary_rows)
    complete = pd.DataFrame(complete_rows)
    knn = pd.DataFrame(knn_rows)
    if len(primary) != 12 or len(complete) != 12 or len(knn) != 48:
        raise RuntimeError(
            "Unexpected Moran result counts: "
            f"primary={len(primary)}, complete={len(complete)}, kNN={len(knn)}."
        )
    primary = apply_primary_fdr(primary, protocol)
    complete = apply_primary_fdr(complete, protocol)
    knn = apply_sensitivity_fdr(knn, protocol)
    primary = primary.sort_values(
        ["Validation_Scheme", "Season_Code", "Count_Response"]
    ).reset_index(drop=True)
    complete = complete.sort_values(
        ["Validation_Scheme", "Season_Code", "Count_Response"]
    ).reset_index(drop=True)
    knn = knn.sort_values(
        [
            "Sample_Definition",
            "Validation_Scheme",
            "Season_Code",
            "Count_Response",
            "K_Neighbors",
        ]
    ).reset_index(drop=True)
    return primary, complete, knn


# =============================================================================
# Comparison tables
# =============================================================================


def build_legacy_comparison(primary: pd.DataFrame) -> pd.DataFrame:
    legacy = pd.read_csv(LEGACY_MORAN_FILE)
    current = primary.loc[
        primary["Validation_Scheme"] == "Temporal",
        [
            "Season_Code",
            "Season_Label",
            "Rate_Scale_Name",
            "GRID_UID_Count",
            "Moran_I",
            "Expected_I",
            "Permutation_P_TwoSided",
            "Permutation_Mean",
            "Permutation_SD",
            "Permutations",
        ],
    ].copy()
    legacy = legacy.rename(
        columns={
            "Season": "Season_Code",
            "Response": "Rate_Scale_Name",
            "GRID_UID_Count": "Legacy_GRID_UID_Count",
            "Moran_I": "Legacy_Moran_I",
            "Expected_I": "Legacy_Expected_I",
            "Permutation_P_TwoSided": "Legacy_Permutation_P_TwoSided",
            "Permutation_Mean": "Legacy_Permutation_Mean",
            "Permutation_SD": "Legacy_Permutation_SD",
            "Permutations": "Legacy_Permutations",
        }
    )
    result = current.merge(
        legacy[
            [
                "Season_Code",
                "Rate_Scale_Name",
                "Legacy_GRID_UID_Count",
                "Legacy_Moran_I",
                "Legacy_Expected_I",
                "Legacy_Permutation_P_TwoSided",
                "Legacy_Permutation_Mean",
                "Legacy_Permutation_SD",
                "Legacy_Permutations",
            ]
        ],
        on=["Season_Code", "Rate_Scale_Name"],
        how="inner",
        validate="one_to_one",
    )
    result = result.rename(
        columns={
            "Moran_I": "Unified_Moran_I",
            "Expected_I": "Unified_Expected_I",
            "Permutation_P_TwoSided": "Unified_Permutation_P_TwoSided",
            "Permutation_Mean": "Unified_Permutation_Mean",
            "Permutation_SD": "Unified_Permutation_SD",
            "Permutations": "Unified_Permutations",
        }
    )
    result["Unified_minus_Legacy_Moran_I"] = (
        result["Unified_Moran_I"] - result["Legacy_Moran_I"]
    )
    result["Moran_I_Reproduced"] = (
        np.abs(result["Unified_minus_Legacy_Moran_I"]) <= 1e-12
    )
    result["Interpretation"] = (
        "Moran_I must reproduce the legacy calculation exactly; permutation "
        "p-values may differ because the unified analysis uses 9,999 rather "
        "than 999 permutations and a shared deterministic seed rule."
    )
    if len(result) != 6 or not result["Moran_I_Reproduced"].all():
        raise ValueError("Unified temporal RF Moran's I did not reproduce legacy values.")
    return result.sort_values(["Season_Code", "Rate_Scale_Name"]).reset_index(drop=True)


def build_time_support_comparison(all_primary: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "Validation_Scheme",
        "Model_Role",
        "Season_Code",
        "Season_Label",
        "Count_Response",
        "Rate_Scale_Name",
        "K_Neighbors",
        "Permutations",
    ]
    metrics = [
        "GRID_UID_Count",
        "Moran_I",
        "Expected_I",
        "Moran_Excess_Over_Expected",
        "Permutation_P_TwoSided",
        "Residual_Mean_Absolute_Value",
        "Residual_Root_Mean_Square",
    ]
    main = all_primary.loc[
        all_primary["Sample_Definition"] == MAIN_SAMPLE, [*keys, *metrics]
    ].rename(columns={metric: f"Main_{metric}" for metric in metrics})
    complete = all_primary.loc[
        all_primary["Sample_Definition"] == COMPLETE_SAMPLE, [*keys, *metrics]
    ].rename(columns={metric: f"Complete25Y_{metric}" for metric in metrics})
    result = main.merge(complete, on=keys, how="inner", validate="one_to_one")
    result["Complete25Y_minus_Main_Moran_I"] = (
        result["Complete25Y_Moran_I"] - result["Main_Moran_I"]
    )
    result["Moran_Direction_Consistent"] = (
        np.sign(result["Complete25Y_Moran_Excess_Over_Expected"])
        == np.sign(result["Main_Moran_Excess_Over_Expected"])
    )
    result["Sensitivity_Note"] = (
        "The main sample uses the locked 4,982-grid common universe and "
        "available-year means. The complete sample uses the identical 4,924 "
        "25-year grids defined by regression Step 08."
    )
    return result.sort_values(
        ["Validation_Scheme", "Season_Code", "Count_Response"]
    ).reset_index(drop=True)


def build_temporal_spatial_comparison(all_primary: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "Sample_Definition",
        "Model_Role",
        "Season_Code",
        "Season_Label",
        "Count_Response",
        "Rate_Scale_Name",
        "GRID_UID_Count",
        "K_Neighbors",
        "Permutations",
    ]
    metrics = [
        "Moran_I",
        "Expected_I",
        "Moran_Excess_Over_Expected",
        "Permutation_P_TwoSided",
        "Residual_Mean_Absolute_Value",
        "Residual_Root_Mean_Square",
    ]
    temporal = all_primary.loc[
        all_primary["Validation_Scheme"] == "Temporal", [*keys, *metrics]
    ].rename(columns={metric: f"Temporal_{metric}" for metric in metrics})
    spatial = all_primary.loc[
        all_primary["Validation_Scheme"] == "Spatial", [*keys, *metrics]
    ].rename(columns={metric: f"Spatial_{metric}" for metric in metrics})
    result = temporal.merge(spatial, on=keys, how="inner", validate="one_to_one")
    result["Spatial_minus_Temporal_Moran_I"] = (
        result["Spatial_Moran_I"] - result["Temporal_Moran_I"]
    )
    result["Spatial_minus_Temporal_Residual_RMSE"] = (
        result["Spatial_Residual_Root_Mean_Square"]
        - result["Temporal_Residual_Root_Mean_Square"]
    )
    result["Comparison_Note"] = (
        "Temporal and spatial OOF answer different validation questions and "
        "are reported separately; the difference is descriptive."
    )
    return result.sort_values(
        ["Sample_Definition", "Season_Code", "Count_Response"]
    ).reset_index(drop=True)


def compare_edge_tables(
    rf_edges: pd.DataFrame,
    regression_path: Path,
    sample_definition: str,
) -> Dict[str, object]:
    """Compare RF and regression kNN edges by semantic identity.

    Neighbor membership and rank must match exactly. Floating-point distances
    are compared with a 1e-6 m absolute tolerance because SciPy/NumPy builds
    and CSV round-tripping can differ at sub-micrometre precision while
    representing the identical spatial graph. Row-standardized weights remain
    subject to a near-machine-precision tolerance.
    """
    regression = pd.read_csv(regression_path, compression="gzip", low_memory=False)
    required = {
        "Source_GRID_UID",
        "Neighbor_Rank",
        "Neighbor_GRID_UID",
        "Distance_m",
        "Distance_km",
        "Row_Standardized_Weight",
    }
    missing_left = sorted(required - set(rf_edges.columns))
    missing_right = sorted(required - set(regression.columns))
    if missing_left or missing_right:
        return {
            "Audit_Item": f"{sample_definition}_KNN8_Edge_Alignment",
            "RF_Value": int(len(rf_edges)),
            "Regression_Value": int(len(regression)),
            "Maximum_Absolute_Difference": np.inf,
            "Passed": False,
            "Details": (
                f"missing_RF_columns={missing_left}; "
                f"missing_regression_columns={missing_right}"
            ),
        }

    def normalize(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        result["Source_GRID_UID"] = result["Source_GRID_UID"].astype(str).str.strip()
        result["Neighbor_GRID_UID"] = (
            result["Neighbor_GRID_UID"].astype(str).str.strip()
        )
        result["Neighbor_Rank"] = pd.to_numeric(
            result["Neighbor_Rank"], errors="raise"
        ).astype(int)
        for column in [
            "Distance_m",
            "Distance_km",
            "Row_Standardized_Weight",
        ]:
            result[column] = pd.to_numeric(result[column], errors="raise")
        return result.sort_values(
            ["Source_GRID_UID", "Neighbor_Rank"]
        ).reset_index(drop=True)

    left = normalize(rf_edges)
    right = normalize(regression)
    same_row_count = len(left) == len(right)
    if same_row_count:
        same_sources = bool(
            np.array_equal(
                left["Source_GRID_UID"].to_numpy(dtype=str),
                right["Source_GRID_UID"].to_numpy(dtype=str),
            )
        )
        same_ranks = bool(
            np.array_equal(
                left["Neighbor_Rank"].to_numpy(dtype=int),
                right["Neighbor_Rank"].to_numpy(dtype=int),
            )
        )
        same_neighbors = bool(
            np.array_equal(
                left["Neighbor_GRID_UID"].to_numpy(dtype=str),
                right["Neighbor_GRID_UID"].to_numpy(dtype=str),
            )
        )
        max_distance_m = maximum_absolute_difference(
            left["Distance_m"], right["Distance_m"]
        )
        max_distance_km = maximum_absolute_difference(
            left["Distance_km"], right["Distance_km"]
        )
        max_weight = maximum_absolute_difference(
            left["Row_Standardized_Weight"],
            right["Row_Standardized_Weight"],
        )
    else:
        same_sources = False
        same_ranks = False
        same_neighbors = False
        max_distance_m = np.inf
        max_distance_km = np.inf
        max_weight = np.inf

    distance_tolerance_m = 1e-6
    distance_tolerance_km = 1e-9
    weight_tolerance = 1e-15
    passed = bool(
        same_row_count
        and same_sources
        and same_ranks
        and same_neighbors
        and max_distance_m <= distance_tolerance_m
        and max_distance_km <= distance_tolerance_km
        and max_weight <= weight_tolerance
    )
    return {
        "Audit_Item": f"{sample_definition}_KNN8_Edge_Alignment",
        "RF_Value": int(len(left)),
        "Regression_Value": int(len(right)),
        "Maximum_Absolute_Difference": max(
            max_distance_m, max_distance_km, max_weight
        ),
        "Passed": passed,
        "Details": (
            f"same_row_count={same_row_count}; same_sources={same_sources}; "
            f"same_ranks={same_ranks}; same_neighbors={same_neighbors}; "
            f"max_distance_m={max_distance_m:.3e} "
            f"(tol={distance_tolerance_m:.1e}); "
            f"max_distance_km={max_distance_km:.3e} "
            f"(tol={distance_tolerance_km:.1e}); "
            f"max_weight={max_weight:.3e} (tol={weight_tolerance:.1e})"
        ),
    }


# =============================================================================
# Main workflow
# =============================================================================


def main() -> int:
    require_files(
        [
            TEMPORAL_OOF_FILE,
            SPATIAL_OOF_FILE,
            LEGACY_MORAN_FILE,
            SPATIAL_BLOCK_FILE,
            REGRESSION_STEP08_SCRIPT,
            REGRESSION_STEP08_METHOD_FILE,
            REGRESSION_INPUT_QA_FILE,
            REGRESSION_TIME_SUPPORT_FILE,
            REGRESSION_MAIN_EDGE_FILE,
            REGRESSION_COMPLETE_EDGE_FILE,
            REGRESSION_PRIMARY_MORAN_FILE,
        ]
    )
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text("", encoding="utf-8")

    log("Starting unified final Hurdle RF residual Moran analysis.")
    log(f"Step 10 code version: {STEP10_CODE_VERSION}.")
    log("Existing temporal and spatial OOF predictions are read; no RF model is trained.")
    log("Spatial blocks are read from 08_Regression_modeling Step 07; no KMeans block generation occurs.")

    protocol = load_regression_protocol()
    blocks, block_metadata = read_and_verify_blocks()
    regression_reference = read_regression_reference()
    log("Verified the locked regression Step 08 protocol and shared assignment.")

    temporal, temporal_qa = read_rf_oof(
        TEMPORAL_OOF_FILE, "Temporal", blocks, protocol
    )
    spatial, spatial_qa = read_rf_oof(
        SPATIAL_OOF_FILE, "Spatial", blocks, protocol
    )
    if temporal_qa["OOF_Key_Signature"] != spatial_qa["OOF_Key_Signature"]:
        raise ValueError("RF temporal and spatial OOF key universes differ.")
    if temporal_qa["OOF_Key_Signature"] != regression_reference[
        "Expected_OOF_Key_Signature"
    ]:
        raise ValueError(
            "RF OOF key universe differs from the locked regression OOF universe."
        )
    observed_signature_temporal = protocol.order_independent_row_signature(
        temporal,
        [
            "GRID_UID",
            "Year",
            "Season",
            "Count_Response",
            "Observed_Rate",
        ],
    )
    observed_signature_spatial = protocol.order_independent_row_signature(
        spatial,
        [
            "GRID_UID",
            "Year",
            "Season",
            "Count_Response",
            "Observed_Rate",
        ],
    )
    if observed_signature_temporal != observed_signature_spatial:
        raise ValueError("RF temporal and spatial observed-rate panels differ.")
    temporal_qa["Observed_Rate_Panel_Signature"] = observed_signature_temporal
    spatial_qa["Observed_Rate_Panel_Signature"] = observed_signature_spatial
    pd.DataFrame([temporal_qa, spatial_qa]).to_csv(
        INPUT_QA_FILE, index=False, encoding="utf-8-sig"
    )
    log(f"Verified two aligned RF OOF files with {len(temporal):,} rows each.")

    temporal_residuals = aggregate_grid_residuals(temporal)
    spatial_residuals = aggregate_grid_residuals(spatial)
    residuals = pd.concat([temporal_residuals, spatial_residuals], ignore_index=True)
    expected_residual_rows = EXPECTED_GRID_COUNT * 6 * 2
    if len(residuals) != expected_residual_rows:
        raise ValueError(
            f"Expected {expected_residual_rows:,} RF grid residual rows, "
            f"found {len(residuals):,}."
        )
    residuals.to_csv(GRID_RESIDUAL_FILE, index=False, compression="gzip")

    residual_qa = (
        residuals.groupby(
            [
                "Validation_Scheme",
                "Model_Role",
                "Season",
                "Season_Label",
                "Count_Response",
                "Rate_Scale_Name",
            ],
            as_index=False,
        )
        .agg(
            GRID_UID_Count=("GRID_UID", "nunique"),
            Years_Observed_Minimum=("Years_Observed", "min"),
            Years_Observed_Median=("Years_Observed", "median"),
            Years_Observed_Maximum=("Years_Observed", "max"),
            Complete_25_Year_GRID_UID_Count=(
                "Years_Observed", lambda values: int((values == 25).sum())
            ),
            Partial_Time_Support_GRID_UID_Count=(
                "Years_Observed", lambda values: int((values < 25).sum())
            ),
            Mean_Grid_Residual_Rate=("Mean_Residual_Rate", "mean"),
            SD_Grid_Residual_Rate=("Mean_Residual_Rate", "std"),
            Minimum_Grid_Residual_Rate=("Mean_Residual_Rate", "min"),
            Maximum_Grid_Residual_Rate=("Mean_Residual_Rate", "max"),
            Mean_Absolute_Grid_Residual_Rate=(
                "Mean_Residual_Rate", lambda values: float(np.mean(np.abs(values)))
            ),
        )
        .sort_values(["Validation_Scheme", "Season", "Count_Response"])
        .reset_index(drop=True)
    )
    residual_qa.to_csv(RESIDUAL_QA_FILE, index=False, encoding="utf-8-sig")

    time_support = build_time_support_audit(
        temporal,
        spatial,
        residuals,
        blocks,
        regression_reference["Time_Support"],
    )
    time_support.to_csv(TIME_SUPPORT_FILE, index=False, encoding="utf-8-sig")
    partial = time_support.loc[~time_support["Complete_25_Years"]].copy()
    partial.to_csv(PARTIAL_TIME_SUPPORT_FILE, index=False, encoding="utf-8-sig")
    log(
        f"Time support matches regression: {int(time_support['Complete_25_Years'].sum()):,} "
        f"complete and {len(partial):,} partial grids."
    )

    complete_uids = set(
        time_support.loc[time_support["Complete_25_Years"], "GRID_UID"].astype(str)
    )
    sample_blocks = {
        MAIN_SAMPLE: blocks.sort_values("GRID_UID").reset_index(drop=True),
        COMPLETE_SAMPLE: blocks.loc[blocks["GRID_UID"].isin(complete_uids)]
        .sort_values("GRID_UID")
        .reset_index(drop=True),
    }

    weight_matrices: Dict[Tuple[str, int], sparse.csr_matrix] = {}
    weight_qa_rows: List[Dict[str, object]] = []
    main_edges: pd.DataFrame | None = None
    complete_edges: pd.DataFrame | None = None
    for sample_definition in SAMPLE_DEFINITIONS:
        for k in [*SENSITIVITY_K_NEIGHBORS, PRIMARY_K_NEIGHBORS]:
            weights, edges, metadata = protocol.build_knn_weights(
                sample_blocks[sample_definition], k, sample_definition
            )
            weight_matrices[(sample_definition, k)] = weights
            metadata["Analysis_Role"] = (
                "Primary" if k == PRIMARY_K_NEIGHBORS else "Weights_Sensitivity"
            )
            weight_qa_rows.append(metadata)
            if sample_definition == MAIN_SAMPLE and k == PRIMARY_K_NEIGHBORS:
                main_edges = edges
            if sample_definition == COMPLETE_SAMPLE and k == PRIMARY_K_NEIGHBORS:
                complete_edges = edges
            log(
                f"Built {sample_definition} k={k} weights: {weights.nnz:,} directed edges."
            )
    pd.DataFrame(weight_qa_rows).sort_values(
        ["Sample_Definition", "K_Neighbors"]
    ).to_csv(WEIGHT_QA_FILE, index=False, encoding="utf-8-sig")
    if main_edges is None or complete_edges is None:
        raise RuntimeError("Primary RF edge tables were not created.")
    main_edges.to_csv(MAIN_EDGE_FILE, index=False, compression="gzip")
    complete_edges.to_csv(COMPLETE_EDGE_FILE, index=False, compression="gzip")

    alignment_rows: List[Dict[str, object]] = [
        compare_edge_tables(main_edges, REGRESSION_MAIN_EDGE_FILE, MAIN_SAMPLE),
        compare_edge_tables(
            complete_edges, REGRESSION_COMPLETE_EDGE_FILE, COMPLETE_SAMPLE
        ),
        {
            "Audit_Item": "OOF_Key_Universe_Alignment",
            "RF_Value": temporal_qa["OOF_Key_Signature"],
            "Regression_Value": regression_reference["Expected_OOF_Key_Signature"],
            "Maximum_Absolute_Difference": np.nan,
            "Passed": bool(
                temporal_qa["OOF_Key_Signature"]
                == regression_reference["Expected_OOF_Key_Signature"]
            ),
            "Details": "Exact order-independent GRID_UID-Year-Season-response signature.",
        },
        {
            "Audit_Item": "Complete_25Y_GRID_UID_Membership",
            "RF_Value": int(time_support["Complete_25_Years"].sum()),
            "Regression_Value": int(
                regression_reference["Time_Support"]["Complete_25_Years"]
                .astype(bool)
                .sum()
            ),
            "Maximum_Absolute_Difference": 0.0,
            "Passed": True,
            "Details": "Exact GRID_UID membership and observed/missing-year strings verified.",
        },
    ]
    preliminary_alignment = pd.DataFrame(alignment_rows)
    preliminary_alignment.to_csv(
        PROTOCOL_ALIGNMENT_FILE, index=False, encoding="utf-8-sig"
    )
    failed_alignment = preliminary_alignment.loc[
        ~preliminary_alignment["Passed"].astype(bool)
    ]
    if not failed_alignment.empty:
        for row in failed_alignment.itertuples(index=False):
            log(
                f"Protocol alignment failed: {row.Audit_Item}; "
                f"details={row.Details}"
            )
        raise ValueError(
            "RF and regression protocol alignment audit failed. "
            f"See {PROTOCOL_ALIGNMENT_FILE}."
        )

    primary, complete, knn = run_moran_tests(
        residuals,
        sample_blocks,
        weight_matrices,
        protocol,
    )
    primary.to_csv(PRIMARY_MORAN_FILE, index=False, encoding="utf-8-sig")
    complete.to_csv(COMPLETE_MORAN_FILE, index=False, encoding="utf-8-sig")
    knn.to_csv(KNN_SENSITIVITY_FILE, index=False, encoding="utf-8-sig")

    legacy_comparison = build_legacy_comparison(primary)
    legacy_comparison.to_csv(
        LEGACY_COMPARISON_FILE, index=False, encoding="utf-8-sig"
    )
    all_primary = pd.concat([primary, complete], ignore_index=True)
    time_comparison = build_time_support_comparison(all_primary)
    time_comparison.to_csv(
        TIME_SUPPORT_COMPARISON_FILE, index=False, encoding="utf-8-sig"
    )
    scheme_comparison = build_temporal_spatial_comparison(all_primary)
    scheme_comparison.to_csv(
        TEMPORAL_SPATIAL_COMPARISON_FILE, index=False, encoding="utf-8-sig"
    )

    # Verify identical primary permutation seeds to regression for every
    # season-response combination. Seeds intentionally do not depend on model
    # role or validation scheme. Use explicit uint64/Python-int conversion:
    # several valid uint32 seeds exceed the signed-int32 maximum on Windows,
    # so pandas ``astype(int)`` is not portable for this audit.
    regression_primary = regression_reference["Primary_Moran"]
    seed_audit_rows: List[Dict[str, object]] = []
    seed_alignment_passed = True

    for row in primary.itertuples(index=False):
        expected_seed = int(
            protocol.deterministic_seed(
                RANDOM_SEED,
                int(row.Season_Code),
                str(row.Count_Response),
                PRIMARY_RESIDUAL_FIELD,
                PRIMARY_K_NEIGHBORS,
                "primary",
            )
        )
        rf_seed = int(row.Permutation_Seed)
        matched = regression_primary.loc[
            (regression_primary["Validation_Scheme"].astype(str) == str(row.Validation_Scheme))
            & (
                pd.to_numeric(
                    regression_primary["Season_Code"], errors="raise"
                ).astype("int64")
                == int(row.Season_Code)
            )
            & (
                regression_primary["Count_Response"].astype(str)
                == str(row.Count_Response)
            )
        ].copy()

        if matched.empty:
            regression_seeds: List[int] = []
        else:
            regression_seed_values = pd.to_numeric(
                matched["Permutation_Seed"], errors="raise"
            )
            if regression_seed_values.isna().any():
                raise ValueError(
                    "Regression primary Moran table contains a missing permutation seed."
                )
            regression_seeds = sorted(
                {
                    int(value)
                    for value in regression_seed_values.astype("uint64").tolist()
                }
            )

        row_passed = bool(
            rf_seed == expected_seed
            and regression_seeds == [expected_seed]
        )
        seed_alignment_passed = seed_alignment_passed and row_passed
        seed_audit_rows.append(
            {
                "Validation_Scheme": str(row.Validation_Scheme),
                "Season_Code": int(row.Season_Code),
                "Season_Label": str(row.Season_Label),
                "Count_Response": str(row.Count_Response),
                "Expected_Deterministic_Seed": expected_seed,
                "RF_Permutation_Seed": rf_seed,
                "Regression_Permutation_Seeds": ";".join(
                    str(value) for value in regression_seeds
                ),
                "Regression_Matched_Row_Count": int(len(matched)),
                "Passed": row_passed,
            }
        )

    seed_audit = pd.DataFrame(seed_audit_rows)
    seed_audit_file = OUTPUT_ROOT / "11b_Primary_Permutation_Seed_Alignment_Audit.csv"
    seed_audit.to_csv(seed_audit_file, index=False, encoding="utf-8-sig")

    alignment_rows.append(
        {
            "Audit_Item": "Primary_Permutation_Seed_Alignment",
            "RF_Value": "12 RF tests checked with uint64-safe conversion",
            "Regression_Value": "matching Step08 seeds",
            "Maximum_Absolute_Difference": 0.0 if seed_alignment_passed else np.nan,
            "Passed": seed_alignment_passed,
            "Details": (
                "Same season-response-k deterministic seed across models and "
                "OOF schemes; explicit uint64 conversion avoids Windows int32 overflow."
            ),
        }
    )
    pd.DataFrame(alignment_rows).to_csv(
        PROTOCOL_ALIGNMENT_FILE, index=False, encoding="utf-8-sig"
    )
    if not seed_alignment_passed:
        raise ValueError(
            "RF primary permutation seed differs from regression; see "
            f"{seed_audit_file.name}."
        )

    method_definition = {
        "Step": "10_Final_Hurdle_RF_Residual_Spatial_Autocorrelation",
        "Created": datetime.now().isoformat(timespec="seconds"),
        "Step10_Code_Version": STEP10_CODE_VERSION,
        "Purpose": (
            "Unified temporal- and spatial-OOF residual Moran evaluation for "
            "the locked final Hurdle RF, aligned exactly to regression Step 08."
        ),
        "No_RF_Model_Training": True,
        "No_RF_Prediction_Recomputation": True,
        "No_Model_Selection": True,
        "No_Spatial_Block_Generation": True,
        "RF_OOF_Inputs": {
            "Temporal": str(TEMPORAL_OOF_FILE),
            "Spatial": str(SPATIAL_OOF_FILE),
            "Expected_Rows_Per_File": EXPECTED_ROWS_PER_OOF_FILE,
            "Observed_Rate_Field": "Observed",
            "Predicted_Rate_Field": "Expected_Prediction",
            "Residual_Rate_Field": "Residual",
        },
        "Shared_Spatial_Assignment": block_metadata,
        "Locked_Regression_Protocol": {
            "Script": str(REGRESSION_STEP08_SCRIPT),
            "Script_SHA256": EXPECTED_REGRESSION_STEP08_SCRIPT_SHA256,
            "Version": EXPECTED_REGRESSION_STEP08_VERSION,
            "Imported_Core_Functions": [
                "deterministic_seed",
                "bh_adjust",
                "order_independent_row_signature",
                "build_knn_weights",
                "morans_i_permutation",
            ],
            "Protocol_Alignment_Audit": str(PROTOCOL_ALIGNMENT_FILE),
        },
        "Residual_Aggregation": {
            "Primary_Field": PRIMARY_RESIDUAL_FIELD,
            "Definition": PRIMARY_RESIDUAL_DEFINITION,
            "Main_Sample": {
                "Name": MAIN_SAMPLE,
                "GRID_UID_Count": EXPECTED_GRID_COUNT,
                "Temporal_Aggregation": "Mean across each grid's available OOF years",
            },
            "Complete_Time_Support_Sensitivity": {
                "Name": COMPLETE_SAMPLE,
                "GRID_UID_Count": EXPECTED_COMPLETE_25Y_GRID_COUNT,
                "Temporal_Aggregation": "Mean across exactly 25 years",
            },
        },
        "Primary_Moran_Test": {
            "Test_Count": 12,
            "Validation_Schemes": VALIDATION_SCHEMES,
            "Coordinates": ["Centroid_X_UTM52_m", "Centroid_Y_UTM52_m"],
            "Weights": "8-nearest-neighbor directed row-standardized",
            "Permutations_Per_Test": PRIMARY_PERMUTATIONS,
            "P_Value": (
                "Two-sided permutation p-value centered on -1/(n-1), with plus-one correction"
            ),
            "FDR": {
                "Within_RF_Global": "BH across 12 main RF tests",
                "Within_Validation_Scheme": "BH across 6 tests per OOF scheme",
                "Within_Scheme_Model": "BH across 6 tests per scheme/model",
                "Future_Cross_Model_Rule": (
                    "Recompute combined FDR in the later model-comparison step across "
                    "all three model roles."
                ),
            },
        },
        "Sensitivity_Analyses": {
            "Complete_25Y": {
                "Test_Count": 12,
                "K_Neighbors": PRIMARY_K_NEIGHBORS,
                "Permutations_Per_Test": PRIMARY_PERMUTATIONS,
            },
            "KNN_Weights": {
                "Test_Count": 48,
                "K_Neighbors": SENSITIVITY_K_NEIGHBORS,
                "Permutations_Per_Test": SENSITIVITY_PERMUTATIONS,
            },
        },
        "Legacy_RF_Moran_Reproduction": {
            "Legacy_File": str(LEGACY_MORAN_FILE),
            "Rule": "Unified temporal main-sample Moran_I must equal legacy Moran_I",
            "Audit_File": str(LEGACY_COMPARISON_FILE),
        },
        "Software": {
            "Python": platform.python_version(),
            "Platform": platform.platform(),
            "NumPy": np.__version__,
            "pandas": pd.__version__,
            "SciPy": scipy.__version__,
        },
    }
    method_file = OUTPUT_ROOT / "00_Method_Definition.json"
    method_file.write_text(
        json.dumps(method_definition, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    software_file = OUTPUT_ROOT / "Software_Environment.json"
    software_file.write_text(
        json.dumps(
            {
                "Created": datetime.now().isoformat(timespec="seconds"),
                "Python": sys.version,
                "Platform": platform.platform(),
                "NumPy": np.__version__,
                "pandas": pd.__version__,
                "SciPy": scipy.__version__,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    log(
        "Step 10 completed: 12 main-sample primary, 12 complete-25-year "
        "time-support sensitivity, and 48 additional kNN-sensitivity Moran "
        "tests were calculated separately for final Hurdle RF temporal and "
        "spatial OOF rate residuals without retraining the model."
    )

    output_paths = [
        method_file,
        INPUT_QA_FILE,
        RESIDUAL_QA_FILE,
        TIME_SUPPORT_FILE,
        PARTIAL_TIME_SUPPORT_FILE,
        WEIGHT_QA_FILE,
        MAIN_EDGE_FILE,
        COMPLETE_EDGE_FILE,
        GRID_RESIDUAL_FILE,
        PRIMARY_MORAN_FILE,
        COMPLETE_MORAN_FILE,
        KNN_SENSITIVITY_FILE,
        LEGACY_COMPARISON_FILE,
        TIME_SUPPORT_COMPARISON_FILE,
        TEMPORAL_SPATIAL_COMPARISON_FILE,
        PROTOCOL_ALIGNMENT_FILE,
        software_file,
        LOG_FILE,
    ]
    manifest = build_output_manifest(output_paths)
    manifest["Step10_Code_Version"] = STEP10_CODE_VERSION
    manifest["Spatial_Mapping_SHA256"] = EXPECTED_SPATIAL_MAPPING_SHA256
    manifest.to_csv(OUTPUT_MANIFEST_FILE, index=False, encoding="utf-8-sig")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
