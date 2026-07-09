# -*- coding: utf-8 -*-
"""
Step 08: unified temporal- and spatial-OOF residual Moran analysis for
locked regression models.

Recommended public location
---------------------------
<CODE_ROOT>/08_Regression_modeling/00_Script/
    08_test_regression_residual_spatial_autocorrelation.py

Inputs
------
Baseline temporal OOF predictions:
<CODE_ROOT>/08_Regression_modeling/
    02_Single_Stage_Count_Family_Comparison/
        Temporal_OOF_Predictions_S{season}_{response}.csv.gz

Final-regression temporal OOF predictions:
<CODE_ROOT>/08_Regression_modeling/
    06d_Final_Stable_Fixed_Effect_Inference/
        17_Final_Selected_OOF_Manifest.csv
        Final_Selected_Inference_OOF_Predictions/
            Final_Selected_Inference_OOF_Predictions_*.csv.gz

Locked spatial assignment and spatial OOF predictions:
<CODE_ROOT>/08_Regression_modeling/
    07_Regression_Spatial_Block_Validation/
        00_Method_Definition.json
        08_Spatial_OOF_Manifest.csv
        Spatial_Block_Assignment.csv
        Baseline_NB1_Spatial_Out_of_Fold_Predictions.csv.gz
        Final_Regression_Spatial_Out_of_Fold_Predictions.csv.gz

Output
------
<CODE_ROOT>/08_Regression_modeling/
    08_Regression_Residual_Spatial_Autocorrelation/

Purpose
-------
This step creates a unified residual-spatial-autocorrelation evaluation for
Baseline_NB1 and Final_Regression using both temporal and spatial OOF
predictions. It does not refit models, change model structures, select
predictors, or regenerate spatial blocks.

The main sample follows the previously implemented final-RF residual-Moran
principle so later regression-versus-RF comparisons use the same definition:

1. retain the locked common 4,982-grid universe;
2. calculate each grid's mean rate residual across its available OOF years;
3. use UTM52N-centroid, directed k-nearest-neighbor, row-standardized weights;
4. use k=8 as primary and k=4/12 as prespecified weight sensitivities.

This release adds a parallel complete-time-support sensitivity restricted to the
4,924 grids observed in all 25 years. Temporal OOF and spatial OOF results are
kept separate throughout. For each season-response pair, identical
permutation seeds are used across model roles and validation schemes to make
comparisons reproducible.

Primary inference
-----------------
For the main 4,982-grid sample, 24 k=8 tests are calculated:
    2 OOF schemes x 2 model roles x 3 seasons x 2 responses.
Each uses 9,999 two-sided permutations. BH-FDR is reported globally across
all 24 tests, within each OOF scheme (12 tests), and within each
OOF-scheme/model family (6 tests).

Sensitivity analyses
--------------------
1. Complete 25-year sample: the same 24 k=8 tests with 9,999 permutations.
2. Spatial weights: k=4 and k=12 for both samples, both OOF schemes and both
   model roles, using 999 permutations per test.

Workflow rule
-------------
08_Regression_modeling Step 07 remains the sole generator of the locked
spatial assignment. This script never reads any output from 09_RF_modeling.
"""

from __future__ import annotations

import hashlib
import json
import math
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
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


# =============================================================================
# Repository-relative paths
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REGRESSION_ROOT = SCRIPT_DIR.parent
CODE_ROOT = REGRESSION_ROOT.parent

STEP02_ROOT = REGRESSION_ROOT / "02_Single_Stage_Count_Family_Comparison"
STEP02_METHOD_FILE = STEP02_ROOT / "00_Method_Definition.json"

STEP06D_ROOT = REGRESSION_ROOT / "06d_Final_Stable_Fixed_Effect_Inference"
STEP06D_OOF_MANIFEST_FILE = STEP06D_ROOT / "17_Final_Selected_OOF_Manifest.csv"
STEP06D_OOF_ROOT = (
    STEP06D_ROOT / "Final_Selected_Inference_OOF_Predictions"
)

STEP07_ROOT = REGRESSION_ROOT / "07_Regression_Spatial_Block_Validation"
STEP07_METHOD_FILE = STEP07_ROOT / "00_Method_Definition.json"
STEP07_OOF_MANIFEST_FILE = STEP07_ROOT / "08_Spatial_OOF_Manifest.csv"
SPATIAL_BLOCK_FILE = STEP07_ROOT / "Spatial_Block_Assignment.csv"

OUTPUT_ROOT = (
    REGRESSION_ROOT / "08_Regression_Residual_Spatial_Autocorrelation"
)
LOG_FILE = OUTPUT_ROOT / "regression_residual_spatial_autocorrelation.log"

INPUT_QA_FILE = OUTPUT_ROOT / "01_Input_and_Residual_QA.csv"
RESIDUAL_QA_FILE = OUTPUT_ROOT / "02_Grid_Residual_Summary_QA.csv"
TIME_SUPPORT_FILE = OUTPUT_ROOT / "02b_Grid_Time_Support_Audit.csv"
PARTIAL_TIME_SUPPORT_FILE = OUTPUT_ROOT / "02c_Partial_Time_Support_Grids.csv"
WEIGHT_QA_FILE = OUTPUT_ROOT / "03_Spatial_Weights_QA.csv"
MAIN_EDGE_FILE = OUTPUT_ROOT / "03b_Main_4982_KNN8_Neighbor_Edges.csv.gz"
COMPLETE_EDGE_FILE = (
    OUTPUT_ROOT / "03c_Complete_4924_KNN8_Neighbor_Edges.csv.gz"
)
GRID_RESIDUAL_FILE = OUTPUT_ROOT / "04_All_Grid_Mean_OOF_Residuals.csv.gz"
PRIMARY_MORAN_FILE = OUTPUT_ROOT / "05_Primary_Main_Sample_Morans_I.csv"
COMPLETE_MORAN_FILE = (
    OUTPUT_ROOT / "06_Complete_25Y_Sensitivity_Morans_I.csv"
)
KNN_SENSITIVITY_FILE = OUTPUT_ROOT / "07_KNN_Sensitivity_Morans_I.csv"
BASELINE_FINAL_COMPARISON_FILE = (
    OUTPUT_ROOT / "08_Baseline_vs_Final_Moran_Comparison.csv"
)
TIME_SUPPORT_COMPARISON_FILE = (
    OUTPUT_ROOT / "09_Time_Support_Sensitivity_Comparison.csv"
)
TEMPORAL_SPATIAL_COMPARISON_FILE = (
    OUTPUT_ROOT / "10_Temporal_vs_Spatial_OOF_Moran_Comparison.csv"
)
OUTPUT_MANIFEST_FILE = OUTPUT_ROOT / "11_Output_Manifest.csv"


# =============================================================================
# Locked settings
# =============================================================================

STEP08_CODE_VERSION = (
    "2026-07-08_RELEASE_OPTIONAL_LEGACY_FOLD_COMPATIBILITY"
)
EXPECTED_STEP07_CODE_VERSION = (
    "2026-07-08_RELEASE_STABLE_POSITIVE_PROBABILITY"
)
EXPECTED_STEP07_PREDICTION_PROTOCOL_ID = (
    "STABLE_LOGSPACE_POSITIVE_PROBABILITY_AND_CONDITIONAL_MEAN_V4"
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
EXPECTED_ROWS_PER_MODEL_ROLE = EXPECTED_ROWS_PER_SEASON_RESPONSE * 6
EXPECTED_YEARS = set(range(2001, 2026))
EXPECTED_BLOCK_GRID_COUNTS = {
    1: 912,
    2: 1042,
    3: 735,
    4: 1222,
    5: 1071,
}

MODEL_ROLES = ["Baseline_NB1", "Final_Regression"]
VALIDATION_SCHEMES = ["Temporal", "Spatial"]
SEASONS = {1: "Spring", 2: "Summer", 3: "Autumn"}
COUNT_RESPONSES = ["Fire_Count", "Burned_Pixel_Count"]
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
PERMUTATION_BATCH_SIZE = 128
RANDOM_SEED = 2026
FDR_ALPHA = 0.05

PRIMARY_RESIDUAL_FIELD = "Mean_Residual_Rate"
PRIMARY_RESIDUAL_DEFINITION = (
    "Arithmetic mean across the relevant OOF years of Observed_Rate minus "
    "Predicted_Rate by GRID_UID"
)

BASELINE_TEMPORAL_FAMILY = "NB1"
BASELINE_MODEL_STRUCTURE = "Single_Stage_NB1"
BASELINE_CANDIDATE_ID = "BASELINE_SINGLE_STAGE_NB1"

COMMON_OUTPUT_COLUMNS = [
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
    "ForestPixelCount",
    "Observed_Count",
    "Predicted_Count",
    "Observed_Rate",
    "Predicted_Rate",
    "Residual_Count",
    "Residual_Rate",
]


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


def deterministic_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % (
        2**32 - 1
    )


def bh_adjust(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    result = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return result

    selected = values[finite]
    order = np.argsort(selected)
    ranked = selected[order]
    m = len(ranked)
    adjusted_ranked = ranked * m / np.arange(1, m + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted_ranked = np.clip(adjusted_ranked, 0.0, 1.0)
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    result[finite] = adjusted
    return result


def order_independent_row_signature(
    frame: pd.DataFrame, columns: Sequence[str]
) -> str:
    selected = frame[list(columns)].copy()
    for column in selected.columns:
        selected[column] = selected[column].astype(str)
    selected = selected.sort_values(list(columns)).reset_index(drop=True)
    payload = selected.to_csv(
        index=False, header=False, lineterminator="\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def maximum_absolute_difference(
    left: Sequence[float], right: Sequence[float]
) -> float:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    if a.shape != b.shape:
        return np.inf
    finite = np.isfinite(a) & np.isfinite(b)
    if not finite.all():
        return np.inf
    if len(a) == 0:
        return 0.0
    return float(np.max(np.abs(a - b)))


def response_order(response: str) -> int:
    return COUNT_RESPONSES.index(str(response))


# =============================================================================
# Locked Step-07 inputs and spatial assignment
# =============================================================================


def read_and_verify_step07_method() -> Dict[str, object]:
    method = json.loads(STEP07_METHOD_FILE.read_text(encoding="utf-8"))
    if str(method.get("Step07_Code_Version", "")) != EXPECTED_STEP07_CODE_VERSION:
        raise ValueError(
            "Step 08 requires the locked Step-07 release method definition.\n"
            f"Expected: {EXPECTED_STEP07_CODE_VERSION}\n"
            f"Observed: {method.get('Step07_Code_Version')}"
        )
    if (
        str(method.get("Spatial_Prediction_Protocol_ID", ""))
        != EXPECTED_STEP07_PREDICTION_PROTOCOL_ID
    ):
        raise ValueError("Unexpected Step-07 spatial prediction protocol.")
    if int(method.get("Expected_Grid_Count", -1)) != EXPECTED_GRID_COUNT:
        raise ValueError("Unexpected Step-07 expected grid count.")
    if (
        int(method.get("Expected_Rows_Per_Season_Response", -1))
        != EXPECTED_ROWS_PER_SEASON_RESPONSE
    ):
        raise ValueError("Unexpected Step-07 season-response row count.")
    return method


def read_and_verify_blocks() -> Tuple[pd.DataFrame, Dict[str, object]]:
    blocks = pd.read_csv(SPATIAL_BLOCK_FILE)
    required = {
        "GRID_UID",
        "Country",
        "GRID_ID",
        *COORDINATE_COLUMNS,
        "Spatial_Block",
    }
    missing = sorted(required - set(blocks.columns))
    if missing:
        raise ValueError(
            "Spatial_Block_Assignment.csv is missing columns:\n"
            + "\n".join(missing)
        )

    blocks = blocks[
        ["GRID_UID", "Country", "GRID_ID", *COORDINATE_COLUMNS, "Spatial_Block"]
    ].copy()
    blocks["GRID_UID"] = blocks["GRID_UID"].astype(str)
    blocks["Country"] = blocks["Country"].astype(str)
    blocks["GRID_ID"] = blocks["GRID_ID"].astype(str)
    blocks["Spatial_Block"] = pd.to_numeric(
        blocks["Spatial_Block"], errors="raise"
    ).astype(int)
    for column in COORDINATE_COLUMNS:
        blocks[column] = pd.to_numeric(blocks[column], errors="coerce")

    if blocks["GRID_UID"].duplicated().any():
        raise ValueError("Duplicate GRID_UID values occur in the block file.")
    if len(blocks) != EXPECTED_GRID_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_GRID_COUNT:,} grids, found {len(blocks):,}."
        )
    if not np.isfinite(blocks[COORDINATE_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError("The spatial block file contains invalid coordinates.")
    if blocks.duplicated(
        ["Centroid_X_UTM52_m", "Centroid_Y_UTM52_m"], keep=False
    ).any():
        raise ValueError("Duplicate UTM centroid coordinate pairs were found.")

    observed_counts = {
        int(block): int(count)
        for block, count in blocks["Spatial_Block"]
        .value_counts()
        .sort_index()
        .items()
    }
    if observed_counts != EXPECTED_BLOCK_GRID_COUNTS:
        raise ValueError(
            "Spatial-block counts differ from the locked design.\n"
            f"Expected: {EXPECTED_BLOCK_GRID_COUNTS}\n"
            f"Observed: {observed_counts}"
        )

    mapping_hash = spatial_mapping_sha256(blocks)
    file_hash = sha256_file(SPATIAL_BLOCK_FILE)
    if mapping_hash != EXPECTED_SPATIAL_MAPPING_SHA256:
        raise ValueError("The semantic spatial mapping differs from the lock.")
    if file_hash != EXPECTED_SPATIAL_BLOCK_FILE_SHA256:
        raise ValueError("The block-file byte hash differs from the lock.")

    blocks = blocks.sort_values("GRID_UID").reset_index(drop=True)
    metadata = {
        "Spatial_Block_File": str(SPATIAL_BLOCK_FILE),
        "Spatial_Block_File_SHA256": file_hash,
        "Spatial_Mapping_SHA256": mapping_hash,
        "GRID_UID_Count": int(len(blocks)),
        "Block_GRID_UID_Counts": observed_counts,
    }
    return blocks, metadata


def read_and_verify_spatial_manifest() -> pd.DataFrame:
    manifest = pd.read_csv(STEP07_OOF_MANIFEST_FILE)
    required = {
        "Model_Role",
        "Output_File",
        "Rows",
        "SHA256",
        "Unique_OOF_Keys",
        "Validation_Scheme",
        "Spatial_Prediction_Protocol_ID",
        "Spatial_Mapping_SHA256",
        "Spatial_Block_File_SHA256",
    }
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(
            "The Step-07 OOF manifest is missing columns:\n"
            + "\n".join(missing)
        )
    if len(manifest) != 2 or set(manifest["Model_Role"].astype(str)) != set(
        MODEL_ROLES
    ):
        raise ValueError("Step-07 manifest must contain exactly two model roles.")

    for row in manifest.itertuples(index=False):
        if str(row.Validation_Scheme) != "Spatial":
            raise ValueError("A Step-07 manifest row is not spatial OOF.")
        if int(row.Rows) != EXPECTED_ROWS_PER_MODEL_ROLE:
            raise ValueError(f"Unexpected spatial OOF rows for {row.Model_Role}.")
        if int(row.Unique_OOF_Keys) != EXPECTED_ROWS_PER_MODEL_ROLE:
            raise ValueError(f"Unexpected spatial OOF keys for {row.Model_Role}.")
        if (
            str(row.Spatial_Prediction_Protocol_ID)
            != EXPECTED_STEP07_PREDICTION_PROTOCOL_ID
        ):
            raise ValueError("A spatial OOF file is not from the required Step-07 release.")
        if str(row.Spatial_Mapping_SHA256) != EXPECTED_SPATIAL_MAPPING_SHA256:
            raise ValueError("A spatial OOF mapping hash is incorrect.")
        if str(row.Spatial_Block_File_SHA256) != EXPECTED_SPATIAL_BLOCK_FILE_SHA256:
            raise ValueError("A spatial OOF block-file hash is incorrect.")
        path = STEP07_ROOT / str(row.Output_File)
        require_files([path])
        if sha256_file(path) != str(row.SHA256):
            raise ValueError(f"SHA256 mismatch for spatial OOF file:\n{path}")

    return manifest.sort_values("Model_Role").reset_index(drop=True)


# =============================================================================
# Temporal OOF inputs
# =============================================================================


def step02_temporal_path(season: int, response: str) -> Path:
    return STEP02_ROOT / f"Temporal_OOF_Predictions_S{season}_{response}.csv.gz"


def final_temporal_oof_path(output_file: str) -> Path:
    candidates = [STEP06D_OOF_ROOT / output_file, STEP06D_ROOT / output_file]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "A Step-06d temporal OOF file was not found:\n"
        + "\n".join(str(path) for path in candidates)
    )


def read_and_verify_final_temporal_manifest() -> pd.DataFrame:
    manifest = pd.read_csv(STEP06D_OOF_MANIFEST_FILE)
    required = {
        "Season_Code",
        "Season_Label",
        "Count_Response",
        "Selected_Candidate_ID",
        "Output_File",
        "Rows",
    }
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(
            "The Step-06d OOF manifest is missing columns:\n"
            + "\n".join(missing)
        )
    expected = {
        (season, response)
        for season in SEASONS
        for response in COUNT_RESPONSES
    }
    observed = {
        (int(row.Season_Code), str(row.Count_Response))
        for row in manifest.itertuples(index=False)
    }
    if len(manifest) != 6 or observed != expected:
        raise ValueError("Step-06d manifest does not contain the six combinations.")
    if not (
        pd.to_numeric(manifest["Rows"], errors="raise").astype(int)
        == EXPECTED_ROWS_PER_SEASON_RESPONSE
    ).all():
        raise ValueError("A Step-06d temporal OOF row count is unexpected.")
    for output_file in manifest["Output_File"].astype(str):
        require_files([final_temporal_oof_path(output_file)])
    return manifest.sort_values(["Season_Code", "Count_Response"]).reset_index(
        drop=True
    )


# =============================================================================
# OOF normalization and validation
# =============================================================================


def attach_block_metadata(
    data: pd.DataFrame, blocks: pd.DataFrame, source_name: str
) -> pd.DataFrame:
    result = data.copy()
    result["GRID_UID"] = result["GRID_UID"].astype(str)
    result["Country"] = result["Country"].astype(str)
    result["GRID_ID"] = result["GRID_ID"].astype(str)

    block_table = blocks.rename(
        columns={
            "Country": "Locked_Country",
            "GRID_ID": "Locked_GRID_ID",
            "Spatial_Block": "Locked_Spatial_Block",
            **{column: f"Locked_{column}" for column in COORDINATE_COLUMNS},
        }
    )
    result = result.merge(
        block_table,
        on="GRID_UID",
        how="left",
        validate="many_to_one",
    )
    if result["Locked_Spatial_Block"].isna().any():
        raise ValueError(f"{source_name} contains a grid absent from Step 07.")
    if not (
        result["Country"].to_numpy()
        == result["Locked_Country"].astype(str).to_numpy()
    ).all():
        raise ValueError(f"Country disagrees with Step 07 in {source_name}.")
    if not (
        result["GRID_ID"].to_numpy()
        == result["Locked_GRID_ID"].astype(str).to_numpy()
    ).all():
        raise ValueError(f"GRID_ID disagrees with Step 07 in {source_name}.")

    if "Spatial_Block" in result.columns:
        observed_block = pd.to_numeric(
            result["Spatial_Block"], errors="coerce"
        )
        if observed_block.notna().any() and not (
            observed_block.astype(int).to_numpy()
            == result["Locked_Spatial_Block"].astype(int).to_numpy()
        ).all():
            raise ValueError(f"Spatial_Block disagrees in {source_name}.")

    for column in COORDINATE_COLUMNS:
        if column in result.columns:
            observed = pd.to_numeric(result[column], errors="coerce")
            locked = pd.to_numeric(result[f"Locked_{column}"], errors="coerce")
            if observed.notna().any():
                difference = maximum_absolute_difference(observed, locked)
                if difference > 1e-8:
                    raise ValueError(f"{column} disagrees in {source_name}.")
        result[column] = pd.to_numeric(
            result[f"Locked_{column}"], errors="raise"
        )

    result["Spatial_Block"] = result["Locked_Spatial_Block"].astype(int)
    drop_columns = [
        "Locked_Country",
        "Locked_GRID_ID",
        "Locked_Spatial_Block",
        *[f"Locked_{column}" for column in COORDINATE_COLUMNS],
    ]
    return result.drop(columns=drop_columns)


def finalize_oof_frame(
    raw: pd.DataFrame,
    blocks: pd.DataFrame,
    model_role: str,
    validation_scheme: str,
    source_name: str,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    required = {
        "GRID_UID",
        "GRID_ID",
        "Country",
        "Year",
        "Season",
        "Temporal_Fold",
        "ForestPixelCount",
        "Count_Response",
        "Observed_Count",
        "Predicted_Count",
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(
            f"{source_name} is missing required columns:\n" + "\n".join(missing)
        )

    data = raw.copy()
    data["GRID_UID"] = data["GRID_UID"].astype(str)
    data["Country"] = data["Country"].astype(str)
    data["GRID_ID"] = data["GRID_ID"].astype(str)
    data["Count_Response"] = data["Count_Response"].astype(str)
    data["Year"] = pd.to_numeric(data["Year"], errors="raise").astype(int)
    data["Season"] = pd.to_numeric(data["Season"], errors="raise").astype(int)
    data["Temporal_Fold"] = pd.to_numeric(
        data["Temporal_Fold"], errors="raise"
    ).astype(int)

    if len(data) != EXPECTED_ROWS_PER_MODEL_ROLE:
        raise ValueError(
            f"Expected {EXPECTED_ROWS_PER_MODEL_ROLE:,} rows in {source_name}, "
            f"found {len(data):,}."
        )
    if set(data["Year"].unique()) != EXPECTED_YEARS:
        raise ValueError(f"Unexpected year universe in {source_name}.")
    if set(data["Season"].unique()) != set(SEASONS):
        raise ValueError(f"Unexpected seasons in {source_name}.")
    if set(data["Count_Response"].unique()) != set(COUNT_RESPONSES):
        raise ValueError(f"Unexpected responses in {source_name}.")
    if set(data["Temporal_Fold"].unique()) != {1, 2, 3, 4, 5}:
        raise ValueError(f"Unexpected temporal folds in {source_name}.")

    key_columns = ["GRID_UID", "Year", "Season", "Count_Response"]
    if data.duplicated(key_columns).any():
        raise ValueError(f"Duplicate OOF keys occur in {source_name}.")

    combination_counts = (
        data.groupby(["Season", "Count_Response"], sort=True).size().to_dict()
    )
    expected_counts = {
        (season, response): EXPECTED_ROWS_PER_SEASON_RESPONSE
        for season in SEASONS
        for response in COUNT_RESPONSES
    }
    if combination_counts != expected_counts:
        raise ValueError(
            f"Season-response row counts differ in {source_name}.\n"
            f"Observed: {combination_counts}"
        )
    grid_counts = (
        data.groupby(["Season", "Count_Response"])["GRID_UID"]
        .nunique()
        .to_dict()
    )
    if grid_counts != {
        key: EXPECTED_GRID_COUNT for key in expected_counts
    }:
        raise ValueError(f"Season-response grid counts differ in {source_name}.")

    data = attach_block_metadata(data, blocks, source_name)

    if validation_scheme == "Spatial":
        if "Validation_Fold" not in data.columns:
            raise ValueError(f"Spatial Validation_Fold missing in {source_name}.")
        validation_fold = pd.to_numeric(
            data["Validation_Fold"], errors="raise"
        ).astype(int)
        if not (
            validation_fold.to_numpy() == data["Spatial_Block"].to_numpy()
        ).all():
            raise ValueError(
                f"Validation_Fold and Spatial_Block disagree in {source_name}."
            )
        data["Validation_Fold"] = validation_fold
    elif validation_scheme == "Temporal":
        # Temporal_Fold is the authoritative field across all Step-02 and
        # Step-06d OOF sources. Some legacy prediction tables also carry an
        # optional Fold column, but after combining candidates that column can
        # be populated for only a subset of rows. Validate it only where a
        # non-missing legacy value exists; never require it to be complete.
        if "Fold" in data.columns:
            raw_legacy_fold = data["Fold"]
            legacy_fold = pd.to_numeric(raw_legacy_fold, errors="coerce")
            nonempty_legacy = raw_legacy_fold.notna() & (
                raw_legacy_fold.astype(str).str.strip() != ""
            )
            invalid_legacy = nonempty_legacy & legacy_fold.isna()
            if invalid_legacy.any():
                raise ValueError(
                    f"Nonnumeric non-missing legacy Fold values occur in "
                    f"{source_name}."
                )
            present_legacy = legacy_fold.notna()
            if present_legacy.any():
                rounded_legacy = np.rint(
                    legacy_fold.loc[present_legacy].to_numpy(dtype=float)
                )
                if not np.allclose(
                    legacy_fold.loc[present_legacy].to_numpy(dtype=float),
                    rounded_legacy,
                    rtol=0.0,
                    atol=0.0,
                ):
                    raise ValueError(
                        f"Noninteger legacy Fold values occur in {source_name}."
                    )
                legacy_fold_int = rounded_legacy.astype(int)
                temporal_fold_int = data.loc[
                    present_legacy, "Temporal_Fold"
                ].to_numpy(dtype=int)
                if not np.array_equal(legacy_fold_int, temporal_fold_int):
                    raise ValueError(
                        f"Non-missing Fold and Temporal_Fold values disagree "
                        f"in {source_name}."
                    )
        data["Validation_Fold"] = data["Temporal_Fold"].astype(int)
    else:
        raise ValueError(f"Unsupported validation scheme: {validation_scheme}")

    numeric_columns = ["ForestPixelCount", "Observed_Count", "Predicted_Count"]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if not np.isfinite(data[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError(f"Non-finite core values occur in {source_name}.")
    if (data["ForestPixelCount"] <= 0).any():
        raise ValueError(f"Non-positive exposure occurs in {source_name}.")
    if (data["Observed_Count"] < 0).any() or (data["Predicted_Count"] < 0).any():
        raise ValueError(f"Negative count values occur in {source_name}.")

    data["Observed_Rate"] = data["Observed_Count"] / data["ForestPixelCount"]
    data["Predicted_Rate"] = data["Predicted_Count"] / data["ForestPixelCount"]
    data["Residual_Count"] = data["Observed_Count"] - data["Predicted_Count"]
    data["Residual_Rate"] = data["Observed_Rate"] - data["Predicted_Rate"]

    if not np.isfinite(
        data[
            [
                "Observed_Rate",
                "Predicted_Rate",
                "Residual_Count",
                "Residual_Rate",
            ]
        ].to_numpy(dtype=float)
    ).all():
        raise ValueError(f"Non-finite derived values occur in {source_name}.")

    data["Season_Label"] = data["Season"].map(SEASONS)
    data["Rate_Scale_Name"] = data["Count_Response"].map(RATE_NAMES)
    data["Validation_Scheme"] = validation_scheme
    data["Model_Role"] = model_role
    data["Model_ID"] = model_role

    if model_role == "Baseline_NB1":
        data["Model_Structure"] = BASELINE_MODEL_STRUCTURE
        data["Candidate_ID"] = BASELINE_CANDIDATE_ID
    else:
        if "Model_Structure" not in data.columns or "Candidate_ID" not in data.columns:
            raise ValueError(
                f"Final model metadata is absent from {source_name}."
            )
        data["Model_Structure"] = data["Model_Structure"].astype(str)
        data["Candidate_ID"] = data["Candidate_ID"].astype(str)

    data = data[COMMON_OUTPUT_COLUMNS].copy()
    key_signature = order_independent_row_signature(data, key_columns)
    panel_signature = order_independent_row_signature(
        data,
        [
            "GRID_UID",
            "Year",
            "Season",
            "Count_Response",
            "Observed_Count",
            "ForestPixelCount",
        ],
    )
    qa = {
        "Validation_Scheme": validation_scheme,
        "Model_Role": model_role,
        "Source_Name": source_name,
        "Rows": int(len(data)),
        "Unique_OOF_Keys": int(len(data)),
        "GRID_UID_Count": int(data["GRID_UID"].nunique()),
        "Minimum_Year": int(data["Year"].min()),
        "Maximum_Year": int(data["Year"].max()),
        "Season_Count": int(data["Season"].nunique()),
        "Response_Count": int(data["Count_Response"].nunique()),
        "Temporal_Fold_Count": int(data["Temporal_Fold"].nunique()),
        "Spatial_Block_Count": int(data["Spatial_Block"].nunique()),
        "OOF_Key_Signature": key_signature,
        "Observed_Panel_Signature": panel_signature,
    }
    return data, qa


def aggregate_grid_residuals(data: pd.DataFrame) -> pd.DataFrame:
    work = data.copy()
    work["Observed_Positive"] = work["Observed_Count"] > 0
    work["Squared_Residual_Rate"] = np.square(
        work["Residual_Rate"].to_numpy(dtype=float)
    )
    work["Absolute_Residual_Rate"] = np.abs(
        work["Residual_Rate"].to_numpy(dtype=float)
    )
    group_columns = [
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
    ]
    grouped = (
        work.groupby(group_columns, as_index=False, sort=True)
        .agg(
            Row_Count=("Year", "size"),
            Years_Observed=("Year", "nunique"),
            First_Year=("Year", "min"),
            Last_Year=("Year", "max"),
            Positive_Observed_Years=("Observed_Positive", "sum"),
            Mean_Observed_Count=("Observed_Count", "mean"),
            Mean_Predicted_Count=("Predicted_Count", "mean"),
            Mean_Residual_Count=("Residual_Count", "mean"),
            Total_Observed_Count=("Observed_Count", "sum"),
            Total_Predicted_Count=("Predicted_Count", "sum"),
            Mean_Observed_Rate=("Observed_Rate", "mean"),
            Mean_Predicted_Rate=("Predicted_Rate", "mean"),
            Mean_Residual_Rate=("Residual_Rate", "mean"),
            Median_Residual_Rate=("Residual_Rate", "median"),
            Residual_Rate_SD=("Residual_Rate", "std"),
            Mean_Absolute_Residual_Rate=("Absolute_Residual_Rate", "mean"),
            Mean_Squared_Residual_Rate=("Squared_Residual_Rate", "mean"),
        )
        .sort_values(
            [
                "Validation_Scheme",
                "Model_Role",
                "Season",
                "Count_Response",
                "GRID_UID",
            ]
        )
        .reset_index(drop=True)
    )
    grouped["Residual_Rate_RMSE_Across_Years"] = np.sqrt(
        grouped["Mean_Squared_Residual_Rate"]
    )
    grouped["Positive_Observed_Years"] = grouped[
        "Positive_Observed_Years"
    ].astype(int)

    source_pair_count = int(
        work[["Validation_Scheme", "Model_Role"]].drop_duplicates().shape[0]
    )
    expected = EXPECTED_GRID_COUNT * 6 * source_pair_count
    if len(grouped) != expected:
        raise ValueError(
            f"Expected {expected:,} grid-residual rows, found {len(grouped):,}."
        )
    counts = grouped.groupby(
        ["Validation_Scheme", "Model_Role", "Season", "Count_Response"]
    )["GRID_UID"].nunique()
    if not (counts == EXPECTED_GRID_COUNT).all():
        raise ValueError("A grid is missing from a residual combination.")
    if not np.isfinite(grouped[PRIMARY_RESIDUAL_FIELD].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite mean rate residuals were generated.")
    return grouped


# =============================================================================
# Read temporal and spatial OOF sources
# =============================================================================


def read_temporal_oof(
    blocks: pd.DataFrame,
) -> Tuple[List[pd.DataFrame], List[Dict[str, object]], pd.DataFrame]:
    require_files([STEP02_METHOD_FILE, STEP06D_OOF_MANIFEST_FILE])
    final_manifest = read_and_verify_final_temporal_manifest()

    baseline_parts: List[pd.DataFrame] = []
    final_parts: List[pd.DataFrame] = []
    source_rows: List[Dict[str, object]] = []
    panel_reference: pd.DataFrame | None = None

    for manifest_row in final_manifest.itertuples(index=False):
        season = int(manifest_row.Season_Code)
        response = str(manifest_row.Count_Response)
        final_path = final_temporal_oof_path(str(manifest_row.Output_File))
        baseline_path = step02_temporal_path(season, response)
        require_files([baseline_path, final_path])

        log(f"Reading final temporal OOF: {SEASONS[season]} {response}.")
        final_raw = pd.read_csv(final_path, compression="gzip", low_memory=False)
        if len(final_raw) != EXPECTED_ROWS_PER_SEASON_RESPONSE:
            raise ValueError(f"Unexpected row count in {final_path}.")
        final_raw["GRID_UID"] = final_raw["GRID_UID"].astype(str)
        final_raw["Year"] = pd.to_numeric(final_raw["Year"], errors="raise").astype(int)
        final_raw["Season"] = pd.to_numeric(
            final_raw["Season"], errors="raise"
        ).astype(int)
        final_raw["Count_Response"] = final_raw["Count_Response"].astype(str)
        if set(final_raw["Season"].unique()) != {season}:
            raise ValueError(f"Unexpected season in {final_path}.")
        if set(final_raw["Count_Response"].unique()) != {response}:
            raise ValueError(f"Unexpected response in {final_path}.")
        if final_raw.duplicated(["GRID_UID", "Year", "Season"]).any():
            raise ValueError(f"Duplicate panel keys in {final_path}.")

        if panel_reference is None:
            panel_reference = final_raw[
                ["GRID_UID", "Country", "GRID_ID", "Year"]
            ].copy()

        log(f"Reading baseline NB1 temporal OOF: {SEASONS[season]} {response}.")
        baseline_all = pd.read_csv(
            baseline_path, compression="gzip", low_memory=False
        )
        required_baseline = {
            "GRID_UID",
            "GRID_ID",
            "Country",
            "Year",
            "Season",
            "Temporal_Fold",
            "ForestPixelCount",
            "Count_Response",
            "Family",
            "Observed_Count",
            "Predicted_Count",
        }
        missing = sorted(required_baseline - set(baseline_all.columns))
        if missing:
            raise ValueError(
                f"{baseline_path.name} is missing columns:\n" + "\n".join(missing)
            )
        baseline_selected = baseline_all.loc[
            baseline_all["Family"].astype(str) == BASELINE_TEMPORAL_FAMILY
        ].copy()
        baseline_selected["GRID_UID"] = baseline_selected["GRID_UID"].astype(str)
        baseline_selected["Year"] = pd.to_numeric(
            baseline_selected["Year"], errors="raise"
        ).astype(int)
        baseline_selected["Season"] = pd.to_numeric(
            baseline_selected["Season"], errors="raise"
        ).astype(int)
        baseline_selected["Count_Response"] = baseline_selected[
            "Count_Response"
        ].astype(str)
        baseline_selected = baseline_selected.loc[
            (baseline_selected["Season"] == season)
            & (baseline_selected["Count_Response"] == response)
        ].copy()

        reference = final_raw[
            [
                "GRID_UID",
                "Year",
                "Season",
                "GRID_ID",
                "Country",
                "Temporal_Fold",
                "ForestPixelCount",
                "Observed_Count",
            ]
        ].rename(
            columns={
                "GRID_ID": "Reference_GRID_ID",
                "Country": "Reference_Country",
                "Temporal_Fold": "Reference_Temporal_Fold",
                "ForestPixelCount": "Reference_ForestPixelCount",
                "Observed_Count": "Reference_Observed_Count",
            }
        )
        baseline_selected = baseline_selected.merge(
            reference,
            on=["GRID_UID", "Year", "Season"],
            how="inner",
            validate="one_to_one",
        )
        if len(baseline_selected) != EXPECTED_ROWS_PER_SEASON_RESPONSE:
            raise ValueError(
                f"Baseline NB1 did not match the exact Step-06d sample for "
                f"{SEASONS[season]} {response}."
            )
        if not (
            baseline_selected["GRID_ID"].astype(str).to_numpy()
            == baseline_selected["Reference_GRID_ID"].astype(str).to_numpy()
        ).all():
            raise ValueError("Baseline and final GRID_ID values disagree.")
        if not (
            baseline_selected["Country"].astype(str).to_numpy()
            == baseline_selected["Reference_Country"].astype(str).to_numpy()
        ).all():
            raise ValueError("Baseline and final country values disagree.")
        if not np.allclose(
            pd.to_numeric(
                baseline_selected["ForestPixelCount"], errors="raise"
            ).to_numpy(dtype=float),
            pd.to_numeric(
                baseline_selected["Reference_ForestPixelCount"], errors="raise"
            ).to_numpy(dtype=float),
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError("Baseline and final exposure values disagree.")
        if not np.allclose(
            pd.to_numeric(
                baseline_selected["Observed_Count"], errors="raise"
            ).to_numpy(dtype=float),
            pd.to_numeric(
                baseline_selected["Reference_Observed_Count"], errors="raise"
            ).to_numpy(dtype=float),
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError("Baseline and final observed counts disagree.")
        if not (
            pd.to_numeric(
                baseline_selected["Temporal_Fold"], errors="raise"
            ).astype(int).to_numpy()
            == pd.to_numeric(
                baseline_selected["Reference_Temporal_Fold"], errors="raise"
            ).astype(int).to_numpy()
        ).all():
            raise ValueError("Baseline and final temporal folds disagree.")

        baseline_selected = baseline_selected.drop(
            columns=[
                "Reference_GRID_ID",
                "Reference_Country",
                "Reference_Temporal_Fold",
                "Reference_ForestPixelCount",
                "Reference_Observed_Count",
            ]
        )
        baseline_parts.append(baseline_selected)
        final_parts.append(final_raw)

        source_rows.extend(
            [
                {
                    "Validation_Scheme": "Temporal",
                    "Model_Role": "Baseline_NB1",
                    "Season_Code": season,
                    "Season_Label": SEASONS[season],
                    "Count_Response": response,
                    "Input_File": str(baseline_path),
                    "Input_File_SHA256": sha256_file(baseline_path),
                    "Raw_File_Rows": int(len(baseline_all)),
                    "Selected_Exact_Sample_Rows": int(len(baseline_selected)),
                    "Source_Filter": "Family == NB1 and exact Step-06d OOF keys",
                },
                {
                    "Validation_Scheme": "Temporal",
                    "Model_Role": "Final_Regression",
                    "Season_Code": season,
                    "Season_Label": SEASONS[season],
                    "Count_Response": response,
                    "Input_File": str(final_path),
                    "Input_File_SHA256": sha256_file(final_path),
                    "Raw_File_Rows": int(len(final_raw)),
                    "Selected_Exact_Sample_Rows": int(len(final_raw)),
                    "Source_Filter": "Step-06d final selected OOF file",
                },
            ]
        )

    baseline_combined = pd.concat(baseline_parts, ignore_index=True)
    final_combined = pd.concat(final_parts, ignore_index=True)
    baseline_frame, baseline_qa = finalize_oof_frame(
        baseline_combined,
        blocks,
        model_role="Baseline_NB1",
        validation_scheme="Temporal",
        source_name="Step-02 NB1 exact Step-06d temporal sample",
    )
    final_frame, final_qa = finalize_oof_frame(
        final_combined,
        blocks,
        model_role="Final_Regression",
        validation_scheme="Temporal",
        source_name="Step-06d final selected temporal OOF",
    )
    if baseline_qa["OOF_Key_Signature"] != final_qa["OOF_Key_Signature"]:
        raise ValueError("Temporal baseline and final OOF key universes differ.")
    if (
        baseline_qa["Observed_Panel_Signature"]
        != final_qa["Observed_Panel_Signature"]
    ):
        raise ValueError("Temporal baseline and final observed panels differ.")
    source_rows.extend([baseline_qa, final_qa])

    if panel_reference is None:
        raise RuntimeError("Temporal panel reference was not created.")
    return [baseline_frame, final_frame], source_rows, panel_reference


def read_spatial_oof(
    blocks: pd.DataFrame, manifest: pd.DataFrame
) -> Tuple[List[pd.DataFrame], List[Dict[str, object]]]:
    frames: List[pd.DataFrame] = []
    qa_rows: List[Dict[str, object]] = []
    for model_role in MODEL_ROLES:
        selected = manifest.loc[manifest["Model_Role"].astype(str) == model_role]
        if len(selected) != 1:
            raise ValueError(f"Spatial manifest did not identify {model_role}.")
        path = STEP07_ROOT / str(selected.iloc[0]["Output_File"])
        log(f"Reading {model_role} spatial OOF predictions.")
        raw = pd.read_csv(path, compression="gzip", low_memory=False)
        frame, qa = finalize_oof_frame(
            raw,
            blocks,
            model_role=model_role,
            validation_scheme="Spatial",
            source_name=path.name,
        )
        qa.update(
            {
                "Input_File": str(path),
                "Input_File_SHA256": sha256_file(path),
                "Raw_File_Rows": int(len(raw)),
                "Selected_Exact_Sample_Rows": int(len(frame)),
                "Source_Filter": "Locked Step-07 release spatial OOF",
            }
        )
        frames.append(frame)
        qa_rows.append(qa)
    if qa_rows[0]["OOF_Key_Signature"] != qa_rows[1]["OOF_Key_Signature"]:
        raise ValueError("Spatial baseline and final OOF key universes differ.")
    if (
        qa_rows[0]["Observed_Panel_Signature"]
        != qa_rows[1]["Observed_Panel_Signature"]
    ):
        raise ValueError("Spatial baseline and final observed panels differ.")
    return frames, qa_rows


# =============================================================================
# Time-support audit
# =============================================================================


def build_time_support_audit(
    panel_reference: pd.DataFrame,
    residuals: pd.DataFrame,
    blocks: pd.DataFrame,
) -> pd.DataFrame:
    reference = panel_reference.copy()
    reference["GRID_UID"] = reference["GRID_UID"].astype(str)
    reference["Year"] = pd.to_numeric(reference["Year"], errors="raise").astype(int)
    year_lists = (
        reference.groupby("GRID_UID")["Year"]
        .agg(lambda values: sorted(set(int(value) for value in values)))
        .rename("Observed_Years_List")
    )
    support = blocks.copy().set_index("GRID_UID")
    support = support.join(year_lists, how="left")
    if support["Observed_Years_List"].isna().any():
        raise ValueError("A locked grid is absent from the temporal panel reference.")

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
    support = support.join(residual_years, how="left")
    support = support.rename(
        columns={
            "min": "Minimum_Years_Observed_Across_All_24_Combinations",
            "max": "Maximum_Years_Observed_Across_All_24_Combinations",
            "nunique": "Distinct_Years_Observed_Counts_Across_Combinations",
        }
    )
    if support[
        [
            "Minimum_Years_Observed_Across_All_24_Combinations",
            "Maximum_Years_Observed_Across_All_24_Combinations",
        ]
    ].isna().any().any():
        raise ValueError("Residual time support could not be attached to all grids.")
    if not (
        support["Minimum_Years_Observed_Across_All_24_Combinations"].astype(int)
        == support["Years_Observed"]
    ).all() or not (
        support["Maximum_Years_Observed_Across_All_24_Combinations"].astype(int)
        == support["Years_Observed"]
    ).all():
        raise ValueError(
            "Temporal support differs between model, response, season, or OOF scheme."
        )

    support = support.drop(columns=["Observed_Years_List", "Missing_Years_List"])
    support = support.reset_index().sort_values("GRID_UID").reset_index(drop=True)

    complete_count = int(support["Complete_25_Years"].sum())
    partial_count = int((~support["Complete_25_Years"]).sum())
    if complete_count != EXPECTED_COMPLETE_25Y_GRID_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_COMPLETE_25Y_GRID_COUNT:,} complete grids, "
            f"found {complete_count:,}."
        )
    if partial_count != EXPECTED_PARTIAL_TIME_SUPPORT_GRID_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_PARTIAL_TIME_SUPPORT_GRID_COUNT} partial grids, "
            f"found {partial_count}."
        )
    return support


# =============================================================================
# Spatial weights
# =============================================================================


def build_knn_weights(
    blocks: pd.DataFrame, k: int, sample_definition: str
) -> Tuple[sparse.csr_matrix, pd.DataFrame, Dict[str, object]]:
    if len(blocks) <= k:
        raise ValueError("Not enough grid centroids for the requested k.")
    local = blocks.sort_values("GRID_UID").reset_index(drop=True).copy()
    coordinates = local[
        ["Centroid_X_UTM52_m", "Centroid_Y_UTM52_m"]
    ].to_numpy(dtype=float)
    tree = cKDTree(coordinates)
    distances, neighbors = tree.query(coordinates, k=k + 1)
    if not np.array_equal(neighbors[:, 0], np.arange(len(local))):
        raise ValueError("The nearest-neighbor query did not return self first.")

    neighbor_indices = neighbors[:, 1:]
    neighbor_distances = distances[:, 1:]
    if (neighbor_indices == np.arange(len(local))[:, None]).any():
        raise ValueError("Self-neighbors remained in the kNN result.")
    rows = np.repeat(np.arange(len(local)), k)
    columns = neighbor_indices.reshape(-1)
    values = np.full(len(rows), 1.0 / k, dtype=float)
    weights = sparse.csr_matrix(
        (values, (rows, columns)), shape=(len(local), len(local))
    )
    row_sums = np.asarray(weights.sum(axis=1)).ravel()
    if not np.allclose(row_sums, 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("The kNN weights are not row-standardized.")
    if weights.diagonal().any():
        raise ValueError("The kNN matrix contains self-weights.")

    binary = weights.copy()
    binary.data = np.ones_like(binary.data)
    reciprocal_directed_edges = int(binary.multiply(binary.T).nnz)
    symmetrized = binary.maximum(binary.T)
    component_count, component_labels = connected_components(
        symmetrized, directed=False, return_labels=True
    )
    component_sizes = np.bincount(component_labels)

    source_index = np.repeat(np.arange(len(local)), k)
    edge_frame = pd.DataFrame(
        {
            "Sample_Definition": sample_definition,
            "Source_GRID_UID": local.iloc[source_index]["GRID_UID"].to_numpy(),
            "Source_Country": local.iloc[source_index]["Country"].to_numpy(),
            "Source_GRID_ID": local.iloc[source_index]["GRID_ID"].to_numpy(),
            "Neighbor_Rank": np.tile(np.arange(1, k + 1), len(local)),
            "Neighbor_GRID_UID": local.iloc[columns]["GRID_UID"].to_numpy(),
            "Neighbor_Country": local.iloc[columns]["Country"].to_numpy(),
            "Neighbor_GRID_ID": local.iloc[columns]["GRID_ID"].to_numpy(),
            "Distance_m": neighbor_distances.reshape(-1),
            "Distance_km": neighbor_distances.reshape(-1) / 1000.0,
            "Row_Standardized_Weight": values,
        }
    )
    kth_distance = neighbor_distances[:, -1] / 1000.0
    metadata = {
        "Sample_Definition": sample_definition,
        "K_Neighbors": int(k),
        "GRID_UID_Count": int(len(local)),
        "Directed_Edge_Count": int(weights.nnz),
        "Reciprocal_Directed_Edge_Count": reciprocal_directed_edges,
        "Reciprocal_Directed_Edge_Proportion": float(
            reciprocal_directed_edges / weights.nnz
        ),
        "Weak_Component_Count": int(component_count),
        "Largest_Weak_Component_Size": int(component_sizes.max()),
        "Smallest_Weak_Component_Size": int(component_sizes.min()),
        "Minimum_Neighbor_Distance_km": float(neighbor_distances.min() / 1000.0),
        "Median_Neighbor_Distance_km": float(
            np.median(neighbor_distances) / 1000.0
        ),
        "Maximum_Neighbor_Distance_km": float(neighbor_distances.max() / 1000.0),
        "Mean_Kth_Neighbor_Distance_km": float(kth_distance.mean()),
        "Median_Kth_Neighbor_Distance_km": float(np.median(kth_distance)),
        "Maximum_Kth_Neighbor_Distance_km": float(kth_distance.max()),
        "Row_Sum_Minimum": float(row_sums.min()),
        "Row_Sum_Maximum": float(row_sums.max()),
        "S0_Total_Weight": float(weights.sum()),
    }
    return weights, edge_frame, metadata


# =============================================================================
# Moran's I
# =============================================================================


def morans_i_permutation(
    values: np.ndarray,
    weights: sparse.csr_matrix,
    permutations: int,
    seed: int,
    batch_size: int = PERMUTATION_BATCH_SIZE,
) -> Dict[str, float]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("Moran values must be one-dimensional.")
    if len(values) != weights.shape[0] or weights.shape[0] != weights.shape[1]:
        raise ValueError("Moran values and weights have incompatible dimensions.")
    if not np.isfinite(values).all():
        raise ValueError("Moran values contain non-finite entries.")
    if permutations < 1:
        raise ValueError("At least one permutation is required.")

    n = len(values)
    z = values - values.mean()
    denominator = float(np.dot(z, z))
    expected_i = -1.0 / (n - 1)
    if denominator <= 0 or not np.isfinite(denominator):
        return {
            "Moran_I": np.nan,
            "Expected_I": expected_i,
            "Moran_Excess_Over_Expected": np.nan,
            "Permutation_P_TwoSided": np.nan,
            "Permutation_P_Greater": np.nan,
            "Permutation_P_Less": np.nan,
            "Permutation_Mean": np.nan,
            "Permutation_SD": np.nan,
            "Permutation_Z_Score": np.nan,
            "Permutation_P_TwoSided_Monte_Carlo_SE": np.nan,
        }

    s0 = float(weights.sum())
    if s0 <= 0:
        raise ValueError("Spatial weights have non-positive total weight.")
    scale = n / s0
    observed_i = float(scale * np.dot(z, weights.dot(z)) / denominator)

    rng = np.random.default_rng(seed)
    permutation_values = np.empty(permutations, dtype=float)
    start = 0
    while start < permutations:
        current = min(batch_size, permutations - start)
        permuted = np.column_stack([rng.permutation(z) for _ in range(current)])
        weighted = weights.dot(permuted)
        numerators = np.einsum("ij,ij->j", permuted, weighted)
        permutation_values[start : start + current] = scale * numerators / denominator
        start += current

    observed_distance = abs(observed_i - expected_i)
    permutation_distance = np.abs(permutation_values - expected_i)
    p_two_sided = (
        1 + int((permutation_distance >= observed_distance).sum())
    ) / (permutations + 1)
    p_greater = (
        1 + int((permutation_values >= observed_i).sum())
    ) / (permutations + 1)
    p_less = (
        1 + int((permutation_values <= observed_i).sum())
    ) / (permutations + 1)
    permutation_mean = float(permutation_values.mean())
    permutation_sd = float(permutation_values.std(ddof=1))
    z_score = (
        float((observed_i - permutation_mean) / permutation_sd)
        if permutation_sd > 0
        else np.nan
    )
    monte_carlo_se = math.sqrt(
        p_two_sided * (1.0 - p_two_sided) / (permutations + 1)
    )
    return {
        "Moran_I": observed_i,
        "Expected_I": expected_i,
        "Moran_Excess_Over_Expected": observed_i - expected_i,
        "Permutation_P_TwoSided": float(p_two_sided),
        "Permutation_P_Greater": float(p_greater),
        "Permutation_P_Less": float(p_less),
        "Permutation_Mean": permutation_mean,
        "Permutation_SD": permutation_sd,
        "Permutation_Z_Score": z_score,
        "Permutation_P_TwoSided_Monte_Carlo_SE": float(monte_carlo_se),
    }


def residual_vector(
    residuals: pd.DataFrame,
    sample_blocks: pd.DataFrame,
    sample_definition: str,
    validation_scheme: str,
    model_role: str,
    season: int,
    response: str,
) -> Tuple[np.ndarray, Dict[str, object]]:
    selected = residuals.loc[
        (residuals["Validation_Scheme"] == validation_scheme)
        & (residuals["Model_Role"] == model_role)
        & (residuals["Season"] == season)
        & (residuals["Count_Response"] == response)
        & (residuals["GRID_UID"].isin(sample_blocks["GRID_UID"]))
    ].copy()
    expected_n = len(sample_blocks)
    if len(selected) != expected_n:
        raise ValueError(
            f"Expected {expected_n:,} residual grids for {sample_definition}, "
            f"{validation_scheme}, {model_role}, {season}, {response}; "
            f"found {len(selected):,}."
        )
    if selected["GRID_UID"].duplicated().any():
        raise ValueError("Duplicate grid residual rows were found.")
    aligned = (
        selected.set_index("GRID_UID")[PRIMARY_RESIDUAL_FIELD]
        .reindex(sample_blocks["GRID_UID"])
    )
    if aligned.isna().any():
        raise ValueError("Residual vector could not be aligned to sample grids.")
    values = aligned.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Aligned residual vector contains non-finite values.")
    metadata = {
        "GRID_UID_Count": int(len(values)),
        "Years_Observed_Minimum": int(selected["Years_Observed"].min()),
        "Years_Observed_Median": float(selected["Years_Observed"].median()),
        "Years_Observed_Maximum": int(selected["Years_Observed"].max()),
        "Complete_25_Year_GRID_UID_Count": int(
            (selected["Years_Observed"] == 25).sum()
        ),
        "Residual_Mean": float(values.mean()),
        "Residual_SD": float(values.std(ddof=1)),
        "Residual_Minimum": float(values.min()),
        "Residual_Maximum": float(values.max()),
        "Residual_Mean_Absolute_Value": float(np.mean(np.abs(values))),
        "Residual_Root_Mean_Square": float(np.sqrt(np.mean(np.square(values)))),
    }
    return values, metadata


def apply_primary_fdr(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["Permutation_P_TwoSided_Global_BH_FDR"] = bh_adjust(
        result["Permutation_P_TwoSided"].to_numpy(dtype=float)
    )
    result["Permutation_P_TwoSided_Validation_Scheme_BH_FDR"] = np.nan
    for _, index in result.groupby("Validation_Scheme").groups.items():
        result.loc[index, "Permutation_P_TwoSided_Validation_Scheme_BH_FDR"] = (
            bh_adjust(result.loc[index, "Permutation_P_TwoSided"].to_numpy(dtype=float))
        )
    result["Permutation_P_TwoSided_Scheme_Model_BH_FDR"] = np.nan
    for _, index in result.groupby(["Validation_Scheme", "Model_Role"]).groups.items():
        result.loc[index, "Permutation_P_TwoSided_Scheme_Model_BH_FDR"] = bh_adjust(
            result.loc[index, "Permutation_P_TwoSided"].to_numpy(dtype=float)
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


def apply_sensitivity_fdr(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["Permutation_P_TwoSided_Sample_K_BH_FDR"] = np.nan
    for _, index in result.groupby(["Sample_Definition", "K_Neighbors"]).groups.items():
        result.loc[index, "Permutation_P_TwoSided_Sample_K_BH_FDR"] = bh_adjust(
            result.loc[index, "Permutation_P_TwoSided"].to_numpy(dtype=float)
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
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    primary_rows: List[Dict[str, object]] = []
    complete_rows: List[Dict[str, object]] = []
    knn_rows: List[Dict[str, object]] = []

    for sample_definition in SAMPLE_DEFINITIONS:
        blocks = sample_blocks[sample_definition]
        for validation_scheme in VALIDATION_SCHEMES:
            for season, season_label in SEASONS.items():
                for response in COUNT_RESPONSES:
                    # The seed intentionally excludes sample, validation scheme,
                    # and model role. Thus the same season-response/k test uses
                    # a common deterministic permutation stream whenever matrix
                    # dimensions permit, improving reproducible comparisons.
                    primary_seed = deterministic_seed(
                        RANDOM_SEED,
                        season,
                        response,
                        PRIMARY_RESIDUAL_FIELD,
                        PRIMARY_K_NEIGHBORS,
                        "primary",
                    )
                    for model_role in MODEL_ROLES:
                        values, vector_metadata = residual_vector(
                            residuals=residuals,
                            sample_blocks=blocks,
                            sample_definition=sample_definition,
                            validation_scheme=validation_scheme,
                            model_role=model_role,
                            season=season,
                            response=response,
                        )
                        log(
                            f"Moran primary: {sample_definition}, "
                            f"{validation_scheme}, {model_role}, "
                            f"{season_label}, {response}, k=8."
                        )
                        moran = morans_i_permutation(
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
                            "Model_Role": model_role,
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
                            sensitivity_seed = deterministic_seed(
                                RANDOM_SEED,
                                season,
                                response,
                                PRIMARY_RESIDUAL_FIELD,
                                k,
                                "weights_sensitivity",
                            )
                            log(
                                f"Moran k sensitivity: {sample_definition}, "
                                f"{validation_scheme}, {model_role}, "
                                f"{season_label}, {response}, k={k}."
                            )
                            sensitivity = morans_i_permutation(
                                values,
                                weight_matrices[(sample_definition, k)],
                                permutations=SENSITIVITY_PERMUTATIONS,
                                seed=sensitivity_seed,
                            )
                            knn_rows.append(
                                {
                                    "Analysis_Type": "Weights_Sensitivity",
                                    "Sample_Definition": sample_definition,
                                    "Validation_Scheme": validation_scheme,
                                    "Model_Role": model_role,
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
                                    "Permutation_Seed": sensitivity_seed,
                                    **vector_metadata,
                                    **sensitivity,
                                }
                            )

    primary = pd.DataFrame(primary_rows)
    complete = pd.DataFrame(complete_rows)
    knn = pd.DataFrame(knn_rows)
    if len(primary) != 24:
        raise RuntimeError(f"Expected 24 main primary tests, found {len(primary)}.")
    if len(complete) != 24:
        raise RuntimeError(
            f"Expected 24 complete-sample tests, found {len(complete)}."
        )
    if len(knn) != 96:
        raise RuntimeError(f"Expected 96 kNN sensitivity tests, found {len(knn)}.")

    primary = apply_primary_fdr(primary)
    complete = apply_primary_fdr(complete)
    knn = apply_sensitivity_fdr(knn)
    sort_columns = [
        "Validation_Scheme",
        "Season_Code",
        "Count_Response",
        "Model_Role",
    ]
    primary = primary.sort_values(sort_columns).reset_index(drop=True)
    complete = complete.sort_values(sort_columns).reset_index(drop=True)
    knn = knn.sort_values(
        [
            "Sample_Definition",
            "Validation_Scheme",
            "Season_Code",
            "Count_Response",
            "Model_Role",
            "K_Neighbors",
        ]
    ).reset_index(drop=True)
    return primary, complete, knn


# =============================================================================
# Comparison tables
# =============================================================================


def build_baseline_final_comparison(all_primary: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "Sample_Definition",
        "Validation_Scheme",
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
        "Permutation_P_TwoSided_Global_BH_FDR",
        "Significant_Global_BH_FDR_0p05",
        "Residual_Mean_Absolute_Value",
        "Residual_Root_Mean_Square",
    ]
    baseline = all_primary.loc[
        all_primary["Model_Role"] == "Baseline_NB1", [*keys, *metrics]
    ].rename(columns={metric: f"Baseline_{metric}" for metric in metrics})
    final = all_primary.loc[
        all_primary["Model_Role"] == "Final_Regression", [*keys, *metrics]
    ].rename(columns={metric: f"Final_{metric}" for metric in metrics})
    result = baseline.merge(final, on=keys, how="inner", validate="one_to_one")
    result["Final_minus_Baseline_Moran_I"] = (
        result["Final_Moran_I"] - result["Baseline_Moran_I"]
    )
    result["Baseline_Absolute_Deviation_From_Expected"] = np.abs(
        result["Baseline_Moran_I"] - result["Baseline_Expected_I"]
    )
    result["Final_Absolute_Deviation_From_Expected"] = np.abs(
        result["Final_Moran_I"] - result["Final_Expected_I"]
    )
    result["Final_minus_Baseline_Absolute_Deviation_From_Expected"] = (
        result["Final_Absolute_Deviation_From_Expected"]
        - result["Baseline_Absolute_Deviation_From_Expected"]
    )
    result["Final_Reduced_Absolute_Residual_Autocorrelation"] = (
        result["Final_Absolute_Deviation_From_Expected"]
        < result["Baseline_Absolute_Deviation_From_Expected"]
    )
    result["Final_Reduced_Residual_RMSE_Across_Grids"] = (
        result["Final_Residual_Root_Mean_Square"]
        < result["Baseline_Residual_Root_Mean_Square"]
    )
    result["Comparison_Note"] = (
        "Descriptive paired comparison on the same grids and OOF scheme; "
        "not a formal test of the difference between two Moran statistics."
    )
    return result.sort_values(
        ["Sample_Definition", "Validation_Scheme", "Season_Code", "Count_Response"]
    ).reset_index(drop=True)


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
        "The main sample preserves the locked 4,982-grid common universe and "
        "averages over available years. The complete sample restricts all "
        "models and OOF schemes to the same 4,924 grids with 25 years."
    )
    return result.sort_values(
        ["Validation_Scheme", "Model_Role", "Season_Code", "Count_Response"]
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
        ["Sample_Definition", "Model_Role", "Season_Code", "Count_Response"]
    ).reset_index(drop=True)


# =============================================================================
# Output manifest
# =============================================================================


def build_output_manifest(paths: Sequence[Path]) -> pd.DataFrame:
    rows = []
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
# Main workflow
# =============================================================================


def main() -> int:
    require_files(
        [
            STEP02_METHOD_FILE,
            STEP06D_OOF_MANIFEST_FILE,
            STEP07_METHOD_FILE,
            STEP07_OOF_MANIFEST_FILE,
            SPATIAL_BLOCK_FILE,
        ]
    )

    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text("", encoding="utf-8")

    log("Starting Step 08 unified regression residual Moran analysis.")
    log(f"Step 08 code version: {STEP08_CODE_VERSION}.")
    log(
        "Temporal and spatial OOF residuals are analyzed separately; no "
        "09_RF_modeling file is read."
    )

    step07_method = read_and_verify_step07_method()
    blocks, block_metadata = read_and_verify_blocks()
    spatial_manifest = read_and_verify_spatial_manifest()
    log("Verified the locked 4,982-grid Step-07 assignment and release spatial OOF files.")

    temporal_frames, temporal_source_rows, panel_reference = read_temporal_oof(blocks)
    spatial_frames, spatial_source_rows = read_spatial_oof(blocks, spatial_manifest)
    all_frames = [*temporal_frames, *spatial_frames]

    signatures = []
    for frame in all_frames:
        signatures.append(
            (
                order_independent_row_signature(
                    frame,
                    ["GRID_UID", "Year", "Season", "Count_Response"],
                ),
                order_independent_row_signature(
                    frame,
                    [
                        "GRID_UID",
                        "Year",
                        "Season",
                        "Count_Response",
                        "Observed_Count",
                        "ForestPixelCount",
                    ],
                ),
            )
        )
    if len({item[0] for item in signatures}) != 1:
        raise ValueError("Temporal and spatial OOF key universes differ.")
    if len({item[1] for item in signatures}) != 1:
        raise ValueError("Temporal and spatial observed OOF panels differ.")

    expected_oof_rows = EXPECTED_ROWS_PER_MODEL_ROLE * 4
    observed_oof_rows = int(sum(len(frame) for frame in all_frames))
    if observed_oof_rows != expected_oof_rows:
        raise ValueError(
            f"Expected {expected_oof_rows:,} combined OOF rows, "
            f"found {observed_oof_rows:,}."
        )

    # The common 4,982-grid workflow may contain partial time support, but the
    # exact GRID_UID-Year availability must be identical for every model role,
    # response, season and OOF scheme. This makes the single reference panel
    # used below valid for the complete-25-year sensitivity definition.
    support_signatures = []
    for frame in all_frames:
        for _, subset in frame.groupby(
            ["Season", "Count_Response"], sort=True
        ):
            support_signatures.append(
                order_independent_row_signature(subset, ["GRID_UID", "Year"])
            )
    if len(set(support_signatures)) != 1:
        raise ValueError(
            "GRID_UID-Year availability differs across model, response, "
            "season, or OOF scheme."
        )

    log(f"Verified and aligned {observed_oof_rows:,} temporal/spatial OOF rows.")

    # Aggregate one source frame at a time to avoid constructing a second
    # multi-million-row combined copy in memory.
    residual_frames = [aggregate_grid_residuals(frame) for frame in all_frames]
    residuals = pd.concat(residual_frames, ignore_index=True)
    expected_residual_rows = (
        EXPECTED_GRID_COUNT * 6 * len(MODEL_ROLES) * len(VALIDATION_SCHEMES)
    )
    if len(residuals) != expected_residual_rows:
        raise ValueError(
            f"Expected {expected_residual_rows:,} combined residual rows, "
            f"found {len(residuals):,}."
        )
    del residual_frames
    residuals.to_csv(GRID_RESIDUAL_FILE, index=False, compression="gzip")
    log(f"Wrote {len(residuals):,} model-scheme-season-response-grid residual rows.")

    source_qa = pd.DataFrame([*temporal_source_rows, *spatial_source_rows])
    source_qa.to_csv(INPUT_QA_FILE, index=False, encoding="utf-8-sig")

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
        .sort_values(
            ["Validation_Scheme", "Model_Role", "Season", "Count_Response"]
        )
        .reset_index(drop=True)
    )
    residual_qa.to_csv(RESIDUAL_QA_FILE, index=False, encoding="utf-8-sig")

    time_support = build_time_support_audit(panel_reference, residuals, blocks)
    time_support.to_csv(TIME_SUPPORT_FILE, index=False, encoding="utf-8-sig")
    partial_support = time_support.loc[~time_support["Complete_25_Years"]].copy()
    partial_support.to_csv(
        PARTIAL_TIME_SUPPORT_FILE, index=False, encoding="utf-8-sig"
    )
    log(
        f"Time support: {int(time_support['Complete_25_Years'].sum()):,} complete "
        f"25-year grids and {len(partial_support):,} partial grids."
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
    if len(sample_blocks[COMPLETE_SAMPLE]) != EXPECTED_COMPLETE_25Y_GRID_COUNT:
        raise ValueError("Complete-grid sample size is incorrect.")

    weight_matrices: Dict[Tuple[str, int], sparse.csr_matrix] = {}
    weight_qa_rows: List[Dict[str, object]] = []
    main_edges: pd.DataFrame | None = None
    complete_edges: pd.DataFrame | None = None
    for sample_definition in SAMPLE_DEFINITIONS:
        for k in [*SENSITIVITY_K_NEIGHBORS, PRIMARY_K_NEIGHBORS]:
            weights, edges, metadata = build_knn_weights(
                sample_blocks[sample_definition], k, sample_definition
            )
            weight_matrices[(sample_definition, k)] = weights
            metadata["Analysis_Role"] = (
                "Primary" if k == PRIMARY_K_NEIGHBORS else "Weights_Sensitivity"
            )
            weight_qa_rows.append(metadata)
            if k == PRIMARY_K_NEIGHBORS and sample_definition == MAIN_SAMPLE:
                main_edges = edges
            if k == PRIMARY_K_NEIGHBORS and sample_definition == COMPLETE_SAMPLE:
                complete_edges = edges
            log(
                f"Built {sample_definition} k={k} weights: "
                f"{weights.nnz:,} directed edges."
            )
    pd.DataFrame(weight_qa_rows).sort_values(
        ["Sample_Definition", "K_Neighbors"]
    ).to_csv(WEIGHT_QA_FILE, index=False, encoding="utf-8-sig")
    if main_edges is None or complete_edges is None:
        raise RuntimeError("A primary edge list was not created.")
    main_edges.to_csv(MAIN_EDGE_FILE, index=False, compression="gzip")
    complete_edges.to_csv(COMPLETE_EDGE_FILE, index=False, compression="gzip")

    primary, complete, knn = run_moran_tests(
        residuals=residuals,
        sample_blocks=sample_blocks,
        weight_matrices=weight_matrices,
    )
    primary.to_csv(PRIMARY_MORAN_FILE, index=False, encoding="utf-8-sig")
    complete.to_csv(COMPLETE_MORAN_FILE, index=False, encoding="utf-8-sig")
    knn.to_csv(KNN_SENSITIVITY_FILE, index=False, encoding="utf-8-sig")

    all_primary = pd.concat([primary, complete], ignore_index=True)
    baseline_final = build_baseline_final_comparison(all_primary)
    baseline_final.to_csv(
        BASELINE_FINAL_COMPARISON_FILE, index=False, encoding="utf-8-sig"
    )
    time_comparison = build_time_support_comparison(all_primary)
    time_comparison.to_csv(
        TIME_SUPPORT_COMPARISON_FILE, index=False, encoding="utf-8-sig"
    )
    scheme_comparison = build_temporal_spatial_comparison(all_primary)
    scheme_comparison.to_csv(
        TEMPORAL_SPATIAL_COMPARISON_FILE, index=False, encoding="utf-8-sig"
    )

    step02_method = json.loads(STEP02_METHOD_FILE.read_text(encoding="utf-8"))
    method_definition = {
        "Step": "08_Regression_Residual_Spatial_Autocorrelation",
        "Created": datetime.now().isoformat(timespec="seconds"),
        "Step08_Code_Version": STEP08_CODE_VERSION,
        "Purpose": (
            "Unified temporal- and spatial-OOF residual Moran evaluation for "
            "locked Baseline_NB1 and Final_Regression models."
        ),
        "No_Model_Refitting": True,
        "No_Model_Selection": True,
        "No_09_RF_modeling_Input": True,
        "Upstream_Inputs": {
            "Baseline_Temporal_OOF_Root": str(STEP02_ROOT),
            "Baseline_Temporal_Family": BASELINE_TEMPORAL_FAMILY,
            "Step02_Method_Step": step02_method.get("step"),
            "Final_Temporal_OOF_Manifest": str(STEP06D_OOF_MANIFEST_FILE),
            "Spatial_OOF_Manifest": str(STEP07_OOF_MANIFEST_FILE),
            "Step07_Code_Version": step07_method.get("Step07_Code_Version"),
            "Spatial_Prediction_Protocol_ID": step07_method.get(
                "Spatial_Prediction_Protocol_ID"
            ),
            **block_metadata,
        },
        "Validation_Schemes": {
            "Temporal": (
                "Five temporal folds; baseline NB1 is read from Step 02 and "
                "restricted to the exact Step-06d final OOF keys; final "
                "regression is read from Step 06d."
            ),
            "Spatial": (
                "Five locked Step-07 spatial blocks; both model roles are "
                "read from Step-07 release spatial OOF files."
            ),
            "Reporting_Rule": (
                "Temporal and spatial OOF Moran results are never pooled into "
                "one statistic and are compared only descriptively."
            ),
        },
        "Residual_Aggregation": {
            "Primary_Field": PRIMARY_RESIDUAL_FIELD,
            "Definition": PRIMARY_RESIDUAL_DEFINITION,
            "Main_Sample": {
                "Name": MAIN_SAMPLE,
                "GRID_UID_Count": EXPECTED_GRID_COUNT,
                "Temporal_Aggregation": (
                    "Arithmetic mean across each grid's available OOF years, "
                    "matching the original final-RF residual-Moran sample rule."
                ),
            },
            "Complete_Time_Support_Sensitivity": {
                "Name": COMPLETE_SAMPLE,
                "GRID_UID_Count": EXPECTED_COMPLETE_25Y_GRID_COUNT,
                "Temporal_Aggregation": (
                    "Arithmetic mean across exactly 25 years for every grid."
                ),
            },
            "Why_Rate_Residual_Is_Primary": (
                "Rate residuals place grids with different ForestPixelCount "
                "exposure on a common scale."
            ),
        },
        "Primary_Moran_Test": {
            "Primary_Sample": MAIN_SAMPLE,
            "Test_Count": 24,
            "Coordinates": ["Centroid_X_UTM52_m", "Centroid_Y_UTM52_m"],
            "Weights": "8-nearest-neighbor directed row-standardized",
            "Permutations_Per_Test": PRIMARY_PERMUTATIONS,
            "P_Value": (
                "Two-sided permutation p-value centered on -1/(n-1), with "
                "plus-one correction."
            ),
            "Permutation_Seed_Rule": (
                "Same deterministic seed for a season-response-k combination "
                "across model roles and temporal/spatial OOF schemes."
            ),
            "FDR": {
                "Global": "BH across all 24 main-sample primary tests",
                "Within_Validation_Scheme": "BH across 12 tests per OOF scheme",
                "Within_Scheme_Model": "BH across 6 tests per scheme/model role",
                "Alpha": FDR_ALPHA,
            },
        },
        "Sensitivity_Analyses": {
            "Complete_25Y": {
                "Test_Count": 24,
                "K_Neighbors": PRIMARY_K_NEIGHBORS,
                "Permutations_Per_Test": PRIMARY_PERMUTATIONS,
            },
            "KNN_Weights": {
                "Test_Count": 96,
                "K_Neighbors": SENSITIVITY_K_NEIGHBORS,
                "Permutations_Per_Test": SENSITIVITY_PERMUTATIONS,
                "Samples": SAMPLE_DEFINITIONS,
            },
        },
        "Future_RF_Alignment": {
            "Required_RF_OOF_Schemes": VALIDATION_SCHEMES,
            "Required_Samples": SAMPLE_DEFINITIONS,
            "Required_Residual": "Observed rate minus predicted rate",
            "Required_Primary_K": PRIMARY_K_NEIGHBORS,
            "Required_Primary_Permutations": PRIMARY_PERMUTATIONS,
            "Note": (
                "Later RF residual Moran recalculation should use this same "
                "protocol without retraining the RF models."
            ),
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
        "Step 08 completed: 24 main-sample primary, 24 complete-25-year "
        "time-support sensitivity, and 96 additional kNN-sensitivity Moran "
        "tests were calculated separately for temporal and spatial OOF rate "
        "residuals."
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
        BASELINE_FINAL_COMPARISON_FILE,
        TIME_SUPPORT_COMPARISON_FILE,
        TEMPORAL_SPATIAL_COMPARISON_FILE,
        software_file,
        LOG_FILE,
    ]
    output_manifest = build_output_manifest(output_paths)
    output_manifest["Step08_Code_Version"] = STEP08_CODE_VERSION
    output_manifest["Spatial_Mapping_SHA256"] = EXPECTED_SPATIAL_MAPPING_SHA256
    output_manifest.to_csv(
        OUTPUT_MANIFEST_FILE, index=False, encoding="utf-8-sig"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        error_text = traceback.format_exc()
        try:
            if OUTPUT_ROOT.exists():
                LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
                with LOG_FILE.open("a", encoding="utf-8") as handle:
                    handle.write("\n" + "=" * 78 + "\n")
                    handle.write("ERROR\n")
                    handle.write(error_text)
                    handle.write("\n" + "=" * 78 + "\n")
        except Exception:
            pass
        print(error_text, file=sys.stderr)
        raise
