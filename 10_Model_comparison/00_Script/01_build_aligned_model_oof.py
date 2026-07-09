# -*- coding: utf-8 -*-
"""
Step 01: build strictly aligned temporal and spatial OOF datasets for model comparison.

Recommended public location
---------------------------
<CODE_ROOT>/10_Model_comparison/00_Script/01_build_aligned_model_oof.py

Purpose
-------
Read the locked out-of-fold predictions from:

1. Baseline single-stage NB1 regression;
2. Final fixed-effect NB1/ZINB1 regression;
3. Final two-stage Hurdle random forest;

and place them on exactly the same GRID_UID-Year-Season-response rows for each
validation scheme. Temporal and spatial OOF predictions remain separate.

This step does not fit, tune, select, rank, or compare any model. It only:

* verifies source provenance and the shared 4,982-grid spatial assignment;
* normalizes model-specific prediction fields to one common definition;
* checks that all models use identical observed outcomes and exposures;
* writes paired wide-format OOF files for all downstream comparisons.

Outputs
-------
<CODE_ROOT>/10_Model_comparison/01_Aligned_Model_OOF/

    00_Method_Definition.json
    01_Input_Source_Manifest.csv
    02_Model_OOF_Normalization_QA.csv
    03_Alignment_and_Panel_QA.csv
    04_Probability_and_Conditional_QA.csv
    05_Aligned_Temporal_OOF.csv.gz
    06_Aligned_Spatial_OOF.csv.gz
    07_Aligned_OOF_Manifest.csv
    08_Column_Definitions.csv
    Software_Environment.json
    model_oof_alignment.log
"""

from __future__ import annotations

import gzip
import hashlib
import json
import platform
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# Repository-relative paths
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_COMPARISON_ROOT = SCRIPT_DIR.parent
CODE_ROOT = MODEL_COMPARISON_ROOT.parent

REGRESSION_ROOT = CODE_ROOT / "08_Regression_modeling"
RF_ROOT = CODE_ROOT / "09_RF_modeling"

STEP02_ROOT = REGRESSION_ROOT / "02_Single_Stage_Count_Family_Comparison"
STEP02_METHOD_FILE = STEP02_ROOT / "00_Method_Definition.json"

STEP06D_ROOT = REGRESSION_ROOT / "06d_Final_Stable_Fixed_Effect_Inference"
STEP06D_MANIFEST_FILE = STEP06D_ROOT / "17_Final_Selected_OOF_Manifest.csv"
STEP06D_OOF_ROOT = STEP06D_ROOT / "Final_Selected_Inference_OOF_Predictions"

STEP07_ROOT = REGRESSION_ROOT / "07_Regression_Spatial_Block_Validation"
STEP07_METHOD_FILE = STEP07_ROOT / "00_Method_Definition.json"
STEP07_MANIFEST_FILE = STEP07_ROOT / "08_Spatial_OOF_Manifest.csv"
SPATIAL_BLOCK_FILE = STEP07_ROOT / "Spatial_Block_Assignment.csv"

RF_VALIDATION_ROOT = RF_ROOT / "06_Final_Hurdle_RF_Validation"
RF_TEMPORAL_OOF_FILE = RF_VALIDATION_ROOT / "Temporal_Out_of_Fold_Predictions.csv.gz"
RF_SPATIAL_OOF_FILE = RF_VALIDATION_ROOT / "Spatial_Out_of_Fold_Predictions.csv.gz"
RF_UNIFIED_MORAN_ROOT = (
    RF_ROOT / "10_Final_Hurdle_RF_Residual_Spatial_Autocorrelation"
)
RF_UNIFIED_MORAN_METHOD_FILE = RF_UNIFIED_MORAN_ROOT / "00_Method_Definition.json"

OUTPUT_ROOT = MODEL_COMPARISON_ROOT / "01_Aligned_Model_OOF"
LOG_FILE = OUTPUT_ROOT / "model_oof_alignment.log"
METHOD_FILE = OUTPUT_ROOT / "00_Method_Definition.json"
SOURCE_MANIFEST_FILE = OUTPUT_ROOT / "01_Input_Source_Manifest.csv"
NORMALIZATION_QA_FILE = OUTPUT_ROOT / "02_Model_OOF_Normalization_QA.csv"
ALIGNMENT_QA_FILE = OUTPUT_ROOT / "03_Alignment_and_Panel_QA.csv"
PROBABILITY_QA_FILE = OUTPUT_ROOT / "04_Probability_and_Conditional_QA.csv"
TEMPORAL_OUTPUT_FILE = OUTPUT_ROOT / "05_Aligned_Temporal_OOF.csv.gz"
SPATIAL_OUTPUT_FILE = OUTPUT_ROOT / "06_Aligned_Spatial_OOF.csv.gz"
OUTPUT_MANIFEST_FILE = OUTPUT_ROOT / "07_Aligned_OOF_Manifest.csv"
COLUMN_DEFINITION_FILE = OUTPUT_ROOT / "08_Column_Definitions.csv"
SOFTWARE_FILE = OUTPUT_ROOT / "Software_Environment.json"


# =============================================================================
# Locked workflow settings
# =============================================================================

STEP01_CODE_VERSION = "2026-06-30_MODEL_COMPARISON_ALIGNED_OOF_V1"

EXPECTED_STEP07_CODE_VERSION = "2026-07-08_RELEASE_STABLE_POSITIVE_PROBABILITY"
EXPECTED_STEP07_PREDICTION_PROTOCOL_ID = (
    "STABLE_LOGSPACE_POSITIVE_PROBABILITY_AND_CONDITIONAL_MEAN_V4"
)
EXPECTED_RF_MORAN_CODE_VERSION = (
    "2026-06-30_RF_UNIFIED_TEMPORAL_SPATIAL_OOF_MORAN_V1B_UINT32_SEED_AUDIT"
)

EXPECTED_SPATIAL_MAPPING_SHA256 = (
    "715f9fe342eec78d936d223a98f5a4d0378281d3a3fa420ed250ba07fe056cfe"
)
EXPECTED_SPATIAL_BLOCK_FILE_SHA256 = (
    "da752eff8f734d29e0ff7b5535c93277b4b146b0b5a84e27efe3f08d17f61a11"
)
EXPECTED_BLOCK_GRID_COUNTS = {1: 912, 2: 1042, 3: 735, 4: 1222, 5: 1071}

EXPECTED_GRID_COUNT = 4_982
EXPECTED_ROWS_PER_SEASON_RESPONSE = 123_729
EXPECTED_ROWS_PER_MODEL = EXPECTED_ROWS_PER_SEASON_RESPONSE * 6
EXPECTED_YEARS = set(range(2001, 2026))
EXPECTED_TEMPORAL_FOLDS = {1, 2, 3, 4, 5}
EXPECTED_SPATIAL_BLOCKS = {1, 2, 3, 4, 5}

SEASONS = {1: "Spring", 2: "Summer", 3: "Autumn"}
COUNT_RESPONSES = ["Fire_Count", "Burned_Pixel_Count"]
RATE_NAMES = {"Fire_Count": "FCD", "Burned_Pixel_Count": "BAD"}
RESPONSE_FROM_RATE = {value: key for key, value in RATE_NAMES.items()}

MODEL_ROLES = ["Baseline_NB1", "Final_Regression", "Final_Hurdle_RF"]
VALIDATION_SCHEMES = ["Temporal", "Spatial"]

BASELINE_TEMPORAL_FAMILY = "NB1"
BASELINE_MODEL_STRUCTURE = "Single_Stage_NB1"
BASELINE_CANDIDATE_ID = "BASELINE_SINGLE_STAGE_NB1"
RF_MODEL_STRUCTURE = "Two_Stage_Hurdle_RF"
RF_CANDIDATE_ID = "LOCKED_FINAL_HURDLE_RF"

COORDINATE_COLUMNS = [
    "Centroid_X_UTM52_m",
    "Centroid_Y_UTM52_m",
    "Longitude",
    "Latitude",
]

KEY_COLUMNS = ["GRID_UID", "Year", "Season", "Count_Response"]
SHARED_COLUMNS = [
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
    "ForestPixelCount",
    "Observed_Count",
    "Observed_Rate",
    "Observed_Zero",
]

MODEL_VALUE_COLUMNS = [
    "Model_Structure",
    "Candidate_ID",
    "Predicted_Count",
    "Predicted_Rate",
    "Predicted_Zero_Probability",
    "Predicted_Positive_Probability",
    "Predicted_Conditional_Positive_Count",
    "Predicted_Conditional_Positive_Rate",
    "Predictive_LogProbability",
    "Residual_Count",
    "Residual_Rate",
    "Positive_Probability_Source",
    "Conditional_Positive_Source",
]

# Numerical tolerances are verification tolerances only. They do not modify
# any prediction used in the output.
ABSOLUTE_METADATA_TOLERANCE = 1e-8
ABSOLUTE_RATE_TOLERANCE = 1e-12
PROBABILITY_SUM_TOLERANCE = 5e-12
CONDITIONAL_IDENTITY_RELATIVE_TOLERANCE = 1e-10


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


def safe_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def maximum_absolute_difference(left: pd.Series, right: pd.Series) -> float:
    a = pd.to_numeric(left, errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(right, errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(a) & np.isfinite(b)
    if not finite.any():
        return 0.0
    return float(np.max(np.abs(a[finite] - b[finite])))


def canonical_mapping_sha256(blocks: pd.DataFrame) -> str:
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
        index=False,
        header=False,
        lineterminator="\n",
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def frame_signature(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    work = frame[list(columns)].copy()
    work = work.sort_values(list(columns[:4])).reset_index(drop=True)
    for column in work.columns:
        if pd.api.types.is_numeric_dtype(work[column]):
            work[column] = pd.to_numeric(work[column], errors="raise")
        else:
            work[column] = work[column].astype(str)
    hashes = pd.util.hash_pandas_object(work, index=False).to_numpy(dtype=np.uint64)
    return hashlib.sha256(hashes.tobytes()).hexdigest()


def step02_temporal_path(season: int, response: str) -> Path:
    return STEP02_ROOT / f"Temporal_OOF_Predictions_S{season}_{response}.csv.gz"


def final_temporal_path(output_file: str) -> Path:
    candidates = [STEP06D_OOF_ROOT / output_file, STEP06D_ROOT / output_file]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "A Step-06d final temporal OOF file was not found:\n"
        + "\n".join(str(path) for path in candidates)
    )


def read_csv_gzip(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, compression="gzip", low_memory=False)


# =============================================================================
# Shared spatial assignment and source manifests
# =============================================================================


def read_and_verify_blocks() -> Tuple[pd.DataFrame, Dict[str, object]]:
    require_files([SPATIAL_BLOCK_FILE])
    blocks = pd.read_csv(SPATIAL_BLOCK_FILE, low_memory=False)
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
            "Spatial_Block_Assignment.csv is missing columns:\n"
            + "\n".join(missing)
        )
    blocks = blocks[list(required)].copy()
    blocks["GRID_UID"] = blocks["GRID_UID"].astype(str)
    blocks["Country"] = blocks["Country"].astype(str)
    blocks["GRID_ID"] = blocks["GRID_ID"].astype(str)
    blocks["Spatial_Block"] = pd.to_numeric(
        blocks["Spatial_Block"], errors="raise"
    ).astype(int)
    for column in COORDINATE_COLUMNS:
        blocks[column] = pd.to_numeric(blocks[column], errors="raise")

    if len(blocks) != EXPECTED_GRID_COUNT or blocks["GRID_UID"].nunique() != EXPECTED_GRID_COUNT:
        raise ValueError("The shared assignment does not contain 4,982 unique grids.")
    if blocks["GRID_UID"].duplicated().any():
        raise ValueError("Duplicate GRID_UID values occur in the shared assignment.")
    counts = {
        int(block): int(count)
        for block, count in blocks["Spatial_Block"].value_counts().sort_index().items()
    }
    if counts != EXPECTED_BLOCK_GRID_COUNTS:
        raise ValueError(
            "Spatial-block counts differ from the locked design.\n"
            f"Expected: {EXPECTED_BLOCK_GRID_COUNTS}\nObserved: {counts}"
        )

    semantic_hash = canonical_mapping_sha256(blocks)
    byte_hash = sha256_file(SPATIAL_BLOCK_FILE)
    if semantic_hash != EXPECTED_SPATIAL_MAPPING_SHA256:
        raise ValueError("The semantic spatial mapping differs from the lock.")
    if byte_hash != EXPECTED_SPATIAL_BLOCK_FILE_SHA256:
        raise ValueError("The spatial-block file byte hash differs from the lock.")

    return (
        blocks.sort_values("GRID_UID").reset_index(drop=True),
        {
            "Path": str(SPATIAL_BLOCK_FILE),
            "SHA256": byte_hash,
            "Semantic_Mapping_SHA256": semantic_hash,
            "Rows": len(blocks),
            "GRID_UID_Count": blocks["GRID_UID"].nunique(),
        },
    )


def verify_upstream_methods() -> Dict[str, object]:
    require_files(
        [
            STEP02_METHOD_FILE,
            STEP06D_MANIFEST_FILE,
            STEP07_METHOD_FILE,
            STEP07_MANIFEST_FILE,
            RF_TEMPORAL_OOF_FILE,
            RF_SPATIAL_OOF_FILE,
        ]
    )

    step07 = safe_json(STEP07_METHOD_FILE)
    observed_step07_version = str(step07.get("Step07_Code_Version", ""))
    if observed_step07_version != EXPECTED_STEP07_CODE_VERSION:
        raise ValueError(
            "Step 07 is not the locked release version.\n"
            f"Expected: {EXPECTED_STEP07_CODE_VERSION}\n"
            f"Observed: {observed_step07_version}"
        )
    observed_protocol = str(step07.get("Spatial_Prediction_Protocol_ID", ""))
    if observed_protocol != EXPECTED_STEP07_PREDICTION_PROTOCOL_ID:
        raise ValueError("Step 07 spatial prediction protocol differs from the required release.")

    rf_moran_version = ""
    rf_moran_verified = False
    if RF_UNIFIED_MORAN_METHOD_FILE.exists():
        rf_method = safe_json(RF_UNIFIED_MORAN_METHOD_FILE)
        rf_moran_version = str(
            rf_method.get(
                "Step10_Code_Version",
                rf_method.get("Code_Version", ""),
            )
        )
        # The exact key differs between public metadata versions. The output
        # file is used as a completion/provenance check rather than as an OOF
        # data source.
        rf_moran_verified = (
            EXPECTED_RF_MORAN_CODE_VERSION in json.dumps(rf_method, ensure_ascii=False)
            or rf_moran_version == EXPECTED_RF_MORAN_CODE_VERSION
        )
        if not rf_moran_verified:
            raise ValueError(
                "The unified RF residual Moran method file is not the locked release version."
            )

    return {
        "Step07_Code_Version": observed_step07_version,
        "Step07_Spatial_Prediction_Protocol_ID": observed_protocol,
        "RF_Unified_Moran_Method_File_Exists": RF_UNIFIED_MORAN_METHOD_FILE.exists(),
        "RF_Unified_Moran_Code_Version": rf_moran_version,
        "RF_Unified_Moran_Release_Verified": rf_moran_verified,
    }


def read_final_temporal_manifest() -> pd.DataFrame:
    manifest = pd.read_csv(STEP06D_MANIFEST_FILE)
    required = {
        "Season_Code",
        "Season_Label",
        "Count_Response",
        "Output_File",
        "Rows",
    }
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(
            "The Step-06d OOF manifest is missing columns:\n" + "\n".join(missing)
        )
    expected = {(season, response) for season in SEASONS for response in COUNT_RESPONSES}
    observed = {
        (int(row.Season_Code), str(row.Count_Response))
        for row in manifest.itertuples(index=False)
    }
    if len(manifest) != 6 or observed != expected:
        raise ValueError("The Step-06d manifest does not contain the six combinations.")
    if not (
        pd.to_numeric(manifest["Rows"], errors="raise").astype(int)
        == EXPECTED_ROWS_PER_SEASON_RESPONSE
    ).all():
        raise ValueError("A Step-06d final temporal OOF row count is unexpected.")
    for output_file in manifest["Output_File"].astype(str):
        require_files([final_temporal_path(output_file)])
    return manifest.sort_values(["Season_Code", "Count_Response"]).reset_index(drop=True)


def read_spatial_manifest() -> pd.DataFrame:
    manifest = pd.read_csv(STEP07_MANIFEST_FILE)
    required = {"Model_Role", "Output_File", "Rows", "SHA256", "Unique_OOF_Keys"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(
            "The Step-07 OOF manifest is missing columns:\n" + "\n".join(missing)
        )
    expected_roles = {"Baseline_NB1", "Final_Regression"}
    if len(manifest) != 2 or set(manifest["Model_Role"].astype(str)) != expected_roles:
        raise ValueError("The Step-07 manifest must contain the two regression roles.")
    for row in manifest.itertuples(index=False):
        path = STEP07_ROOT / str(row.Output_File)
        require_files([path])
        if int(row.Rows) != EXPECTED_ROWS_PER_MODEL:
            raise ValueError(f"Unexpected spatial OOF rows for {row.Model_Role}.")
        if int(row.Unique_OOF_Keys) != EXPECTED_ROWS_PER_MODEL:
            raise ValueError(f"Unexpected spatial OOF keys for {row.Model_Role}.")
        if sha256_file(path) != str(row.SHA256):
            raise ValueError(f"SHA256 mismatch for spatial OOF file:\n{path}")
    return manifest.sort_values("Model_Role").reset_index(drop=True)


# =============================================================================
# Metadata and prediction normalization
# =============================================================================


def attach_locked_metadata(
    data: pd.DataFrame,
    blocks: pd.DataFrame,
    source_name: str,
) -> pd.DataFrame:
    result = data.copy()
    result["GRID_UID"] = result["GRID_UID"].astype(str)
    result["Country"] = result["Country"].astype(str)
    result["GRID_ID"] = result["GRID_ID"].astype(str)

    locked = blocks.rename(
        columns={
            "Country": "Locked_Country",
            "GRID_ID": "Locked_GRID_ID",
            "Spatial_Block": "Locked_Spatial_Block",
            **{column: f"Locked_{column}" for column in COORDINATE_COLUMNS},
        }
    )
    result = result.merge(locked, on="GRID_UID", how="left", validate="many_to_one")
    if result["Locked_Spatial_Block"].isna().any():
        raise ValueError(f"{source_name} contains a grid absent from Step 07.")
    if not (
        result["Country"].astype(str).to_numpy()
        == result["Locked_Country"].astype(str).to_numpy()
    ).all():
        raise ValueError(f"Country disagrees with Step 07 in {source_name}.")
    if not (
        result["GRID_ID"].astype(str).to_numpy()
        == result["Locked_GRID_ID"].astype(str).to_numpy()
    ).all():
        raise ValueError(f"GRID_ID disagrees with Step 07 in {source_name}.")

    if "Spatial_Block" in result.columns:
        observed = pd.to_numeric(result["Spatial_Block"], errors="coerce")
        present = observed.notna()
        if present.any() and not (
            observed.loc[present].astype(int).to_numpy()
            == result.loc[present, "Locked_Spatial_Block"].astype(int).to_numpy()
        ).all():
            raise ValueError(f"Spatial_Block disagrees in {source_name}.")

    for column in COORDINATE_COLUMNS:
        if column in result.columns:
            observed = pd.to_numeric(result[column], errors="coerce")
            present = observed.notna()
            if present.any():
                difference = maximum_absolute_difference(
                    observed.loc[present], result.loc[present, f"Locked_{column}"]
                )
                if difference > ABSOLUTE_METADATA_TOLERANCE:
                    raise ValueError(f"{column} disagrees in {source_name}.")
        result[column] = pd.to_numeric(result[f"Locked_{column}"], errors="raise")

    result["Spatial_Block"] = result["Locked_Spatial_Block"].astype(int)
    result["Country"] = result["Locked_Country"].astype(str)
    result["GRID_ID"] = result["Locked_GRID_ID"].astype(str)
    drop_columns = [
        "Locked_Country",
        "Locked_GRID_ID",
        "Locked_Spatial_Block",
        *[f"Locked_{column}" for column in COORDINATE_COLUMNS],
    ]
    return result.drop(columns=drop_columns)


def stable_regression_positive_probability(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    p0 = pd.to_numeric(data["Predicted_Zero_Probability"], errors="coerce").to_numpy(
        dtype=float
    )
    if not np.isfinite(p0).all() or np.any((p0 < 0.0) | (p0 > 1.0)):
        raise ValueError("Regression zero probabilities are invalid.")

    if "Predicted_Positive_Probability" in data.columns:
        p1_source = pd.to_numeric(
            data["Predicted_Positive_Probability"], errors="coerce"
        ).to_numpy(dtype=float)
        if not np.isfinite(p1_source).all() or np.any(
            (p1_source < 0.0) | (p1_source > 1.0)
        ):
            raise ValueError("Source positive probabilities are invalid.")
        if float(np.max(np.abs((p0 + p1_source) - 1.0))) > PROBABILITY_SUM_TOLERANCE:
            raise ValueError("Source zero and positive probabilities do not sum to one.")
        source = np.full(len(data), "Source_Stable_Positive_Probability", dtype=object)
        return p1_source, source

    p1 = np.clip(1.0 - p0, 0.0, 1.0)
    source = np.full(len(data), "One_Minus_Zero_Probability", dtype=object)

    # For an observed zero, Predictive_LogProbability is log P(Y=0), so
    # -expm1(log P0) recovers a small positive probability without cancellation.
    if "Predictive_LogProbability" in data.columns:
        observed = pd.to_numeric(data["Observed_Count"], errors="raise").to_numpy(
            dtype=float
        )
        log_probability = pd.to_numeric(
            data["Predictive_LogProbability"], errors="coerce"
        ).to_numpy(dtype=float)
        zero_rows = (observed == 0.0) & np.isfinite(log_probability)
        if zero_rows.any():
            stable = -np.expm1(log_probability[zero_rows])
            valid = np.isfinite(stable) & (stable >= 0.0) & (stable <= 1.0)
            selected_indices = np.flatnonzero(zero_rows)[valid]
            p1[selected_indices] = stable[valid]
            source[selected_indices] = "Recovered_From_Zero_Row_LogProbability"

    return p1, source


def conditional_from_expected(
    predicted_count: np.ndarray,
    exposure: np.ndarray,
    positive_probability: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    conditional_count = np.full(len(predicted_count), np.nan, dtype=float)
    valid = (
        np.isfinite(predicted_count)
        & np.isfinite(positive_probability)
        & (positive_probability > 0.0)
    )
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        conditional_count[valid] = predicted_count[valid] / positive_probability[valid]
    conditional_count[~np.isfinite(conditional_count)] = np.nan
    conditional_rate = conditional_count / exposure
    available = np.isfinite(conditional_count) & np.isfinite(conditional_rate)
    return conditional_count, conditional_rate, available


def validate_common_frame(data: pd.DataFrame, source_name: str) -> None:
    if len(data) != EXPECTED_ROWS_PER_MODEL:
        raise ValueError(
            f"Expected {EXPECTED_ROWS_PER_MODEL:,} rows in {source_name}, "
            f"found {len(data):,}."
        )
    if data.duplicated(KEY_COLUMNS).any():
        raise ValueError(f"Duplicate OOF keys occur in {source_name}.")
    if set(data["Year"].unique()) != EXPECTED_YEARS:
        raise ValueError(f"Unexpected year universe in {source_name}.")
    if set(data["Season"].unique()) != set(SEASONS):
        raise ValueError(f"Unexpected season universe in {source_name}.")
    if set(data["Count_Response"].unique()) != set(COUNT_RESPONSES):
        raise ValueError(f"Unexpected response universe in {source_name}.")
    counts = data.groupby(["Season", "Count_Response"]).size().to_dict()
    expected = {
        (season, response): EXPECTED_ROWS_PER_SEASON_RESPONSE
        for season in SEASONS
        for response in COUNT_RESPONSES
    }
    if counts != expected:
        raise ValueError(
            f"Season-response row counts differ in {source_name}.\nObserved: {counts}"
        )
    grid_counts = data.groupby(["Season", "Count_Response"])["GRID_UID"].nunique()
    if not (grid_counts == EXPECTED_GRID_COUNT).all():
        raise ValueError(f"A season-response combination lacks 4,982 grids in {source_name}.")


def normalize_regression_oof(
    raw: pd.DataFrame,
    blocks: pd.DataFrame,
    model_role: str,
    validation_scheme: str,
    source_name: str,
) -> Tuple[pd.DataFrame, Dict[str, object], Dict[str, object]]:
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
        "Predicted_Zero_Probability",
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"{source_name} is missing columns:\n" + "\n".join(missing))

    data = raw.copy()
    data["GRID_UID"] = data["GRID_UID"].astype(str)
    data["GRID_ID"] = data["GRID_ID"].astype(str)
    data["Country"] = data["Country"].astype(str)
    data["Count_Response"] = data["Count_Response"].astype(str)
    data["Year"] = pd.to_numeric(data["Year"], errors="raise").astype(int)
    data["Season"] = pd.to_numeric(data["Season"], errors="raise").astype(int)
    data["Temporal_Fold"] = pd.to_numeric(
        data["Temporal_Fold"], errors="raise"
    ).astype(int)

    data = attach_locked_metadata(data, blocks, source_name)

    if validation_scheme == "Temporal":
        if set(data["Temporal_Fold"].unique()) != EXPECTED_TEMPORAL_FOLDS:
            raise ValueError(f"Unexpected temporal folds in {source_name}.")
        if "Fold" in data.columns:
            legacy = pd.to_numeric(data["Fold"], errors="coerce")
            present = legacy.notna()
            if present.any() and not (
                np.rint(legacy.loc[present].to_numpy(dtype=float)).astype(int)
                == data.loc[present, "Temporal_Fold"].to_numpy(dtype=int)
            ).all():
                raise ValueError(f"Fold and Temporal_Fold disagree in {source_name}.")
        data["Validation_Fold"] = data["Temporal_Fold"].astype(int)
    elif validation_scheme == "Spatial":
        if "Validation_Fold" not in data.columns:
            raise ValueError(f"Validation_Fold is missing from {source_name}.")
        data["Validation_Fold"] = pd.to_numeric(
            data["Validation_Fold"], errors="raise"
        ).astype(int)
        if not (
            data["Validation_Fold"].to_numpy(dtype=int)
            == data["Spatial_Block"].to_numpy(dtype=int)
        ).all():
            raise ValueError(f"Validation_Fold and Spatial_Block disagree in {source_name}.")
    else:
        raise ValueError(f"Unsupported validation scheme: {validation_scheme}")

    for column in ["ForestPixelCount", "Observed_Count", "Predicted_Count"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if not np.isfinite(
        data[["ForestPixelCount", "Observed_Count", "Predicted_Count"]].to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError(f"Non-finite core prediction values occur in {source_name}.")
    if (data["ForestPixelCount"] <= 0).any():
        raise ValueError(f"Non-positive exposure occurs in {source_name}.")
    if (data["Observed_Count"] < 0).any() or (data["Predicted_Count"] < 0).any():
        raise ValueError(f"Negative counts occur in {source_name}.")

    exposure = data["ForestPixelCount"].to_numpy(dtype=float)
    observed_count = data["Observed_Count"].to_numpy(dtype=float)
    predicted_count = data["Predicted_Count"].to_numpy(dtype=float)
    data["Observed_Rate"] = observed_count / exposure
    data["Predicted_Rate"] = predicted_count / exposure
    data["Observed_Zero"] = (observed_count == 0.0).astype(int)

    positive_probability, probability_source = stable_regression_positive_probability(data)
    data["Predicted_Zero_Probability"] = pd.to_numeric(
        data["Predicted_Zero_Probability"], errors="raise"
    ).to_numpy(dtype=float)
    data["Predicted_Positive_Probability"] = positive_probability
    data["Positive_Probability_Source"] = probability_source

    if {
        "Predicted_Conditional_Positive_Count",
        "Predicted_Conditional_Positive_Rate",
    }.issubset(data.columns):
        conditional_count = pd.to_numeric(
            data["Predicted_Conditional_Positive_Count"], errors="coerce"
        ).to_numpy(dtype=float)
        conditional_rate = pd.to_numeric(
            data["Predicted_Conditional_Positive_Rate"], errors="coerce"
        ).to_numpy(dtype=float)
        conditional_source = np.full(
            len(data), "Source_Stable_Conditional_Positive_Mean", dtype=object
        )
    else:
        conditional_count, conditional_rate, available = conditional_from_expected(
            predicted_count, exposure, positive_probability
        )
        conditional_source = np.where(
            available,
            "Derived_Expected_Divided_By_Positive_Probability",
            "Unavailable_Zero_or_Nonfinite_Positive_Probability",
        ).astype(object)

    data["Predicted_Conditional_Positive_Count"] = conditional_count
    data["Predicted_Conditional_Positive_Rate"] = conditional_rate
    data["Conditional_Positive_Source"] = conditional_source

    if "Predictive_LogProbability" in data.columns:
        data["Predictive_LogProbability"] = pd.to_numeric(
            data["Predictive_LogProbability"], errors="coerce"
        )
    else:
        data["Predictive_LogProbability"] = np.nan

    data["Residual_Count"] = observed_count - predicted_count
    data["Residual_Rate"] = data["Observed_Rate"] - data["Predicted_Rate"]
    data["Season_Label"] = data["Season"].map(SEASONS)
    data["Rate_Scale_Name"] = data["Count_Response"].map(RATE_NAMES)
    data["Validation_Scheme"] = validation_scheme
    data["Model_Role"] = model_role

    if model_role == "Baseline_NB1":
        data["Model_Structure"] = BASELINE_MODEL_STRUCTURE
        data["Candidate_ID"] = BASELINE_CANDIDATE_ID
    elif model_role == "Final_Regression":
        if "Model_Structure" not in data.columns or "Candidate_ID" not in data.columns:
            raise ValueError(f"Final regression metadata is missing in {source_name}.")
        data["Model_Structure"] = data["Model_Structure"].astype(str)
        data["Candidate_ID"] = data["Candidate_ID"].astype(str)
    else:
        raise ValueError(f"Unexpected regression model role: {model_role}")

    validate_common_frame(data, source_name)

    core_columns = SHARED_COLUMNS + ["Model_Role", *MODEL_VALUE_COLUMNS]
    data = data[core_columns].sort_values(KEY_COLUMNS).reset_index(drop=True)

    finite_conditional = np.isfinite(
        data["Predicted_Conditional_Positive_Rate"].to_numpy(dtype=float)
    )
    identity_rows = finite_conditional & (
        data["Predicted_Positive_Probability"].to_numpy(dtype=float) > 0.0
    )
    identity_relative_error = np.nan
    if identity_rows.any():
        reconstructed = (
            data.loc[identity_rows, "Predicted_Conditional_Positive_Count"].to_numpy(
                dtype=float
            )
            * data.loc[identity_rows, "Predicted_Positive_Probability"].to_numpy(
                dtype=float
            )
        )
        target = data.loc[identity_rows, "Predicted_Count"].to_numpy(dtype=float)
        denominator = np.maximum(np.abs(target), np.finfo(float).tiny)
        identity_relative_error = float(
            np.max(np.abs(reconstructed - target) / denominator)
        )
        if identity_relative_error > CONDITIONAL_IDENTITY_RELATIVE_TOLERANCE:
            raise ValueError(
                f"Conditional-positive identity failed in {source_name}: "
                f"{identity_relative_error:.3e}."
            )

    normalization_qa = {
        "Validation_Scheme": validation_scheme,
        "Model_Role": model_role,
        "Source_Name": source_name,
        "Rows": len(data),
        "Unique_OOF_Keys": len(data),
        "GRID_UID_Count": data["GRID_UID"].nunique(),
        "Positive_Observed_Rows": int((data["Observed_Count"] > 0).sum()),
        "Zero_Observed_Rows": int((data["Observed_Count"] == 0).sum()),
        "Predicted_Positive_Probability_Zero_Rows": int(
            (data["Predicted_Positive_Probability"] == 0.0).sum()
        ),
        "Finite_Conditional_Positive_Rows": int(finite_conditional.sum()),
        "Missing_Conditional_Positive_Rows": int((~finite_conditional).sum()),
        "Maximum_Conditional_Identity_Relative_Error": identity_relative_error,
        "OOF_Key_Signature": frame_signature(data, KEY_COLUMNS),
        "Observed_Panel_Signature": frame_signature(
            data,
            [
                "GRID_UID",
                "Year",
                "Season",
                "Count_Response",
                "Observed_Count",
                "ForestPixelCount",
            ],
        ),
    }
    probability_qa = probability_summary(data, model_role, validation_scheme)
    return data, normalization_qa, probability_qa


def normalize_rf_oof(
    raw: pd.DataFrame,
    blocks: pd.DataFrame,
    reference_panel: pd.DataFrame,
    validation_scheme: str,
    source_name: str,
) -> Tuple[pd.DataFrame, Dict[str, object], Dict[str, object]]:
    required = {
        "GRID_UID",
        "Country",
        "GRID_ID",
        "Year",
        "Season",
        "Temporal_Fold",
        "Spatial_Block",
        "Validation_Scheme",
        "Validation_Fold",
        "Response",
        "Observed",
        "Occurrence_Probability",
        "Conditional_Severity",
        "Expected_Prediction",
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"{source_name} is missing columns:\n" + "\n".join(missing))

    data = raw.copy()
    data["GRID_UID"] = data["GRID_UID"].astype(str)
    data["Country"] = data["Country"].astype(str)
    data["GRID_ID"] = data["GRID_ID"].astype(str)
    data["Year"] = pd.to_numeric(data["Year"], errors="raise").astype(int)
    data["Season"] = pd.to_numeric(data["Season"], errors="raise").astype(int)
    data["Temporal_Fold"] = pd.to_numeric(
        data["Temporal_Fold"], errors="raise"
    ).astype(int)
    data["Spatial_Block"] = pd.to_numeric(
        data["Spatial_Block"], errors="raise"
    ).astype(int)
    data["Validation_Fold"] = pd.to_numeric(
        data["Validation_Fold"], errors="raise"
    ).astype(int)
    data["Response"] = data["Response"].astype(str)
    if not set(data["Response"].unique()).issubset(set(RESPONSE_FROM_RATE)):
        raise ValueError(f"Unexpected RF response names occur in {source_name}.")
    data["Count_Response"] = data["Response"].map(RESPONSE_FROM_RATE)
    data["Rate_Scale_Name"] = data["Response"]

    source_schemes = set(data["Validation_Scheme"].astype(str).unique())
    if source_schemes != {validation_scheme}:
        raise ValueError(
            f"RF validation scheme mismatch in {source_name}: {source_schemes}."
        )
    if validation_scheme == "Temporal":
        if not (
            data["Validation_Fold"].to_numpy(dtype=int)
            == data["Temporal_Fold"].to_numpy(dtype=int)
        ).all():
            raise ValueError(f"RF temporal folds disagree in {source_name}.")
    elif validation_scheme == "Spatial":
        if not (
            data["Validation_Fold"].to_numpy(dtype=int)
            == data["Spatial_Block"].to_numpy(dtype=int)
        ).all():
            raise ValueError(f"RF spatial folds disagree in {source_name}.")
    else:
        raise ValueError(f"Unsupported validation scheme: {validation_scheme}")

    data = attach_locked_metadata(data, blocks, source_name)

    reference = reference_panel[
        KEY_COLUMNS
        + [
            "ForestPixelCount",
            "Observed_Count",
            "Observed_Rate",
            "Observed_Zero",
        ]
    ].copy()
    data = data.merge(reference, on=KEY_COLUMNS, how="inner", validate="one_to_one")
    if len(data) != EXPECTED_ROWS_PER_MODEL:
        raise ValueError(
            f"RF OOF did not match the exact regression panel in {source_name}."
        )

    for column in ["Observed", "Occurrence_Probability", "Conditional_Severity", "Expected_Prediction"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if not np.isfinite(
        data[["Observed", "Occurrence_Probability", "Conditional_Severity", "Expected_Prediction"]].to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError(f"Non-finite RF prediction values occur in {source_name}.")
    if np.any(
        (data["Occurrence_Probability"].to_numpy(dtype=float) < 0.0)
        | (data["Occurrence_Probability"].to_numpy(dtype=float) > 1.0)
    ):
        raise ValueError(f"RF occurrence probabilities fall outside [0,1] in {source_name}.")
    if (data["Conditional_Severity"] < 0).any() or (data["Expected_Prediction"] < 0).any():
        raise ValueError(f"Negative RF predictions occur in {source_name}.")

    observed_difference = maximum_absolute_difference(data["Observed"], data["Observed_Rate"])
    if observed_difference > ABSOLUTE_RATE_TOLERANCE:
        raise ValueError(
            f"RF and regression observed rates disagree in {source_name}: "
            f"{observed_difference:.3e}."
        )

    reconstructed_expected = (
        data["Occurrence_Probability"].to_numpy(dtype=float)
        * data["Conditional_Severity"].to_numpy(dtype=float)
    )
    expected = data["Expected_Prediction"].to_numpy(dtype=float)
    max_expected_difference = float(np.max(np.abs(reconstructed_expected - expected)))
    if max_expected_difference > 5e-10:
        raise ValueError(
            f"RF expected prediction is not occurrence × severity in {source_name}: "
            f"{max_expected_difference:.3e}."
        )

    exposure = data["ForestPixelCount"].to_numpy(dtype=float)
    data["Predicted_Rate"] = expected
    data["Predicted_Count"] = expected * exposure
    data["Predicted_Positive_Probability"] = data["Occurrence_Probability"]
    data["Predicted_Zero_Probability"] = 1.0 - data["Occurrence_Probability"]
    data["Predicted_Conditional_Positive_Rate"] = data["Conditional_Severity"]
    data["Predicted_Conditional_Positive_Count"] = (
        data["Conditional_Severity"].to_numpy(dtype=float) * exposure
    )
    data["Predictive_LogProbability"] = np.nan
    data["Residual_Count"] = (
        data["Observed_Count"].to_numpy(dtype=float)
        - data["Predicted_Count"].to_numpy(dtype=float)
    )
    data["Residual_Rate"] = (
        data["Observed_Rate"].to_numpy(dtype=float)
        - data["Predicted_Rate"].to_numpy(dtype=float)
    )
    data["Positive_Probability_Source"] = "RF_Direct_Occurrence_Probability"
    data["Conditional_Positive_Source"] = "RF_Direct_Conditional_Severity"
    data["Season_Label"] = data["Season"].map(SEASONS)
    data["Validation_Scheme"] = validation_scheme
    data["Model_Role"] = "Final_Hurdle_RF"
    data["Model_Structure"] = RF_MODEL_STRUCTURE
    data["Candidate_ID"] = RF_CANDIDATE_ID

    validate_common_frame(data, source_name)
    core_columns = SHARED_COLUMNS + ["Model_Role", *MODEL_VALUE_COLUMNS]
    data = data[core_columns].sort_values(KEY_COLUMNS).reset_index(drop=True)

    normalization_qa = {
        "Validation_Scheme": validation_scheme,
        "Model_Role": "Final_Hurdle_RF",
        "Source_Name": source_name,
        "Rows": len(data),
        "Unique_OOF_Keys": len(data),
        "GRID_UID_Count": data["GRID_UID"].nunique(),
        "Positive_Observed_Rows": int((data["Observed_Count"] > 0).sum()),
        "Zero_Observed_Rows": int((data["Observed_Count"] == 0).sum()),
        "Predicted_Positive_Probability_Zero_Rows": int(
            (data["Predicted_Positive_Probability"] == 0.0).sum()
        ),
        "Finite_Conditional_Positive_Rows": int(
            np.isfinite(data["Predicted_Conditional_Positive_Rate"]).sum()
        ),
        "Missing_Conditional_Positive_Rows": int(
            (~np.isfinite(data["Predicted_Conditional_Positive_Rate"])).sum()
        ),
        "RF_Maximum_Observed_Rate_Difference_vs_Regression": observed_difference,
        "RF_Maximum_Expected_Product_Identity_Absolute_Error": max_expected_difference,
        "OOF_Key_Signature": frame_signature(data, KEY_COLUMNS),
        "Observed_Panel_Signature": frame_signature(
            data,
            [
                "GRID_UID",
                "Year",
                "Season",
                "Count_Response",
                "Observed_Count",
                "ForestPixelCount",
            ],
        ),
    }
    probability_qa = probability_summary(data, "Final_Hurdle_RF", validation_scheme)
    return data, normalization_qa, probability_qa


def probability_summary(
    data: pd.DataFrame,
    model_role: str,
    validation_scheme: str,
) -> Dict[str, object]:
    p0 = data["Predicted_Zero_Probability"].to_numpy(dtype=float)
    p1 = data["Predicted_Positive_Probability"].to_numpy(dtype=float)
    conditional_rate = data["Predicted_Conditional_Positive_Rate"].to_numpy(
        dtype=float
    )
    conditional_count = data["Predicted_Conditional_Positive_Count"].to_numpy(
        dtype=float
    )
    probability_error = float(np.max(np.abs((p0 + p1) - 1.0)))
    if probability_error > PROBABILITY_SUM_TOLERANCE:
        raise ValueError(
            f"Probability complement identity failed for {model_role} {validation_scheme}."
        )
    finite_conditional = np.isfinite(conditional_rate) & np.isfinite(conditional_count)
    return {
        "Validation_Scheme": validation_scheme,
        "Model_Role": model_role,
        "Rows": len(data),
        "Minimum_Zero_Probability": float(np.min(p0)),
        "Maximum_Zero_Probability": float(np.max(p0)),
        "Minimum_Positive_Probability": float(np.min(p1)),
        "Maximum_Positive_Probability": float(np.max(p1)),
        "Maximum_Probability_Complement_Error": probability_error,
        "Finite_Conditional_Positive_Rows": int(finite_conditional.sum()),
        "Missing_Conditional_Positive_Rows": int((~finite_conditional).sum()),
        "Maximum_Finite_Conditional_Positive_Count": (
            float(np.max(conditional_count[finite_conditional]))
            if finite_conditional.any()
            else np.nan
        ),
        "Maximum_Finite_Conditional_Positive_Rate": (
            float(np.max(conditional_rate[finite_conditional]))
            if finite_conditional.any()
            else np.nan
        ),
        "Positive_Probability_Source_Levels": ";".join(
            sorted(data["Positive_Probability_Source"].astype(str).unique())
        ),
        "Conditional_Positive_Source_Levels": ";".join(
            sorted(data["Conditional_Positive_Source"].astype(str).unique())
        ),
    }


# =============================================================================
# Input reading
# =============================================================================


def read_temporal_regression(
    blocks: pd.DataFrame,
    manifest: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    baseline_parts: List[pd.DataFrame] = []
    final_parts: List[pd.DataFrame] = []
    source_rows: List[Dict[str, object]] = []

    for row in manifest.itertuples(index=False):
        season = int(row.Season_Code)
        response = str(row.Count_Response)
        baseline_path = step02_temporal_path(season, response)
        final_path = final_temporal_path(str(row.Output_File))
        require_files([baseline_path, final_path])

        log(f"Reading temporal regression OOF: {SEASONS[season]} {response}.")
        final_raw = read_csv_gzip(final_path)
        if len(final_raw) != EXPECTED_ROWS_PER_SEASON_RESPONSE:
            raise ValueError(f"Unexpected row count in {final_path}.")
        final_raw["GRID_UID"] = final_raw["GRID_UID"].astype(str)
        final_raw["Year"] = pd.to_numeric(final_raw["Year"], errors="raise").astype(int)
        final_raw["Season"] = pd.to_numeric(final_raw["Season"], errors="raise").astype(int)
        final_raw["Count_Response"] = final_raw["Count_Response"].astype(str)

        baseline_all = read_csv_gzip(baseline_path)
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
            "Predicted_Zero_Probability",
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
        baseline_selected["Count_Response"] = baseline_selected["Count_Response"].astype(str)
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
                f"Baseline NB1 did not match the exact final sample for "
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
            raise ValueError("Baseline and final Country values disagree.")
        for observed_column, reference_column in [
            ("ForestPixelCount", "Reference_ForestPixelCount"),
            ("Observed_Count", "Reference_Observed_Count"),
        ]:
            if not np.allclose(
                pd.to_numeric(
                    baseline_selected[observed_column], errors="raise"
                ).to_numpy(dtype=float),
                pd.to_numeric(
                    baseline_selected[reference_column], errors="raise"
                ).to_numpy(dtype=float),
                rtol=0.0,
                atol=0.0,
            ):
                raise ValueError(f"Baseline and final {observed_column} values disagree.")
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
                source_manifest_row(
                    baseline_path,
                    "Temporal",
                    "Baseline_NB1",
                    raw_rows=len(baseline_all),
                    selected_rows=len(baseline_selected),
                    source_filter="Family == NB1 and exact Step-06d final OOF keys",
                ),
                source_manifest_row(
                    final_path,
                    "Temporal",
                    "Final_Regression",
                    raw_rows=len(final_raw),
                    selected_rows=len(final_raw),
                    source_filter="Step-06d final selected temporal OOF",
                ),
            ]
        )

    baseline_raw = pd.concat(baseline_parts, ignore_index=True)
    final_raw = pd.concat(final_parts, ignore_index=True)
    baseline, baseline_qa, baseline_probability = normalize_regression_oof(
        baseline_raw,
        blocks,
        "Baseline_NB1",
        "Temporal",
        "Step-02 NB1 exact final temporal sample",
    )
    final, final_qa, final_probability = normalize_regression_oof(
        final_raw,
        blocks,
        "Final_Regression",
        "Temporal",
        "Step-06d final selected temporal OOF",
    )
    return (
        baseline,
        final,
        source_rows,
        [baseline_qa, final_qa],
        [baseline_probability, final_probability],
    )


def read_spatial_regression(
    blocks: pd.DataFrame,
    manifest: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    frames: Dict[str, pd.DataFrame] = {}
    source_rows: List[Dict[str, object]] = []
    qa_rows: List[Dict[str, object]] = []
    probability_rows: List[Dict[str, object]] = []

    for model_role in ["Baseline_NB1", "Final_Regression"]:
        selected = manifest.loc[manifest["Model_Role"].astype(str) == model_role]
        if len(selected) != 1:
            raise ValueError(f"Step-07 manifest did not uniquely identify {model_role}.")
        path = STEP07_ROOT / str(selected.iloc[0]["Output_File"])
        log(f"Reading spatial regression OOF: {model_role}.")
        raw = read_csv_gzip(path)
        frame, qa, probability = normalize_regression_oof(
            raw,
            blocks,
            model_role,
            "Spatial",
            path.name,
        )
        frames[model_role] = frame
        qa_rows.append(qa)
        probability_rows.append(probability)
        source_rows.append(
            source_manifest_row(
                path,
                "Spatial",
                model_role,
                raw_rows=len(raw),
                selected_rows=len(frame),
                source_filter="Locked Step-07 release spatial OOF",
            )
        )

    return (
        frames["Baseline_NB1"],
        frames["Final_Regression"],
        source_rows,
        qa_rows,
        probability_rows,
    )


def source_manifest_row(
    path: Path,
    validation_scheme: str,
    model_role: str,
    raw_rows: int,
    selected_rows: int,
    source_filter: str,
) -> Dict[str, object]:
    return {
        "Validation_Scheme": validation_scheme,
        "Model_Role": model_role,
        "Input_File": str(path),
        "Input_File_SHA256": sha256_file(path),
        "Input_File_Size_Bytes": path.stat().st_size,
        "Raw_File_Rows": int(raw_rows),
        "Selected_Aligned_Rows": int(selected_rows),
        "Source_Filter": source_filter,
    }


# =============================================================================
# Pairwise alignment and output assembly
# =============================================================================


def verify_model_alignment(
    frames: Mapping[str, pd.DataFrame],
    validation_scheme: str,
) -> List[Dict[str, object]]:
    reference = frames["Final_Regression"].sort_values(KEY_COLUMNS).reset_index(drop=True)
    rows: List[Dict[str, object]] = []

    for model_role, frame in frames.items():
        candidate = frame.sort_values(KEY_COLUMNS).reset_index(drop=True)
        key_match = reference[KEY_COLUMNS].equals(candidate[KEY_COLUMNS])
        if not key_match:
            raise ValueError(
                f"OOF keys do not align for {model_role} {validation_scheme}."
            )
        metadata_mismatch = 0
        for column in [
            "Country",
            "GRID_ID",
            "Season_Label",
            "Temporal_Fold",
            "Spatial_Block",
            "Validation_Fold",
            "Rate_Scale_Name",
            "Observed_Zero",
        ]:
            if not reference[column].astype(str).equals(candidate[column].astype(str)):
                metadata_mismatch += int(
                    (reference[column].astype(str) != candidate[column].astype(str)).sum()
                )
        numeric_max_differences: Dict[str, float] = {}
        for column in [
            *COORDINATE_COLUMNS,
            "ForestPixelCount",
            "Observed_Count",
            "Observed_Rate",
        ]:
            difference = maximum_absolute_difference(reference[column], candidate[column])
            numeric_max_differences[column] = difference
            tolerance = (
                ABSOLUTE_METADATA_TOLERANCE
                if column in COORDINATE_COLUMNS
                else ABSOLUTE_RATE_TOLERANCE
            )
            if difference > tolerance:
                raise ValueError(
                    f"Shared field {column} differs for {model_role} "
                    f"{validation_scheme}: {difference:.3e}."
                )
        if metadata_mismatch:
            raise ValueError(
                f"Shared categorical metadata differs for {model_role} "
                f"{validation_scheme}: {metadata_mismatch} cells."
            )
        rows.append(
            {
                "Validation_Scheme": validation_scheme,
                "Reference_Model": "Final_Regression",
                "Compared_Model": model_role,
                "Rows": len(candidate),
                "OOF_Key_Match": key_match,
                "Categorical_Metadata_Mismatch_Count": metadata_mismatch,
                **{
                    f"Maximum_Absolute_Difference_{column}": value
                    for column, value in numeric_max_differences.items()
                },
                "Observed_Panel_Signature": frame_signature(
                    candidate,
                    [
                        "GRID_UID",
                        "Year",
                        "Season",
                        "Count_Response",
                        "Observed_Count",
                        "ForestPixelCount",
                    ],
                ),
            }
        )
    return rows


def prefixed_model_frame(frame: pd.DataFrame, model_role: str) -> pd.DataFrame:
    selected = frame[KEY_COLUMNS + MODEL_VALUE_COLUMNS].copy()
    rename = {
        column: f"{model_role}_{column}"
        for column in MODEL_VALUE_COLUMNS
    }
    return selected.rename(columns=rename)


def build_wide_aligned(
    frames: Mapping[str, pd.DataFrame],
    validation_scheme: str,
) -> pd.DataFrame:
    reference = (
        frames["Final_Regression"]
        .sort_values(KEY_COLUMNS)
        .reset_index(drop=True)
    )
    wide = reference[SHARED_COLUMNS].copy()
    for model_role in MODEL_ROLES:
        model_values = prefixed_model_frame(frames[model_role], model_role)
        wide = wide.merge(
            model_values,
            on=KEY_COLUMNS,
            how="inner",
            validate="one_to_one",
        )
    if len(wide) != EXPECTED_ROWS_PER_MODEL:
        raise ValueError(
            f"Expected {EXPECTED_ROWS_PER_MODEL:,} aligned rows for "
            f"{validation_scheme}, found {len(wide):,}."
        )
    if wide.duplicated(KEY_COLUMNS).any():
        raise ValueError(f"Duplicate aligned keys occur for {validation_scheme}.")
    return wide.sort_values(KEY_COLUMNS).reset_index(drop=True)


def write_gzip_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, compression="gzip")


def aligned_manifest_row(
    frame: pd.DataFrame,
    path: Path,
    validation_scheme: str,
) -> Dict[str, object]:
    return {
        "Validation_Scheme": validation_scheme,
        "Output_File": path.name,
        "Rows": len(frame),
        "Unique_OOF_Keys": frame[KEY_COLUMNS].drop_duplicates().shape[0],
        "GRID_UID_Count": frame["GRID_UID"].nunique(),
        "Season_Response_Combinations": frame[
            ["Season", "Count_Response"]
        ].drop_duplicates().shape[0],
        "Model_Count": len(MODEL_ROLES),
        "Models": ";".join(MODEL_ROLES),
        "OOF_Key_Signature": frame_signature(frame, KEY_COLUMNS),
        "Observed_Panel_Signature": frame_signature(
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
        "SHA256": sha256_file(path),
        "File_Size_Bytes": path.stat().st_size,
    }


# =============================================================================
# Column definitions and metadata
# =============================================================================


def column_definitions() -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    shared_definitions = {
        "GRID_UID": "Unique country-qualified grid identifier.",
        "Country": "Country label locked to the Step-07 shared grid assignment.",
        "GRID_ID": "Original grid identifier within country.",
        "Year": "Observation year, 2001-2025.",
        "Season": "Season code: 1 Spring, 2 Summer, 3 Autumn.",
        "Season_Label": "English season label.",
        "Temporal_Fold": "Locked five-fold temporal validation assignment.",
        "Spatial_Block": "Locked five-block spatial validation assignment generated in regression Step 07.",
        "Validation_Scheme": "Temporal or Spatial OOF validation.",
        "Validation_Fold": "Held-out temporal fold or held-out spatial block for this OOF prediction.",
        "Centroid_X_UTM52_m": "Grid centroid easting in UTM zone 52N, metres.",
        "Centroid_Y_UTM52_m": "Grid centroid northing in UTM zone 52N, metres.",
        "Longitude": "Grid centroid longitude.",
        "Latitude": "Grid centroid latitude.",
        "Count_Response": "Fire_Count or Burned_Pixel_Count.",
        "Rate_Scale_Name": "FCD or BAD.",
        "ForestPixelCount": "Forest exposure in 500 m forest-pixel units.",
        "Observed_Count": "Observed count response.",
        "Observed_Rate": "Observed count divided by ForestPixelCount.",
        "Observed_Zero": "1 when Observed_Count equals zero, otherwise 0.",
    }
    for column, definition in shared_definitions.items():
        rows.append(
            {
                "Column": column,
                "Scope": "Shared",
                "Definition": definition,
                "Directly_Comparable_Across_All_Models": True,
            }
        )

    model_definitions = {
        "Model_Structure": "Locked model structure label.",
        "Candidate_ID": "Locked model/candidate identifier.",
        "Predicted_Count": "OOF unconditional expected count.",
        "Predicted_Rate": "OOF unconditional expected rate.",
        "Predicted_Zero_Probability": "OOF probability of a zero response.",
        "Predicted_Positive_Probability": "OOF probability of a positive response.",
        "Predicted_Conditional_Positive_Count": "OOF expected count conditional on a positive response.",
        "Predicted_Conditional_Positive_Rate": "OOF expected rate conditional on a positive response.",
        "Predictive_LogProbability": "Log probability assigned to the observed count; unavailable for RF and not comparable between regression and RF.",
        "Residual_Count": "Observed_Count minus Predicted_Count.",
        "Residual_Rate": "Observed_Rate minus Predicted_Rate.",
        "Positive_Probability_Source": "How the positive probability was obtained or stabilized.",
        "Conditional_Positive_Source": "How the conditional-positive prediction was obtained.",
    }
    for model_role in MODEL_ROLES:
        for suffix, definition in model_definitions.items():
            comparable = suffix != "Predictive_LogProbability"
            rows.append(
                {
                    "Column": f"{model_role}_{suffix}",
                    "Scope": model_role,
                    "Definition": definition,
                    "Directly_Comparable_Across_All_Models": comparable,
                }
            )
    return pd.DataFrame(rows)


# =============================================================================
# Main workflow
# =============================================================================


def main() -> int:
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text("", encoding="utf-8")

    log("Starting Step 01 aligned model OOF construction.")
    log(f"Code version: {STEP01_CODE_VERSION}.")
    log("No model is fitted, tuned, selected, or ranked in this step.")

    method_versions = verify_upstream_methods()
    blocks, block_metadata = read_and_verify_blocks()
    temporal_manifest = read_final_temporal_manifest()
    spatial_manifest = read_spatial_manifest()
    log("Verified the locked 4,982-grid assignment and upstream OOF manifests.")

    (
        temporal_baseline,
        temporal_final,
        temporal_source_rows,
        temporal_normalization_rows,
        temporal_probability_rows,
    ) = read_temporal_regression(blocks, temporal_manifest)

    (
        spatial_baseline,
        spatial_final,
        spatial_source_rows,
        spatial_normalization_rows,
        spatial_probability_rows,
    ) = read_spatial_regression(blocks, spatial_manifest)

    log("Reading final Hurdle RF temporal OOF predictions.")
    rf_temporal_raw = read_csv_gzip(RF_TEMPORAL_OOF_FILE)
    rf_temporal, rf_temporal_qa, rf_temporal_probability = normalize_rf_oof(
        rf_temporal_raw,
        blocks,
        temporal_final,
        "Temporal",
        RF_TEMPORAL_OOF_FILE.name,
    )
    log("Reading final Hurdle RF spatial OOF predictions.")
    rf_spatial_raw = read_csv_gzip(RF_SPATIAL_OOF_FILE)
    rf_spatial, rf_spatial_qa, rf_spatial_probability = normalize_rf_oof(
        rf_spatial_raw,
        blocks,
        spatial_final,
        "Spatial",
        RF_SPATIAL_OOF_FILE.name,
    )

    source_rows = [
        *temporal_source_rows,
        *spatial_source_rows,
        source_manifest_row(
            RF_TEMPORAL_OOF_FILE,
            "Temporal",
            "Final_Hurdle_RF",
            raw_rows=len(rf_temporal_raw),
            selected_rows=len(rf_temporal),
            source_filter="Locked final RF temporal OOF",
        ),
        source_manifest_row(
            RF_SPATIAL_OOF_FILE,
            "Spatial",
            "Final_Hurdle_RF",
            raw_rows=len(rf_spatial_raw),
            selected_rows=len(rf_spatial),
            source_filter="Locked final RF spatial OOF",
        ),
    ]

    temporal_frames = {
        "Baseline_NB1": temporal_baseline,
        "Final_Regression": temporal_final,
        "Final_Hurdle_RF": rf_temporal,
    }
    spatial_frames = {
        "Baseline_NB1": spatial_baseline,
        "Final_Regression": spatial_final,
        "Final_Hurdle_RF": rf_spatial,
    }

    alignment_rows = [
        *verify_model_alignment(temporal_frames, "Temporal"),
        *verify_model_alignment(spatial_frames, "Spatial"),
    ]

    temporal_wide = build_wide_aligned(temporal_frames, "Temporal")
    spatial_wide = build_wide_aligned(spatial_frames, "Spatial")

    # The observed panels must be identical across validation schemes; only the
    # held-out fold and predictions differ.
    temporal_panel_signature = frame_signature(
        temporal_wide,
        [
            "GRID_UID",
            "Year",
            "Season",
            "Count_Response",
            "Observed_Count",
            "ForestPixelCount",
        ],
    )
    spatial_panel_signature = frame_signature(
        spatial_wide,
        [
            "GRID_UID",
            "Year",
            "Season",
            "Count_Response",
            "Observed_Count",
            "ForestPixelCount",
        ],
    )
    if temporal_panel_signature != spatial_panel_signature:
        raise ValueError("Temporal and spatial OOF observed panels differ.")
    alignment_rows.append(
        {
            "Validation_Scheme": "Temporal_vs_Spatial",
            "Reference_Model": "All_Models",
            "Compared_Model": "All_Models",
            "Rows": len(temporal_wide),
            "OOF_Key_Match": temporal_wide[KEY_COLUMNS].equals(
                spatial_wide[KEY_COLUMNS]
            ),
            "Categorical_Metadata_Mismatch_Count": 0,
            "Observed_Panel_Signature": temporal_panel_signature,
            "Spatial_Observed_Panel_Signature": spatial_panel_signature,
            "Observed_Panels_Identical": True,
        }
    )

    write_gzip_csv(temporal_wide, TEMPORAL_OUTPUT_FILE)
    write_gzip_csv(spatial_wide, SPATIAL_OUTPUT_FILE)
    log("Wrote paired temporal and spatial aligned OOF datasets.")

    source_manifest = pd.DataFrame(source_rows).sort_values(
        ["Validation_Scheme", "Model_Role", "Input_File"]
    )
    source_manifest.to_csv(SOURCE_MANIFEST_FILE, index=False, encoding="utf-8-sig")

    normalization_qa = pd.DataFrame(
        [
            *temporal_normalization_rows,
            rf_temporal_qa,
            *spatial_normalization_rows,
            rf_spatial_qa,
        ]
    ).sort_values(["Validation_Scheme", "Model_Role"])
    normalization_qa.to_csv(
        NORMALIZATION_QA_FILE, index=False, encoding="utf-8-sig"
    )

    alignment_qa = pd.DataFrame(alignment_rows)
    alignment_qa.to_csv(ALIGNMENT_QA_FILE, index=False, encoding="utf-8-sig")

    probability_qa = pd.DataFrame(
        [
            *temporal_probability_rows,
            rf_temporal_probability,
            *spatial_probability_rows,
            rf_spatial_probability,
        ]
    ).sort_values(["Validation_Scheme", "Model_Role"])
    probability_qa.to_csv(
        PROBABILITY_QA_FILE, index=False, encoding="utf-8-sig"
    )

    output_manifest = pd.DataFrame(
        [
            aligned_manifest_row(temporal_wide, TEMPORAL_OUTPUT_FILE, "Temporal"),
            aligned_manifest_row(spatial_wide, SPATIAL_OUTPUT_FILE, "Spatial"),
        ]
    )
    output_manifest["Step01_Code_Version"] = STEP01_CODE_VERSION
    output_manifest["Spatial_Mapping_SHA256"] = EXPECTED_SPATIAL_MAPPING_SHA256
    output_manifest["Spatial_Block_File_SHA256"] = (
        EXPECTED_SPATIAL_BLOCK_FILE_SHA256
    )
    output_manifest.to_csv(
        OUTPUT_MANIFEST_FILE, index=False, encoding="utf-8-sig"
    )

    column_definitions().to_csv(
        COLUMN_DEFINITION_FILE, index=False, encoding="utf-8-sig"
    )

    method = {
        "Step": "01_Build_Aligned_Model_OOF",
        "Code_Version": STEP01_CODE_VERSION,
        "Created": datetime.now().isoformat(timespec="seconds"),
        "Purpose": (
            "Create strict paired wide-format temporal and spatial OOF datasets "
            "for Baseline NB1, Final Regression, and Final Hurdle RF."
        ),
        "No_Model_Fitting": True,
        "No_Model_Selection": True,
        "No_Performance_Ranking": True,
        "Models": MODEL_ROLES,
        "Validation_Schemes": VALIDATION_SCHEMES,
        "Rows_Per_Validation_Scheme": EXPECTED_ROWS_PER_MODEL,
        "Rows_Per_Season_Response": EXPECTED_ROWS_PER_SEASON_RESPONSE,
        "Shared_GRID_UID_Count": EXPECTED_GRID_COUNT,
        "Shared_Key": KEY_COLUMNS,
        "Primary_Output_Format": (
            "One row per shared GRID_UID-Year-Season-response key, with all "
            "three model predictions stored in model-prefixed columns."
        ),
        "Baseline_Temporal_Source": (
            "Step-02 Family == NB1 restricted to exact Step-06d final OOF keys"
        ),
        "Final_Regression_Temporal_Source": "Step-06d final selected OOF files",
        "Regression_Spatial_Source": "Step-07 release aligned spatial OOF files",
        "RF_Source": "Step-06 final Hurdle RF temporal and spatial OOF files",
        "Spatial_Assignment_Source": "08_Regression_modeling Step 07",
        "Spatial_Assignment": block_metadata,
        "Upstream_Method_Versions": method_versions,
        "Probability_Normalization": {
            "Regression_Spatial": (
                "Use Step-07 release stable positive probabilities and stable "
                "conditional-positive means directly."
            ),
            "Regression_Temporal": (
                "Use one minus zero probability, with zero-observation rows "
                "stabilized by -expm1(Predictive_LogProbability), because that "
                "log probability equals log P(Y=0) on zero rows."
            ),
            "RF": (
                "Use direct occurrence probability and direct conditional "
                "severity from the final two-stage Hurdle RF."
            ),
        },
        "Comparability_Limits": [
            (
                "Predictive_LogProbability is retained for regression models "
                "but is unavailable for RF and must not be used for three-model comparison."
            ),
            (
                "Temporal and spatial OOF represent different generalization tasks "
                "and must remain separate in all downstream analyses."
            ),
            (
                "This step aligns predictions only; it does not calculate or rank "
                "model performance."
            ),
        ],
        "Output_Files": {
            "Temporal": str(TEMPORAL_OUTPUT_FILE),
            "Spatial": str(SPATIAL_OUTPUT_FILE),
        },
    }
    METHOD_FILE.write_text(
        json.dumps(method, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    software = {
        "Created": datetime.now().isoformat(timespec="seconds"),
        "Python": sys.version,
        "Platform": platform.platform(),
        "NumPy": np.__version__,
        "pandas": pd.__version__,
    }
    SOFTWARE_FILE.write_text(
        json.dumps(software, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    log(
        "Step 01 completed: 742,374 temporal and 742,374 spatial OOF rows "
        "were strictly aligned across Baseline NB1, Final Regression, and "
        "Final Hurdle RF."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
