# -*- coding: utf-8 -*-
"""
Step 07: generate the shared five-fold spatial-block assignment and validate
both the baseline and final fixed-effect regression models.

Recommended public location
---------------------------
<CODE_ROOT>/08_Regression_modeling/00_Script/
    07_run_regression_spatial_block_validation.py

Inputs
------
<CODE_ROOT>/00_Input_data/Base_Table_2001_2025.csv.bz2

<CODE_ROOT>/08_Regression_modeling/00_Script/
    02_compare_single_stage_count_families.py
    04_compare_nb1_hurdle_zinb.py
    05_refine_component_specific_predictors.py
    06c_audit_parameter_identifiability_and_stabilize_inference.py

<CODE_ROOT>/08_Regression_modeling/
    06d_Final_Stable_Fixed_Effect_Inference/
        14_Final_Six_Fixed_Effect_Inference_Structures.csv
        17_Final_Selected_OOF_Manifest.csv
        Final_Selected_Inference_OOF_Predictions/

Output
------
<CODE_ROOT>/08_Regression_modeling/
    07_Regression_Spatial_Block_Validation/

Purpose
-------
This step is the first and only place in the published workflow where the
shared spatial blocks are generated. The assignment is built directly from
the eligible grid centroids in Base_Table_2001_2025.csv.bz2 using the same locked
algorithm previously used by the final Hurdle RF validation:

1. retain panel rows with ForestArea_km2 > 2.5;
2. reproduce the locked shared validation universe by excluding rows that are
   incomplete for the original final-RF validation inputs, responses, or
   coordinates; this removes 525 records belonging to seven fully incomplete
   grids and leaves the locked 4,982-grid universe;
3. standardize UTM52N centroid X and Y;
4. run five-cluster KMeans with random_state=2026 and n_init=50;
5. renumber clusters west to east as Spatial_Block 1-5.

The generated Spatial_Block_Assignment.csv is then used for five-fold spatial
out-of-fold validation of two locked regression roles:

1. Baseline_NB1: the Step-02 reference single-stage NB1 structure;
2. Final_Regression: the six fixed NB1/ZINB1 structures finalized in Step 06d.

No model structure or predictor is selected here. Each fold trains on four
spatial blocks and predicts the held-out fifth block. Later RF steps must read
this Step-07 assignment rather than regenerate spatial blocks.

Release implementation note
---------
The release implementation keeps the accepted fit parameters unchanged and replaces only the
positive-probability post-processing. It computes NB1 P(Y > 0) directly with
-expm1(log P(Y = 0)), avoiding catastrophic cancellation when the stored zero
probability rounds to 1.0. Existing accepted parameter checkpoints are
therefore refreshed without refitting.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import platform
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np
import pandas as pd
import scipy
import sklearn
import statsmodels
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error


# =============================================================================
# Repository-relative paths
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

FINAL_REGRESSION_ROOT = (
    REGRESSION_ROOT
    / "06d_Final_Stable_Fixed_Effect_Inference"
)
FINAL_STRUCTURE_CSV = (
    FINAL_REGRESSION_ROOT
    / "14_Final_Six_Fixed_Effect_Inference_Structures.csv"
)
FINAL_OOF_MANIFEST_CSV = (
    FINAL_REGRESSION_ROOT
    / "17_Final_Selected_OOF_Manifest.csv"
)
FINAL_TEMPORAL_OOF_ROOT = (
    FINAL_REGRESSION_ROOT
    / "Final_Selected_Inference_OOF_Predictions"
)

OUTPUT_ROOT = (
    REGRESSION_ROOT
    / "07_Regression_Spatial_Block_Validation"
)
SPATIAL_BLOCK_FILE = OUTPUT_ROOT / "Spatial_Block_Assignment.csv"
SPATIAL_BLOCK_SUMMARY_FILE = OUTPUT_ROOT / "Spatial_Block_Summary.csv"
SPATIAL_BLOCK_QA_FILE = OUTPUT_ROOT / "Spatial_Block_Generation_QA.csv"
SPATIAL_BLOCK_METADATA_FILE = (
    OUTPUT_ROOT / "Spatial_Block_Generation_Metadata.json"
)
SPATIAL_BLOCK_EXCLUDED_RECORDS_FILE = (
    OUTPUT_ROOT / "Spatial_Block_Eligibility_Excluded_Records.csv"
)
SPATIAL_BLOCK_EXCLUDED_GRIDS_FILE = (
    OUTPUT_ROOT / "Spatial_Block_Eligibility_Excluded_Grids.csv"
)
CHECKPOINT_ROOT = OUTPUT_ROOT / "Checkpoints"
FOLD_PREDICTION_ROOT = CHECKPOINT_ROOT / "Fold_Predictions"
FOLD_PARAMETER_ROOT = CHECKPOINT_ROOT / "Fold_Parameters"
FAILED_FIT_ROOT = CHECKPOINT_ROOT / "Failed_Fit_Diagnostics"
LOG_FILE = OUTPUT_ROOT / "regression_spatial_block_validation.log"

BASELINE_OOF_FILE = (
    OUTPUT_ROOT
    / "Baseline_NB1_Spatial_Out_of_Fold_Predictions.csv.gz"
)
FINAL_OOF_FILE = (
    OUTPUT_ROOT
    / "Final_Regression_Spatial_Out_of_Fold_Predictions.csv.gz"
)


# =============================================================================
# Locked workflow settings
# =============================================================================

STEP07_CODE_VERSION = "2026-07-08_RELEASE_STABLE_POSITIVE_PROBABILITY"
SPATIAL_BLOCK_COUNT = 5
SPATIAL_BLOCK_RANDOM_SEED = 2026
SPATIAL_BLOCK_N_INIT = 50
SPATIAL_BLOCK_DESIGN_ID = "KMEANS_UTM52_XY_5BLOCKS_SEED2026_WEST_TO_EAST_COMMON_UNIVERSE_V2"
SPATIAL_FIT_PROTOCOL_ID = (
    "REGRESSION_SPATIAL_OOF_STABLE_NB1_ZINB1_STEP06D_SCORE5E-5_RESCUE_V3"
)
# The fitted likelihood and optimizer are unchanged. The release implementation changes
# only probability/prediction post-processing. Keeping the fit protocol ID
# unchanged allows the 60 accepted parameter checkpoints to be reused.
SPATIAL_PREDICTION_PROTOCOL_ID = (
    "STABLE_LOGSPACE_POSITIVE_PROBABILITY_AND_CONDITIONAL_MEAN_V4"
)
# Step 06d adopted this scale-independent threshold for the final stable
# fixed-effect structures. Spatial folds use the same acceptance rule rather
# than the earlier, more restrictive Step-06c screening threshold.
SPATIAL_RELATIVE_GRADIENT_MAX = 5e-5

# Canonical SHA256 of sorted lines: GRID_UID,Spatial_Block\n without a header.
# This locks the semantic grid-to-block mapping while remaining independent of
# CSV byte-order marks, float formatting, and operating-system line endings.
EXPECTED_SPATIAL_MAPPING_SHA256 = (
    "715f9fe342eec78d936d223a98f5a4d0378281d3a3fa420ed250ba07fe056cfe"
)

# Historical byte-level hash of the same assignment produced by the earlier RF
# workflow. It is recorded for compatibility only; the semantic hash above is
# the authoritative check.
REFERENCE_SPATIAL_BLOCK_FILE_SHA256 = (
    "da752eff8f734d29e0ff7b5535c93277b4b146b0b5a84e27efe3f08d17f61a11"
)

EXPECTED_BLOCK_GRID_COUNTS = {
    1: 912,
    2: 1042,
    3: 735,
    4: 1222,
    5: 1071,
}

COORDINATE_COLUMNS = [
    "Centroid_X_UTM52_m",
    "Centroid_Y_UTM52_m",
    "Longitude",
    "Latitude",
]

SEASONS = {
    1: "Spring",
    2: "Summer",
    3: "Autumn",
}

RATE_NAMES = {
    "Fire_Count": "FCD",
    "Burned_Pixel_Count": "BAD",
}

COUNT_RESPONSES = [
    "Fire_Count",
    "Burned_Pixel_Count",
]

COUNTRY_LEVELS = [
    "China",
    "NK",
    "Russia",
]

COUNTRY_TERMS = [
    "Country_NK",
    "Country_Russia",
]

SPATIAL_FOLDS = [1, 2, 3, 4, 5]

FOREST_PIXEL_AREA_KM2 = 0.25
FOREST_AREA_THRESHOLD_KM2 = 2.5

# The original RF Step 06 generated spatial blocks after its row-level
# complete-case filter. The seven additional forest-eligible grids identified in the audit
# have no complete model row because static DEM/Slope/Aspect/POP fields are
# missing. To preserve the already validated RF partition while moving its
# generation upstream into 08, Step 07 now formalizes that exact shared
# validation universe independently of any 09 output file.
SHARED_SPATIAL_ELIGIBILITY_PREDICTORS = [
    "BD",
    "ND",
    "NE",
    "EVI",
    "PTC",
    "Temp",
    "Pre",
    "Rhum",
    "Wind",
    "SSRD",
    "LtgProxy",
    "SPEI3",
    "SPEI12",
    "DEM",
    "Slope",
    "Aspect",
    "POP",
    "Dis_Farm",
    "Road_dens",
    "Dis_Railway",
]
EXPECTED_FOREST_ELIGIBLE_GRID_COUNT_BEFORE_COMPLETENESS = 4_989
EXPECTED_SPATIAL_ELIGIBILITY_EXCLUDED_RECORD_COUNT = 525
EXPECTED_SPATIAL_ELIGIBILITY_EXCLUDED_GRID_UIDS = [
    "Russia|BE-90",
    "Russia|BF-91",
    "Russia|BJ-92",
    "Russia|BJ-93",
    "Russia|CH-84",
    "Russia|CM-75",
    "Russia|CT-67",
]

# The Step-02 family-screening reference structure is the locked baseline.
BASELINE_CONTINUOUS_PREDICTORS = [
    "BD",
    "PTC",
    "FWI",
    "LtgProxy",
    "SPEI3",
    "DEM",
    "Slope",
    "Aspect",
    "POP",
    "Dis_Farm",
    "Road_dens",
]

BASELINE_TRANSFORMATION_SCHEME = "Audit_Guided_Log1p"
BASELINE_CANDIDATE_ID = "BASELINE_SINGLE_STAGE_NB1"

EXPECTED_SEASON_RESPONSE_ROWS = 123_729
EXPECTED_GRID_COUNT = 4_982
EXPECTED_COMBINATIONS = 6
EXPECTED_ROWS_PER_MODEL_ROLE = (
    EXPECTED_SEASON_RESPONSE_ROWS * EXPECTED_COMBINATIONS
)

PREDICTION_COLUMNS = [
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
    "Centroid_X_UTM52_m",
    "Centroid_Y_UTM52_m",
    "Longitude",
    "Latitude",
    "Count_Response",
    "Rate_Scale_Name",
    "Model_Role",
    "Model_ID",
    "Model_Structure",
    "Candidate_ID",
    "Transformation_Scheme",
    "ForestPixelCount",
    "Observed_Count",
    "Predicted_Count",
    "Observed_Rate",
    "Predicted_Rate",
    "Observed_Zero",
    "Predicted_Zero_Probability",
    "Predicted_Positive_Probability",
    "Predicted_Conditional_Positive_Count",
    "Predicted_Conditional_Positive_Rate",
    "Predictive_LogProbability",
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
            "Required files are missing:\n"
            + "\n".join(str(path) for path in missing)
        )


def load_module(path: Path, module_name: str):
    if not path.exists():
        raise FileNotFoundError(f"Required script not found:\n{path}")

    specification = importlib.util.spec_from_file_location(module_name, path)
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


def safe_name(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def model_definition_signature(definition: Mapping[str, object]) -> str:
    fields = [
        "Season_Code",
        "Count_Response",
        "Model_Role",
        "Model_Structure",
        "Candidate_ID",
        "Structural_Zero_Predictors",
        "Structural_Zero_Country_Terms",
        "Count_Predictors",
        "Count_Country_Terms",
        "Transformation_Scheme",
    ]
    payload = {field: str(definition.get(field, "")) for field in fields}
    text = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()



def safe_rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Compute RMSE without squaring unscaled extreme residuals."""
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    finite = np.isfinite(observed) & np.isfinite(predicted)
    residual = predicted[finite] - observed[finite]
    if len(residual) == 0:
        return np.nan
    scale = float(np.max(np.abs(residual)))
    if scale == 0.0:
        return 0.0
    return float(scale * np.sqrt(np.mean(np.square(residual / scale))))


def safe_r2(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Compute R-squared without directly squaring unscaled extremes."""
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    finite = np.isfinite(observed) & np.isfinite(predicted)
    observed = observed[finite]
    predicted = predicted[finite]
    if len(observed) < 2 or np.allclose(observed, observed[0]):
        return np.nan

    residual = observed - predicted
    centered = observed - observed.mean()
    residual_scale = float(np.max(np.abs(residual)))
    centered_scale = float(np.max(np.abs(centered)))
    if centered_scale == 0.0 or not np.isfinite(centered_scale):
        return np.nan
    if residual_scale == 0.0:
        return 1.0
    if not np.isfinite(residual_scale):
        return np.nan

    residual_sum = float(
        np.dot(residual / residual_scale, residual / residual_scale)
    )
    centered_sum = float(
        np.dot(centered / centered_scale, centered / centered_scale)
    )
    if residual_sum <= 0.0 or centered_sum <= 0.0:
        return np.nan

    log_ratio = (
        2.0 * (np.log(residual_scale) - np.log(centered_scale))
        + np.log(residual_sum)
        - np.log(centered_sum)
    )
    if log_ratio > np.log(np.finfo(float).max):
        return -np.inf
    return float(1.0 - np.exp(log_ratio))


def safe_spearman(observed: np.ndarray, predicted: np.ndarray) -> float:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    finite = np.isfinite(observed) & np.isfinite(predicted)
    observed = observed[finite]
    predicted = predicted[finite]
    if (
        len(observed) < 3
        or np.allclose(observed, observed[0])
        or np.allclose(predicted, predicted[0])
    ):
        return np.nan
    return float(stats.spearmanr(observed, predicted)[0])


def calibration_parameters(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> Tuple[float, float]:
    """Return intercept and slope using centered, scale-normalized moments."""
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    finite = np.isfinite(observed) & np.isfinite(predicted)
    observed = observed[finite]
    predicted = predicted[finite]
    if len(observed) < 2 or np.allclose(predicted, predicted[0]):
        return np.nan, np.nan

    x_scale = float(np.max(np.abs(predicted)))
    y_scale = float(np.max(np.abs(observed)))
    if x_scale == 0.0 or y_scale == 0.0:
        return np.nan, np.nan

    x = predicted / x_scale
    y = observed / y_scale
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = float(np.dot(x_centered, x_centered))
    if denominator <= 0.0 or not np.isfinite(denominator):
        return np.nan, np.nan
    slope_scaled = float(np.dot(x_centered, y_centered) / denominator)
    slope = slope_scaled * y_scale / x_scale
    intercept = float(observed.mean() - slope * predicted.mean())
    return intercept, float(slope)


# =============================================================================
# Spatial-block integrity
# =============================================================================


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
        index=False,
        header=False,
        lineterminator="\n",
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_spatial_grid_source() -> Tuple[pd.DataFrame, Dict[str, object]]:
    required = {
        "GRID_UID",
        "Country",
        "GRID_ID",
        "Year",
        "Season",
        "Fire_Count",
        "Burned_Pixel_Count",
        "ForestPixelCount",
        "ForestArea_km2",
        *SHARED_SPATIAL_ELIGIBILITY_PREDICTORS,
        *COORDINATE_COLUMNS,
    }

    header = pd.read_csv(INPUT_CSV, nrows=0)
    missing = sorted(required - set(header.columns))
    if missing:
        raise ValueError(
            "Base_Table_2001_2025.csv.bz2 is missing spatial-block fields:\n"
            + "\n".join(missing)
        )

    source = pd.read_csv(INPUT_CSV, usecols=sorted(required), low_memory=False)
    source["GRID_UID"] = source["GRID_UID"].astype(str)

    exposure = pd.to_numeric(source["ForestPixelCount"], errors="coerce")
    forest_area = pd.to_numeric(source["ForestArea_km2"], errors="coerce")
    if exposure.isna().any() or forest_area.isna().any():
        raise ValueError("Forest exposure fields contain nonnumeric values.")

    area_error = np.abs(forest_area - exposure * FOREST_PIXEL_AREA_KM2)
    max_area_error = float(area_error.max()) if len(area_error) else np.nan
    if not np.isfinite(max_area_error) or max_area_error > 1e-9:
        raise ValueError(
            "ForestArea_km2 is inconsistent with ForestPixelCount * 0.25."
        )

    forest_eligible = source.loc[
        forest_area > FOREST_AREA_THRESHOLD_KM2
    ].copy()
    if forest_eligible.empty:
        raise ValueError("No rows remain after the forest-area threshold.")

    forest_eligible_grid_count = int(forest_eligible["GRID_UID"].nunique())
    if (
        forest_eligible_grid_count
        != EXPECTED_FOREST_ELIGIBLE_GRID_COUNT_BEFORE_COMPLETENESS
    ):
        raise ValueError(
            "The forest-area filter did not reproduce the expected pre-"
            "completeness grid universe.\n"
            f"Expected {EXPECTED_FOREST_ELIGIBLE_GRID_COUNT_BEFORE_COMPLETENESS:,} "
            f"grids, observed {forest_eligible_grid_count:,}."
        )

    # Reproduce the row-level completeness screen that was applied before the
    # original RF spatial blocks were generated. FCD and BAD are included here
    # because the old code screened both derived responses together.
    numeric_columns = [
        "Fire_Count",
        "Burned_Pixel_Count",
        "ForestPixelCount",
        *SHARED_SPATIAL_ELIGIBILITY_PREDICTORS,
        *COORDINATE_COLUMNS,
    ]
    for column in numeric_columns:
        forest_eligible[column] = pd.to_numeric(
            forest_eligible[column], errors="coerce"
        )

    forest_eligible["FCD"] = np.where(
        forest_eligible["ForestPixelCount"] > 0,
        forest_eligible["Fire_Count"] / forest_eligible["ForestPixelCount"],
        np.nan,
    )
    forest_eligible["BAD"] = np.where(
        forest_eligible["ForestPixelCount"] > 0,
        forest_eligible["Burned_Pixel_Count"]
        / forest_eligible["ForestPixelCount"],
        np.nan,
    )

    completeness_columns = [
        *SHARED_SPATIAL_ELIGIBILITY_PREDICTORS,
        "FCD",
        "BAD",
        *COORDINATE_COLUMNS,
    ]
    missing_matrix = forest_eligible[completeness_columns].isna()
    incomplete = missing_matrix.any(axis=1)

    excluded_records = forest_eligible.loc[
        incomplete,
        [
            "GRID_UID",
            "Country",
            "GRID_ID",
            "Year",
            "Season",
            "ForestPixelCount",
            "ForestArea_km2",
            *SHARED_SPATIAL_ELIGIBILITY_PREDICTORS,
            *COORDINATE_COLUMNS,
        ],
    ].copy()
    if not excluded_records.empty:
        excluded_records["Missing_Eligibility_Fields"] = [
            ";".join(
                column
                for column, is_missing in zip(
                    completeness_columns,
                    row,
                )
                if bool(is_missing)
            )
            for row in missing_matrix.loc[incomplete].to_numpy()
        ]
    excluded_records.to_csv(
        SPATIAL_BLOCK_EXCLUDED_RECORDS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    if len(excluded_records) != EXPECTED_SPATIAL_ELIGIBILITY_EXCLUDED_RECORD_COUNT:
        raise ValueError(
            "The shared spatial-universe completeness screen excluded an "
            "unexpected number of panel records.\n"
            f"Expected {EXPECTED_SPATIAL_ELIGIBILITY_EXCLUDED_RECORD_COUNT:,}, "
            f"observed {len(excluded_records):,}. See "
            f"{SPATIAL_BLOCK_EXCLUDED_RECORDS_FILE.name}."
        )

    model_eligible = forest_eligible.loc[~incomplete].copy()
    all_forest_grids = set(forest_eligible["GRID_UID"].astype(str).unique())
    retained_grids = set(model_eligible["GRID_UID"].astype(str).unique())
    fully_excluded_grid_uids = sorted(all_forest_grids - retained_grids)

    if fully_excluded_grid_uids != EXPECTED_SPATIAL_ELIGIBILITY_EXCLUDED_GRID_UIDS:
        raise ValueError(
            "The fully excluded grid set differs from the locked shared "
            "validation universe.\nExpected:\n"
            + "\n".join(EXPECTED_SPATIAL_ELIGIBILITY_EXCLUDED_GRID_UIDS)
            + "\nObserved:\n"
            + "\n".join(fully_excluded_grid_uids)
        )

    excluded_grid_summary = (
        excluded_records.groupby(
            ["GRID_UID", "Country", "GRID_ID"], as_index=False
        )
        .agg(
            Excluded_Record_Count=("Year", "size"),
            First_Excluded_Year=("Year", "min"),
            Last_Excluded_Year=("Year", "max"),
            Season_Count=("Season", "nunique"),
            ForestPixelCount=("ForestPixelCount", "first"),
            ForestArea_km2=("ForestArea_km2", "first"),
        )
        .sort_values("GRID_UID")
        .reset_index(drop=True)
    )
    excluded_grid_summary["Exclusion_Reason"] = (
        "No complete panel row under the locked shared validation "
        "eligibility fields"
    )
    excluded_grid_summary.to_csv(
        SPATIAL_BLOCK_EXCLUDED_GRIDS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    stable_columns = ["Country", "GRID_ID", *COORDINATE_COLUMNS]
    conflicting_grids = []
    grouped = model_eligible.groupby("GRID_UID", sort=False)
    for column in stable_columns:
        conflict = grouped[column].nunique(dropna=False) > 1
        if conflict.any():
            conflicting_grids.extend(conflict.index[conflict].astype(str).tolist())
    conflicting_grids = sorted(set(conflicting_grids))
    if conflicting_grids:
        conflict_rows = model_eligible.loc[
            model_eligible["GRID_UID"].isin(conflicting_grids),
            ["GRID_UID", *stable_columns],
        ].sort_values(["GRID_UID", "Country", "GRID_ID"])
        conflict_rows.to_csv(
            OUTPUT_ROOT / "ERROR_Conflicting_Grid_Coordinates.csv",
            index=False,
            encoding="utf-8-sig",
        )
        raise ValueError(
            "Country, GRID_ID, or coordinates vary within GRID_UID. "
            "See ERROR_Conflicting_Grid_Coordinates.csv."
        )

    grids = (
        model_eligible[["GRID_UID", "Country", "GRID_ID", *COORDINATE_COLUMNS]]
        .drop_duplicates("GRID_UID")
        .sort_values("GRID_UID")
        .reset_index(drop=True)
    )

    if grids["GRID_UID"].duplicated().any():
        raise RuntimeError("Duplicate GRID_UID values remained after grid extraction.")
    if len(grids) != EXPECTED_GRID_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_GRID_COUNT:,} eligible grids, "
            f"but found {len(grids):,}."
        )

    source_metadata = {
        "Input_Path": str(INPUT_CSV),
        "Input_File_SHA256": sha256_file(INPUT_CSV),
        "Input_Rows": int(len(source)),
        "Forest_Eligible_Rows_Before_Completeness": int(len(forest_eligible)),
        "Forest_Eligible_GRID_UID_Count_Before_Completeness": (
            forest_eligible_grid_count
        ),
        "Spatial_Eligibility_Excluded_Record_Count": int(len(excluded_records)),
        "Spatial_Eligibility_Fully_Excluded_GRID_UID_Count": int(
            len(fully_excluded_grid_uids)
        ),
        "Spatial_Eligibility_Fully_Excluded_GRID_UIDs": fully_excluded_grid_uids,
        "Model_Eligible_Rows_After_Completeness": int(len(model_eligible)),
        "Eligible_GRID_UID_Count": int(grids["GRID_UID"].nunique()),
        "Forest_Area_Threshold_km2": FOREST_AREA_THRESHOLD_KM2,
        "Forest_Pixel_Area_km2": FOREST_PIXEL_AREA_KM2,
        "Spatial_Eligibility_Predictors": SHARED_SPATIAL_ELIGIBILITY_PREDICTORS,
        "Spatial_Eligibility_Response_Fields": ["FCD", "BAD"],
        "Spatial_Eligibility_Coordinate_Fields": COORDINATE_COLUMNS,
        "Maximum_Forest_Area_Consistency_Error": max_area_error,
        "Coordinate_Conflict_GRID_UID_Count": 0,
    }
    return grids, source_metadata


def generate_and_verify_spatial_blocks() -> Tuple[pd.DataFrame, Dict[str, object]]:
    grids, source_metadata = read_spatial_grid_source()

    coordinates = grids[
        ["Centroid_X_UTM52_m", "Centroid_Y_UTM52_m"]
    ].to_numpy(dtype=float)
    coordinate_mean = coordinates.mean(axis=0)
    coordinate_sd = coordinates.std(axis=0)
    if np.any(~np.isfinite(coordinate_sd)) or np.any(coordinate_sd <= 0):
        raise ValueError("Invalid coordinate standard deviation.")

    standardized = (coordinates - coordinate_mean) / coordinate_sd
    kmeans = KMeans(
        n_clusters=SPATIAL_BLOCK_COUNT,
        random_state=SPATIAL_BLOCK_RANDOM_SEED,
        n_init=SPATIAL_BLOCK_N_INIT,
    )
    raw_labels = kmeans.fit_predict(standardized)

    centroid_x = {
        raw_label: float(coordinates[raw_labels == raw_label, 0].mean())
        for raw_label in range(SPATIAL_BLOCK_COUNT)
    }
    ordered_raw_labels = sorted(centroid_x, key=centroid_x.get)
    label_map = {
        raw_label: index + 1
        for index, raw_label in enumerate(ordered_raw_labels)
    }
    grids["Spatial_Block"] = np.asarray(
        [label_map[int(label)] for label in raw_labels],
        dtype=int,
    )

    if set(grids["Spatial_Block"].unique()) != set(SPATIAL_FOLDS):
        raise ValueError("Generated spatial blocks are not exactly 1-5.")

    observed_counts = {
        int(block): int(count)
        for block, count in grids["Spatial_Block"].value_counts().sort_index().items()
    }
    if observed_counts != EXPECTED_BLOCK_GRID_COUNTS:
        raise ValueError(
            "Generated spatial-block sizes differ from the locked design.\n"
            f"Expected: {EXPECTED_BLOCK_GRID_COUNTS}\n"
            f"Observed: {observed_counts}"
        )

    mapping_hash = spatial_mapping_sha256(grids)
    if mapping_hash != EXPECTED_SPATIAL_MAPPING_SHA256:
        raise ValueError(
            "Generated GRID_UID-to-Spatial_Block mapping differs from the "
            "locked design.\n"
            f"Expected semantic SHA256: {EXPECTED_SPATIAL_MAPPING_SHA256}\n"
            f"Observed semantic SHA256: {mapping_hash}"
        )

    output_columns = [
        "GRID_UID",
        "Country",
        "GRID_ID",
        *COORDINATE_COLUMNS,
        "Spatial_Block",
    ]
    grids = grids[output_columns].copy()
    grids.to_csv(
        SPATIAL_BLOCK_FILE,
        index=False,
        encoding="utf-8-sig",
    )
    file_hash = sha256_file(SPATIAL_BLOCK_FILE)

    block_summary = (
        grids.groupby(["Spatial_Block", "Country"], as_index=False)
        .agg(
            GRID_UID_Count=("GRID_UID", "nunique"),
            X_Min=("Centroid_X_UTM52_m", "min"),
            X_Max=("Centroid_X_UTM52_m", "max"),
            Y_Min=("Centroid_Y_UTM52_m", "min"),
            Y_Max=("Centroid_Y_UTM52_m", "max"),
            Longitude_Min=("Longitude", "min"),
            Longitude_Max=("Longitude", "max"),
            Latitude_Min=("Latitude", "min"),
            Latitude_Max=("Latitude", "max"),
        )
        .sort_values(["Spatial_Block", "Country"])
        .reset_index(drop=True)
    )
    block_summary.to_csv(
        SPATIAL_BLOCK_SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    qa_rows = []
    for block in SPATIAL_FOLDS:
        subset = grids.loc[grids["Spatial_Block"] == block]
        qa_rows.append(
            {
                "Spatial_Block": block,
                "GRID_UID_Count": int(subset["GRID_UID"].nunique()),
                "Expected_GRID_UID_Count": EXPECTED_BLOCK_GRID_COUNTS[block],
                "Grid_Count_Matches_Locked_Design": bool(
                    subset["GRID_UID"].nunique() == EXPECTED_BLOCK_GRID_COUNTS[block]
                ),
                "Country_Count": int(subset["Country"].nunique()),
                "Centroid_X_Mean": float(subset["Centroid_X_UTM52_m"].mean()),
                "Centroid_Y_Mean": float(subset["Centroid_Y_UTM52_m"].mean()),
                "Semantic_Mapping_SHA256": mapping_hash,
                "Semantic_Mapping_Hash_Verified": True,
            }
        )
    pd.DataFrame(qa_rows).to_csv(
        SPATIAL_BLOCK_QA_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    metadata = {
        "Design_ID": SPATIAL_BLOCK_DESIGN_ID,
        "Generated_At": datetime.now().isoformat(timespec="seconds"),
        "Generation_Stage": "08_Regression_modeling Step 07",
        "Downstream_Rule": (
            "09_RF_modeling must read this assignment and must not rerun KMeans."
        ),
        "Algorithm": {
            "Eligible_Grid_Filter": (
                "ForestArea_km2 > 2.5 followed by the locked shared "
                "row-level completeness screen"
            ),
            "Forest_Eligible_Grid_Count_Before_Completeness": (
                EXPECTED_FOREST_ELIGIBLE_GRID_COUNT_BEFORE_COMPLETENESS
            ),
            "Expected_Excluded_Record_Count": (
                EXPECTED_SPATIAL_ELIGIBILITY_EXCLUDED_RECORD_COUNT
            ),
            "Expected_Fully_Excluded_GRID_UIDs": (
                EXPECTED_SPATIAL_ELIGIBILITY_EXCLUDED_GRID_UIDS
            ),
            "Coordinate_Columns": [
                "Centroid_X_UTM52_m",
                "Centroid_Y_UTM52_m",
            ],
            "Coordinate_Standardization": "population mean and SD (ddof=0)",
            "Clustering": "sklearn.cluster.KMeans",
            "n_clusters": SPATIAL_BLOCK_COUNT,
            "random_state": SPATIAL_BLOCK_RANDOM_SEED,
            "n_init": SPATIAL_BLOCK_N_INIT,
            "Final_Label_Order": "cluster centroid X from west to east",
        },
        "Coordinate_Mean": {
            "Centroid_X_UTM52_m": float(coordinate_mean[0]),
            "Centroid_Y_UTM52_m": float(coordinate_mean[1]),
        },
        "Coordinate_SD": {
            "Centroid_X_UTM52_m": float(coordinate_sd[0]),
            "Centroid_Y_UTM52_m": float(coordinate_sd[1]),
        },
        "Raw_Cluster_Centroid_X": {
            str(key): value for key, value in centroid_x.items()
        },
        "Raw_to_Final_Label_Map": {
            str(key): int(value) for key, value in label_map.items()
        },
        "Block_GRID_UID_Counts": {
            str(key): int(value) for key, value in observed_counts.items()
        },
        "Expected_Semantic_Mapping_SHA256": EXPECTED_SPATIAL_MAPPING_SHA256,
        "Observed_Semantic_Mapping_SHA256": mapping_hash,
        "Semantic_Mapping_Hash_Verified": True,
        "Generated_File_SHA256": file_hash,
        "Reference_Earlier_RF_File_SHA256": REFERENCE_SPATIAL_BLOCK_FILE_SHA256,
        "Generated_File_Byte_Hash_Matches_Earlier_RF_File": bool(
            file_hash == REFERENCE_SPATIAL_BLOCK_FILE_SHA256
        ),
        **source_metadata,
    }
    SPATIAL_BLOCK_METADATA_FILE.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return grids, metadata


# =============================================================================
# Locked model definitions
# =============================================================================


def read_final_structure() -> pd.DataFrame:
    structure = pd.read_csv(FINAL_STRUCTURE_CSV)
    required = {
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
        "Transformation_Scheme",
    }
    missing = sorted(required - set(structure.columns))
    if missing:
        raise ValueError(
            "The Step-06d final structure table is missing columns:\n"
            + "\n".join(missing)
        )

    expected = {
        (season, response)
        for season in SEASONS
        for response in COUNT_RESPONSES
    }
    observed = {
        (int(row.Season_Code), str(row.Count_Response))
        for row in structure.itertuples(index=False)
    }
    if len(structure) != EXPECTED_COMBINATIONS or observed != expected:
        raise ValueError(
            "The Step-06d structure table does not contain exactly the six "
            "expected season-response combinations."
        )

    return structure.sort_values(
        ["Season_Code", "Count_Response"]
    ).reset_index(drop=True)


def build_model_definitions(final_structure: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for season in SEASONS:
        for response in COUNT_RESPONSES:
            rows.append(
                {
                    "Season_Code": season,
                    "Season_Label": SEASONS[season],
                    "Count_Response": response,
                    "Rate_Scale_Name": RATE_NAMES[response],
                    "Model_Role": "Baseline_NB1",
                    "Model_ID": "Baseline_NB1",
                    "Model_Structure": "Single_Stage_NB1",
                    "Candidate_ID": BASELINE_CANDIDATE_ID,
                    "Structural_Zero_Predictors": "",
                    "Structural_Zero_Country_Terms": "",
                    "Count_Predictors": ";".join(
                        BASELINE_CONTINUOUS_PREDICTORS
                    ),
                    "Count_Country_Terms": ";".join(COUNTRY_TERMS),
                    "Transformation_Scheme": BASELINE_TRANSFORMATION_SCHEME,
                    "Definition_Source": (
                        "Step-02 reference single-stage NB1 structure"
                    ),
                }
            )

    for row in final_structure.itertuples(index=False):
        rows.append(
            {
                "Season_Code": int(row.Season_Code),
                "Season_Label": str(row.Season_Label),
                "Count_Response": str(row.Count_Response),
                "Rate_Scale_Name": str(row.Rate_Scale_Name),
                "Model_Role": "Final_Regression",
                "Model_ID": "Final_Regression",
                "Model_Structure": str(row.Model_Structure),
                "Candidate_ID": str(row.Selected_Inference_Candidate_ID),
                "Structural_Zero_Predictors": (
                    "" if pd.isna(row.Structural_Zero_Predictors)
                    else str(row.Structural_Zero_Predictors)
                ),
                "Structural_Zero_Country_Terms": (
                    "" if pd.isna(row.Structural_Zero_Country_Terms)
                    else str(row.Structural_Zero_Country_Terms)
                ),
                "Count_Predictors": str(row.Count_Predictors),
                "Count_Country_Terms": (
                    "" if pd.isna(row.Count_Country_Terms)
                    else str(row.Count_Country_Terms)
                ),
                "Transformation_Scheme": str(row.Transformation_Scheme),
                "Definition_Source": (
                    "Step-06d final stable fixed-effect inference structure"
                ),
            }
        )

    definitions = pd.DataFrame(rows)
    if len(definitions) != 12:
        raise RuntimeError("Exactly 12 locked model definitions were expected.")

    definitions["Definition_Order"] = np.arange(1, len(definitions) + 1)
    definitions["Definition_Signature"] = [
        model_definition_signature(record)
        for record in definitions.to_dict(orient="records")
    ]
    return definitions


# =============================================================================
# Exact analysis samples
# =============================================================================


def final_oof_path(output_file: str) -> Path:
    candidates = [
        FINAL_TEMPORAL_OOF_ROOT / output_file,
        FINAL_REGRESSION_ROOT / output_file,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "A Step-06d final temporal OOF file listed in the manifest was not found:\n"
        + "\n".join(str(path) for path in candidates)
    )


def read_base_table(model_definitions: pd.DataFrame) -> pd.DataFrame:
    all_predictors = sorted(
        {
            predictor
            for column in [
                "Structural_Zero_Predictors",
                "Count_Predictors",
            ]
            for value in model_definitions[column]
            for predictor in parse_list(value)
        }
    )

    required = {
        "GRID_UID",
        "GRID_ID",
        "Country",
        "Year",
        "Season",
        "Fire_Count",
        "Burned_Pixel_Count",
        "ForestPixelCount",
        "ForestArea_km2",
        *all_predictors,
    }

    header = pd.read_csv(INPUT_CSV, nrows=0)
    missing = sorted(required - set(header.columns))
    if missing:
        raise ValueError(
            "Base_Table_2001_2025.csv.bz2 is missing required columns:\n"
            + "\n".join(missing)
        )

    data = pd.read_csv(INPUT_CSV, usecols=sorted(required))
    data["GRID_UID"] = data["GRID_UID"].astype(str)

    if data.duplicated(["GRID_UID", "Year", "Season"], keep=False).any():
        raise ValueError("Duplicate GRID_UID-Year-Season records were found.")

    exposure = pd.to_numeric(data["ForestPixelCount"], errors="coerce")
    forest_area = pd.to_numeric(data["ForestArea_km2"], errors="coerce")

    if exposure.isna().any() or forest_area.isna().any():
        raise ValueError("Forest exposure fields contain nonnumeric values.")

    area_error = np.abs(forest_area - exposure * FOREST_PIXEL_AREA_KM2)
    if float(area_error.max()) > 1e-9:
        raise ValueError(
            "ForestArea_km2 is inconsistent with ForestPixelCount * 0.25."
        )

    data = data.loc[
        forest_area > FOREST_AREA_THRESHOLD_KM2
    ].copy().reset_index(drop=True)

    if (pd.to_numeric(data["ForestPixelCount"], errors="coerce") <= 0).any():
        raise ValueError("ForestPixelCount must be positive after filtering.")

    unexpected_countries = sorted(
        set(data["Country"].dropna().astype(str)) - set(COUNTRY_LEVELS)
    )
    if unexpected_countries:
        raise ValueError(
            "Unexpected Country values: " + ", ".join(unexpected_countries)
        )

    unexpected_seasons = sorted(
        set(data["Season"].dropna().astype(int)) - set(SEASONS)
    )
    if unexpected_seasons:
        raise ValueError(
            "Unexpected Season values: "
            + ", ".join(map(str, unexpected_seasons))
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

    return data


def read_final_oof_reference(
    season: int,
    response: str,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    selected = manifest.loc[
        (manifest["Season_Code"].astype(int) == season)
        & (manifest["Count_Response"].astype(str) == response)
    ]
    if len(selected) != 1:
        raise ValueError(
            f"The Step-06d OOF manifest did not uniquely identify "
            f"{SEASONS[season]} {response}."
        )

    path = final_oof_path(str(selected.iloc[0]["Output_File"]))
    oof = pd.read_csv(path, compression="gzip")
    required = {
        "GRID_UID",
        "GRID_ID",
        "Country",
        "Year",
        "Season",
        "Temporal_Fold",
        "ForestPixelCount",
        "Observed_Count",
        "Count_Response",
    }
    missing = sorted(required - set(oof.columns))
    if missing:
        raise ValueError(
            f"{path.name} is missing required columns:\n"
            + "\n".join(missing)
        )

    oof["GRID_UID"] = oof["GRID_UID"].astype(str)
    if oof.duplicated(["GRID_UID", "Year", "Season"], keep=False).any():
        raise ValueError(f"Duplicate panel keys were found in {path.name}.")
    if set(oof["Season"].astype(int)) != {season}:
        raise ValueError(f"Unexpected Season values in {path.name}.")
    if set(oof["Count_Response"].astype(str)) != {response}:
        raise ValueError(f"Unexpected Count_Response values in {path.name}.")
    if set(oof["Temporal_Fold"].astype(int)) != set(SPATIAL_FOLDS):
        raise ValueError(f"Unexpected temporal folds in {path.name}.")
    if len(oof) != EXPECTED_SEASON_RESPONSE_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_SEASON_RESPONSE_ROWS:,} rows in {path.name}, "
            f"but found {len(oof):,}."
        )

    return oof[
        [
            "GRID_UID",
            "GRID_ID",
            "Country",
            "Year",
            "Season",
            "Temporal_Fold",
            "ForestPixelCount",
            "Observed_Count",
        ]
    ].copy()


def exact_analysis_sample(
    base_data: pd.DataFrame,
    blocks: pd.DataFrame,
    manifest: pd.DataFrame,
    model_definitions: pd.DataFrame,
    season: int,
    response: str,
) -> pd.DataFrame:
    reference = read_final_oof_reference(
        season=season,
        response=response,
        manifest=manifest,
    )

    keys = reference.rename(
        columns={
            "GRID_ID": "OOF_GRID_ID",
            "Country": "OOF_Country",
            "ForestPixelCount": "OOF_ForestPixelCount",
            "Observed_Count": "OOF_Observed_Count",
        }
    )

    model_data = base_data.loc[
        base_data["Season"].astype(int) == season
    ].merge(
        keys,
        on=["GRID_UID", "Year", "Season"],
        how="inner",
        validate="one_to_one",
    )

    if len(model_data) != len(reference):
        raise ValueError(
            f"Base table and final OOF sample do not match for "
            f"{SEASONS[season]} {response}: "
            f"{len(model_data):,} versus {len(reference):,}."
        )

    if not (
        model_data["GRID_ID"].astype(str).to_numpy()
        == model_data["OOF_GRID_ID"].astype(str).to_numpy()
    ).all():
        raise ValueError("GRID_ID disagrees with the final OOF reference.")

    if not (
        model_data["Country"].astype(str).to_numpy()
        == model_data["OOF_Country"].astype(str).to_numpy()
    ).all():
        raise ValueError("Country disagrees with the final OOF reference.")

    if not np.allclose(
        model_data[response].to_numpy(dtype=float),
        model_data["OOF_Observed_Count"].to_numpy(dtype=float),
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError("Observed counts disagree with the final OOF reference.")

    if not np.allclose(
        model_data["ForestPixelCount"].to_numpy(dtype=float),
        model_data["OOF_ForestPixelCount"].to_numpy(dtype=float),
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError(
            "ForestPixelCount disagrees with the final OOF reference."
        )

    local_definitions = model_definitions.loc[
        (model_definitions["Season_Code"].astype(int) == season)
        & (model_definitions["Count_Response"].astype(str) == response)
    ]
    required_predictors = sorted(
        {
            predictor
            for column in [
                "Structural_Zero_Predictors",
                "Count_Predictors",
            ]
            for value in local_definitions[column]
            for predictor in parse_list(value)
        }
    )

    complete_columns = [
        "Country",
        response,
        "ForestPixelCount",
        *required_predictors,
    ]
    complete = (
        model_data[complete_columns]
        .replace([np.inf, -np.inf], np.nan)
        .notna()
        .all(axis=1)
    )
    if not complete.all():
        model_data.loc[
            ~complete,
            [
                "GRID_UID",
                "GRID_ID",
                "Country",
                "Year",
                "Season",
                *required_predictors,
            ],
        ].to_csv(
            OUTPUT_ROOT
            / f"ERROR_Incomplete_Spatial_Sample_S{season}_{response}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        raise ValueError(
            f"The locked common sample contains incomplete predictors for "
            f"{SEASONS[season]} {response}."
        )

    block_table = blocks.rename(
        columns={
            "Country": "Block_Country",
            "GRID_ID": "Block_GRID_ID",
        }
    )
    model_data = model_data.merge(
        block_table,
        on="GRID_UID",
        how="left",
        validate="many_to_one",
    )

    if model_data["Spatial_Block"].isna().any():
        missing_grids = sorted(
            model_data.loc[
                model_data["Spatial_Block"].isna(), "GRID_UID"
            ].unique()
        )
        raise ValueError(
            "Some model grids are absent from Spatial_Block_Assignment.csv:\n"
            + "\n".join(missing_grids[:50])
        )

    if not (
        model_data["Country"].astype(str).to_numpy()
        == model_data["Block_Country"].astype(str).to_numpy()
    ).all():
        raise ValueError("Country disagrees with Spatial_Block_Assignment.csv.")

    if not (
        model_data["GRID_ID"].astype(str).to_numpy()
        == model_data["Block_GRID_ID"].astype(str).to_numpy()
    ).all():
        raise ValueError("GRID_ID disagrees with Spatial_Block_Assignment.csv.")

    model_data["Spatial_Block"] = model_data["Spatial_Block"].astype(int)

    if set(model_data["Spatial_Block"].unique()) != set(SPATIAL_FOLDS):
        raise ValueError(
            f"Not all five spatial blocks occur in {SEASONS[season]} {response}."
        )

    if model_data.groupby("GRID_UID")["Spatial_Block"].nunique().max() != 1:
        raise ValueError("At least one GRID_UID belongs to multiple spatial blocks.")

    if len(model_data) != EXPECTED_SEASON_RESPONSE_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_SEASON_RESPONSE_ROWS:,} exact sample rows, "
            f"but found {len(model_data):,}."
        )

    if model_data["GRID_UID"].nunique() != EXPECTED_GRID_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_GRID_COUNT:,} unique grids, but found "
            f"{model_data['GRID_UID'].nunique():,}."
        )

    drop_columns = [
        "OOF_GRID_ID",
        "OOF_Country",
        "OOF_ForestPixelCount",
        "OOF_Observed_Count",
        "Block_Country",
        "Block_GRID_ID",
    ]
    return (
        model_data.drop(columns=drop_columns)
        .sort_values(["GRID_UID", "Year"])
        .reset_index(drop=True)
    )


# =============================================================================
# Fitting and prediction
# =============================================================================


def definition_components(definition: Mapping[str, object]) -> Dict[str, object]:
    return {
        "zero_predictors": parse_list(
            definition["Structural_Zero_Predictors"]
        ),
        "zero_country_terms": parse_list(
            definition["Structural_Zero_Country_Terms"]
        ),
        "count_predictors": parse_list(definition["Count_Predictors"]),
        "count_country_terms": parse_list(
            definition["Count_Country_Terms"]
        ),
        "transformation_scheme": str(
            definition["Transformation_Scheme"]
        ),
        "model_structure": str(definition["Model_Structure"]),
    }


def checkpoint_prefix(
    definition: Mapping[str, object],
    fold: int,
) -> str:
    return (
        f"{safe_name(SPATIAL_BLOCK_DESIGN_ID)}_"
        f"{safe_name(SPATIAL_FIT_PROTOCOL_ID)}_"
        f"{safe_name(definition['Model_Role'])}_"
        f"S{int(definition['Season_Code'])}_"
        f"{safe_name(definition['Count_Response'])}_"
        f"{safe_name(definition['Candidate_ID'])}_"
        f"{safe_name(definition['Definition_Signature'])}_"
        f"SpatialFold{fold}"
    )


def fold_prediction_path(
    definition: Mapping[str, object],
    fold: int,
) -> Path:
    return FOLD_PREDICTION_ROOT / f"{checkpoint_prefix(definition, fold)}.csv.gz"


def fold_parameter_paths(
    definition: Mapping[str, object],
    fold: int,
) -> Tuple[Path, Path]:
    prefix = checkpoint_prefix(definition, fold)
    return (
        FOLD_PARAMETER_ROOT / f"{prefix}.npz",
        FOLD_PARAMETER_ROOT / f"{prefix}.json",
    )


def optimizer_passes_spatial_rule(status: Mapping[str, object]) -> bool:
    """Return whether an optimizer result satisfies the Step-06d spatial rule."""
    try:
        relative_gradient = float(status.get("Relative_Gradient", np.nan))
    except (TypeError, ValueError):
        relative_gradient = np.nan
    return bool(
        status.get("Converged", False)
        and np.isfinite(relative_gradient)
        and relative_gradient <= SPATIAL_RELATIVE_GRADIENT_MAX
        and not status.get("Parameter_Bound_Hit", True)
        and np.isfinite(float(status.get("Negative_LogLikelihood", np.nan)))
    )


def _deduplicate_starts(
    labeled_starts: Iterable[Tuple[str, np.ndarray]],
) -> List[Tuple[str, np.ndarray]]:
    unique: List[Tuple[str, np.ndarray]] = []
    seen = set()
    for label, start in labeled_starts:
        array = np.asarray(start, dtype=float)
        if not np.isfinite(array).all():
            continue
        key = tuple(np.round(array, 8))
        if key in seen:
            continue
        seen.add(key)
        unique.append((label, array))
    return unique


def optimize_spatial_fold_with_rescue(
    step06c: object,
    training_start: np.ndarray,
    neutral_start: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    exposure: np.ndarray,
    model: str,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Fit deterministically and retry near-converged spatial-fold solutions.

    Step 06c's multistart helper selects the lowest objective even when a
    numerically equivalent candidate has a better standardized score. Spatial
    folds can therefore be rejected unnecessarily. This wrapper preserves the
    same likelihood, bounds and L-BFGS-B optimizer, but gives priority to
    candidates satisfying the final Step-06d standardized-score rule.
    """
    initial_theta, initial_status = step06c.optimize_multistart(
        [training_start, neutral_start],
        y,
        x,
        z,
        exposure,
        model,
    )
    initial_status = dict(initial_status)
    initial_status["Spatial_Attempt_Label"] = "Initial_multistart_best"
    attempts: List[Tuple[np.ndarray, Dict[str, object]]] = [
        (np.asarray(initial_theta, dtype=float), initial_status)
    ]
    internal_initial_attempts = int(initial_status.get("Attempt_Count", 1))

    if not optimizer_passes_spatial_rule(initial_status):
        train = np.asarray(training_start, dtype=float)
        neutral = np.asarray(neutral_start, dtype=float)
        best = np.asarray(initial_theta, dtype=float)
        rescue_starts = _deduplicate_starts(
            [
                ("Restart_from_initial_solution", best),
                ("Blend_75pct_training", 0.75 * train + 0.25 * neutral),
                ("Blend_50pct_training", 0.50 * train + 0.50 * neutral),
                ("Blend_25pct_training", 0.25 * train + 0.75 * neutral),
                ("Initial_solution_alpha_minus_0p5", np.r_[best[:-1], best[-1] - 0.5]),
                ("Initial_solution_alpha_plus_0p5", np.r_[best[:-1], best[-1] + 0.5]),
                ("Neutral_with_initial_alpha", np.r_[neutral[:-1], best[-1]]),
            ]
        )
        for label, start in rescue_starts:
            theta_i, status_i = step06c.optimize_strict(
                start,
                y,
                x,
                z,
                exposure,
                model,
            )
            status_i = dict(status_i)
            status_i["Spatial_Attempt_Label"] = label
            attempts.append((np.asarray(theta_i, dtype=float), status_i))
            # Once an accepted restart is found, refine it once more. This is
            # deterministic and often reduces the score by another order.
            if optimizer_passes_spatial_rule(status_i):
                theta_refined, status_refined = step06c.optimize_strict(
                    np.asarray(theta_i, dtype=float),
                    y,
                    x,
                    z,
                    exposure,
                    model,
                )
                status_refined = dict(status_refined)
                status_refined["Spatial_Attempt_Label"] = f"{label}_refined"
                attempts.append(
                    (np.asarray(theta_refined, dtype=float), status_refined)
                )
                break

    accepted = [
        item for item in attempts if optimizer_passes_spatial_rule(item[1])
    ]
    if accepted:
        selected_theta, selected_status = min(
            accepted,
            key=lambda item: float(item[1]["Negative_LogLikelihood"]),
        )
    else:
        def failure_key(item: Tuple[np.ndarray, Mapping[str, object]]) -> Tuple[float, ...]:
            status = item[1]
            try:
                rel = float(status.get("Relative_Gradient", np.inf))
            except (TypeError, ValueError):
                rel = np.inf
            try:
                nll = float(status.get("Negative_LogLikelihood", np.inf))
            except (TypeError, ValueError):
                nll = np.inf
            return (
                0.0 if status.get("Converged", False) else 1.0,
                0.0 if not status.get("Parameter_Bound_Hit", True) else 1.0,
                rel if np.isfinite(rel) else np.inf,
                nll if np.isfinite(nll) else np.inf,
            )
        selected_theta, selected_status = min(attempts, key=failure_key)

    selected_status = dict(selected_status)
    attempt_summaries = []
    for _, status in attempts:
        attempt_summaries.append(
            {
                "Label": str(status.get("Spatial_Attempt_Label", "")),
                "Converged": bool(status.get("Converged", False)),
                "Optimizer_Success": bool(status.get("Optimizer_Success", False)),
                "Status_Code": status.get("Status_Code"),
                "Negative_LogLikelihood": status.get("Negative_LogLikelihood"),
                "Gradient_Infinity_Norm": status.get("Gradient_Infinity_Norm"),
                "Relative_Gradient": status.get("Relative_Gradient"),
                "Parameter_Bound_Hit": bool(status.get("Parameter_Bound_Hit", True)),
                "Accepted_By_Spatial_Rule": optimizer_passes_spatial_rule(status),
                "Message": str(status.get("Message", "")),
            }
        )
    selected_status["Spatial_Rescue_Used"] = bool(len(attempts) > 1)
    selected_status["Spatial_Initial_Internal_Attempt_Count"] = internal_initial_attempts
    selected_status["Spatial_Explicit_Attempt_Count"] = len(attempts)
    selected_status["Spatial_Total_Optimization_Attempt_Count"] = (
        internal_initial_attempts + max(len(attempts) - 1, 0)
    )
    selected_status["Spatial_Accepted_Attempt_Count"] = len(accepted)
    selected_status["Spatial_Selected_Attempt_Label"] = str(
        selected_status.get("Spatial_Attempt_Label", "")
    )
    selected_status["Spatial_Attempt_Summary_JSON"] = json.dumps(
        attempt_summaries, ensure_ascii=False
    )
    return np.asarray(selected_theta, dtype=float), selected_status


def stable_spatial_predictions(
    step06c: object,
    theta: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    exposure: np.ndarray,
    model: str,
) -> Dict[str, np.ndarray]:
    """Return stable NB1/ZINB1 predictions, including P(Y > 0).

    The earlier implementation obtained P(Y > 0) as ``1 - P(Y=0)``.
    When P(Y=0) rounded to exactly 1.0, that subtraction lost all precision
    and produced an artificial near-zero denominator. The release implementation derives the NB1
    positive probability directly as ``-expm1(log P_NB1(Y=0))``.

    For ZINB1,
        P(Y > 0) = P(not structural zero) * P_NB1(Y > 0),
    while
        E(Y | Y > 0) = mu / P_NB1(Y > 0).
    The structural-zero probability cancels from the conditional mean, so no
    division by an underflowed mixture probability is required.
    """
    theta = np.asarray(theta, dtype=float)
    y = np.asarray(y, dtype=float)
    exposure = np.asarray(exposure, dtype=float)
    kx = x.shape[1]
    tiny = np.finfo(float).tiny
    log_tiny = float(np.log(tiny))

    if model == "Single_Stage_NB1":
        alpha, mu, log_probability, _, _ = step06c.nb1_parts(
            y, x, exposure, theta[:kx], theta[-1]
        )
        nb_log_zero = -(mu / alpha) * np.log1p(alpha)
        nb_positive = -np.expm1(np.minimum(nb_log_zero, 0.0))
        nb_positive = np.clip(nb_positive, tiny, 1.0)
        positive_probability = nb_positive
        zero_probability = np.clip(np.exp(nb_log_zero), 0.0, 1.0)
        predicted_count = mu
        conditional_positive_count = mu / nb_positive

    elif model == "ZINB1":
        kz = z.shape[1]
        gamma = theta[:kz]
        beta = theta[kz:kz + kx]
        alpha, mu, nb_log_probability, _, _ = step06c.nb1_parts(
            y, x, exposure, beta, theta[-1]
        )
        nb_log_zero = -(mu / alpha) * np.log1p(alpha)
        nb_positive = -np.expm1(np.minimum(nb_log_zero, 0.0))
        nb_positive = np.clip(nb_positive, tiny, 1.0)

        xi = z @ gamma
        log_pi = -np.logaddexp(0.0, -xi)
        log_one_minus_pi = -np.logaddexp(0.0, xi)
        log_positive = log_one_minus_pi + np.log(nb_positive)
        positive_probability = np.exp(np.clip(log_positive, log_tiny, 0.0))
        log_zero = np.logaddexp(log_pi, log_one_minus_pi + nb_log_zero)
        zero_probability = np.clip(np.exp(log_zero), 0.0, 1.0)
        predicted_count = np.exp(log_one_minus_pi) * mu
        conditional_positive_count = mu / nb_positive
        log_probability = np.where(
            y == 0,
            log_zero,
            log_one_minus_pi + nb_log_probability,
        )
    else:
        raise ValueError(f"Unsupported locked model structure: {model}")

    conditional_positive_rate = conditional_positive_count / exposure
    arrays = {
        "Predicted_Count": np.asarray(predicted_count, dtype=float),
        "Predicted_Zero_Probability": np.asarray(zero_probability, dtype=float),
        "Predicted_Positive_Probability": np.asarray(positive_probability, dtype=float),
        "Predicted_Conditional_Positive_Count": np.asarray(
            conditional_positive_count, dtype=float
        ),
        "Predicted_Conditional_Positive_Rate": np.asarray(
            conditional_positive_rate, dtype=float
        ),
        "Predictive_LogProbability": np.asarray(log_probability, dtype=float),
        "NB1_Positive_Probability": np.asarray(nb_positive, dtype=float),
    }

    for name, values in arrays.items():
        if not np.isfinite(values).all():
            raise FloatingPointError(
                f"Stable spatial prediction field {name} contains non-finite values."
            )
    if (arrays["Predicted_Count"] < 0).any():
        raise FloatingPointError("Spatial-fold predictions contain negative counts.")
    for name in [
        "Predicted_Zero_Probability",
        "Predicted_Positive_Probability",
        "NB1_Positive_Probability",
    ]:
        values = arrays[name]
        if ((values < 0) | (values > 1)).any():
            raise FloatingPointError(f"{name} falls outside [0, 1].")
    if (arrays["Predicted_Conditional_Positive_Count"] < 0).any():
        raise FloatingPointError(
            "Conditional-positive predictions contain negative counts."
        )
    return arrays


def build_spatial_prediction_frame(
    validation: pd.DataFrame,
    definition: Mapping[str, object],
    fold: int,
    outputs: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    response = str(definition["Count_Response"])
    exposure = validation["ForestPixelCount"].to_numpy(dtype=float)
    observed = validation[response].to_numpy(dtype=float)
    predicted_count = np.asarray(outputs["Predicted_Count"], dtype=float)

    prediction = validation[
        [
            "GRID_UID",
            "Country",
            "GRID_ID",
            "Year",
            "Season",
            "Temporal_Fold",
            "Spatial_Block",
            "Centroid_X_UTM52_m",
            "Centroid_Y_UTM52_m",
            "Longitude",
            "Latitude",
            "ForestPixelCount",
        ]
    ].copy()
    prediction["Season_Label"] = str(definition["Season_Label"])
    prediction["Validation_Scheme"] = "Spatial"
    prediction["Validation_Fold"] = int(fold)
    prediction["Count_Response"] = response
    prediction["Rate_Scale_Name"] = str(definition["Rate_Scale_Name"])
    prediction["Model_Role"] = str(definition["Model_Role"])
    prediction["Model_ID"] = str(definition["Model_ID"])
    prediction["Model_Structure"] = str(definition["Model_Structure"])
    prediction["Candidate_ID"] = str(definition["Candidate_ID"])
    prediction["Transformation_Scheme"] = str(
        definition["Transformation_Scheme"]
    )
    prediction["Observed_Count"] = observed
    prediction["Predicted_Count"] = predicted_count
    prediction["Observed_Rate"] = observed / exposure
    prediction["Predicted_Rate"] = predicted_count / exposure
    prediction["Observed_Zero"] = (observed == 0).astype(int)
    prediction["Predicted_Zero_Probability"] = outputs[
        "Predicted_Zero_Probability"
    ]
    prediction["Predicted_Positive_Probability"] = outputs[
        "Predicted_Positive_Probability"
    ]
    prediction["Predicted_Conditional_Positive_Count"] = outputs[
        "Predicted_Conditional_Positive_Count"
    ]
    prediction["Predicted_Conditional_Positive_Rate"] = outputs[
        "Predicted_Conditional_Positive_Rate"
    ]
    prediction["Predictive_LogProbability"] = outputs[
        "Predictive_LogProbability"
    ]
    prediction["Residual_Count"] = observed - predicted_count
    prediction["Residual_Rate"] = (
        prediction["Observed_Rate"] - prediction["Predicted_Rate"]
    )
    return prediction[PREDICTION_COLUMNS].copy()


def fit_spatial_fold(
    step04: object,
    step05: object,
    step06c: object,
    data: pd.DataFrame,
    definition: Mapping[str, object],
    fold: int,
    spatial_mapping_sha256_value: str,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    prediction_path = fold_prediction_path(definition, fold)
    parameter_npz, parameter_json = fold_parameter_paths(definition, fold)

    training = data.loc[data["Spatial_Block"] != fold].copy()
    validation = data.loc[data["Spatial_Block"] == fold].copy()
    if training.empty or validation.empty:
        raise ValueError(
            f"Empty training or validation sample in spatial fold {fold}."
        )

    components = definition_components(definition)
    count_predictors = components["count_predictors"]
    count_country_terms = components["count_country_terms"]
    zero_predictors = components["zero_predictors"]
    zero_country_terms = components["zero_country_terms"]
    scheme = str(components["transformation_scheme"])
    model = str(components["model_structure"])
    response = str(definition["Count_Response"])

    x_training, x_validation, x_names = step06c.train_valid_design(
        training,
        validation,
        count_predictors,
        scheme,
        count_country_terms,
    )
    if model == "ZINB1":
        z_training, z_validation, z_names = step06c.train_valid_design(
            training,
            validation,
            zero_predictors,
            scheme,
            zero_country_terms,
        )
    elif model == "Single_Stage_NB1":
        z_training = np.empty((len(training), 0), dtype=float)
        z_validation = np.empty((len(validation), 0), dtype=float)
        z_names = []
    else:
        raise ValueError(f"Unsupported locked model structure: {model}")

    x_rank = int(np.linalg.matrix_rank(x_training))
    z_rank = int(np.linalg.matrix_rank(z_training)) if model == "ZINB1" else 0
    if x_rank != x_training.shape[1]:
        raise ValueError(
            f"Count design is rank deficient for {definition['Model_Role']} "
            f"{definition['Season_Label']} {response}, fold {fold}."
        )
    if model == "ZINB1" and z_rank != z_training.shape[1]:
        raise ValueError(
            f"Structural-zero design is rank deficient for "
            f"{definition['Model_Role']} {definition['Season_Label']} "
            f"{response}, fold {fold}."
        )

    expected_keys = set(
        map(
            tuple,
            validation[["GRID_UID", "Year", "Season"]].itertuples(
                index=False, name=None
            ),
        )
    )
    checkpoint_status: Dict[str, object] | None = None
    checkpoint_theta: np.ndarray | None = None

    if prediction_path.exists() and parameter_npz.exists() and parameter_json.exists():
        old_prediction = pd.read_csv(prediction_path, compression="gzip")
        status = json.loads(parameter_json.read_text(encoding="utf-8"))
        missing_checkpoint_columns = sorted(
            set(PREDICTION_COLUMNS) - set(old_prediction.columns)
        )
        signature_matches = (
            str(status.get("Definition_Signature", ""))
            == str(definition["Definition_Signature"])
        )
        design_matches = (
            str(status.get("Spatial_Block_Design_ID", ""))
            == SPATIAL_BLOCK_DESIGN_ID
        )
        mapping_matches = (
            str(status.get("Spatial_Mapping_SHA256", ""))
            == spatial_mapping_sha256_value
        )
        fit_protocol_matches = (
            str(status.get("Spatial_Fit_Protocol_ID", ""))
            == SPATIAL_FIT_PROTOCOL_ID
        )
        accepted_fit = bool(status.get("Fold_Fit_Accepted", False))
        observed_keys = set(
            map(
                tuple,
                old_prediction[["GRID_UID", "Year", "Season"]].itertuples(
                    index=False, name=None
                ),
            )
        ) if not missing_checkpoint_columns else set()
        checkpoint_integrity_ok = bool(
            not missing_checkpoint_columns
            and signature_matches
            and design_matches
            and mapping_matches
            and fit_protocol_matches
            and accepted_fit
            and len(old_prediction) == len(validation)
            and set(old_prediction["Validation_Fold"].astype(int)) == {fold}
            and observed_keys == expected_keys
        )

        if checkpoint_integrity_ok:
            with np.load(parameter_npz) as loaded:
                if "theta" not in loaded:
                    raise ValueError(f"theta is missing from {parameter_npz.name}.")
                checkpoint_theta = np.asarray(loaded["theta"], dtype=float)
            if not np.isfinite(checkpoint_theta).all():
                raise ValueError(f"Non-finite theta in {parameter_npz.name}.")
            checkpoint_status = dict(status)

            prediction_protocol_matches = (
                str(status.get("Spatial_Prediction_Protocol_ID", ""))
                == SPATIAL_PREDICTION_PROTOCOL_ID
            )
            if prediction_protocol_matches:
                checkpoint_status["Checkpoint_Reused"] = True
                checkpoint_status["Prediction_Checkpoint_Refreshed"] = False
                log(
                    f"Reused accepted checkpoint: {definition['Model_Role']} "
                    f"{definition['Season_Label']} {response} fold {fold}."
                )
                return old_prediction[PREDICTION_COLUMNS].copy(), checkpoint_status
            log(
                f"Refreshing predictions from accepted parameters: "
                f"{definition['Model_Role']} {definition['Season_Label']} "
                f"{response} fold {fold}."
            )
        else:
            log(
                f"Ignored incompatible checkpoint and refitting: "
                f"{prediction_path.name}."
            )
            prediction_path.unlink(missing_ok=True)
            parameter_npz.unlink(missing_ok=True)
            parameter_json.unlink(missing_ok=True)

    y_training = training[response].to_numpy(dtype=float)
    exposure_training = training["ForestPixelCount"].to_numpy(dtype=float)
    reused_accepted_fit = checkpoint_theta is not None and checkpoint_status is not None

    if reused_accepted_fit:
        theta = np.asarray(checkpoint_theta, dtype=float)
        optimizer_status = dict(checkpoint_status)
    else:
        training_start = step06c.training_start(
            step04,
            step05,
            y_training,
            x_training,
            z_training,
            exposure_training,
            model,
        )
        neutral_start = step06c.neutral_start(
            y_training,
            x_training,
            z_training,
            exposure_training,
            model,
        )
        theta, optimizer_status = optimize_spatial_fold_with_rescue(
            step06c,
            training_start,
            neutral_start,
            y_training,
            x_training,
            z_training,
            exposure_training,
            model,
        )

    y_validation = validation[response].to_numpy(dtype=float)
    exposure_validation = validation["ForestPixelCount"].to_numpy(dtype=float)
    outputs = stable_spatial_predictions(
        step06c,
        theta,
        y_validation,
        x_validation,
        z_validation,
        exposure_validation,
        model,
    )
    prediction = build_spatial_prediction_frame(
        validation=validation,
        definition=definition,
        fold=fold,
        outputs=outputs,
    )

    fold_status: Dict[str, object] = {
        **optimizer_status,
        "Model_Role": str(definition["Model_Role"]),
        "Model_ID": str(definition["Model_ID"]),
        "Season_Code": int(definition["Season_Code"]),
        "Season_Label": str(definition["Season_Label"]),
        "Count_Response": response,
        "Rate_Scale_Name": str(definition["Rate_Scale_Name"]),
        "Model_Structure": model,
        "Candidate_ID": str(definition["Candidate_ID"]),
        "Definition_Signature": str(definition["Definition_Signature"]),
        "Spatial_Block_Design_ID": SPATIAL_BLOCK_DESIGN_ID,
        "Spatial_Mapping_SHA256": spatial_mapping_sha256_value,
        "Step07_Code_Version": STEP07_CODE_VERSION,
        "Spatial_Fit_Protocol_ID": SPATIAL_FIT_PROTOCOL_ID,
        "Spatial_Prediction_Protocol_ID": SPATIAL_PREDICTION_PROTOCOL_ID,
        "Spatial_Relative_Gradient_Max": SPATIAL_RELATIVE_GRADIENT_MAX,
        "Transformation_Scheme": scheme,
        "Spatial_Fold": int(fold),
        "Training_Rows": int(len(training)),
        "Validation_Rows": int(len(validation)),
        "Training_GRID_UID_Count": int(training["GRID_UID"].nunique()),
        "Validation_GRID_UID_Count": int(validation["GRID_UID"].nunique()),
        "Training_Positive_Rows": int((y_training > 0).sum()),
        "Validation_Positive_Rows": int((y_validation > 0).sum()),
        "Count_Design_Columns": int(x_training.shape[1]),
        "Count_Design_Rank": x_rank,
        "Count_Design_Full_Rank": bool(x_rank == x_training.shape[1]),
        "Structural_Zero_Design_Columns": int(z_training.shape[1]),
        "Structural_Zero_Design_Rank": z_rank,
        "Structural_Zero_Design_Full_Rank": bool(
            model != "ZINB1" or z_rank == z_training.shape[1]
        ),
        "Count_Parameter_Names": ";".join(x_names),
        "Structural_Zero_Parameter_Names": ";".join(z_names),
        "Checkpoint_Reused": bool(reused_accepted_fit),
        "Prediction_Checkpoint_Refreshed": bool(reused_accepted_fit),
        "Minimum_Predicted_Positive_Probability": float(
            np.min(outputs["Predicted_Positive_Probability"])
        ),
        "Minimum_NB1_Positive_Probability": float(
            np.min(outputs["NB1_Positive_Probability"])
        ),
        "Maximum_Conditional_Positive_Count": float(
            np.max(outputs["Predicted_Conditional_Positive_Count"])
        ),
        "Maximum_Conditional_Positive_Rate": float(
            np.max(outputs["Predicted_Conditional_Positive_Rate"])
        ),
    }
    predictive_log_probability = np.asarray(
        outputs["Predictive_LogProbability"], dtype=float
    )
    fold_status["Validation_Total_Predictive_LogLikelihood"] = float(
        predictive_log_probability.sum()
    )
    fold_status["Validation_Mean_Negative_LogLikelihood"] = float(
        -predictive_log_probability.mean()
    )
    fold_status["Spatial_Standardized_Score_Acceptable"] = bool(
        np.isfinite(float(fold_status.get("Relative_Gradient", np.nan)))
        and float(fold_status.get("Relative_Gradient", np.nan))
        <= SPATIAL_RELATIVE_GRADIENT_MAX
    )
    fold_status["Fold_Fit_Accepted"] = bool(
        fold_status.get("Converged", False)
        and fold_status["Spatial_Standardized_Score_Acceptable"]
        and not fold_status.get("Parameter_Bound_Hit", True)
        and fold_status["Count_Design_Full_Rank"]
        and fold_status["Structural_Zero_Design_Full_Rank"]
    )

    if not fold_status["Fold_Fit_Accepted"]:
        FAILED_FIT_ROOT.mkdir(parents=True, exist_ok=True)
        failed_prefix = checkpoint_prefix(definition, fold)
        failed_json = FAILED_FIT_ROOT / f"FAILED_{failed_prefix}.json"
        failed_npz = FAILED_FIT_ROOT / f"FAILED_{failed_prefix}.npz"
        np.savez_compressed(failed_npz, theta=np.asarray(theta, dtype=float))
        failed_json.write_text(
            json.dumps(fold_status, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        raise RuntimeError(
            f"The spatial-fold fit did not satisfy the locked Step-06d "
            f"stability rules: {definition['Model_Role']} "
            f"{definition['Season_Label']} {response}, fold {fold}. "
            f"Diagnostics were saved to {failed_json}."
        )

    prediction.to_csv(prediction_path, index=False, compression="gzip")
    if not parameter_npz.exists() or not reused_accepted_fit:
        np.savez_compressed(parameter_npz, theta=np.asarray(theta, dtype=float))
    parameter_json.write_text(
        json.dumps(fold_status, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    action = "Refreshed predictions" if reused_accepted_fit else "Completed"
    log(
        f"{action}: {definition['Model_Role']} "
        f"{definition['Season_Label']} {response} fold {fold}; "
        f"train={len(training):,}, validation={len(validation):,}."
    )
    return prediction, fold_status


# =============================================================================
# Metrics and output assembly
# =============================================================================


def calculate_metrics(
    step02: object,
    prediction: pd.DataFrame,
) -> Dict[str, float]:
    observed_count = prediction["Observed_Count"].to_numpy(dtype=float)
    predicted_count = prediction["Predicted_Count"].to_numpy(dtype=float)
    exposure = prediction["ForestPixelCount"].to_numpy(dtype=float)
    predicted_zero_probability = prediction[
        "Predicted_Zero_Probability"
    ].to_numpy(dtype=float)
    log_probability = prediction["Predictive_LogProbability"].to_numpy(
        dtype=float
    )

    result = dict(
        step02.prediction_metrics(
            observed_count=observed_count,
            predicted_count=predicted_count,
            exposure=exposure,
            predicted_zero_probability=predicted_zero_probability,
            log_probability=log_probability,
        )
    )

    observed_rate = observed_count / exposure
    predicted_rate = predicted_count / exposure
    conditional_positive_count = prediction[
        "Predicted_Conditional_Positive_Count"
    ].to_numpy(dtype=float)
    conditional_positive_rate = prediction[
        "Predicted_Conditional_Positive_Rate"
    ].to_numpy(dtype=float)

    result.update(
        {
            "Count_Mean_Bias": float(np.mean(predicted_count - observed_count)),
            "Rate_Mean_Bias": float(np.mean(predicted_rate - observed_rate)),
            "Count_Spearman": safe_spearman(observed_count, predicted_count),
            "Rate_Spearman": safe_spearman(observed_rate, predicted_rate),
            "Predicted_to_Observed_Mean_Rate_Ratio": (
                float(predicted_rate.mean() / observed_rate.mean())
                if observed_rate.mean() > 0
                else np.nan
            ),
        }
    )

    positive = observed_count > 0
    result["Conditional_Positive_Observed_N"] = int(positive.sum())

    if positive.any():
        positive_count_intercept, positive_count_slope = calibration_parameters(
            observed_count[positive], conditional_positive_count[positive]
        )
        positive_rate_intercept, positive_rate_slope = calibration_parameters(
            observed_rate[positive], conditional_positive_rate[positive]
        )

        result.update(
            {
                "Conditional_Positive_Count_RMSE": safe_rmse(
                    observed_count[positive], conditional_positive_count[positive]
                ),
                "Conditional_Positive_Count_MAE": float(
                    mean_absolute_error(
                        observed_count[positive], conditional_positive_count[positive]
                    )
                ),
                "Conditional_Positive_Count_R2": safe_r2(
                    observed_count[positive], conditional_positive_count[positive]
                ),
                "Conditional_Positive_Count_Spearman": safe_spearman(
                    observed_count[positive], conditional_positive_count[positive]
                ),
                "Conditional_Positive_Count_Calibration_Intercept": (
                    positive_count_intercept
                ),
                "Conditional_Positive_Count_Calibration_Slope": positive_count_slope,
                "Conditional_Positive_Rate_RMSE": safe_rmse(
                    observed_rate[positive], conditional_positive_rate[positive]
                ),
                "Conditional_Positive_Rate_MAE": float(
                    mean_absolute_error(
                        observed_rate[positive], conditional_positive_rate[positive]
                    )
                ),
                "Conditional_Positive_Rate_R2": safe_r2(
                    observed_rate[positive], conditional_positive_rate[positive]
                ),
                "Conditional_Positive_Rate_Spearman": safe_spearman(
                    observed_rate[positive], conditional_positive_rate[positive]
                ),
                "Conditional_Positive_Rate_Calibration_Intercept": positive_rate_intercept,
                "Conditional_Positive_Rate_Calibration_Slope": positive_rate_slope,
            }
        )
    else:
        for name in [
            "Conditional_Positive_Count_RMSE",
            "Conditional_Positive_Count_MAE",
            "Conditional_Positive_Count_R2",
            "Conditional_Positive_Count_Spearman",
            "Conditional_Positive_Count_Calibration_Intercept",
            "Conditional_Positive_Count_Calibration_Slope",
            "Conditional_Positive_Rate_RMSE",
            "Conditional_Positive_Rate_MAE",
            "Conditional_Positive_Rate_R2",
            "Conditional_Positive_Rate_Spearman",
            "Conditional_Positive_Rate_Calibration_Intercept",
            "Conditional_Positive_Rate_Calibration_Slope",
        ]:
            result[name] = np.nan

    return result


def write_combined_oof(
    definitions: pd.DataFrame,
    model_role: str,
    output_path: Path,
) -> Dict[str, object]:
    selected = definitions.loc[
        definitions["Model_Role"].astype(str) == model_role
    ].sort_values(["Season_Code", "Count_Response"])

    total_rows = 0
    header_written = False
    key_tracker: set[Tuple[str, int, int, str]] = set()

    with gzip.open(output_path, "wt", encoding="utf-8", newline="") as handle:
        for definition in selected.to_dict(orient="records"):
            for fold in SPATIAL_FOLDS:
                path = fold_prediction_path(definition, fold)
                frame = pd.read_csv(path, compression="gzip")
                frame = frame[PREDICTION_COLUMNS]

                keys = set(
                    (
                        str(row.GRID_UID),
                        int(row.Year),
                        int(row.Season),
                        str(row.Count_Response),
                    )
                    for row in frame.itertuples(index=False)
                )
                overlap = key_tracker.intersection(keys)
                if overlap:
                    raise ValueError(
                        f"Duplicate OOF keys encountered while assembling {model_role}."
                    )
                key_tracker.update(keys)

                frame.to_csv(
                    handle,
                    index=False,
                    header=not header_written,
                )
                header_written = True
                total_rows += len(frame)

    if total_rows != EXPECTED_ROWS_PER_MODEL_ROLE:
        raise ValueError(
            f"Expected {EXPECTED_ROWS_PER_MODEL_ROLE:,} rows for {model_role}, "
            f"but wrote {total_rows:,}."
        )

    return {
        "Model_Role": model_role,
        "Output_File": output_path.name,
        "Rows": total_rows,
        "SHA256": sha256_file(output_path),
        "Unique_OOF_Keys": len(key_tracker),
    }


def pooled_comparison_table(pooled: pd.DataFrame) -> pd.DataFrame:
    key = [
        "Season_Code",
        "Season_Label",
        "Count_Response",
        "Rate_Scale_Name",
    ]

    selected_metrics = [
        "Mean_Negative_LogLikelihood",
        "Rate_RMSE",
        "Rate_MAE",
        "Rate_R2",
        "Rate_Spearman",
        "Rate_Mean_Bias",
        "Predicted_to_Observed_Mean_Rate_Ratio",
        "Zero_Probability_Brier",
        "Occurrence_ROC_AUC",
        "Occurrence_PR_AUC",
        "Conditional_Positive_Rate_RMSE",
        "Conditional_Positive_Rate_MAE",
        "Conditional_Positive_Rate_R2",
        "Conditional_Positive_Rate_Spearman",
    ]

    baseline = pooled.loc[
        pooled["Model_Role"] == "Baseline_NB1",
        [*key, *selected_metrics],
    ].copy()
    final = pooled.loc[
        pooled["Model_Role"] == "Final_Regression",
        [*key, *selected_metrics],
    ].copy()

    baseline = baseline.rename(
        columns={name: f"Baseline_{name}" for name in selected_metrics}
    )
    final = final.rename(
        columns={name: f"Final_{name}" for name in selected_metrics}
    )

    result = baseline.merge(final, on=key, how="inner", validate="one_to_one")
    for name in selected_metrics:
        result[f"Final_minus_Baseline_{name}"] = (
            result[f"Final_{name}"] - result[f"Baseline_{name}"]
        )

    result["Interpretation_Note"] = (
        "Descriptive spatial-OOF comparison only; final cross-model synthesis "
        "is reserved for Step 10."
    )
    return result


# =============================================================================
# Main workflow
# =============================================================================


def main() -> int:
    for path in [
        OUTPUT_ROOT,
        CHECKPOINT_ROOT,
        FOLD_PREDICTION_ROOT,
        FOLD_PARAMETER_ROOT,
        FAILED_FIT_ROOT,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    LOG_FILE.write_text("", encoding="utf-8")
    log("Starting Step 07 shared spatial-block generation and regression validation.")
    log(f"Step 07 code version: {STEP07_CODE_VERSION}.")
    log(f"Spatial prediction protocol: {SPATIAL_PREDICTION_PROTOCOL_ID}.")
    log(
        "Accepted parameter checkpoints will be reused; only stable "
        "probabilities, conditional-positive predictions, and summaries are rebuilt."
    )
    log("Spatial blocks are generated inside 08_Regression_modeling; no 09_RF_modeling result is read.")
    log(
        "The locked shared universe is reconstructed from 4,989 forest-eligible "
        "grids by excluding 525 incomplete records from seven fully incomplete "
        "grids, leaving 4,982 grids."
    )

    require_files(
        [
            INPUT_CSV,
            STEP02_SCRIPT,
            STEP04_SCRIPT,
            STEP05_SCRIPT,
            STEP06C_SCRIPT,
            FINAL_STRUCTURE_CSV,
            FINAL_OOF_MANIFEST_CSV,
        ]
    )

    step02 = load_module(STEP02_SCRIPT, "step02_spatial_validation")
    step04 = load_module(STEP04_SCRIPT, "step04_spatial_validation")
    step05 = load_module(STEP05_SCRIPT, "step05_spatial_validation")
    step06c = load_module(STEP06C_SCRIPT, "step06c_spatial_validation")

    blocks, block_hash_metadata = generate_and_verify_spatial_blocks()
    spatial_mapping_hash = str(
        block_hash_metadata["Observed_Semantic_Mapping_SHA256"]
    )
    log(
        "Generated the shared spatial-block assignment in 08 and verified "
        f"semantic SHA256 {spatial_mapping_hash}."
    )

    final_structure = read_final_structure()
    definitions = build_model_definitions(final_structure)
    definitions.to_csv(
        OUTPUT_ROOT / "03_Locked_Model_Definitions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    manifest = pd.read_csv(FINAL_OOF_MANIFEST_CSV)
    required_manifest_columns = {
        "Season_Code",
        "Count_Response",
        "Output_File",
    }
    missing_manifest = sorted(
        required_manifest_columns - set(manifest.columns)
    )
    if missing_manifest:
        raise ValueError(
            "Step-06d OOF manifest is missing columns:\n"
            + "\n".join(missing_manifest)
        )

    base_data = read_base_table(definitions)

    samples: Dict[Tuple[int, str], pd.DataFrame] = {}
    qa_rows: List[Dict[str, object]] = []
    block_summary_frames: List[pd.DataFrame] = []

    for season in SEASONS:
        for response in COUNT_RESPONSES:
            sample = exact_analysis_sample(
                base_data=base_data,
                blocks=blocks,
                manifest=manifest,
                model_definitions=definitions,
                season=season,
                response=response,
            )
            samples[(season, response)] = sample

            qa_rows.append(
                {
                    "Season_Code": season,
                    "Season_Label": SEASONS[season],
                    "Count_Response": response,
                    "Rate_Scale_Name": RATE_NAMES[response],
                    "Rows": int(len(sample)),
                    "GRID_UID_Count": int(sample["GRID_UID"].nunique()),
                    "Year_Count": int(sample["Year"].nunique()),
                    "Minimum_Year": int(sample["Year"].min()),
                    "Maximum_Year": int(sample["Year"].max()),
                    "Country_Count": int(sample["Country"].nunique()),
                    "Spatial_Block_Count": int(
                        sample["Spatial_Block"].nunique()
                    ),
                    "Positive_Rows": int((sample[response] > 0).sum()),
                    "Zero_Rows": int((sample[response] == 0).sum()),
                    "Observed_Total_Count": float(sample[response].sum()),
                    "Duplicate_Panel_Keys": int(
                        sample.duplicated(
                            ["GRID_UID", "Year", "Season"]
                        ).sum()
                    ),
                    "Missing_Spatial_Block_Rows": int(
                        sample["Spatial_Block"].isna().sum()
                    ),
                    "Spatial_Mapping_SHA256_Verified": True,
                }
            )

            summary = (
                sample.groupby(["Spatial_Block", "Country"], as_index=False)
                .agg(
                    Rows=("GRID_UID", "size"),
                    GRID_UID_Count=("GRID_UID", "nunique"),
                    Year_Count=("Year", "nunique"),
                    Positive_Rows=(response, lambda values: int((values > 0).sum())),
                    Observed_Total_Count=(response, "sum"),
                )
            )
            summary.insert(0, "Count_Response", response)
            summary.insert(0, "Season_Label", SEASONS[season])
            summary.insert(0, "Season_Code", season)
            block_summary_frames.append(summary)

    qa = pd.DataFrame(qa_rows)
    qa.to_csv(
        OUTPUT_ROOT / "01_Data_and_Block_QA.csv",
        index=False,
        encoding="utf-8-sig",
    )

    block_summary = pd.concat(block_summary_frames, ignore_index=True)
    block_summary.to_csv(
        OUTPUT_ROOT / "02_Spatial_Block_Sample_Summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fit_status_rows: List[Dict[str, object]] = []
    fold_metric_rows: List[Dict[str, object]] = []

    for definition in definitions.sort_values(
        ["Model_Role", "Season_Code", "Count_Response"]
    ).to_dict(orient="records"):
        season = int(definition["Season_Code"])
        response = str(definition["Count_Response"])
        sample = samples[(season, response)]

        for fold in SPATIAL_FOLDS:
            prediction, fit_status = fit_spatial_fold(
                step04=step04,
                step05=step05,
                step06c=step06c,
                data=sample,
                definition=definition,
                fold=fold,
                spatial_mapping_sha256_value=spatial_mapping_hash,
            )
            fit_status_rows.append(fit_status)

            fold_metric_rows.append(
                {
                    "Validation_Scheme": "Spatial",
                    "Model_Role": str(definition["Model_Role"]),
                    "Model_ID": str(definition["Model_ID"]),
                    "Season_Code": season,
                    "Season_Label": str(definition["Season_Label"]),
                    "Count_Response": response,
                    "Rate_Scale_Name": str(definition["Rate_Scale_Name"]),
                    "Model_Structure": str(definition["Model_Structure"]),
                    "Candidate_ID": str(definition["Candidate_ID"]),
                    "Definition_Signature": str(
                        definition["Definition_Signature"]
                    ),
                    "Spatial_Fold": fold,
                    "Training_Rows": int((sample["Spatial_Block"] != fold).sum()),
                    "Validation_Rows": int((sample["Spatial_Block"] == fold).sum()),
                    "Training_GRID_UID_Count": int(
                        sample.loc[
                            sample["Spatial_Block"] != fold, "GRID_UID"
                        ].nunique()
                    ),
                    "Validation_GRID_UID_Count": int(
                        sample.loc[
                            sample["Spatial_Block"] == fold, "GRID_UID"
                        ].nunique()
                    ),
                    **calculate_metrics(step02, prediction),
                }
            )

    fit_status = pd.DataFrame(fit_status_rows)
    fit_status.to_csv(
        OUTPUT_ROOT / "04_Spatial_Fold_Fit_Status.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fold_metrics = pd.DataFrame(fold_metric_rows)
    fold_metrics.to_csv(
        OUTPUT_ROOT / "05_Spatial_Fold_Metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pooled_rows: List[Dict[str, object]] = []
    for definition in definitions.sort_values(
        ["Model_Role", "Season_Code", "Count_Response"]
    ).to_dict(orient="records"):
        frames = [
            pd.read_csv(
                fold_prediction_path(definition, fold),
                compression="gzip",
            )
            for fold in SPATIAL_FOLDS
        ]
        prediction = pd.concat(frames, ignore_index=True)

        if len(prediction) != EXPECTED_SEASON_RESPONSE_ROWS:
            raise ValueError(
                f"Pooled OOF row count mismatch for {definition['Model_Role']} "
                f"{definition['Season_Label']} {definition['Count_Response']}."
            )
        if prediction.duplicated(
            ["GRID_UID", "Year", "Season", "Count_Response"],
            keep=False,
        ).any():
            raise ValueError("Duplicate pooled spatial OOF keys were found.")

        pooled_rows.append(
            {
                "Validation_Scheme": "Spatial",
                "Model_Role": str(definition["Model_Role"]),
                "Model_ID": str(definition["Model_ID"]),
                "Season_Code": int(definition["Season_Code"]),
                "Season_Label": str(definition["Season_Label"]),
                "Count_Response": str(definition["Count_Response"]),
                "Rate_Scale_Name": str(definition["Rate_Scale_Name"]),
                "Model_Structure": str(definition["Model_Structure"]),
                "Candidate_ID": str(definition["Candidate_ID"]),
                "Definition_Signature": str(
                    definition["Definition_Signature"]
                ),
                "Spatial_Fold_Count": int(
                    prediction["Validation_Fold"].nunique()
                ),
                **calculate_metrics(step02, prediction),
            }
        )

    pooled_metrics = pd.DataFrame(pooled_rows)
    pooled_metrics.to_csv(
        OUTPUT_ROOT / "06_Pooled_Spatial_OOF_Metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    comparison = pooled_comparison_table(pooled_metrics)
    comparison.to_csv(
        OUTPUT_ROOT / "07_Baseline_vs_Final_Spatial_Comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    output_manifest_rows = [
        write_combined_oof(
            definitions=definitions,
            model_role="Baseline_NB1",
            output_path=BASELINE_OOF_FILE,
        ),
        write_combined_oof(
            definitions=definitions,
            model_role="Final_Regression",
            output_path=FINAL_OOF_FILE,
        ),
    ]
    output_manifest = pd.DataFrame(output_manifest_rows)
    output_manifest["Validation_Scheme"] = "Spatial"
    output_manifest["Spatial_Block_Design_ID"] = SPATIAL_BLOCK_DESIGN_ID
    output_manifest["Spatial_Fit_Protocol_ID"] = SPATIAL_FIT_PROTOCOL_ID
    output_manifest["Spatial_Prediction_Protocol_ID"] = (
        SPATIAL_PREDICTION_PROTOCOL_ID
    )
    output_manifest["Spatial_Mapping_SHA256"] = spatial_mapping_hash
    output_manifest["Spatial_Block_File_SHA256"] = sha256_file(
        SPATIAL_BLOCK_FILE
    )
    output_manifest.to_csv(
        OUTPUT_ROOT / "08_Spatial_OOF_Manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )

    method_definition = {
        "Step": "07_Regression_Spatial_Block_Validation",
        "Created": datetime.now().isoformat(timespec="seconds"),
        "Purpose": (
            "Five-fold spatial OOF validation for the locked Baseline_NB1 "
            "and Final_Regression models."
        ),
        "No_Model_Selection_Performed": True,
        "Step07_Code_Version": STEP07_CODE_VERSION,
        "Spatial_Fit_Protocol_ID": SPATIAL_FIT_PROTOCOL_ID,
        "Spatial_Prediction_Protocol_ID": SPATIAL_PREDICTION_PROTOCOL_ID,
        "Stable_Positive_Probability_Method": (
            "NB1 P(Y>0) = -expm1(log P_NB1(Y=0)); ZINB1 P(Y>0) "
            "= (1-pi) * P_NB1(Y>0); E(Y|Y>0) = mu / P_NB1(Y>0)"
        ),
        "Spatial_Relative_Gradient_Max": SPATIAL_RELATIVE_GRADIENT_MAX,
        "Spatial_Validation": {
            "Fold_Count": 5,
            "Training_Blocks_Per_Fold": 4,
            "Validation_Blocks_Per_Fold": 1,
            "GRID_UID_Held_Together_Across_Years": True,
            "Assignment_Generated_Here": True,
            "Later_RF_Steps_Must_Read_This_Assignment": True,
            **block_hash_metadata,
        },
        "Baseline_NB1": {
            "Model_Structure": "Single_Stage_NB1",
            "Count_Predictors": BASELINE_CONTINUOUS_PREDICTORS,
            "Country_Fixed_Effects": COUNTRY_TERMS,
            "Transformation_Scheme": BASELINE_TRANSFORMATION_SCHEME,
            "Log1p_Predictors": [
                "LtgProxy",
                "POP",
                "Dis_Farm",
                "Road_dens",
            ],
            "Definition_Source": "Step 02",
            "Likelihood_Implementation": (
                "Step-06c explicit stable NB1 likelihood"
            ),
        },
        "Final_Regression": {
            "Definition_Source": str(FINAL_STRUCTURE_CSV),
            "Structures": final_structure.to_dict(orient="records"),
            "Likelihood_Implementation": (
                "Step-06c explicit stable NB1/ZINB1 likelihood"
            ),
        },
        "Exposure": "ForestPixelCount",
        "Offset_Equivalent": "log(ForestPixelCount)",
        "Analysis_Sample_Source": (
            "Exact row keys and temporal-fold labels from Step-06d final OOF files"
        ),
        "Expected_Rows_Per_Season_Response": EXPECTED_SEASON_RESPONSE_ROWS,
        "Expected_Grid_Count": EXPECTED_GRID_COUNT,
        "Fit_Count": {
            "Baseline_NB1": 30,
            "Final_Regression": 30,
            "Total": 60,
        },
        "Software": {
            "Python": platform.python_version(),
            "NumPy": np.__version__,
            "pandas": pd.__version__,
            "SciPy": scipy.__version__,
            "scikit-learn": sklearn.__version__,
            "statsmodels": statsmodels.__version__,
        },
    }
    (OUTPUT_ROOT / "00_Method_Definition.json").write_text(
        json.dumps(
            method_definition,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    software_environment = {
        "Python": platform.python_version(),
        "Platform": platform.platform(),
        "NumPy": np.__version__,
        "pandas": pd.__version__,
        "SciPy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
        "statsmodels": statsmodels.__version__,
    }
    (OUTPUT_ROOT / "Software_Environment.json").write_text(
        json.dumps(software_environment, indent=2),
        encoding="utf-8",
    )

    log(
        "Step 07 completed: the shared spatial blocks were generated in 08, "
        "then 30 baseline and 30 final-regression spatial-fold results were "
        "assembled into two aligned OOF files using stable positive-probability "
        "post-processing."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
