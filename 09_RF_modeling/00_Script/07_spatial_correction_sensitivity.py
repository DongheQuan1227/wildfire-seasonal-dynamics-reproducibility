# -*- coding: utf-8 -*-

# Public reproducibility version.
# Input paths are resolved relative to the code repository.
# ForestPixelCount stores the original pixel count; ForestArea_km2 stores physical area.


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
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import sparse
from scipy.spatial import cKDTree, distance
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


# =============================================================================
# Paths and settings
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_ROOT = SCRIPT_DIR.parent
CODE_ROOT = MODEL_ROOT.parent
INPUT_CSV = CODE_ROOT / "00_Input_data" / "Base_Table_2001_2025.csv.bz2"


# Shared spatial-block dependency. The assignment is generated once upstream
# by 08_Regression_modeling Step 07 and must never be regenerated inside 09.
SHARED_SPATIAL_BLOCK_FILE = (
    CODE_ROOT
    / "08_Regression_modeling"
    / "07_Regression_Spatial_Block_Validation"
    / "Spatial_Block_Assignment.csv"
)
SHARED_SPATIAL_BLOCK_WORKFLOW_VERSION = (
    "2026-06-30_RF_READS_08_STEP07_SHARED_BLOCKS_V1"
)
EXPECTED_SPATIAL_GRID_COUNT = 4_982
EXPECTED_SPATIAL_BLOCK_GRID_COUNTS = {
    1: 912,
    2: 1_042,
    3: 735,
    4: 1_222,
    5: 1_071,
}
EXPECTED_SPATIAL_MAPPING_SHA256 = (
    "715f9fe342eec78d936d223a98f5a4d0378281d3a3fa420ed250ba07fe056cfe"
)
EXPECTED_SPATIAL_BLOCK_FILE_SHA256 = (
    "da752eff8f734d29e0ff7b5535c93277b4b146b0b5a84e27efe3f08d17f61a11"
)
OUTPUT_ROOT = MODEL_ROOT / "07_Spatial_Correction_Sensitivity"
LOG_FILE = OUTPUT_ROOT / "spatial_correction_sensitivity.log"

# ForestPixelCount is the number of 500 m x 500 m forest pixels.
# ForestArea_km2 is ForestPixelCount * 0.25; models use ForestArea_km2 > 2.5.
FOREST_AREA_THRESHOLD_KM2 = 2.5
RANDOM_SEED = 2026
N_ESTIMATORS = 300
SPATIAL_BLOCK_COUNT = 5
RBF_BASIS_COUNT = 12
MORAN_K_NEIGHBORS = 8
MORAN_PERMUTATIONS = 999

SEASONS = {
    1: "Spring",
    2: "Summer",
    3: "Autumn",
}

RESPONSES = ["FCD", "BAD"]

TEMPORAL_FOLDS = {
    1: [2024, 2002, 2006, 2004, 2009],
    2: [2016, 2015, 2007, 2018, 2014],
    3: [2025, 2021, 2013, 2019, 2003],
    4: [2010, 2023, 2017, 2011, 2005],
    5: [2020, 2022, 2001, 2012, 2008],
}

COUNTRY_DUMMY_COLUMNS = [
    "Country_China",
    "Country_NK",
    "Country_Russia",
]

CONTINUOUS_PREDICTORS = [
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

BASE_MODEL_COLUMNS = (
    CONTINUOUS_PREDICTORS
    + COUNTRY_DUMMY_COLUMNS
)

COORDINATE_COLUMNS = [
    "Centroid_X_UTM52_m",
    "Centroid_Y_UTM52_m",
    "Longitude",
    "Latitude",
]

MODEL_VARIANTS = {
    "Drivers_Only": {
        "include_xy": False,
        "include_rbf": False,
        "interpretation": (
            "Selected environmental, vegetation, terrain, anthropogenic, "
            "and country predictors only."
        ),
    },
    "Drivers_XY": {
        "include_xy": True,
        "include_rbf": False,
        "interpretation": (
            "Driver model plus standardized UTM centroid X and Y as "
            "linear spatial controls."
        ),
    },
    "Drivers_XY_RBF12": {
        "include_xy": True,
        "include_rbf": True,
        "interpretation": (
            "Driver model plus standardized UTM X/Y and 12 Gaussian "
            "radial-basis spatial controls."
        ),
    },
}

# Locked after the previous tuning step.
LOCKED_CONFIGS: Dict[Tuple[int, str], Dict[str, object]] = {
    (1, "FCD"): {
        "occurrence_name": "O1_Baseline",
        "occurrence_max_features": "sqrt",
        "occurrence_min_samples_leaf": 5,
        "occurrence_max_samples": 0.70,
        "severity_name": "S3_Leaf10",
        "severity_max_features": "sqrt",
        "severity_min_samples_leaf": 10,
        "severity_max_samples": 0.80,
    },
    (1, "BAD"): {
        "occurrence_name": "O4_Features50_Leaf20",
        "occurrence_max_features": 0.50,
        "occurrence_min_samples_leaf": 20,
        "occurrence_max_samples": 0.70,
        "severity_name": "S3_Leaf10",
        "severity_max_features": "sqrt",
        "severity_min_samples_leaf": 10,
        "severity_max_samples": 0.80,
    },
    (2, "FCD"): {
        "occurrence_name": "O2_Leaf20",
        "occurrence_max_features": "sqrt",
        "occurrence_min_samples_leaf": 20,
        "occurrence_max_samples": 0.70,
        "severity_name": "S3_Leaf10",
        "severity_max_features": "sqrt",
        "severity_min_samples_leaf": 10,
        "severity_max_samples": 0.80,
    },
    (2, "BAD"): {
        "occurrence_name": "O4_Features50_Leaf20",
        "occurrence_max_features": 0.50,
        "occurrence_min_samples_leaf": 20,
        "occurrence_max_samples": 0.70,
        "severity_name": "S3_Leaf10",
        "severity_max_features": "sqrt",
        "severity_min_samples_leaf": 10,
        "severity_max_samples": 0.80,
    },
    (3, "FCD"): {
        "occurrence_name": "O2_Leaf20",
        "occurrence_max_features": "sqrt",
        "occurrence_min_samples_leaf": 20,
        "occurrence_max_samples": 0.70,
        "severity_name": "S3_Leaf10",
        "severity_max_features": "sqrt",
        "severity_min_samples_leaf": 10,
        "severity_max_samples": 0.80,
    },
    (3, "BAD"): {
        "occurrence_name": "O1_Baseline",
        "occurrence_max_features": "sqrt",
        "occurrence_min_samples_leaf": 5,
        "occurrence_max_samples": 0.70,
        "severity_name": "S5_Features50_Leaf10",
        "severity_max_features": 0.50,
        "severity_min_samples_leaf": 10,
        "severity_max_samples": 0.80,
    },
}


# =============================================================================
# Logging
# =============================================================================

def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as file:
        file.write(line + "\n")


# =============================================================================
# Metrics
# =============================================================================

def safe_spearman(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> float:
    if len(observed) < 3:
        return np.nan

    if np.std(observed) == 0 or np.std(predicted) == 0:
        return np.nan

    return float(
        spearmanr(
            observed,
            predicted,
            nan_policy="omit",
        ).statistic
    )


def safe_roc_auc(
    labels: np.ndarray,
    scores: np.ndarray,
) -> float:
    if len(np.unique(labels)) < 2:
        return np.nan

    return float(roc_auc_score(labels, scores))


def safe_pr_auc(
    labels: np.ndarray,
    scores: np.ndarray,
) -> float:
    if len(np.unique(labels)) < 2:
        return np.nan

    return float(average_precision_score(labels, scores))


def calculate_metrics(
    observed: np.ndarray,
    expected_prediction: np.ndarray,
    occurrence_probability: np.ndarray,
    conditional_severity: np.ndarray,
) -> Dict[str, float]:
    expected_prediction = np.clip(
        expected_prediction,
        0.0,
        None,
    )

    occurrence_probability = np.clip(
        occurrence_probability,
        1e-6,
        1 - 1e-6,
    )

    conditional_severity = np.clip(
        conditional_severity,
        0.0,
        None,
    )

    positive = observed > 0
    zero = observed == 0
    occurrence = positive.astype(int)

    result: Dict[str, float] = {
        "N": int(len(observed)),
        "Positive_N": int(positive.sum()),
        "Zero_N": int(zero.sum()),
        "Observed_Mean": float(observed.mean()),
        "Predicted_Mean": float(expected_prediction.mean()),
        "Mean_Bias": float(
            expected_prediction.mean() - observed.mean()
        ),
        "Absolute_Mean_Bias": float(
            abs(expected_prediction.mean() - observed.mean())
        ),
        "RMSE_All": float(
            math.sqrt(
                mean_squared_error(
                    observed,
                    expected_prediction,
                )
            )
        ),
        "MAE_All": float(
            mean_absolute_error(
                observed,
                expected_prediction,
            )
        ),
        "R2_All": float(
            r2_score(
                observed,
                expected_prediction,
            )
        ),
        "Spearman_All": safe_spearman(
            observed,
            expected_prediction,
        ),
        "Occurrence_Prevalence": float(occurrence.mean()),
        "Mean_Predicted_Probability": float(
            occurrence_probability.mean()
        ),
        "Occurrence_Probability_Bias": float(
            occurrence_probability.mean()
            - occurrence.mean()
        ),
        "Occurrence_ROC_AUC": safe_roc_auc(
            occurrence,
            occurrence_probability,
        ),
        "Occurrence_PR_AUC": safe_pr_auc(
            occurrence,
            occurrence_probability,
        ),
        "Occurrence_Brier": float(
            brier_score_loss(
                occurrence,
                occurrence_probability,
            )
        ),
        "Occurrence_LogLoss": float(
            log_loss(
                occurrence,
                occurrence_probability,
                labels=[0, 1],
            )
        ),
    }

    if positive.any():
        observed_positive = observed[positive]
        expected_positive = expected_prediction[positive]
        severity_positive = conditional_severity[positive]

        result.update(
            {
                "Positive_Observed_Mean": float(
                    observed_positive.mean()
                ),
                "Expected_Prediction_on_Positive_Mean": float(
                    expected_positive.mean()
                ),
                "Conditional_Severity_Prediction_Mean": float(
                    severity_positive.mean()
                ),
                "Expected_RMSE_Positive": float(
                    math.sqrt(
                        mean_squared_error(
                            observed_positive,
                            expected_positive,
                        )
                    )
                ),
                "Expected_MAE_Positive": float(
                    mean_absolute_error(
                        observed_positive,
                        expected_positive,
                    )
                ),
                "Expected_Spearman_Positive": safe_spearman(
                    observed_positive,
                    expected_positive,
                ),
                "Conditional_Severity_RMSE": float(
                    math.sqrt(
                        mean_squared_error(
                            observed_positive,
                            severity_positive,
                        )
                    )
                ),
                "Conditional_Severity_MAE": float(
                    mean_absolute_error(
                        observed_positive,
                        severity_positive,
                    )
                ),
                "Conditional_Severity_Spearman": safe_spearman(
                    observed_positive,
                    severity_positive,
                ),
            }
        )
    else:
        for column in [
            "Positive_Observed_Mean",
            "Expected_Prediction_on_Positive_Mean",
            "Conditional_Severity_Prediction_Mean",
            "Expected_RMSE_Positive",
            "Expected_MAE_Positive",
            "Expected_Spearman_Positive",
            "Conditional_Severity_RMSE",
            "Conditional_Severity_MAE",
            "Conditional_Severity_Spearman",
        ]:
            result[column] = np.nan

    if zero.any():
        zero_predictions = expected_prediction[zero]

        result.update(
            {
                "Zero_Predicted_Mean": float(
                    zero_predictions.mean()
                ),
                "Zero_Predicted_P95": float(
                    np.quantile(
                        zero_predictions,
                        0.95,
                    )
                ),
            }
        )
    else:
        result["Zero_Predicted_Mean"] = np.nan
        result["Zero_Predicted_P95"] = np.nan

    return result


# =============================================================================
# Data preparation
# =============================================================================

def prepare_data() -> pd.DataFrame:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Base table not found:\n{INPUT_CSV}"
        )

    data = pd.read_csv(INPUT_CSV)

    required = {
        "GRID_UID",
        "Country",
        "GRID_ID",
        "Year",
        "Season",
        "Fire_Count",
        "Burned_Pixel_Count",
        "ForestPixelCount",
        *CONTINUOUS_PREDICTORS,
        *COORDINATE_COLUMNS,
    }

    missing_columns = sorted(
        required - set(data.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(missing_columns)
        )

    duplicate_panel = data.duplicated(
        ["GRID_UID", "Year", "Season"],
        keep=False,
    )

    if duplicate_panel.any():
        data.loc[duplicate_panel].to_csv(
            OUTPUT_ROOT / "ERROR_Duplicate_Panel_Records.csv",
            index=False,
        )
        raise ValueError(
            "Duplicate GRID_UID-Year-Season records were found."
        )

    data["ForestArea_km2"] = data["ForestPixelCount"] * 0.25

    data["FCD"] = np.where(
        data["ForestPixelCount"] > 0,
        data["Fire_Count"] / data["ForestPixelCount"],
        np.nan,
    )

    data["BAD"] = np.where(
        data["ForestPixelCount"] > 0,
        data["Burned_Pixel_Count"] / data["ForestPixelCount"],
        np.nan,
    )

    year_to_fold: Dict[int, int] = {}

    for fold, years in TEMPORAL_FOLDS.items():
        for year in years:
            year_to_fold[int(year)] = int(fold)

    data["Temporal_Fold"] = (
        data["Year"].map(year_to_fold).astype(int)
    )

    data = data.loc[
        data["ForestArea_km2"] > FOREST_AREA_THRESHOLD_KM2
    ].copy()

    country_dummies = pd.get_dummies(
        data["Country"],
        prefix="Country",
        dtype=float,
    )

    for column in COUNTRY_DUMMY_COLUMNS:
        if column not in country_dummies.columns:
            country_dummies[column] = 0.0

    country_dummies = country_dummies[
        COUNTRY_DUMMY_COLUMNS
    ]

    data = pd.concat(
        [
            data.reset_index(drop=True),
            country_dummies.reset_index(drop=True),
        ],
        axis=1,
    )

    incomplete = data[
        BASE_MODEL_COLUMNS
        + RESPONSES
        + COORDINATE_COLUMNS
    ].isna().any(axis=1)

    excluded = data.loc[
        incomplete,
        [
            "GRID_UID",
            "Country",
            "GRID_ID",
            "Year",
            "Season",
            "ForestPixelCount",
            "ForestArea_km2",
            *CONTINUOUS_PREDICTORS,
            *COORDINATE_COLUMNS,
        ],
    ].copy()

    excluded.to_csv(
        OUTPUT_ROOT / "Excluded_Missing_Records.csv",
        index=False,
        encoding="utf-8-sig",
    )

    model_data = data.loc[
        ~incomplete
    ].copy()

    return model_data


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
        index=False,
        header=False,
        lineterminator="\n",
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_shared_spatial_blocks(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Read and verify the single Step-07 spatial-block assignment.

    No KMeans model is fitted here. The exact file generated upstream in
    08_Regression_modeling is copied into this RF result directory for
    provenance after its grid universe, metadata, block counts, semantic hash,
    and byte hash have been verified against the local RF analysis sample.
    """
    if not SHARED_SPATIAL_BLOCK_FILE.exists():
        raise FileNotFoundError(
            "The shared Step-07 spatial-block assignment is missing:\n"
            f"{SHARED_SPATIAL_BLOCK_FILE}"
        )

    required = [
        "GRID_UID",
        "Country",
        "GRID_ID",
        *COORDINATE_COLUMNS,
        "Spatial_Block",
    ]
    shared = pd.read_csv(SHARED_SPATIAL_BLOCK_FILE)
    missing = sorted(set(required) - set(shared.columns))
    if missing:
        raise ValueError(
            "The shared Step-07 assignment is missing columns:\n"
            + "\n".join(missing)
        )

    shared = shared[required].copy()
    shared["GRID_UID"] = shared["GRID_UID"].astype(str)
    shared["Spatial_Block"] = pd.to_numeric(
        shared["Spatial_Block"], errors="raise"
    ).astype(int)

    if shared["GRID_UID"].duplicated().any():
        raise ValueError(
            "Duplicate GRID_UID values were found in the shared assignment."
        )
    if len(shared) != EXPECTED_SPATIAL_GRID_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_SPATIAL_GRID_COUNT:,} shared grids, "
            f"but found {len(shared):,}."
        )

    block_counts = {
        int(block): int(count)
        for block, count in shared["Spatial_Block"]
        .value_counts()
        .sort_index()
        .items()
    }
    if block_counts != EXPECTED_SPATIAL_BLOCK_GRID_COUNTS:
        raise ValueError(
            "Shared spatial-block counts differ from the locked design.\n"
            f"Expected: {EXPECTED_SPATIAL_BLOCK_GRID_COUNTS}\n"
            f"Observed: {block_counts}"
        )

    mapping_hash = spatial_mapping_sha256(shared)
    if mapping_hash != EXPECTED_SPATIAL_MAPPING_SHA256:
        raise ValueError(
            "Shared GRID_UID-to-Spatial_Block mapping differs from the locked "
            "design.\n"
            f"Expected: {EXPECTED_SPATIAL_MAPPING_SHA256}\n"
            f"Observed: {mapping_hash}"
        )

    source_file_hash = sha256_file(SHARED_SPATIAL_BLOCK_FILE)
    if source_file_hash != EXPECTED_SPATIAL_BLOCK_FILE_SHA256:
        raise ValueError(
            "The shared assignment byte hash differs from the locked file.\n"
            f"Expected: {EXPECTED_SPATIAL_BLOCK_FILE_SHA256}\n"
            f"Observed: {source_file_hash}"
        )

    local = (
        data[["GRID_UID", "Country", "GRID_ID", *COORDINATE_COLUMNS]]
        .drop_duplicates("GRID_UID")
        .sort_values("GRID_UID")
        .reset_index(drop=True)
    )
    local["GRID_UID"] = local["GRID_UID"].astype(str)
    if local["GRID_UID"].duplicated().any():
        raise ValueError("Duplicate GRID_UID values remained in the RF grid table.")
    if len(local) != EXPECTED_SPATIAL_GRID_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_SPATIAL_GRID_COUNT:,} RF grids, "
            f"but found {len(local):,}."
        )

    merged = shared.merge(
        local,
        on="GRID_UID",
        how="outer",
        suffixes=("_Shared", "_RF"),
        indicator=True,
        validate="one_to_one",
    )
    if not (merged["_merge"] == "both").all():
        missing_in_rf = merged.loc[
            merged["_merge"] == "left_only", "GRID_UID"
        ].astype(str).tolist()
        missing_in_shared = merged.loc[
            merged["_merge"] == "right_only", "GRID_UID"
        ].astype(str).tolist()
        raise ValueError(
            "RF and shared spatial-grid universes differ.\n"
            f"Only in shared assignment: {missing_in_rf[:20]}\n"
            f"Only in RF data: {missing_in_shared[:20]}"
        )

    for column in ["Country", "GRID_ID"]:
        mismatch = (
            merged[f"{column}_Shared"].astype(str)
            != merged[f"{column}_RF"].astype(str)
        )
        if mismatch.any():
            raise ValueError(
                f"{column} disagrees between RF data and the shared assignment."
            )

    for column in COORDINATE_COLUMNS:
        shared_values = pd.to_numeric(
            merged[f"{column}_Shared"], errors="raise"
        ).to_numpy(dtype=float)
        rf_values = pd.to_numeric(
            merged[f"{column}_RF"], errors="raise"
        ).to_numpy(dtype=float)
        if not np.allclose(shared_values, rf_values, rtol=0.0, atol=1e-9):
            raise ValueError(
                f"{column} disagrees between RF data and the shared assignment."
            )

    # Copy, rather than rewrite, so the downstream RF copy is byte-identical
    # to the assignment first generated and locked by 08 Step 07.
    local_assignment = OUTPUT_ROOT / "Spatial_Block_Assignment.csv"
    shutil.copy2(SHARED_SPATIAL_BLOCK_FILE, local_assignment)
    copied_hash = sha256_file(local_assignment)
    if copied_hash != source_file_hash:
        raise RuntimeError("Copied spatial-block file is not byte-identical.")

    block_summary = (
        shared.groupby(["Spatial_Block", "Country"], as_index=False)
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
        OUTPUT_ROOT / "Spatial_Block_Summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    provenance = {
        "Workflow_Version": SHARED_SPATIAL_BLOCK_WORKFLOW_VERSION,
        "Source_Stage": "08_Regression_modeling Step 07",
        "Source_File": str(SHARED_SPATIAL_BLOCK_FILE),
        "Generation_Performed_in_09_RF_modeling": False,
        "RF_Action": "Read, verify, and copy the locked upstream assignment",
        "GRID_UID_Count": int(len(shared)),
        "Block_GRID_UID_Counts": {
            str(key): int(value) for key, value in block_counts.items()
        },
        "Semantic_Mapping_SHA256": mapping_hash,
        "Source_File_SHA256": source_file_hash,
        "Copied_File_SHA256": copied_hash,
        "RF_Grid_Universe_and_Metadata_Verified": True,
    }
    (OUTPUT_ROOT / "Spatial_Block_Source_Metadata.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    log(
        "Read the locked 4,982-grid spatial-block assignment from "
        "08_Regression_modeling Step 07; no KMeans block generation was "
        "performed in 09_RF_modeling."
    )
    return shared[["GRID_UID", "Spatial_Block"]].copy()

# =============================================================================
# Spatial controls
# =============================================================================

def make_design_matrices(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    variant_name: str,
    seed: int,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    List[str],
    Dict[str, object],
]:
    variant = MODEL_VARIANTS[variant_name]

    X_train_base = training[
        BASE_MODEL_COLUMNS
    ].to_numpy(dtype=np.float32)

    X_valid_base = validation[
        BASE_MODEL_COLUMNS
    ].to_numpy(dtype=np.float32)

    feature_names = list(BASE_MODEL_COLUMNS)

    metadata: Dict[str, object] = {
        "Variant": variant_name,
        "Spatial_Control_Count": 0,
        "RBF_Count": 0,
        "RBF_Sigma": np.nan,
    }

    if not variant["include_xy"]:
        return (
            X_train_base,
            X_valid_base,
            feature_names,
            metadata,
        )

    unique_training_grids = (
        training[
            [
                "GRID_UID",
                "Centroid_X_UTM52_m",
                "Centroid_Y_UTM52_m",
            ]
        ]
        .drop_duplicates("GRID_UID")
        .sort_values("GRID_UID")
    )

    unique_coordinates = unique_training_grids[
        [
            "Centroid_X_UTM52_m",
            "Centroid_Y_UTM52_m",
        ]
    ].to_numpy(dtype=float)

    coordinate_mean = unique_coordinates.mean(axis=0)
    coordinate_sd = unique_coordinates.std(axis=0)

    if np.any(coordinate_sd <= 0):
        raise ValueError(
            "Invalid coordinate standard deviation."
        )

    train_coordinates = training[
        [
            "Centroid_X_UTM52_m",
            "Centroid_Y_UTM52_m",
        ]
    ].to_numpy(dtype=float)

    valid_coordinates = validation[
        [
            "Centroid_X_UTM52_m",
            "Centroid_Y_UTM52_m",
        ]
    ].to_numpy(dtype=float)

    unique_standardized = (
        unique_coordinates - coordinate_mean
    ) / coordinate_sd

    train_xy = (
        train_coordinates - coordinate_mean
    ) / coordinate_sd

    valid_xy = (
        valid_coordinates - coordinate_mean
    ) / coordinate_sd

    X_train_parts = [
        X_train_base,
        train_xy.astype(np.float32),
    ]

    X_valid_parts = [
        X_valid_base,
        valid_xy.astype(np.float32),
    ]

    feature_names.extend(
        [
            "Spatial_Control_X_Standardized",
            "Spatial_Control_Y_Standardized",
        ]
    )

    metadata["Spatial_Control_Count"] = 2

    if variant["include_rbf"]:
        number_of_centers = min(
            RBF_BASIS_COUNT,
            len(unique_standardized),
        )

        kmeans = KMeans(
            n_clusters=number_of_centers,
            random_state=seed,
            n_init=30,
        )

        kmeans.fit(unique_standardized)
        centers = kmeans.cluster_centers_

        if len(centers) > 1:
            center_distances = distance.squareform(
                distance.pdist(centers)
            )
            center_distances[
                center_distances == 0
            ] = np.nan

            nearest_center_distance = np.nanmin(
                center_distances,
                axis=1,
            )

            sigma = float(
                1.5
                * np.nanmedian(
                    nearest_center_distance
                )
            )
        else:
            sigma = 1.0

        if not np.isfinite(sigma) or sigma <= 0:
            sigma = 1.0

        train_squared_distance = (
            (
                train_xy[:, None, :]
                - centers[None, :, :]
            )
            ** 2
        ).sum(axis=2)

        valid_squared_distance = (
            (
                valid_xy[:, None, :]
                - centers[None, :, :]
            )
            ** 2
        ).sum(axis=2)

        train_rbf = np.exp(
            -train_squared_distance
            / (2.0 * sigma**2)
        )

        valid_rbf = np.exp(
            -valid_squared_distance
            / (2.0 * sigma**2)
        )

        X_train_parts.append(
            train_rbf.astype(np.float32)
        )

        X_valid_parts.append(
            valid_rbf.astype(np.float32)
        )

        feature_names.extend(
            [
                f"Spatial_Control_RBF_{index:02d}"
                for index in range(
                    1,
                    number_of_centers + 1,
                )
            ]
        )

        metadata["Spatial_Control_Count"] = (
            2 + number_of_centers
        )
        metadata["RBF_Count"] = number_of_centers
        metadata["RBF_Sigma"] = sigma

    return (
        np.hstack(X_train_parts),
        np.hstack(X_valid_parts),
        feature_names,
        metadata,
    )


# =============================================================================
# Model fitting
# =============================================================================

def fit_hurdle(
    X_train: np.ndarray,
    X_valid: np.ndarray,
    y_train: np.ndarray,
    config: Dict[str, object],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    occurrence_train = (
        y_train > 0
    ).astype(np.int8)

    positive_train = y_train > 0

    if positive_train.sum() < 20:
        raise ValueError(
            "Too few positive training observations."
        )

    occurrence_model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_features=config[
            "occurrence_max_features"
        ],
        min_samples_leaf=int(
            config[
                "occurrence_min_samples_leaf"
            ]
        ),
        max_samples=float(
            config[
                "occurrence_max_samples"
            ]
        ),
        bootstrap=True,
        class_weight=None,
        n_jobs=-1,
        random_state=seed + 1,
        criterion="gini",
    )

    occurrence_model.fit(
        X_train,
        occurrence_train,
    )

    occurrence_probability = (
        occurrence_model.predict_proba(
            X_valid
        )[:, 1]
    )

    severity_model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_features=config[
            "severity_max_features"
        ],
        min_samples_leaf=int(
            config[
                "severity_min_samples_leaf"
            ]
        ),
        max_samples=float(
            config[
                "severity_max_samples"
            ]
        ),
        bootstrap=True,
        n_jobs=-1,
        random_state=seed + 2,
        criterion="squared_error",
    )

    severity_model.fit(
        X_train[positive_train],
        y_train[positive_train],
    )

    conditional_severity = np.clip(
        severity_model.predict(
            X_valid
        ),
        0.0,
        None,
    )

    expected_prediction = (
        occurrence_probability
        * conditional_severity
    )

    return (
        occurrence_probability,
        conditional_severity,
        expected_prediction,
    )


# =============================================================================
# Validation
# =============================================================================

def run_validation(
    data: pd.DataFrame,
    scheme: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: List[Dict[str, object]] = []
    metadata_rows: List[Dict[str, object]] = []
    prediction_frames: List[pd.DataFrame] = []

    if scheme == "Temporal":
        fold_column = "Temporal_Fold"
        fold_values = range(1, 6)
    elif scheme == "Spatial":
        fold_column = "Spatial_Block"
        fold_values = range(
            1,
            SPATIAL_BLOCK_COUNT + 1,
        )
    else:
        raise ValueError(
            f"Unknown scheme: {scheme}"
        )

    for season, season_label in SEASONS.items():
        season_data = data.loc[
            data["Season"] == season
        ].copy()

        for response in RESPONSES:
            config = LOCKED_CONFIGS[
                (season, response)
            ]

            for fold in fold_values:
                training = season_data.loc[
                    season_data[fold_column] != fold
                ].copy()

                validation = season_data.loc[
                    season_data[fold_column] == fold
                ].copy()

                if validation.empty:
                    raise ValueError(
                        f"Empty validation fold: "
                        f"{scheme}, {fold}"
                    )

                y_train = training[
                    response
                ].to_numpy(dtype=np.float64)

                y_valid = validation[
                    response
                ].to_numpy(dtype=np.float64)

                for variant_index, variant_name in enumerate(
                    MODEL_VARIANTS
                ):
                    seed = (
                        RANDOM_SEED
                        + season * 100_000
                        + (
                            10_000
                            if response == "BAD"
                            else 0
                        )
                        + (
                            5_000
                            if scheme == "Spatial"
                            else 0
                        )
                        + fold * 100
                        + variant_index * 10
                    )

                    (
                        X_train,
                        X_valid,
                        feature_names,
                        transform_metadata,
                    ) = make_design_matrices(
                        training=training,
                        validation=validation,
                        variant_name=variant_name,
                        seed=seed,
                    )

                    (
                        probability,
                        conditional,
                        expected,
                    ) = fit_hurdle(
                        X_train=X_train,
                        X_valid=X_valid,
                        y_train=y_train,
                        config=config,
                        seed=seed,
                    )

                    validation_label = (
                        ",".join(
                            map(
                                str,
                                sorted(
                                    validation[
                                        "Year"
                                    ].unique()
                                ),
                            )
                        )
                        if scheme == "Temporal"
                        else str(fold)
                    )

                    metric_rows.append(
                        {
                            "Validation_Scheme": scheme,
                            "Model_Variant": variant_name,
                            "Season": season,
                            "Season_Label": season_label,
                            "Response": response,
                            "Fold": fold,
                            "Validation_Label": validation_label,
                            "Occurrence_Config": config[
                                "occurrence_name"
                            ],
                            "Severity_Config": config[
                                "severity_name"
                            ],
                            "Total_Feature_Count": len(
                                feature_names
                            ),
                            "Spatial_Control_Count":
                                transform_metadata[
                                    "Spatial_Control_Count"
                                ],
                            "Training_Rows": len(training),
                            "Validation_Rows": len(validation),
                            **calculate_metrics(
                                observed=y_valid,
                                expected_prediction=expected,
                                occurrence_probability=probability,
                                conditional_severity=conditional,
                            ),
                        }
                    )

                    metadata_rows.append(
                        {
                            "Validation_Scheme": scheme,
                            "Model_Variant": variant_name,
                            "Season": season,
                            "Response": response,
                            "Fold": fold,
                            **transform_metadata,
                        }
                    )

                    prediction_frame = validation[
                        [
                            "GRID_UID",
                            "Country",
                            "GRID_ID",
                            "Year",
                            "Season",
                            "Temporal_Fold",
                            "Spatial_Block",
                            *COORDINATE_COLUMNS,
                        ]
                    ].copy()

                    prediction_frame[
                        "Validation_Scheme"
                    ] = scheme

                    prediction_frame[
                        "Validation_Fold"
                    ] = fold

                    prediction_frame[
                        "Model_Variant"
                    ] = variant_name

                    prediction_frame[
                        "Response"
                    ] = response

                    prediction_frame[
                        "Observed"
                    ] = y_valid

                    prediction_frame[
                        "Occurrence_Probability"
                    ] = probability

                    prediction_frame[
                        "Conditional_Severity"
                    ] = conditional

                    prediction_frame[
                        "Expected_Prediction"
                    ] = expected

                    prediction_frame[
                        "Residual"
                    ] = y_valid - expected

                    prediction_frames.append(
                        prediction_frame
                    )

                    log(
                        f"{scheme}: {season_label} "
                        f"{response}, fold {fold}, "
                        f"{variant_name} completed."
                    )

    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(metadata_rows),
        pd.concat(
            prediction_frames,
            ignore_index=True,
        ),
    )


# =============================================================================
# Moran's I
# =============================================================================

def build_knn_weights(
    coordinates: np.ndarray,
    k: int,
) -> sparse.csr_matrix:
    tree = cKDTree(coordinates)

    _, neighbors = tree.query(
        coordinates,
        k=k + 1,
    )

    rows = np.repeat(
        np.arange(len(coordinates)),
        k,
    )

    columns = neighbors[:, 1:].reshape(-1)

    values = np.full(
        len(rows),
        1.0 / k,
        dtype=float,
    )

    return sparse.csr_matrix(
        (
            values,
            (rows, columns),
        ),
        shape=(
            len(coordinates),
            len(coordinates),
        ),
    )


def morans_i_permutation(
    residuals: np.ndarray,
    weights: sparse.csr_matrix,
    permutations: int,
    seed: int,
) -> Dict[str, float]:
    residuals = np.asarray(
        residuals,
        dtype=float,
    )

    if not np.isfinite(residuals).all():
        raise ValueError(
            "Residuals contain non-finite values."
        )

    centered = residuals - residuals.mean()
    denominator = float(
        np.dot(centered, centered)
    )

    if denominator <= 0:
        return {
            "Moran_I": np.nan,
            "Expected_I": np.nan,
            "Permutation_P_TwoSided": np.nan,
            "Permutation_Mean": np.nan,
            "Permutation_SD": np.nan,
        }

    observed_i = float(
        np.dot(
            centered,
            weights.dot(centered),
        )
        / denominator
    )

    expected_i = -1.0 / (
        len(centered) - 1
    )

    rng = np.random.default_rng(seed)
    permutation_values = np.empty(
        permutations,
        dtype=float,
    )

    for index in range(permutations):
        permuted = rng.permutation(
            centered
        )

        permutation_values[index] = float(
            np.dot(
                permuted,
                weights.dot(permuted),
            )
            / denominator
        )

    observed_distance = abs(
        observed_i - expected_i
    )

    permutation_distance = np.abs(
        permutation_values - expected_i
    )

    p_value = (
        1
        + int(
            (
                permutation_distance
                >= observed_distance
            ).sum()
        )
    ) / (
        permutations + 1
    )

    return {
        "Moran_I": observed_i,
        "Expected_I": expected_i,
        "Permutation_P_TwoSided": float(
            p_value
        ),
        "Permutation_Mean": float(
            permutation_values.mean()
        ),
        "Permutation_SD": float(
            permutation_values.std(ddof=1)
        ),
    }


def run_moran(
    temporal_predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for variant_index, variant_name in enumerate(
        MODEL_VARIANTS
    ):
        for season, season_label in SEASONS.items():
            for response in RESPONSES:
                subset = temporal_predictions.loc[
                    (
                        temporal_predictions[
                            "Model_Variant"
                        ] == variant_name
                    )
                    & (
                        temporal_predictions[
                            "Season"
                        ] == season
                    )
                    & (
                        temporal_predictions[
                            "Response"
                        ] == response
                    )
                ].copy()

                grid_residuals = (
                    subset.groupby(
                        [
                            "GRID_UID",
                            "Country",
                            "GRID_ID",
                            *COORDINATE_COLUMNS,
                        ],
                        as_index=False,
                    )
                    .agg(
                        Mean_Residual=(
                            "Residual",
                            "mean",
                        ),
                        Mean_Observed=(
                            "Observed",
                            "mean",
                        ),
                        Mean_Predicted=(
                            "Expected_Prediction",
                            "mean",
                        ),
                        Years=("Year", "nunique"),
                    )
                    .sort_values("GRID_UID")
                    .reset_index(drop=True)
                )

                coordinates = grid_residuals[
                    [
                        "Centroid_X_UTM52_m",
                        "Centroid_Y_UTM52_m",
                    ]
                ].to_numpy(dtype=float)

                weights = build_knn_weights(
                    coordinates=coordinates,
                    k=MORAN_K_NEIGHBORS,
                )

                result = morans_i_permutation(
                    residuals=grid_residuals[
                        "Mean_Residual"
                    ].to_numpy(dtype=float),
                    weights=weights,
                    permutations=MORAN_PERMUTATIONS,
                    seed=(
                        RANDOM_SEED
                        + variant_index * 1_000
                        + season * 100
                        + (
                            1
                            if response == "FCD"
                            else 2
                        )
                    ),
                )

                rows.append(
                    {
                        "Model_Variant": variant_name,
                        "Season": season,
                        "Season_Label": season_label,
                        "Response": response,
                        "GRID_UID_Count": len(
                            grid_residuals
                        ),
                        "Residual_Aggregation": (
                            "Mean temporal OOF residual "
                            "by GRID_UID"
                        ),
                        "Weights": (
                            f"{MORAN_K_NEIGHBORS}-nearest-neighbor "
                            "row-standardized"
                        ),
                        "Permutations":
                            MORAN_PERMUTATIONS,
                        **result,
                    }
                )

                grid_residuals.to_csv(
                    OUTPUT_ROOT
                    / (
                        f"Temporal_Grid_Residuals_"
                        f"{variant_name}_S{season}_"
                        f"{response}.csv"
                    ),
                    index=False,
                    encoding="utf-8-sig",
                )

    return pd.DataFrame(rows)


# =============================================================================
# Summaries
# =============================================================================

def summarize_fold_metrics(
    fold_metrics: pd.DataFrame,
) -> pd.DataFrame:
    grouping = [
        "Validation_Scheme",
        "Model_Variant",
        "Season",
        "Season_Label",
        "Response",
        "Occurrence_Config",
        "Severity_Config",
    ]

    excluded_numeric = {
        "Season",
        "Fold",
        "Total_Feature_Count",
        "Spatial_Control_Count",
        "Training_Rows",
        "Validation_Rows",
    }

    numeric_columns = [
        column
        for column in fold_metrics.select_dtypes(
            include=[np.number]
        ).columns
        if column not in excluded_numeric
    ]

    rows: List[Dict[str, object]] = []

    for group_values, group in fold_metrics.groupby(
        grouping,
        sort=False,
    ):
        row = dict(
            zip(
                grouping,
                group_values,
            )
        )

        row["Folds"] = len(group)

        for column in numeric_columns:
            row[f"{column}_Mean"] = float(
                group[column].mean()
            )
            row[f"{column}_SD"] = float(
                group[column].std(ddof=1)
            )

        rows.append(row)

    return pd.DataFrame(rows)


def pooled_oof_metrics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    grouping = [
        "Validation_Scheme",
        "Model_Variant",
        "Season",
        "Response",
    ]

    for group_values, group in predictions.groupby(
        grouping,
        sort=False,
    ):
        (
            scheme,
            variant,
            season,
            response,
        ) = group_values

        metrics = calculate_metrics(
            observed=group[
                "Observed"
            ].to_numpy(dtype=float),
            expected_prediction=group[
                "Expected_Prediction"
            ].to_numpy(dtype=float),
            occurrence_probability=group[
                "Occurrence_Probability"
            ].to_numpy(dtype=float),
            conditional_severity=group[
                "Conditional_Severity"
            ].to_numpy(dtype=float),
        )

        rows.append(
            {
                "Validation_Scheme": scheme,
                "Model_Variant": variant,
                "Season": season,
                "Season_Label": SEASONS[
                    int(season)
                ],
                "Response": response,
                **metrics,
            }
        )

    return pd.DataFrame(rows)


def make_comparison_table(
    pooled_metrics: pd.DataFrame,
    moran_results: pd.DataFrame,
) -> pd.DataFrame:
    baseline_metrics = (
        pooled_metrics.loc[
            pooled_metrics[
                "Model_Variant"
            ] == "Drivers_Only"
        ]
        .copy()
    )

    baseline_moran = (
        moran_results.loc[
            moran_results[
                "Model_Variant"
            ] == "Drivers_Only"
        ]
        .copy()
    )

    rows: List[Dict[str, object]] = []

    for _, candidate in pooled_metrics.iterrows():
        mask = (
            (
                baseline_metrics[
                    "Validation_Scheme"
                ]
                == candidate[
                    "Validation_Scheme"
                ]
            )
            & (
                baseline_metrics["Season"]
                == candidate["Season"]
            )
            & (
                baseline_metrics["Response"]
                == candidate["Response"]
            )
        )

        baseline_match = baseline_metrics.loc[
            mask
        ]

        if len(baseline_match) != 1:
            raise RuntimeError(
                "Could not uniquely match pooled baseline."
            )

        baseline = baseline_match.iloc[0]

        row = {
            "Validation_Scheme": candidate[
                "Validation_Scheme"
            ],
            "Season": candidate["Season"],
            "Season_Label": candidate[
                "Season_Label"
            ],
            "Response": candidate["Response"],
            "Model_Variant": candidate[
                "Model_Variant"
            ],
            "R2_All": candidate["R2_All"],
            "Delta_R2_vs_Drivers_Only": (
                candidate["R2_All"]
                - baseline["R2_All"]
            ),
            "RMSE_All": candidate[
                "RMSE_All"
            ],
            "Delta_RMSE_vs_Drivers_Only": (
                candidate["RMSE_All"]
                - baseline["RMSE_All"]
            ),
            "MAE_All": candidate[
                "MAE_All"
            ],
            "Delta_MAE_vs_Drivers_Only": (
                candidate["MAE_All"]
                - baseline["MAE_All"]
            ),
            "Occurrence_ROC_AUC": candidate[
                "Occurrence_ROC_AUC"
            ],
            "Delta_ROC_AUC_vs_Drivers_Only": (
                candidate[
                    "Occurrence_ROC_AUC"
                ]
                - baseline[
                    "Occurrence_ROC_AUC"
                ]
            ),
            "Occurrence_PR_AUC": candidate[
                "Occurrence_PR_AUC"
            ],
            "Delta_PR_AUC_vs_Drivers_Only": (
                candidate[
                    "Occurrence_PR_AUC"
                ]
                - baseline[
                    "Occurrence_PR_AUC"
                ]
            ),
        }

        if (
            candidate["Validation_Scheme"]
            == "Temporal"
        ):
            moran_mask = (
                (
                    moran_results[
                        "Model_Variant"
                    ]
                    == candidate[
                        "Model_Variant"
                    ]
                )
                & (
                    moran_results["Season"]
                    == candidate["Season"]
                )
                & (
                    moran_results["Response"]
                    == candidate["Response"]
                )
            )

            candidate_moran = (
                moran_results.loc[
                    moran_mask
                ].iloc[0]
            )

            baseline_moran_mask = (
                (
                    baseline_moran["Season"]
                    == candidate["Season"]
                )
                & (
                    baseline_moran["Response"]
                    == candidate["Response"]
                )
            )

            baseline_moran_row = (
                baseline_moran.loc[
                    baseline_moran_mask
                ].iloc[0]
            )

            baseline_i = float(
                baseline_moran_row["Moran_I"]
            )

            candidate_i = float(
                candidate_moran["Moran_I"]
            )

            row.update(
                {
                    "Residual_Moran_I":
                        candidate_i,
                    "Delta_Moran_I_vs_Drivers_Only": (
                        candidate_i - baseline_i
                    ),
                    "Absolute_Moran_I_Reduction_Percent": (
                        100.0
                        * (
                            abs(baseline_i)
                            - abs(candidate_i)
                        )
                        / abs(baseline_i)
                        if baseline_i != 0
                        else np.nan
                    ),
                    "Moran_P": candidate_moran[
                        "Permutation_P_TwoSided"
                    ],
                }
            )
        else:
            row.update(
                {
                    "Residual_Moran_I": np.nan,
                    "Delta_Moran_I_vs_Drivers_Only":
                        np.nan,
                    "Absolute_Moran_I_Reduction_Percent":
                        np.nan,
                    "Moran_P": np.nan,
                }
            )

        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOG_FILE.write_text(
        "",
        encoding="utf-8",
    )

    log("=" * 78)
    log("Starting spatial-correction sensitivity analysis")
    log(f"Input: {INPUT_CSV}")
    log(
        "Forest threshold: ForestArea_km2 > 2.5 km2 "
        "(equivalent to > 2.5 km2)"
    )
    log(f"Trees per RF: {N_ESTIMATORS}")
    log("=" * 78)

    data = prepare_data()

    spatial_blocks = load_shared_spatial_blocks(
        data
    )

    data = data.merge(
        spatial_blocks,
        on="GRID_UID",
        how="left",
        validate="many_to_one",
    )

    if data["Spatial_Block"].isna().any():
        raise RuntimeError(
            "Spatial block assignment is incomplete."
        )

    variant_rows = []

    for variant_name, definition in (
        MODEL_VARIANTS.items()
    ):
        variant_rows.append(
            {
                "Model_Variant": variant_name,
                **definition,
                "Spatial_controls_are_drivers":
                    False,
                "Include_in_driver_importance":
                    False,
            }
        )

    pd.DataFrame(variant_rows).to_csv(
        OUTPUT_ROOT
        / "00_Model_Variant_Definitions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    config_rows = []

    for (season, response), config in (
        LOCKED_CONFIGS.items()
    ):
        config_rows.append(
            {
                "Season": season,
                "Season_Label":
                    SEASONS[season],
                "Response": response,
                **config,
            }
        )

    pd.DataFrame(config_rows).to_csv(
        OUTPUT_ROOT
        / "01_Locked_Model_Configurations.csv",
        index=False,
        encoding="utf-8-sig",
    )

    (
        temporal_metrics,
        temporal_metadata,
        temporal_predictions,
    ) = run_validation(
        data=data,
        scheme="Temporal",
    )

    (
        spatial_metrics,
        spatial_metadata,
        spatial_predictions,
    ) = run_validation(
        data=data,
        scheme="Spatial",
    )

    fold_metrics = pd.concat(
        [
            temporal_metrics,
            spatial_metrics,
        ],
        ignore_index=True,
    )

    transform_metadata = pd.concat(
        [
            temporal_metadata,
            spatial_metadata,
        ],
        ignore_index=True,
    )

    predictions = pd.concat(
        [
            temporal_predictions,
            spatial_predictions,
        ],
        ignore_index=True,
    )

    fold_metrics.to_csv(
        OUTPUT_ROOT / "02_Fold_Metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summarize_fold_metrics(
        fold_metrics
    ).to_csv(
        OUTPUT_ROOT / "03_Fold_Summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pooled_metrics = pooled_oof_metrics(
        predictions
    )

    pooled_metrics.to_csv(
        OUTPUT_ROOT / "04_Pooled_OOF_Metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    transform_metadata.to_csv(
        OUTPUT_ROOT
        / "05_Spatial_Transform_Metadata.csv",
        index=False,
        encoding="utf-8-sig",
    )

    moran_results = run_moran(
        temporal_predictions
    )

    moran_results.to_csv(
        OUTPUT_ROOT / "06_Residual_Morans_I.csv",
        index=False,
        encoding="utf-8-sig",
    )

    comparison = make_comparison_table(
        pooled_metrics=pooled_metrics,
        moran_results=moran_results,
    )

    comparison.to_csv(
        OUTPUT_ROOT
        / "07_Comparison_vs_Drivers_Only.csv",
        index=False,
        encoding="utf-8-sig",
    )

    predictions.to_csv(
        OUTPUT_ROOT
        / "All_OOF_Predictions.csv.gz",
        index=False,
        compression="gzip",
    )

    qa = pd.DataFrame(
        [
            [
                "ForestArea_threshold_rule",
                "ForestArea_km2 > 2.5",
            ],
            [
                "Equivalent_forest_area_rule",
                "ForestArea_km2 > 2.5",
            ],
            [
                "Rows_used",
                len(data),
            ],
            [
                "Unique_GRID_UIDs_used",
                data["GRID_UID"].nunique(),
            ],
            [
                "Driver_predictor_count",
                len(CONTINUOUS_PREDICTORS),
            ],
            [
                "Country_dummy_count",
                len(COUNTRY_DUMMY_COLUMNS),
            ],
            [
                "Model_variant_count",
                len(MODEL_VARIANTS),
            ],
            [
                "RBF_basis_count",
                RBF_BASIS_COUNT,
            ],
            [
                "Trees_per_RF",
                N_ESTIMATORS,
            ],
            [
                "Moran_k_neighbors",
                MORAN_K_NEIGHBORS,
            ],
            [
                "Moran_permutations",
                MORAN_PERMUTATIONS,
            ],
        ],
        columns=["Check", "Value"],
    )

    qa.to_csv(
        OUTPUT_ROOT / "08_Data_QA_Summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    environment = {
        "Created": datetime.now().isoformat(
            timespec="seconds"
        ),
        "Python": sys.version,
        "Platform": platform.platform(),
        "NumPy": np.__version__,
        "pandas": pd.__version__,
        "SciPy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "Input_file": str(INPUT_CSV),
        "Forest_threshold": {
            "field": "ForestArea_km2",
            "rule": "ForestArea_km2 > 2.5",
            "field_unit": (
                "count of 500 m x 500 m forest pixels"
            ),
            "equivalent_area_rule":
                "ForestArea_km2 > 2.5",
        },
        "Trees_per_RF": N_ESTIMATORS,
        "Model_variants": MODEL_VARIANTS,
        "Spatial_block_assignment": {
            "method": (
                "Read and verify the single assignment generated upstream "
                "by 08_Regression_modeling Step 07; no KMeans block "
                "generation is performed in 09_RF_modeling"
            ),
            "source_file": str(SHARED_SPATIAL_BLOCK_FILE),
            "workflow_version": SHARED_SPATIAL_BLOCK_WORKFLOW_VERSION,
            "semantic_mapping_sha256": EXPECTED_SPATIAL_MAPPING_SHA256,
            "file_sha256": EXPECTED_SPATIAL_BLOCK_FILE_SHA256,
        },
        "RBF_method": {
            "coordinate_standardization":
                "Training-grid mean and standard deviation only",
            "centers":
                "KMeans centers fitted on unique training GRID_UID coordinates",
            "basis":
                "Gaussian radial basis functions",
            "basis_count":
                RBF_BASIS_COUNT,
            "bandwidth": (
                "1.5 times the median nearest-neighbor distance "
                "among fitted RBF centers"
            ),
            "leakage_control": (
                "Spatial standardization and RBF centers are fitted "
                "inside each training fold only"
            ),
        },
        "Interpretation": (
            "XY and RBF variables are nuisance spatial controls. "
            "They must not be ranked or interpreted as ecological drivers."
        ),
    }

    with (
        OUTPUT_ROOT
        / "Software_Environment_and_Method.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            environment,
            file,
            ensure_ascii=False,
            indent=2,
        )




    log("=" * 78)
    log(
        "Spatial-correction sensitivity analysis "
        "completed successfully."
    )
    log("=" * 78)

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
            print(
                error_text,
                file=sys.stderr,
            )

        raise
