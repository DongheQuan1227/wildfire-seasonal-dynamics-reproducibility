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
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
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
OUTPUT_ROOT = MODEL_ROOT / "09_Country_Effect_Sensitivity"
LOG_FILE = OUTPUT_ROOT / "country_effect_sensitivity.log"

# ForestPixelCount is the number of 500 m x 500 m forest pixels.
# ForestArea_km2 is ForestPixelCount * 0.25; models use ForestArea_km2 > 2.5.
FOREST_AREA_THRESHOLD_KM2 = 2.5
RANDOM_SEED = 2026
N_ESTIMATORS = 300
SPATIAL_BLOCK_COUNT = 5
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

COUNTRY_DUMMY_COLUMNS = [
    "Country_China",
    "Country_NK",
    "Country_Russia",
]

MODEL_VARIANTS = {
    "With_Country": (
        CONTINUOUS_PREDICTORS
        + COUNTRY_DUMMY_COLUMNS
    ),
    "Without_Country": list(
        CONTINUOUS_PREDICTORS
    ),
}

COORDINATE_COLUMNS = [
    "Centroid_X_UTM52_m",
    "Centroid_Y_UTM52_m",
    "Longitude",
    "Latitude",
]

# Locked after the previous tuning step. These settings are not re-tuned
# after removing Country, so the sensitivity isolates the Country effect.
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
# Metric helpers
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

    return float(
        roc_auc_score(labels, scores)
    )


def safe_pr_auc(
    labels: np.ndarray,
    scores: np.ndarray,
) -> float:
    if len(np.unique(labels)) < 2:
        return np.nan

    return float(
        average_precision_score(
            labels,
            scores,
        )
    )


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
        "Predicted_Mean": float(
            expected_prediction.mean()
        ),
        "Mean_Bias": float(
            expected_prediction.mean()
            - observed.mean()
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
        "Occurrence_Prevalence": float(
            occurrence.mean()
        ),
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
        expected_positive = (
            expected_prediction[positive]
        )
        conditional_positive = (
            conditional_severity[positive]
        )

        result.update(
            {
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
                "Expected_Spearman_Positive":
                    safe_spearman(
                        observed_positive,
                        expected_positive,
                    ),
                "Conditional_Severity_RMSE": float(
                    math.sqrt(
                        mean_squared_error(
                            observed_positive,
                            conditional_positive,
                        )
                    )
                ),
                "Conditional_Severity_MAE": float(
                    mean_absolute_error(
                        observed_positive,
                        conditional_positive,
                    )
                ),
                "Conditional_Severity_Spearman":
                    safe_spearman(
                        observed_positive,
                        conditional_positive,
                    ),
            }
        )
    else:
        for column in [
            "Expected_RMSE_Positive",
            "Expected_MAE_Positive",
            "Expected_Spearman_Positive",
            "Conditional_Severity_RMSE",
            "Conditional_Severity_MAE",
            "Conditional_Severity_Spearman",
        ]:
            result[column] = np.nan

    if zero.any():
        zero_predictions = (
            expected_prediction[zero]
        )

        result["Zero_Predicted_Mean"] = float(
            zero_predictions.mean()
        )
        result["Zero_Predicted_P95"] = float(
            np.quantile(
                zero_predictions,
                0.95,
            )
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
            OUTPUT_ROOT
            / "ERROR_Duplicate_Panel_Records.csv",
            index=False,
        )
        raise ValueError(
            "Duplicate GRID_UID-Year-Season "
            "records were found."
        )

    data["ForestArea_km2"] = data["ForestPixelCount"] * 0.25

    data["FCD"] = np.where(
        data["ForestPixelCount"] > 0,
        data["Fire_Count"]
        / data["ForestPixelCount"],
        np.nan,
    )

    data["BAD"] = np.where(
        data["ForestPixelCount"] > 0,
        data["Burned_Pixel_Count"]
        / data["ForestPixelCount"],
        np.nan,
    )

    year_to_fold: Dict[int, int] = {}

    for fold, years in TEMPORAL_FOLDS.items():
        for year in years:
            year_to_fold[int(year)] = int(fold)

    data["Temporal_Fold"] = (
        data["Year"]
        .map(year_to_fold)
        .astype(int)
    )

    data = data.loc[
        data["ForestArea_km2"]
        > FOREST_AREA_THRESHOLD_KM2
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
            country_dummies.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    required_model_columns = (
        CONTINUOUS_PREDICTORS
        + COUNTRY_DUMMY_COLUMNS
        + RESPONSES
        + COORDINATE_COLUMNS
    )

    incomplete = data[
        required_model_columns
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
        OUTPUT_ROOT
        / "Excluded_Missing_Records.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return data.loc[
        ~incomplete
    ].copy()


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
# Hurdle RF
# =============================================================================

def fit_predict_hurdle(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    response: str,
    predictors: Sequence[str],
    config: Dict[str, object],
    seed: int,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    X_train = training[
        list(predictors)
    ].to_numpy(dtype=np.float32)

    X_valid = validation[
        list(predictors)
    ].to_numpy(dtype=np.float32)

    y_train = training[
        response
    ].to_numpy(dtype=np.float64)

    occurrence_train = (
        y_train > 0
    ).astype(np.int8)

    positive_train = y_train > 0

    occurrence_model = (
        RandomForestClassifier(
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

    severity_model = (
        RandomForestRegressor(
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

def run_validation_scheme(
    data: pd.DataFrame,
    scheme: str,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    metric_rows: List[Dict[str, object]] = []
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
            f"Unknown validation scheme: {scheme}"
        )

    for season, season_label in (
        SEASONS.items()
    ):
        season_data = data.loc[
            data["Season"] == season
        ].copy()

        for response in RESPONSES:
            config = LOCKED_CONFIGS[
                (season, response)
            ]

            for fold in fold_values:
                training = season_data.loc[
                    season_data[
                        fold_column
                    ] != fold
                ].copy()

                validation = season_data.loc[
                    season_data[
                        fold_column
                    ] == fold
                ].copy()

                for variant_index, (
                    variant_name,
                    predictors,
                ) in enumerate(
                    MODEL_VARIANTS.items()
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
                        probability,
                        conditional,
                        expected,
                    ) = fit_predict_hurdle(
                        training=training,
                        validation=validation,
                        response=response,
                        predictors=predictors,
                        config=config,
                        seed=seed,
                    )

                    y_valid = validation[
                        response
                    ].to_numpy(dtype=float)

                    metric_rows.append(
                        {
                            "Validation_Scheme":
                                scheme,
                            "Model_Variant":
                                variant_name,
                            "Season": season,
                            "Season_Label":
                                season_label,
                            "Response": response,
                            "Fold": fold,
                            "Predictor_Count":
                                len(predictors),
                            "Occurrence_Config":
                                config[
                                    "occurrence_name"
                                ],
                            "Severity_Config":
                                config[
                                    "severity_name"
                                ],
                            **calculate_metrics(
                                observed=y_valid,
                                expected_prediction=
                                    expected,
                                occurrence_probability=
                                    probability,
                                conditional_severity=
                                    conditional,
                            ),
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
        pd.concat(
            prediction_frames,
            ignore_index=True,
        ),
    )


# =============================================================================
# Pooled and country-level summaries
# =============================================================================

def pooled_metrics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    grouping = [
        "Validation_Scheme",
        "Model_Variant",
        "Season",
        "Response",
    ]

    for group_values, group in (
        predictions.groupby(
            grouping,
            sort=False,
        )
    ):
        (
            scheme,
            variant,
            season,
            response,
        ) = group_values

        rows.append(
            {
                "Validation_Scheme":
                    scheme,
                "Model_Variant":
                    variant,
                "Season": season,
                "Season_Label":
                    SEASONS[int(season)],
                "Response":
                    response,
                **calculate_metrics(
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
                ),
            }
        )

    return pd.DataFrame(rows)


def country_residual_summary(
    temporal_predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    grouping = [
        "Model_Variant",
        "Season",
        "Response",
        "Country",
    ]

    for group_values, group in (
        temporal_predictions.groupby(
            grouping,
            sort=False,
        )
    ):
        (
            variant,
            season,
            response,
            country,
        ) = group_values

        residual = group[
            "Residual"
        ].to_numpy(dtype=float)

        observed = group[
            "Observed"
        ].to_numpy(dtype=float)

        predicted = group[
            "Expected_Prediction"
        ].to_numpy(dtype=float)

        rows.append(
            {
                "Model_Variant": variant,
                "Season": season,
                "Season_Label":
                    SEASONS[int(season)],
                "Response": response,
                "Country": country,
                "N": len(group),
                "Positive_N": int(
                    (observed > 0).sum()
                ),
                "Observed_Mean": float(
                    observed.mean()
                ),
                "Predicted_Mean": float(
                    predicted.mean()
                ),
                "Mean_Residual": float(
                    residual.mean()
                ),
                "Mean_Absolute_Residual":
                    float(
                        np.abs(residual).mean()
                    ),
                "RMSE": float(
                    math.sqrt(
                        np.mean(
                            residual**2
                        )
                    )
                ),
            }
        )

    return pd.DataFrame(rows)


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

    columns = neighbors[
        :, 1:
    ].reshape(-1)

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
    centered = (
        np.asarray(
            residuals,
            dtype=float,
        )
        - np.mean(residuals)
    )

    denominator = float(
        np.dot(centered, centered)
    )

    if denominator <= 0:
        return {
            "Moran_I": np.nan,
            "Expected_I": np.nan,
            "Permutation_P_TwoSided":
                np.nan,
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
        "Permutation_P_TwoSided":
            float(p_value),
    }


def residual_moran(
    temporal_predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for variant_index, variant in enumerate(
        MODEL_VARIANTS
    ):
        for season, season_label in (
            SEASONS.items()
        ):
            for response in RESPONSES:
                subset = temporal_predictions.loc[
                    (
                        temporal_predictions[
                            "Model_Variant"
                        ] == variant
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

                grid_residual = (
                    subset.groupby(
                        [
                            "GRID_UID",
                            *COORDINATE_COLUMNS,
                        ],
                        as_index=False,
                    )
                    .agg(
                        Mean_Residual=(
                            "Residual",
                            "mean",
                        )
                    )
                    .sort_values("GRID_UID")
                )

                coordinates = grid_residual[
                    [
                        "Centroid_X_UTM52_m",
                        "Centroid_Y_UTM52_m",
                    ]
                ].to_numpy(dtype=float)

                weights = build_knn_weights(
                    coordinates,
                    MORAN_K_NEIGHBORS,
                )

                result = morans_i_permutation(
                    residuals=grid_residual[
                        "Mean_Residual"
                    ].to_numpy(dtype=float),
                    weights=weights,
                    permutations=
                        MORAN_PERMUTATIONS,
                    seed=(
                        RANDOM_SEED
                        + variant_index
                        * 1_000
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
                        "Model_Variant":
                            variant,
                        "Season":
                            season,
                        "Season_Label":
                            season_label,
                        "Response":
                            response,
                        "GRID_UID_Count":
                            len(grid_residual),
                        "Weights": (
                            f"{MORAN_K_NEIGHBORS}-nearest-neighbor "
                            "row-standardized"
                        ),
                        "Permutations":
                            MORAN_PERMUTATIONS,
                        **result,
                    }
                )

    return pd.DataFrame(rows)


# =============================================================================
# Comparison table
# =============================================================================

def make_comparison(
    pooled: pd.DataFrame,
    moran: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    metrics = [
        "RMSE_All",
        "MAE_All",
        "R2_All",
        "Spearman_All",
        "Occurrence_ROC_AUC",
        "Occurrence_PR_AUC",
        "Occurrence_Brier",
        "Conditional_Severity_RMSE",
        "Conditional_Severity_Spearman",
    ]

    for (
        scheme,
        season,
        response,
    ), group in pooled.groupby(
        [
            "Validation_Scheme",
            "Season",
            "Response",
        ]
    ):
        with_country = group.loc[
            group["Model_Variant"]
            == "With_Country"
        ].iloc[0]

        without_country = group.loc[
            group["Model_Variant"]
            == "Without_Country"
        ].iloc[0]

        row = {
            "Validation_Scheme": scheme,
            "Season": season,
            "Season_Label":
                SEASONS[int(season)],
            "Response": response,
        }

        for metric in metrics:
            row[
                f"With_Country_{metric}"
            ] = with_country[metric]

            row[
                f"Without_Country_{metric}"
            ] = without_country[metric]

            row[
                f"Without_minus_With_{metric}"
            ] = (
                without_country[metric]
                - with_country[metric]
            )

        if scheme == "Temporal":
            with_moran = moran.loc[
                (
                    moran["Model_Variant"]
                    == "With_Country"
                )
                & (
                    moran["Season"]
                    == season
                )
                & (
                    moran["Response"]
                    == response
                )
            ].iloc[0]

            without_moran = moran.loc[
                (
                    moran["Model_Variant"]
                    == "Without_Country"
                )
                & (
                    moran["Season"]
                    == season
                )
                & (
                    moran["Response"]
                    == response
                )
            ].iloc[0]

            row["With_Country_Moran_I"] = (
                with_moran["Moran_I"]
            )

            row[
                "Without_Country_Moran_I"
            ] = without_moran["Moran_I"]

            row[
                "Without_minus_With_Moran_I"
            ] = (
                without_moran["Moran_I"]
                - with_moran["Moran_I"]
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
    log("Starting Country-effect sensitivity analysis")
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

    temporal_metrics, temporal_predictions = (
        run_validation_scheme(
            data,
            "Temporal",
        )
    )

    spatial_metrics, spatial_predictions = (
        run_validation_scheme(
            data,
            "Spatial",
        )
    )

    fold_metrics = pd.concat(
        [
            temporal_metrics,
            spatial_metrics,
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

    pooled = pooled_metrics(
        predictions
    )

    temporal_only_predictions = (
        predictions.loc[
            predictions[
                "Validation_Scheme"
            ] == "Temporal"
        ].copy()
    )

    country_summary = (
        country_residual_summary(
            temporal_only_predictions
        )
    )

    moran = residual_moran(
        temporal_only_predictions
    )

    comparison = make_comparison(
        pooled=pooled,
        moran=moran,
    )

    pd.DataFrame(
        [
            {
                "Model_Variant": name,
                "Predictor_Count":
                    len(predictors),
                "Predictors":
                    ";".join(predictors),
                "Country_Dummies_Included":
                    (
                        name
                        == "With_Country"
                    ),
            }
            for name, predictors
            in MODEL_VARIANTS.items()
        ]
    ).to_csv(
        OUTPUT_ROOT
        / "00_Model_Variant_Definitions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fold_metrics.to_csv(
        OUTPUT_ROOT
        / "01_Fold_Metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pooled.to_csv(
        OUTPUT_ROOT
        / "02_Pooled_OOF_Metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    comparison.to_csv(
        OUTPUT_ROOT
        / "03_Comparison_Without_vs_With_Country.csv",
        index=False,
        encoding="utf-8-sig",
    )

    country_summary.to_csv(
        OUTPUT_ROOT
        / "04_Temporal_OOF_Country_Residual_Summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    moran.to_csv(
        OUTPUT_ROOT
        / "05_Temporal_OOF_Residual_Morans_I.csv",
        index=False,
        encoding="utf-8-sig",
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
            ["Rows_used", len(data)],
            [
                "Unique_GRID_UIDs_used",
                data["GRID_UID"].nunique(),
            ],
            [
                "Continuous_predictor_count",
                len(CONTINUOUS_PREDICTORS),
            ],
            [
                "Country_dummy_count",
                len(COUNTRY_DUMMY_COLUMNS),
            ],
            [
                "Trees_per_RF",
                N_ESTIMATORS,
            ],
            [
                "Parameters_re_tuned_after_removing_Country",
                False,
            ],
            [
                "Sensitivity_interpretation",
                (
                    "Locked parameters are used in both variants "
                    "to isolate the incremental contribution of Country."
                ),
            ],
        ],
        columns=["Check", "Value"],
    )

    qa.to_csv(
        OUTPUT_ROOT
        / "06_Data_QA_Summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    predictions.to_csv(
        OUTPUT_ROOT
        / "All_OOF_Predictions.csv.gz",
        index=False,
        compression="gzip",
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
            "rule": "ForestArea_km2 > 2.5",
            "unit": "km2",
            "equivalent_area": "> 2.5 km2",
        },
        "Trees_per_RF": N_ESTIMATORS,
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
        "Validation": [
            "Five-fold grouped temporal cross-validation",
            "Five-block spatial cross-validation",
        ],
        "Country_sensitivity": (
            "With_Country versus Without_Country using the same "
            "locked RF hyperparameters and validation folds."
        ),
        "Important_note": (
            "Country is treated as a broad contextual control, not "
            "as a causal environmental driver."
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
    log("Country-effect sensitivity completed successfully.")
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
