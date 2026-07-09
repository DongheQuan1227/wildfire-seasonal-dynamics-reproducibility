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
from scipy.spatial import cKDTree
from scipy import sparse
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
OUTPUT_ROOT = MODEL_ROOT / "06_Final_Hurdle_RF_Validation"
LOG_FILE = OUTPUT_ROOT / "final_hurdle_RF_validation.log"

FOREST_AREA_THRESHOLD_KM2 = 2.5
RANDOM_SEED = 2026
N_ESTIMATORS = 500
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

MODEL_PREDICTORS = CONTINUOUS_PREDICTORS + COUNTRY_DUMMY_COLUMNS

COORDINATE_COLUMNS = [
    "Centroid_X_UTM52_m",
    "Centroid_Y_UTM52_m",
    "Longitude",
    "Latitude",
]

# Locked after the exploratory tuning step.
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
        MODEL_PREDICTORS
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
# Model fitting
# =============================================================================

def fit_hurdle_model(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    response: str,
    config: Dict[str, object],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_train = training[
        MODEL_PREDICTORS
    ].to_numpy(dtype=np.float32)

    X_valid = validation[
        MODEL_PREDICTORS
    ].to_numpy(dtype=np.float32)

    y_train = training[
        response
    ].to_numpy(dtype=np.float64)

    occurrence_train = (
        y_train > 0
    ).astype(np.int8)

    positive_train = y_train > 0

    if positive_train.sum() < 20:
        raise ValueError(
            "Too few positive training rows."
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


def run_validation_scheme(
    data: pd.DataFrame,
    scheme: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: List[Dict[str, object]] = []
    prediction_frames: List[pd.DataFrame] = []

    for season, season_label in SEASONS.items():
        season_data = data.loc[
            data["Season"] == season
        ].copy()

        for response in RESPONSES:
            config = LOCKED_CONFIGS[
                (season, response)
            ]

            if scheme == "Temporal":
                fold_values = range(1, 6)
                fold_column = "Temporal_Fold"
            elif scheme == "Spatial":
                fold_values = range(
                    1,
                    SPATIAL_BLOCK_COUNT + 1,
                )
                fold_column = "Spatial_Block"
            else:
                raise ValueError(
                    f"Unknown validation scheme: {scheme}"
                )

            for fold in fold_values:
                training = season_data.loc[
                    season_data[fold_column] != fold
                ]

                validation = season_data.loc[
                    season_data[fold_column] == fold
                ]

                if validation.empty:
                    raise ValueError(
                        f"Empty validation fold: "
                        f"{scheme}, fold {fold}"
                    )

                y_valid = validation[
                    response
                ].to_numpy(dtype=np.float64)

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
                )

                (
                    probability,
                    conditional,
                    expected,
                ) = fit_hurdle_model(
                    training=training,
                    validation=validation,
                    response=response,
                    config=config,
                    seed=seed,
                )

                fold_label = (
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
                        "Season": season,
                        "Season_Label":
                            season_label,
                        "Response": response,
                        "Fold": fold,
                        "Validation_Label":
                            fold_label,
                        "Occurrence_Config":
                            config[
                                "occurrence_name"
                            ],
                        "Severity_Config":
                            config[
                                "severity_name"
                            ],
                        "Training_Rows":
                            len(training),
                        "Validation_Rows":
                            len(validation),
                        **calculate_metrics(
                            observed=y_valid,
                            expected_prediction=expected,
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
                    f"{response}, fold {fold} completed."
                )

    return (
        pd.DataFrame(metric_rows),
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
    if len(coordinates) <= k:
        raise ValueError(
            "Not enough coordinates for requested k."
        )

    tree = cKDTree(coordinates)
    distances, neighbors = tree.query(
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

    weights = sparse.csr_matrix(
        (
            values,
            (rows, columns),
        ),
        shape=(
            len(coordinates),
            len(coordinates),
        ),
    )

    return weights


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

    valid = np.isfinite(residuals)

    if not valid.all():
        raise ValueError(
            "Residual array contains non-finite values."
        )

    z = residuals - residuals.mean()
    denominator = float(np.dot(z, z))

    if denominator <= 0:
        return {
            "Moran_I": np.nan,
            "Expected_I": np.nan,
            "Permutation_P_TwoSided": np.nan,
            "Permutation_Mean": np.nan,
            "Permutation_SD": np.nan,
        }

    observed_i = float(
        np.dot(z, weights.dot(z))
        / denominator
    )

    expected_i = -1.0 / (
        len(z) - 1
    )

    rng = np.random.default_rng(seed)
    permutation_values = np.empty(
        permutations,
        dtype=float,
    )

    for index in range(permutations):
        permuted = rng.permutation(z)

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
        "Permutation_Mean": float(
            permutation_values.mean()
        ),
        "Permutation_SD": float(
            permutation_values.std(ddof=1)
        ),
    }


def run_residual_moran(
    temporal_predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for season, season_label in SEASONS.items():
        for response in RESPONSES:
            subset = temporal_predictions.loc[
                (
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
                        "Country",
                        "GRID_ID",
                        *COORDINATE_COLUMNS,
                    ],
                    as_index=False,
                )
                .agg(
                    Mean_Residual=("Residual", "mean"),
                    Mean_Observed=("Observed", "mean"),
                    Mean_Predicted=(
                        "Expected_Prediction",
                        "mean",
                    ),
                    Years=("Year", "nunique"),
                )
                .sort_values("GRID_UID")
                .reset_index(drop=True)
            )

            coordinates = grid_residual[
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
                residuals=grid_residual[
                    "Mean_Residual"
                ].to_numpy(dtype=float),
                weights=weights,
                permutations=MORAN_PERMUTATIONS,
                seed=(
                    RANDOM_SEED
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
                    "Season": season,
                    "Season_Label":
                        season_label,
                    "Response": response,
                    "GRID_UID_Count":
                        len(grid_residual),
                    "Residual_Aggregation":
                        "Mean temporal OOF residual by GRID_UID",
                    "Weights":
                        f"{MORAN_K_NEIGHBORS}-nearest-neighbor row-standardized",
                    "Permutations":
                        MORAN_PERMUTATIONS,
                    **result,
                }
            )

            grid_residual.to_csv(
                OUTPUT_ROOT
                / (
                    f"Temporal_OOF_Grid_Mean_Residuals_"
                    f"S{season}_{response}.csv"
                ),
                index=False,
                encoding="utf-8-sig",
            )

    return pd.DataFrame(rows)


# =============================================================================
# Summaries
# =============================================================================

def summarize_metrics(
    fold_metrics: pd.DataFrame,
) -> pd.DataFrame:
    excluded = {
        "Validation_Scheme",
        "Season",
        "Season_Label",
        "Response",
        "Fold",
        "Validation_Label",
        "Occurrence_Config",
        "Severity_Config",
    }

    numeric_columns = [
        column
        for column in fold_metrics.select_dtypes(
            include=[np.number]
        ).columns
        if column not in {
            "Season",
            "Fold",
            "Training_Rows",
            "Validation_Rows",
        }
    ]

    rows: List[Dict[str, object]] = []

    grouping = [
        "Validation_Scheme",
        "Season",
        "Season_Label",
        "Response",
        "Occurrence_Config",
        "Severity_Config",
    ]

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
    log("Starting locked final hurdle-RF validation")
    log(f"Input: {INPUT_CSV}")
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
        OUTPUT_ROOT / "Locked_Model_Configurations.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        {
            "Order": range(
                1,
                len(CONTINUOUS_PREDICTORS) + 1,
            ),
            "Predictor":
                CONTINUOUS_PREDICTORS,
        }
    ).to_csv(
        OUTPUT_ROOT / "Selected_Continuous_Predictors.csv",
        index=False,
        encoding="utf-8-sig",
    )

    temporal_metrics, temporal_predictions = (
        run_validation_scheme(
            data=data,
            scheme="Temporal",
        )
    )

    spatial_metrics, spatial_predictions = (
        run_validation_scheme(
            data=data,
            scheme="Spatial",
        )
    )

    all_metrics = pd.concat(
        [
            temporal_metrics,
            spatial_metrics,
        ],
        ignore_index=True,
    )

    all_predictions = pd.concat(
        [
            temporal_predictions,
            spatial_predictions,
        ],
        ignore_index=True,
    )

    all_metrics.to_csv(
        OUTPUT_ROOT / "Validation_Fold_Metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summarize_metrics(
        all_metrics
    ).to_csv(
        OUTPUT_ROOT / "Validation_Summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    all_predictions.to_csv(
        OUTPUT_ROOT
        / "All_Out_of_Fold_Predictions.csv.gz",
        index=False,
        compression="gzip",
    )

    temporal_predictions.to_csv(
        OUTPUT_ROOT
        / "Temporal_Out_of_Fold_Predictions.csv.gz",
        index=False,
        compression="gzip",
    )

    spatial_predictions.to_csv(
        OUTPUT_ROOT
        / "Spatial_Out_of_Fold_Predictions.csv.gz",
        index=False,
        compression="gzip",
    )

    moran_results = run_residual_moran(
        temporal_predictions
    )

    moran_results.to_csv(
        OUTPUT_ROOT
        / "Temporal_OOF_Residual_Morans_I.csv",
        index=False,
        encoding="utf-8-sig",
    )

    qa = pd.DataFrame(
        [
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
                "Total_model_column_count",
                len(MODEL_PREDICTORS),
            ],
            ["Trees_per_RF", N_ESTIMATORS],
            [
                "Temporal_fold_count",
                len(TEMPORAL_FOLDS),
            ],
            [
                "Spatial_block_count",
                SPATIAL_BLOCK_COUNT,
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
        OUTPUT_ROOT / "Data_QA_Summary.csv",
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
        "Trees_per_RF": N_ESTIMATORS,
        "ForestArea_threshold_km2":
            FOREST_AREA_THRESHOLD_KM2,
        "Temporal_folds": TEMPORAL_FOLDS,
        "Spatial_blocks": {
            "method": (
                "Read and verify the single assignment generated upstream "
                "by 08_Regression_modeling Step 07; no KMeans block "
                "generation is performed in 09_RF_modeling"
            ),
            "source_file": str(SHARED_SPATIAL_BLOCK_FILE),
            "workflow_version": SHARED_SPATIAL_BLOCK_WORKFLOW_VERSION,
            "semantic_mapping_sha256": EXPECTED_SPATIAL_MAPPING_SHA256,
            "file_sha256": EXPECTED_SPATIAL_BLOCK_FILE_SHA256,
            "count": SPATIAL_BLOCK_COUNT,
            "all_years_of_each_GRID_UID_in_same_block": True,
        },
        "Residual_Morans_I": {
            "residuals": (
                "Mean temporal out-of-fold residual "
                "by GRID_UID"
            ),
            "weights": (
                f"{MORAN_K_NEIGHBORS}-nearest-neighbor "
                "row-standardized"
            ),
            "permutations":
                MORAN_PERMUTATIONS,
            "p_value":
                "Two-sided permutation p-value",
        },
        "Important_note": (
            "Hyperparameters were selected in a previous "
            "temporal-CV tuning step. Temporal validation here "
            "uses the same year folds and should be interpreted "
            "as locked cross-validated performance rather than "
            "a fully independent nested-CV estimate. Spatial "
            "block validation provides a distinct assessment "
            "of transfer to unseen locations."
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
    log("Final validation completed successfully.")
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
            print(error_text, file=sys.stderr)

        raise
