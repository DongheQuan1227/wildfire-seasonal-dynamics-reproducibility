# -*- coding: utf-8 -*-
"""
Model Comparison Step 04: compare residual spatial autocorrelation.

This script reads the locked regression Step 08 and final Hurdle RF Step 10
Moran outputs. It does not refit a model, regenerate OOF predictions, rebuild
spatial weights, or rerun a permutation test.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# Paths and locked versions
# =============================================================================

CODE_ROOT = Path(
    os.environ.get(
        "MODEL_COMPARISON_CODE_ROOT",
        str(Path(__file__).resolve().parents[2]),
    )
).resolve()
MODEL_COMPARISON_ROOT = CODE_ROOT / "10_Model_comparison"

REGRESSION_ROOT = (
    CODE_ROOT
    / "08_Regression_modeling"
    / "08_Regression_Residual_Spatial_Autocorrelation"
)
RF_ROOT = (
    CODE_ROOT
    / "09_RF_modeling"
    / "10_Final_Hurdle_RF_Residual_Spatial_Autocorrelation"
)
STEP02_ROOT = MODEL_COMPARISON_ROOT / "02_Unified_Prediction_Metrics"
STEP03_ROOT = MODEL_COMPARISON_ROOT / "03_Paired_Cluster_Bootstrap"

OUTPUT_ROOT = (
    MODEL_COMPARISON_ROOT
    / "04_Residual_Spatial_Autocorrelation_Comparison"
)
LOG_FILE = OUTPUT_ROOT / "residual_spatial_autocorrelation_comparison.log"

CODE_VERSION = "2026-06-30_MODEL_COMPARISON_RESIDUAL_MORAN_V1"
EXPECTED_REGRESSION_VERSION = (
    "2026-07-08_RELEASE_OPTIONAL_LEGACY_FOLD_COMPATIBILITY"
)
EXPECTED_RF_VERSION = (
    "2026-06-30_RF_UNIFIED_TEMPORAL_SPATIAL_OOF_MORAN_"
    "V1B_UINT32_SEED_AUDIT"
)
EXPECTED_STEP02_VERSION = (
    "2026-06-30_MODEL_COMPARISON_UNIFIED_METRICS_"
    "V1A_ZERO_POSITIVE_FOLD_COMPATIBILITY"
)
EXPECTED_STEP03_VERSION = (
    "2026-06-30_MODEL_COMPARISON_PAIRED_GRID_CLUSTER_BOOTSTRAP_V1"
)
EXPECTED_SPATIAL_MAPPING_SHA256 = (
    "715f9fe342eec78d936d223a98f5a4d0378281d3a3fa420ed250ba07fe056cfe"
)

MODELS = ["Baseline_NB1", "Final_Regression", "Final_Hurdle_RF"]
VALIDATION_SCHEMES = ["Temporal", "Spatial"]
SEASONS = {1: "Spring", 2: "Summer", 3: "Autumn"}
RESPONSES = ["Fire_Count", "Burned_Pixel_Count"]
MODEL_PAIRS = [
    ("Final_Regression_vs_Baseline_NB1", "Final_Regression", "Baseline_NB1"),
    ("Final_Hurdle_RF_vs_Baseline_NB1", "Final_Hurdle_RF", "Baseline_NB1"),
    ("Final_Hurdle_RF_vs_Final_Regression", "Final_Hurdle_RF", "Final_Regression"),
]

PRIMARY_FILE = "05_Primary_Main_Sample_Morans_I.csv"
COMPLETE_FILE = "06_Complete_25Y_Sensitivity_Morans_I.csv"
KNN_FILE = "07_KNN_Sensitivity_Morans_I.csv"
REGRESSION_MANIFEST_FILE = "11_Output_Manifest.csv"
RF_MANIFEST_FILE = "12_Output_Manifest.csv"

STEP02_OVERALL_FILE = STEP02_ROOT / "02_Overall_Unified_Prediction_Metrics.csv"
STEP02_METHOD_FILE = STEP02_ROOT / "00_Method_Definition.json"
STEP02_MANIFEST_FILE = STEP02_ROOT / "09_Output_Manifest.csv"
STEP03_BOOTSTRAP_FILE = STEP03_ROOT / "03_Paired_Bootstrap_Difference_Summary.csv"
STEP03_METHOD_FILE = STEP03_ROOT / "00_Method_Definition.json"
STEP03_MANIFEST_FILE = STEP03_ROOT / "12_Output_Manifest.csv"

HASH_CHUNK_SIZE = 1024 * 1024


# =============================================================================
# Helpers
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


def bh_adjust(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    adjusted = np.full(array.shape, np.nan, dtype=float)
    valid = np.isfinite(array)
    if not valid.any():
        return adjusted

    p = array[valid]
    order = np.argsort(p, kind="mergesort")
    ranked = p[order]
    n = len(ranked)
    raw = ranked * n / np.arange(1, n + 1, dtype=float)
    monotone = np.minimum.accumulate(raw[::-1])[::-1]
    monotone = np.clip(monotone, 0.0, 1.0)
    restored = np.empty(n, dtype=float)
    restored[order] = monotone
    adjusted[valid] = restored
    return adjusted


def parse_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n", "", "nan", "none"}:
        return False
    raise ValueError(f"Cannot parse boolean: {value!r}")


def add_audit(
    rows: List[Dict[str, object]],
    category: str,
    check: str,
    passed: bool,
    observed: object,
    expected: object,
    detail: str = "",
) -> None:
    rows.append(
        {
            "Category": category,
            "Check": check,
            "Passed": bool(passed),
            "Observed": observed,
            "Expected": expected,
            "Detail": detail,
        }
    )


def verify_manifest_file(
    root: Path,
    manifest_path: Path,
    required_names: Sequence[str],
    source_label: str,
) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    relative_col = "Relative_Path"
    hash_col = "SHA256"
    file_col = "Output_File" if "Output_File" in manifest.columns else "File"

    rows: List[Dict[str, object]] = []
    for name in required_names:
        match = manifest.loc[
            (manifest[file_col].astype(str) == name)
            | (manifest[relative_col].astype(str) == name)
        ]
        path = root / name
        if len(match) != 1 or not path.exists():
            rows.append(
                {
                    "Source": source_label,
                    "File": name,
                    "Exists": path.exists(),
                    "Manifest_Row_Count": len(match),
                    "Expected_SHA256": "",
                    "Actual_SHA256": sha256_file(path) if path.exists() else "",
                    "Passed": False,
                }
            )
            continue
        expected_hash = str(match.iloc[0][hash_col])
        actual_hash = sha256_file(path)
        rows.append(
            {
                "Source": source_label,
                "File": name,
                "Exists": True,
                "Manifest_Row_Count": 1,
                "Expected_SHA256": expected_hash,
                "Actual_SHA256": actual_hash,
                "Passed": actual_hash == expected_hash,
            }
        )
    return pd.DataFrame(rows)


def expected_key_set(models: Sequence[str]) -> set[Tuple[object, ...]]:
    return {
        (scheme, model, season, response)
        for scheme in VALIDATION_SCHEMES
        for model in models
        for season in SEASONS
        for response in RESPONSES
    }


def validate_moran_table(
    data: pd.DataFrame,
    expected_models: Sequence[str],
    expected_rows: int,
    expected_sample: str,
    expected_k: int,
    expected_permutations: int,
    table_name: str,
) -> None:
    required = {
        "Sample_Definition",
        "Validation_Scheme",
        "Model_Role",
        "Season_Code",
        "Season_Label",
        "Count_Response",
        "K_Neighbors",
        "Permutations",
        "Permutation_Seed",
        "GRID_UID_Count",
        "Residual_Mean_Absolute_Value",
        "Residual_Root_Mean_Square",
        "Moran_I",
        "Expected_I",
        "Permutation_P_TwoSided",
        "Autocorrelation_Direction",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"{table_name} missing columns: {missing}")
    if len(data) != expected_rows:
        raise ValueError(
            f"{table_name}: expected {expected_rows} rows, observed {len(data)}."
        )
    if set(data["Model_Role"].astype(str)) != set(expected_models):
        raise ValueError(f"{table_name}: unexpected model roles.")
    if set(data["Sample_Definition"].astype(str)) != {expected_sample}:
        raise ValueError(f"{table_name}: unexpected sample definition.")
    if set(data["K_Neighbors"].astype(int)) != {expected_k}:
        raise ValueError(f"{table_name}: unexpected k.")
    if set(data["Permutations"].astype(int)) != {expected_permutations}:
        raise ValueError(f"{table_name}: unexpected permutation count.")
    observed_keys = set(
        zip(
            data["Validation_Scheme"].astype(str),
            data["Model_Role"].astype(str),
            data["Season_Code"].astype(int),
            data["Count_Response"].astype(str),
        )
    )
    if observed_keys != expected_key_set(expected_models):
        raise ValueError(f"{table_name}: incomplete or duplicated key universe.")


def add_combined_fdr(data: pd.DataFrame, family_prefix: str) -> pd.DataFrame:
    result = data.copy()
    p_col = "Permutation_P_TwoSided"
    result[f"{family_prefix}_Global_BH_FDR"] = bh_adjust(result[p_col])
    result[f"{family_prefix}_Validation_Scheme_BH_FDR"] = (
        result.groupby("Validation_Scheme", sort=False)[p_col]
        .transform(lambda values: bh_adjust(values))
    )
    result[f"{family_prefix}_Scheme_Model_BH_FDR"] = (
        result.groupby(["Validation_Scheme", "Model_Role"], sort=False)[p_col]
        .transform(lambda values: bh_adjust(values))
    )
    result[f"{family_prefix}_Global_Significant_0p05"] = (
        result[f"{family_prefix}_Global_BH_FDR"] <= 0.05
    )
    result[f"{family_prefix}_Validation_Scheme_Significant_0p05"] = (
        result[f"{family_prefix}_Validation_Scheme_BH_FDR"] <= 0.05
    )
    result[f"{family_prefix}_Scheme_Model_Significant_0p05"] = (
        result[f"{family_prefix}_Scheme_Model_BH_FDR"] <= 0.05
    )
    return result


def paired_moran_differences(data: pd.DataFrame, sample_label: str) -> pd.DataFrame:
    index_columns = [
        "Validation_Scheme",
        "Season_Code",
        "Season_Label",
        "Count_Response",
        "Rate_Scale_Name",
    ]
    value_columns = [
        "Moran_I",
        "Expected_I",
        "Moran_Excess_Over_Expected",
        "Residual_Mean_Absolute_Value",
        "Residual_Root_Mean_Square",
        "Permutation_P_TwoSided",
        "GRID_UID_Count",
        "K_Neighbors",
        "Permutations",
    ]
    wide = data.pivot(index=index_columns, columns="Model_Role", values=value_columns)
    rows: List[Dict[str, object]] = []
    for index_values, record in wide.iterrows():
        key = dict(zip(index_columns, index_values))
        for pair_name, candidate, reference in MODEL_PAIRS:
            candidate_i = float(record[("Moran_I", candidate)])
            reference_i = float(record[("Moran_I", reference)])
            difference = candidate_i - reference_i
            percent_reduction = (
                np.nan
                if abs(reference_i) <= np.finfo(float).tiny
                else 100.0 * (reference_i - candidate_i) / abs(reference_i)
            )
            if difference < 0:
                conclusion = "Candidate_lower_residual_Moran_I"
            elif difference > 0:
                conclusion = "Reference_lower_residual_Moran_I"
            else:
                conclusion = "Equal_residual_Moran_I"

            row: Dict[str, object] = {
                **key,
                "Sample_Definition": sample_label,
                "Model_Pair": pair_name,
                "Candidate_Model": candidate,
                "Reference_Model": reference,
                "Candidate_Moran_I": candidate_i,
                "Reference_Moran_I": reference_i,
                "Candidate_Minus_Reference_Moran_I": difference,
                "Moran_I_Percent_Reduction_Positive_Favors_Candidate": percent_reduction,
                "Descriptive_Moran_Conclusion": conclusion,
                "Moran_Difference_Inference": (
                    "Descriptive paired difference; no confidence interval is assigned."
                ),
            }
            for metric in [
                "Residual_Mean_Absolute_Value",
                "Residual_Root_Mean_Square",
            ]:
                candidate_value = float(record[(metric, candidate)])
                reference_value = float(record[(metric, reference)])
                row[f"Candidate_{metric}"] = candidate_value
                row[f"Reference_{metric}"] = reference_value
                row[f"Candidate_Minus_Reference_{metric}"] = (
                    candidate_value - reference_value
                )
            row["Candidate_Permutation_P_TwoSided"] = float(
                record[("Permutation_P_TwoSided", candidate)]
            )
            row["Reference_Permutation_P_TwoSided"] = float(
                record[("Permutation_P_TwoSided", reference)]
            )
            row["GRID_UID_Count"] = int(record[("GRID_UID_Count", candidate)])
            row["K_Neighbors"] = int(record[("K_Neighbors", candidate)])
            row["Permutations"] = int(record[("Permutations", candidate)])
            rows.append(row)
    return pd.DataFrame(rows)


def build_all_k_table(
    primary: pd.DataFrame,
    complete: pd.DataFrame,
    sensitivity: pd.DataFrame,
) -> pd.DataFrame:
    selected_columns = [
        "Analysis_Type",
        "Sample_Definition",
        "Validation_Scheme",
        "Model_Role",
        "Season_Code",
        "Season_Label",
        "Count_Response",
        "Rate_Scale_Name",
        "K_Neighbors",
        "Permutations",
        "Permutation_Seed",
        "GRID_UID_Count",
        "Residual_Mean_Absolute_Value",
        "Residual_Root_Mean_Square",
        "Moran_I",
        "Expected_I",
        "Moran_Excess_Over_Expected",
        "Permutation_P_TwoSided",
        "Autocorrelation_Direction",
    ]
    all_k = pd.concat(
        [
            primary[selected_columns],
            complete[selected_columns],
            sensitivity[selected_columns],
        ],
        ignore_index=True,
    )
    all_k["Combined_Sample_K_BH_FDR"] = (
        all_k.groupby(["Sample_Definition", "K_Neighbors"], sort=False)[
            "Permutation_P_TwoSided"
        ].transform(lambda values: bh_adjust(values))
    )
    all_k["Combined_Sample_Scheme_K_BH_FDR"] = (
        all_k.groupby(
            ["Sample_Definition", "Validation_Scheme", "K_Neighbors"],
            sort=False,
        )["Permutation_P_TwoSided"].transform(lambda values: bh_adjust(values))
    )
    all_k["Combined_Sample_K_Significant_0p05"] = (
        all_k["Combined_Sample_K_BH_FDR"] <= 0.05
    )
    all_k["Moran_I_Rank_Lower_Is_Better"] = (
        all_k.groupby(
            [
                "Sample_Definition",
                "Validation_Scheme",
                "Season_Code",
                "Count_Response",
                "K_Neighbors",
            ]
        )["Moran_I"]
        .rank(ascending=True, method="min")
        .astype(int)
    )
    return all_k.sort_values(
        [
            "Sample_Definition",
            "Validation_Scheme",
            "Season_Code",
            "Count_Response",
            "K_Neighbors",
            "Model_Role",
        ]
    ).reset_index(drop=True)


def build_knn_stability(all_k: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, object]] = []
    group_columns = [
        "Sample_Definition",
        "Validation_Scheme",
        "Model_Role",
        "Season_Code",
        "Season_Label",
        "Count_Response",
        "Rate_Scale_Name",
    ]
    for key, group in all_k.groupby(group_columns, sort=True):
        group = group.sort_values("K_Neighbors")
        by_k = group.set_index("K_Neighbors")
        if set(by_k.index.astype(int)) != {4, 8, 12}:
            raise ValueError(f"Missing k setting for {key}.")
        morans = by_k["Moran_I"].astype(float)
        ranks = by_k["Moran_I_Rank_Lower_Is_Better"].astype(int)
        rows.append(
            {
                **dict(zip(group_columns, key)),
                "K4_Moran_I": float(morans.loc[4]),
                "K8_Moran_I": float(morans.loc[8]),
                "K12_Moran_I": float(morans.loc[12]),
                "Moran_I_Minimum_Across_K": float(morans.min()),
                "Moran_I_Maximum_Across_K": float(morans.max()),
                "Moran_I_Range_Across_K": float(morans.max() - morans.min()),
                "Moran_I_Direction_Consistent": bool(
                    (group["Autocorrelation_Direction"].astype(str) == "Positive").all()
                    or (group["Autocorrelation_Direction"].astype(str) == "Negative").all()
                ),
                "All_K_Combined_FDR_Significant_0p05": bool(
                    group["Combined_Sample_K_Significant_0p05"].all()
                ),
                "K4_Model_Rank": int(ranks.loc[4]),
                "K8_Model_Rank": int(ranks.loc[8]),
                "K12_Model_Rank": int(ranks.loc[12]),
                "Model_Rank_Constant_Across_K": len(set(ranks.tolist())) == 1,
            }
        )
    stability = pd.DataFrame(rows)

    winner_rows: List[Dict[str, object]] = []
    winner_groups = [
        "Sample_Definition",
        "Validation_Scheme",
        "Season_Code",
        "Season_Label",
        "Count_Response",
        "Rate_Scale_Name",
    ]
    for key, group in all_k.groupby(winner_groups, sort=True):
        winners: Dict[int, str] = {}
        for k, k_group in group.groupby("K_Neighbors"):
            minimum = k_group["Moran_I"].min()
            names = sorted(
                k_group.loc[
                    np.isclose(k_group["Moran_I"], minimum, rtol=0, atol=1e-15),
                    "Model_Role",
                ].astype(str)
            )
            winners[int(k)] = ";".join(names)
        winner_rows.append(
            {
                **dict(zip(winner_groups, key)),
                "K4_Lowest_Moran_Model": winners[4],
                "K8_Lowest_Moran_Model": winners[8],
                "K12_Lowest_Moran_Model": winners[12],
                "Lowest_Moran_Model_Consistent_Across_K": len(set(winners.values())) == 1,
            }
        )
    return stability, pd.DataFrame(winner_rows)


def build_time_support_summary(
    primary: pd.DataFrame,
    complete: pd.DataFrame,
) -> pd.DataFrame:
    keys = [
        "Validation_Scheme",
        "Model_Role",
        "Season_Code",
        "Season_Label",
        "Count_Response",
        "Rate_Scale_Name",
    ]
    main = primary[
        keys
        + [
            "GRID_UID_Count",
            "Moran_I",
            "Expected_I",
            "Permutation_P_TwoSided",
            "Residual_Mean_Absolute_Value",
            "Residual_Root_Mean_Square",
        ]
    ].rename(
        columns={
            "GRID_UID_Count": "Main_GRID_UID_Count",
            "Moran_I": "Main_Moran_I",
            "Expected_I": "Main_Expected_I",
            "Permutation_P_TwoSided": "Main_Permutation_P_TwoSided",
            "Residual_Mean_Absolute_Value": "Main_Residual_Mean_Absolute_Value",
            "Residual_Root_Mean_Square": "Main_Residual_Root_Mean_Square",
        }
    )
    comp = complete[
        keys
        + [
            "GRID_UID_Count",
            "Moran_I",
            "Expected_I",
            "Permutation_P_TwoSided",
            "Residual_Mean_Absolute_Value",
            "Residual_Root_Mean_Square",
        ]
    ].rename(
        columns={
            "GRID_UID_Count": "Complete25Y_GRID_UID_Count",
            "Moran_I": "Complete25Y_Moran_I",
            "Expected_I": "Complete25Y_Expected_I",
            "Permutation_P_TwoSided": "Complete25Y_Permutation_P_TwoSided",
            "Residual_Mean_Absolute_Value": "Complete25Y_Residual_Mean_Absolute_Value",
            "Residual_Root_Mean_Square": "Complete25Y_Residual_Root_Mean_Square",
        }
    )
    result = main.merge(comp, on=keys, how="inner", validate="one_to_one")
    result["Complete25Y_minus_Main_Moran_I"] = (
        result["Complete25Y_Moran_I"] - result["Main_Moran_I"]
    )
    result["Moran_Direction_Consistent"] = (
        np.sign(result["Complete25Y_Moran_I"])
        == np.sign(result["Main_Moran_I"])
    )
    return result


def build_temporal_spatial_summary(
    primary: pd.DataFrame,
    complete: pd.DataFrame,
) -> pd.DataFrame:
    combined = pd.concat([primary, complete], ignore_index=True)
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
    value_columns = [
        "Moran_I",
        "Expected_I",
        "Permutation_P_TwoSided",
        "Residual_Mean_Absolute_Value",
        "Residual_Root_Mean_Square",
    ]
    wide = combined.pivot(index=keys, columns="Validation_Scheme", values=value_columns)
    rows: List[Dict[str, object]] = []
    for index_values, record in wide.iterrows():
        row = dict(zip(keys, index_values))
        for metric in value_columns:
            row[f"Temporal_{metric}"] = float(record[(metric, "Temporal")])
            row[f"Spatial_{metric}"] = float(record[(metric, "Spatial")])
        row["Spatial_minus_Temporal_Moran_I"] = (
            row["Spatial_Moran_I"] - row["Temporal_Moran_I"]
        )
        row["Spatial_minus_Temporal_Residual_RMSE"] = (
            row["Spatial_Residual_Root_Mean_Square"]
            - row["Temporal_Residual_Root_Mean_Square"]
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_joint_summary(primary: pd.DataFrame, overall: pd.DataFrame) -> pd.DataFrame:
    prediction_columns = [
        "Validation_Scheme",
        "Season",
        "Season_Label",
        "Count_Response",
        "Rate_Scale_Name",
        "Model",
        "Rate_RMSE",
        "Rate_MAE",
        "Rate_R2",
        "Rate_Spearman",
        "Rate_Mean_Bias",
        "Occurrence_Brier_Score",
        "Occurrence_PR_AUC",
        "Occurrence_ROC_AUC",
        "Positive_Conditional_Rate_RMSE",
        "Positive_Conditional_Rate_MAE",
        "Occurrence_Expected_Calibration_Error",
        "Rate_Binned_Calibration_MAE",
    ]
    prediction = overall[prediction_columns].rename(
        columns={"Season": "Season_Code", "Model": "Model_Role"}
    )
    moran_columns = [
        "Validation_Scheme",
        "Season_Code",
        "Season_Label",
        "Count_Response",
        "Rate_Scale_Name",
        "Model_Role",
        "Sample_Definition",
        "GRID_UID_Count",
        "K_Neighbors",
        "Permutations",
        "Moran_I",
        "Expected_I",
        "Permutation_P_TwoSided",
        "Combined_Primary_Global_BH_FDR",
        "Residual_Mean_Absolute_Value",
        "Residual_Root_Mean_Square",
    ]
    joint = prediction.merge(
        primary[moran_columns],
        on=[
            "Validation_Scheme",
            "Season_Code",
            "Season_Label",
            "Count_Response",
            "Rate_Scale_Name",
            "Model_Role",
        ],
        how="inner",
        validate="one_to_one",
    )
    group = ["Validation_Scheme", "Season_Code", "Count_Response"]
    for metric, lower in [
        ("Rate_RMSE", True),
        ("Rate_MAE", True),
        ("Occurrence_Brier_Score", True),
        ("Moran_I", True),
        ("Rate_Spearman", False),
    ]:
        joint[f"{metric}_Rank"] = (
            joint.groupby(group)[metric]
            .rank(ascending=lower, method="min")
            .astype(int)
        )
    joint["Lower_Moran_and_Lower_RMSE"] = (
        (joint["Moran_I_Rank"] == 1) & (joint["Rate_RMSE_Rank"] == 1)
    )
    return joint.sort_values(group + ["Model_Role"]).reset_index(drop=True)


def build_tradeoff_table(
    paired_moran: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> pd.DataFrame:
    metrics = [
        "Rate_RMSE",
        "Rate_MAE",
        "Occurrence_Brier_Score",
        "Positive_Conditional_Rate_RMSE",
        "Positive_Conditional_Rate_MAE",
    ]
    selected = bootstrap.loc[bootstrap["Metric"].isin(metrics)].copy()
    key_columns = [
        "Validation_Scheme",
        "Season",
        "Season_Label",
        "Count_Response",
        "Model_Pair",
        "Candidate_Model",
        "Reference_Model",
    ]
    value_columns = [
        "Candidate_Minus_Reference",
        "Difference_CI_Lower",
        "Difference_CI_Upper",
        "Probability_Candidate_Better",
        "Stable_Conclusion",
    ]
    wide = selected.pivot(index=key_columns, columns="Metric", values=value_columns)
    flattened = wide.copy()
    flattened.columns = [f"{metric}_{value}" for value, metric in flattened.columns]
    flattened = flattened.reset_index().rename(columns={"Season": "Season_Code"})

    moran_columns = [
        "Validation_Scheme",
        "Season_Code",
        "Season_Label",
        "Count_Response",
        "Model_Pair",
        "Candidate_Model",
        "Reference_Model",
        "Candidate_Moran_I",
        "Reference_Moran_I",
        "Candidate_Minus_Reference_Moran_I",
        "Moran_I_Percent_Reduction_Positive_Favors_Candidate",
        "Descriptive_Moran_Conclusion",
    ]
    result = paired_moran[moran_columns].merge(
        flattened,
        on=[
            "Validation_Scheme",
            "Season_Code",
            "Season_Label",
            "Count_Response",
            "Model_Pair",
            "Candidate_Model",
            "Reference_Model",
        ],
        how="inner",
        validate="one_to_one",
    )

    categories: List[str] = []
    for _, row in result.iterrows():
        prediction = row["Rate_RMSE_Stable_Conclusion"]
        moran_delta = float(row["Candidate_Minus_Reference_Moran_I"])
        if prediction == "Candidate_stably_better":
            categories.append(
                "Candidate_stably_lower_RMSE_and_lower_Moran"
                if moran_delta < 0
                else "Candidate_stably_lower_RMSE_but_higher_Moran"
            )
        elif prediction == "Reference_stably_better":
            categories.append(
                "Candidate_lower_Moran_but_reference_stably_lower_RMSE"
                if moran_delta < 0
                else "Reference_stably_lower_RMSE_and_lower_Moran"
            )
        else:
            categories.append(
                "No_stable_RMSE_difference_candidate_lower_Moran"
                if moran_delta < 0
                else "No_stable_RMSE_difference_reference_lower_Moran"
                if moran_delta > 0
                else "No_stable_RMSE_difference_equal_Moran"
            )
    result["Primary_RMSE_Moran_Tradeoff_Category"] = categories
    result["Interpretation_Note"] = (
        "RMSE stability is based on paired GRID_UID-cluster bootstrap; "
        "Moran difference is descriptive and has no confidence interval."
    )
    return result


def write_manifest() -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for path in sorted(OUTPUT_ROOT.iterdir()):
        if path.is_file() and path.name != "13_Output_Manifest.csv":
            rows.append(
                {
                    "File": path.name,
                    "Relative_Path": path.name,
                    "File_Size_Bytes": int(path.stat().st_size),
                    "SHA256": sha256_file(path),
                    "Step04_Code_Version": CODE_VERSION,
                }
            )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(
        OUTPUT_ROOT / "13_Output_Manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return manifest


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text("", encoding="utf-8")

    log("Starting unified residual-spatial-autocorrelation model comparison.")
    log(f"Step 04 code version: {CODE_VERSION}.")
    log("No model, OOF prediction, spatial weight, or Moran permutation is recomputed.")
    log("Temporal and spatial OOF are compared separately.")

    required_paths = [
        REGRESSION_ROOT / "00_Method_Definition.json",
        REGRESSION_ROOT / REGRESSION_MANIFEST_FILE,
        REGRESSION_ROOT / PRIMARY_FILE,
        REGRESSION_ROOT / COMPLETE_FILE,
        REGRESSION_ROOT / KNN_FILE,
        RF_ROOT / "00_Method_Definition.json",
        RF_ROOT / RF_MANIFEST_FILE,
        RF_ROOT / PRIMARY_FILE,
        RF_ROOT / COMPLETE_FILE,
        RF_ROOT / KNN_FILE,
        RF_ROOT / "11_Regression_Protocol_Alignment_Audit.csv",
        RF_ROOT / "11b_Primary_Permutation_Seed_Alignment_Audit.csv",
        STEP02_METHOD_FILE,
        STEP02_MANIFEST_FILE,
        STEP02_OVERALL_FILE,
        STEP03_METHOD_FILE,
        STEP03_MANIFEST_FILE,
        STEP03_BOOTSTRAP_FILE,
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Required inputs are missing:\n" + "\n".join(missing))

    audit_rows: List[Dict[str, object]] = []

    regression_method = read_json(REGRESSION_ROOT / "00_Method_Definition.json")
    rf_method = read_json(RF_ROOT / "00_Method_Definition.json")
    step02_method = read_json(STEP02_METHOD_FILE)
    step03_method = read_json(STEP03_METHOD_FILE)

    add_audit(
        audit_rows,
        "Version",
        "Regression_Step08_version",
        regression_method.get("Step08_Code_Version") == EXPECTED_REGRESSION_VERSION,
        regression_method.get("Step08_Code_Version"),
        EXPECTED_REGRESSION_VERSION,
    )
    add_audit(
        audit_rows,
        "Version",
        "RF_Step10_version",
        rf_method.get("Step10_Code_Version") == EXPECTED_RF_VERSION,
        rf_method.get("Step10_Code_Version"),
        EXPECTED_RF_VERSION,
    )
    add_audit(
        audit_rows,
        "Version",
        "Step02_version",
        step02_method.get("Code_Version") == EXPECTED_STEP02_VERSION,
        step02_method.get("Code_Version"),
        EXPECTED_STEP02_VERSION,
    )
    add_audit(
        audit_rows,
        "Version",
        "Step03_version",
        step03_method.get("Code_Version") == EXPECTED_STEP03_VERSION,
        step03_method.get("Code_Version"),
        EXPECTED_STEP03_VERSION,
    )

    regression_mapping = pd.read_csv(
        REGRESSION_ROOT / REGRESSION_MANIFEST_FILE
    )["Spatial_Mapping_SHA256"].astype(str).unique().tolist()
    rf_mapping = pd.read_csv(
        RF_ROOT / RF_MANIFEST_FILE
    )["Spatial_Mapping_SHA256"].astype(str).unique().tolist()
    add_audit(
        audit_rows,
        "Spatial_Protocol",
        "Shared_spatial_mapping_hash",
        regression_mapping == [EXPECTED_SPATIAL_MAPPING_SHA256]
        and rf_mapping == [EXPECTED_SPATIAL_MAPPING_SHA256],
        f"Regression={regression_mapping}; RF={rf_mapping}",
        EXPECTED_SPATIAL_MAPPING_SHA256,
    )

    hash_audits = pd.concat(
        [
            verify_manifest_file(
                REGRESSION_ROOT,
                REGRESSION_ROOT / REGRESSION_MANIFEST_FILE,
                ["00_Method_Definition.json", PRIMARY_FILE, COMPLETE_FILE, KNN_FILE],
                "Regression_Step08",
            ),
            verify_manifest_file(
                RF_ROOT,
                RF_ROOT / RF_MANIFEST_FILE,
                ["00_Method_Definition.json", PRIMARY_FILE, COMPLETE_FILE, KNN_FILE],
                "RF_Step10",
            ),
            verify_manifest_file(
                STEP02_ROOT,
                STEP02_MANIFEST_FILE,
                ["00_Method_Definition.json", "02_Overall_Unified_Prediction_Metrics.csv"],
                "Model_Comparison_Step02",
            ),
            verify_manifest_file(
                STEP03_ROOT,
                STEP03_MANIFEST_FILE,
                ["00_Method_Definition.json", "03_Paired_Bootstrap_Difference_Summary.csv"],
                "Model_Comparison_Step03",
            ),
        ],
        ignore_index=True,
    )
    add_audit(
        audit_rows,
        "Hashes",
        "All_required_upstream_hashes_match",
        bool(hash_audits["Passed"].all()),
        int(hash_audits["Passed"].sum()),
        len(hash_audits),
    )

    protocol_audit = pd.read_csv(
        RF_ROOT / "11_Regression_Protocol_Alignment_Audit.csv"
    )
    seed_audit = pd.read_csv(
        RF_ROOT / "11b_Primary_Permutation_Seed_Alignment_Audit.csv"
    )
    add_audit(
        audit_rows,
        "Spatial_Protocol",
        "RF_regression_protocol_alignment_passed",
        protocol_audit["Passed"].map(parse_bool).all(),
        int(protocol_audit["Passed"].map(parse_bool).sum()),
        len(protocol_audit),
    )
    add_audit(
        audit_rows,
        "Spatial_Protocol",
        "Primary_seed_alignment_passed",
        seed_audit["Passed"].map(parse_bool).all(),
        int(seed_audit["Passed"].map(parse_bool).sum()),
        len(seed_audit),
    )

    reg_primary = pd.read_csv(REGRESSION_ROOT / PRIMARY_FILE)
    rf_primary = pd.read_csv(RF_ROOT / PRIMARY_FILE)
    reg_complete = pd.read_csv(REGRESSION_ROOT / COMPLETE_FILE)
    rf_complete = pd.read_csv(RF_ROOT / COMPLETE_FILE)
    reg_knn = pd.read_csv(REGRESSION_ROOT / KNN_FILE)
    rf_knn = pd.read_csv(RF_ROOT / KNN_FILE)

    validate_moran_table(
        reg_primary,
        ["Baseline_NB1", "Final_Regression"],
        24,
        "Main_4982_Available_Years",
        8,
        9999,
        "Regression primary",
    )
    validate_moran_table(
        rf_primary,
        ["Final_Hurdle_RF"],
        12,
        "Main_4982_Available_Years",
        8,
        9999,
        "RF primary",
    )
    validate_moran_table(
        reg_complete,
        ["Baseline_NB1", "Final_Regression"],
        24,
        "Complete_4924_25_Years",
        8,
        9999,
        "Regression complete",
    )
    validate_moran_table(
        rf_complete,
        ["Final_Hurdle_RF"],
        12,
        "Complete_4924_25_Years",
        8,
        9999,
        "RF complete",
    )

    primary = pd.concat([reg_primary, rf_primary], ignore_index=True)
    complete = pd.concat([reg_complete, rf_complete], ignore_index=True)
    knn = pd.concat([reg_knn, rf_knn], ignore_index=True)

    validate_moran_table(
        primary,
        MODELS,
        36,
        "Main_4982_Available_Years",
        8,
        9999,
        "Combined primary",
    )
    validate_moran_table(
        complete,
        MODELS,
        36,
        "Complete_4924_25_Years",
        8,
        9999,
        "Combined complete",
    )
    if len(knn) != 144:
        raise ValueError(f"Combined kNN sensitivity expected 144 rows, got {len(knn)}.")
    expected_knn_keys = {
        (sample, scheme, model, season, response, k)
        for sample in ["Main_4982_Available_Years", "Complete_4924_25_Years"]
        for scheme in VALIDATION_SCHEMES
        for model in MODELS
        for season in SEASONS
        for response in RESPONSES
        for k in [4, 12]
    }
    observed_knn_keys = set(
        zip(
            knn["Sample_Definition"].astype(str),
            knn["Validation_Scheme"].astype(str),
            knn["Model_Role"].astype(str),
            knn["Season_Code"].astype(int),
            knn["Count_Response"].astype(str),
            knn["K_Neighbors"].astype(int),
        )
    )
    if observed_knn_keys != expected_knn_keys:
        raise ValueError("Combined kNN sensitivity key universe is incomplete.")

    # Same randomization and weight constants across the three models.
    for label, table in [("Primary", primary), ("Complete", complete), ("kNN", knn)]:
        group_columns = [
            "Sample_Definition",
            "Validation_Scheme",
            "Season_Code",
            "Count_Response",
            "K_Neighbors",
        ]
        consistent = table.groupby(group_columns).agg(
            Seed_N=("Permutation_Seed", "nunique"),
            Expected_I_N=("Expected_I", "nunique"),
            Grid_N=("GRID_UID_Count", "nunique"),
            Permutation_N=("Permutations", "nunique"),
        )
        passed = bool((consistent == 1).all().all())
        add_audit(
            audit_rows,
            "Moran_Alignment",
            f"{label}_same_seed_expectedI_grid_and_permutations_across_models",
            passed,
            int((consistent == 1).all(axis=1).sum()),
            len(consistent),
        )

    primary = add_combined_fdr(primary, "Combined_Primary")
    complete = add_combined_fdr(complete, "Combined_Complete25Y")

    primary_pairs = paired_moran_differences(
        primary, "Main_4982_Available_Years"
    )
    complete_pairs = paired_moran_differences(
        complete, "Complete_4924_25_Years"
    )
    all_k = build_all_k_table(primary, complete, knn)
    knn_stability, knn_winners = build_knn_stability(all_k)
    time_support = build_time_support_summary(primary, complete)
    temporal_spatial = build_temporal_spatial_summary(primary, complete)

    overall_metrics = pd.read_csv(STEP02_OVERALL_FILE)
    bootstrap = pd.read_csv(STEP03_BOOTSTRAP_FILE)
    if len(overall_metrics) != 36:
        raise ValueError("Step 02 overall metrics must contain 36 rows.")
    if len(bootstrap) != 432:
        raise ValueError("Step 03 bootstrap summary must contain 432 rows.")

    joint = build_joint_summary(primary, overall_metrics)
    tradeoff = build_tradeoff_table(primary_pairs, bootstrap)

    # Descriptive model wins under lower Moran I.
    rank_rows = primary[
        [
            "Validation_Scheme",
            "Season_Code",
            "Season_Label",
            "Count_Response",
            "Rate_Scale_Name",
            "Model_Role",
            "Moran_I",
        ]
    ].copy()
    rank_rows["Moran_I_Rank_Lower_Is_Better"] = (
        rank_rows.groupby(
            ["Validation_Scheme", "Season_Code", "Count_Response"]
        )["Moran_I"]
        .rank(ascending=True, method="min")
        .astype(int)
    )
    win_summary = (
        rank_rows.assign(Lowest_Moran_I=rank_rows["Moran_I_Rank_Lower_Is_Better"] == 1)
        .groupby(["Validation_Scheme", "Model_Role"], as_index=False)
        .agg(
            Comparison_N=("Lowest_Moran_I", "size"),
            Lowest_Moran_I_N=("Lowest_Moran_I", "sum"),
            Mean_Moran_I=("Moran_I", "mean"),
            Median_Moran_I=("Moran_I", "median"),
        )
    )

    add_audit(audit_rows, "Output", "Primary_rows", len(primary) == 36, len(primary), 36)
    add_audit(audit_rows, "Output", "Complete_rows", len(complete) == 36, len(complete), 36)
    add_audit(audit_rows, "Output", "Primary_pair_rows", len(primary_pairs) == 36, len(primary_pairs), 36)
    add_audit(audit_rows, "Output", "Complete_pair_rows", len(complete_pairs) == 36, len(complete_pairs), 36)
    add_audit(audit_rows, "Output", "All_k_rows", len(all_k) == 216, len(all_k), 216)
    add_audit(audit_rows, "Output", "KNN_stability_rows", len(knn_stability) == 72, len(knn_stability), 72)
    add_audit(audit_rows, "Output", "KNN_winner_rows", len(knn_winners) == 24, len(knn_winners), 24)
    add_audit(audit_rows, "Output", "Time_support_rows", len(time_support) == 36, len(time_support), 36)
    add_audit(audit_rows, "Output", "Temporal_spatial_rows", len(temporal_spatial) == 36, len(temporal_spatial), 36)
    add_audit(audit_rows, "Output", "Joint_summary_rows", len(joint) == 36, len(joint), 36)
    add_audit(audit_rows, "Output", "Tradeoff_rows", len(tradeoff) == 36, len(tradeoff), 36)

    audit = pd.DataFrame(audit_rows)
    if not audit["Passed"].all():
        audit.to_csv(
            OUTPUT_ROOT / "01_Input_Integrity_Audit.csv",
            index=False,
            encoding="utf-8-sig",
        )
        failed = audit.loc[~audit["Passed"]]
        raise ValueError("Step 04 input/output audit failed:\n" + failed.to_string(index=False))

    method = {
        "Step": "04_Compare_Residual_Spatial_Autocorrelation",
        "Code_Version": CODE_VERSION,
        "Created": datetime.now().isoformat(timespec="seconds"),
        "Purpose": (
            "Compare residual spatial autocorrelation among Baseline NB1, "
            "Final Regression, and Final Hurdle RF under locked temporal and "
            "spatial OOF protocols."
        ),
        "No_Model_Fitting": True,
        "No_OOF_Prediction_Recomputation": True,
        "No_Spatial_Weight_Recomputation": True,
        "No_Moran_Permutation_Recomputation": True,
        "No_Model_Selection": True,
        "Input_Versions": {
            "Regression_Step08": EXPECTED_REGRESSION_VERSION,
            "RF_Step10": EXPECTED_RF_VERSION,
            "Model_Comparison_Step02": EXPECTED_STEP02_VERSION,
            "Model_Comparison_Step03": EXPECTED_STEP03_VERSION,
        },
        "Models": MODELS,
        "Validation_Schemes": VALIDATION_SCHEMES,
        "Primary_Sample": {
            "Name": "Main_4982_Available_Years",
            "K_Neighbors": 8,
            "Permutations": 9999,
            "Combined_FDR": (
                "BH recomputed jointly across all 36 three-model primary tests; "
                "additional scheme and scheme-model families are reported."
            ),
        },
        "Complete_Time_Support_Sensitivity": {
            "Name": "Complete_4924_25_Years",
            "K_Neighbors": 8,
            "Permutations": 9999,
            "Combined_FDR": "BH recomputed jointly across all 36 tests.",
        },
        "KNN_Sensitivity": {
            "K_Neighbors": [4, 8, 12],
            "K4_K12_Permutations": 999,
            "K8_Permutations": 9999,
            "FDR_Family": (
                "BH within each sample definition and k across 36 model tests; "
                "a global 216-test family is intentionally not used because "
                "permutation resolutions differ between k=8 and k=4/12."
            ),
        },
        "Moran_Model_Difference": (
            "Candidate minus reference. Negative values indicate lower residual "
            "spatial autocorrelation for the candidate. Differences are descriptive; "
            "no unsupported confidence interval is assigned."
        ),
        "Joint_Interpretation": (
            "Prediction-error uncertainty comes from Step 03 paired GRID_UID-cluster "
            "bootstrap. Moran differences remain descriptive. Lower prediction error "
            "and lower residual Moran I are reported jointly rather than collapsed "
            "into a single score."
        ),
    }
    with (OUTPUT_ROOT / "00_Method_Definition.json").open("w", encoding="utf-8") as file:
        json.dump(method, file, ensure_ascii=False, indent=2)

    audit.to_csv(
        OUTPUT_ROOT / "01_Input_Integrity_Audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    hash_audits.to_csv(
        OUTPUT_ROOT / "01b_Upstream_Hash_Audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    primary.to_csv(
        OUTPUT_ROOT / "02_Combined_Primary_Main_Sample_Morans_I.csv",
        index=False,
        encoding="utf-8-sig",
    )
    primary_pairs.to_csv(
        OUTPUT_ROOT / "03_Primary_Paired_Model_Moran_Differences.csv",
        index=False,
        encoding="utf-8-sig",
    )
    complete.to_csv(
        OUTPUT_ROOT / "04_Combined_Complete25Y_Morans_I.csv",
        index=False,
        encoding="utf-8-sig",
    )
    complete_pairs.to_csv(
        OUTPUT_ROOT / "05_Complete25Y_Paired_Model_Moran_Differences.csv",
        index=False,
        encoding="utf-8-sig",
    )
    all_k.to_csv(
        OUTPUT_ROOT / "06_All_KNN_Settings_Morans_I.csv",
        index=False,
        encoding="utf-8-sig",
    )
    knn_stability.to_csv(
        OUTPUT_ROOT / "07_KNN_Stability_Summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    knn_winners.to_csv(
        OUTPUT_ROOT / "07b_KNN_Lowest_Moran_Model_Stability.csv",
        index=False,
        encoding="utf-8-sig",
    )
    time_support.to_csv(
        OUTPUT_ROOT / "08_Time_Support_Sensitivity_Summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    temporal_spatial.to_csv(
        OUTPUT_ROOT / "09_Temporal_vs_Spatial_OOF_Moran_Summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    joint.to_csv(
        OUTPUT_ROOT / "10_Prediction_Performance_and_Moran_Joint_Summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    tradeoff.to_csv(
        OUTPUT_ROOT / "11_Paired_Prediction_Moran_Tradeoff.csv",
        index=False,
        encoding="utf-8-sig",
    )
    rank_rows.to_csv(
        OUTPUT_ROOT / "12_Primary_Moran_Descriptive_Ranks.csv",
        index=False,
        encoding="utf-8-sig",
    )
    win_summary.to_csv(
        OUTPUT_ROOT / "12b_Primary_Moran_Lowest_Count_Summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    environment = {
        "Created": datetime.now().isoformat(timespec="seconds"),
        "Python": sys.version,
        "Platform": platform.platform(),
        "NumPy": np.__version__,
        "pandas": pd.__version__,
        "Code_Root": str(CODE_ROOT),
    }
    with (OUTPUT_ROOT / "Software_Environment.json").open("w", encoding="utf-8") as file:
        json.dump(environment, file, ensure_ascii=False, indent=2)

    log(
        "Step 04 completed: residual Moran results were jointly compared for "
        "Baseline NB1, Final Regression, and Final Hurdle RF under temporal and "
        "spatial OOF, including 4,982-grid primary, 4,924-grid complete-time, "
        "and k=4/8/12 sensitivity results, without recomputing any model or Moran test."
    )
    write_manifest()
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
