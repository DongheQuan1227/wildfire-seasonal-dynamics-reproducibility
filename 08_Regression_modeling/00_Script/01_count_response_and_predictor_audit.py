# -*- coding: utf-8 -*-
"""
Count-response and predictor audit for wildfire detection modeling.

Recommended public location
---------------------------
<CODE_ROOT>/08_Regression_modeling/00_Script/
    01_count_response_and_predictor_audit.py

Input
-----
<CODE_ROOT>/00_Input_data/Base_Table_2001_2025.csv.bz2

Output
------
<CODE_ROOT>/08_Regression_modeling/
    01_Count_Response_and_Predictor_Audit/

Fire_Count and Burned_Pixel_Count are treated as seasonal detection
counts. ForestPixelCount is the exposure variable for later count models:

    offset = log(ForestPixelCount)

FCD and BAD are calculated only as descriptive rate-scale quantities and
are not assumed to be bounded beta-distributed proportions. This script
fits no final model and uses no random-forest variable-selection result.
"""

from __future__ import annotations

import json
import math
import platform
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import stats


# =============================================================================
# Paths and settings
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REGRESSION_ROOT = SCRIPT_DIR.parent
CODE_ROOT = REGRESSION_ROOT.parent
INPUT_CSV = CODE_ROOT / "00_Input_data" / "Base_Table_2001_2025.csv.bz2"
OUTPUT_ROOT = REGRESSION_ROOT / "01_Count_Response_and_Predictor_Audit"
LOG_FILE = OUTPUT_ROOT / "count_response_predictor_audit.log"

RANDOM_SEED = 2026
FOREST_PIXEL_AREA_KM2 = 0.25
FOREST_AREA_THRESHOLD_KM2 = 2.5
INTEGER_TOLERANCE = 1e-9
CORRELATION_SAMPLE_MAX = 100_000
HIGH_CORRELATION_THRESHOLD = 0.80
VERY_HIGH_CORRELATION_THRESHOLD = 0.90
VIF_WARNING_THRESHOLD = 5.0
VIF_HIGH_THRESHOLD = 10.0
HIGH_MISSINGNESS_THRESHOLD = 0.20
SKEWNESS_TRANSFORM_THRESHOLD = 2.0
NEAR_ZERO_VARIANCE_TOLERANCE = 1e-12

SEASONS = {1: "Spring", 2: "Summer", 3: "Autumn"}
COUNT_RESPONSES = {"Fire_Count": "FCD", "Burned_Pixel_Count": "BAD"}
COUNTRY_LEVELS = {"China", "NK", "Russia"}

HARD_EXCLUDE_COLUMNS = {
    "GRID_UID", "GRID_ID", "ZONE_ID", "Country", "Year", "Season",
    "Fire_Count", "Burned_Pixel_Count", "FCD", "BAD", "Total_FRP_MW",
    "FRP", "Total_BA_km2", "BA_km2", "ForestPixelCount",
    "ForestArea_km2", "ForestArea", "Log_ForestPixelCount",
    "Centroid_X_UTM52_m", "Centroid_Y_UTM52_m", "Longitude", "Latitude",
    "X", "Y", "POINT_X", "POINT_Y", "Temporal_Fold", "Spatial_Block",
    "Fold", "Prediction", "Predicted", "Residual",
}

KNOWN_GROUPS = {
    "Forest_structure": {"BD", "ND", "NE", "EVI", "PTC"},
    "Meteorology": {"Temp", "Pre", "Rhum", "Wind", "SSRD", "LtgProxy"},
    "Drought": {"SPEI1", "SPEI3", "SPEI6", "SPEI12", "SPEI24"},
    "Fire_weather": {"FFMC", "DMC", "DC", "ISI", "BUI", "FWI"},
    "Terrain": {"DEM", "Slope", "Aspect"},
    "Anthropogenic": {
        "POP", "Dis_Farm", "Dis_Build", "Road_dens", "Dis_Railway",
        "Dis_Power",
    },
}


# =============================================================================
# Helpers
# =============================================================================

def log(message: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}"
    print(line, flush=True)
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def scientific_group(variable: str) -> str:
    for group, members in KNOWN_GROUPS.items():
        if variable in members:
            return group
    return "Other_candidate"


def finite_series(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric[np.isfinite(numeric)]


def safe_skew(values: pd.Series) -> float:
    finite = finite_series(values)
    if len(finite) < 3 or float(finite.std(ddof=0)) <= 0:
        return np.nan
    return float(stats.skew(finite.to_numpy(float), bias=False))


def q(values: pd.Series, probability: float) -> float:
    finite = finite_series(values)
    return float(finite.quantile(probability)) if len(finite) else np.nan


def integer_mask(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return (
        numeric.notna()
        & np.isfinite(numeric)
        & (np.abs(numeric - np.round(numeric)) <= INTEGER_TOLERANCE)
    )


def season_groups(data: pd.DataFrame) -> List[Tuple[int, str, pd.DataFrame]]:
    groups: List[Tuple[int, str, pd.DataFrame]] = [(0, "Overall", data)]
    groups.extend(
        (code, label, data.loc[data["Season"] == code])
        for code, label in SEASONS.items()
    )
    return groups


def safe_spearman(first: pd.Series, second: pd.Series) -> Tuple[float, float, int]:
    frame = pd.DataFrame({"x": first, "y": second}).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if len(frame) < 3 or frame["x"].std(ddof=0) <= 0 or frame["y"].std(ddof=0) <= 0:
        return np.nan, np.nan, int(len(frame))
    result = stats.spearmanr(frame["x"], frame["y"])
    return float(result.statistic), float(result.pvalue), int(len(frame))


# =============================================================================
# Input preparation
# =============================================================================

def prepare_data() -> Tuple[pd.DataFrame, int, int]:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Base table not found:\n{INPUT_CSV}")

    raw = pd.read_csv(INPUT_CSV)
    input_rows = len(raw)
    required = {
        "GRID_UID", "GRID_ID", "Country", "Year", "Season", "Fire_Count",
        "Burned_Pixel_Count", "ForestPixelCount", "ForestArea_km2",
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError("Missing required columns:\n" + "\n".join(missing))

    unexpected_countries = sorted(set(raw["Country"].dropna().astype(str)) - COUNTRY_LEVELS)
    if unexpected_countries:
        raise ValueError("Unexpected Country values: " + ", ".join(unexpected_countries))

    unexpected_seasons = sorted(set(raw["Season"].dropna().astype(int)) - set(SEASONS))
    if unexpected_seasons:
        raise ValueError("Unexpected Season values: " + ", ".join(map(str, unexpected_seasons)))

    duplicates = raw.duplicated(["GRID_UID", "Year", "Season"], keep=False)
    if duplicates.any():
        raw.loc[duplicates].to_csv(
            OUTPUT_ROOT / "ERROR_Duplicate_Panel_Records.csv",
            index=False,
            encoding="utf-8-sig",
        )
        raise ValueError("Duplicate GRID_UID-Year-Season records were found.")

    exposure = pd.to_numeric(raw["ForestPixelCount"], errors="coerce")
    forest_area = pd.to_numeric(raw["ForestArea_km2"], errors="coerce")
    if exposure.isna().any() or forest_area.isna().any():
        raise ValueError("ForestPixelCount or ForestArea_km2 contains nonnumeric values.")

    area_error = np.abs(forest_area - exposure * FOREST_PIXEL_AREA_KM2)
    if float(area_error.max()) > 1e-9:
        raise ValueError("ForestArea_km2 is inconsistent with ForestPixelCount * 0.25.")

    keep = forest_area > FOREST_AREA_THRESHOLD_KM2
    data = raw.loc[keep].copy().reset_index(drop=True)
    excluded_rows = int((~keep).sum())

    if (data["ForestPixelCount"].astype(float) <= 0).any():
        raise ValueError("ForestPixelCount must be positive after filtering.")

    data["FCD"] = data["Fire_Count"].astype(float) / data["ForestPixelCount"].astype(float)
    data["BAD"] = data["Burned_Pixel_Count"].astype(float) / data["ForestPixelCount"].astype(float)
    data["Log_ForestPixelCount"] = np.log(data["ForestPixelCount"].astype(float))

    return data, input_rows, excluded_rows


# =============================================================================
# Response and exposure audits
# =============================================================================

def anomaly_records(data: pd.DataFrame) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for response in COUNT_RESPONSES:
        values = pd.to_numeric(data[response], errors="coerce")
        valid_integer = integer_mask(values)
        bad = values.isna() | ~np.isfinite(values) | (values < 0) | ~valid_integer
        if not bad.any():
            continue
        part = data.loc[
            bad,
            ["GRID_UID", "GRID_ID", "Country", "Year", "Season", response,
             "ForestPixelCount", "ForestArea_km2"],
        ].copy()
        part = part.rename(columns={response: "Observed_Value"})
        part.insert(5, "Response", response)
        part["Issue_Nonnumeric_or_Nonfinite"] = (
            values.loc[bad].isna() | ~np.isfinite(values.loc[bad])
        ).to_numpy()
        part["Issue_Negative"] = (values.loc[bad] < 0).to_numpy()
        part["Issue_Noninteger"] = (~valid_integer.loc[bad]).to_numpy()
        rows.append(part)

    columns = [
        "GRID_UID", "GRID_ID", "Country", "Year", "Season", "Response",
        "Observed_Value", "ForestPixelCount", "ForestArea_km2",
        "Issue_Nonnumeric_or_Nonfinite", "Issue_Negative", "Issue_Noninteger",
    ]
    return pd.concat(rows, ignore_index=True)[columns] if rows else pd.DataFrame(columns=columns)


def count_distribution_audit(data: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for season_code, season_label, subset in season_groups(data):
        exposure = subset["ForestPixelCount"].astype(float)
        for response, rate_name in COUNT_RESPONSES.items():
            values = subset[response].astype(float)
            rates = values / exposure
            positive = values[values > 0]
            positive_rates = rates[values > 0]
            mean = float(values.mean())
            variance = float(values.var(ddof=1))
            rows.append({
                "Season_Code": season_code,
                "Season_Label": season_label,
                "Count_Response": response,
                "Rate_Scale_Name": rate_name,
                "N": int(len(values)),
                "Negative_Count": int((values < 0).sum()),
                "Noninteger_Count": int((~integer_mask(values)).sum()),
                "Zero_Count": int((values == 0).sum()),
                "Zero_Proportion": float((values == 0).mean()),
                "Positive_Count": int((values > 0).sum()),
                "Positive_Proportion": float((values > 0).mean()),
                "Count_Min": float(values.min()),
                "Count_Q50": q(values, 0.50),
                "Count_Q95": q(values, 0.95),
                "Count_Q99": q(values, 0.99),
                "Count_Max": float(values.max()),
                "Count_Mean": mean,
                "Count_Variance": variance,
                "Count_Variance_to_Mean": variance / mean if mean > 0 else np.nan,
                "Count_Skewness": safe_skew(values),
                "Count_Greater_Than_Exposure_Count": int((values > exposure).sum()),
                "Count_Greater_Than_Exposure_Proportion": float((values > exposure).mean()),
                "Rate_Min": float(rates.min()),
                "Rate_Q50": q(rates, 0.50),
                "Rate_Q95": q(rates, 0.95),
                "Rate_Q99": q(rates, 0.99),
                "Rate_Max": float(rates.max()),
                "Rate_Mean": float(rates.mean()),
                "Rate_Skewness": safe_skew(rates),
                "Positive_Count_Min": float(positive.min()) if len(positive) else np.nan,
                "Positive_Count_Median": float(positive.median()) if len(positive) else np.nan,
                "Positive_Count_Q95": float(positive.quantile(0.95)) if len(positive) else np.nan,
                "Positive_Count_Max": float(positive.max()) if len(positive) else np.nan,
                "Positive_Count_Mean": float(positive.mean()) if len(positive) else np.nan,
                "Positive_Count_Variance": float(positive.var(ddof=1)) if len(positive) > 1 else np.nan,
                "Positive_Count_Skewness": safe_skew(positive),
                "Positive_Rate_Median": float(positive_rates.median()) if len(positive_rates) else np.nan,
                "Positive_Rate_Q95": float(positive_rates.quantile(0.95)) if len(positive_rates) else np.nan,
                "Positive_Rate_Max": float(positive_rates.max()) if len(positive_rates) else np.nan,
                "Descriptive_Overdispersion_Flag": bool(mean > 0 and variance / mean > 1.5),
            })
    return pd.DataFrame(rows)


def poisson_diagnostics(data: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for season_code, season_label, subset in season_groups(data):
        exposure = subset["ForestPixelCount"].astype(float).to_numpy()
        for response, rate_name in COUNT_RESPONSES.items():
            observed = subset[response].astype(float).to_numpy()
            total_exposure = float(exposure.sum())
            total_count = float(observed.sum())
            rate_hat = total_count / total_exposure
            expected = exposure * rate_hat
            valid = expected > 0
            pearson = float(np.sum((observed[valid] - expected[valid]) ** 2 / expected[valid]))
            df = max(int(valid.sum()) - 1, 1)
            observed_zero = float((observed == 0).mean())
            expected_zero = float(np.exp(-expected).mean())
            log_likelihood = float(np.sum(
                observed * np.log(np.maximum(expected, np.finfo(float).tiny))
                - expected - scipy.special.gammaln(observed + 1.0)
            ))
            rows.append({
                "Season_Code": season_code,
                "Season_Label": season_label,
                "Count_Response": response,
                "Rate_Scale_Name": rate_name,
                "N": int(len(observed)),
                "Total_Count": total_count,
                "Total_Exposure_ForestPixels": total_exposure,
                "Intercept_Only_Rate_per_ForestPixel": rate_hat,
                "Observed_Zero_Proportion": observed_zero,
                "Poisson_Expected_Zero_Proportion": expected_zero,
                "Observed_minus_Expected_Zero_Proportion": observed_zero - expected_zero,
                "Observed_to_Expected_Zero_Ratio": observed_zero / expected_zero if expected_zero > 0 else np.nan,
                "Pearson_Chi_Square": pearson,
                "Pearson_Degrees_of_Freedom": df,
                "Pearson_Dispersion": pearson / df,
                "Poisson_LogLikelihood": log_likelihood,
                "Overdispersion_Flag_Dispersion_above_1_5": bool(pearson / df > 1.5),
                "Excess_Zero_Flag_Observed_above_Expected": bool(observed_zero > expected_zero),
                "Interpretation": (
                    "Intercept-only exposure-adjusted Poisson diagnostic; "
                    "final conclusions require predictor-adjusted model comparison."
                ),
            })
    return pd.DataFrame(rows)


def exposure_audit(data: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for season_code, season_label, subset in season_groups(data):
        exposure = subset["ForestPixelCount"].astype(float)
        area = subset["ForestArea_km2"].astype(float)
        rows.append({
            "Season_Code": season_code,
            "Season_Label": season_label,
            "N": int(len(subset)),
            "Exposure_Min_ForestPixels": float(exposure.min()),
            "Exposure_Q01": q(exposure, 0.01),
            "Exposure_Q50": q(exposure, 0.50),
            "Exposure_Q95": q(exposure, 0.95),
            "Exposure_Q99": q(exposure, 0.99),
            "Exposure_Max_ForestPixels": float(exposure.max()),
            "Exposure_Mean_ForestPixels": float(exposure.mean()),
            "Exposure_SD_ForestPixels": float(exposure.std(ddof=1)),
            "Exposure_Skewness": safe_skew(exposure),
            "ForestArea_Min_km2": float(area.min()),
            "ForestArea_Median_km2": float(area.median()),
            "ForestArea_Max_km2": float(area.max()),
            "Log_Exposure_Min": float(np.log(exposure).min()),
            "Log_Exposure_Max": float(np.log(exposure).max()),
        })
    return pd.DataFrame(rows)


def country_season_audit(data: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (country, season), subset in data.groupby(["Country", "Season"]):
        exposure = subset["ForestPixelCount"].astype(float)
        for response, rate_name in COUNT_RESPONSES.items():
            values = subset[response].astype(float)
            rates = values / exposure
            rows.append({
                "Country": country,
                "Season_Code": int(season),
                "Season_Label": SEASONS[int(season)],
                "Count_Response": response,
                "Rate_Scale_Name": rate_name,
                "N": int(len(subset)),
                "Unique_GRID_UIDs": int(subset["GRID_UID"].nunique()),
                "Zero_Count": int((values == 0).sum()),
                "Zero_Proportion": float((values == 0).mean()),
                "Count_Mean": float(values.mean()),
                "Count_Variance": float(values.var(ddof=1)),
                "Count_Max": float(values.max()),
                "Rate_Mean": float(rates.mean()),
                "Rate_Max": float(rates.max()),
                "Exposure_Mean": float(exposure.mean()),
                "Exposure_Median": float(exposure.median()),
            })
    return pd.DataFrame(rows)


def count_exposure_relationships(data: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for season_code, season_label, subset in season_groups(data):
        exposure = subset["ForestPixelCount"].astype(float)
        for response, rate_name in COUNT_RESPONSES.items():
            count = subset[response].astype(float)
            rate = count / exposure
            count_rho, count_p, count_n = safe_spearman(count, exposure)
            rate_rho, rate_p, rate_n = safe_spearman(rate, exposure)
            rows.append({
                "Season_Code": season_code,
                "Season_Label": season_label,
                "Count_Response": response,
                "Rate_Scale_Name": rate_name,
                "Count_vs_Exposure_Spearman_Rho": count_rho,
                "Count_vs_Exposure_P": count_p,
                "Count_vs_Exposure_N": count_n,
                "Rate_vs_Exposure_Spearman_Rho": rate_rho,
                "Rate_vs_Exposure_P": rate_p,
                "Rate_vs_Exposure_N": rate_n,
                "Interpretation_Note": (
                    "ForestPixelCount remains the log-offset regardless of correlation size."
                ),
            })
    return pd.DataFrame(rows)


# =============================================================================
# Predictor audit
# =============================================================================

def build_inventory(data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    rows: List[Dict[str, object]] = []
    candidates: List[str] = []
    for variable in data.columns:
        series = data[variable]
        numeric = pd.api.types.is_numeric_dtype(series)
        excluded = variable in HARD_EXCLUDE_COLUMNS
        nonmissing = int(series.notna().sum())
        missing_rate = float(series.isna().mean())
        unique = int(series.nunique(dropna=True))
        minimum = maximum = mean = sd = skewness = zero_rate = negative_rate = np.nan
        near_zero = False
        if numeric:
            finite = finite_series(series)
            if len(finite):
                minimum = float(finite.min())
                maximum = float(finite.max())
                mean = float(finite.mean())
                sd = float(finite.std(ddof=1)) if len(finite) > 1 else 0.0
                skewness = safe_skew(finite)
                zero_rate = float((finite == 0).mean())
                negative_rate = float((finite < 0).mean())
                near_zero = unique <= 1 or sd <= NEAR_ZERO_VARIANCE_TOLERANCE

        reason = ""
        if excluded:
            reason = "Identifier_response_exposure_coordinate_or_model_field"
        elif not numeric:
            reason = "Non_numeric"
        elif near_zero:
            reason = "Constant_or_near_zero_variance"
        elif missing_rate > HIGH_MISSINGNESS_THRESHOLD:
            reason = "Missing_rate_above_20_percent"

        candidate = reason == ""
        if candidate:
            candidates.append(variable)
        rows.append({
            "Variable": variable,
            "Data_Type": str(series.dtype),
            "Is_Numeric": bool(numeric),
            "Scientific_Group": scientific_group(variable) if numeric and not excluded else "Not_applicable",
            "Nonmissing_Count": nonmissing,
            "Missing_Count": int(series.isna().sum()),
            "Missing_Rate": missing_rate,
            "Unique_Nonmissing_Count": unique,
            "Minimum": minimum,
            "Q01": q(series, 0.01) if numeric else np.nan,
            "Median": q(series, 0.50) if numeric else np.nan,
            "Q99": q(series, 0.99) if numeric else np.nan,
            "Maximum": maximum,
            "Mean": mean,
            "SD": sd,
            "Skewness": skewness,
            "Zero_Rate": zero_rate,
            "Negative_Rate": negative_rate,
            "Near_Zero_Variance": near_zero,
            "Hard_Exclusion_Reason": reason,
            "Regression_Candidate": candidate,
        })
    return pd.DataFrame(rows), candidates


def complete_sample(data: pd.DataFrame, variables: Sequence[str]) -> pd.DataFrame:
    sample = data.loc[:, variables].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sample) > CORRELATION_SAMPLE_MAX:
        sample = sample.sample(CORRELATION_SAMPLE_MAX, random_state=RANDOM_SEED)
    return sample


def correlation_and_vif(sample: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    spearman = sample.corr(method="spearman")
    pair_rows: List[Dict[str, object]] = []
    columns = list(spearman.columns)
    for i, first in enumerate(columns):
        for second in columns[i + 1:]:
            rho = float(spearman.loc[first, second])
            if abs(rho) >= HIGH_CORRELATION_THRESHOLD:
                pair_rows.append({
                    "Variable_1": first,
                    "Group_1": scientific_group(first),
                    "Variable_2": second,
                    "Group_2": scientific_group(second),
                    "Spearman_Rho": rho,
                    "Absolute_Rho": abs(rho),
                    "Above_0_90": bool(abs(rho) >= VERY_HIGH_CORRELATION_THRESHOLD),
                    "Selection_Note": (
                        "Retain one scientifically justified representation in the same "
                        "count-model candidate set; do not use RF importance for this decision."
                    ),
                })
    high_pairs = pd.DataFrame(pair_rows)
    if high_pairs.empty:
        high_pairs = pd.DataFrame(columns=[
            "Variable_1", "Group_1", "Variable_2", "Group_2", "Spearman_Rho",
            "Absolute_Rho", "Above_0_90", "Selection_Note",
        ])
    else:
        high_pairs = high_pairs.sort_values("Absolute_Rho", ascending=False).reset_index(drop=True)

    standardized = (sample - sample.mean()) / sample.std(ddof=0)
    correlation = standardized.corr().to_numpy(float)
    inverse = np.linalg.pinv(correlation, hermitian=True)
    vif_values = np.diag(inverse)
    condition = float(np.linalg.cond(correlation))
    vif = pd.DataFrame({
        "Variable": sample.columns,
        "Scientific_Group": [scientific_group(v) for v in sample.columns],
        "VIF": vif_values,
        "VIF_Above_5": vif_values >= VIF_WARNING_THRESHOLD,
        "VIF_Above_10": vif_values >= VIF_HIGH_THRESHOLD,
        "Correlation_Matrix_Condition_Number": condition,
        "Complete_Sample_N": len(sample),
    }).sort_values(["VIF", "Variable"], ascending=[False, True]).reset_index(drop=True)
    return spearman, high_pairs, vif


def transformation_flags(inventory: pd.DataFrame, candidates: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for _, item in inventory.loc[inventory["Variable"].isin(candidates)].iterrows():
        variable = str(item["Variable"])
        suggestion = "None_initially"
        reason = ""
        if variable == "Aspect":
            suggestion = "Keep_as_existing_cosine_aspect_index"
            reason = "Aspect is already a cosine-based index and must not be transformed again."
        elif np.isfinite(item["Skewness"]) and abs(float(item["Skewness"])) >= SKEWNESS_TRANSFORM_THRESHOLD:
            if np.isfinite(item["Minimum"]) and float(item["Minimum"]) >= 0:
                suggestion = "Compare_log1p_with_original_scale"
                reason = "Strong right skewness in a nonnegative predictor."
            else:
                suggestion = "Compare_Yeo_Johnson_with_original_scale"
                reason = "Strong skewness with zero or negative values."
        rows.append({
            "Variable": variable,
            "Scientific_Group": item["Scientific_Group"],
            "Minimum": item["Minimum"],
            "Maximum": item["Maximum"],
            "Skewness": item["Skewness"],
            "Suggested_Transformation": suggestion,
            "Reason": reason,
        })
    return pd.DataFrame(rows).sort_values(["Scientific_Group", "Variable"]).reset_index(drop=True)


def screening_summary(
    inventory: pd.DataFrame,
    candidates: Sequence[str],
    high_pairs: pd.DataFrame,
    vif: pd.DataFrame,
) -> pd.DataFrame:
    correlated = set(high_pairs.get("Variable_1", [])) | set(high_pairs.get("Variable_2", []))
    vif_lookup = vif.set_index("Variable")["VIF"].to_dict()
    rows: List[Dict[str, object]] = []
    for _, item in inventory.loc[inventory["Variable"].isin(candidates)].iterrows():
        variable = str(item["Variable"])
        vif_value = float(vif_lookup.get(variable, np.nan))
        issues: List[str] = []
        if variable in correlated:
            issues.append("High_pairwise_correlation")
        if np.isfinite(vif_value) and vif_value >= 10:
            issues.append("VIF_above_10")
        elif np.isfinite(vif_value) and vif_value >= 5:
            issues.append("VIF_5_to_10")
        if np.isfinite(item["Skewness"]) and abs(float(item["Skewness"])) >= SKEWNESS_TRANSFORM_THRESHOLD:
            issues.append("Strong_skewness")
        if variable == "Aspect":
            issues.append("Already_cosine_aspect_index")
        rows.append({
            "Variable": variable,
            "Scientific_Group": item["Scientific_Group"],
            "Missing_Rate": item["Missing_Rate"],
            "Skewness": item["Skewness"],
            "VIF": vif_value,
            "Has_High_Correlation_Pair": variable in correlated,
            "Screening_Issues": ";".join(issues),
            "Proceed_to_Count_Model_Set_Comparison": True,
            "Automatic_Final_Selection": False,
            "Decision_Rule": (
                "Final inclusion will be determined using mechanism, collinearity, "
                "transformations, cross-validation, information criteria, and diagnostics."
            ),
        })
    return pd.DataFrame(rows).sort_values(["Scientific_Group", "Variable"]).reset_index(drop=True)


def missingness_by_country_season(data: pd.DataFrame, candidates: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (country, season), subset in data.groupby(["Country", "Season"]):
        for variable in candidates:
            rows.append({
                "Country": country,
                "Season_Code": int(season),
                "Season_Label": SEASONS[int(season)],
                "Variable": variable,
                "N": int(len(subset)),
                "Missing_Count": int(subset[variable].isna().sum()),
                "Missing_Rate": float(subset[variable].isna().mean()),
            })
    return pd.DataFrame(rows)


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)

    log("Starting count-response and predictor audit.")
    log(f"Input: {INPUT_CSV}")
    log("Responses: Fire_Count and Burned_Pixel_Count.")
    log("Exposure: ForestPixelCount; later offset = log(ForestPixelCount).")

    data, input_rows, excluded_rows = prepare_data()
    anomalies = anomaly_records(data)
    response_audit = count_distribution_audit(data)
    poisson_audit = poisson_diagnostics(data)
    exposure_summary = exposure_audit(data)
    country_summary = country_season_audit(data)
    exposure_relationships = count_exposure_relationships(data)

    inventory, candidates = build_inventory(data)
    if not candidates:
        raise ValueError("No regression candidate predictors were identified.")
    sample = complete_sample(data, candidates)
    if len(sample) < 100:
        raise ValueError("Too few complete rows for correlation and VIF analysis.")
    correlation, high_pairs, vif = correlation_and_vif(sample)
    transformations = transformation_flags(inventory, candidates)
    screening = screening_summary(inventory, candidates, high_pairs, vif)
    missingness = missingness_by_country_season(data, candidates)

    qa = pd.DataFrame([
        ("Input_file", str(INPUT_CSV)),
        ("Input_row_count", input_rows),
        ("Rows_after_ForestArea_km2_strictly_above_2_5", len(data)),
        ("Rows_excluded_by_forest_area_threshold", excluded_rows),
        ("Unique_GRID_UIDs_after_threshold", data["GRID_UID"].nunique()),
        ("Year_min", int(data["Year"].min())),
        ("Year_max", int(data["Year"].max())),
        ("Forest_pixel_area_km2", FOREST_PIXEL_AREA_KM2),
        ("Forest_area_threshold_km2_strictly_greater_than", FOREST_AREA_THRESHOLD_KM2),
        ("Minimum_ForestPixelCount_after_threshold", float(data["ForestPixelCount"].min())),
        ("Count_response_anomaly_record_count", len(anomalies)),
        ("Regression_candidate_count", len(candidates)),
        ("Correlation_and_VIF_complete_sample_N", len(sample)),
        ("Count_model_exposure", "ForestPixelCount"),
        ("Count_model_offset", "log(ForestPixelCount)"),
        ("Country_role", "Contextual fixed effect"),
        ("GRID_UID_role", "Repeated-measure clustering identifier"),
        ("Year_role", "Shared annual clustering identifier"),
        ("FCD_BAD_role", "Descriptive and final comparison scales only"),
    ], columns=["Metric", "Value"])

    definition = {
        "workflow_stage": "Count-response distribution and candidate-predictor audit",
        "uses_random_forest_variable_selection": False,
        "count_responses": list(COUNT_RESPONSES),
        "rate_scales": {
            "FCD": "Fire_Count / ForestPixelCount",
            "BAD": "Burned_Pixel_Count / ForestPixelCount",
        },
        "rate_interpretation": (
            "Seasonal cumulative detection density; not assumed to be bounded by one."
        ),
        "exposure": "ForestPixelCount",
        "offset_for_later_count_models": "log(ForestPixelCount)",
        "forest_filter": "ForestArea_km2 > 2.5",
        "initial_count_model_candidates": "Poisson, NB1, and NB2",
        "advanced_count_model_candidates": (
            "Hurdle negative binomial and zero-inflated negative binomial"
        ),
        "future_grouping_candidates": ["GRID_UID", "Year"],
        "correlation_rule": "Absolute Spearman rho >= 0.80 is flagged.",
        "vif_rule": "VIF >= 5 is flagged; VIF >= 10 is strongly flagged.",
        "aspect_rule": (
            "Aspect is already a cosine-based index and is not transformed again."
        ),
        "poisson_diagnostic_note": (
            "Intercept-only exposure-adjusted Poisson diagnostics are descriptive."
        ),
    }
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    with (OUTPUT_ROOT / "00_Audit_Definition.json").open("w", encoding="utf-8") as handle:
        json.dump(definition, handle, indent=2)
    qa.to_csv(OUTPUT_ROOT / "01_Data_QA_Summary.csv", index=False, encoding="utf-8-sig")
    response_audit.to_csv(OUTPUT_ROOT / "02_Count_Response_Distribution_Audit.csv", index=False, encoding="utf-8-sig")
    poisson_audit.to_csv(OUTPUT_ROOT / "03_Intercept_Only_Poisson_Diagnostics.csv", index=False, encoding="utf-8-sig")
    exposure_summary.to_csv(OUTPUT_ROOT / "04_Exposure_Distribution_Audit.csv", index=False, encoding="utf-8-sig")
    country_summary.to_csv(OUTPUT_ROOT / "05_Response_by_Country_Season.csv", index=False, encoding="utf-8-sig")
    exposure_relationships.to_csv(OUTPUT_ROOT / "06_Count_Exposure_Relationships.csv", index=False, encoding="utf-8-sig")
    inventory.to_csv(OUTPUT_ROOT / "07_Full_Column_Inventory.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({
        "Variable": candidates,
        "Scientific_Group": [scientific_group(v) for v in candidates],
    }).to_csv(OUTPUT_ROOT / "08_Regression_Candidate_Pool.csv", index=False, encoding="utf-8-sig")
    correlation.to_csv(OUTPUT_ROOT / "09_Spearman_Correlation_Matrix.csv", encoding="utf-8-sig")
    high_pairs.to_csv(OUTPUT_ROOT / "10_High_Correlation_Pairs.csv", index=False, encoding="utf-8-sig")
    vif.to_csv(OUTPUT_ROOT / "11_VIF_Diagnostics.csv", index=False, encoding="utf-8-sig")
    transformations.to_csv(OUTPUT_ROOT / "12_Transformation_Flags.csv", index=False, encoding="utf-8-sig")
    screening.to_csv(OUTPUT_ROOT / "13_Regression_Screening_Summary.csv", index=False, encoding="utf-8-sig")
    missingness.to_csv(OUTPUT_ROOT / "14_Missingness_by_Country_Season.csv", index=False, encoding="utf-8-sig")
    anomalies.to_csv(OUTPUT_ROOT / "15_Noninteger_or_Negative_Response_Records.csv", index=False, encoding="utf-8-sig")
    inventory.loc[~inventory["Regression_Candidate"]].to_csv(
        OUTPUT_ROOT / "16_Excluded_or_Noncandidate_Columns.csv",
        index=False,
        encoding="utf-8-sig",
    )
    with (OUTPUT_ROOT / "Software_Environment.json").open("w", encoding="utf-8") as handle:
        json.dump(environment, handle, indent=2)

    log(f"Rows retained: {len(data):,}; candidate predictors: {len(candidates)}.")
    log(f"Count-response anomaly records: {len(anomalies)}.")
    log("Audit completed successfully.")
    log("Next step: compare predictor-adjusted Poisson, NB1, and NB2 models.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
