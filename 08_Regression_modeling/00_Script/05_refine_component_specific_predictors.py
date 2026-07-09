# -*- coding: utf-8 -*-
"""
Component-specific predictor-block refinement for the formally selected
wildfire count models.

Public location
---------------
<CODE_ROOT>/08_Regression_modeling/00_Script/
    05_refine_component_specific_predictors.py

This step keeps the Step-04b distribution structures fixed. Five ZINB1
models use inexpensive component screens followed by actual joint ZINB1
confirmation. The Summer Burned_Pixel_Count single-stage NB1 model is
refined directly. Country fixed effects and the ForestPixelCount exposure
are retained in every applicable component. Final clustered inference is applied in Step 06d.
"""

from __future__ import annotations

import importlib.util
import json
import math
import platform
import re
import shutil
import sys
import warnings
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy
import sklearn
import statsmodels
from scipy import special
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP


# =============================================================================
# Paths
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REGRESSION_ROOT = SCRIPT_DIR.parent
CODE_ROOT = REGRESSION_ROOT.parent

INPUT_CSV = CODE_ROOT / "00_Input_data" / "Base_Table_2001_2025.csv.bz2"
STEP04_SCRIPT = SCRIPT_DIR / "04_compare_nb1_hurdle_zinb.py"
MATCHED_CONFIGURATION_CSV = (
    REGRESSION_ROOT
    / "04_NB1_Hurdle_ZINB_Matched_Comparison"
    / "02_Matched_Predictor_Configurations.csv"
)
FORMAL_ROOT = REGRESSION_ROOT / "04b_Formal_Count_Structure_Selection"
FORMAL_SELECTION_CSV = FORMAL_ROOT / "11_Final_Selected_Count_Structure.csv"
OUTPUT_ROOT = REGRESSION_ROOT / "05_Component_Specific_Predictor_Refinement"
CHECKPOINT_ROOT = OUTPUT_ROOT / "_checkpoints"
LOG_FILE = OUTPUT_ROOT / "component_specific_predictor_refinement.log"


# =============================================================================
# Settings
# =============================================================================

FOREST_PIXEL_AREA_KM2 = 0.25
FOREST_AREA_THRESHOLD_KM2 = 2.5

SEASONS = {1: "Spring", 2: "Summer", 3: "Autumn"}
COUNT_RESPONSES = {"Fire_Count": "FCD", "Burned_Pixel_Count": "BAD"}
TEMPORAL_FOLDS = {
    1: [2024, 2002, 2006, 2004, 2009],
    2: [2016, 2015, 2007, 2018, 2014],
    3: [2025, 2021, 2013, 2019, 2003],
    4: [2010, 2023, 2017, 2011, 2005],
    5: [2020, 2022, 2001, 2012, 2008],
}

COUNTRY_LEVELS = ["China", "NK", "Russia"]
LOG1P_VARIABLES = {"ND", "NE", "LtgProxy", "POP", "Dis_Farm", "Road_dens"}
SPEI_VARIABLES = {"SPEI1", "SPEI3", "SPEI6", "SPEI12", "SPEI24"}
BLOCK_TEMPLATES = {
    "Forest_Structure": ["BD", "PTC"],
    "Terrain": ["DEM", "Slope", "Aspect"],
    "Human_Access": ["POP", "Dis_Farm", "Road_dens"],
    "Lightning": ["LtgProxy"],
}

FIT_MAXITER = 1000
ZINB_OPTIMIZERS = ["bfgs", "lbfgs"]
PROBABILITY_EPSILON = 1e-12

SCREEN_ABS_EQUIVALENCE = 0.0001
SCREEN_REL_EQUIVALENCE_PERCENT = 0.01
FINAL_ABS_NLL_EQUIVALENCE = 0.0001
FINAL_REL_NLL_EQUIVALENCE_PERCENT = 0.01

TOTAL_COUNT_RATIO_LOWER = 0.10
TOTAL_COUNT_RATIO_UPPER = 10.0
MAX_RMSE_MULTIPLE_VS_BASELINE = 10.0
MAX_RATE_MAX_MULTIPLE = 100.0


# =============================================================================
# Step-04 function loader
# =============================================================================

def load_step04_module():
    if not STEP04_SCRIPT.exists():
        raise FileNotFoundError(
            f"Required Step-04 script not found:\n{STEP04_SCRIPT}"
        )

    module_name = "step04_count_models"

    spec = importlib.util.spec_from_file_location(
        module_name,
        STEP04_SCRIPT,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Could not create an import specification for Step 04."
        )

    module = importlib.util.module_from_spec(spec)

    # Register the module before execution. The Step-04 script defines a
    # dataclass, and Python's dataclasses module resolves type information
    # through sys.modules while the class decorator is running.
    previous_module = sys.modules.get(module_name)
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception:
        if previous_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module
        raise

    required = [
        "fit_nb1",
        "single_stage_outputs",
        "fit_binary_glm",
        "stable_expit",
        "count_mean",
        "nb1_logpmf_and_logzero",
        "stable_log_one_minus_exp",
        "prediction_metrics",
        "convergence_status",
        "warning_text",
    ]

    missing = [
        name
        for name in required
        if not hasattr(module, name)
    ]

    if missing:
        raise RuntimeError(
            "Step-04 script is missing required functions: "
            + ", ".join(missing)
        )

    return module


# =============================================================================
# Utilities
# =============================================================================

def log(message: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def parse_predictors(value: object) -> List[str]:
    if pd.isna(value):
        return []
    return list(dict.fromkeys(item.strip() for item in str(value).split(";") if item.strip()))


def join_predictors(values: Sequence[str]) -> str:
    return ";".join(values)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def temporal_fold_mapping() -> Dict[int, int]:
    result: Dict[int, int] = {}
    for fold, years in TEMPORAL_FOLDS.items():
        for year in years:
            if year in result:
                raise ValueError(f"Year {year} occurs in multiple folds.")
            result[int(year)] = int(fold)
    return result


def preserve_checkpoints_and_reset_output() -> None:
    temporary = OUTPUT_ROOT.parent / "_temporary_step05_checkpoints"
    if temporary.exists():
        shutil.rmtree(temporary)
    if OUTPUT_ROOT.exists():
        if CHECKPOINT_ROOT.exists():
            shutil.move(str(CHECKPOINT_ROOT), str(temporary))
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    if temporary.exists():
        shutil.move(str(temporary), str(CHECKPOINT_ROOT))


def robust_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def safe_roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan


def safe_pr_auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else np.nan


# =============================================================================
# Inputs and data
# =============================================================================

def read_definitions() -> pd.DataFrame:
    if not FORMAL_SELECTION_CSV.exists() or not MATCHED_CONFIGURATION_CSV.exists():
        raise FileNotFoundError("Step-04/04b definition files are missing.")
    selected = pd.read_csv(FORMAL_SELECTION_CSV)
    configs = pd.read_csv(MATCHED_CONFIGURATION_CSV)
    definitions = selected[[
        "Season_Code", "Season_Label", "Count_Response", "Rate_Scale_Name",
        "Model_Structure",
    ]].merge(
        configs[[
            "Season_Code", "Season_Label", "Count_Response", "Rate_Scale_Name",
            "Config_ID", "Climate_Representation", "Drought_Representation",
            "Transformation_Scheme", "Continuous_Predictors", "Maximum_VIF",
        ]],
        on=["Season_Code", "Season_Label", "Count_Response", "Rate_Scale_Name"],
        how="left",
        validate="one_to_one",
    )
    expected = {(s, r) for s in SEASONS for r in COUNT_RESPONSES}
    observed = {(int(x.Season_Code), str(x.Count_Response)) for x in definitions.itertuples()}
    if observed != expected:
        raise ValueError("Definitions do not contain exactly six season-response models.")
    if set(definitions["Model_Structure"]) - {"ZINB1", "Single_Stage_NB1"}:
        raise ValueError("Unexpected formal model structure.")
    return definitions.sort_values(["Season_Code", "Count_Response"]).reset_index(drop=True)


def prepare_data(definitions: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(INPUT_CSV)
    predictors = sorted({p for value in definitions["Continuous_Predictors"] for p in parse_predictors(value)})
    required = {
        "GRID_UID", "GRID_ID", "Country", "Year", "Season", "Fire_Count",
        "Burned_Pixel_Count", "ForestPixelCount", "ForestArea_km2", *predictors,
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError("Base table is missing columns:\n" + "\n".join(missing))
    duplicates = data.duplicated(["GRID_UID", "Year", "Season"], keep=False)
    if duplicates.any():
        raise ValueError("Duplicate GRID_UID-Year-Season records were found.")
    exposure = pd.to_numeric(data["ForestPixelCount"], errors="coerce")
    area = pd.to_numeric(data["ForestArea_km2"], errors="coerce")
    if exposure.isna().any() or area.isna().any():
        raise ValueError("Forest exposure variables contain nonnumeric values.")
    if float(np.abs(area - exposure * FOREST_PIXEL_AREA_KM2).max()) > 1e-9:
        raise ValueError("ForestArea_km2 is inconsistent with ForestPixelCount * 0.25.")
    data = data.loc[area > FOREST_AREA_THRESHOLD_KM2].copy()
    if set(data["Country"].dropna().astype(str)) - set(COUNTRY_LEVELS):
        raise ValueError("Unexpected Country values were found.")
    for response in COUNT_RESPONSES:
        values = pd.to_numeric(data[response], errors="coerce")
        invalid = values.isna() | ~np.isfinite(values) | (values < 0) | (np.abs(values - np.round(values)) > 1e-9)
        if invalid.any():
            raise ValueError(f"{response} contains invalid counts.")
        data[response] = values.astype(int)
    data["Temporal_Fold"] = data["Year"].astype(int).map(temporal_fold_mapping())
    if data["Temporal_Fold"].isna().any():
        raise ValueError("At least one year is absent from the temporal-fold definition.")
    complete_columns = ["Country", "Fire_Count", "Burned_Pixel_Count", "ForestPixelCount", *predictors]
    complete = data[complete_columns].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    excluded = data.loc[~complete, ["GRID_UID", "GRID_ID", "Country", "Year", "Season", *predictors]].copy()
    excluded.to_csv(OUTPUT_ROOT / "Excluded_Incomplete_Refinement_Records.csv", index=False, encoding="utf-8-sig")
    data = data.loc[complete].copy().reset_index(drop=True)
    data["Temporal_Fold"] = data["Temporal_Fold"].astype(int)
    return data, excluded


# =============================================================================
# Predictor blocks and matrices
# =============================================================================

def predictor_blocks(predictors: Sequence[str]) -> Dict[str, List[str]]:
    predictors = list(dict.fromkeys(predictors))
    remaining = set(predictors)
    blocks: Dict[str, List[str]] = {}
    for name, template in BLOCK_TEMPLATES.items():
        values = [v for v in template if v in remaining]
        if values:
            blocks[name] = values
            remaining -= set(values)
    drought = [v for v in predictors if v in SPEI_VARIABLES and v in remaining]
    if drought:
        blocks["Drought"] = drought
        remaining -= set(drought)
    weather = [v for v in predictors if v in remaining]
    if weather:
        blocks["Weather_Fire_Weather"] = weather
        remaining -= set(weather)
    if remaining or set(v for values in blocks.values() for v in values) != set(predictors):
        raise RuntimeError("Predictor-block assignment failed.")
    return blocks


def omission_candidates(full: Sequence[str], blocks: Dict[str, List[str]], prefix: str) -> pd.DataFrame:
    rows = [{"Candidate_ID": f"{prefix}_FULL", "Omitted_Block": "None", "Predictors": join_predictors(full), "Predictor_Count": len(full)}]
    for name, removed in blocks.items():
        retained = [v for v in full if v not in set(removed)]
        if retained:
            rows.append({"Candidate_ID": f"{prefix}_DROP_{name}", "Omitted_Block": name, "Predictors": join_predictors(retained), "Predictor_Count": len(retained)})
    return pd.DataFrame(rows)


def transform(frame: pd.DataFrame, predictors: Sequence[str], scheme: str) -> pd.DataFrame:
    result = frame.loc[:, predictors].astype(float).copy() if predictors else pd.DataFrame(index=frame.index)
    if scheme == "Raw":
        return result
    if scheme != "Audit_Guided_Log1p":
        raise ValueError(f"Unknown transformation scheme: {scheme}")
    for variable in predictors:
        if variable in LOG1P_VARIABLES:
            if float(result[variable].min()) < 0:
                raise ValueError(f"{variable} cannot use log1p because it contains negatives.")
            result[variable] = np.log1p(result[variable].to_numpy(dtype=float))
    return result


def country_dummies(frame: pd.DataFrame) -> np.ndarray:
    country = frame["Country"].astype(str)
    return np.column_stack([(country == "NK").astype(float), (country == "Russia").astype(float)])


def matrices(train: pd.DataFrame, valid: pd.DataFrame, predictors: Sequence[str], scheme: str) -> Tuple[np.ndarray, np.ndarray]:
    train_cont = transform(train, predictors, scheme)
    valid_cont = transform(valid, predictors, scheme)
    if predictors:
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_cont.to_numpy(dtype=float))
        valid_scaled = scaler.transform(valid_cont.to_numpy(dtype=float))
    else:
        train_scaled = np.empty((len(train), 0))
        valid_scaled = np.empty((len(valid), 0))
    return (
        np.column_stack([np.ones(len(train)), train_scaled, country_dummies(train)]),
        np.column_stack([np.ones(len(valid)), valid_scaled, country_dummies(valid)]),
    )


# =============================================================================
# Screening metrics
# =============================================================================

def binary_metrics(observed_zero: np.ndarray, predicted_zero: np.ndarray) -> Dict[str, float]:
    y = np.asarray(observed_zero, dtype=float)
    p = np.clip(np.asarray(predicted_zero, dtype=float), PROBABILITY_EPSILON, 1 - PROBABILITY_EPSILON)
    logp = y * np.log(p) + (1 - y) * np.log1p(-p)
    return {
        "Binary_LogLoss": float(-logp.mean()),
        "Binary_Brier": float(np.mean((p - y) ** 2)),
        "Zero_ROC_AUC": safe_roc_auc(y, p),
        "Zero_PR_AUC": safe_pr_auc(y, p),
        "Observed_Zero_Proportion": float(y.mean()),
        "Predicted_Zero_Proportion": float(p.mean()),
    }


def count_metrics(step04, predictions: pd.DataFrame) -> Dict[str, float]:
    return step04.prediction_metrics(
        predictions["Observed_Count"].to_numpy(dtype=float),
        predictions["Predicted_Count"].to_numpy(dtype=float),
        predictions["ForestPixelCount"].to_numpy(dtype=float),
        predictions["Predicted_Zero_Probability"].to_numpy(dtype=float),
        predictions["Predictive_LogProbability"].to_numpy(dtype=float),
    )


def make_prediction_frame(validation: pd.DataFrame, response: str, structure: str, candidate: str, outputs: Dict[str, object]) -> pd.DataFrame:
    result = validation[["GRID_UID", "GRID_ID", "Country", "Year", "Season", "Temporal_Fold", "ForestPixelCount"]].copy()
    observed = validation[response].to_numpy(dtype=float)
    exposure = validation["ForestPixelCount"].to_numpy(dtype=float)
    predicted = np.asarray(outputs["Predicted_Count"], dtype=float)
    result["Count_Response"] = response
    result["Rate_Scale_Name"] = COUNT_RESPONSES[response]
    result["Model_Structure"] = structure
    result["Candidate_ID"] = candidate
    result["Observed_Count"] = observed
    result["Predicted_Count"] = predicted
    result["Observed_Rate"] = observed / exposure
    result["Predicted_Rate"] = predicted / exposure
    result["Predicted_Zero_Probability"] = np.asarray(outputs["Predicted_Zero_Probability"], dtype=float)
    result["Predictive_LogProbability"] = np.asarray(outputs["Predictive_LogProbability"], dtype=float)
    return result


# =============================================================================
# Component-specific ZINB1
# =============================================================================

def zinb_outputs(step04, parameters: np.ndarray, observed: np.ndarray, x_count: np.ndarray, x_zero: np.ndarray, exposure: np.ndarray) -> Dict[str, object]:
    parameters = np.asarray(parameters, dtype=float)
    kz, kc = x_zero.shape[1], x_count.shape[1]
    if len(parameters) != kz + kc + 1:
        raise ValueError("Unexpected component-specific ZINB1 parameter length.")
    gamma = parameters[:kz]
    beta = parameters[kz:kz + kc]
    alpha = float(parameters[-1])
    pi = step04.stable_expit(x_zero @ gamma)
    mean = step04.count_mean(np.concatenate([beta, [alpha]]), x_count, exposure)
    base_logp, base_logzero = step04.nb1_logpmf_and_logzero(observed, mean, alpha)
    y = np.asarray(observed, dtype=float)
    zero = y == 0
    logp = np.empty_like(y)
    log_pi = np.log(pi)
    log_main = np.log1p(-pi)
    logp[zero] = np.logaddexp(log_pi[zero], log_main[zero] + base_logzero[zero])
    logp[~zero] = log_main[~zero] + base_logp[~zero]
    base_zero = np.exp(np.clip(base_logzero, -745.0, 0.0))
    return {
        "Predicted_Count": (1 - pi) * mean,
        "Predicted_Zero_Probability": pi + (1 - pi) * base_zero,
        "Predictive_LogProbability": logp,
        "Alpha": alpha,
    }


def zinb_objective_gradient(step04, transformed: np.ndarray, y: np.ndarray, xc: np.ndarray, xz: np.ndarray, exposure: np.ndarray) -> Tuple[float, np.ndarray]:
    y = np.asarray(y, dtype=float)
    kz, kc = xz.shape[1], xc.shape[1]
    gamma = transformed[:kz]
    beta = transformed[kz:kz + kc]
    alpha = float(np.exp(transformed[-1]))
    pi = step04.stable_expit(xz @ gamma)
    log_pi = np.log(pi)
    log_main = np.log1p(-pi)
    eta = np.clip(xc @ beta + np.log(exposure), -745.0, 700.0)
    mean = np.clip(np.exp(eta), np.finfo(float).tiny, 1e300)
    size = np.clip(mean / alpha, np.finfo(float).tiny, 1e300)
    log_s = -math.log1p(alpha)
    log_f = math.log(alpha) - math.log1p(alpha)
    p_s = 1.0 / (1.0 + alpha)
    p_f = alpha / (1.0 + alpha)
    dig = special.digamma(y + size) - special.digamma(size) + log_s
    base_logp = special.gammaln(y + size) - special.gammaln(size) - special.gammaln(y + 1.0) + size * log_s + y * log_f
    base_logzero = size * log_s
    zero = y == 0
    positive = ~zero
    logp = np.empty_like(y)
    logp[positive] = log_main[positive] + base_logp[positive]
    logp[zero] = np.logaddexp(log_pi[zero], log_main[zero] + base_logzero[zero])
    if not np.isfinite(logp).all():
        return 1e100, np.zeros_like(transformed)
    infl_mult = np.empty_like(y)
    infl_mult[positive] = -pi[positive]
    if zero.any():
        log_one_minus_base_zero = step04.stable_log_one_minus_exp(base_logzero[zero])
        infl_mult[zero] = np.exp(np.clip(log_pi[zero] + log_main[zero] + log_one_minus_base_zero - logp[zero], -745.0, 700.0))
    gamma_score = xz.T @ infl_mult
    beta_mult = np.empty_like(y)
    logalpha_score = np.empty_like(y)
    beta_mult[positive] = size[positive] * dig[positive]
    logalpha_score[positive] = -size[positive] * dig[positive] - size[positive] * p_f + y[positive] * p_s
    if zero.any():
        posterior = np.exp(np.clip(log_main[zero] + base_logzero[zero] - logp[zero], -745.0, 0.0))
        beta_mult[zero] = posterior * size[zero] * log_s
        logalpha_score[zero] = posterior * (-size[zero]) * (log_s + p_f)
    score = np.concatenate([xz.T @ infl_mult, xc.T @ beta_mult, [logalpha_score.sum()]])
    n = float(len(y))
    objective = float(-logp.sum() / n)
    gradient = -score / n
    if not np.isfinite(objective) or not np.isfinite(gradient).all():
        return 1e100, np.zeros_like(transformed)
    return objective, gradient


def fit_zinb_component(step04, y: np.ndarray, xc: np.ndarray, xz: np.ndarray, exposure: np.ndarray) -> Tuple[np.ndarray, Dict[str, object]]:
    nb_result, nb_status = step04.fit_nb1(y, xc, exposure)
    zero_result, zero_status = step04.fit_binary_glm((np.asarray(y) == 0).astype(float), xz)
    count_start = np.asarray(nb_result.params, dtype=float)
    zero_start = np.asarray(zero_result.params, dtype=float)
    starts = [
        np.concatenate([zero_start, count_start]),
        np.concatenate([zero_start * 0.5, count_start]),
        np.concatenate([zero_start * 0.25, count_start]),
    ]
    model = ZeroInflatedNegativeBinomialP(y, xc, exog_infl=xz, exposure=exposure, inflation="logit", p=1, missing="raise")
    attempts: List[Dict[str, object]] = []
    best_params = None
    best_status = None
    best_ll = -np.inf
    for start_index, start in enumerate(starts, start=1):
        for optimizer in ZINB_OPTIMIZERS:
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                try:
                    result = model.fit(start_params=start, method=optimizer, maxiter=FIT_MAXITER, disp=0, full_output=True)
                    params = np.asarray(result.params, dtype=float)
                    converged = step04.convergence_status(result)
                    stable_ll = np.nan
                    if converged and np.isfinite(params).all() and params[-1] > 0:
                        stable_ll = float(np.asarray(zinb_outputs(step04, params, y, xc, xz, exposure)["Predictive_LogProbability"]).sum())
                    valid = bool(np.isfinite(stable_ll))
                    status = {
                        "Optimizer": optimizer, "Start_Index": start_index,
                        "Converged": converged, "Valid_Fit": valid,
                        "Alpha_Estimate": float(params[-1]) if np.isfinite(params[-1]) else np.nan,
                        "Stable_LogLikelihood": stable_ll,
                        "Warning_Text": step04.warning_text(captured),
                    }
                    attempts.append(status)
                    if valid and stable_ll > best_ll:
                        best_params, best_status, best_ll = params, status, stable_ll
                        break
                except Exception as error:
                    attempts.append({"Optimizer": optimizer, "Start_Index": start_index, "Converged": False, "Valid_Fit": False, "Alpha_Estimate": np.nan, "Stable_LogLikelihood": np.nan, "Warning_Text": f"{type(error).__name__}: {error}"})
        if best_params is not None:
            break
    if best_params is not None:
        best_status = dict(best_status)
        best_status.update({"Attempt_Count": len(attempts), "Failed_Attempt_Count": sum(not x["Valid_Fit"] for x in attempts), "Count_Start_Optimizer": nb_status.get("Optimizer", ""), "Zero_Start_Optimizer": zero_status.get("Optimizer", "")})
        return best_params, best_status

    # Stable custom fallback with separate count and inflation designs.
    beta = count_start[:-1]
    alpha = float(np.clip(count_start[-1], 1e-6, 1e6))
    low_zero = np.zeros_like(zero_start)
    low_zero[0] = -3.0
    custom_starts = [
        np.concatenate([low_zero, beta, [math.log(alpha)]]),
        np.concatenate([zero_start * 0.25, beta, [math.log(alpha)]]),
        np.concatenate([zero_start * 0.50, beta, [math.log(alpha)]]),
    ]
    bounds = [(-25.0, 25.0)] * xz.shape[1] + [(-30.0, 30.0)] * xc.shape[1] + [(math.log(1e-6), math.log(1e6))]
    custom_attempts = []
    best = None
    best_objective = np.inf
    for start_index, start in enumerate(custom_starts, start=1):
        optimization = minimize(
            fun=lambda parameters: zinb_objective_gradient(step04, parameters, y, xc, xz, exposure),
            x0=start, method="L-BFGS-B", jac=True, bounds=bounds,
            options={"maxiter": 2500, "maxls": 80, "ftol": 1e-12, "gtol": 1e-6},
        )
        objective, gradient = zinb_objective_gradient(step04, optimization.x, y, xc, xz, exposure)
        maximum_gradient = float(np.max(np.abs(gradient)))
        valid = bool((optimization.success or maximum_gradient <= 1e-4) and np.isfinite(objective) and np.isfinite(optimization.x).all())
        custom_attempts.append({"Start_Index": start_index, "Valid_Fit": valid, "Objective": objective, "Maximum_Absolute_Gradient": maximum_gradient, "Message": str(optimization.message)})
        if valid and objective < best_objective:
            best, best_objective = optimization, objective
        if valid and optimization.success and maximum_gradient <= 1e-5:
            break
    if best is None:
        raise RuntimeError("All component-specific ZINB1 fits failed: " + json.dumps({"statsmodels": attempts, "custom": custom_attempts}, default=str))
    transformed = np.asarray(best.x, dtype=float)
    kz = xz.shape[1]
    params = np.concatenate([transformed[:kz], transformed[kz:-1], [float(np.exp(transformed[-1]))]])
    return params, {
        "Optimizer": "Custom_L-BFGS-B_analytic_gradient", "Start_Index": np.nan,
        "Converged": True, "Valid_Fit": True, "Alpha_Estimate": float(params[-1]),
        "Stable_LogLikelihood": float(-best_objective * len(y)),
        "Warning_Text": str(best.message), "Attempt_Count": len(custom_attempts),
        "Failed_Attempt_Count": sum(not x["Valid_Fit"] for x in custom_attempts),
        "Statsmodels_Failed_Attempt_Count": len(attempts),
    }


# =============================================================================
# CV runners and checkpoints
# =============================================================================

def checkpoint_paths(stage: str, season: int, response: str, candidate: str) -> Tuple[Path, Path, Path]:
    folder = CHECKPOINT_ROOT / safe_name(stage) / f"S{season}_{response}"
    folder.mkdir(parents=True, exist_ok=True)
    stem = safe_name(candidate)
    return folder / f"{stem}_fold.csv", folder / f"{stem}_status.csv", folder / f"{stem}_predictions.csv.gz"


def load_valid_checkpoint(stage: str, season: int, response: str, candidate: str, expected_rows: int):
    fold_path, status_path, prediction_path = checkpoint_paths(stage, season, response, candidate)
    if not (fold_path.exists() and status_path.exists() and prediction_path.exists()):
        return None
    try:
        fold = pd.read_csv(fold_path)
        status = pd.read_csv(status_path)
        predictions = pd.read_csv(prediction_path, compression="gzip")
    except Exception:
        return None
    if len(fold) != 5 or len(status) != 5 or len(predictions) != expected_rows:
        return None
    return fold, status, predictions


def save_checkpoint(stage: str, season: int, response: str, candidate: str, fold: pd.DataFrame, status: pd.DataFrame, predictions: pd.DataFrame) -> None:
    fold_path, status_path, prediction_path = checkpoint_paths(stage, season, response, candidate)
    fold.to_csv(fold_path, index=False, encoding="utf-8-sig")
    status.to_csv(status_path, index=False, encoding="utf-8-sig")
    predictions.to_csv(prediction_path, index=False, compression="gzip")


def run_zero_screen(step04, season_data: pd.DataFrame, season: int, response: str, candidate: str, predictors: Sequence[str], scheme: str):
    stage = "Zero_Screen"
    cached = load_valid_checkpoint(stage, season, response, candidate, len(season_data))
    if cached is not None:
        log(f"Reused zero-screen checkpoint: {SEASONS[season]} {response} {candidate}.")
        return cached
    fold_rows, status_rows, prediction_tables = [], [], []
    for fold in range(1, 6):
        valid_mask = season_data["Temporal_Fold"] == fold
        train = season_data.loc[~valid_mask].reset_index(drop=True)
        valid = season_data.loc[valid_mask].reset_index(drop=True)
        x_train, x_valid = matrices(train, valid, predictors, scheme)
        y_train = (train[response].to_numpy(dtype=float) == 0).astype(float)
        y_valid = (valid[response].to_numpy(dtype=float) == 0).astype(float)
        result, status = step04.fit_binary_glm(y_train, x_train)
        p = step04.stable_expit(x_valid @ np.asarray(result.params, dtype=float))
        fold_rows.append({"Fold": fold, "Candidate_ID": candidate, **binary_metrics(y_valid, p)})
        status_rows.append({"Fold": fold, "Candidate_ID": candidate, **status})
        prediction = valid[["GRID_UID", "GRID_ID", "Country", "Year", "Season", "Temporal_Fold"]].copy()
        prediction["Candidate_ID"] = candidate
        prediction["Observed_Zero"] = y_valid
        prediction["Predicted_Zero_Probability"] = p
        prediction_tables.append(prediction)
    result = pd.DataFrame(fold_rows), pd.DataFrame(status_rows), pd.concat(prediction_tables, ignore_index=True)
    save_checkpoint(stage, season, response, candidate, *result)
    return result


def run_count_cv(step04, season_data: pd.DataFrame, season: int, response: str, candidate: str, count_predictors: Sequence[str], scheme: str, structure: str, zero_predictors: Sequence[str] | None = None):
    stage = "Joint_ZINB" if structure == "ZINB1" else "NB1_Refinement"
    cached = load_valid_checkpoint(stage, season, response, candidate, len(season_data))
    if cached is not None:
        log(f"Reused {stage} checkpoint: {SEASONS[season]} {response} {candidate}.")
        return cached
    fold_rows, status_rows, prediction_tables = [], [], []
    for fold in range(1, 6):
        valid_mask = season_data["Temporal_Fold"] == fold
        train = season_data.loc[~valid_mask].reset_index(drop=True)
        valid = season_data.loc[valid_mask].reset_index(drop=True)
        xc_train, xc_valid = matrices(train, valid, count_predictors, scheme)
        y_train = train[response].to_numpy(dtype=float)
        y_valid = valid[response].to_numpy(dtype=float)
        e_train = train["ForestPixelCount"].to_numpy(dtype=float)
        e_valid = valid["ForestPixelCount"].to_numpy(dtype=float)
        if structure == "ZINB1":
            xz_train, xz_valid = matrices(train, valid, zero_predictors or [], scheme)
            params, status = fit_zinb_component(step04, y_train, xc_train, xz_train, e_train)
            outputs = zinb_outputs(step04, params, y_valid, xc_valid, xz_valid, e_valid)
        else:
            result, status = step04.fit_nb1(y_train, xc_train, e_train)
            outputs = step04.single_stage_outputs(result, y_valid, xc_valid, e_valid)
        metrics = step04.prediction_metrics(y_valid, np.asarray(outputs["Predicted_Count"]), e_valid, np.asarray(outputs["Predicted_Zero_Probability"]), np.asarray(outputs["Predictive_LogProbability"]))
        fold_rows.append({"Fold": fold, "Candidate_ID": candidate, **metrics})
        status_rows.append({"Fold": fold, "Candidate_ID": candidate, **status})
        prediction_tables.append(make_prediction_frame(valid, response, structure, candidate, outputs))
        if structure == "ZINB1":
            log(f"Completed joint ZINB fold {fold}: {SEASONS[season]} {response} {candidate}; NLL={metrics['Mean_Negative_LogLikelihood']:.6f}.")
    result = pd.DataFrame(fold_rows), pd.DataFrame(status_rows), pd.concat(prediction_tables, ignore_index=True)
    save_checkpoint(stage, season, response, candidate, *result)
    return result


# =============================================================================
# Ranking and stability
# =============================================================================

def rank_screen(table: pd.DataFrame, loss: str, secondary: Sequence[str]) -> pd.DataFrame:
    parts = []
    for _, subset in table.groupby(["Season_Code", "Count_Response"], sort=True):
        subset = subset.copy()
        minimum = float(subset[loss].min())
        subset["Delta_Loss"] = subset[loss] - minimum
        subset["Relative_Delta_Loss_pct"] = subset["Delta_Loss"] / minimum * 100.0
        subset["Loss_Equivalent"] = (subset["Delta_Loss"] <= SCREEN_ABS_EQUIVALENCE) & (subset["Relative_Delta_Loss_pct"] <= SCREEN_REL_EQUIVALENCE_PERCENT)
        subset = subset.sort_values(["Loss_Equivalent", "Predictor_Count", *secondary, "Candidate_ID"], ascending=[False, True, *([True] * len(secondary)), True])
        subset["Screening_Rank"] = np.arange(1, len(subset) + 1)
        parts.append(subset)
    return pd.concat(parts, ignore_index=True)


def baseline_predictions(season: int, response: str, structure: str, candidate: str) -> pd.DataFrame:
    path = FORMAL_ROOT / f"Selected_OOF_Predictions_S{season}_{response}.csv.gz"
    source = pd.read_csv(path, compression="gzip")
    if set(source["Model_Structure"].astype(str)) != {structure}:
        raise ValueError("Baseline OOF model structure does not match Step 04b.")
    source["Candidate_ID"] = candidate
    source["Count_Response"] = response
    source["Rate_Scale_Name"] = COUNT_RESPONSES[response]
    return source


def fold_metrics_from_predictions(step04, predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold in range(1, 6):
        rows.append({"Fold": fold, **count_metrics(step04, predictions.loc[predictions["Temporal_Fold"] == fold])})
    return pd.DataFrame(rows)


def summarize_candidate(step04, season: int, response: str, candidate_definition: Dict[str, object], predictions: pd.DataFrame, fold_metrics: pd.DataFrame, fit_status: pd.DataFrame, baseline_pooled: Dict[str, float], baseline_fold: pd.Series) -> Dict[str, object]:
    pooled = count_metrics(step04, predictions)
    ratio = float(pooled["Predicted_to_Observed_Total_Count_Ratio"])
    rmse_multiple = float(pooled["Rate_RMSE"] / baseline_pooled["Rate_RMSE"])
    fold_rmse_multiple = fold_metrics.set_index("Fold")["Rate_RMSE"] / baseline_fold
    numerical_valid = bool(np.isfinite(predictions[["Predicted_Count", "Predicted_Rate", "Predicted_Zero_Probability", "Predictive_LogProbability"]].to_numpy(dtype=float)).all() and (predictions["Predicted_Count"] >= -1e-10).all() and predictions["Predicted_Zero_Probability"].between(-1e-10, 1 + 1e-10).all())
    ratio_valid = bool(np.isfinite(ratio) and TOTAL_COUNT_RATIO_LOWER <= ratio <= TOTAL_COUNT_RATIO_UPPER and fold_metrics["Predicted_to_Observed_Total_Count_Ratio"].between(TOTAL_COUNT_RATIO_LOWER, TOTAL_COUNT_RATIO_UPPER).all())
    rmse_valid = bool(np.isfinite(rmse_multiple) and rmse_multiple <= MAX_RMSE_MULTIPLE_VS_BASELINE and fold_rmse_multiple.le(MAX_RMSE_MULTIPLE_VS_BASELINE).all())
    max_rate_valid = bool((predictions["Predicted_Rate"].max() / max(predictions["Observed_Rate"].max(), np.finfo(float).tiny)) <= MAX_RATE_MAX_MULTIPLE)
    converged = bool(fit_status["Converged"].map(robust_bool).all())
    valid_fit = bool(fit_status["Valid_Fit"].map(robust_bool).all())
    stable = converged and valid_fit and numerical_valid and ratio_valid and rmse_valid and max_rate_valid
    return {
        "Season_Code": season, "Season_Label": SEASONS[season], "Count_Response": response,
        "Rate_Scale_Name": COUNT_RESPONSES[response], **candidate_definition, **pooled,
        "All_Folds_Converged": converged, "All_Folds_Valid": valid_fit,
        "Numerical_Predictions_Valid": numerical_valid, "Total_Count_Ratio_Valid": ratio_valid,
        "Rate_RMSE_Multiple_vs_Full_Baseline": rmse_multiple,
        "Rate_RMSE_Stability_Valid": rmse_valid, "Maximum_Rate_Stability_Valid": max_rate_valid,
        "Stable_for_Final_Selection": stable,
    }


def rank_final(table: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, subset in table.groupby(["Season_Code", "Count_Response"], sort=True):
        subset = subset.copy()
        stable = subset.loc[subset["Stable_for_Final_Selection"]].copy()
        if stable.empty:
            raise RuntimeError("No stable Step-05 candidate remains.")
        minimum = float(stable["Mean_Negative_LogLikelihood"].min())
        stable["Delta_NLL"] = stable["Mean_Negative_LogLikelihood"] - minimum
        stable["Relative_Delta_NLL_pct"] = stable["Delta_NLL"] / minimum * 100.0
        stable["NLL_Equivalent"] = (stable["Delta_NLL"] <= FINAL_ABS_NLL_EQUIVALENCE) & (stable["Relative_Delta_NLL_pct"] <= FINAL_REL_NLL_EQUIVALENCE_PERCENT)
        stable["Absolute_Total_Count_Calibration_Error"] = np.abs(stable["Predicted_to_Observed_Total_Count_Ratio"] - 1.0)
        stable = stable.sort_values(["NLL_Equivalent", "Zero_Probability_Brier", "Rate_RMSE", "Total_Component_Predictor_Count", "Absolute_Total_Count_Calibration_Error", "Candidate_ID"], ascending=[False, True, True, True, True, True])
        stable["Final_Refinement_Rank"] = np.arange(1, len(stable) + 1)
        unstable = subset.loc[~subset["Stable_for_Final_Selection"]].copy()
        unstable[["Delta_NLL", "Relative_Delta_NLL_pct", "Absolute_Total_Count_Calibration_Error", "Final_Refinement_Rank"]] = np.nan
        unstable["NLL_Equivalent"] = False
        parts.extend([stable, unstable])
    return pd.concat(parts, ignore_index=True).sort_values(["Season_Code", "Count_Response", "Final_Refinement_Rank", "Candidate_ID"], na_position="last").reset_index(drop=True)


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    preserve_checkpoints_and_reset_output()
    step04 = load_step04_module()
    log("Starting component-specific predictor refinement.")
    definitions = read_definitions()
    data, excluded = prepare_data(definitions)
    log(f"Prepared {len(data):,} rows and {data['GRID_UID'].nunique():,} grids; excluded {len(excluded):,} rows.")

    block_rows, screen_definition_rows = [], []
    zero_fold_all, zero_status_all, zero_pooled_rows = [], [], []
    count_fold_all, count_status_all, count_pooled_rows = [], [], []

    # Screening for the five ZINB1 models.
    for definition in definitions.itertuples(index=False):
        season = int(definition.Season_Code)
        response = str(definition.Count_Response)
        if str(definition.Model_Structure) != "ZINB1":
            continue
        scheme = str(definition.Transformation_Scheme)
        full = parse_predictors(definition.Continuous_Predictors)
        blocks = predictor_blocks(full)
        for name, variables in blocks.items():
            block_rows.append({"Season_Code": season, "Season_Label": SEASONS[season], "Count_Response": response, "Block_Name": name, "Block_Variables": join_predictors(variables), "Block_Variable_Count": len(variables)})
        season_data = data.loc[data["Season"] == season].reset_index(drop=True)
        for candidate in omission_candidates(full, blocks, "ZERO").itertuples(index=False):
            predictors = parse_predictors(candidate.Predictors)
            screen_definition_rows.append({"Season_Code": season, "Season_Label": SEASONS[season], "Count_Response": response, "Screening_Component": "Structural_Zero_Proxy", "Candidate_ID": candidate.Candidate_ID, "Omitted_Block": candidate.Omitted_Block, "Predictors": candidate.Predictors, "Predictor_Count": candidate.Predictor_Count, "Transformation_Scheme": scheme})
            fold, status, predictions = run_zero_screen(step04, season_data, season, response, candidate.Candidate_ID, predictors, scheme)
            fold.insert(0, "Season_Code", season); fold.insert(1, "Season_Label", SEASONS[season]); fold.insert(2, "Count_Response", response); fold["Predictor_Count"] = candidate.Predictor_Count
            status.insert(0, "Season_Code", season); status.insert(1, "Season_Label", SEASONS[season]); status.insert(2, "Count_Response", response)
            zero_fold_all.append(fold); zero_status_all.append(status)
            zero_pooled_rows.append({"Season_Code": season, "Season_Label": SEASONS[season], "Count_Response": response, "Candidate_ID": candidate.Candidate_ID, "Omitted_Block": candidate.Omitted_Block, "Predictors": candidate.Predictors, "Predictor_Count": candidate.Predictor_Count, **binary_metrics(predictions["Observed_Zero"].to_numpy(), predictions["Predicted_Zero_Probability"].to_numpy())})
        for candidate in omission_candidates(full, blocks, "COUNT").itertuples(index=False):
            predictors = parse_predictors(candidate.Predictors)
            screen_definition_rows.append({"Season_Code": season, "Season_Label": SEASONS[season], "Count_Response": response, "Screening_Component": "Count_NB1_Proxy", "Candidate_ID": candidate.Candidate_ID, "Omitted_Block": candidate.Omitted_Block, "Predictors": candidate.Predictors, "Predictor_Count": candidate.Predictor_Count, "Transformation_Scheme": scheme})
            fold, status, predictions = run_count_cv(step04, season_data, season, response, candidate.Candidate_ID, predictors, scheme, "Single_Stage_NB1")
            fold.insert(0, "Season_Code", season); fold.insert(1, "Season_Label", SEASONS[season]); fold.insert(2, "Count_Response", response); fold["Predictor_Count"] = candidate.Predictor_Count
            status.insert(0, "Season_Code", season); status.insert(1, "Season_Label", SEASONS[season]); status.insert(2, "Count_Response", response)
            count_fold_all.append(fold); count_status_all.append(status)
            count_pooled_rows.append({"Season_Code": season, "Season_Label": SEASONS[season], "Count_Response": response, "Candidate_ID": candidate.Candidate_ID, "Omitted_Block": candidate.Omitted_Block, "Predictors": candidate.Predictors, "Predictor_Count": candidate.Predictor_Count, **count_metrics(step04, predictions)})

    zero_pooled = pd.DataFrame(zero_pooled_rows)
    count_pooled = pd.DataFrame(count_pooled_rows)
    zero_ranking = rank_screen(zero_pooled, "Binary_LogLoss", ["Binary_Brier"])
    count_ranking = rank_screen(count_pooled, "Mean_Negative_LogLikelihood", ["Zero_Probability_Brier", "Rate_RMSE"])
    selected_zero = {(int(x.Season_Code), str(x.Count_Response)): x for x in zero_ranking.loc[zero_ranking["Screening_Rank"] == 1].itertuples(index=False)}
    selected_count = {(int(x.Season_Code), str(x.Count_Response)): x for x in count_ranking.loc[count_ranking["Screening_Rank"] == 1].itertuples(index=False)}

    final_definition_rows, final_fold_all, final_status_all, final_summary_rows = [], [], [], []
    prediction_store: Dict[Tuple[int, str, str], pd.DataFrame] = {}

    for definition in definitions.itertuples(index=False):
        season = int(definition.Season_Code)
        response = str(definition.Count_Response)
        structure = str(definition.Model_Structure)
        scheme = str(definition.Transformation_Scheme)
        full = parse_predictors(definition.Continuous_Predictors)
        season_data = data.loc[data["Season"] == season].reset_index(drop=True)
        baseline_id = "JOINT_FULL_FULL" if structure == "ZINB1" else "NB1_FULL"
        baseline = baseline_predictions(season, response, structure, baseline_id)
        baseline_pooled = count_metrics(step04, baseline)
        baseline_fold_metrics = fold_metrics_from_predictions(step04, baseline)
        baseline_fold = baseline_fold_metrics.set_index("Fold")["Rate_RMSE"]
        baseline_status = pd.DataFrame([{"Fold": f, "Candidate_ID": baseline_id, "Optimizer": "Reused_Step04b", "Converged": True, "Valid_Fit": True, "Warning_Text": "Reused formal OOF baseline."} for f in range(1, 6)])

        if structure == "ZINB1":
            z = selected_zero[(season, response)]
            c = selected_count[(season, response)]
            options = [
                ("JOINT_FULL_FULL", full, full, "None", "None", "Reused_Step04b"),
                ("JOINT_SCREENED_ZERO_FULL_COUNT", parse_predictors(z.Predictors), full, z.Omitted_Block, "None", "New_Joint_Fit"),
                ("JOINT_FULL_ZERO_SCREENED_COUNT", full, parse_predictors(c.Predictors), "None", c.Omitted_Block, "New_Joint_Fit"),
                ("JOINT_SCREENED_ZERO_SCREENED_COUNT", parse_predictors(z.Predictors), parse_predictors(c.Predictors), z.Omitted_Block, c.Omitted_Block, "New_Joint_Fit"),
            ]
            unique = {}
            for option in options:
                unique.setdefault((tuple(option[1]), tuple(option[2])), option)
            candidates = list(unique.values())
        else:
            blocks = predictor_blocks(full)
            for name, variables in blocks.items():
                block_rows.append({"Season_Code": season, "Season_Label": SEASONS[season], "Count_Response": response, "Block_Name": name, "Block_Variables": join_predictors(variables), "Block_Variable_Count": len(variables)})
            candidates = [(x.Candidate_ID, [], parse_predictors(x.Predictors), "Not_applicable", x.Omitted_Block, "Reused_Step04b" if x.Candidate_ID == "NB1_FULL" else "New_NB1_Fit") for x in omission_candidates(full, blocks, "NB1").itertuples(index=False)]

        for candidate_id, zero_predictors, count_predictors, zero_omitted, count_omitted, source in candidates:
            definition_row = {
                "Candidate_ID": candidate_id, "Model_Structure": structure,
                "Structural_Zero_Predictors": join_predictors(zero_predictors),
                "Count_Predictors": join_predictors(count_predictors),
                "Structural_Zero_Predictor_Count": len(zero_predictors),
                "Count_Predictor_Count": len(count_predictors),
                "Total_Component_Predictor_Count": len(zero_predictors) + len(count_predictors),
                "Structural_Zero_Omitted_Block": zero_omitted,
                "Count_Omitted_Block": count_omitted,
                "Transformation_Scheme": scheme, "Prediction_Source": source,
            }
            final_definition_rows.append({"Season_Code": season, "Season_Label": SEASONS[season], "Count_Response": response, **definition_row})
            if candidate_id == baseline_id:
                predictions = baseline.copy()
                fold_metrics = baseline_fold_metrics.copy(); fold_metrics["Candidate_ID"] = candidate_id
                status = baseline_status.copy()
            else:
                fold_metrics, status, predictions = run_count_cv(step04, season_data, season, response, candidate_id, count_predictors, scheme, structure, zero_predictors)
            fold_metrics.insert(0, "Season_Code", season); fold_metrics.insert(1, "Season_Label", SEASONS[season]); fold_metrics.insert(2, "Count_Response", response)
            status.insert(0, "Season_Code", season); status.insert(1, "Season_Label", SEASONS[season]); status.insert(2, "Count_Response", response)
            final_fold_all.append(fold_metrics); final_status_all.append(status)
            final_summary_rows.append(summarize_candidate(step04, season, response, definition_row, predictions, fold_metrics, status, baseline_pooled, baseline_fold))
            prediction_store[(season, response, candidate_id)] = predictions

    final_fold = pd.concat(final_fold_all, ignore_index=True)
    final_status = pd.concat(final_status_all, ignore_index=True)
    final_pooled = pd.DataFrame(final_summary_rows)
    final_ranking = rank_final(final_pooled)
    final_selected = final_ranking.loc[final_ranking["Final_Refinement_Rank"] == 1].copy().sort_values(["Season_Code", "Count_Response"]).reset_index(drop=True)
    final_selected["Selection_Status"] = "Formal component-specific predictor structure selected by temporal OOF confirmation"
    final_selected["Next_Step"] = "Audit panel dependence and finalize GRID_UID-Year two-way clustered inference."

    manifest_rows = []
    for row in final_selected.itertuples(index=False):
        predictions = prediction_store[(int(row.Season_Code), str(row.Count_Response), str(row.Candidate_ID))].copy()
        destination = OUTPUT_ROOT / f"Refined_Selected_OOF_Predictions_S{row.Season_Code}_{row.Count_Response}.csv.gz"
        predictions["Final_Selected_Candidate_ID"] = row.Candidate_ID
        predictions.to_csv(destination, index=False, compression="gzip")
        manifest_rows.append({"Season_Code": row.Season_Code, "Season_Label": row.Season_Label, "Count_Response": row.Count_Response, "Model_Structure": row.Model_Structure, "Selected_Candidate_ID": row.Candidate_ID, "Prediction_File": destination.name, "Rows": len(predictions)})

    qa = pd.DataFrame([
        ("Rows_in_common_complete_case_sample", len(data)),
        ("Excluded_incomplete_rows", len(excluded)),
        ("Unique_GRID_UIDs", data["GRID_UID"].nunique()),
        ("Formal_ZINB1_model_count", int((definitions["Model_Structure"] == "ZINB1").sum())),
        ("Formal_single_stage_NB1_model_count", int((definitions["Model_Structure"] == "Single_Stage_NB1").sum())),
        ("Zero_screen_candidate_count", len(zero_pooled)),
        ("Count_screen_candidate_count", len(count_pooled)),
        ("Final_confirmation_candidate_count", len(final_pooled)),
        ("Final_selected_model_count", len(final_selected)),
        ("Final_inference_stage", "Step 06d GRID_UID-Year two-way clustered inference"),
        ("Exposure", "ForestPixelCount"),
    ], columns=["Metric", "Value"])

    method = {
        "workflow_stage": "Component-specific predictor-block refinement",
        "formal_structure_source": str(FORMAL_SELECTION_CSV),
        "screening_note": "Zero-indicator logit and single-stage NB1 are screening proxies only; final ZINB1 selection uses joint temporal OOF likelihood.",
        "predictor_blocks": {"Forest_Structure": ["BD", "PTC"], "Terrain": ["DEM", "Slope", "Aspect"], "Human_Access": ["POP", "Dis_Farm", "Road_dens"], "Lightning": ["LtgProxy"], "Drought": "selected SPEI", "Weather_Fire_Weather": "remaining selected weather variables"},
        "country_effect": "Retained in every component; China reference",
        "exposure": "ForestPixelCount",
        "equivalent_offset": "log(ForestPixelCount)",
        "screen_equivalence": {"absolute": SCREEN_ABS_EQUIVALENCE, "relative_percent": SCREEN_REL_EQUIVALENCE_PERCENT},
        "final_nll_equivalence": {"absolute": FINAL_ABS_NLL_EQUIVALENCE, "relative_percent": FINAL_REL_NLL_EQUIVALENCE_PERCENT},
        "final_selection_order": ["OOF NLL equivalence", "zero Brier", "rate RMSE", "total component predictor count", "total-count calibration"],
        "final_inference_stage": "Step 06d GRID_UID-Year two-way clustered inference",
    }
    environment = {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__, "statsmodels": statsmodels.__version__, "run_timestamp": datetime.now().isoformat(timespec="seconds")}

    with (OUTPUT_ROOT / "00_Method_Definition.json").open("w", encoding="utf-8") as handle:
        json.dump(method, handle, indent=2)
    qa.to_csv(OUTPUT_ROOT / "01_Data_QA_Summary.csv", index=False, encoding="utf-8-sig")
    definitions.to_csv(OUTPUT_ROOT / "02_Formal_Model_and_Base_Predictor_Definitions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(block_rows).drop_duplicates().to_csv(OUTPUT_ROOT / "03_Predictor_Block_Definitions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(screen_definition_rows).to_csv(OUTPUT_ROOT / "04_Screening_Candidate_Definitions.csv", index=False, encoding="utf-8-sig")
    pd.concat(zero_fold_all, ignore_index=True).to_csv(OUTPUT_ROOT / "05_Zero_Component_Screen_Fold_Metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat(zero_status_all, ignore_index=True).to_csv(OUTPUT_ROOT / "06_Zero_Component_Screen_Fit_Status.csv", index=False, encoding="utf-8-sig")
    zero_ranking.to_csv(OUTPUT_ROOT / "07_Zero_Component_Screen_Ranking.csv", index=False, encoding="utf-8-sig")
    pd.concat(count_fold_all, ignore_index=True).to_csv(OUTPUT_ROOT / "08_Count_Component_Screen_Fold_Metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat(count_status_all, ignore_index=True).to_csv(OUTPUT_ROOT / "09_Count_Component_Screen_Fit_Status.csv", index=False, encoding="utf-8-sig")
    count_ranking.to_csv(OUTPUT_ROOT / "10_Count_Component_Screen_Ranking.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(final_definition_rows).to_csv(OUTPUT_ROOT / "11_Final_Candidate_Definitions.csv", index=False, encoding="utf-8-sig")
    final_fold.to_csv(OUTPUT_ROOT / "12_Final_Confirmation_Fold_Metrics.csv", index=False, encoding="utf-8-sig")
    final_status.to_csv(OUTPUT_ROOT / "13_Final_Confirmation_Fit_Status.csv", index=False, encoding="utf-8-sig")
    final_pooled.to_csv(OUTPUT_ROOT / "14_Final_Confirmation_Pooled_Metrics.csv", index=False, encoding="utf-8-sig")
    final_ranking.to_csv(OUTPUT_ROOT / "15_Final_Refinement_Ranking.csv", index=False, encoding="utf-8-sig")
    final_selected.to_csv(OUTPUT_ROOT / "16_Final_Component_Specific_Predictor_Structure.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(manifest_rows).to_csv(OUTPUT_ROOT / "17_Selected_OOF_Prediction_Manifest.csv", index=False, encoding="utf-8-sig")
    with (OUTPUT_ROOT / "Software_Environment.json").open("w", encoding="utf-8") as handle:
        json.dump(environment, handle, indent=2)

    if CHECKPOINT_ROOT.exists():
        shutil.rmtree(CHECKPOINT_ROOT)
    log("Component-specific predictor refinement completed successfully.")
    for row in final_selected.itertuples(index=False):
        log(f"Selected: {row.Season_Label} {row.Count_Response}; {row.Model_Structure}; {row.Candidate_ID}; zero={row.Structural_Zero_Predictor_Count}; count={row.Count_Predictor_Count}.")
    log("Next step: audit panel dependence and finalize GRID_UID-Year two-way clustered inference.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
