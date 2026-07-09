# -*- coding: utf-8 -*-
"""
Step 06c: parameter identifiability and stable-likelihood audit.

Public location:
  <CODE_ROOT>/08_Regression_modeling/00_Script/
  06c_audit_parameter_identifiability_and_stabilize_inference.py

This script stabilizes the selected fixed mean structures and:
- reuses the exact Step-05 samples and temporal folds;
- evaluates NB1/ZINB1 with an explicit stable likelihood;
- optimizes log(alpha) with analytic scores;
- obtains a numerical Hessian from the analytic total score;
- calculates GRID_UID-Year two-way clustered covariance;
- tests limited simplifications for Spring FC, Summer BA, and Autumn BA;
- confirms those alternatives with the same five temporal folds.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import platform
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy
import sklearn
import statsmodels
from scipy import optimize, stats
from scipy.special import digamma, expit, gammaln
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
REGRESSION_ROOT = SCRIPT_DIR.parent
CODE_ROOT = REGRESSION_ROOT.parent
STEP04_SCRIPT = SCRIPT_DIR / "04_compare_nb1_hurdle_zinb.py"
STEP05_SCRIPT = SCRIPT_DIR / "05_refine_component_specific_predictors.py"
STEP06B_SCRIPT = SCRIPT_DIR / "06b_fit_full_data_models_and_two_way_cluster_inference.py"
STEP05_ROOT = REGRESSION_ROOT / "05_Component_Specific_Predictor_Refinement"
STEP06B_ROOT = REGRESSION_ROOT / "06b_Full_Data_Fixed_Model_and_Two_Way_Cluster_Inference"
STEP06B_COEF = STEP06B_ROOT / "06_All_Coefficients_and_Clustered_Inference.csv"
OUTPUT_ROOT = REGRESSION_ROOT / "06c_Parameter_Identifiability_and_Stable_Likelihood_Audit"
CHECKPOINT_ROOT = OUTPUT_ROOT / "Checkpoints"
OOF_CHECKPOINT_ROOT = CHECKPOINT_ROOT / "OOF_Folds"
FULL_CHECKPOINT_ROOT = CHECKPOINT_ROOT / "Full_Data"
COVARIANCE_ROOT = OUTPUT_ROOT / "Candidate_Covariance_Matrices"
SELECTED_OOF_ROOT = OUTPUT_ROOT / "Selected_Inference_Model_OOF_Predictions"
LOG_FILE = OUTPUT_ROOT / "parameter_identifiability_stable_likelihood.log"

SEASONS = {1: "Spring", 2: "Summer", 3: "Autumn"}
RATE_NAMES = {"Fire_Count": "FCD", "Burned_Pixel_Count": "BAD"}
COUNTRY_TERMS = ["Country_NK", "Country_Russia"]
LOG1P_VARIABLES = {"ND", "NE", "LtgProxy", "POP", "Dis_Farm", "Road_dens"}
FOLDS = [1, 2, 3, 4, 5]

EPS = 1e-12
COEF_BOUND = 50.0
LOG_ALPHA_BOUNDS = (-12.0, 12.0)
MAXITER = 3000
MAX_GRAD = 1e-3
MAX_REL_GRAD = 1e-6
MAX_INFO_CONDITION = 1e12
INFO_RCOND = 1e-10
EXTREME_COEF = 25.0
PSD_REL_TOL = 1e-8
OOF_REL_TOL = 0.005
OOF_ABS_TOL = 1e-4


def log(message: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}"
    print(line, flush=True)
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def parse_list(value: object) -> List[str]:
    if pd.isna(value) or not str(value).strip():
        return []
    return list(dict.fromkeys(x.strip() for x in str(value).split(";") if x.strip()))


def safe_name(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def load_module(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
        raise
    return module


def signature(record: Mapping[str, object]) -> str:
    text = json.dumps(dict(record), sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def candidate_definitions(structure: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for d in structure.itertuples(index=False):
        s, r, m = int(d.Season_Code), str(d.Count_Response), str(d.Model_Structure)
        zp = parse_list(d.Structural_Zero_Predictors)
        cp = parse_list(d.Count_Predictors)
        base = dict(
            Season_Code=s, Season_Label=SEASONS[s], Count_Response=r,
            Rate_Scale_Name=RATE_NAMES[r], Model_Structure=m,
            Transformation_Scheme=str(d.Transformation_Scheme),
            Step05_Candidate_ID=str(d.Candidate_ID),
            Structural_Zero_Predictors=";".join(zp), Count_Predictors=";".join(cp),
            Structural_Zero_Country_Terms=";".join(COUNTRY_TERMS) if m == "ZINB1" else "",
            Count_Country_Terms=";".join(COUNTRY_TERMS),
        )

        def make(**updates: object) -> Dict[str, object]:
            item = base.copy()
            item.update(updates)
            return item

        targeted = (s, r) in {
            (1, "Fire_Count"),
            (2, "Burned_Pixel_Count"),
            (3, "Burned_Pixel_Count"),
        }
        local = [make(
            Candidate_ID="BASELINE_STRICT",
            Candidate_Type="Baseline",
            Targeted_OOF_Required=targeted,
            Terms_Removed_from_Step05="",
            Scientific_Role="Strict stable-likelihood refit of the Step-05 structure.",
        )]
        if (s, r, m) == (1, "Fire_Count", "ZINB1"):
            no_pop = ";".join(x for x in zp if x != "POP")
            local += [
                make(
                    Candidate_ID="ZERO_DROP_COUNTRY_NK",
                    Candidate_Type="Targeted_simplification",
                    Targeted_OOF_Required=True,
                    Structural_Zero_Country_Terms="Country_Russia",
                    Terms_Removed_from_Step05="Structural_Zero:Country_NK",
                    Scientific_Role="Remove the non-estimable NK zero-component contrast.",
                ),
                make(
                    Candidate_ID="ZERO_DROP_POP",
                    Candidate_Type="Targeted_simplification",
                    Targeted_OOF_Required=True,
                    Structural_Zero_Predictors=no_pop,
                    Terms_Removed_from_Step05="Structural_Zero:POP",
                    Scientific_Role="Audit the large Spring FC zero-component POP coefficient.",
                ),
                make(
                    Candidate_ID="ZERO_DROP_COUNTRY_NK_AND_POP",
                    Candidate_Type="Targeted_simplification",
                    Targeted_OOF_Required=True,
                    Structural_Zero_Predictors=no_pop,
                    Structural_Zero_Country_Terms="Country_Russia",
                    Terms_Removed_from_Step05="Structural_Zero:Country_NK;Structural_Zero:POP",
                    Scientific_Role="Remove both Spring FC boundary candidates.",
                ),
                make(
                    Candidate_ID="ZERO_DROP_ALL_COUNTRY",
                    Candidate_Type="Diagnostic_sensitivity",
                    Targeted_OOF_Required=True,
                    Structural_Zero_Country_Terms="",
                    Terms_Removed_from_Step05="Structural_Zero:Country_NK;Structural_Zero:Country_Russia",
                    Scientific_Role="Retain Country only in the count component.",
                ),
                make(
                    Candidate_ID="ZERO_DROP_ALL_COUNTRY_AND_POP",
                    Candidate_Type="Diagnostic_sensitivity",
                    Targeted_OOF_Required=True,
                    Structural_Zero_Predictors=no_pop,
                    Structural_Zero_Country_Terms="",
                    Terms_Removed_from_Step05="Structural_Zero:Country_NK;Structural_Zero:Country_Russia;Structural_Zero:POP",
                    Scientific_Role="Conservative Spring FC zero-component simplification.",
                ),
            ]
        if (s, r, m) == (2, "Burned_Pixel_Count", "Single_Stage_NB1"):
            local += [
                make(
                    Candidate_ID="COUNT_DROP_COUNTRY_NK",
                    Candidate_Type="Targeted_simplification",
                    Targeted_OOF_Required=True,
                    Count_Country_Terms="Country_Russia",
                    Terms_Removed_from_Step05="Count:Country_NK",
                    Scientific_Role="Remove only the non-estimable NK Summer BA contrast.",
                ),
                make(
                    Candidate_ID="COUNT_DROP_ALL_COUNTRY",
                    Candidate_Type="Diagnostic_sensitivity",
                    Targeted_OOF_Required=True,
                    Count_Country_Terms="",
                    Terms_Removed_from_Step05="Count:Country_NK;Count:Country_Russia",
                    Scientific_Role="Diagnostic only unless the partial-country model remains unstable.",
                ),
            ]
        if (s, r, m) == (3, "Burned_Pixel_Count", "ZINB1"):
            local += [
                make(
                    Candidate_ID="ZERO_DROP_COUNTRY_NK",
                    Candidate_Type="Targeted_simplification",
                    Targeted_OOF_Required=True,
                    Structural_Zero_Country_Terms="Country_Russia",
                    Terms_Removed_from_Step05="Structural_Zero:Country_NK",
                    Scientific_Role="Remove only the non-estimable NK Autumn BA zero-component contrast.",
                ),
                make(
                    Candidate_ID="ZERO_DROP_ALL_COUNTRY",
                    Candidate_Type="Diagnostic_sensitivity",
                    Targeted_OOF_Required=True,
                    Structural_Zero_Country_Terms="",
                    Terms_Removed_from_Step05="Structural_Zero:Country_NK;Structural_Zero:Country_Russia",
                    Scientific_Role="Diagnostic only unless the partial-country model remains unstable.",
                ),
            ]
        for item in local:
            item["Candidate_Signature"] = signature(item)
            rows.append(item)
    out = pd.DataFrame(rows)
    out["Candidate_Order"] = out.groupby(["Season_Code", "Count_Response"]).cumcount() + 1
    return out

def transform(frame: pd.DataFrame, predictors: Sequence[str], scheme: str) -> pd.DataFrame:
    if not predictors:
        return pd.DataFrame(index=frame.index)
    result = frame.loc[:, list(predictors)].astype(float).copy()
    if scheme == "Raw":
        return result
    if scheme != "Audit_Guided_Log1p":
        raise ValueError(scheme)
    for name in predictors:
        if name in LOG1P_VARIABLES:
            if float(result[name].min()) < 0:
                raise ValueError(f"Negative values in {name}")
            result[name] = np.log1p(result[name].to_numpy(float))
    return result


def country_matrix(frame: pd.DataFrame, terms: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    c = frame["Country"].astype(str)
    cols, names = [], []
    for term in terms:
        if term == "Country_NK":
            cols.append((c == "NK").astype(float).to_numpy())
        elif term == "Country_Russia":
            cols.append((c == "Russia").astype(float).to_numpy())
        else:
            raise ValueError(term)
        names.append(term)
    return (np.column_stack(cols) if cols else np.empty((len(frame), 0))), names


def fit_scaler(frame: pd.DataFrame, predictors: Sequence[str], scheme: str):
    t = transform(frame, predictors, scheme)
    if not predictors:
        return None, np.empty((len(frame), 0))
    scaler = StandardScaler()
    return scaler, scaler.fit_transform(t.to_numpy(float))


def apply_scaler(frame: pd.DataFrame, predictors: Sequence[str], scheme: str, scaler):
    if not predictors:
        return np.empty((len(frame), 0))
    return scaler.transform(transform(frame, predictors, scheme).to_numpy(float))


def assemble(frame: pd.DataFrame, continuous: np.ndarray, predictors: Sequence[str], terms: Sequence[str]):
    cm, cn = country_matrix(frame, terms)
    x = np.column_stack([np.ones(len(frame)), continuous, cm])
    names = ["Intercept", *predictors, *cn]
    if x.shape[1] != len(names):
        raise RuntimeError("Design mismatch")
    return x, names


def full_design(frame: pd.DataFrame, predictors: Sequence[str], scheme: str, terms: Sequence[str]):
    scaler, cont = fit_scaler(frame, predictors, scheme)
    x, names = assemble(frame, cont, predictors, terms)
    rows = []
    if scaler is not None:
        for i, name in enumerate(predictors):
            rows.append(dict(Variable=name, Transformation_Scheme=scheme,
                             Log1p_Applied=bool(scheme == "Audit_Guided_Log1p" and name in LOG1P_VARIABLES),
                             Transformed_Mean=float(scaler.mean_[i]), Transformed_Scale=float(scaler.scale_[i])))
    return x, names, pd.DataFrame(rows)


def train_valid_design(train: pd.DataFrame, valid: pd.DataFrame, predictors: Sequence[str], scheme: str, terms: Sequence[str]):
    scaler, tr_cont = fit_scaler(train, predictors, scheme)
    va_cont = apply_scaler(valid, predictors, scheme, scaler)
    xtr, names = assemble(train, tr_cont, predictors, terms)
    xva, names2 = assemble(valid, va_cont, predictors, terms)
    if names != names2:
        raise RuntimeError("Training/validation design mismatch")
    return xtr, xva, names


def nb1_parts(y: np.ndarray, x: np.ndarray, exposure: np.ndarray, beta: np.ndarray, log_alpha: float):
    alpha = float(np.exp(np.clip(log_alpha, *LOG_ALPHA_BOUNDS)))
    eta = x @ beta + np.log(exposure)
    mu = np.exp(np.clip(eta, -50.0, 50.0))
    size = mu / alpha
    logp = -np.log1p(alpha)
    logq = np.log(alpha) - np.log1p(alpha)
    ll = gammaln(y + size) - gammaln(size) - gammaln(y + 1.0) + size * logp + y * logq
    a = digamma(y + size) - digamma(size) + logp
    score_eta = size * a
    score_loga = -size * a - mu / (1.0 + alpha) + y / (1.0 + alpha)
    return alpha, mu, ll, score_eta, score_loga


def loglike_score_obs(theta: np.ndarray, y: np.ndarray, x: np.ndarray, z: np.ndarray, exposure: np.ndarray, model: str):
    kx = x.shape[1]
    if model == "Single_Stage_NB1":
        alpha, mu, ll, se, sa = nb1_parts(y, x, exposure, theta[:kx], theta[-1])
        return ll, np.column_stack([se[:, None] * x, sa])
    if model != "ZINB1":
        raise ValueError(model)
    kz = z.shape[1]
    gamma, beta, loga = theta[:kz], theta[kz:kz + kx], theta[-1]
    alpha, mu, nbll, se, sa = nb1_parts(y, x, exposure, beta, loga)
    xi = z @ gamma
    pi = expit(xi)
    logpi = -np.logaddexp(0.0, -xi)
    log1mpi = -np.logaddexp(0.0, xi)
    zero = y == 0
    pos = ~zero
    ll = np.empty(len(y))
    sg = np.empty((len(y), kz)); sb = np.empty((len(y), kx)); sla = np.empty(len(y))
    ll[pos] = log1mpi[pos] + nbll[pos]
    sg[pos] = -pi[pos, None] * z[pos]
    sb[pos] = se[pos, None] * x[pos]
    sla[pos] = sa[pos]
    logp0 = np.logaddexp(logpi[zero], log1mpi[zero] + nbll[zero])
    ll[zero] = logp0
    tau = np.exp(logpi[zero] - logp0)
    w = 1.0 - tau
    sg[zero] = (tau - pi[zero])[:, None] * z[zero]
    sb[zero] = (w * se[zero])[:, None] * x[zero]
    sla[zero] = w * sa[zero]
    return ll, np.column_stack([sg, sb, sla])


def objective(theta, y, x, z, exposure, model):
    try:
        ll, score = loglike_score_obs(theta, y, x, z, exposure, model)
        if not np.isfinite(ll).all() or not np.isfinite(score).all():
            raise FloatingPointError
        return -float(ll.sum()), -score.sum(axis=0)
    except Exception:
        return 1e100, np.zeros_like(theta)


def optimize_strict(start: np.ndarray, y: np.ndarray, x: np.ndarray, z: np.ndarray, exposure: np.ndarray, model: str):
    bounds = [(-COEF_BOUND, COEF_BOUND)] * (len(start) - 1) + [LOG_ALPHA_BOUNDS]
    fun = lambda t: objective(t, y, x, z, exposure, model)[0]
    jac = lambda t: objective(t, y, x, z, exposure, model)[1]
    result = optimize.minimize(fun, start, jac=jac, method="L-BFGS-B", bounds=bounds,
                               options={"maxiter": MAXITER, "ftol": 1e-12, "gtol": 1e-7, "maxls": 60, "maxcor": 30})
    theta = np.asarray(result.x, float)
    value, grad = objective(theta, y, x, z, exposure, model)
    score = loglike_score_obs(theta, y, x, z, exposure, model)[1]
    grad_inf = float(np.max(np.abs(grad)))
    rel_grad = grad_inf / max(float(np.sqrt(np.sum(score ** 2))), np.finfo(float).tiny)
    lower = np.array([b[0] for b in bounds]); upper = np.array([b[1] for b in bounds])
    hit = bool(np.any(np.abs(theta - lower) <= 0.01) or np.any(np.abs(theta - upper) <= 0.01))
    status = dict(Optimizer="L-BFGS-B", Optimizer_Success=bool(result.success),
                  Converged=bool((result.success or grad_inf <= MAX_GRAD) and np.isfinite(value)),
                  Status_Code=int(result.status), Message=str(result.message), Iteration_Count=int(result.nit),
                  Function_Evaluation_Count=int(result.nfev), Gradient_Evaluation_Count=int(result.njev),
                  Negative_LogLikelihood=float(value), LogLikelihood=float(-value),
                  Gradient_Infinity_Norm=grad_inf, Relative_Gradient=rel_grad,
                  Gradient_Within_Tolerance=bool(grad_inf <= MAX_GRAD or rel_grad <= MAX_REL_GRAD),
                  Parameter_Bound_Hit=hit, Alpha_Estimate=float(np.exp(theta[-1])))
    return theta, status



def optimize_multistart(starts, y, x, z, exposure, model):
    results = []
    seen = set()
    for start in starts:
        start = np.asarray(start, float)
        key = tuple(np.round(start, 8))
        if key in seen:
            continue
        seen.add(key)
        theta, status = optimize_strict(start, y, x, z, exposure, model)
        results.append((theta, status))
    if not results:
        raise RuntimeError("No optimization starts were supplied.")
    theta, status = min(results, key=lambda item: item[1]["Negative_LogLikelihood"])
    status = dict(status)
    status["Attempt_Count"] = len(results)
    status["Successful_Attempt_Count"] = int(sum(bool(item[1]["Converged"]) for item in results))
    return theta, status

def neutral_start(y, x, z, exposure, model):
    beta = np.zeros(x.shape[1]); beta[0] = np.log(max(float(y.sum()) / float(exposure.sum()), 1e-10))
    if model == "Single_Stage_NB1":
        return np.r_[beta, 0.0]
    gamma = np.zeros(z.shape[1])
    p = np.clip(float(np.mean(y == 0)) - 0.05, 0.05, 0.95)
    gamma[0] = stats.logistic.ppf(p)
    return np.r_[gamma, beta, 0.0]


def training_start(step04, step05, y, x, z, exposure, model):
    try:
        if model == "Single_Stage_NB1":
            result, _ = step04.fit_nb1(y, x, exposure)
            raw = np.asarray(result.params, float)
        else:
            raw, _ = step05.fit_zinb_component(step04, y, x, z, exposure)
            raw = np.asarray(raw, float)
        if not np.isfinite(raw).all() or raw[-1] <= 0:
            raise ValueError
        return np.r_[raw[:-1], np.log(raw[-1])]
    except Exception:
        return neutral_start(y, x, z, exposure, model)


def step06b_start(coef: pd.DataFrame, season: int, response: str, model: str, znames: Sequence[str], xnames: Sequence[str]):
    sub = coef[(coef.Season_Code.astype(int) == season) & (coef.Count_Response.astype(str) == response)]
    lookup = {(str(r.Component), str(r.Parameter)): float(r.Estimate) for r in sub.itertuples(index=False)}
    values = []
    if model == "ZINB1":
        values += [lookup.get(("Structural_Zero", n), 0.0) for n in znames]
    values += [lookup.get(("Count", n), 0.0) for n in xnames]
    alpha = lookup.get(("Dispersion", "alpha"), 1.0)
    if not np.isfinite(alpha) or alpha <= 0:
        alpha = 1.0
    return np.asarray([*values, np.log(alpha)], float)


def stable_predictions(theta, y, x, z, exposure, model):
    kx = x.shape[1]
    if model == "Single_Stage_NB1":
        _, mu, ll, _, _ = nb1_parts(y, x, exposure, theta[:kx], theta[-1])
        _, _, ll0, _, _ = nb1_parts(np.zeros_like(y), x, exposure, theta[:kx], theta[-1])
        return dict(Predicted_Count=mu, Predicted_Zero_Probability=np.clip(np.exp(ll0), EPS, 1.0), Predictive_LogProbability=ll)
    kz = z.shape[1]
    gamma, beta = theta[:kz], theta[kz:kz + kx]
    _, mu, nbll, _, _ = nb1_parts(y, x, exposure, beta, theta[-1])
    _, _, nbll0, _, _ = nb1_parts(np.zeros_like(y), x, exposure, beta, theta[-1])
    xi = z @ gamma
    logpi = -np.logaddexp(0.0, -xi); log1mpi = -np.logaddexp(0.0, xi)
    logzero = np.logaddexp(logpi, log1mpi + nbll0)
    return dict(Predicted_Count=(1.0 - expit(xi)) * mu,
                Predicted_Zero_Probability=np.clip(np.exp(logzero), EPS, 1.0),
                Predictive_LogProbability=np.where(y == 0, logzero, log1mpi + nbll))


def prediction_frame(valid, response, model, fold, out, candidate):
    result = valid[["GRID_UID", "GRID_ID", "Country", "Year", "Season", "ForestPixelCount"]].copy()
    y = valid[response].to_numpy(float); exposure = valid.ForestPixelCount.to_numpy(float)
    result["Temporal_Fold"] = fold; result["Count_Response"] = response; result["Rate_Scale_Name"] = RATE_NAMES[response]
    result["Model_Structure"] = model; result["Candidate_ID"] = candidate
    result["Observed_Count"] = y; result["Predicted_Count"] = out["Predicted_Count"]
    result["Observed_Rate"] = y / exposure; result["Predicted_Rate"] = out["Predicted_Count"] / exposure
    result["Predicted_Zero_Probability"] = out["Predicted_Zero_Probability"]
    result["Predictive_LogProbability"] = out["Predictive_LogProbability"]
    return result


def total_score(theta, y, x, z, exposure, model):
    return loglike_score_obs(theta, y, x, z, exposure, model)[1].sum(axis=0)


def numerical_hessian(theta, y, x, z, exposure, model):
    p = len(theta); h = np.empty((p, p))
    for j in range(p):
        step = 1e-4 * (1.0 + abs(theta[j]))
        up = theta.copy(); down = theta.copy(); up[j] += step; down[j] -= step
        h[:, j] = (total_score(up, y, x, z, exposure, model) - total_score(down, y, x, z, exposure, model)) / (2.0 * step)
    return (h + h.T) / 2.0


def cluster_meat(score: np.ndarray, groups: Sequence[object]):
    codes, uniques = pd.factorize(groups, sort=False)
    n, p = score.shape; g = len(uniques)
    sums = np.zeros((g, p)); np.add.at(sums, codes, score)
    correction = g / (g - 1.0) * (n - 1.0) / (n - p)
    return correction * (sums.T @ sums), g, correction


def covariance_bundle(theta, y, x, z, exposure, model, grid, year):
    ll, score = loglike_score_obs(theta, y, x, z, exposure, model)
    info = -numerical_hessian(theta, y, x, z, exposure, model)
    info = (info + info.T) / 2.0
    eig = np.linalg.eigvalsh(info); scale = max(float(np.max(np.abs(eig))), 1.0)
    rank = int(np.linalg.matrix_rank(info, tol=INFO_RCOND * scale)); condition = float(np.linalg.cond(info))
    direct = rank == len(theta) and np.isfinite(condition) and condition <= MAX_INFO_CONDITION
    bread = np.linalg.inv(info) if direct else np.linalg.pinv(info, rcond=INFO_RCOND)
    bread = (bread + bread.T) / 2.0
    mg, gg, cg = cluster_meat(score, grid); my, gy, cy = cluster_meat(score, year)
    inter = pd.MultiIndex.from_arrays([np.asarray(grid), np.asarray(year)])
    mi, gi, ci = cluster_meat(score, inter)
    cg_cov = bread @ mg @ bread.T; cy_cov = bread @ my @ bread.T; ci_cov = bread @ mi @ bread.T
    tw = cg_cov + cy_cov - ci_cov; tw = (tw + tw.T) / 2.0
    tw_eig = np.linalg.eigvalsh(tw); tw_scale = max(float(np.max(np.abs(tw_eig))), np.finfo(float).tiny)
    tw_psd = bool(float(tw_eig.min()) >= -PSD_REL_TOL * tw_scale)
    tw_diag = bool((np.diag(tw) > 0).all()); tw_finite = bool(np.isfinite(tw).all())
    grad = score.sum(axis=0); grad_inf = float(np.max(np.abs(grad)))
    rel_grad = grad_inf / max(float(np.sqrt(np.sum(score ** 2))), np.finfo(float).tiny)
    diagnostics = dict(LogLikelihood=float(ll.sum()), Parameter_Count=len(theta), Gradient_Infinity_Norm=grad_inf,
                       Relative_Gradient=rel_grad, Gradient_Within_Tolerance=bool(grad_inf <= MAX_GRAD or rel_grad <= MAX_REL_GRAD),
                       Information_Rank=rank, Information_Full_Rank=bool(rank == len(theta)),
                       Information_Minimum_Eigenvalue=float(eig.min()), Information_Maximum_Eigenvalue=float(eig.max()),
                       Information_Condition_Number=condition, Information_Condition_Acceptable=bool(np.isfinite(condition) and condition <= MAX_INFO_CONDITION),
                       Bread_Inversion_Method="Direct_inverse" if direct else "Moore_Penrose_pseudoinverse",
                       GRID_UID_Cluster_Count=gg, Year_Cluster_Count=gy, Intersection_Cluster_Count=gi,
                       GRID_UID_CR1_Correction=cg, Year_CR1_Correction=cy, Intersection_CR1_Correction=ci,
                       Two_Way_Minimum_Eigenvalue=float(tw_eig.min()), Two_Way_Maximum_Absolute_Eigenvalue=tw_scale,
                       Two_Way_Positive_Semidefinite=tw_psd, Two_Way_All_Diagonal_Positive=tw_diag,
                       Two_Way_Condition_Number=float(np.linalg.cond(tw)), Raw_Two_Way_Covariance_Usable=bool(tw_finite and tw_psd and tw_diag))
    return {"Model_Based": bread, "GRID_UID_Cluster": (cg_cov + cg_cov.T) / 2.0,
            "Year_Cluster": (cy_cov + cy_cov.T) / 2.0, "GRID_UID_Year_Intersection": (ci_cov + ci_cov.T) / 2.0,
            "Two_Way_Cluster": tw, "Information_Matrix": info}, diagnostics


def metadata(model, znames, xnames):
    rows = []
    if model == "ZINB1":
        rows += [dict(Component="Structural_Zero", Parameter=n, Parameter_Label=f"inflate_{n}", Effect_Scale="Excess_Zero_Odds_Ratio") for n in znames]
    rows += [dict(Component="Count", Parameter=n, Parameter_Label=n, Effect_Scale="Count_Rate_Ratio") for n in xnames]
    rows.append(dict(Component="Dispersion", Parameter="log_alpha", Parameter_Label="log_alpha", Effect_Scale="Log_NB1_Dispersion"))
    return pd.DataFrame(rows)


def coef_table(theta, meta, cov, df):
    out = meta.copy(); out["Estimate"] = theta
    diag = np.diag(cov); se = np.where(np.isfinite(diag) & (diag > 0), np.sqrt(diag), np.nan)
    t = theta / se; crit = stats.t.ppf(0.975, df=df)
    out["Two_Way_Cluster_SE"] = se; out["Two_Way_Cluster_t"] = t; out["Two_Way_Cluster_df"] = df
    out["Two_Way_Cluster_p"] = 2.0 * stats.t.sf(np.abs(t), df=df)
    out["Two_Way_Cluster_CI_Lower"] = theta - crit * se; out["Two_Way_Cluster_CI_Upper"] = theta + crit * se
    out["Exponentiated_Estimate"] = np.exp(np.clip(theta, -700, 700))
    return out


def attach_fold(model_data, oof):
    fmap = oof[["GRID_UID", "Year", "Season", "Temporal_Fold"]].copy(); fmap["GRID_UID"] = fmap.GRID_UID.astype(str)
    result = model_data.merge(fmap, on=["GRID_UID", "Year", "Season"], how="left", validate="one_to_one")
    if result.Temporal_Fold.isna().any() or set(result.Temporal_Fold.astype(int)) != set(FOLDS):
        raise ValueError("Temporal folds could not be attached")
    result["Temporal_Fold"] = result.Temporal_Fold.astype(int)
    return result


def sparsity_rows(data, season, response):
    rows = []
    for country, sub in data.groupby("Country", sort=True):
        pos = int((sub[response] > 0).sum()); zero = int((sub[response] == 0).sum()); n = len(sub); prop = pos / n
        rows.append(dict(Season_Code=season, Season_Label=SEASONS[season], Count_Response=response, Rate_Scale_Name=RATE_NAMES[response],
                         Country=country, Rows=n, Positive_Row_Count=pos, Zero_Row_Count=zero,
                         Observed_Total_Count=float(sub[response].sum()), Event_Proportion=prop,
                         Complete_Categorical_Separation=bool(pos == 0 or zero == 0),
                         Quasi_Categorical_Separation=bool(pos > 0 and zero > 0 and (min(pos, zero) < 10 or prop < .001 or prop > .999))))
    return rows


def overlap_rows(data, season, response, predictors, scheme):
    if not predictors:
        return []
    transformed = transform(data, predictors, scheme); positive = data[response].to_numpy(float) > 0; rows = []
    for name in predictors:
        a, b = transformed[name].to_numpy(float)[positive], transformed[name].to_numpy(float)[~positive]
        amin, amax, bmin, bmax = float(a.min()), float(a.max()), float(b.min()), float(b.max())
        lower, upper = max(amin, bmin), min(amax, bmax)
        pooled = max(max(amax, bmax) - min(amin, bmin), np.finfo(float).tiny)
        rows.append(dict(Season_Code=season, Season_Label=SEASONS[season], Count_Response=response, Variable=name,
                         Transformation_Scheme=scheme, Positive_Min=amin, Positive_Max=amax, Zero_Min=bmin, Zero_Max=bmax,
                         Range_Overlap_Fraction=max(0.0, upper - lower) / pooled,
                         Complete_Univariate_Threshold_Separation=bool(upper < lower)))
    return rows


def candidate_components(c):
    return dict(zp=parse_list(c.Structural_Zero_Predictors), cp=parse_list(c.Count_Predictors),
                zc=parse_list(c.Structural_Zero_Country_Terms), cc=parse_list(c.Count_Country_Terms),
                scheme=str(c.Transformation_Scheme), model=str(c.Model_Structure))


def full_checkpoint(c):
    prefix = f"S{int(c.Season_Code)}_{safe_name(c.Count_Response)}_{safe_name(c.Candidate_ID)}_{c.Candidate_Signature}"
    return FULL_CHECKPOINT_ROOT / f"{prefix}.npz", FULL_CHECKPOINT_ROOT / f"{prefix}.json"


def oof_checkpoint(c, fold):
    return OOF_CHECKPOINT_ROOT / f"S{int(c.Season_Code)}_{safe_name(c.Count_Response)}_{safe_name(c.Candidate_ID)}_{c.Candidate_Signature}_Fold{fold}.csv.gz"


def fit_full_candidate(c, data, source_coef):
    comp = candidate_components(c)
    x, xnames, xscale = full_design(data, comp["cp"], comp["scheme"], comp["cc"])
    if comp["model"] == "ZINB1":
        z, znames, zscale = full_design(data, comp["zp"], comp["scheme"], comp["zc"])
    else:
        z, znames, zscale = np.empty((len(data), 0)), [], pd.DataFrame()
    y = data[c.Count_Response].to_numpy(float); exposure = data.ForestPixelCount.to_numpy(float)
    start = step06b_start(source_coef, int(c.Season_Code), str(c.Count_Response), comp["model"], znames, xnames)
    npz, js = full_checkpoint(c)
    if npz.exists() and js.exists():
        theta = np.load(npz)["theta"]; status = json.loads(js.read_text(encoding="utf-8")); log(f"Reused full checkpoint: {c.Season_Label} {c.Count_Response} {c.Candidate_ID}")
    else:
        neutral = neutral_start(y, x, z, exposure, comp["model"])
        theta, status = optimize_multistart([start, neutral], y, x, z, exposure, comp["model"])
        np.savez_compressed(npz, theta=theta); js.write_text(json.dumps(status, indent=2), encoding="utf-8")
    covs, diag = covariance_bundle(theta, y, x, z, exposure, comp["model"], data.GRID_UID.astype(str), data.Year.astype(int))
    meta = metadata(comp["model"], znames, xnames); df = min(diag["GRID_UID_Cluster_Count"] - 1, diag["Year_Cluster_Count"] - 1)
    coefs = coef_table(theta, meta, covs["Two_Way_Cluster"], df)
    non_disp = coefs.loc[coefs.Component != "Dispersion", "Estimate"].to_numpy(float)
    extreme = int((np.abs(non_disp) >= EXTREME_COEF).sum())
    row = {"Season_Code": c.Season_Code, "Season_Label": c.Season_Label, "Count_Response": c.Count_Response,
           "Rate_Scale_Name": c.Rate_Scale_Name, "Model_Structure": c.Model_Structure, "Candidate_ID": c.Candidate_ID,
           "Candidate_Type": c.Candidate_Type, "Candidate_Signature": c.Candidate_Signature,
           "Terms_Removed_from_Step05": c.Terms_Removed_from_Step05, "N": len(data), **status, **diag,
           "Extreme_Coefficient_Count": extreme,
           "Maximum_Absolute_NonDispersion_Coefficient": float(np.max(np.abs(non_disp)))}
    row["Full_Data_Stable"] = bool(row["Converged"] and row["Gradient_Within_Tolerance"] and row["Information_Full_Rank"]
                                    and row["Information_Condition_Acceptable"] and row["Raw_Two_Way_Covariance_Usable"]
                                    and not row["Parameter_Bound_Hit"] and extreme == 0)
    coefs.insert(0, "Candidate_ID", c.Candidate_ID); coefs.insert(0, "Model_Structure", c.Model_Structure)
    coefs.insert(0, "Count_Response", c.Count_Response); coefs.insert(0, "Season_Label", c.Season_Label); coefs.insert(0, "Season_Code", c.Season_Code)
    scales = []
    for component, table in [("Count", xscale), ("Structural_Zero", zscale)]:
        if not table.empty:
            table = table.copy(); table.insert(0, "Component", component); table.insert(0, "Candidate_ID", c.Candidate_ID)
            table.insert(0, "Count_Response", c.Count_Response); table.insert(0, "Season_Code", c.Season_Code); scales.append(table)
    scale_out = pd.concat(scales, ignore_index=True) if scales else pd.DataFrame()
    prefix = f"S{int(c.Season_Code)}_{safe_name(c.Count_Response)}_{safe_name(c.Candidate_ID)}"
    labels = meta.Parameter_Label.astype(str).tolist()
    for name, matrix in covs.items():
        table = pd.DataFrame(matrix, index=labels, columns=labels); table.index.name = "Parameter"
        table.to_csv(COVARIANCE_ROOT / f"{prefix}_{name}.csv", encoding="utf-8-sig")
    return row, coefs, scale_out


def run_oof(c, data, step04, step05):
    comp = candidate_components(c); response = str(c.Count_Response); predictions, status_rows = [], []
    for fold in FOLDS:
        path = oof_checkpoint(c, fold)
        if path.exists():
            pred = pd.read_csv(path, compression="gzip"); predictions.append(pred)
            status_rows.append(dict(Season_Code=c.Season_Code, Season_Label=c.Season_Label, Count_Response=response,
                                    Candidate_ID=c.Candidate_ID, Temporal_Fold=fold, Checkpoint_Reused=True,
                                    Converged=True, Gradient_Within_Tolerance=True, Parameter_Bound_Hit=False,
                                    Negative_LogLikelihood=float(-pred.Predictive_LogProbability.sum())))
            log(f"Reused OOF checkpoint: {c.Season_Label} {response} {c.Candidate_ID} fold {fold}")
            continue
        train = data[data.Temporal_Fold != fold].copy(); valid = data[data.Temporal_Fold == fold].copy()
        xtr, xva, xnames = train_valid_design(train, valid, comp["cp"], comp["scheme"], comp["cc"])
        if comp["model"] == "ZINB1":
            ztr, zva, znames = train_valid_design(train, valid, comp["zp"], comp["scheme"], comp["zc"])
        else:
            ztr, zva = np.empty((len(train), 0)), np.empty((len(valid), 0))
        ytr = train[response].to_numpy(float); etr = train.ForestPixelCount.to_numpy(float)
        start = training_start(step04, step05, ytr, xtr, ztr, etr, comp["model"])
        neutral = neutral_start(ytr, xtr, ztr, etr, comp["model"])
        theta, st = optimize_multistart([start, neutral], ytr, xtr, ztr, etr, comp["model"])
        yva = valid[response].to_numpy(float); eva = valid.ForestPixelCount.to_numpy(float)
        out = stable_predictions(theta, yva, xva, zva, eva, comp["model"])
        pred = prediction_frame(valid, response, comp["model"], fold, out, c.Candidate_ID)
        pred.to_csv(path, index=False, compression="gzip"); predictions.append(pred)
        status_rows.append({"Season_Code": c.Season_Code, "Season_Label": c.Season_Label, "Count_Response": response,
                            "Candidate_ID": c.Candidate_ID, "Temporal_Fold": fold, "Checkpoint_Reused": False, **st})
        log(f"Completed OOF fold {fold}: {c.Season_Label} {response} {c.Candidate_ID}")
    pred = pd.concat(predictions, ignore_index=True).sort_values(["GRID_UID", "Year", "Season"]).reset_index(drop=True)
    return pred, pd.DataFrame(status_rows)


def make_selection(candidates, full, metrics):
    key = ["Season_Code", "Count_Response", "Candidate_ID"]
    result = candidates.merge(full, on=key, how="left", validate="one_to_one", suffixes=("", "_Full"))
    result = result.merge(metrics, on=key, how="left", validate="one_to_one", suffixes=("", "_OOF"))
    result["OOF_NLL_Reference"] = np.nan; result["OOF_NLL_Noninferiority_Tolerance"] = np.nan; result["OOF_Noninferior"] = False
    for (_, _), idx in result.groupby(["Season_Code", "Count_Response"]).groups.items():
        group = result.loc[idx]; base = group[group.Candidate_ID == "BASELINE_STRICT"]
        if base.empty:
            continue
        nll = float(base.Mean_Negative_LogLikelihood.iloc[0])
        tol = max(OOF_ABS_TOL, OOF_REL_TOL * nll)
        result.loc[idx, "OOF_NLL_Reference"] = nll; result.loc[idx, "OOF_NLL_Noninferiority_Tolerance"] = tol
        result.loc[idx, "OOF_Noninferior"] = result.loc[idx, "Mean_Negative_LogLikelihood"] <= nll + tol
    result["Eligible_for_Selection"] = result.Full_Data_Stable.astype(bool) & result.OOF_Noninferior.astype(bool)
    result["Removed_Term_Count"] = result.Terms_Removed_from_Step05.fillna("").apply(lambda v: len([x for x in str(v).split(";") if x.strip()]))
    result["Selected"] = False; result["Selection_Status"] = "Not selected"
    for _, idx in result.groupby(["Season_Code", "Count_Response"]).groups.items():
        group = result.loc[idx]; eligible = group[group.Eligible_for_Selection].copy()
        if eligible.empty:
            result.loc[idx, "Selection_Status"] = "Unresolved: no stable OOF-noninferior candidate"; continue
        eligible["Baseline_Preference"] = (eligible.Candidate_ID != "BASELINE_STRICT").astype(int)
        chosen = eligible.sort_values(["Removed_Term_Count", "Baseline_Preference", "Mean_Negative_LogLikelihood", "Candidate_Order"]).iloc[0]
        result.loc[chosen.name, "Selected"] = True; result.loc[chosen.name, "Selection_Status"] = "Selected stable fixed-effect inference structure"
        other = result.index.isin(idx) & (result.index != chosen.name)
        result.loc[other, "Selection_Status"] = "Not selected after stability and temporal OOF comparison"
    return result


def main() -> int:
    for path in [OUTPUT_ROOT, CHECKPOINT_ROOT, OOF_CHECKPOINT_ROOT, FULL_CHECKPOINT_ROOT, COVARIANCE_ROOT, SELECTED_OOF_ROOT]:
        path.mkdir(parents=True, exist_ok=True)
    log("Starting Step 06c stable-likelihood audit.")
    step04 = load_module(STEP04_SCRIPT, "step04_06c")
    step05 = load_module(STEP05_SCRIPT, "step05_06c")
    step06b = load_module(STEP06B_SCRIPT, "step06b_06c")
    if not STEP06B_COEF.exists():
        raise FileNotFoundError(STEP06B_COEF)
    structure = step06b.read_final_structure(); base = step06b.read_base_table(structure); source_coef = pd.read_csv(STEP06B_COEF)
    candidates = candidate_definitions(structure)
    candidates.to_csv(OUTPUT_ROOT / "04_Candidate_Definitions.csv", index=False, encoding="utf-8-sig")

    samples, source_oofs = {}, {}
    sparse, overlap = [], []
    for d in structure.itertuples(index=False):
        s, r = int(d.Season_Code), str(d.Count_Response)
        predictors = list(dict.fromkeys(parse_list(d.Structural_Zero_Predictors) + parse_list(d.Count_Predictors)))
        data, oof = step06b.exact_model_sample(base, s, r, str(d.Model_Structure), predictors)
        data = attach_fold(data, oof); samples[(s, r)] = data; source_oofs[(s, r)] = oof
        sparse += sparsity_rows(data, s, r)
        overlap += overlap_rows(data, s, r, parse_list(d.Structural_Zero_Predictors), str(d.Transformation_Scheme))
    pd.DataFrame(sparse).to_csv(OUTPUT_ROOT / "02_Country_Outcome_Sparsity_and_Separation.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(overlap).to_csv(OUTPUT_ROOT / "03_Continuous_Predictor_Overlap_Audit.csv", index=False, encoding="utf-8-sig")

    full_rows, coef_tables, scale_tables, fold_tables, metric_rows = [], [], [], [], []
    prediction_cache = {}
    for c in candidates.itertuples(index=False):
        data = samples[(int(c.Season_Code), str(c.Count_Response))]
        log(f"Full fit: {c.Season_Label} {c.Count_Response} {c.Candidate_ID}")
        row, coefs, scales = fit_full_candidate(c, data, source_coef)
        full_rows.append(row); coef_tables.append(coefs)
        if not scales.empty:
            scale_tables.append(scales)
        if bool(c.Targeted_OOF_Required):
            log(f"OOF confirmation: {c.Season_Label} {c.Count_Response} {c.Candidate_ID}")
            pred, fold_status = run_oof(c, data, step04, step05); fold_tables.append(fold_status)
            prediction_cache[(int(c.Season_Code), str(c.Count_Response), str(c.Candidate_ID))] = pred
            source = "Strict stable-likelihood five-fold refit"
        else:
            pred = source_oofs[(int(c.Season_Code), str(c.Count_Response))].copy(); source = "Reused Step-05 selected OOF baseline"
        metrics = step04.prediction_metrics(pred.Observed_Count.to_numpy(float), pred.Predicted_Count.to_numpy(float),
                                            pred.ForestPixelCount.to_numpy(float), pred.Predicted_Zero_Probability.to_numpy(float),
                                            pred.Predictive_LogProbability.to_numpy(float))
        metric_rows.append(dict(Season_Code=c.Season_Code, Season_Label=c.Season_Label, Count_Response=c.Count_Response,
                                Rate_Scale_Name=c.Rate_Scale_Name, Model_Structure=c.Model_Structure,
                                Candidate_ID=c.Candidate_ID, **metrics, OOF_Source=source))

    full = pd.DataFrame(full_rows); coefs = pd.concat(coef_tables, ignore_index=True)
    scales = pd.concat(scale_tables, ignore_index=True) if scale_tables else pd.DataFrame()
    folds = pd.concat(fold_tables, ignore_index=True) if fold_tables else pd.DataFrame()
    metrics = pd.DataFrame(metric_rows)
    comparison = make_selection(candidates, full, metrics)
    selected = comparison[comparison.Selected.astype(bool)].copy()

    final_rows, manifest = [], []
    for row in selected.itertuples(index=False):
        final_rows.append(dict(Season_Code=row.Season_Code, Season_Label=row.Season_Label, Count_Response=row.Count_Response,
                               Rate_Scale_Name=row.Rate_Scale_Name, Model_Structure=row.Model_Structure,
                               Selected_Inference_Candidate_ID=row.Candidate_ID,
                               Structural_Zero_Predictors=row.Structural_Zero_Predictors,
                               Structural_Zero_Country_Terms=row.Structural_Zero_Country_Terms,
                               Count_Predictors=row.Count_Predictors, Count_Country_Terms=row.Count_Country_Terms,
                               Transformation_Scheme=row.Transformation_Scheme, Terms_Removed_from_Step05=row.Terms_Removed_from_Step05,
                               Full_Data_Stable=row.Full_Data_Stable, OOF_Noninferior=row.OOF_Noninferior,
                               Mean_Negative_LogLikelihood=row.Mean_Negative_LogLikelihood, Selection_Status=row.Selection_Status))
        dest = SELECTED_OOF_ROOT / f"Selected_Inference_OOF_Predictions_S{int(row.Season_Code)}_{row.Count_Response}.csv.gz"
        key = (int(row.Season_Code), str(row.Count_Response), str(row.Candidate_ID))
        if key in prediction_cache:
            pred = prediction_cache[key]; source = "Strict stable-likelihood refit"
        else:
            pred = pd.read_csv(step06b.selected_oof_path(int(row.Season_Code), str(row.Count_Response)), compression="gzip")
            pred["Candidate_ID"] = row.Candidate_ID; source = "Copied Step-05 selected OOF"
        pred.to_csv(dest, index=False, compression="gzip")
        manifest.append(dict(Season_Code=row.Season_Code, Season_Label=row.Season_Label, Count_Response=row.Count_Response,
                             Selected_Inference_Candidate_ID=row.Candidate_ID, OOF_Source=source, Output_File=dest.name, Rows=len(pred)))
    final = pd.DataFrame(final_rows); manifest = pd.DataFrame(manifest)

    global_qa = pd.DataFrame([
        ("Expected_model_count", 6), ("Candidate_count", len(candidates)),
        ("Targeted_OOF_candidate_count", int(candidates.Targeted_OOF_Required.astype(bool).sum())),
        ("Strict_full_data_fit_count", len(full)), ("Strict_full_data_stable_count", int(full.Full_Data_Stable.astype(bool).sum())),
        ("Selected_inference_structure_count", len(selected)), ("All_six_inference_structures_resolved", len(selected) == 6),
        ("Step06b_Summer_BA_derivative_diagnostics_reused", False),
        ("Stable_likelihood_parameterization", "log(alpha)")], columns=["Metric", "Value"])
    method = dict(workflow_stage="Parameter identifiability and stable likelihood audit",
                  likelihoods="Explicit NB1 and ZINB1; size=mu/alpha; log-sum-exp mixture",
                  dispersion_parameterization="log(alpha)", optimizer="L-BFGS-B with analytic gradient",
                  hessian="Central differences of analytic total score",
                  cluster_covariance="GRID_UID + Year - GRID_UID-by-Year",
                  targeted_models=["Spring Fire_Count", "Summer Burned_Pixel_Count", "Autumn Burned_Pixel_Count"],
                  oof_validation="Same five temporal folds as Steps 03-05",
                  oof_noninferiority=dict(relative=OOF_REL_TOL, absolute=OOF_ABS_TOL))
    environment = dict(python=sys.version, platform=platform.platform(), numpy=np.__version__, pandas=pd.__version__,
                       scipy=scipy.__version__, scikit_learn=sklearn.__version__, statsmodels=statsmodels.__version__,
                       run_timestamp=datetime.now().isoformat(timespec="seconds"))
    (OUTPUT_ROOT / "00_Method_Definition.json").write_text(json.dumps(method, indent=2), encoding="utf-8")
    global_qa.to_csv(OUTPUT_ROOT / "01_Global_QA_Summary.csv", index=False, encoding="utf-8-sig")
    full.to_csv(OUTPUT_ROOT / "05_Strict_Full_Data_Fit_and_Stability.csv", index=False, encoding="utf-8-sig")
    coefs.to_csv(OUTPUT_ROOT / "06_Candidate_Coefficients_and_Two_Way_Inference.csv", index=False, encoding="utf-8-sig")
    scales.to_csv(OUTPUT_ROOT / "07_Candidate_Transformation_and_Scaling.csv", index=False, encoding="utf-8-sig")
    folds.to_csv(OUTPUT_ROOT / "08_Targeted_OOF_Fold_Fit_Status.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(OUTPUT_ROOT / "09_Targeted_OOF_Performance.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(OUTPUT_ROOT / "10_Candidate_Comparison_and_Formal_Selection.csv", index=False, encoding="utf-8-sig")
    final.to_csv(OUTPUT_ROOT / "11_Final_Stable_Fixed_Effect_Inference_Structures.csv", index=False, encoding="utf-8-sig")
    manifest.to_csv(OUTPUT_ROOT / "12_Selected_Inference_OOF_Manifest.csv", index=False, encoding="utf-8-sig")
    (OUTPUT_ROOT / "Software_Environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    log("Step 06c completed.")
    for row in selected.itertuples(index=False):
        log(f"Selected: {row.Season_Label} {row.Count_Response} {row.Candidate_ID}; stable={row.Full_Data_Stable}; OOF_noninferior={row.OOF_Noninferior}")
    if len(selected) != 6:
        log("WARNING: at least one fixed-effect inference structure remains unresolved. Do not proceed to final reporting yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
