# -*- coding: utf-8 -*-
r"""
Reproducible fire-atmospheric-circulation analysis and figure generation, 2001-2025.

Annual fire activity is calculated from canonical event tables and seasonal
activity is calculated directly from the canonical base table. Climate windows use the exact DOY
season definitions (44-160, 161-218 and 269-331), including leap years.

Run this file from any working directory:
    python atmospheric_circulation_analysis_and_figures.py

The script locates Code/00_Input_data relative to its own position and writes
processed data, statistical tables, QA reports, logs and figures under
05_Atmospheric_circulation/output/.
"""
from __future__ import annotations

import calendar
import csv
import json
import math
import platform
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_DIR.parent
INPUT = CODE_ROOT / "00_Input_data"
PROCESSED = SCRIPT_DIR / "output" / "data_processed"
RESULTS = SCRIPT_DIR / "output" / "results"
LOGS = SCRIPT_DIR / "output" / "logs"
for p in (PROCESSED, RESULTS, LOGS):
    p.mkdir(parents=True, exist_ok=True)

START_YEAR = 2001
END_YEAR = 2025
BOOTSTRAP_REPS = 2000
RANDOM_SEED = 20260620

BASE_TABLE_FILE = INPUT / "Base_Table_2001_2025.csv.bz2"
FIRE_EVENT_FILE = INPUT / "Fire_Event_Table_2001_2025.csv"
BA_EVENT_FILE = INPUT / "Burned_Pixel_Event_Table_2001_2025.csv"
AO_FILE = INPUT / "ao_monthly_noaa.csv"
PDO_FILE = INPUT / "pdo_monthly_noaa.csv"
NINO_FILE = INPUT / "nino34_monthly_noaa.csv"
AO_FILL_FILE = INPUT / "ao_official_fill_2025.csv"

# Fire seasons are defined by day of year (DOY), matching the fire-data
# processing exactly in both common and leap years.
PERIOD_DOY_WINDOWS = {
    "S1": (44, 160),
    "S2": (161, 218),
    "S3": (269, 331),
}
PERIOD_CODES = ["AnnualFullYear", "S1", "S2", "S3"]
PERIOD_LABELS = {
    "AnnualFullYear": "Annual",
    "FireSeasonTotal": "Three fire seasons",
    "S1": "Spring",
    "S2": "Summer",
    "S3": "Autumn",
}
PRIMARY_PERIODS = ["AnnualFullYear", "S1", "S2", "S3"]
FIRE_METRICS = {"FC": "Fire count", "BA_km2": "Burned area"}
INDEX_LABELS = {"AO": "AO", "PDO": "PDO", "Nino34": "Nino 3.4"}
WINDOW_ORDER = {"Current": 0, "Lag30": 1, "Lag60": 2, "Lag90": 3, "PreWinter": 4}


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def read_noaa_csv(path: Path) -> dict[tuple[int, int], float | None]:
    require(path)
    out = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            try:
                d = datetime.strptime(row[0].strip(), "%Y-%m-%d")
                value = float(row[1])
            except (ValueError, TypeError):
                continue
            out[(d.year, d.month)] = None if abs(value) >= 90 else value
    return out


def apply_ao_fill(ao: dict) -> pd.DataFrame:
    require(AO_FILL_FILE)
    fill = pd.read_csv(AO_FILL_FILE, encoding="utf-8-sig")
    applied = []
    for row in fill.itertuples(index=False):
        key = (int(row.Year), int(row.Month))
        if ao.get(key) is None:
            ao[key] = float(row.AO)
            applied.append({
                "Year": key[0], "Month": key[1], "AO": float(row.AO),
                "Source": row.Source, "SourceURL": row.SourceURL,
            })
    return pd.DataFrame(applied)


def iter_months(start: date, end: date):
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield y, m
        if m == 12:
            y, m = y + 1, 1
        else:
            m += 1


def overlap_days(y: int, m: int, start: date, end: date) -> int:
    month_start = date(y, m, 1)
    month_end = date(y, m, calendar.monthrange(y, m)[1])
    s, e = max(month_start, start), min(month_end, end)
    return 0 if s > e else (e - s).days + 1


def weighted_value(lookup: dict, start: date, end: date):
    numerator, n_days, weight_parts = 0.0, 0, []
    for y, m in iter_months(start, end):
        days = overlap_days(y, m, start, end)
        if days <= 0:
            continue
        value = lookup.get((y, m))
        if value is None or not np.isfinite(value):
            raise RuntimeError(f"Missing monthly index value: {y}-{m:02d}")
        numerator += float(value) * days
        n_days += days
        weight_parts.append(f"{y}-{m:02d}:{days}")
    if n_days == 0:
        raise RuntimeError(f"Empty window: {start} to {end}")
    return numerator / n_days, n_days, "|".join(weight_parts)


def period_dates(year: int, code: str):
    if code == "AnnualFullYear":
        return date(year, 1, 1), date(year, 12, 31)
    start_doy, end_doy = PERIOD_DOY_WINDOWS[code]
    first_day = date(year, 1, 1)
    return (
        first_day + timedelta(days=start_doy - 1),
        first_day + timedelta(days=end_doy - 1),
    )


def lag_windows(year: int, code: str):
    start, end = period_dates(year, code)
    out = [
        ("Current", "Concurrent fire season", start, end),
        ("Lag30", "30 days before season", start - timedelta(days=30), start - timedelta(days=1)),
        ("Lag60", "60 days before season", start - timedelta(days=60), start - timedelta(days=1)),
        ("Lag90", "90 days before season", start - timedelta(days=90), start - timedelta(days=1)),
    ]
    if code == "S1":
        out.append(("PreWinter", "Previous Dec 1 to current Feb 12", date(year - 1, 12, 1), date(year, 2, 12)))
    return out


def bh_adjust(values) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    adjusted = np.full(len(p), np.nan)
    valid = np.isfinite(p)
    x = p[valid]
    if len(x) == 0:
        return adjusted
    order = np.argsort(x)
    ranked = x[order]
    n = len(ranked)
    adj = ranked * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.minimum(adj, 1.0)
    restored = np.empty(n)
    restored[order] = adj
    adjusted[valid] = restored
    return adjusted


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 5 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return len(x), np.nan, np.nan
    result = stats.spearmanr(x, y)
    return len(x), float(result.statistic), float(result.pvalue)


def residualize(target, covariate):
    X = np.column_stack([np.ones(len(target)), covariate])
    beta, _, _, _ = np.linalg.lstsq(X, target, rcond=None)
    return target - X @ beta


def partial_spearman(x, y, years):
    x, y, years = map(lambda z: np.asarray(z, float), (x, y, years))
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(years)
    x, y, years = x[ok], y[ok], years[ok]
    n = len(x)
    if n < 6:
        return n, np.nan, np.nan
    rx = residualize(stats.rankdata(x), stats.rankdata(years))
    ry = residualize(stats.rankdata(y), stats.rankdata(years))
    rho = float(np.corrcoef(rx, ry)[0, 1])
    df = n - 3
    t = rho * math.sqrt(df / max(1e-15, 1 - rho**2))
    p = float(2 * stats.t.sf(abs(t), df))
    return n, rho, p


def bootstrap_spearman_ci(x, y, reps=BOOTSTRAP_REPS, seed=RANDOM_SEED):
    """Vectorized percentile bootstrap CI for Spearman rho."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return np.nan, np.nan, 0
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, len(x), size=(reps, len(x)))
    xb = x[ids]
    yb = y[ids]
    rx = stats.rankdata(xb, axis=1, method="average")
    ry = stats.rankdata(yb, axis=1, method="average")
    rx = rx - rx.mean(axis=1, keepdims=True)
    ry = ry - ry.mean(axis=1, keepdims=True)
    denom = np.sqrt((rx * rx).sum(axis=1) * (ry * ry).sum(axis=1))
    valid = denom > 0
    estimates = np.full(reps, np.nan)
    estimates[valid] = (rx[valid] * ry[valid]).sum(axis=1) / denom[valid]
    estimates = estimates[np.isfinite(estimates)]
    if len(estimates) == 0:
        return np.nan, np.nan, 0
    lo, hi = np.percentile(estimates, [2.5, 97.5])
    return float(lo), float(hi), int(len(estimates))


def add_fdr(df: pd.DataFrame, p_col: str, family_cols: list[str], family_out: str):
    df = df.copy()
    df["p_FDR_global"] = bh_adjust(df[p_col])
    df[family_out] = np.nan
    for _, ids in df.groupby(family_cols).groups.items():
        df.loc[ids, family_out] = bh_adjust(df.loc[ids, p_col])
    return df


def analyze_tests(data: pd.DataFrame, definitions: list[dict], partial=False):
    rows = []
    for i, test in enumerate(definitions):
        mask = np.ones(len(data), dtype=bool)
        for col, val in test["filters"].items():
            mask &= data[col].astype(str).eq(str(val)).to_numpy()
        sub = data.loc[mask].sort_values("Year")
        for metric, metric_label in FIRE_METRICS.items():
            for index, index_label in INDEX_LABELS.items():
                if partial:
                    n, rho, p = partial_spearman(sub[metric], sub[index], sub["Year"])
                    row = {"n": n, "rho_partial": rho, "p_partial": p}
                else:
                    n, rho, p = spearman(sub[metric], sub[index])
                    lo, hi, valid_reps = bootstrap_spearman_ci(
                        sub[metric], sub[index], seed=RANDOM_SEED + i * 100 + len(rows)
                    )
                    row = {"n": n, "rho": rho, "p_raw": p,
                           "bootstrap_CI_low": lo, "bootstrap_CI_high": hi,
                           "bootstrap_valid_reps": valid_reps}
                rows.append({
                    **{k: v for k, v in test.items() if k != "filters"},
                    "FireMetric": metric, "FireMetricLabel": metric_label,
                    "ClimateIndex": index, "ClimateIndexLabel": index_label,
                    **row,
                })
    return pd.DataFrame(rows)


def loyo_summary(data: pd.DataFrame, result_df: pd.DataFrame, value_col="rho"):
    rows = []
    for test in result_df.itertuples(index=False):
        mask = data["PeriodCode"].astype(str).eq(str(test.PeriodCode))
        if hasattr(test, "WindowCode") and pd.notna(test.WindowCode):
            mask &= data["WindowCode"].astype(str).eq(str(test.WindowCode))
        sub = data.loc[mask].sort_values("Year")
        rhos = []
        years = []
        for omitted in sub["Year"].astype(int):
            reduced = sub[sub["Year"] != omitted]
            _, rho, _ = spearman(reduced[test.FireMetric], reduced[test.ClimateIndex])
            rhos.append(rho); years.append(omitted)
        rhos = np.asarray(rhos, float)
        full = float(getattr(test, value_col))
        same = (rhos > 0) if full > 0 else (rhos < 0)
        changes = np.abs(rhos - full)
        rows.append({
            "PeriodCode": test.PeriodCode, "Period": test.Period,
            "FireMetric": test.FireMetric, "ClimateIndex": test.ClimateIndex,
            "WindowCode": getattr(test, "WindowCode", "Current"),
            "FullSampleRho": full, "LOYO_Rho_Min": float(np.nanmin(rhos)),
            "LOYO_Rho_Max": float(np.nanmax(rhos)), "LOYO_Rho_Mean": float(np.nanmean(rhos)),
            "DirectionConsistencyPercent": float(np.mean(same) * 100),
            "MostInfluentialYear": int(years[int(np.nanargmax(changes))]),
            "MaximumAbsoluteRhoChange": float(np.nanmax(changes)),
        })
    return pd.DataFrame(rows)


def load_fire_activity_from_canonical() -> pd.DataFrame:
    """Build annual and seasonal regional FC/BA series directly from canonical data."""
    for path in (BASE_TABLE_FILE, FIRE_EVENT_FILE, BA_EVENT_FILE):
        require(path)

    base = pd.read_csv(
        BASE_TABLE_FILE,
        usecols=["Year", "Season", "Fire_Count", "Burned_Pixel_Count"],
        encoding="utf-8-sig",
        low_memory=False,
    )
    base["Year"] = pd.to_numeric(base["Year"], errors="raise").astype(int)
    base["Season"] = pd.to_numeric(base["Season"], errors="raise").astype(int)
    base["Fire_Count"] = pd.to_numeric(base["Fire_Count"], errors="raise")
    base["Burned_Pixel_Count"] = pd.to_numeric(
        base["Burned_Pixel_Count"], errors="raise"
    )
    seasonal = (
        base.groupby(["Year", "Season"], as_index=False)
        .agg(FC=("Fire_Count", "sum"), BurnedPixelCount=("Burned_Pixel_Count", "sum"))
    )

    fire_events = pd.read_csv(
        FIRE_EVENT_FILE, usecols=["Fire_ID", "Year"],
        encoding="utf-8-sig", low_memory=False,
    )
    ba_events = pd.read_csv(
        BA_EVENT_FILE, usecols=["Burned_Pixel_ID", "Year"],
        encoding="utf-8-sig", low_memory=False,
    )
    fire_events["Year"] = pd.to_numeric(fire_events["Year"], errors="raise").astype(int)
    ba_events["Year"] = pd.to_numeric(ba_events["Year"], errors="raise").astype(int)
    if fire_events["Fire_ID"].duplicated().any():
        raise RuntimeError("Duplicate Fire_ID values in canonical fire event table.")
    if ba_events["Burned_Pixel_ID"].duplicated().any():
        raise RuntimeError("Duplicate Burned_Pixel_ID values in canonical BA event table.")

    annual_fc = fire_events.groupby("Year").size().reindex(range(START_YEAR, END_YEAR + 1), fill_value=0)
    annual_ba = ba_events.groupby("Year").size().reindex(range(START_YEAR, END_YEAR + 1), fill_value=0)
    seasonal_lookup = {
        (int(row.Year), int(row.Season)): row
        for row in seasonal.itertuples(index=False)
    }

    rows = []
    for year in range(START_YEAR, END_YEAR + 1):
        annual_pixels = int(annual_ba.loc[year])
        rows.append({
            "Year": year,
            "PeriodCode": "AnnualFullYear",
            "Period": PERIOD_LABELS["AnnualFullYear"],
            "FC": int(annual_fc.loc[year]),
            "BurnedPixelCount": annual_pixels,
            "BA_km2": annual_pixels * 0.25,
            "BA_ha": annual_pixels * 25.0,
        })

        season_rows = []
        for season, code in [(1, "S1"), (2, "S2"), (3, "S3")]:
            row = seasonal_lookup[(year, season)]
            pixels = int(round(float(row.BurnedPixelCount)))
            item = {
                "Year": year,
                "PeriodCode": code,
                "Period": PERIOD_LABELS[code],
                "FC": int(round(float(row.FC))),
                "BurnedPixelCount": pixels,
                "BA_km2": pixels * 0.25,
                "BA_ha": pixels * 25.0,
            }
            rows.append(item)
            season_rows.append(item)

        total_pixels = sum(item["BurnedPixelCount"] for item in season_rows)
        rows.append({
            "Year": year,
            "PeriodCode": "FireSeasonTotal",
            "Period": PERIOD_LABELS["FireSeasonTotal"],
            "FC": sum(item["FC"] for item in season_rows),
            "BurnedPixelCount": total_pixels,
            "BA_km2": total_pixels * 0.25,
            "BA_ha": total_pixels * 25.0,
        })

    output = pd.DataFrame(rows)
    expected = (END_YEAR - START_YEAR + 1) * 5
    if len(output) != expected:
        raise RuntimeError(f"Expected {expected} fire-activity rows; found {len(output)}.")
    if output.duplicated(["Year", "PeriodCode"]).any():
        raise RuntimeError("Duplicate Year + PeriodCode rows in canonical fire activity.")
    return output


# ---------- Inputs ----------
for path in (
    BASE_TABLE_FILE, FIRE_EVENT_FILE, BA_EVENT_FILE,
    AO_FILE, PDO_FILE, NINO_FILE, AO_FILL_FILE,
):
    require(path)
fire = load_fire_activity_from_canonical()
fire["Year"] = pd.to_numeric(fire["Year"], errors="raise").astype(int)
fire["PeriodCode"] = fire["PeriodCode"].astype(str)
fire.to_csv(
    PROCESSED / "00_fire_activity_from_canonical_data.csv",
    index=False, encoding="utf-8-sig"
)

ao, pdo, nino = read_noaa_csv(AO_FILE), read_noaa_csv(PDO_FILE), read_noaa_csv(NINO_FILE)
ao_fill_applied = apply_ao_fill(ao)
lookups = {"AO": ao, "PDO": pdo, "Nino34": nino}

# ---------- Clean monthly table ----------
monthly_rows = []
missing = []
for year in range(START_YEAR, END_YEAR + 1):
    for month in range(1, 13):
        row = {"Date": f"{year:04d}-{month:02d}-01", "Year": year, "Month": month}
        for name, lookup in lookups.items():
            value = lookup.get((year, month))
            row[name] = value
            if value is None or not np.isfinite(value):
                missing.append((year, month, name))
        monthly_rows.append(row)
if missing:
    raise RuntimeError(f"Missing monthly values remain: {missing[:20]}")
monthly = pd.DataFrame(monthly_rows)
monthly.to_csv(PROCESSED / "01_climate_monthly_clean_2001_2025.csv", index=False, encoding="utf-8-sig")
ao_fill_applied.to_csv(PROCESSED / "00_ao_fill_applied.csv", index=False, encoding="utf-8-sig")

# ---------- Concurrent/seasonal indices ----------
concurrent_rows = []
for year in range(START_YEAR, END_YEAR + 1):
    period_cache = {}
    for code in PERIOD_CODES:
        start, end = period_dates(year, code)
        row = {"Year": year, "PeriodCode": code, "Period": PERIOD_LABELS[code],
               "WindowStart": start.isoformat(), "WindowEnd": end.isoformat()}
        for index, lookup in lookups.items():
            value, days, weights = weighted_value(lookup, start, end)
            row[index] = value
        row["N_days"] = days; row["MonthWeights"] = weights
        concurrent_rows.append(row); period_cache[code] = row
    total_days = sum(period_cache[c]["N_days"] for c in ["S1", "S2", "S3"])
    combined = {"Year": year, "PeriodCode": "FireSeasonTotal",
                "Period": PERIOD_LABELS["FireSeasonTotal"], "WindowStart": "Discontinuous",
                "WindowEnd": "Discontinuous", "N_days": total_days,
                "MonthWeights": "Union of S1, S2 and S3"}
    for index in INDEX_LABELS:
        combined[index] = sum(period_cache[c][index] * period_cache[c]["N_days"] for c in ["S1","S2","S3"]) / total_days
    concurrent_rows.append(combined)
concurrent = pd.DataFrame(concurrent_rows)
order = {"AnnualFullYear":0,"FireSeasonTotal":1,"S1":2,"S2":3,"S3":4}
concurrent["_order"] = concurrent["PeriodCode"].map(order)
concurrent = concurrent.sort_values(["Year","_order"]).drop(columns="_order")
concurrent.to_csv(PROCESSED / "02_climate_concurrent_periods_2001_2025.csv", index=False, encoding="utf-8-sig")

# Remove any old climate columns from fire input before merging.
fire_base = fire.drop(columns=[c for c in ["AO","PDO","Nino34","N_days","MonthWeights","Index_N_days","Index_MonthWeights"] if c in fire.columns], errors="ignore")
fire_concurrent = fire_base.merge(concurrent, on=["Year","PeriodCode"], how="left", validate="one_to_one", suffixes=("_fire","_climate"))
if "Period_fire" in fire_concurrent.columns:
    fire_concurrent["Period"] = fire_concurrent["Period_fire"]
    fire_concurrent = fire_concurrent.drop(columns=[c for c in ["Period_fire","Period_climate"] if c in fire_concurrent.columns])
if fire_concurrent[["FC","BA_km2","AO","PDO","Nino34"]].isna().any().any():
    raise RuntimeError("Missing values after concurrent merge.")
fire_concurrent.to_csv(PROCESSED / "03_fire_climate_concurrent_2001_2025.csv", index=False, encoding="utf-8-sig")

# ---------- Lagged indices ----------
lag_rows = []
for year in range(START_YEAR, END_YEAR + 1):
    for period in ["S1","S2","S3"]:
        for window_code, window_label, start, end in lag_windows(year, period):
            row = {"Year":year,"PeriodCode":period,"Period":PERIOD_LABELS[period],
                   "WindowCode":window_code,"WindowLabel":window_label,
                   "WindowStart":start.isoformat(),"WindowEnd":end.isoformat()}
            for index, lookup in lookups.items():
                value, days, weights = weighted_value(lookup,start,end)
                row[index]=value
            row["N_days"]=days; row["MonthWeights"]=weights
            lag_rows.append(row)
lagged = pd.DataFrame(lag_rows)
lagged["_order"] = lagged["WindowCode"].map(WINDOW_ORDER)
lagged = lagged.sort_values(["Year","PeriodCode","_order"]).drop(columns="_order")
lagged.to_csv(PROCESSED / "04_climate_lagged_windows_2001_2025.csv", index=False, encoding="utf-8-sig")
fire_seasonal = fire_base[fire_base["PeriodCode"].isin(["S1","S2","S3"])].copy()
fire_lagged = fire_seasonal.merge(lagged, on=["Year","PeriodCode"], how="left", validate="one_to_many", suffixes=("_fire","_climate"))
if "Period_fire" in fire_lagged.columns:
    fire_lagged["Period"] = fire_lagged["Period_fire"]
    fire_lagged = fire_lagged.drop(columns=[c for c in ["Period_fire","Period_climate"] if c in fire_lagged.columns])
if fire_lagged[["FC","BA_km2","AO","PDO","Nino34"]].isna().any().any():
    raise RuntimeError("Missing values after lagged merge.")
fire_lagged.to_csv(PROCESSED / "05_fire_climate_lagged_2001_2025.csv", index=False, encoding="utf-8-sig")

# ---------- Concurrent statistics ----------
concurrent_defs = [{"PeriodCode": p, "Period": PERIOD_LABELS[p], "filters":{"PeriodCode":p}} for p in PRIMARY_PERIODS]
raw_concurrent = analyze_tests(fire_concurrent, concurrent_defs, partial=False)
raw_concurrent = add_fdr(raw_concurrent, "p_raw", ["PeriodCode"], "p_FDR_within_period")
partial_concurrent = analyze_tests(fire_concurrent, concurrent_defs, partial=True)
partial_concurrent = add_fdr(partial_concurrent, "p_partial", ["PeriodCode"], "p_FDR_within_period")
raw_concurrent.to_csv(RESULTS / "01_concurrent_spearman.csv", index=False, encoding="utf-8-sig")
partial_concurrent.to_csv(RESULTS / "02_concurrent_partial_spearman_year.csv", index=False, encoding="utf-8-sig")
loyo_summary(fire_concurrent, raw_concurrent).to_csv(RESULTS / "03_concurrent_leave_one_year_out.csv", index=False, encoding="utf-8-sig")

# Supplementary three-season aggregate
supp_defs = [{"PeriodCode":"FireSeasonTotal","Period":PERIOD_LABELS["FireSeasonTotal"],"filters":{"PeriodCode":"FireSeasonTotal"}}]
supp = analyze_tests(fire_concurrent, supp_defs, partial=False)
supp = add_fdr(supp, "p_raw", ["PeriodCode"], "p_FDR_within_period")
supp.to_csv(RESULTS / "04_fire_season_total_spearman_supplementary.csv", index=False, encoding="utf-8-sig")

# ---------- Lagged statistics ----------
lag_defs = []
for period in ["S1","S2","S3"]:
    windows = ["Current","Lag30","Lag60","Lag90"] + (["PreWinter"] if period=="S1" else [])
    for window in windows:
        lag_defs.append({"PeriodCode":period,"Period":PERIOD_LABELS[period],
                         "WindowCode":window,
                         "WindowLabel":fire_lagged.loc[(fire_lagged.PeriodCode==period)&(fire_lagged.WindowCode==window),"WindowLabel"].iloc[0],
                         "filters":{"PeriodCode":period,"WindowCode":window}})
raw_lag = analyze_tests(fire_lagged, lag_defs, partial=False)
raw_lag = add_fdr(raw_lag, "p_raw", ["PeriodCode","FireMetric","ClimateIndex"], "p_FDR_lag_family")
partial_lag = analyze_tests(fire_lagged, lag_defs, partial=True)
partial_lag = add_fdr(partial_lag, "p_partial", ["PeriodCode","FireMetric","ClimateIndex"], "p_FDR_lag_family")
raw_lag.to_csv(RESULTS / "05_lagged_spearman.csv", index=False, encoding="utf-8-sig")
partial_lag.to_csv(RESULTS / "06_lagged_partial_spearman_year.csv", index=False, encoding="utf-8-sig")
lag_loyo = loyo_summary(fire_lagged, raw_lag)
lag_loyo.to_csv(RESULTS / "07_lagged_leave_one_year_out.csv", index=False, encoding="utf-8-sig")

# ---------- Window collinearity ----------
corr_rows = []
for period in ["S1","S2","S3"]:
    for index in INDEX_LABELS:
        wide = lagged[lagged.PeriodCode==period].pivot(index="Year",columns="WindowCode",values=index)
        corr = wide.corr(method="spearman")
        for w1 in corr.index:
            for w2 in corr.columns:
                corr_rows.append({"PeriodCode":period,"Period":PERIOD_LABELS[period],
                                  "ClimateIndex":index,"Window1":w1,"Window2":w2,
                                  "SpearmanRho":corr.loc[w1,w2]})
pd.DataFrame(corr_rows).to_csv(RESULTS / "08_lag_window_intercorrelations.csv", index=False, encoding="utf-8-sig")

# ---------- Key relationships for main scatter panels ----------
key_specs = [
    ("S1","BA_km2","Nino34","PreWinter","Spring BA vs preceding-winter Nino 3.4"),
    ("S2","BA_km2","Nino34","Lag60","Summer BA vs 60-day antecedent Nino 3.4"),
]
key_rows = []
for period, metric, index, window, label in key_specs:
    sub = fire_lagged[(fire_lagged.PeriodCode==period)&(fire_lagged.WindowCode==window)]
    result = raw_lag[(raw_lag.PeriodCode==period)&(raw_lag.FireMetric==metric)&(raw_lag.ClimateIndex==index)&(raw_lag.WindowCode==window)].iloc[0]
    for row in sub.itertuples(index=False):
        key_rows.append({"Relationship":label,"Year":int(row.Year),"PeriodCode":period,
                         "FireMetric":metric,"ClimateIndex":index,"WindowCode":window,
                         "FireValue":float(getattr(row,metric)),"IndexValue":float(getattr(row,index)),
                         "rho":result.rho,"p_raw":result.p_raw,
                         "p_FDR_lag_family":result.p_FDR_lag_family})
pd.DataFrame(key_rows).to_csv(RESULTS / "09_key_relationship_scatter_data.csv", index=False, encoding="utf-8-sig")

# ---------- Validation and QA ----------
qa = []
qa.append({"Check":"Fire input rows","Observed":len(fire),"Expected":125,"Pass":len(fire)==125})
qa.append({"Check":"Monthly climate rows","Observed":len(monthly),"Expected":300,"Pass":len(monthly)==300})
qa.append({"Check":"Concurrent merged rows","Observed":len(fire_concurrent),"Expected":125,"Pass":len(fire_concurrent)==125})
qa.append({"Check":"Lagged merged rows","Observed":len(fire_lagged),"Expected":325,"Pass":len(fire_lagged)==325})
qa.append({"Check":"Concurrent tests","Observed":len(raw_concurrent),"Expected":24,"Pass":len(raw_concurrent)==24})
qa.append({"Check":"Lagged tests","Observed":len(raw_lag),"Expected":78,"Pass":len(raw_lag)==78})
qa_df = pd.DataFrame(qa)
qa_df.to_csv(RESULTS / "00_QA_summary.csv", index=False, encoding="utf-8-sig")
if not qa_df["Pass"].all():
    raise RuntimeError("At least one QA check failed; inspect output/results/00_QA_summary.csv")

# ---------- Machine-readable run metadata ----------
metadata = {
    "study_period":"2001-2025",
    "bootstrap_reps":BOOTSTRAP_REPS,
    "random_seed":RANDOM_SEED,
    "python":sys.version,
    "platform":platform.platform(),
    "numpy":np.__version__,
    "pandas":pd.__version__,
    "scipy":scipy.__version__,
    "primary_periods":PRIMARY_PERIODS,
    "lag_windows":["Current","Lag30","Lag60","Lag90","PreWinter (spring only)"],
}
(LOGS / "analysis_run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=True),encoding="utf-8")

# ---------- Human-readable summary ----------
strongest = raw_lag.loc[raw_lag["rho"].abs().sort_values(ascending=False).index].head(15)
with (LOGS / "analysis_summary.txt").open("w",encoding="utf-8") as f:
    f.write("FIRE-ATMOSPHERIC-CIRCULATION REPRODUCIBLE ANALYSIS\n")
    f.write("="*78+"\n")
    f.write(f"Study period: {START_YEAR}-{END_YEAR}\n")
    f.write(f"Bootstrap repetitions: {BOOTSTRAP_REPS}\n")
    f.write("All QA checks passed.\n\n")
    f.write("Strongest lagged relationships by absolute Spearman rho:\n")
    f.write(strongest[["Period","FireMetricLabel","ClimateIndexLabel","WindowCode","rho","p_raw","p_FDR_lag_family","p_FDR_global"]].to_string(index=False))

print("Analysis completed successfully.")
print("Code root:", CODE_ROOT)
print("Processed data:", PROCESSED)
print("Results:", RESULTS)
print("QA: all checks passed")


# ============================================================================
# FIGURE GENERATION (combined with the analysis pipeline)
# ============================================================================

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

available_fonts = {font.name for font in font_manager.fontManager.ttflist}
figure_font = "Times New Roman" if "Times New Roman" in available_fonts else "serif"

plt.rcParams.update({
    "font.family": figure_font,
    "font.size": 14.0,
    "axes.titlesize": 16.0,
    "axes.labelsize": 15.0,
    "xtick.labelsize": 14.0,
    "ytick.labelsize": 14.0,
    "legend.fontsize": 13.0,
    "figure.titlesize": 17.0,
    "axes.unicode_minus": True,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

MAIN_DIR = SCRIPT_DIR / "output" / "figures" / "main_text"
SUPP_DIR = SCRIPT_DIR / "output" / "figures" / "supplementary"
for path in (MAIN_DIR, SUPP_DIR, LOGS):
    path.mkdir(parents=True, exist_ok=True)

DPI = 600
PERIOD_ORDER = ["Annual", "Spring", "Summer", "Autumn"]
METRIC_ORDER = ["Fire count", "Burned area"]
INDEX_ORDER = ["AO", "PDO", "Nino 3.4"]
WINDOW_ORDER_SPRING = ["Current", "Lag30", "Lag60", "Lag90", "PreWinter"]
WINDOW_ORDER_OTHER = ["Current", "Lag30", "Lag60", "Lag90"]

# Fixed display range used for panel (d) in the manuscript figure.
PANEL_D_YLIM = (-10, 120)


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required result file not found: {path}\n"
            "The analysis stage in this combined script did not produce the required file."
        )


def save_figure(fig: plt.Figure, folder: Path, stem: str) -> None:
    fig.savefig(
        folder / f"{stem}.tif",
        dpi=DPI,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.savefig(folder / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def significance_mark(p_raw: float, p_family: float, p_global: float) -> str:
    """Return a compact hierarchy of inferential symbols."""
    if np.isfinite(p_global) and p_global < 0.05:
        return "**"
    if np.isfinite(p_family) and p_family < 0.05:
        return "*"
    if np.isfinite(p_raw) and p_raw < 0.05:
        return "+"
    return ""


UNICODE_MINUS = "\N{MINUS SIGN}"


def format_number(value: float, digits: int = 2) -> str:
    """Format a number using the mathematical minus sign (U+2212)."""
    if not np.isfinite(value):
        return "NA"
    return f"{float(value):.{digits}f}".replace("-", UNICODE_MINUS)


def add_two_tier_period_metric_yaxis(
    ax: plt.Axes,
    periods: list[str],
    inner_fontsize: float = 14.5,
    outer_fontsize: float = 14.5,
    outer_x: float = -0.17,
) -> None:
    """Show FC/BA beside the axis and one period label for each row pair."""
    ax.set_yticks(np.arange(len(periods) * 2))
    ax.set_yticklabels(["FC", "BA"] * len(periods), fontsize=inner_fontsize)
    ax.tick_params(axis="y", length=0, pad=5)

    for index, period in enumerate(periods):
        center = index * 2 + 0.5
        ax.text(
            outer_x,
            center,
            period,
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=outer_fontsize,
            clip_on=False,
        )


def add_two_tier_period_metric_xaxis(
    ax: plt.Axes,
    periods: list[str],
    inner_fontsize: float = 14.5,
    outer_fontsize: float = 14.5,
    outer_y: float = -0.070,
) -> None:
    """Show FC/BA below cells and one centered period label for each pair."""
    ax.set_xticks(np.arange(len(periods) * 2))
    ax.set_xticklabels(["FC", "BA"] * len(periods), fontsize=inner_fontsize)
    ax.tick_params(axis="x", length=0, pad=5)

    for index, period in enumerate(periods):
        center = index * 2 + 0.5
        ax.text(
            center,
            outer_y,
            period,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=outer_fontsize,
            clip_on=False,
        )


def robust_line(ax, x, y) -> None:
    """Add a Theil-Sen trend line as a descriptive visual guide."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 3 or np.unique(x).size < 2:
        return
    slope, intercept, _, _ = stats.theilslopes(y, x, alpha=0.95)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, intercept + slope * x_line, linewidth=1.2)


# ---------------------------------------------------------------------------
# Read analysis outputs
# ---------------------------------------------------------------------------
concurrent_path = RESULTS / "01_concurrent_spearman.csv"
lagged_path = RESULTS / "05_lagged_spearman.csv"
partial_lagged_path = RESULTS / "06_lagged_partial_spearman_year.csv"
lagged_data_path = PROCESSED / "05_fire_climate_lagged_2001_2025.csv"

for path in (concurrent_path, lagged_path, partial_lagged_path, lagged_data_path):
    require(path)

concurrent = pd.read_csv(concurrent_path, encoding="utf-8-sig")
lagged = pd.read_csv(lagged_path, encoding="utf-8-sig")
partial_lagged = pd.read_csv(partial_lagged_path, encoding="utf-8-sig")
lagged_data = pd.read_csv(lagged_data_path, encoding="utf-8-sig")

# ---------------------------------------------------------------------------
# Main Figure 1: compact concurrent correlation matrix
# ---------------------------------------------------------------------------
row_definitions = []
for period in PERIOD_ORDER:
    for metric in METRIC_ORDER:
        row_definitions.append((period, metric, f"{period} - {metric}"))

matrix = np.full((len(row_definitions), len(INDEX_ORDER)), np.nan)
labels = np.full(matrix.shape, "", dtype=object)

for i, (period, metric, _) in enumerate(row_definitions):
    for j, index_label in enumerate(INDEX_ORDER):
        row = concurrent[
            (concurrent["Period"] == period)
            & (concurrent["FireMetricLabel"] == metric)
            & (concurrent["ClimateIndexLabel"] == index_label)
        ]
        if row.empty:
            continue
        r = row.iloc[0]
        matrix[i, j] = float(r["rho"])
        mark = significance_mark(
            float(r["p_raw"]),
            float(r["p_FDR_within_period"]),
            float(r["p_FDR_global"]),
        )
        labels[i, j] = f"{format_number(float(r['rho']), 2)}{mark}"

fig, ax = plt.subplots(figsize=(9.4, 7.2))
image = ax.imshow(matrix, vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(INDEX_ORDER)), INDEX_ORDER)
add_two_tier_period_metric_yaxis(
    ax,
    periods=PERIOD_ORDER,
    inner_fontsize=14.5,
    outer_fontsize=14.5,
    outer_x=-0.17,
)
ax.set_xlabel("Atmospheric-circulation index")
ax.set_ylabel(
    "Fire metric and analysis period",
    fontsize=15.0,
    labelpad=105,
)
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        ax.text(j, i, labels[i, j], ha="center", va="center", fontsize=15.0)
colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.04)
colorbar.set_label("Spearman rho")
colorbar.ax.tick_params(labelsize=14.0)
fig.subplots_adjust(
    left=0.30,
    right=0.90,
    bottom=0.11,
    top=0.98,
)
save_figure(fig, MAIN_DIR, "Main_Figure_1_Concurrent_correlations")

# ---------------------------------------------------------------------------
# Main Figure 2: Nino 3.4 lag profiles and two key scatterplots
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.5))

for ax, period_code, period_label, windows in [
    (axes[0, 0], "S1", "Spring", WINDOW_ORDER_SPRING),
    (axes[0, 1], "S2", "Summer", WINDOW_ORDER_OTHER),
]:
    for metric_label, marker in [("Fire count", "o"), ("Burned area", "s")]:
        raw = lagged[
            (lagged["PeriodCode"] == period_code)
            & (lagged["ClimateIndex"] == "Nino34")
            & (lagged["FireMetricLabel"] == metric_label)
        ].copy()
        raw["WindowCode"] = pd.Categorical(raw["WindowCode"], windows, ordered=True)
        raw = raw.sort_values("WindowCode")

        partial = partial_lagged[
            (partial_lagged["PeriodCode"] == period_code)
            & (partial_lagged["ClimateIndex"] == "Nino34")
            & (partial_lagged["FireMetricLabel"] == metric_label)
        ].copy()
        partial["WindowCode"] = pd.Categorical(partial["WindowCode"], windows, ordered=True)
        partial = partial.sort_values("WindowCode")

        x = np.arange(len(raw))
        line = ax.plot(
            x,
            raw["rho"],
            marker=marker,
            linewidth=1.4,
            label=f"{metric_label}, raw",
        )[0]
        ax.fill_between(
            x,
            raw["bootstrap_CI_low"].to_numpy(dtype=float),
            raw["bootstrap_CI_high"].to_numpy(dtype=float),
            alpha=0.12,
        )
        ax.plot(
            x,
            partial["rho_partial"],
            marker=marker,
            markerfacecolor="none",
            linestyle="--",
            linewidth=1.1,
            color=line.get_color(),
            label=f"{metric_label}, year-controlled",
        )

    ax.axhline(0, linewidth=0.8, linestyle=":")
    ax.set_xticks(np.arange(len(windows)), windows, rotation=25, ha="right")
    ax.set_ylim(-0.85, 0.25)
    ax.set_ylabel("Spearman rho")
    ax.set_title(f"{period_label}: concurrent and antecedent Nino 3.4")
    ax.legend(fontsize=12.5, frameon=False)

scatter_specs = [
    (axes[1, 0], "S1", "PreWinter", "Spring BA vs preceding-winter Nino 3.4"),
    (axes[1, 1], "S2", "Lag60", "Summer BA vs 60-day antecedent Nino 3.4"),
]

for ax, period_code, window_code, title in scatter_specs:
    data = lagged_data[
        (lagged_data["PeriodCode"] == period_code)
        & (lagged_data["WindowCode"] == window_code)
    ].sort_values("Year")
    result = lagged[
        (lagged["PeriodCode"] == period_code)
        & (lagged["WindowCode"] == window_code)
        & (lagged["FireMetric"] == "BA_km2")
        & (lagged["ClimateIndex"] == "Nino34")
    ].iloc[0]

    ax.scatter(data["Nino34"], data["BA_km2"], s=28)
    robust_line(ax, data["Nino34"], data["BA_km2"])
    ax.set_xlabel("Nino 3.4 index")
    ax.set_ylabel(r"Burned area (km$^2$)")
    ax.set_title(title)

    # Set the y-axis range only for panel (d).
    if period_code == "S2" and window_code == "Lag60":
        ax.set_ylim(*PANEL_D_YLIM)
    ax.text(
        0.97,
        0.97,
        (
            f"rho = {format_number(float(result['rho']), 2)}\n"
            f"p = {format_number(float(result['p_raw']), 3)}\n"
            "lag-family FDR = "
            f"{format_number(float(result['p_FDR_lag_family']), 3)}"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=12.5,
        bbox={"boxstyle": "round", "alpha": 0.75},
    )

for label, ax in zip(["(a)", "(b)", "(c)", "(d)"], axes.flat):
    ax.text(-0.12, 1.05, label, transform=ax.transAxes, fontsize=17)

fig.tight_layout()
save_figure(fig, MAIN_DIR, "Main_Figure_2_Nino34_lagged_response")

# ---------------------------------------------------------------------------
# Supplementary Figure S2: compact six-column correlation matrix
# ---------------------------------------------------------------------------
# For spring, the displayed Lag90 row uses the predefined PreWinter window
# (1 December of the previous year to 12 February of the current year),
# allowing a common four-row layout across seasons.
display_windows = ["Current", "Lag30", "Lag60", "Lag90"]
period_definitions = [
    ("S1", "Spring"),
    ("S2", "Summer"),
    ("S3", "Autumn"),
]
column_definitions = [
    (period_code, period_label, metric_label)
    for period_code, period_label in period_definitions
    for metric_label in METRIC_ORDER
]

row_labels = [
    f"{index_label} - {window_label}"
    for index_label in INDEX_ORDER
    for window_label in display_windows
]

matrix = np.full((len(row_labels), len(column_definitions)), np.nan)
text_labels = np.full(matrix.shape, "", dtype=object)

row_number = 0
for index_label in INDEX_ORDER:
    for displayed_window in display_windows:
        for col_number, (period_code, period_label, metric_label) in enumerate(column_definitions):
            source_window = (
                "PreWinter"
                if period_code == "S1" and displayed_window == "Lag90"
                else displayed_window
            )

            row = lagged[
                (lagged["PeriodCode"] == period_code)
                & (lagged["ClimateIndexLabel"] == index_label)
                & (lagged["WindowCode"] == source_window)
                & (lagged["FireMetricLabel"] == metric_label)
            ]

            if not row.empty:
                result = row.iloc[0]
                matrix[row_number, col_number] = float(result["rho"])
                mark = significance_mark(
                    float(result["p_raw"]),
                    float(result["p_FDR_lag_family"]),
                    float(result["p_FDR_global"]),
                )
                text_labels[row_number, col_number] = (
                    f"{format_number(float(result['rho']), 2)}{mark}"
                )
        row_number += 1

fig, ax = plt.subplots(
    figsize=(13.5, 9.5),
    constrained_layout=False,
)

image = ax.imshow(
    matrix,
    vmin=-1,
    vmax=1,
    aspect="auto",
)

add_two_tier_period_metric_xaxis(
    ax,
    periods=[period_label for _, period_label in period_definitions],
    inner_fontsize=14.5,
    outer_fontsize=14.5,
    outer_y=-0.070,
)

ax.set_yticks(
    range(len(row_labels)),
    row_labels,
    fontsize=14.0,
)

for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        ax.text(
            j,
            i,
            text_labels[i, j],
            ha="center",
            va="center",
            fontsize=15.0,
        )

# Separate atmospheric indices and seasonal column groups.
for y_position in [3.5, 7.5]:
    ax.axhline(y_position, linewidth=0.8, color="black")
for x_position in [1.5, 3.5]:
    ax.axvline(x_position, linewidth=0.8, color="black")

colorbar = fig.colorbar(
    image,
    ax=ax,
    fraction=0.035,
    pad=0.025,
    shrink=0.86,
    aspect=25,
)
colorbar.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])
colorbar.set_label(
    "Spearman rho",
    fontsize=15.0,
    labelpad=9,
)
colorbar.ax.tick_params(
    labelsize=14.0,
)

fig.subplots_adjust(
    left=0.18,
    right=0.91,
    top=0.97,
    bottom=0.15,
)

save_figure(
    fig,
    SUPP_DIR,
    "Supplementary_Figure_S2_Concurrent_and_antecedent_correlations",
)

(LOGS / "figure_generation_summary.txt").write_text(
    "Figure generation completed successfully.\n"
    "Main figures: 2\n"
    "Supplementary figures: 1\n"
    "Formats: 600 dpi TIFF and vector PDF\n",
    encoding="utf-8",
)

print("Figure generation completed successfully.")
print("Main-text figures:", MAIN_DIR)
print("Supplementary figures:", SUPP_DIR)
