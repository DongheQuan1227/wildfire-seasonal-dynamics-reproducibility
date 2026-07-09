#!/usr/bin/env python3
"""Validate key generated results against the manuscript values."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = json.loads((ROOT / "validation" / "expected_results.json").read_text(encoding="utf-8"))


class Checks:
    def __init__(self, allow_partial: bool) -> None:
        self.allow_partial = allow_partial
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.messages: list[str] = []

    def ok(self, name: str, condition: bool, detail: str = "") -> None:
        if condition:
            self.passed += 1
            self.messages.append(f"PASS  {name}" + (f": {detail}" if detail else ""))
        else:
            self.failed += 1
            self.messages.append(f"FAIL  {name}" + (f": {detail}" if detail else ""))

    def missing(self, name: str, path: Path) -> None:
        if self.allow_partial:
            self.skipped += 1
            self.messages.append(f"SKIP  {name}: missing {path.relative_to(ROOT)}")
        else:
            self.failed += 1
            self.messages.append(f"FAIL  {name}: missing {path.relative_to(ROOT)}")


def close(a: float, b: float, tol: float = 1e-5) -> bool:
    return bool(np.isfinite(a) and abs(float(a) - float(b)) <= tol)


def read_or_missing(checks: Checks, name: str, rel: str) -> pd.DataFrame | None:
    path = ROOT / rel
    if not path.is_file():
        checks.missing(name, path)
        return None
    return pd.read_csv(path)


def check_input_manifest(checks: Checks) -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "verify_input_manifest.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    checks.ok("input data checksums", result.returncode == 0, result.stdout.strip() or result.stderr.strip())


def check_fai(checks: Checks) -> None:
    df = read_or_missing(checks, "FAI peak output", "02_FAI_season_delineation/output/Supplementary_FAI_peak_results.csv")
    if df is None:
        return
    peaks = [int(x) for x in df["Day_number"].tolist()]
    checks.ok("FAI peaks", peaks == EXPECTED["fai_peaks_doy"], f"observed={peaks}")


def check_frp(checks: Checks) -> None:
    df = read_or_missing(checks, "FRP trend output", "03_FRP_trends/output/frp_trend_results_2001_2025.csv")
    if df is None:
        return
    def row(season: str, metric: str) -> pd.Series:
        return df[(df["Season"] == season) & (df["Metric"] == metric)].iloc[0]
    exp = EXPECTED["frp"]
    checks.ok("annual Fire Count tau", close(row("Annual", "Fire_Count")["Tau"], exp["annual_fire_count_tau"], 1e-6))
    checks.ok("annual median FRP tau", close(row("Annual", "Median_FRP_MW")["Tau"], exp["annual_median_frp_tau"], 1e-6))
    checks.ok("annual median FRP FDR", close(row("Annual", "Median_FRP_MW")["P_FDR"], exp["annual_median_frp_fdr"], 1e-6))
    checks.ok("spring median FRP tau", close(row("Spring", "Median_FRP_MW")["Tau"], exp["spring_median_frp_tau"], 1e-6))
    checks.ok("spring P95 FRP tau", close(row("Spring", "P95_FRP_MW")["Tau"], exp["spring_p95_frp_tau"], 1e-6))


def check_country_shares(checks: Checks) -> None:
    df = read_or_missing(checks, "country-season trend output", "04_Country_season_trends/output/country_season_fc_ba_trend_results_2001_2025.csv")
    if df is None:
        return
    exp = EXPECTED["country_shares"]
    for period in ["Spring", "Summer", "Autumn"]:
        row = df[(df["Period"] == period) & (df["Country"] == "Russia")].iloc[0]
        for metric, column in [("FC", "FC_share_pct"), ("BA", "BA_share_pct")]:
            key = f"Russia_{period}_{metric}"
            checks.ok(f"{key} share", close(row[column], exp[key], 1e-6), f"observed={row[column]:.6f}")


def check_circulation(checks: Checks) -> None:
    concurrent = read_or_missing(checks, "concurrent circulation output", "05_Atmospheric_circulation/output/results/01_concurrent_spearman.csv")
    lagged = read_or_missing(checks, "lagged circulation output", "05_Atmospheric_circulation/output/results/05_lagged_spearman.csv")
    exp = EXPECTED["circulation_rho"]
    if concurrent is not None:
        value = concurrent[(concurrent.PeriodCode == "S1") & (concurrent.FireMetric == "BA_km2") & (concurrent.ClimateIndex == "Nino34")].iloc[0]["rho"]
        checks.ok("spring BA–Niño3.4 concurrent rho", close(value, exp["Spring_BA_Nino34_concurrent"], 1e-6))
    if lagged is not None:
        definitions = [
            ("Spring BA–Niño3.4 prewinter rho", "S1", "BA_km2", "PreWinter", "Spring_BA_Nino34_prewinter"),
            ("summer FC–Niño3.4 lag30 rho", "S2", "FC", "Lag30", "Summer_FC_Nino34_lag30"),
            ("summer BA–Niño3.4 lag60 rho", "S2", "BA_km2", "Lag60", "Summer_BA_Nino34_lag60"),
        ]
        for name, period, metric, window, key in definitions:
            value = lagged[(lagged.PeriodCode == period) & (lagged.FireMetric == metric) & (lagged.ClimateIndex == "Nino34") & (lagged.WindowCode == window)].iloc[0]["rho"]
            checks.ok(name, close(value, exp[key], 1e-6))


def check_admin1(checks: Checks) -> None:
    df = read_or_missing(checks, "administrative hotspot output", "07_Admin1_hotspot_recurrence/output/admin1_hotspot_recurrence_summary_2001_2025.csv")
    if df is None:
        return
    exp = EXPECTED["admin1"]
    def row(metric: str, region: str) -> pd.Series:
        return df[(df["Metric"] == metric) & (df["Administrative region"] == region)].iloc[0]
    checks.ok("Primorskiy FC hotspot years", int(row("FC", "Primorskiy")["Hotspot years n"]) == exp["Primorskiy_FC_hotspot_years"])
    checks.ok("Primorskiy BA hotspot years", int(row("BA", "Primorskiy")["Hotspot years n"]) == exp["Primorskiy_BA_hotspot_years"])
    checks.ok("Khabarovskiy FC Sen slope", close(row("FC", "Khabarovskiy")["Sen slope (percentage points/year)"], exp["Khabarovskiy_FC_sen_slope"], 1e-4))
    checks.ok("Khabarovskiy BA Sen slope", close(row("BA", "Khabarovskiy")["Sen slope (percentage points/year)"], exp["Khabarovskiy_BA_sen_slope"], 1e-4))


def check_model_comparison(checks: Checks) -> None:
    df = read_or_missing(checks, "model-comparison main table", "10_Model_comparison/06_Model_Comparison_Tables_Figures/03_Main_Table_RF_vs_Regression_Paired_Comparison.csv")
    if df is None:
        return
    exp = EXPECTED["model_comparison"]
    temporal = df[df["Validation_Scheme"] == "Temporal"].sort_values(["Season", "Count_Response"])
    overall_col = "Rate_RMSE_Percent_Improvement_Positive_Is_Better"
    positive_col = "Positive_Conditional_Rate_RMSE_Percent_Improvement_Positive_Is_Better"
    observed_overall = sorted(np.round(temporal[overall_col].astype(float).to_numpy(), 1).tolist())
    expected_overall = sorted(exp["temporal_overall_improvement_pct"])
    observed_positive = sorted(np.round(temporal[positive_col].astype(float).to_numpy(), 1).tolist())
    expected_positive = sorted(exp["temporal_positive_improvement_pct"])
    checks.ok("temporal overall RF improvements", observed_overall == expected_overall, f"observed={observed_overall}")
    checks.ok("temporal positive RF improvements", observed_positive == expected_positive, f"observed={observed_positive}")
    spatial = df[df["Validation_Scheme"] == "Spatial"]
    checks.ok("spatial maximum RF improvement", close(spatial[overall_col].max(), exp["spatial_overall_max_improvement_pct"], 0.15))
    checks.ok("spatial minimum RF improvement", close(spatial[overall_col].min(), exp["spatial_overall_min_improvement_pct"], 0.15))


def check_sem(checks: Checks) -> None:
    df = read_or_missing(checks, "SEM total-effect output", "11_SEM_mechanism_analysis/05_SEM_Effect_Decomposition_and_Path_Diagrams/05_Primary_Total_Effects_to_FAI.csv")
    if df is None:
        return
    exp = EXPECTED["sem_total_effects"]
    expected_pairs = {tuple(key.split("_", 1)) for key in exp}
    total = df.copy()
    if "Effect_Type" in total.columns:
        total = total[total["Effect_Type"].astype(str).eq("Total")].copy()
    total = total[
        total.apply(
            lambda row: (str(row["Season_Label"]), str(row["Predictor"])) in expected_pairs,
            axis=1,
        )
    ].copy()
    checks.ok("SEM total-effect row count", len(total) == 12, f"observed={len(total)}")
    for key, expected_value in exp.items():
        season, predictor = key.split("_", 1)
        row = total[(total["Season_Label"] == season) & (total["Predictor"] == predictor)]
        if row.empty:
            checks.ok(f"SEM {key}", False, "row not found")
            continue
        checks.ok(f"SEM {key}", close(row.iloc[0]["Estimate"], expected_value, 0.002), f"observed={row.iloc[0]['Estimate']}")


def check_primary_artifacts(checks: Checks) -> None:
    paths = [
        "08_Regression_modeling/10_Manuscript_Figure_08/Figure_08_Final_Regression_Effects.tif",
        "09_RF_modeling/13_Manuscript_Figure_09/Figure_09_Variable_Importance_and_Key_SHAP.tif",
        "09_RF_modeling/14_Figure_S6_Top5_SHAP_Profiles/Figure_S6_Top5_SHAP_Profiles.tif",
        "10_Model_comparison/07_Manuscript_Figure_10/Figure_10_Unified_OOF_Model_Comparison.tif",
        "10_Model_comparison/08_Supplementary_Figure_S7_Annual_OOF_Predictions/Figure_S7_Annual_OOF_Predictions.tif",
        "11_SEM_mechanism_analysis/06_Manuscript_Figure_11D/Figure_11D_Total_Effects.tif"
    ]
    # Figure scripts also produce PDF/PNG and may use slightly different final names.
    # Treat a directory with at least one TIFF as a valid artifact when the exact name differs.
    for rel in paths:
        path = ROOT / rel
        if path.is_file():
            checks.ok(f"artifact {path.name}", True)
            continue
        directory = path.parent
        candidates = list(directory.glob("*.tif")) + list(directory.glob("*.tiff")) if directory.is_dir() else []
        if candidates:
            checks.ok(f"artifact directory {directory.relative_to(ROOT)}", True, candidates[0].name)
        else:
            checks.missing(f"artifact {path.name}", path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-partial", action="store_true", help="Skip checks whose output files have not yet been generated")
    args = parser.parse_args()
    checks = Checks(args.allow_partial)

    check_input_manifest(checks)
    check_fai(checks)
    check_frp(checks)
    check_country_shares(checks)
    check_circulation(checks)
    check_admin1(checks)
    check_model_comparison(checks)
    check_sem(checks)
    check_primary_artifacts(checks)

    print("\n".join(checks.messages))
    print(f"\nSummary: {checks.passed} passed, {checks.failed} failed, {checks.skipped} skipped")
    return 0 if checks.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
