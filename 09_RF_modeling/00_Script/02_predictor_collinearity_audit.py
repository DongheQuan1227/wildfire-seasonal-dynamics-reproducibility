# -*- coding: utf-8 -*-

# Public reproducibility version.
# Input paths are resolved relative to the code repository.
# ForestPixelCount stores the original pixel count; ForestArea_km2 stores physical area.


from __future__ import annotations

import json
import math
import platform
import shutil
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy
import sklearn


# =============================================================================
# Paths and settings
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_ROOT = SCRIPT_DIR.parent
CODE_ROOT = MODEL_ROOT.parent
INPUT_CSV = CODE_ROOT / "00_Input_data" / "Base_Table_2001_2025.csv.bz2"

OUTPUT_ROOT = MODEL_ROOT / "02_Predictor_Collinearity_Audit"
LOG_FILE = OUTPUT_ROOT / "predictor_collinearity_audit.log"

FOREST_AREA_THRESHOLD_KM2 = 2.5
RANDOM_SEED = 2026
MAX_VIF_ROWS = 100_000

SEASON_LABELS = {
    0: "All_Seasons",
    1: "Spring",
    2: "Summer",
    3: "Autumn",
}

CORRELATION_THRESHOLDS = [0.70, 0.80, 0.90]

# This audit deliberately includes the complete FWI family and all SPEI
# timescales, even though the earlier exploratory RF excluded ISI, BUI,
# and FWI. The purpose here is to quantify redundancy before selecting
# a scientifically defensible predictor set.
PREDICTOR_FAMILIES: Dict[str, List[str]] = {
    "Landscape_Vegetation": [
        "BD",
        "ND",
        "NE",
        "EVI",
        "PTC",
    ],
    "Meteorology": [
        "Temp",
        "Pre",
        "Rhum",
        "Wind",
        "SSRD",
    ],
    "Lightning": [
        "LtgProxy",
    ],
    "Fire_Weather": [
        "FFMC",
        "DMC",
        "DC",
        "ISI",
        "BUI",
        "FWI",
    ],
    "Drought": [
        "SPEI1",
        "SPEI3",
        "SPEI6",
        "SPEI12",
        "SPEI24",
    ],
    "Terrain": [
        "DEM",
        "Slope",
        "Aspect",
    ],
    "Anthropogenic": [
        "POP",
        "Dis_Farm",
        "Dis_Build",
        "Road_dens",
        "Dis_Railway",
        "Dis_Power",
    ],
}

PREDICTORS: List[str] = [
    predictor
    for family_predictors in PREDICTOR_FAMILIES.values()
    for predictor in family_predictors
]

IDENTIFIER_COLUMNS = [
    "GRID_UID",
    "Country",
    "GRID_ID",
    "Year",
    "Season",
    "ForestPixelCount",
]

# Country is categorical and is intentionally excluded from correlation
# and VIF calculations. It will continue to be represented by dummy
# variables in the models.


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
# Metadata helpers
# =============================================================================

def predictor_to_family() -> Dict[str, str]:
    mapping: Dict[str, str] = {}

    for family, variables in PREDICTOR_FAMILIES.items():
        for variable in variables:
            if variable in mapping:
                raise ValueError(
                    f"Predictor appears in more than one family: {variable}"
                )
            mapping[variable] = family

    return mapping


FAMILY_MAP = predictor_to_family()


def validate_input_columns(df: pd.DataFrame) -> None:
    required = set(IDENTIFIER_COLUMNS + PREDICTORS)
    missing = sorted(required - set(df.columns))

    if missing:
        raise ValueError(
            "Input table is missing required columns:\n"
            + "\n".join(missing)
        )

    duplicate_predictors = sorted(
        {
            predictor
            for predictor in PREDICTORS
            if PREDICTORS.count(predictor) > 1
        }
    )

    if duplicate_predictors:
        raise ValueError(
            "Duplicate predictors were defined:\n"
            + "\n".join(duplicate_predictors)
        )


# =============================================================================
# Correlation helpers
# =============================================================================

def matrix_to_long(
    matrix: pd.DataFrame,
    value_name: str,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for i, variable_1 in enumerate(matrix.columns):
        for j in range(i + 1, len(matrix.columns)):
            variable_2 = matrix.columns[j]
            value = matrix.iloc[i, j]

            rows.append(
                {
                    "Variable_1": variable_1,
                    "Family_1": FAMILY_MAP[variable_1],
                    "Variable_2": variable_2,
                    "Family_2": FAMILY_MAP[variable_2],
                    value_name: value,
                    f"Absolute_{value_name}": (
                        abs(value) if pd.notna(value) else np.nan
                    ),
                }
            )

    return pd.DataFrame(rows)


def union_find_clusters(
    correlation_matrix: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    variables = list(correlation_matrix.columns)
    parent = {variable: variable for variable in variables}

    def find(variable: str) -> str:
        while parent[variable] != variable:
            parent[variable] = parent[parent[variable]]
            variable = parent[variable]
        return variable

    def union(a: str, b: str) -> None:
        root_a = find(a)
        root_b = find(b)

        if root_a != root_b:
            parent[root_b] = root_a

    for i, variable_1 in enumerate(variables):
        for j in range(i + 1, len(variables)):
            variable_2 = variables[j]
            correlation = correlation_matrix.iloc[i, j]

            if pd.notna(correlation) and abs(correlation) >= threshold:
                union(variable_1, variable_2)

    groups: Dict[str, List[str]] = {}

    for variable in variables:
        root = find(variable)
        groups.setdefault(root, []).append(variable)

    sorted_groups = sorted(
        groups.values(),
        key=lambda members: (-len(members), ",".join(sorted(members))),
    )

    rows: List[Dict[str, object]] = []

    for cluster_number, members in enumerate(sorted_groups, start=1):
        members = sorted(members)
        cluster_name = f"T{int(threshold * 100):02d}_C{cluster_number:02d}"

        for variable in members:
            partner_correlations = [
                abs(correlation_matrix.loc[variable, other])
                for other in members
                if other != variable
                and pd.notna(correlation_matrix.loc[variable, other])
            ]

            rows.append(
                {
                    "Threshold": threshold,
                    "Cluster_ID": cluster_name,
                    "Cluster_Size": len(members),
                    "Variable": variable,
                    "Family": FAMILY_MAP[variable],
                    "Cluster_Members": ";".join(members),
                    "Mean_Absolute_Correlation_within_Cluster": (
                        float(np.mean(partner_correlations))
                        if partner_correlations
                        else np.nan
                    ),
                    "Maximum_Absolute_Correlation_within_Cluster": (
                        float(np.max(partner_correlations))
                        if partner_correlations
                        else np.nan
                    ),
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# VIF and condition diagnostics
# =============================================================================

@dataclass
class VIFResult:
    table: pd.DataFrame
    diagnostics: Dict[str, object]


def safe_correlation_matrix(values: np.ndarray) -> np.ndarray:
    matrix = np.corrcoef(values, rowvar=False)
    matrix = np.asarray(matrix, dtype=float)

    # Numerical cleanup.
    matrix = (matrix + matrix.T) / 2.0
    np.fill_diagonal(matrix, 1.0)

    return matrix


def calculate_vif_from_correlation(
    correlation_matrix: np.ndarray,
    variable_names: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    number_of_variables = len(variable_names)

    for index, variable in enumerate(variable_names):
        other_indices = [
            other
            for other in range(number_of_variables)
            if other != index
        ]

        r_yx = correlation_matrix[index, other_indices]
        r_xx = correlation_matrix[
            np.ix_(other_indices, other_indices)
        ]

        inverse_method = "inverse"

        try:
            inverse_r_xx = np.linalg.inv(r_xx)
        except np.linalg.LinAlgError:
            inverse_r_xx = np.linalg.pinv(
                r_xx,
                rcond=1e-12,
            )
            inverse_method = "pseudoinverse"

        r_squared = float(
            r_yx @ inverse_r_xx @ r_yx.T
        )

        # Numerical values can exceed the mathematical range by tiny amounts.
        r_squared = max(0.0, min(1.0, r_squared))
        denominator = 1.0 - r_squared

        if denominator <= 1e-12:
            vif = np.inf
            tolerance = 0.0
        else:
            vif = 1.0 / denominator
            tolerance = denominator

        if math.isinf(vif):
            category = "Infinite"
        elif vif >= 10:
            category = "High"
        elif vif >= 5:
            category = "Moderate"
        else:
            category = "Low"

        rows.append(
            {
                "Predictor": variable,
                "Family": FAMILY_MAP[variable],
                "VIF": vif,
                "Tolerance": tolerance,
                "R_squared_against_other_predictors": r_squared,
                "VIF_Category": category,
                "Inverse_Method": inverse_method,
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["VIF", "Predictor"],
        ascending=[False, True],
    )


def calculate_vif(
    data: pd.DataFrame,
    season_code: int,
) -> VIFResult:
    complete = data[PREDICTORS].dropna().copy()

    original_complete_rows = len(complete)

    if original_complete_rows == 0:
        raise ValueError(
            f"No complete predictor rows are available for season {season_code}."
        )

    if len(complete) > MAX_VIF_ROWS:
        complete = complete.sample(
            n=MAX_VIF_ROWS,
            random_state=RANDOM_SEED + season_code,
        )

    standard_deviations = complete.std(ddof=0)
    zero_variance_variables = standard_deviations[
        standard_deviations <= 0
    ].index.tolist()

    variables_used = [
        predictor
        for predictor in PREDICTORS
        if predictor not in zero_variance_variables
    ]

    standardized = (
        complete[variables_used]
        - complete[variables_used].mean()
    ) / complete[variables_used].std(ddof=0)

    correlation_matrix = safe_correlation_matrix(
        standardized.to_numpy(dtype=float)
    )

    eigenvalues = np.linalg.eigvalsh(correlation_matrix)
    minimum_eigenvalue = float(np.min(eigenvalues))
    maximum_eigenvalue = float(np.max(eigenvalues))

    if minimum_eigenvalue <= 0:
        condition_number = np.inf
    else:
        condition_number = float(
            math.sqrt(maximum_eigenvalue / minimum_eigenvalue)
        )

    determinant = float(
        np.linalg.det(correlation_matrix)
    )

    vif_table = calculate_vif_from_correlation(
        correlation_matrix=correlation_matrix,
        variable_names=variables_used,
    )

    for variable in zero_variance_variables:
        vif_table = pd.concat(
            [
                vif_table,
                pd.DataFrame(
                    [
                        {
                            "Predictor": variable,
                            "Family": FAMILY_MAP[variable],
                            "VIF": np.inf,
                            "Tolerance": 0.0,
                            "R_squared_against_other_predictors": 1.0,
                            "VIF_Category": "Zero_Variance",
                            "Inverse_Method": "Not_applicable",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    diagnostics = {
        "Season_Code": season_code,
        "Season_Label": SEASON_LABELS[season_code],
        "Rows_available_after_forest_filter": len(data),
        "Complete_rows_before_VIF_sampling": original_complete_rows,
        "Rows_used_for_VIF": len(complete),
        "Predictors_defined": len(PREDICTORS),
        "Predictors_used_in_matrix": len(variables_used),
        "Zero_variance_predictor_count": len(zero_variance_variables),
        "Zero_variance_predictors": ";".join(zero_variance_variables),
        "Minimum_eigenvalue": minimum_eigenvalue,
        "Maximum_eigenvalue": maximum_eigenvalue,
        "Correlation_matrix_determinant": determinant,
        "Condition_index_sqrt_lambda_max_over_lambda_min": condition_number,
        "VIF_ge_5_count": int(
            (
                np.isfinite(vif_table["VIF"])
                & (vif_table["VIF"] >= 5)
            ).sum()
        ),
        "VIF_ge_10_count": int(
            (
                np.isfinite(vif_table["VIF"])
                & (vif_table["VIF"] >= 10)
            ).sum()
        ),
        "Infinite_VIF_count": int(
            np.isinf(vif_table["VIF"]).sum()
        ),
    }

    return VIFResult(
        table=vif_table,
        diagnostics=diagnostics,
    )


# =============================================================================
# Family summaries
# =============================================================================

def make_family_pair_summary(
    long_correlations: pd.DataFrame,
    season_code: int,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    family_pairs = sorted(
        {
            tuple(sorted([family_1, family_2]))
            for family_1, family_2 in zip(
                long_correlations["Family_1"],
                long_correlations["Family_2"],
            )
        }
    )

    for family_1, family_2 in family_pairs:
        mask = (
            (
                long_correlations["Family_1"] == family_1
            )
            & (
                long_correlations["Family_2"] == family_2
            )
        ) | (
            (
                long_correlations["Family_1"] == family_2
            )
            & (
                long_correlations["Family_2"] == family_1
            )
        )

        subset = long_correlations.loc[mask].copy()
        absolute_values = subset["Absolute_Spearman_Rho"].dropna()

        if absolute_values.empty:
            continue

        maximum_index = subset[
            "Absolute_Spearman_Rho"
        ].idxmax()

        maximum_row = subset.loc[maximum_index]

        rows.append(
            {
                "Season_Code": season_code,
                "Season_Label": SEASON_LABELS[season_code],
                "Family_1": family_1,
                "Family_2": family_2,
                "Pair_Count": len(absolute_values),
                "Mean_Absolute_Spearman_Rho": float(
                    absolute_values.mean()
                ),
                "Median_Absolute_Spearman_Rho": float(
                    absolute_values.median()
                ),
                "Maximum_Absolute_Spearman_Rho": float(
                    absolute_values.max()
                ),
                "Maximum_Pair": (
                    f"{maximum_row['Variable_1']}~"
                    f"{maximum_row['Variable_2']}"
                ),
                "Pairs_abs_rho_ge_0_70": int(
                    (absolute_values >= 0.70).sum()
                ),
                "Pairs_abs_rho_ge_0_80": int(
                    (absolute_values >= 0.80).sum()
                ),
                "Pairs_abs_rho_ge_0_90": int(
                    (absolute_values >= 0.90).sum()
                ),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# Per-season audit
# =============================================================================

def audit_one_season(
    filtered_data: pd.DataFrame,
    season_code: int,
) -> Dict[str, pd.DataFrame]:
    season_label = SEASON_LABELS[season_code]
    file_prefix = f"S{season_code}_{season_label}_"

    if season_code == 0:
        data = filtered_data.copy()
    else:
        data = filtered_data.loc[
            filtered_data["Season"] == season_code
        ].copy()

    log(
        f"{season_label}: {len(data):,} rows after ForestArea_km2 > "
        f"{FOREST_AREA_THRESHOLD_KM2}."
    )

    # Pairwise complete Spearman correlations.
    spearman_matrix = data[PREDICTORS].corr(
        method="spearman",
        min_periods=100,
    )

    spearman_matrix.to_csv(
        OUTPUT_ROOT / f"{file_prefix}Spearman_Correlation_Matrix.csv",
        encoding="utf-8-sig",
    )

    long_correlations = matrix_to_long(
        spearman_matrix,
        value_name="Spearman_Rho",
    )

    long_correlations = long_correlations.sort_values(
        [
            "Absolute_Spearman_Rho",
            "Variable_1",
            "Variable_2",
        ],
        ascending=[False, True, True],
    )

    long_correlations.to_csv(
        OUTPUT_ROOT / f"{file_prefix}Spearman_Correlation_Long.csv",
        index=False,
        encoding="utf-8-sig",
    )

    high_pairs = long_correlations.loc[
        long_correlations["Absolute_Spearman_Rho"] >= 0.70
    ].copy()

    high_pairs.insert(0, "Season_Code", season_code)
    high_pairs.insert(1, "Season_Label", season_label)

    high_pairs.to_csv(
        OUTPUT_ROOT / f"{file_prefix}High_Correlation_Pairs_abs_rho_ge_0_70.csv",
        index=False,
        encoding="utf-8-sig",
    )

    cluster_tables: List[pd.DataFrame] = []

    for threshold in CORRELATION_THRESHOLDS:
        clusters = union_find_clusters(
            correlation_matrix=spearman_matrix,
            threshold=threshold,
        )

        clusters.insert(0, "Season_Code", season_code)
        clusters.insert(1, "Season_Label", season_label)

        clusters.to_csv(
            OUTPUT_ROOT
            / f"{file_prefix}Correlation_Clusters_abs_rho_ge_{threshold:.2f}.csv",
            index=False,
            encoding="utf-8-sig",
        )

        cluster_tables.append(clusters)

    vif_result = calculate_vif(
        data=data,
        season_code=season_code,
    )

    vif_table = vif_result.table.copy()
    vif_table.insert(0, "Season_Code", season_code)
    vif_table.insert(1, "Season_Label", season_label)

    vif_table.to_csv(
        OUTPUT_ROOT / f"{file_prefix}VIF_Table.csv",
        index=False,
        encoding="utf-8-sig",
    )

    family_summary = make_family_pair_summary(
        long_correlations=long_correlations,
        season_code=season_code,
    )

    family_summary.to_csv(
        OUTPUT_ROOT / f"{file_prefix}Family_Correlation_Summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    missingness = pd.DataFrame(
        {
            "Predictor": PREDICTORS,
            "Family": [FAMILY_MAP[p] for p in PREDICTORS],
            "Rows": len(data),
            "Missing_N": [
                int(data[p].isna().sum())
                for p in PREDICTORS
            ],
            "Missing_Percent": [
                float(data[p].isna().mean() * 100)
                for p in PREDICTORS
            ],
            "Unique_Nonmissing_Values": [
                int(data[p].nunique(dropna=True))
                for p in PREDICTORS
            ],
        }
    )

    missingness.insert(0, "Season_Code", season_code)
    missingness.insert(1, "Season_Label", season_label)

    missingness.to_csv(
        OUTPUT_ROOT / f"{file_prefix}Predictor_Missingness_and_Unique_Values.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return {
        "high_pairs": high_pairs,
        "clusters": pd.concat(cluster_tables, ignore_index=True),
        "vif": vif_table,
        "family_summary": family_summary,
        "missingness": missingness,
        "condition": pd.DataFrame([vif_result.diagnostics]),
    }


# =============================================================================
# Summary outputs
# =============================================================================

def build_text_summary(
    high_pairs: pd.DataFrame,
    vif_table: pd.DataFrame,
    condition_table: pd.DataFrame,
) -> None:
    lines: List[str] = []

    lines.append("Predictor collinearity audit")
    lines.append("=" * 78)
    lines.append(
        "Country was excluded from correlation and VIF calculations because it "
        "is categorical and will be dummy-coded in the models."
    )
    lines.append(
        "No predictor was automatically removed. The outputs are intended to "
        "support a scientifically justified selection in the next step."
    )
    lines.append("")

    for season_code in [0, 1, 2, 3]:
        season_label = SEASON_LABELS[season_code]
        season_pairs = high_pairs.loc[
            high_pairs["Season_Code"] == season_code
        ]
        season_vif = vif_table.loc[
            vif_table["Season_Code"] == season_code
        ]
        season_condition = condition_table.loc[
            condition_table["Season_Code"] == season_code
        ]

        lines.append(f"{season_label}")
        lines.append("-" * 78)
        lines.append(
            f"High-correlation pairs (|Spearman rho| >= 0.70): "
            f"{len(season_pairs)}"
        )
        lines.append(
            f"Predictors with VIF >= 5: "
            f"{int((season_vif['VIF'] >= 5).sum())}"
        )
        lines.append(
            f"Predictors with VIF >= 10: "
            f"{int((season_vif['VIF'] >= 10).sum())}"
        )

        if not season_condition.empty:
            condition_value = season_condition.iloc[0][
                "Condition_index_sqrt_lambda_max_over_lambda_min"
            ]
            lines.append(
                f"Condition index: {condition_value}"
            )

        if not season_pairs.empty:
            top_pairs = season_pairs.head(10)

            lines.append("Top correlation pairs:")
            for row in top_pairs.itertuples(index=False):
                lines.append(
                    f"  {row.Variable_1} ~ {row.Variable_2}: "
                    f"rho={row.Spearman_Rho:.6f}"
                )

        lines.append("")

    (OUTPUT_ROOT / "Collinearity_Audit_Summary.txt").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )




# =============================================================================
# Main
# =============================================================================

def main() -> int:
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text("", encoding="utf-8")

    log("=" * 78)
    log("Starting predictor collinearity audit")
    log(f"Input: {INPUT_CSV}")
    log(f"Output: {OUTPUT_ROOT}")
    log("=" * 78)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Base table not found:\n{INPUT_CSV}"
        )

    data = pd.read_csv(INPUT_CSV)
    validate_input_columns(data)

    duplicate_panel = data.duplicated(
        ["GRID_UID", "Year", "Season"],
        keep=False,
    )

    if duplicate_panel.any():
        data.loc[duplicate_panel].to_csv(
            OUTPUT_ROOT / "ERROR_Duplicate_Panel_Records.csv",
            index=False,
            encoding="utf-8-sig",
        )
        raise ValueError(
            "Duplicate GRID_UID-Year-Season records were found."
        )

    filtered = data.loc[
        data["ForestArea_km2"] > FOREST_AREA_THRESHOLD_KM2
    ].copy()

    metadata_rows = []

    for order, predictor in enumerate(PREDICTORS, start=1):
        metadata_rows.append(
            {
                "Order": order,
                "Predictor": predictor,
                "Family": FAMILY_MAP[predictor],
                "Included_in_collinearity_audit": True,
                "Categorical": False,
            }
        )

    metadata_rows.append(
        {
            "Order": len(metadata_rows) + 1,
            "Predictor": "Country",
            "Family": "Country_Fixed_Effect",
            "Included_in_collinearity_audit": False,
            "Categorical": True,
        }
    )

    pd.DataFrame(metadata_rows).to_csv(
        OUTPUT_ROOT / "00_Predictor_Metadata.csv",
        index=False,
        encoding="utf-8-sig",
    )

    qa_summary = pd.DataFrame(
        [
            ["Input_rows", len(data)],
            ["Input_columns", len(data.columns)],
            ["Unique_GRID_UIDs", data["GRID_UID"].nunique()],
            ["Years", data["Year"].nunique()],
            ["Seasons", data["Season"].nunique()],
            [
                "Rows_after_ForestArea_km2_gt2_5",
                len(filtered),
            ],
            [
                "Unique_GRID_UIDs_after_ForestArea_km2_gt2_5",
                filtered["GRID_UID"].nunique(),
            ],
            ["Continuous_predictors_audited", len(PREDICTORS)],
            [
                "Country_excluded_from_numeric_audit",
                True,
            ],
            ["Maximum_rows_used_for_each_VIF", MAX_VIF_ROWS],
        ],
        columns=["Check", "Value"],
    )

    qa_summary.to_csv(
        OUTPUT_ROOT / "01_QA_Summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    all_results: Dict[str, List[pd.DataFrame]] = {
        "high_pairs": [],
        "clusters": [],
        "vif": [],
        "family_summary": [],
        "missingness": [],
        "condition": [],
    }

    for season_code in [0, 1, 2, 3]:
        result = audit_one_season(
            filtered_data=filtered,
            season_code=season_code,
        )

        for key in all_results:
            all_results[key].append(result[key])

    combined_high_pairs = pd.concat(
        all_results["high_pairs"],
        ignore_index=True,
    )

    combined_clusters = pd.concat(
        all_results["clusters"],
        ignore_index=True,
    )

    combined_vif = pd.concat(
        all_results["vif"],
        ignore_index=True,
    )

    combined_family_summary = pd.concat(
        all_results["family_summary"],
        ignore_index=True,
    )

    combined_missingness = pd.concat(
        all_results["missingness"],
        ignore_index=True,
    )

    combined_condition = pd.concat(
        all_results["condition"],
        ignore_index=True,
    )

    combined_high_pairs.to_csv(
        OUTPUT_ROOT / "02_All_High_Correlation_Pairs.csv",
        index=False,
        encoding="utf-8-sig",
    )

    combined_vif.to_csv(
        OUTPUT_ROOT / "03_All_VIF_Results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    combined_condition.to_csv(
        OUTPUT_ROOT / "04_Condition_Diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    combined_clusters.to_csv(
        OUTPUT_ROOT / "05_All_Correlation_Clusters.csv",
        index=False,
        encoding="utf-8-sig",
    )

    combined_family_summary.to_csv(
        OUTPUT_ROOT / "06_All_Family_Correlation_Summaries.csv",
        index=False,
        encoding="utf-8-sig",
    )

    combined_missingness.to_csv(
        OUTPUT_ROOT / "07_All_Predictor_Missingness.csv",
        index=False,
        encoding="utf-8-sig",
    )

    build_text_summary(
        high_pairs=combined_high_pairs,
        vif_table=combined_vif,
        condition_table=combined_condition,
    )

    environment = {
        "Created": datetime.now().isoformat(timespec="seconds"),
        "Python": sys.version,
        "Platform": platform.platform(),
        "NumPy": np.__version__,
        "pandas": pd.__version__,
        "SciPy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "Input_file": str(INPUT_CSV),
        "ForestArea_threshold_km2": FOREST_AREA_THRESHOLD_KM2,
        "Predictor_count": len(PREDICTORS),
        "VIF_max_rows": MAX_VIF_ROWS,
        "Correlation_method": "Spearman pairwise complete observations",
        "VIF_method": (
            "Standard Pearson-correlation VIF on complete cases; "
            "maximum 100,000 reproducibly sampled rows per season."
        ),
    }

    with (
        OUTPUT_ROOT / "Software_Environment_and_Method.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            environment,
            file,
            ensure_ascii=False,
            indent=2,
        )

    log("=" * 78)
    log("Predictor collinearity audit completed successfully.")
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
