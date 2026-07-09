# -*- coding: utf-8 -*-
"""
Panel-dependence and clustered-inference audit.

Recommended public location
---------------------------
<CODE_ROOT>/08_Regression_modeling/00_Script/
    06_audit_panel_dependence_and_clustered_inference.py

Inputs
------
<CODE_ROOT>/08_Regression_modeling/
    05_Component_Specific_Predictor_Refinement/
        16_Final_Component_Specific_Predictor_Structure.csv
        Refined_Selected_OOF_Predictions_S*_*.csv.gz

Output
------
<CODE_ROOT>/08_Regression_modeling/
    06_Panel_Dependence_and_Clustered_Inference_Audit/

Purpose
-------
This step does not change the fixed mean structures selected in Step 05.

It audits whether temporal out-of-fold residual dependence remains at the:

1. GRID_UID level;
2. Year level;
3. within-grid one-year lag.

Three process-specific residuals are examined:

1. occurrence residual:
       I(Y > 0) - P(Y > 0)

2. unconditional rate residual:
       observed count / exposure - predicted count / exposure

3. positive-magnitude log residual:
       log1p(observed positive count)
       - log1p(predicted count conditional on Y > 0)

The script also estimates uncertainty in OOF performance under GRID_UID and
Year cluster bootstrap resampling. These diagnostics are used to justify the final covariance structure. They are not used to alter the selected mean models.

Country remains a fixed contextual effect, and the selected mean structures are not changed.
"""

from __future__ import annotations

import json
import math
import platform
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy
from scipy.stats import spearmanr


# =============================================================================
# Repository-relative paths
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REGRESSION_ROOT = SCRIPT_DIR.parent

SOURCE_ROOT = (
    REGRESSION_ROOT
    / "05_Component_Specific_Predictor_Refinement"
)

FINAL_STRUCTURE_CSV = (
    SOURCE_ROOT
    / "16_Final_Component_Specific_Predictor_Structure.csv"
)

OUTPUT_ROOT = (
    REGRESSION_ROOT
    / "06_Panel_Dependence_and_Clustered_Inference_Audit"
)

LOG_FILE = (
    OUTPUT_ROOT
    / "panel_dependence_clustered_inference.log"
)


# =============================================================================
# Expected analysis structure
# =============================================================================

SEASONS = {
    1: "Spring",
    2: "Summer",
    3: "Autumn",
}

COUNT_RESPONSES = {
    "Fire_Count": "FCD",
    "Burned_Pixel_Count": "BAD",
}

EXPECTED_YEARS = set(
    range(
        2001,
        2026,
    )
)

EXPECTED_FOLDS = {
    1,
    2,
    3,
    4,
    5,
}

COUNTRY_LEVELS = {
    "China",
    "NK",
    "Russia",
}


# =============================================================================
# Reproducibility and thresholds
# =============================================================================

RANDOM_SEED = 2026
CLUSTER_BOOTSTRAP_REPLICATES = 2000
PROBABILITY_EPSILON = 1e-12

# Descriptive dependence categories.
ICC_NEGLIGIBLE_MAX = 0.01
ICC_WEAK_MAX = 0.05
ICC_MODERATE_MAX = 0.20

AUTOCORRELATION_NEGLIGIBLE_MAX = 0.10
AUTOCORRELATION_MODERATE_MAX = 0.30

# These rules identify dependence that must be reflected in final covariance estimation.
ICC_TEST_TRIGGER = 0.05
ABS_LAG1_TEST_TRIGGER = 0.20
CLUSTER_SE_INFLATION_TEST_TRIGGER = 1.50

# The number of Year clusters is small enough that year-cluster uncertainty
# should be interpreted cautiously.
YEAR_CLUSTER_CAUTION_THRESHOLD = 30


# =============================================================================
# Logging
# =============================================================================

def log(message: str) -> None:
    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    line = f"[{timestamp}] {message}"
    print(line, flush=True)

    with LOG_FILE.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(
            line + "\n"
        )


# =============================================================================
# General helpers
# =============================================================================

def finite_or_nan(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan

    if not np.isfinite(result):
        return np.nan

    return result


def safe_ratio(
    numerator: float,
    denominator: float,
) -> float:
    if (
        not np.isfinite(numerator)
        or not np.isfinite(denominator)
        or denominator == 0
    ):
        return np.nan

    return float(
        numerator / denominator
    )


def classify_icc(value: float) -> str:
    if not np.isfinite(value):
        return "Not_estimable"

    absolute = abs(
        float(value)
    )

    if absolute < ICC_NEGLIGIBLE_MAX:
        return "Negligible"

    if absolute < ICC_WEAK_MAX:
        return "Weak"

    if absolute < ICC_MODERATE_MAX:
        return "Moderate"

    return "Strong"


def classify_autocorrelation(
    value: float,
) -> str:
    if not np.isfinite(value):
        return "Not_estimable"

    absolute = abs(
        float(value)
    )

    if absolute < AUTOCORRELATION_NEGLIGIBLE_MAX:
        return "Negligible"

    if absolute < AUTOCORRELATION_MODERATE_MAX:
        return "Moderate"

    return "Strong"


def required_columns_present(
    data: pd.DataFrame,
    required: Iterable[str],
    table_name: str,
) -> None:
    missing = sorted(
        set(required)
        - set(data.columns)
    )

    if missing:
        raise ValueError(
            f"{table_name} is missing required columns:\n"
            + "\n".join(
                missing
            )
        )


def prediction_path(
    season_code: int,
    response: str,
) -> Path:
    return (
        SOURCE_ROOT
        / (
            "Refined_Selected_OOF_Predictions_"
            f"S{season_code}_{response}.csv.gz"
        )
    )


# =============================================================================
# Input validation
# =============================================================================

PREDICTION_REQUIRED_COLUMNS = {
    "GRID_UID",
    "GRID_ID",
    "Country",
    "Year",
    "Season",
    "Temporal_Fold",
    "ForestPixelCount",
    "Count_Response",
    "Rate_Scale_Name",
    "Model_Structure",
    "Observed_Count",
    "Predicted_Count",
    "Observed_Rate",
    "Predicted_Rate",
    "Predicted_Zero_Probability",
    "Predictive_LogProbability",
}


def read_final_structure() -> pd.DataFrame:
    if not FINAL_STRUCTURE_CSV.exists():
        raise FileNotFoundError(
            "Step-05 final structure table was not found:\n"
            f"{FINAL_STRUCTURE_CSV}"
        )

    structure = pd.read_csv(
        FINAL_STRUCTURE_CSV
    )

    required_columns_present(
        structure,
        {
            "Season_Code",
            "Season_Label",
            "Count_Response",
            "Rate_Scale_Name",
            "Candidate_ID",
            "Model_Structure",
            "Structural_Zero_Predictors",
            "Count_Predictors",
            "Transformation_Scheme",
        },
        "Step-05 final structure table",
    )

    expected = {
        (
            season_code,
            response,
        )
        for season_code in SEASONS
        for response in COUNT_RESPONSES
    }

    observed = {
        (
            int(row.Season_Code),
            str(row.Count_Response),
        )
        for row in structure.itertuples(
            index=False
        )
    }

    if observed != expected:
        raise ValueError(
            "The Step-05 structure table does not contain exactly "
            "the expected six season-response combinations."
        )

    if len(structure) != 6:
        raise ValueError(
            "The Step-05 structure table must contain exactly six rows."
        )

    return structure.sort_values(
        [
            "Season_Code",
            "Count_Response",
        ]
    ).reset_index(
        drop=True
    )


def validate_prediction_data(
    data: pd.DataFrame,
    season_code: int,
    response: str,
    expected_structure: str,
    source_path: Path,
) -> None:
    required_columns_present(
        data,
        PREDICTION_REQUIRED_COLUMNS,
        source_path.name,
    )

    if data.empty:
        raise ValueError(
            f"Prediction file is empty: {source_path}"
        )

    if set(
        data[
            "Season"
        ].astype(int)
    ) != {
        season_code
    }:
        raise ValueError(
            f"Unexpected season values in {source_path.name}."
        )

    if set(
        data[
            "Count_Response"
        ].astype(str)
    ) != {
        response
    }:
        raise ValueError(
            f"Unexpected response values in {source_path.name}."
        )

    if set(
        data[
            "Model_Structure"
        ].astype(str)
    ) != {
        expected_structure
    }:
        raise ValueError(
            f"Unexpected model structure in {source_path.name}."
        )

    if set(
        data[
            "Rate_Scale_Name"
        ].astype(str)
    ) != {
        COUNT_RESPONSES[
            response
        ]
    }:
        raise ValueError(
            f"Unexpected rate-scale name in {source_path.name}."
        )

    if set(
        data[
            "Temporal_Fold"
        ].astype(int)
    ) != EXPECTED_FOLDS:
        raise ValueError(
            f"Unexpected temporal folds in {source_path.name}."
        )

    if set(
        data[
            "Year"
        ].astype(int)
    ) != EXPECTED_YEARS:
        raise ValueError(
            f"Unexpected year coverage in {source_path.name}."
        )

    unexpected_countries = (
        set(
            data[
                "Country"
            ].dropna().astype(str)
        )
        - COUNTRY_LEVELS
    )

    if unexpected_countries:
        raise ValueError(
            "Unexpected Country values in "
            f"{source_path.name}: "
            + ", ".join(
                sorted(
                    unexpected_countries
                )
            )
        )

    if data.duplicated(
        [
            "GRID_UID",
            "Year",
            "Season",
        ]
    ).any():
        raise ValueError(
            f"Duplicate GRID_UID-Year-Season rows in {source_path.name}."
        )

    numeric_columns = [
        "ForestPixelCount",
        "Observed_Count",
        "Predicted_Count",
        "Observed_Rate",
        "Predicted_Rate",
        "Predicted_Zero_Probability",
        "Predictive_LogProbability",
    ]

    numeric = data[
        numeric_columns
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if not np.isfinite(
        numeric.to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError(
            f"Non-finite numeric values in {source_path.name}."
        )

    if (
        numeric[
            "ForestPixelCount"
        ]
        <= 0
    ).any():
        raise ValueError(
            f"Nonpositive exposure in {source_path.name}."
        )

    if (
        numeric[
            "Observed_Count"
        ]
        < 0
    ).any():
        raise ValueError(
            f"Negative observed count in {source_path.name}."
        )

    if (
        numeric[
            "Predicted_Count"
        ]
        < 0
    ).any():
        raise ValueError(
            f"Negative predicted count in {source_path.name}."
        )

    if (
        ~numeric[
            "Predicted_Zero_Probability"
        ].between(
            0.0,
            1.0,
            inclusive="both",
        )
    ).any():
        raise ValueError(
            f"Invalid zero probabilities in {source_path.name}."
        )

    observed_rate_check = (
        numeric[
            "Observed_Count"
        ]
        / numeric[
            "ForestPixelCount"
        ]
    )

    predicted_rate_check = (
        numeric[
            "Predicted_Count"
        ]
        / numeric[
            "ForestPixelCount"
        ]
    )

    if not np.allclose(
        observed_rate_check,
        numeric[
            "Observed_Rate"
        ],
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError(
            f"Observed rate is inconsistent in {source_path.name}."
        )

    if not np.allclose(
        predicted_rate_check,
        numeric[
            "Predicted_Rate"
        ],
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError(
            f"Predicted rate is inconsistent in {source_path.name}."
        )


# =============================================================================
# Residual construction
# =============================================================================

def add_process_residuals(
    data: pd.DataFrame,
) -> pd.DataFrame:
    result = data.copy()

    observed_count = pd.to_numeric(
        result[
            "Observed_Count"
        ],
        errors="raise",
    ).to_numpy(
        dtype=float
    )

    predicted_count = pd.to_numeric(
        result[
            "Predicted_Count"
        ],
        errors="raise",
    ).to_numpy(
        dtype=float
    )

    observed_rate = pd.to_numeric(
        result[
            "Observed_Rate"
        ],
        errors="raise",
    ).to_numpy(
        dtype=float
    )

    predicted_rate = pd.to_numeric(
        result[
            "Predicted_Rate"
        ],
        errors="raise",
    ).to_numpy(
        dtype=float
    )

    predicted_zero = np.clip(
        pd.to_numeric(
            result[
                "Predicted_Zero_Probability"
            ],
            errors="raise",
        ).to_numpy(
            dtype=float
        ),
        PROBABILITY_EPSILON,
        1.0
        - PROBABILITY_EPSILON,
    )

    observed_event = (
        observed_count > 0
    ).astype(float)

    predicted_event = (
        1.0
        - predicted_zero
    )

    occurrence_residual = (
        observed_event
        - predicted_event
    )

    rate_residual = (
        observed_rate
        - predicted_rate
    )

    count_residual = (
        observed_count
        - predicted_count
    )

    conditional_positive_prediction = (
        predicted_count
        / np.clip(
            predicted_event,
            PROBABILITY_EPSILON,
            None,
        )
    )

    positive_mask = (
        observed_count > 0
    )

    positive_log_residual = np.full(
        len(result),
        np.nan,
        dtype=float,
    )

    positive_log_residual[
        positive_mask
    ] = (
        np.log1p(
            observed_count[
                positive_mask
            ]
        )
        - np.log1p(
            conditional_positive_prediction[
                positive_mask
            ]
        )
    )

    result[
        "Observed_Event"
    ] = observed_event

    result[
        "Predicted_Event_Probability"
    ] = predicted_event

    result[
        "Occurrence_Residual"
    ] = occurrence_residual

    result[
        "Count_Residual"
    ] = count_residual

    result[
        "Rate_Residual"
    ] = rate_residual

    result[
        "Predicted_Positive_Conditional_Count"
    ] = conditional_positive_prediction

    result[
        "Positive_Log_Magnitude_Residual"
    ] = positive_log_residual

    result[
        "Negative_LogProbability"
    ] = (
        -pd.to_numeric(
            result[
                "Predictive_LogProbability"
            ],
            errors="raise",
        )
    )

    result[
        "Occurrence_Brier_Contribution"
    ] = np.square(
        occurrence_residual
    )

    result[
        "Rate_Squared_Error"
    ] = np.square(
        rate_residual
    )

    result[
        "Rate_Absolute_Error"
    ] = np.abs(
        rate_residual
    )

    return result


# =============================================================================
# Residual summaries
# =============================================================================

def residual_summary(
    values: np.ndarray,
) -> Dict[str, float]:
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    if len(values) == 0:
        return {
            "N": 0,
            "Mean": np.nan,
            "SD": np.nan,
            "RMSE": np.nan,
            "MAE": np.nan,
            "Median": np.nan,
            "P05": np.nan,
            "P95": np.nan,
        }

    return {
        "N": int(
            len(values)
        ),
        "Mean": float(
            values.mean()
        ),
        "SD": float(
            values.std(
                ddof=1
            )
        ) if len(values) > 1 else np.nan,
        "RMSE": float(
            np.sqrt(
                np.mean(
                    np.square(
                        values
                    )
                )
            )
        ),
        "MAE": float(
            np.mean(
                np.abs(
                    values
                )
            )
        ),
        "Median": float(
            np.median(
                values
            )
        ),
        "P05": float(
            np.quantile(
                values,
                0.05,
            )
        ),
        "P95": float(
            np.quantile(
                values,
                0.95,
            )
        ),
    }


def pooled_residual_rows(
    data: pd.DataFrame,
    season_code: int,
    response: str,
    model_structure: str,
) -> List[Dict[str, object]]:
    residual_definitions = {
        "Occurrence": (
            "Occurrence_Residual",
            "All rows",
        ),
        "Unconditional_Rate": (
            "Rate_Residual",
            "All rows",
        ),
        "Positive_Magnitude": (
            "Positive_Log_Magnitude_Residual",
            "Observed positive rows only",
        ),
    }

    rows: List[
        Dict[str, object]
    ] = []

    for process, (
        residual_column,
        analysis_rows,
    ) in residual_definitions.items():
        summary = residual_summary(
            data[
                residual_column
            ].to_numpy(
                dtype=float
            )
        )

        rows.append(
            {
                "Season_Code": (
                    season_code
                ),
                "Season_Label": (
                    SEASONS[
                        season_code
                    ]
                ),
                "Count_Response": (
                    response
                ),
                "Rate_Scale_Name": (
                    COUNT_RESPONSES[
                        response
                    ]
                ),
                "Model_Structure": (
                    model_structure
                ),
                "Process": (
                    process
                ),
                "Residual_Column": (
                    residual_column
                ),
                "Analysis_Rows": (
                    analysis_rows
                ),
                **summary,
            }
        )

    return rows


# =============================================================================
# One-way ICC
# =============================================================================

def one_way_icc(
    data: pd.DataFrame,
    value_column: str,
    group_column: str,
) -> Dict[str, float]:
    subset = data[
        [
            value_column,
            group_column,
        ]
    ].dropna()

    if subset.empty:
        return {
            "N": 0,
            "Group_Count": 0,
            "Mean_Group_Size": np.nan,
            "Minimum_Group_Size": np.nan,
            "Maximum_Group_Size": np.nan,
            "MS_Between": np.nan,
            "MS_Within": np.nan,
            "Effective_Group_Size_n0": np.nan,
            "Between_Variance_Signed": np.nan,
            "Between_Variance_Truncated": np.nan,
            "Within_Variance": np.nan,
            "ICC_Signed": np.nan,
            "ICC_Truncated": np.nan,
            "Design_Effect_Truncated": np.nan,
        }

    grouped = subset.groupby(
        group_column,
        sort=False,
    )[
        value_column
    ]

    counts = grouped.size().astype(
        float
    )

    means = grouped.mean().astype(
        float
    )

    group_count = int(
        len(counts)
    )

    sample_size = int(
        counts.sum()
    )

    if (
        group_count < 2
        or sample_size <= group_count
    ):
        return {
            "N": sample_size,
            "Group_Count": group_count,
            "Mean_Group_Size": float(
                counts.mean()
            ),
            "Minimum_Group_Size": float(
                counts.min()
            ),
            "Maximum_Group_Size": float(
                counts.max()
            ),
            "MS_Between": np.nan,
            "MS_Within": np.nan,
            "Effective_Group_Size_n0": np.nan,
            "Between_Variance_Signed": np.nan,
            "Between_Variance_Truncated": np.nan,
            "Within_Variance": np.nan,
            "ICC_Signed": np.nan,
            "ICC_Truncated": np.nan,
            "Design_Effect_Truncated": np.nan,
        }

    grand_mean = float(
        np.average(
            means.to_numpy(),
            weights=counts.to_numpy(),
        )
    )

    ss_between = float(
        np.sum(
            counts.to_numpy()
            * np.square(
                means.to_numpy()
                - grand_mean
            )
        )
    )

    merged_mean = subset[
        group_column
    ].map(
        means
    ).to_numpy(
        dtype=float
    )

    residual = (
        subset[
            value_column
        ].to_numpy(
            dtype=float
        )
        - merged_mean
    )

    ss_within = float(
        np.sum(
            np.square(
                residual
            )
        )
    )

    df_between = (
        group_count - 1
    )

    df_within = (
        sample_size
        - group_count
    )

    ms_between = (
        ss_between
        / df_between
    )

    ms_within = (
        ss_within
        / df_within
    )

    n0 = (
        sample_size
        - float(
            np.sum(
                np.square(
                    counts.to_numpy()
                )
            )
        )
        / sample_size
    ) / df_between

    between_signed = (
        (
            ms_between
            - ms_within
        )
        / n0
    )

    between_truncated = max(
        0.0,
        between_signed,
    )

    denominator_signed = (
        between_signed
        + ms_within
    )

    icc_signed = (
        between_signed
        / denominator_signed
        if denominator_signed > 0
        else np.nan
    )

    denominator_truncated = (
        between_truncated
        + ms_within
    )

    icc_truncated = (
        between_truncated
        / denominator_truncated
        if denominator_truncated > 0
        else np.nan
    )

    mean_group_size = float(
        counts.mean()
    )

    design_effect = (
        1.0
        + (
            mean_group_size
            - 1.0
        )
        * icc_truncated
        if np.isfinite(
            icc_truncated
        )
        else np.nan
    )

    return {
        "N": sample_size,
        "Group_Count": group_count,
        "Mean_Group_Size": mean_group_size,
        "Minimum_Group_Size": float(
            counts.min()
        ),
        "Maximum_Group_Size": float(
            counts.max()
        ),
        "MS_Between": float(
            ms_between
        ),
        "MS_Within": float(
            ms_within
        ),
        "Effective_Group_Size_n0": float(
            n0
        ),
        "Between_Variance_Signed": float(
            between_signed
        ),
        "Between_Variance_Truncated": float(
            between_truncated
        ),
        "Within_Variance": float(
            ms_within
        ),
        "ICC_Signed": float(
            icc_signed
        ) if np.isfinite(
            icc_signed
        ) else np.nan,
        "ICC_Truncated": float(
            icc_truncated
        ) if np.isfinite(
            icc_truncated
        ) else np.nan,
        "Design_Effect_Truncated": float(
            design_effect
        ) if np.isfinite(
            design_effect
        ) else np.nan,
    }


def dependence_icc_rows(
    data: pd.DataFrame,
    season_code: int,
    response: str,
    model_structure: str,
) -> List[Dict[str, object]]:
    residual_definitions = {
        "Occurrence": (
            "Occurrence_Residual",
            "All rows",
        ),
        "Unconditional_Rate": (
            "Rate_Residual",
            "All rows",
        ),
        "Positive_Magnitude": (
            "Positive_Log_Magnitude_Residual",
            "Observed positive rows only",
        ),
    }

    grouping_definitions = {
        "GRID_UID": "GRID_UID",
        "Year": "Year",
    }

    rows: List[
        Dict[str, object]
    ] = []

    for process, (
        residual_column,
        analysis_rows,
    ) in residual_definitions.items():
        for level, group_column in (
            grouping_definitions.items()
        ):
            estimates = one_way_icc(
                data=data,
                value_column=(
                    residual_column
                ),
                group_column=(
                    group_column
                ),
            )

            rows.append(
                {
                    "Season_Code": (
                        season_code
                    ),
                    "Season_Label": (
                        SEASONS[
                            season_code
                        ]
                    ),
                    "Count_Response": (
                        response
                    ),
                    "Rate_Scale_Name": (
                        COUNT_RESPONSES[
                            response
                        ]
                    ),
                    "Model_Structure": (
                        model_structure
                    ),
                    "Process": (
                        process
                    ),
                    "Residual_Column": (
                        residual_column
                    ),
                    "Analysis_Rows": (
                        analysis_rows
                    ),
                    "Grouping_Level": (
                        level
                    ),
                    **estimates,
                    "Dependence_Category": (
                        classify_icc(
                            estimates[
                                "ICC_Truncated"
                            ]
                        )
                    ),
                }
            )

    return rows


# =============================================================================
# Within-grid lag-one residual correlation
# =============================================================================

def within_grid_lag1(
    data: pd.DataFrame,
    residual_column: str,
) -> Dict[str, float]:
    subset = (
        data[
            [
                "GRID_UID",
                "Year",
                residual_column,
            ]
        ]
        .dropna()
        .sort_values(
            [
                "GRID_UID",
                "Year",
            ]
        )
        .copy()
    )

    if subset.empty:
        return {
            "Pair_Count": 0,
            "Grid_Count_with_Pairs": 0,
            "Lag1_Pearson": np.nan,
            "Lag1_Spearman": np.nan,
        }

    subset[
        "Previous_Year"
    ] = (
        subset.groupby(
            "GRID_UID"
        )[
            "Year"
        ].shift(1)
    )

    subset[
        "Previous_Residual"
    ] = (
        subset.groupby(
            "GRID_UID"
        )[
            residual_column
        ].shift(1)
    )

    pairs = subset.loc[
        (
            subset[
                "Year"
            ]
            - subset[
                "Previous_Year"
            ]
        )
        == 1
    ].dropna(
        subset=[
            residual_column,
            "Previous_Residual",
        ]
    )

    if len(pairs) < 3:
        return {
            "Pair_Count": int(
                len(pairs)
            ),
            "Grid_Count_with_Pairs": int(
                pairs[
                    "GRID_UID"
                ].nunique()
            ),
            "Lag1_Pearson": np.nan,
            "Lag1_Spearman": np.nan,
        }

    current = pairs[
        residual_column
    ].to_numpy(
        dtype=float
    )

    previous = pairs[
        "Previous_Residual"
    ].to_numpy(
        dtype=float
    )

    pearson = float(
        np.corrcoef(
            previous,
            current,
        )[
            0,
            1,
        ]
    )

    spearman = spearmanr(
        previous,
        current,
        nan_policy="omit",
    )

    return {
        "Pair_Count": int(
            len(pairs)
        ),
        "Grid_Count_with_Pairs": int(
            pairs[
                "GRID_UID"
            ].nunique()
        ),
        "Lag1_Pearson": (
            pearson
        ),
        "Lag1_Spearman": float(
            spearman.statistic
        ),
    }


def lag1_rows(
    data: pd.DataFrame,
    season_code: int,
    response: str,
    model_structure: str,
) -> List[Dict[str, object]]:
    definitions = {
        "Occurrence": (
            "Occurrence_Residual",
            "All rows",
        ),
        "Unconditional_Rate": (
            "Rate_Residual",
            "All rows",
        ),
        "Positive_Magnitude": (
            "Positive_Log_Magnitude_Residual",
            "Observed positive rows only",
        ),
    }

    rows: List[
        Dict[str, object]
    ] = []

    for process, (
        residual_column,
        analysis_rows,
    ) in definitions.items():
        result = within_grid_lag1(
            data=data,
            residual_column=(
                residual_column
            ),
        )

        rows.append(
            {
                "Season_Code": (
                    season_code
                ),
                "Season_Label": (
                    SEASONS[
                        season_code
                    ]
                ),
                "Count_Response": (
                    response
                ),
                "Rate_Scale_Name": (
                    COUNT_RESPONSES[
                        response
                    ]
                ),
                "Model_Structure": (
                    model_structure
                ),
                "Process": (
                    process
                ),
                "Residual_Column": (
                    residual_column
                ),
                "Analysis_Rows": (
                    analysis_rows
                ),
                **result,
                "Dependence_Category": (
                    classify_autocorrelation(
                        result[
                            "Lag1_Spearman"
                        ]
                    )
                ),
            }
        )

    return rows


# =============================================================================
# Annual and country-year calibration
# =============================================================================

def aggregate_calibration(
    data: pd.DataFrame,
    group_columns: Sequence[str],
) -> pd.DataFrame:
    grouped = data.groupby(
        list(
            group_columns
        ),
        dropna=False,
        sort=True,
    )

    result = grouped.agg(
        N=(
            "Observed_Count",
            "size",
        ),
        Observed_Total_Count=(
            "Observed_Count",
            "sum",
        ),
        Predicted_Total_Count=(
            "Predicted_Count",
            "sum",
        ),
        Observed_Event_Proportion=(
            "Observed_Event",
            "mean",
        ),
        Predicted_Event_Probability_Mean=(
            "Predicted_Event_Probability",
            "mean",
        ),
        Occurrence_Residual_Mean=(
            "Occurrence_Residual",
            "mean",
        ),
        Rate_Residual_Mean=(
            "Rate_Residual",
            "mean",
        ),
        Rate_RMSE=(
            "Rate_Squared_Error",
            lambda values: float(
                np.sqrt(
                    np.mean(
                        values
                    )
                )
            ),
        ),
        Rate_MAE=(
            "Rate_Absolute_Error",
            "mean",
        ),
        Mean_Negative_LogLikelihood=(
            "Negative_LogProbability",
            "mean",
        ),
        Occurrence_Brier=(
            "Occurrence_Brier_Contribution",
            "mean",
        ),
    ).reset_index()

    result[
        "Predicted_to_Observed_Total_Count_Ratio"
    ] = np.where(
        result[
            "Observed_Total_Count"
        ]
        > 0,
        result[
            "Predicted_Total_Count"
        ]
        / result[
            "Observed_Total_Count"
        ],
        np.nan,
    )

    result[
        "Event_Probability_Calibration_Difference"
    ] = (
        result[
            "Predicted_Event_Probability_Mean"
        ]
        - result[
            "Observed_Event_Proportion"
        ]
    )

    return result


# =============================================================================
# Cluster bootstrap
# =============================================================================

METRIC_LABELS = [
    "Mean_Negative_LogLikelihood",
    "Occurrence_Brier",
    "Rate_RMSE",
    "Rate_MAE",
    "Predicted_to_Observed_Total_Count_Ratio",
]


def metric_point_estimates(
    data: pd.DataFrame,
) -> Dict[str, float]:
    observed_total = float(
        data[
            "Observed_Count"
        ].sum()
    )

    predicted_total = float(
        data[
            "Predicted_Count"
        ].sum()
    )

    return {
        "Mean_Negative_LogLikelihood": float(
            data[
                "Negative_LogProbability"
            ].mean()
        ),
        "Occurrence_Brier": float(
            data[
                "Occurrence_Brier_Contribution"
            ].mean()
        ),
        "Rate_RMSE": float(
            np.sqrt(
                data[
                    "Rate_Squared_Error"
                ].mean()
            )
        ),
        "Rate_MAE": float(
            data[
                "Rate_Absolute_Error"
            ].mean()
        ),
        "Predicted_to_Observed_Total_Count_Ratio": (
            safe_ratio(
                predicted_total,
                observed_total,
            )
        ),
    }


def iid_standard_errors(
    data: pd.DataFrame,
    point_estimates: Dict[
        str,
        float,
    ],
) -> Dict[str, float]:
    sample_size = int(
        len(data)
    )

    if sample_size <= 1:
        return {
            metric: np.nan
            for metric in METRIC_LABELS
        }

    nll = data[
        "Negative_LogProbability"
    ].to_numpy(
        dtype=float
    )

    brier = data[
        "Occurrence_Brier_Contribution"
    ].to_numpy(
        dtype=float
    )

    squared_error = data[
        "Rate_Squared_Error"
    ].to_numpy(
        dtype=float
    )

    absolute_error = data[
        "Rate_Absolute_Error"
    ].to_numpy(
        dtype=float
    )

    observed = data[
        "Observed_Count"
    ].to_numpy(
        dtype=float
    )

    predicted = data[
        "Predicted_Count"
    ].to_numpy(
        dtype=float
    )

    nll_se = float(
        np.std(
            nll,
            ddof=1,
        )
        / math.sqrt(
            sample_size
        )
    )

    brier_se = float(
        np.std(
            brier,
            ddof=1,
        )
        / math.sqrt(
            sample_size
        )
    )

    mse = float(
        squared_error.mean()
    )

    mse_se = float(
        np.std(
            squared_error,
            ddof=1,
        )
        / math.sqrt(
            sample_size
        )
    )

    rmse = float(
        point_estimates[
            "Rate_RMSE"
        ]
    )

    rmse_se = (
        mse_se
        / (
            2.0
            * rmse
        )
        if rmse > 0
        else np.nan
    )

    mae_se = float(
        np.std(
            absolute_error,
            ddof=1,
        )
        / math.sqrt(
            sample_size
        )
    )

    mean_observed = float(
        observed.mean()
    )

    ratio = float(
        point_estimates[
            "Predicted_to_Observed_Total_Count_Ratio"
        ]
    )

    if (
        mean_observed > 0
        and np.isfinite(
            ratio
        )
    ):
        ratio_influence = (
            predicted
            - ratio
            * observed
        ) / mean_observed

        ratio_se = float(
            np.std(
                ratio_influence,
                ddof=1,
            )
            / math.sqrt(
                sample_size
            )
        )
    else:
        ratio_se = np.nan

    return {
        "Mean_Negative_LogLikelihood": (
            nll_se
        ),
        "Occurrence_Brier": (
            brier_se
        ),
        "Rate_RMSE": (
            rmse_se
        ),
        "Rate_MAE": (
            mae_se
        ),
        "Predicted_to_Observed_Total_Count_Ratio": (
            ratio_se
        ),
    }


def cluster_sufficient_statistics(
    data: pd.DataFrame,
    cluster_column: str,
) -> pd.DataFrame:
    return (
        data.groupby(
            cluster_column,
            sort=False,
        )
        .agg(
            N=(
                "Observed_Count",
                "size",
            ),
            NLL_Sum=(
                "Negative_LogProbability",
                "sum",
            ),
            Brier_Sum=(
                "Occurrence_Brier_Contribution",
                "sum",
            ),
            Rate_SE_Sum=(
                "Rate_Squared_Error",
                "sum",
            ),
            Rate_AE_Sum=(
                "Rate_Absolute_Error",
                "sum",
            ),
            Observed_Count_Sum=(
                "Observed_Count",
                "sum",
            ),
            Predicted_Count_Sum=(
                "Predicted_Count",
                "sum",
            ),
        )
        .reset_index()
    )


def cluster_bootstrap_metrics(
    data: pd.DataFrame,
    cluster_column: str,
    replicate_count: int,
    seed: int,
) -> Tuple[
    Dict[str, np.ndarray],
    int,
]:
    statistics = (
        cluster_sufficient_statistics(
            data=data,
            cluster_column=(
                cluster_column
            ),
        )
    )

    cluster_count = int(
        len(statistics)
    )

    if cluster_count < 2:
        return (
            {
                metric: np.full(
                    replicate_count,
                    np.nan,
                )
                for metric in METRIC_LABELS
            },
            cluster_count,
        )

    values = statistics[
        [
            "N",
            "NLL_Sum",
            "Brier_Sum",
            "Rate_SE_Sum",
            "Rate_AE_Sum",
            "Observed_Count_Sum",
            "Predicted_Count_Sum",
        ]
    ].to_numpy(
        dtype=float
    )

    rng = np.random.default_rng(
        seed
    )

    results = {
        metric: np.empty(
            replicate_count,
            dtype=float,
        )
        for metric in METRIC_LABELS
    }

    batch_size = 100

    for start in range(
        0,
        replicate_count,
        batch_size,
    ):
        stop = min(
            start
            + batch_size,
            replicate_count,
        )

        current_size = (
            stop - start
        )

        indices = rng.integers(
            low=0,
            high=cluster_count,
            size=(
                current_size,
                cluster_count,
            ),
        )

        sampled = values[
            indices
        ].sum(
            axis=1
        )

        sample_n = sampled[
            :,
            0,
        ]

        nll_sum = sampled[
            :,
            1,
        ]

        brier_sum = sampled[
            :,
            2,
        ]

        rate_se_sum = sampled[
            :,
            3,
        ]

        rate_ae_sum = sampled[
            :,
            4,
        ]

        observed_sum = sampled[
            :,
            5,
        ]

        predicted_sum = sampled[
            :,
            6,
        ]

        results[
            "Mean_Negative_LogLikelihood"
        ][
            start:stop
        ] = (
            nll_sum
            / sample_n
        )

        results[
            "Occurrence_Brier"
        ][
            start:stop
        ] = (
            brier_sum
            / sample_n
        )

        results[
            "Rate_RMSE"
        ][
            start:stop
        ] = np.sqrt(
            rate_se_sum
            / sample_n
        )

        results[
            "Rate_MAE"
        ][
            start:stop
        ] = (
            rate_ae_sum
            / sample_n
        )

        results[
            "Predicted_to_Observed_Total_Count_Ratio"
        ][
            start:stop
        ] = np.where(
            observed_sum > 0,
            predicted_sum
            / observed_sum,
            np.nan,
        )

    return (
        results,
        cluster_count,
    )


def bootstrap_rows(
    data: pd.DataFrame,
    season_code: int,
    response: str,
    model_structure: str,
) -> List[Dict[str, object]]:
    point = metric_point_estimates(
        data
    )

    iid_se = iid_standard_errors(
        data=data,
        point_estimates=(
            point
        ),
    )

    rows: List[
        Dict[str, object]
    ] = []

    cluster_definitions = [
        (
            "GRID_UID",
            "GRID_UID",
            RANDOM_SEED
            + season_code
            * 100
            + (
                1
                if response
                == "Fire_Count"
                else 2
            ),
        ),
        (
            "Year",
            "Year",
            RANDOM_SEED
            + season_code
            * 1000
            + (
                11
                if response
                == "Fire_Count"
                else 12
            ),
        ),
    ]

    for (
        cluster_label,
        cluster_column,
        seed,
    ) in cluster_definitions:
        (
            distributions,
            cluster_count,
        ) = cluster_bootstrap_metrics(
            data=data,
            cluster_column=(
                cluster_column
            ),
            replicate_count=(
                CLUSTER_BOOTSTRAP_REPLICATES
            ),
            seed=seed,
        )

        for metric in METRIC_LABELS:
            values = distributions[
                metric
            ]

            finite = values[
                np.isfinite(
                    values
                )
            ]

            if len(finite) > 1:
                cluster_se = float(
                    np.std(
                        finite,
                        ddof=1,
                    )
                )

                lower = float(
                    np.quantile(
                        finite,
                        0.025,
                    )
                )

                upper = float(
                    np.quantile(
                        finite,
                        0.975,
                    )
                )
            else:
                cluster_se = np.nan
                lower = np.nan
                upper = np.nan

            iid_metric_se = float(
                iid_se[
                    metric
                ]
            )

            inflation = (
                cluster_se
                / iid_metric_se
                if (
                    np.isfinite(
                        cluster_se
                    )
                    and np.isfinite(
                        iid_metric_se
                    )
                    and iid_metric_se > 0
                )
                else np.nan
            )

            rows.append(
                {
                    "Season_Code": (
                        season_code
                    ),
                    "Season_Label": (
                        SEASONS[
                            season_code
                        ]
                    ),
                    "Count_Response": (
                        response
                    ),
                    "Rate_Scale_Name": (
                        COUNT_RESPONSES[
                            response
                        ]
                    ),
                    "Model_Structure": (
                        model_structure
                    ),
                    "Cluster_Level": (
                        cluster_label
                    ),
                    "Cluster_Count": (
                        cluster_count
                    ),
                    "Bootstrap_Replicates": (
                        CLUSTER_BOOTSTRAP_REPLICATES
                    ),
                    "Metric": (
                        metric
                    ),
                    "Point_Estimate": (
                        point[
                            metric
                        ]
                    ),
                    "IID_Approximate_SE": (
                        iid_metric_se
                    ),
                    "Cluster_Bootstrap_SE": (
                        cluster_se
                    ),
                    "Cluster_to_IID_SE_Ratio": (
                        inflation
                    ),
                    "Bootstrap_CI_Lower_2.5pct": (
                        lower
                    ),
                    "Bootstrap_CI_Upper_97.5pct": (
                        upper
                    ),
                    "Small_Cluster_Count_Caution": bool(
                        cluster_count
                        < YEAR_CLUSTER_CAUTION_THRESHOLD
                    ),
                }
            )

    return rows


# =============================================================================
# Clustered-inference recommendation
# =============================================================================

def maximum_cluster_inflation(
    bootstrap: pd.DataFrame,
    season_code: int,
    response: str,
    metric: str,
    cluster_level: str,
) -> float:
    subset = bootstrap.loc[
        (
            bootstrap[
                "Season_Code"
            ]
            == season_code
        )
        & (
            bootstrap[
                "Count_Response"
            ]
            == response
        )
        & (
            bootstrap[
                "Metric"
            ]
            == metric
        )
        & (
            bootstrap[
                "Cluster_Level"
            ]
            == cluster_level
        ),
        "Cluster_to_IID_SE_Ratio",
    ]

    if subset.empty:
        return np.nan

    return finite_or_nan(
        subset.iloc[0]
    )


def extract_icc(
    icc: pd.DataFrame,
    season_code: int,
    response: str,
    process: str,
    grouping_level: str,
) -> float:
    subset = icc.loc[
        (
            icc[
                "Season_Code"
            ]
            == season_code
        )
        & (
            icc[
                "Count_Response"
            ]
            == response
        )
        & (
            icc[
                "Process"
            ]
            == process
        )
        & (
            icc[
                "Grouping_Level"
            ]
            == grouping_level
        ),
        "ICC_Truncated",
    ]

    if subset.empty:
        return np.nan

    return finite_or_nan(
        subset.iloc[0]
    )


def extract_lag1(
    lag1: pd.DataFrame,
    season_code: int,
    response: str,
    process: str,
) -> float:
    subset = lag1.loc[
        (
            lag1[
                "Season_Code"
            ]
            == season_code
        )
        & (
            lag1[
                "Count_Response"
            ]
            == response
        )
        & (
            lag1[
                "Process"
            ]
            == process
        ),
        "Lag1_Spearman",
    ]

    if subset.empty:
        return np.nan

    return finite_or_nan(
        subset.iloc[0]
    )


def process_signal(
    grid_icc: float,
    year_icc: float,
    lag1_value: float,
    grid_inflation: float,
    year_inflation: float,
) -> Dict[str, object]:
    grid_trigger = bool(
        (
            np.isfinite(
                grid_icc
            )
            and grid_icc
            >= ICC_TEST_TRIGGER
        )
        or (
            np.isfinite(
                lag1_value
            )
            and abs(
                lag1_value
            )
            >= ABS_LAG1_TEST_TRIGGER
        )
        or (
            np.isfinite(
                grid_inflation
            )
            and grid_inflation
            >= CLUSTER_SE_INFLATION_TEST_TRIGGER
        )
    )

    year_trigger = bool(
        (
            np.isfinite(
                year_icc
            )
            and year_icc
            >= ICC_TEST_TRIGGER
        )
        or (
            np.isfinite(
                year_inflation
            )
            and year_inflation
            >= CLUSTER_SE_INFLATION_TEST_TRIGGER
        )
    )

    if grid_trigger and year_trigger:
        signal = (
            "Test crossed GRID_UID and Year dependence"
        )
    elif grid_trigger:
        signal = (
            "Test GRID_UID dependence"
        )
    elif year_trigger:
        signal = (
            "Test Year dependence"
        )
    else:
        signal = (
            "No strong trigger beyond clustered-inference baseline"
        )

    return {
        "GRID_UID_Test_Trigger": (
            grid_trigger
        ),
        "Year_Test_Trigger": (
            year_trigger
        ),
        "Dependence_Signal": (
            signal
        ),
    }


def dependence_decision_table(
    structure: pd.DataFrame,
    icc: pd.DataFrame,
    lag1: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> pd.DataFrame:
    decision_rows: List[
        Dict[str, object]
    ] = []

    for definition in structure.itertuples(
        index=False
    ):
        season_code = int(
            definition.Season_Code
        )

        response = str(
            definition.Count_Response
        )

        model_structure = str(
            definition.Model_Structure
        )

        occurrence_grid_icc = extract_icc(
            icc,
            season_code,
            response,
            "Occurrence",
            "GRID_UID",
        )

        occurrence_year_icc = extract_icc(
            icc,
            season_code,
            response,
            "Occurrence",
            "Year",
        )

        occurrence_lag1 = extract_lag1(
            lag1,
            season_code,
            response,
            "Occurrence",
        )

        occurrence_grid_inflation = (
            maximum_cluster_inflation(
                bootstrap,
                season_code,
                response,
                "Occurrence_Brier",
                "GRID_UID",
            )
        )

        occurrence_year_inflation = (
            maximum_cluster_inflation(
                bootstrap,
                season_code,
                response,
                "Occurrence_Brier",
                "Year",
            )
        )

        count_grid_icc = extract_icc(
            icc,
            season_code,
            response,
            "Unconditional_Rate",
            "GRID_UID",
        )

        count_year_icc = extract_icc(
            icc,
            season_code,
            response,
            "Unconditional_Rate",
            "Year",
        )

        count_lag1 = extract_lag1(
            lag1,
            season_code,
            response,
            "Unconditional_Rate",
        )

        count_grid_inflation = (
            maximum_cluster_inflation(
                bootstrap,
                season_code,
                response,
                "Rate_RMSE",
                "GRID_UID",
            )
        )

        count_year_inflation = (
            maximum_cluster_inflation(
                bootstrap,
                season_code,
                response,
                "Rate_RMSE",
                "Year",
            )
        )

        positive_grid_icc = extract_icc(
            icc,
            season_code,
            response,
            "Positive_Magnitude",
            "GRID_UID",
        )

        positive_year_icc = extract_icc(
            icc,
            season_code,
            response,
            "Positive_Magnitude",
            "Year",
        )

        positive_lag1 = extract_lag1(
            lag1,
            season_code,
            response,
            "Positive_Magnitude",
        )

        occurrence_signal = process_signal(
            grid_icc=(
                occurrence_grid_icc
            ),
            year_icc=(
                occurrence_year_icc
            ),
            lag1_value=(
                occurrence_lag1
            ),
            grid_inflation=(
                occurrence_grid_inflation
            ),
            year_inflation=(
                occurrence_year_inflation
            ),
        )

        count_signal = process_signal(
            grid_icc=max(
                [
                    value
                    for value in [
                        count_grid_icc,
                        positive_grid_icc,
                    ]
                    if np.isfinite(
                        value
                    )
                ],
                default=np.nan,
            ),
            year_icc=max(
                [
                    value
                    for value in [
                        count_year_icc,
                        positive_year_icc,
                    ]
                    if np.isfinite(
                        value
                    )
                ],
                default=np.nan,
            ),
            lag1_value=max(
                [
                    abs(
                        value
                    )
                    for value in [
                        count_lag1,
                        positive_lag1,
                    ]
                    if np.isfinite(
                        value
                    )
                ],
                default=np.nan,
            ),
            grid_inflation=(
                count_grid_inflation
            ),
            year_inflation=(
                count_year_inflation
            ),
        )

        decision_rows.append(
            {
                "Season_Code": (
                    season_code
                ),
                "Season_Label": (
                    SEASONS[
                        season_code
                    ]
                ),
                "Count_Response": (
                    response
                ),
                "Rate_Scale_Name": (
                    COUNT_RESPONSES[
                        response
                    ]
                ),
                "Model_Structure": (
                    model_structure
                ),
                "Occurrence_GRID_UID_ICC": (
                    occurrence_grid_icc
                ),
                "Occurrence_Year_ICC": (
                    occurrence_year_icc
                ),
                "Occurrence_Lag1_Spearman": (
                    occurrence_lag1
                ),
                "Occurrence_GRID_UID_Brier_SE_Inflation": (
                    occurrence_grid_inflation
                ),
                "Occurrence_Year_Brier_SE_Inflation": (
                    occurrence_year_inflation
                ),
                "Occurrence_GRID_UID_Test_Trigger": (
                    occurrence_signal[
                        "GRID_UID_Test_Trigger"
                    ]
                ),
                "Occurrence_Year_Test_Trigger": (
                    occurrence_signal[
                        "Year_Test_Trigger"
                    ]
                ),
                "Occurrence_Dependence_Signal": (
                    occurrence_signal[
                        "Dependence_Signal"
                    ]
                ),
                "Rate_GRID_UID_ICC": (
                    count_grid_icc
                ),
                "Rate_Year_ICC": (
                    count_year_icc
                ),
                "Rate_Lag1_Spearman": (
                    count_lag1
                ),
                "Positive_Magnitude_GRID_UID_ICC": (
                    positive_grid_icc
                ),
                "Positive_Magnitude_Year_ICC": (
                    positive_year_icc
                ),
                "Positive_Magnitude_Lag1_Spearman": (
                    positive_lag1
                ),
                "Count_GRID_UID_RMSE_SE_Inflation": (
                    count_grid_inflation
                ),
                "Count_Year_RMSE_SE_Inflation": (
                    count_year_inflation
                ),
                "Count_GRID_UID_Test_Trigger": (
                    count_signal[
                        "GRID_UID_Test_Trigger"
                    ]
                ),
                "Count_Year_Test_Trigger": (
                    count_signal[
                        "Year_Test_Trigger"
                    ]
                ),
                "Count_Dependence_Signal": (
                    count_signal[
                        "Dependence_Signal"
                    ]
                ),
                "Interpretation": (
                    "Dependence signals justify clustered covariance "
                    "but do not alter the selected mean structure."
                ),
            }
        )


    return pd.DataFrame(
        decision_rows
    )


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    if OUTPUT_ROOT.exists():
        shutil.rmtree(
            OUTPUT_ROOT
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=False,
    )

    log(
        "Starting panel-dependence and clustered-inference audit."
    )

    log(
        f"Source: {SOURCE_ROOT}"
    )

    log(
        f"Output: {OUTPUT_ROOT}"
    )

    structure = read_final_structure()

    qa_rows: List[
        Dict[str, object]
    ] = []

    manifest_rows: List[
        Dict[str, object]
    ] = []

    pooled_rows: List[
        Dict[str, object]
    ] = []

    icc_rows: List[
        Dict[str, object]
    ] = []

    lag_rows: List[
        Dict[str, object]
    ] = []

    annual_tables: List[
        pd.DataFrame
    ] = []

    country_year_tables: List[
        pd.DataFrame
    ] = []

    bootstrap_result_rows: List[
        Dict[str, object]
    ] = []

    for definition in structure.itertuples(
        index=False
    ):
        season_code = int(
            definition.Season_Code
        )

        response = str(
            definition.Count_Response
        )

        model_structure = str(
            definition.Model_Structure
        )

        source_path = prediction_path(
            season_code=(
                season_code
            ),
            response=response,
        )

        if not source_path.exists():
            raise FileNotFoundError(
                "Step-05 selected OOF prediction file was not found:\n"
                f"{source_path}"
            )

        log(
            f"Reading {source_path.name}."
        )

        data = pd.read_csv(
            source_path,
            compression="gzip",
        )

        validate_prediction_data(
            data=data,
            season_code=(
                season_code
            ),
            response=response,
            expected_structure=(
                model_structure
            ),
            source_path=(
                source_path
            ),
        )

        data = add_process_residuals(
            data
        )

        qa_rows.append(
            {
                "Season_Code": (
                    season_code
                ),
                "Season_Label": (
                    SEASONS[
                        season_code
                    ]
                ),
                "Count_Response": (
                    response
                ),
                "Rate_Scale_Name": (
                    COUNT_RESPONSES[
                        response
                    ]
                ),
                "Model_Structure": (
                    model_structure
                ),
                "Rows": (
                    len(data)
                ),
                "Unique_GRID_UIDs": (
                    data[
                        "GRID_UID"
                    ].nunique()
                ),
                "Unique_Years": (
                    data[
                        "Year"
                    ].nunique()
                ),
                "Unique_Countries": (
                    data[
                        "Country"
                    ].nunique()
                ),
                "Positive_Row_Count": int(
                    (
                        data[
                            "Observed_Count"
                        ]
                        > 0
                    ).sum()
                ),
                "Observed_Zero_Proportion": float(
                    (
                        data[
                            "Observed_Count"
                        ]
                        == 0
                    ).mean()
                ),
            }
        )

        manifest_rows.append(
            {
                "Season_Code": (
                    season_code
                ),
                "Season_Label": (
                    SEASONS[
                        season_code
                    ]
                ),
                "Count_Response": (
                    response
                ),
                "Model_Structure": (
                    model_structure
                ),
                "Final_Candidate_ID": (
                    definition.Candidate_ID
                ),
                "Source_Prediction_File": (
                    source_path.name
                ),
                "Source_Size_Bytes": (
                    source_path.stat().st_size
                ),
                "Rows": (
                    len(data)
                ),
            }
        )

        pooled_rows.extend(
            pooled_residual_rows(
                data=data,
                season_code=(
                    season_code
                ),
                response=response,
                model_structure=(
                    model_structure
                ),
            )
        )

        icc_rows.extend(
            dependence_icc_rows(
                data=data,
                season_code=(
                    season_code
                ),
                response=response,
                model_structure=(
                    model_structure
                ),
            )
        )

        lag_rows.extend(
            lag1_rows(
                data=data,
                season_code=(
                    season_code
                ),
                response=response,
                model_structure=(
                    model_structure
                ),
            )
        )

        annual = aggregate_calibration(
            data=data,
            group_columns=[
                "Year",
            ],
        )

        annual.insert(
            0,
            "Model_Structure",
            model_structure,
        )

        annual.insert(
            0,
            "Rate_Scale_Name",
            COUNT_RESPONSES[
                response
            ],
        )

        annual.insert(
            0,
            "Count_Response",
            response,
        )

        annual.insert(
            0,
            "Season_Label",
            SEASONS[
                season_code
            ],
        )

        annual.insert(
            0,
            "Season_Code",
            season_code,
        )

        annual_tables.append(
            annual
        )

        country_year = aggregate_calibration(
            data=data,
            group_columns=[
                "Country",
                "Year",
            ],
        )

        country_year.insert(
            0,
            "Model_Structure",
            model_structure,
        )

        country_year.insert(
            0,
            "Rate_Scale_Name",
            COUNT_RESPONSES[
                response
            ],
        )

        country_year.insert(
            0,
            "Count_Response",
            response,
        )

        country_year.insert(
            0,
            "Season_Label",
            SEASONS[
                season_code
            ],
        )

        country_year.insert(
            0,
            "Season_Code",
            season_code,
        )

        country_year_tables.append(
            country_year
        )

        log(
            f"Running cluster bootstrap: "
            f"{SEASONS[season_code]} {response}."
        )

        bootstrap_result_rows.extend(
            bootstrap_rows(
                data=data,
                season_code=(
                    season_code
                ),
                response=response,
                model_structure=(
                    model_structure
                ),
            )
        )

        log(
            f"Completed audit: "
            f"{SEASONS[season_code]} {response}; "
            f"{len(data):,} rows."
        )

    qa = pd.DataFrame(
        qa_rows
    )

    manifest = pd.DataFrame(
        manifest_rows
    )

    pooled = pd.DataFrame(
        pooled_rows
    )

    icc = pd.DataFrame(
        icc_rows
    )

    lag1 = pd.DataFrame(
        lag_rows
    )

    annual = pd.concat(
        annual_tables,
        ignore_index=True,
    )

    country_year = pd.concat(
        country_year_tables,
        ignore_index=True,
    )

    bootstrap = pd.DataFrame(
        bootstrap_result_rows
    )

    decisions = dependence_decision_table(
        structure=structure,
        icc=icc,
        lag1=lag1,
        bootstrap=bootstrap,
    )

    residual_definition = pd.DataFrame(
        [
            {
                "Process": (
                    "Occurrence"
                ),
                "Residual": (
                    "I(Observed_Count > 0) "
                    "- (1 - Predicted_Zero_Probability)"
                ),
                "Rows": (
                    "All validation rows"
                ),
                "Model_Component": (
                    "Structural-zero / occurrence component"
                ),
            },
            {
                "Process": (
                    "Unconditional_Rate"
                ),
                "Residual": (
                    "Observed_Rate - Predicted_Rate"
                ),
                "Rows": (
                    "All validation rows"
                ),
                "Model_Component": (
                    "Overall expected-count/rate component"
                ),
            },
            {
                "Process": (
                    "Positive_Magnitude"
                ),
                "Residual": (
                    "log1p(Observed_Count) - "
                    "log1p(Predicted_Count / P(Y > 0))"
                ),
                "Rows": (
                    "Observed positive validation rows only"
                ),
                "Model_Component": (
                    "Positive fire-magnitude component"
                ),
            },
        ]
    )

    rules = pd.DataFrame(
        [
            {
                "Rule_ID": "D01",
                "Quantity": "One-way ICC",
                "Trigger": f"ICC >= {ICC_TEST_TRIGGER}",
                "Purpose": (
                    "Identify meaningful GRID_UID or Year residual clustering."
                ),
            },
            {
                "Rule_ID": "D02",
                "Quantity": "Within-grid lag-one Spearman correlation",
                "Trigger": f"|rho| >= {ABS_LAG1_TEST_TRIGGER}",
                "Purpose": (
                    "Identify persistent within-grid temporal dependence."
                ),
            },
            {
                "Rule_ID": "D03",
                "Quantity": "Cluster-to-IID standard-error ratio",
                "Trigger": f"ratio >= {CLUSTER_SE_INFLATION_TEST_TRIGGER}",
                "Purpose": (
                    "Identify material uncertainty inflation under GRID_UID "
                    "or Year cluster resampling."
                ),
            },
            {
                "Rule_ID": "D04",
                "Quantity": "Final covariance structure",
                "Trigger": "Repeated GRID_UID observations and shared annual shocks",
                "Purpose": (
                    "Use GRID_UID-Year two-way cluster-robust covariance in "
                    "the final NB1/ZINB1 inference."
                ),
            },
        ]
    )

    global_qa = pd.DataFrame(
        [
            (
                "Expected_model_count",
                6,
            ),
            (
                "Observed_model_count",
                len(qa),
            ),
            (
                "All_prediction_files_validated",
                len(qa) == 6,
            ),
            (
                "Total_prediction_rows",
                int(
                    qa[
                        "Rows"
                    ].sum()
                ),
            ),
            (
                "ICC_result_row_count",
                len(icc),
            ),
            (
                "Lag1_result_row_count",
                len(lag1),
            ),
            (
                "Cluster_bootstrap_result_row_count",
                len(bootstrap),
            ),
            (
                "Cluster_bootstrap_replicates_per_level",
                CLUSTER_BOOTSTRAP_REPLICATES,
            ),
            (
                "Final_covariance_structure",
                "GRID_UID-Year two-way cluster-robust",
            ),
            (
                "Fixed_effect_structures_changed",
                False,
            ),
        ],
        columns=[
            "Metric",
            "Value",
        ],
    )

    method = {
        "workflow_stage":
            "Panel-dependence and clustered-inference audit",
        "source_final_structure":
            str(FINAL_STRUCTURE_CSV),
        "source_prediction_directory":
            str(SOURCE_ROOT),
        "selected_mean_structures_changed":
            False,
        "residual_processes": {
            "occurrence":
                "I(Y > 0) - P(Y > 0)",
            "unconditional_rate":
                "Observed_Rate - Predicted_Rate",
            "positive_magnitude":
                "log1p(observed positive count) - "
                "log1p(predicted count conditional on Y > 0)",
        },
        "dependence_diagnostics": [
            "Unbalanced one-way ICC by GRID_UID",
            "Unbalanced one-way ICC by Year",
            "Within-GRID_UID one-year lag residual correlation",
            "Annual and Country-Year OOF calibration",
            "GRID_UID cluster bootstrap",
            "Year cluster bootstrap",
        ],
        "cluster_bootstrap_replicates":
            CLUSTER_BOOTSTRAP_REPLICATES,
        "decision_thresholds": {
            "icc":
                ICC_TEST_TRIGGER,
            "absolute_lag1_spearman":
                ABS_LAG1_TEST_TRIGGER,
            "cluster_to_iid_se_ratio":
                CLUSTER_SE_INFLATION_TEST_TRIGGER,
        },
        "year_cluster_caution":
            "Only 25 Year clusters are available; final inference uses "
            "small-sample caution and 24 degrees of freedom.",
        "final_inference_recommendation":
            "Retain the selected NB1/ZINB1 mean structures and use "
            "GRID_UID-Year two-way cluster-robust covariance.",
        "next_step":
            "Fit the selected mean structures on the full data and calculate "
            "two-way cluster-robust inference.",
    }

    environment = {
        "python": (
            sys.version
        ),
        "platform": (
            platform.platform()
        ),
        "numpy": (
            np.__version__
        ),
        "pandas": (
            pd.__version__
        ),
        "scipy": (
            scipy.__version__
        ),
        "run_timestamp": (
            datetime.now().isoformat(
                timespec="seconds"
            )
        ),
    }

    with (
        OUTPUT_ROOT
        / "00_Method_Definition.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            method,
            handle,
            indent=2,
        )

    global_qa.to_csv(
        OUTPUT_ROOT
        / "01_Global_QA_Summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    qa.to_csv(
        OUTPUT_ROOT
        / "02_Model_Data_QA.csv",
        index=False,
        encoding="utf-8-sig",
    )

    manifest.to_csv(
        OUTPUT_ROOT
        / "03_Source_Prediction_Manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )

    residual_definition.to_csv(
        OUTPUT_ROOT
        / "04_Residual_Process_Definitions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pooled.to_csv(
        OUTPUT_ROOT
        / "05_Pooled_Residual_Summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    icc.to_csv(
        OUTPUT_ROOT
        / "06_GRID_UID_and_Year_ICC.csv",
        index=False,
        encoding="utf-8-sig",
    )

    lag1.to_csv(
        OUTPUT_ROOT
        / "07_Within_GRID_UID_Lag1_Dependence.csv",
        index=False,
        encoding="utf-8-sig",
    )

    annual.to_csv(
        OUTPUT_ROOT
        / "08_Annual_OOF_Calibration.csv",
        index=False,
        encoding="utf-8-sig",
    )

    country_year.to_csv(
        OUTPUT_ROOT
        / "09_Country_Year_OOF_Calibration.csv",
        index=False,
        encoding="utf-8-sig",
    )

    bootstrap.to_csv(
        OUTPUT_ROOT
        / "10_Cluster_Bootstrap_Performance_Uncertainty.csv",
        index=False,
        encoding="utf-8-sig",
    )

    decisions.to_csv(
        OUTPUT_ROOT
        / "11_Process_Specific_Dependence_Decisions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    rules.to_csv(
        OUTPUT_ROOT
        / "12_Dependence_Decision_Rules.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with (
        OUTPUT_ROOT
        / "Software_Environment.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            environment,
            handle,
            indent=2,
        )

    log(
        "Panel-dependence and clustered-inference audit completed successfully."
    )

    for row in decisions.itertuples(
        index=False
    ):
        log(
            f"Decision: {row.Season_Label} "
            f"{row.Count_Response}; "
            f"occurrence={row.Occurrence_Dependence_Signal}; "
            f"count={row.Count_Dependence_Signal}."
        )

    log(
        "Next step: fit the selected mean structures on the full data and "
        "calculate GRID_UID-Year two-way cluster-robust inference."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
