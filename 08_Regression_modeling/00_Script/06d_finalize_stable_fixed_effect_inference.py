# -*- coding: utf-8 -*-
"""
Step 06d: finalize stable fixed-effect inference structures.

Recommended public location
---------------------------
<CODE_ROOT>/08_Regression_modeling/00_Script/
    06d_finalize_stable_fixed_effect_inference.py

Purpose
-------
This step reuses all Step-06c fits, folds, and checkpoints. It does not
repeat the 15 existing candidate searches.

It:

1. applies a scale-independent standardized-score convergence rule;
2. allows a clearly documented mild non-positive-semidefinite correction
   for multiway clustered covariance;
3. adds only the missing Summer Fire_Count structural-zero country
   simplification candidates;
4. confirms those new candidates with the same five temporal folds;
5. calculates paired fold-level NLL differences and a one-standard-error
   diagnostic;
6. selects the smallest identifiable simplification when the Step-05
   baseline is not identifiable;
7. performs a limited Newton-style likelihood polishing of the six selected
   full-data fits;
8. writes the final six-row fixed-effect inference structure table.

The selected fixed mean structures are finalized here.

Selection priority
-------------------
1. Retain the Step-05 baseline when it is identifiable and numerically stable.
2. When the baseline is not identifiable, select among stable alternatives.
3. Prefer the fewest removed terms.
4. Use temporal OOF NLL and paired one-standard-error results as secondary
   predictive diagnostics.
5. Never retain a rank-deficient, boundary, or extreme-parameter model only
   because its OOF NLL is slightly lower.

The Step-05 OOF predictions remain the predictive reference unless a selected
inference simplification was explicitly re-fitted under the same temporal
folds.
"""

from __future__ import annotations

import importlib.util
import json
import math
import platform
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy
import sklearn
import statsmodels
from scipy import stats


# =============================================================================
# Repository-relative paths
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REGRESSION_ROOT = SCRIPT_DIR.parent

STEP04_SCRIPT = (
    SCRIPT_DIR
    / "04_compare_nb1_hurdle_zinb.py"
)

STEP05_SCRIPT = (
    SCRIPT_DIR
    / "05_refine_component_specific_predictors.py"
)

STEP06B_SCRIPT = (
    SCRIPT_DIR
    / "06b_fit_full_data_models_and_two_way_cluster_inference.py"
)

STEP06C_SCRIPT = (
    SCRIPT_DIR
    / "06c_audit_parameter_identifiability_and_stabilize_inference.py"
)

STEP05_ROOT = (
    REGRESSION_ROOT
    / "05_Component_Specific_Predictor_Refinement"
)

STEP06B_ROOT = (
    REGRESSION_ROOT
    / "06b_Full_Data_Fixed_Model_and_Two_Way_Cluster_Inference"
)

STEP06C_ROOT = (
    REGRESSION_ROOT
    / "06c_Parameter_Identifiability_and_Stable_Likelihood_Audit"
)

STEP06C_CANDIDATES = (
    STEP06C_ROOT
    / "04_Candidate_Definitions.csv"
)

STEP06C_FULL_STATUS = (
    STEP06C_ROOT
    / "05_Strict_Full_Data_Fit_and_Stability.csv"
)

STEP06C_COEFFICIENTS = (
    STEP06C_ROOT
    / "06_Candidate_Coefficients_and_Two_Way_Inference.csv"
)

STEP06C_FOLD_STATUS = (
    STEP06C_ROOT
    / "08_Targeted_OOF_Fold_Fit_Status.csv"
)

STEP06C_OOF_METRICS = (
    STEP06C_ROOT
    / "09_Targeted_OOF_Performance.csv"
)

STEP06C_FULL_CHECKPOINT_ROOT = (
    STEP06C_ROOT
    / "Checkpoints"
    / "Full_Data"
)

STEP06C_OOF_CHECKPOINT_ROOT = (
    STEP06C_ROOT
    / "Checkpoints"
    / "OOF_Folds"
)

STEP06B_COEFFICIENTS = (
    STEP06B_ROOT
    / "06_All_Coefficients_and_Clustered_Inference.csv"
)

OUTPUT_ROOT = (
    REGRESSION_ROOT
    / "06d_Final_Stable_Fixed_Effect_Inference"
)

CHECKPOINT_ROOT = (
    OUTPUT_ROOT
    / "Checkpoints"
)

NEW_FULL_CHECKPOINT_ROOT = (
    CHECKPOINT_ROOT
    / "New_Summer_FC_Full_Data"
)

NEW_OOF_CHECKPOINT_ROOT = (
    CHECKPOINT_ROOT
    / "New_Summer_FC_OOF_Folds"
)

CANDIDATE_COVARIANCE_ROOT = (
    OUTPUT_ROOT
    / "New_Summer_FC_Candidate_Covariance_Matrices"
)

FINAL_COVARIANCE_ROOT = (
    OUTPUT_ROOT
    / "Final_Selected_Covariance_Matrices"
)

SELECTED_OOF_ROOT = (
    OUTPUT_ROOT
    / "Final_Selected_Inference_OOF_Predictions"
)

POLISHED_PARAMETER_ROOT = (
    OUTPUT_ROOT
    / "Final_Selected_Polished_Parameters"
)

LOG_FILE = (
    OUTPUT_ROOT
    / "finalize_stable_fixed_effect_inference.log"
)


# =============================================================================
# Fixed workflow settings
# =============================================================================

SEASONS = {
    1: "Spring",
    2: "Summer",
    3: "Autumn",
}

RATE_NAMES = {
    "Fire_Count": "FCD",
    "Burned_Pixel_Count": "BAD",
}

FOLDS = [
    1,
    2,
    3,
    4,
    5,
]

# Revised numerical rules.
STANDARDIZED_SCORE_MAX = 5e-5
MAX_INFORMATION_CONDITION = 1e12
EXTREME_COEFFICIENT_ABS = 25.0

# Mild multiway-cluster covariance correction rule.
MAX_MILD_NON_PSD_RATIO = 1e-3
PSD_EIGENVALUE_FLOOR_RATIO = 1e-12

# Newton polishing.
MAX_NEWTON_ITERATIONS = 4
MAX_NEWTON_COMPONENT_STEP = 1.0
MIN_LIKELIHOOD_IMPROVEMENT = 1e-9

# OOF operational checks.
MIN_ACCEPTABLE_TOTAL_RATIO = 0.50
MAX_ACCEPTABLE_TOTAL_RATIO = 2.00


# =============================================================================
# Logging and generic helpers
# =============================================================================

def log(message: str) -> None:
    line = (
        f"[{datetime.now():%Y-%m-%d %H:%M:%S}] "
        f"{message}"
    )

    print(
        line,
        flush=True,
    )

    with LOG_FILE.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(
            line + "\n"
        )


def load_module(
    path: Path,
    module_name: str,
):
    if not path.exists():
        raise FileNotFoundError(
            f"Required script not found:\n{path}"
        )

    specification = (
        importlib.util.spec_from_file_location(
            module_name,
            path,
        )
    )

    if (
        specification is None
        or specification.loader is None
    ):
        raise RuntimeError(
            f"Could not import {path.name}."
        )

    module = (
        importlib.util.module_from_spec(
            specification
        )
    )

    previous = sys.modules.get(
        module_name
    )

    sys.modules[
        module_name
    ] = module

    try:
        specification.loader.exec_module(
            module
        )
    except Exception:
        if previous is None:
            sys.modules.pop(
                module_name,
                None,
            )
        else:
            sys.modules[
                module_name
            ] = previous

        raise

    return module


def require_files(
    paths: Iterable[Path],
) -> None:
    missing = [
        path
        for path in paths
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Required files are missing:\n"
            + "\n".join(
                str(path)
                for path in missing
            )
        )


def parse_list(
    value: object,
) -> List[str]:
    if (
        pd.isna(
            value
        )
        or not str(
            value
        ).strip()
    ):
        return []

    return list(
        dict.fromkeys(
            item.strip()
            for item in str(
                value
            ).split(
                ";"
            )
            if item.strip()
        )
    )


def removed_term_count(
    value: object,
) -> int:
    return len(
        parse_list(
            value
        )
    )


def safe_name(
    value: object,
) -> str:
    return re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        str(
            value
        ),
    )


def finite_float(
    value: object,
) -> float:
    try:
        result = float(
            value
        )
    except (
        TypeError,
        ValueError,
    ):
        return np.nan

    if not np.isfinite(
        result
    ):
        return np.nan

    return result


# =============================================================================
# New Summer Fire_Count candidates
# =============================================================================

def new_summer_fc_candidates(
    existing_candidates: pd.DataFrame,
    step06c: object,
) -> pd.DataFrame:
    baseline = existing_candidates.loc[
        (
            existing_candidates[
                "Season_Code"
            ].astype(int)
            == 2
        )
        & (
            existing_candidates[
                "Count_Response"
            ].astype(str)
            == "Fire_Count"
        )
        & (
            existing_candidates[
                "Candidate_ID"
            ].astype(str)
            == "BASELINE_STRICT"
        )
    ]

    if len(
        baseline
    ) != 1:
        raise ValueError(
            "The Summer Fire_Count baseline candidate "
            "could not be uniquely identified."
        )

    base = baseline.iloc[
        0
    ].to_dict()

    definitions = [
        {
            "Candidate_ID":
                "ZERO_DROP_COUNTRY_NK",
            "Candidate_Type":
                "Targeted_simplification",
            "Targeted_OOF_Required":
                True,
            "Structural_Zero_Country_Terms":
                "Country_Russia",
            "Terms_Removed_from_Step05":
                "Structural_Zero:Country_NK",
            "Scientific_Role":
                "Remove only the Summer FC structural-zero NK contrast.",
        },
        {
            "Candidate_ID":
                "ZERO_DROP_COUNTRY_RUSSIA",
            "Candidate_Type":
                "Targeted_simplification",
            "Targeted_OOF_Required":
                True,
            "Structural_Zero_Country_Terms":
                "Country_NK",
            "Terms_Removed_from_Step05":
                "Structural_Zero:Country_Russia",
            "Scientific_Role":
                "Remove only the Summer FC structural-zero Russia contrast.",
        },
        {
            "Candidate_ID":
                "ZERO_DROP_ALL_COUNTRY",
            "Candidate_Type":
                "Targeted_simplification",
            "Targeted_OOF_Required":
                True,
            "Structural_Zero_Country_Terms":
                "",
            "Terms_Removed_from_Step05":
                (
                    "Structural_Zero:Country_NK;"
                    "Structural_Zero:Country_Russia"
                ),
            "Scientific_Role":
                "Retain Country only in the Summer FC count component.",
        },
    ]

    rows: List[
        Dict[str, object]
    ] = []

    next_order = int(
        existing_candidates.loc[
            (
                existing_candidates[
                    "Season_Code"
                ].astype(int)
                == 2
            )
            & (
                existing_candidates[
                    "Count_Response"
                ].astype(str)
                == "Fire_Count"
            ),
            "Candidate_Order",
        ].max()
    ) + 1

    for offset, updates in enumerate(
        definitions
    ):
        record = dict(
            base
        )

        record.update(
            updates
        )

        record.pop(
            "Candidate_Signature",
            None,
        )

        record.pop(
            "Candidate_Order",
            None,
        )

        record[
            "Candidate_Signature"
        ] = step06c.signature(
            record
        )

        record[
            "Candidate_Order"
        ] = (
            next_order
            + offset
        )

        rows.append(
            record
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# Exact samples
# =============================================================================

def build_exact_samples(
    step06b: object,
    step06c: object,
) -> Tuple[
    pd.DataFrame,
    Dict[
        Tuple[int, str],
        pd.DataFrame,
    ],
    Dict[
        Tuple[int, str],
        pd.DataFrame,
    ],
]:
    structure = (
        step06b.read_final_structure()
    )

    base_data = (
        step06b.read_base_table(
            structure
        )
    )

    samples: Dict[
        Tuple[int, str],
        pd.DataFrame
    ] = {}

    source_oof: Dict[
        Tuple[int, str],
        pd.DataFrame
    ] = {}

    for definition in structure.itertuples(
        index=False
    ):
        season_code = int(
            definition.Season_Code
        )

        response = str(
            definition.Count_Response
        )

        predictors = list(
            dict.fromkeys(
                parse_list(
                    definition.Structural_Zero_Predictors
                )
                + parse_list(
                    definition.Count_Predictors
                )
            )
        )

        model_data, oof = (
            step06b.exact_model_sample(
                base_data,
                season_code,
                response,
                str(
                    definition.Model_Structure
                ),
                predictors,
            )
        )

        model_data = step06c.attach_fold(
            model_data,
            oof,
        )

        samples[
            (
                season_code,
                response,
            )
        ] = model_data

        source_oof[
            (
                season_code,
                response,
            )
        ] = oof

    return (
        structure,
        samples,
        source_oof,
    )


# =============================================================================
# Existing and new OOF predictions
# =============================================================================

def existing_checkpoint_path(
    candidate: pd.Series,
    fold: int,
) -> Path:
    return (
        STEP06C_OOF_CHECKPOINT_ROOT
        / (
            f"S{int(candidate['Season_Code'])}_"
            f"{safe_name(candidate['Count_Response'])}_"
            f"{safe_name(candidate['Candidate_ID'])}_"
            f"{candidate['Candidate_Signature']}_"
            f"Fold{fold}.csv.gz"
        )
    )


def read_existing_candidate_oof(
    candidate: pd.Series,
    source_oof: Mapping[
        Tuple[int, str],
        pd.DataFrame,
    ],
) -> pd.DataFrame:
    season_code = int(
        candidate[
            "Season_Code"
        ]
    )

    response = str(
        candidate[
            "Count_Response"
        ]
    )

    paths = [
        existing_checkpoint_path(
            candidate,
            fold,
        )
        for fold in FOLDS
    ]

    if all(
        path.exists()
        for path in paths
    ):
        result = pd.concat(
            [
                pd.read_csv(
                    path,
                    compression="gzip",
                )
                for path in paths
            ],
            ignore_index=True,
        )
    elif (
        str(
            candidate[
                "Candidate_ID"
            ]
        )
        == "BASELINE_STRICT"
    ):
        result = source_oof[
            (
                season_code,
                response,
            )
        ].copy()

        result[
            "Candidate_ID"
        ] = (
            candidate[
                "Candidate_ID"
            ]
        )
    else:
        raise FileNotFoundError(
            "Existing Step-06c OOF checkpoints are missing for:\n"
            f"{candidate['Season_Label']} "
            f"{response} "
            f"{candidate['Candidate_ID']}"
        )

    if len(
        result
    ) == 0:
        raise ValueError(
            "An OOF prediction table is empty."
        )

    return (
        result.sort_values(
            [
                "GRID_UID",
                "Year",
                "Season",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# =============================================================================
# Revised stability assessment
# =============================================================================

def revised_stability_table(
    full_status: pd.DataFrame,
) -> pd.DataFrame:
    result = full_status.copy()

    minimum_eigenvalue = pd.to_numeric(
        result[
            "Two_Way_Minimum_Eigenvalue"
        ],
        errors="coerce",
    )

    maximum_absolute_eigenvalue = pd.to_numeric(
        result[
            "Two_Way_Maximum_Absolute_Eigenvalue"
        ],
        errors="coerce",
    )

    non_psd_ratio = np.where(
        (
            np.isfinite(
                minimum_eigenvalue
            )
            & np.isfinite(
                maximum_absolute_eigenvalue
            )
            & (
                maximum_absolute_eigenvalue
                > 0
            )
            & (
                minimum_eigenvalue
                < 0
            )
        ),
        (
            -minimum_eigenvalue
            / maximum_absolute_eigenvalue
        ),
        0.0,
    )

    result[
        "Two_Way_Non_PSD_Ratio"
    ] = non_psd_ratio

    result[
        "Standardized_Score_Acceptable"
    ] = (
        pd.to_numeric(
            result[
                "Relative_Gradient"
            ],
            errors="coerce",
        )
        <= STANDARDIZED_SCORE_MAX
    )

    result[
        "Information_Identifiable"
    ] = (
        result[
            "Information_Full_Rank"
        ].astype(bool)
        & (
            pd.to_numeric(
                result[
                    "Information_Condition_Number"
                ],
                errors="coerce",
            )
            <= MAX_INFORMATION_CONDITION
        )
    )

    result[
        "Two_Way_Raw_or_Mild_PSD_Correction_Acceptable"
    ] = (
        result[
            "Raw_Two_Way_Covariance_Usable"
        ].astype(bool)
        | (
            result[
                "Two_Way_All_Diagonal_Positive"
            ].astype(bool)
            & (
                result[
                    "Two_Way_Non_PSD_Ratio"
                ]
                <= MAX_MILD_NON_PSD_RATIO
            )
        )
    )

    result[
        "Mild_PSD_Correction_Required"
    ] = (
        ~result[
            "Raw_Two_Way_Covariance_Usable"
        ].astype(bool)
        & result[
            "Two_Way_Raw_or_Mild_PSD_Correction_Acceptable"
        ].astype(bool)
    )

    result[
        "No_Boundary_or_Extreme_Parameter"
    ] = (
        ~result[
            "Parameter_Bound_Hit"
        ].astype(bool)
        & (
            pd.to_numeric(
                result[
                    "Extreme_Coefficient_Count"
                ],
                errors="coerce",
            )
            == 0
        )
        & (
            pd.to_numeric(
                result[
                    "Maximum_Absolute_NonDispersion_Coefficient"
                ],
                errors="coerce",
            )
            < EXTREME_COEFFICIENT_ABS
        )
    )

    result[
        "Revised_Full_Data_Stable"
    ] = (
        result[
            "Converged"
        ].astype(bool)
        & result[
            "Standardized_Score_Acceptable"
        ].astype(bool)
        & result[
            "Information_Identifiable"
        ].astype(bool)
        & result[
            "Two_Way_Raw_or_Mild_PSD_Correction_Acceptable"
        ].astype(bool)
        & result[
            "No_Boundary_or_Extreme_Parameter"
        ].astype(bool)
    )

    result[
        "Revised_Stability_Reason"
    ] = np.where(
        result[
            "Revised_Full_Data_Stable"
        ],
        (
            "Converged; standardized score acceptable; "
            "information matrix identifiable; no boundary/extreme "
            "coefficient; clustered covariance usable directly or "
            "with only a mild PSD sensitivity correction."
        ),
        (
            "At least one identifiability, boundary, standardized-score, "
            "or covariance criterion failed."
        ),
    )

    return result


# =============================================================================
# OOF metrics and paired fold NLL
# =============================================================================

def candidate_metrics(
    step04: object,
    candidate: pd.Series,
    prediction: pd.DataFrame,
    source: str,
) -> Dict[str, object]:
    metrics = step04.prediction_metrics(
        prediction[
            "Observed_Count"
        ].to_numpy(
            dtype=float
        ),
        prediction[
            "Predicted_Count"
        ].to_numpy(
            dtype=float
        ),
        prediction[
            "ForestPixelCount"
        ].to_numpy(
            dtype=float
        ),
        prediction[
            "Predicted_Zero_Probability"
        ].to_numpy(
            dtype=float
        ),
        prediction[
            "Predictive_LogProbability"
        ].to_numpy(
            dtype=float
        ),
    )

    return {
        "Season_Code": (
            candidate[
                "Season_Code"
            ]
        ),
        "Season_Label": (
            candidate[
                "Season_Label"
            ]
        ),
        "Count_Response": (
            candidate[
                "Count_Response"
            ]
        ),
        "Rate_Scale_Name": (
            candidate[
                "Rate_Scale_Name"
            ]
        ),
        "Model_Structure": (
            candidate[
                "Model_Structure"
            ]
        ),
        "Candidate_ID": (
            candidate[
                "Candidate_ID"
            ]
        ),
        **metrics,
        "OOF_Source": (
            source
        ),
    }


def fold_nll_table(
    candidates: pd.DataFrame,
    prediction_cache: Mapping[
        Tuple[int, str, str],
        pd.DataFrame,
    ],
) -> pd.DataFrame:
    rows: List[
        Dict[str, object]
    ] = []

    for candidate in candidates.itertuples(
        index=False
    ):
        key = (
            int(
                candidate.Season_Code
            ),
            str(
                candidate.Count_Response
            ),
            str(
                candidate.Candidate_ID
            ),
        )

        if key not in prediction_cache:
            continue

        prediction = (
            prediction_cache[
                key
            ]
        )

        grouped = prediction.groupby(
            "Temporal_Fold",
            sort=True,
        )

        for fold, subset in grouped:
            rows.append(
                {
                    "Season_Code": (
                        candidate.Season_Code
                    ),
                    "Season_Label": (
                        candidate.Season_Label
                    ),
                    "Count_Response": (
                        candidate.Count_Response
                    ),
                    "Candidate_ID": (
                        candidate.Candidate_ID
                    ),
                    "Temporal_Fold": int(
                        fold
                    ),
                    "Validation_Row_Count": int(
                        len(
                            subset
                        )
                    ),
                    "Fold_Mean_Negative_LogLikelihood": float(
                        -subset[
                            "Predictive_LogProbability"
                        ].mean()
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def paired_fold_comparison(
    fold_nll: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[
        Dict[str, object]
    ] = []

    for (
        season_code,
        response,
    ), group in fold_nll.groupby(
        [
            "Season_Code",
            "Count_Response",
        ]
    ):
        baseline = (
            group.loc[
                group[
                    "Candidate_ID"
                ]
                == "BASELINE_STRICT",
                [
                    "Temporal_Fold",
                    "Fold_Mean_Negative_LogLikelihood",
                ],
            ]
            .rename(
                columns={
                    "Fold_Mean_Negative_LogLikelihood":
                        "Baseline_Fold_NLL"
                }
            )
        )

        if len(
            baseline
        ) != 5:
            raise ValueError(
                f"Baseline fold NLL is incomplete for "
                f"season={season_code}, response={response}."
            )

        for candidate_id, candidate_group in group.groupby(
            "Candidate_ID",
            sort=False,
        ):
            merged = candidate_group.merge(
                baseline,
                on="Temporal_Fold",
                how="inner",
                validate="one_to_one",
            )

            if len(
                merged
            ) != 5:
                raise ValueError(
                    f"Candidate fold NLL is incomplete for "
                    f"{season_code} {response} {candidate_id}."
                )

            difference = (
                merged[
                    "Fold_Mean_Negative_LogLikelihood"
                ]
                - merged[
                    "Baseline_Fold_NLL"
                ]
            ).to_numpy(
                dtype=float
            )

            mean_difference = float(
                difference.mean()
            )

            standard_error = float(
                difference.std(
                    ddof=1
                )
                / math.sqrt(
                    len(
                        difference
                    )
                )
            )

            rows.append(
                {
                    "Season_Code": (
                        season_code
                    ),
                    "Season_Label": (
                        SEASONS[
                            int(
                                season_code
                            )
                        ]
                    ),
                    "Count_Response": (
                        response
                    ),
                    "Candidate_ID": (
                        candidate_id
                    ),
                    "Fold_Count": int(
                        len(
                            difference
                        )
                    ),
                    "Mean_Paired_NLL_Difference_vs_Baseline": (
                        mean_difference
                    ),
                    "SE_Paired_NLL_Difference": (
                        standard_error
                    ),
                    "One_SE_Noninferior": bool(
                        mean_difference
                        <= standard_error
                    ),
                    "Candidate_Better_Fold_Count": int(
                        np.sum(
                            difference
                            < 0
                        )
                    ),
                    "Maximum_Fold_NLL_Deterioration": float(
                        difference.max()
                    ),
                    "Minimum_Fold_NLL_Difference": float(
                        difference.min()
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def fold_fit_validity(
    existing_fold_status: pd.DataFrame,
    new_fold_status: pd.DataFrame,
) -> pd.DataFrame:
    tables = [
        table
        for table in [
            existing_fold_status,
            new_fold_status,
        ]
        if not table.empty
    ]

    if not tables:
        return pd.DataFrame()

    status = pd.concat(
        tables,
        ignore_index=True,
        sort=False,
    )

    rows: List[
        Dict[str, object]
    ] = []

    for (
        season_code,
        response,
        candidate_id,
    ), group in status.groupby(
        [
            "Season_Code",
            "Count_Response",
            "Candidate_ID",
        ]
    ):
        rows.append(
            {
                "Season_Code": (
                    season_code
                ),
                "Count_Response": (
                    response
                ),
                "Candidate_ID": (
                    candidate_id
                ),
                "Fold_Status_Row_Count": int(
                    len(
                        group
                    )
                ),
                "All_Five_Folds_Present": bool(
                    set(
                        group[
                            "Temporal_Fold"
                        ].astype(int)
                    )
                    == set(
                        FOLDS
                    )
                ),
                "All_Folds_Converged": bool(
                    group[
                        "Converged"
                    ].astype(bool).all()
                ),
                "Any_Fold_Parameter_Bound_Hit": bool(
                    group.get(
                        "Parameter_Bound_Hit",
                        pd.Series(
                            False,
                            index=group.index,
                        ),
                    ).astype(bool).any()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# Formal candidate selection
# =============================================================================

def select_candidates(
    candidates: pd.DataFrame,
    stability: pd.DataFrame,
    metrics: pd.DataFrame,
    paired: pd.DataFrame,
    fold_validity: pd.DataFrame,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    key = [
        "Season_Code",
        "Count_Response",
        "Candidate_ID",
    ]

    comparison = candidates.merge(
        stability[
            key
            + [
                "Revised_Full_Data_Stable",
                "Mild_PSD_Correction_Required",
                "Standardized_Score_Acceptable",
                "Information_Identifiable",
                "No_Boundary_or_Extreme_Parameter",
                "Two_Way_Non_PSD_Ratio",
                "Information_Condition_Number",
                "Relative_Gradient",
                "Maximum_Absolute_NonDispersion_Coefficient",
            ]
        ],
        on=key,
        how="left",
        validate="one_to_one",
    )

    comparison = comparison.merge(
        metrics,
        on=key,
        how="left",
        validate="one_to_one",
        suffixes=(
            "",
            "_OOF",
        ),
    )

    # `paired` also contains Season_Label. Merging it directly would cause
    # pandas to rename the existing candidate column to Season_Label_x and
    # Season_Label_y, so later code could no longer access Season_Label.
    paired_for_merge = paired.drop(
        columns=[
            column
            for column in [
                "Season_Label",
            ]
            if column in paired.columns
        ]
    )

    comparison = comparison.merge(
        paired_for_merge,
        on=key,
        how="left",
        validate="one_to_one",
    )

    required_descriptive_columns = [
        "Season_Label",
        "Rate_Scale_Name",
        "Model_Structure",
    ]

    missing_descriptive_columns = [
        column
        for column in required_descriptive_columns
        if column not in comparison.columns
    ]

    if missing_descriptive_columns:
        raise RuntimeError(
            "Candidate comparison lost required descriptive columns after "
            "merging: "
            + ", ".join(
                missing_descriptive_columns
            )
        )

    if not fold_validity.empty:
        comparison = comparison.merge(
            fold_validity,
            on=key,
            how="left",
            validate="one_to_one",
        )
    else:
        comparison[
            "All_Five_Folds_Present"
        ] = np.nan

        comparison[
            "All_Folds_Converged"
        ] = np.nan

        comparison[
            "Any_Fold_Parameter_Bound_Hit"
        ] = np.nan

    comparison[
        "Removed_Term_Count"
    ] = comparison[
        "Terms_Removed_from_Step05"
    ].apply(
        removed_term_count
    )

    total_ratio = pd.to_numeric(
        comparison[
            "Predicted_to_Observed_Total_Count_Ratio"
        ],
        errors="coerce",
    )

    comparison[
        "OOF_Operationally_Valid"
    ] = (
        np.isfinite(
            pd.to_numeric(
                comparison[
                    "Mean_Negative_LogLikelihood"
                ],
                errors="coerce",
            )
        )
        & np.isfinite(
            pd.to_numeric(
                comparison[
                    "Rate_RMSE"
                ],
                errors="coerce",
            )
        )
        & total_ratio.between(
            MIN_ACCEPTABLE_TOTAL_RATIO,
            MAX_ACCEPTABLE_TOTAL_RATIO,
            inclusive="both",
        )
    )

    comparison[
        "Eligible_for_Inference_Selection"
    ] = (
        comparison[
            "Revised_Full_Data_Stable"
        ].astype(bool)
        & comparison[
            "OOF_Operationally_Valid"
        ].astype(bool)
    )

    comparison[
        "Selected"
    ] = False

    comparison[
        "Selection_Status"
    ] = (
        "Not selected"
    )

    selection_rows: List[
        Dict[str, object]
    ] = []

    for (
        season_code,
        response,
    ), indices in comparison.groupby(
        [
            "Season_Code",
            "Count_Response",
        ]
    ).groups.items():
        group = comparison.loc[
            indices
        ].copy()

        baseline = group.loc[
            group[
                "Candidate_ID"
            ]
            == "BASELINE_STRICT"
        ]

        if len(
            baseline
        ) != 1:
            raise ValueError(
                f"Baseline candidate is not unique for "
                f"{season_code} {response}."
            )

        baseline_stable = bool(
            baseline[
                "Revised_Full_Data_Stable"
            ].iloc[
                0
            ]
        )

        if baseline_stable:
            chosen = baseline.iloc[
                0
            ]

            principle = (
                "Retained identifiable Step-05 baseline."
            )
        else:
            eligible = group.loc[
                (
                    group[
                        "Candidate_ID"
                    ]
                    != "BASELINE_STRICT"
                )
                & group[
                    "Eligible_for_Inference_Selection"
                ].astype(bool)
            ].copy()

            if eligible.empty:
                comparison.loc[
                    indices,
                    "Selection_Status",
                ] = (
                    "Unresolved: no identifiable stable alternative"
                )

                selection_rows.append(
                    {
                        "Season_Code": (
                            season_code
                        ),
                        "Season_Label": (
                            SEASONS[
                                int(
                                    season_code
                                )
                            ]
                        ),
                        "Count_Response": (
                            response
                        ),
                        "Resolved": (
                            False
                        ),
                        "Selected_Candidate_ID": (
                            ""
                        ),
                        "Selection_Principle": (
                            "No stable alternative was available."
                        ),
                    }
                )

                continue

            eligible[
                "Targeted_Preference"
            ] = (
                eligible[
                    "Candidate_Type"
                ].astype(str)
                .str.contains(
                    "Diagnostic",
                    case=False,
                    na=False,
                )
                .astype(int)
            )

            eligible[
                "One_SE_Preference"
            ] = (
                ~eligible[
                    "One_SE_Noninferior"
                ].fillna(
                    False
                ).astype(bool)
            ).astype(int)

            chosen = (
                eligible.sort_values(
                    [
                        "Removed_Term_Count",
                        "Targeted_Preference",
                        "One_SE_Preference",
                        "Mean_Negative_LogLikelihood",
                        "Candidate_Order",
                    ]
                )
                .iloc[
                    0
                ]
            )

            principle = (
                "Baseline was not identifiable. Selected the stable "
                "alternative with the fewest removed terms; temporal "
                "OOF and paired one-SE diagnostics were secondary."
            )

        selected_index = chosen.name

        comparison.loc[
            selected_index,
            "Selected",
        ] = True

        comparison.loc[
            selected_index,
            "Selection_Status",
        ] = (
            "Selected final fixed-effect inference structure"
        )

        comparison.loc[
            (
                comparison.index.isin(
                    indices
                )
                & (
                    comparison.index
                    != selected_index
                )
            ),
            "Selection_Status",
        ] = (
            "Not selected after identifiability-first comparison"
        )

        selection_rows.append(
            {
                "Season_Code": (
                    season_code
                ),
                "Season_Label": (
                    SEASONS[
                        int(
                            season_code
                        )
                    ]
                ),
                "Count_Response": (
                    response
                ),
                "Resolved": (
                    True
                ),
                "Selected_Candidate_ID": (
                    chosen[
                        "Candidate_ID"
                    ]
                ),
                "Baseline_Stable": (
                    baseline_stable
                ),
                "Removed_Term_Count": (
                    chosen[
                        "Removed_Term_Count"
                    ]
                ),
                "One_SE_Noninferior": (
                    chosen.get(
                        "One_SE_Noninferior",
                        np.nan,
                    )
                ),
                "Mean_Negative_LogLikelihood": (
                    chosen[
                        "Mean_Negative_LogLikelihood"
                    ]
                ),
                "Mild_PSD_Correction_Required": (
                    chosen[
                        "Mild_PSD_Correction_Required"
                    ]
                ),
                "Selection_Principle": (
                    principle
                ),
            }
        )

    return (
        comparison,
        pd.DataFrame(
            selection_rows
        ),
    )


# =============================================================================
# Selected parameter loading and Newton polishing
# =============================================================================

def checkpoint_path_for_candidate(
    candidate: pd.Series,
    is_new_candidate: bool,
) -> Path:
    root = (
        NEW_FULL_CHECKPOINT_ROOT
        if is_new_candidate
        else STEP06C_FULL_CHECKPOINT_ROOT
    )

    return (
        root
        / (
            f"S{int(candidate['Season_Code'])}_"
            f"{safe_name(candidate['Count_Response'])}_"
            f"{safe_name(candidate['Candidate_ID'])}_"
            f"{candidate['Candidate_Signature']}.npz"
        )
    )


def load_checkpoint_parameters(
    candidate: pd.Series,
    new_candidate_ids: set[
        Tuple[int, str, str]
    ],
) -> np.ndarray:
    key = (
        int(
            candidate[
                "Season_Code"
            ]
        ),
        str(
            candidate[
                "Count_Response"
            ]
        ),
        str(
            candidate[
                "Candidate_ID"
            ]
        ),
    )

    path = checkpoint_path_for_candidate(
        candidate,
        key in new_candidate_ids,
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Selected candidate checkpoint was not found:\n{path}"
        )

    checkpoint = np.load(
        path,
        allow_pickle=False,
    )

    if "theta" not in checkpoint.files:
        raise ValueError(
            f"Checkpoint does not contain theta:\n{path}"
        )

    return np.asarray(
        checkpoint[
            "theta"
        ],
        dtype=float,
    )


def selected_design(
    candidate: pd.Series,
    model_data: pd.DataFrame,
    step06c: object,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[str],
    List[str],
]:
    components = step06c.candidate_components(
        candidate
    )

    count_design, count_names, _ = (
        step06c.full_design(
            model_data,
            components[
                "cp"
            ],
            components[
                "scheme"
            ],
            components[
                "cc"
            ],
        )
    )

    if (
        components[
            "model"
        ]
        == "ZINB1"
    ):
        zero_design, zero_names, _ = (
            step06c.full_design(
                model_data,
                components[
                    "zp"
                ],
                components[
                    "scheme"
                ],
                components[
                    "zc"
                ],
            )
        )
    else:
        zero_design = np.empty(
            (
                len(
                    model_data
                ),
                0,
            ),
            dtype=float,
        )

        zero_names = []

    observed = model_data[
        candidate[
            "Count_Response"
        ]
    ].to_numpy(
        dtype=float
    )

    exposure = model_data[
        "ForestPixelCount"
    ].to_numpy(
        dtype=float
    )

    return (
        observed,
        exposure,
        count_design,
        zero_design,
        count_names,
        zero_names,
    )


def log_likelihood_value(
    step06c: object,
    parameters: np.ndarray,
    observed: np.ndarray,
    count_design: np.ndarray,
    zero_design: np.ndarray,
    exposure: np.ndarray,
    model_structure: str,
) -> float:
    return float(
        step06c.loglike_score_obs(
            parameters,
            observed,
            count_design,
            zero_design,
            exposure,
            model_structure,
        )[
            0
        ].sum()
    )


def standardized_score(
    step06c: object,
    parameters: np.ndarray,
    observed: np.ndarray,
    count_design: np.ndarray,
    zero_design: np.ndarray,
    exposure: np.ndarray,
    model_structure: str,
) -> Tuple[
    float,
    float,
]:
    score_observations = (
        step06c.loglike_score_obs(
            parameters,
            observed,
            count_design,
            zero_design,
            exposure,
            model_structure,
        )[
            1
        ]
    )

    total = score_observations.sum(
        axis=0
    )

    infinity_norm = float(
        np.max(
            np.abs(
                total
            )
        )
    )

    scale = max(
        float(
            np.sqrt(
                np.sum(
                    np.square(
                        score_observations
                    )
                )
            )
        ),
        np.finfo(float).tiny,
    )

    return (
        infinity_norm,
        infinity_norm
        / scale,
    )


def newton_polish(
    step06c: object,
    initial_parameters: np.ndarray,
    observed: np.ndarray,
    count_design: np.ndarray,
    zero_design: np.ndarray,
    exposure: np.ndarray,
    model_structure: str,
) -> Tuple[
    np.ndarray,
    Dict[str, object],
]:
    current = np.asarray(
        initial_parameters,
        dtype=float,
    ).copy()

    initial_log_likelihood = (
        log_likelihood_value(
            step06c,
            current,
            observed,
            count_design,
            zero_design,
            exposure,
            model_structure,
        )
    )

    accepted_steps = 0
    attempted_iterations = 0

    for iteration in range(
        1,
        MAX_NEWTON_ITERATIONS
        + 1,
    ):
        attempted_iterations = iteration

        score = step06c.total_score(
            current,
            observed,
            count_design,
            zero_design,
            exposure,
            model_structure,
        )

        hessian = step06c.numerical_hessian(
            current,
            observed,
            count_design,
            zero_design,
            exposure,
            model_structure,
        )

        information = (
            -(
                hessian
                + hessian.T
            )
            / 2.0
        )

        direction = (
            np.linalg.pinv(
                information,
                rcond=1e-10,
            )
            @ score
        )

        maximum_component = float(
            np.max(
                np.abs(
                    direction
                )
            )
        )

        if (
            not np.isfinite(
                maximum_component
            )
            or maximum_component
            == 0
        ):
            break

        if (
            maximum_component
            > MAX_NEWTON_COMPONENT_STEP
        ):
            direction = (
                direction
                * (
                    MAX_NEWTON_COMPONENT_STEP
                    / maximum_component
                )
            )

        current_log_likelihood = (
            log_likelihood_value(
                step06c,
                current,
                observed,
                count_design,
                zero_design,
                exposure,
                model_structure,
            )
        )

        accepted = False

        for step_size in [
            1.0,
            0.5,
            0.25,
            0.125,
            0.0625,
            0.03125,
        ]:
            candidate = (
                current
                + step_size
                * direction
            )

            candidate[
                :-1
            ] = np.clip(
                candidate[
                    :-1
                ],
                -step06c.COEF_BOUND,
                step06c.COEF_BOUND,
            )

            candidate[
                -1
            ] = np.clip(
                candidate[
                    -1
                ],
                step06c.LOG_ALPHA_BOUNDS[
                    0
                ],
                step06c.LOG_ALPHA_BOUNDS[
                    1
                ],
            )

            candidate_log_likelihood = (
                log_likelihood_value(
                    step06c,
                    candidate,
                    observed,
                    count_design,
                    zero_design,
                    exposure,
                    model_structure,
                )
            )

            if (
                candidate_log_likelihood
                >= (
                    current_log_likelihood
                    + MIN_LIKELIHOOD_IMPROVEMENT
                )
            ):
                current = candidate
                accepted = True
                accepted_steps += 1
                break

        if not accepted:
            break

        _, relative = standardized_score(
            step06c,
            current,
            observed,
            count_design,
            zero_design,
            exposure,
            model_structure,
        )

        if (
            relative
            <= STANDARDIZED_SCORE_MAX
        ):
            break

    final_log_likelihood = (
        log_likelihood_value(
            step06c,
            current,
            observed,
            count_design,
            zero_design,
            exposure,
            model_structure,
        )
    )

    gradient_infinity, relative_gradient = (
        standardized_score(
            step06c,
            current,
            observed,
            count_design,
            zero_design,
            exposure,
            model_structure,
        )
    )

    return (
        current,
        {
            "Newton_Attempted_Iterations": (
                attempted_iterations
            ),
            "Newton_Accepted_Steps": (
                accepted_steps
            ),
            "Initial_LogLikelihood": (
                initial_log_likelihood
            ),
            "Final_LogLikelihood": (
                final_log_likelihood
            ),
            "LogLikelihood_Improvement": (
                final_log_likelihood
                - initial_log_likelihood
            ),
            "Final_Gradient_Infinity_Norm": (
                gradient_infinity
            ),
            "Final_Relative_Gradient": (
                relative_gradient
            ),
        },
    )


def clip_covariance_to_psd(
    covariance: np.ndarray,
) -> Tuple[
    np.ndarray,
    int,
]:
    symmetric = (
        np.asarray(
            covariance,
            dtype=float,
        )
        + np.asarray(
            covariance,
            dtype=float,
        ).T
    ) / 2.0

    eigenvalues, eigenvectors = (
        np.linalg.eigh(
            symmetric
        )
    )

    scale = max(
        float(
            np.max(
                np.abs(
                    eigenvalues
                )
            )
        ),
        1.0,
    )

    floor = (
        scale
        * PSD_EIGENVALUE_FLOOR_RATIO
    )

    clipped = np.maximum(
        eigenvalues,
        floor,
    )

    adjusted_count = int(
        np.sum(
            eigenvalues
            < floor
        )
    )

    adjusted = (
        eigenvectors
        @ np.diag(
            clipped
        )
        @ eigenvectors.T
    )

    return (
        (
            adjusted
            + adjusted.T
        )
        / 2.0,
        adjusted_count,
    )


def covariance_acceptance(
    diagnostics: Mapping[str, object],
) -> Dict[str, object]:
    minimum = finite_float(
        diagnostics[
            "Two_Way_Minimum_Eigenvalue"
        ]
    )

    maximum = finite_float(
        diagnostics[
            "Two_Way_Maximum_Absolute_Eigenvalue"
        ]
    )

    ratio = (
        max(
            0.0,
            -minimum,
        )
        / maximum
        if (
            np.isfinite(
                minimum
            )
            and np.isfinite(
                maximum
            )
            and maximum > 0
        )
        else np.inf
    )

    raw_usable = bool(
        diagnostics[
            "Raw_Two_Way_Covariance_Usable"
        ]
    )

    mild = bool(
        (
            not raw_usable
        )
        and bool(
            diagnostics[
                "Two_Way_All_Diagonal_Positive"
            ]
        )
        and ratio
        <= MAX_MILD_NON_PSD_RATIO
    )

    return {
        "Two_Way_Non_PSD_Ratio": (
            ratio
        ),
        "Raw_Two_Way_Covariance_Usable": (
            raw_usable
        ),
        "Mild_PSD_Correction_Acceptable": (
            mild
        ),
        "Covariance_Acceptable": bool(
            raw_usable
            or mild
        ),
    }


# =============================================================================
# Final selected full-data output
# =============================================================================

def finalize_selected_candidate(
    candidate: pd.Series,
    model_data: pd.DataFrame,
    step06c: object,
    new_candidate_ids: set[
        Tuple[int, str, str]
    ],
) -> Tuple[
    Dict[str, object],
    pd.DataFrame,
]:
    (
        observed,
        exposure,
        count_design,
        zero_design,
        count_names,
        zero_names,
    ) = selected_design(
        candidate,
        model_data,
        step06c,
    )

    initial_parameters = (
        load_checkpoint_parameters(
            candidate,
            new_candidate_ids,
        )
    )

    polished_parameters, polish_status = (
        newton_polish(
            step06c,
            initial_parameters,
            observed,
            count_design,
            zero_design,
            exposure,
            str(
                candidate[
                    "Model_Structure"
                ]
            ),
        )
    )

    covariance_matrices, diagnostics = (
        step06c.covariance_bundle(
            polished_parameters,
            observed,
            count_design,
            zero_design,
            exposure,
            str(
                candidate[
                    "Model_Structure"
                ]
            ),
            model_data[
                "GRID_UID"
            ].astype(str),
            model_data[
                "Year"
            ].astype(int),
        )
    )

    covariance_status = covariance_acceptance(
        diagnostics
    )

    raw_two_way = covariance_matrices[
        "Two_Way_Cluster"
    ]

    psd_two_way, clipped_count = (
        clip_covariance_to_psd(
            raw_two_way
        )
    )

    if covariance_status[
        "Raw_Two_Way_Covariance_Usable"
    ]:
        inference_covariance = (
            raw_two_way
        )

        inference_source = (
            "Raw_GRID_UID_Year_Two_Way_Cluster"
        )
    elif covariance_status[
        "Mild_PSD_Correction_Acceptable"
    ]:
        inference_covariance = (
            psd_two_way
        )

        inference_source = (
            "PSD_Adjusted_GRID_UID_Year_Two_Way_Cluster_Sensitivity"
        )
    else:
        inference_covariance = (
            psd_two_way
        )

        inference_source = (
            "Unresolved_Covariance_PSD_Diagnostic_Only"
        )

    metadata = step06c.metadata(
        str(
            candidate[
                "Model_Structure"
            ]
        ),
        zero_names,
        count_names,
    )

    degrees_of_freedom = int(
        min(
            diagnostics[
                "GRID_UID_Cluster_Count"
            ]
            - 1,
            diagnostics[
                "Year_Cluster_Count"
            ]
            - 1,
        )
    )

    coefficients = step06c.coef_table(
        polished_parameters,
        metadata,
        inference_covariance,
        degrees_of_freedom,
    )

    coefficients.insert(
        0,
        "Inference_Covariance_Source",
        inference_source,
    )

    coefficients.insert(
        0,
        "Candidate_ID",
        candidate[
            "Candidate_ID"
        ],
    )

    coefficients.insert(
        0,
        "Model_Structure",
        candidate[
            "Model_Structure"
        ],
    )

    coefficients.insert(
        0,
        "Rate_Scale_Name",
        candidate[
            "Rate_Scale_Name"
        ],
    )

    coefficients.insert(
        0,
        "Count_Response",
        candidate[
            "Count_Response"
        ],
    )

    coefficients.insert(
        0,
        "Season_Label",
        candidate[
            "Season_Label"
        ],
    )

    coefficients.insert(
        0,
        "Season_Code",
        candidate[
            "Season_Code"
        ],
    )

    non_dispersion = coefficients.loc[
        coefficients[
            "Component"
        ]
        != "Dispersion",
        "Estimate",
    ].to_numpy(
        dtype=float
    )

    extreme_count = int(
        np.sum(
            np.abs(
                non_dispersion
            )
            >= EXTREME_COEFFICIENT_ABS
        )
    )

    final_stable = bool(
        polish_status[
            "Final_Relative_Gradient"
        ]
        <= STANDARDIZED_SCORE_MAX
        and diagnostics[
            "Information_Full_Rank"
        ]
        and diagnostics[
            "Information_Condition_Number"
        ]
        <= MAX_INFORMATION_CONDITION
        and covariance_status[
            "Covariance_Acceptable"
        ]
        and extreme_count
        == 0
        and float(
            np.max(
                np.abs(
                    non_dispersion
                )
            )
        )
        < EXTREME_COEFFICIENT_ABS
    )

    parameter_labels = (
        metadata[
            "Parameter_Label"
        ].astype(str).tolist()
    )

    prefix = (
        f"S{int(candidate['Season_Code'])}_"
        f"{safe_name(candidate['Count_Response'])}_"
        f"{safe_name(candidate['Candidate_ID'])}"
    )

    matrix_bundle = {
        **covariance_matrices,
        "Two_Way_Cluster_PSD_Adjusted": (
            psd_two_way
        ),
        "Inference_Covariance_Used": (
            inference_covariance
        ),
    }

    for name, matrix in (
        matrix_bundle.items()
    ):
        table = pd.DataFrame(
            matrix,
            index=parameter_labels,
            columns=parameter_labels,
        )

        table.index.name = (
            "Parameter"
        )

        table.to_csv(
            FINAL_COVARIANCE_ROOT
            / f"{prefix}_{name}.csv",
            encoding="utf-8-sig",
        )

    np.savez_compressed(
        POLISHED_PARAMETER_ROOT
        / f"{prefix}_Polished_Parameters.npz",
        theta=polished_parameters,
    )

    diagnostics_row = {
        "Season_Code": (
            candidate[
                "Season_Code"
            ]
        ),
        "Season_Label": (
            candidate[
                "Season_Label"
            ]
        ),
        "Count_Response": (
            candidate[
                "Count_Response"
            ]
        ),
        "Rate_Scale_Name": (
            candidate[
                "Rate_Scale_Name"
            ]
        ),
        "Model_Structure": (
            candidate[
                "Model_Structure"
            ]
        ),
        "Candidate_ID": (
            candidate[
                "Candidate_ID"
            ]
        ),
        **polish_status,
        **diagnostics,
        **covariance_status,
        "PSD_Clipped_Eigenvalue_Count": (
            clipped_count
        ),
        "Inference_Covariance_Source": (
            inference_source
        ),
        "Extreme_Coefficient_Count": (
            extreme_count
        ),
        "Maximum_Absolute_NonDispersion_Coefficient": float(
            np.max(
                np.abs(
                    non_dispersion
                )
            )
        ),
        "Final_Selected_Fit_Stable": (
            final_stable
        ),
    }

    return (
        diagnostics_row,
        coefficients,
    )


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    # Preserve the new full-data and OOF checkpoints across reruns.
    for path in [
        OUTPUT_ROOT,
        CHECKPOINT_ROOT,
        NEW_FULL_CHECKPOINT_ROOT,
        NEW_OOF_CHECKPOINT_ROOT,
    ]:
        path.mkdir(
            parents=True,
            exist_ok=True,
        )

    # Rebuild only derived outputs that can be recreated from checkpoints.
    for path in [
        CANDIDATE_COVARIANCE_ROOT,
        FINAL_COVARIANCE_ROOT,
        SELECTED_OOF_ROOT,
        POLISHED_PARAMETER_ROOT,
    ]:
        if path.exists():
            shutil.rmtree(
                path
            )

        path.mkdir(
            parents=True,
            exist_ok=False,
        )

    if LOG_FILE.exists():
        LOG_FILE.unlink()

    require_files(
        [
            STEP04_SCRIPT,
            STEP05_SCRIPT,
            STEP06B_SCRIPT,
            STEP06C_SCRIPT,
            STEP06C_CANDIDATES,
            STEP06C_FULL_STATUS,
            STEP06C_COEFFICIENTS,
            STEP06C_FOLD_STATUS,
            STEP06C_OOF_METRICS,
            STEP06B_COEFFICIENTS,
        ]
    )

    log(
        "Starting Step 06d final stable fixed-effect inference workflow."
    )

    step04 = load_module(
        STEP04_SCRIPT,
        "step04_06d",
    )

    step05 = load_module(
        STEP05_SCRIPT,
        "step05_06d",
    )

    step06b = load_module(
        STEP06B_SCRIPT,
        "step06b_06d",
    )

    step06c = load_module(
        STEP06C_SCRIPT,
        "step06c_06d",
    )

    # Redirect only new Step-06d candidate outputs.
    step06c.FULL_CHECKPOINT_ROOT = (
        NEW_FULL_CHECKPOINT_ROOT
    )

    step06c.OOF_CHECKPOINT_ROOT = (
        NEW_OOF_CHECKPOINT_ROOT
    )

    step06c.COVARIANCE_ROOT = (
        CANDIDATE_COVARIANCE_ROOT
    )

    existing_candidates = pd.read_csv(
        STEP06C_CANDIDATES
    )

    existing_full_status = pd.read_csv(
        STEP06C_FULL_STATUS
    )

    existing_fold_status = pd.read_csv(
        STEP06C_FOLD_STATUS
    )

    existing_metrics = pd.read_csv(
        STEP06C_OOF_METRICS
    )

    source_coefficients = pd.read_csv(
        STEP06B_COEFFICIENTS
    )

    structure, samples, source_oof = (
        build_exact_samples(
            step06b,
            step06c,
        )
    )

    new_candidates = new_summer_fc_candidates(
        existing_candidates,
        step06c,
    )

    all_candidates = pd.concat(
        [
            existing_candidates,
            new_candidates,
        ],
        ignore_index=True,
    )

    new_candidate_keys = {
        (
            int(
                row.Season_Code
            ),
            str(
                row.Count_Response
            ),
            str(
                row.Candidate_ID
            ),
        )
        for row in new_candidates.itertuples(
            index=False
        )
    }

    prediction_cache: Dict[
        Tuple[int, str, str],
        pd.DataFrame
    ] = {}

    metric_rows: List[
        Dict[str, object]
    ] = []

    # Reuse all existing Step-06c OOF predictions.
    for _, candidate in existing_candidates.iterrows():
        prediction = read_existing_candidate_oof(
            candidate,
            source_oof,
        )

        key = (
            int(
                candidate[
                    "Season_Code"
                ]
            ),
            str(
                candidate[
                    "Count_Response"
                ]
            ),
            str(
                candidate[
                    "Candidate_ID"
                ]
            ),
        )

        prediction_cache[
            key
        ] = prediction

        metric_rows.append(
            candidate_metrics(
                step04,
                candidate,
                prediction,
                (
                    "Reused Step-06c strict OOF checkpoint"
                    if any(
                        existing_checkpoint_path(
                            candidate,
                            fold,
                        ).exists()
                        for fold in FOLDS
                    )
                    else "Reused Step-05 selected OOF baseline"
                ),
            )
        )

    # Fit only the three missing Summer Fire_Count candidates.
    new_full_rows: List[
        Dict[str, object]
    ] = []

    new_coefficient_tables: List[
        pd.DataFrame
    ] = []

    new_fold_status_tables: List[
        pd.DataFrame
    ] = []

    for candidate in new_candidates.itertuples(
        index=False
    ):
        model_data = samples[
            (
                int(
                    candidate.Season_Code
                ),
                str(
                    candidate.Count_Response
                ),
            )
        ]

        log(
            f"New Summer FC full fit: {candidate.Candidate_ID}."
        )

        row, coefficients, _ = (
            step06c.fit_full_candidate(
                candidate,
                model_data,
                source_coefficients,
            )
        )

        new_full_rows.append(
            row
        )

        new_coefficient_tables.append(
            coefficients
        )

        log(
            f"New Summer FC five-fold OOF: {candidate.Candidate_ID}."
        )

        prediction, fold_status = (
            step06c.run_oof(
                candidate,
                model_data,
                step04,
                step05,
            )
        )

        new_fold_status_tables.append(
            fold_status
        )

        key = (
            int(
                candidate.Season_Code
            ),
            str(
                candidate.Count_Response
            ),
            str(
                candidate.Candidate_ID
            ),
        )

        prediction_cache[
            key
        ] = prediction

        candidate_series = pd.Series(
            candidate._asdict()
        )

        metric_rows.append(
            candidate_metrics(
                step04,
                candidate_series,
                prediction,
                "New Step-06d strict stable-likelihood five-fold refit",
            )
        )

    new_full_status = pd.DataFrame(
        new_full_rows
    )

    new_coefficients = (
        pd.concat(
            new_coefficient_tables,
            ignore_index=True,
        )
        if new_coefficient_tables
        else pd.DataFrame()
    )

    new_fold_status = (
        pd.concat(
            new_fold_status_tables,
            ignore_index=True,
        )
        if new_fold_status_tables
        else pd.DataFrame()
    )

    combined_full_status = pd.concat(
        [
            existing_full_status,
            new_full_status,
        ],
        ignore_index=True,
        sort=False,
    )

    revised_stability = revised_stability_table(
        combined_full_status
    )

    combined_metrics = pd.DataFrame(
        metric_rows
    )

    fold_nll = fold_nll_table(
        all_candidates,
        prediction_cache,
    )

    paired = paired_fold_comparison(
        fold_nll
    )

    fold_validity = fold_fit_validity(
        existing_fold_status,
        new_fold_status,
    )

    comparison, selection_summary = (
        select_candidates(
            all_candidates,
            revised_stability,
            combined_metrics,
            paired,
            fold_validity,
        )
    )

    selected = comparison.loc[
        comparison[
            "Selected"
        ].astype(bool)
    ].copy()

    log(
        f"Provisional selected structure count: {len(selected)}."
    )

    # Newton polish the selected full-data fits.
    final_diagnostic_rows: List[
        Dict[str, object]
    ] = []

    final_coefficient_tables: List[
        pd.DataFrame
    ] = []

    selected_oof_manifest_rows: List[
        Dict[str, object]
    ] = []

    for _, candidate in selected.iterrows():
        key = (
            int(
                candidate[
                    "Season_Code"
                ]
            ),
            str(
                candidate[
                    "Count_Response"
                ]
            ),
        )

        log(
            f"Newton polishing selected model: "
            f"{candidate['Season_Label']} "
            f"{candidate['Count_Response']} "
            f"{candidate['Candidate_ID']}."
        )

        diagnostics_row, coefficients = (
            finalize_selected_candidate(
                candidate,
                samples[
                    key
                ],
                step06c,
                new_candidate_keys,
            )
        )

        final_diagnostic_rows.append(
            diagnostics_row
        )

        final_coefficient_tables.append(
            coefficients
        )

        prediction_key = (
            int(
                candidate[
                    "Season_Code"
                ]
            ),
            str(
                candidate[
                    "Count_Response"
                ]
            ),
            str(
                candidate[
                    "Candidate_ID"
                ]
            ),
        )

        prediction = (
            prediction_cache[
                prediction_key
            ].copy()
        )

        destination = (
            SELECTED_OOF_ROOT
            / (
                "Final_Selected_Inference_OOF_Predictions_"
                f"S{int(candidate['Season_Code'])}_"
                f"{candidate['Count_Response']}.csv.gz"
            )
        )

        prediction.to_csv(
            destination,
            index=False,
            compression="gzip",
        )

        selected_oof_manifest_rows.append(
            {
                "Season_Code": (
                    candidate[
                        "Season_Code"
                    ]
                ),
                "Season_Label": (
                    candidate[
                        "Season_Label"
                    ]
                ),
                "Count_Response": (
                    candidate[
                        "Count_Response"
                    ]
                ),
                "Selected_Candidate_ID": (
                    candidate[
                        "Candidate_ID"
                    ]
                ),
                "Output_File": (
                    destination.name
                ),
                "Rows": int(
                    len(
                        prediction
                    )
                ),
                "OOF_Source": (
                    combined_metrics.loc[
                        (
                            combined_metrics[
                                "Season_Code"
                            ].astype(int)
                            == int(
                                candidate[
                                    "Season_Code"
                                ]
                            )
                        )
                        & (
                            combined_metrics[
                                "Count_Response"
                            ].astype(str)
                            == str(
                                candidate[
                                    "Count_Response"
                                ]
                            )
                        )
                        & (
                            combined_metrics[
                                "Candidate_ID"
                            ].astype(str)
                            == str(
                                candidate[
                                    "Candidate_ID"
                                ]
                            )
                        ),
                        "OOF_Source",
                    ].iloc[
                        0
                    ]
                ),
            }
        )

    final_diagnostics = pd.DataFrame(
        final_diagnostic_rows
    )

    final_coefficients = (
        pd.concat(
            final_coefficient_tables,
            ignore_index=True,
        )
        if final_coefficient_tables
        else pd.DataFrame()
    )

    selected_oof_manifest = pd.DataFrame(
        selected_oof_manifest_rows
    )

    final_selected_stable_keys = {
        (
            int(
                row.Season_Code
            ),
            str(
                row.Count_Response
            ),
            str(
                row.Candidate_ID
            ),
        )
        for row in final_diagnostics.loc[
            final_diagnostics[
                "Final_Selected_Fit_Stable"
            ].astype(bool)
        ].itertuples(
            index=False
        )
    }

    final_structure_rows: List[
        Dict[str, object]
    ] = []

    for _, row in selected.iterrows():
        candidate_key = (
            int(
                row[
                    "Season_Code"
                ]
            ),
            str(
                row[
                    "Count_Response"
                ]
            ),
            str(
                row[
                    "Candidate_ID"
                ]
            ),
        )

        final_structure_rows.append(
            {
                "Season_Code": (
                    row[
                        "Season_Code"
                    ]
                ),
                "Season_Label": (
                    row[
                        "Season_Label"
                    ]
                ),
                "Count_Response": (
                    row[
                        "Count_Response"
                    ]
                ),
                "Rate_Scale_Name": (
                    row[
                        "Rate_Scale_Name"
                    ]
                ),
                "Model_Structure": (
                    row[
                        "Model_Structure"
                    ]
                ),
                "Selected_Inference_Candidate_ID": (
                    row[
                        "Candidate_ID"
                    ]
                ),
                "Structural_Zero_Predictors": (
                    row[
                        "Structural_Zero_Predictors"
                    ]
                ),
                "Structural_Zero_Country_Terms": (
                    row[
                        "Structural_Zero_Country_Terms"
                    ]
                ),
                "Count_Predictors": (
                    row[
                        "Count_Predictors"
                    ]
                ),
                "Count_Country_Terms": (
                    row[
                        "Count_Country_Terms"
                    ]
                ),
                "Transformation_Scheme": (
                    row[
                        "Transformation_Scheme"
                    ]
                ),
                "Terms_Removed_from_Step05": (
                    row[
                        "Terms_Removed_from_Step05"
                    ]
                ),
                "Revised_Full_Data_Stable_before_Polish": (
                    row[
                        "Revised_Full_Data_Stable"
                    ]
                ),
                "Final_Selected_Fit_Stable_after_Polish": bool(
                    candidate_key
                    in final_selected_stable_keys
                ),
                "One_SE_Noninferior": (
                    row[
                        "One_SE_Noninferior"
                    ]
                ),
                "Mean_Negative_LogLikelihood": (
                    row[
                        "Mean_Negative_LogLikelihood"
                    ]
                ),
                "Mild_PSD_Correction_Required": (
                    row[
                        "Mild_PSD_Correction_Required"
                    ]
                ),
                "Selection_Status": (
                    row[
                        "Selection_Status"
                    ]
                ),
            }
        )

    final_structure = pd.DataFrame(
        final_structure_rows
    ).sort_values(
        [
            "Season_Code",
            "Count_Response",
        ]
    )

    resolved_count = int(
        len(
            final_structure
        )
    )

    stable_after_polish_count = int(
        final_structure[
            "Final_Selected_Fit_Stable_after_Polish"
        ].astype(bool).sum()
    ) if not final_structure.empty else 0

    all_six_resolved = bool(
        resolved_count
        == 6
        and stable_after_polish_count
        == 6
    )

    global_qa = pd.DataFrame(
        [
            (
                "Existing_Step06c_candidate_count",
                len(
                    existing_candidates
                ),
            ),
            (
                "New_Summer_FC_candidate_count",
                len(
                    new_candidates
                ),
            ),
            (
                "Existing_candidate_fits_reused",
                True,
            ),
            (
                "Existing_candidate_OOF_folds_reused",
                True,
            ),
            (
                "New_Summer_FC_fold_fit_count",
                int(
                    len(
                        new_fold_status
                    )
                ),
            ),
            (
                "Provisional_selected_structure_count",
                resolved_count,
            ),
            (
                "Stable_selected_structure_count_after_polish",
                stable_after_polish_count,
            ),
            (
                "All_six_final_fixed_effect_structures_resolved",
                all_six_resolved,
            ),
            (
                "Mean_structure_type",
                "Fixed NB1/ZINB1",
            ),
            (
                "Standardized_score_threshold",
                STANDARDIZED_SCORE_MAX,
            ),
            (
                "Maximum_mild_non_PSD_ratio",
                MAX_MILD_NON_PSD_RATIO,
            ),
        ],
        columns=[
            "Metric",
            "Value",
        ],
    )

    rules = pd.DataFrame(
        [
            {
                "Rule_ID": (
                    "S01"
                ),
                "Rule": (
                    "Standardized score"
                ),
                "Criterion": (
                    f"Relative gradient <= {STANDARDIZED_SCORE_MAX}"
                ),
                "Rationale": (
                    "Scale-independent convergence criterion appropriate "
                    "for large samples and differently scaled parameters."
                ),
            },
            {
                "Rule_ID": (
                    "S02"
                ),
                "Rule": (
                    "Information matrix"
                ),
                "Criterion": (
                    "Full rank and condition number <= 1e12"
                ),
                "Rationale": (
                    "Reject rank-deficient or practically unidentified fits."
                ),
            },
            {
                "Rule_ID": (
                    "S03"
                ),
                "Rule": (
                    "Boundary parameters"
                ),
                "Criterion": (
                    "No parameter bound hit and no |coefficient| >= 25"
                ),
                "Rationale": (
                    "Reject complete or quasi-separation solutions."
                ),
            },
            {
                "Rule_ID": (
                    "S04"
                ),
                "Rule": (
                    "Two-way clustered covariance"
                ),
                "Criterion": (
                    "Raw PSD, or negative-eigenvalue ratio <= 0.001 "
                    "with positive diagonal and explicit PSD sensitivity"
                ),
                "Rationale": (
                    "Allows only negligible finite-sample multiway "
                    "cluster covariance non-PSD behavior."
                ),
            },
            {
                "Rule_ID": (
                    "S05"
                ),
                "Rule": (
                    "Selection priority"
                ),
                "Criterion": (
                    "Identifiability first; fewest removed terms; "
                    "paired one-SE and pooled OOF metrics secondary"
                ),
                "Rationale": (
                    "Do not preserve a non-estimable coefficient solely "
                    "for a small predictive NLL advantage."
                ),
            },
        ]
    )

    method = {
        "workflow_stage":
            "Final stable fixed-effect inference structure selection",
        "existing_step06c_candidates_reused":
            True,
        "new_models_fitted":
            "Only three Summer Fire_Count structural-zero country candidates",
        "standardized_score_threshold":
            STANDARDIZED_SCORE_MAX,
        "mild_non_psd_ratio_threshold":
            MAX_MILD_NON_PSD_RATIO,
        "newton_polishing_iterations":
            MAX_NEWTON_ITERATIONS,
        "paired_fold_diagnostic":
            "Candidate minus baseline mean NLL across the same five "
            "temporal validation folds; one-SE noninferiority when "
            "mean difference <= SE of paired differences.",
        "selection_priority": [
            "Retain stable Step-05 baseline",
            "If baseline is unidentified, select a stable alternative",
            "Prefer the fewest removed terms",
            "Use paired one-SE and pooled OOF metrics as secondary diagnostics",
        ],
        "mean_structure_type":
            "Fixed NB1/ZINB1",
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
        "scikit_learn": (
            sklearn.__version__
        ),
        "statsmodels": (
            statsmodels.__version__
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

    rules.to_csv(
        OUTPUT_ROOT
        / "02_Revised_Stability_and_Selection_Rules.csv",
        index=False,
        encoding="utf-8-sig",
    )

    all_candidates.to_csv(
        OUTPUT_ROOT
        / "03_All_Candidate_Definitions_Including_New_Summer_FC.csv",
        index=False,
        encoding="utf-8-sig",
    )

    revised_stability.to_csv(
        OUTPUT_ROOT
        / "04_Revised_Full_Data_Stability_Assessment.csv",
        index=False,
        encoding="utf-8-sig",
    )

    new_full_status.to_csv(
        OUTPUT_ROOT
        / "05_New_Summer_FC_Full_Data_Fit_Status.csv",
        index=False,
        encoding="utf-8-sig",
    )

    new_coefficients.to_csv(
        OUTPUT_ROOT
        / "06_New_Summer_FC_Coefficients.csv",
        index=False,
        encoding="utf-8-sig",
    )

    new_fold_status.to_csv(
        OUTPUT_ROOT
        / "07_New_Summer_FC_OOF_Fold_Status.csv",
        index=False,
        encoding="utf-8-sig",
    )

    combined_metrics.to_csv(
        OUTPUT_ROOT
        / "08_All_Candidate_OOF_Performance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fold_nll.to_csv(
        OUTPUT_ROOT
        / "09_Fold_Level_Mean_NLL.csv",
        index=False,
        encoding="utf-8-sig",
    )

    paired.to_csv(
        OUTPUT_ROOT
        / "10_Paired_Fold_NLL_One_SE_Diagnostic.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fold_validity.to_csv(
        OUTPUT_ROOT
        / "11_Fold_Fit_Validity.csv",
        index=False,
        encoding="utf-8-sig",
    )

    comparison.to_csv(
        OUTPUT_ROOT
        / "12_Identifiability_First_Candidate_Comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    selection_summary.to_csv(
        OUTPUT_ROOT
        / "13_Formal_Selection_Summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    final_structure.to_csv(
        OUTPUT_ROOT
        / "14_Final_Six_Fixed_Effect_Inference_Structures.csv",
        index=False,
        encoding="utf-8-sig",
    )

    final_diagnostics.to_csv(
        OUTPUT_ROOT
        / "15_Final_Selected_Newton_Polish_and_Covariance_Diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    final_coefficients.to_csv(
        OUTPUT_ROOT
        / "16_Final_Selected_Coefficients_and_Two_Way_Inference.csv",
        index=False,
        encoding="utf-8-sig",
    )

    selected_oof_manifest.to_csv(
        OUTPUT_ROOT
        / "17_Final_Selected_OOF_Manifest.csv",
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
        "Step 06d completed."
    )

    for row in final_structure.itertuples(
        index=False
    ):
        log(
            f"Final structure: {row.Season_Label} "
            f"{row.Count_Response}; "
            f"{row.Selected_Inference_Candidate_ID}; "
            f"stable_after_polish="
            f"{row.Final_Selected_Fit_Stable_after_Polish}."
        )

    if not all_six_resolved:
        log(
            "WARNING: all six final stable structures were not resolved. "
            "Do not proceed to final model comparison."
        )
    else:
        log(
            "All six final stable fixed-effect inference structures "
            "were resolved. Proceed to regression-versus-RF model comparison."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
