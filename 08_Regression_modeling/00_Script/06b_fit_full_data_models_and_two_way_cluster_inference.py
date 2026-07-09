# -*- coding: utf-8 -*-
"""
Full-data final fixed-effect fits and two-way cluster-robust inference.

Recommended public location
---------------------------
<CODE_ROOT>/08_Regression_modeling/00_Script/
    06b_fit_full_data_models_and_two_way_cluster_inference.py

Inputs
------
<CODE_ROOT>/00_Input_data/
    Base_Table_2001_2025.csv.bz2

<CODE_ROOT>/08_Regression_modeling/00_Script/
    04_compare_nb1_hurdle_zinb.py
    05_refine_component_specific_predictors.py

<CODE_ROOT>/08_Regression_modeling/
    05_Component_Specific_Predictor_Refinement/
        16_Final_Component_Specific_Predictor_Structure.csv
        Refined_Selected_OOF_Predictions_S*_*.csv.gz

    06_Panel_Dependence_and_Clustered_Inference_Audit/
        11_Process_Specific_Dependence_Decisions.csv

Output
------
<CODE_ROOT>/08_Regression_modeling/
    06b_Full_Data_Fixed_Model_and_Two_Way_Cluster_Inference/

Purpose
-------
This step:

1. keeps the Step-05 distribution and predictor structures fixed;
2. fits the six selected models to their full 2001-2025 analysis samples;
3. calculates model-based, GRID_UID-clustered, Year-clustered, and
   GRID_UID-Year two-way cluster-robust covariance matrices;
4. audits score, Hessian, information-matrix, and covariance stability;
5. writes coefficient-readiness and clustered-covariance diagnostics.

This script retains the selected fixed mean structures and evaluates only
their full-data inference stability.

Important interpretation
------------------------
The structural-zero coefficients of a ZINB1 model describe the odds of
belonging to the excess-zero process. A positive structural-zero coefficient
therefore indicates higher excess-zero odds, not higher fire occurrence.

Continuous predictors are reported on the standardized transformed scale
used in Step 05. Country is retained as a fixed effect with China as the
reference category. ForestPixelCount remains the count exposure, equivalent
to offset log(ForestPixelCount).

Two-way clustered inference uses the Cameron-Gelbach-Miller inclusion-
exclusion form:

    Cov(GRID_UID) + Cov(Year) - Cov(GRID_UID x Year)

with CR1-type finite-cluster corrections. Because only 25 Year clusters are
available, t inference uses df = min(G_grid - 1, G_year - 1) = 24.
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
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy
import sklearn
import statsmodels
from scipy import stats
from sklearn.preprocessing import StandardScaler
from statsmodels.discrete.count_model import (
    ZeroInflatedNegativeBinomialP,
)
from statsmodels.discrete.discrete_model import (
    NegativeBinomial,
)


# =============================================================================
# Repository-relative paths
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REGRESSION_ROOT = SCRIPT_DIR.parent
CODE_ROOT = REGRESSION_ROOT.parent

INPUT_CSV = (
    CODE_ROOT
    / "00_Input_data"
    / "Base_Table_2001_2025.csv.bz2"
)

STEP04_SCRIPT = (
    SCRIPT_DIR
    / "04_compare_nb1_hurdle_zinb.py"
)

STEP05_SCRIPT = (
    SCRIPT_DIR
    / "05_refine_component_specific_predictors.py"
)

STEP05_ROOT = (
    REGRESSION_ROOT
    / "05_Component_Specific_Predictor_Refinement"
)

FINAL_STRUCTURE_CSV = (
    STEP05_ROOT
    / "16_Final_Component_Specific_Predictor_Structure.csv"
)

PANEL_AUDIT_ROOT = (
    REGRESSION_ROOT
    / "06_Panel_Dependence_and_Clustered_Inference_Audit"
)

PANEL_DECISIONS_CSV = (
    PANEL_AUDIT_ROOT
    / "11_Process_Specific_Dependence_Decisions.csv"
)

OUTPUT_ROOT = (
    REGRESSION_ROOT
    / "06b_Full_Data_Fixed_Model_and_Two_Way_Cluster_Inference"
)

COVARIANCE_ROOT = (
    OUTPUT_ROOT
    / "Covariance_Matrices"
)

PREDICTION_ROOT = (
    OUTPUT_ROOT
    / "Full_Data_Predictions"
)

LOG_FILE = (
    OUTPUT_ROOT
    / "full_data_two_way_cluster_inference.log"
)


# =============================================================================
# Fixed workflow settings
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

COUNTRY_LEVELS = [
    "China",
    "NK",
    "Russia",
]

COUNTRY_PARAMETER_NAMES = [
    "Country_NK",
    "Country_Russia",
]

LOG1P_VARIABLES = {
    "ND",
    "NE",
    "LtgProxy",
    "POP",
    "Dis_Farm",
    "Road_dens",
}

FOREST_PIXEL_AREA_KM2 = 0.25
FOREST_AREA_THRESHOLD_KM2 = 2.5

PROBABILITY_EPSILON = 1e-12

# Hessian and covariance diagnostics.
INFORMATION_RCOND = 1e-10
NUMERICAL_HESSIAN_RELATIVE_STEP = 1e-5
PSD_RELATIVE_TOLERANCE = 1e-8
SCORE_SUM_RELATIVE_TOLERANCE = 1e-4


# =============================================================================
# Logging
# =============================================================================

def log(message: str) -> None:
    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    line = f"[{timestamp}] {message}"
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


# =============================================================================
# Generic helpers
# =============================================================================

def safe_name(value: str) -> str:
    return re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        str(value),
    )


def parse_predictors(
    value: object,
) -> List[str]:
    if pd.isna(value):
        return []

    return list(
        dict.fromkeys(
            item.strip()
            for item in str(value).split(";")
            if item.strip()
        )
    )


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


def robust_bool(value: object) -> bool:
    if isinstance(
        value,
        (
            bool,
            np.bool_,
        ),
    ):
        return bool(value)

    if isinstance(
        value,
        (
            int,
            np.integer,
        ),
    ):
        return bool(value)

    if isinstance(
        value,
        (
            float,
            np.floating,
        ),
    ):
        if np.isnan(value):
            return False

        return bool(value)

    text = str(value).strip().lower()

    if text in {
        "true",
        "t",
        "1",
        "yes",
        "y",
    }:
        return True

    if text in {
        "false",
        "f",
        "0",
        "no",
        "n",
        "",
        "nan",
        "none",
    }:
        return False

    raise ValueError(
        f"Cannot interpret boolean value: {value!r}"
    )


def load_module(
    path: Path,
    module_name: str,
):
    if not path.exists():
        raise FileNotFoundError(
            f"Required script not found:\n{path}"
        )

    spec = (
        importlib.util.spec_from_file_location(
            module_name,
            path,
        )
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            f"Could not create an import specification for {path.name}."
        )

    module = (
        importlib.util.module_from_spec(
            spec
        )
    )

    previous = sys.modules.get(
        module_name
    )

    sys.modules[
        module_name
    ] = module

    try:
        spec.loader.exec_module(
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


def finite_or_nan(
    value: object,
) -> float:
    try:
        number = float(
            value
        )
    except (
        TypeError,
        ValueError,
    ):
        return np.nan

    if not np.isfinite(
        number
    ):
        return np.nan

    return number


def convergence_status(
    result: object,
) -> bool:
    retvals = getattr(
        result,
        "mle_retvals",
        {},
    )

    if isinstance(
        retvals,
        dict,
    ):
        if "converged" in retvals:
            return bool(
                retvals[
                    "converged"
                ]
            )

        if "warnflag" in retvals:
            return (
                int(
                    retvals[
                        "warnflag"
                    ]
                )
                == 0
            )

    return bool(
        getattr(
            result,
            "converged",
            True,
        )
    )


def warning_contains_hessian(
    warning_text: object,
) -> bool:
    return (
        "hessian"
        in str(
            warning_text
        ).lower()
    )


# =============================================================================
# Source definitions and exact analysis samples
# =============================================================================

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
            season,
            response,
        )
        for season in SEASONS
        for response in COUNT_RESPONSES
    }

    observed = {
        (
            int(
                row.Season_Code
            ),
            str(
                row.Count_Response
            ),
        )
        for row in structure.itertuples(
            index=False
        )
    }

    if observed != expected:
        raise ValueError(
            "The Step-05 final structure table does not contain "
            "exactly the six expected season-response combinations."
        )

    if len(
        structure
    ) != 6:
        raise ValueError(
            "The Step-05 final structure table must contain six rows."
        )

    return (
        structure.sort_values(
            [
                "Season_Code",
                "Count_Response",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def selected_oof_path(
    season_code: int,
    response: str,
) -> Path:
    return (
        STEP05_ROOT
        / (
            "Refined_Selected_OOF_Predictions_"
            f"S{season_code}_{response}.csv.gz"
        )
    )


def read_base_table(
    structure: pd.DataFrame,
) -> pd.DataFrame:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Base table not found:\n{INPUT_CSV}"
        )

    selected_predictors = sorted(
        {
            predictor
            for column in [
                "Structural_Zero_Predictors",
                "Count_Predictors",
            ]
            for value in structure[
                column
            ]
            for predictor in parse_predictors(
                value
            )
        }
    )

    required = {
        "GRID_UID",
        "GRID_ID",
        "Country",
        "Year",
        "Season",
        "Fire_Count",
        "Burned_Pixel_Count",
        "ForestPixelCount",
        "ForestArea_km2",
        *selected_predictors,
    }

    header = pd.read_csv(
        INPUT_CSV,
        nrows=0,
    )

    missing = sorted(
        required
        - set(
            header.columns
        )
    )

    if missing:
        raise ValueError(
            "Base table is missing required columns:\n"
            + "\n".join(
                missing
            )
        )

    data = pd.read_csv(
        INPUT_CSV,
        usecols=sorted(
            required
        ),
    )

    if data.duplicated(
        [
            "GRID_UID",
            "Year",
            "Season",
        ],
        keep=False,
    ).any():
        raise ValueError(
            "Duplicate GRID_UID-Year-Season records were found "
            "in the base table."
        )

    data[
        "GRID_UID"
    ] = (
        data[
            "GRID_UID"
        ].astype(str)
    )

    exposure = pd.to_numeric(
        data[
            "ForestPixelCount"
        ],
        errors="coerce",
    )

    forest_area = pd.to_numeric(
        data[
            "ForestArea_km2"
        ],
        errors="coerce",
    )

    if (
        exposure.isna().any()
        or forest_area.isna().any()
    ):
        raise ValueError(
            "Forest exposure variables contain nonnumeric values."
        )

    if float(
        np.abs(
            forest_area
            - exposure
            * FOREST_PIXEL_AREA_KM2
        ).max()
    ) > 1e-9:
        raise ValueError(
            "ForestArea_km2 is inconsistent with "
            "ForestPixelCount * 0.25."
        )

    data = (
        data.loc[
            forest_area
            > FOREST_AREA_THRESHOLD_KM2
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    unexpected_countries = (
        set(
            data[
                "Country"
            ]
            .dropna()
            .astype(str)
        )
        - set(
            COUNTRY_LEVELS
        )
    )

    if unexpected_countries:
        raise ValueError(
            "Unexpected Country values were found: "
            + ", ".join(
                sorted(
                    unexpected_countries
                )
            )
        )

    for response in COUNT_RESPONSES:
        values = pd.to_numeric(
            data[
                response
            ],
            errors="coerce",
        )

        invalid = (
            values.isna()
            | ~np.isfinite(
                values
            )
            | (
                values < 0
            )
            | (
                np.abs(
                    values
                    - np.round(
                        values
                    )
                )
                > 1e-9
            )
        )

        if invalid.any():
            raise ValueError(
                f"{response} contains invalid count values."
            )

        data[
            response
        ] = values.astype(
            int
        )

    if (
        pd.to_numeric(
            data[
                "ForestPixelCount"
            ],
            errors="coerce",
        )
        <= 0
    ).any():
        raise ValueError(
            "ForestPixelCount must be positive."
        )

    return data


def exact_model_sample(
    base_data: pd.DataFrame,
    season_code: int,
    response: str,
    model_structure: str,
    selected_predictors: Sequence[str],
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    oof_path = selected_oof_path(
        season_code,
        response,
    )

    if not oof_path.exists():
        raise FileNotFoundError(
            "Selected Step-05 OOF prediction file was not found:\n"
            f"{oof_path}"
        )

    oof = pd.read_csv(
        oof_path,
        compression="gzip",
    )

    required_columns_present(
        oof,
        {
            "GRID_UID",
            "GRID_ID",
            "Country",
            "Year",
            "Season",
            "ForestPixelCount",
            "Observed_Count",
            "Model_Structure",
        },
        oof_path.name,
    )

    if set(
        oof[
            "Model_Structure"
        ].astype(str)
    ) != {
        model_structure
    }:
        raise ValueError(
            f"Unexpected model structure in {oof_path.name}."
        )

    if set(
        oof[
            "Season"
        ].astype(int)
    ) != {
        season_code
    }:
        raise ValueError(
            f"Unexpected season values in {oof_path.name}."
        )

    if oof.duplicated(
        [
            "GRID_UID",
            "Year",
            "Season",
        ],
        keep=False,
    ).any():
        raise ValueError(
            f"Duplicate panel keys were found in {oof_path.name}."
        )

    oof[
        "GRID_UID"
    ] = (
        oof[
            "GRID_UID"
        ].astype(str)
    )

    keys = (
        oof[
            [
                "GRID_UID",
                "Year",
                "Season",
                "Observed_Count",
                "ForestPixelCount",
            ]
        ]
        .copy()
        .rename(
            columns={
                "Observed_Count":
                    "OOF_Observed_Count",
                "ForestPixelCount":
                    "OOF_ForestPixelCount",
            }
        )
    )

    model_data = (
        base_data.loc[
            base_data[
                "Season"
            ]
            .astype(int)
            .eq(
                season_code
            )
        ]
        .merge(
            keys,
            on=[
                "GRID_UID",
                "Year",
                "Season",
            ],
            how="inner",
            validate="one_to_one",
        )
    )

    if len(
        model_data
    ) != len(
        oof
    ):
        raise ValueError(
            f"The base table and OOF sample do not match for "
            f"{SEASONS[season_code]} {response}: "
            f"{len(model_data):,} versus {len(oof):,} rows."
        )

    if not np.allclose(
        model_data[
            response
        ].to_numpy(
            dtype=float
        ),
        model_data[
            "OOF_Observed_Count"
        ].to_numpy(
            dtype=float
        ),
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError(
            f"Observed counts disagree with the Step-05 OOF file for "
            f"{SEASONS[season_code]} {response}."
        )

    if not np.allclose(
        model_data[
            "ForestPixelCount"
        ].to_numpy(
            dtype=float
        ),
        model_data[
            "OOF_ForestPixelCount"
        ].to_numpy(
            dtype=float
        ),
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError(
            f"ForestPixelCount disagrees with the Step-05 OOF file for "
            f"{SEASONS[season_code]} {response}."
        )

    complete_columns = [
        "Country",
        response,
        "ForestPixelCount",
        *selected_predictors,
    ]

    complete = (
        model_data[
            complete_columns
        ]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .notna()
        .all(
            axis=1
        )
    )

    if not complete.all():
        excluded = model_data.loc[
            ~complete,
            [
                "GRID_UID",
                "GRID_ID",
                "Country",
                "Year",
                "Season",
                *selected_predictors,
            ],
        ].copy()

        excluded.to_csv(
            OUTPUT_ROOT
            / (
                "ERROR_Incomplete_Exact_Sample_"
                f"S{season_code}_{response}.csv"
            ),
            index=False,
            encoding="utf-8-sig",
        )

        raise ValueError(
            f"The exact Step-05 sample contains incomplete selected "
            f"predictors for {SEASONS[season_code]} {response}."
        )

    model_data = (
        model_data.sort_values(
            [
                "GRID_UID",
                "Year",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return (
        model_data,
        oof,
    )


# =============================================================================
# Transformations and design matrices
# =============================================================================

def transformed_continuous(
    frame: pd.DataFrame,
    predictors: Sequence[str],
    scheme: str,
) -> pd.DataFrame:
    if not predictors:
        return pd.DataFrame(
            index=frame.index
        )

    result = (
        frame.loc[
            :,
            predictors,
        ]
        .astype(float)
        .copy()
    )

    if scheme == "Raw":
        return result

    if scheme != (
        "Audit_Guided_Log1p"
    ):
        raise ValueError(
            f"Unknown transformation scheme: {scheme}"
        )

    for variable in predictors:
        if variable not in LOG1P_VARIABLES:
            continue

        if float(
            result[
                variable
            ].min()
        ) < 0:
            raise ValueError(
                f"{variable} contains negative values and "
                "cannot use log1p."
            )

        result[
            variable
        ] = np.log1p(
            result[
                variable
            ].to_numpy(
                dtype=float
            )
        )

    return result


def build_full_design(
    frame: pd.DataFrame,
    predictors: Sequence[str],
    scheme: str,
    component: str,
) -> Tuple[
    np.ndarray,
    List[str],
    pd.DataFrame,
]:
    predictors = list(
        predictors
    )

    transformed = (
        transformed_continuous(
            frame=frame,
            predictors=predictors,
            scheme=scheme,
        )
    )

    scaler_rows: List[
        Dict[str, object]
    ] = []

    if predictors:
        scaler = StandardScaler()

        scaled = scaler.fit_transform(
            transformed.to_numpy(
                dtype=float
            )
        )

        for index, variable in enumerate(
            predictors
        ):
            scaler_rows.append(
                {
                    "Component": (
                        component
                    ),
                    "Variable": (
                        variable
                    ),
                    "Transformation_Scheme": (
                        scheme
                    ),
                    "Log1p_Applied": bool(
                        scheme
                        == "Audit_Guided_Log1p"
                        and variable
                        in LOG1P_VARIABLES
                    ),
                    "Transformed_Mean": float(
                        scaler.mean_[
                            index
                        ]
                    ),
                    "Transformed_Scale": float(
                        scaler.scale_[
                            index
                        ]
                    ),
                }
            )
    else:
        scaled = np.empty(
            (
                len(frame),
                0,
            ),
            dtype=float,
        )

    country = (
        frame[
            "Country"
        ].astype(str)
    )

    country_dummies = np.column_stack(
        [
            (
                country
                == "NK"
            )
            .astype(float)
            .to_numpy(),
            (
                country
                == "Russia"
            )
            .astype(float)
            .to_numpy(),
        ]
    )

    design = np.column_stack(
        [
            np.ones(
                len(frame),
                dtype=float,
            ),
            scaled,
            country_dummies,
        ]
    )

    names = [
        "Intercept",
        *predictors,
        *COUNTRY_PARAMETER_NAMES,
    ]

    if design.shape[1] != len(
        names
    ):
        raise RuntimeError(
            "Design-matrix column count does not match parameter names."
        )

    return (
        design,
        names,
        pd.DataFrame(
            scaler_rows
        ),
    )


# =============================================================================
# Full-data prediction frames
# =============================================================================

def prediction_frame(
    model_data: pd.DataFrame,
    response: str,
    model_structure: str,
    outputs: Dict[
        str,
        object,
    ],
) -> pd.DataFrame:
    result = model_data[
        [
            "GRID_UID",
            "GRID_ID",
            "Country",
            "Year",
            "Season",
            "ForestPixelCount",
        ]
    ].copy()

    observed = (
        model_data[
            response
        ].to_numpy(
            dtype=float
        )
    )

    exposure = (
        model_data[
            "ForestPixelCount"
        ].to_numpy(
            dtype=float
        )
    )

    predicted_count = np.asarray(
        outputs[
            "Predicted_Count"
        ],
        dtype=float,
    )

    result[
        "Count_Response"
    ] = response

    result[
        "Rate_Scale_Name"
    ] = COUNT_RESPONSES[
        response
    ]

    result[
        "Model_Structure"
    ] = model_structure

    result[
        "Observed_Count"
    ] = observed

    result[
        "Predicted_Count"
    ] = predicted_count

    result[
        "Observed_Rate"
    ] = (
        observed
        / exposure
    )

    result[
        "Predicted_Rate"
    ] = (
        predicted_count
        / exposure
    )

    result[
        "Predicted_Zero_Probability"
    ] = np.asarray(
        outputs[
            "Predicted_Zero_Probability"
        ],
        dtype=float,
    )

    result[
        "Predictive_LogProbability"
    ] = np.asarray(
        outputs[
            "Predictive_LogProbability"
        ],
        dtype=float,
    )

    return result


# =============================================================================
# Score and Hessian calculations
# =============================================================================

def numerical_score_obs(
    model: object,
    parameters: np.ndarray,
) -> np.ndarray:
    parameters = np.asarray(
        parameters,
        dtype=float,
    )

    base = np.asarray(
        model.loglikeobs(
            parameters
        ),
        dtype=float,
    )

    sample_size = len(
        base
    )

    parameter_count = len(
        parameters
    )

    result = np.empty(
        (
            sample_size,
            parameter_count,
        ),
        dtype=float,
    )

    for index in range(
        parameter_count
    ):
        step = (
            NUMERICAL_HESSIAN_RELATIVE_STEP
            * (
                1.0
                + abs(
                    parameters[
                        index
                    ]
                )
            )
        )

        upper = parameters.copy()
        lower = parameters.copy()

        upper[
            index
        ] += step

        lower[
            index
        ] -= step

        upper_loglike = np.asarray(
            model.loglikeobs(
                upper
            ),
            dtype=float,
        )

        lower_loglike = np.asarray(
            model.loglikeobs(
                lower
            ),
            dtype=float,
        )

        result[
            :,
            index,
        ] = (
            upper_loglike
            - lower_loglike
        ) / (
            2.0
            * step
        )

    return result


def total_score(
    model: object,
    parameters: np.ndarray,
) -> np.ndarray:
    try:
        value = np.asarray(
            model.score(
                parameters
            ),
            dtype=float,
        )

        if (
            value.ndim == 1
            and len(
                value
            )
            == len(
                parameters
            )
            and np.isfinite(
                value
            ).all()
        ):
            return value
    except Exception:
        pass

    return np.asarray(
        model.score_obs(
            parameters
        ),
        dtype=float,
    ).sum(
        axis=0
    )


def numerical_hessian_from_score(
    model: object,
    parameters: np.ndarray,
) -> np.ndarray:
    parameters = np.asarray(
        parameters,
        dtype=float,
    )

    parameter_count = len(
        parameters
    )

    hessian = np.empty(
        (
            parameter_count,
            parameter_count,
        ),
        dtype=float,
    )

    for index in range(
        parameter_count
    ):
        step = (
            NUMERICAL_HESSIAN_RELATIVE_STEP
            * (
                1.0
                + abs(
                    parameters[
                        index
                    ]
                )
            )
        )

        upper = parameters.copy()
        lower = parameters.copy()

        upper[
            index
        ] += step

        lower[
            index
        ] -= step

        upper_score = total_score(
            model,
            upper,
        )

        lower_score = total_score(
            model,
            lower,
        )

        hessian[
            :,
            index,
        ] = (
            upper_score
            - lower_score
        ) / (
            2.0
            * step
        )

    return (
        hessian
        + hessian.T
    ) / 2.0


def obtain_score_and_hessian(
    model: object,
    parameters: np.ndarray,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    str,
    str,
]:
    score_method = (
        "Analytic_score_obs"
    )

    try:
        score_observations = np.asarray(
            model.score_obs(
                parameters
            ),
            dtype=float,
        )

        if (
            score_observations.ndim != 2
            or score_observations.shape[
                1
            ] != len(
                parameters
            )
            or not np.isfinite(
                score_observations
            ).all()
        ):
            raise ValueError(
                "Analytic score_obs returned an invalid array."
            )
    except Exception:
        score_observations = (
            numerical_score_obs(
                model,
                parameters,
            )
        )

        score_method = (
            "Central_difference_loglikeobs"
        )

    hessian_method = (
        "Analytic_model_hessian"
    )

    try:
        hessian = np.asarray(
            model.hessian(
                parameters
            ),
            dtype=float,
        )

        if (
            hessian.shape
            != (
                len(
                    parameters
                ),
                len(
                    parameters
                ),
            )
            or not np.isfinite(
                hessian
            ).all()
        ):
            raise ValueError(
                "Analytic Hessian returned an invalid array."
            )

        hessian = (
            hessian
            + hessian.T
        ) / 2.0
    except Exception:
        hessian = (
            numerical_hessian_from_score(
                model,
                parameters,
            )
        )

        hessian_method = (
            "Central_difference_total_score"
        )

    return (
        score_observations,
        hessian,
        score_method,
        hessian_method,
    )


# =============================================================================
# Cluster-robust covariance
# =============================================================================

def factorized_groups(
    values: Sequence[object],
) -> Tuple[
    np.ndarray,
    int,
]:
    codes, uniques = pd.factorize(
        values,
        sort=False,
    )

    if (
        codes < 0
    ).any():
        raise ValueError(
            "Missing cluster labels were found."
        )

    return (
        codes.astype(
            int
        ),
        int(
            len(
                uniques
            )
        ),
    )


def cluster_meat(
    score_observations: np.ndarray,
    groups: Sequence[object],
) -> Tuple[
    np.ndarray,
    int,
    float,
]:
    score_observations = np.asarray(
        score_observations,
        dtype=float,
    )

    sample_size, parameter_count = (
        score_observations.shape
    )

    codes, cluster_count = (
        factorized_groups(
            groups
        )
    )

    if cluster_count < 2:
        raise ValueError(
            "At least two clusters are required."
        )

    cluster_scores = np.zeros(
        (
            cluster_count,
            parameter_count,
        ),
        dtype=float,
    )

    np.add.at(
        cluster_scores,
        codes,
        score_observations,
    )

    raw_meat = (
        cluster_scores.T
        @ cluster_scores
    )

    denominator = (
        sample_size
        - parameter_count
    )

    if denominator <= 0:
        raise ValueError(
            "The sample size must exceed the parameter count."
        )

    correction = (
        cluster_count
        / (
            cluster_count
            - 1.0
        )
        * (
            sample_size
            - 1.0
        )
        / denominator
    )

    return (
        correction
        * raw_meat,
        cluster_count,
        float(
            correction
        ),
    )


def covariance_from_meat(
    bread: np.ndarray,
    meat: np.ndarray,
) -> np.ndarray:
    covariance = (
        bread
        @ meat
        @ bread.T
    )

    return (
        covariance
        + covariance.T
    ) / 2.0


def nearest_psd_by_eigenvalue_clipping(
    covariance: np.ndarray,
) -> Tuple[
    np.ndarray,
    int,
]:
    covariance = (
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
            covariance
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
        * 1e-12
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

    adjusted = (
        adjusted
        + adjusted.T
    ) / 2.0

    return (
        adjusted,
        adjusted_count,
    )


def safe_standard_errors(
    covariance: np.ndarray,
) -> np.ndarray:
    diagonal = np.diag(
        covariance
    )

    result = np.full(
        len(
            diagonal
        ),
        np.nan,
        dtype=float,
    )

    valid = (
        np.isfinite(
            diagonal
        )
        & (
            diagonal > 0
        )
    )

    result[
        valid
    ] = np.sqrt(
        diagonal[
            valid
        ]
    )

    return result


def covariance_diagnostics(
    covariance: np.ndarray,
    label: str,
) -> Dict[str, object]:
    covariance = np.asarray(
        covariance,
        dtype=float,
    )

    finite = bool(
        np.isfinite(
            covariance
        ).all()
    )

    if finite:
        symmetric = (
            covariance
            + covariance.T
        ) / 2.0

        eigenvalues = np.linalg.eigvalsh(
            symmetric
        )

        maximum_absolute_eigenvalue = max(
            float(
                np.max(
                    np.abs(
                        eigenvalues
                    )
                )
            ),
            np.finfo(float).tiny,
        )

        tolerance = (
            PSD_RELATIVE_TOLERANCE
            * maximum_absolute_eigenvalue
        )

        minimum_eigenvalue = float(
            eigenvalues.min()
        )

        positive_semidefinite = bool(
            minimum_eigenvalue
            >= -tolerance
        )

        positive_diagonal = bool(
            (
                np.diag(
                    symmetric
                )
                > 0
            ).all()
        )

        condition_number = float(
            np.linalg.cond(
                symmetric
            )
        )
    else:
        minimum_eigenvalue = np.nan
        maximum_absolute_eigenvalue = np.nan
        positive_semidefinite = False
        positive_diagonal = False
        condition_number = np.nan

    return {
        f"{label}_Finite": (
            finite
        ),
        f"{label}_Minimum_Eigenvalue": (
            minimum_eigenvalue
        ),
        f"{label}_Maximum_Absolute_Eigenvalue": (
            maximum_absolute_eigenvalue
        ),
        f"{label}_Positive_Semidefinite": (
            positive_semidefinite
        ),
        f"{label}_All_Diagonal_Positive": (
            positive_diagonal
        ),
        f"{label}_Condition_Number": (
            condition_number
        ),
    }


def robust_covariance_bundle(
    model: object,
    parameters: np.ndarray,
    grid_uid: Sequence[object],
    year: Sequence[object],
) -> Tuple[
    Dict[str, np.ndarray],
    Dict[str, object],
]:
    (
        score_observations,
        hessian,
        score_method,
        hessian_method,
    ) = obtain_score_and_hessian(
        model=model,
        parameters=parameters,
    )

    sample_size, parameter_count = (
        score_observations.shape
    )

    score_sum = (
        score_observations.sum(
            axis=0
        )
    )

    score_scale = max(
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

    maximum_absolute_score_sum = float(
        np.max(
            np.abs(
                score_sum
            )
        )
    )

    relative_score_sum = (
        maximum_absolute_score_sum
        / score_scale
    )

    information = (
        -(
            hessian
            + hessian.T
        )
        / 2.0
    )

    information_eigenvalues = (
        np.linalg.eigvalsh(
            information
        )
    )

    information_rank = int(
        np.linalg.matrix_rank(
            information,
            tol=(
                INFORMATION_RCOND
                * max(
                    float(
                        np.max(
                            np.abs(
                                information_eigenvalues
                            )
                        )
                    ),
                    1.0,
                )
            ),
        )
    )

    information_condition = float(
        np.linalg.cond(
            information
        )
    )

    if (
        information_rank
        == parameter_count
        and np.isfinite(
            information_condition
        )
        and information_condition
        < (
            1.0
            / INFORMATION_RCOND
        )
    ):
        bread = np.linalg.inv(
            information
        )

        inversion_method = (
            "Direct_inverse"
        )
    else:
        bread = np.linalg.pinv(
            information,
            rcond=(
                INFORMATION_RCOND
            ),
        )

        inversion_method = (
            "Moore_Penrose_pseudoinverse"
        )

    bread = (
        bread
        + bread.T
    ) / 2.0

    grid_meat, grid_clusters, grid_correction = (
        cluster_meat(
            score_observations=(
                score_observations
            ),
            groups=grid_uid,
        )
    )

    year_meat, year_clusters, year_correction = (
        cluster_meat(
            score_observations=(
                score_observations
            ),
            groups=year,
        )
    )

    interaction_labels = pd.MultiIndex.from_arrays(
        [
            np.asarray(
                grid_uid
            ),
            np.asarray(
                year
            ),
        ]
    )

    (
        interaction_meat,
        interaction_clusters,
        interaction_correction,
    ) = cluster_meat(
        score_observations=(
            score_observations
        ),
        groups=(
            interaction_labels
        ),
    )

    model_based = bread

    grid_covariance = covariance_from_meat(
        bread,
        grid_meat,
    )

    year_covariance = covariance_from_meat(
        bread,
        year_meat,
    )

    interaction_covariance = (
        covariance_from_meat(
            bread,
            interaction_meat,
        )
    )

    two_way_raw = (
        grid_covariance
        + year_covariance
        - interaction_covariance
    )

    two_way_raw = (
        two_way_raw
        + two_way_raw.T
    ) / 2.0

    (
        two_way_psd,
        clipped_eigenvalue_count,
    ) = nearest_psd_by_eigenvalue_clipping(
        two_way_raw
    )

    covariance_matrices = {
        "Model_Based": (
            model_based
        ),
        "GRID_UID_Cluster": (
            grid_covariance
        ),
        "Year_Cluster": (
            year_covariance
        ),
        "GRID_UID_Year_Intersection": (
            interaction_covariance
        ),
        "Two_Way_Cluster_Raw": (
            two_way_raw
        ),
        "Two_Way_Cluster_PSD_Sensitivity": (
            two_way_psd
        ),
    }

    diagnostics: Dict[
        str,
        object,
    ] = {
        "N": (
            sample_size
        ),
        "Parameter_Count": (
            parameter_count
        ),
        "Score_Method": (
            score_method
        ),
        "Hessian_Method": (
            hessian_method
        ),
        "Maximum_Absolute_Score_Sum": (
            maximum_absolute_score_sum
        ),
        "Relative_Maximum_Score_Sum": (
            relative_score_sum
        ),
        "Score_Sum_Within_Tolerance": bool(
            relative_score_sum
            <= SCORE_SUM_RELATIVE_TOLERANCE
        ),
        "Information_Rank": (
            information_rank
        ),
        "Information_Full_Rank": bool(
            information_rank
            == parameter_count
        ),
        "Information_Minimum_Eigenvalue": float(
            information_eigenvalues.min()
        ),
        "Information_Maximum_Eigenvalue": float(
            information_eigenvalues.max()
        ),
        "Information_Condition_Number": (
            information_condition
        ),
        "Bread_Inversion_Method": (
            inversion_method
        ),
        "GRID_UID_Cluster_Count": (
            grid_clusters
        ),
        "Year_Cluster_Count": (
            year_clusters
        ),
        "Intersection_Cluster_Count": (
            interaction_clusters
        ),
        "GRID_UID_CR1_Correction": (
            grid_correction
        ),
        "Year_CR1_Correction": (
            year_correction
        ),
        "Intersection_CR1_Correction": (
            interaction_correction
        ),
        "Two_Way_PSD_Clipped_Eigenvalue_Count": (
            clipped_eigenvalue_count
        ),
    }

    for label, covariance in (
        covariance_matrices.items()
    ):
        diagnostics.update(
            covariance_diagnostics(
                covariance=covariance,
                label=label,
            )
        )

    raw_usable = bool(
        diagnostics[
            "Two_Way_Cluster_Raw_Finite"
        ]
        and diagnostics[
            "Two_Way_Cluster_Raw_All_Diagonal_Positive"
        ]
        and diagnostics[
            "Two_Way_Cluster_Raw_Positive_Semidefinite"
        ]
    )

    diagnostics[
        "Raw_Two_Way_Covariance_Usable"
    ] = raw_usable

    diagnostics[
        "PSD_Sensitivity_Required"
    ] = bool(
        not raw_usable
    )

    return (
        covariance_matrices,
        diagnostics,
    )


# =============================================================================
# Parameter names and coefficient tables
# =============================================================================

def parameter_metadata(
    model_structure: str,
    zero_names: Sequence[str],
    count_names: Sequence[str],
) -> pd.DataFrame:
    rows: List[
        Dict[str, object]
    ] = []

    if model_structure == "ZINB1":
        for name in zero_names:
            rows.append(
                {
                    "Component": (
                        "Structural_Zero"
                    ),
                    "Parameter": (
                        name
                    ),
                    "Parameter_Label": (
                        f"inflate_{name}"
                    ),
                    "Effect_Scale": (
                        "Excess_Zero_Odds_Ratio"
                    ),
                }
            )

    for name in count_names:
        rows.append(
            {
                "Component": (
                    "Count"
                ),
                "Parameter": (
                    name
                ),
                "Parameter_Label": (
                    name
                ),
                "Effect_Scale": (
                    "Count_Rate_Ratio"
                ),
            }
        )

    rows.append(
        {
            "Component": (
                "Dispersion"
            ),
            "Parameter": (
                "alpha"
            ),
            "Parameter_Label": (
                "alpha"
            ),
            "Effect_Scale": (
                "NB1_Dispersion"
            ),
        }
    )

    return pd.DataFrame(
        rows
    )


def inference_columns(
    estimate: np.ndarray,
    standard_error: np.ndarray,
    degrees_of_freedom: int,
    prefix: str,
) -> Dict[str, np.ndarray]:
    estimate = np.asarray(
        estimate,
        dtype=float,
    )

    standard_error = np.asarray(
        standard_error,
        dtype=float,
    )

    statistic = np.full(
        len(
            estimate
        ),
        np.nan,
        dtype=float,
    )

    valid = (
        np.isfinite(
            estimate
        )
        & np.isfinite(
            standard_error
        )
        & (
            standard_error > 0
        )
    )

    statistic[
        valid
    ] = (
        estimate[
            valid
        ]
        / standard_error[
            valid
        ]
    )

    p_value = np.full(
        len(
            estimate
        ),
        np.nan,
        dtype=float,
    )

    p_value[
        valid
    ] = (
        2.0
        * stats.t.sf(
            np.abs(
                statistic[
                    valid
                ]
            ),
            df=(
                degrees_of_freedom
            ),
        )
    )

    critical = float(
        stats.t.ppf(
            0.975,
            df=(
                degrees_of_freedom
            ),
        )
    )

    lower = np.full(
        len(
            estimate
        ),
        np.nan,
        dtype=float,
    )

    upper = np.full(
        len(
            estimate
        ),
        np.nan,
        dtype=float,
    )

    lower[
        valid
    ] = (
        estimate[
            valid
        ]
        - critical
        * standard_error[
            valid
        ]
    )

    upper[
        valid
    ] = (
        estimate[
            valid
        ]
        + critical
        * standard_error[
            valid
        ]
    )

    return {
        f"{prefix}_SE": (
            standard_error
        ),
        f"{prefix}_t": (
            statistic
        ),
        f"{prefix}_df": np.full(
            len(
                estimate
            ),
            degrees_of_freedom,
            dtype=int,
        ),
        f"{prefix}_p": (
            p_value
        ),
        f"{prefix}_CI_Lower": (
            lower
        ),
        f"{prefix}_CI_Upper": (
            upper
        ),
    }


def coefficient_table(
    parameters: np.ndarray,
    metadata: pd.DataFrame,
    covariance_matrices: Dict[
        str,
        np.ndarray,
    ],
    diagnostics: Dict[
        str,
        object,
    ],
) -> pd.DataFrame:
    parameters = np.asarray(
        parameters,
        dtype=float,
    )

    if len(
        metadata
    ) != len(
        parameters
    ):
        raise ValueError(
            "Parameter metadata length does not match parameter vector."
        )

    result = metadata.copy()

    result[
        "Estimate"
    ] = parameters

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

    covariance_prefixes = [
        (
            "Model_Based",
            "Model_Based",
        ),
        (
            "GRID_UID_Cluster",
            "GRID_UID_Cluster",
        ),
        (
            "Year_Cluster",
            "Year_Cluster",
        ),
        (
            "Two_Way_Cluster_Raw",
            "Two_Way_Raw",
        ),
        (
            "Two_Way_Cluster_PSD_Sensitivity",
            "Two_Way_PSD_Sensitivity",
        ),
    ]

    for covariance_key, prefix in (
        covariance_prefixes
    ):
        standard_error = safe_standard_errors(
            covariance_matrices[
                covariance_key
            ]
        )

        columns = inference_columns(
            estimate=parameters,
            standard_error=(
                standard_error
            ),
            degrees_of_freedom=(
                degrees_of_freedom
            ),
            prefix=prefix,
        )

        for column, values in (
            columns.items()
        ):
            result[
                column
            ] = values

    effect_mask = (
        result[
            "Component"
        ]
        != "Dispersion"
    )

    result[
        "Exponentiated_Estimate"
    ] = np.where(
        effect_mask,
        np.exp(
            np.clip(
                result[
                    "Estimate"
                ].to_numpy(
                    dtype=float
                ),
                -700.0,
                700.0,
            )
        ),
        np.nan,
    )

    raw_lower = result[
        "Two_Way_Raw_CI_Lower"
    ].to_numpy(
        dtype=float
    )

    raw_upper = result[
        "Two_Way_Raw_CI_Upper"
    ].to_numpy(
        dtype=float
    )

    result[
        "Two_Way_Raw_Exponentiated_CI_Lower"
    ] = np.where(
        effect_mask,
        np.exp(
            np.clip(
                raw_lower,
                -700.0,
                700.0,
            )
        ),
        np.nan,
    )

    result[
        "Two_Way_Raw_Exponentiated_CI_Upper"
    ] = np.where(
        effect_mask,
        np.exp(
            np.clip(
                raw_upper,
                -700.0,
                700.0,
            )
        ),
        np.nan,
    )

    psd_lower = result[
        "Two_Way_PSD_Sensitivity_CI_Lower"
    ].to_numpy(
        dtype=float
    )

    psd_upper = result[
        "Two_Way_PSD_Sensitivity_CI_Upper"
    ].to_numpy(
        dtype=float
    )

    result[
        "Two_Way_PSD_Exponentiated_CI_Lower"
    ] = np.where(
        effect_mask,
        np.exp(
            np.clip(
                psd_lower,
                -700.0,
                700.0,
            )
        ),
        np.nan,
    )

    result[
        "Two_Way_PSD_Exponentiated_CI_Upper"
    ] = np.where(
        effect_mask,
        np.exp(
            np.clip(
                psd_upper,
                -700.0,
                700.0,
            )
        ),
        np.nan,
    )

    result[
        "Raw_Two_Way_Covariance_Usable"
    ] = bool(
        diagnostics[
            "Raw_Two_Way_Covariance_Usable"
        ]
    )

    preferred_inference = (
        "Two_Way_Raw"
        if bool(
            diagnostics[
                "Raw_Two_Way_Covariance_Usable"
            ]
        )
        else "Two_Way_PSD_Sensitivity_only_pending_review"
    )

    result[
        "Preferred_Inference_for_Review"
    ] = preferred_inference

    result[
        "Interpretation_Note"
    ] = np.where(
        result[
            "Component"
        ]
        == "Structural_Zero",
        (
            "Positive coefficient or odds ratio above 1 indicates "
            "higher excess-zero odds and therefore lower occurrence "
            "propensity, all else equal."
        ),
        np.where(
            result[
                "Component"
            ]
            == "Count",
            (
                "Positive coefficient or rate ratio above 1 indicates "
                "a higher expected count per ForestPixelCount exposure, "
                "all else equal."
            ),
            (
                "Alpha is the NB1 dispersion parameter and is not "
                "interpreted as an effect ratio."
            ),
        ),
    )

    return result


# =============================================================================
# Covariance-file writer
# =============================================================================

def write_covariance_matrix(
    covariance: np.ndarray,
    parameter_labels: Sequence[str],
    path: Path,
) -> None:
    table = pd.DataFrame(
        covariance,
        index=parameter_labels,
        columns=parameter_labels,
    )

    table.index.name = (
        "Parameter"
    )

    table.to_csv(
        path,
        encoding="utf-8-sig",
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

    COVARIANCE_ROOT.mkdir(
        parents=True,
        exist_ok=False,
    )

    PREDICTION_ROOT.mkdir(
        parents=True,
        exist_ok=False,
    )

    log(
        "Starting full-data fixed-model fitting and "
        "two-way cluster-robust inference."
    )

    log(
        f"Base table: {INPUT_CSV}"
    )

    log(
        f"Step-05 structure: {FINAL_STRUCTURE_CSV}"
    )

    log(
        f"Output: {OUTPUT_ROOT}"
    )

    step04 = load_module(
        STEP04_SCRIPT,
        "step04_final_inference_module",
    )

    step05 = load_module(
        STEP05_SCRIPT,
        "step05_final_inference_module",
    )

    required_step04 = [
        "fit_nb1",
        "single_stage_outputs",
        "prediction_metrics",
    ]

    required_step05 = [
        "fit_zinb_component",
        "zinb_outputs",
    ]

    missing_step04 = [
        name
        for name in required_step04
        if not hasattr(
            step04,
            name,
        )
    ]

    missing_step05 = [
        name
        for name in required_step05
        if not hasattr(
            step05,
            name,
        )
    ]

    if missing_step04:
        raise RuntimeError(
            "Step-04 script is missing required functions: "
            + ", ".join(
                missing_step04
            )
        )

    if missing_step05:
        raise RuntimeError(
            "Step-05 script is missing required functions: "
            + ", ".join(
                missing_step05
            )
        )

    structure = read_final_structure()

    base_data = read_base_table(
        structure
    )

    if PANEL_DECISIONS_CSV.exists():
        panel_decisions = pd.read_csv(
            PANEL_DECISIONS_CSV
        )
    else:
        panel_decisions = None

    qa_rows: List[
        Dict[str, object]
    ] = []

    fit_status_rows: List[
        Dict[str, object]
    ] = []

    metric_rows: List[
        Dict[str, object]
    ] = []

    diagnostic_rows: List[
        Dict[str, object]
    ] = []

    coefficient_tables: List[
        pd.DataFrame
    ] = []

    scaler_tables: List[
        pd.DataFrame
    ] = []

    manifest_rows: List[
        Dict[str, object]
    ] = []

    for definition in structure.itertuples(
        index=False
    ):
        season_code = int(
            definition.Season_Code
        )

        season_label = (
            SEASONS[
                season_code
            ]
        )

        response = str(
            definition.Count_Response
        )

        model_structure = str(
            definition.Model_Structure
        )

        candidate_id = str(
            definition.Candidate_ID
        )

        transformation_scheme = str(
            definition.Transformation_Scheme
        )

        zero_predictors = parse_predictors(
            definition.Structural_Zero_Predictors
        )

        count_predictors = parse_predictors(
            definition.Count_Predictors
        )

        selected_predictors = list(
            dict.fromkeys(
                [
                    *zero_predictors,
                    *count_predictors,
                ]
            )
        )

        log(
            f"Preparing exact Step-05 sample: "
            f"{season_label} {response}."
        )

        (
            model_data,
            source_oof,
        ) = exact_model_sample(
            base_data=base_data,
            season_code=(
                season_code
            ),
            response=response,
            model_structure=(
                model_structure
            ),
            selected_predictors=(
                selected_predictors
            ),
        )

        (
            count_design,
            count_names,
            count_scaler,
        ) = build_full_design(
            frame=model_data,
            predictors=(
                count_predictors
            ),
            scheme=(
                transformation_scheme
            ),
            component=(
                "Count"
            ),
        )

        if model_structure == "ZINB1":
            (
                zero_design,
                zero_names,
                zero_scaler,
            ) = build_full_design(
                frame=model_data,
                predictors=(
                    zero_predictors
                ),
                scheme=(
                    transformation_scheme
                ),
                component=(
                    "Structural_Zero"
                ),
            )
        elif (
            model_structure
            == "Single_Stage_NB1"
        ):
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

            zero_scaler = pd.DataFrame()
        else:
            raise ValueError(
                f"Unexpected model structure: {model_structure}"
            )

        for scaler_table in [
            count_scaler,
            zero_scaler,
        ]:
            if scaler_table.empty:
                continue

            scaler_table = (
                scaler_table.copy()
            )

            scaler_table.insert(
                0,
                "Model_Structure",
                model_structure,
            )

            scaler_table.insert(
                0,
                "Count_Response",
                response,
            )

            scaler_table.insert(
                0,
                "Season_Label",
                season_label,
            )

            scaler_table.insert(
                0,
                "Season_Code",
                season_code,
            )

            scaler_tables.append(
                scaler_table
            )

        observed = (
            model_data[
                response
            ].to_numpy(
                dtype=float
            )
        )

        exposure = (
            model_data[
                "ForestPixelCount"
            ].to_numpy(
                dtype=float
            )
        )

        log(
            f"Fitting full-data {model_structure}: "
            f"{season_label} {response}; "
            f"N={len(model_data):,}."
        )

        if (
            model_structure
            == "Single_Stage_NB1"
        ):
            (
                fit_result,
                fit_status,
            ) = step04.fit_nb1(
                observed,
                count_design,
                exposure,
            )

            parameters = np.asarray(
                fit_result.params,
                dtype=float,
            )

            outputs = (
                step04.single_stage_outputs(
                    fit_result,
                    observed,
                    count_design,
                    exposure,
                )
            )

            inference_model = (
                NegativeBinomial(
                    observed,
                    count_design,
                    loglike_method="nb1",
                    exposure=exposure,
                    missing="raise",
                    check_rank=False,
                )
            )

            fitted_converged = (
                convergence_status(
                    fit_result
                )
            )
        else:
            (
                parameters,
                fit_status,
            ) = (
                step05.fit_zinb_component(
                    step04,
                    observed,
                    count_design,
                    zero_design,
                    exposure,
                )
            )

            parameters = np.asarray(
                parameters,
                dtype=float,
            )

            outputs = (
                step05.zinb_outputs(
                    step04,
                    parameters,
                    observed,
                    count_design,
                    zero_design,
                    exposure,
                )
            )

            inference_model = (
                ZeroInflatedNegativeBinomialP(
                    observed,
                    count_design,
                    exog_infl=(
                        zero_design
                    ),
                    exposure=exposure,
                    inflation="logit",
                    p=1,
                    missing="raise",
                )
            )

            fitted_converged = robust_bool(
                fit_status.get(
                    "Converged",
                    True,
                )
            )

        if not np.isfinite(
            parameters
        ).all():
            raise RuntimeError(
                f"Non-finite full-data parameters for "
                f"{season_label} {response}."
            )

        full_predictions = prediction_frame(
            model_data=model_data,
            response=response,
            model_structure=(
                model_structure
            ),
            outputs=outputs,
        )

        full_prediction_path = (
            PREDICTION_ROOT
            / (
                "Full_Data_Predictions_"
                f"S{season_code}_{response}.csv.gz"
            )
        )

        full_predictions.to_csv(
            full_prediction_path,
            index=False,
            compression="gzip",
        )

        full_metrics = (
            step04.prediction_metrics(
                full_predictions[
                    "Observed_Count"
                ].to_numpy(
                    dtype=float
                ),
                full_predictions[
                    "Predicted_Count"
                ].to_numpy(
                    dtype=float
                ),
                full_predictions[
                    "ForestPixelCount"
                ].to_numpy(
                    dtype=float
                ),
                full_predictions[
                    "Predicted_Zero_Probability"
                ].to_numpy(
                    dtype=float
                ),
                full_predictions[
                    "Predictive_LogProbability"
                ].to_numpy(
                    dtype=float
                ),
            )
        )

        metadata = parameter_metadata(
            model_structure=(
                model_structure
            ),
            zero_names=(
                zero_names
            ),
            count_names=(
                count_names
            ),
        )

        if len(
            metadata
        ) != len(
            parameters
        ):
            raise RuntimeError(
                f"Parameter metadata mismatch for "
                f"{season_label} {response}: "
                f"{len(metadata)} labels versus "
                f"{len(parameters)} estimates."
            )

        log(
            f"Calculating score, Hessian, and clustered covariance: "
            f"{season_label} {response}."
        )

        (
            covariance_matrices,
            diagnostics,
        ) = robust_covariance_bundle(
            model=inference_model,
            parameters=parameters,
            grid_uid=(
                model_data[
                    "GRID_UID"
                ].astype(str)
            ),
            year=(
                model_data[
                    "Year"
                ].astype(int)
            ),
        )

        coefficients = coefficient_table(
            parameters=parameters,
            metadata=metadata,
            covariance_matrices=(
                covariance_matrices
            ),
            diagnostics=diagnostics,
        )

        coefficients.insert(
            0,
            "Candidate_ID",
            candidate_id,
        )

        coefficients.insert(
            0,
            "Transformation_Scheme",
            transformation_scheme,
        )

        coefficients.insert(
            0,
            "Model_Structure",
            model_structure,
        )

        coefficients.insert(
            0,
            "Rate_Scale_Name",
            COUNT_RESPONSES[
                response
            ],
        )

        coefficients.insert(
            0,
            "Count_Response",
            response,
        )

        coefficients.insert(
            0,
            "Season_Label",
            season_label,
        )

        coefficients.insert(
            0,
            "Season_Code",
            season_code,
        )

        coefficient_tables.append(
            coefficients
        )

        parameter_labels = (
            metadata[
                "Parameter_Label"
            ]
            .astype(str)
            .tolist()
        )

        model_prefix = (
            f"S{season_code}_"
            f"{response}"
        )

        for covariance_name, covariance in (
            covariance_matrices.items()
        ):
            write_covariance_matrix(
                covariance=covariance,
                parameter_labels=(
                    parameter_labels
                ),
                path=(
                    COVARIANCE_ROOT
                    / (
                        f"{model_prefix}_"
                        f"{covariance_name}.csv"
                    )
                ),
            )

        qa_rows.append(
            {
                "Season_Code": (
                    season_code
                ),
                "Season_Label": (
                    season_label
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
                "Candidate_ID": (
                    candidate_id
                ),
                "Rows": int(
                    len(
                        model_data
                    )
                ),
                "Unique_GRID_UIDs": int(
                    model_data[
                        "GRID_UID"
                    ].nunique()
                ),
                "Unique_Years": int(
                    model_data[
                        "Year"
                    ].nunique()
                ),
                "Positive_Row_Count": int(
                    (
                        model_data[
                            response
                        ]
                        > 0
                    ).sum()
                ),
                "Observed_Zero_Proportion": float(
                    (
                        model_data[
                            response
                        ]
                        == 0
                    ).mean()
                ),
                "Count_Predictor_Count": int(
                    len(
                        count_predictors
                    )
                ),
                "Structural_Zero_Predictor_Count": int(
                    len(
                        zero_predictors
                    )
                ),
                "Parameter_Count": int(
                    len(
                        parameters
                    )
                ),
                "Exact_OOF_Sample_Reused": (
                    True
                ),
            }
        )

        fit_status_rows.append(
            {
                "Season_Code": (
                    season_code
                ),
                "Season_Label": (
                    season_label
                ),
                "Count_Response": (
                    response
                ),
                "Model_Structure": (
                    model_structure
                ),
                "Candidate_ID": (
                    candidate_id
                ),
                "Converged": (
                    fitted_converged
                ),
                "Valid_Fit": robust_bool(
                    fit_status.get(
                        "Valid_Fit",
                        True,
                    )
                ),
                "Optimizer": fit_status.get(
                    "Optimizer",
                    "",
                ),
                "Attempt_Count": fit_status.get(
                    "Attempt_Count",
                    np.nan,
                ),
                "Failed_Attempt_Count": fit_status.get(
                    "Failed_Attempt_Count",
                    np.nan,
                ),
                "Alpha_Estimate": fit_status.get(
                    "Alpha_Estimate",
                    parameters[
                        -1
                    ],
                ),
                "Stable_LogLikelihood": fit_status.get(
                    "Stable_LogLikelihood",
                    np.sum(
                        np.asarray(
                            outputs[
                                "Predictive_LogProbability"
                            ],
                            dtype=float,
                        )
                    ),
                ),
                "Warning_Text": fit_status.get(
                    "Warning_Text",
                    "",
                ),
                "Hessian_Warning_in_Fit": (
                    warning_contains_hessian(
                        fit_status.get(
                            "Warning_Text",
                            "",
                        )
                    )
                ),
            }
        )

        metric_rows.append(
            {
                "Season_Code": (
                    season_code
                ),
                "Season_Label": (
                    season_label
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
                "Candidate_ID": (
                    candidate_id
                ),
                **full_metrics,
                "Metric_Scope": (
                    "Full-data in-sample fit; not OOF performance"
                ),
            }
        )

        diagnostic_rows.append(
            {
                "Season_Code": (
                    season_code
                ),
                "Season_Label": (
                    season_label
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
                "Candidate_ID": (
                    candidate_id
                ),
                "Fit_Converged": (
                    fitted_converged
                ),
                "Fit_Valid": robust_bool(
                    fit_status.get(
                        "Valid_Fit",
                        True,
                    )
                ),
                "Fit_Hessian_Warning": (
                    warning_contains_hessian(
                        fit_status.get(
                            "Warning_Text",
                            "",
                        )
                    )
                ),
                **diagnostics,
                "Inference_Status": (
                    "Raw two-way clustered covariance usable"
                    if diagnostics[
                        "Raw_Two_Way_Covariance_Usable"
                    ]
                    else (
                        "Raw two-way covariance requires review; "
                        "PSD-adjusted sensitivity is provided but "
                        "must not be treated as final without review"
                    )
                ),
            }
        )

        manifest_rows.append(
            {
                "Season_Code": (
                    season_code
                ),
                "Season_Label": (
                    season_label
                ),
                "Count_Response": (
                    response
                ),
                "Model_Structure": (
                    model_structure
                ),
                "Source_OOF_File": (
                    selected_oof_path(
                        season_code,
                        response,
                    ).name
                ),
                "Source_OOF_Rows": int(
                    len(
                        source_oof
                    )
                ),
                "Full_Data_Prediction_File": (
                    full_prediction_path.name
                ),
                "Full_Data_Prediction_Rows": int(
                    len(
                        full_predictions
                    )
                ),
            }
        )

        log(
            f"Completed: {season_label} {response}; "
            f"converged={fitted_converged}; "
            f"raw_two_way_usable="
            f"{diagnostics['Raw_Two_Way_Covariance_Usable']}."
        )

    qa = pd.DataFrame(
        qa_rows
    )

    fit_status_table = pd.DataFrame(
        fit_status_rows
    )

    metrics = pd.DataFrame(
        metric_rows
    )

    diagnostics_table = pd.DataFrame(
        diagnostic_rows
    )

    coefficients = pd.concat(
        coefficient_tables,
        ignore_index=True,
    )

    if scaler_tables:
        scalers = pd.concat(
            scaler_tables,
            ignore_index=True,
        )
    else:
        scalers = pd.DataFrame(
            columns=[
                "Season_Code",
                "Season_Label",
                "Count_Response",
                "Model_Structure",
                "Component",
                "Variable",
                "Transformation_Scheme",
                "Log1p_Applied",
                "Transformed_Mean",
                "Transformed_Scale",
            ]
        )

    manifest = pd.DataFrame(
        manifest_rows
    )

    publication_readiness = (
        diagnostics_table[
            [
                "Season_Code",
                "Season_Label",
                "Count_Response",
                "Rate_Scale_Name",
                "Model_Structure",
                "Candidate_ID",
                "Fit_Converged",
                "Fit_Valid",
                "Fit_Hessian_Warning",
                "Score_Sum_Within_Tolerance",
                "Information_Full_Rank",
                "Information_Condition_Number",
                "Bread_Inversion_Method",
                "Raw_Two_Way_Covariance_Usable",
                "PSD_Sensitivity_Required",
                "Inference_Status",
            ]
        ]
        .copy()
    )

    publication_readiness[
        "Ready_for_Direct_Coefficient_Interpretation"
    ] = (
        publication_readiness[
            "Fit_Converged"
        ].astype(bool)
        & publication_readiness[
            "Fit_Valid"
        ].astype(bool)
        & publication_readiness[
            "Score_Sum_Within_Tolerance"
        ].astype(bool)
        & publication_readiness[
            "Raw_Two_Way_Covariance_Usable"
        ].astype(bool)
    )

    publication_readiness[
        "Decision_Note"
    ] = np.where(
        publication_readiness[
            "Ready_for_Direct_Coefficient_Interpretation"
        ],
        (
            "The raw two-way clustered covariance is numerically usable. "
            "Coefficient interpretation can proceed after the stable-likelihood "
            "and identifiability checks are completed."
        ),
        (
            "Do not report p-values or confidence intervals as final until "
            "the numerical issue is resolved by the stable-likelihood and "
            "identifiability audit."
        ),
    )

    global_qa = pd.DataFrame(
        [
            (
                "Expected_model_count",
                6,
            ),
            (
                "Observed_model_count",
                len(
                    qa
                ),
            ),
            (
                "All_models_converged",
                bool(
                    fit_status_table[
                        "Converged"
                    ].astype(bool).all()
                ),
            ),
            (
                "All_models_valid",
                bool(
                    fit_status_table[
                        "Valid_Fit"
                    ].astype(bool).all()
                ),
            ),
            (
                "Raw_two_way_covariance_usable_model_count",
                int(
                    diagnostics_table[
                        "Raw_Two_Way_Covariance_Usable"
                    ].astype(bool).sum()
                ),
            ),
            (
                "PSD_sensitivity_required_model_count",
                int(
                    diagnostics_table[
                        "PSD_Sensitivity_Required"
                    ].astype(bool).sum()
                ),
            ),
            (
                "Exact_Step05_OOF_samples_reused",
                True,
            ),
            (
                "Mean_structure_type",
                "Fixed NB1/ZINB1",
            ),
            (
                "Fixed_effect_structures_changed",
                False,
            ),
            (
                "Primary_inference_adjustment",
                "GRID_UID and Year two-way clustered covariance",
            ),
            (
                "Smallest_cluster_dimension_df",
                24,
            ),
        ],
        columns=[
            "Metric",
            "Value",
        ],
    )

    method = {
        "workflow_stage":
            "Full-data fixed-effect fitting and two-way clustered inference",
        "source_structure":
            str(
                FINAL_STRUCTURE_CSV
            ),
        "exact_sample_source":
            "Step-05 selected temporal OOF prediction files",
        "fixed_effect_structures_changed":
            False,
        "mean_structure_type":
            "Fixed NB1/ZINB1",
        "country_effect":
            "Fixed effect; China reference",
        "exposure":
            "ForestPixelCount",
        "equivalent_offset":
            "log(ForestPixelCount)",
        "continuous_predictor_scale":
            "Standardized after the Step-05 transformation",
        "cluster_covariance": {
            "dimensions": [
                "GRID_UID",
                "Year",
            ],
            "formula":
                "Cov(GRID_UID) + Cov(Year) - Cov(GRID_UID x Year)",
            "finite_sample_correction":
                "CR1-type correction for each cluster dimension",
            "inference_distribution":
                "Student t",
            "degrees_of_freedom":
                24,
        },
        "raw_covariance_policy":
            "Use raw two-way covariance only when finite, positive "
            "on the diagonal, and positive semidefinite within tolerance.",
        "psd_sensitivity_policy":
            "Eigenvalue-clipped PSD covariance is diagnostic sensitivity "
            "only and is not automatically treated as final inference.",
        "next_step":
            "Run the stable-likelihood and parameter-identifiability audit.",
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

    qa.to_csv(
        OUTPUT_ROOT
        / "02_Model_Data_QA.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fit_status_table.to_csv(
        OUTPUT_ROOT
        / "03_Full_Data_Fit_Status.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metrics.to_csv(
        OUTPUT_ROOT
        / "04_Full_Data_In_Sample_Metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    diagnostics_table.to_csv(
        OUTPUT_ROOT
        / "05_Hessian_Score_and_Covariance_Diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    coefficients.to_csv(
        OUTPUT_ROOT
        / "06_All_Coefficients_and_Clustered_Inference.csv",
        index=False,
        encoding="utf-8-sig",
    )

    publication_readiness.to_csv(
        OUTPUT_ROOT
        / "07_Model_Inference_Readiness.csv",
        index=False,
        encoding="utf-8-sig",
    )

    scalers.to_csv(
        OUTPUT_ROOT
        / "08_Continuous_Predictor_Transformation_and_Scaling.csv",
        index=False,
        encoding="utf-8-sig",
    )

    manifest.to_csv(
        OUTPUT_ROOT
        / "09_Full_Data_Prediction_Manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )

    structure.to_csv(
        OUTPUT_ROOT
        / "11_Fixed_Effect_Structures_Held_Constant.csv",
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
        "Full-data fixed-model and two-way clustered inference "
        "completed successfully."
    )

    for row in publication_readiness.itertuples(
        index=False
    ):
        log(
            f"Readiness: {row.Season_Label} "
            f"{row.Count_Response}; "
            f"raw_two_way_usable="
            f"{row.Raw_Two_Way_Covariance_Usable}; "
            f"direct_interpretation_ready="
            f"{row.Ready_for_Direct_Coefficient_Interpretation}."
        )

    log(
        "Next step: review output 07, then run the stable-likelihood and "
        "parameter-identifiability audit."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
