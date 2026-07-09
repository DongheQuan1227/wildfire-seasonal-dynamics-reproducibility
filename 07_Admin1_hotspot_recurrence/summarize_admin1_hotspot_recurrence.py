# -*- coding: utf-8 -*-
"""
Summarize annual Getis-Ord Gi* hotspot recurrence by first-level
administrative region.

Input
-----
admin1_annual_gistar_class_counts_2001_2025.csv

Output
------
output/admin1_hotspot_recurrence_summary_2001_2025.csv

The summary includes hotspot-year frequency, longest consecutive period,
mean and peak hotspot extent, Mann-Kendall trend statistics, and Sen's slope.
"""

from pathlib import Path
from collections import Counter
import math
import statistics
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_DIR.parent
INPUT_DIR = CODE_ROOT / "00_Input_data"
INPUT = INPUT_DIR / "admin1_annual_gistar_class_counts_2001_2025.csv"
OUTPUT_DIR = SCRIPT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
YEARS = list(range(2001, 2026))


def mann_kendall_test(years, values):
    pairs = [
        (int(year), float(value))
        for year, value in zip(years, values)
        if value is not None and math.isfinite(float(value))
    ]

    n = len(pairs)
    if n < 3:
        return None, None, None, None

    x_years = [item[0] for item in pairs]
    x_values = [item[1] for item in pairs]

    s_value = 0
    slopes = []

    for i in range(n - 1):
        for j in range(i + 1, n):
            difference = x_values[j] - x_values[i]
            if difference > 0:
                s_value += 1
            elif difference < 0:
                s_value -= 1

            year_difference = x_years[j] - x_years[i]
            if year_difference != 0:
                slopes.append(difference / year_difference)

    denominator = n * (n - 1) / 2.0
    tau = s_value / denominator if denominator > 0 else 0.0

    tie_counts = Counter(round(value, 12) for value in x_values)
    tie_term = sum(
        count * (count - 1) * (2 * count + 5)
        for count in tie_counts.values()
        if count > 1
    )

    variance = (
        n * (n - 1) * (2 * n + 5) - tie_term
    ) / 18.0

    if variance <= 0:
        z_value = 0.0
        p_value = 1.0
    else:
        standard_error = math.sqrt(variance)
        if s_value > 0:
            z_value = (s_value - 1) / standard_error
        elif s_value < 0:
            z_value = (s_value + 1) / standard_error
        else:
            z_value = 0.0

        p_value = math.erfc(abs(z_value) / math.sqrt(2.0))

    sen_slope = statistics.median(slopes) if slopes else 0.0
    return tau, z_value, p_value, sen_slope


def trend_interpretation(sen_slope, p_value):
    if sen_slope is None or p_value is None:
        return "Insufficient data"
    if p_value < 0.05:
        if sen_slope > 0:
            return "Significant increase"
        if sen_slope < 0:
            return "Significant decrease"
    return "No significant trend"


def longest_consecutive_period(hotspot_years):
    sorted_years = sorted(set(hotspot_years))
    if not sorted_years:
        return "", 0

    periods = []
    start = sorted_years[0]
    previous = sorted_years[0]

    for year in sorted_years[1:]:
        if year == previous + 1:
            previous = year
            continue
        periods.append((start, previous))
        start = year
        previous = year

    periods.append((start, previous))
    maximum_length = max(end - begin + 1 for begin, end in periods)

    labels = []
    for begin, end in periods:
        if end - begin + 1 != maximum_length:
            continue
        if begin == end:
            labels.append(f"{begin} (1 year)")
        else:
            labels.append(f"{begin}–{end} ({end - begin + 1} years)")

    return "; ".join(labels), maximum_length


def summarize_metric(metric, metric_data):
    summary_rows = []

    for (country, region), region_data in metric_data.groupby(
        ["Country", "Administrative region"], sort=False
    ):
        valid_years = []
        hotspot_years = []
        hotspot_99_years = []
        extent_years = []
        extent_values = []

        for year in YEARS:
            annual = region_data[region_data["Year"] == year]
            if annual.empty:
                continue

            total_values = annual["Total valid cells"].unique()
            if len(total_values) != 1:
                raise ValueError(
                    f"Inconsistent total cells: {metric}/{country}/{region}/{year}"
                )

            total_cells = int(total_values[0])
            hotspot_cells = int(
                annual.loc[annual["HS_BIN"] > 0, "Count"].sum()
            )
            hotspot_99_cells = int(
                annual.loc[annual["HS_BIN"] == 3, "Count"].sum()
            )

            if total_cells <= 0:
                continue

            hotspot_extent = hotspot_cells / total_cells * 100.0
            valid_years.append(year)
            extent_years.append(year)
            extent_values.append(hotspot_extent)

            if hotspot_cells > 0:
                hotspot_years.append(year)
            if hotspot_99_cells > 0:
                hotspot_99_years.append(year)

        years_analyzed = len(valid_years)
        hotspot_year_n = len(hotspot_years)
        hotspot_99_year_n = len(hotspot_99_years)

        longest_period, longest_length = longest_consecutive_period(
            hotspot_years
        )

        mean_extent = (
            statistics.mean(extent_values) if extent_values else 0.0
        )

        if extent_values:
            peak_extent = max(extent_values)
            if math.isclose(
                peak_extent, 0.0, rel_tol=0.0, abs_tol=1e-12
            ):
                peak_year_text = ""
            else:
                peak_year_text = ", ".join(
                    str(year)
                    for year, extent in zip(extent_years, extent_values)
                    if math.isclose(
                        extent,
                        peak_extent,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                )
        else:
            peak_extent = 0.0
            peak_year_text = ""

        mk_tau, mk_z, mk_p, sen_slope = mann_kendall_test(
            extent_years, extent_values
        )

        summary_rows.append({
            "Metric": metric,
            "Country": country,
            "Administrative region": region,
            "Years analyzed": years_analyzed,
            "Hotspot years n": hotspot_year_n,
            "Hotspot years percentage": (
                hotspot_year_n / years_analyzed * 100.0
                if years_analyzed else 0.0
            ),
            "99% hotspot years n": hotspot_99_year_n,
            "99% hotspot years percentage": (
                hotspot_99_year_n / years_analyzed * 100.0
                if years_analyzed else 0.0
            ),
            "First hotspot year": (
                min(hotspot_years) if hotspot_years else ""
            ),
            "Last hotspot year": (
                max(hotspot_years) if hotspot_years else ""
            ),
            "Longest consecutive hotspot period": longest_period,
            "Longest consecutive hotspot years": longest_length,
            "Mean annual hotspot extent (%)": mean_extent,
            "Peak hotspot extent (%)": peak_extent,
            "Peak year": peak_year_text,
            "Mann-Kendall tau": mk_tau,
            "Mann-Kendall Z": mk_z,
            "Sen slope (percentage points/year)": sen_slope,
            "p-value": mk_p,
            "Trend": trend_interpretation(sen_slope, mk_p),
        })

    ranked = sorted(
        summary_rows,
        key=lambda row: (
            -row["Mean annual hotspot extent (%)"],
            -row["99% hotspot years n"],
            -row["Hotspot years n"],
            -row["Peak hotspot extent (%)"],
            row["Country"].lower(),
            row["Administrative region"].lower(),
        ),
    )

    for rank, row in enumerate(ranked, start=1):
        row["Rank"] = rank

    return ranked


def main():
    data = pd.read_csv(INPUT, encoding="utf-8-sig")

    required = {
        "Metric", "Country", "Administrative region", "Year",
        "HS_BIN", "Count", "Total valid cells",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    all_rows = []
    for metric in ["FC", "BA"]:
        all_rows.extend(
            summarize_metric(metric, data[data["Metric"] == metric])
        )

    columns = [
        "Rank", "Metric", "Country", "Administrative region",
        "Years analyzed", "Hotspot years n", "Hotspot years percentage",
        "99% hotspot years n", "99% hotspot years percentage",
        "First hotspot year", "Last hotspot year",
        "Longest consecutive hotspot period",
        "Longest consecutive hotspot years",
        "Mean annual hotspot extent (%)",
        "Peak hotspot extent (%)", "Peak year",
        "Mann-Kendall tau", "Mann-Kendall Z",
        "Sen slope (percentage points/year)", "p-value", "Trend",
    ]

    output_data = pd.DataFrame(all_rows)[columns]

    for column in [
        "Mean annual hotspot extent (%)",
        "Peak hotspot extent (%)",
        "Mann-Kendall tau",
        "Mann-Kendall Z",
        "Sen slope (percentage points/year)",
    ]:
        output_data[column] = output_data[column].round(4)

    output_data["p-value"] = output_data["p-value"].round(6)

    output = (
        OUTPUT_DIR
        / "admin1_hotspot_recurrence_summary_2001_2025.csv"
    )
    output_data.to_csv(output, index=False, encoding="utf-8-sig")
    print(output)


if __name__ == "__main__":
    main()
