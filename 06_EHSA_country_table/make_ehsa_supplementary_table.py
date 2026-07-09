# -*- coding: utf-8 -*-
"""
Merge Fire Count and Burned Area EHSA country-composition tables.

No ArcPy is required.
Only Python's standard csv module is used.

Outputs:
1. Compact table: each cell is n (percentage%)
2. Detailed table: count and percentage in separate columns
"""

import csv
from pathlib import Path


# ============================================================
# 1. PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_DIR.parent
INPUT_DIR = CODE_ROOT / "00_Input_data"

FC_CSV = INPUT_DIR / "FC_EHSA_all17_country_percentages_wide.csv"
BA_CSV = INPUT_DIR / "BA_EHSA_all17_country_percentages_wide.csv"

OUT_DIR = SCRIPT_DIR / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COMPACT_CSV = OUT_DIR / "Table_S5_EHSA_counts_percentages_FC_BA_nonzero_categories.csv"
DETAILED_CSV = OUT_DIR / "Table_S5_EHSA_counts_percentages_FC_BA_nonzero_categories_detailed.csv"

PERCENT_DECIMALS = 2


# ============================================================
# 2. ORDER
# ============================================================

COUNTRIES = ["China", "North Korea", "Russia"]

EHSA_CATEGORY_ORDER = [
    "New Hot Spot",
    "Consecutive Hot Spot",
    "Intensifying Hot Spot",
    "Persistent Hot Spot",
    "Diminishing Hot Spot",
    "Sporadic Hot Spot",
    "Oscillating Hot Spot",
    "Historical Hot Spot",
    "New Cold Spot",
    "Consecutive Cold Spot",
    "Intensifying Cold Spot",
    "Persistent Cold Spot",
    "Diminishing Cold Spot",
    "Sporadic Cold Spot",
    "Oscillating Cold Spot",
    "Historical Cold Spot",
    "No Pattern Detected",
]


# ============================================================
# 3. READ INPUT CSV FILES
# ============================================================

def read_input_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Input file not found:\n{path}")

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames:
            raise ValueError(f"No header found in:\n{path}")

        data = {}

        for row in reader:
            category = row["EHSA_Category"].strip()

            if not category:
                continue

            data[category] = {}

            for country in COUNTRIES:
                data[category][country] = {
                    "count": int(round(float(row[f"{country}_Count"] or 0))),
                    "percentage": float(row[f"{country}_Percentage"] or 0),
                }

    # Missing categories are treated as zero.
    for category in EHSA_CATEGORY_ORDER:
        if category not in data:
            data[category] = {
                country: {
                    "count": 0,
                    "percentage": 0.0,
                }
                for country in COUNTRIES
            }

    return data


# ============================================================
# 4. OUTPUT
# ============================================================

def format_value(count, percentage):
    return f"{count:,} ({percentage:.{PERCENT_DECIMALS}f}%)"


def calculate_total(dataset, country):
    return sum(
        dataset[category][country]["count"]
        for category in EHSA_CATEGORY_ORDER
    )


def get_display_categories(fc, ba):
    """
    Keep a category if at least one count is greater than zero
    across FC/BA and China/North Korea/Russia.

    Categories that are zero everywhere are omitted.
    """
    displayed = []

    for category in EHSA_CATEGORY_ORDER:
        total_count = 0

        for dataset in (fc, ba):
            for country in COUNTRIES:
                total_count += dataset[category][country]["count"]

        if total_count > 0:
            displayed.append(category)

    return displayed


def write_compact_table(fc, ba, display_categories):
    with COMPACT_CSV.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.writer(f)

        # Two-level header.
        writer.writerow([
            "EHSA category",
            "Fire Count (FC)",
            "",
            "",
            "Burned Area (BA)",
            "",
            "",
        ])

        writer.writerow([
            "",
            "China",
            "North Korea",
            "Russia",
            "China",
            "North Korea",
            "Russia",
        ])

        for category in display_categories:
            row = [category]

            for dataset in (fc, ba):
                for country in COUNTRIES:
                    item = dataset[category][country]

                    row.append(
                        format_value(
                            item["count"],
                            item["percentage"],
                        )
                    )

            writer.writerow(row)

        total_row = ["Total"]

        for dataset in (fc, ba):
            for country in COUNTRIES:
                total_row.append(
                    format_value(
                        calculate_total(dataset, country),
                        100.0,
                    )
                )

        writer.writerow(total_row)

        writer.writerow([])
        writer.writerow([
            "Note:",
            (
                "Values are the number of grid cells, with percentages "
                "in parentheses. Percentages were calculated using all "
                "EHSA grid cells within each country, including "
                "No Pattern Detected."
            ),
        ])


def write_detailed_table(fc, ba, display_categories):
    with DETAILED_CSV.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.writer(f)

        header = ["EHSA category"]

        for metric in ("FC", "BA"):
            for country in COUNTRIES:
                header.extend([
                    f"{metric}—{country} count",
                    f"{metric}—{country} percentage (%)",
                ])

        writer.writerow(header)

        for category in display_categories:
            row = [category]

            for dataset in (fc, ba):
                for country in COUNTRIES:
                    item = dataset[category][country]

                    row.extend([
                        item["count"],
                        round(
                            item["percentage"],
                            PERCENT_DECIMALS,
                        ),
                    ])

            writer.writerow(row)

        total_row = ["Total"]

        for dataset in (fc, ba):
            for country in COUNTRIES:
                total_row.extend([
                    calculate_total(dataset, country),
                    100.0,
                ])

        writer.writerow(total_row)


def main():
    print("=" * 80)
    print("Reading EHSA country-composition CSV files")
    print("=" * 80)

    fc = read_input_csv(FC_CSV)
    ba = read_input_csv(BA_CSV)

    display_categories = get_display_categories(
        fc,
        ba,
    )

    omitted_categories = [
        category
        for category in EHSA_CATEGORY_ORDER
        if category not in display_categories
    ]

    write_compact_table(
        fc,
        ba,
        display_categories,
    )

    write_detailed_table(
        fc,
        ba,
        display_categories,
    )

    print("\nDisplayed categories:")
    for category in display_categories:
        print(f"  {category}")

    if omitted_categories:
        print("\nOmitted because all FC/BA country counts were zero:")
        for category in omitted_categories:
            print(f"  {category}")
    else:
        print("\nNo all-zero categories were omitted.")

    print("\nCompleted successfully.")
    print(f"Compact supplementary table:\n{COMPACT_CSV}")
    print(f"Detailed verification table:\n{DETAILED_CSV}")


if __name__ == "__main__":
    main()
