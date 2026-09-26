"""
SMART INVENTORY MANAGEMENT & DEMAND FORECASTING SYSTEM
Feature Engineering - Step 2

Purpose:
    Create historical lag features from Units Sold.

Input:
    feature_engineered_step1.csv

Target for future forecasting model:
    Units Sold

Important:
    - Lag features are created separately for each Store ID + Product ID.
    - No ML model is trained in this step.
    - Existing Demand Forecast is NOT used as the model target.
    - Original input file is never modified.
"""

from pathlib import Path
import sys

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(r"C:\SIM&DFS")

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "feature_engineered_step1.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "feature_engineered_step2.csv"
)

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

CHART_FILE = (
    FIGURES_DIR
    / "feature_engineering_step2_lag_features.png"
)


# ============================================================
# LAG FEATURES
# ============================================================

LAG_PERIODS = [1, 7, 14, 30]

NEW_FEATURES = [
    f"Units_Sold_Lag_{lag}"
    for lag in LAG_PERIODS
]


# ============================================================
# HELPER FUNCTION
# ============================================================

def print_section(title):
    """Print a formatted section heading."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    print_section(
        "SMART INVENTORY MANAGEMENT & DEMAND FORECASTING SYSTEM\n"
        "FEATURE ENGINEERING - STEP 2: HISTORICAL LAG FEATURES"
    )

    # --------------------------------------------------------
    # 1. INPUT FILE CHECK
    # --------------------------------------------------------

    print("\n--- INPUT FILE CHECK ---")
    print(f"Input file: {INPUT_FILE}")

    if not INPUT_FILE.exists():
        print("\nERROR: Input dataset was not found.")
        print(f"Expected path: {INPUT_FILE}")
        sys.exit(1)

    print("Input dataset found successfully.")

    # --------------------------------------------------------
    # 2. CREATE OUTPUT DIRECTORIES
    # --------------------------------------------------------

    try:
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print("\nERROR: Could not create required directories.")
        print(f"Details: {e}")
        sys.exit(1)

    # --------------------------------------------------------
    # 3. LOAD DATASET
    # --------------------------------------------------------

    print_section("1. LOADING FEATURE-ENGINEERED STEP 1 DATASET")

    try:
        df = pd.read_csv(INPUT_FILE)
    except Exception as e:
        print("\nERROR: Could not load the input CSV file.")
        print(f"Details: {e}")
        sys.exit(1)

    rows_before, columns_before = df.shape

    print(f"Rows before feature engineering    : {rows_before:,}")
    print(f"Columns before feature engineering : {columns_before}")

    # --------------------------------------------------------
    # 4. REQUIRED COLUMN CHECK
    # --------------------------------------------------------

    print_section("2. REQUIRED COLUMN VALIDATION")

    required_columns = [
        "Date",
        "Store ID",
        "Product ID",
        "Units Sold",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        print("ERROR: Required columns are missing:")

        for column in missing_columns:
            print(f"  - {column}")

        sys.exit(1)

    print("All required columns are present.")

    print("\nRequired columns:")
    for column in required_columns:
        print(f"  ✓ {column}")

    # --------------------------------------------------------
    # 5. DATE VALIDATION
    # --------------------------------------------------------

    print_section("3. DATE VALIDATION")

    try:
        df["Date"] = pd.to_datetime(
            df["Date"],
            errors="coerce"
        )
    except Exception as e:
        print("\nERROR: Date conversion failed.")
        print(f"Details: {e}")
        sys.exit(1)

    invalid_dates = df["Date"].isna().sum()

    print(f"Invalid Date values: {invalid_dates}")

    if invalid_dates > 0:
        print(
            "\nERROR: Invalid dates were found. "
            "Lag features will not be created."
        )
        sys.exit(1)

    print("Date validation passed.")

    # --------------------------------------------------------
    # 6. UNITS SOLD VALIDATION
    # --------------------------------------------------------

    print_section("4. TARGET COLUMN VALIDATION")

    try:
        df["Units Sold"] = pd.to_numeric(
            df["Units Sold"],
            errors="coerce"
        )
    except Exception as e:
        print("\nERROR: Units Sold conversion failed.")
        print(f"Details: {e}")
        sys.exit(1)

    invalid_units_sold = df["Units Sold"].isna().sum()

    print(f"Invalid Units Sold values: {invalid_units_sold}")

    if invalid_units_sold > 0:
        print(
            "\nERROR: Invalid/missing Units Sold values found."
        )
        sys.exit(1)

    print("Units Sold validation passed.")

    # --------------------------------------------------------
    # 7. CHECK STORE-PRODUCT COMBINATIONS
    # --------------------------------------------------------

    print_section("5. STORE-PRODUCT SERIES INSPECTION")

    unique_stores = df["Store ID"].nunique()
    unique_products = df["Product ID"].nunique()

    store_product_combinations = (
        df[["Store ID", "Product ID"]]
        .drop_duplicates()
        .shape[0]
    )

    print(f"Unique Stores              : {unique_stores}")
    print(f"Unique Products            : {unique_products}")
    print(
        f"Unique Store-Product pairs : "
        f"{store_product_combinations}"
    )

    # --------------------------------------------------------
    # 8. SORT DATA CORRECTLY
    # --------------------------------------------------------

    print_section("6. SORTING DATA FOR TIME-SERIES LAGS")

    print(
        "Sorting by Store ID → Product ID → Date..."
    )

    try:
        df = df.sort_values(
            by=["Store ID", "Product ID", "Date"],
            kind="stable"
        ).reset_index(drop=True)
    except Exception as e:
        print("\nERROR: Dataset sorting failed.")
        print(f"Details: {e}")
        sys.exit(1)

    print("Sorting completed successfully.")

    # --------------------------------------------------------
    # 9. CREATE LAG FEATURES
    # --------------------------------------------------------

    print_section("7. CREATING HISTORICAL LAG FEATURES")

    print(
        "\nLag features will be calculated independently "
        "for each Store ID + Product ID."
    )

    try:
        grouped_units_sold = df.groupby(
            ["Store ID", "Product ID"],
            sort=False
        )["Units Sold"]

        for lag in LAG_PERIODS:

            feature_name = f"Units_Sold_Lag_{lag}"

            df[feature_name] = grouped_units_sold.shift(lag)

            print(
                f"Created: {feature_name:<22}"
                f" → previous {lag} record(s)"
            )

    except Exception as e:
        print("\nERROR: Lag feature creation failed.")
        print(f"Details: {e}")
        sys.exit(1)

    # --------------------------------------------------------
    # 10. VERIFY NEW FEATURES
    # --------------------------------------------------------

    print_section("8. NEW FEATURE VALIDATION")

    missing_features = [
        feature
        for feature in NEW_FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        print("ERROR: Expected lag features are missing:")

        for feature in missing_features:
            print(f"  - {feature}")

        sys.exit(1)

    print("All lag features were created successfully.")

    print("\nNewly created features:")

    for feature in NEW_FEATURES:
        print(f"  ✓ {feature}")

    # --------------------------------------------------------
    # 11. LAG MISSING VALUES
    # --------------------------------------------------------

    print_section("9. LAG FEATURE MISSING VALUE ANALYSIS")

    print(
        "Initial rows in each Store-Product series naturally "
        "have missing lag values because historical records "
        "do not exist before the first observation."
    )

    for feature in NEW_FEATURES:

        missing_count = df[feature].isna().sum()

        missing_percentage = (
            missing_count / len(df) * 100
        )

        print(
            f"{feature:<22}: "
            f"{missing_count:,} missing "
            f"({missing_percentage:.2f}%)"
        )

    # --------------------------------------------------------
    # 12. FIRST 20 ROWS PREVIEW
    # --------------------------------------------------------

    print_section("10. LAG FEATURE PREVIEW")

    preview_columns = [
        "Date",
        "Store ID",
        "Product ID",
        "Units Sold",
    ] + NEW_FEATURES

    print(
        df[preview_columns]
        .head(20)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # 13. LAG STATISTICS
    # --------------------------------------------------------

    print_section("11. LAG FEATURE STATISTICS")

    print(
        df[NEW_FEATURES]
        .describe()
        .round(2)
        .to_string()
    )

    # --------------------------------------------------------
    # 14. CHECK ROW COUNT
    # --------------------------------------------------------

    rows_after, columns_after = df.shape

    print_section("12. DATASET SIZE VALIDATION")

    print(f"Rows before : {rows_before:,}")
    print(f"Rows after  : {rows_after:,}")

    print(f"Columns before : {columns_before}")
    print(f"Columns after  : {columns_after}")

    if rows_before != rows_after:
        print(
            "\nERROR: Row count changed unexpectedly."
        )
        sys.exit(1)

    expected_columns_after = (
        columns_before + len(NEW_FEATURES)
    )

    if columns_after != expected_columns_after:

        print(
            "\nERROR: Unexpected number of columns."
        )

        print(
            f"Expected columns: {expected_columns_after}"
        )

        print(
            f"Actual columns  : {columns_after}"
        )

        sys.exit(1)

    print("\nDataset size validation passed.")

    # --------------------------------------------------------
    # 15. CREATE ONE PROFESSIONAL CHART
    # --------------------------------------------------------

    print_section("13. CREATING LAG FEATURE CHART")

    try:

        # Select one Store-Product series that actually exists.
        first_store = df["Store ID"].iloc[0]
        first_product = df["Product ID"].iloc[0]

        series_df = df[
            (df["Store ID"] == first_store)
            & (df["Product ID"] == first_product)
        ].copy()

        series_df = series_df.sort_values("Date")

        # Plot a limited number of observations for readability.
        plot_df = series_df.head(60)

        plt.figure(figsize=(14, 7))

        plt.plot(
            plot_df["Date"],
            plot_df["Units Sold"],
            marker="o",
            linewidth=2,
            label="Actual Units Sold"
        )

        plt.plot(
            plot_df["Date"],
            plot_df["Units_Sold_Lag_1"],
            linestyle="--",
            linewidth=1.5,
            label="Lag 1"
        )

        plt.title(
            "Historical Units Sold and Lag-1 Feature\n"
            "Feature Engineering Step 2",
            fontsize=14,
            fontweight="bold"
        )

        plt.xlabel("Date")
        plt.ylabel("Units Sold")

        plt.legend()

        plt.grid(
            True,
            alpha=0.3
        )

        plt.xticks(
            rotation=45
        )

        plt.tight_layout()

        plt.savefig(
            CHART_FILE,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    except Exception as e:

        print("\nERROR: Chart creation failed.")
        print(f"Details: {e}")
        sys.exit(1)

    print(
        f"Chart saved successfully:\n"
        f"{CHART_FILE}"
    )

    # --------------------------------------------------------
    # 16. SAVE OUTPUT DATASET
    # --------------------------------------------------------

    print_section(
        "14. SAVING FEATURE-ENGINEERED STEP 2 DATASET"
    )

    try:

        df.to_csv(
            OUTPUT_FILE,
            index=False
        )

    except Exception as e:

        print(
            "\nERROR: Could not save output dataset."
        )

        print(f"Details: {e}")

        sys.exit(1)

    print(
        "Feature-engineered Step 2 dataset saved successfully:"
    )

    print(OUTPUT_FILE)

    # --------------------------------------------------------
    # 17. FINAL SUMMARY
    # --------------------------------------------------------

    print_section(
        "FEATURE ENGINEERING STEP 2 - FINAL SUMMARY"
    )

    print("Status: SUCCESS")

    print("\nInput dataset:")
    print(INPUT_FILE)

    print("\nOutput dataset:")
    print(OUTPUT_FILE)

    print("\nChart:")
    print(CHART_FILE)

    print(f"\nRows:")
    print(f"{rows_after:,}")

    print(f"\nColumns:")
    print(columns_after)

    print("\nNew lag features created:")

    for feature in NEW_FEATURES:
        print(f"  ✓ {feature}")

    print("\nLag grouping:")
    print("  Store ID + Product ID")

    print("\nFuture forecasting target:")
    print("  Units Sold")

    print("\nExisting Demand Forecast:")
    print("  Kept as an existing dataset column.")
    print("  NOT used as our model prediction.")

    print("\nML Model:")
    print("  NOT trained in Feature Engineering Step 2.")

    print("\nOriginal Step 1 dataset:")
    print("  NOT modified.")

    print(
        "\nFeature Engineering Step 2 "
        "completed successfully."
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()