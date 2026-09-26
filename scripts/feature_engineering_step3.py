"""
SMART INVENTORY MANAGEMENT & DEMAND FORECASTING SYSTEM
Feature Engineering - Step 3

Purpose:
    Create leakage-safe rolling demand features from Units Sold.

Input:
    feature_engineered_step2.csv

Target for future forecasting model:
    Units Sold

Important:
    - Rolling features use ONLY historical observations.
    - Current day's Units Sold is excluded using shift(1).
    - Features are calculated independently for each Store ID + Product ID.
    - No ML model is trained in this step.
    - Existing Demand Forecast is NOT used as our model prediction.
    - Original Step 2 dataset is never modified.
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
    / "feature_engineered_step2.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "feature_engineered_step3.csv"
)

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

CHART_FILE = (
    FIGURES_DIR
    / "feature_engineering_step3_rolling_features.png"
)


# ============================================================
# ROLLING WINDOWS
# ============================================================

ROLLING_WINDOWS = [7, 14, 30]

NEW_FEATURES = []

for window in ROLLING_WINDOWS:
    NEW_FEATURES.append(
        f"Units_Sold_Rolling_Mean_{window}"
    )

    NEW_FEATURES.append(
        f"Units_Sold_Rolling_Std_{window}"
    )


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
        "FEATURE ENGINEERING - STEP 3: LEAKAGE-SAFE ROLLING FEATURES"
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
    # 2. CREATE REQUIRED DIRECTORIES
    # --------------------------------------------------------

    try:
        OUTPUT_FILE.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        FIGURES_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

    except Exception as e:

        print(
            "\nERROR: Could not create required directories."
        )

        print(f"Details: {e}")

        sys.exit(1)

    # --------------------------------------------------------
    # 3. LOAD DATASET
    # --------------------------------------------------------

    print_section(
        "1. LOADING FEATURE-ENGINEERED STEP 2 DATASET"
    )

    try:

        df = pd.read_csv(INPUT_FILE)

    except Exception as e:

        print(
            "\nERROR: Could not load the input CSV file."
        )

        print(f"Details: {e}")

        sys.exit(1)

    rows_before, columns_before = df.shape

    print(
        f"Rows before feature engineering    : "
        f"{rows_before:,}"
    )

    print(
        f"Columns before feature engineering : "
        f"{columns_before}"
    )

    # --------------------------------------------------------
    # 4. REQUIRED COLUMN CHECK
    # --------------------------------------------------------

    print_section(
        "2. REQUIRED COLUMN VALIDATION"
    )

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

        print(
            "ERROR: Required columns are missing:"
        )

        for column in missing_columns:
            print(f"  - {column}")

        sys.exit(1)

    print(
        "All required columns are present."
    )

    for column in required_columns:
        print(f"  ✓ {column}")

    # --------------------------------------------------------
    # 5. DATE VALIDATION
    # --------------------------------------------------------

    print_section(
        "3. DATE VALIDATION"
    )

    try:

        df["Date"] = pd.to_datetime(
            df["Date"],
            errors="coerce"
        )

    except Exception as e:

        print(
            "\nERROR: Date conversion failed."
        )

        print(f"Details: {e}")

        sys.exit(1)

    invalid_dates = df["Date"].isna().sum()

    print(
        f"Invalid Date values: {invalid_dates}"
    )

    if invalid_dates > 0:

        print(
            "\nERROR: Invalid Date values found."
        )

        sys.exit(1)

    print(
        "Date validation passed."
    )

    # --------------------------------------------------------
    # 6. TARGET VALIDATION
    # --------------------------------------------------------

    print_section(
        "4. TARGET COLUMN VALIDATION"
    )

    try:

        df["Units Sold"] = pd.to_numeric(
            df["Units Sold"],
            errors="coerce"
        )

    except Exception as e:

        print(
            "\nERROR: Units Sold conversion failed."
        )

        print(f"Details: {e}")

        sys.exit(1)

    invalid_units_sold = df["Units Sold"].isna().sum()

    print(
        f"Invalid Units Sold values: "
        f"{invalid_units_sold}"
    )

    if invalid_units_sold > 0:

        print(
            "\nERROR: Invalid or missing "
            "Units Sold values found."
        )

        sys.exit(1)

    print(
        "Units Sold validation passed."
    )

    # --------------------------------------------------------
    # 7. SORT DATA
    # --------------------------------------------------------

    print_section(
        "5. SORTING DATA FOR TIME-SERIES ROLLING FEATURES"
    )

    print(
        "Sorting by Store ID → Product ID → Date..."
    )

    try:

        df = df.sort_values(
            by=[
                "Store ID",
                "Product ID",
                "Date"
            ],
            kind="stable"
        ).reset_index(drop=True)

    except Exception as e:

        print(
            "\nERROR: Dataset sorting failed."
        )

        print(f"Details: {e}")

        sys.exit(1)

    print(
        "Sorting completed successfully."
    )

    # --------------------------------------------------------
    # 8. CHECK DUPLICATE STORE-PRODUCT-DATE RECORDS
    # --------------------------------------------------------

    print_section(
        "6. TIME-SERIES DUPLICATE CHECK"
    )

    duplicate_count = df.duplicated(
        subset=[
            "Store ID",
            "Product ID",
            "Date"
        ]
    ).sum()

    print(
        "Duplicate Store-Product-Date records:"
        f" {duplicate_count:,}"
    )

    if duplicate_count > 0:

        print(
            "\nWARNING: Duplicate Store-Product-Date "
            "records exist."
        )

        print(
            "Rolling features will still be created "
            "in the existing row order."
        )

    else:

        print(
            "No duplicate Store-Product-Date "
            "records found."
        )

    # --------------------------------------------------------
    # 9. CREATE GROUPED HISTORICAL SERIES
    # --------------------------------------------------------

    print_section(
        "7. CREATING LEAKAGE-SAFE ROLLING FEATURES"
    )

    print(
        "\nRolling calculation strategy:"
    )

    print(
        "  1. Group by Store ID + Product ID"
    )

    print(
        "  2. Sort chronologically by Date"
    )

    print(
        "  3. Shift Units Sold by 1 observation"
    )

    print(
        "  4. Calculate rolling statistics"
    )

    print(
        "\nThis ensures the current day's Units Sold "
        "is NOT included in its own rolling feature."
    )

    try:

        grouped_units_sold = (
            df.groupby(
                [
                    "Store ID",
                    "Product ID"
                ],
                sort=False
            )["Units Sold"]
        )

        historical_units_sold = (
            grouped_units_sold.shift(1)
        )

        for window in ROLLING_WINDOWS:

            mean_feature = (
                f"Units_Sold_Rolling_Mean_{window}"
            )

            std_feature = (
                f"Units_Sold_Rolling_Std_{window}"
            )

            df[mean_feature] = (
                historical_units_sold
                .groupby(
                    [
                        df["Store ID"],
                        df["Product ID"]
                    ],
                    sort=False
                )
                .rolling(
                    window=window,
                    min_periods=window
                )
                .mean()
                .reset_index(
                    level=[0, 1],
                    drop=True
                )
            )

            df[std_feature] = (
                historical_units_sold
                .groupby(
                    [
                        df["Store ID"],
                        df["Product ID"]
                    ],
                    sort=False
                )
                .rolling(
                    window=window,
                    min_periods=window
                )
                .std()
                .reset_index(
                    level=[0, 1],
                    drop=True
                )
            )

            print(
                f"Created: {mean_feature}"
            )

            print(
                f"Created: {std_feature}"
            )

    except Exception as e:

        print(
            "\nERROR: Rolling feature creation failed."
        )

        print(f"Details: {e}")

        sys.exit(1)

    # --------------------------------------------------------
    # 10. VERIFY FEATURES
    # --------------------------------------------------------

    print_section(
        "8. NEW FEATURE VALIDATION"
    )

    missing_features = [
        feature
        for feature in NEW_FEATURES
        if feature not in df.columns
    ]

    if missing_features:

        print(
            "ERROR: Expected rolling features are missing:"
        )

        for feature in missing_features:
            print(f"  - {feature}")

        sys.exit(1)

    print(
        "All rolling features were created successfully."
    )

    print(
        "\nNewly created features:"
    )

    for feature in NEW_FEATURES:
        print(f"  ✓ {feature}")

    # --------------------------------------------------------
    # 11. MISSING VALUE ANALYSIS
    # --------------------------------------------------------

    print_section(
        "9. ROLLING FEATURE MISSING VALUE ANALYSIS"
    )

    print(
        "Initial rows of each Store-Product series "
        "naturally contain NaN values because a complete "
        "historical window is not yet available."
    )

    print(
        "\nExpected minimum history:"
    )

    for window in ROLLING_WINDOWS:

        expected_missing = (
            window * df[
                ["Store ID", "Product ID"]
            ]
            .drop_duplicates()
            .shape[0]
        )

        print(
            f"  Window {window:>2}: "
            f"approximately {expected_missing:,} "
            f"initial rows"
        )

    print(
        "\nActual missing values:"
    )

    for feature in NEW_FEATURES:

        missing_count = df[feature].isna().sum()

        missing_percentage = (
            missing_count / len(df) * 100
        )

        print(
            f"  {feature:<32}: "
            f"{missing_count:,} "
            f"({missing_percentage:.2f}%)"
        )

    # --------------------------------------------------------
    # 12. FIRST 40 ROWS PREVIEW
    # --------------------------------------------------------

    print_section(
        "10. ROLLING FEATURE PREVIEW"
    )

    preview_columns = [
        "Date",
        "Store ID",
        "Product ID",
        "Units Sold",
    ] + NEW_FEATURES

    print(
        df[preview_columns]
        .head(40)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # 13. STATISTICS
    # --------------------------------------------------------

    print_section(
        "11. ROLLING FEATURE STATISTICS"
    )

    print(
        df[NEW_FEATURES]
        .describe()
        .round(2)
        .to_string()
    )

    # --------------------------------------------------------
    # 14. LEAKAGE SAFETY CHECK
    # --------------------------------------------------------

    print_section(
        "12. DATA LEAKAGE SAFETY CHECK"
    )

    print(
        "Checking first valid rolling Mean_7 value..."
    )

    first_store = df["Store ID"].iloc[0]
    first_product = df["Product ID"].iloc[0]

    first_series = df[
        (df["Store ID"] == first_store)
        & (df["Product ID"] == first_product)
    ].copy()

    first_series = first_series.sort_values(
        "Date"
    ).reset_index(drop=True)

    first_valid_index = (
        first_series[
            "Units_Sold_Rolling_Mean_7"
        ]
        .first_valid_index()
    )

    if first_valid_index is None:

        print(
            "ERROR: No valid Rolling Mean 7 value found."
        )

        sys.exit(1)

    print(
        f"First valid Rolling Mean 7 index: "
        f"{first_valid_index}"
    )

    if first_valid_index < 7:

        print(
            "\nERROR: Rolling feature became valid "
            "before 7 historical observations."
        )

        sys.exit(1)

    print(
        "✓ Rolling Mean 7 requires the previous "
        "7 observations."
    )

    print(
        "✓ Current Units Sold is excluded."
    )

    print(
        "✓ Leakage safety check passed."
    )

    # --------------------------------------------------------
    # 15. DATASET SIZE VALIDATION
    # --------------------------------------------------------

    rows_after, columns_after = df.shape

    print_section(
        "13. DATASET SIZE VALIDATION"
    )

    print(
        f"Rows before : {rows_before:,}"
    )

    print(
        f"Rows after  : {rows_after:,}"
    )

    print(
        f"Columns before : {columns_before}"
    )

    print(
        f"Columns after  : {columns_after}"
    )

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
            "\nERROR: Unexpected column count."
        )

        print(
            f"Expected: {expected_columns_after}"
        )

        print(
            f"Actual: {columns_after}"
        )

        sys.exit(1)

    print(
        "\nDataset size validation passed."
    )

    # --------------------------------------------------------
    # 16. CREATE PROFESSIONAL CHART
    # --------------------------------------------------------

    print_section(
        "14. CREATING ROLLING FEATURE CHART"
    )

    try:

        plot_df = first_series.head(90).copy()

        plt.figure(
            figsize=(14, 7)
        )

        plt.plot(
            plot_df["Date"],
            plot_df["Units Sold"],
            marker="o",
            linewidth=1.8,
            label="Actual Units Sold"
        )

        plt.plot(
            plot_df["Date"],
            plot_df[
                "Units_Sold_Rolling_Mean_7"
            ],
            linewidth=2,
            label="7-Period Historical Rolling Mean"
        )

        plt.plot(
            plot_df["Date"],
            plot_df[
                "Units_Sold_Rolling_Mean_30"
            ],
            linewidth=2,
            label="30-Period Historical Rolling Mean"
        )

        plt.title(
            "Historical Demand and Leakage-Safe Rolling Means\n"
            "Feature Engineering Step 3",
            fontsize=14,
            fontweight="bold"
        )

        plt.xlabel(
            "Date"
        )

        plt.ylabel(
            "Units Sold"
        )

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

        print(
            "\nERROR: Chart creation failed."
        )

        print(
            f"Details: {e}"
        )

        sys.exit(1)

    print(
        "Chart saved successfully:"
    )

    print(
        CHART_FILE
    )

    # --------------------------------------------------------
    # 17. SAVE OUTPUT DATASET
    # --------------------------------------------------------

    print_section(
        "15. SAVING FEATURE-ENGINEERED STEP 3 DATASET"
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

        print(
            f"Details: {e}"
        )

        sys.exit(1)

    print(
        "Feature-engineered Step 3 dataset saved successfully:"
    )

    print(
        OUTPUT_FILE
    )

    # --------------------------------------------------------
    # 18. FINAL SUMMARY
    # --------------------------------------------------------

    print_section(
        "FEATURE ENGINEERING STEP 3 - FINAL SUMMARY"
    )

    print(
        "Status: SUCCESS"
    )

    print(
        "\nInput dataset:"
    )

    print(
        INPUT_FILE
    )

    print(
        "\nOutput dataset:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        "\nChart:"
    )

    print(
        CHART_FILE
    )

    print(
        f"\nRows:"
    )

    print(
        f"{rows_after:,}"
    )

    print(
        f"\nColumns:"
    )

    print(
        columns_after
    )

    print(
        "\nNew rolling features created:"
    )

    for feature in NEW_FEATURES:
        print(f"  ✓ {feature}")

    print(
        "\nRolling windows:"
    )

    print(
        "  7, 14, 30 observations"
    )

    print(
        "\nGrouping:"
    )

    print(
        "  Store ID + Product ID"
    )

    print(
        "\nLeakage protection:"
    )

    print(
        "  Current Units Sold excluded using shift(1)"
    )

    print(
        "  Full historical window required"
    )

    print(
        "\nFuture forecasting target:"
    )

    print(
        "  Units Sold"
    )

    print(
        "\nExisting Demand Forecast:"
    )

    print(
        "  Kept as an existing dataset column."
    )

    print(
        "  NOT used as our model prediction."
    )

    print(
        "\nML Model:"
    )

    print(
        "  NOT trained in Feature Engineering Step 3."
    )

    print(
        "\nOriginal Step 2 dataset:"
    )

    print(
        "  NOT modified."
    )

    print(
        "\nFeature Engineering Step 3 "
        "completed successfully."
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()