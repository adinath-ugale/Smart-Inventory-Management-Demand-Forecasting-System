"""
SMART INVENTORY MANAGEMENT & DEMAND FORECASTING SYSTEM
Feature Engineering - Step 1

Purpose:
    Create basic time-based features from the Date column.

Target for future forecasting model:
    Units Sold

Important:
    - No ML model is trained in this step.
    - Existing Demand Forecast is NOT used as our model target/prediction.
    - Original cleaned dataset is never modified.
"""

from pathlib import Path
import sys

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(r"C:\SIM&DFS")

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "cleaned_inventory.csv"

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "feature_engineered_step1.csv"
)

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

CHART_FILE = FIGURES_DIR / "feature_engineering_time_features.png"


# ============================================================
# FEATURE LIST
# ============================================================

NEW_FEATURES = [
    "Year",
    "Month",
    "Month_Name",
    "Day",
    "Day_of_Week",
    "Day_Name",
    "Week_of_Year",
    "Quarter",
    "Is_Weekend",
]


# ============================================================
# HELPER FUNCTION
# ============================================================

def print_section(title):
    """Print a clear section heading."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    print_section(
        "SMART INVENTORY MANAGEMENT & DEMAND FORECASTING SYSTEM\n"
        "FEATURE ENGINEERING - STEP 1: TIME-BASED FEATURES"
    )

    # --------------------------------------------------------
    # 1. CHECK INPUT FILE
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
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print("\nERROR: Could not create required directories.")
        print(f"Details: {e}")
        sys.exit(1)

    # --------------------------------------------------------
    # 3. LOAD DATASET
    # --------------------------------------------------------

    print_section("1. LOADING CLEANED DATASET")

    try:
        df = pd.read_csv(INPUT_FILE)
    except Exception as e:
        print("\nERROR: Could not load the CSV file.")
        print(f"Details: {e}")
        sys.exit(1)

    rows_before, columns_before = df.shape

    print(f"Rows before feature engineering    : {rows_before:,}")
    print(f"Columns before feature engineering : {columns_before}")

    # --------------------------------------------------------
    # 4. CHECK DATE COLUMN
    # --------------------------------------------------------

    print_section("2. DATE COLUMN INSPECTION")

    if "Date" not in df.columns:
        print("ERROR: 'Date' column does not exist in the dataset.")
        sys.exit(1)

    print(f"Date column data type before conversion: {df['Date'].dtype}")

    print("\nFirst 5 Date values before conversion:")
    print(df["Date"].head().to_string(index=False))

    # --------------------------------------------------------
    # 5. CONVERT DATE TO DATETIME
    # --------------------------------------------------------

    print("\nConverting Date column to datetime...")

    try:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    except Exception as e:
        print("\nERROR: Date conversion failed.")
        print(f"Details: {e}")
        sys.exit(1)

    # --------------------------------------------------------
    # 6. VERIFY INVALID DATES
    # --------------------------------------------------------

    invalid_dates = df["Date"].isna().sum()

    print(f"\nInvalid Date values after conversion: {invalid_dates}")

    if invalid_dates > 0:
        print("\nERROR: Invalid Date values were found.")
        print("The dataset will NOT be saved.")
        sys.exit(1)

    print("Date validation passed: No invalid Date values found.")

    print(f"Date column data type after conversion: {df['Date'].dtype}")

    print(f"\nMinimum Date: {df['Date'].min()}")
    print(f"Maximum Date: {df['Date'].max()}")

    # --------------------------------------------------------
    # 7. CREATE TIME-BASED FEATURES
    # --------------------------------------------------------

    print_section("3. CREATING TIME-BASED FEATURES")

    try:
        df["Year"] = df["Date"].dt.year

        df["Month"] = df["Date"].dt.month

        df["Month_Name"] = df["Date"].dt.month_name()

        df["Day"] = df["Date"].dt.day

        df["Day_of_Week"] = df["Date"].dt.dayofweek

        df["Day_Name"] = df["Date"].dt.day_name()

        df["Week_of_Year"] = df["Date"].dt.isocalendar().week.astype(int)

        df["Quarter"] = df["Date"].dt.quarter

        df["Is_Weekend"] = df["Date"].dt.dayofweek >= 5

    except Exception as e:
        print("\nERROR: Feature creation failed.")
        print(f"Details: {e}")
        sys.exit(1)

    print("\nNewly created features:")
    for feature in NEW_FEATURES:
        print(f"  - {feature}")

    # --------------------------------------------------------
    # 8. VERIFY NEW FEATURES
    # --------------------------------------------------------

    print_section("4. FEATURE VALIDATION")

    missing_features = [
        feature
        for feature in NEW_FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        print("ERROR: Some features were not created:")
        for feature in missing_features:
            print(f"  - {feature}")
        sys.exit(1)

    print("All expected time-based features were created successfully.")

    # --------------------------------------------------------
    # 9. FIRST 10 ROWS OF NEW FEATURES
    # --------------------------------------------------------

    print_section("5. FIRST 10 ROWS OF NEW TIME FEATURES")

    preview_columns = ["Date"] + NEW_FEATURES

    print(
        df[preview_columns]
        .head(10)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # 10. UNIQUE VALUES / RANGES
    # --------------------------------------------------------

    print_section("6. FEATURE VALUE SUMMARY")

    print(f"Year range       : {df['Year'].min()} - {df['Year'].max()}")
    print(f"Month values     : {sorted(df['Month'].unique().tolist())}")

    print(
        f"Month names      : "
        f"{df['Month_Name'].unique().tolist()}"
    )

    print(
        f"Day range        : "
        f"{df['Day'].min()} - {df['Day'].max()}"
    )

    print(
        f"Day_of_Week      : "
        f"{sorted(df['Day_of_Week'].unique().tolist())}"
    )

    print(
        f"Day names        : "
        f"{df['Day_Name'].unique().tolist()}"
    )

    print(
        f"Week_of_Year range: "
        f"{df['Week_of_Year'].min()} - "
        f"{df['Week_of_Year'].max()}"
    )

    print(
        f"Quarter values   : "
        f"{sorted(df['Quarter'].unique().tolist())}"
    )

    print(
        f"Is_Weekend values: "
        f"{df['Is_Weekend'].unique().tolist()}"
    )

    print("\nWeekend record counts:")
    print(df["Is_Weekend"].value_counts().sort_index())

    # --------------------------------------------------------
    # 11. CHECK MISSING VALUES
    # --------------------------------------------------------

    print_section("7. MISSING VALUE VALIDATION")

    new_feature_missing = df[NEW_FEATURES].isna().sum()

    print("Missing values in newly created features:")

    for feature, missing_count in new_feature_missing.items():
        print(f"  {feature:<18}: {missing_count}")

    total_new_feature_missing = new_feature_missing.sum()

    if total_new_feature_missing > 0:
        print(
            "\nERROR: Unexpected missing values were introduced "
            "in the new features."
        )
        sys.exit(1)

    print("\nValidation passed: No missing values introduced.")

    # --------------------------------------------------------
    # 12. CHECK ROW COUNT
    # --------------------------------------------------------

    rows_after, columns_after = df.shape

    print_section("8. DATASET SIZE VALIDATION")

    print(f"Rows before : {rows_before:,}")
    print(f"Rows after  : {rows_after:,}")

    print(f"Columns before : {columns_before}")
    print(f"Columns after  : {columns_after}")

    if rows_before != rows_after:
        print("\nERROR: Row count changed unexpectedly.")
        sys.exit(1)

    expected_columns_after = columns_before + len(NEW_FEATURES)

    if columns_after != expected_columns_after:
        print(
            "\nWARNING: Column count is different from the "
            "expected count."
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
    # 13. CREATE ONE PROFESSIONAL CHART
    # --------------------------------------------------------

    print_section("9. CREATING TEMPORAL STRUCTURE CHART")

    try:
        monthly_counts = (
            df.set_index("Date")
            .resample("MS")
            .size()
        )

        plt.figure(figsize=(12, 6))

        plt.plot(
            monthly_counts.index,
            monthly_counts.values,
            marker="o",
            linewidth=2
        )

        plt.title(
            "Monthly Record Counts - Feature Engineering Step 1",
            fontsize=14,
            fontweight="bold"
        )

        plt.xlabel("Month")
        plt.ylabel("Number of Records")

        plt.grid(True, alpha=0.3)

        plt.xticks(rotation=45)

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

    print(f"Chart saved successfully:")
    print(CHART_FILE)

    # --------------------------------------------------------
    # 14. SAVE FEATURE-ENGINEERED DATASET
    # --------------------------------------------------------

    print_section("10. SAVING FEATURE-ENGINEERED DATASET")

    try:
        df.to_csv(
            OUTPUT_FILE,
            index=False
        )
    except Exception as e:
        print("\nERROR: Could not save feature-engineered dataset.")
        print(f"Details: {e}")
        sys.exit(1)

    print("Feature-engineered dataset saved successfully:")
    print(OUTPUT_FILE)

    # --------------------------------------------------------
    # 15. FINAL SUMMARY
    # --------------------------------------------------------

    print_section("FEATURE ENGINEERING STEP 1 - FINAL SUMMARY")

    print("Status: SUCCESS")

    print(f"\nInput dataset:")
    print(f"  {INPUT_FILE}")

    print(f"\nOutput dataset:")
    print(f"  {OUTPUT_FILE}")

    print(f"\nChart:")
    print(f"  {CHART_FILE}")

    print(f"\nRows:")
    print(f"  {rows_after:,}")

    print(f"\nColumns:")
    print(f"  {columns_after}")

    print("\nNew features created:")
    for feature in NEW_FEATURES:
        print(f"  ✓ {feature}")

    print("\nTarget for future forecasting model:")
    print("  Units Sold")

    print("\nExisting Demand Forecast:")
    print("  Kept as an existing dataset column.")
    print("  NOT used as our model target/prediction.")

    print("\nML Model:")
    print("  NOT trained in Feature Engineering Step 1.")

    print("\nOriginal cleaned_inventory.csv:")
    print("  NOT modified.")

    print("\nFeature Engineering Step 1 completed successfully.")


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()