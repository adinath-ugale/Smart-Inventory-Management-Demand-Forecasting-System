r"""
SMART INVENTORY MANAGEMENT & DEMAND FORECASTING SYSTEM
Feature Engineering - Step 4: Demand Trend Features

Purpose:
    Add exactly four new leakage-safe demand trend and lag features
    to the Step 3 dataset:
    1. Units_Sold_Lag_3
    2. Units_Sold_Lag_21
    3. Units_Sold_Rolling_Median_7
    4. Units_Sold_Trend_7

Input:
    C:\SIM&DFS\data\processed\feature_engineered_step3.csv

Output:
    C:\SIM&DFS\data\processed\feature_engineered_step4.csv

Chart:
    C:\SIM&DFS\reports\figures\feature_engineering_step4_trend_features.png

Prediction target:
    Units Sold (Demand Forecast is NOT the model target and is NOT used).

Leakage Prevention:
    - All features are calculated strictly within each Store ID + Product ID group.
    - All rolling calculations use historical observations only (shifted by 1).
    - Current day's Units Sold is never included in rolling or lag features.
    - Future observations are never used.
"""

from pathlib import Path
import sys

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
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
    / "feature_engineered_step3.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "feature_engineered_step4.csv"
)

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

CHART_FILE = (
    FIGURES_DIR
    / "feature_engineering_step4_trend_features.png"
)


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

NEW_FEATURES = [
    "Units_Sold_Lag_3",
    "Units_Sold_Lag_21",
    "Units_Sold_Rolling_Median_7",
    "Units_Sold_Trend_7",
]

EXPECTED_ROWS = 73100
EXPECTED_INPUT_COLS = 34
EXPECTED_OUTPUT_COLS = 38
EXPECTED_GROUPS = 100

EXPECTED_NANS = {
    "Units_Sold_Lag_3": 3 * EXPECTED_GROUPS,            # 300
    "Units_Sold_Lag_21": 21 * EXPECTED_GROUPS,          # 2,100
    "Units_Sold_Rolling_Median_7": 7 * EXPECTED_GROUPS,  # 700
    "Units_Sold_Trend_7": 7 * EXPECTED_GROUPS,           # 700
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_section(title: str):
    """Print a formatted section heading."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def calculate_linear_trend_slope(window: np.ndarray) -> float:
    """
    Calculate the linear regression slope over the previous 7 observations.
    
    Given:
        x = [1, 2, 3, 4, 5, 6, 7]
        y = previous 7 Units Sold values
    
    In simple linear regression (OLS), the slope beta_1 is:
        beta_1 = Cov(x, y) / Var(x)
               = sum((x_i - x_mean) * y_i) / sum((x_i - x_mean)^2)
    
    For fixed x = [1, 2, 3, 4, 5, 6, 7]:
        x_mean = 4.0
        x_i - x_mean = [-3, -2, -1, 0, 1, 2, 3]
        sum((x_i - x_mean)^2) = 9 + 4 + 1 + 0 + 1 + 4 + 9 = 28.0
        
    Therefore:
        beta_1 = sum((x_i - 4) * y_i) / 28.0
        
    This analytical formula is mathematically exact and highly efficient.
    """
    weights = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]) / 28.0
    return float(np.dot(window, weights))


# ============================================================
# MAIN
# ============================================================

def main():

    print_section(
        "SMART INVENTORY MANAGEMENT & DEMAND FORECASTING SYSTEM\n"
        "FEATURE ENGINEERING - STEP 4: DEMAND TREND FEATURES"
    )

    # --------------------------------------------------------
    # 1. INPUT FILE CHECK
    # --------------------------------------------------------
    print("\n--- INPUT FILE CHECK ---")
    print(f"Input file: {INPUT_FILE}")

    if not INPUT_FILE.exists():
        print(f"\nERROR: Input dataset was not found at {INPUT_FILE}")
        sys.exit(1)

    print("Input dataset found successfully.")

    # --------------------------------------------------------
    # 2. CREATE REQUIRED DIRECTORIES
    # --------------------------------------------------------
    try:
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"\nERROR: Could not create required directories: {e}")
        sys.exit(1)

    # --------------------------------------------------------
    # 3. LOAD DATASET
    # --------------------------------------------------------
    print_section("1. LOADING FEATURE-ENGINEERED STEP 3 DATASET")

    try:
        df = pd.read_csv(INPUT_FILE)
    except Exception as e:
        print(f"\nERROR: Could not load the input CSV file: {e}")
        sys.exit(1)

    rows_before, columns_before = df.shape

    print(f"Rows before feature engineering    : {rows_before:,} (Expected: {EXPECTED_ROWS:,})")
    print(f"Columns before feature engineering : {columns_before} (Expected: {EXPECTED_INPUT_COLS})")

    if rows_before != EXPECTED_ROWS:
        print(f"\nERROR: Expected {EXPECTED_ROWS:,} rows, but got {rows_before:,}.")
        sys.exit(1)

    if columns_before != EXPECTED_INPUT_COLS:
        print(f"\nERROR: Expected {EXPECTED_INPUT_COLS} columns, but got {columns_before}.")
        sys.exit(1)

    print("Input dataset dimensions verified successfully.")

    # --------------------------------------------------------
    # 4. REQUIRED COLUMN VALIDATION
    # --------------------------------------------------------
    print_section("2. REQUIRED COLUMN VALIDATION")

    required_columns = [
        "Date",
        "Store ID",
        "Product ID",
        "Units Sold",
        "Demand Forecast",
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        print(f"ERROR: Required columns are missing: {missing_columns}")
        sys.exit(1)

    print("All required columns are present:")
    for col in required_columns:
        print(f"  [OK] {col}")

    # --------------------------------------------------------
    # 5. DATE VALIDATION
    # --------------------------------------------------------
    print_section("3. DATE VALIDATION")

    try:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    except Exception as e:
        print(f"\nERROR: Date conversion failed: {e}")
        sys.exit(1)

    invalid_dates = df["Date"].isna().sum()
    print(f"Invalid Date count: {invalid_dates} (Expected: 0)")

    if invalid_dates > 0:
        print(f"\nERROR: Found {invalid_dates} invalid Date values.")
        sys.exit(1)

    print("Date validation passed.")

    # --------------------------------------------------------
    # 6. TARGET COLUMN VALIDATION
    # --------------------------------------------------------
    print_section("4. TARGET COLUMN VALIDATION (Units Sold)")

    try:
        df["Units Sold"] = pd.to_numeric(df["Units Sold"], errors="coerce")
    except Exception as e:
        print(f"\nERROR: Units Sold conversion failed: {e}")
        sys.exit(1)

    invalid_units_sold = df["Units Sold"].isna().sum()
    negative_units_sold = (df["Units Sold"] < 0).sum()

    print(f"Invalid / missing Units Sold values: {invalid_units_sold} (Expected: 0)")
    print(f"Negative Units Sold values          : {negative_units_sold} (Expected: 0)")

    if invalid_units_sold > 0 or negative_units_sold > 0:
        print("\nERROR: Units Sold contains missing or negative values.")
        sys.exit(1)

    print("Units Sold validation passed (all values non-negative and non-null).")

    # --------------------------------------------------------
    # 7. SORT DATA
    # --------------------------------------------------------
    print_section("5. SORTING DATA CHRONOLOGICALLY PER STORE AND PRODUCT")

    print("Sorting by Store ID -> Product ID -> Date (kind='stable')...")

    try:
        df = df.sort_values(
            by=["Store ID", "Product ID", "Date"],
            kind="stable"
        ).reset_index(drop=True)
    except Exception as e:
        print(f"\nERROR: Dataset sorting failed: {e}")
        sys.exit(1)

    print("Sorting completed successfully.")

    # --------------------------------------------------------
    # 8. CHECK DUPLICATES & STORE-PRODUCT GROUPS
    # --------------------------------------------------------
    print_section("6. DUPLICATE & GROUP VALIDATION")

    duplicate_count = df.duplicated(
        subset=["Store ID", "Product ID", "Date"]
    ).sum()

    print(f"Duplicate Store ID + Product ID + Date records: {duplicate_count:,} (Expected: 0)")

    if duplicate_count > 0:
        print(f"\nERROR: Found {duplicate_count} duplicate records.")
        sys.exit(1)

    total_groups = df.groupby(["Store ID", "Product ID"], sort=False).ngroups
    print(f"Store-Product unique groups: {total_groups} (Expected: {EXPECTED_GROUPS})")

    if total_groups != EXPECTED_GROUPS:
        print(f"\nERROR: Expected {EXPECTED_GROUPS} groups, but found {total_groups}.")
        sys.exit(1)

    print("Duplicate Check: PASSED")
    print("Group Count Check: PASSED")

    # --------------------------------------------------------
    # 9. CREATE LEAKAGE-SAFE DEMAND TREND FEATURES
    # --------------------------------------------------------
    print_section("7. CREATING LEAKAGE-SAFE DEMAND TREND & LAG FEATURES")

    print("Feature Engineering Strategy:")
    print("  1. Units_Sold_Lag_3: groupby(['Store ID', 'Product ID'])['Units Sold'].shift(3)")
    print("  2. Units_Sold_Lag_21: groupby(['Store ID', 'Product ID'])['Units Sold'].shift(21)")
    print("  3. Historical Units Sold: shift(1) observation per group (excludes current day)")
    print("  4. Units_Sold_Rolling_Median_7: rolling median over previous 7 observations (min_periods=7)")
    print("  5. Units_Sold_Trend_7: linear regression slope over previous 7 observations (min_periods=7)")
    print("\nStrict Leakage Rule: Current day's Units Sold is NEVER included.")
    print("Forecast Rule: Demand Forecast column is NOT used.")

    try:
        # Grouped Units Sold Series
        grouped_units_sold = df.groupby(
            ["Store ID", "Product ID"],
            sort=False
        )["Units Sold"]

        # Feature 1: Units_Sold_Lag_3
        print("\nCreating Units_Sold_Lag_3...")
        df["Units_Sold_Lag_3"] = grouped_units_sold.shift(3)
        print("  [OK] Created: Units_Sold_Lag_3")

        # Feature 2: Units_Sold_Lag_21
        print("Creating Units_Sold_Lag_21...")
        df["Units_Sold_Lag_21"] = grouped_units_sold.shift(21)
        print("  [OK] Created: Units_Sold_Lag_21")

        # Historical series: shift by 1 to strictly exclude current observation
        # Conceptually: historical_units_sold(t) = UnitsSold(t-1)
        historical_units_sold = grouped_units_sold.shift(1)

        grouped_historical = historical_units_sold.groupby(
            [df["Store ID"], df["Product ID"]],
            sort=False
        )

        # Feature 3: Units_Sold_Rolling_Median_7
        print("Creating Units_Sold_Rolling_Median_7...")
        df["Units_Sold_Rolling_Median_7"] = (
            grouped_historical
            .rolling(window=7, min_periods=7)
            .median()
            .reset_index(level=[0, 1], drop=True)
        )
        print("  [OK] Created: Units_Sold_Rolling_Median_7")

        # Feature 4: Units_Sold_Trend_7
        print("Creating Units_Sold_Trend_7 (Linear regression slope over previous 7 days)...")
        df["Units_Sold_Trend_7"] = (
            grouped_historical
            .rolling(window=7, min_periods=7)
            .apply(calculate_linear_trend_slope, raw=True)
            .reset_index(level=[0, 1], drop=True)
        )
        print("  [OK] Created: Units_Sold_Trend_7")

    except Exception as e:
        print(f"\nERROR: Feature creation failed: {e}")
        sys.exit(1)

    # --------------------------------------------------------
    # 10. NEW FEATURE EXISTENCE VALIDATION
    # --------------------------------------------------------
    print_section("8. NEW FEATURE EXISTENCE VALIDATION")

    missing_features = [f for f in NEW_FEATURES if f not in df.columns]

    if missing_features:
        print(f"ERROR: The following expected features are missing: {missing_features}")
        sys.exit(1)

    print("All four new features exist in the dataframe:")
    for feature in NEW_FEATURES:
        print(f"  [OK] {feature}")

    # --------------------------------------------------------
    # 11. MISSING VALUE & STRUCTURAL NAN VALIDATION
    # --------------------------------------------------------
    print_section("9. STRUCTURAL MISSING VALUE VALIDATION")

    print("Initial observations naturally produce structural NaNs because")
    print("the required historical window is not yet fully available.")
    print("These structural NaNs MUST NOT be filled with zero.\n")

    nan_validation_passed = True

    for feature, expected_nan in EXPECTED_NANS.items():
        actual_nan = int(df[feature].isna().sum())
        nan_pct = (actual_nan / len(df)) * 100
        status = "PASSED" if actual_nan == expected_nan else "FAILED"
        print(f"  {feature:<30}: Actual NaNs = {actual_nan:,} | Expected = {expected_nan:,} ({nan_pct:.2f}%) -> {status}")

        if actual_nan != expected_nan:
            nan_validation_passed = False

    if not nan_validation_passed:
        print("\nERROR: Structural missing value counts did not match expected counts.")
        sys.exit(1)

    print("\nAll structural missing values exactly match expected counts.")

    # Check for unexpected missing values across the entire dataset
    print("\n--- UNEXPECTED MISSING VALUE CHECK ---")
    
    known_nans_cols = {
        "Demand Forecast": 673,
        "Units_Sold_Lag_1": 100,
        "Units_Sold_Lag_7": 700,
        "Units_Sold_Lag_14": 1400,
        "Units_Sold_Lag_30": 3000,
        "Units_Sold_Rolling_Mean_7": 700,
        "Units_Sold_Rolling_Std_7": 700,
        "Units_Sold_Rolling_Mean_14": 1400,
        "Units_Sold_Rolling_Std_14": 1400,
        "Units_Sold_Rolling_Mean_30": 3000,
        "Units_Sold_Rolling_Std_30": 3000,
        **EXPECTED_NANS,
    }

    all_missing = df.isna().sum()
    unexpected_found = False

    for col in df.columns:
        missing_cnt = int(all_missing[col])
        if col in known_nans_cols:
            if missing_cnt != known_nans_cols[col]:
                print(f"  WARNING: Column {col} has {missing_cnt} NaNs, expected {known_nans_cols[col]}")
                unexpected_found = True
        else:
            if missing_cnt > 0:
                print(f"  ERROR: Column {col} has unexpected {missing_cnt} NaNs!")
                unexpected_found = True

    if unexpected_found:
        print("\nERROR: Unexpected missing values detected.")
        sys.exit(1)

    print("No unexpected missing values found beyond structural initial-history NaNs.")

    # --------------------------------------------------------
    # 12. EXPLICIT DATA LEAKAGE VALIDATION
    # --------------------------------------------------------
    print_section("10. EXPLICIT DATA LEAKAGE VALIDATION")

    print("Checking leakage rules across all 100 Store-Product groups:")
    print("  1. Rolling Median 7: first valid value occurs only at observation index 7.")
    print("  2. Trend 7: first valid value occurs only at observation index 7.")
    print("  3. Current day's Units Sold is strictly excluded.")
    print("  4. Index 7 value matches exact calculation from historical days 0..6.")

    weights = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]) / 28.0
    leakage_passed = True

    for (store, prod), group in df.groupby(["Store ID", "Product ID"], sort=False):
        group_start = group.index[0]

        first_med_idx = group["Units_Sold_Rolling_Median_7"].first_valid_index() - group_start
        first_trend_idx = group["Units_Sold_Trend_7"].first_valid_index() - group_start

        if first_med_idx != 7:
            print(f"LEAKAGE FAILURE: Store {store}, Product {prod} - Rolling Median first valid index is {first_med_idx} (Expected: 7)")
            leakage_passed = False
            break

        if first_trend_idx != 7:
            print(f"LEAKAGE FAILURE: Store {store}, Product {prod} - Trend first valid index is {first_trend_idx} (Expected: 7)")
            leakage_passed = False
            break

        hist_vals = group["Units Sold"].iloc[:7].values
        expected_median = float(np.median(hist_vals))
        actual_median = float(group["Units_Sold_Rolling_Median_7"].iloc[7])

        expected_trend = float(np.dot(hist_vals, weights))
        actual_trend = float(group["Units_Sold_Trend_7"].iloc[7])

        if abs(expected_median - actual_median) > 1e-6:
            print(f"VALUE MISMATCH: Store {store}, Product {prod} - Rolling Median {actual_median} != Expected {expected_median}")
            leakage_passed = False
            break

        if abs(expected_trend - actual_trend) > 1e-6:
            print(f"VALUE MISMATCH: Store {store}, Product {prod} - Trend {actual_trend} != Expected {expected_trend}")
            leakage_passed = False
            break

    if not leakage_passed:
        print("\nERROR: Data leakage checks failed.")
        sys.exit(1)

    print("\n[OK] Validated for all 100 Store-Product groups:")
    print("  - Observation indices 0 to 6 are NaN.")
    print("  - Observation index 7 is the first valid value.")
    print("  - Uses exactly the 7 previous observations (t-7 to t-1).")
    print("  - Current day t observation is completely excluded.")
    print("\nLEAKAGE CHECK: PASSED")

    # --------------------------------------------------------
    # 13. TREND VALIDATION & DESCRIPTIVE STATISTICS
    # --------------------------------------------------------
    print_section("11. TREND FEATURE DESCRIPTIVE STATISTICS")

    stats_df = df[NEW_FEATURES].describe().loc[["count", "mean", "std", "min", "max"]]
    print(stats_df.round(4).to_string())

    print("\n--- SAMPLE PREVIEW (First 25 Rows of Store S001, Product P0001) ---")
    preview_cols = [
        "Date",
        "Store ID",
        "Product ID",
        "Units Sold",
        "Units_Sold_Lag_3",
        "Units_Sold_Lag_21",
        "Units_Sold_Rolling_Median_7",
        "Units_Sold_Trend_7",
    ]

    preview_df = df[preview_cols].head(25).copy()
    preview_df["Date"] = preview_df["Date"].dt.strftime("%Y-%m-%d")
    print(preview_df.to_string(index=False))

    # --------------------------------------------------------
    # 14. DATASET SIZE & ROW/COL INTEGRITY CHECK
    # --------------------------------------------------------
    print_section("12. DATASET SIZE & ROW/COL INTEGRITY CHECK")

    rows_after, columns_after = df.shape

    print(f"Rows before    : {rows_before:,}")
    print(f"Rows after     : {rows_after:,}")
    print(f"Columns before : {columns_before}")
    print(f"Columns after  : {columns_after}")

    if rows_before != rows_after:
        print("\nERROR: Row count changed unexpectedly.")
        sys.exit(1)

    if columns_after != EXPECTED_OUTPUT_COLS:
        print(f"\nERROR: Unexpected column count: {columns_after} (Expected: {EXPECTED_OUTPUT_COLS})")
        sys.exit(1)

    print("\nRow Count Check: PASSED")
    print("Column Count Check: PASSED")

    # --------------------------------------------------------
    # 15. PROFESSIONAL CHART CREATION
    # --------------------------------------------------------
    print_section("13. CREATING DEMAND TREND VISUALIZATION CHART")

    try:
        first_store = df["Store ID"].iloc[0]
        first_product = df["Product ID"].iloc[0]

        plot_df = df[
            (df["Store ID"] == first_store)
            & (df["Product ID"] == first_product)
        ].copy().head(90)

        fig, (ax1, ax2) = plt.subplots(
            2, 1,
            figsize=(14, 9),
            sharex=True,
            gridspec_kw={"height_ratios": [2, 1]}
        )

        # Subplot 1: Actual Units Sold & 7-Period Rolling Median
        ax1.plot(
            plot_df["Date"],
            plot_df["Units Sold"],
            marker="o",
            markersize=4,
            linewidth=1.5,
            color="#1f77b4",
            label="Actual Units Sold",
            alpha=0.75
        )

        ax1.plot(
            plot_df["Date"],
            plot_df["Units_Sold_Rolling_Median_7"],
            linewidth=2.5,
            color="#ff7f0e",
            label="7-Period Historical Rolling Median"
        )

        ax1.set_ylabel("Units Sold", fontsize=12, fontweight="bold")
        ax1.set_title(
            f"Demand Trend & Rolling Median Analysis | Store: {first_store} - Product: {first_product}\n"
            "Feature Engineering Step 4: Demand Trend Features",
            fontsize=14,
            fontweight="bold"
        )
        ax1.legend(loc="upper right", framealpha=0.9)
        ax1.grid(True, linestyle="--", alpha=0.5)

        # Subplot 2: 7-Period Demand Trend (Slope)
        ax2.plot(
            plot_df["Date"],
            plot_df["Units_Sold_Trend_7"],
            linewidth=2.0,
            color="#2ca02c",
            label="7-Period Demand Trend (Slope)"
        )

        ax2.axhline(
            0,
            color="red",
            linestyle="--",
            linewidth=1.2,
            alpha=0.75,
            label="Zero Trend (Stable Demand)"
        )

        ax2.set_xlabel("Date", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Demand Slope", fontsize=12, fontweight="bold")
        ax2.legend(loc="upper right", framealpha=0.9)
        ax2.grid(True, linestyle="--", alpha=0.5)

        plt.xticks(rotation=45)
        plt.tight_layout()

        plt.savefig(
            CHART_FILE,
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

        print(f"Chart saved successfully at:\n  {CHART_FILE}")

    except Exception as e:
        print(f"\nERROR: Chart creation failed: {e}")
        sys.exit(1)

    # --------------------------------------------------------
    # 16. SAVE OUTPUT DATASET
    # --------------------------------------------------------
    print_section("14. SAVING FEATURE-ENGINEERED STEP 4 DATASET")

    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

    try:
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"Feature-engineered Step 4 dataset saved successfully at:\n  {OUTPUT_FILE}")
    except Exception as e:
        print(f"\nERROR: Could not save output dataset: {e}")
        sys.exit(1)

    if not INPUT_FILE.exists():
        print(f"\nCRITICAL ERROR: Input file {INPUT_FILE} was removed or modified!")
        sys.exit(1)

    # --------------------------------------------------------
    # 17. FINAL TERMINAL SUMMARY
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING STEP 4 SUMMARY")
    print("=" * 34)
    print("\nInput:")
    print("feature_engineered_step3.csv")
    print(f"\nRows before: {rows_before:,}")
    print(f"Columns before: {columns_before}")
    print("\nNew Features:\n")
    print("1. Units_Sold_Lag_3")
    print("2. Units_Sold_Lag_21")
    print("3. Units_Sold_Rolling_Median_7")
    print("4. Units_Sold_Trend_7")
    print(f"\nRows after: {rows_after:,}")
    print(f"Columns after: {columns_after}")
    print("\nExpected structural NaNs:")
    print("Lag 3: 300")
    print("Lag 21: 2,100")
    print("Rolling Median 7: 700")
    print("Trend 7: 700")
    print("\nLeakage Check: PASSED")
    print("Row Count Check: PASSED")
    print("Column Count Check: PASSED")
    print("Duplicate Check: PASSED")
    print("\nOutput:")
    print(str(OUTPUT_FILE))
    print("\nChart:")
    print(str(CHART_FILE))
    print("\nSTATUS: SUCCESS")


if __name__ == "__main__":
    main()
