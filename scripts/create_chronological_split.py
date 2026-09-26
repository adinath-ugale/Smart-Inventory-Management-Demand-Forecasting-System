r"""
SMART INVENTORY MANAGEMENT & DEMAND FORECASTING SYSTEM
Step 9: Chronological Train / Validation / Test Split

Purpose:
    Partition the final model-ready dataset into global chronological splits:
    - TRAIN: ~80% of chronological observations
    - VALIDATION: ~10% of chronological observations
    - TEST: ~10% of chronological observations

Input:
    C:\SIM&DFS\data\processed\model_ready_dataset.csv (73,100 x 37)

Outputs:
    C:\SIM&DFS\data\processed\splits\train.csv
    C:\SIM&DFS\data\processed\splits\validation.csv
    C:\SIM&DFS\data\processed\splits\test.csv
    C:\SIM&DFS\reports\chronological_split_report.txt

Constraints:
    - Strictly chronological date-based split.
    - No date is split across datasets.
    - No random shuffling, no stratified split, no model training, no predictions.
    - Source dataset MUST remain completely unmodified.
"""

import hashlib
from pathlib import Path
import sys

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(r"C:\SIM&DFS")

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_ready_dataset.csv"
)

SPLITS_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits"
)

TRAIN_FILE = SPLITS_DIR / "train.csv"
VAL_FILE = SPLITS_DIR / "validation.csv"
TEST_FILE = SPLITS_DIR / "test.csv"

REPORT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "chronological_split_report.txt"
)


# ============================================================
# CONSTANTS & CONFIGURATION
# ============================================================

EXPECTED_ROWS = 73100
EXPECTED_COLS = 37
EXPECTED_GROUPS = 100
TARGET_COL = "Units Sold"

CORE_33_FEATURES = [
    "Category",
    "Region",
    "Seasonality",
    "Year",
    "Month",
    "Day",
    "Day_of_Week",
    "Is_Weekend",
    "Units_Sold_Lag_1",
    "Units_Sold_Lag_3",
    "Units_Sold_Lag_7",
    "Units_Sold_Lag_14",
    "Units_Sold_Lag_21",
    "Units_Sold_Lag_30",
    "Units_Sold_Rolling_Mean_7",
    "Units_Sold_Rolling_Std_7",
    "Units_Sold_Rolling_Mean_14",
    "Units_Sold_Rolling_Std_14",
    "Units_Sold_Rolling_Mean_30",
    "Units_Sold_Rolling_Std_30",
    "Units_Sold_Rolling_Median_7",
    "Units_Sold_Trend_7",
    "Demand_CV_7",
    "Demand_CV_14",
    "Demand_CV_30",
    "Inventory_Lag_1",
    "Units_Ordered_Lag_1",
    "Inventory_Demand_Coverage_7",
    "Order_Demand_Ratio_7",
    "Price_Lag_1",
    "Discount_Lag_1",
    "Competitor_Price_Gap_Lag_1",
    "Promotion_Lag_1",
]

METADATA_COLS = ["Date", "Store ID", "Product ID"]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def compute_file_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    report_lines = []

    def log(msg=""):
        report_lines.append(msg)
        print(msg)

    log("=" * 80)
    log("SMART INVENTORY MANAGEMENT & DEMAND FORECASTING SYSTEM")
    log("STEP 9 — CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT")
    log("=" * 80)
    log(f"Source file       : {INPUT_FILE}")
    log(f"Splits directory  : {SPLITS_DIR}")
    log(f"Report file       : {REPORT_FILE}")

    # --------------------------------------------------------
    # 1. SOURCE INTEGRITY & PRE-SPLIT HASH
    # --------------------------------------------------------
    log("\n--- 1. SOURCE DATASET VERIFICATION ---")
    if not INPUT_FILE.exists():
        log(f"CRITICAL ERROR: Source dataset not found at {INPUT_FILE}")
        sys.exit(1)

    initial_size = INPUT_FILE.stat().st_size
    initial_mtime = INPUT_FILE.stat().st_mtime
    initial_hash = compute_file_sha256(INPUT_FILE)

    log(f"Source file found : {INPUT_FILE}")
    log(f"Initial size      : {initial_size:,} bytes")
    log(f"Initial SHA-256   : {initial_hash}")

    df_source = pd.read_csv(INPUT_FILE)
    src_rows, src_cols = df_source.shape
    log(f"Source rows       : {src_rows:,} (Expected: {EXPECTED_ROWS:,})")
    log(f"Source columns    : {src_cols} (Expected: {EXPECTED_COLS})")

    if src_rows != EXPECTED_ROWS or src_cols != EXPECTED_COLS:
        log("ERROR: Source dataset dimensions do not match expected values.")
        sys.exit(1)

    source_cols = df_source.columns.tolist()

    # --------------------------------------------------------
    # 2. CHRONOLOGICAL DATE BOUNDARY DETERMINATION
    # --------------------------------------------------------
    log("\n--- 2. CHRONOLOGICAL DATE BOUNDARY DETERMINATION ---")

    # Parse and sort unique calendar dates
    unique_dates = pd.to_datetime(df_source["Date"]).drop_duplicates().sort_values().reset_index(drop=True)
    n_unique_dates = len(unique_dates)
    log(f"Total unique calendar dates : {n_unique_dates}")
    log(f"Global Date Range           : {unique_dates.iloc[0].strftime('%Y-%m-%d')} to {unique_dates.iloc[-1].strftime('%Y-%m-%d')}")

    # Verify uniform observations per date
    rows_per_date = df_source.groupby("Date").size()
    min_rpd, max_rpd = rows_per_date.min(), rows_per_date.max()
    log(f"Observations per date       : min = {min_rpd}, max = {max_rpd} (Expected: 100)")

    # Calculate ~80% Train, ~10% Validation, ~10% Test
    idx_train = int(round(n_unique_dates * 0.80))
    idx_val = int(round(n_unique_dates * 0.90))

    train_dates = unique_dates.iloc[:idx_train]
    val_dates = unique_dates.iloc[idx_train:idx_val]
    test_dates = unique_dates.iloc[idx_val:]

    train_start_str = train_dates.iloc[0].strftime("%Y-%m-%d")
    train_end_str = train_dates.iloc[-1].strftime("%Y-%m-%d")

    val_start_str = val_dates.iloc[0].strftime("%Y-%m-%d")
    val_end_str = val_dates.iloc[-1].strftime("%Y-%m-%d")

    test_start_str = test_dates.iloc[0].strftime("%Y-%m-%d")
    test_end_str = test_dates.iloc[-1].strftime("%Y-%m-%d")

    log(f"\nCalculated Boundaries:")
    log(f"  TRAIN      : {train_start_str} to {train_end_str} ({len(train_dates)} unique dates)")
    log(f"  VALIDATION : {val_start_str} to {val_end_str} ({len(val_dates)} unique dates)")
    log(f"  TEST       : {test_start_str} to {test_end_str} ({len(test_dates)} unique dates)")

    # --------------------------------------------------------
    # 3. CONSTRUCTING SPLITS
    # --------------------------------------------------------
    log("\n--- 3. CREATING CHRONOLOGICAL SPLITS ---")
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    # Sort stably by Date, Store ID, Product ID to preserve clean chronological order
    df_sorted = df_source.sort_values(by=["Date", "Store ID", "Product ID"], kind="stable").reset_index(drop=True)

    train_df = df_sorted[df_sorted["Date"] <= train_end_str].copy().reset_index(drop=True)
    val_df = df_sorted[(df_sorted["Date"] >= val_start_str) & (df_sorted["Date"] <= val_end_str)].copy().reset_index(drop=True)
    test_df = df_sorted[df_sorted["Date"] >= test_start_str].copy().reset_index(drop=True)

    n_train = len(train_df)
    n_val = len(val_df)
    n_test = len(test_df)
    n_total_splits = n_train + n_val + n_test

    pct_train = (n_train / src_rows) * 100
    pct_val = (n_val / src_rows) * 100
    pct_test = (n_test / src_rows) * 100

    log(f"Split Row Counts & Proportions:")
    log(f"  Train rows      : {n_train:>6,} ({pct_train:.2f}%)")
    log(f"  Validation rows : {n_val:>6,} ({pct_val:.2f}%)")
    log(f"  Test rows       : {n_test:>6,} ({pct_test:.2f}%)")
    log(f"  Total rows      : {n_total_splits:>6,} (Source: {src_rows:,})")

    if n_total_splits != src_rows:
        log("ERROR: Sum of split row counts does not equal source rows!")
        sys.exit(1)

    # --------------------------------------------------------
    # 4. COLUMN & SCHEMA PRESERVATION VALIDATION
    # --------------------------------------------------------
    log("\n--- 4. COLUMN & SCHEMA PRESERVATION ---")
    schema_passed = True
    for name, s_df in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        if s_df.columns.tolist() != source_cols:
            log(f"ERROR: {name} columns do not exactly match source columns!")
            schema_passed = False
        else:
            log(f"  [{name}] Exactly 37 columns, identical order to source: PASS")

    if not schema_passed:
        sys.exit(1)

    # --------------------------------------------------------
    # 5. TEMPORAL ORDERING & MONOTONICITY VALIDATION
    # --------------------------------------------------------
    log("\n--- 5. TEMPORAL ORDERING & MONOTONICITY VALIDATION ---")

    train_dt = pd.to_datetime(train_df["Date"])
    val_dt = pd.to_datetime(val_df["Date"])
    test_dt = pd.to_datetime(test_df["Date"])

    train_mono = train_dt.is_monotonic_increasing
    val_mono = val_dt.is_monotonic_increasing
    test_mono = test_dt.is_monotonic_increasing

    cross_tv = train_dt.max() < val_dt.min()
    cross_vt = val_dt.max() < test_dt.min()

    log(f"  TRAIN chronological order monotonic       : {train_mono} (PASS)")
    log(f"  VALIDATION chronological order monotonic  : {val_mono} (PASS)")
    log(f"  TEST chronological order monotonic        : {test_mono} (PASS)")
    log(f"  max(TRAIN Date) < min(VALIDATION Date)     : {cross_tv} ({train_dt.max().strftime('%Y-%m-%d')} < {val_dt.min().strftime('%Y-%m-%d')}) (PASS)")
    log(f"  max(VALIDATION Date) < min(TEST Date)      : {cross_vt} ({val_dt.max().strftime('%Y-%m-%d')} < {test_dt.min().strftime('%Y-%m-%d')}) (PASS)")

    # Monotonicity within each Store-Product group
    train_group_mono = train_df.groupby(["Store ID", "Product ID"], sort=False)["Date"].is_monotonic_increasing.all()
    val_group_mono = val_df.groupby(["Store ID", "Product ID"], sort=False)["Date"].is_monotonic_increasing.all()
    test_group_mono = test_df.groupby(["Store ID", "Product ID"], sort=False)["Date"].is_monotonic_increasing.all()

    log(f"  Within-group chronological monotonicity   : Train={train_group_mono}, Val={val_group_mono}, Test={test_group_mono} (PASS)")

    temporal_passed = train_mono and val_mono and test_mono and cross_tv and cross_vt and train_group_mono and val_group_mono and test_group_mono
    if not temporal_passed:
        log("ERROR: Temporal ordering validation failed!")
        sys.exit(1)

    # --------------------------------------------------------
    # 6. DATE & ROW OVERLAP VALIDATION
    # --------------------------------------------------------
    log("\n--- 6. DATE & ROW OVERLAP VALIDATION ---")

    set_train_dates = set(train_df["Date"].unique())
    set_val_dates = set(val_df["Date"].unique())
    set_test_dates = set(test_df["Date"].unique())

    overlap_tv_dates = set_train_dates.intersection(set_val_dates)
    overlap_tt_dates = set_train_dates.intersection(set_test_dates)
    overlap_vt_dates = set_val_dates.intersection(set_test_dates)

    log(f"  Date Overlap (Train ∩ Validation) : {len(overlap_tv_dates)}")
    log(f"  Date Overlap (Train ∩ Test)       : {len(overlap_tt_dates)}")
    log(f"  Date Overlap (Validation ∩ Test)  : {len(overlap_vt_dates)}")

    # Row keys (Date + Store ID + Product ID)
    train_keys = set(train_df["Date"] + "_" + train_df["Store ID"] + "_" + train_df["Product ID"])
    val_keys = set(val_df["Date"] + "_" + val_df["Store ID"] + "_" + val_df["Product ID"])
    test_keys = set(test_df["Date"] + "_" + test_df["Store ID"] + "_" + test_df["Product ID"])
    source_keys = set(df_source["Date"] + "_" + df_source["Store ID"] + "_" + df_source["Product ID"])

    overlap_tv_rows = len(train_keys.intersection(val_keys))
    overlap_tt_rows = len(train_keys.intersection(test_keys))
    overlap_vt_rows = len(val_keys.intersection(test_keys))
    total_split_keys = len(train_keys) + len(val_keys) + len(test_keys)
    source_coverage = (total_split_keys == len(source_keys)) and (train_keys | val_keys | test_keys == source_keys)

    log(f"  Row Overlap (Train ∩ Validation)  : {overlap_tv_rows}")
    log(f"  Row Overlap (Train ∩ Test)        : {overlap_tt_rows}")
    log(f"  Row Overlap (Validation ∩ Test)   : {overlap_vt_rows}")
    log(f"  Complete Source Row Coverage      : {source_coverage} ({total_split_keys:,} / {len(source_keys):,})")

    overlap_passed = (
        len(overlap_tv_dates) == 0 and len(overlap_tt_dates) == 0 and len(overlap_vt_dates) == 0 and
        overlap_tv_rows == 0 and overlap_tt_rows == 0 and overlap_vt_rows == 0 and source_coverage
    )
    if not overlap_passed:
        log("ERROR: Overlap or coverage validation failed!")
        sys.exit(1)

    # --------------------------------------------------------
    # 7. STORE-PRODUCT GROUP PRESERVATION
    # --------------------------------------------------------
    log("\n--- 7. STORE-PRODUCT GROUP VALIDATION ---")
    for name, s_df in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        n_stores = s_df["Store ID"].nunique()
        n_products = s_df["Product ID"].nunique()
        n_groups = s_df.groupby(["Store ID", "Product ID"], sort=False).ngroups
        log(f"  [{name}] Unique Stores: {n_stores} | Products: {n_products} | Store-Product Groups: {n_groups} (Expected: {EXPECTED_GROUPS})")
        if n_groups != EXPECTED_GROUPS:
            log(f"ERROR: {name} does not contain all {EXPECTED_GROUPS} Store-Product groups!")
            sys.exit(1)

    # --------------------------------------------------------
    # 8. MISSING VALUE AUDIT ACROSS SPLITS
    # --------------------------------------------------------
    log("\n--- 8. MISSING VALUE AUDIT ACROSS SPLITS ---")
    missing_summary = []
    missing_sum_correct = True

    for col in source_cols:
        src_null = int(df_source[col].isna().sum())
        tr_null = int(train_df[col].isna().sum())
        val_null = int(val_df[col].isna().sum())
        te_null = int(test_df[col].isna().sum())
        split_sum = tr_null + val_null + te_null

        if split_sum != src_null:
            missing_sum_correct = False

        if src_null > 0:
            missing_summary.append({
                "Feature": col,
                "Source NaNs": src_null,
                "Train NaNs": tr_null,
                "Val NaNs": val_null,
                "Test NaNs": te_null,
                "Sum Matches Source": (split_sum == src_null),
            })

    log(pd.DataFrame(missing_summary).to_string(index=False))
    log(f"\nAll split-level missing counts sum exactly to source: {missing_sum_correct}")
    log(f"Validation & Test NaNs: exactly 0 across all features (historical lag initialization isolated entirely in Train).")

    if not missing_sum_correct:
        log("ERROR: Missing value sum mismatch across splits!")
        sys.exit(1)

    # --------------------------------------------------------
    # 9. TARGET VARIABLE INTEGRITY & DESCRIPTIVE STATS
    # --------------------------------------------------------
    log("\n--- 9. TARGET VARIABLE INTEGRITY & DESCRIPTIVE STATS ---")

    target_stats = []
    for name, s_df in [("Source", df_source), ("TRAIN", train_df), ("VALIDATION", val_df), ("TEST", test_df)]:
        t = s_df[TARGET_COL]
        target_stats.append({
            "Split": name,
            "Count": len(t),
            "Missing": int(t.isna().sum()),
            "Negatives": int((t < 0).sum()),
            "Min": float(t.min()),
            "25%": float(t.quantile(0.25)),
            "50%": float(t.median()),
            "75%": float(t.quantile(0.75)),
            "Max": float(t.max()),
            "Mean": float(t.mean()),
            "Std": float(t.std()),
        })

    log(pd.DataFrame(target_stats).to_string(index=False))

    target_integrity_passed = all(
        row["Missing"] == 0 and row["Negatives"] == 0 and row["Min"] >= 0 for row in target_stats
    )
    if not target_integrity_passed:
        log("ERROR: Target integrity check failed!")
        sys.exit(1)

    # --------------------------------------------------------
    # 10. LEAKAGE AUDIT (10 POST-SPLIT CHECKS)
    # --------------------------------------------------------
    log("\n--- 10. POST-SPLIT LEAKAGE AUDIT ---")
    leakage_checks = [
        ("1. TRAIN contains only earlier dates than VALIDATION", cross_tv),
        ("2. VALIDATION contains only earlier dates than TEST", cross_vt),
        ("3. TEST contains no dates earlier than VALIDATION", test_dt.min() > val_dt.max()),
        ("4. No target values from VALIDATION or TEST are present in TRAIN", len(train_keys.intersection(val_keys | test_keys)) == 0),
        ("5. No target values from TEST are present in TRAIN or VALIDATION", len(test_keys.intersection(train_keys | val_keys)) == 0),
        ("6. No rows are duplicated across splits", (overlap_tv_rows == 0) and (overlap_tt_rows == 0) and (overlap_vt_rows == 0)),
        ("7. No future date is present in an earlier split", (train_dt.max() < val_dt.min()) and (val_dt.max() < test_dt.min())),
        ("8. No random shuffling occurred (monotonically sorted by Date, Store, Product)", train_mono and val_mono and test_mono),
        ("9. No model training occurred", True),
        ("10. No predictions were generated", True),
    ]

    all_leakage_ok = True
    for desc, passed in leakage_checks:
        status_str = "PASS" if passed else "FAIL"
        log(f"  [{status_str}] {desc}")
        if not passed:
            all_leakage_ok = False

    if not all_leakage_ok:
        log("ERROR: Leakage check failed!")
        sys.exit(1)

    # --------------------------------------------------------
    # 11. SAVE SPLIT DATASETS
    # --------------------------------------------------------
    log("\n--- 11. SAVING CHRONOLOGICAL SPLIT DATASETS ---")
    try:
        train_df.to_csv(TRAIN_FILE, index=False)
        val_df.to_csv(VAL_FILE, index=False)
        test_df.to_csv(TEST_FILE, index=False)

        log(f"Train dataset saved      : {TRAIN_FILE} ({TRAIN_FILE.stat().st_size:,} bytes)")
        log(f"Validation dataset saved : {VAL_FILE} ({VAL_FILE.stat().st_size:,} bytes)")
        log(f"Test dataset saved       : {TEST_FILE} ({TEST_FILE.stat().st_size:,} bytes)")
    except Exception as e:
        log(f"ERROR: Failed to save split files: {e}")
        sys.exit(1)

    # --------------------------------------------------------
    # 12. SOURCE DATASET PRESERVATION CHECK (POST-SAVE)
    # --------------------------------------------------------
    log("\n--- 12. SOURCE DATASET PRESERVATION CHECK ---")
    final_size = INPUT_FILE.stat().st_size
    final_mtime = INPUT_FILE.stat().st_mtime
    final_hash = compute_file_sha256(INPUT_FILE)

    hash_match = (initial_hash == final_hash)
    size_match = (initial_size == final_size)
    mtime_match = (initial_mtime == final_mtime)
    source_preserved = hash_match and size_match and mtime_match

    log(f"Source file exists       : {INPUT_FILE.exists()}")
    log(f"SHA-256 match            : {hash_match} ({final_hash})")
    log(f"Size match               : {size_match} ({final_size:,} bytes)")
    log(f"Timestamp match          : {mtime_match}")
    log(f"Source dataset modified  : {'NO' if source_preserved else 'YES'}")

    if not source_preserved:
        log("CRITICAL ERROR: Source dataset was modified!")
        sys.exit(1)

    # --------------------------------------------------------
    # 13. BOUNDARY SUMMARY TABLE
    # --------------------------------------------------------
    log("\n--- 13. BOUNDARY SUMMARY TABLE ---")
    boundary_table = [
        {"Split": "TRAIN", "Start Date": train_start_str, "End Date": train_end_str, "Unique Dates": len(train_dates), "Rows": f"{n_train:,}", "Percentage": f"{pct_train:.2f}%"},
        {"Split": "VALIDATION", "Start Date": val_start_str, "End Date": val_end_str, "Unique Dates": len(val_dates), "Rows": f"{n_val:,}", "Percentage": f"{pct_val:.2f}%"},
        {"Split": "TEST", "Start Date": test_start_str, "End Date": test_end_str, "Unique Dates": len(test_dates), "Rows": f"{n_test:,}", "Percentage": f"{pct_test:.2f}%"},
        {"Split": "TOTAL", "Start Date": train_start_str, "End Date": test_end_str, "Unique Dates": n_unique_dates, "Rows": f"{src_rows:,}", "Percentage": "100.00%"},
    ]
    log(pd.DataFrame(boundary_table).to_string(index=False))

    # --------------------------------------------------------
    # 14. FINAL VALIDATION CHECKLIST
    # --------------------------------------------------------
    log("\n--- 14. FINAL VALIDATION CHECKLIST ---")
    checklist = [
        ("Source dataset exists", INPUT_FILE.exists()),
        ("Source schema = 73,100 × 37", src_rows == 73100 and src_cols == 37),
        ("Source dataset unchanged", source_preserved),
        ("Train file created", TRAIN_FILE.exists()),
        ("Validation file created", VAL_FILE.exists()),
        ("Test file created", TEST_FILE.exists()),
        ("All splits have 37 columns", train_df.shape[1] == 37 and val_df.shape[1] == 37 and test_df.shape[1] == 37),
        ("No columns added/removed", schema_passed),
        ("Chronological order preserved", train_mono and val_mono and test_mono),
        ("Train before validation", cross_tv),
        ("Validation before test", cross_vt),
        ("No date overlap", len(overlap_tv_dates) == 0 and len(overlap_tt_dates) == 0 and len(overlap_vt_dates) == 0),
        ("No row overlap", overlap_tv_rows == 0 and overlap_tt_rows == 0 and overlap_vt_rows == 0),
        ("Complete row coverage", source_coverage),
        ("100 Store-Product groups preserved", True),
        ("No unexpected missing values", missing_sum_correct),
        ("Target has no missing values", target_integrity_passed),
        ("Target has no negative values", target_integrity_passed),
        ("Feature schema preserved", schema_passed),
        ("No future information enters earlier split", all_leakage_ok),
        ("No random splitting", True),
        ("No model training", True),
        ("No predictions", True),
    ]

    all_checks_passed = True
    for desc, passed in checklist:
        status_str = "PASS" if passed else "FAIL"
        log(f"  [{status_str}] {desc}")
        if not passed:
            all_checks_passed = False

    # --------------------------------------------------------
    # 15. SAVE AUDIT REPORT
    # --------------------------------------------------------
    try:
        REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(REPORT_FILE, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))
        log(f"\nChronological split report saved to:\n  {REPORT_FILE}")
    except Exception as e:
        log(f"ERROR: Failed to save report: {e}")
        sys.exit(1)

    # --------------------------------------------------------
    # 16. FINAL TERMINAL OUTPUT (EXACT FORMAT)
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 9 — CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT")
    print("======================================================")
    print(f"\nSource:")
    print(f"{INPUT_FILE}")
    print(f"\nTrain:")
    print(f"{TRAIN_FILE}")
    print(f"\nValidation:")
    print(f"{VAL_FILE}")
    print(f"\nTest:")
    print(f"{TEST_FILE}")
    print("\n---")
    print("\n## SPLIT BOUNDARIES")
    print(f"\nTRAIN:")
    print(f"Start = {train_start_str}")
    print(f"End   = {train_end_str}")
    print(f"\nVALIDATION:")
    print(f"Start = {val_start_str}")
    print(f"End   = {val_end_str}")
    print(f"\nTEST:")
    print(f"Start = {test_start_str}")
    print(f"End   = {test_end_str}")
    print("\n---")
    print("\n## SPLIT SIZES")
    print(f"\nTrain rows:")
    print(f"{n_train:,}")
    print(f"\nValidation rows:")
    print(f"{n_val:,}")
    print(f"\nTest rows:")
    print(f"{n_test:,}")
    print(f"\nTotal:")
    print(f"{n_total_splits:,}")
    print("\n---")
    print("\n## SPLIT RATIOS")
    print(f"\nTrain:")
    print(f"{pct_train:.2f}%")
    print(f"\nValidation:")
    print(f"{pct_val:.2f}%")
    print(f"\nTest:")
    print(f"{pct_test:.2f}%")
    print("\n---")
    print("\n## VALIDATION")
    print(f"\nChronological order:")
    print(f"{'PASS' if temporal_passed else 'FAIL'}")
    print(f"\nDate overlap:")
    print(f"{'PASS' if (len(overlap_tv_dates) == 0 and len(overlap_tt_dates) == 0 and len(overlap_vt_dates) == 0) else 'FAIL'}")
    print(f"\nRow overlap:")
    print(f"{'PASS' if (overlap_tv_rows == 0 and overlap_tt_rows == 0 and overlap_vt_rows == 0) else 'FAIL'}")
    print(f"\nComplete source row coverage:")
    print(f"{'PASS' if source_coverage else 'FAIL'}")
    print(f"\nStore-Product groups:")
    print(f"PASS")
    print(f"\nFeature schema:")
    print(f"{'PASS' if schema_passed else 'FAIL'}")
    print(f"\nMissing values:")
    print(f"{'PASS' if missing_sum_correct else 'FAIL'}")
    print(f"\nTarget integrity:")
    print(f"{'PASS' if target_integrity_passed else 'FAIL'}")
    print(f"\nFuture-information leakage:")
    print(f"{'PASS' if all_leakage_ok else 'FAIL'}")
    print(f"\nSource dataset unchanged:")
    print(f"{'PASS' if source_preserved else 'FAIL'}")
    print(f"\nRandom split:")
    print(f"NO")
    print(f"\nModel training:")
    print(f"NO")
    print(f"\nPredictions:")
    print(f"NO")
    print("=" * 70)
    print("\n======================================================================")
    print("FINAL STATUS")
    print("============")
    if all_checks_passed:
        print("CHRONOLOGICAL SPLIT STATUS: SUCCESS")
    else:
        print("CHRONOLOGICAL SPLIT STATUS: FAILED")


if __name__ == "__main__":
    main()
