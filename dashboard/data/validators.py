"""
Startup and Runtime Data & Policy Safeguards for the SIM&DFS Step 6 Dashboard.
Enforces TEST checksum protection, pre-decision date cutoffs (Date < D),
forbidden input blocking (Demand Forecast, Future Demand, Units Ordered as On-Order),
and frozen formula consistency verification.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from dashboard.config.policy_config import (
    ALLOWED_LEAD_TIMES,
    ALLOWED_POLICY_STATUSES,
    DATA_FILES,
    FORBIDDEN_POLICY_INPUTS,
    FROZEN_BASE_LEAD_TIME_DAYS,
    FROZEN_BASE_SERVICE_LEVEL,
    FROZEN_BASE_Z_VALUE,
    PROTECTED_TEST_SHA256,
    TEST_START_DATE,
)


class TestIntegrityError(RuntimeError):
    """Raised when the reserved TEST dataset checksum mismatches or TEST rows leak."""


class LeakageViolationError(ValueError):
    """Raised when decision-date leakage, forbidden inputs, or unverified inventory position is detected."""


class DatasetSchemaError(RuntimeError):
    """Raised when required columns or data types are missing or invalid."""


def calculate_file_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file without modifying it."""
    if not filepath.exists():
        raise FileNotFoundError(f"Protected file not found: {filepath}")
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_test_split_checksum(
    test_path: Optional[Path] = None,
    expected_sha256: str = PROTECTED_TEST_SHA256,
) -> Dict[str, Any]:
    """
    Verify that data/processed/splits/test.csv matches the protected SHA-256 checksum.
    Raises TestIntegrityError if tampered.
    """
    target_path = test_path or DATA_FILES["test_split_protected"]
    actual_sha256 = calculate_file_sha256(target_path)
    if actual_sha256 != expected_sha256:
        raise TestIntegrityError(
            "TEST INTEGRITY FAILURE\n\n"
            "The reserved TEST dataset does not match the protected checksum.\n"
            f"Expected: {expected_sha256}\n"
            f"Actual  : {actual_sha256}\n"
            "Dashboard development/evaluation has been halted."
        )
    return {
        "status": "PASS",
        "path": str(target_path),
        "expected_sha256": expected_sha256,
        "actual_sha256": actual_sha256,
    }


def assert_no_date_leakage(
    historical_dates: Iterable[pd.Timestamp | str],
    decision_date: pd.Timestamp | str,
) -> bool:
    """
    Enforce Section 33 Date-Based Leakage Protection:
      assert all(historical_dates < decision_date)
    Raises LeakageViolationError (subclass of ValueError) if any historical date >= decision_date.
    """
    dec_dt = pd.Timestamp(decision_date)
    hist_dts = [pd.Timestamp(d) for d in historical_dates]
    if not all(h_dt < dec_dt for h_dt in hist_dts):
        violating = [str(h_dt.date()) for h_dt in hist_dts if h_dt >= dec_dt][:5]
        raise LeakageViolationError(
            f"DATE LEAKAGE BLOCKED: Historical demand dates must satisfy Date < {dec_dt.date()}. "
            f"Violating dates detected: {violating}"
        )
    if dec_dt >= pd.Timestamp(TEST_START_DATE):
        raise LeakageViolationError(
            f"TEST ACCESS BLOCKED: Decision date {dec_dt.date()} falls in the reserved TEST period "
            f"(>= {TEST_START_DATE})."
        )
    return True


def assert_no_forbidden_policy_inputs(input_fields: Sequence[str]) -> bool:
    """
    Enforce Sections 34 & 35: Block Demand Forecast, Future_Demand, or TEST Units Sold
    from entering policy calculations.
    """
    forbidden_set = {f.lower() for f in FORBIDDEN_POLICY_INPUTS}
    for field in input_fields:
        if field.strip().lower() in forbidden_set:
            raise LeakageViolationError(
                f"FORBIDDEN POLICY INPUT BLOCKED: '{field}' is strictly prohibited as an input to the "
                "frozen Step 5.7 inventory policy."
            )
    return True


def assert_units_ordered_not_used_as_on_order(
    inventory_level: float = 0.0,
    effective_inventory_position: float = 0.0,
    units_ordered: float = 0.0,
    include_units_ordered_in_position: bool = False,
) -> bool:
    """
    Enforce Section 9 & Section 45 Test 8: Block treating Units Ordered as On-Order Inventory
    (Inventory Position = Inventory Level + Units Ordered).
    """
    if include_units_ordered_in_position or abs(effective_inventory_position - inventory_level) > 1e-6:
        raise LeakageViolationError(
            "UNVERIFIED INVENTORY POSITION BLOCKED: 'Units Ordered' timing/on-order semantics are "
            "unverified in the source dataset. Calculating Inventory Position = Inventory Level + Units Ordered "
            "is strictly prohibited under the frozen Step 5.7 policy."
        )
    return True


def compute_frozen_policy_metrics(
    demand_baseline: Optional[float] = None,
    demand_std: Optional[float] = None,
    inventory_level: Optional[float] = None,
    current_inventory: Optional[float] = None,
    lead_time_days: int = FROZEN_BASE_LEAD_TIME_DAYS,
    service_level: float = FROZEN_BASE_SERVICE_LEVEL,
    z_score: float = FROZEN_BASE_Z_VALUE,
    z_value: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Evaluate the exact frozen Step 5.7 inventory formulas:
      - Lead_Time_Demand = Demand_Baseline * Lead_Time_Days
      - Safety_Stock     = Z * Demand_Std * sqrt(Lead_Time_Days)
      - Reorder_Point    = Lead_Time_Demand + Safety_Stock
      - Policy_Status    = 'INSUFFICIENT_DATA' if inputs missing else ('REORDER' if Inventory <= ROP else 'MONITOR')
    """
    if lead_time_days not in ALLOWED_LEAD_TIMES:
        raise ValueError(
            f"Lead time {lead_time_days} not in allowed scenario set {ALLOWED_LEAD_TIMES}"
        )

    inv_val = inventory_level if inventory_level is not None else current_inventory
    eff_z = z_value if z_value is not None else z_score

    if (
        inv_val is None
        or demand_baseline is None
        or demand_std is None
        or pd.isna(inv_val)
        or pd.isna(demand_baseline)
        or pd.isna(demand_std)
    ):
        return {
            "lead_time_demand": None,
            "safety_stock": None,
            "reorder_point": None,
            "policy_status": "INSUFFICIENT_DATA",
        }

    ltd = round(float(demand_baseline) * float(lead_time_days), 4)
    # CRITICAL: Frozen formula uses square root of lead time: Z * Demand_Std * sqrt(L)
    ss = round(float(eff_z) * float(demand_std) * math.sqrt(float(lead_time_days)), 4)
    rop = round(ltd + ss, 4)

    status = "REORDER" if float(inv_val) <= rop else "MONITOR"
    return {
        "lead_time_demand": ltd,
        "safety_stock": ss,
        "reorder_point": rop,
        "policy_status": status,
    }


def validate_development_dataset_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate the development inventory optimization dataset at startup (Section 42).
    Ensures required columns exist, types are valid, and zero TEST rows are present.
    """
    required_cols = [
        "Date",
        "Store ID",
        "Product ID",
        "Store_Product_ID",
        "Category",
        "Region",
        "Inventory Level",
        "Units Ordered",
        "Units Sold",
        "Historical_Demand_Count",
        "Demand_Baseline",
        "Demand_Std",
        "Demand_Baseline_Source",
        "Demand_Std_Source",
        "Lead_Time_Days",
        "Lead_Time_Demand",
        "Service_Level",
        "Service_Level_Z",
        "Safety_Stock",
        "Reorder_Point",
        "Current_Inventory",
        "inventory_level_time_semantics",
        "Inventory_Action",
        "Historical_Cutoff_Date",
        "Demand_History_Leakage_Check",
        "Test_Contamination_Check",
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise DatasetSchemaError(f"Dataset is missing required columns: {missing_cols}")

    if "Demand Forecast" in df.columns:
        raise LeakageViolationError(
            "Demand Forecast column must not be present in the policy dataset."
        )

    dates = pd.to_datetime(df["Date"], errors="raise")
    if (dates >= pd.Timestamp(TEST_START_DATE)).any():
        raise TestIntegrityError(
            f"TEST rows (>= {TEST_START_DATE}) detected inside development dataset!"
        )

    if not pd.api.types.is_numeric_dtype(df["Units Sold"]):
        raise DatasetSchemaError("'Units Sold' must be numeric.")
    if not pd.api.types.is_numeric_dtype(df["Inventory Level"]):
        raise DatasetSchemaError("'Inventory Level' must be numeric.")

    invalid_actions = set(df["Inventory_Action"].dropna().unique()) - set(
        ALLOWED_POLICY_STATUSES
    )
    if invalid_actions:
        raise DatasetSchemaError(
            f"Invalid Inventory_Action values detected: {invalid_actions}"
        )

    if not (df["Demand_History_Leakage_Check"] == "PASS").all():
        raise LeakageViolationError(
            "Demand_History_Leakage_Check contains non-PASS rows."
        )
    if not (df["Test_Contamination_Check"] == "PASS").all():
        raise LeakageViolationError("Test_Contamination_Check contains non-PASS rows.")

    return True, ["All dataset schema, type, and TEST-isolation checks PASSED."]
