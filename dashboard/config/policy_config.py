"""
Centralized Read-Only Path Mapping and Frozen Step 5.7 Policy Constants
for the Smart Inventory Management & Demand Forecasting System (SIM&DFS).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Union

# Resolve project root dynamically relative to this file (C:\SIM&DFS)
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# Protected TEST split checksum (Step 9 / Step 5 / Step 5.5 / Step 5.6 / Step 5.7)
PROTECTED_TEST_SHA256: str = (
    "4fd35bc0da8548ed50e347d6d0a00f925d75a40ee3382f79f51dff3462edbe96"
)

# Development date boundaries (TEST period 2023-10-21 to 2024-01-01 is strictly locked)
DEV_START_DATE: str = "2022-01-01"
DEV_END_DATE: str = "2023-10-20"
DEV_DATE_MIN: str = DEV_START_DATE
DEV_DATE_MAX: str = DEV_END_DATE
TEST_START_DATE: str = "2023-10-21"
TEST_END_DATE: str = "2024-01-01"

# Frozen Base Policy Parameters (Step 5.7)
FROZEN_POLICY_VERSION: str = "STEP_5_7_FROZEN_v1.0"
FROZEN_BASE_LEAD_TIME_DAYS: int = 4
FROZEN_BASE_SERVICE_LEVEL: float = 0.95
FROZEN_BASE_Z_VALUE: float = 1.6449

# Aliases used across policy_reader and UI components
FROZEN_LEAD_TIME_DAYS: int = FROZEN_BASE_LEAD_TIME_DAYS
FROZEN_SERVICE_LEVEL: float = FROZEN_BASE_SERVICE_LEVEL
FROZEN_Z_SCORE: float = FROZEN_BASE_Z_VALUE

ALLOWED_LEAD_TIMES: List[int] = [2, 4, 7]
ALLOWED_SERVICE_LEVELS: List[float] = [0.90, 0.95, 0.99]
SCENARIO_LEAD_TIMES: List[int] = ALLOWED_LEAD_TIMES
SCENARIO_SERVICE_LEVELS: List[float] = ALLOWED_SERVICE_LEVELS

FROZEN_Z_MAP: Dict[str, float] = {
    "0.90": 1.2816,
    "0.95": 1.6449,
    "0.99": 2.3263,
}
SCENARIO_Z_SCORES: Dict[float, float] = {
    0.90: 1.2816,
    0.95: 1.6449,
    0.99: 2.3263,
}

ALLOWED_FALLBACK_SOURCES: List[str] = [
    "STORE_PRODUCT",
    "PRODUCT",
    "STORE",
    "GLOBAL_TRAIN",
    "INSUFFICIENT_DATA",
]

ALLOWED_POLICY_STATUSES: List[str] = [
    "REORDER",
    "MONITOR",
    "INSUFFICIENT_DATA",
]
ALLOWED_POLICY_STATUS: List[str] = ALLOWED_POLICY_STATUSES

ALLOWED_HISTORY_WINDOWS: List[Union[int, str]] = [30, 90, 180, "All"]

NAVIGATION_PAGES: List[str] = [
    "Overview",
    "Demand Analysis",
    "Inventory Decision",
    "Scenario Comparison",
    "Backtest & Validation",
]

STATUS_COLORS: Dict[str, str] = {
    "REORDER": "#dc2626",
    "MONITOR": "#16a34a",
    "INSUFFICIENT_DATA": "#d97706",
}

FORBIDDEN_POLICY_INPUTS: List[str] = [
    "Demand Forecast",
    "Future_Demand",
    "Future_Demand_2D",
    "Future_Demand_4D",
    "Future_Demand_7D",
    "Future Units Sold",
    "TEST Units Sold",
]
FORBIDDEN_POLICY_INPUT_COLUMNS: List[str] = FORBIDDEN_POLICY_INPUTS

# Centralized Read-Only Data File Mapping (Section 17)
DATA_FILES: Dict[str, Path] = {
    "policy_config": PROJECT_ROOT
    / "data"
    / "processed"
    / "inventory_optimization"
    / "final_policy"
    / "final_policy_locked_config.json",
    "inventory_dataset": PROJECT_ROOT
    / "data"
    / "processed"
    / "inventory_optimization_dataset.csv",
    "development_dataset": PROJECT_ROOT
    / "data"
    / "processed"
    / "inventory_optimization"
    / "inventory_optimization_train_val.csv",
    "scenario_summary": PROJECT_ROOT
    / "reports"
    / "step5_inventory_optimization"
    / "scenario_sensitivity_summary.csv",
    "scenario_backtest": PROJECT_ROOT
    / "data"
    / "processed"
    / "inventory_optimization"
    / "backtest"
    / "scenario_backtest_comparison.csv",
    "horizon_backtest": PROJECT_ROOT
    / "data"
    / "processed"
    / "inventory_optimization"
    / "backtest"
    / "horizon_backtest_summary.csv",
    "store_product_backtest": PROJECT_ROOT
    / "data"
    / "processed"
    / "inventory_optimization"
    / "backtest"
    / "store_product_backtest.csv",
    "base_policy_outcomes": PROJECT_ROOT
    / "data"
    / "processed"
    / "inventory_optimization"
    / "backtest"
    / "base_policy_outcomes.csv",
    "base_decision_outcome_matrix": PROJECT_ROOT
    / "data"
    / "processed"
    / "inventory_optimization"
    / "backtest"
    / "base_decision_outcome_matrix.csv",
    "zero_inventory_diagnostic": PROJECT_ROOT
    / "data"
    / "processed"
    / "inventory_optimization"
    / "backtest"
    / "zero_inventory_diagnostic.csv",
    "validation_checks": PROJECT_ROOT
    / "data"
    / "processed"
    / "inventory_optimization"
    / "backtest"
    / "backtest_validation_checks.csv",
    "step5_validation_checks": PROJECT_ROOT
    / "reports"
    / "step5_inventory_optimization"
    / "validation_checks_report.csv",
    "step5_5_validation_checks": PROJECT_ROOT
    / "data"
    / "processed"
    / "inventory_optimization"
    / "trigger_diagnosis"
    / "diagnostic_validation_checks.csv",
    "leakage_audit": PROJECT_ROOT
    / "data"
    / "processed"
    / "inventory_optimization"
    / "backtest"
    / "backtest_leakage_audit.csv",
    "trigger_four_way": PROJECT_ROOT
    / "data"
    / "processed"
    / "inventory_optimization"
    / "trigger_diagnosis"
    / "base_four_way_decomposition.csv",
    "test_split_protected": PROJECT_ROOT
    / "data"
    / "processed"
    / "splits"
    / "test.csv",
}
