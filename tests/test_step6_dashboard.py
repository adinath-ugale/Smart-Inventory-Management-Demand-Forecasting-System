"""
Automated Verification & Acceptance Test Suite for STEP 6 — SIM&DFS Dashboard.

Executes:
- Section 45: Tests 1..8 (Unit & Integration Policy Tests)
- Section 50: Acceptance Criteria 1..15 (Full System & Governance Verification)
"""

import math
import py_compile
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.config.policy_config import (
    ALLOWED_FALLBACK_SOURCES,
    ALLOWED_POLICY_STATUS,
    DATA_FILES,
    DEV_DATE_MAX,
    DEV_DATE_MIN,
    FROZEN_LEAD_TIME_DAYS,
    FROZEN_POLICY_VERSION,
    FROZEN_SERVICE_LEVEL,
    FROZEN_Z_SCORE,
    NAVIGATION_PAGES,
    PROTECTED_TEST_SHA256,
)
from dashboard.data.loaders import (
    load_backtest_results,
    load_development_dataset,
    load_inventory_dataset,
    load_leakage_audit,
    load_policy_config,
    load_scenario_results,
    load_trigger_diagnosis_results,
    load_validation_results,
)
from dashboard.data.validators import (
    assert_no_date_leakage,
    assert_no_forbidden_policy_inputs,
    assert_units_ordered_not_used_as_on_order,
    compute_frozen_policy_metrics,
    verify_test_split_checksum,
)
from dashboard.policy.policy_reader import (
    get_entity_selectors,
    get_historical_demand_series,
    get_policy_config,
    get_policy_record,
    get_scenario_matrix_for_record,
    get_system_overview_kpis,
)


def run_all_tests() -> None:
    print("=" * 76)
    print("STEP 6 — SIM&DFS DASHBOARD AUTOMATED TEST & ACCEPTANCE SUITE")
    print("=" * 76)

    passed_tests = []

    # ------------------------------------------------------------------
    # TEST 1: Policy Config Integrity
    # ------------------------------------------------------------------
    cfg = get_policy_config()
    assert cfg.policy_version == "STEP_5_7_FROZEN_v1.0"
    assert cfg.base_lead_time_days == 4
    assert cfg.base_service_level == 0.95
    assert cfg.base_z_score == 1.6449
    assert cfg.order_quantity_supported is False
    assert cfg.demand_forecast_used_in_policy is False
    assert cfg.units_ordered_used_as_on_order is False
    passed_tests.append(("Test 1 — Policy Config Integrity", "PASS"))

    # ------------------------------------------------------------------
    # TEST 2: Safety Stock Formula (sqrt(L), NEVER linear L)
    # ------------------------------------------------------------------
    metrics_t2 = compute_frozen_policy_metrics(
        demand_baseline=136.46,
        demand_std=64.79,
        inventory_level=274.43,
        lead_time_days=4,
        service_level=0.95,
        z_score=1.6449,
    )
    expected_ss = round(1.6449 * 64.79 * math.sqrt(4), 4)
    wrong_linear_ss = round(1.6449 * 64.79 * 4, 4)
    assert expected_ss == 213.1461, f"Expected 213.1461, got {expected_ss}"
    assert metrics_t2["safety_stock"] == 213.1461
    assert metrics_t2["safety_stock"] != wrong_linear_ss
    passed_tests.append(("Test 2 — Safety Stock Formula (sqrt(L) = 213.1461)", "PASS"))

    # ------------------------------------------------------------------
    # TEST 3: Reorder Point Formula
    # ------------------------------------------------------------------
    assert metrics_t2["lead_time_demand"] == 545.84
    assert metrics_t2["reorder_point"] == 758.9861
    passed_tests.append(("Test 3 — Reorder Point Formula (LTD=545.84, ROP=758.9861)", "PASS"))

    # ------------------------------------------------------------------
    # TEST 4: Decision Rule
    # ------------------------------------------------------------------
    assert metrics_t2["policy_status"] == "REORDER"
    metrics_monitor = compute_frozen_policy_metrics(
        demand_baseline=50.0,
        demand_std=10.0,
        inventory_level=450.0,
        lead_time_days=4,
        service_level=0.95,
        z_score=1.6449,
    )
    assert metrics_monitor["policy_status"] == "MONITOR"
    passed_tests.append(("Test 4 — Decision Rule (REORDER vs MONITOR)", "PASS"))

    # ------------------------------------------------------------------
    # TEST 5: Cold-Start Day 1 (2022-01-01 -> None/NaN & INSUFFICIENT_DATA)
    # ------------------------------------------------------------------
    dev_df = load_development_dataset()
    rec_day1 = get_policy_record("S001", "P0001", "2022-01-01", df=dev_df)
    assert rec_day1.demand_baseline is None
    assert rec_day1.demand_std is None
    assert rec_day1.lead_time_demand is None
    assert rec_day1.safety_stock is None
    assert rec_day1.reorder_point is None
    assert rec_day1.policy_status == "INSUFFICIENT_DATA"
    assert rec_day1.demand_baseline_source == "INSUFFICIENT_DATA"
    assert rec_day1.demand_std_source == "INSUFFICIENT_DATA"
    passed_tests.append(("Test 5 — Cold-Start Day 1 (2022-01-01 -> INSUFFICIENT_DATA)", "PASS"))

    # ------------------------------------------------------------------
    # TEST 6: Date Leakage Check (Raises error on t' >= t)
    # ------------------------------------------------------------------
    leakage_caught = False
    try:
        assert_no_date_leakage(["2022-05-10", "2022-05-11"], "2022-05-10")
    except ValueError:
        leakage_caught = True
    assert leakage_caught, "assert_no_date_leakage failed to raise ValueError on same/future date!"
    passed_tests.append(("Test 6 — Date Leakage Guard (t' < t enforced)", "PASS"))

    # ------------------------------------------------------------------
    # TEST 7: Visualization Window Does Not Change Policy
    # ------------------------------------------------------------------
    rec_base = get_policy_record("S001", "P0001", "2023-10-20", df=dev_df)
    for w in [30, 90, 180, None]:
        hist_slice, _ = get_historical_demand_series("S001", "P0001", "2023-10-20", window_days=w, df=dev_df)
        if w is not None:
            assert len(hist_slice) == w
        rec_after = get_policy_record("S001", "P0001", "2023-10-20", df=dev_df)
        assert rec_after.demand_baseline == rec_base.demand_baseline
        assert rec_after.demand_std == rec_base.demand_std
        assert rec_after.safety_stock == rec_base.safety_stock
        assert rec_after.reorder_point == rec_base.reorder_point
        assert rec_after.policy_status == rec_base.policy_status
    passed_tests.append(("Test 7 — Visualization Window Isolation (30/90/180/All)", "PASS"))

    # ------------------------------------------------------------------
    # TEST 8: TEST Split Protection & SHA-256 Verification
    # ------------------------------------------------------------------
    test_chk = verify_test_split_checksum()
    assert test_chk["status"] == "PASS"
    assert test_chk["actual_sha256"] == PROTECTED_TEST_SHA256
    assert "TEST" not in set(dev_df["Split_Label"].unique())
    assert dev_df["Date_Str"].min() == DEV_DATE_MIN
    assert dev_df["Date_Str"].max() == DEV_DATE_MAX
    assert len(dev_df) == 65800
    passed_tests.append(("Test 8 — TEST Split SHA-256 & Zero Leakage Protection", "PASS"))

    # ------------------------------------------------------------------
    # SECTION 50: FINAL ACCEPTANCE CRITERIA (1..15)
    # ------------------------------------------------------------------
    required_files = [
        PROJECT_ROOT / "app.py",
        PROJECT_ROOT / "dashboard" / "__init__.py",
        PROJECT_ROOT / "dashboard" / "config" / "policy_config.py",
        PROJECT_ROOT / "dashboard" / "data" / "loaders.py",
        PROJECT_ROOT / "dashboard" / "data" / "validators.py",
        PROJECT_ROOT / "dashboard" / "data" / "schemas.py",
        PROJECT_ROOT / "dashboard" / "policy" / "policy_reader.py",
        PROJECT_ROOT / "dashboard" / "components" / "header.py",
        PROJECT_ROOT / "dashboard" / "components" / "kpi_cards.py",
        PROJECT_ROOT / "dashboard" / "components" / "policy_status.py",
        PROJECT_ROOT / "dashboard" / "components" / "demand_chart.py",
        PROJECT_ROOT / "dashboard" / "components" / "inventory_chart.py",
        PROJECT_ROOT / "dashboard" / "components" / "safety_stock.py",
        PROJECT_ROOT / "dashboard" / "components" / "fallback_badge.py",
        PROJECT_ROOT / "dashboard" / "components" / "scenario_table.py",
        PROJECT_ROOT / "dashboard" / "components" / "backtest_panel.py",
        PROJECT_ROOT / "dashboard" / "components" / "validation_panel.py",
        PROJECT_ROOT / "dashboard" / "pages" / "overview.py",
        PROJECT_ROOT / "dashboard" / "pages" / "demand.py",
        PROJECT_ROOT / "dashboard" / "pages" / "inventory.py",
        PROJECT_ROOT / "dashboard" / "pages" / "scenarios.py",
        PROJECT_ROOT / "dashboard" / "pages" / "backtest.py",
        PROJECT_ROOT / "docs" / "STEP6_DASHBOARD_README.md",
    ]
    for rf in required_files:
        assert rf.exists(), f"Missing required file: {rf}"
        if rf.suffix == ".py":
            py_compile.compile(str(rf), doraise=True)

    assert len(NAVIGATION_PAGES) == 5
    scen_list = get_scenario_matrix_for_record(rec_base)
    assert len(scen_list) == 9
    base_scen = [s for s in scen_list if s.is_base_policy]
    assert len(base_scen) == 1 and base_scen[0].scenario_id == "LT4_SL95"

    overview_kpis = get_system_overview_kpis("2023-10-20", df=dev_df)
    assert overview_kpis["total_rows"] == 65800
    assert overview_kpis["reorder_count"] == 65700
    assert overview_kpis["monitor_count"] == 0
    assert overview_kpis["insufficient_count"] == 100
    assert overview_kpis["reorder_pct"] == 99.85

    for py_file in (PROJECT_ROOT / "dashboard").rglob("*.py"):
        src = py_file.read_text(encoding="utf-8")
        assert "st.slider(" not in src, f"Forbidden st.slider found in {py_file}"
        assert "st.number_input(" not in src, f"Forbidden st.number_input found in {py_file}"

    passed_tests.append(("Acceptance Criteria 1..15 — Complete System & UI Governance", "PASS"))

    for name, status in passed_tests:
        print(f"  [{status}] {name}")
    print("=" * 76)
    print(f"ALL {len(passed_tests)} TEST SUITES & 15 ACCEPTANCE CRITERIA PASSED SUCCESSFULLY.")
    print("=" * 76)


if __name__ == "__main__":
    run_all_tests()
