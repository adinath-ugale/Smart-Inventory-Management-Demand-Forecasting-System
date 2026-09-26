r"""
SMART INVENTORY MANAGEMENT & DEMAND FORECASTING SYSTEM (SIM&DFS)
STEP 6 — PRODUCTION DASHBOARD APPLICATION (FROZEN STEP 5.7 POLICY)

Entrypoint: `C:\SIM&DFS\app.py`
Run via: `py -m streamlit run app.py`
"""

import streamlit as st

from dashboard.config.policy_config import (
    DEV_DATE_MAX,
    DEV_DATE_MIN,
    FROZEN_LEAD_TIME_DAYS,
    FROZEN_POLICY_VERSION,
    FROZEN_SERVICE_LEVEL,
    FROZEN_Z_SCORE,
    NAVIGATION_PAGES,
)
from dashboard.data.loaders import (
    load_backtest_results,
    load_development_dataset,
    load_leakage_audit,
    load_trigger_diagnosis_results,
    load_validation_results,
)
from dashboard.data.validators import verify_test_split_checksum
from dashboard.pages.backtest import render_backtest_validation_page
from dashboard.pages.demand import render_demand_analysis_page
from dashboard.pages.inventory import render_inventory_decision_page
from dashboard.pages.overview import render_overview_page
from dashboard.pages.scenarios import render_scenario_comparison_page
from dashboard.policy.policy_reader import (
    get_entity_selectors,
    get_policy_config,
    get_policy_record,
)


@st.cache_data(show_spinner=False)
def _cached_startup_bundle():
    """Load and validate all frozen Step 5 / 5.5 / 5.6 / 5.7 datasets once."""
    test_check = verify_test_split_checksum()
    policy_cfg = get_policy_config()
    dev_df = load_development_dataset()
    selectors = get_entity_selectors(dev_df)
    backtest_data = load_backtest_results()
    trigger_data = load_trigger_diagnosis_results()
    val_df = load_validation_results()
    leakage_df = load_leakage_audit()
    return {
        "test_check": test_check,
        "policy_cfg": policy_cfg,
        "dev_df": dev_df,
        "selectors": selectors,
        "backtest_data": backtest_data,
        "trigger_data": trigger_data,
        "val_df": val_df,
        "leakage_df": leakage_df,
    }


def main() -> None:
    st.set_page_config(
        page_title="SIM&DFS — Frozen Step 5.7 Inventory Dashboard",
        page_icon="📦",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # 1. Startup TEST split SHA-256 verification and dataset schema check
    try:
        bundle = _cached_startup_bundle()
    except Exception as exc:
        st.error(
            f"🚨 **CRITICAL POLICY / DATA INTEGRITY FAILURE AT STARTUP**:\n\n`{exc}`\n\n"
            "Dashboard execution halted to protect frozen policy and TEST split integrity."
        )
        st.stop()
        return

    test_check = bundle["test_check"]
    policy_cfg = bundle["policy_cfg"]
    dev_df = bundle["dev_df"]
    selectors = bundle["selectors"]
    backtest_data = bundle["backtest_data"]
    trigger_data = bundle["trigger_data"]
    val_df = bundle["val_df"]
    leakage_df = bundle["leakage_df"]

    # 2. Sidebar Layout (Strictly Read-Only Policy Controls + Entity/Date Selectors)
    with st.sidebar:
        st.markdown(
            """
            <div style="padding:8px 0 4px 0;">
                <div style="font-size:11px; color:#38bdf8; font-weight:700; letter-spacing:1px;">
                    SIM&amp;DFS STEP 6 APPLICATION
                </div>
                <div style="font-size:18px; font-weight:700; color:#f8fafc;">
                    Inventory Decision Suite
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        selected_page = st.radio(
            "Navigation",
            options=NAVIGATION_PAGES,
            index=0,
        )

        st.markdown("---")
        st.markdown("#### 🎯 Entity & Causal Date Selector")

        selected_store = st.selectbox(
            "Store ID",
            options=selectors["stores"],
            index=0,
        )
        selected_product = st.selectbox(
            "Product ID",
            options=selectors["products"],
            index=0,
        )

        # Default to a mid-series date with complete >30d history (`2023-10-20` latest validation date)
        default_date_idx = len(selectors["dates"]) - 1
        selected_date = st.selectbox(
            f"Decision Date ({DEV_DATE_MIN} to {DEV_DATE_MAX})",
            options=selectors["dates"],
            index=default_date_idx,
            help="Restricted strictly to TRAIN + VALIDATION dates (2022-01-01 to 2023-10-20). Protected TEST dates are excluded.",
        )

        st.markdown("---")
        st.markdown(
            f"""
            <div style="background:#0f172a; border:1px solid #334155; border-radius:8px; padding:12px; font-size:12px; color:#e2e8f0;">
                <div style="font-weight:700; color:#38bdf8; margin-bottom:6px;">
                    🔒 Frozen Policy ({FROZEN_POLICY_VERSION})
                </div>
                <div>• <strong>Lead Time (L):</strong> <code>{FROZEN_LEAD_TIME_DAYS} days</code> (Locked)</div>
                <div>• <strong>Service Level:</strong> <code>{int(FROZEN_SERVICE_LEVEL*100)}%</code> (Locked)</div>
                <div>• <strong>Z-Score:</strong> <code>{FROZEN_Z_SCORE}</code> (Locked)</div>
                <div>• <strong>Safety Stock:</strong> <code>Z × Std × √L</code></div>
                <div>• <strong>Reorder Point:</strong> <code>LTD + SS</code></div>
                <div>• <strong>Trigger Rule:</strong> <code>Inv ≤ ROP ⇒ REORDER</code></div>
                <div>• <strong>Order Quantity:</strong> <code>Unsupported (False)</code></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("📋 Policy Assumptions & Guardrails", expanded=False):
            st.markdown(
                """
                1. **No Policy Mutation**: Lead Time, Service Level, Z-score, and formulas are frozen from Step 5.7.
                2. **Causal Boundary (`t' < t`)**: Only strictly prior observations enter `Demand_Baseline` and `Demand_Std`.
                3. **Cold-Start Preservation**: Day 1 (`2022-01-01`) preserves `NaN` and returns `INSUFFICIENT_DATA`.
                4. **On-Hand Evaluation**: `Units Ordered` is never added to `Inventory Level`.
                5. **TEST Split Isolation**: `test.csv` (`2023-10-21`..`2024-01-01`) is verified by SHA-256 (`4fd35bc...`) and never displayed.
                """
            )

    # 3. Retrieve frozen policy record for the active selection
    record = get_policy_record(
        store_id=selected_store,
        product_id=selected_product,
        decision_date=selected_date,
        df=dev_df,
    )

    # 4. Dispatch to the selected page
    if selected_page == "Overview":
        render_overview_page(df=dev_df, policy_cfg=policy_cfg, record=record)
    elif selected_page == "Demand Analysis":
        render_demand_analysis_page(df=dev_df, policy_cfg=policy_cfg, record=record)
    elif selected_page == "Inventory Decision":
        render_inventory_decision_page(df=dev_df, policy_cfg=policy_cfg, record=record)
    elif selected_page == "Scenario Comparison":
        render_scenario_comparison_page(
            policy_cfg=policy_cfg,
            record=record,
            backtest_data=backtest_data,
        )
    elif selected_page == "Backtest & Validation":
        render_backtest_validation_page(
            policy_cfg=policy_cfg,
            record=record,
            backtest_data=backtest_data,
            trigger_data=trigger_data,
            val_df=val_df,
            leakage_df=leakage_df,
            test_check_info=test_check,
        )


if __name__ == "__main__":
    main()
