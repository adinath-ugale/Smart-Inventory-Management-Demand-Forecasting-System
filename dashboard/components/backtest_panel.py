"""Step 5.6 historical policy backtest and Step 5.5 trigger-rate diagnosis panels."""

from typing import Dict
import pandas as pd
import streamlit as st


def render_backtest_confusion_and_metrics(backtest_data: Dict[str, pd.DataFrame]) -> None:
    """
    Render the Step 5.6 historical backtest decision outcome matrix (`base_decision_outcome_matrix`),
    `base_policy_outcomes`, and `horizon_backtest` summary.
    """
    col1, col2 = st.columns([1.05, 1.0])
    with col1:
        st.markdown("#### 📊 Step 5.6 Base Policy Decision vs. Future Lead-Time Demand Matrix (`4-Day` Horizon)")
        matrix_df = backtest_data["base_decision_outcome_matrix"]
        st.dataframe(matrix_df, use_container_width=True, hide_index=True)

        st.markdown("#### 📈 Multi-Horizon Backtest Summary (`2d`, `4d`, `7d`)")
        st.dataframe(backtest_data["horizon_backtest"], use_container_width=True, hide_index=True)

    with col2:
        st.markdown("#### 🔄 Step 5.6 Base Policy Outcomes & Zero-Inventory Diagnostic")
        st.dataframe(backtest_data["base_policy_outcomes"], use_container_width=True, hide_index=True)
        st.markdown("#### 🛡️ Zero-Inventory Semantics Diagnostic")
        st.dataframe(backtest_data["zero_inventory_diagnostic"], use_container_width=True, hide_index=True)


def render_trigger_diagnosis_panel(trigger_data: Dict[str, pd.DataFrame]) -> None:
    """
    Render the Step 5.5 Trigger-Rate Diagnosis four-way decomposition explaining why the base policy triggers
    `REORDER` on `65,700 / 65,700` active post-cold-start rows (`99.85%` of all `65,800` rows).
    """
    four_way_df = trigger_data["four_way"]
    step55_checks = trigger_data["validation_checks"]

    st.markdown("#### 🔍 Step 5.5 Trigger-Rate Diagnosis (`Four-Way Decomposition`)")
    st.markdown(
        """
        Step 5.5 audited why `REORDER` triggers on `65,700` of `65,700` active post-cold-start rows (`99.85%` of `65,800` total rows):
        - **Bucket A (`Current_Inventory <= LTD`)**: `65,355 rows` (`99.47%` of active rows) — on-hand inventory (`mean 274.52 units`) is already below 4-day Lead-Time Demand (`mean 546.80 units`) before adding any Safety Stock.
        - **Bucket B (`LTD < Current_Inventory <= ROP`)**: `345 rows` (`0.53%` of active rows) — Safety Stock (`mean 357.27 units`, `ROP mean 904.07 units`) triggers reorder to buffer lead-time demand volatility.
        - **Bucket C (`Current_Inventory > ROP`)**: `0 rows` under `L = 4` (`10,165 rows` under `LT2_SL90`).
        - **Bucket D (`Cold-Start Day 1 INSUFFICIENT_DATA`)**: `100 rows` (`2022-01-01`).
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Step 5.5 Four-Way Trigger Decomposition**")
        st.dataframe(four_way_df, use_container_width=True, hide_index=True)
    with col2:
        st.markdown("**Step 5.5 Trigger Diagnosis Validation Suite (`12 / 12 PASS`)**")
        st.dataframe(step55_checks, use_container_width=True, hide_index=True)
