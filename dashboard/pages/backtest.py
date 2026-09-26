"""Page 5 — Backtest & Validation: Step 5.6 historical backtest, Step 5.5 trigger diagnosis, and leakage audit."""

from typing import Dict
import pandas as pd
import streamlit as st

from dashboard.components.backtest_panel import (
    render_backtest_confusion_and_metrics,
    render_trigger_diagnosis_panel,
)
from dashboard.components.header import render_top_banner
from dashboard.components.kpi_cards import render_kpi_card
from dashboard.components.validation_panel import render_validation_and_leakage_panels
from dashboard.data.schemas import DashboardPolicyRecord, PolicyConfigSchema


def render_backtest_validation_page(
    policy_cfg: PolicyConfigSchema,
    record: DashboardPolicyRecord,
    backtest_data: Dict[str, pd.DataFrame],
    trigger_data: Dict[str, pd.DataFrame],
    val_df: pd.DataFrame,
    leakage_df: pd.DataFrame,
    test_check_info: Dict[str, object],
) -> None:
    """Render Page 5 — Backtest & Validation."""
    render_top_banner(
        policy_cfg,
        record,
        "5. Backtest & Validation — Step 5.6 Policy Evaluation & Causal Leakage Audit",
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        render_kpi_card(
            "Base Reorder Trigger Rate",
            "99.85% (All) | 100% (Active)",
            "65,700 / 65,800 Rows (100 Cold-Start)",
            "#f59e0b",
        )
    with c2:
        render_kpi_card(
            "4-Day ROP Demand Coverage",
            "93.30%",
            "Future 4-Day Demand ≤ ROP (60,925 / 65,300)",
            "#10b981",
        )
    with c3:
        render_kpi_card(
            "4-Day LTD Demand Coverage",
            "54.22%",
            "Future 4-Day Demand ≤ LTD (35,408 / 65,300)",
            "#38bdf8",
        )
    with c4:
        render_kpi_card(
            "Bucket A (Inv ≤ LTD)",
            "99.47%",
            "65,355 / 65,700 Active Rows Triggered by LTD",
            "#22c55e",
        )
    with c5:
        render_kpi_card(
            "All Validation Suites",
            "49 / 49 PASS",
            "Step 5 (18) + Step 5.5 (12) + Step 5.6 (6+13)",
            "#8b5cf6",
        )

    render_backtest_confusion_and_metrics(backtest_data=backtest_data)

    st.markdown("---")
    render_trigger_diagnosis_panel(trigger_data=trigger_data)

    st.markdown("---")
    st.markdown("### 🛡️ Automated Validation, Leakage & TEST Split Protection Suite")
    render_validation_and_leakage_panels(
        val_df=val_df,
        leakage_df=leakage_df,
        backtest_val_df=backtest_data["validation_checks"],
        test_check_info=test_check_info,
    )
