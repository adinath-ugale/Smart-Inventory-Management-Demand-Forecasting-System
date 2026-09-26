"""Page 4 — Scenario Comparison: Pre-approved 3x3 sensitivity matrix (Lead Time x Service Level)."""

from typing import Dict
import pandas as pd
import streamlit as st

from dashboard.components.header import render_top_banner
from dashboard.components.kpi_cards import format_num, render_kpi_card
from dashboard.components.scenario_table import (
    render_scenario_comparison_table,
    render_scenario_tradeoff_charts,
)
from dashboard.data.schemas import DashboardPolicyRecord, PolicyConfigSchema
from dashboard.policy.policy_reader import get_scenario_matrix_for_record


def render_scenario_comparison_page(
    policy_cfg: PolicyConfigSchema,
    record: DashboardPolicyRecord,
    backtest_data: Dict[str, pd.DataFrame],
) -> None:
    """Render Page 4 — Scenario Comparison."""
    render_top_banner(
        policy_cfg,
        record,
        "4. Scenario Comparison — Pre-Approved 3×3 Lead Time × Service Level Grid",
    )

    st.info(
        "🔒 **Frozen Policy Governance**: The 9 scenarios below (`Lead Time ∈ {2, 4, 7} days × Service Level ∈ {90%, 95%, 99%}`) "
        "are pre-approved sensitivity views from Step 5 and Step 5.6. They are **read-only** comparisons and do **not** modify "
        "the active frozen base policy (`LT = 4 days, SL = 95%, Z = 1.6449`)."
    )

    scenarios = get_scenario_matrix_for_record(record=record, backtest_data=backtest_data)

    # Identify base scenario
    base_scen = next((s for s in scenarios if s.is_base_policy), scenarios[0])
    min_scen = scenarios[0]   # LT2_SL90
    max_scen = scenarios[-1]  # LT7_SL99

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi_card(
            "⭐ Base Policy (LT4_SL95)",
            f"ROP: {format_num(base_scen.reorder_point, 1)}",
            f"Status: {base_scen.policy_status} | Hist Reorder: {base_scen.historical_reorder_rate}%",
            "#10b981",
        )
    with c2:
        render_kpi_card(
            "Lowest Buffer (LT2_SL90)",
            f"ROP: {format_num(min_scen.reorder_point, 1)}",
            f"Status: {min_scen.policy_status} | Hist Reorder: {min_scen.historical_reorder_rate}%",
            "#38bdf8",
        )
    with c3:
        render_kpi_card(
            "Highest Buffer (LT7_SL99)",
            f"ROP: {format_num(max_scen.reorder_point, 1)}",
            f"Status: {max_scen.policy_status} | Hist Reorder: {max_scen.historical_reorder_rate}%",
            "#f59e0b",
        )
    with c4:
        agree_count = sum(1 for s in scenarios if s.policy_status == record.policy_status)
        render_kpi_card(
            "Scenario Consensus on Date",
            f"{agree_count} / 9 Scenarios",
            f"Agree with Base Status ({record.policy_status})",
            "#a855f7",
        )

    st.markdown(
        f"#### 📊 9-Scenario Matrix for `{record.store_id} × {record.product_id}` on `{record.date}` + Historical Backtest Metrics (`65,800` Rows)"
    )
    scen_df = render_scenario_comparison_table(scenarios=scenarios, record=record)

    render_scenario_tradeoff_charts(scen_df=scen_df)
