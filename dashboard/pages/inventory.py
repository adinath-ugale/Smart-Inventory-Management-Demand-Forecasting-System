"""Page 3 — Inventory Decision: Full Step 5.7 decision card, formula verification, and trajectory."""

import pandas as pd
import streamlit as st

from dashboard.components.fallback_badge import render_fallback_badge
from dashboard.components.header import render_top_banner
from dashboard.components.inventory_chart import (
    render_inventory_trajectory_chart,
    render_inventory_vs_thresholds_bar,
)
from dashboard.components.kpi_cards import format_num, render_kpi_card
from dashboard.components.policy_status import render_policy_status_banner
from dashboard.components.safety_stock import render_safety_stock_breakdown
from dashboard.data.schemas import DashboardPolicyRecord, PolicyConfigSchema
from dashboard.policy.policy_reader import get_entity_trajectory_up_to_date


def render_inventory_decision_page(
    df: pd.DataFrame,
    policy_cfg: PolicyConfigSchema,
    record: DashboardPolicyRecord,
) -> None:
    """Render Page 3 — Inventory Decision."""
    render_top_banner(policy_cfg, record, "3. Inventory Decision — Frozen Step 5.7 Replenishment Engine")

    render_policy_status_banner(record)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        render_kpi_card(
            "On-Hand Inventory Level",
            f"{record.inventory_level:,.2f}",
            "Effective Position (On-Hand Only)",
            "#38bdf8",
        )
    with c2:
        render_kpi_card(
            "Lead-Time Demand (4d)",
            format_num(record.lead_time_demand, 2),
            f"Baseline ({format_num(record.demand_baseline, 2)}) × 4",
            "#3b82f6",
        )
    with c3:
        render_kpi_card(
            "Safety Stock (95% SL)",
            format_num(record.safety_stock, 2),
            f"1.6449 × {format_num(record.demand_std, 2)} × √4",
            "#f59e0b",
        )
    with c4:
        render_kpi_card(
            "Reorder Point (ROP)",
            format_num(record.reorder_point, 2),
            "LTD + Safety Stock",
            "#ec4899",
        )
    with c5:
        render_kpi_card(
            "Inventory Gap to ROP",
            format_num(record.inventory_gap_to_rop, 2),
            "Inventory Level - ROP",
            "#dc2626" if record.policy_status == "REORDER" else "#16a34a",
        )
    with c6:
        render_kpi_card(
            "Est. Days of Supply",
            format_num(record.days_of_supply_estimate, 2, " d"),
            f"Stockout Prob: {format_num(record.estimated_stockout_probability, 3)}",
            "#8b5cf6",
        )

    render_safety_stock_breakdown(record)
    render_fallback_badge(
        baseline_source=record.demand_baseline_source,
        std_source=record.demand_std_source,
        historical_count=record.historical_demand_count,
    )

    col1, col2 = st.columns([1.0, 1.15])
    with col1:
        render_inventory_vs_thresholds_bar(record)

    with col2:
        traj_df = get_entity_trajectory_up_to_date(
            store_id=record.store_id,
            product_id=record.product_id,
            decision_date=record.date,
            window_days=90,
            df=df,
        )
        render_inventory_trajectory_chart(traj_df=traj_df, record=record)

    st.markdown("#### 🛡️ Inventory Semantics & Reference-Only Field Audit")
    st.markdown(
        f"""
        - **On-Hand vs. On-Order Semantics (`inventory_level_time_semantics = "unknown"`)**:
          `Inventory Level` (`{record.inventory_level:,.2f} units`) is evaluated directly against `Reorder_Point` (`{format_num(record.reorder_point, 2)} units`).
        - **Why `Units Ordered` (`{record.units_ordered_reference:,.2f} units` on `{record.date}`) Is NOT Added**:
          Dataset diagnostics in Step 5 proved that `Units Ordered` represents same-day replenishment activity/records without verified delivery arrival timestamps or open pipeline tracking. Adding `Units Ordered` to `Inventory Level` is strictly forbidden (`assert_units_ordered_not_used_as_on_order`).
        - **Why Order Quantity (`EOQ / Order-Up-To`) Is Unsupported (`order_quantity_supported = False`)**:
          Neither fixed ordering cost ($K$), unit holding cost ($h$), penalty cost ($p$), nor supplier minimum order quantity (MOQ) is observed in the dataset.
        """
    )
