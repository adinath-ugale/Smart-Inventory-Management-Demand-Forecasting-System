"""Policy status banner component for REORDER, MONITOR, and INSUFFICIENT_DATA states."""

import streamlit as st
from dashboard.config.policy_config import STATUS_COLORS
from dashboard.data.schemas import DashboardPolicyRecord


def render_policy_status_banner(record: DashboardPolicyRecord) -> None:
    """
    Render the visual policy status card (`REORDER`, `MONITOR`, or `INSUFFICIENT_DATA`)
    with explicit formula justification and order-quantity disclaimer.
    """
    status = record.policy_status
    color = STATUS_COLORS.get(status, "#6b7280")

    if status == "REORDER":
        headline = "🔴 REORDER TRIGGERED — Inventory Level ≤ Reorder Point"
        detail = (
            f"On-hand <strong>Inventory Level ({record.inventory_level:,.2f} units)</strong> is at or below the "
            f"frozen <strong>Reorder Point ({record.reorder_point:,.2f} units)</strong> "
            f"(Gap: <code>{record.inventory_gap_to_rop:+,.2f} units</code>). "
            "Replenishment review is triggered under the frozen Step 5.7 policy."
        )
        bg = "rgba(220, 38, 38, 0.12)"
    elif status == "MONITOR":
        headline = "🟢 MONITOR — Inventory Level > Reorder Point"
        detail = (
            f"On-hand <strong>Inventory Level ({record.inventory_level:,.2f} units)</strong> is above the "
            f"frozen <strong>Reorder Point ({record.reorder_point:,.2f} units)</strong> "
            f"(Buffer above ROP: <code>{record.inventory_gap_to_rop:+,.2f} units</code>). "
            "No replenishment trigger is required today."
        )
        bg = "rgba(22, 163, 74, 0.12)"
    else:
        headline = "🟡 INSUFFICIENT_DATA — Cold-Start Causal Boundary (Day 1)"
        detail = (
            f"Decision date <code>{record.date}</code> has <strong>{record.historical_demand_count} prior historical days</strong> "
            "(`t' < t`). Because zero prior days exist without peeking into same-day or future demand, "
            "<code>Demand_Baseline</code>, <code>Demand_Std</code>, <code>Safety_Stock</code>, and <code>Reorder_Point</code> "
            "are intentionally preserved as <code>NaN</code> (never converted to 0)."
        )
        bg = "rgba(217, 119, 6, 0.14)"

    st.markdown(
        f"""
        <div style="
            background: {bg};
            border: 1.5px solid {color};
            border-left: 6px solid {color};
            border-radius: 10px;
            padding: 16px 20px;
            margin-bottom: 16px;
        ">
            <div style="font-size:17px; font-weight:700; color:#f8fafc; margin-bottom:6px;">
                {headline}
            </div>
            <div style="font-size:13.5px; color:#e2e8f0; line-height:1.5;">
                {detail}
            </div>
            <div style="font-size:12px; color:#94a3b8; margin-top:8px;">
                <strong>Order Quantity Policy:</strong> <code>order_quantity_supported = False</code> —
                SIM&amp;DFS triggers <code>REORDER</code> vs. <code>MONITOR</code> decisions only; exact EOQ / order-up-to quantities
                are intentionally unsupported because ordering costs, holding costs, and supplier batch constraints are unobserved.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
