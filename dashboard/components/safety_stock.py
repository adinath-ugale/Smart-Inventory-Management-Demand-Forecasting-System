"""Safety stock and reorder point mathematical breakdown panel for the SIM&DFS Step 6 Dashboard."""

import math
import streamlit as st
from dashboard.data.schemas import DashboardPolicyRecord


def render_safety_stock_breakdown(record: DashboardPolicyRecord) -> None:
    """
    Render the step-by-step frozen mathematical decomposition of `Lead_Time_Demand`,
    `Safety_Stock`, and `Reorder_Point` for the selected `DashboardPolicyRecord`.
    """
    sqrt_l = math.sqrt(record.lead_time_days)
    if record.policy_status == "INSUFFICIENT_DATA" or record.demand_baseline is None or record.demand_std is None:
        st.markdown(
            f"""
            <div style="background:#0f172a; border:1px solid #334155; border-radius:8px; padding:16px;">
                <div style="font-size:15px; font-weight:700; color:#f8fafc; margin-bottom:8px;">
                    📐 Frozen Step 5.7 Mathematical Formula Breakdown
                </div>
                <div style="font-size:13px; color:#fde68a;">
                    Cold-Start Day 1 (<code>{record.date}</code>): Prior historical demand count is <code>0</code>.
                    All formula outputs remain <code>NaN</code> and policy status is <code>INSUFFICIENT_DATA</code>.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f"""
        <div style="background:#0f172a; border:1px solid #334155; border-radius:10px; padding:18px; margin-bottom:14px;">
            <div style="font-size:15px; font-weight:700; color:#38bdf8; margin-bottom:10px;">
                📐 Frozen Step 5.7 Mathematical Formula Verification (Read-Only)
            </div>
            <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap:12px; font-size:13px; color:#e2e8f0;">
                <div style="background:#1e293b; padding:12px; border-radius:6px;">
                    <div style="color:#94a3b8; font-size:11.5px;">1. Lead-Time Demand (LTD)</div>
                    <div style="font-weight:700; font-size:14px; margin:4px 0;">
                        <code>LTD = Demand_Baseline × L</code>
                    </div>
                    <div>
                        <code>{record.demand_baseline:,.4f} × {record.lead_time_days} = <strong>{record.lead_time_demand:,.4f}</strong> units</code>
                    </div>
                </div>
                <div style="background:#1e293b; padding:12px; border-radius:6px;">
                    <div style="color:#94a3b8; font-size:11.5px;">2. Safety Stock (SS — Square-Root Lead Time Rule)</div>
                    <div style="font-weight:700; font-size:14px; margin:4px 0;">
                        <code>SS = Z × Demand_Std × √L</code>
                    </div>
                    <div>
                        <code>{record.z_score} × {record.demand_std:,.4f} × √{record.lead_time_days} ({sqrt_l:.2f}) = <strong>{record.safety_stock:,.4f}</strong> units</code>
                    </div>
                </div>
                <div style="background:#1e293b; padding:12px; border-radius:6px;">
                    <div style="color:#94a3b8; font-size:11.5px;">3. Reorder Point (ROP)</div>
                    <div style="font-weight:700; font-size:14px; margin:4px 0;">
                        <code>ROP = LTD + SS</code>
                    </div>
                    <div>
                        <code>{record.lead_time_demand:,.4f} + {record.safety_stock:,.4f} = <strong>{record.reorder_point:,.4f}</strong> units</code>
                    </div>
                </div>
                <div style="background:#1e293b; padding:12px; border-radius:6px;">
                    <div style="color:#94a3b8; font-size:11.5px;">4. Decision Trigger Condition</div>
                    <div style="font-weight:700; font-size:14px; margin:4px 0;">
                        <code>Inventory_Level ≤ ROP ⇒ REORDER</code>
                    </div>
                    <div>
                        <code>{record.inventory_level:,.2f} vs. {record.reorder_point:,.2f} ⇒ <strong>{record.policy_status}</strong></code>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
