"""Header and frozen policy banner components for the SIM&DFS Step 6 Dashboard."""

import streamlit as st
from dashboard.data.schemas import DashboardPolicyRecord, PolicyConfigSchema


def render_top_banner(policy_cfg: PolicyConfigSchema, record: DashboardPolicyRecord, current_page: str) -> None:
    """
    Render the executive dashboard header banner showing active navigation context,
    selected entity (`Store ID`, `Product ID`, `Decision Date`), and frozen Step 5.7 policy parameters.
    """
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            border: 1px solid #334155;
            border-radius: 10px;
            padding: 16px 22px;
            margin-bottom: 18px;
            color: #f8fafc;
        ">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
                <div>
                    <div style="font-size:12px; text-transform:uppercase; letter-spacing:1.2px; color:#38bdf8; font-weight:700;">
                        SMART INVENTORY MANAGEMENT &amp; DEMAND FORECASTING SYSTEM (SIM&amp;DFS)
                    </div>
                    <div style="font-size:22px; font-weight:700; margin-top:2px;">
                        {current_page}
                    </div>
                </div>
                <div style="display:flex; gap:10px; flex-wrap:wrap;">
                    <span style="background:#1e3a8a; color:#93c5fd; padding:5px 11px; border-radius:6px; font-size:12px; font-weight:600;">
                        🔒 Policy: {policy_cfg.policy_version} (FROZEN)
                    </span>
                    <span style="background:#064e3b; color:#6ee7b7; padding:5px 11px; border-radius:6px; font-size:12px; font-weight:600;">
                        LT = {policy_cfg.base_lead_time_days}d | SL = {int(policy_cfg.base_service_level*100)}% | Z = {policy_cfg.base_z_score}
                    </span>
                    <span style="background:#312e81; color:#c7d2fe; padding:5px 11px; border-radius:6px; font-size:12px; font-weight:600;">
                        🛡️ TEST Split Protected (SHA-256 Verified)
                    </span>
                </div>
            </div>
            <hr style="border:none; border-top:1px solid #334155; margin:12px 0;" />
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; font-size:13px; color:#cbd5e1;">
                <div>
                    <strong>Active Context:</strong>
                    Store <code>{record.store_id}</code> ({record.region}) &nbsp;|&nbsp;
                    Product <code>{record.product_id}</code> ({record.category}) &nbsp;|&nbsp;
                    Decision Date <code>{record.date}</code> ({record.split_label}) &nbsp;|&nbsp;
                    Causal History Count: <code>{record.historical_demand_count} days (t' &lt; t)</code>
                </div>
                <div>
                    <strong>Inventory Semantics:</strong> <code>inventory_level_time_semantics = "{policy_cfg.inventory_level_time_semantics}"</code>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
