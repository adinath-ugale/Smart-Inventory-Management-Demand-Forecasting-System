"""Page 1 — Overview: System-wide KPIs, decision date snapshot, and entity portfolio matrix."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.header import render_top_banner
from dashboard.components.kpi_cards import format_num, render_kpi_card
from dashboard.components.policy_status import render_policy_status_banner
from dashboard.data.schemas import DashboardPolicyRecord, PolicyConfigSchema
from dashboard.policy.policy_reader import get_system_overview_kpis


def render_overview_page(
    df: pd.DataFrame,
    policy_cfg: PolicyConfigSchema,
    record: DashboardPolicyRecord,
) -> None:
    """Render Page 1 — Overview."""
    render_top_banner(policy_cfg, record, "1. Overview — Executive System & Portfolio Summary")

    kpis = get_system_overview_kpis(decision_date=record.date, df=df)
    snap = kpis["date_snapshot"]

    st.markdown("### 📌 Selected Store × Product Decision Snapshot")
    render_policy_status_banner(record)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        render_kpi_card(
            "Current Inventory",
            f"{record.inventory_level:,.2f}",
            f"Store {record.store_id} × {record.product_id}",
            "#38bdf8",
        )
    with c2:
        render_kpi_card(
            "Demand Baseline",
            format_num(record.demand_baseline, 2, " /d"),
            f"Source: {record.demand_baseline_source}",
            "#10b981",
        )
    with c3:
        render_kpi_card(
            "Lead-Time Demand (4d)",
            format_num(record.lead_time_demand, 2),
            "LTD = Baseline × 4",
            "#3b82f6",
        )
    with c4:
        render_kpi_card(
            "Safety Stock (SS)",
            format_num(record.safety_stock, 2),
            "SS = 1.6449 × Std × √4",
            "#f59e0b",
        )
    with c5:
        render_kpi_card(
            "Reorder Point (ROP)",
            format_num(record.reorder_point, 2),
            "ROP = LTD + SS",
            "#ec4899",
        )
    with c6:
        gap_color = "#dc2626" if record.policy_status == "REORDER" else "#16a34a"
        render_kpi_card(
            "Policy Action",
            record.policy_status,
            f"Gap to ROP: {format_num(record.inventory_gap_to_rop, 2)}",
            gap_color,
        )

    st.markdown("---")
    st.markdown(
        f"### 🌐 Portfolio Snapshot on `{record.date}` (`100` Store × Product Entities) vs. Full Development Benchmark (`65,800` Rows)"
    )

    p1, p2, p3, p4 = st.columns(4)
    with p1:
        render_kpi_card(
            f"Date Snapshot ({record.date}) REORDER",
            f"{snap['reorder_count']} / {snap['total_skus']} ({snap['reorder_pct']}%)",
            f"MONITOR: {snap['monitor_count']} | INSUFFICIENT: {snap['insufficient_count']}",
            "#ef4444",
        )
    with p2:
        render_kpi_card(
            "Full Benchmark REORDER Rate",
            f"{kpis['reorder_pct']}% ({kpis['reorder_count']:,})",
            f"MONITOR: {kpis['monitor_pct']}% ({kpis['monitor_count']:,}) | Cold-Start: {kpis['insufficient_count']}",
            "#f97316",
        )
    with p3:
        render_kpi_card(
            "Benchmark Avg Inventory vs ROP",
            f"{kpis['avg_inventory_level']:,.1f} vs {kpis['avg_reorder_point']:,.1f}",
            f"Avg LTD: {kpis['avg_lead_time_demand']:,.1f} | Avg SS: {kpis['avg_safety_stock']:,.1f}",
            "#8b5cf6",
        )
    with p4:
        render_kpi_card(
            "Development Scope (TRAIN + VAL)",
            f"{kpis['total_rows']:,} Rows",
            f"TRAIN: {kpis['train_rows']:,} | VAL: {kpis['validation_rows']:,} (TEST Excluded)",
            "#14b8a6",
        )

    col_left, col_right = st.columns([1.15, 1.0])

    with col_left:
        st.markdown(f"#### 📋 All 100 Store × Product Policy Decisions on `{record.date}`")
        day_df = df[df["Date_Str"] == record.date][
            [
                "Store ID",
                "Product ID",
                "Category",
                "Region",
                "Inventory Level",
                "Demand_Baseline",
                "Safety_Stock",
                "Reorder_Point",
                "Inventory_Action",
                "Demand_Std_Source",
            ]
        ].sort_values(["Store ID", "Product ID"])
        st.dataframe(day_df, use_container_width=True, hide_index=True, height=340)

    with col_right:
        st.markdown("#### 🧭 Causal Fallback Hierarchy Usage (`65,800` Development Rows)")
        fb = kpis["fallback_summary"]
        fb_df = pd.DataFrame(
            [
                {"Fallback Tier": k, "Row Count": v, "Share (%)": round(100.0 * v / kpis["total_rows"], 2)}
                for k, v in fb.items()
            ]
        )
        fig = go.Figure(
            go.Bar(
                x=fb_df["Fallback Tier"],
                y=fb_df["Row Count"],
                marker_color=["#10b981", "#3b82f6", "#8b5cf6", "#f59e0b", "#ef4444"],
                text=[f"{r:,} ({p}%)" for r, p in zip(fb_df["Row Count"], fb_df["Share (%)"])],
                textposition="auto",
            )
        )
        fig.update_layout(
            title="Demand Volatility (Std) Fallback Tier Distribution",
            xaxis_title="Fallback Source Tier",
            yaxis_title="Rows",
            template="plotly_dark",
            height=340,
            margin=dict(l=20, r=20, t=45, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)
