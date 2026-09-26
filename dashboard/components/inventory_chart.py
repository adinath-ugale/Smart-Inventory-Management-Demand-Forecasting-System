"""Inventory trajectory and threshold comparison charts for the SIM&DFS Step 6 Dashboard."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dashboard.data.schemas import DashboardPolicyRecord


def render_inventory_vs_thresholds_bar(record: DashboardPolicyRecord) -> None:
    """
    Render a side-by-side horizontal bar comparison of `Inventory Level` vs.
    `Reorder Point (ROP)`, `Lead-Time Demand (LTD)`, and `Safety Stock (SS)` on `decision_date`.
    """
    if record.policy_status == "INSUFFICIENT_DATA" or record.reorder_point is None:
        st.info(
            "Inventory threshold comparison chart is unavailable on Cold-Start Day 1 (`INSUFFICIENT_DATA`) "
            "because `Safety_Stock` and `Reorder_Point` are preserved as `NaN`."
        )
        return

    labels = [
        "Safety Stock (SS)",
        "Lead-Time Demand (LTD)",
        "Reorder Point (ROP = LTD + SS)",
        "Current Inventory Level",
    ]
    values = [
        float(record.safety_stock or 0.0),
        float(record.lead_time_demand or 0.0),
        float(record.reorder_point or 0.0),
        float(record.inventory_level),
    ]
    inv_color = "#dc2626" if record.policy_status == "REORDER" else "#16a34a"
    colors = ["#f59e0b", "#3b82f6", "#ec4899", inv_color]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{v:,.2f} units" for v in values],
            textposition="auto",
        )
    )
    fig.add_vline(
        x=float(record.reorder_point),
        line_dash="dash",
        line_color="#ec4899",
        annotation_text=f"ROP = {record.reorder_point:,.2f}",
        annotation_position="top right",
    )
    fig.update_layout(
        title=f"Decision Snapshot ({record.date}): On-Hand Inventory Level vs. Frozen ROP Components",
        xaxis_title="Units",
        template="plotly_dark",
        height=320,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_inventory_trajectory_chart(
    traj_df: pd.DataFrame,
    record: DashboardPolicyRecord,
) -> None:
    """
    Render the historical trajectory (`Date <= decision_date`) of `Inventory Level`,
    `Reorder_Point`, and `Safety_Stock` for the selected Store × Product.
    """
    if traj_df.empty:
        return

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=traj_df["Date_Str"],
            y=traj_df["Inventory Level"],
            mode="lines+markers",
            name="Inventory Level (On-Hand)",
            line=dict(color="#38bdf8", width=2.2),
            marker=dict(size=4),
        )
    )

    if "Reorder_Point" in traj_df.columns:
        fig.add_trace(
            go.Scatter(
                x=traj_df["Date_Str"],
                y=traj_df["Reorder_Point"],
                mode="lines",
                name="Frozen Reorder Point (ROP)",
                line=dict(color="#ec4899", width=2.2, dash="dash"),
            )
        )

    if "Safety_Stock" in traj_df.columns:
        fig.add_trace(
            go.Scatter(
                x=traj_df["Date_Str"],
                y=traj_df["Safety_Stock"],
                mode="lines",
                name="Frozen Safety Stock (SS)",
                line=dict(color="#f59e0b", width=1.8, dash="dot"),
            )
        )

    # Highlight REORDER trigger dates on the trajectory
    reorder_pts = traj_df[traj_df["Inventory_Action"] == "REORDER"]
    if not reorder_pts.empty:
        fig.add_trace(
            go.Scatter(
                x=reorder_pts["Date_Str"],
                y=reorder_pts["Inventory Level"],
                mode="markers",
                name="REORDER Triggered (Inv ≤ ROP)",
                marker=dict(color="#dc2626", size=7, symbol="diamond"),
            )
        )

    fig.update_layout(
        title=f"Historical Inventory Level vs. Reorder Point Trajectory (Up to {record.date})",
        xaxis_title="Date (<= Decision Date)",
        yaxis_title="Units",
        template="plotly_dark",
        height=390,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)
