"""Demand history and distribution charts for the SIM&DFS Step 6 Dashboard."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dashboard.data.schemas import DashboardPolicyRecord


def render_demand_history_chart(
    hist_df: pd.DataFrame,
    record: DashboardPolicyRecord,
    window_label: str = "90 Days",
) -> None:
    """
    Render the historical daily demand chart (`Date < decision_date`) overlaid with
    the frozen `Demand_Baseline` and `±1 Demand_Std` volatility band.
    """
    if hist_df.empty:
        st.warning(
            f"No strictly prior historical demand (`Date < {record.date}`) exists for "
            f"Store `{record.store_id}` × Product `{record.product_id}` (Cold-Start Day 1)."
        )
        return

    fig = go.Figure()

    # Historical daily Units Sold (t' < t)
    fig.add_trace(
        go.Scatter(
            x=hist_df["Date_Str"],
            y=hist_df["Units Sold"],
            mode="lines+markers",
            name="Historical Units Sold (t' < t)",
            line=dict(color="#38bdf8", width=1.8),
            marker=dict(size=4, color="#38bdf8"),
        )
    )

    # Historical 7-day rolling mean from dataset (if available)
    if "Demand_Rolling_7D_Mean" in hist_df.columns:
        fig.add_trace(
            go.Scatter(
                x=hist_df["Date_Str"],
                y=hist_df["Demand_Rolling_7D_Mean"],
                mode="lines",
                name="7-Day Prior Rolling Mean",
                line=dict(color="#a855f7", width=2.0, dash="dot"),
            )
        )

    # Frozen Demand_Baseline & ±1 Std band as of decision_date
    if record.demand_baseline is not None:
        fig.add_hline(
            y=record.demand_baseline,
            line_dash="dash",
            line_color="#10b981",
            annotation_text=f"Frozen Demand Baseline = {record.demand_baseline:,.2f} ({record.demand_baseline_source})",
            annotation_position="top left",
        )
        if record.demand_std is not None:
            upper = record.demand_baseline + record.demand_std
            lower = max(0.0, record.demand_baseline - record.demand_std)
            fig.add_hrect(
                y0=lower,
                y1=upper,
                fillcolor="#10b981",
                opacity=0.10,
                line_width=0,
                annotation_text=f"±1σ Band ([{lower:,.1f}, {upper:,.1f}])",
                annotation_position="bottom right",
            )

    fig.update_layout(
        title=(
            f"Causal Historical Daily Demand (t' < {record.date}) — Window: {window_label} "
            f"(Visualization Only; Policy Frozen)"
        ),
        xaxis_title="Historical Date (t' < Decision Date)",
        yaxis_title="Units Sold / Day",
        template="plotly_dark",
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_demand_distribution_chart(
    hist_df: pd.DataFrame,
    record: DashboardPolicyRecord,
) -> None:
    """Render the empirical histogram of strictly prior daily demand (`t' < t`)."""
    if hist_df.empty:
        return

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=hist_df["Units Sold"],
            nbinsx=25,
            name="Prior Daily Units Sold",
            marker_color="#6366f1",
            opacity=0.85,
        )
    )
    if record.demand_baseline is not None:
        fig.add_vline(
            x=record.demand_baseline,
            line_dash="dash",
            line_color="#10b981",
            annotation_text=f"Mean: {record.demand_baseline:,.1f}",
        )
    fig.update_layout(
        title=f"Prior Daily Demand Distribution (N = {len(hist_df)} displayed days, t' < {record.date})",
        xaxis_title="Daily Units Sold",
        yaxis_title="Frequency (Days)",
        template="plotly_dark",
        height=340,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)
