"""Pre-computed scenario comparison matrix and tradeoff charts for the SIM&DFS Step 6 Dashboard."""

from typing import List
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dashboard.data.schemas import DashboardPolicyRecord, ScenarioRecordSchema


def render_scenario_comparison_table(
    scenarios: List[ScenarioRecordSchema],
    record: DashboardPolicyRecord,
) -> pd.DataFrame:
    """
    Render the 9-row pre-approved scenario table (`LT ∈ {2,4,7} × SL ∈ {90%,95%,99%}`)
    highlighting the frozen Step 5.7 Base Policy (`LT4_SL95`).
    """
    rows = []
    for s in scenarios:
        rows.append(
            {
                "Scenario_ID": f"⭐ {s.scenario_id} (BASE)" if s.is_base_policy else s.scenario_id,
                "Raw_ID": s.scenario_id,
                "Lead_Time_Days": s.lead_time_days,
                "Service_Level": f"{int(round(s.service_level * 100))}%",
                "Z_Score": s.z_score,
                "Snapshot_LTD": s.lead_time_demand,
                "Snapshot_Safety_Stock": s.safety_stock,
                "Snapshot_ROP": s.reorder_point,
                "Snapshot_Status": s.policy_status,
                "Hist_Reorder_Rate_%": s.historical_reorder_rate,
                "Hist_Stockout_Coverage_%": s.historical_stockout_coverage,
                "Hist_Reorder_Precision_%": s.historical_reorder_precision,
                "Hist_False_Reorder_%": s.historical_false_reorder_rate,
                "Hist_Avg_Safety_Stock": s.historical_avg_safety_stock,
                "Hist_Avg_ROP": s.historical_avg_rop,
                "Is_Base": s.is_base_policy,
            }
        )

    df = pd.DataFrame(rows)
    display_cols = [
        "Scenario_ID",
        "Lead_Time_Days",
        "Service_Level",
        "Z_Score",
        "Snapshot_LTD",
        "Snapshot_Safety_Stock",
        "Snapshot_ROP",
        "Snapshot_Status",
        "Hist_Reorder_Rate_%",
        "Hist_Stockout_Coverage_%",
        "Hist_Reorder_Precision_%",
        "Hist_False_Reorder_%",
    ]

    st.dataframe(
        df[display_cols],
        use_container_width=True,
        hide_index=True,
    )
    return df


def render_scenario_tradeoff_charts(scen_df: pd.DataFrame) -> None:
    """
    Render visual tradeoff charts across the 9 pre-computed scenarios:
    1. Reorder Trigger Rate (%) vs. Stockout Coverage Rate (%)
    2. Average Reorder Point & Safety Stock across scenarios
    """
    col1, col2 = st.columns(2)

    with col1:
        fig1 = go.Figure()
        for _, r in scen_df.iterrows():
            is_base = bool(r["Is_Base"])
            fig1.add_trace(
                go.Scatter(
                    x=[r["Hist_Reorder_Rate_%"]],
                    y=[r["Hist_Stockout_Coverage_%"]],
                    mode="markers+text",
                    name=str(r["Raw_ID"]),
                    text=[str(r["Raw_ID"])],
                    textposition="top center",
                    marker=dict(
                        size=15 if is_base else 10,
                        color="#10b981" if is_base else "#38bdf8",
                        symbol="star" if is_base else "circle",
                        line=dict(width=2, color="#f8fafc"),
                    ),
                )
            )
        fig1.update_layout(
            title="Historical Tradeoff: Reorder Rate (%) vs. Stockout Coverage (%)",
            xaxis_title="Historical Reorder Trigger Rate (%)",
            yaxis_title="Stockout Capture / Coverage Rate (%)",
            template="plotly_dark",
            height=370,
            showlegend=False,
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        colors = [
            "#10b981" if b else "#6366f1" for b in scen_df["Is_Base"].tolist()
        ]
        fig2 = go.Figure()
        fig2.add_trace(
            go.Bar(
                x=scen_df["Raw_ID"],
                y=scen_df["Hist_Avg_ROP"],
                name="Historical Avg ROP",
                marker_color=colors,
                text=[f"{v:,.1f}" for v in scen_df["Hist_Avg_ROP"]],
                textposition="auto",
            )
        )
        fig2.update_layout(
            title="Historical Average Reorder Point (ROP) Across Pre-Approved Scenarios",
            xaxis_title="Scenario (LT × SL)",
            yaxis_title="Avg Reorder Point (Units)",
            template="plotly_dark",
            height=370,
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig2, use_container_width=True)
