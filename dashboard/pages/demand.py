"""Page 2 — Demand Analysis: Strictly prior historical demand visualization and Step 13 forecast diagnostics note."""

from typing import Optional
import pandas as pd
import streamlit as st

from dashboard.components.demand_chart import (
    render_demand_distribution_chart,
    render_demand_history_chart,
)
from dashboard.components.fallback_badge import render_fallback_badge
from dashboard.components.header import render_top_banner
from dashboard.components.kpi_cards import format_num, render_kpi_card
from dashboard.config.policy_config import ALLOWED_HISTORY_WINDOWS
from dashboard.data.schemas import DashboardPolicyRecord, PolicyConfigSchema
from dashboard.policy.policy_reader import get_historical_demand_series


def render_demand_analysis_page(
    df: pd.DataFrame,
    policy_cfg: PolicyConfigSchema,
    record: DashboardPolicyRecord,
) -> None:
    """Render Page 2 — Demand Analysis."""
    render_top_banner(policy_cfg, record, "2. Demand Analysis — Causal Historical Demand & Variability")

    st.info(
        "🔒 **Visualization Window Isolation Rule**: Selecting `30`, `90`, `180`, or `All` below changes **only** the displayed "
        "chart slice (`t' < Decision Date`). It **never** recalculates or overrides the frozen Step 5.7 "
        "`Demand_Baseline`, `Demand_Std`, `Safety_Stock`, or `Reorder_Point`."
    )

    window_choice = st.radio(
        "Select Historical Demand Chart Display Window (Visualization Only):",
        options=ALLOWED_HISTORY_WINDOWS,
        index=1,  # Default 90 days
        horizontal=True,
        format_func=lambda x: f"Last {x} Prior Days" if isinstance(x, int) else "All Prior Days (t' < t)",
    )
    window_days: Optional[int] = int(window_choice) if isinstance(window_choice, int) else None
    window_label = f"Last {window_days} Prior Days" if window_days is not None else "All Prior Days"

    hist_df, _ = get_historical_demand_series(
        store_id=record.store_id,
        product_id=record.product_id,
        decision_date=record.date,
        window_days=window_days,
        df=df,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        render_kpi_card(
            "Prior Causal Days (t' < t)",
            f"{record.historical_demand_count} Days",
            f"Displayed in chart: {len(hist_df)} days",
            "#38bdf8",
        )
    with c2:
        render_kpi_card(
            "Frozen Demand Baseline",
            format_num(record.demand_baseline, 2, " u/d"),
            f"Source: {record.demand_baseline_source}",
            "#10b981",
        )
    with c3:
        render_kpi_card(
            "Frozen Demand Std (σ)",
            format_num(record.demand_std, 2, " u/d"),
            f"Source: {record.demand_std_source}",
            "#f59e0b",
        )
    with c4:
        cv = (
            round(record.demand_std / record.demand_baseline, 3)
            if (record.demand_baseline and record.demand_std and record.demand_baseline > 0)
            else None
        )
        render_kpi_card(
            "Coefficient of Variation (CV)",
            format_num(cv, 3),
            "σ / Demand_Baseline",
            "#a855f7",
        )
    with c5:
        render_kpi_card(
            "Same-Day Units Sold (t)",
            f"{record.units_sold:,.2f} u",
            "Excluded from Day-t Baseline (No Leakage)",
            "#64748b",
        )

    render_fallback_badge(
        baseline_source=record.demand_baseline_source,
        std_source=record.demand_std_source,
        historical_count=record.historical_demand_count,
    )

    render_demand_history_chart(hist_df=hist_df, record=record, window_label=window_label)

    col1, col2 = st.columns([1.15, 1.0])
    with col1:
        render_demand_distribution_chart(hist_df=hist_df, record=record)

    with col2:
        st.markdown("#### 🧪 Why Dynamic ML Forecasts Are Excluded from Replenishment Policy")
        st.markdown(
            f"""
            - **Step 11–13 Diagnostic Finding**: Across `33` locked engineered features and `46` entity-aware features,
              Ridge, Random Forest, and HistGradientBoosting achieved near-zero dynamic skill over entity historical mean (`R² ≈ -0.0017`, `MAE ≈ 68.79`),
              because daily demand variations in this retail dataset behave as high-variance stationary stochastic noise around Store × Product baselines.
            - **Forbidden Column Protection**: The dataset column `Demand Forecast` (`{format_num(record.demand_forecast_reference, 2)}` on `{record.date}`)
              is **strictly blocked** (`FORBIDDEN_POLICY_INPUT_COLUMNS`) from entering `Demand_Baseline`, `Safety_Stock`, or `Reorder_Point`.
            - **Causal Rolling Estimator**: SIM&DFS uses strictly prior (`t' < t`) 28-day rolling mean (`Demand_Baseline`) and 28-day rolling standard deviation (`Demand_Std`)
              backed by the 5-tier causal fallback hierarchy.
            """
        )
