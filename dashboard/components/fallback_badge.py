"""Fallback hierarchy badge component for the SIM&DFS Step 6 Dashboard."""

import streamlit as st


FALLBACK_DESCRIPTIONS = {
    "STORE_PRODUCT": (
        "Tier 1 — Store × Product Entity History",
        "#10b981",
        "Computed directly from this Store × Product's strictly prior history (≥ 7 prior days for baseline mean, ≥ 30 prior days for volatility std).",
    ),
    "PRODUCT": (
        "Tier 2 — Product Cross-Store Fallback",
        "#3b82f6",
        "Store × Product prior history is below threshold (< 7 days for mean or < 30 days for std); uses cross-store Product prior historical distribution (`t' < t`).",
    ),
    "STORE": (
        "Tier 3 — Store Cross-Product Fallback",
        "#8b5cf6",
        "Uses Store-level prior historical demand distribution (`t' < t`).",
    ),
    "GLOBAL_TRAIN": (
        "Tier 4 — Global Training Prior Fallback",
        "#f59e0b",
        "Uses global prior historical distribution (`t' < t`) when entity-specific sample depth is not yet reached.",
    ),
    "INSUFFICIENT_DATA": (
        "Tier 5 — Cold-Start Insufficient Data (`NaN` Preserved)",
        "#ef4444",
        "Day 1 cold-start (`2022-01-01`, 0 prior observations). Parameters remain `NaN` to strictly prevent same-day or future data leakage.",
    ),
}


def render_fallback_badge(
    baseline_source: str,
    std_source: str,
    historical_count: int,
) -> None:
    """Render an informative card explaining the active fallback tier for Demand_Baseline and Demand_Std."""
    b_title, b_color, b_desc = FALLBACK_DESCRIPTIONS.get(
        baseline_source, ("Unknown Source", "#64748b", "")
    )
    s_title, s_color, s_desc = FALLBACK_DESCRIPTIONS.get(
        std_source, ("Unknown Source", "#64748b", "")
    )

    st.markdown(
        f"""
        <div style="
            background:#0f172a;
            border:1px solid #1e293b;
            border-radius:8px;
            padding:14px 18px;
            margin-bottom:14px;
        ">
            <div style="font-size:13px; font-weight:700; color:#e2e8f0; margin-bottom:8px;">
                🧭 Causal Estimation &amp; Fallback Hierarchy Status (Prior Days Available: <code>{historical_count}</code>)
            </div>
            <div style="display:grid; grid-template-columns: 1fr 1fr; gap:12px;">
                <div style="background:#1e293b; padding:10px 14px; border-radius:6px; border-left:4px solid {b_color};">
                    <div style="font-size:11.5px; color:#94a3b8;">Demand Baseline Source (Min 7 prior days)</div>
                    <div style="font-size:14px; font-weight:700; color:#f8fafc; margin:2px 0;"><code>{baseline_source}</code> — {b_title}</div>
                    <div style="font-size:11.5px; color:#cbd5e1;">{b_desc}</div>
                </div>
                <div style="background:#1e293b; padding:10px 14px; border-radius:6px; border-left:4px solid {s_color};">
                    <div style="font-size:11.5px; color:#94a3b8;">Demand Volatility (Std) Source (Min 30 prior days)</div>
                    <div style="font-size:14px; font-weight:700; color:#f8fafc; margin:2px 0;"><code>{std_source}</code> — {s_title}</div>
                    <div style="font-size:11.5px; color:#cbd5e1;">{s_desc}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
