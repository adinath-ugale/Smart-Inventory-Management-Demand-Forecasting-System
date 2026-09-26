"""KPI card rendering utilities for the SIM&DFS Step 6 Dashboard."""

from typing import Optional
import streamlit as st


def render_kpi_card(
    title: str,
    value: str,
    subtitle: str = "",
    accent_color: str = "#38bdf8",
) -> None:
    """Render a styled KPI metric card."""
    st.markdown(
        f"""
        <div style="
            background-color: #0f172a;
            border: 1px solid #1e293b;
            border-left: 4px solid {accent_color};
            border-radius: 8px;
            padding: 14px 16px;
            margin-bottom: 10px;
            min-height: 98px;
        ">
            <div style="font-size:12px; color:#94a3b8; text-transform:uppercase; font-weight:600; letter-spacing:0.6px;">
                {title}
            </div>
            <div style="font-size:23px; font-weight:700; color:#f8fafc; margin-top:4px; margin-bottom:3px;">
                {value}
            </div>
            <div style="font-size:11.5px; color:#64748b;">
                {subtitle}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_num(val: Optional[float], decimals: int = 2, suffix: str = "") -> str:
    """Safely format nullable float values for display (never converts None/NaN to 0)."""
    if val is None:
        return "N/A (Cold-Start)"
    return f"{val:,.{decimals}f}{suffix}"
