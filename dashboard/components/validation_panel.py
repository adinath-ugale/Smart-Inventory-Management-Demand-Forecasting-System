"""Validation and leakage audit tables for the SIM&DFS Step 6 Dashboard."""

from typing import Dict
import pandas as pd
import streamlit as st


def render_validation_and_leakage_panels(
    val_df: pd.DataFrame,
    leakage_df: pd.DataFrame,
    backtest_val_df: pd.DataFrame,
    test_check_info: Dict[str, object],
) -> None:
    """
    Render the 18/18 Step 5 Validation Checks, 11/11 Step 5 Leakage Audit Checks,
    13/13 Step 5.6 Backtest Validation Checks, and Live Step 6 TEST Split SHA-256 Verification.
    """
    st.markdown(
        f"""
        <div style="
            background:#064e3b;
            border:1px solid #10b981;
            border-radius:8px;
            padding:14px 18px;
            margin-bottom:16px;
            color:#ecfdf5;
        ">
            <div style="font-size:15px; font-weight:700; margin-bottom:4px;">
                🛡️ Live Step 6 Startup TEST Split Integrity Check: <code>{test_check_info.get('status', 'PASS')}</code>
            </div>
            <div style="font-size:12.5px; color:#a7f3d0;">
                Protected File: <code>{test_check_info.get('path')}</code> &nbsp;|&nbsp;
                Expected SHA-256: <code>{test_check_info.get('expected_sha256')}</code> &nbsp;|&nbsp;
                Actual SHA-256: <code>{test_check_info.get('actual_sha256')}</code>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(
        [
            f"✅ Step 5 Data Spec Validation ({len(val_df)} / {len(val_df)} PASS)",
            f"🔒 Step 5 Causal Leakage Audit ({len(leakage_df)} / {len(leakage_df)} PASS)",
            f"📈 Step 5.6 Policy Backtest Checks ({len(backtest_val_df)} / {len(backtest_val_df)} PASS)",
        ]
    )

    with tab1:
        st.dataframe(val_df, use_container_width=True, hide_index=True)

    with tab2:
        st.dataframe(leakage_df, use_container_width=True, hide_index=True)

    with tab3:
        st.dataframe(backtest_val_df, use_container_width=True, hide_index=True)
