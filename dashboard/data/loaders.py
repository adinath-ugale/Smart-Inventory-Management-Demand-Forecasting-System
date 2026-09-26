"""
Centralized, Cached Read-Only Data Loaders for the SIM&DFS Step 6 Dashboard (Section 18).
Ensures large CSVs and JSON configurations are loaded once, validated at startup,
and restricted strictly to the TRAIN + VALIDATION development population (Date <= 2023-10-20).
"""

from __future__ import annotations

import functools
import json
from typing import Dict

import pandas as pd
import streamlit as st

from dashboard.config.policy_config import DATA_FILES, DEV_END_DATE, TEST_START_DATE
from dashboard.data.schemas import PolicyConfigSchema
from dashboard.data.validators import (
    TestIntegrityError,
    validate_development_dataset_schema,
    verify_test_split_checksum,
)


@st.cache_data(show_spinner=False)
@functools.lru_cache(maxsize=1)
def load_policy_config() -> PolicyConfigSchema:
    """Load and validate the frozen Step 5.7 policy configuration JSON."""
    verify_test_split_checksum()
    cfg_path = DATA_FILES["policy_config"]
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Frozen policy configuration file not found: {cfg_path}"
        )
    with open(cfg_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return PolicyConfigSchema.from_dict(payload)


@st.cache_data(show_spinner=False)
@functools.lru_cache(maxsize=1)
def load_development_dataset() -> pd.DataFrame:
    """
    Load and validate the 65,800-row TRAIN + VALIDATION inventory optimization dataset.
    Strictly blocks any TEST rows (Date >= 2023-10-21).
    """
    verify_test_split_checksum()
    dev_path = DATA_FILES["development_dataset"]
    if not dev_path.exists():
        raise FileNotFoundError(f"Development dataset not found: {dev_path}")
    df = pd.read_csv(dev_path)
    df["Date_dt"] = pd.to_datetime(df["Date"])
    df["Date_Str"] = df["Date_dt"].dt.strftime("%Y-%m-%d")
    if "Split_Label" not in df.columns and "Split_Partition" in df.columns:
        df["Split_Label"] = df["Split_Partition"]
    if (df["Date_dt"] >= pd.Timestamp(TEST_START_DATE)).any():
        raise TestIntegrityError("TEST rows detected in development_dataset!")
    validate_development_dataset_schema(df)
    return df.sort_values(["Date_dt", "Store ID", "Product ID"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
@functools.lru_cache(maxsize=1)
def load_inventory_dataset() -> pd.DataFrame:
    """
    Load the primary inventory_optimization_dataset.csv and immediately filter
    out the locked TEST partition so only TRAIN + VALIDATION (Date <= 2023-10-20,
    65,800 rows) is ever accessible to the dashboard UI.
    """
    verify_test_split_checksum()
    inv_path = DATA_FILES["inventory_dataset"]
    if not inv_path.exists():
        raise FileNotFoundError(f"Primary inventory dataset not found: {inv_path}")
    df = pd.read_csv(inv_path)
    df["Date_dt"] = pd.to_datetime(df["Date"])
    df["Date_Str"] = df["Date_dt"].dt.strftime("%Y-%m-%d")
    if "Split_Label" not in df.columns and "Split_Partition" in df.columns:
        df["Split_Label"] = df["Split_Partition"]
    df_safe = df[
        (df["Split_Partition"].isin(["TRAIN", "VALIDATION"]))
        & (df["Date_dt"] <= pd.Timestamp(DEV_END_DATE))
    ].copy()
    validate_development_dataset_schema(df_safe)
    return df_safe.sort_values(["Date_dt", "Store ID", "Product ID"]).reset_index(
        drop=True
    )


@st.cache_data(show_spinner=False)
@functools.lru_cache(maxsize=1)
def load_scenario_results() -> Dict[str, pd.DataFrame]:
    """Load the 9-scenario sensitivity summary and 9-scenario backtest comparison."""
    verify_test_split_checksum()
    scen_sum_path = DATA_FILES["scenario_summary"]
    scen_bt_path = DATA_FILES["scenario_backtest"]
    if not scen_sum_path.exists() or not scen_bt_path.exists():
        raise FileNotFoundError(
            "Scenario summary or backtest comparison CSV not found."
        )
    return {
        "scenario_sensitivity_summary": pd.read_csv(scen_sum_path),
        "scenario_backtest_comparison": pd.read_csv(scen_bt_path),
    }


@st.cache_data(show_spinner=False)
@functools.lru_cache(maxsize=1)
def load_backtest_results() -> Dict[str, pd.DataFrame]:
    """Load all validated Step 5.5 and Step 5.6 historical backtest evidence tables."""
    verify_test_split_checksum()
    return {
        "scenario_backtest": pd.read_csv(DATA_FILES["scenario_backtest"]),
        "horizon_backtest": pd.read_csv(DATA_FILES["horizon_backtest"]),
        "store_product_backtest": pd.read_csv(DATA_FILES["store_product_backtest"]),
        "base_policy_outcomes": pd.read_csv(DATA_FILES["base_policy_outcomes"]),
        "base_decision_outcome_matrix": pd.read_csv(
            DATA_FILES["base_decision_outcome_matrix"]
        ),
        "zero_inventory_diagnostic": pd.read_csv(
            DATA_FILES["zero_inventory_diagnostic"]
        ),
        "trigger_four_way": pd.read_csv(DATA_FILES["trigger_four_way"]),
        "validation_checks": pd.read_csv(DATA_FILES["validation_checks"]),
    }


@st.cache_data(show_spinner=False)
@functools.lru_cache(maxsize=1)
def load_trigger_diagnosis_results() -> Dict[str, pd.DataFrame]:
    """Load Step 5.5 trigger-rate diagnosis four-way decomposition and validation checks."""
    verify_test_split_checksum()
    return {
        "four_way": pd.read_csv(DATA_FILES["trigger_four_way"]),
        "validation_checks": pd.read_csv(DATA_FILES["step5_5_validation_checks"]),
    }


@st.cache_data(show_spinner=False)
@functools.lru_cache(maxsize=1)
def load_validation_results() -> pd.DataFrame:
    """Load the 18/18 Step 5 validation checks table."""
    verify_test_split_checksum()
    return pd.read_csv(DATA_FILES["step5_validation_checks"])


@st.cache_data(show_spinner=False)
@functools.lru_cache(maxsize=1)
def load_leakage_audit() -> pd.DataFrame:
    """Load the 13-point Step 5.6 leakage audit CSV."""
    verify_test_split_checksum()
    audit_path = DATA_FILES["leakage_audit"]
    if not audit_path.exists():
        raise FileNotFoundError(f"Leakage audit file not found: {audit_path}")
    return pd.read_csv(audit_path)
