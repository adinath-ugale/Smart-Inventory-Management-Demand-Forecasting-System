"""
Read-only frozen inventory policy readers for the SIM&DFS Step 6 Dashboard.

Consumes the frozen Step 5.7 policy configuration and Step 5 / 5.5 / 5.6
validated dataset outputs without recalculating, tuning, or mutating any
inventory parameters.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd

from dashboard.config.policy_config import (
    ALLOWED_FALLBACK_SOURCES,
    ALLOWED_POLICY_STATUS,
    FROZEN_LEAD_TIME_DAYS,
    FROZEN_SERVICE_LEVEL,
    FROZEN_Z_SCORE,
    SCENARIO_LEAD_TIMES,
    SCENARIO_SERVICE_LEVELS,
    SCENARIO_Z_SCORES,
)
from dashboard.data.loaders import (
    load_backtest_results,
    load_development_dataset,
    load_policy_config,
)
from dashboard.data.schemas import (
    DashboardPolicyRecord,
    PolicyConfigSchema,
    ScenarioRecordSchema,
)
from dashboard.data.validators import (
    assert_no_date_leakage,
    assert_no_forbidden_policy_inputs,
    assert_units_ordered_not_used_as_on_order,
    compute_frozen_policy_metrics,
)


def get_policy_config() -> PolicyConfigSchema:
    """Return the frozen Step 5.7 policy configuration schema."""
    return load_policy_config()


def get_entity_selectors(df: Optional[pd.DataFrame] = None) -> Dict[str, List[str]]:
    """
    Return sorted unique Store IDs, Product IDs, Categories, Regions, and valid Decision Dates
    from the development dataset (`TRAIN + VALIDATION` only).
    """
    if df is None:
        df = load_development_dataset()

    stores = sorted(df["Store ID"].astype(str).unique().tolist())
    products = sorted(df["Product ID"].astype(str).unique().tolist())
    categories = sorted(df["Category"].astype(str).unique().tolist())
    regions = sorted(df["Region"].astype(str).unique().tolist())
    dates = sorted(df["Date_Str"].astype(str).unique().tolist())

    return {
        "stores": stores,
        "products": products,
        "categories": categories,
        "regions": regions,
        "dates": dates,
        "min_date": dates[0],
        "max_date": dates[-1],
    }


def _resolve_fallback_source(raw_source: object, is_missing: bool) -> str:
    """
    Resolve fallback source string into one of ALLOWED_FALLBACK_SOURCES.
    On Cold-Start Day 1 (2022-01-01) when baseline/std is NaN, returns 'INSUFFICIENT_DATA'.
    """
    if is_missing or pd.isna(raw_source):
        return "INSUFFICIENT_DATA"
    src = str(raw_source).strip()
    if src in ALLOWED_FALLBACK_SOURCES:
        return src
    return "INSUFFICIENT_DATA"


def get_policy_record(
    store_id: str,
    product_id: str,
    decision_date: str,
    df: Optional[pd.DataFrame] = None,
) -> DashboardPolicyRecord:
    """
    Retrieve the locked `DashboardPolicyRecord` for a given `(store_id, product_id, decision_date)`
    from the frozen development dataset, enforcing strict date leakage validation (`t' < t`)
    and formula consistency checks.
    """
    if df is None:
        df = load_development_dataset()

    assert_no_forbidden_policy_inputs(list(df.columns))

    entity_df = df[
        (df["Store ID"] == str(store_id)) & (df["Product ID"] == str(product_id))
    ].sort_values("Date_Str")

    if entity_df.empty:
        raise KeyError(
            f"No records found for Store ID='{store_id}', Product ID='{product_id}'."
        )

    target_rows = entity_df[entity_df["Date_Str"] == str(decision_date)]
    if target_rows.empty:
        raise KeyError(
            f"Decision date '{decision_date}' not found for ({store_id}, {product_id}). "
            "Note: TEST dates (2023-10-21..2024-01-01) are strictly excluded."
        )

    row = target_rows.iloc[0]

    # Enforce strict historical causal boundary: all historical dates used prior to decision_date must satisfy t' < t
    prior_dates = entity_df[entity_df["Date_Str"] < str(decision_date)]["Date_Str"].tolist()
    assert_no_date_leakage(prior_dates, str(decision_date))

    hist_count = int(row["Historical_Demand_Count"])
    if len(prior_dates) != hist_count:
        raise AssertionError(
            f"Historical_Demand_Count mismatch for ({store_id}, {product_id}, {decision_date}): "
            f"expected {len(prior_dates)} prior dates, found {hist_count}."
        )

    inv_level = float(row["Inventory Level"])
    units_sold = float(row["Units Sold"])
    units_ordered_ref = float(row["Units Ordered"])

    # Enforce that Units Ordered is never added to Inventory Level
    assert_units_ordered_not_used_as_on_order(
        inventory_level=inv_level,
        effective_inventory_position=inv_level,
        units_ordered=units_ordered_ref,
    )

    is_baseline_missing = bool(pd.isna(row["Demand_Baseline"]) or pd.isna(row["Demand_Std"]))
    demand_baseline = None if is_baseline_missing else float(row["Demand_Baseline"])
    demand_std = None if is_baseline_missing else float(row["Demand_Std"])

    baseline_source = _resolve_fallback_source(row["Demand_Baseline_Source"], is_baseline_missing)
    std_source = _resolve_fallback_source(row["Demand_Std_Source"], is_baseline_missing)

    raw_status = str(row["Inventory_Action"]).strip()
    if raw_status not in ALLOWED_POLICY_STATUS:
        raise ValueError(f"Invalid policy status '{raw_status}' in dataset.")

    lead_time_demand = None if pd.isna(row["Lead_Time_Demand"]) else float(row["Lead_Time_Demand"])
    safety_stock = None if pd.isna(row["Safety_Stock"]) else float(row["Safety_Stock"])
    reorder_point = None if pd.isna(row["Reorder_Point"]) else float(row["Reorder_Point"])
    stockout_prob = (
        None
        if pd.isna(row.get("Estimated_Stockout_Risk", None))
        else float(row["Estimated_Stockout_Risk"])
    )

    # Cross-verify frozen formula mathematical identity
    recomputed = compute_frozen_policy_metrics(
        demand_baseline=demand_baseline,
        demand_std=demand_std,
        inventory_level=inv_level,
        lead_time_days=FROZEN_LEAD_TIME_DAYS,
        service_level=FROZEN_SERVICE_LEVEL,
        z_score=FROZEN_Z_SCORE,
    )
    if recomputed["policy_status"] != raw_status:
        raise AssertionError(
            f"Policy status mismatch for ({store_id}, {product_id}, {decision_date}): "
            f"dataset='{raw_status}' vs frozen rule='{recomputed['policy_status']}'."
        )
    if reorder_point is not None and recomputed["reorder_point"] is not None:
        if abs(reorder_point - float(recomputed["reorder_point"])) > 1e-3:
            raise AssertionError(
                f"Reorder point mismatch: dataset={reorder_point} vs frozen formula={recomputed['reorder_point']}."
            )

    inventory_gap_to_rop = (
        round(inv_level - reorder_point, 4) if reorder_point is not None else None
    )
    days_of_supply = (
        round(inv_level / demand_baseline, 2)
        if (demand_baseline is not None and demand_baseline > 0)
        else None
    )

    record = DashboardPolicyRecord(
        date=str(decision_date),
        store_id=str(store_id),
        product_id=str(product_id),
        category=str(row["Category"]),
        region=str(row["Region"]),
        split_label=str(row["Split_Label"]),
        inventory_level=round(inv_level, 4),
        units_sold=round(units_sold, 4),
        units_ordered_reference=round(units_ordered_ref, 4),
        demand_forecast_reference=None,
        historical_demand_count=hist_count,
        demand_baseline=round(demand_baseline, 4) if demand_baseline is not None else None,
        demand_baseline_source=baseline_source,
        demand_std=round(demand_std, 4) if demand_std is not None else None,
        demand_std_source=std_source,
        lead_time_days=FROZEN_LEAD_TIME_DAYS,
        service_level=FROZEN_SERVICE_LEVEL,
        z_score=FROZEN_Z_SCORE,
        lead_time_demand=round(lead_time_demand, 4) if lead_time_demand is not None else None,
        safety_stock=round(safety_stock, 4) if safety_stock is not None else None,
        reorder_point=round(reorder_point, 4) if reorder_point is not None else None,
        inventory_gap_to_rop=inventory_gap_to_rop,
        days_of_supply_estimate=days_of_supply,
        estimated_stockout_probability=round(stockout_prob, 4) if stockout_prob is not None else None,
        policy_status=raw_status,
        order_quantity_supported=False,
        inventory_level_time_semantics=str(row.get("inventory_level_time_semantics", "unknown")),
    )
    record.validate()
    return record


def get_historical_demand_series(
    store_id: str,
    product_id: str,
    decision_date: str,
    window_days: Optional[int] = None,
    df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Return strictly historical demand slices (`Date < decision_date`) and the current decision date row
    for visualization.
    """
    if df is None:
        df = load_development_dataset()

    entity_df = df[
        (df["Store ID"] == str(store_id)) & (df["Product ID"] == str(product_id))
    ].sort_values("Date_Str")

    current_rows = entity_df[entity_df["Date_Str"] == str(decision_date)]
    if current_rows.empty:
        raise KeyError(f"Decision date '{decision_date}' not found for ({store_id}, {product_id}).")

    current_row = current_rows.iloc[0]

    hist_df = entity_df[entity_df["Date_Str"] < str(decision_date)].copy()
    assert_no_date_leakage(hist_df["Date_Str"].tolist(), str(decision_date))

    if window_days is not None and window_days > 0 and len(hist_df) > window_days:
        hist_df = hist_df.tail(int(window_days)).copy()

    return hist_df, current_row


def get_entity_trajectory_up_to_date(
    store_id: str,
    product_id: str,
    decision_date: str,
    window_days: Optional[int] = 90,
    df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Return the Store × Product trajectory up to and including `decision_date` (`Date <= decision_date`)
    for plotting the historical evolution of Inventory Level vs. Reorder Point and Safety Stock.
    """
    if df is None:
        df = load_development_dataset()

    entity_df = df[
        (df["Store ID"] == str(store_id))
        & (df["Product ID"] == str(product_id))
        & (df["Date_Str"] <= str(decision_date))
    ].sort_values("Date_Str").copy()

    prior_dates = entity_df[entity_df["Date_Str"] < str(decision_date)]["Date_Str"].tolist()
    assert_no_date_leakage(prior_dates, str(decision_date))

    if window_days is not None and window_days > 0 and len(entity_df) > window_days:
        entity_df = entity_df.tail(int(window_days)).copy()

    return entity_df


def get_scenario_matrix_for_record(
    record: DashboardPolicyRecord,
    backtest_data: Optional[Dict[str, pd.DataFrame]] = None,
) -> List[ScenarioRecordSchema]:
    """
    Return the 9 pre-approved scenario rows (`LT ∈ {2,4,7} × SL ∈ {0.90,0.95,0.99}`)
    combining the selected `(Store, Product, Decision Date)` snapshot with the frozen
    Step 5.6 historical backtest metrics (`scenario_backtest_comparison.csv`).
    """
    if backtest_data is None:
        backtest_data = load_backtest_results()

    scen_df = backtest_data["scenario_backtest"]
    results: List[ScenarioRecordSchema] = []

    for lt in SCENARIO_LEAD_TIMES:
        for sl in SCENARIO_SERVICE_LEVELS:
            sl_pct = int(round(sl * 100))
            scen_id = f"LT{lt}_SL{sl_pct}"
            z_val = SCENARIO_Z_SCORES[sl]
            is_base = (lt == FROZEN_LEAD_TIME_DAYS) and (abs(sl - FROZEN_SERVICE_LEVEL) < 1e-6)

            metrics = compute_frozen_policy_metrics(
                demand_baseline=record.demand_baseline,
                demand_std=record.demand_std,
                inventory_level=record.inventory_level,
                lead_time_days=lt,
                service_level=sl,
                z_score=z_val,
            )

            match = scen_df[scen_df["Scenario"] == scen_id]
            if not match.empty:
                m_row = match.iloc[0]
                hist_reorder_rate = round(float(m_row["Policy_Population_Trigger_Rate_Pct"]), 2)
                hist_coverage = round(float(m_row["Future_Demand_Coverage_Rate"]) * 100.0, 2)
                hist_precision = round(float(m_row["LTD_Coverage_Rate_Pct"]), 2)
                hist_false_rate = round(float(m_row["Future_Demand_Exceeds_ROP_Rate"]) * 100.0, 2)
                hist_avg_ss = round(float(m_row["Mean_Safety_Stock"]), 2)
                hist_avg_rop = round(float(m_row["Mean_Reorder_Point"]), 2)
            else:
                hist_reorder_rate = None
                hist_coverage = None
                hist_precision = None
                hist_false_rate = None
                hist_avg_ss = None
                hist_avg_rop = None

            item = ScenarioRecordSchema(
                scenario_id=scen_id,
                lead_time_days=lt,
                service_level=sl,
                z_score=z_val,
                is_base_policy=is_base,
                lead_time_demand=metrics["lead_time_demand"],
                safety_stock=metrics["safety_stock"],
                reorder_point=metrics["reorder_point"],
                policy_status=str(metrics["policy_status"]),
                historical_reorder_rate=hist_reorder_rate,
                historical_stockout_coverage=hist_coverage,
                historical_reorder_precision=hist_precision,
                historical_false_reorder_rate=hist_false_rate,
                historical_avg_safety_stock=hist_avg_ss,
                historical_avg_rop=hist_avg_rop,
            )
            item.validate()
            results.append(item)

    return results


def get_system_overview_kpis(
    decision_date: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
) -> Dict[str, object]:
    """
    Compute aggregate KPIs for the Overview page across all 65,800 development rows
    as well as for the selected decision date snapshot (100 Store × Product pairs).
    """
    if df is None:
        df = load_development_dataset()

    total_rows = len(df)
    status_counts = df["Inventory_Action"].value_counts().to_dict()
    reorder_count = int(status_counts.get("REORDER", 0))
    monitor_count = int(status_counts.get("MONITOR", 0))
    insufficient_count = int(status_counts.get("INSUFFICIENT_DATA", 0))

    valid_df = df[df["Inventory_Action"] != "INSUFFICIENT_DATA"]

    fallback_summary = {
        "STORE_PRODUCT": int(
            valid_df["Demand_Std_Source"].eq("STORE_PRODUCT").sum()
        ),
        "PRODUCT": int(valid_df["Demand_Std_Source"].eq("PRODUCT").sum()),
        "STORE": int(valid_df["Demand_Std_Source"].eq("STORE").sum()),
        "GLOBAL_TRAIN": int(
            valid_df["Demand_Std_Source"].eq("GLOBAL_TRAIN").sum()
        ),
        "INSUFFICIENT_DATA": insufficient_count,
    }

    date_snapshot = {}
    if decision_date is not None:
        sub = df[df["Date_Str"] == str(decision_date)]
        sub_counts = sub["Inventory_Action"].value_counts().to_dict()
        date_snapshot = {
            "date": str(decision_date),
            "total_skus": len(sub),
            "reorder_count": int(sub_counts.get("REORDER", 0)),
            "monitor_count": int(sub_counts.get("MONITOR", 0)),
            "insufficient_count": int(sub_counts.get("INSUFFICIENT_DATA", 0)),
            "reorder_pct": round(
                100.0 * int(sub_counts.get("REORDER", 0)) / max(len(sub), 1), 2
            ),
            "avg_inventory": round(float(sub["Inventory Level"].mean()), 2)
            if not sub.empty
            else 0.0,
            "avg_rop": round(float(sub["Reorder_Point"].dropna().mean()), 2)
            if not sub["Reorder_Point"].dropna().empty
            else None,
            "avg_safety_stock": round(float(sub["Safety_Stock"].dropna().mean()), 2)
            if not sub["Safety_Stock"].dropna().empty
            else None,
        }

    return {
        "total_rows": total_rows,
        "train_rows": int((df["Split_Label"] == "TRAIN").sum()),
        "validation_rows": int((df["Split_Label"] == "VALIDATION").sum()),
        "stores_count": int(df["Store ID"].nunique()),
        "products_count": int(df["Product ID"].nunique()),
        "store_product_pairs": int(df.groupby(["Store ID", "Product ID"]).ngroups),
        "reorder_count": reorder_count,
        "monitor_count": monitor_count,
        "insufficient_count": insufficient_count,
        "reorder_pct": round(100.0 * reorder_count / total_rows, 2),
        "monitor_pct": round(100.0 * monitor_count / total_rows, 2),
        "insufficient_pct": round(100.0 * insufficient_count / total_rows, 2),
        "avg_demand_baseline": round(float(valid_df["Demand_Baseline"].mean()), 2),
        "avg_demand_std": round(float(valid_df["Demand_Std"].mean()), 2),
        "avg_lead_time_demand": round(float(valid_df["Lead_Time_Demand"].mean()), 2),
        "avg_safety_stock": round(float(valid_df["Safety_Stock"].mean()), 2),
        "avg_reorder_point": round(float(valid_df["Reorder_Point"].mean()), 2),
        "avg_inventory_level": round(float(df["Inventory Level"].mean()), 2),
        "fallback_summary": fallback_summary,
        "date_snapshot": date_snapshot,
    }
