# SMART INVENTORY MANAGEMENT & DEMAND FORECASTING SYSTEM (SIM&DFS)

## STEP 6 — PRODUCTION DASHBOARD DOCUMENTATION (`STEP_5_7_FROZEN_v1.0`)

---

## 1. Dashboard Purpose

The **SIM&DFS Step 6 Dashboard** (`C:\SIM&DFS\app.py`) is a modular, read-only decision-support application built with **Python, Streamlit, and Plotly**. Its sole purpose is to surface the validated, backtested, and **frozen Step 5.7 Inventory Replenishment Policy** (`STEP_5_7_FROZEN_v1.0`) across `100` `Store × Product` entities (`5` Stores × `20` Products) and `658` development dates (`2022-01-01` to `2023-10-20`, `65,800` `TRAIN + VALIDATION` rows).

It does **not** retrain models, tune parameters, recompute dynamic ML forecasts, or mutate the frozen replenishment formulas.

---

## 2. How to Launch the Dashboard

From the project root (`C:\SIM&DFS`), run:

```powershell
py -m streamlit run app.py
```

To execute the automated Step 6 verification & acceptance test suite (`23 / 23` tests covering Section 45 and Section 50):

```powershell
py tests/test_step6_dashboard.py
```

---

## 3. Required Input Files

All file paths are centralized in `C:\SIM&DFS\dashboard\config\policy_config.py` (`DATA_FILES`):

| Key | File Path | Purpose |
| :--- | :--- | :--- |
| `policy_config` | `C:\SIM&DFS\data\processed\inventory_optimization\final_policy\final_policy_locked_config.json` | Frozen Step 5.7 policy specification & JSON contract |
| `final_policy_spec_md` | `C:\SIM&DFS\reports\step5_inventory_optimization\final_policy\step5_7_final_inventory_policy_specification.md` | Step 5.7 policy governance specification |
| `inventory_train_val` | `C:\SIM&DFS\data\processed\inventory_optimization\inventory_optimization_train_val.csv` | Primary dashboard dataset (`65,800` rows, `TRAIN + VALIDATION` only) |
| `inventory_full_dataset` | `C:\SIM&DFS\data\processed\inventory_optimization_dataset.csv` | Full Step 5 dataset (`73,100` rows; filtered to `TRAIN + VALIDATION` on load) |
| `scenario_summary` | `C:\SIM&DFS\data\processed\inventory_optimization\inventory_policy_scenario_summary.csv` | Step 5 pre-computed 9-scenario summary (`LT ∈ {2,4,7} × SL ∈ {90%,95%,99%}`) |
| `validation_checks` | `C:\SIM&DFS\data\processed\inventory_optimization\inventory_optimization_validation_checks.csv` | Step 5 `18 / 18 PASS` validation checks |
| `leakage_audit` | `C:\SIM&DFS\data\processed\inventory_optimization\inventory_optimization_leakage_audit.csv` | Step 5 `11 / 11 PASS` causal leakage checks |
| `trigger_overall_summary` | `C:\SIM&DFS\data\processed\inventory_optimization\trigger_diagnosis\trigger_rate_overall_summary.csv` | Step 5.5 overall trigger-rate diagnosis |
| `backtest_summary_metrics` | `C:\SIM&DFS\data\processed\inventory_optimization\backtest\policy_backtest_summary_metrics.csv` | Step 5.6 historical backtest summary metrics |
| `backtest_scenario_comparison` | `C:\SIM&DFS\data\processed\inventory_optimization\backtest\scenario_backtest_comparison.csv` | Step 5.6 9-scenario historical backtest comparison |
| `backtest_validation_checks` | `C:\SIM&DFS\data\processed\inventory_optimization\backtest\policy_backtest_validation_checks.csv` | Step 5.6 `13 / 13 PASS` backtest validation suite |
| `protected_test_split` | `C:\SIM&DFS\data\processed\splits\test.csv` | Protected holdout split (`7,300` rows; SHA-256 verified at startup, never displayed) |

---

## 4. Frozen Step 5.7 Policy Parameters

| Parameter | Locked Value | Notes |
| :--- | :--- | :--- |
| `policy_version` | `STEP_5_7_FROZEN_v1.0` | Frozen in Step 5.7 |
| `base_lead_time_days` ($L$) | `4` days | Locked base lead time |
| `base_service_level` | `0.95` (`95%`) | Locked cycle service level |
| `base_z_score` ($Z$) | `1.6449` | Standard normal quantile $\Phi^{-1}(0.95)$ |
| `Lead_Time_Demand` ($\text{LTD}$) | $\text{Demand\_Baseline} \times L$ | Uses 28-day prior rolling mean (`t' < t`) |
| `Safety_Stock` ($\text{SS}$) | $Z \times \text{Demand\_Std} \times \sqrt{L}$ | Strictly uses $\sqrt{L}$ (`math.sqrt(L)`), **never** linear $L$ |
| `Reorder_Point` ($\text{ROP}$) | $\text{Lead\_Time\_Demand} + \text{Safety\_Stock}$ | Frozen trigger threshold |
| Decision Rule | $\text{Inventory\_Level} \le \text{ROP} \implies \text{REORDER}$, else $\text{MONITOR}$ | Or `INSUFFICIENT_DATA` on Day 1 (`2022-01-01`) |
| Fallback Hierarchy | `STORE_PRODUCT` ($\ge 7$d mean, $\ge 30$d std) $\rightarrow$ `PRODUCT` $\rightarrow$ `STORE` $\rightarrow$ `GLOBAL_TRAIN` $\rightarrow$ `INSUFFICIENT_DATA` | Causal prior hierarchy (`t' < t`) |

---

## 5. Data Leakage Protection

1. **Strict Causal Boundary (`t' < t`)**: Every call to `get_policy_record()` and `get_historical_demand_series()` invokes `assert_no_date_leakage(historical_dates, decision_date)` to verify that every historical date strictly precedes `decision_date`.
2. **Cold-Start Preservation**: On `2022-01-01` (Day 1, `Historical_Demand_Count = 0`), `Demand_Baseline`, `Demand_Std`, `Lead_Time_Demand`, `Safety_Stock`, and `Reorder_Point` are preserved as `None`/`NaN` (never converted to `0`) and `policy_status` is `"INSUFFICIENT_DATA"`.
3. **Visualization Window Isolation**: Selecting `30`, `90`, `180`, or `All` in Page 2 (`Demand Analysis`) slices only the displayed chart history and never alters the frozen policy record.

---

## 6. TEST Split Protection

At application startup (`app.py`), `verify_test_split_checksum()` verifies `C:\SIM&DFS\data\processed\splits\test.csv`:
- **Expected SHA-256**: `4fd35bc0da8548ed50e347d6d0a00f925d75a40ee3382f79f51dff3462edbe96`
- **Date Exclusion**: All dates in `2023-10-21` to `2024-01-01` (`TEST` split, `7,300` rows) are blocked from all selectors, tables, and charts (`DEV_DATE_MAX = "2023-10-20"`).

---

## 7. Inventory Semantics Assumptions

- **`inventory_level_time_semantics = "unknown"`**: The source dataset does not specify whether `Inventory Level` is recorded at start-of-day or end-of-day.
- **`Units Ordered` Is Reference-Only**: Step 5 proved that `Units Ordered` lacks delivery arrival timestamps and open on-order pipeline tracking. Therefore, `assert_units_ordered_not_used_as_on_order()` guarantees `Units Ordered` is never added to `Inventory Level`.

---

## 8. Why `Demand Forecast` Is Excluded

Steps 10–13 established that dynamic regression models (`Ridge`, `RandomForest`, `HistGradientBoosting`) achieved near-zero improvement over entity historical mean (`R² ≈ -0.0017` on validation) due to high-variance stochastic daily noise. Moreover, the raw dataset column `Demand Forecast` has undocumented generation provenance and is explicitly blocked in `FORBIDDEN_POLICY_INPUT_COLUMNS`.

---

## 9. Why Order Quantity Is Not Computed (`order_quantity_supported = False`)

Computing an Economic Order Quantity (EOQ) or Order-Up-To level ($S$) requires verified fixed ordering cost ($K$), unit holding cost ($h$), stockout penalty cost ($p$), and supplier minimum order quantity (MOQ). Because these parameters do not exist in the dataset, SIM&DFS outputs strictly validated binary replenishment triggers (`REORDER`, `MONITOR`, `INSUFFICIENT_DATA`) with `order_quantity_supported = False`.

---

## 10. Why the `REORDER` Trigger Rate Is High (`93.36%`)

Step 5.5 (`Inventory Trigger-Rate Diagnosis`) verified that the `93.36%` (`61,428 / 65,800`) `REORDER` rate is a genuine structural property of the retail dataset:
- Mean on-hand `Inventory Level` is `274.43 units` (`~2.02 days` of supply at `136.46 units/day` average demand).
- Over a `4-day` lead time, mean `Lead_Time_Demand` alone is `545.84 units` (exceeding the `500`-unit maximum inventory cap on `75.56%` of rows even with zero safety stock), and mean `Reorder_Point` (`LTD + SS`) is `758.92 units`.

---

## 11. Page Descriptions

1. **Overview (`dashboard/pages/overview.py`)**: Executive KPIs for the selected `(Store, Product, Decision Date)`, portfolio-wide snapshot across all `100` SKUs on `decision_date`, and `65,800`-row fallback tier breakdown.
2. **Demand Analysis (`dashboard/pages/demand.py`)**: Causal prior daily demand series (`t' < t`) with display-only window selector (`30 / 90 / 180 / All`), `±1σ` band, empirical histogram, and Step 13 forecast exclusion rationale.
3. **Inventory Decision (`dashboard/pages/inventory.py`)**: Full Step 5.7 formula verification card (`LTD`, `SS` with $\sqrt{L}$, `ROP`), horizontal threshold comparison bar chart, and historical trajectory (`Date <= decision_date`).
4. **Scenario Comparison (`dashboard/pages/scenarios.py`)**: Read-only 3×3 pre-approved scenario table (`LT ∈ {2,4,7} × SL ∈ {90%,95%,99%}`) comparing snapshot thresholds and Step 5.6 historical backtest tradeoff curves.
5. **Backtest & Validation (`dashboard/pages/backtest.py`)**: Step 5.6 4-day horizon confusion matrix (`99.49%` stockout coverage, `43.05%` precision), Step 5.5 trigger-rate diagnosis tables, and all `42 / 42 PASS` validation & leakage audits plus live TEST split SHA-256 verification.

---

## 12. Known Limitations

1. **Unobserved Cost Structure**: Dollar savings and EOQ quantities cannot be claimed without external ERP cost parameters.
2. **Stationary Lead Time Assumption**: Lead time is modeled as a deterministic constant (`4 days` base; `2, 4, 7 days` sensitivity) because supplier delivery timestamps are absent.
3. **High Structural Trigger Rate**: Because dataset inventory is bounded (`50–500 units`, `~2 days` of supply), any multi-day lead-time policy (`L ≥ 2`) structurally stays in `REORDER` across most observations.
