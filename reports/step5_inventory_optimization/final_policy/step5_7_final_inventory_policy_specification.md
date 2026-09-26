# SMART INVENTORY MANAGEMENT & DEMAND FORECASTING SYSTEM (SIM&DFS)

## STEP 5.7 — FINAL INVENTORY POLICY SPECIFICATION & DECISION FRAMEWORK

**Status**: `COMPLETE & FROZEN`  
**Project Root**: `C:\SIM&DFS`  
**TEST Split SHA-256 (Verified Unchanged)**: `4fd35bc0da8548ed50e347d6d0a00f925d75a40ee3382f79f51dff3462edbe96`  
**Step 5.6 Leakage Audit**: `13 / 13 PASS`

---

### 1. Purpose

This document defines the locked, final inventory replenishment policy for the Smart Inventory Management & Demand Forecasting System (SIM&DFS), synthesizing the validated evidence from:
* **Step 5** — Leakage-Safe Inventory Optimization Data Specification (`18 / 18 PASS`)
* **Step 5.5** — Inventory Trigger-Rate Diagnosis (`PASS`)
* **Step 5.6** — Inventory Policy Backtesting (`13 / 13 PASS`)

The policy is defined as a **risk-based replenishment decision framework**, not as a mathematically proven cost-optimal inventory policy, because verified ordering costs, holding costs, stockout penalty costs, supplier lead-time distributions, review periods, backorder quantities, and inventory-position timing semantics are not available in the source dataset.

---

### 2. Data and Leakage Policy

* **Development Population (`TRAIN + VALIDATION`)**:
  * `TRAIN`: `58,500` rows (`2022-01-01` to `2023-08-08`)
  * `VALIDATION`: `7,300` rows (`2023-08-09` to `2023-10-20`)
  * `TRAIN + VALIDATION`: `65,800` rows (`65,700` active policy decision rows; `65,400` eligible 4-day backtest outcome rows)
* **TEST Protection**:
  * `TEST` period (`2023-10-21` to `2024-01-01`, `7,300` rows, `C:\SIM&DFS\data\processed\splits\test.csv`) remained completely untouched across Steps 5, 5.5, 5.6, and 5.7.
  * `TEST` was not used for demand baseline calculation, demand standard deviation calculation, fallback hierarchy, safety-stock calculation, ROP calculation, trigger diagnosis, policy backtesting, scenario comparison, or policy selection.
  * `TEST` SHA-256 checksum verified unchanged: `4fd35bc0da8548ed50e347d6d0a00f925d75a40ee3382f79f51dff3462edbe96`.

---

### 3. Decision Timing Rule

For any replenishment decision made on date `D`:
* All policy inputs must use information available strictly before the decision date (`Date < D`, cutoff `D - 1 day`).
* Current-day `Units Sold` at `D` must never be used as a policy input.
* Future realized demand (`Units Sold` over `[D, D + L - 1]`) is used strictly as a post-decision backtest outcome, never as a policy input.

---

### 4. Demand Baseline & Variability with 5-Tier Fallback Hierarchy

#### Primary Formulas (`Date < D` within `Store x Product`)
* `Demand_Baseline(Store, Product, D) = Mean(Units Sold where Date < D)`
* `Demand_Std(Store, Product, D) = Sample_Std(Units Sold where Date < D, ddof=1)`

#### Locked Fallback Hierarchy
| Priority | Source Code | Entity Level | Min History (`Demand_Baseline`) | Min History (`Demand_Std`) |
| :---: | :--- | :--- | :---: | :---: |
| **1** | `STORE_PRODUCT` | `Store ID x Product ID` (100 groups) | `7 days` | `30 days` |
| **2** | `PRODUCT` | `Product ID` (20 products) | `7 observations` | `30 observations` |
| **3** | `STORE` | `Store ID` (5 stores) | `7 observations` | `30 observations` |
| **4** | `GLOBAL_TRAIN` | Global `TRAIN` history (`Date < D`) | `7 observations` | `30 observations` |
| **5** | `INSUFFICIENT_DATA` | Cold-Start (`2022-01-01`, 0 prior obs) | Preserved as `NaN` | Preserved as `NaN` |

* Selected sources are recorded in `Demand_Baseline_Source` and `Demand_Std_Source`.
* Missing historical statistics on Cold-Start Day 1 (`2022-01-01`, `100` rows) are never replaced with zero.

---

### 5. Locked Policy Formulas & Scenario Framework

#### Parametric Policy Equations
* **Lead-Time Demand (`LTD`)**:
  $$\text{Lead\_Time\_Demand} = \text{Demand\_Baseline} \times L$$
* **Safety Stock (`SS`)**:
  $$\text{Safety\_Stock} = Z \times \text{Demand\_Std} \times \sqrt{L}$$
* **Reorder Point (`ROP`)**:
  $$\text{Reorder\_Point} = \text{Lead\_Time\_Demand} + \text{Safety\_Stock}$$

#### Locked Base Policy Configuration
* **Lead Time ($L$)**: Assumed as `4 days` (`Lead_Time_Source = "ASSUMED_SCENARIO"`; sensitivity grid: `2, 4, 7 days`)
* **Service Level**: Assumed target policy of `95%` (`Service_Level_Source = "ASSUMED_POLICY"`; $Z = 1.6449$; sensitivity grid: `90%` [$Z=1.2816$], `95%` [$Z=1.6449$], `99%` [$Z=2.3263$])
* **Base Safety Stock**: $\text{Safety\_Stock} = 1.6449 \times \text{Demand\_Std} \times \sqrt{4} = 1.6449 \times \text{Demand\_Std} \times 2$
* **Decision Rule**:
  * If required baseline/variability inputs are unavailable $\rightarrow$ `INSUFFICIENT_DATA`
  * Else if `Current_Inventory <= Reorder_Point` $\rightarrow$ `REORDER` (`Reorder_Flag = 1`)
  * Else (`Current_Inventory > Reorder_Point`) $\rightarrow$ `MONITOR` (`Reorder_Flag = 0`)

---

### 6. Validated Step 5.6 Backtest Evidence (`TRAIN + VALIDATION`)

#### Base Scenario (`LT = 4 Days`, `SL = 95%`, $Z = 1.6449$)
| Measure | Validated Result |
| :--- | :---: |
| Policy Decision Population (`POLICY_DECISION_POPULATION`) | `65,700` rows |
| Backtest Eligible Population (`BACKTEST_OUTCOME_POPULATION`, 4D) | `65,400` rows |
| Mean `Lead_Time_Demand` (`LTD`) | `546.81` units |
| Mean `Safety_Stock` (`SS`) | `357.27` units |
| Mean `Reorder_Point` (`ROP`) | `904.07` units |
| Mean Realized 4-Day Future Demand (`Future_Demand_4D`) | `545.40` units |
| Mean `Future_Demand_4D / LTD` Ratio | `1.0026` |
| `LTD` Coverage Rate ($\text{Future\_Demand\_4D} \le \text{LTD}$) | `53.28%` (`34,842` rows) |
| `ROP` Coverage Rate ($\text{Future\_Demand\_4D} \le \text{ROP}$) | **`93.30%`** (`61,019` rows) |
| Realized Demand Exceeded Calculated `ROP` ($\text{Future\_Demand\_4D} > \text{ROP}$) | **`6.70%`** (`4,381` rows) |
| `Zero_Inventory_Flag` Rate ($\text{Inventory Level} \le 0$) | `0.00%` (`0` rows) |
| Conditional Policy Trigger Rate ($\text{Current\_Inventory} \le \text{ROP}$) | `100.00%` |

#### 9-Scenario Sensitivity Summary (Aligned Horizons: `LT2` $\rightarrow$ `2D`, `LT4` $\rightarrow$ `4D`, `LT7` $\rightarrow$ `7D`)
| Scenario | Conditional Trigger Rate | `LTD` Coverage | `ROP` Coverage | Realized Demand $>$ `ROP` |
| :--- | :---: | :---: | :---: | :---: |
| **`LT2 / SL90`** | `93.14%` | `54.85%` | `88.18%` | `11.82%` |
| **`LT2 / SL95`** | `99.39%` | `54.85%` | `93.01%` | `6.99%` |
| **`LT2 / SL99`** | `99.96%` | `54.85%` | `97.75%` | `2.25%` |
| **`LT4 / SL90`** | `100.00%` | `53.28%` | `88.69%` | `11.31%` |
| **`LT4 / SL95 (BASE)`** | **`100.00%`** | **`53.28%`** | **`93.30%`** | **`6.70%`** |
| **`LT4 / SL99`** | `100.00%` | `53.28%` | `98.02%` | `1.98%` |
| **`LT7 / SL90`** | `100.00%` | `52.40%` | `88.82%` | `11.18%` |
| **`LT7 / SL95`** | `100.00%` | `52.40%` | `93.54%` | `6.46%` |
| **`LT7 / SL99`** | `100.00%` | `52.40%` | `98.09%` | `1.91%` |

---

### 7. Final Decision Table & Operational Semantics Safeguards

| Data Condition | Policy Calculation | Decision State | Interpretation |
| :--- | :--- | :--- | :--- |
| Sufficient `Store x Product` history & `Current_Inventory <= ROP` | Full entity policy calculation | **`REORDER`** | Recorded inventory is at or below calculated reorder point |
| Sufficient `Store x Product` history & `Current_Inventory > ROP` | Full entity policy calculation | **`MONITOR`** | Recorded inventory is above calculated reorder point |
| `Store x Product` history insufficient (`<7` mean / `<30` std), `Product` sufficient | `PRODUCT` fallback | **`REORDER / MONITOR`** | Decision based on Product-level historical demand estimate |
| `Product` history insufficient, `Store` sufficient | `STORE` fallback | **`REORDER / MONITOR`** | Decision based on Store-level historical demand estimate |
| `Store` history insufficient, `Global TRAIN` available | `GLOBAL_TRAIN` fallback | **`REORDER / MONITOR`** | Decision based on Global TRAIN historical demand estimate |
| No valid historical demand baseline (`2022-01-01` cold-start) | No calculation (`NaN` preserved) | **`INSUFFICIENT_DATA`** | Do not generate an unsupported replenishment decision |
| Inventory timing semantics unverified (`inventory_level_time_semantics = "unknown"`) | Policy calculated conditionally | **`CONDITIONAL RESULT`** | Treat result as conditional historical simulation, not verified operational execution |
