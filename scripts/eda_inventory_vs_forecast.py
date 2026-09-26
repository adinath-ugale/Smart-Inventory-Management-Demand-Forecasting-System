import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==============================================================
# SMART INVENTORY MANAGEMENT & DEMAND FORECASTING SYSTEM
# EDA STEP 15: INVENTORY LEVEL VS DEMAND FORECAST ANALYSIS
# ==============================================================

print("=" * 70)
print("SMART INVENTORY MANAGEMENT & DEMAND FORECASTING SYSTEM")
print("EDA STEP 15: INVENTORY LEVEL VS DEMAND FORECAST ANALYSIS")
print("=" * 70)

# --------------------------------------------------------------
# 1. PATHS
# --------------------------------------------------------------

INPUT_FILE = Path(
    r"C:\SIM&DFS\data\processed\cleaned_inventory.csv"
)

OUTPUT_DIR = Path(
    r"C:\SIM&DFS\reports\figures"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "inventory_vs_forecast.png"

# --------------------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------------------

try:
    df = pd.read_csv(INPUT_FILE)

except FileNotFoundError:
    print("\nERROR: Dataset file not found.")
    print(f"Expected file:\n{INPUT_FILE}")
    raise SystemExit(1)

print("\nDataset loaded successfully.")
print(f"Rows: {len(df):,}")

# --------------------------------------------------------------
# 3. CHECK REQUIRED COLUMNS
# --------------------------------------------------------------

required_columns = [
    "Category",
    "Region",
    "Inventory Level",
    "Demand Forecast"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:

    print("\nERROR: Required columns are missing:")

    for col in missing_columns:
        print(f"- {col}")

    raise SystemExit(1)

# --------------------------------------------------------------
# 4. OVERALL INVENTORY VS FORECAST
# --------------------------------------------------------------

total_inventory = df["Inventory Level"].sum()
total_forecast = df["Demand Forecast"].sum()

inventory_forecast_difference = (
    total_inventory - total_forecast
)

if total_forecast > 0:

    inventory_forecast_ratio = (
        total_inventory / total_forecast
    )

else:

    inventory_forecast_ratio = 0

print("\n--- OVERALL INVENTORY VS FORECAST ---")

print(
    f"Total Inventory Level : "
    f"{total_inventory:,.0f}"
)

print(
    f"Total Forecast Demand : "
    f"{total_forecast:,.0f}"
)

print(
    f"Inventory-Forecast Difference : "
    f"{inventory_forecast_difference:,.0f}"
)

print(
    f"Inventory-to-Forecast Ratio : "
    f"{inventory_forecast_ratio:.2f}"
)

# --------------------------------------------------------------
# 5. CATEGORY-WISE ANALYSIS
# --------------------------------------------------------------

category_summary = (
    df.groupby("Category")
    .agg(
        Total_Inventory=("Inventory Level", "sum"),
        Total_Forecast=("Demand Forecast", "sum")
    )
    .reset_index()
)

category_summary["Inventory_Forecast_Difference"] = (
    category_summary["Total_Inventory"]
    - category_summary["Total_Forecast"]
)

category_summary["Inventory_to_Forecast_Ratio"] = (
    category_summary["Total_Inventory"]
    / category_summary["Total_Forecast"]
)

print("\n--- CATEGORY-WISE INVENTORY VS FORECAST ---")

print(
    category_summary.to_string(index=False)
)

# --------------------------------------------------------------
# 6. CATEGORY EXTREMES
# --------------------------------------------------------------

highest_ratio_category = category_summary.loc[
    category_summary["Inventory_to_Forecast_Ratio"].idxmax()
]

lowest_ratio_category = category_summary.loc[
    category_summary["Inventory_to_Forecast_Ratio"].idxmin()
]

print("\n--- CATEGORY COVERAGE EXTREMES ---")

print(
    f"Highest inventory/forecast ratio : "
    f"{highest_ratio_category['Category']} "
    f"({highest_ratio_category['Inventory_to_Forecast_Ratio']:.2f})"
)

print(
    f"Lowest inventory/forecast ratio  : "
    f"{lowest_ratio_category['Category']} "
    f"({lowest_ratio_category['Inventory_to_Forecast_Ratio']:.2f})"
)

# --------------------------------------------------------------
# 7. REGION-WISE ANALYSIS
# --------------------------------------------------------------

region_summary = (
    df.groupby("Region")
    .agg(
        Total_Inventory=("Inventory Level", "sum"),
        Total_Forecast=("Demand Forecast", "sum")
    )
    .reset_index()
)

region_summary["Inventory_Forecast_Difference"] = (
    region_summary["Total_Inventory"]
    - region_summary["Total_Forecast"]
)

region_summary["Inventory_to_Forecast_Ratio"] = (
    region_summary["Total_Inventory"]
    / region_summary["Total_Forecast"]
)

print("\n--- REGION-WISE INVENTORY VS FORECAST ---")

print(
    region_summary.to_string(index=False)
)

# --------------------------------------------------------------
# 8. REGION EXTREMES
# --------------------------------------------------------------

highest_ratio_region = region_summary.loc[
    region_summary["Inventory_to_Forecast_Ratio"].idxmax()
]

lowest_ratio_region = region_summary.loc[
    region_summary["Inventory_to_Forecast_Ratio"].idxmin()
]

print("\n--- REGION COVERAGE EXTREMES ---")

print(
    f"Highest inventory/forecast ratio : "
    f"{highest_ratio_region['Region']} "
    f"({highest_ratio_region['Inventory_to_Forecast_Ratio']:.2f})"
)

print(
    f"Lowest inventory/forecast ratio  : "
    f"{lowest_ratio_region['Region']} "
    f"({lowest_ratio_region['Inventory_to_Forecast_Ratio']:.2f})"
)

# --------------------------------------------------------------
# 9. INTERPRETATION
# --------------------------------------------------------------

print("\n--- INTERPRETATION ---")

if total_inventory > total_forecast:

    print(
        "Total recorded inventory is higher than total "
        "forecast demand."
    )

elif total_inventory < total_forecast:

    print(
        "Total recorded inventory is lower than total "
        "forecast demand."
    )

else:

    print(
        "Total recorded inventory is equal to total "
        "forecast demand."
    )

print(
    "The inventory-to-forecast ratio provides a descriptive "
    "measure of inventory coverage relative to forecast demand."
)

print(
    "It should not be treated as a final reorder decision because "
    "time period, lead time, safety stock, and inventory flow "
    "must also be considered."
)

# --------------------------------------------------------------
# 10. CREATE CHART
# --------------------------------------------------------------

plt.figure(figsize=(11, 7))

x = range(len(category_summary))

bar_width = 0.35

plt.bar(
    [i - bar_width / 2 for i in x],
    category_summary["Total_Inventory"],
    width=bar_width,
    label="Inventory Level"
)

plt.bar(
    [i + bar_width / 2 for i in x],
    category_summary["Total_Forecast"],
    width=bar_width,
    label="Demand Forecast"
)

plt.title(
    "Inventory Level vs Demand Forecast by Category",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel(
    "Category",
    fontsize=12
)

plt.ylabel(
    "Units",
    fontsize=12
)

plt.xticks(
    list(x),
    category_summary["Category"],
    rotation=30
)

plt.legend()

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.5
)

plt.tight_layout()

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# --------------------------------------------------------------
# 11. OUTPUT
# --------------------------------------------------------------

print("\n--- OUTPUT ---")

print(
    "Inventory vs Forecast chart saved to:"
)

print(
    OUTPUT_FILE
)

print("\nEDA Step 15 completed successfully.")
print("=" * 70)