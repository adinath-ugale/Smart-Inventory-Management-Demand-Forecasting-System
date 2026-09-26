import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==============================================================
# SMART INVENTORY MANAGEMENT & DEMAND FORECASTING SYSTEM
# EDA STEP 13: INVENTORY UTILIZATION ANALYSIS
# ==============================================================

print("=" * 70)
print("SMART INVENTORY MANAGEMENT & DEMAND FORECASTING SYSTEM")
print("EDA STEP 13: INVENTORY UTILIZATION ANALYSIS")
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

OUTPUT_FILE = OUTPUT_DIR / "inventory_utilization_by_category.png"

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
    "Units Sold"
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
# 4. OVERALL INVENTORY UTILIZATION
# --------------------------------------------------------------

total_inventory = df["Inventory Level"].sum()
total_units_sold = df["Units Sold"].sum()

if total_inventory > 0:

    overall_utilization = (
        total_units_sold / total_inventory
    ) * 100

else:

    overall_utilization = 0

print("\n--- OVERALL INVENTORY UTILIZATION ---")

print(
    f"Total Inventory Level : "
    f"{total_inventory:,.0f}"
)

print(
    f"Total Units Sold      : "
    f"{total_units_sold:,.0f}"
)

print(
    f"Overall Utilization   : "
    f"{overall_utilization:.2f}%"
)

# --------------------------------------------------------------
# 5. CATEGORY-WISE UTILIZATION
# --------------------------------------------------------------

category_summary = (
    df.groupby("Category")
    .agg(
        Total_Inventory=("Inventory Level", "sum"),
        Total_Units_Sold=("Units Sold", "sum"),
        Average_Inventory=("Inventory Level", "mean"),
        Average_Units_Sold=("Units Sold", "mean")
    )
    .reset_index()
)

category_summary["Utilization_Percentage"] = (
    category_summary["Total_Units_Sold"]
    / category_summary["Total_Inventory"]
) * 100

print("\n--- CATEGORY-WISE INVENTORY UTILIZATION ---")

print(
    category_summary.to_string(index=False)
)

# --------------------------------------------------------------
# 6. HIGHEST & LOWEST CATEGORY UTILIZATION
# --------------------------------------------------------------

highest_category = category_summary.loc[
    category_summary["Utilization_Percentage"].idxmax()
]

lowest_category = category_summary.loc[
    category_summary["Utilization_Percentage"].idxmin()
]

print("\n--- CATEGORY UTILIZATION EXTREMES ---")

print(
    f"Highest utilization : "
    f"{highest_category['Category']} "
    f"({highest_category['Utilization_Percentage']:.2f}%)"
)

print(
    f"Lowest utilization  : "
    f"{lowest_category['Category']} "
    f"({lowest_category['Utilization_Percentage']:.2f}%)"
)

# --------------------------------------------------------------
# 7. REGION-WISE UTILIZATION
# --------------------------------------------------------------

region_summary = (
    df.groupby("Region")
    .agg(
        Total_Inventory=("Inventory Level", "sum"),
        Total_Units_Sold=("Units Sold", "sum"),
        Average_Inventory=("Inventory Level", "mean"),
        Average_Units_Sold=("Units Sold", "mean")
    )
    .reset_index()
)

region_summary["Utilization_Percentage"] = (
    region_summary["Total_Units_Sold"]
    / region_summary["Total_Inventory"]
) * 100

print("\n--- REGION-WISE INVENTORY UTILIZATION ---")

print(
    region_summary.to_string(index=False)
)

# --------------------------------------------------------------
# 8. HIGHEST & LOWEST REGION UTILIZATION
# --------------------------------------------------------------

highest_region = region_summary.loc[
    region_summary["Utilization_Percentage"].idxmax()
]

lowest_region = region_summary.loc[
    region_summary["Utilization_Percentage"].idxmin()
]

print("\n--- REGION UTILIZATION EXTREMES ---")

print(
    f"Highest utilization : "
    f"{highest_region['Region']} "
    f"({highest_region['Utilization_Percentage']:.2f}%)"
)

print(
    f"Lowest utilization  : "
    f"{lowest_region['Region']} "
    f"({lowest_region['Utilization_Percentage']:.2f}%)"
)

# --------------------------------------------------------------
# 9. INTERPRETATION
# --------------------------------------------------------------

print("\n--- INTERPRETATION ---")

print(
    "Inventory utilization represents the relationship between "
    "units sold and available inventory level."
)

print(
    "A higher utilization ratio indicates that a larger amount "
    "of sales occurred relative to the recorded inventory level."
)

print(
    "A lower utilization ratio indicates comparatively lower "
    "sales relative to the recorded inventory level."
)

# --------------------------------------------------------------
# 10. CREATE CATEGORY UTILIZATION CHART
# --------------------------------------------------------------

plt.figure(figsize=(11, 7))

plt.bar(
    category_summary["Category"],
    category_summary["Utilization_Percentage"]
)

plt.title(
    "Inventory Utilization by Category",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel(
    "Category",
    fontsize=12
)

plt.ylabel(
    "Inventory Utilization (%)",
    fontsize=12
)

plt.xticks(
    rotation=30
)

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
    "Inventory utilization chart saved to:"
)

print(
    OUTPUT_FILE
)

print("\nEDA Step 13 completed successfully.")
print("=" * 70)