import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==============================================================
# SMART INVENTORY MANAGEMENT & DEMAND FORECASTING SYSTEM
# EDA STEP 14: UNITS ORDERED VS UNITS SOLD ANALYSIS
# ==============================================================

print("=" * 70)
print("SMART INVENTORY MANAGEMENT & DEMAND FORECASTING SYSTEM")
print("EDA STEP 14: UNITS ORDERED VS UNITS SOLD ANALYSIS")
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

OUTPUT_FILE = OUTPUT_DIR / "ordered_vs_sold.png"

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
    "Units Ordered",
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
# 4. OVERALL ANALYSIS
# --------------------------------------------------------------

total_ordered = df["Units Ordered"].sum()
total_sold = df["Units Sold"].sum()

order_sales_difference = (
    total_ordered - total_sold
)

if total_sold > 0:

    order_sales_ratio = (
        total_ordered / total_sold
    )

else:

    order_sales_ratio = 0

if total_sold > 0:

    excess_order_percentage = (
        order_sales_difference / total_sold
    ) * 100

else:

    excess_order_percentage = 0

print("\n--- OVERALL ORDER VS SALES ---")

print(
    f"Total Units Ordered : "
    f"{total_ordered:,.0f}"
)

print(
    f"Total Units Sold    : "
    f"{total_sold:,.0f}"
)

print(
    f"Order-Sales Difference : "
    f"{order_sales_difference:,.0f}"
)

print(
    f"Order-to-Sales Ratio : "
    f"{order_sales_ratio:.2f}"
)

print(
    f"Difference as % of Sales : "
    f"{excess_order_percentage:.2f}%"
)

# --------------------------------------------------------------
# 5. CATEGORY-WISE ANALYSIS
# --------------------------------------------------------------

category_summary = (
    df.groupby("Category")
    .agg(
        Total_Ordered=("Units Ordered", "sum"),
        Total_Sold=("Units Sold", "sum")
    )
    .reset_index()
)

category_summary["Order_Sales_Difference"] = (
    category_summary["Total_Ordered"]
    - category_summary["Total_Sold"]
)

category_summary["Order_to_Sales_Ratio"] = (
    category_summary["Total_Ordered"]
    / category_summary["Total_Sold"]
)

print("\n--- CATEGORY-WISE ORDER VS SALES ---")

print(
    category_summary.to_string(index=False)
)

# --------------------------------------------------------------
# 6. HIGHEST CATEGORY DIFFERENCE
# --------------------------------------------------------------

highest_category_difference = category_summary.loc[
    category_summary["Order_Sales_Difference"].idxmax()
]

lowest_category_difference = category_summary.loc[
    category_summary["Order_Sales_Difference"].idxmin()
]

print("\n--- CATEGORY ORDER DIFFERENCE EXTREMES ---")

print(
    f"Highest difference : "
    f"{highest_category_difference['Category']} "
    f"({highest_category_difference['Order_Sales_Difference']:,.0f} units)"
)

print(
    f"Lowest difference  : "
    f"{lowest_category_difference['Category']} "
    f"({lowest_category_difference['Order_Sales_Difference']:,.0f} units)"
)

# --------------------------------------------------------------
# 7. REGION-WISE ANALYSIS
# --------------------------------------------------------------

region_summary = (
    df.groupby("Region")
    .agg(
        Total_Ordered=("Units Ordered", "sum"),
        Total_Sold=("Units Sold", "sum")
    )
    .reset_index()
)

region_summary["Order_Sales_Difference"] = (
    region_summary["Total_Ordered"]
    - region_summary["Total_Sold"]
)

region_summary["Order_to_Sales_Ratio"] = (
    region_summary["Total_Ordered"]
    / region_summary["Total_Sold"]
)

print("\n--- REGION-WISE ORDER VS SALES ---")

print(
    region_summary.to_string(index=False)
)

# --------------------------------------------------------------
# 8. REGION DIFFERENCE EXTREMES
# --------------------------------------------------------------

highest_region_difference = region_summary.loc[
    region_summary["Order_Sales_Difference"].idxmax()
]

lowest_region_difference = region_summary.loc[
    region_summary["Order_Sales_Difference"].idxmin()
]

print("\n--- REGION ORDER DIFFERENCE EXTREMES ---")

print(
    f"Highest difference : "
    f"{highest_region_difference['Region']} "
    f"({highest_region_difference['Order_Sales_Difference']:,.0f} units)"
)

print(
    f"Lowest difference  : "
    f"{lowest_region_difference['Region']} "
    f"({lowest_region_difference['Order_Sales_Difference']:,.0f} units)"
)

# --------------------------------------------------------------
# 9. INTERPRETATION
# --------------------------------------------------------------

print("\n--- INTERPRETATION ---")

if total_ordered > total_sold:

    print(
        "Total units ordered are higher than total units sold."
    )

elif total_ordered < total_sold:

    print(
        "Total units ordered are lower than total units sold."
    )

else:

    print(
        "Total units ordered are equal to total units sold."
    )

print(
    "The difference between ordered and sold units can help "
    "identify potential inventory accumulation or supply gaps."
)

# --------------------------------------------------------------
# 10. CREATE CHART
# --------------------------------------------------------------

plt.figure(figsize=(11, 7))

x = range(len(category_summary))

bar_width = 0.35

plt.bar(
    [i - bar_width / 2 for i in x],
    category_summary["Total_Ordered"],
    width=bar_width,
    label="Units Ordered"
)

plt.bar(
    [i + bar_width / 2 for i in x],
    category_summary["Total_Sold"],
    width=bar_width,
    label="Units Sold"
)

plt.title(
    "Units Ordered vs Units Sold by Category",
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
    "Ordered vs Sold chart saved to:"
)

print(
    OUTPUT_FILE
)

print("\nEDA Step 14 completed successfully.")
print("=" * 70)