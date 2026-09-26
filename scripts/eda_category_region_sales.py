import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==============================================================
# SMART INVENTORY MANAGEMENT & DEMAND FORECASTING SYSTEM
# EDA STEP 12: CATEGORY-WISE REGIONAL SALES ANALYSIS
# ==============================================================

print("=" * 70)
print("SMART INVENTORY MANAGEMENT & DEMAND FORECASTING SYSTEM")
print("EDA STEP 12: CATEGORY-WISE REGIONAL SALES ANALYSIS")
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

OUTPUT_FILE = OUTPUT_DIR / "category_region_sales_heatmap.png"

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
# 4. CATEGORY × REGION SALES
# --------------------------------------------------------------

category_region = (
    df.groupby(
        ["Category", "Region"]
    )["Units Sold"]
    .sum()
    .reset_index()
)

# --------------------------------------------------------------
# 5. DISPLAY DETAILED SUMMARY
# --------------------------------------------------------------

print("\n--- CATEGORY × REGION SALES ---")

print(
    category_region.to_string(index=False)
)

# --------------------------------------------------------------
# 6. CREATE PIVOT TABLE
# --------------------------------------------------------------

sales_matrix = (
    category_region
    .pivot(
        index="Category",
        columns="Region",
        values="Units Sold"
    )
    .fillna(0)
)

print("\n--- CATEGORY × REGION SALES MATRIX ---")

print(
    sales_matrix.to_string()
)

# --------------------------------------------------------------
# 7. HIGHEST CATEGORY FOR EACH REGION
# --------------------------------------------------------------

print("\n--- HIGHEST-SELLING CATEGORY BY REGION ---")

for region in sales_matrix.columns:

    category = sales_matrix[region].idxmax()
    sales = sales_matrix[region].max()

    print(
        f"{region:<10}: "
        f"{category} "
        f"({sales:,.0f} units)"
    )

# --------------------------------------------------------------
# 8. LOWEST CATEGORY FOR EACH REGION
# --------------------------------------------------------------

print("\n--- LOWEST-SELLING CATEGORY BY REGION ---")

for region in sales_matrix.columns:

    category = sales_matrix[region].idxmin()
    sales = sales_matrix[region].min()

    print(
        f"{region:<10}: "
        f"{category} "
        f"({sales:,.0f} units)"
    )

# --------------------------------------------------------------
# 9. HIGHEST CATEGORY-REGION COMBINATION
# --------------------------------------------------------------

highest_combination = category_region.loc[
    category_region["Units Sold"].idxmax()
]

print("\n--- HIGHEST CATEGORY-REGION COMBINATION ---")

print(
    f"Category : {highest_combination['Category']}"
)

print(
    f"Region   : {highest_combination['Region']}"
)

print(
    f"Units Sold: "
    f"{highest_combination['Units Sold']:,.0f}"
)

# --------------------------------------------------------------
# 10. LOWEST CATEGORY-REGION COMBINATION
# --------------------------------------------------------------

lowest_combination = category_region.loc[
    category_region["Units Sold"].idxmin()
]

print("\n--- LOWEST CATEGORY-REGION COMBINATION ---")

print(
    f"Category : {lowest_combination['Category']}"
)

print(
    f"Region   : {lowest_combination['Region']}"
)

print(
    f"Units Sold: "
    f"{lowest_combination['Units Sold']:,.0f}"
)

# --------------------------------------------------------------
# 11. CREATE HEATMAP
# --------------------------------------------------------------

plt.figure(figsize=(12, 7))

plt.imshow(
    sales_matrix.values,
    aspect="auto"
)

plt.title(
    "Category × Region Sales Heatmap",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel(
    "Region",
    fontsize=12
)

plt.ylabel(
    "Category",
    fontsize=12
)

plt.xticks(
    range(len(sales_matrix.columns)),
    sales_matrix.columns
)

plt.yticks(
    range(len(sales_matrix.index)),
    sales_matrix.index
)

# Add values inside cells
for i in range(len(sales_matrix.index)):

    for j in range(len(sales_matrix.columns)):

        value = sales_matrix.iloc[i, j]

        plt.text(
            j,
            i,
            f"{value:,.0f}",
            ha="center",
            va="center",
            fontsize=9
        )

plt.colorbar(
    label="Units Sold"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# --------------------------------------------------------------
# 12. OUTPUT
# --------------------------------------------------------------

print("\n--- OUTPUT ---")

print(
    "Category × Region heatmap saved to:"
)

print(
    OUTPUT_FILE
)

print("\nEDA Step 12 completed successfully.")
print("=" * 70)