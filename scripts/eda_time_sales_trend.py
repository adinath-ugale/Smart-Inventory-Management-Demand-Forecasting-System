import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==============================================================
# SMART INVENTORY MANAGEMENT & DEMAND FORECASTING SYSTEM
# EDA STEP 10: TIME-BASED SALES & DEMAND TREND ANALYSIS
# ==============================================================

print("=" * 70)
print("SMART INVENTORY MANAGEMENT & DEMAND FORECASTING SYSTEM")
print("EDA STEP 10: TIME-BASED SALES & DEMAND TREND ANALYSIS")
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

OUTPUT_FILE = OUTPUT_DIR / "monthly_sales_demand_trend.png"

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
    "Date",
    "Units Sold",
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
# 4. CONVERT DATE COLUMN
# --------------------------------------------------------------

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)

invalid_dates = df["Date"].isna().sum()

if invalid_dates > 0:
    print(f"\nWarning: {invalid_dates} invalid dates found.")
    df = df.dropna(subset=["Date"])

# --------------------------------------------------------------
# 5. CREATE MONTH COLUMN
# --------------------------------------------------------------

df["Month"] = df["Date"].dt.to_period("M")

# --------------------------------------------------------------
# 6. MONTHLY SALES & FORECAST AGGREGATION
# --------------------------------------------------------------

monthly = (
    df.groupby("Month")
    .agg(
        Actual_Sales=("Units Sold", "sum"),
        Forecast_Demand=("Demand Forecast", "sum")
    )
    .reset_index()
)

# Convert Period to timestamp for plotting
monthly["Month_Date"] = monthly["Month"].dt.to_timestamp()

# --------------------------------------------------------------
# 7. FORECAST DIFFERENCE
# --------------------------------------------------------------

monthly["Forecast_Difference"] = (
    monthly["Forecast_Demand"]
    - monthly["Actual_Sales"]
)

monthly["Forecast_Error_Percentage"] = (
    monthly["Forecast_Difference"]
    / monthly["Actual_Sales"].replace(0, pd.NA)
) * 100

# --------------------------------------------------------------
# 8. MONTHLY SALES SUMMARY
# --------------------------------------------------------------

print("\n--- MONTHLY SALES & DEMAND FORECAST ---")

display_table = monthly[
    [
        "Month",
        "Actual_Sales",
        "Forecast_Demand",
        "Forecast_Difference"
    ]
].copy()

print(
    display_table.to_string(index=False)
)

# --------------------------------------------------------------
# 9. HIGHEST & LOWEST SALES MONTH
# --------------------------------------------------------------

highest_sales_row = monthly.loc[
    monthly["Actual_Sales"].idxmax()
]

lowest_sales_row = monthly.loc[
    monthly["Actual_Sales"].idxmin()
]

print("\n--- SALES TREND SUMMARY ---")

print(
    f"Highest-sales month : "
    f"{highest_sales_row['Month']} "
    f"({highest_sales_row['Actual_Sales']:,} units)"
)

print(
    f"Lowest-sales month  : "
    f"{lowest_sales_row['Month']} "
    f"({lowest_sales_row['Actual_Sales']:,} units)"
)

# --------------------------------------------------------------
# 10. TOTAL SALES & FORECAST
# --------------------------------------------------------------

total_actual = monthly["Actual_Sales"].sum()
total_forecast = monthly["Forecast_Demand"].sum()

print("\n--- TOTALS ---")

print(
    f"Total Actual Sales    : "
    f"{total_actual:,.0f}"
)

print(
    f"Total Forecast Demand : "
    f"{total_forecast:,.0f}"
)

# --------------------------------------------------------------
# 11. TREND DIRECTION
# --------------------------------------------------------------

first_month_sales = monthly.iloc[0]["Actual_Sales"]
last_month_sales = monthly.iloc[-1]["Actual_Sales"]

if last_month_sales > first_month_sales:
    trend_direction = "increasing"
elif last_month_sales < first_month_sales:
    trend_direction = "decreasing"
else:
    trend_direction = "stable"

print("\n--- OVERALL TREND ---")

print(
    f"Sales trend from first to last month: "
    f"{trend_direction.upper()}"
)

print(
    f"First month sales : {first_month_sales:,.0f}"
)

print(
    f"Last month sales  : {last_month_sales:,.0f}"
)

# --------------------------------------------------------------
# 12. MONTHLY FORECAST BIAS
# --------------------------------------------------------------

overestimated_months = (
    monthly["Forecast_Difference"] > 0
).sum()

underestimated_months = (
    monthly["Forecast_Difference"] < 0
).sum()

accurate_months = (
    monthly["Forecast_Difference"] == 0
).sum()

print("\n--- MONTHLY FORECAST BIAS ---")

print(
    f"Months with forecast overestimation  : "
    f"{overestimated_months}"
)

print(
    f"Months with forecast underestimation  : "
    f"{underestimated_months}"
)

print(
    f"Months with exact forecast            : "
    f"{accurate_months}"
)

# --------------------------------------------------------------
# 13. INTERPRETATION
# --------------------------------------------------------------

print("\n--- INTERPRETATION ---")

if trend_direction == "increasing":
    print(
        "Actual sales show an increasing trend from "
        "the first month to the last month."
    )

elif trend_direction == "decreasing":
    print(
        "Actual sales show a decreasing trend from "
        "the first month to the last month."
    )

else:
    print(
        "Actual sales remain relatively stable from "
        "the first month to the last month."
    )

if total_forecast > total_actual:
    print(
        "Across the complete dataset, forecast demand "
        "is higher than actual sales."
    )

elif total_forecast < total_actual:
    print(
        "Across the complete dataset, forecast demand "
        "is lower than actual sales."
    )

else:
    print(
        "Across the complete dataset, forecast demand "
        "matches actual sales."
    )

# --------------------------------------------------------------
# 14. CREATE CHART
# --------------------------------------------------------------

plt.figure(figsize=(14, 7))

plt.plot(
    monthly["Month_Date"],
    monthly["Actual_Sales"],
    marker="o",
    linewidth=2,
    label="Actual Sales"
)

plt.plot(
    monthly["Month_Date"],
    monthly["Forecast_Demand"],
    marker="o",
    linewidth=2,
    label="Demand Forecast"
)

plt.title(
    "Monthly Actual Sales vs Demand Forecast",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel(
    "Month",
    fontsize=12
)

plt.ylabel(
    "Units",
    fontsize=12
)

plt.grid(
    True,
    linestyle="--",
    alpha=0.5
)

plt.legend()

plt.xticks(
    rotation=45
)

plt.tight_layout()

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# --------------------------------------------------------------
# 15. OUTPUT
# --------------------------------------------------------------

print("\n--- OUTPUT ---")

print(
    "Monthly sales trend chart saved to:"
)

print(
    OUTPUT_FILE
)

print("\nEDA Step 10 completed successfully.")
print("=" * 70)