import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==============================================================
# SMART INVENTORY MANAGEMENT & DEMAND FORECASTING SYSTEM
# EDA STEP 11: DAY-OF-WEEK SALES PATTERN ANALYSIS
# ==============================================================

print("=" * 70)
print("SMART INVENTORY MANAGEMENT & DEMAND FORECASTING SYSTEM")
print("EDA STEP 11: DAY-OF-WEEK SALES PATTERN ANALYSIS")
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

OUTPUT_FILE = OUTPUT_DIR / "weekday_sales_pattern.png"

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
# 4. CONVERT DATE
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
# 5. CREATE WEEKDAY INFORMATION
# --------------------------------------------------------------

df["Weekday"] = df["Date"].dt.day_name()

weekday_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

df["Weekday"] = pd.Categorical(
    df["Weekday"],
    categories=weekday_order,
    ordered=True
)

# --------------------------------------------------------------
# 6. WEEKDAY SALES ANALYSIS
# --------------------------------------------------------------

weekday_summary = (
    df.groupby(
        "Weekday",
        observed=False
    )
    .agg(
        Total_Units_Sold=("Units Sold", "sum"),
        Average_Units_Sold=("Units Sold", "mean"),
        Total_Forecast=("Demand Forecast", "sum"),
        Average_Forecast=("Demand Forecast", "mean")
    )
    .reset_index()
)

# --------------------------------------------------------------
# 7. FORECAST DIFFERENCE
# --------------------------------------------------------------

weekday_summary["Forecast_Difference"] = (
    weekday_summary["Total_Forecast"]
    - weekday_summary["Total_Units_Sold"]
)

# --------------------------------------------------------------
# 8. DISPLAY SUMMARY
# --------------------------------------------------------------

print("\n--- WEEKDAY SALES SUMMARY ---")

print(
    weekday_summary.to_string(index=False)
)

# --------------------------------------------------------------
# 9. HIGHEST AND LOWEST SALES WEEKDAY
# --------------------------------------------------------------

highest_sales = weekday_summary.loc[
    weekday_summary["Total_Units_Sold"].idxmax()
]

lowest_sales = weekday_summary.loc[
    weekday_summary["Total_Units_Sold"].idxmin()
]

print("\n--- WEEKDAY SALES EXTREMES ---")

print(
    f"Highest-sales weekday : "
    f"{highest_sales['Weekday']} "
    f"({highest_sales['Total_Units_Sold']:,.0f} units)"
)

print(
    f"Lowest-sales weekday  : "
    f"{lowest_sales['Weekday']} "
    f"({lowest_sales['Total_Units_Sold']:,.0f} units)"
)

# --------------------------------------------------------------
# 10. AVERAGE SALES EXTREMES
# --------------------------------------------------------------

highest_average = weekday_summary.loc[
    weekday_summary["Average_Units_Sold"].idxmax()
]

lowest_average = weekday_summary.loc[
    weekday_summary["Average_Units_Sold"].idxmin()
]

print("\n--- AVERAGE DAILY SALES EXTREMES ---")

print(
    f"Highest average sales : "
    f"{highest_average['Weekday']} "
    f"({highest_average['Average_Units_Sold']:.2f} units/record)"
)

print(
    f"Lowest average sales  : "
    f"{lowest_average['Weekday']} "
    f"({lowest_average['Average_Units_Sold']:.2f} units/record)"
)

# --------------------------------------------------------------
# 11. FORECAST BIAS BY WEEKDAY
# --------------------------------------------------------------

print("\n--- WEEKDAY FORECAST DIFFERENCE ---")

for _, row in weekday_summary.iterrows():

    difference = row["Forecast_Difference"]

    if difference > 0:
        interpretation = "Overestimated"
    elif difference < 0:
        interpretation = "Underestimated"
    else:
        interpretation = "Exact"

    print(
        f"{row['Weekday']:<10} : "
        f"{difference:,.0f} units "
        f"({interpretation})"
    )

# --------------------------------------------------------------
# 12. OVERALL WEEKDAY PATTERN
# --------------------------------------------------------------

max_avg = weekday_summary["Average_Units_Sold"].max()
min_avg = weekday_summary["Average_Units_Sold"].min()

weekday_variation = (
    ((max_avg - min_avg) / min_avg) * 100
)

print("\n--- WEEKDAY PATTERN ---")

print(
    f"Difference between highest and lowest "
    f"average weekday sales: {weekday_variation:.2f}%"
)

if weekday_variation < 5:
    print(
        "Sales are relatively consistent across "
        "the days of the week."
    )

elif weekday_variation < 15:
    print(
        "Sales show a moderate variation across "
        "the days of the week."
    )

else:
    print(
        "Sales show a relatively high variation "
        "across the days of the week."
    )

# --------------------------------------------------------------
# 13. CREATE CHART
# --------------------------------------------------------------

plt.figure(figsize=(12, 7))

plt.bar(
    weekday_summary["Weekday"],
    weekday_summary["Total_Units_Sold"]
)

plt.title(
    "Total Units Sold by Day of Week",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel(
    "Day of Week",
    fontsize=12
)

plt.ylabel(
    "Total Units Sold",
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
# 14. OUTPUT
# --------------------------------------------------------------

print("\n--- OUTPUT ---")

print(
    "Weekday sales pattern chart saved to:"
)

print(
    OUTPUT_FILE
)

print("\nEDA Step 11 completed successfully.")
print("=" * 70)