import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path("C:/SIM&DFS")

INPUT_FILE = BASE_DIR / "data/raw/retail_store_inventory.csv"
OUTPUT_DIR = BASE_DIR / "data/processed"
OUTPUT_FILE = OUTPUT_DIR / "cleaned_inventory_data.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("DAY 2 - DATA CLEANING")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print(f"\nOriginal shape: {df.shape}")


# ============================================================
# 1. MISSING VALUES
# ============================================================

print("\n--- MISSING VALUES ---")

missing = df.isnull().sum()

print(missing)

print("\nTotal missing values:", missing.sum())


# ============================================================
# 2. DUPLICATES
# ============================================================

print("\n--- DUPLICATES ---")

duplicates = df.duplicated().sum()

print("Duplicate rows:", duplicates)

if duplicates > 0:
    df = df.drop_duplicates()
    print("Duplicates removed.")

else:
    print("No duplicates found.")


# ============================================================
# 3. DATE CONVERSION
# ============================================================

print("\n--- DATE CONVERSION ---")

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

print("Date datatype:", df["Date"].dtype)

print("Minimum date:", df["Date"].min())

print("Maximum date:", df["Date"].max())

print("Invalid dates:", df["Date"].isna().sum())


# ============================================================
# 4. CHECK NEGATIVE VALUES
# ============================================================

print("\n--- NEGATIVE VALUES ---")

numeric_columns = df.select_dtypes(include="number").columns

negative_counts = (df[numeric_columns] < 0).sum()

print(negative_counts)


# ============================================================
# 5. OUTLIER DETECTION USING IQR
# ============================================================

print("\n--- OUTLIER DETECTION ---")

outlier_summary = {}

for column in numeric_columns:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = ((df[column] < lower_bound) |
                (df[column] > upper_bound)).sum()

    outlier_summary[column] = outliers

    print(f"{column}: {outliers} outliers")


# ============================================================
# 6. FEATURE ENGINEERING
# ============================================================

print("\n--- FEATURE ENGINEERING ---")

df["Year"] = df["Date"].dt.year

df["Month"] = df["Date"].dt.month

df["Day"] = df["Date"].dt.day

df["DayOfWeek"] = df["Date"].dt.dayofweek

df["IsWeekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)

df["Quarter"] = df["Date"].dt.quarter

print("Created features:")
print("- Year")
print("- Month")
print("- Day")
print("- DayOfWeek")
print("- IsWeekend")
print("- Quarter")

# ============================================================
# 7. INVENTORY METRICS
# ============================================================

print("\n--- INVENTORY METRICS ---")

# Difference between current inventory and units sold
df["Stock_Gap"] = (
    df["Inventory Level"] - df["Units Sold"]
)

# Estimated inventory remaining after sales
df["Estimated_End_Inventory"] = (
    df["Inventory Level"] - df["Units Sold"]
)

# Difference between forecasted demand and current inventory
df["Reorder_Gap"] = (
    df["Demand Forecast"] - df["Inventory Level"]
)

# Difference between actual sales and forecasted demand
df["Forecast_Error"] = (
    df["Units Sold"] - df["Demand Forecast"]
)

print("Created inventory metrics:")
print("- Stock_Gap")
print("- Estimated_End_Inventory")
print("- Reorder_Gap")
print("- Forecast_Error")

# ============================================================
# 7. SORT DATA
# ============================================================

df = df.sort_values(
    by=["Store ID", "Product ID", "Date"]
).reset_index(drop=True)


# ============================================================
# 8. SAVE CLEAN DATASET
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df.to_csv(OUTPUT_FILE, index=False)

print("\n--- FINAL DATASET ---")

print("Final shape:", df.shape)

print("Output file:", OUTPUT_FILE)

print("\n" + "=" * 60)
print("DATA CLEANING COMPLETE")
print("=" * 60)