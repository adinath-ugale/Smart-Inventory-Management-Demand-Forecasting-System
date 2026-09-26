import pandas as pd

INPUT_FILE = r"C:\SIM&DFS\data\raw\retail_store_inventory.csv"
OUTPUT_FILE = r"C:\SIM&DFS\data\processed\cleaned_inventory.csv"

print("=" * 60)
print("SMART INVENTORY DATA CLEANING")
print("=" * 60)

# 1. Load dataset
df = pd.read_csv(INPUT_FILE)

print(f"\nOriginal rows: {len(df):,}")
print(f"Original columns: {len(df.columns)}")

# 2. Convert Date
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# 3. Check invalid dates
invalid_dates = df["Date"].isna().sum()
print(f"Invalid dates: {invalid_dates}")

# 4. Remove exact duplicates
duplicates = df.duplicated().sum()
print(f"Duplicate rows: {duplicates}")

if duplicates > 0:
    df = df.drop_duplicates()

# 5. Check missing values
missing_total = df.isna().sum().sum()
print(f"Missing values: {missing_total}")

# 6. Validate numerical columns
numeric_columns = [
    "Inventory Level",
    "Units Sold",
    "Units Ordered",
    "Demand Forecast",
    "Price",
    "Discount",
    "Holiday/Promotion",
    "Competitor Pricing"
]

print("\n--- NEGATIVE VALUES CHECK ---")

for column in numeric_columns:
    negative_count = (df[column] < 0).sum()
    print(f"{column}: {negative_count}")

# 7. Demand Forecast validation
negative_demand = (df["Demand Forecast"] < 0).sum()

print(f"\nNegative Demand Forecast records: {negative_demand}")

# Negative demand forecasts are not logically valid.
# Convert them to NaN so they are treated as invalid/missing values
# instead of incorrectly treating them as zero demand.
if negative_demand > 0:
    df.loc[df["Demand Forecast"] < 0, "Demand Forecast"] = pd.NA

print(
    f"Negative Demand Forecast after cleaning: "
    f"{(df['Demand Forecast'] < 0).sum()}"
)

print(
    f"Missing Demand Forecast after cleaning: "
    f"{df['Demand Forecast'].isna().sum()}"
)

# 8. Final logical data validation

print("\n--- FINAL LOGICAL VALIDATION ---")

# Numeric business constraints
negative_inventory = (df["Inventory Level"] < 0).sum()
negative_units_sold = (df["Units Sold"] < 0).sum()
negative_units_ordered = (df["Units Ordered"] < 0).sum()
non_positive_price = (df["Price"] <= 0).sum()
negative_competitor_price = (df["Competitor Pricing"] <= 0).sum()

# Discount range validation
discount_below_zero = (df["Discount"] < 0).sum()
discount_above_20 = (df["Discount"] > 20).sum()

print(f"Negative Inventory Level: {negative_inventory}")
print(f"Negative Units Sold: {negative_units_sold}")
print(f"Negative Units Ordered: {negative_units_ordered}")
print(f"Non-positive Price: {non_positive_price}")
print(f"Negative/zero Competitor Pricing: {negative_competitor_price}")
print(f"Discount below 0: {discount_below_zero}")
print(f"Discount above 20%: {discount_above_20}")

# Expected categorical values
expected_categories = {
    "Category": {
        "Groceries",
        "Toys",
        "Electronics",
        "Furniture",
        "Clothing"
    },
    "Region": {
        "North",
        "South",
        "East",
        "West"
    },
    "Weather Condition": {
        "Sunny",
        "Cloudy",
        "Rainy",
        "Snowy"
    },
    "Seasonality": {
        "Winter",
        "Spring",
        "Summer",
        "Autumn"
    }
}

print("\n--- CATEGORICAL VALUE CHECK ---")

for column, expected_values in expected_categories.items():
    actual_values = set(df[column].dropna().unique())
    unexpected_values = actual_values - expected_values

    print(f"\n{column}:")
    print(f"Unique values: {sorted(actual_values)}")
    print(f"Unexpected values: {sorted(unexpected_values)}")

# 9. Save cleaned dataset
df.to_csv(OUTPUT_FILE, index=False)

print("\n--- CLEANING RESULT ---")
print(f"Final rows: {len(df):,}")
print(f"Final columns: {len(df.columns)}")
print(f"Saved to: {OUTPUT_FILE}")

print("\n" + "=" * 60)
print("DATA CLEANING COMPLETE")
print("=" * 60)