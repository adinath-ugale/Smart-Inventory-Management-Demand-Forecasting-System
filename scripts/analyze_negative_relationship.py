import pandas as pd

# Load cleaned dataset
file_path = r"C:\SIM&DFS\data\processed\cleaned_inventory.csv"
df = pd.read_csv(file_path)

# Select negative Demand Forecast records
negative_df = df[df["Demand Forecast"] < 0].copy()

print("=" * 60)
print("NEGATIVE DEMAND vs ACTUAL SALES ANALYSIS")
print("=" * 60)

print(f"\nNegative Demand Forecast records: {len(negative_df)}")

# ---------------------------------------------------------
# 1. Units Sold statistics
# ---------------------------------------------------------
print("\n--- Units Sold Statistics ---")
print(negative_df["Units Sold"].describe())

# ---------------------------------------------------------
# 2. Units Sold = 0 vs > 0
# ---------------------------------------------------------
zero_sales = (negative_df["Units Sold"] == 0).sum()
positive_sales = (negative_df["Units Sold"] > 0).sum()

print("\n--- Units Sold Distribution ---")
print(f"Units Sold = 0 : {zero_sales}")
print(f"Units Sold > 0 : {positive_sales}")

# ---------------------------------------------------------
# 3. Inventory statistics
# ---------------------------------------------------------
print("\n--- Inventory Level Statistics ---")
print(negative_df["Inventory Level"].describe())

# ---------------------------------------------------------
# 4. Units Ordered statistics
# ---------------------------------------------------------
print("\n--- Units Ordered Statistics ---")
print(negative_df["Units Ordered"].describe())

# ---------------------------------------------------------
# 5. Relationship summary
# ---------------------------------------------------------
print("\n--- Negative Forecast vs Sales Summary ---")

summary = negative_df[
    ["Demand Forecast", "Units Sold", "Inventory Level", "Units Ordered"]
].describe()

print(summary)

# ---------------------------------------------------------
# 6. Sample records
# ---------------------------------------------------------
print("\n--- Sample Records ---")

print(
    negative_df[
        [
            "Date",
            "Store ID",
            "Product ID",
            "Category",
            "Region",
            "Inventory Level",
            "Units Sold",
            "Units Ordered",
            "Demand Forecast",
        ]
    ].head(20).to_string(index=False)
)

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)