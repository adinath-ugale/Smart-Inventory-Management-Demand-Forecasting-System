import pandas as pd

INPUT_FILE = "C:/SIM&DFS/data/raw/retail_store_inventory.csv"

df = pd.read_csv(INPUT_FILE)

# Find negative demand forecast records
negative_demand = df[df["Demand Forecast"] < 0]

print("=" * 60)
print("NEGATIVE DEMAND FORECAST ANALYSIS")
print("=" * 60)

print("\nNegative records:", len(negative_demand))

print("\n--- Negative Demand Statistics ---")

print(
    negative_demand["Demand Forecast"].describe()
)

print("\n--- Sample Negative Records ---")

print(
    negative_demand[
        [
            "Date",
            "Store ID",
            "Product ID",
            "Category",
            "Region",
            "Inventory Level",
            "Units Sold",
            "Units Ordered",
            "Demand Forecast"
        ]
    ].head(20).to_string(index=False)
)

print("\n--- Demand Forecast Values ---")

print(
    negative_demand["Demand Forecast"]
    .value_counts()
    .sort_index()
    .head(30)
)

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)