import pandas as pd

INPUT_FILE = r"C:\SIM&DFS\data\raw\retail_store_inventory.csv"

df = pd.read_csv(INPUT_FILE)

print("=" * 60)
print("DISCOUNT DATA ANALYSIS")
print("=" * 60)

print("\n--- DISCOUNT SUMMARY ---")
print(df["Discount"].describe())

print("\n--- UNIQUE DISCOUNT VALUES ---")
print(sorted(df["Discount"].unique())[:30])

print("\n--- MIN / MAX ---")
print(f"Minimum Discount: {df['Discount'].min()}")
print(f"Maximum Discount: {df['Discount'].max()}")

print("\n--- SAMPLE VALUES ---")
print(df["Discount"].head(20).to_list())

print("\n" + "=" * 60)