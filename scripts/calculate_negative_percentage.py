import pandas as pd

file_path = r"C:\SIM&DFS\data\processed\cleaned_inventory.csv"

df = pd.read_csv(file_path)

negative_count = (df["Demand Forecast"] < 0).sum()
total_count = len(df)

negative_percentage = (negative_count / total_count) * 100

print("=" * 60)
print("NEGATIVE DEMAND FORECAST PROPORTION")
print("=" * 60)

print(f"\nTotal records          : {total_count:,}")
print(f"Negative forecast     : {negative_count:,}")
print(f"Valid/non-negative    : {total_count - negative_count:,}")
print(f"Negative percentage   : {negative_percentage:.2f}%")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)