import pandas as pd

file_path = r"C:\SIM&DFS\data\raw\retail_store_inventory.csv"

df = pd.read_csv(file_path)

print("\n" + "=" * 60)
print("DATASET INSPECTION")
print("=" * 60)

print(f"\nRows: {df.shape[0]:,}")
print(f"Columns: {df.shape[1]}")

print("\n--- COLUMN NAMES ---")
for i, column in enumerate(df.columns, 1):
    print(f"{i}. {column}")

print("\n--- DATA TYPES ---")
print(df.dtypes.to_string())

print("\n--- MISSING VALUES ---")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)

missing_table = pd.DataFrame({
    "Missing": missing,
    "Missing %": missing_pct
})

print(missing_table.to_string())

print("\n--- DUPLICATES ---")
print(f"Duplicate rows: {df.duplicated().sum():,}")

print("\n--- DATE COLUMNS ---")
date_columns = []

for column in df.columns:
    if "date" in column.lower():
        date_columns.append(column)

if date_columns:
    for column in date_columns:
        converted = pd.to_datetime(df[column], errors="coerce")

        print(f"\nColumn: {column}")
        print(f"Valid dates: {converted.notna().sum():,}")
        print(f"Invalid dates: {converted.isna().sum():,}")

        if converted.notna().any():
            print(f"Minimum date: {converted.min()}")
            print(f"Maximum date: {converted.max()}")
else:
    print("No column containing 'date' was found.")

print("\n--- NUMERIC COLUMNS ---")
print(df.select_dtypes(include="number").columns.tolist())

print("\n--- FIRST 5 ROWS ---")
print(df.head().to_string())

print("\n--- BASIC STATISTICS ---")
print(df.describe(include="all").transpose().to_string())

print("\n" + "=" * 60)
print("INSPECTION COMPLETE")
print("=" * 60)