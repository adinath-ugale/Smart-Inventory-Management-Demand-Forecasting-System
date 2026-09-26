from pathlib import Path
import pandas as pd


# ============================================================
# SMART INVENTORY MANAGEMENT & DEMAND FORECASTING SYSTEM
# EDA STEP 1: DATASET OVERVIEW & DESCRIPTIVE STATISTICS
# ============================================================

# Project paths
PROJECT_ROOT = Path(r"C:\SIM&DFS")
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_inventory.csv"


def load_dataset(file_path: Path) -> pd.DataFrame:
    """Load the cleaned inventory dataset."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {file_path}"
        )

    df = pd.read_csv(file_path)

    # Convert Date column to datetime for reliable date analysis
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    return df


def print_section(title: str) -> None:
    """Print a consistent section heading."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    print("=" * 70)
    print("SMART INVENTORY MANAGEMENT & DEMAND FORECASTING SYSTEM")
    print("EDA STEP 1: DATASET OVERVIEW & DESCRIPTIVE STATISTICS")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------------
    df = load_dataset(DATA_PATH)

    # --------------------------------------------------------
    # 2. Dataset shape
    # --------------------------------------------------------
    print_section("1. DATASET SHAPE")

    rows, columns = df.shape

    print(f"Rows    : {rows:,}")
    print(f"Columns : {columns:,}")

    # --------------------------------------------------------
    # 3. Column names
    # --------------------------------------------------------
    print_section("2. COLUMN NAMES")

    for index, column in enumerate(df.columns, start=1):
        print(f"{index:2}. {column}")

    # --------------------------------------------------------
    # 4. Data types
    # --------------------------------------------------------
    print_section("3. DATA TYPES")

    dtype_summary = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values
    })

    print(dtype_summary.to_string(index=False))

    # --------------------------------------------------------
    # 5. Date range
    # --------------------------------------------------------
    print_section("4. DATE RANGE")

    if "Date" in df.columns:
        valid_dates = df["Date"].dropna()

        if not valid_dates.empty:
            print(f"Minimum Date : {valid_dates.min().date()}")
            print(f"Maximum Date : {valid_dates.max().date()}")
            print(f"Date Records : {len(valid_dates):,}")
        else:
            print("No valid dates found.")

    # --------------------------------------------------------
    # 6. Unique counts
    # --------------------------------------------------------
    print_section("5. UNIQUE VALUE COUNTS")

    for column in df.columns:
        print(f"{column:<25}: {df[column].nunique(dropna=True):,}")

    # --------------------------------------------------------
    # 7. Categorical unique values
    # --------------------------------------------------------
    categorical_columns = [
        "Store ID",
        "Product ID",
        "Category",
        "Region",
        "Weather Condition",
        "Holiday/Promotion",
        "Seasonality"
    ]

    print_section("6. CATEGORICAL UNIQUE VALUES")

    for column in categorical_columns:
        if column not in df.columns:
            continue

        print(f"\n--- {column} ---")

        values = df[column].dropna().unique()

        for value in sorted(values, key=lambda x: str(x)):
            count = (df[column] == value).sum()
            print(f"{str(value):<25} : {count:,}")

    # --------------------------------------------------------
    # 8. Descriptive statistics
    # --------------------------------------------------------
    print_section("7. NUMERIC DESCRIPTIVE STATISTICS")

    numeric_df = df.select_dtypes(include="number")

    if numeric_df.empty:
        print("No numeric columns found.")
    else:
        statistics = numeric_df.describe().T

        statistics = statistics[
            ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
        ]

        print(statistics.to_string(float_format=lambda x: f"{x:,.2f}"))

    # --------------------------------------------------------
    # 9. Missing values
    # --------------------------------------------------------
    print_section("8. MISSING VALUE COUNTS")

    missing_counts = df.isna().sum()

    missing_summary = pd.DataFrame({
        "Column": df.columns,
        "Missing Values": missing_counts.values,
        "Missing %": (missing_counts.values / len(df)) * 100
    })

    print(
        missing_summary.to_string(
            index=False,
            formatters={
                "Missing %": lambda x: f"{x:.2f}%"
            }
        )
    )

    # --------------------------------------------------------
    # 10. Concise EDA summary
    # --------------------------------------------------------
    print_section("9. CONCISE EDA SUMMARY")

    print(f"Dataset contains {rows:,} rows and {columns} columns.")

    if "Date" in df.columns and not df["Date"].dropna().empty:
        print(
            f"Data covers the period from "
            f"{df['Date'].min().date()} to {df['Date'].max().date()}."
        )

    print(
        f"Numeric columns analyzed: {len(numeric_df)}"
    )

    total_missing = int(df.isna().sum().sum())

    print(
        f"Total missing values in cleaned dataset: {total_missing:,}"
    )

    if "Demand Forecast" in df.columns:
        demand_missing = int(df["Demand Forecast"].isna().sum())

        print(
            f"Missing Demand Forecast values: {demand_missing:,}"
        )

    print(
        "\nNo charts or dataset modifications are performed in this step."
    )

    print("\nEDA Step 1 completed successfully.")


if __name__ == "__main__":
    main()