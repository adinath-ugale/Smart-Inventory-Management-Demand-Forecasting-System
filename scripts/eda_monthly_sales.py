from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# SMART INVENTORY MANAGEMENT & DEMAND FORECASTING SYSTEM
# EDA STEP 2: MONTHLY UNITS SOLD TREND
# ============================================================

PROJECT_ROOT = Path(r"C:\SIM&DFS")

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cleaned_inventory.csv"
)

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
OUTPUT_PATH = FIGURES_DIR / "monthly_units_sold_trend.png"


def load_dataset(file_path: Path) -> pd.DataFrame:
    """Load the cleaned inventory dataset."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {file_path}"
        )

    df = pd.read_csv(file_path)

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    return df


def create_monthly_sales_summary(
    df: pd.DataFrame
) -> pd.DataFrame:
    """Aggregate total Units Sold by month."""

    monthly_sales = (
        df.set_index("Date")
        .resample("MS")["Units Sold"]
        .sum()
        .reset_index()
    )

    return monthly_sales


def create_chart(
    monthly_sales: pd.DataFrame,
    output_path: Path
) -> None:
    """Create and save the monthly Units Sold trend chart."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure(figsize=(12, 6))

    plt.plot(
        monthly_sales["Date"],
        monthly_sales["Units Sold"],
        linewidth=2
    )

    plt.title(
        "Monthly Units Sold Trend",
        fontsize=16,
        fontweight="bold"
    )

    plt.xlabel("Month")
    plt.ylabel("Total Units Sold")

    plt.grid(
        True,
        alpha=0.3
    )

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def main() -> None:
    print("=" * 70)
    print("SMART INVENTORY MANAGEMENT & DEMAND FORECASTING SYSTEM")
    print("EDA STEP 2: MONTHLY UNITS SOLD TREND")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------------
    df = load_dataset(DATA_PATH)

    print("\nDataset loaded successfully.")
    print(f"Rows: {len(df):,}")

    # --------------------------------------------------------
    # 2. Create monthly aggregation
    # --------------------------------------------------------
    monthly_sales = create_monthly_sales_summary(df)

    print("\nMonthly aggregation created.")
    print(f"Number of months: {len(monthly_sales):,}")

    # --------------------------------------------------------
    # 3. Display monthly summary
    # --------------------------------------------------------
    print("\n--- MONTHLY UNITS SOLD ---")

    print(
        monthly_sales.to_string(
            index=False,
            formatters={
                "Units Sold": lambda x: f"{x:,.0f}"
            }
        )
    )

    # --------------------------------------------------------
    # 4. Basic trend statistics
    # --------------------------------------------------------
    highest_month = monthly_sales.loc[
        monthly_sales["Units Sold"].idxmax()
    ]

    lowest_month = monthly_sales.loc[
        monthly_sales["Units Sold"].idxmin()
    ]

    print("\n--- TREND SUMMARY ---")

    print(
        f"Average monthly Units Sold : "
        f"{monthly_sales['Units Sold'].mean():,.2f}"
    )

    print(
        f"Highest monthly Units Sold : "
        f"{highest_month['Units Sold']:,.0f} "
        f"({highest_month['Date'].strftime('%Y-%m')})"
    )

    print(
        f"Lowest monthly Units Sold  : "
        f"{lowest_month['Units Sold']:,.0f} "
        f"({lowest_month['Date'].strftime('%Y-%m')})"
    )

    # --------------------------------------------------------
    # 5. Create chart
    # --------------------------------------------------------
    create_chart(
        monthly_sales,
        OUTPUT_PATH
    )

    print("\n--- OUTPUT ---")
    print(f"Chart saved to:")
    print(OUTPUT_PATH)

    print("\nEDA Step 2 completed successfully.")


if __name__ == "__main__":
    main()

