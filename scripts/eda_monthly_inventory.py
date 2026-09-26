from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# SMART INVENTORY MANAGEMENT & DEMAND FORECASTING SYSTEM
# EDA STEP 3: MONTHLY INVENTORY LEVEL TREND
# ============================================================

PROJECT_ROOT = Path(r"C:\SIM&DFS")

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cleaned_inventory.csv"
)

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

OUTPUT_PATH = (
    FIGURES_DIR
    / "monthly_inventory_level_trend.png"
)


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


def create_monthly_inventory_summary(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate monthly average Inventory Level.

    Inventory Level represents stock position, so monthly
    average is more meaningful than monthly summation.
    """

    monthly_inventory = (
        df.set_index("Date")
        .resample("MS")["Inventory Level"]
        .mean()
        .reset_index()
    )

    return monthly_inventory


def create_chart(
    monthly_inventory: pd.DataFrame,
    output_path: Path
) -> None:
    """Create and save the monthly inventory trend chart."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure(figsize=(12, 6))

    plt.plot(
        monthly_inventory["Date"],
        monthly_inventory["Inventory Level"],
        linewidth=2
    )

    plt.title(
        "Monthly Average Inventory Level Trend",
        fontsize=16,
        fontweight="bold"
    )

    plt.xlabel("Month")
    plt.ylabel("Average Inventory Level")

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
    print("EDA STEP 3: MONTHLY INVENTORY LEVEL TREND")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------------
    df = load_dataset(DATA_PATH)

    print("\nDataset loaded successfully.")
    print(f"Rows: {len(df):,}")

    # --------------------------------------------------------
    # 2. Monthly aggregation
    # --------------------------------------------------------
    monthly_inventory = create_monthly_inventory_summary(df)

    print("\nMonthly inventory aggregation created.")
    print(
        f"Number of months: "
        f"{len(monthly_inventory):,}"
    )

    # --------------------------------------------------------
    # 3. Display monthly summary
    # --------------------------------------------------------
    print("\n--- MONTHLY AVERAGE INVENTORY LEVEL ---")

    print(
        monthly_inventory.to_string(
            index=False,
            formatters={
                "Inventory Level": lambda x: f"{x:,.2f}"
            }
        )
    )

    # --------------------------------------------------------
    # 4. Trend statistics
    # --------------------------------------------------------
    highest_month = monthly_inventory.loc[
        monthly_inventory["Inventory Level"].idxmax()
    ]

    lowest_month = monthly_inventory.loc[
        monthly_inventory["Inventory Level"].idxmin()
    ]

    print("\n--- TREND SUMMARY ---")

    print(
        f"Average monthly Inventory Level : "
        f"{monthly_inventory['Inventory Level'].mean():,.2f}"
    )

    print(
        f"Highest monthly average         : "
        f"{highest_month['Inventory Level']:,.2f} "
        f"({highest_month['Date'].strftime('%Y-%m')})"
    )

    print(
        f"Lowest monthly average          : "
        f"{lowest_month['Inventory Level']:,.2f} "
        f"({lowest_month['Date'].strftime('%Y-%m')})"
    )

    # --------------------------------------------------------
    # 5. Create chart
    # --------------------------------------------------------
    create_chart(
        monthly_inventory,
        OUTPUT_PATH
    )

    print("\n--- OUTPUT ---")
    print("Chart saved to:")
    print(OUTPUT_PATH)

    print("\nEDA Step 3 completed successfully.")


if __name__ == "__main__":
    main()