from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# SMART INVENTORY MANAGEMENT & DEMAND FORECASTING SYSTEM
# EDA STEP 6: REGION-WISE SALES ANALYSIS
# ============================================================

PROJECT_ROOT = Path(r"C:\SIM&DFS")

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cleaned_inventory.csv"
)

FIGURES_DIR = (
    PROJECT_ROOT
    / "reports"
    / "figures"
)

OUTPUT_PATH = (
    FIGURES_DIR
    / "region_units_sold.png"
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


def analyze_region_sales(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate total and average Units Sold
    for each region.
    """

    region_summary = (
        df.groupby("Region")["Units Sold"]
        .agg(
            Total_Units_Sold="sum",
            Average_Units_Sold="mean"
        )
        .reset_index()
        .sort_values(
            by="Total_Units_Sold",
            ascending=False
        )
    )

    return region_summary


def create_chart(
    region_summary: pd.DataFrame,
    output_path: Path
) -> None:
    """Create and save the region-wise Units Sold bar chart."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure(figsize=(10, 6))

    bars = plt.bar(
        region_summary["Region"],
        region_summary["Total_Units_Sold"]
    )

    plt.title(
        "Units Sold by Region",
        fontsize=16,
        fontweight="bold"
    )

    plt.xlabel(
        "Region",
        fontsize=12
    )

    plt.ylabel(
        "Total Units Sold",
        fontsize=12
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    # Add value labels above each bar
    for bar in bars:
        height = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:,.0f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold"
        )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def main() -> None:

    print("=" * 70)
    print(
        "SMART INVENTORY MANAGEMENT & DEMAND FORECASTING SYSTEM"
    )
    print("EDA STEP 6: REGION-WISE SALES ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------------
    df = load_dataset(DATA_PATH)

    print("\nDataset loaded successfully.")
    print(f"Rows: {len(df):,}")

    # --------------------------------------------------------
    # 2. Region analysis
    # --------------------------------------------------------
    region_summary = analyze_region_sales(df)

    print("\n--- UNITS SOLD BY REGION ---")

    print(
        region_summary.to_string(
            index=False,
            formatters={
                "Total_Units_Sold":
                    lambda x: f"{x:,.0f}",
                "Average_Units_Sold":
                    lambda x: f"{x:,.2f}"
            }
        )
    )

    # --------------------------------------------------------
    # 3. Create chart
    # --------------------------------------------------------
    create_chart(
        region_summary,
        OUTPUT_PATH
    )

    # --------------------------------------------------------
    # 4. Trend summary
    # --------------------------------------------------------
    total_units_sold = (
        region_summary["Total_Units_Sold"].sum()
    )

    highest_region = region_summary.iloc[0]
    lowest_region = region_summary.iloc[-1]

    print("\n--- REGION SALES SUMMARY ---")

    print(
        f"Total Units Sold (all regions): "
        f"{total_units_sold:,.0f}"
    )

    print(
        f"Highest-selling region       : "
        f"{highest_region['Region']} "
        f"({highest_region['Total_Units_Sold']:,.0f} units)"
    )

    print(
        f"Lowest-selling region        : "
        f"{lowest_region['Region']} "
        f"({lowest_region['Total_Units_Sold']:,.0f} units)"
    )

    # --------------------------------------------------------
    # 5. Output
    # --------------------------------------------------------
    print("\n--- OUTPUT ---")

    print("Chart saved to:")
    print(OUTPUT_PATH)

    print("\nEDA Step 6 completed successfully.")


if __name__ == "__main__":
    main()