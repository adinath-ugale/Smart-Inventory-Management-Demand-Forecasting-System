from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# SMART INVENTORY MANAGEMENT & DEMAND FORECASTING SYSTEM
# EDA STEP 5: CATEGORY-WISE SALES ANALYSIS
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
    / "category_units_sold.png"
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


def analyze_category_sales(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate total and average Units Sold
    for each product category.
    """

    category_summary = (
        df.groupby("Category")["Units Sold"]
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

    return category_summary


def create_chart(
    category_summary: pd.DataFrame,
    output_path: Path
) -> None:
    """Create and save the category-wise Units Sold bar chart."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure(figsize=(12, 6))

    bars = plt.bar(
        category_summary["Category"],
        category_summary["Total_Units_Sold"]
    )

    plt.title(
        "Units Sold by Category",
        fontsize=16,
        fontweight="bold"
    )

    plt.xlabel(
        "Category",
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
    print("EDA STEP 5: CATEGORY-WISE SALES ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------------
    df = load_dataset(DATA_PATH)

    print("\nDataset loaded successfully.")
    print(f"Rows: {len(df):,}")

    # --------------------------------------------------------
    # 2. Category analysis
    # --------------------------------------------------------
    category_summary = analyze_category_sales(df)

    print("\n--- UNITS SOLD BY CATEGORY ---")

    print(
        category_summary.to_string(
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
        category_summary,
        OUTPUT_PATH
    )

    # --------------------------------------------------------
    # 4. Trend summary
    # --------------------------------------------------------
    total_units_sold = (
        category_summary["Total_Units_Sold"].sum()
    )

    highest_category = category_summary.iloc[0]
    lowest_category = category_summary.iloc[-1]

    print("\n--- CATEGORY SALES SUMMARY ---")

    print(
        f"Total Units Sold (all categories): "
        f"{total_units_sold:,.0f}"
    )

    print(
        f"Highest-selling category         : "
        f"{highest_category['Category']} "
        f"({highest_category['Total_Units_Sold']:,.0f} units)"
    )

    print(
        f"Lowest-selling category          : "
        f"{lowest_category['Category']} "
        f"({lowest_category['Total_Units_Sold']:,.0f} units)"
    )

    # --------------------------------------------------------
    # 5. Output
    # --------------------------------------------------------
    print("\n--- OUTPUT ---")

    print("Chart saved to:")
    print(OUTPUT_PATH)

    print("\nEDA Step 5 completed successfully.")


if __name__ == "__main__":
    main()