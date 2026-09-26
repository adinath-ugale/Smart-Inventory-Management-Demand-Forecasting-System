from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# SMART INVENTORY MANAGEMENT & DEMAND FORECASTING SYSTEM
# EDA STEP 7: INVENTORY LEVEL VS UNITS SOLD RELATIONSHIP
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
    / "inventory_vs_units_sold.png"
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


def calculate_statistics(
    df: pd.DataFrame
) -> dict:
    """
    Calculate summary statistics and correlation
    between Inventory Level and Units Sold.
    """

    inventory = df["Inventory Level"]
    units_sold = df["Units Sold"]

    correlation = inventory.corr(units_sold)

    statistics = {
        "Inventory_Mean": inventory.mean(),
        "Inventory_Median": inventory.median(),
        "Inventory_Std": inventory.std(),
        "Inventory_Min": inventory.min(),
        "Inventory_Max": inventory.max(),

        "Units_Sold_Mean": units_sold.mean(),
        "Units_Sold_Median": units_sold.median(),
        "Units_Sold_Std": units_sold.std(),
        "Units_Sold_Min": units_sold.min(),
        "Units_Sold_Max": units_sold.max(),

        "Correlation": correlation
    }

    return statistics


def interpret_correlation(
    correlation: float
) -> str:
    """Provide a basic interpretation of Pearson correlation."""

    absolute_correlation = abs(correlation)

    if absolute_correlation < 0.20:
        strength = "very weak"
    elif absolute_correlation < 0.40:
        strength = "weak"
    elif absolute_correlation < 0.60:
        strength = "moderate"
    elif absolute_correlation < 0.80:
        strength = "strong"
    else:
        strength = "very strong"

    if correlation > 0:
        direction = "positive"
    elif correlation < 0:
        direction = "negative"
    else:
        direction = "no"

    if correlation == 0:
        return "There is no linear correlation between Inventory Level and Units Sold."

    return (
        f"There is a {strength} {direction} linear relationship "
        f"between Inventory Level and Units Sold."
    )


def create_chart(
    df: pd.DataFrame,
    output_path: Path
) -> None:
    """Create and save the Inventory Level vs Units Sold scatter plot."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure(figsize=(10, 6))

    plt.scatter(
        df["Inventory Level"],
        df["Units Sold"],
        alpha=0.35,
        s=20
    )

    plt.title(
        "Inventory Level vs Units Sold",
        fontsize=16,
        fontweight="bold"
    )

    plt.xlabel(
        "Inventory Level",
        fontsize=12
    )

    plt.ylabel(
        "Units Sold",
        fontsize=12
    )

    plt.grid(
        alpha=0.3
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
    print(
        "EDA STEP 7: INVENTORY LEVEL VS UNITS SOLD RELATIONSHIP"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------------
    df = load_dataset(DATA_PATH)

    print("\nDataset loaded successfully.")
    print(f"Rows: {len(df):,}")

    # --------------------------------------------------------
    # 2. Calculate statistics
    # --------------------------------------------------------
    statistics = calculate_statistics(df)

    print("\n--- INVENTORY LEVEL STATISTICS ---")

    print(
        f"Mean Inventory Level   : "
        f"{statistics['Inventory_Mean']:,.2f}"
    )

    print(
        f"Median Inventory Level : "
        f"{statistics['Inventory_Median']:,.2f}"
    )

    print(
        f"Std. Deviation         : "
        f"{statistics['Inventory_Std']:,.2f}"
    )

    print(
        f"Minimum Inventory      : "
        f"{statistics['Inventory_Min']:,.0f}"
    )

    print(
        f"Maximum Inventory      : "
        f"{statistics['Inventory_Max']:,.0f}"
    )

    # --------------------------------------------------------
    # 3. Units Sold statistics
    # --------------------------------------------------------
    print("\n--- UNITS SOLD STATISTICS ---")

    print(
        f"Mean Units Sold        : "
        f"{statistics['Units_Sold_Mean']:,.2f}"
    )

    print(
        f"Median Units Sold      : "
        f"{statistics['Units_Sold_Median']:,.2f}"
    )

    print(
        f"Std. Deviation         : "
        f"{statistics['Units_Sold_Std']:,.2f}"
    )

    print(
        f"Minimum Units Sold     : "
        f"{statistics['Units_Sold_Min']:,.0f}"
    )

    print(
        f"Maximum Units Sold     : "
        f"{statistics['Units_Sold_Max']:,.0f}"
    )

    # --------------------------------------------------------
    # 4. Correlation analysis
    # --------------------------------------------------------
    correlation = statistics["Correlation"]

    print("\n--- CORRELATION ANALYSIS ---")

    print(
        f"Pearson Correlation: "
        f"{correlation:.4f}"
    )

    print(
        "\nInterpretation:"
    )

    print(
        interpret_correlation(correlation)
    )

    # --------------------------------------------------------
    # 5. Create chart
    # --------------------------------------------------------
    create_chart(
        df,
        OUTPUT_PATH
    )

    # --------------------------------------------------------
    # 6. Output
    # --------------------------------------------------------
    print("\n--- OUTPUT ---")

    print("Scatter plot saved to:")
    print(OUTPUT_PATH)

    print("\nEDA Step 7 completed successfully.")


if __name__ == "__main__":
    main()