from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# SMART INVENTORY MANAGEMENT & DEMAND FORECASTING SYSTEM
# EDA STEP 4: MONTHLY DEMAND FORECAST TREND
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
    / "monthly_demand_forecast_trend.png"
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


def create_monthly_forecast_summary(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate monthly average Demand Forecast.

    Missing Demand Forecast values remain NaN and are not
    treated as zero. The aggregation uses only available
    forecast observations.
    """

    monthly_forecast = (
        df.set_index("Date")
        .resample("MS")
        .agg(
            Average_Demand_Forecast=(
                "Demand Forecast",
                "mean"
            ),
            Available_Forecast_Records=(
                "Demand Forecast",
                "count"
            )
        )
        .reset_index()
    )

    return monthly_forecast


def create_chart(
    monthly_forecast: pd.DataFrame,
    output_path: Path
) -> None:
    """Create and save the monthly Demand Forecast trend chart."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure(figsize=(12, 6))

    plt.plot(
        monthly_forecast["Date"],
        monthly_forecast["Average_Demand_Forecast"],
        linewidth=2
    )

    plt.title(
        "Monthly Average Demand Forecast Trend",
        fontsize=16,
        fontweight="bold"
    )

    plt.xlabel("Month")
    plt.ylabel("Average Demand Forecast")

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
    print("EDA STEP 4: MONTHLY DEMAND FORECAST TREND")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------------
    df = load_dataset(DATA_PATH)

    print("\nDataset loaded successfully.")
    print(f"Rows: {len(df):,}")

    # --------------------------------------------------------
    # 2. Report missing forecasts
    # --------------------------------------------------------
    total_forecast = len(df)

    missing_forecast = int(
        df["Demand Forecast"].isna().sum()
    )

    available_forecast = (
        total_forecast - missing_forecast
    )

    print("\n--- DEMAND FORECAST AVAILABILITY ---")

    print(
        f"Total records             : "
        f"{total_forecast:,}"
    )

    print(
        f"Available forecasts       : "
        f"{available_forecast:,}"
    )

    print(
        f"Missing forecasts         : "
        f"{missing_forecast:,}"
    )

    print(
        f"Missing percentage        : "
        f"{(missing_forecast / total_forecast) * 100:.2f}%"
    )

    # --------------------------------------------------------
    # 3. Monthly aggregation
    # --------------------------------------------------------
    monthly_forecast = create_monthly_forecast_summary(df)

    print("\nMonthly forecast aggregation created.")

    print(
        f"Number of months: "
        f"{len(monthly_forecast):,}"
    )

    # --------------------------------------------------------
    # 4. Display monthly summary
    # --------------------------------------------------------
    print("\n--- MONTHLY DEMAND FORECAST ---")

    print(
        monthly_forecast.to_string(
            index=False,
            formatters={
                "Average_Demand_Forecast":
                    lambda x: (
                        f"{x:,.2f}"
                        if pd.notna(x)
                        else "NaN"
                    ),
                "Available_Forecast_Records":
                    lambda x: f"{x:,}"
            }
        )
    )

    # --------------------------------------------------------
    # 5. Trend statistics
    # --------------------------------------------------------
    valid_months = monthly_forecast.dropna(
        subset=["Average_Demand_Forecast"]
    )

    highest_month = valid_months.loc[
        valid_months["Average_Demand_Forecast"].idxmax()
    ]

    lowest_month = valid_months.loc[
        valid_months["Average_Demand_Forecast"].idxmin()
    ]

    print("\n--- TREND SUMMARY ---")

    print(
        f"Overall average Demand Forecast : "
        f"{df['Demand Forecast'].mean():,.2f}"
    )

    print(
        f"Highest monthly average         : "
        f"{highest_month['Average_Demand_Forecast']:,.2f} "
        f"({highest_month['Date'].strftime('%Y-%m')})"
    )

    print(
        f"Lowest monthly average          : "
        f"{lowest_month['Average_Demand_Forecast']:,.2f} "
        f"({lowest_month['Date'].strftime('%Y-%m')})"
    )

    # --------------------------------------------------------
    # 6. Create chart
    # --------------------------------------------------------
    create_chart(
        monthly_forecast,
        OUTPUT_PATH
    )

    print("\n--- OUTPUT ---")

    print("Chart saved to:")
    print(OUTPUT_PATH)

    print("\nEDA Step 4 completed successfully.")


if __name__ == "__main__":
    main()