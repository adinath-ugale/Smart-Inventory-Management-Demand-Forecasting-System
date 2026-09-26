from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# SMART INVENTORY MANAGEMENT & DEMAND FORECASTING SYSTEM
# EDA STEP 9: DEMAND FORECAST VS ACTUAL SALES
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
    / "actual_vs_forecast.png"
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


def calculate_metrics(
    df: pd.DataFrame
) -> dict:
    """
    Calculate MAE, RMSE and MAPE between
    actual Units Sold and Demand Forecast.
    """

    actual = df["Units Sold"]
    forecast = df["Demand Forecast"]

    # --------------------------------------------------------
    # MAE
    # --------------------------------------------------------
    mae = np.mean(
        np.abs(actual - forecast)
    )

    # --------------------------------------------------------
    # RMSE
    # --------------------------------------------------------
    rmse = np.sqrt(
        np.mean(
            (actual - forecast) ** 2
        )
    )

    # --------------------------------------------------------
    # MAPE
    # Exclude rows where actual demand is zero
    # --------------------------------------------------------
    non_zero_actual = actual != 0

    actual_mape = actual[non_zero_actual]
    forecast_mape = forecast[non_zero_actual]

    mape = np.mean(
        np.abs(
            (actual_mape - forecast_mape)
            / actual_mape
        )
    ) * 100

    # --------------------------------------------------------
    # Forecast bias
    # Positive = overestimation
    # Negative = underestimation
    # --------------------------------------------------------
    forecast_error = forecast - actual

    mean_forecast_error = forecast_error.mean()

    total_actual = actual.sum()
    total_forecast = forecast.sum()

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "Mean_Forecast_Error": mean_forecast_error,
        "Total_Actual": total_actual,
        "Total_Forecast": total_forecast,
        "MAPE_Records": len(actual_mape),
        "Zero_Actual_Records": len(actual) - len(actual_mape)
    }


def interpret_forecast(
    mean_forecast_error: float
) -> str:
    """
    Interpret whether the forecast generally
    overestimates or underestimates actual sales.
    """

    if mean_forecast_error > 0:
        return (
            "The Demand Forecast generally overestimates "
            "actual Units Sold."
        )

    elif mean_forecast_error < 0:
        return (
            "The Demand Forecast generally underestimates "
            "actual Units Sold."
        )

    else:
        return (
            "The Demand Forecast has no overall directional "
            "bias compared with actual Units Sold."
        )


def create_chart(
    df: pd.DataFrame,
    output_path: Path
) -> None:
    """
    Create and save an Actual vs Forecast comparison chart.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Aggregate by date so the chart remains readable.
    daily_summary = (
        df.groupby("Date")[
            ["Units Sold", "Demand Forecast"]
        ]
        .sum()
        .reset_index()
        .sort_values("Date")
    )

    plt.figure(figsize=(12, 6))

    plt.plot(
        daily_summary["Date"],
        daily_summary["Units Sold"],
        label="Actual Units Sold",
        linewidth=2
    )

    plt.plot(
        daily_summary["Date"],
        daily_summary["Demand Forecast"],
        label="Demand Forecast",
        linewidth=2
    )

    plt.title(
        "Actual Units Sold vs Demand Forecast",
        fontsize=16,
        fontweight="bold"
    )

    plt.xlabel(
        "Date",
        fontsize=12
    )

    plt.ylabel(
        "Units",
        fontsize=12
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.xticks(
        rotation=45
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
        "EDA STEP 9: DEMAND FORECAST VS ACTUAL SALES"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------------
    df = load_dataset(DATA_PATH)

    print("\nDataset loaded successfully.")
    print(f"Rows: {len(df):,}")

    # --------------------------------------------------------
    # 2. Required columns check
    # --------------------------------------------------------
    required_columns = [
        "Date",
        "Units Sold",
        "Demand Forecast"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    # --------------------------------------------------------
    # 3. Calculate forecast metrics
    # --------------------------------------------------------
    metrics = calculate_metrics(df)

    print("\n--- FORECAST ACCURACY METRICS ---")

    print(
        f"MAE  : "
        f"{metrics['MAE']:,.2f}"
    )

    print(
        f"RMSE : "
        f"{metrics['RMSE']:,.2f}"
    )

    print(
        f"MAPE : "
        f"{metrics['MAPE']:.2f}%"
    )

    # --------------------------------------------------------
    # 4. Actual vs Forecast totals
    # --------------------------------------------------------
    print("\n--- ACTUAL VS FORECAST ---")

    print(
        f"Total Actual Units Sold    : "
        f"{metrics['Total_Actual']:,.0f}"
    )

    print(
        f"Total Forecast Units       : "
        f"{metrics['Total_Forecast']:,.0f}"
    )

    print(
        f"Mean Forecast Error        : "
        f"{metrics['Mean_Forecast_Error']:,.2f}"
    )

    # --------------------------------------------------------
    # 5. Forecast direction
    # --------------------------------------------------------
    print("\n--- FORECAST BIAS ---")

    print(
        interpret_forecast(
            metrics["Mean_Forecast_Error"]
        )
    )

    # --------------------------------------------------------
    # 6. MAPE information
    # --------------------------------------------------------
    print("\n--- MAPE INFORMATION ---")

    print(
        f"Records used for MAPE     : "
        f"{metrics['MAPE_Records']:,}"
    )

    print(
        f"Zero-actual records       : "
        f"{metrics['Zero_Actual_Records']:,}"
    )

    print(
        "Zero-actual records are excluded from MAPE "
        "because percentage error is undefined when "
        "actual demand is zero."
    )

    # --------------------------------------------------------
    # 7. Create chart
    # --------------------------------------------------------
    create_chart(
        df,
        OUTPUT_PATH
    )

    # --------------------------------------------------------
    # 8. Output
    # --------------------------------------------------------
    print("\n--- OUTPUT ---")

    print("Actual vs Forecast chart saved to:")
    print(OUTPUT_PATH)

    print("\nEDA Step 9 completed successfully.")


if __name__ == "__main__":
    main()