from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# FILE PATHS
# ============================================================

PROJECT_FOLDER = Path(__file__).resolve().parents[1]

FEATURE_FILE = (
    PROJECT_FOLDER
    / "data"
    / "processed"
    / "btcusdt_1h_features.csv"
)

REPORT_FILE = (
    PROJECT_FOLDER
    / "reports"
    / "hourly_movement_summary.csv"
)

FIGURE_FOLDER = (
    PROJECT_FOLDER
    / "reports"
    / "figures"
)

EVENT_RATE_FIGURE = (
    FIGURE_FOLDER
    / "movement_event_rate_by_utc_hour.png"
)

MEDIAN_RATIO_FIGURE = (
    FIGURE_FOLDER
    / "median_movement_ratio_by_utc_hour.png"
)

OBSERVATION_FIGURE = (
    FIGURE_FOLDER
    / "observations_by_utc_hour.png"
)


# ============================================================
# LOAD FEATURE DATA
# ============================================================

def load_feature_data() -> pd.DataFrame:
    """
    Load the feature dataset and restore required data types.
    """

    df = pd.read_csv(FEATURE_FILE)

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        format="ISO8601",
        utc=True,
    )

    df["movement_ratio"] = pd.to_numeric(
        df["movement_ratio"],
        errors="coerce",
    )

    df["movement_event"] = pd.to_numeric(
        df["movement_event"],
        errors="coerce",
    )

    df["utc_hour"] = pd.to_numeric(
        df["utc_hour"],
        errors="raise",
    ).astype(int)

    if df["gap_safe_movement_row"].dtype == "object":
        df["gap_safe_movement_row"] = (
            df["gap_safe_movement_row"]
            .astype(str)
            .str.lower()
            .map(
                {
                    "true": True,
                    "false": False,
                }
            )
        )

    df["gap_safe_movement_row"] = (
        df["gap_safe_movement_row"]
        .fillna(False)
        .astype(bool)
    )

    return df


# ============================================================
# SELECT PRIMARY ANALYSIS DATA
# ============================================================

def select_gap_safe_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select movement rows whose Bollinger Band calculation
    was not affected by missing hourly periods.
    """

    analysis_df = df.loc[
        df["gap_safe_movement_row"]
        & df["movement_event"].notna()
        & df["movement_ratio"].notna()
    ].copy()

    return analysis_df


# ============================================================
# CALCULATE HOURLY SUMMARY
# ============================================================

def create_hourly_summary(
    analysis_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate descriptive movement statistics for each
    UTC hour.
    """

    hourly_summary = (
        analysis_df
        .groupby("utc_hour")
        .agg(
            observations=(
                "movement_event",
                "size",
            ),
            movement_events=(
                "movement_event",
                "sum",
            ),
            movement_event_rate=(
                "movement_event",
                "mean",
            ),
            mean_movement_ratio=(
                "movement_ratio",
                "mean",
            ),
            median_movement_ratio=(
                "movement_ratio",
                "median",
            ),
            percentile_90_ratio=(
                "movement_ratio",
                lambda values: values.quantile(0.90),
            ),
            maximum_movement_ratio=(
                "movement_ratio",
                "max",
            ),
        )
        .reset_index()
    )

    hourly_summary["movement_event_rate"] = (
        hourly_summary["movement_event_rate"]
        * 100
    )

    hourly_summary = hourly_summary.sort_values(
        "utc_hour"
    ).reset_index(drop=True)

    hourly_summary["event_rate_rank"] = (
        hourly_summary["movement_event_rate"]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    return hourly_summary


# ============================================================
# CREATE EVENT-RATE CHART
# ============================================================

def create_event_rate_chart(
    hourly_summary: pd.DataFrame,
) -> None:
    """
    Create a bar chart showing movement-event rates
    for every UTC hour.
    """

    overall_rate = (
        hourly_summary["movement_events"].sum()
        / hourly_summary["observations"].sum()
        * 100
    )

    plt.figure(figsize=(12, 6))

    plt.bar(
        hourly_summary["utc_hour"],
        hourly_summary["movement_event_rate"],
        edgecolor="black",
    )

    plt.axhline(
        overall_rate,
        linestyle="--",
        linewidth=2,
        label=f"Overall event rate = {overall_rate:.2f}%",
    )

    plt.title(
        "BTC Movement-Event Rate by UTC Hour\n"
        "Gap-Safe Hourly Candles, 2021–2025"
    )

    plt.xlabel("UTC Hour")
    plt.ylabel("Movement-Event Rate (%)")

    plt.xticks(
        range(24),
        [f"{hour:02d}" for hour in range(24)],
    )

    plt.legend()
    plt.tight_layout()

    FIGURE_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        EVENT_RATE_FIGURE,
        dpi=200,
    )

    plt.close()


# ============================================================
# CREATE MEDIAN-RATIO CHART
# ============================================================

def create_median_ratio_chart(
    hourly_summary: pd.DataFrame,
) -> None:
    """
    Create a chart of the typical movement ratio for
    every UTC hour.
    """

    plt.figure(figsize=(12, 6))

    plt.bar(
        hourly_summary["utc_hour"],
        hourly_summary["median_movement_ratio"],
        edgecolor="black",
    )

    plt.title(
        "Median BTC Movement Ratio by UTC Hour\n"
        "Gap-Safe Hourly Candles, 2021–2025"
    )

    plt.xlabel("UTC Hour")
    plt.ylabel(
        "Median Candle Range / Previous BB Width"
    )

    plt.xticks(
        range(24),
        [f"{hour:02d}" for hour in range(24)],
    )

    plt.tight_layout()

    plt.savefig(
        MEDIAN_RATIO_FIGURE,
        dpi=200,
    )

    plt.close()


# ============================================================
# CREATE OBSERVATION-COUNT CHART
# ============================================================

def create_observation_chart(
    hourly_summary: pd.DataFrame,
) -> None:
    """
    Verify that every UTC hour has a similar number of
    usable observations.
    """

    plt.figure(figsize=(12, 6))

    plt.bar(
        hourly_summary["utc_hour"],
        hourly_summary["observations"],
        edgecolor="black",
    )

    plt.title(
        "Valid Observations by UTC Hour\n"
        "Gap-Safe Hourly Candles, 2021–2025"
    )

    plt.xlabel("UTC Hour")
    plt.ylabel("Number of Valid Candles")

    plt.xticks(
        range(24),
        [f"{hour:02d}" for hour in range(24)],
    )

    plt.tight_layout()

    plt.savefig(
        OBSERVATION_FIGURE,
        dpi=200,
    )

    plt.close()


# ============================================================
# SAVE REPORT
# ============================================================

def save_hourly_summary(
    hourly_summary: pd.DataFrame,
) -> None:
    """
    Save the hourly descriptive-statistics report.
    """

    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    hourly_summary.to_csv(
        REPORT_FILE,
        index=False,
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    df = load_feature_data()

    analysis_df = select_gap_safe_data(df)

    hourly_summary = create_hourly_summary(
        analysis_df
    )

    save_hourly_summary(
        hourly_summary
    )

    create_event_rate_chart(
        hourly_summary
    )

    create_median_ratio_chart(
        hourly_summary
    )

    create_observation_chart(
        hourly_summary
    )

    display_columns = [
        "utc_hour",
        "observations",
        "movement_events",
        "movement_event_rate",
        "median_movement_ratio",
        "mean_movement_ratio",
        "percentile_90_ratio",
        "event_rate_rank",
    ]

    display_summary = hourly_summary[
        display_columns
    ].copy()

    decimal_columns = [
        "movement_event_rate",
        "median_movement_ratio",
        "mean_movement_ratio",
        "percentile_90_ratio",
    ]

    display_summary[decimal_columns] = (
        display_summary[decimal_columns]
        .round(4)
    )

    print("HOURLY MOVEMENT SUMMARY")

    print(
        display_summary.to_string(
            index=False
        )
    )

    highest_hour = hourly_summary.loc[
        hourly_summary["movement_event_rate"].idxmax()
    ]

    lowest_hour = hourly_summary.loc[
        hourly_summary["movement_event_rate"].idxmin()
    ]

    print("\nHIGHEST MOVEMENT-EVENT HOUR")

    print(
        f"UTC hour: "
        f"{int(highest_hour['utc_hour']):02d}:00"
    )

    print(
        f"Event rate: "
        f"{highest_hour['movement_event_rate']:.2f}%"
    )

    print("\nLOWEST MOVEMENT-EVENT HOUR")

    print(
        f"UTC hour: "
        f"{int(lowest_hour['utc_hour']):02d}:00"
    )

    print(
        f"Event rate: "
        f"{lowest_hour['movement_event_rate']:.2f}%"
    )

    rate_difference = (
        highest_hour["movement_event_rate"]
        - lowest_hour["movement_event_rate"]
    )

    print(
        "\nHighest-minus-lowest difference:",
        f"{rate_difference:.2f} percentage points",
    )

    print(
        f"\nReport saved to:\n{REPORT_FILE}"
    )

    print(
        f"\nEvent-rate chart saved to:\n"
        f"{EVENT_RATE_FIGURE}"
    )

    print(
        f"\nMedian-ratio chart saved to:\n"
        f"{MEDIAN_RATIO_FIGURE}"
    )

    print(
        f"\nObservation chart saved to:\n"
        f"{OBSERVATION_FIGURE}"
    )


if __name__ == "__main__":
    main()