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
    / "movement_event_rate_by_year_and_hour.csv"
)

FIGURE_FOLDER = (
    PROJECT_FOLDER
    / "reports"
    / "figures"
)

HEATMAP_FILE = (
    FIGURE_FOLDER
    / "movement_event_rate_by_year_and_hour.png"
)

YEARLY_PEAK_FILE = (
    PROJECT_FOLDER
    / "reports"
    / "yearly_peak_hours.csv"
)


# ============================================================
# LOAD FEATURE DATA
# ============================================================

def load_feature_data() -> pd.DataFrame:
    """
    Load feature data and restore the required data types.
    """

    df = pd.read_csv(FEATURE_FILE)

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        format="ISO8601",
        utc=True,
    )

    df["movement_event"] = pd.to_numeric(
        df["movement_event"],
        errors="coerce",
    )

    df["movement_ratio"] = pd.to_numeric(
        df["movement_ratio"],
        errors="coerce",
    )

    df["utc_hour"] = pd.to_numeric(
        df["utc_hour"],
        errors="raise",
    ).astype(int)

    df["year"] = pd.to_numeric(
        df["year"],
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
    Select rows unaffected by missing-hour gaps.
    """

    analysis_df = df.loc[
        df["gap_safe_movement_row"]
        & df["movement_event"].notna()
        & df["movement_ratio"].notna()
    ].copy()

    return analysis_df


# ============================================================
# CREATE YEAR-HOUR SUMMARY
# ============================================================

def create_year_hour_summary(
    analysis_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate movement statistics for every combination
    of year and UTC hour.
    """

    summary = (
        analysis_df
        .groupby(
            [
                "year",
                "utc_hour",
            ]
        )
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
            median_movement_ratio=(
                "movement_ratio",
                "median",
            ),
        )
        .reset_index()
    )

    summary["movement_event_rate"] = (
        summary["movement_event_rate"]
        * 100
    )

    summary = summary.sort_values(
        [
            "year",
            "utc_hour",
        ]
    ).reset_index(drop=True)

    return summary


# ============================================================
# IDENTIFY YEARLY PEAK AND LOW HOURS
# ============================================================

def create_yearly_peak_report(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Identify the highest-rate and lowest-rate UTC hour
    within each year.
    """

    rows = []

    for year, year_data in summary.groupby("year"):
        highest_row = year_data.loc[
            year_data["movement_event_rate"].idxmax()
        ]

        lowest_row = year_data.loc[
            year_data["movement_event_rate"].idxmin()
        ]

        rows.append(
            {
                "year": int(year),
                "highest_hour": int(
                    highest_row["utc_hour"]
                ),
                "highest_event_rate": float(
                    highest_row["movement_event_rate"]
                ),
                "lowest_hour": int(
                    lowest_row["utc_hour"]
                ),
                "lowest_event_rate": float(
                    lowest_row["movement_event_rate"]
                ),
                "highest_minus_lowest": float(
                    highest_row["movement_event_rate"]
                    - lowest_row["movement_event_rate"]
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# CREATE HEATMAP
# ============================================================

def create_heatmap(
    summary: pd.DataFrame,
) -> None:
    """
    Create a year-by-hour heatmap of movement-event rates.
    """

    pivot_table = summary.pivot(
        index="year",
        columns="utc_hour",
        values="movement_event_rate",
    )

    FIGURE_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(14, 5))

    image = plt.imshow(
        pivot_table.values,
        aspect="auto",
    )

    plt.colorbar(
        image,
        label="Movement-Event Rate (%)",
    )

    plt.title(
        "BTC Movement-Event Rate by Year and UTC Hour"
    )

    plt.xlabel("UTC Hour")
    plt.ylabel("Year")

    plt.xticks(
        range(len(pivot_table.columns)),
        [
            f"{hour:02d}"
            for hour in pivot_table.columns
        ],
    )

    plt.yticks(
        range(len(pivot_table.index)),
        pivot_table.index,
    )

    for row_index in range(
        len(pivot_table.index)
    ):
        for column_index in range(
            len(pivot_table.columns)
        ):
            value = pivot_table.iloc[
                row_index,
                column_index,
            ]

            plt.text(
                column_index,
                row_index,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=7,
            )

    plt.tight_layout()

    plt.savefig(
        HEATMAP_FILE,
        dpi=200,
    )

    plt.close()


# ============================================================
# SAVE REPORTS
# ============================================================

def save_reports(
    summary: pd.DataFrame,
    yearly_peaks: pd.DataFrame,
) -> None:
    """
    Save the year-hour summary and yearly peak report.
    """

    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        REPORT_FILE,
        index=False,
    )

    yearly_peaks.to_csv(
        YEARLY_PEAK_FILE,
        index=False,
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    df = load_feature_data()

    analysis_df = select_gap_safe_data(df)

    summary = create_year_hour_summary(
        analysis_df
    )

    yearly_peaks = create_yearly_peak_report(
        summary
    )

    save_reports(
        summary,
        yearly_peaks,
    )

    create_heatmap(
        summary
    )

    display_peaks = yearly_peaks.copy()

    decimal_columns = [
        "highest_event_rate",
        "lowest_event_rate",
        "highest_minus_lowest",
    ]

    display_peaks[decimal_columns] = (
        display_peaks[decimal_columns]
        .round(2)
    )

    print("YEARLY PEAK AND LOW HOURS")

    print(
        display_peaks.to_string(
            index=False
        )
    )

    print("\nEVENT RATES FOR UTC HOURS 13–17")

    important_hours = summary.loc[
        summary["utc_hour"].between(
            13,
            17,
        ),
        [
            "year",
            "utc_hour",
            "observations",
            "movement_events",
            "movement_event_rate",
            "median_movement_ratio",
        ],
    ].copy()

    important_hours[
        [
            "movement_event_rate",
            "median_movement_ratio",
        ]
    ] = important_hours[
        [
            "movement_event_rate",
            "median_movement_ratio",
        ]
    ].round(4)

    print(
        important_hours.to_string(
            index=False
        )
    )

    print(
        f"\nYear-hour report saved to:\n{REPORT_FILE}"
    )

    print(
        f"\nYearly peak report saved to:\n{YEARLY_PEAK_FILE}"
    )

    print(
        f"\nHeatmap saved to:\n{HEATMAP_FILE}"
    )


if __name__ == "__main__":
    main()