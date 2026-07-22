from pathlib import Path

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
    / "gap_sensitivity_by_hour.csv"
)


# ============================================================
# LOAD FEATURE DATA
# ============================================================

def load_feature_data() -> pd.DataFrame:
    """
    Load the feature dataset and restore important data types.
    """

    df = pd.read_csv(FEATURE_FILE)

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        format="ISO8601",
        utc=True,
    )

    boolean_columns = [
        "valid_movement_row",
        "gap_safe_movement_row",
        "strict_gap_safe_movement_row",
    ]

    for column in boolean_columns:
        if df[column].dtype == "object":
            df[column] = (
                df[column]
                .astype(str)
                .str.lower()
                .map(
                    {
                        "true": True,
                        "false": False,
                    }
                )
            )

        df[column] = df[column].fillna(False).astype(bool)

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

    return df


# ============================================================
# SUMMARIZE ONE ANALYSIS VERSION
# ============================================================

def summarize_by_hour(
    df: pd.DataFrame,
    row_mask: pd.Series,
    label: str,
) -> pd.DataFrame:
    """
    Calculate hourly sample size, movement-event count,
    event rate, and median movement ratio.
    """

    selected = df.loc[
        row_mask
        & df["movement_event"].notna()
    ].copy()

    summary = (
        selected
        .groupby("utc_hour")
        .agg(
            **{
                f"{label}_observations": (
                    "movement_event",
                    "size",
                ),
                f"{label}_events": (
                    "movement_event",
                    "sum",
                ),
                f"{label}_event_rate": (
                    "movement_event",
                    "mean",
                ),
                f"{label}_median_ratio": (
                    "movement_ratio",
                    "median",
                ),
            }
        )
        .reset_index()
    )

    summary[f"{label}_event_rate"] = (
        summary[f"{label}_event_rate"] * 100
    )

    return summary


# ============================================================
# CREATE GAP-SENSITIVITY REPORT
# ============================================================

def create_gap_sensitivity_report(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare hourly results using three different treatments
    of candles influenced by missing periods.
    """

    all_summary = summarize_by_hour(
        df=df,
        row_mask=df["valid_movement_row"],
        label="all",
    )

    gap_safe_summary = summarize_by_hour(
        df=df,
        row_mask=df["gap_safe_movement_row"],
        label="gap_safe",
    )

    strict_summary = summarize_by_hour(
        df=df,
        row_mask=df["strict_gap_safe_movement_row"],
        label="strict",
    )

    report = all_summary.merge(
        gap_safe_summary,
        on="utc_hour",
        how="outer",
    )

    report = report.merge(
        strict_summary,
        on="utc_hour",
        how="outer",
    )

    report = report.sort_values(
        "utc_hour"
    ).reset_index(drop=True)

    report["gap_safe_rate_difference"] = (
        report["gap_safe_event_rate"]
        - report["all_event_rate"]
    )

    report["strict_rate_difference"] = (
        report["strict_event_rate"]
        - report["all_event_rate"]
    )

    report["absolute_gap_safe_difference"] = (
        report["gap_safe_rate_difference"].abs()
    )

    report["absolute_strict_difference"] = (
        report["strict_rate_difference"].abs()
    )

    report["excluded_gap_safe_rows"] = (
        report["all_observations"]
        - report["gap_safe_observations"]
    )

    report["excluded_strict_rows"] = (
        report["all_observations"]
        - report["strict_observations"]
    )

    return report


# ============================================================
# SAVE REPORT
# ============================================================

def save_report(
    report: pd.DataFrame,
) -> None:
    """
    Save the hourly gap-sensitivity comparison.
    """

    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report.to_csv(
        REPORT_FILE,
        index=False,
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    df = load_feature_data()

    report = create_gap_sensitivity_report(df)

    save_report(report)

    maximum_gap_safe_difference = (
        report["absolute_gap_safe_difference"].max()
    )

    maximum_strict_difference = (
        report["absolute_strict_difference"].max()
    )

    largest_gap_safe_hour = int(
        report.loc[
            report["absolute_gap_safe_difference"].idxmax(),
            "utc_hour",
        ]
    )

    largest_strict_hour = int(
        report.loc[
            report["absolute_strict_difference"].idxmax(),
            "utc_hour",
        ]
    )

    print("GAP-SENSITIVITY RESULTS")

    print(
        "\nLargest hourly difference after excluding "
        "gap-affected rows:"
    )

    print(
        f"UTC hour: {largest_gap_safe_hour:02d}:00"
    )

    print(
        f"Absolute difference: "
        f"{maximum_gap_safe_difference:.4f} percentage points"
    )

    print(
        "\nLargest hourly difference using strict exclusion:"
    )

    print(
        f"UTC hour: {largest_strict_hour:02d}:00"
    )

    print(
        f"Absolute difference: "
        f"{maximum_strict_difference:.4f} percentage points"
    )

    display_columns = [
        "utc_hour",
        "all_observations",
        "all_event_rate",
        "gap_safe_observations",
        "gap_safe_event_rate",
        "gap_safe_rate_difference",
        "strict_observations",
        "strict_event_rate",
        "strict_rate_difference",
    ]

    display_report = report[
        display_columns
    ].copy()

    rate_columns = [
        "all_event_rate",
        "gap_safe_event_rate",
        "gap_safe_rate_difference",
        "strict_event_rate",
        "strict_rate_difference",
    ]

    display_report[rate_columns] = (
        display_report[rate_columns].round(4)
    )

    print("\nHOURLY COMPARISON")

    print(
        display_report.to_string(
            index=False
        )
    )

    print(
        f"\nReport saved to:\n{REPORT_FILE}"
    )


if __name__ == "__main__":
    main()