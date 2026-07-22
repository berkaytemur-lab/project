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

FIGURE_FOLDER = (
    PROJECT_FOLDER
    / "reports"
    / "figures"
)

HISTOGRAM_FILE = (
    FIGURE_FOLDER
    / "movement_ratio_distribution.png"
)

SUMMARY_FILE = (
    PROJECT_FOLDER
    / "reports"
    / "movement_ratio_summary.csv"
)


# ============================================================
# PROJECT SETTINGS
# ============================================================

MOVEMENT_THRESHOLD = 0.60


# ============================================================
# LOAD DATA
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

    boolean_columns = [
        "valid_movement_row",
        "gap_safe_movement_row",
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

        df[column] = (
            df[column]
            .fillna(False)
            .astype(bool)
        )

    return df


# ============================================================
# SELECT PRIMARY ANALYSIS DATA
# ============================================================

def select_gap_safe_rows(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select rows whose movement ratio is available and whose
    previous Bollinger Band window is not affected by gaps.
    """

    analysis_df = df.loc[
        df["gap_safe_movement_row"]
        & df["movement_ratio"].notna()
        & df["movement_event"].notna()
    ].copy()

    return analysis_df


# ============================================================
# CREATE NUMERICAL SUMMARY
# ============================================================

def create_summary(
    analysis_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate descriptive statistics for movement ratios.
    """

    movement_ratio = analysis_df["movement_ratio"]

    summary = {
        "observations": len(movement_ratio),
        "mean": movement_ratio.mean(),
        "median": movement_ratio.median(),
        "standard_deviation": movement_ratio.std(),
        "minimum": movement_ratio.min(),
        "percentile_25": movement_ratio.quantile(0.25),
        "percentile_75": movement_ratio.quantile(0.75),
        "percentile_90": movement_ratio.quantile(0.90),
        "percentile_95": movement_ratio.quantile(0.95),
        "percentile_99": movement_ratio.quantile(0.99),
        "maximum": movement_ratio.max(),
        "skewness": movement_ratio.skew(),
        "movement_events": int(
            analysis_df["movement_event"].sum()
        ),
        "movement_event_rate_percent": (
            analysis_df["movement_event"].mean()
            * 100
        ),
    }

    return pd.DataFrame(
        {
            "statistic": summary.keys(),
            "value": summary.values(),
        }
    )


# ============================================================
# CREATE HISTOGRAM
# ============================================================

def create_histogram(
    analysis_df: pd.DataFrame,
) -> None:
    """
    Create a histogram of movement ratios.

    The chart is limited to the 99th percentile so that
    extreme outliers do not compress the main distribution.
    """

    movement_ratio = analysis_df["movement_ratio"]

    percentile_99 = movement_ratio.quantile(0.99)

    chart_data = movement_ratio[
        movement_ratio <= percentile_99
    ]

    FIGURE_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.hist(
        chart_data,
        bins=60,
        edgecolor="black",
    )

    plt.axvline(
        MOVEMENT_THRESHOLD,
        linestyle="--",
        linewidth=2,
        label="Movement threshold = 0.60",
    )

    plt.title(
        "Distribution of BTC Hourly Movement Ratios\n"
        "Gap-Safe Rows, Limited to the 99th Percentile"
    )

    plt.xlabel(
        "Candle Range / Previous Bollinger Band Width"
    )

    plt.ylabel(
        "Number of Hourly Candles"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        HISTOGRAM_FILE,
        dpi=200,
    )

    plt.close()


# ============================================================
# SAVE SUMMARY
# ============================================================

def save_summary(
    summary_df: pd.DataFrame,
) -> None:
    """
    Save the descriptive statistics report.
    """

    SUMMARY_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_df.to_csv(
        SUMMARY_FILE,
        index=False,
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    df = load_feature_data()

    analysis_df = select_gap_safe_rows(df)

    summary_df = create_summary(
        analysis_df
    )

    save_summary(
        summary_df
    )

    create_histogram(
        analysis_df
    )

    print("MOVEMENT-RATIO DISTRIBUTION")

    print(
        summary_df.to_string(
            index=False
        )
    )

    below_threshold = int(
        (
            analysis_df["movement_ratio"]
            < MOVEMENT_THRESHOLD
        ).sum()
    )

    at_or_above_threshold = int(
        (
            analysis_df["movement_ratio"]
            >= MOVEMENT_THRESHOLD
        ).sum()
    )

    print("\nTHRESHOLD COUNTS")

    print(
        "Below 0.60:",
        f"{below_threshold:,}",
    )

    print(
        "At or above 0.60:",
        f"{at_or_above_threshold:,}",
    )

    print(
        f"\nSummary saved to:\n{SUMMARY_FILE}"
    )

    print(
        f"\nHistogram saved to:\n{HISTOGRAM_FILE}"
    )


if __name__ == "__main__":
    main()