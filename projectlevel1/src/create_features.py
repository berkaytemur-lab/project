from pathlib import Path

import pandas as pd


# ============================================================
# FILE PATHS
# ============================================================

PROJECT_FOLDER = Path(__file__).resolve().parents[1]

CLEANED_FILE = (
    PROJECT_FOLDER
    / "data"
    / "processed"
    / "btcusdt_1h_cleaned.csv"
)

FEATURE_FILE = (
    PROJECT_FOLDER
    / "data"
    / "processed"
    / "btcusdt_1h_features.csv"
)


# ============================================================
# PROJECT SETTINGS
# ============================================================

BB_LENGTH = 20
BB_MULTIPLIER = 2.0
MOVEMENT_THRESHOLD = 0.60


# ============================================================
# LOAD CLEANED DATA
# ============================================================

def load_cleaned_data() -> pd.DataFrame:
    """
    Load the cleaned BTCUSDT hourly dataset and convert
    timestamp columns into UTC datetime values.
    """

    df = pd.read_csv(CLEANED_FILE)

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        format="ISO8601",
        utc=True,
    )

    df["close_time"] = pd.to_datetime(
        df["close_time"],
        format="ISO8601",
        utc=True,
    )

    df = df.sort_values(
        "open_time"
    ).reset_index(drop=True)

    return df


# ============================================================
# IDENTIFY TIME GAPS
# ============================================================

def add_gap_flags(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Identify missing-hour boundaries.

    Missing hours are not filled or interpolated.

    gap_before_current:
        The current candle does not begin exactly one hour
        after the previous available candle.

    gap_after_current:
        The next available candle does not begin exactly
        one hour after the current candle.
    """

    df = df.copy()

    df["time_difference"] = (
        df["open_time"].diff()
    )

    df["next_time_difference"] = (
        df["open_time"].shift(-1)
        - df["open_time"]
    )

    df["follows_previous_hour"] = (
        df["time_difference"]
        == pd.Timedelta(hours=1)
    )

    df["followed_by_next_hour"] = (
        df["next_time_difference"]
        == pd.Timedelta(hours=1)
    )

    df["gap_before_current"] = (
        ~df["follows_previous_hour"]
    )

    df["gap_after_current"] = (
        ~df["followed_by_next_hour"]
    )

    # The first row has no previous candle, but this is not
    # treated as a dataset gap.
    df.loc[
        df.index[0],
        "gap_before_current",
    ] = False

    # The last row has no next candle because the collection
    # period ends there. This is not treated as a dataset gap.
    df.loc[
        df.index[-1],
        "gap_after_current",
    ] = False

    df["adjacent_to_gap"] = (
        df["gap_before_current"]
        | df["gap_after_current"]
    )

    return df


# ============================================================
# CALCULATE BOLLINGER BANDS
# ============================================================

def add_bollinger_bands(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate Bollinger Bands using the selected settings:

    Length: 20
    Basis: SMA
    Source: close
    Standard-deviation multiplier: 2.0
    Offset: 0

    Bollinger Bands are calculated using the previous
    20 available dataframe rows, matching the normal
    row-based rolling calculation.

    Windows containing a time gap are flagged but are
    not automatically deleted.
    """

    df = df.copy()

    df["bb_basis"] = (
        df["close"]
        .rolling(
            window=BB_LENGTH,
            min_periods=BB_LENGTH,
        )
        .mean()
    )

    df["bb_standard_deviation"] = (
        df["close"]
        .rolling(
            window=BB_LENGTH,
            min_periods=BB_LENGTH,
        )
        .std(ddof=0)
    )

    df["bb_upper"] = (
        df["bb_basis"]
        + BB_MULTIPLIER
        * df["bb_standard_deviation"]
    )

    df["bb_lower"] = (
        df["bb_basis"]
        - BB_MULTIPLIER
        * df["bb_standard_deviation"]
    )

    df["bb_width"] = (
        df["bb_upper"]
        - df["bb_lower"]
    )

    # A 20-candle window contains 19 internal time intervals.
    # Count how many of those intervals contain a gap.
    df["bb_window_gap_count"] = (
        df["gap_before_current"]
        .astype("int64")
        .rolling(
            window=BB_LENGTH - 1,
            min_periods=BB_LENGTH - 1,
        )
        .sum()
    )

    df["bb_window_available"] = (
        df["bb_width"].notna()
    )

    df["bb_window_gap_affected"] = (
        df["bb_window_available"]
        & (df["bb_window_gap_count"] > 0)
    )

    df["bb_window_gap_safe"] = (
        df["bb_window_available"]
        & (df["bb_window_gap_count"] == 0)
    )

    return df


# ============================================================
# CREATE MOVEMENT FEATURES
# ============================================================

def add_movement_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create the movement variables.

    The current candle range is compared with the Bollinger
    Band width calculated on the previous available candle.

    Gap-related rows remain in the dataset and are marked
    with explicit flags for later sensitivity analysis.
    """

    df = df.copy()

    df["candle_range"] = (
        df["high"]
        - df["low"]
    )

    df["previous_bb_width"] = (
        df["bb_width"].shift(1)
    )

    df["previous_bb_gap_affected"] = (
        df["bb_window_gap_affected"]
        .shift(
            periods=1,
            fill_value=False,
        )
    )

    df["previous_bb_gap_safe"] = (
        df["bb_window_gap_safe"]
        .shift(
            periods=1,
            fill_value=False,
        )
    )

    # A movement ratio can be calculated whenever the
    # previous available candle has a valid BB width.
    df["valid_movement_row"] = (
        df["previous_bb_width"].notna()
        & (df["previous_bb_width"] > 0)
    )

    df["movement_ratio"] = float("nan")

    df.loc[
        df["valid_movement_row"],
        "movement_ratio",
    ] = (
        df.loc[
            df["valid_movement_row"],
            "candle_range",
        ]
        / df.loc[
            df["valid_movement_row"],
            "previous_bb_width",
        ]
    )

    # Nullable integer allows unavailable movement labels
    # to remain empty.
    df["movement_event"] = pd.Series(
        pd.NA,
        index=df.index,
        dtype="Int64",
    )

    df.loc[
        df["valid_movement_row"],
        "movement_event",
    ] = (
        df.loc[
            df["valid_movement_row"],
            "movement_ratio",
        ]
        >= MOVEMENT_THRESHOLD
    ).astype("int64")

    # Main gap-effect flag:
    # - current candle begins after a gap, or
    # - previous BB window contains a gap.
    df["movement_gap_affected"] = (
        df["gap_before_current"]
        | df["previous_bb_gap_affected"]
    )

    # Gap-safe movement rows:
    # previous BB contains no gaps and the current candle
    # immediately follows the previous hour.
    df["gap_safe_movement_row"] = (
        df["valid_movement_row"]
        & ~df["movement_gap_affected"]
    )

    # Extra-conservative version:
    # also exclude the candle immediately before a gap.
    df["strict_gap_affected"] = (
        df["movement_gap_affected"]
        | df["gap_after_current"]
    )

    df["strict_gap_safe_movement_row"] = (
        df["valid_movement_row"]
        & ~df["strict_gap_affected"]
    )

    df["utc_hour"] = (
        df["open_time"].dt.hour
    )

    df["year"] = (
        df["open_time"].dt.year
    )

    return df


# ============================================================
# VALIDATE FEATURES
# ============================================================

def validate_features(
    df: pd.DataFrame,
) -> None:
    """
    Validate derived features before saving.
    """

    negative_ranges = int(
        (df["candle_range"] < 0).sum()
    )

    if negative_ranges > 0:
        raise ValueError(
            "Negative candle ranges were detected."
        )

    invalid_event_assignments = int(
        (
            ~df["valid_movement_row"]
            & df["movement_event"].notna()
        ).sum()
    )

    if invalid_event_assignments > 0:
        raise ValueError(
            "Movement labels were assigned to invalid rows."
        )

    negative_ratios = int(
        (
            df.loc[
                df["valid_movement_row"],
                "movement_ratio",
            ]
            < 0
        ).sum()
    )

    if negative_ratios > 0:
        raise ValueError(
            "Negative movement ratios were detected."
        )

    invalid_gap_safe_rows = int(
        (
            df["gap_safe_movement_row"]
            & ~df["valid_movement_row"]
        ).sum()
    )

    if invalid_gap_safe_rows > 0:
        raise ValueError(
            "Invalid gap-safe movement rows were detected."
        )


# ============================================================
# CALCULATE EVENT RATE
# ============================================================

def calculate_event_rate(
    df: pd.DataFrame,
    row_mask: pd.Series,
) -> float:
    """
    Calculate the percentage of movement events among
    the selected rows.
    """

    selected_events = df.loc[
        row_mask,
        "movement_event",
    ].dropna()

    if selected_events.empty:
        return float("nan")

    return float(
        selected_events.mean() * 100
    )


# ============================================================
# SAVE FEATURE DATA
# ============================================================

def save_feature_data(
    df: pd.DataFrame,
) -> None:
    """
    Save the feature dataset.
    """

    FEATURE_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        FEATURE_FILE,
        index=False,
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    df = load_cleaned_data()

    df = add_gap_flags(df)

    df = add_bollinger_bands(df)

    df = add_movement_features(df)

    validate_features(df)

    save_feature_data(df)

    gap_count = int(
        df["gap_before_current"].sum()
    )

    available_bb_count = int(
        df["bb_window_available"].sum()
    )

    gap_affected_bb_count = int(
        df["bb_window_gap_affected"].sum()
    )

    valid_movement_count = int(
        df["valid_movement_row"].sum()
    )

    gap_safe_movement_count = int(
        df["gap_safe_movement_row"].sum()
    )

    strict_gap_safe_count = int(
        df["strict_gap_safe_movement_row"].sum()
    )

    movement_event_count = int(
        df.loc[
            df["valid_movement_row"],
            "movement_event",
        ].sum()
    )

    all_event_rate = calculate_event_rate(
        df,
        df["valid_movement_row"],
    )

    gap_safe_event_rate = calculate_event_rate(
        df,
        df["gap_safe_movement_row"],
    )

    strict_gap_safe_event_rate = calculate_event_rate(
        df,
        df["strict_gap_safe_movement_row"],
    )

    print(f"Rows loaded: {len(df):,}")
    print(f"First candle: {df['open_time'].min()}")
    print(f"Last candle: {df['open_time'].max()}")

    print("\nGAP SUMMARY")

    print(
        "Separate time-gap blocks:",
        gap_count,
    )

    print(
        "Candles adjacent to gaps:",
        int(df["adjacent_to_gap"].sum()),
    )

    print("\nBOLLINGER BAND SUMMARY")

    print(
        "Available Bollinger Band values:",
        f"{available_bb_count:,}",
    )

    print(
        "BB windows affected by gaps:",
        f"{gap_affected_bb_count:,}",
    )

    print(
        "Gap-safe BB windows:",
        f"{int(df['bb_window_gap_safe'].sum()):,}",
    )

    print("\nMOVEMENT SUMMARY")

    print(
        "All rows available for movement analysis:",
        f"{valid_movement_count:,}",
    )

    print(
        "Gap-safe movement rows:",
        f"{gap_safe_movement_count:,}",
    )

    print(
        "Strict gap-safe movement rows:",
        f"{strict_gap_safe_count:,}",
    )

    print(
        "Total movement events:",
        f"{movement_event_count:,}",
    )

    print(
        "Event rate using all available rows:",
        f"{all_event_rate:.2f}%",
    )

    print(
        "Event rate excluding gap-affected rows:",
        f"{gap_safe_event_rate:.2f}%",
    )

    print(
        "Event rate using strict gap exclusion:",
        f"{strict_gap_safe_event_rate:.2f}%",
    )

    print("\nROWS AROUND TIME GAPS")

    gap_rows = df.loc[
        df["adjacent_to_gap"],
        [
            "open_time",
            "time_difference",
            "next_time_difference",
            "gap_before_current",
            "gap_after_current",
        ],
    ]

    print(
        gap_rows.to_string(index=False)
    )

    print("\nLAST 5 FEATURE ROWS")

    display_columns = [
        "open_time",
        "high",
        "low",
        "candle_range",
        "previous_bb_width",
        "movement_ratio",
        "movement_event",
        "movement_gap_affected",
        "gap_safe_movement_row",
        "utc_hour",
        "year",
    ]

    print(
        df[display_columns]
        .tail()
        .to_string(index=False)
    )

    print(
        f"\nFeature data saved to:\n{FEATURE_FILE}"
    )


if __name__ == "__main__":
    main()