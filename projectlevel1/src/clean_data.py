from pathlib import Path

import pandas as pd


PROJECT_FOLDER = Path(__file__).resolve().parents[1]

RAW_FILE = (
    PROJECT_FOLDER
    / "data"
    / "raw"
    / "btcusdt_1h_2021_2025_raw.csv"
)

PROCESSED_FILE = (
    PROJECT_FOLDER
    / "data"
    / "processed"
    / "btcusdt_1h_cleaned.csv"
)

REMOVED_ROWS_FILE = (
    PROJECT_FOLDER
    / "reports"
    / "removed_candles.csv"
)

MINIMUM_DURATION_MS = 59 * 60 * 1000


def main() -> None:
    df = pd.read_csv(RAW_FILE)

    original_row_count = len(df)

    # Convert timestamps from milliseconds to UTC datetime.
    df["open_time"] = pd.to_datetime(
        df["open_time"],
        unit="ms",
        utc=True,
    )

    df["close_time"] = pd.to_datetime(
        df["close_time"],
        unit="ms",
        utc=True,
    )

    # Calculate actual candle duration.
    df["duration_ms"] = (
        df["close_time"] - df["open_time"]
    ).dt.total_seconds() * 1000

    # Define the cleaning rules.
    df["valid_volume"] = df["volume"] > 0

    df["valid_trade_count"] = (
        df["number_of_trades"] > 0
    )

    df["valid_duration"] = (
        df["duration_ms"] >= MINIMUM_DURATION_MS
    )

    df["usable_candle"] = (
        df["valid_volume"]
        & df["valid_trade_count"]
        & df["valid_duration"]
    )

    removed_rows = df.loc[
        ~df["usable_candle"]
    ].copy()

    cleaned_df = df.loc[
        df["usable_candle"]
    ].copy()

    # The API ignore field contains no useful information.
    cleaned_df = cleaned_df.drop(
    columns=[
        "ignore",
        "valid_volume",
        "valid_trade_count",
        "valid_duration",
        "usable_candle",
    ]
)

    cleaned_df = cleaned_df.sort_values(
        "open_time"
    ).reset_index(drop=True)

    PROCESSED_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REMOVED_ROWS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cleaned_df.to_csv(
        PROCESSED_FILE,
        index=False,
    )

    removed_rows.to_csv(
        REMOVED_ROWS_FILE,
        index=False,
    )

    print("CLEANING SUMMARY")
    print(f"Rows before cleaning: {original_row_count:,}")
    print(f"Rows removed: {len(removed_rows):,}")
    print(f"Rows after cleaning: {len(cleaned_df):,}")

    print("\nREMOVED CANDLES")

    print(
        removed_rows[
            [
                "open_time",
                "close_time",
                "duration_ms",
                "volume",
                "number_of_trades",
            ]
        ].to_string(index=False)
    )

    print(f"\nProcessed data saved to:\n{PROCESSED_FILE}")
    print(f"\nRemoved-row report saved to:\n{REMOVED_ROWS_FILE}")


if __name__ == "__main__":
    main()