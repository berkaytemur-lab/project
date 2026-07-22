from pathlib import Path

import pandas as pd


PROJECT_FOLDER = Path(__file__).resolve().parents[1]

RAW_FILE = (
    PROJECT_FOLDER
    / "data"
    / "raw"
    / "btcusdt_1h_2021_2025_raw.csv"
)


def main() -> None:
    df = pd.read_csv(RAW_FILE)

    df["open_time_utc"] = pd.to_datetime(
        df["open_time"],
        unit="ms",
        utc=True,
    )

    df["close_time_utc"] = pd.to_datetime(
        df["close_time"],
        unit="ms",
        utc=True,
    )

    df["duration_ms"] = (
        df["close_time"] - df["open_time"]
    )

    invalid_duration = (
        df["duration_ms"] != 3_599_999
    )

    zero_trading = (
        (df["volume"] == 0)
        | (df["number_of_trades"] == 0)
    )

    display_columns = [
        "open_time_utc",
        "close_time_utc",
        "duration_ms",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "number_of_trades",
    ]

    print("INVALID-DURATION CANDLES")

    print(
        df.loc[
            invalid_duration,
            display_columns,
        ].to_string(index=False)
    )

    print("\nSURROUNDING ROWS")

    anomaly_indices = df.index[
        invalid_duration | zero_trading
    ]

    for index in anomaly_indices:
        start = max(0, index - 1)
        end = min(len(df), index + 2)

        print(
            f"\nRows around dataframe index {index}:"
        )

        print(
            df.loc[
                start:end - 1,
                display_columns,
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()