from pathlib import Path

import pandas as pd


PROJECT_FOLDER = Path(__file__).resolve().parents[1]

RAW_FILE = (
    PROJECT_FOLDER
    / "data"
    / "raw"
    / "btcusdt_1h_2021_2025_raw.csv"
)

REPORT_FILE = (
    PROJECT_FOLDER
    / "reports"
    / "missing_timestamps.csv"
)


def main() -> None:
    df = pd.read_csv(RAW_FILE)

    # Convert the raw millisecond timestamp only for validation.
    open_times = pd.to_datetime(
        df["open_time"],
        unit="ms",
        utc=True,
    )

    expected_times = pd.date_range(
        start="2021-01-01 00:00:00",
        end="2026-01-01 00:00:00",
        freq="1h",
        inclusive="left",
        tz="UTC",
    )

    actual_times = pd.DatetimeIndex(open_times)

    missing_times = expected_times.difference(
        actual_times
    )

    duplicate_times = actual_times[
        actual_times.duplicated()
    ]

    print(f"Rows: {len(df):,}")
    print(f"Expected rows: {len(expected_times):,}")
    print(f"Missing timestamps: {len(missing_times)}")
    print(f"Duplicate timestamps: {len(duplicate_times)}")

    print(f"\nFirst timestamp: {actual_times.min()}")
    print(f"Last timestamp:  {actual_times.max()}")

    if len(missing_times) > 0:
        missing_df = pd.DataFrame(
            {
                "missing_open_time_utc": missing_times,
                "utc_hour": missing_times.hour,
            }
        )

        REPORT_FILE.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        missing_df.to_csv(
            REPORT_FILE,
            index=False,
        )

        print("\nMissing timestamps:")
        print(missing_df.to_string(index=False))

        print("\nMissing timestamps by UTC hour:")
        print(
            missing_df["utc_hour"]
            .value_counts()
            .sort_index()
            .to_string()
        )

        print(f"\nReport saved to:\n{REPORT_FILE}")

    if len(duplicate_times) > 0:
        print("\nDuplicate timestamps:")
        print(duplicate_times)


if __name__ == "__main__":
    main()