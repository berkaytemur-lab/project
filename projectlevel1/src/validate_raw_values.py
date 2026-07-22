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

    price_columns = [
        "open",
        "high",
        "low",
        "close",
    ]

    negative_price_rows = (
        df[price_columns] <= 0
    ).any(axis=1)

    invalid_high_rows = (
        (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["high"] < df["low"])
    )

    invalid_low_rows = (
        (df["low"] > df["open"])
        | (df["low"] > df["close"])
        | (df["low"] > df["high"])
    )

    negative_volume_rows = (
        (df["volume"] < 0)
        | (df["quote_volume"] < 0)
        | (df["taker_buy_base_volume"] < 0)
        | (df["taker_buy_quote_volume"] < 0)
    )

    negative_trade_rows = (
        df["number_of_trades"] < 0
    )

    invalid_duration_rows = (
        df["close_time"] - df["open_time"]
        != 3_599_999
    )

    invalid_open_alignment = (
        df["open_time"] % 3_600_000 != 0
    )

    invalid_taker_base_rows = (
        df["taker_buy_base_volume"]
        > df["volume"]
    )

    invalid_taker_quote_rows = (
        df["taker_buy_quote_volume"]
        > df["quote_volume"]
    )

    zero_volume_rows = (
        (df["volume"] == 0)
        | (df["number_of_trades"] == 0)
    )

    checks = {
        "Non-positive price rows": negative_price_rows.sum(),
        "Invalid high rows": invalid_high_rows.sum(),
        "Invalid low rows": invalid_low_rows.sum(),
        "Negative volume rows": negative_volume_rows.sum(),
        "Negative trade-count rows": negative_trade_rows.sum(),
        "Invalid candle-duration rows": invalid_duration_rows.sum(),
        "Misaligned hourly timestamps": invalid_open_alignment.sum(),
        "Taker base exceeds total": invalid_taker_base_rows.sum(),
        "Taker quote exceeds total": invalid_taker_quote_rows.sum(),
        "Zero-volume or zero-trade rows": zero_volume_rows.sum(),
    }

    print("VALIDATION RESULTS")

    for check_name, count in checks.items():
        print(f"{check_name}: {count}")

    print("\nIGNORE COLUMN VALUES")
    print(df["ignore"].value_counts(dropna=False))

    if zero_volume_rows.any():
        zero_rows = df.loc[
            zero_volume_rows,
            [
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "number_of_trades",
            ],
        ].copy()

        zero_rows["open_time"] = pd.to_datetime(
            zero_rows["open_time"],
            unit="ms",
            utc=True,
        )

        print("\nZERO-VOLUME OR ZERO-TRADE CANDLES")
        print(zero_rows.to_string(index=False))


if __name__ == "__main__":
    main()