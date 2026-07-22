import requests
import pandas as pd
from pathlib import Path


# ============================================================
# GENERAL SETTINGS
# ============================================================

SYMBOL = "BTCUSDT"
INTERVALS = ["15m", "1h", "1d"]
LIMIT = 1000

API_URL = "https://api.binance.me/api/v1/klines"

OUTPUT_FOLDER = Path(__file__).resolve().parent / "data"


# ============================================================
# TRADINGVIEW BOLLINGER BAND SETTINGS
# ============================================================

BB_LENGTH = 20
BB_MULTIPLIER = 2.0


# ============================================================
# TRADINGVIEW PVO SETTINGS
# ============================================================

PVO_FAST_LENGTH = 12
PVO_SLOW_LENGTH = 26
PVO_SIGNAL_LENGTH = 9


def download_candles(interval: str) -> list:
    params = {
        "symbol": SYMBOL,
        "interval": interval,
        "limit": LIMIT,
    }

    session = requests.Session()
    session.trust_env = False

    response = session.get(
        API_URL,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    if not isinstance(data, list):
        raise ValueError(f"Unexpected API response: {data}")

    if len(data) == 0:
        raise ValueError(f"No candle data returned for {interval}.")

    return data


def create_dataframe(data: list) -> pd.DataFrame:
    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "ignore",
    ]

    df = pd.DataFrame(data, columns=columns)

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ]

    df[numeric_columns] = df[numeric_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

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

    df = df.drop(columns=["ignore"])

    df = df.sort_values("open_time").reset_index(drop=True)

    return df


def remove_unfinished_candle(
    df: pd.DataFrame,
) -> pd.DataFrame:
    current_time = pd.Timestamp.now(tz="UTC")

    if (
        not df.empty
        and df.iloc[-1]["close_time"] > current_time
    ):
        df = df.iloc[:-1].copy()

    return df.reset_index(drop=True)


def calculate_bollinger_bands(
    df: pd.DataFrame,
) -> pd.DataFrame:
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

    return df


def tradingview_ema(
    series: pd.Series,
    length: int,
) -> pd.Series:
    return series.ewm(
        span=length,
        adjust=False,
        min_periods=1,
    ).mean()


def calculate_pvo(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df["pvo_fast_volume_ema"] = tradingview_ema(
        df["volume"],
        PVO_FAST_LENGTH,
    )

    df["pvo_slow_volume_ema"] = tradingview_ema(
        df["volume"],
        PVO_SLOW_LENGTH,
    )

    df["pvo"] = (
        100
        * (
            df["pvo_fast_volume_ema"]
            - df["pvo_slow_volume_ema"]
        )
        / df["pvo_slow_volume_ema"]
    )

    df["pvo_signal"] = tradingview_ema(
        df["pvo"],
        PVO_SIGNAL_LENGTH,
    )

    df["pvo_histogram"] = (
        df["pvo"]
        - df["pvo_signal"]
    )

    return df


def process_interval(interval: str) -> None:
    print(f"\nDownloading {SYMBOL} {interval} data...")

    candle_data = download_candles(interval)

    df = create_dataframe(candle_data)

    df = remove_unfinished_candle(df)

    df = calculate_bollinger_bands(df)

    df = calculate_pvo(df)

    output_file = (
        OUTPUT_FOLDER
        / f"{SYMBOL.lower()}_{interval}_bb_pvo.csv"
    )

    df.to_csv(
        output_file,
        index=False,
    )

    display_columns = [
        "open_time",
        "close",
        "volume",
        "bb_basis",
        "bb_upper",
        "bb_lower",
        "pvo",
        "pvo_signal",
        "pvo_histogram",
    ]

    print(f"Completed candles: {len(df)}")
    print(f"Saved to: {output_file}")

    print(
        df[display_columns]
        .tail(3)
        .to_string(index=False)
    )


def main() -> None:
    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    for interval in INTERVALS:
        try:
            process_interval(interval)

        except requests.exceptions.Timeout:
            print(f"{interval}: API connection timed out.")

        except requests.exceptions.HTTPError as error:
            print(f"{interval}: HTTP error: {error}")

        except requests.exceptions.ConnectionError as error:
            print(f"{interval}: Connection error: {error}")

        except Exception as error:
            print(f"{interval}: Unexpected error: {error}")


if __name__ == "__main__":
    main()