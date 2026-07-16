import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


# ============================================================
# COLLECTION SETTINGS
# ============================================================

SYMBOL = "BTCUSDT"
INTERVAL = "1h"
LIMIT = 1000

START_DATE = pd.Timestamp(
    "2021-01-01 00:00:00",
    tz="UTC",
)

# Exclusive endpoint:
# 2026-01-01 00:00 is not included.
END_DATE = pd.Timestamp(
    "2026-01-01 00:00:00",
    tz="UTC",
)

API_URL = "https://api.binance.me/api/v1/klines"

PROJECT_FOLDER = Path(__file__).resolve().parents[1]

RAW_DATA_FOLDER = PROJECT_FOLDER / "data" / "raw"

RAW_DATA_FILE = (
    RAW_DATA_FOLDER
    / "btcusdt_1h_2021_2025_raw.csv"
)

METADATA_FILE = (
    RAW_DATA_FOLDER
    / "btcusdt_1h_2021_2025_metadata.json"
)


# One hour in milliseconds.
INTERVAL_MILLISECONDS = 60 * 60 * 1000


# ============================================================
# RAW API COLUMN NAMES
# ============================================================

RAW_COLUMNS = [
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


# ============================================================
# TIME CONVERSION
# ============================================================

def timestamp_to_milliseconds(
    timestamp: pd.Timestamp,
) -> int:
    """
    Convert a UTC pandas Timestamp to Unix milliseconds.
    """
    return int(timestamp.timestamp() * 1000)


# ============================================================
# API RESPONSE HANDLING
# ============================================================

def extract_candles(payload: object) -> list:
    """
    Extract candle rows from the Binance TR API response.

    Some responses may be returned directly as a list.
    Other responses may contain the candles inside a
    dictionary under the 'data' field.
    """

    if isinstance(payload, list):
        candles = payload

    elif isinstance(payload, dict):
        if payload.get("code") not in (None, 0):
            raise ValueError(
                "API returned an error: "
                f"{payload.get('msg', payload)}"
            )

        candles = payload.get("data")

    else:
        raise ValueError(
            "Unexpected API response type: "
            f"{type(payload).__name__}"
        )

    if not isinstance(candles, list):
        raise ValueError(
            f"Unexpected candle data: {candles}"
        )

    return candles


# ============================================================
# DOWNLOAD ONE PAGE
# ============================================================

def download_page(
    session: requests.Session,
    start_time_ms: int,
    end_time_ms: int,
    max_retries: int = 3,
) -> list:
    """
    Download one page containing up to LIMIT candles.
    """

    params = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "startTime": start_time_ms,
        "endTime": end_time_ms,
        "limit": LIMIT,
    }

    for attempt in range(1, max_retries + 1):
        try:
            response = session.get(
                API_URL,
                params=params,
                timeout=30,
            )

            response.raise_for_status()

            payload = response.json()

            return extract_candles(payload)

        except (
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.HTTPError,
            requests.exceptions.JSONDecodeError,
        ) as error:
            if attempt == max_retries:
                raise RuntimeError(
                    "API request failed after "
                    f"{max_retries} attempts."
                ) from error

            wait_seconds = attempt * 2

            print(
                f"Request failed: {error}\n"
                f"Retrying in {wait_seconds} seconds..."
            )

            time.sleep(wait_seconds)

    return []


# ============================================================
# DOWNLOAD ALL CANDLES
# ============================================================

def download_all_candles() -> list:
    """
    Download all hourly candles between START_DATE
    and END_DATE using pagination.
    """

    start_time_ms = timestamp_to_milliseconds(
        START_DATE
    )

    # END_DATE is exclusive, so subtract one millisecond.
    final_end_time_ms = (
        timestamp_to_milliseconds(END_DATE) - 1
    )

    all_candles = []

    session = requests.Session()

    # Prevent requests from using unwanted system proxies.
    session.trust_env = False

    current_start_time_ms = start_time_ms
    request_number = 0

    while current_start_time_ms <= final_end_time_ms:
        request_number += 1

        candles = download_page(
            session=session,
            start_time_ms=current_start_time_ms,
            end_time_ms=final_end_time_ms,
        )

        if not candles:
            print(
                "No more candles were returned by the API."
            )
            break

        all_candles.extend(candles)

        first_open_time_ms = int(candles[0][0])
        last_open_time_ms = int(candles[-1][0])

        first_time = pd.to_datetime(
            first_open_time_ms,
            unit="ms",
            utc=True,
        )

        last_time = pd.to_datetime(
            last_open_time_ms,
            unit="ms",
            utc=True,
        )

        print(
            f"Request {request_number}: "
            f"{len(candles)} candles | "
            f"{first_time} → {last_time}"
        )

        next_start_time_ms = (
            last_open_time_ms
            + INTERVAL_MILLISECONDS
        )

        if next_start_time_ms <= current_start_time_ms:
            raise RuntimeError(
                "Pagination did not advance. "
                "The download was stopped to prevent "
                "an infinite loop."
            )

        current_start_time_ms = next_start_time_ms

        # Small delay between requests.
        time.sleep(0.2)

    session.close()

    return all_candles


# ============================================================
# CREATE RAW DATAFRAME
# ============================================================

def create_raw_dataframe(
    candles: list,
) -> pd.DataFrame:
    """
    Create the raw dataframe without converting timestamps,
    changing numeric values, or calculating indicators.
    """

    if not candles:
        raise ValueError(
            "No candle data was downloaded."
        )

    for row_number, candle in enumerate(
        candles,
        start=1,
    ):
        if len(candle) != len(RAW_COLUMNS):
            raise ValueError(
                f"Row {row_number} contains "
                f"{len(candle)} values instead of "
                f"{len(RAW_COLUMNS)}."
            )

    return pd.DataFrame(
        candles,
        columns=RAW_COLUMNS,
    )


# ============================================================
# VALIDATE RAW DOWNLOAD
# ============================================================

def validate_raw_dataframe(
    df: pd.DataFrame,
) -> None:
    """
    Check that the collected raw dataset is structurally valid.
    This function does not modify the data.
    """

    if df.empty:
        raise ValueError(
            "The raw dataframe is empty."
        )

    if df["open_time"].isna().any():
        raise ValueError(
            "Missing open_time values were detected."
        )

    duplicate_count = int(
        df["open_time"].duplicated().sum()
    )

    if duplicate_count > 0:
        raise ValueError(
            f"{duplicate_count} duplicate candle "
            "timestamps were detected."
        )

    open_times = pd.to_numeric(
        df["open_time"],
        errors="coerce",
    )

    if open_times.isna().any():
        raise ValueError(
            "Invalid open_time values were detected."
        )

    if not open_times.is_monotonic_increasing:
        raise ValueError(
            "Candles are not ordered chronologically."
        )

    start_time_ms = timestamp_to_milliseconds(
        START_DATE
    )

    end_time_ms = timestamp_to_milliseconds(
        END_DATE
    )

    if open_times.iloc[0] < start_time_ms:
        raise ValueError(
            "The dataset begins before START_DATE."
        )

    if open_times.iloc[-1] >= end_time_ms:
        raise ValueError(
            "The dataset contains candles at or after "
            "END_DATE."
        )


# ============================================================
# SAVE RAW DATA
# ============================================================

def save_raw_data(
    df: pd.DataFrame,
) -> None:
    """
    Save the raw candle values to CSV.
    """

    RAW_DATA_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        RAW_DATA_FILE,
        index=False,
    )


# ============================================================
# SAVE COLLECTION METADATA
# ============================================================

def save_metadata(
    df: pd.DataFrame,
) -> None:
    """
    Save information describing where, when, and how
    the raw data was collected.
    """

    first_open_time = pd.to_datetime(
        int(df["open_time"].iloc[0]),
        unit="ms",
        utc=True,
    )

    last_open_time = pd.to_datetime(
        int(df["open_time"].iloc[-1]),
        unit="ms",
        utc=True,
    )

    metadata = {
        "source": "Binance TR public market API",
        "api_url": API_URL,
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "requested_start_date": START_DATE.isoformat(),
        "requested_end_date_exclusive": (
            END_DATE.isoformat()
        ),
        "first_downloaded_candle": (
            first_open_time.isoformat()
        ),
        "last_downloaded_candle": (
            last_open_time.isoformat()
        ),
        "number_of_rows": int(len(df)),
        "columns": RAW_COLUMNS,
        "collection_time_utc": (
            datetime.now(timezone.utc).isoformat()
        ),
        "raw_file": RAW_DATA_FILE.name,
    }

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8",
    ) as metadata_file:
        json.dump(
            metadata,
            metadata_file,
            indent=4,
            ensure_ascii=False,
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print(
        f"Downloading {SYMBOL} {INTERVAL} candles..."
    )

    print(
        f"Period: {START_DATE} → "
        f"{END_DATE} exclusive"
    )

    candles = download_all_candles()

    df = create_raw_dataframe(candles)

    validate_raw_dataframe(df)

    save_raw_data(df)

    save_metadata(df)

    expected_rows = int(
        (
            END_DATE - START_DATE
        ).total_seconds()
        / 3600
    )

    print("\nDownload completed.")

    print(f"Downloaded rows: {len(df):,}")
    print(f"Expected hourly periods: {expected_rows:,}")

    print(f"Raw data saved to:\n{RAW_DATA_FILE}")

    print(f"Metadata saved to:\n{METADATA_FILE}")


if __name__ == "__main__":
    main()