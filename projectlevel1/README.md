## Data Acquisition

- Source: Binance public market API
- Endpoint: https://api.binance.me/api/v1/klines
- Symbol: BTCUSDT
- Interval: 1 hour
- Requested period: 2021-01-01 00:00 UTC to 2026-01-01 00:00 UTC, exclusive
- Unit of analysis: One completed BTCUSDT hourly candle
- Raw rows downloaded: 43,810
- Expected hourly periods: 43,824
- Missing timestamps: 14
- Duplicate timestamps: 0

The raw API response was preserved in `data/raw`. Missing candles were not filled or interpolated. Their timestamps are documented in `reports/missing_timestamps.csv`.