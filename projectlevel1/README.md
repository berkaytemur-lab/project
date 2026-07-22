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


## Limitations

- The analysis uses BTCUSDT one-hour candles from one Binance data source.
- The findings describe the 2021–2025 period and may change in future market conditions.
- A movement event depends on the selected definition: the candle range must equal at least 60% of the previous candle's Bollinger Band width.
- UTC hour provides useful information about movement probability but is not sufficient to predict individual movement events reliably.
- The analysis identifies association, not causation.
- Other possible factors, including market regime, trading volume, news, weekday, and recent price behavior, were not included.
- Seventeen hourly periods were unavailable. Sensitivity analysis showed that excluding gap-affected observations changed hourly event rates by less than 0.08 percentage points.