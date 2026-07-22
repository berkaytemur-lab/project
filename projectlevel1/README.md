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


## Final Interpretation

### What the evidence supports

BTCUSDT movement events were not distributed evenly across UTC hours during 2021–2025. A movement event was defined as an hourly candle range equal to at least 60% of the previous candle's Bollinger Band width.

The overall movement-event rate was 9.16%. The highest rate occurred at 14:00 UTC, where 25.52% of candles were movement events. The lowest rate occurred at 05:00 UTC, where the rate was 2.69%. Movement rates were generally elevated between 13:00 and 16:00 UTC and relatively low between 03:00 and 07:00 UTC.

The yearly analysis showed that 14:00 UTC was the highest-rate hour in 2022, 2023, 2024, and 2025. The pattern was weaker in 2021, which shows that its strength changes between market periods.

A logistic-regression model trained on 2021–2024 correctly identified 14:00 UTC as the highest-risk hour in the unseen 2025 data. For predicting hourly movement rates, the model achieved an MAE of 2.49 percentage points and an RMSE of 2.99 percentage points. This was better than the constant-rate baseline, which produced an MAE of 5.01 and an RMSE of 6.59 percentage points.

### What the evidence does not support

The analysis does not prove that the UTC hour causes BTC movements. It only demonstrates an association between UTC hour and the historical frequency of movement events.

UTC hour alone was not sufficient to predict individual movement events. At the standard 50% classification threshold, logistic regression predicted no movement events and performed identically to the majority-class baseline.

The results also do not demonstrate that trading during the identified hours would be profitable. Transaction costs, movement direction, entry timing, execution, and risk management were not analyzed.

### Alternative explanations

The hourly pattern may be related to differences in global trading activity, market-session overlap, scheduled economic announcements, liquidity, or other recurring events. These explanations were not tested in this project and therefore remain speculation.

### Data-quality effect

Seventeen hourly periods were unavailable. The sensitivity analysis showed that excluding Bollinger Band windows affected by these gaps changed hourly event rates by no more than 0.08 percentage points. The missing periods therefore did not materially change the main conclusion.

### Practical implication

UTC hour can be used as a contextual risk variable indicating when large relative movements have historically been more frequent. It should not be used as a standalone trading signal.

### Next test

A future project should test whether variables such as trading volume, current Bollinger Band width, previous price movement, weekday, and market regime improve prediction beyond UTC hour alone.