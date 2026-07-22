# Data Dictionary

## Unit of Analysis

Each row represents one completed and usable one-hour BTCUSDT candle from Binance, identified by its UTC opening time.

| Column | Data Type | Unit | Meaning | Used in Project |
|---|---|---|---|---|
| open_time | datetime | UTC | The UTC date and time when the one-hour candle begins. | Yes — hour grouping |
| open | float | USDT per BTC | The BTC price at the beginning of the one-hour candle. | Yes — validation |
| high | float | USDT per BTC | The highest BTC price reached during the one-hour candle. | Yes — candle range |
| low | float | USDT per BTC | The lowest BTC price reached during the one-hour candle. | Yes — candle range |
| close | float | USDT per BTC | The BTC price at the end of the one-hour candle. | Yes — Bollinger Bands |
| volume | float | BTC | The total amount of BTC traded during the one-hour candle. | Yes — data-quality checks |
| close_time | datetime | UTC | The UTC date and time when the candle ends. | Yes — candle-duration validation |
| quote_volume | float | USDT | The total USDT value traded during the one-hour candle. | No |
| number_of_trades | integer | Trades | The total number of trades recorded during the one-hour candle. | Yes — data-quality checks |
| taker_buy_base_volume | float | BTC | The amount of BTC bought through taker buy orders during the candle. | No |
| taker_buy_quote_volume | float | USDT | The USDT value of BTC bought through taker buy orders during the candle. | No |
| duration_ms | float | Milliseconds | The measured duration between the candle opening and closing timestamps. | Yes — data cleaning |