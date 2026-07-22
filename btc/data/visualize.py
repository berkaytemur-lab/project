from pathlib import Path

import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

# Choose: "15m", "1h", or "1d"
INTERVAL = "1h"

# Number of candles displayed
CANDLES_TO_SHOW = 50

# TradingView: Lock price to bar ratio
PRICE_TO_BAR_RATIO = 328.0


# visualize.py and CSV files are in the same folder
SCRIPT_FOLDER = Path(__file__).resolve().parent

CSV_FILE = (
    SCRIPT_FOLDER
    / f"btcusdt_{INTERVAL}_bb_pvo.csv"
)


# ============================================================
# LOAD CSV DATA
# ============================================================

def load_data() -> pd.DataFrame:
    if not CSV_FILE.exists():
        raise FileNotFoundError(
            f"CSV file was not found:\n{CSV_FILE}"
        )

    df = pd.read_csv(
        CSV_FILE,
        parse_dates=["open_time"],
    )

    required_columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "bb_basis",
        "bb_upper",
        "bb_lower",
        "pvo",
        "pvo_signal",
        "pvo_histogram",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing CSV columns: {missing_columns}"
        )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "bb_basis",
        "bb_upper",
        "bb_lower",
        "pvo",
        "pvo_signal",
        "pvo_histogram",
    ]

    df[numeric_columns] = df[numeric_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "open",
            "high",
            "low",
            "close",
            "bb_basis",
            "bb_upper",
            "bb_lower",
            "pvo",
            "pvo_signal",
            "pvo_histogram",
        ]
    )

    df = df.sort_values("open_time")

    df = df.tail(CANDLES_TO_SHOW).copy()

    # mplfinance requires datetime index
    df = df.set_index("open_time")

    return df


# ============================================================
# PVO HISTOGRAM COLORS
# ============================================================

def create_histogram_colors(
    df: pd.DataFrame,
) -> list[str]:

    colors = []

    histogram = df["pvo_histogram"]

    for index in range(len(histogram)):
        current_value = histogram.iloc[index]

        if index == 0:
            previous_value = current_value
        else:
            previous_value = histogram.iloc[index - 1]

        if current_value >= 0:
            if current_value > previous_value:
                colors.append("#26a69a")
            else:
                colors.append("#b2dfdb")

        else:
            if current_value > previous_value:
                colors.append("#ffcdd2")
            else:
                colors.append("#ff5252")

    return colors


# ============================================================
# CREATE CHART
# ============================================================

def plot_chart(df: pd.DataFrame) -> None:
    histogram_colors = create_histogram_colors(df)

    additional_plots = [
        # Bollinger upper band
        mpf.make_addplot(
            df["bb_upper"],
            panel=0,
            color="#f23645",
            width=1.2,
        ),

        # Bollinger basis
        mpf.make_addplot(
            df["bb_basis"],
            panel=0,
            color="#2962ff",
            width=1.2,
        ),

        # Bollinger lower band
        mpf.make_addplot(
            df["bb_lower"],
            panel=0,
            color="#089981",
            width=1.2,
        ),

        # PVO histogram
        mpf.make_addplot(
            df["pvo_histogram"],
            panel=1,
            type="bar",
            color=histogram_colors,
            alpha=0.85,
            ylabel="PVO %",
        ),

        # PVO line
        mpf.make_addplot(
            df["pvo"],
            panel=1,
            color="#2962ff",
            width=1.0,
        ),

        # PVO signal line
        mpf.make_addplot(
            df["pvo_signal"],
            panel=1,
            color="#ff6d00",
            width=1.0,
        ),
    ]

    market_colors = mpf.make_marketcolors(
        up="#089981",
        down="#f23645",
        edge="inherit",
        wick="inherit",
    )

    chart_style = mpf.make_mpf_style(
        base_mpf_style="nightclouds",
        marketcolors=market_colors,
        gridstyle="--",
        gridcolor="#292929",
        facecolor="#000000",
        figcolor="#000000",
    )

    figure, axes = mpf.plot(
        df,
        type="candle",
        style=chart_style,
        addplot=additional_plots,
        panel_ratios=(3, 1),
        figsize=(16, 9),
        title=(
            f"BTCUSDT {INTERVAL} — "
            f"Bollinger Bands and PVO"
        ),
        ylabel="Price",
        volume=False,
        datetime_format="%Y-%m-%d %H:%M",
        xrotation=20,
        tight_layout=True,
        returnfig=True,
    )

    price_axis = axes[0]
    pvo_axis = axes[2]

    # Fill Bollinger Band area
    price_axis.fill_between(
        range(len(df)),
        df["bb_lower"].to_numpy(),
        df["bb_upper"].to_numpy(),
        alpha=0.12,
        color="#d4b800",
    )

    # TradingView-style locked price-to-bar ratio
    price_axis.set_aspect(
        1 / PRICE_TO_BAR_RATIO,
        adjustable="datalim",
        anchor="C",
    )

    # PVO zero line
    pvo_axis.axhline(
        y=0,
        linestyle="--",
        linewidth=0.8,
        alpha=0.7,
        color="#787b86",
    )

    plt.show()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    try:
        df = load_data()

        print(f"Reading: {CSV_FILE}")
        print(f"Interval: {INTERVAL}")
        print(f"Showing candles: {len(df)}")
        print(
            f"Price-to-bar ratio: "
            f"{PRICE_TO_BAR_RATIO}"
        )

        plot_chart(df)

    except Exception as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()