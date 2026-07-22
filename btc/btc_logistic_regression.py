from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# SETTINGS
# ============================================================

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

CSV_PATH = BASE_DIR / "data" / "btcusdt_1h_bb_pvo.csv"
OUTPUT_DIRECTORY = BASE_DIR / "outputs"
TIME_COLUMN = "open_time"
CLOSE_COLUMN = "close"
VOLUME_COLUMN = "volume"

# TradingView-style Bollinger Bands
BB_LENGTH = 20
BB_MULTIPLIER = 2.0

# TradingView-style PVO
PVO_FAST_LENGTH = 12
PVO_SLOW_LENGTH = 26
PVO_SIGNAL_LENGTH = 9

# Example:
# If each candle is 30 minutes and HORIZON = 2,
# the model predicts approximately one hour ahead.
HORIZON = 2

# Target = 1 when future return is above this amount.
# 0.002 means +0.20%.
TARGET_RETURN_THRESHOLD = 0.002

# Final 20% of the data is reserved for testing.
TEST_RATIO = 0.20

# Model enters a trade only when probability is at least 55%.
BUY_PROBABILITY_THRESHOLD = 0.55

# Example total entry + exit cost.
# Replace this with your actual fees and estimated slippage.
ROUND_TRIP_COST = 0.001

MINIMUM_REQUIRED_ROWS = 300


# ============================================================
# DATA LOADING
# ============================================================

def parse_timestamp(series: pd.Series) -> pd.Series:
    """
    Parse string, second-based, or millisecond-based timestamps.
    """
    numeric_series = pd.to_numeric(series, errors="coerce")

    if numeric_series.notna().mean() > 0.90:
        maximum_value = numeric_series.max()

        if maximum_value > 10_000_000_000:
            return pd.to_datetime(
                numeric_series,
                unit="ms",
                utc=True,
                errors="coerce",
            )

        return pd.to_datetime(
            numeric_series,
            unit="s",
            utc=True,
            errors="coerce",
        )

    return pd.to_datetime(series, utc=True, errors="coerce")


def load_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file was not found: {csv_path.resolve()}\n"
            "Place your file inside data/btc.csv or change CSV_PATH."
        )

    df = pd.read_csv(csv_path)

    # Make column names consistent.
    df.columns = [
        str(column).strip().lower().replace(" ", "_")
        for column in df.columns
    ]

    required_columns = {
        TIME_COLUMN,
        CLOSE_COLUMN,
        VOLUME_COLUMN,
    }

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}\n"
            f"Available columns: {list(df.columns)}"
        )

    df[TIME_COLUMN] = parse_timestamp(df[TIME_COLUMN])

    numeric_columns = [
        column
        for column in ["open", "high", "low", "close", "volume"]
        if column in df.columns
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = (
        df.dropna(subset=[TIME_COLUMN, CLOSE_COLUMN, VOLUME_COLUMN])
        .sort_values(TIME_COLUMN)
        .drop_duplicates(subset=[TIME_COLUMN], keep="last")
        .reset_index(drop=True)
    )

    if (df[CLOSE_COLUMN] <= 0).any():
        raise ValueError("The close column contains zero or negative values.")

    if (df[VOLUME_COLUMN] < 0).any():
        raise ValueError("The volume column contains negative values.")

    return df


# ============================================================
# INDICATOR CALCULATION
# ============================================================

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    close = df[CLOSE_COLUMN]
    volume = df[VOLUME_COLUMN]

    # --------------------------------------------------------
    # Bollinger Bands: 20-period SMA and 2 standard deviations
    # ddof=0 uses population standard deviation.
    # --------------------------------------------------------

    df["bb_basis"] = close.rolling(
        window=BB_LENGTH,
        min_periods=BB_LENGTH,
    ).mean()

    df["bb_std"] = close.rolling(
        window=BB_LENGTH,
        min_periods=BB_LENGTH,
    ).std(ddof=0)

    df["bb_upper"] = (
        df["bb_basis"] + BB_MULTIPLIER * df["bb_std"]
    )

    df["bb_lower"] = (
        df["bb_basis"] - BB_MULTIPLIER * df["bb_std"]
    )

    band_range = (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
    safe_basis = df["bb_basis"].replace(0, np.nan)

    # 0 means lower band, 0.5 means basis, 1 means upper band.
    df["bb_position"] = (
        (close - df["bb_lower"]) / band_range
    )

    # Relative band width.
    df["bb_width"] = band_range / safe_basis

    # Distance from the Bollinger basis.
    df["bb_basis_distance"] = (
        close / safe_basis - 1
    )

    # --------------------------------------------------------
    # Percentage Volume Oscillator
    # PVO = 100 × (fast volume EMA - slow volume EMA)
    #             / slow volume EMA
    # --------------------------------------------------------

    df["volume_ema_fast"] = volume.ewm(
        span=PVO_FAST_LENGTH,
        adjust=False,
        min_periods=PVO_FAST_LENGTH,
    ).mean()

    df["volume_ema_slow"] = volume.ewm(
        span=PVO_SLOW_LENGTH,
        adjust=False,
        min_periods=PVO_SLOW_LENGTH,
    ).mean()

    safe_volume_ema_slow = df["volume_ema_slow"].replace(0, np.nan)

    df["pvo"] = (
        100
        * (
            df["volume_ema_fast"]
            - df["volume_ema_slow"]
        )
        / safe_volume_ema_slow
    )

    df["pvo_signal"] = df["pvo"].ewm(
        span=PVO_SIGNAL_LENGTH,
        adjust=False,
        min_periods=PVO_SIGNAL_LENGTH,
    ).mean()

    df["pvo_histogram"] = (
        df["pvo"] - df["pvo_signal"]
    )

    return df


# ============================================================
# TARGET CREATION
# ============================================================

def create_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Future price after HORIZON candles.
    df["future_close"] = df[CLOSE_COLUMN].shift(-HORIZON)

    # Return between the current close and future close.
    df["future_return"] = (
        df["future_close"] / df[CLOSE_COLUMN] - 1
    )

    # Do not incorrectly label the final rows, because their
    # future prices do not exist.
    df["target"] = np.where(
        df["future_return"].notna(),
        (
            df["future_return"]
            > TARGET_RETURN_THRESHOLD
        ).astype(int),
        np.nan,
    )

    return df


# ============================================================
# MODEL EVALUATION
# ============================================================

def calculate_max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return np.nan

    running_maximum = equity.cummax()
    drawdown = equity / running_maximum - 1

    return float(drawdown.min())


def print_model_metrics(
    y_train: pd.Series,
    y_test: pd.Series,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    majority_class = int(y_train.mean() >= 0.50)
    baseline_predictions = np.full(
        shape=len(y_test),
        fill_value=majority_class,
        dtype=int,
    )

    print("\n" + "=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)

    print(f"Training rows: {len(y_train):,}")
    print(f"Testing rows:  {len(y_test):,}")

    print(
        "Training positive-target percentage: "
        f"{y_train.mean():.2%}"
    )

    print(
        "Testing positive-target percentage:  "
        f"{y_test.mean():.2%}"
    )

    print("\n" + "=" * 60)
    print("CLASSIFICATION RESULTS")
    print("=" * 60)

    print(
        f"Majority baseline accuracy: "
        f"{accuracy_score(y_test, baseline_predictions):.4f}"
    )

    print(
        f"Model accuracy:           "
        f"{accuracy_score(y_test, predictions):.4f}"
    )

    print(
        f"Balanced accuracy:        "
        f"{balanced_accuracy_score(y_test, predictions):.4f}"
    )

    print(
        f"Precision:                "
        f"{precision_score(y_test, predictions, zero_division=0):.4f}"
    )

    print(
        f"Recall:                   "
        f"{recall_score(y_test, predictions, zero_division=0):.4f}"
    )

    print(
        f"F1 score:                 "
        f"{f1_score(y_test, predictions, zero_division=0):.4f}"
    )

    if y_test.nunique() == 2:
        print(
            f"ROC AUC:                  "
            f"{roc_auc_score(y_test, probabilities):.4f}"
        )
    else:
        print("ROC AUC:                  Cannot calculate; one class only.")

    print("\nConfusion matrix:")
    print("[[true negatives, false positives],")
    print(" [false negatives, true positives]]")

    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1],
        )
    )

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            digits=4,
            zero_division=0,
        )
    )


# ============================================================
# SIMPLE NON-OVERLAPPING BACKTEST
# ============================================================

def run_backtest(results: pd.DataFrame) -> pd.DataFrame:
    """
    Uses every HORIZON-th row so positions do not overlap.

    prediction = 1: enter a long trade
    prediction = 0: remain in cash
    """
    trades = results.iloc[::HORIZON].copy()

    trades["strategy_return"] = np.where(
        trades["prediction"] == 1,
        trades["future_return"] - ROUND_TRIP_COST,
        0.0,
    )

    trades["buy_hold_return"] = trades["future_return"]

    trades["strategy_equity"] = (
        1 + trades["strategy_return"]
    ).cumprod()

    trades["buy_hold_equity"] = (
        1 + trades["buy_hold_return"]
    ).cumprod()

    executed_trades = trades.loc[
        trades["prediction"] == 1
    ].copy()

    print("\n" + "=" * 60)
    print("BASIC BACKTEST")
    print("=" * 60)

    print(f"Non-overlapping periods: {len(trades):,}")
    print(f"Executed long trades:    {len(executed_trades):,}")

    if not executed_trades.empty:
        profitable_trades = (
            executed_trades["strategy_return"] > 0
        ).mean()

        print(f"Trade win rate:          {profitable_trades:.2%}")
        print(
            "Average trade return:    "
            f"{executed_trades['strategy_return'].mean():.4%}"
        )
    else:
        print("Trade win rate:          No trades")
        print("Average trade return:    No trades")

    strategy_total_return = (
        trades["strategy_equity"].iloc[-1] - 1
    )

    buy_hold_total_return = (
        trades["buy_hold_equity"].iloc[-1] - 1
    )

    strategy_max_drawdown = calculate_max_drawdown(
        trades["strategy_equity"]
    )

    buy_hold_max_drawdown = calculate_max_drawdown(
        trades["buy_hold_equity"]
    )

    print(
        f"Strategy total return:   "
        f"{strategy_total_return:.2%}"
    )

    print(
        f"Buy-and-hold return:     "
        f"{buy_hold_total_return:.2%}"
    )

    print(
        f"Strategy max drawdown:   "
        f"{strategy_max_drawdown:.2%}"
    )

    print(
        f"Buy-and-hold drawdown:   "
        f"{buy_hold_max_drawdown:.2%}"
    )

    return trades


# ============================================================
# PLOTTING
# ============================================================

def plot_results(trades: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 6))

    plt.plot(
        trades[TIME_COLUMN],
        trades["strategy_equity"],
        label="Logistic regression strategy",
    )

    plt.plot(
        trades[TIME_COLUMN],
        trades["buy_hold_equity"],
        label="Buy and hold",
    )

    plt.title("Out-of-Sample Equity Curve")
    plt.xlabel("Time")
    plt.ylabel("Growth of 1 unit")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_path = OUTPUT_DIRECTORY / "equity_curve.png"
    plt.savefig(output_path, dpi=150)
    plt.show()

    print(f"\nEquity chart saved to: {output_path.resolve()}")


# ============================================================
# MAIN PROGRAM
# ============================================================

def main() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    print(f"Reading data from: {CSV_PATH.resolve()}")

    df = load_data(CSV_PATH)
    df = calculate_indicators(df)
    df = create_target(df)

    features = [
        "bb_position",
        "bb_width",
        "bb_basis_distance",
        "pvo",
        "pvo_signal",
        "pvo_histogram",
    ]

    model_data = df.dropna(
        subset=features + ["target", "future_return"]
    ).copy()

    model_data["target"] = model_data["target"].astype(int)

    if len(model_data) < MINIMUM_REQUIRED_ROWS:
        raise ValueError(
            f"Only {len(model_data)} usable rows remain.\n"
            f"At least {MINIMUM_REQUIRED_ROWS} rows are recommended."
        )

    split_index = int(
        len(model_data) * (1 - TEST_RATIO)
    )

    train_data = model_data.iloc[:split_index].copy()
    test_data = model_data.iloc[split_index:].copy()

    if train_data.empty or test_data.empty:
        raise ValueError("Training or testing dataset is empty.")

    if train_data["target"].nunique() < 2:
        raise ValueError(
            "The training target contains only one class.\n"
            "Reduce TARGET_RETURN_THRESHOLD or use more data."
        )

    X_train = train_data[features]
    y_train = train_data["target"]

    X_test = test_data[features]
    y_test = test_data["target"]

    # StandardScaler is fitted only on training data because it
    # is inside the model pipeline.
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logistic_regression",
                LogisticRegression(
                    max_iter=2_000,
                    solver="lbfgs",
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]

    predictions = (
        probabilities >= BUY_PROBABILITY_THRESHOLD
    ).astype(int)

    print_model_metrics(
        y_train=y_train,
        y_test=y_test,
        predictions=predictions,
        probabilities=probabilities,
    )

    # --------------------------------------------------------
    # Examine the learned logistic-regression coefficients.
    # Positive coefficient:
    # higher feature values increase predicted probability.
    # --------------------------------------------------------

    logistic_model = model.named_steps["logistic_regression"]

    coefficient_table = pd.DataFrame(
        {
            "feature": features,
            "coefficient": logistic_model.coef_[0],
        }
    )

    coefficient_table["absolute_coefficient"] = (
        coefficient_table["coefficient"].abs()
    )

    coefficient_table = coefficient_table.sort_values(
        "absolute_coefficient",
        ascending=False,
    )

    print("\n" + "=" * 60)
    print("MODEL COEFFICIENTS")
    print("=" * 60)

    print(
        coefficient_table[
            ["feature", "coefficient"]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # Store test predictions.
    # --------------------------------------------------------

    results = test_data[
        [
            TIME_COLUMN,
            CLOSE_COLUMN,
            "future_close",
            "future_return",
            "target",
        ]
    ].copy()

    results["probability"] = probabilities
    results["prediction"] = predictions

    prediction_path = (
        OUTPUT_DIRECTORY / "test_predictions.csv"
    )

    results.to_csv(prediction_path, index=False)

    coefficient_path = (
        OUTPUT_DIRECTORY / "model_coefficients.csv"
    )

    coefficient_table.to_csv(
        coefficient_path,
        index=False,
    )

    trades = run_backtest(results)

    trade_path = OUTPUT_DIRECTORY / "backtest_trades.csv"
    trades.to_csv(trade_path, index=False)

    print("\n" + "=" * 60)
    print("OUTPUT FILES")
    print("=" * 60)

    print(f"Predictions:  {prediction_path.resolve()}")
    print(f"Trades:       {trade_path.resolve()}")
    print(f"Coefficients: {coefficient_path.resolve()}")

    plot_results(trades)


if __name__ == "__main__":
    main()