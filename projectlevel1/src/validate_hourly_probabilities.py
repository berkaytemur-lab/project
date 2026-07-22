from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)


# ============================================================
# FILE PATHS
# ============================================================

PROJECT_FOLDER = Path(__file__).resolve().parents[1]

FEATURE_FILE = (
    PROJECT_FOLDER
    / "data"
    / "processed"
    / "btcusdt_1h_features.csv"
)

MODEL_PROBABILITY_FILE = (
    PROJECT_FOLDER
    / "reports"
    / "logistic_probability_by_hour.csv"
)

COMPARISON_FILE = (
    PROJECT_FOLDER
    / "reports"
    / "hourly_probability_validation_2025.csv"
)

EVALUATION_FILE = (
    PROJECT_FOLDER
    / "reports"
    / "hourly_probability_metrics.csv"
)

FIGURE_FOLDER = (
    PROJECT_FOLDER
    / "reports"
    / "figures"
)

COMPARISON_FIGURE = (
    FIGURE_FOLDER
    / "predicted_vs_actual_hourly_rates_2025.png"
)


# ============================================================
# VALIDATION SETTINGS
# ============================================================

TRAIN_END_YEAR = 2024
TEST_YEAR = 2025


# ============================================================
# LOAD FEATURE DATA
# ============================================================

def load_feature_data() -> pd.DataFrame:
    """
    Load the feature dataset and restore required data types.
    """

    df = pd.read_csv(FEATURE_FILE)

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        format="ISO8601",
        utc=True,
    )

    df["movement_event"] = pd.to_numeric(
        df["movement_event"],
        errors="coerce",
    )

    df["utc_hour"] = pd.to_numeric(
        df["utc_hour"],
        errors="raise",
    ).astype(int)

    df["year"] = pd.to_numeric(
        df["year"],
        errors="raise",
    ).astype(int)

    if df["gap_safe_movement_row"].dtype == "object":
        df["gap_safe_movement_row"] = (
            df["gap_safe_movement_row"]
            .astype(str)
            .str.lower()
            .map(
                {
                    "true": True,
                    "false": False,
                }
            )
        )

    df["gap_safe_movement_row"] = (
        df["gap_safe_movement_row"]
        .fillna(False)
        .astype(bool)
    )

    return df


# ============================================================
# LOAD MODEL PROBABILITIES
# ============================================================

def load_model_probabilities() -> pd.DataFrame:
    """
    Load hourly probabilities estimated by the logistic
    regression trained on 2021-2024 data.
    """

    probability_df = pd.read_csv(
        MODEL_PROBABILITY_FILE
    )

    probability_df["utc_hour"] = pd.to_numeric(
        probability_df["utc_hour"],
        errors="raise",
    ).astype(int)

    probability_df["predicted_probability"] = (
        pd.to_numeric(
            probability_df["predicted_probability"],
            errors="raise",
        )
    )

    return probability_df[
        [
            "utc_hour",
            "predicted_probability",
        ]
    ].copy()


# ============================================================
# SELECT GAP-SAFE DATA
# ============================================================

def select_analysis_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select rows with valid, gap-safe movement labels.
    """

    analysis_df = df.loc[
        df["gap_safe_movement_row"]
        & df["movement_event"].notna()
    ].copy()

    analysis_df["movement_event"] = (
        analysis_df["movement_event"]
        .astype(int)
    )

    return analysis_df


# ============================================================
# CALCULATE ACTUAL 2025 HOURLY RATES
# ============================================================

def create_actual_test_rates(
    analysis_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate actual movement-event rates for every UTC
    hour in the unseen 2025 test period.
    """

    test_df = analysis_df.loc[
        analysis_df["year"] == TEST_YEAR
    ].copy()

    if test_df.empty:
        raise ValueError(
            "The 2025 test dataset is empty."
        )

    actual_rates = (
        test_df
        .groupby("utc_hour")
        .agg(
            test_observations=(
                "movement_event",
                "size",
            ),
            test_movement_events=(
                "movement_event",
                "sum",
            ),
            actual_event_rate=(
                "movement_event",
                "mean",
            ),
        )
        .reset_index()
    )

    return actual_rates


# ============================================================
# CALCULATE TRAINING BASELINE
# ============================================================

def calculate_training_baseline(
    analysis_df: pd.DataFrame,
) -> float:
    """
    Use the overall 2021-2024 movement-event rate as the
    constant probability baseline.
    """

    train_df = analysis_df.loc[
        analysis_df["year"] <= TRAIN_END_YEAR
    ].copy()

    if train_df.empty:
        raise ValueError(
            "The training dataset is empty."
        )

    baseline_probability = float(
        train_df["movement_event"].mean()
    )

    return baseline_probability


# ============================================================
# CREATE COMPARISON TABLE
# ============================================================

def create_comparison(
    actual_rates: pd.DataFrame,
    model_probabilities: pd.DataFrame,
    baseline_probability: float,
) -> pd.DataFrame:
    """
    Compare actual 2025 rates with the logistic model and
    constant baseline probabilities.
    """

    comparison = actual_rates.merge(
        model_probabilities,
        on="utc_hour",
        how="inner",
    )

    comparison["baseline_probability"] = (
        baseline_probability
    )

    comparison["actual_event_rate_percent"] = (
        comparison["actual_event_rate"] * 100
    )

    comparison["model_probability_percent"] = (
        comparison["predicted_probability"] * 100
    )

    comparison["baseline_probability_percent"] = (
        comparison["baseline_probability"] * 100
    )

    comparison["model_error_percentage_points"] = (
        comparison["model_probability_percent"]
        - comparison["actual_event_rate_percent"]
    )

    comparison[
        "baseline_error_percentage_points"
    ] = (
        comparison["baseline_probability_percent"]
        - comparison["actual_event_rate_percent"]
    )

    comparison["absolute_model_error"] = (
        comparison[
            "model_error_percentage_points"
        ].abs()
    )

    comparison["absolute_baseline_error"] = (
        comparison[
            "baseline_error_percentage_points"
        ].abs()
    )

    comparison = comparison.sort_values(
        "utc_hour"
    ).reset_index(drop=True)

    return comparison


# ============================================================
# CALCULATE VALIDATION METRICS
# ============================================================

def create_evaluation_metrics(
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate MAE, RMSE, and correlation across the 24
    UTC-hour groups.
    """

    actual = comparison[
        "actual_event_rate_percent"
    ]

    model_prediction = comparison[
        "model_probability_percent"
    ]

    baseline_prediction = comparison[
        "baseline_probability_percent"
    ]

    model_mae = mean_absolute_error(
        actual,
        model_prediction,
    )

    baseline_mae = mean_absolute_error(
        actual,
        baseline_prediction,
    )

    model_rmse = np.sqrt(
        mean_squared_error(
            actual,
            model_prediction,
        )
    )

    baseline_rmse = np.sqrt(
        mean_squared_error(
            actual,
            baseline_prediction,
        )
    )

    model_correlation = actual.corr(
        model_prediction
    )

    metrics = {
        "model_mae_percentage_points": model_mae,
        "baseline_mae_percentage_points": baseline_mae,
        "model_rmse_percentage_points": model_rmse,
        "baseline_rmse_percentage_points": baseline_rmse,
        "mae_improvement_percentage_points": (
            baseline_mae - model_mae
        ),
        "rmse_improvement_percentage_points": (
            baseline_rmse - model_rmse
        ),
        "model_actual_correlation": model_correlation,
    }

    return pd.DataFrame(
        {
            "metric": metrics.keys(),
            "value": metrics.values(),
        }
    )


# ============================================================
# CREATE COMPARISON FIGURE
# ============================================================

def create_comparison_figure(
    comparison: pd.DataFrame,
) -> None:
    """
    Compare actual 2025 event rates with model and baseline
    probabilities for every UTC hour.
    """

    FIGURE_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(
        figsize=(13, 7)
    )

    plt.plot(
        comparison["utc_hour"],
        comparison["actual_event_rate_percent"],
        marker="o",
        linewidth=2,
        label="Actual 2025 event rate",
    )

    plt.plot(
        comparison["utc_hour"],
        comparison["model_probability_percent"],
        marker="o",
        linewidth=2,
        label="Logistic prediction from 2021-2024",
    )

    plt.plot(
        comparison["utc_hour"],
        comparison["baseline_probability_percent"],
        linestyle="--",
        linewidth=2,
        label="Constant training-rate baseline",
    )

    plt.title(
        "Predicted vs Actual BTC Movement Rates by UTC Hour\n"
        "Training: 2021-2024 | Test: 2025"
    )

    plt.xlabel("UTC Hour")
    plt.ylabel("Movement-Event Rate (%)")

    plt.xticks(
        range(24),
        [
            f"{hour:02d}"
            for hour in range(24)
        ],
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        COMPARISON_FIGURE,
        dpi=200,
    )

    plt.close()


# ============================================================
# SAVE REPORTS
# ============================================================

def save_reports(
    comparison: pd.DataFrame,
    evaluation: pd.DataFrame,
) -> None:
    """
    Save the validation comparison and evaluation metrics.
    """

    COMPARISON_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_csv(
        COMPARISON_FILE,
        index=False,
    )

    evaluation.to_csv(
        EVALUATION_FILE,
        index=False,
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    df = load_feature_data()

    analysis_df = select_analysis_data(df)

    model_probabilities = (
        load_model_probabilities()
    )

    actual_rates = create_actual_test_rates(
        analysis_df
    )

    baseline_probability = (
        calculate_training_baseline(
            analysis_df
        )
    )

    comparison = create_comparison(
        actual_rates=actual_rates,
        model_probabilities=model_probabilities,
        baseline_probability=baseline_probability,
    )

    evaluation = create_evaluation_metrics(
        comparison
    )

    save_reports(
        comparison=comparison,
        evaluation=evaluation,
    )

    create_comparison_figure(
        comparison
    )

    print("HOURLY PROBABILITY VALIDATION")

    display_columns = [
        "utc_hour",
        "test_observations",
        "actual_event_rate_percent",
        "model_probability_percent",
        "baseline_probability_percent",
        "absolute_model_error",
        "absolute_baseline_error",
    ]

    display_comparison = comparison[
        display_columns
    ].copy()

    decimal_columns = [
        "actual_event_rate_percent",
        "model_probability_percent",
        "baseline_probability_percent",
        "absolute_model_error",
        "absolute_baseline_error",
    ]

    display_comparison[decimal_columns] = (
        display_comparison[decimal_columns]
        .round(4)
    )

    print(
        display_comparison.to_string(
            index=False
        )
    )

    print("\nVALIDATION METRICS")

    display_evaluation = evaluation.copy()

    display_evaluation["value"] = (
        display_evaluation["value"]
        .round(4)
    )

    print(
        display_evaluation.to_string(
            index=False
        )
    )

    model_best_hour = int(
        comparison.loc[
            comparison[
                "model_probability_percent"
            ].idxmax(),
            "utc_hour",
        ]
    )

    actual_best_hour = int(
        comparison.loc[
            comparison[
                "actual_event_rate_percent"
            ].idxmax(),
            "utc_hour",
        ]
    )

    print(
        "\nModel's highest-risk hour:",
        f"{model_best_hour:02d}:00 UTC",
    )

    print(
        "Actual highest-risk hour in 2025:",
        f"{actual_best_hour:02d}:00 UTC",
    )

    print(
        f"\nComparison saved to:\n"
        f"{COMPARISON_FILE}"
    )

    print(
        f"\nMetrics saved to:\n"
        f"{EVALUATION_FILE}"
    )

    print(
        f"\nFigure saved to:\n"
        f"{COMPARISON_FIGURE}"
    )


if __name__ == "__main__":
    main()