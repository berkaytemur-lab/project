from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


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

EVALUATION_FILE = (
    PROJECT_FOLDER
    / "reports"
    / "logistic_regression_evaluation.csv"
)

HOURLY_PROBABILITY_FILE = (
    PROJECT_FOLDER
    / "reports"
    / "logistic_probability_by_hour.csv"
)

FIGURE_FOLDER = (
    PROJECT_FOLDER
    / "reports"
    / "figures"
)

CONFUSION_MATRIX_FILE = (
    FIGURE_FOLDER
    / "logistic_regression_confusion_matrix.png"
)

PROBABILITY_FIGURE_FILE = (
    FIGURE_FOLDER
    / "logistic_probability_by_utc_hour.png"
)


# ============================================================
# MODEL SETTINGS
# ============================================================

TRAIN_END_YEAR = 2024
TEST_YEAR = 2025

CLASSIFICATION_THRESHOLD = 0.50


# ============================================================
# LOAD FEATURE DATA
# ============================================================

def load_feature_data() -> pd.DataFrame:
    """
    Load the feature dataset and restore the required
    data types.
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
# SELECT ANALYSIS DATA
# ============================================================

def select_gap_safe_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select rows with valid movement labels whose previous
    Bollinger Band window was not affected by a time gap.
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
# CREATE TIME-BASED TRAIN-TEST SPLIT
# ============================================================

def create_train_test_split(
    analysis_df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """
    Use 2021–2024 for training and 2025 for testing.
    """

    train_df = analysis_df.loc[
        analysis_df["year"] <= TRAIN_END_YEAR
    ].copy()

    test_df = analysis_df.loc[
        analysis_df["year"] == TEST_YEAR
    ].copy()

    if train_df.empty:
        raise ValueError(
            "The training dataset is empty."
        )

    if test_df.empty:
        raise ValueError(
            "The test dataset is empty."
        )

    if train_df["open_time"].max() >= test_df["open_time"].min():
        raise ValueError(
            "Training and test periods overlap."
        )

    X_train = train_df[
        [
            "utc_hour",
        ]
    ]

    y_train = train_df[
        "movement_event"
    ]

    X_test = test_df[
        [
            "utc_hour",
        ]
    ]

    y_test = test_df[
        "movement_event"
    ]

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


# ============================================================
# CREATE MODEL
# ============================================================

def create_model() -> Pipeline:
    """
    Create a logistic-regression pipeline.

    UTC hour is treated as a categorical variable using
    one-hot encoding.
    """

    model = Pipeline(
        steps=[
            (
                "hour_encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
            ),
            (
                "logistic_regression",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    return model


# ============================================================
# CREATE PREDICTIONS
# ============================================================

def create_predictions(
    model: Pipeline,
    X_test: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """
    Generate movement probabilities and binary predictions.
    """

    probabilities = pd.Series(
        model.predict_proba(X_test)[:, 1],
        index=X_test.index,
        name="movement_probability",
    )

    predictions = (
        probabilities
        >= CLASSIFICATION_THRESHOLD
    ).astype(int)

    predictions.name = "predicted_event"

    return probabilities, predictions


# ============================================================
# CALCULATE BASELINE
# ============================================================

def calculate_baseline_accuracy(
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[int, float]:
    """
    Calculate the majority-class baseline using only the
    training data.
    """

    majority_class = int(
        y_train.mode().iloc[0]
    )

    baseline_predictions = pd.Series(
        majority_class,
        index=y_test.index,
        dtype="int64",
    )

    baseline_accuracy = accuracy_score(
        y_test,
        baseline_predictions,
    )

    return (
        majority_class,
        baseline_accuracy,
    )


# ============================================================
# CREATE EVALUATION REPORT
# ============================================================

def create_evaluation_report(
    y_train: pd.Series,
    y_test: pd.Series,
    predictions: pd.Series,
) -> pd.DataFrame:
    """
    Calculate logistic-regression evaluation metrics.
    """

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[
            0,
            1,
        ],
    )

    true_negative = int(matrix[0, 0])
    false_positive = int(matrix[0, 1])
    false_negative = int(matrix[1, 0])
    true_positive = int(matrix[1, 1])

    model_accuracy = accuracy_score(
        y_test,
        predictions,
    )

    majority_class, baseline_accuracy = (
        calculate_baseline_accuracy(
            y_train=y_train,
            y_test=y_test,
        )
    )

    results = {
        "training_rows": len(y_train),
        "test_rows": len(y_test),
        "training_event_rate_percent": (
            y_train.mean() * 100
        ),
        "test_event_rate_percent": (
            y_test.mean() * 100
        ),
        "classification_threshold": (
            CLASSIFICATION_THRESHOLD
        ),
        "baseline_majority_class": majority_class,
        "baseline_accuracy_percent": (
            baseline_accuracy * 100
        ),
        "model_accuracy_percent": (
            model_accuracy * 100
        ),
        "accuracy_improvement_percentage_points": (
            model_accuracy
            - baseline_accuracy
        ) * 100,
        "predicted_movement_events": int(
            predictions.sum()
        ),
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_positive": true_positive,
    }

    return pd.DataFrame(
        {
            "metric": results.keys(),
            "value": results.values(),
        }
    )


# ============================================================
# CREATE HOURLY PROBABILITY REPORT
# ============================================================

def create_hourly_probability_report(
    model: Pipeline,
) -> pd.DataFrame:
    """
    Estimate movement-event probability for every UTC hour.
    """

    hour_df = pd.DataFrame(
        {
            "utc_hour": range(24),
        }
    )

    hour_df["predicted_probability"] = (
        model.predict_proba(hour_df)[:, 1]
    )

    hour_df[
        "predicted_probability_percent"
    ] = (
        hour_df["predicted_probability"]
        * 100
    )

    hour_df["predicted_class_at_0_50"] = (
        hour_df["predicted_probability"]
        >= CLASSIFICATION_THRESHOLD
    ).astype(int)

    hour_df["probability_rank"] = (
        hour_df[
            "predicted_probability"
        ]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    return hour_df


# ============================================================
# CREATE CONFUSION-MATRIX FIGURE
# ============================================================

def create_confusion_matrix_figure(
    evaluation_df: pd.DataFrame,
) -> None:
    """
    Create the logistic-regression confusion matrix.
    """

    values = dict(
        zip(
            evaluation_df["metric"],
            evaluation_df["value"],
        )
    )

    matrix = [
        [
            values["true_negative"],
            values["false_positive"],
        ],
        [
            values["false_negative"],
            values["true_positive"],
        ],
    ]

    FIGURE_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(
        figsize=(7, 6)
    )

    image = plt.imshow(matrix)

    plt.colorbar(
        image,
        label="Number of Test Candles",
    )

    plt.xticks(
        [
            0,
            1,
        ],
        [
            "Predicted 0",
            "Predicted 1",
        ],
    )

    plt.yticks(
        [
            0,
            1,
        ],
        [
            "Actual 0",
            "Actual 1",
        ],
    )

    plt.title(
        "Logistic Regression Confusion Matrix\n"
        "Test Period: 2025"
    )

    for row_index in range(2):
        for column_index in range(2):
            plt.text(
                column_index,
                row_index,
                f"{int(matrix[row_index][column_index]):,}",
                ha="center",
                va="center",
                fontsize=14,
            )

    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")

    plt.tight_layout()

    plt.savefig(
        CONFUSION_MATRIX_FILE,
        dpi=200,
    )

    plt.close()


# ============================================================
# CREATE HOURLY PROBABILITY FIGURE
# ============================================================

def create_probability_figure(
    hourly_probability_df: pd.DataFrame,
) -> None:
    """
    Plot logistic-regression movement probability by hour.
    """

    plt.figure(
        figsize=(12, 6)
    )

    plt.bar(
        hourly_probability_df["utc_hour"],
        hourly_probability_df[
            "predicted_probability_percent"
        ],
        edgecolor="black",
    )

    plt.axhline(
        CLASSIFICATION_THRESHOLD * 100,
        linestyle="--",
        linewidth=2,
        label="Classification threshold = 50%",
    )

    plt.title(
        "Logistic Regression Movement Probability "
        "by UTC Hour\n"
        "Model Trained on 2021–2024"
    )

    plt.xlabel("UTC Hour")
    plt.ylabel(
        "Predicted Movement Probability (%)"
    )

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
        PROBABILITY_FIGURE_FILE,
        dpi=200,
    )

    plt.close()


# ============================================================
# SAVE REPORTS
# ============================================================

def save_reports(
    evaluation_df: pd.DataFrame,
    hourly_probability_df: pd.DataFrame,
) -> None:
    """
    Save evaluation and hourly-probability reports.
    """

    EVALUATION_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    evaluation_df.to_csv(
        EVALUATION_FILE,
        index=False,
    )

    hourly_probability_df.to_csv(
        HOURLY_PROBABILITY_FILE,
        index=False,
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    df = load_feature_data()

    analysis_df = select_gap_safe_data(df)

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = create_train_test_split(
        analysis_df
    )

    model = create_model()

    model.fit(
        X_train,
        y_train,
    )

    probabilities, predictions = (
        create_predictions(
            model=model,
            X_test=X_test,
        )
    )

    evaluation_df = create_evaluation_report(
        y_train=y_train,
        y_test=y_test,
        predictions=predictions,
    )

    hourly_probability_df = (
        create_hourly_probability_report(
            model
        )
    )

    save_reports(
        evaluation_df=evaluation_df,
        hourly_probability_df=(
            hourly_probability_df
        ),
    )

    create_confusion_matrix_figure(
        evaluation_df
    )

    create_probability_figure(
        hourly_probability_df
    )

    print("LOGISTIC REGRESSION EVALUATION")

    display_evaluation = (
        evaluation_df.copy()
    )

    display_evaluation["value"] = (
        display_evaluation["value"]
        .round(4)
    )

    print(
        display_evaluation.to_string(
            index=False
        )
    )

    print(
        "\nPREDICTED MOVEMENT PROBABILITY "
        "BY UTC HOUR"
    )

    display_probabilities = (
        hourly_probability_df[
            [
                "utc_hour",
                "predicted_probability_percent",
                "predicted_class_at_0_50",
                "probability_rank",
            ]
        ].copy()
    )

    display_probabilities[
        "predicted_probability_percent"
    ] = (
        display_probabilities[
            "predicted_probability_percent"
        ].round(4)
    )

    print(
        display_probabilities.to_string(
            index=False
        )
    )

    print(
        f"\nEvaluation saved to:\n"
        f"{EVALUATION_FILE}"
    )

    print(
        f"\nHourly probabilities saved to:\n"
        f"{HOURLY_PROBABILITY_FILE}"
    )

    print(
        f"\nConfusion matrix saved to:\n"
        f"{CONFUSION_MATRIX_FILE}"
    )

    print(
        f"\nProbability chart saved to:\n"
        f"{PROBABILITY_FIGURE_FILE}"
    )


if __name__ == "__main__":
    main()