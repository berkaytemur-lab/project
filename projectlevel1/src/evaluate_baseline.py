from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


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

REPORT_FILE = (
    PROJECT_FOLDER
    / "reports"
    / "baseline_evaluation.csv"
)

FIGURE_FOLDER = (
    PROJECT_FOLDER
    / "reports"
    / "figures"
)

CONFUSION_MATRIX_FILE = (
    FIGURE_FOLDER
    / "baseline_confusion_matrix.png"
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
# SELECT PRIMARY ANALYSIS ROWS
# ============================================================

def select_gap_safe_rows(
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
# CREATE TIME-BASED SPLIT
# ============================================================

def create_train_test_split(
    analysis_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Use 2021–2024 as training data and 2025 as unseen
    test data.
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

    return train_df, test_df


# ============================================================
# CREATE BASELINE PREDICTIONS
# ============================================================

def create_baseline_predictions(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[int, pd.Series]:
    """
    Learn the most common class from the training data
    and predict that class for every test observation.
    """

    majority_class = int(
        train_df["movement_event"]
        .mode()
        .iloc[0]
    )

    predictions = pd.Series(
        majority_class,
        index=test_df.index,
        dtype="int64",
    )

    return majority_class, predictions


# ============================================================
# CALCULATE CONFUSION MATRIX
# ============================================================

def calculate_confusion_matrix(
    actual: pd.Series,
    predicted: pd.Series,
) -> dict:
    """
    Calculate binary-classification confusion-matrix values.
    """

    true_negative = int(
        (
            (actual == 0)
            & (predicted == 0)
        ).sum()
    )

    false_positive = int(
        (
            (actual == 0)
            & (predicted == 1)
        ).sum()
    )

    false_negative = int(
        (
            (actual == 1)
            & (predicted == 0)
        ).sum()
    )

    true_positive = int(
        (
            (actual == 1)
            & (predicted == 1)
        ).sum()
    )

    return {
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_positive": true_positive,
    }


# ============================================================
# CALCULATE EVALUATION METRICS
# ============================================================

def calculate_evaluation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    majority_class: int,
    predictions: pd.Series,
) -> pd.DataFrame:
    """
    Calculate the baseline accuracy and confusion matrix.
    """

    actual = test_df["movement_event"]

    confusion = calculate_confusion_matrix(
        actual=actual,
        predicted=predictions,
    )

    correct_predictions = (
        predictions == actual
    ).sum()

    accuracy = (
        correct_predictions
        / len(test_df)
    )

    train_event_rate = (
        train_df["movement_event"].mean()
    )

    test_event_rate = (
        test_df["movement_event"].mean()
    )

    results = {
        "training_rows": len(train_df),
        "test_rows": len(test_df),
        "training_event_rate_percent": (
            train_event_rate * 100
        ),
        "test_event_rate_percent": (
            test_event_rate * 100
        ),
        "baseline_majority_class": majority_class,
        "baseline_accuracy_percent": (
            accuracy * 100
        ),
        "true_negative": confusion["true_negative"],
        "false_positive": confusion["false_positive"],
        "false_negative": confusion["false_negative"],
        "true_positive": confusion["true_positive"],
    }

    return pd.DataFrame(
        {
            "metric": results.keys(),
            "value": results.values(),
        }
    )


# ============================================================
# CREATE CONFUSION-MATRIX FIGURE
# ============================================================

def create_confusion_matrix_figure(
    evaluation_df: pd.DataFrame,
) -> None:
    """
    Create and save a visual confusion matrix.
    """

    metric_values = dict(
        zip(
            evaluation_df["metric"],
            evaluation_df["value"],
        )
    )

    matrix = [
        [
            metric_values["true_negative"],
            metric_values["false_positive"],
        ],
        [
            metric_values["false_negative"],
            metric_values["true_positive"],
        ],
    ]

    FIGURE_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(7, 6))

    image = plt.imshow(matrix)

    plt.colorbar(
        image,
        label="Number of Test Candles",
    )

    plt.xticks(
        [0, 1],
        [
            "Predicted 0",
            "Predicted 1",
        ],
    )

    plt.yticks(
        [0, 1],
        [
            "Actual 0",
            "Actual 1",
        ],
    )

    plt.title(
        "Baseline Confusion Matrix\n"
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
# SAVE REPORT
# ============================================================

def save_evaluation(
    evaluation_df: pd.DataFrame,
) -> None:
    """
    Save the baseline evaluation report.
    """

    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    evaluation_df.to_csv(
        REPORT_FILE,
        index=False,
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    df = load_feature_data()

    analysis_df = select_gap_safe_rows(df)

    train_df, test_df = create_train_test_split(
        analysis_df
    )

    majority_class, predictions = (
        create_baseline_predictions(
            train_df=train_df,
            test_df=test_df,
        )
    )

    evaluation_df = calculate_evaluation(
        train_df=train_df,
        test_df=test_df,
        majority_class=majority_class,
        predictions=predictions,
    )

    save_evaluation(
        evaluation_df
    )

    create_confusion_matrix_figure(
        evaluation_df
    )

    print("TIME-BASED DATA SPLIT")

    print(
        "Training period:",
        train_df["open_time"].min(),
        "to",
        train_df["open_time"].max(),
    )

    print(
        "Test period:",
        test_df["open_time"].min(),
        "to",
        test_df["open_time"].max(),
    )

    print("\nBASELINE EVALUATION")

    display_df = evaluation_df.copy()

    display_df["value"] = (
        display_df["value"].round(4)
    )

    print(
        display_df.to_string(
            index=False
        )
    )

    print(
        "\nBaseline prediction:",
        majority_class,
    )

    print(
        "0 means no movement event."
    )

    print(
        "1 means movement event."
    )

    print(
        f"\nReport saved to:\n{REPORT_FILE}"
    )

    print(
        f"\nConfusion matrix saved to:\n"
        f"{CONFUSION_MATRIX_FILE}"
    )


if __name__ == "__main__":
    main()