from pathlib import Path

import pandas as pd


PROJECT_FOLDER = Path(__file__).resolve().parents[1]

RAW_FILE = (
    PROJECT_FOLDER
    / "data"
    / "raw"
    / "btcusdt_1h_2021_2025_raw.csv"
)


def main() -> None:
    df = pd.read_csv(RAW_FILE)

    print("SHAPE")
    print(df.shape)

    print("\nCOLUMN NAMES")
    print(df.columns.tolist())

    print("\nDATA TYPES")
    print(df.dtypes)

    print("\nFIRST 5 ROWS")
    print(df.head().to_string(index=False))

    print("\nLAST 5 ROWS")
    print(df.tail().to_string(index=False))

    print("\nMISSING VALUES")
    print(df.isna().sum())

    print("\nBASIC NUMERICAL SUMMARY")
    print(df.describe().to_string())


if __name__ == "__main__":
    main()