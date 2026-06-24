# src/feature_engineering.py

import pandas as pd
import numpy as np
import os


def load_processed_data(filename="cleaned_stock_data.csv"):
    path = os.path.join("data", "processed", filename)
    df = pd.read_csv(path, parse_dates=["Date"])
    print(f"Loaded {len(df)} rows.")
    return df


def create_target_column(df):
    """
    Target = what the AI will predict.

    Logic:
      If tomorrow's Close price > today's Close → BUY  (1)
      If tomorrow's Close price < today's Close → SELL (0)

    We shift Close by -1 to get "tomorrow's price" for each row.
    """
    print("Creating target column...")

    df["Tomorrow_Close"] = df["Close"].shift(-1)

    df["Target"] = np.where(
        df["Tomorrow_Close"] > df["Close"], 1, 0
    )
    # 1 = BUY, 0 = SELL

    # Drop the last row (no tomorrow for the final day)
    df.dropna(subset=["Tomorrow_Close"], inplace=True)

    print(f"Target created. BUY days: {df['Target'].sum()} | SELL days: {(df['Target']==0).sum()}")
    return df


def select_features(df):
    """
    Choose which columns the AI will learn from.
    These are called 'features'.
    """
    print("Selecting features...")

    # Drop rows that still have NaN (from MA20, MA50, Volatility)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # These are the inputs the AI will use
    feature_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "MA20",
        "MA50",
        "Daily_Return",
        "Volatility",
        "Price_Range"
    ]

    X = df[feature_columns]   # Features (inputs)
    y = df["Target"]          # Target (output: 1=BUY, 0=SELL)

    print(f"Final dataset: {X.shape[0]} rows, {X.shape[1]} features.")
    return X, y, df


def save_features(X, y, df):
    path = os.path.join("data", "processed", "features.csv")

    # Save everything together for reference
    final = df[["Date", "Close", "Target"]].copy()
    final = pd.concat([final.reset_index(drop=True), X.reset_index(drop=True)], axis=1)
    final.to_csv(path, index=False)
    print(f"Features saved to {path}")


# Run directly to test
if __name__ == "__main__":
    df = load_processed_data()
    df = create_target_column(df)
    X, y, df = select_features(df)
    save_features(X, y, df)

    print("\nFeature columns:")
    print(X.columns.tolist())

    print("\nFirst 3 rows of X:")
    print(X.head(3))

    print("\nFirst 3 values of y (1=BUY, 0=SELL):")
    print(y.head(3).tolist())