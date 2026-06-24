# src/data_preprocessing.py

import pandas as pd
import numpy as np
import os

def load_raw_data(filename="stock_data.csv"):
    path = os.path.join("data", "raw", filename)
    df = pd.read_csv(path, parse_dates=["Date"])
    print(f"Loaded {len(df)} rows from {path}")
    return df


def clean_data(df):
    print("Cleaning data...")

    # Drop rows where any value is missing
    before = len(df)
    df.dropna(inplace=True)
    after = len(df)
    print(f"Removed {before - after} missing rows.")

    # Make sure numeric columns are actually numbers
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sort by date oldest to newest
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    print("Data cleaned.")
    return df


def add_technical_indicators(df):
    print("Adding technical indicators...")

    # Moving Averages (trend detectors)
    df["MA20"]  = df["Close"].rolling(window=20).mean()   # 20-day average
    df["MA50"]  = df["Close"].rolling(window=50).mean()   # 50-day average

    # Daily Return (how much price changed each day in %)
    df["Daily_Return"] = df["Close"].pct_change() * 100

    # Volatility (how risky the stock is — std dev of last 20 days)
    df["Volatility"] = df["Daily_Return"].rolling(window=20).std()

    # Price Range (difference between daily high and low)
    df["Price_Range"] = df["High"] - df["Low"]

    print("Indicators added.")
    return df


def save_processed_data(df, filename="cleaned_stock_data.csv"):
    path = os.path.join("data", "processed", filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved cleaned data to {path}")


def preprocess(filename="stock_data.csv"):
    df = load_raw_data(filename)
    df = clean_data(df)
    df = add_technical_indicators(df)
    save_processed_data(df)
    return df


# Run directly to test
if __name__ == "__main__":
    df = preprocess()

    print("\nFirst 5 rows of cleaned data:")
    print(df.head())

    print("\nColumn list:")
    print(df.columns.tolist())

    print("\nAny missing values?")
    print(df.isnull().sum())