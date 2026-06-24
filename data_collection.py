# src/data_collection.py

import yfinance as yf
import pandas as pd
import os

def download_stock_data(ticker, start, end):
    print(f"Downloading data for {ticker}...")

    df = yf.download(ticker, start=start, end=end)

    if df.empty:
        print("No data found. Check ticker symbol.")
        return df

    # Fix column names (yfinance sometimes creates multi-level columns)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Keep only what we need
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.reset_index(inplace=True)

    print(f"Downloaded {len(df)} rows.")
    return df


def save_raw_data(df, filename="stock_data.csv"):
    path = os.path.join("data", "raw", filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved to {path}")


# Run this file directly to test
if __name__ == "__main__":
    df = download_stock_data(
        ticker="AAPL",
        start="2022-01-01",
        end="2024-12-31"
    )
    print(df.head())
    save_raw_data(df)