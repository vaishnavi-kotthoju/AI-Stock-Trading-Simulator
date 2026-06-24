# src/prediction.py

import pandas as pd
import numpy as np
import os
import joblib


def load_model_and_scaler():
    model  = joblib.load("models/trained_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    print("Model and scaler loaded.")
    return model, scaler


def load_latest_data(filename="features.csv"):
    path = os.path.join("data", "processed", filename)
    df   = pd.read_csv(path, parse_dates=["Date"])
    return df


def predict_signal(model, scaler, df, days=5):
    """
    Predicts BUY/SELL for the last N days.
    Also adds HOLD when the model is not confident enough.
    """
    feature_columns = [
        "Open", "High", "Low", "Close", "Volume",
        "MA20", "MA50", "Daily_Return", "Volatility", "Price_Range"
    ]

    # Take the last N rows
    recent = df.tail(days).copy()
    X      = recent[feature_columns]

    # Scale
    X_scaled = scaler.transform(X)

    # Get prediction AND probability
    predictions  = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)  # [[sell%, buy%], ...]

    results = []
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        sell_prob = prob[0] * 100
        buy_prob  = prob[1] * 100
        date      = recent.iloc[i]["Date"]
        close     = recent.iloc[i]["Close"]

        # HOLD if model isn't confident enough (within 45-55% range)
        if 45 <= buy_prob <= 55:
            signal = "HOLD"
        elif pred == 1:
            signal = "BUY"
        else:
            signal = "SELL"

        results.append({
            "Date"      : date.strftime("%Y-%m-%d"),
            "Close"     : round(close, 2),
            "BUY %"     : round(buy_prob, 1),
            "SELL %"    : round(sell_prob, 1),
            "Signal"    : signal
        })

    return pd.DataFrame(results)


def predict_today(model, scaler, df):
    """
    Predicts signal for the single most recent day.
    """
    feature_columns = [
        "Open", "High", "Low", "Close", "Volume",
        "MA20", "MA50", "Daily_Return", "Volatility", "Price_Range"
    ]

    latest   = df.iloc[[-1]]
    X        = latest[feature_columns]
    X_scaled = scaler.transform(X)

    pred  = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0]

    buy_prob  = proba[1] * 100
    sell_prob = proba[0] * 100

    if 45 <= buy_prob <= 55:
        signal = "HOLD"
    elif pred == 1:
        signal = "BUY"
    else:
        signal = "SELL"

    print("\n" + "="*40)
    print("  LATEST PREDICTION")
    print("="*40)
    print(f"  Date       : {df.iloc[-1]['Date'].strftime('%Y-%m-%d')}")
    print(f"  Close      : ${df.iloc[-1]['Close']:.2f}")
    print(f"  BUY  prob  : {buy_prob:.1f}%")
    print(f"  SELL prob  : {sell_prob:.1f}%")
    print(f"  Signal     : {signal}")
    print("="*40)

    return signal, buy_prob, sell_prob


if __name__ == "__main__":
    model, scaler = load_model_and_scaler()
    df            = load_latest_data()

    # Predict last 5 days
    print("\nLast 5 days predictions:")
    results = predict_signal(model, scaler, df, days=5)
    print(results.to_string(index=False))

    # Predict today
    signal, buy_prob, sell_prob = predict_today(model, scaler, df)