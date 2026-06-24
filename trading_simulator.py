# src/trading_simulator.py

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_data_and_model():
    df     = pd.read_csv("data/processed/features.csv", parse_dates=["Date"])
    model  = joblib.load("models/trained_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return df, model, scaler


def generate_signals(df, model, scaler):
    feature_columns = [
        "Open", "High", "Low", "Close", "Volume",
        "MA20", "MA50", "Daily_Return", "Volatility", "Price_Range"
    ]

    X        = df[feature_columns]
    X_scaled = scaler.transform(X)
    proba    = model.predict_proba(X_scaled)

    buy_probs = proba[:, 1] * 100

    signals = []
    for bp in buy_probs:
        if bp >= 55:
            signals.append("BUY")
        elif bp <= 45:
            signals.append("SELL")
        else:
            signals.append("HOLD")

    df = df.copy()
    df["Signal"]   = signals
    df["BUY_Prob"] = buy_probs.round(1)
    return df


def run_simulation(df, starting_balance=100000):
    """
    Simulates a trader following AI signals.

    Rules:
      BUY  → buy as many shares as possible
      SELL → sell all shares
      HOLD → do nothing
    """
    balance       = starting_balance
    shares        = 0
    portfolio     = []
    trades        = []
    total_trades  = 0
    wins          = 0
    losses        = 0
    buy_price     = 0

    for _, row in df.iterrows():
        price  = row["Close"]
        signal = row["Signal"]
        date   = row["Date"]

        # BUY
        if signal == "BUY" and balance >= price:
            shares_to_buy = int(balance // price)
            cost          = shares_to_buy * price
            balance      -= cost
            shares       += shares_to_buy
            buy_price     = price
            total_trades += 1
            trades.append({
                "Date"   : date,
                "Action" : "BUY",
                "Price"  : round(price, 2),
                "Shares" : shares_to_buy,
                "Balance": round(balance, 2)
            })

        # SELL
        elif signal == "SELL" and shares > 0:
            revenue  = shares * price
            balance += revenue
            profit   = revenue - (shares * buy_price)

            if profit > 0:
                wins += 1
            else:
                losses += 1

            total_trades += 1
            trades.append({
                "Date"   : date,
                "Action" : "SELL",
                "Price"  : round(price, 2),
                "Shares" : shares,
                "Balance": round(balance, 2)
            })
            shares = 0

        # Portfolio value = cash + value of shares held
        portfolio_value = balance + (shares * price)
        portfolio.append({
            "Date"            : date,
            "Portfolio_Value" : round(portfolio_value, 2),
            "Close"           : round(price, 2)
        })

    # If still holding shares at end, sell at last price
    if shares > 0:
        final_price = df.iloc[-1]["Close"]
        balance    += shares * final_price

    # Final stats
    final_value  = balance
    total_profit = final_value - starting_balance
    roi          = (total_profit / starting_balance) * 100

    return {
        "trades"         : pd.DataFrame(trades),
        "portfolio"      : pd.DataFrame(portfolio),
        "starting"       : starting_balance,
        "final"          : round(final_value, 2),
        "profit"         : round(total_profit, 2),
        "roi"            : round(roi, 2),
        "total_trades"   : total_trades,
        "wins"           : wins,
        "losses"         : losses
    }


def print_summary(results):
    print("\n" + "="*45)
    print("        TRADING SIMULATION SUMMARY")
    print("="*45)
    print(f"  Starting Balance : ₹{results['starting']:,.2f}")
    print(f"  Final Balance    : ₹{results['final']:,.2f}")
    print(f"  Total Profit     : ₹{results['profit']:,.2f}")
    print(f"  ROI              : {results['roi']}%")
    print(f"  Total Trades     : {results['total_trades']}")
    print(f"  Winning Trades   : {results['wins']}")
    print(f"  Losing Trades    : {results['losses']}")
    print("="*45)

    print("\nLast 10 trades:")
    if len(results["trades"]) > 0:
        print(results["trades"].tail(10).to_string(index=False))
    else:
        print("No trades executed.")


def plot_portfolio(results):
    portfolio = results["portfolio"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Portfolio value over time
    ax1.plot(
        portfolio["Date"],
        portfolio["Portfolio_Value"],
        color="green", linewidth=1.5, label="Portfolio Value"
    )
    ax1.axhline(
        y=results["starting"],
        color="red", linestyle="--", label="Starting Balance"
    )
    ax1.set_title("Portfolio Value Over Time")
    ax1.set_ylabel("Value (₹)")
    ax1.legend()
    ax1.tick_params(axis="x", rotation=45)

    # Stock price over time
    ax2.plot(
        portfolio["Date"],
        portfolio["Close"],
        color="steelblue", linewidth=1.5, label="AAPL Close Price"
    )
    ax2.set_title("Stock Price Over Time")
    ax2.set_ylabel("Price ($)")
    ax2.legend()
    ax2.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    os.makedirs("reports/Screenshots", exist_ok=True)
    plt.savefig("reports/Screenshots/portfolio_performance.png")
    plt.close()
    print("\nPortfolio chart saved to reports/Screenshots/")


if __name__ == "__main__":
    df, model, scaler = load_data_and_model()
    df                = generate_signals(df, model, scaler)
    results           = run_simulation(df, starting_balance=100000)
    print_summary(results)
    plot_portfolio(results)