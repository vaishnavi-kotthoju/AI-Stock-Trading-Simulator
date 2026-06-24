# dashboard/app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import yfinance as yf
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from trading_simulator import generate_signals, run_simulation

# ── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Stock Trading Simulator",
    page_icon="📈",
    layout="wide"
)

# ── Load Model ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    model  = joblib.load("models/trained_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler

# ── Load & Process Stock Data ────────────────────────────────
@st.cache_data
def load_stock(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.reset_index(inplace=True)

    # Indicators
    df["MA20"]         = df["Close"].rolling(20).mean()
    df["MA50"]         = df["Close"].rolling(50).mean()
    df["Daily_Return"] = df["Close"].pct_change() * 100
    df["Volatility"]   = df["Daily_Return"].rolling(20).std()
    df["Price_Range"]  = df["High"] - df["Low"]

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# ════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════
st.sidebar.title("⚙️ Settings")
st.sidebar.markdown("---")

ticker = st.sidebar.selectbox(
    "Select Stock",
    ["AAPL", "TSLA", "GOOGL", "MSFT", "AMZN",
     "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"]
)

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2022-01-01"))
end_date   = st.sidebar.date_input("End Date",   pd.to_datetime("2024-12-31"))

starting_balance = st.sidebar.number_input(
    "Starting Balance (₹)",
    min_value=10000,
    max_value=10000000,
    value=100000,
    step=10000
)

run_button = st.sidebar.button("🚀 Run Simulation", use_container_width=True)

# ════════════════════════════════════════════════════════════
#  MAIN PAGE
# ════════════════════════════════════════════════════════════
st.title("📈 AI Stock Trading Simulator")
st.markdown("*Powered by Random Forest ML + Technical Indicators*")
st.markdown("---")

if run_button:
    with st.spinner("Downloading stock data and running simulation..."):
        model, scaler = load_model()
        df = load_stock(ticker, str(start_date), str(end_date))
        df = generate_signals(df, model, scaler)
        results = run_simulation(df, starting_balance=starting_balance)

    # ── Row 1: Key Metrics ───────────────────────────────────
    st.subheader("📊 Simulation Results")
    col1, col2, col3, col4, col5 = st.columns(5)

    profit_color = "normal" if results["profit"] >= 0 else "inverse"

    col1.metric("Starting Balance", f"₹{results['starting']:,.0f}")
    col2.metric("Final Balance",    f"₹{results['final']:,.0f}")
    col3.metric("Total Profit",     f"₹{results['profit']:,.0f}",
                delta=f"{results['roi']}%")
    col4.metric("Total Trades",     results["total_trades"])
    col5.metric("Win / Loss",       f"{results['wins']} / {results['losses']}")

    st.markdown("---")

    # ── Row 2: Charts ────────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("💹 Stock Price + Moving Averages")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["Date"], y=df["Close"],
            name="Close Price", line=dict(color="white", width=1.5)
        ))
        fig.add_trace(go.Scatter(
            x=df["Date"], y=df["MA20"],
            name="MA20", line=dict(color="orange", width=1, dash="dot")
        ))
        fig.add_trace(go.Scatter(
            x=df["Date"], y=df["MA50"],
            name="MA50", line=dict(color="cyan", width=1, dash="dot")
        ))
        fig.update_layout(
            template="plotly_dark",
            height=350,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("💰 Portfolio Value Over Time")
        portfolio = results["portfolio"]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=portfolio["Date"], y=portfolio["Portfolio_Value"],
            fill="tozeroy", name="Portfolio",
            line=dict(color="lime", width=2)
        ))
        fig2.add_hline(
            y=starting_balance,
            line_dash="dash", line_color="red",
            annotation_text="Starting Balance"
        )
        fig2.update_layout(
            template="plotly_dark",
            height=350,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # ── Row 3: Signals + Trades ──────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("🤖 AI Signal Distribution")
        signal_counts = df["Signal"].value_counts().reset_index()
        signal_counts.columns = ["Signal", "Count"]
        colors = {"BUY": "green", "SELL": "red", "HOLD": "gray"}
        fig3 = px.bar(
            signal_counts, x="Signal", y="Count",
            color="Signal",
            color_discrete_map=colors,
            template="plotly_dark"
        )
        fig3.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig3, use_container_width=True)

    with col_b:
        st.subheader("🎯 Latest AI Prediction")
        latest      = df.iloc[-1]
        buy_prob    = latest["BUY_Prob"]
        sell_prob   = 100 - buy_prob
        signal      = latest["Signal"]

        signal_colors = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}
        st.markdown(f"### {signal_colors.get(signal, '⚪')} Signal: **{signal}**")
        st.markdown(f"**Date:** {latest['Date'].strftime('%Y-%m-%d')}")
        st.markdown(f"**Close Price:** ${latest['Close']:.2f}")

        fig4 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=buy_prob,
            title={"text": "BUY Confidence %"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": "lime"},
                "steps": [
                    {"range": [0,  45], "color": "red"},
                    {"range": [45, 55], "color": "gray"},
                    {"range": [55, 100],"color": "green"},
                ]
            }
        ))
        fig4.update_layout(
            height=250,
            template="plotly_dark",
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")

    # ── Row 4: Trade History ─────────────────────────────────
    st.subheader("📋 Trade History")
    if len(results["trades"]) > 0:
        trades = results["trades"].copy()
        trades["Action"] = trades["Action"].apply(
            lambda x: f"🟢 {x}" if x == "BUY" else f"🔴 {x}"
        )
        st.dataframe(trades, use_container_width=True, height=300)
    else:
        st.info("No trades were executed.")

else:
    # ── Welcome Screen ───────────────────────────────────────
    st.markdown