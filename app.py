# app.py – Streamlit-only Vol-Risk-Premium & Cheap-Vol scanner
# -----------------------------------------------------------
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ssl, os, sys

# ---------- Streamlit Cloud SSL fix (aiohttp not needed) ----------
ssl._create_default_https_context = ssl._create_unverified_context

# ---------- config ----------
TICKERS = {"SPY": "SPY", "GOLD": "GLD", "YIELD": "TLT"}  # TLT ≈ 10-yr
STOCK_CACHE = "stock_cache.parquet"
OPT_CACHE   = "opt_cache.parquet"
CACHE_MINS  = 30


# ---------- helpers ----------
@st.cache_data(ttl=60*CACHE_MINS)
def load_history(ticker: str, period: str = "6mo"):
    """Return daily price series with Date index."""
    return yf.download(ticker, period=period, progress=False)["Adj Close"]

@st.cache_data(ttl=60*CACHE_MINS)
def get_options_iv(ticker: str) -> float:
    """
    Quick 30-day IV proxy: average of closest two expiries ATM implied vol.
    Falls back to 20-day realised if options unavailable.
    """
    tk = yf.Ticker(ticker)
    try:
        opts = tk.options
        if not opts:
            raise RuntimeError("No options chain")
        exp1, exp2 = opts[0], opts[1]
        chain1 = tk.option_chain(exp1).calls
        chain2 = tk.option_chain(exp2).calls
        atm1 = chain1.iloc[(chain1.strike - chain1.lastPrice).abs().argsort()[:1]]
        atm2 = chain2.iloc[(chain2.strike - chain2.lastPrice).abs().argsort()[:1]]
        # crude linear interp to 30 days
        d1 = (pd.to_datetime(exp1) - datetime.now()).days
        d2 = (pd.to_datetime(exp2) - datetime.now()).days
        iv1, iv2 = atm1.impliedVolatility.iloc[0], atm2.impliedVolatility.iloc[0]
        iv30 = np.interp(30, [d1, d2], [iv1, iv2]) * 100
        return float(iv30)
    except Exception as e:
        # fallback: 20-day realised
        px = load_history(ticker, "40d")
        rt = np.log(px / px.shift(1)).dropna()
        return float(rt.std() * np.sqrt(252) * 100)

def realised_vol(ticker: str, days: int = 21) -> float:
    px = load_history(ticker, "6mo")
    rt = np.log(px / px.shift(1)).dropna()
    return float(rt.rolling(days).std().iloc[-1] * np.sqrt(252) * 100)

def cheap_vol_rank(ticker: str) -> float:
    """1-yr percentile of 30-day IV (0 = cheapest, 1 = richest)."""
    px = load_history(ticker, "1y")
    # use 20-day RV as IV proxy history (no historic options)
    rt = np.log(px / px.shift(1)).dropna()
    rv_hist = rt.rolling(20).std() * np.sqrt(252) * 100
    current_iv = get_options_iv(ticker)
    return float((rv_hist <= current_iv).mean())

# ---------- core ----------
def build_vrp_table():
    rows = []
    for asset, sym in TICKERS.items():
        iv  = get_options_iv(sym)
        rv  = realised_vol(sym, 21)
        rows.append(
            {
                "Asset": asset,
                "Ticker": sym,
                "30d IV (%)": round(iv, 2),
                "21d RV (%)": round(rv, 2),
                "VRP (%)":    round(iv - rv, 2),
            }
        )
    return pd.DataFrame(rows)

def build_cheap_vol_table():
    rows = []
    for asset, sym in TICKERS.items():
        pct = cheap_vol_rank(sym)
        rows.append(
            {
                "Asset": asset,
                "Ticker": sym,
                "IV 1-yr percentile": round(pct * 100, 1),
                "Cheap?": "✅" if pct <= 0.15 else "❌",
            }
        )
    return pd.DataFrame(rows).sort_values("IV 1-yr percentile")

# ---------- UI ----------
st.set_page_config(page_title="Cheap-Vol & VRP Monitor", layout="centered")
st.title("Real-Time Volatility Risk Premium & Cheap-Vol Scanner")
st.markdown("Data refreshed every 30 min via Yahoo Finance (no keys required).")

with st.spinner("Fetching latest data..."):
    vrp_df = build_vrp_table()
    cheap_df = build_cheap_vol_table()

st.subheader("Volatility Risk Premium (VRP)")
st.dataframe(vrp_df, use_container_width=True)

st.subheader("Cheap-Vol Ranking (lower percentile = cheaper)")
st.dataframe(cheap_df, use_container_width=True)

st.caption("Last updated: " + datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"))