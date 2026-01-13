
           import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# --- BACKEND: Data Logic ---

def get_clean_data(ticker, period="60d"):
    """Fetch data and force single-level columns to avoid MultiIndex errors."""
    df = yf.download(ticker, period=period, interval="1d", multi_level_index=False)
    if df.empty:
        return pd.Series(dtype=float)
    # Ensure it's a Series and not a 1-column DataFrame
    return df['Close'].squeeze()

def get_vrp_data(asset_ticker, vol_ticker):
    """Calculates VRP: Difference between Implied and Realized Vol."""
    price_series = get_clean_data(asset_ticker)
    iv_series = get_clean_data(vol_ticker)
    
    if price_series.empty or iv_series.empty:
        return pd.DataFrame()

    # Calculate 21-day Annualized Realized Volatility
    log_return = np.log(price_series / price_series.shift(1))
    rv = log_return.rolling(window=21).std() * np.sqrt(252) * 100
    
    # Align IV and RV into a single DataFrame
    combined = pd.DataFrame({
        'Realized Vol (21d)': rv,
        'Implied Vol (Market)': iv_series
    }).dropna()
    
    combined['VRP'] = combined['Implied Vol (Market)'] - combined['Realized Vol (21d)']
    return combined

def screen_cheap_iv(tickers):
    """Screens for tickers where IV < RV (Potential Naked Buys)."""
    results = []
    for t in tickers:
        try:
            # Get Price and calc RV
            price_series = get_clean_data(t, period="30d")
            if price_series.empty: continue
            rv = (np.log(price_series / price_series.shift(1)).std() * np.sqrt(252) * 100)
            
            # Get Options Chain (using first available expiry)
            ticker_obj = yf.Ticker(t)
            expiries = ticker_obj.options
            if not expiries: continue
            
            chain = ticker_obj.option_chain(expiries[0])
            current_price = price_series.iloc[-1]
            
            # Find At-The-Money (ATM) Call IV
            atm_call = chain.calls.iloc[(chain.calls['strike'] - current_price).abs().argsort()[:1]]
            iv = atm_call['impliedVolatility'].values[0] * 100
            
            results.append({
                "Ticker": t,
                "Price": round(current_price, 2),
                "IV %": round(iv, 2),
                "RV %": round(rv, 2),
                "VRP": round(iv - rv, 2),
                "Signal": "🟢 CHEAP" if iv < rv else "🔴 EXPENSIVE"
            })
        except Exception:
            continue
    return pd.DataFrame(results)

# --- FRONTEND: Dashboard ---

st.set_page_config(page_title="VRP Terminal 2026", layout="wide")
st.title("📊 Cross-Asset Volatility Risk Premium")
st.caption("Daily VRP Dashboard: SPY, Gold, and 10Y Yields")

# 1. Main Dashboard Gauges
col1, col2, col3 = st.columns(3)
assets = {"SPY": "^VIX", "GLD": "^GVZ", "10Y Yields (^TNX)": "^TYVIX"}

for i, (asset, vol_idx) in enumerate(assets.items()):
    with [col1, col2, col3][i]:
        df = get_vrp_data(asset.split(" ")[0], vol_idx)
        if not df.empty:
            latest_vrp = df['VRP'].iloc[-1]
            status = "Bearish (Overpriced)" if latest_vrp > 0 else "Bullish (Underpriced)"
            st.metric(asset, f"{latest_vrp:.2f}", delta=status, delta_color="inverse")
            st.line_chart(df[['Realized Vol (21d)', 'Implied Vol (Market)']])
        else:
            st.error(f"Data unavailable for {asset}")

# 2. Options Screener
st.markdown("---")
st.subheader("🎯 Cheap IV Screener: Naked Buy Opportunities")
st.write("Calculates ATM IV vs 30-day Realized Vol. Negative VRP suggests options are underpriced relative to movement.")

watchlist = ["AAPL", "TSLA", "NVDA", "AMD", "MSFT", "GOOGL", "AMZN", "META", "COIN", "MARA"]
if st.button("🚀 Run Options Screen"):
    with st.spinner("Calculating Volatility Ratios..."):
        screen_df = screen_cheap_iv(watchlist)
        if not screen_df.empty:
            # Highlight cheap rows
            st.dataframe(screen_df.sort_values("VRP"), use_container_width=True)
        else:
            st.warning("No option data found for watchlist.")