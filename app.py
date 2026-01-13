import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# --- BACKEND: Data & Math ---
def get_vrp_data(ticker_symbol, vol_index_symbol):
    # Fetch Price Data for Realized Vol (RV)
    data = yf.download(ticker_symbol, period="60d", interval="1d")
    log_returns = np.log(data['Close'] / data['Close'].shift(1))
    rv = log_returns.rolling(window=21).std() * np.sqrt(252) * 100
    
    # Fetch Implied Vol (IV) Index Data
    iv_data = yf.download(vol_index_symbol, period="60d", interval="1d")
    
    df = pd.DataFrame({
        'Realized Vol (21d)': rv,
        'Implied Vol (Market)': iv_data['Close']
    }).dropna()
    df['VRP'] = df['Implied Vol (Market)'] - df['Realized Vol (21d)']
    return df

def screen_cheap_options(tickers):
    results = []
    for t in tickers:
        try:
            stock = yf.Ticker(t)
            # Calculate 21-day RV
            hist = stock.history(period="30d")
            rv = hist['Close'].pct_change().std() * np.sqrt(252) * 100
            
            # Get IV from ATM Option
            expiries = stock.options
            if not expiries: continue
            chain = stock.option_chain(expiries[0]) # Nearest expiry
            # Filter for At-The-Money (ATM)
            current_price = hist['Close'].iloc[-1]
            atm_option = chain.calls.iloc[(chain.calls['strike'] - current_price).abs().argsort()[:1]]
            iv = atm_option['impliedVolatility'].values[0] * 100
            
            results.append({
                "Ticker": t,
                "IV %": round(iv, 2),
                "RV %": round(rv, 2),
                "VRP (IV-RV)": round(iv - rv, 2),
                "Status": "CHEAP (Buy)" if iv < rv else "EXPENSIVE (Sell)"
            })
        except: continue
    return pd.DataFrame(results)

# --- FRONTEND: Streamlit UI ---
st.set_page_config(page_title="VRP Dashboard 2026", layout="wide")
st.title("🛡️ Cross-Asset Volatility Risk Premium")

col1, col2, col3 = st.columns(3)

# Display VRP Gauges
assets = {"SPY": "^VIX", "Gold (GLD)": "^GVZ", "10Y Yields": "^TYVIX"}
for i, (name, vol_ticker) in enumerate(assets.items()):
    df = get_vrp_data(name.split(" ")[0], vol_ticker)
    latest = df.iloc[-1]
    with [col1, col2, col3][i]:
        st.metric(name, f"{latest['VRP']:.2f}", delta="Bearish (High VRP)" if latest['VRP'] > 0 else "Bullish (Low VRP)")
        st.line_chart(df[['Realized Vol (21d)', 'Implied Vol (Market)']])

st.markdown("---")
st.subheader("🔍 Cheap IV Screener (Potential Naked Buys)")
watchlist = ["AAPL", "TSLA", "NVDA", "AMD", "MSFT", "GOOGL", "AMZN", "META"]
if st.button("Run Screener"):
    screening_df = screen_cheap_options(watchlist)
    st.table(screening_df.sort_values("VRP (IV-RV)"))