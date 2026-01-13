import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import time
from yfinance.exceptions import YFRateLimitError

# ================= CONFIG =================
TICKERS = ["AAPL", "TSLA", "SPY", "NVDA"]       # Watchlist
IV_CHEAP_THRESHOLD = 0.25                       # 25% IV → potentially cheap
NEAR_ATM_DOLLAR = 10                             # fallback for low-price stocks
NEAR_ATM_PCT = 0.02                             # ±2% of price for high-priced stocks
HISTORY_FILE = "iv_history.csv"                 # For IV rank tracking
MAX_RETRIES = 5
RETRY_DELAY = 10  # seconds

sns.set_style("darkgrid")

# ================= HELPERS =================
def get_current_price(ticker):
    """Reliable latest price using history fallback."""
    try:
        stock = yf.Ticker(ticker)
        price = stock.info.get('regularMarketPrice') or stock.info.get('currentPrice')
        if price is None:
            # fallback to last close
            price = stock.history(period="1d")['Close'][-1]
        return round(price, 2)
    except Exception as e:
        print(f"[Price Error] {ticker}: {e}")
        return None

def get_near_atm_iv(ticker):
    """Get average IV for near-ATM options."""
    try:
        stock = yf.Ticker(ticker)
        expirations = stock.options
        if not expirations:
            return None

        nearest_exp = expirations[0]
        chain = stock.option_chain(nearest_exp)
        current_price = get_current_price(ticker)
        if current_price is None:
            return None

        options = pd.concat([chain.calls, chain.puts])
        # Apply % or $ filter
        near_atm = options[
            (abs(options['strike'] - current_price) <= NEAR_ATM_DOLLAR) |
            (abs(options['strike'] - current_price) <= current_price * NEAR_ATM_PCT)
        ]
        if near_atm.empty:
            return None

        avg_iv = near_atm['impliedVolatility'].mean()
        median_iv = near_atm['impliedVolatility'].median()

        return {
            'ticker': ticker,
            'current_price': current_price,
            'nearest_exp': nearest_exp,
            'avg_near_atm_iv': avg_iv,
            'median_near_atm_iv': median_iv,
            'iv_percent': f"{avg_iv:.1%}",
            'is_cheap': avg_iv < IV_CHEAP_THRESHOLD,
            'num_contracts': len(near_atm)
        }

    except Exception as e:
        print(f"[IV Error] {ticker}: {e}")
        return None

def get_near_atm_iv_with_retry(ticker):
    """Retry wrapper for rate limits."""
    for attempt in range(MAX_RETRIES):
        try:
            result = get_near_atm_iv(ticker)
            if result:
                return result
            else:
                print(f"[No Options] {ticker}")
                return None
        except YFRateLimitError:
            print(f"[Rate Limited] {ticker}. Retry {attempt+1}/{MAX_RETRIES} after {RETRY_DELAY}s")
            time.sleep(RETRY_DELAY)
        except Exception as e:
            print(f"[Error] {ticker}: {e}")
            break
    return None

def plot_iv_smile(ticker, exp_date=None):
    """Plot IV vs Strike (smile/skew)."""
    try:
        stock = yf.Ticker(ticker)
        if not exp_date:
            if not stock.options:
                print(f"No options for {ticker}")
                return
            exp_date = stock.options[0]

        chain = stock.option_chain(exp_date)
        current_price = get_current_price(ticker)

        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=chain.calls, x='strike', y='impliedVolatility', 
                        label='Calls', alpha=0.6, color='blue')
        sns.scatterplot(data=chain.puts, x='strike', y='impliedVolatility', 
                        label='Puts', alpha=0.6, color='red')
        plt.axvline(current_price, color='black', linestyle='--', 
                    label=f'Current Price ≈ ${current_price:.0f}')
        plt.title(f"{ticker} IV Smile/Skew - Exp: {exp_date}")
        plt.xlabel("Strike Price")
        plt.ylabel("Implied Volatility")
        plt.legend()
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"[Plot Error] {ticker}: {e}")

def update_iv_history(results):
    """Append today's results to CSV."""
    today = datetime.now().strftime("%Y-%m-%d")
    df_new = pd.DataFrame(results)
    df_new['date'] = today

    if os.path.exists(HISTORY_FILE):
        df_old = pd.read_csv(HISTORY_FILE)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_csv(HISTORY_FILE, index=False)
    print(f"[History Updated] {HISTORY_FILE}")

# ================= MAIN SCANNER =================
def scan_watchlist():
    print(f"\n=== IV Scan Report - {datetime.now().strftime('%Y-%m-%d %H:%M')} ===\n")
    results = []

    for ticker in TICKERS:
        data = get_near_atm_iv_with_retry(ticker)
        if data:
            results.append(data)
            status = "CHEAP!" if data['is_cheap'] else "Normal/High"
            print(f"{ticker:6} | Price: ${data['current_price']:7.2f} | "
                  f"Nearest Exp: {data['nearest_exp']} | "
                  f"Avg IV: {data['iv_percent']:>6} → {status}")
        else:
            print(f"{ticker:6} → No data / no options")

    if results:
        update_iv_history(results)

    print("\nDone. Use plot_iv_smile('AAPL') to visualize any ticker.")
    return results

# ================= RUN IT =================
if __name__ == "__main__":
    scan_watchlist()
    # Example: plot_iv_smile("AAPL")