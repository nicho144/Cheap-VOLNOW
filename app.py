# vrp_platform.py  –  all-in-one financial vol risk premium engine
# ---------------------------------------------------------------
# pip install yfinance aiohttp pandas numpy scipy pyarrow fastapi uvicorn exchange-calendars schedule
# ---------------------------------------------------------------
from __future__ import annotations
import asyncio, aiohttp, yfinance as yf, pandas as pd, numpy as np, os, json, logging
from datetime import datetime, timedelta
from typing import Dict, List
from pathlib import Path
import exchange_calendars as xc
from scipy import linalg
from fastapi import FastAPI
import schedule, time, sys

# ---------- config ----------
TICKERS = dict(SPY="SPY", GOLD="GLD", YIELD="ZN=F")  # 10-yr T-note future
CBOE_VOL_MAP = dict(SPY="VIX", GLD="GVZ", SLV="VXSLV", OIL="OVX", EUR="EVZ", GBP="BPVIX")
DATA_DIR = Path("data")
PARQUET_FILE = DATA_DIR / "vrp_store.parquet"
CHEAP_PC = 0.15  # bottom 15 % == cheap
os.makedirs(DATA_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
NY = xc.get_calendar("NYSE")
# ----------------------------

# ---------- fetch ----------
async def _fetch(url: str) -> dict:
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as s:
        async with s.get(url) as r:
            r.raise_for_status()
            return await r.json()

async def last_close(symbol: str) -> float:
    tk = yf.Ticker(symbol)
    return float(tk.history(period="5d").Close.iloc[-1])

async def cboe_iv(sym: str) -> float:
    url = f"https://cdn.cboe.com/api/global/delayed_quotes/options/_X{sym}.json"
    data = await _fetch(url)
    return float(data["data"]["current_price"])

async def realised_vol(symbol: str, days: int = 21) -> float:
    tk = yf.Ticker(symbol)
    px = tk.history(period=f"{int(days*1.5)}d").Close
    rt = np.log(px / px.shift(1)).dropna()
    return float(rt.std() * np.sqrt(252) * 100)

# ---------- metrics ----------
def har_rv(rv_series: pd.Series) -> float:
    df = pd.DataFrame({"rv": rv_series})
    df["rv1"] = df.rv.shift(1)
    df["rv5"] = df.rv.rolling(5).mean().shift(1)
    df["rv22"] = df.rv.rolling(22).mean().shift(1)
    df = df.dropna()
    if len(df) < 10:
        return float(df.rv.mean())  # fallback
    y, X = df.rv, df[["rv1", "rv5", "rv22"]]
    β, *_ = linalg.lstsq(X, y)
    forecast = β @ np.array([df.rv1.iloc[-1], df.rv5.iloc[-1], df.rv22.iloc[-1]])
    return float(forecast)

def vrp(iv: float, rv_expected: float) -> float:
    return iv - rv_expected

# ---------- scanner ----------
async def _iv_range(cboe_sym: str) -> dict:
    url = f"https://cdn.cboe.com/api/global/delayed_quotes/historical_data/_X{cboe_sym}.json"
    js = await _fetch(url)
    series = pd.Series({pd.to_datetime(d["date"]): float(d["close"]) for d in js["data"]})
    latest, low, high = series.iloc[-1], series.min(), series.max()
    return dict(symbol=cboe_sym, latest=latest, low=low, high=high,
                pct=(latest - low) / (high - low))

async def scan_cheap_vol() -> pd.DataFrame:
    tasks = [asyncio.create_task(_iv_range(s)) for s in CBOE_VOL_MAP.values()]
    rows = await asyncio.gather(*tasks)
    df = pd.DataFrame(rows)
    cheap = df[df["pct"] <= CHEAP_PC]
    return cheap.sort_values("pct")

# ---------- API ----------
app = FastAPI(title="VRP-Platform")

@app.get("/vrp/{asset}")
async def vrp_endpoint(asset: str):
    asset = asset.upper()
    if asset not in TICKERS:
        return {"error": "asset must be SPY, GOLD or YIELD"}
    sym = TICKERS[asset]
    iv, rv = await asyncio.gather(
        cboe_iv(CBOE_VOL_MAP.get(sym, "VIX")),
        realised_vol(sym),
    )
    tk = yf.Ticker(sym)
    rv_series = tk.history(period="6mo").Close.pct_change().rolling(21).std() * np.sqrt(252) * 100
    har = har_rv(rv_series.dropna())
    return {
        "asset": asset,
        "iv": round(iv, 2),
        "rv_21d": round(rv, 2),
        "rv_har": round(har, 2),
        "vrp_classic": round(vrp(iv, rv), 2),
        "vrp_har": round(vrp(iv, har), 2),
    }

@app.get("/cheap_vol")
async def cheap_vol():
    return (await scan_cheap_vol()).to_dict(orient="records")

# ---------- scheduler ----------
async def snapshot_job():
    records = []
    for asset, sym in TICKERS.items():
        iv, rv = await asyncio.gather(cboe_iv(CBOE_VOL_MAP.get(sym, "VIX")), realised_vol(sym))
        tk = yf.Ticker(sym)
        rv_series = tk.history(period="6mo").Close.pct_change().rolling(21).std() * np.sqrt(252) * 100
        har = har_rv(rv_series.dropna())
        records.append({
            "ts": datetime.utcnow(),
            "asset": asset,
            "iv": iv,
            "rv": rv,
            "rv_har": har,
            "vrp": vrp(iv, har),
        })
    df = pd.DataFrame(records)
    if PARQUET_FILE.exists():
        old = pd.read_parquet(PARQUET_FILE)
        df = pd.concat([old, df], ignore_index=True)
    df.to_parquet(PARQUET_FILE, engine="pyarrow")
    logging.info("VRP snapshot saved (%s)", df.iloc[-1]["ts"])

def run_scheduler():
    schedule.every().day.at("16:30").do(lambda: asyncio.run(snapshot_job()))
    logging.info("Scheduler started – next snapshot at 16:30 ET")
    while True:
        schedule.run_pending()
        time.sleep(30)

# ---------- entry ----------
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "api":
        # launch:  python vrp_platform.py api
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        run_scheduler()