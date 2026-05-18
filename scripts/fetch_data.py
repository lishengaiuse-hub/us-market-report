"""
fetch_data.py
Fetches all market data needed for the weekly report.
Data sources:
  - yfinance       : indices, ETFs, commodities, DXY, 10Y yield (free, no key)
  - FRED API       : macro economic data (free, optional FRED_API_KEY env var)
  - CNN F&G proxy  : Fear & Greed scraped from CNN (fallback: alternative.me crypto index)
  - pizzint.watch  : Pentagon Pizza Index (scraped, fallback provided)
"""

import os
import json
import time
import logging
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

FRED_API_KEY = os.environ.get("FRED_API_KEY", "")
TODAY = datetime.utcnow()
YEAR_START = f"{TODAY.year}-01-01"
ONE_YEAR_AGO = (TODAY - timedelta(days=365)).strftime("%Y-%m-%d")
TWELVE_MONTHS_AGO = (TODAY - timedelta(days=370)).strftime("%Y-%m-%d")


# ─────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────

def safe(fn, default=None):
    try:
        return fn()
    except Exception as e:
        log.warning(f"safe() caught: {e}")
        return default


def calculate_rsi(prices: list, period: int = 14) -> float | None:
    """Wilder's smoothed RSI."""
    if len(prices) < period + 2:
        return None
    s = pd.Series(prices, dtype=float)
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    val = float(rsi.iloc[-1])
    return round(val, 2) if not np.isnan(val) else None


def ytd_pct(hist: pd.DataFrame) -> float | None:
    if hist is None or len(hist) < 2:
        return None
    first = float(hist["Close"].iloc[0])
    last = float(hist["Close"].iloc[-1])
    if first == 0:
        return None
    return round((last - first) / first * 100, 2)


def twelve_month_trend(hist: pd.DataFrame, n_points: int = 12) -> list[float]:
    """Return ~12 evenly-spaced closing prices for sparkline charts."""
    if hist is None or hist.empty:
        return []
    closes = hist["Close"].dropna()
    if len(closes) < 2:
        return []
    idx = np.linspace(0, len(closes) - 1, n_points, dtype=int)
    return [round(float(closes.iloc[i]), 2) for i in idx]


def fred(series_id: str, limit: int = 3) -> tuple[float | None, str | None]:
    """Fetch latest value from FRED."""
    if not FRED_API_KEY:
        return None, None
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "limit": limit,
        "sort_order": "desc",
        "observation_start": ONE_YEAR_AGO,
    }
    try:
        r = requests.get(url, params=params, timeout=12)
        obs = r.json().get("observations", [])
        for o in obs:
            try:
                return round(float(o["value"]), 4), o["date"]
            except (ValueError, KeyError):
                continue
    except Exception as e:
        log.warning(f"FRED {series_id}: {e}")
    return None, None


# ─────────────────────────────────────────────
# INDIVIDUAL FETCHERS
# ─────────────────────────────────────────────

def fetch_index(symbol: str, label: str) -> dict:
    log.info(f"Fetching index: {symbol}")
    tk = yf.Ticker(symbol)
    hist_1y = safe(lambda: tk.history(start=TWELVE_MONTHS_AGO), pd.DataFrame())
    hist_10y = safe(lambda: tk.history(period="10y"), pd.DataFrame())
    info = safe(lambda: tk.info, {})

    price = safe(lambda: float(hist_1y["Close"].iloc[-1]))
    prev = safe(lambda: float(hist_1y["Close"].iloc[-2]))
    chg_pct = round((price - prev) / prev * 100, 2) if price and prev else None
    rsi = calculate_rsi(hist_1y["Close"].tolist() if not hist_1y.empty else [])
    pe = safe(lambda: round(float(info.get("trailingPE") or info.get("forwardPE") or 0), 2)) or None
    ytd = ytd_pct(safe(lambda: tk.history(start=YEAR_START), None))
    trend_10y = twelve_month_trend(hist_10y, n_points=24)

    return {
        "symbol": symbol, "label": label,
        "price": price, "change_pct": chg_pct,
        "pe": pe, "rsi": rsi, "ytd": ytd,
        "trend_10y": trend_10y,
        "date": TODAY.strftime("%b %d, %Y"),
    }


def fetch_vix() -> dict:
    log.info("Fetching VIX / VXN")
    vix_hist = safe(lambda: yf.Ticker("^VIX").history(period="5d"), pd.DataFrame())
    vxn_hist = safe(lambda: yf.Ticker("^VXN").history(period="5d"), pd.DataFrame())
    return {
        "vix": round(float(vix_hist["Close"].iloc[-1]), 2) if not vix_hist.empty else None,
        "vxn": round(float(vxn_hist["Close"].iloc[-1]), 2) if not vxn_hist.empty else None,
        "date": TODAY.strftime("%b %d, %Y"),
    }


def fetch_fear_greed() -> dict:
    log.info("Fetching Fear & Greed Index")
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        data = r.json()["data"][0]
        return {
            "value": int(data["value"]),
            "label": data["value_classification"],
            "date": datetime.utcfromtimestamp(int(data["timestamp"])).strftime("%b %d, %Y"),
        }
    except Exception as e:
        log.warning(f"F&G API: {e}")
        return {"value": None, "label": "N/A", "date": TODAY.strftime("%b %d, %Y")}


def fetch_macro() -> dict:
    log.info("Fetching macro data via FRED + yfinance fallback")

    # ── FRED (requires FRED_API_KEY env var) ──────────────────────
    fed_rate, fed_date = fred("FEDFUNDS")
    t10y_fred, t10y_date_fred = fred("DGS10")
    dxy_fred, dxy_date_fred = fred("DTWEXBGS")
    cpi, cpi_date = fred("CPIAUCSL")
    cpi_hist_val, _ = fred("CPIAUCSL", limit=14)
    core_cpi, core_date = fred("CPILFESL")
    core_hist, _ = fred("CPILFESL", limit=14)
    unrate, ur_date = fred("UNRATE")
    icsa, icsa_date = fred("ICSA")
    nfp, nfp_date = fred("PAYEMS")
    nfp_prev, _ = fred("PAYEMS", limit=3)
    ism_mfg, ism_mfg_date = fred("NAPM")
    ism_svc, ism_svc_date = fred("NMFCI")

    cpi_yoy = round((cpi - cpi_hist_val) / cpi_hist_val * 100, 2) if cpi and cpi_hist_val else None
    core_yoy = round((core_cpi - core_hist) / core_hist * 100, 2) if core_cpi and core_hist else None
    # PAYEMS monthly change: level in thousands, limit=3 gets ~2 months prior
    nfp_change = round(nfp - nfp_prev) if nfp and nfp_prev else None  # already in K

    # ── yfinance fallback for market-priced data ───────────────────
    # 10-Year Treasury yield via ^TNX (CBOE index, value = yield %)
    t10y, t10y_date = t10y_fred, t10y_date_fred
    if t10y is None:
        log.info("FRED DGS10 unavailable; falling back to ^TNX (yfinance)")
        tnx_hist = safe(lambda: yf.Ticker("^TNX").history(period="5d"), pd.DataFrame())
        if not tnx_hist.empty:
            val = float(tnx_hist["Close"].iloc[-1])
            t10y = round(val if val < 20 else val / 10, 2)
            t10y_date = tnx_hist.index[-1].strftime("%Y-%m-%d")

    # DXY dollar index via DX-Y.NYB (ICE Dollar Index)
    dxy, dxy_date = dxy_fred, dxy_date_fred
    if dxy is None:
        log.info("FRED DTWEXBGS unavailable; falling back to DX-Y.NYB (yfinance)")
        dxy_hist = safe(lambda: yf.Ticker("DX-Y.NYB").history(period="5d"), pd.DataFrame())
        if not dxy_hist.empty:
            dxy = round(float(dxy_hist["Close"].iloc[-1]), 2)
            dxy_date = dxy_hist.index[-1].strftime("%Y-%m-%d")

    return {
        "fed_rate": fed_rate, "fed_date": fed_date,
        "t10y": t10y, "t10y_date": t10y_date,
        "dxy": dxy, "dxy_date": dxy_date,
        "cpi_yoy": cpi_yoy, "cpi_date": cpi_date,
        "core_cpi_yoy": core_yoy, "core_date": core_date,
        "unrate": unrate, "ur_date": ur_date,
        "nfp_change": nfp_change, "nfp_date": nfp_date,
        "icsa": int(icsa / 1000) if icsa else None, "icsa_date": icsa_date,
        "ism_mfg": ism_mfg, "ism_mfg_date": ism_mfg_date,
        "ism_svc": ism_svc, "ism_svc_date": ism_svc_date,
        "date": TODAY.strftime("%b %d, %Y"),
    }


def fetch_cme_fedwatch() -> dict:
    """Approximate FedWatch via 30-day Fed Funds Futures (ZQ)."""
    log.info("Fetching CME FedWatch proxy")
    # ZQ=F is the nearest Fed Funds futures contract
    try:
        tk = yf.Ticker("ZQM26.CBT")  # adjust contract month as needed
        hist = tk.history(period="5d")
        if not hist.empty:
            price = float(hist["Close"].iloc[-1])
            implied_rate = round(100 - price, 3)
            return {"implied_rate": implied_rate, "cuts_priced": None, "date": TODAY.strftime("%b %d, %Y")}
    except Exception as e:
        log.warning(f"FedWatch proxy: {e}")
    return {"implied_rate": None, "cuts_priced": 0, "date": TODAY.strftime("%b %d, %Y")}


def fetch_sector(symbol: str, name_en: str, name_cn: str) -> dict:
    log.info(f"Fetching sector: {symbol}")
    tk = yf.Ticker(symbol)
    hist_1y = safe(lambda: tk.history(start=TWELVE_MONTHS_AGO), pd.DataFrame())
    hist_ytd = safe(lambda: tk.history(start=YEAR_START), pd.DataFrame())

    price = safe(lambda: float(hist_1y["Close"].iloc[-1]))
    rsi = calculate_rsi(hist_1y["Close"].tolist() if not hist_1y.empty else [])
    ytd = ytd_pct(hist_ytd)
    trend = twelve_month_trend(hist_1y, n_points=12)

    # Momentum classification based on RSI + YTD
    if rsi and ytd:
        if rsi > 55 and ytd > 5:
            momentum = "Leading"
        elif rsi > 50 and ytd > 0:
            momentum = "Improving"
        elif rsi < 45 and ytd < 0:
            momentum = "Lagging"
        else:
            momentum = "Weakening"
    else:
        momentum = "Neutral"

    return {
        "symbol": symbol, "name_en": name_en, "name_cn": name_cn,
        "price": price, "ytd": ytd, "rsi": rsi,
        "momentum": momentum, "trend": trend,
    }


def fetch_semiconductor() -> dict:
    log.info("Fetching semiconductor data")
    # SOX proxy via SOXX ETF
    soxx = yf.Ticker("SOXX")
    hist_1y = safe(lambda: soxx.history(start=TWELVE_MONTHS_AGO), pd.DataFrame())
    hist_ytd = safe(lambda: soxx.history(start=YEAR_START), pd.DataFrame())
    info = safe(lambda: soxx.info, {})

    price = safe(lambda: float(hist_1y["Close"].iloc[-1]))
    rsi = calculate_rsi(hist_1y["Close"].tolist() if not hist_1y.empty else [])
    ytd = ytd_pct(hist_ytd)
    trend = twelve_month_trend(hist_1y, n_points=12)
    week52_low = safe(lambda: round(float(info.get("fiftyTwoWeekLow", 0)), 2))
    week52_high = safe(lambda: round(float(info.get("fiftyTwoWeekHigh", 0)), 2))

    # SOX index (approximated via ^SOX)
    sox_tk = yf.Ticker("^SOX")
    sox_hist = safe(lambda: sox_tk.history(start=TWELVE_MONTHS_AGO), pd.DataFrame())
    sox_price = safe(lambda: round(float(sox_hist["Close"].iloc[-1]), 0))
    sox_rsi = calculate_rsi(sox_hist["Close"].tolist() if sox_hist is not None and not sox_hist.empty else [])
    sox_trend = twelve_month_trend(sox_hist, n_points=12) if sox_hist is not None else []

    return {
        "soxx_price": price, "soxx_rsi": rsi, "soxx_ytd": ytd,
        "soxx_trend": trend, "soxx_52w_low": week52_low, "soxx_52w_high": week52_high,
        "sox_price": sox_price, "sox_rsi": sox_rsi, "sox_trend": sox_trend,
        "date": TODAY.strftime("%b %d, %Y"),
    }


def fetch_commodity(symbol: str, name_en: str, name_cn: str, emoji: str, unit: str,
                    price_scale: float = 1.0) -> dict:
    """
    price_scale: multiplier applied to raw yfinance price and trend values.
    Use 0.01 for grain futures quoted in cents/bushel (ZS=F, ZC=F) to convert to USD.
    """
    log.info(f"Fetching commodity: {symbol}")
    tk = yf.Ticker(symbol)
    hist_1y = safe(lambda: tk.history(start=TWELVE_MONTHS_AGO), pd.DataFrame())
    hist_ytd = safe(lambda: tk.history(start=YEAR_START), pd.DataFrame())

    raw_price = safe(lambda: float(hist_1y["Close"].iloc[-1]))
    price = round(raw_price * price_scale, 2) if raw_price is not None else None
    rsi = calculate_rsi(hist_1y["Close"].tolist() if not hist_1y.empty else [])
    ytd = ytd_pct(hist_ytd)   # ratio-based, unaffected by scale
    raw_trend = twelve_month_trend(hist_1y, n_points=12)
    trend = [round(x * price_scale, 2) for x in raw_trend]

    return {
        "symbol": symbol, "name_en": name_en, "name_cn": name_cn,
        "emoji": emoji, "unit": unit,
        "price": price, "ytd": ytd, "rsi": rsi, "trend": trend,
        "date": TODAY.strftime("%b %d, %Y"),
    }


def fetch_pizza_index() -> dict:
    """
    Scrape pizzint.watch for current DOUGHCON level.
    Falls back to DOUGHCON 5 (normal) if unavailable.
    """
    log.info("Fetching Pentagon Pizza Index")
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; MarketReportBot/1.0)"}
        r = requests.get("https://pizzint.watch", headers=headers, timeout=12)
        text = r.text.upper()
        for lvl in [1, 2, 3, 4, 5]:
            if f"DOUGHCON {lvl}" in text or f"DOUGHCON{lvl}" in text:
                return {
                    "level": lvl,
                    "status": ["最高警戒", "极端飙升", "显著上升", "轻微异常", "正常"][lvl - 1],
                    "date": TODAY.strftime("%b %d, %Y"),
                    "source": "pizzint.watch",
                }
    except Exception as e:
        log.warning(f"PizzINT scrape failed: {e}")

    return {
        "level": 5,
        "status": "正常 (数据暂不可用)",
        "date": TODAY.strftime("%b %d, %Y"),
        "source": "fallback",
    }


# ─────────────────────────────────────────────
# MAIN ORCHESTRATOR
# ─────────────────────────────────────────────

SECTORS = [
    ("XLE", "Energy",                   "能源"),
    ("XLI", "Industrials",              "工业"),
    ("XLB", "Materials",                "材料"),
    ("XLP", "Consumer Staples",         "必需消费"),
    ("XLU", "Utilities",                "公用事业"),
    ("XLF", "Financials",               "金融"),
    ("XLV", "Health Care",              "医疗"),
    ("XLC", "Comm. Services",           "通信"),
    ("XLY", "Consumer Discr.",          "非必需消费"),
    ("XLK", "Info. Technology",         "信息技术"),
]

COMMODITIES = [
    # (symbol, name_en, name_cn, emoji, unit, price_scale)
    ("GC=F",  "Gold",          "黄金", "🥇", "USD/oz",           1.0),
    ("SI=F",  "Silver",        "白银", "🥈", "USD/oz",           1.0),
    ("REMX",  "Rare Earth",    "稀土", "⚗️",  "USD (ETF)",        1.0),
    ("BTU",   "Thermal Coal",  "煤炭", "⚫", "USD/share (BTU)",  1.0),   # KOL delisted → Peabody Energy
    ("CT=F",  "Cotton #2",     "棉花", "🌿", "cents/lb",         1.0),
    ("ZS=F",  "Soybeans",      "大豆", "🫘", "USD/bu",           0.01),  # yfinance returns cents/bu
]


def run() -> dict:
    data = {}

    # Indices
    data["spx"] = fetch_index("^GSPC", "S&P 500")
    time.sleep(0.5)
    data["ndx"] = fetch_index("^NDX", "Nasdaq 100")
    time.sleep(0.5)

    # Volatility
    data["volatility"] = fetch_vix()
    time.sleep(0.5)

    # Sentiment
    data["fear_greed"] = fetch_fear_greed()
    time.sleep(0.5)

    # Macro
    data["macro"] = fetch_macro()
    time.sleep(0.5)

    # FedWatch proxy
    data["fedwatch"] = fetch_cme_fedwatch()
    time.sleep(0.5)

    # Sectors
    data["sectors"] = []
    for sym, en, cn in SECTORS:
        data["sectors"].append(fetch_sector(sym, en, cn))
        time.sleep(0.3)

    # Semiconductor
    data["semiconductor"] = fetch_semiconductor()
    time.sleep(0.5)

    # Commodities
    data["commodities"] = []
    for sym, en, cn, emoji, unit, scale in COMMODITIES:
        data["commodities"].append(fetch_commodity(sym, en, cn, emoji, unit, scale))
        time.sleep(0.3)

    # Pizza Index
    data["pizza"] = fetch_pizza_index()

    # Report metadata
    data["generated_at"] = TODAY.strftime("%b %d, %Y — %H:%M UTC")
    data["report_date"] = TODAY.strftime("%b %d, %Y")
    data["report_day"] = TODAY.strftime("%A").upper()

    return data


if __name__ == "__main__":
    result = run()
    output_path = os.path.join(os.path.dirname(__file__), "..", "output", "data.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    log.info(f"Data written to {output_path}")
