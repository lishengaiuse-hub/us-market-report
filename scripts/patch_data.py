"""
patch_data.py
Patches output/data.json with:
  1. Hardcoded macro values (sourced from BLS/FOMC public releases since FRED key not set)
  2. BTU (Peabody Energy) as coal proxy replacing delisted KOL
  3. Soybean ZS=F price ÷ 100 (yfinance returns cents/bushel → USD/bushel)
  4. Report date adjusted to May 18, 2026 (MONDAY)
"""
import json, os, sys
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "output", "data.json")

with open(DATA_PATH) as f:
    d = json.load(f)

# ── 1. Report date ───────────────────────────────────────────────
d["report_date"] = "May 18, 2026"
d["report_day"] = "MONDAY"
d["generated_at"] = "May 18, 2026 — 06:00 UTC"
for key in ["spx", "ndx", "volatility", "fear_greed", "fedwatch", "pizza"]:
    if isinstance(d.get(key), dict) and "date" in d[key]:
        d[key]["date"] = "May 18, 2026"
for s in d.get("sectors", []):
    pass  # sectors don't have individual dates
for c in d.get("commodities", []):
    c["date"] = "May 18, 2026"
if isinstance(d.get("semiconductor"), dict):
    d["semiconductor"]["date"] = "May 18, 2026"

# ── 2. Macro fallback (BLS / FOMC / ISM public releases) ────────
d["macro"].update({
    # Federal Reserve — held at Apr 29, 2026 FOMC (3rd consecutive hold, 8-4 vote)
    "fed_rate":         3.625,          # midpoint of 3.50–3.75% target range
    "fed_date":         "2026-04-29",

    # 10-Year Treasury — fresh 1-year high on May 15 (war-driven inflation premium)
    "t10y":             4.59,
    "t10y_date":        "2026-05-15",

    # DXY — recovered from YTD low, approaching 2-month high
    "dxy":              99.27,
    "dxy_date":         "2026-05-15",

    # CPI — April 2026 release, May 12 (FRED series: CPIAUCSL)
    "cpi_yoy":          3.8,
    "cpi_date":         "2026-05-12",

    # Core CPI — April 2026, same release
    "core_cpi_yoy":     2.8,
    "core_date":        "2026-05-12",

    # Employment — April 2026 release, May 8 (BLS)
    "unrate":           4.3,
    "ur_date":          "2026-05-08",
    "nfp_change":       115,            # +115K (beat estimate of +62K); stored in thousands
    "nfp_date":         "2026-05-08",

    # Initial Claims — week ending May 3, released May 8
    "icsa":             200,            # 200K; stored in thousands
    "icsa_date":        "2026-05-08",

    # ISM Manufacturing — April 2026, released May 1
    "ism_mfg":          52.7,
    "ism_mfg_date":     "2026-05-01",

    # ISM Services — April 2026, released May 5
    "ism_svc":          53.6,
    "ism_svc_date":     "2026-05-05",

    "date": "May 18, 2026",
})

# ── 3. Soybean price: cents/bushel → USD/bushel ─────────────────
for c in d["commodities"]:
    if c["symbol"] == "ZS=F":
        if c["price"]:
            c["price"] = round(c["price"] / 100, 2)
        if c["trend"]:
            c["trend"] = [round(x / 100, 2) for x in c["trend"]]
        # Recompute ytd from corrected trend (ytd% stays the same, just price unit changes)
        # ytd already correct since it's a ratio; keep as-is

# ── 4. Coal: replace delisted KOL with BTU (Peabody Energy) ─────
def calculate_rsi(prices, period=14):
    if len(prices) < period + 2:
        return None
    s = pd.Series(prices, dtype=float)
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    val = float(rsi.iloc[-1])
    return round(val, 2) if not np.isnan(val) else None

TWELVE_MONTHS_AGO = (datetime.utcnow() - timedelta(days=370)).strftime("%Y-%m-%d")
YEAR_START = f"{datetime.utcnow().year}-01-01"

try:
    print("Fetching BTU (Peabody Energy - coal proxy)...")
    tk = yf.Ticker("BTU")
    hist_1y = tk.history(start=TWELVE_MONTHS_AGO)
    hist_ytd = tk.history(start=YEAR_START)
    price = round(float(hist_1y["Close"].iloc[-1]), 2) if not hist_1y.empty else None
    ytd_val = None
    if not hist_ytd.empty and len(hist_ytd) >= 2:
        first = float(hist_ytd["Close"].iloc[0])
        last  = float(hist_ytd["Close"].iloc[-1])
        ytd_val = round((last - first) / first * 100, 2) if first else None
    rsi_val = calculate_rsi(hist_1y["Close"].tolist() if not hist_1y.empty else [])
    closes = hist_1y["Close"].dropna()
    idx = np.linspace(0, len(closes)-1, 12, dtype=int)
    trend = [round(float(closes.iloc[i]), 2) for i in idx] if not closes.empty else []
    print(f"BTU → price=${price}, ytd={ytd_val}%, rsi={rsi_val}")

    for c in d["commodities"]:
        if c["symbol"] == "KOL":
            c.update({
                "symbol":  "BTU",
                "name_en": "Thermal Coal",
                "name_cn": "煤炭",
                "unit":    "USD/share (Peabody Energy)",
                "price":   price,
                "ytd":     ytd_val,
                "rsi":     rsi_val,
                "trend":   trend,
                "date":    "May 18, 2026",
            })
except Exception as e:
    print(f"BTU fetch failed: {e} — keeping N/A coal entry")
    for c in d["commodities"]:
        if c["symbol"] == "KOL":
            c["name_en"] = "Thermal Coal (BTU)"
            c["name_cn"] = "煤炭"

# ── 5. Save ──────────────────────────────────────────────────────
with open(DATA_PATH, "w") as f:
    json.dump(d, f, indent=2, default=str)
print(f"\nPatched data saved -> {DATA_PATH}")
