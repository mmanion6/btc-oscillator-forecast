import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import plotly.graph_objects as go
import yfinance as yf
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="BTC Forecast", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
    "<style>#MainMenu{visibility:hidden;}footer{visibility:hidden;}header{visibility:hidden;}</style>",
    unsafe_allow_html=True,
)

# Auto-refresh every 60 seconds
st_autorefresh(interval=60_000, key="btc_refresh")

st.title("Bitcoin Damped-Oscillator Model — Full Cycle History & Forecast (2012–2040)")
st.caption("Live data · auto-refreshes every 60 s · hover or drag to explore · scroll to zoom")

# ── Pre-2014 history from CSV (static, cached 24h) ───────────────────────────
CUTOVER = pd.Timestamp("2014-09-17")  # first date yfinance has BTC-USD

@st.cache_data(ttl=86400)
def load_csv_history():
    df = pd.read_csv("BTC_All_graph_coinmarketcap.csv", sep=";")
    df["price"] = pd.to_numeric(
        df["price"].astype(str).str.replace(",", "").str.strip(), errors="coerce"
    )
    df = df.dropna(subset=["price"]).reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df[df["timestamp"] < CUTOVER][["timestamp", "price"]].copy()

# ── 2014-present daily closes + today's live price (5-min cache) ─────────────
@st.cache_data(ttl=300)
def load_yf_data():
    # Full daily history from yfinance
    hist = yf.download("BTC-USD", period="max", interval="1d", progress=False, auto_adjust=True)
    hist = hist[["Close"]].reset_index()
    hist.columns = ["timestamp", "price"]
    hist["timestamp"] = pd.to_datetime(hist["timestamp"]).dt.tz_localize(None)
    hist["price"] = hist["price"].astype(float)
    hist = hist.dropna().sort_values("timestamp").reset_index(drop=True)

    # Today's intraday — get the freshest price available
    live_price, live_ts = None, None
    try:
        intra = yf.download("BTC-USD", period="1d", interval="1m", progress=False, auto_adjust=True)
        if not intra.empty:
            live_price = float(intra["Close"].iloc[-1])
            ts = intra.index[-1]
            live_ts = ts.tz_localize(None) if ts.tzinfo else pd.Timestamp(ts)
    except Exception:
        pass

    # Append live price if it's newer than the last daily bar
    if live_price and live_ts and live_ts > hist["timestamp"].max():
        hist = pd.concat(
            [hist, pd.DataFrame({"timestamp": [live_ts], "price": [live_price]})],
            ignore_index=True,
        )
        return hist, live_price, live_ts

    return hist, live_price, hist["timestamp"].max()

csv_df = load_csv_history()
yf_df, live_price, live_ts = load_yf_data()

# Merge: CSV pre-2014 + yfinance 2014-now (drop overlap)
df = pd.concat([csv_df, yf_df], ignore_index=True)
df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)

last_date = df["timestamp"].max()
genesis = pd.to_datetime("2009-01-01")
df["t"] = (df["timestamp"] - genesis).dt.total_seconds() / (24 * 3600 * 365.25)
df["log_price"] = np.log10(df["price"].astype(float))

# ── Model ─────────────────────────────────────────────────────────────────────
def btc_damped_osc(t, a, b, A, gamma, omega, phi):
    return a + b * np.log(t) + A * np.exp(-gamma * t) * np.sin(omega * t + phi)

def osc_component(t, A, gamma, omega, phi):
    return A * np.exp(-gamma * t) * np.sin(omega * t + phi)

popt, _ = curve_fit(
    btc_damped_osc, df["t"], df["log_price"],
    p0=[-18, 1.15, 0.5, 0.05, 1.57, 0], maxfev=5000,
)
a, b, A, gamma, omega, phi = popt

# ── Unified timeline ──────────────────────────────────────────────────────────
chart_start = pd.to_datetime("2012-01-01")
chart_end   = pd.to_datetime("2040-12-31")
all_dates   = pd.date_range(chart_start, chart_end, freq="D")
all_t       = ((all_dates - genesis).total_seconds() / (24 * 3600)) / 365.25

all_pred_price = 10 ** btc_damped_osc(all_t, *popt)
all_osc        = osc_component(all_t, A, gamma, omega, phi)
all_trend      = 10 ** (a + b * np.log(all_t))
thresh         = 0.20 * np.abs(A * np.exp(-gamma * all_t))

all_df = pd.DataFrame({
    "timestamp":  all_dates,
    "pred_price": all_pred_price,
    "osc":        all_osc,
    "trend":      all_trend,
    "is_hist":    all_dates <= last_date,
})
all_df["regime"] = np.where(
    all_df["osc"] < -thresh, "buy",
    np.where(all_df["osc"] > thresh, "ath", "consolidation"),
)

future_df = all_df[all_df["timestamp"] > last_date].copy().reset_index(drop=True)

# ── Peak / trough detection ───────────────────────────────────────────────────
hist_chart = df[df["timestamp"] >= chart_start].copy().reset_index(drop=True)

peaks_idx, _ = find_peaks(hist_chart["price"].values,
                           prominence=hist_chart["price"].values.max() * 0.10, distance=180)
hist_aths = hist_chart.iloc[peaks_idx][["timestamp", "price"]].copy()

trough_idx, _ = find_peaks(-hist_chart["price"].values,
                            prominence=hist_chart["price"].values.max() * 0.02, distance=180)
hist_lows = hist_chart.iloc[trough_idx][["timestamp", "price"]].copy()
hist_lows = hist_lows[hist_lows["price"] < hist_chart["price"].max() * 0.50]

analysis_start = pd.to_datetime("2026-04-08")
fut_an = future_df[future_df["timestamp"] >= analysis_start].copy().reset_index(drop=True)
fp_idx, _ = find_peaks(fut_an["osc"].values, prominence=0.05, distance=200)
fut_aths = fut_an.iloc[fp_idx][["timestamp", "pred_price"]].copy()
ft_idx, _ = find_peaks(-fut_an["osc"].values, prominence=0.05, distance=200)
fut_lows = fut_an.iloc[ft_idx][["timestamp", "pred_price"]].copy()

# ── Halvings ──────────────────────────────────────────────────────────────────
halving_list = [
    ("2012-11-28", "2012 Halving"),
    ("2016-07-09", "2016 Halving"),
    ("2020-05-11", "2020 Halving"),
    ("2024-04-19", "2024 Halving"),
    ("2028-04-20", "2028 Halving"),
    ("2032-04-20", "2032 Halving"),
    ("2036-04-19", "2036 Halving"),
    ("2040-04-19", "2040 Halving"),
]

# ── Helper: price label ───────────────────────────────────────────────────────
def plabel(p):
    if p >= 1e6:  return f"${p/1e6:.2f}M"
    if p >= 1e3:  return f"${p/1e3:.0f}K"
    return f"${p:,.0f}"

# ── Build Plotly figure ───────────────────────────────────────────────────────
fig = go.Figure()

# 1. Regime background bands
regime_cfg = {
    "buy":           ("Buy Zone",           "rgba(180,240,180,0.45)"),
    "consolidation": ("Consolidation",      "rgba(180,210,255,0.40)"),
    "ath":           ("ATH / Sell Zone",    "rgba(255,210,160,0.45)"),
}

for regime, (label, color) in regime_cfg.items():
    mask = all_df["regime"] == regime
    segs = all_df[mask].copy()
    # Trace as a filled area by stacking y2=very high, y=very low
    fig.add_trace(go.Scatter(
        x=pd.concat([all_df["timestamp"][mask], all_df["timestamp"][mask].iloc[::-1]]),
        y=[1e1] * mask.sum() + [1e9] * mask.sum(),
        fill="toself",
        fillcolor=color,
        line=dict(width=0),
        mode="none",
        name=label,
        legendgroup=regime,
        showlegend=True,
        hoverinfo="skip",
    ))

# 2. Regime-colored model fit lines (historical)
line_cfg = {
    "buy":           ("#00aa00", "dash"),
    "consolidation": ("#1e90ff", "dot"),
    "ath":           ("#e07000", "solid"),
}
label_hist = {
    "buy": "Buy Zone — hist fit",
    "consolidation": "Consolidation — hist fit",
    "ath": "ATH/Sell — hist fit",
}
label_fut = {
    "buy": "Buy Zone — forecast",
    "consolidation": "Consolidation — forecast",
    "ath": "ATH/Sell — forecast",
}

for regime, (color, dash) in line_cfg.items():
    for subset, lw, lbl in [
