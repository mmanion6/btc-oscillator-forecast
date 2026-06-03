import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import yfinance as yf

st.set_page_config(page_title="BTC Forecast", layout="wide", initial_sidebar_state="collapsed")

hide = """<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}</style>"""
st.markdown(hide, unsafe_allow_html=True)

st.title("Bitcoin Damped-Oscillator Model — Full Cycle History & Forecast (2012–2040)")
st.caption("Live data • Buy / Consolidation / ATH Zones")

# ── Real-time price (cached 5 min) ───────────────────────────────────────────
@st.cache_data(ttl=300)
def get_live_price():
    """Returns (price, timestamp) using 1-min data for near-real-time freshness."""
    try:
        tick = yf.download("BTC-USD", period="1d", interval="1m", progress=False)
        if not tick.empty:
            price = float(tick["Close"].iloc[-1])
            ts    = tick.index[-1].tz_localize(None) if tick.index[-1].tzinfo else tick.index[-1]
            return price, pd.Timestamp(ts)
    except Exception:
        pass
    try:
        tick = yf.download("BTC-USD", period="2d", interval="1d", progress=False)
        if not tick.empty:
            price = float(tick["Close"].iloc[-1])
            ts    = pd.Timestamp(tick.index[-1].date())
            return price, ts
    except Exception:
        pass
    return None, None

# ── Historical data (cached 1 hour) ──────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_data():
    df = pd.read_csv("BTC_All_graph_coinmarketcap.csv", sep=";")
    df["price"] = pd.to_numeric(
        df["price"].astype(str).str.replace(",", "").str.strip(), errors="coerce"
    )
    df = df.dropna(subset=["price"]).reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

df = load_data()

# Splice in live price
live_price, live_ts = get_live_price()
if live_price and live_ts and live_ts > df["timestamp"].max():
    df = pd.concat(
        [df, pd.DataFrame({"timestamp": [live_ts], "price": [live_price]})],
        ignore_index=True,
    )

last_date = df["timestamp"].max()

genesis = pd.to_datetime("2009-01-01")
df["t"]         = (df["timestamp"] - genesis).dt.total_seconds() / (24 * 3600 * 365.25)
df["log_price"] = np.log10(df["price"])

# ── Model fit ─────────────────────────────────────────────────────────────────
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
all_envelope   = A * np.exp(-gamma * all_t)
thresh         = 0.20 * np.abs(all_envelope)

all_df = pd.DataFrame({
    "timestamp":     all_dates,
    "pred_price":    all_pred_price,
    "osc":           all_osc,
    "trend_price":   all_trend,
    "consol_thresh": thresh,
    "is_hist":       all_dates <= last_date,
})
all_df["regime"] = np.where(
    all_df["osc"] < -all_df["consol_thresh"], "buy",
    np.where(all_df["osc"] > all_df["consol_thresh"], "ath", "consolidation"),
)

future_df = all_df[all_df["timestamp"] > last_date].copy().reset_index(drop=True)

# ── Historical ATHs & Cycle Lows ──────────────────────────────────────────────
hist_chart = df[df["timestamp"] >= chart_start].copy().reset_index(drop=True)

hist_peaks_idx, _ = find_peaks(
    hist_chart["price"].values,
    prominence=hist_chart["price"].values.max() * 0.10, distance=180,
)
hist_aths = hist_chart.iloc[hist_peaks_idx][["timestamp", "price"]].copy()

hist_trough_idx, _ = find_peaks(
    -hist_chart["price"].values,
    prominence=hist_chart["price"].values.max() * 0.02, distance=180,
)
hist_lows = hist_chart.iloc[hist_trough_idx][["timestamp", "price"]].copy()
hist_lows = hist_lows[hist_lows["price"] < hist_chart["price"].max() * 0.50].copy()

# ── Future ATHs & Cycle Lows ──────────────────────────────────────────────────
analysis_start = pd.to_datetime("2026-04-08")
fut_analysis   = future_df[future_df["timestamp"] >= analysis_start].copy().reset_index(drop=True)

fut_peaks_idx, _  = find_peaks(fut_analysis["osc"].values, prominence=0.05, distance=200)
fut_aths          = fut_analysis.iloc[fut_peaks_idx][["timestamp", "pred_price"]].copy()
fut_trough_idx, _ = find_peaks(-fut_analysis["osc"].values, prominence=0.05, distance=200)
fut_lows          = fut_analysis.iloc[fut_trough_idx][["timestamp", "pred_price"]].copy()

# ── Y-axis range controls ─────────────────────────────────────────────────────
# Build a list of "nice" price tick values for the slider
PRICE_OPTIONS = [
    100, 200, 500,
    1_000, 2_000, 5_000,
    10_000, 20_000, 50_000,
    100_000, 200_000, 500_000,
    1_000_000, 2_000_000, 5_000_000,
    10_000_000, 20_000_000, 50_000_000,
    100_000_000,
]

def fmt_price(p):
    if p >= 1_000_000:  return f"${p/1_000_000:.0f}M"
    if p >= 1_000:      return f"${p/1_000:.0f}K"
    return f"${p:,.0f}"

price_labels = [fmt_price(p) for p in PRICE_OPTIONS]

col_lo, col_hi = st.columns(2)
with col_lo:
    y_min_label = st.select_slider(
        "Y-axis minimum price",
        options=price_labels,
        value="$100",
    )
with col_hi:
    y_max_label = st.select_slider(
        "Y-axis maximum price",
        options=price_labels,
        value="$100M",
    )

y_min = PRICE_OPTIONS[price_labels.index(y_min_label)]
y_max = PRICE_OPTIONS[price_labels.index(y_max_label)]
if y_min >= y_max:
    st.warning("Minimum must be less than maximum — adjusting.")
    y_max = PRICE_OPTIONS[min(price_labels.index(y_min_label) + 3, len(PRICE_OPTIONS) - 1)]

# ── Chart ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(22, 12))

# Regime background shading
for regime, color in {"buy": "#d4f4d4", "consolidation": "#e0f0ff", "ath": "#ffe6cc"}.items():
    mask = all_df["regime"] == regime
    ax.fill_between(all_df["timestamp"], y_min * 0.5, all_df["pred_price"] * 2,
                    where=mask, color=color, alpha=0.75, zorder=1)

# Actual price
hist_chart_plot = df[df["timestamp"] >= chart_start]
ax.plot(hist_chart_plot["timestamp"], hist_chart_plot["price"],
        color="crimson", linewidth=1.6, zorder=3, label="Actual BTC Price")

# Helper: regime-colored model segments
def plot_regime_segs(df_in, regime_val, color, linestyle, linewidth, zorder, label=None):
    in_seg, seg_start, first = False, None, True
    for pos in range(len(df_in)):
        row = df_in.iloc[pos]
        if row["regime"] == regime_val and not in_seg:
            seg_start, in_seg = pos, True
        elif row["regime"] != regime_val and in_seg:
            seg = df_in.iloc[seg_start:pos + 1]
            lbl = label if first else "_nolegend_"
            ax.plot(seg["timestamp"], seg["pred_price"],
                    color=color, linestyle=linestyle, linewidth=linewidth, zorder=zorder, label=lbl)
            in_seg, first = False, False
    if in_seg:
        seg = df_in.iloc[seg_start:]
        lbl = label if first else "_nolegend_"
        ax.plot(seg["timestamp"], seg["pred_price"],
                color=color, linestyle=linestyle, linewidth=linewidth, zorder=zorder, label=lbl)

hist_model_df = all_df[all_df["is_hist"]].copy().reset_index(drop=True)
plot_regime_segs(hist_model_df, "buy",           "#00aa00", "--", 2.2, 5, "Buy Zone — historical fit")
plot_regime_segs(hist_model_df, "consolidation", "#1e90ff", ":",  2.0, 5, "Consolidation — historical fit")
plot_regime_segs(hist_model_df, "ath",           "#e07000", "-",  2.2, 5, "ATH/Sell — historical fit")

fut_model_df = all_df[~all_df["is_hist"]].copy().reset_index(drop=True)
plot_regime_segs(fut_model_df, "buy",           "limegreen",  "--", 3.5, 5, "Buy Zone — forecast")
plot_regime_segs(fut_model_df, "consolidation", "dodgerblue", ":",  2.5, 5, "Consolidation — forecast")
plot_regime_segs(fut_model_df, "ath",           "darkorange", "-",  3.0, 5, "ATH/Sell — forecast")

# Pure log trend
ax.plot(all_dates, all_trend, color="navy", linestyle=":", linewidth=1.4,
        alpha=0.6, label="Pure Log Trend", zorder=4)

# Present line
ax.axvline(last_date, color="gray", linestyle=":", linewidth=2.2, label="Present", zorder=4)

# Halvings
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
for d_str, _ in halving_list:
    dt = pd.to_datetime(d_str)
    if chart_start <= dt <= chart_end:
        ax.axvline(dt, color="purple", linestyle="--", alpha=0.6, linewidth=1.4)

# CAGR boxes
for i in range(len(halving_list) - 1):
    s  = pd.to_datetime(halving_list[i][0])
    e  = pd.to_datetime(halving_list[i + 1][0])
    t1 = (s - genesis).total_seconds() / (24 * 3600 * 365.25)
    t2 = (e - genesis).total_seconds() / (24 * 3600 * 365.25)
    p1 = 10 ** (a + b * np.log(t1))
    p2 = 10 ** (a + b * np.log(t2))
    cagr = (p2 / p1) ** (1 / (t2 - t1)) - 1
    cycle_name = f"{halving_list[i][0][:4]}–{halving_list[i + 1][0][:4]}"
    mid = s + (e - s) / 2
    cagr_y = max(y_min * 1.5, 450)
    ax.text(mid, cagr_y, f"{cycle_name}\n{cagr*100:.1f}% CAGR",
            ha="center", va="bottom", fontsize=9, color="purple", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.92, edgecolor="purple"))

# Historical ATH markers
for _, r in hist_aths.iterrows():
    if y_min <= r["price"] <= y_max * 2:
        ax.plot(r["timestamp"], r["price"],
                "*", color="gold", markersize=14, markeredgecolor="darkred",
                markeredgewidth=1.5, zorder=7)
        p = r["price"]
        p_str = f"${p/1e6:.2f}M" if p >= 1e6 else f"${p/1e3:.0f}K" if p >= 1e4 else f"${p:,.0f}"
        ax.annotate(f"ATH\n{p_str}",
                    xy=(r["timestamp"], r["price"]),
                    xytext=(0, 28), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, color="darkred", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="darkred", lw=1.0),
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                              alpha=0.92, edgecolor="darkred"), zorder=8)

# Historical Cycle Low markers
for _, r in hist_lows.iterrows():
    if y_min <= r["price"] <= y_max * 2:
        ax.plot(r["timestamp"], r["price"],
                "v", color="limegreen", markersize=10, markeredgecolor="darkgreen",
                markeredgewidth=1.4, zorder=7)
        p = r["price"]
        p_str = f"${p/1e3:.1f}K" if p >= 1e3 else f"${p:,.0f}"
        ax.annotate(f"Cycle Low\n{p_str}",
                    xy=(r["timestamp"], r["price"]),
                    xytext=(0, -30), textcoords="offset points",
                    ha="center", va="top", fontsize=7.5, color="darkgreen", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="darkgreen", lw=1.0),
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="honeydew",
                              alpha=0.92, edgecolor="darkgreen"), zorder=8)

# Future ATH markers
for _, r in fut_aths.iterrows():
    if y_min <= r["pred_price"] <= y_max * 2:
        ax.plot(r["timestamp"], r["pred_price"],
                "*", color="gold", markersize=15, markeredgecolor="darkred",
                markeredgewidth=1.5, zorder=7)
        p = r["pred_price"]
        p_str = f"${p/1e6:.2f}M" if p >= 1e6 else f"${p/1e3:.0f}K"
        ax.annotate(f"Pred ATH\n{p_str}",
                    xy=(r["timestamp"], r["pred_price"]),
                    xytext=(0, 28), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, color="darkred", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="darkred", lw=1.0),
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                              alpha=0.92, edgecolor="darkred"), zorder=8)

# Future Cycle Low markers
for _, r in fut_lows.iterrows():
    if y_min <= r["pred_price"] <= y_max * 2:
        ax.plot(r["timestamp"], r["pred_price"],
                "v", color="limegreen", markersize=10, markeredgecolor="darkgreen",
                markeredgewidth=1.4, zorder=7)
        p = r["pred_price"]
        p_str = f"${p/1e6:.2f}M" if p >= 1e6 else f"${p/1e3:.0f}K"
        ax.annotate(f"Pred Low\n{p_str}",
                    xy=(r["timestamp"], r["pred_price"]),
                    xytext=(0, -30), textcoords="offset points",
                    ha="center", va="top", fontsize=7.5, color="darkgreen", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="darkgreen", lw=1.0),
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="honeydew",
                              alpha=0.92, edgecolor="darkgreen"), zorder=8)

# ── Axes formatting ───────────────────────────────────────────────────────────
ax.set_yscale("log")
ax.set_ylim(y_min, y_max)

def usd_formatter(x, pos):
    if x >= 1_000_000: return f"${x/1_000_000:.1f}M"
    if x >= 10_000:    return f"${x/1_000:.0f}K"
    return f"${x:.0f}"

ax.yaxis.set_major_formatter(FuncFormatter(usd_formatter))
ax.set_xlim(chart_start, chart_end)
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

# Equation box
equation = (
    r"$\log_{10}(P(t)) = a + b \cdot \ln(t) + A \cdot e^{-\gamma t} \cdot \sin(\omega t + \phi)$"
    f"\na={a:.3f}  b={b:.3f}  A={A:.3f}  γ={gamma:.4f}  ω={omega:.4f}  φ={phi:.4f}"
)
ax.text(0.01, 0.975, equation, transform=ax.transAxes,
        fontsize=10.5, va="top", ha="left", family="monospace",
        bbox=dict(boxstyle="round,pad=0.7", facecolor="white", alpha=0.95, edgecolor="#1f77b4"))

# Live price badge (top-right)
if live_price:
    ax.text(0.99, 0.975,
            f"Live BTC: ${live_price:,.0f}",
            transform=ax.transAxes, fontsize=11, va="top", ha="right",
            fontweight="bold", color="crimson",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.95, edgecolor="crimson"))

legend_handles = [
    plt.Line2D([0], [0], color="crimson",    linewidth=1.6,                  label="Actual BTC Price"),
    plt.Line2D([0], [0], color="#00aa00",    linewidth=2.2, linestyle="--",  label="Buy Zone — historical fit"),
    plt.Line2D([0], [0], color="#1e90ff",    linewidth=2.0, linestyle=":",   label="Consolidation — historical fit"),
    plt.Line2D([0], [0], color="#e07000",    linewidth=2.2,                  label="ATH/Sell — historical fit"),
    plt.Line2D([0], [0], color="limegreen",  linewidth=3.5, linestyle="--",  label="Buy Zone — forecast"),
    plt.Line2D([0], [0], color="dodgerblue", linewidth=2.5, linestyle=":",   label="Consolidation — forecast"),
    plt.Line2D([0], [0], color="darkorange", linewidth=3.0,                  label="ATH/Sell — forecast"),
    plt.Line2D([0], [0], color="navy",       linewidth=1.4, linestyle=":",   label="Pure Log Trend", alpha=0.6),
    plt.Line2D([0], [0], color="gray",       linewidth=2.2, linestyle=":",   label="Present"),
    plt.Line2D([0], [0], marker="*",         color="gold",  markersize=12,   markeredgecolor="darkred",   linewidth=0, label="ATH (actual / predicted)"),
    plt.Line2D([0], [0], marker="v",         color="limegreen", markersize=9, markeredgecolor="darkgreen", linewidth=0, label="Cycle Low (actual / predicted)"),
]
ax.legend(handles=legend_handles, fontsize=8.5, loc="lower right", framealpha=0.95, ncol=2)

ax.set_title("Bitcoin Damped-Oscillator Model — Full Cycle History & Forecast (2012–2040)",
             fontsize=15, pad=16)
ax.set_xlabel("Year", fontsize=11)
ax.set_ylabel("Bitcoin Price (USD, log scale)", fontsize=11)
ax.grid(True, which="both", linestyle="--", alpha=0.28)

plt.tight_layout()
st.pyplot(fig, use_container_width=True)

live_str = f"${live_price:,.0f}" if live_price else "unavailable"
st.caption(
    f"Live price: {live_str}  ·  "
    f"Last data: {last_date.date()}  ·  "
    f"Cycle ≈ {2*np.pi/omega:.2f} years  ·  "
    f"Refreshes every 5 min"
)
