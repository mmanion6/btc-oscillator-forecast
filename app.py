import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import plotly.graph_objects as go
import yfinance as yf

st.set_page_config(page_title="BTC Forecast", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
    "<style>#MainMenu{visibility:hidden;}footer{visibility:hidden;}header{visibility:hidden;}</style>",
    unsafe_allow_html=True,
)
st.title("Bitcoin Damped-Oscillator Model — Full Cycle History & Forecast (2012–2040)")
st.caption("Live data · hover or drag to explore · scroll to zoom")

# ── Real-time price (5-min cache) ─────────────────────────────────────────────
@st.cache_data(ttl=300)
def get_live_price():
    try:
        tick = yf.download("BTC-USD", period="1d", interval="1m", progress=False)
        if not tick.empty:
            price = float(tick["Close"].iloc[-1])
            ts = tick.index[-1].tz_localize(None) if tick.index[-1].tzinfo else tick.index[-1]
            return price, pd.Timestamp(ts)
    except Exception:
        pass
    try:
        tick = yf.download("BTC-USD", period="2d", interval="1d", progress=False)
        if not tick.empty:
            price = float(tick["Close"].iloc[-1])
            return price, pd.Timestamp(tick.index[-1].date())
    except Exception:
        pass
    return None, None


# ── Historical CSV (1-hour cache) ─────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_data():
    df = pd.read_csv("BTC_All_graph_coinmarketcap.csv", sep=";")
    df["price"] = pd.to_numeric(
        df["price"].astype(str).str.replace(",", "").str.strip(), errors="coerce"
    )
    df = df.dropna(subset=["price"]).reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


df = load_data()
live_price, live_ts = get_live_price()
if live_price and live_ts and live_ts > df["timestamp"].max():
    df = pd.concat(
        [df, pd.DataFrame({"timestamp": [live_ts], "price": [live_price]})],
        ignore_index=True,
    )

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
        (all_df[all_df["is_hist"]],  2, label_hist[regime]),
        (all_df[~all_df["is_hist"]], 3, label_fut[regime]),
    ]:
        seg = subset[subset["regime"] == regime]
        if seg.empty:
            continue
        hover = [
            f"<b>{lbl}</b><br>Date: {d.strftime('%Y-%m-%d')}<br>Model: {plabel(p)}"
            for d, p in zip(seg["timestamp"], seg["pred_price"])
        ]
        fig.add_trace(go.Scatter(
            x=seg["timestamp"], y=seg["pred_price"],
            mode="lines",
            line=dict(color=color, width=lw, dash=dash),
            name=lbl,
            legendgroup=lbl,
            showlegend=True,
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hover,
        ))

# 3. Actual BTC price
hist_plot = df[df["timestamp"] >= chart_start]
hover_actual = [
    f"<b>Actual BTC</b><br>Date: {d.strftime('%Y-%m-%d')}<br>Price: {plabel(p)}"
    for d, p in zip(hist_plot["timestamp"], hist_plot["price"])
]
fig.add_trace(go.Scatter(
    x=hist_plot["timestamp"], y=hist_plot["price"],
    mode="lines",
    line=dict(color="crimson", width=2),
    name="Actual BTC Price",
    hovertemplate="%{customdata}<extra></extra>",
    customdata=hover_actual,
))

# 4. Pure log trend
fig.add_trace(go.Scatter(
    x=all_dates, y=all_trend,
    mode="lines",
    line=dict(color="navy", width=1.5, dash="dot"),
    opacity=0.6,
    name="Pure Log Trend",
    hovertemplate="Trend: %{y:$,.0f}<br>%{x|%Y-%m-%d}<extra></extra>",
))

# 5. Halving vertical lines
for d_str, lbl in halving_list:
    dt = pd.to_datetime(d_str)
    if chart_start <= dt <= chart_end:
        fig.add_vline(x=dt, line=dict(color="purple", dash="dash", width=1.4),
                      opacity=0.6,
                      annotation_text=lbl, annotation_position="top",
                      annotation_font=dict(color="purple", size=9))

# 6. Present line
fig.add_vline(x=last_date, line=dict(color="gray", dash="dot", width=2.5),
              annotation_text="Present", annotation_position="top right",
              annotation_font=dict(color="gray", size=10))

# 7. CAGR annotations
for i in range(len(halving_list) - 1):
    s  = pd.to_datetime(halving_list[i][0])
    e  = pd.to_datetime(halving_list[i + 1][0])
    t1 = (s - genesis).total_seconds() / (24 * 3600 * 365.25)
    t2 = (e - genesis).total_seconds() / (24 * 3600 * 365.25)
    p1 = 10 ** (a + b * np.log(t1))
    p2 = 10 ** (a + b * np.log(t2))
    cagr = (p2 / p1) ** (1 / (t2 - t1)) - 1
    mid  = s + (e - s) / 2
    cycle_name = f"{halving_list[i][0][:4]}–{halving_list[i+1][0][:4]}"
    fig.add_annotation(
        x=mid, y=np.log10(500),
        text=f"<b>{cycle_name}</b><br>{cagr*100:.1f}% CAGR",
        showarrow=False, font=dict(color="purple", size=9),
        bgcolor="white", bordercolor="purple", borderwidth=1, borderpad=4,
        yref="y",
    )

# 8. Historical ATH markers
fig.add_trace(go.Scatter(
    x=hist_aths["timestamp"], y=hist_aths["price"],
    mode="markers+text",
    marker=dict(symbol="star", size=16, color="gold", line=dict(color="darkred", width=1.5)),
    text=[plabel(p) for p in hist_aths["price"]],
    textposition="top center",
    textfont=dict(color="darkred", size=9),
    name="Historical ATH",
    hovertemplate="<b>ATH</b><br>Date: %{x|%Y-%m-%d}<br>Price: %{y:$,.0f}<extra></extra>",
))

# 9. Historical Cycle Low markers
fig.add_trace(go.Scatter(
    x=hist_lows["timestamp"], y=hist_lows["price"],
    mode="markers+text",
    marker=dict(symbol="triangle-down", size=12, color="limegreen",
                line=dict(color="darkgreen", width=1.4)),
    text=[plabel(p) for p in hist_lows["price"]],
    textposition="bottom center",
    textfont=dict(color="darkgreen", size=9),
    name="Historical Cycle Low",
    hovertemplate="<b>Cycle Low</b><br>Date: %{x|%Y-%m-%d}<br>Price: %{y:$,.0f}<extra></extra>",
))

# 10. Future ATH markers
if not fut_aths.empty:
    fig.add_trace(go.Scatter(
        x=fut_aths["timestamp"], y=fut_aths["pred_price"],
        mode="markers+text",
        marker=dict(symbol="star", size=18, color="gold", line=dict(color="darkred", width=1.5)),
        text=[plabel(p) for p in fut_aths["pred_price"]],
        textposition="top center",
        textfont=dict(color="darkred", size=9),
        name="Predicted ATH",
        hovertemplate="<b>Predicted ATH</b><br>Date: %{x|%Y-%m-%d}<br>Model: %{y:$,.0f}<extra></extra>",
    ))

# 11. Future Cycle Low markers
if not fut_lows.empty:
    fig.add_trace(go.Scatter(
        x=fut_lows["timestamp"], y=fut_lows["pred_price"],
        mode="markers+text",
        marker=dict(symbol="triangle-down", size=14, color="limegreen",
                    line=dict(color="darkgreen", width=1.4)),
        text=[plabel(p) for p in fut_lows["pred_price"]],
        textposition="bottom center",
        textfont=dict(color="darkgreen", size=9),
        name="Predicted Cycle Low",
        hovertemplate="<b>Predicted Low</b><br>Date: %{x|%Y-%m-%d}<br>Model: %{y:$,.0f}<extra></extra>",
    ))

# 12. Live price marker
if live_price:
    fig.add_trace(go.Scatter(
        x=[live_ts], y=[live_price],
        mode="markers",
        marker=dict(symbol="circle", size=12, color="crimson", line=dict(color="white", width=2)),
        name=f"Live: {plabel(live_price)}",
        hovertemplate=f"<b>Live BTC</b><br>{plabel(live_price)}<extra></extra>",
    ))

# ── Layout ────────────────────────────────────────────────────────────────────
eq_str = (
    f"log₁₀(P) = {a:.3f} + {b:.3f}·ln(t) + {A:.3f}·e^(−{gamma:.4f}t)·sin({omega:.4f}t + {phi:.4f})"
)
fig.update_layout(
    title=dict(
        text="Bitcoin Damped-Oscillator Model — Full Cycle History & Forecast (2012–2040)",
        font=dict(size=17),
    ),
    xaxis=dict(
        range=[chart_start, chart_end],
        title="Year",
        showgrid=True, gridcolor="rgba(0,0,0,0.08)",
        rangeslider=dict(visible=True, thickness=0.04),
    ),
    yaxis=dict(
        type="log",
        range=[np.log10(100), np.log10(100_000_000)],
        title="Bitcoin Price (USD, log scale)",
        showgrid=True, gridcolor="rgba(0,0,0,0.08)",
        tickformat="$,.0f",
        tickvals=[100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000],
        ticktext=["$100", "$1K", "$10K", "$100K", "$1M", "$10M", "$100M"],
    ),
    hovermode="x unified",
    legend=dict(
        orientation="v", x=1.01, y=1,
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#ccc", borderwidth=1,
        font=dict(size=10),
    ),
    height=750,
    margin=dict(l=70, r=180, t=70, b=80),
    annotations=[
        dict(
            text=eq_str,
            xref="paper", yref="paper",
            x=0.01, y=0.99,
            showarrow=False,
            font=dict(size=10, family="monospace"),
            bgcolor="white", bordercolor="#1f77b4", borderwidth=1, borderpad=6,
            align="left",
        )
    ],
    plot_bgcolor="white",
    paper_bgcolor="white",
)

st.plotly_chart(fig, use_container_width=True)

live_str = plabel(live_price) if live_price else "unavailable"
st.caption(
    f"Live price: {live_str}  ·  Last data: {last_date.date()}  ·  "
    f"Cycle ≈ {2*np.pi/omega:.2f} yr  ·  Refreshes every 5 min  ·  "
    "Drag to zoom · scroll to zoom price · double-click to reset"
)
