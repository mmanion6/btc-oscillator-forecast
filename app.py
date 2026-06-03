"""
BTC-USD Damped Oscillator Financial Forecaster
Power-law + damped harmonic oscillator model for Bitcoin price prediction
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
import yfinance as yf
from scipy.optimize import curve_fit
from datetime import datetime, timedelta
import io

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="BTC Damped Oscillator Forecaster",
    page_icon="₿",
    layout="wide",
)

st.title("₿ Bitcoin Damped Oscillator Financial Forecaster")
st.markdown(
    "Power-law + damped harmonic oscillator model fit to historical BTC-USD daily closes, "
    "with regime-colored forecasts through 2040."
)

# ─────────────────────────────────────────────
# Sidebar controls
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("Model Settings")
    forecast_end = st.slider("Forecast end year", 2027, 2040, 2040)
    damping_fix = st.checkbox("Fix damping γ (prevent over-damping)", value=True)
    show_ci = st.checkbox("Show ±1σ confidence band", value=True)
    theme = st.selectbox("Chart theme", ["dark", "light"], index=0)
    st.markdown("---")
    st.markdown("**Model:**")
    st.latex(r"\log_{10}(P) = a + b\ln(t) + A e^{-\gamma t}\sin(\omega t + \phi)")

# ─────────────────────────────────────────────
# Data download
# ─────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_btc():
    df = yf.download("BTC-USD", start="2010-07-18", auto_adjust=True, progress=False)
    df = df[["Close"]].dropna()
    df.index = pd.to_datetime(df.index)
    df = df[df["Close"] > 0]
    return df

with st.spinner("Downloading BTC-USD data…"):
    df = load_btc()

st.success(f"Loaded {len(df):,} daily closes  ·  {df.index[0].date()} → {df.index[-1].date()}")

# ─────────────────────────────────────────────
# Model definition
# ─────────────────────────────────────────────
GENESIS = pd.Timestamp("2009-01-03")  # Bitcoin genesis block

def days_since_genesis(ts):
    if isinstance(ts, pd.DatetimeIndex):
        return (ts - GENESIS).days.astype(float)
    return float((ts - GENESIS).days)

def damped_osc(t, a, b, A, gamma, omega, phi):
    """log10(Price) = a + b*ln(t) + A*exp(-gamma*t)*sin(omega*t + phi)"""
    return a + b * np.log(t) + A * np.exp(-gamma * t) * np.sin(omega * t + phi)

# ─────────────────────────────────────────────
# Fit the model
# ─────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fit_model(prices_json: str, fix_gamma: bool):
    df_fit = pd.read_json(io.StringIO(prices_json))
    df_fit.index = pd.to_datetime(df_fit.index)

    t = days_since_genesis(df_fit.index)
    y = np.log10(df_fit["Close"].values.ravel())

    # Initial guesses
    p0 = [
        -17.0,   # a  intercept
         2.5,    # b  log-slope
         0.5,    # A  amplitude
         1e-4,   # gamma damping (very slow)
         0.0017, # omega ~1 cycle / 3650 days ≈ 3.7 yr
        -1.5,    # phi  phase
    ]
    # Bounds
    lo = [-30, 0.5,  0.01, 1e-6, 5e-4, -2*np.pi]
    hi = [ -5, 5.0,  3.0,  5e-3, 5e-3,  2*np.pi]

    if fix_gamma:
        # Fix gamma ~ 0 (very slow damping) by tightening bounds
        lo[3] = 5e-6
        hi[3] = 3e-4

    try:
        popt, pcov = curve_fit(
            damped_osc, t, y,
            p0=p0, bounds=(lo, hi),
            maxfev=50000,
            method="trf",
        )
        perr = np.sqrt(np.diag(pcov))
    except Exception as e:
        st.error(f"Curve fit failed: {e}")
        st.stop()

    return popt, perr

prices_json = df[["Close"]].to_json()
with st.spinner("Fitting model…"):
    popt, perr = fit_model(prices_json, damping_fix)

a, b, A, gamma, omega, phi = popt

# ─────────────────────────────────────────────
# Regime classification helpers
# ─────────────────────────────────────────────
def oscillator_value(t_arr, A, gamma, omega, phi):
    return A * np.exp(-gamma * t_arr) * np.sin(omega * t_arr + phi)

def classify_regime(osc_val, A_ref):
    """Return regime label based on oscillator deviation."""
    norm = osc_val / max(abs(A_ref), 1e-9)
    if norm > 0.45:
        return "ATH/Sell Zone"
    elif norm > 0.10:
        return "Consolidation"
    elif norm < -0.35:
        return "Buy Zone"
    else:
        return "Consolidation"

REGIME_COLORS = {
    "Buy Zone":      "#1a6b2e",  # dark green
    "Consolidation": "#2e4a7a",  # dark blue
    "ATH/Sell Zone": "#8b1a1a",  # dark red
}
REGIME_LINE_COLORS = {
    "Buy Zone":      "#00e676",
    "Consolidation": "#64b5f6",
    "ATH/Sell Zone": "#ff5252",
}
REGIME_BG_ALPHA = 0.12

# ─────────────────────────────────────────────
# Generate time arrays
# ─────────────────────────────────────────────
today = pd.Timestamp.today().normalize()
t_hist = days_since_genesis(df.index)
y_hist = np.log10(df["Close"].values.ravel())
y_fit  = damped_osc(t_hist, *popt)

forecast_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1),
                               end=f"{forecast_end}-12-31", freq="D")
t_fore = days_since_genesis(forecast_dates)
y_fore = damped_osc(t_fore, *popt)

# Full timeline for background shading
all_dates = df.index.append(forecast_dates)
t_all    = days_since_genesis(all_dates)
osc_all  = oscillator_value(t_all, A, gamma, omega, phi)

# ─────────────────────────────────────────────
# Confidence band (±1σ from parameter uncertainty)
# ─────────────────────────────────────────────
def model_band(t_arr, popt, perr, nsigma=1):
    n_samples = 500
    samples = np.random.default_rng(42).normal(
        loc=popt, scale=perr, size=(n_samples, len(popt))
    )
    curves = np.array([damped_osc(t_arr, *s) for s in samples])
    lo = np.percentile(curves, 100*(0.5 - 0.3413*nsigma*2), axis=0)
    hi = np.percentile(curves, 100*(0.5 + 0.3413*nsigma*2), axis=0)
    return lo, hi

if show_ci:
    ci_lo, ci_hi = model_band(t_fore, popt, perr)

# ─────────────────────────────────────────────
# Bitcoin halving dates (milestone markers)
# ─────────────────────────────────────────────
HALVINGS = {
    "Genesis":      pd.Timestamp("2009-01-03"),
    "Halving 1":    pd.Timestamp("2012-11-28"),
    "Halving 2":    pd.Timestamp("2016-07-09"),
    "Halving 3":    pd.Timestamp("2020-05-11"),
    "Halving 4":    pd.Timestamp("2024-04-20"),
    "Halving 5\n(est)": pd.Timestamp("2028-03-01"),
}

# ─────────────────────────────────────────────
# Detect cycle highs / lows in fit curve
# ─────────────────────────────────────────────
def find_extrema(dates, y_vals, order=180):
    from scipy.signal import argrelextrema
    highs = argrelextrema(y_vals, np.greater, order=order)[0]
    lows  = argrelextrema(y_vals, np.less,    order=order)[0]
    return (
        [(dates[i], 10**y_vals[i]) for i in highs],
        [(dates[i], 10**y_vals[i]) for i in lows],
    )

all_y_fit = np.concatenate([y_fit, y_fore])
cycle_highs, cycle_lows = find_extrema(all_dates, all_y_fit)

# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────
plt_style = "dark_background" if theme == "dark" else "seaborn-v0_8-whitegrid"
plt.style.use(plt_style)

BG  = "#0d0d0d" if theme == "dark" else "#f8f8f8"
FG  = "#e0e0e0" if theme == "dark" else "#1a1a1a"
GRID = "#2a2a2a" if theme == "dark" else "#cccccc"

fig, ax = plt.subplots(figsize=(18, 9), facecolor=BG)
ax.set_facecolor(BG)

# ── Regime background shading ──────────────────────────────────────────────
prev_regime = None
seg_start   = all_dates[0]

def shade_segment(ax, x0, x1, regime):
    c = REGIME_COLORS.get(regime, "#333333")
    ax.axvspan(x0, x1, color=c, alpha=REGIME_BG_ALPHA, linewidth=0)

for i, (dt, osc_v) in enumerate(zip(all_dates, osc_all)):
    regime = classify_regime(osc_v, A)
    if regime != prev_regime:
        if prev_regime is not None:
            shade_segment(ax, seg_start, dt, prev_regime)
        seg_start  = dt
        prev_regime = regime
shade_segment(ax, seg_start, all_dates[-1], prev_regime)

# ── Historical prices (scatter, thin) ─────────────────────────────────────
ax.semilogy(df.index, df["Close"].values.ravel(),
            color="#888888", linewidth=0.4, alpha=0.6, zorder=2, label="BTC Close (daily)")
