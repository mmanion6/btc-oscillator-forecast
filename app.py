import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import yfinance as yf
from datetime import datetime

# Robust scipy import with fallback message
try:
    from scipy.optimize import curve_fit
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    st.error("⚠️ scipy is not installed. Please check your requirements.txt and redeploy.")
    st.stop()

st.set_page_config(page_title="TSLA Oscillator Forecast", layout="wide", initial_sidebar_state="collapsed")

hide = """<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}</style>"""
st.markdown(hide, unsafe_allow_html=True)

st.title("Tesla Damped-Oscillator Model — Full Cycle History & Forecast (2012–2040)")
st.caption("Live daily update • Buy / Consolidation / ATH Zones")

# ====================== DATA ======================
@st.cache_data(ttl=86400)
def load_data():
    df = yf.download('TSLA', period='max', interval='1d', progress=False)
    df = df[['Close']].reset_index()
    df.columns = ['timestamp', 'price']
    df = df.dropna(subset=['price']).reset_index(drop=True)
    return df

df = load_data()

genesis = pd.to_datetime('2010-06-29')  # Tesla IPO date
df['t'] = (df['timestamp'] - genesis).dt.total_seconds() / (24 * 3600 * 365.25)
df['log_price'] = np.log10(df['price'])

# ====================== MODEL ======================
def damped_osc(t, a, b, A, gamma, omega, phi):
    return a + b * np.log(t) + A * np.exp(-gamma * t) * np.sin(omega * t + phi)

# More robust fitting with bounds
p0 = [-2.0, 2.5, 0.3, 0.01, 1.57, 0.0]
bounds = ([-10, 0.1, -2, 0.0001, 0.1, -np.pi], [5, 10, 2, 0.5, 10, np.pi])

popt, _ = curve_fit(damped_osc, df['t'], df['log_price'], p0=p0, bounds=bounds, maxfev=10000)
a, b, A, gamma, omega, phi = popt

# ====================== UNIFIED TIMELINE ======================
chart_start = pd.to_datetime('2012-01-01')
chart_end = pd.to_datetime('2040-12-31')
all_dates = pd.date_range(chart_start, chart_end, freq='D')
all_t = ((all_dates - genesis).total_seconds() / (24*3600)) / 365.25

all_pred_price = 10 ** damped_osc(all_t, *popt)
all_osc = A * np.exp(-gamma * all_t) * np.sin(omega * all_t + phi)
all_trend = 10 ** (a + b * np.log(all_t))

thresh = 0.20 * np.abs(A * np.exp(-gamma * all_t))

all_df = pd.DataFrame({
    'timestamp': all_dates,
    'pred_price': all_pred_price,
    'osc': all_osc,
    'trend_price': all_trend,
    'is_hist': all_dates <= df['timestamp'].max()
})

all_df['regime'] = np.where(all_df['osc'] < -thresh, 'buy',
                   np.where(all_df['osc'] > thresh, 'ath', 'consolidation'))

future_df = all_df[~all_df['is_hist']].copy()

# ====================== CHART ======================
fig, ax = plt.subplots(figsize=(22, 12))

# Background regime shading
regime_colors = {'buy': '#d4f4d4', 'consolidation': '#e0f0ff', 'ath': '#ffe6cc'}
for regime, color in regime_colors.items():
    mask = all_df['regime'] == regime
    ax.fill_between(all_df['timestamp'], 3, all_df['pred_price']*2, 
                    where=mask, color=color, alpha=0.75, zorder=1)

# Actual price
ax.plot(df['timestamp'], df['price'], color='crimson', linewidth=2.2, label='Actual TSLA Price', zorder=6)

# Regime lines
def plot_regime(df_in, regime, color, ls, lw, label):
    mask = df_in['regime'] == regime
    ax.plot(df_in['timestamp'][mask], df_in['pred_price'][mask], 
            color=color, linestyle=ls, linewidth=lw, label=label, zorder=5)

hist = all_df[all_df['is_hist']]
fut = all_df[~all_df['is_hist']]

plot_regime(hist, 'buy', '#00aa00', '--', 2.2, 'Buy Zone — historical')
plot_regime(hist, 'consolidation', '#1e90ff', ':', 2.0, 'Consolidation — historical')
plot_regime(hist, 'ath', '#e07000', '-', 2.2, 'ATH/Sell — historical')

plot_regime(fut, 'buy', 'limegreen', '--', 3.5, 'Buy Zone — forecast')
plot_regime(fut, 'consolidation', 'dodgerblue', ':', 2.5, 'Consolidation — forecast')
plot_regime(fut, 'ath', 'darkorange', '-', 3.2, 'ATH/Sell — forecast')

ax.plot(all_dates, all_trend, color='navy', linestyle=':', linewidth=1.8, alpha=0.7, label='Pure Log Trend')
ax.axvline(df['timestamp'].max(), color='gray', linestyle=':', linewidth=3, label='Present')

# CAGR blocks at bottom
cagr_milestones = [('2012','2016'), ('2016','2020'), ('2020','2024'), ('2024','2028'), ('2028','2032')]
for i in range(len(cagr_milestones)-1):
    s = pd.to_datetime(cagr_milestones[i][0] + '-01-01')
    e = pd.to_datetime(cagr_milestones[i+1][0] + '-01-01')
    t1 = (s - genesis).total_seconds() / (24*3600*365.25)
    t2 = (e - genesis).total_seconds() / (24*3600*365.25)
    p1 = 10 ** (a + b * np.log(t1))
    p2 = 10 ** (a + b * np.log(t2))
    cagr = (p2 / p1) ** (1 / (t2 - t1)) - 1
    mid = s + (e - s)/2
    ax.text(mid, 450, f"{cagr_milestones[i][0]}–{cagr_milestones[i+1][0]}\n{cagr*100:.1f}% CAGR",
            ha='center', va='bottom', fontsize=9, color='purple', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.92))

# Equation box
eq = f"log₁₀(P(t)) = a + b · ln(t) + A · e^(-γt) · sin(ωt + φ)\na={a:.3f}, b={b:.3f}, A={A:.3f}, γ={gamma:.4f}, ω={omega:.4f}, φ={phi:.4f}"
ax.text(0.02, 0.96, eq, transform=ax.transAxes, fontsize=10.5, va='top', ha='left',
        bbox=dict(boxstyle="round,pad=0.8", facecolor="white", alpha=0.95))

ax.set_yscale('log')
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x/1e6:.1f}M' if x>=1e6 else f'${x/1e3:.0f}K' if x>=1e3 else f'${x:.0f}'))

ax.set_xlim(chart_start, chart_end)
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

ax.set_title('Tesla Damped-Oscillator Model — Full Cycle History & Forecast (2012–2040)', fontsize=15)
ax.set_xlabel('Year')
ax.set_ylabel('Tesla Price (USD, log scale)')

ax.legend(loc='lower right', fontsize=9, ncol=2, framealpha=0.95)
ax.grid(True, which='both', linestyle='--', alpha=0.25)

st.pyplot(fig, use_container_width=True)

st.caption(f"Last data: {df['timestamp'].max().date()} | Cycle ≈ {2*np.pi/omega:.2f} years | Updated daily")
