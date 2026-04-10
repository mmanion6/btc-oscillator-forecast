import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import yfinance as yf

st.set_page_config(page_title="BTC Oscillator Forecast", layout="wide", initial_sidebar_state="collapsed")

hide = """<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}</style>"""
st.markdown(hide, unsafe_allow_html=True)

st.title("Bitcoin Damped-Oscillator Model — Regime Analysis & Long-Term Forecast (2012–2040)")
st.caption("Live daily update • Buy / Consolidation / ATH Zones")

# ====================== DATA ======================
@st.cache_data(ttl=86400)
def load_data():
    df = pd.read_csv('BTC_All_graph_coinmarketcap.csv', sep=';')
    df['price'] = pd.to_numeric(df['price'].astype(str).str.replace(',', '').str.strip(), errors='coerce')
    df = df.dropna(subset=['price']).reset_index(drop=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    try:
        today = yf.download('BTC-USD', period='2d', interval='1d', progress=False)['Close']
        latest_price = float(today.iloc[-1])
        latest_date = today.index[-1].date()
        if df['timestamp'].max().date() < latest_date:
            df = pd.concat([df, pd.DataFrame({'timestamp': [pd.Timestamp(latest_date)], 'price': [latest_price]})], ignore_index=True)
    except:
        pass
    return df

df = load_data()

# Model
genesis = pd.to_datetime('2009-01-01')
df['t'] = (df['timestamp'] - genesis).dt.total_seconds() / (24*3600*365.25)
df['log_price'] = np.log10(df['price'])

def btc_damped_osc(t, a, b, A, gamma, omega, phi):
    return a + b * np.log(t) + A * np.exp(-gamma * t) * np.sin(omega * t + phi)

popt, _ = curve_fit(btc_damped_osc, df['t'], df['log_price'], p0=[-18, 1.15, 0.5, 0.05, 1.57, 0], maxfev=5000)
a, b, A, gamma, omega, phi = popt

# Unified series
chart_start = pd.to_datetime('2012-01-01')
chart_end = pd.to_datetime('2040-12-31')
all_dates = pd.date_range(chart_start, chart_end, freq='D')
all_t = ((all_dates - genesis).total_seconds() / (24*3600)) / 365.25

all_pred_price = 10 ** btc_damped_osc(all_t, *popt)
all_osc = A * np.exp(-gamma * all_t) * np.sin(omega * all_t + phi)
all_trend = 10 ** (a + b * np.log(all_t))

CONSOL = 0.20
thresh = CONSOL * np.abs(A * np.exp(-gamma * all_t))

all_df = pd.DataFrame({
    'timestamp': all_dates,
    'pred_price': all_pred_price,
    'osc': all_osc,
    'trend_price': all_trend,
    'is_hist': all_dates <= df['timestamp'].max()
})

all_df['regime'] = np.where(all_df['osc'] < -thresh, 'buy',
                   np.where(all_df['osc'] > thresh, 'ath', 'consolidation'))

# ====================== CHART ======================
fig, ax = plt.subplots(figsize=(20, 11))

# Background shading
colors = {'buy': ('limegreen', 0.12), 'consolidation': ('dodgerblue', 0.08), 'ath': ('orange', 0.10)}
for regime, (col, alp) in colors.items():
    mask = all_df['regime'] == regime
    ax.fill_between(all_df['timestamp'], all_df['pred_price']*0.85, all_df['pred_price']*1.18,
                    where=mask, color=col, alpha=alp, zorder=1)

# Actual price
ax.plot(df['timestamp'], df['price'], color='crimson', linewidth=2.0, label='Actual BTC Price', zorder=6)

# Regime lines (historical + forecast)
def plot_regime(df_in, regime, color, style, width, label):
    mask = df_in['regime'] == regime
    ax.plot(df_in['timestamp'][mask], df_in['pred_price'][mask],
            color=color, linestyle=style, linewidth=width, label=label, zorder=5)

hist_df = all_df[all_df['is_hist']]
fut_df = all_df[~all_df['is_hist']]

plot_regime(hist_df, 'buy', '#00cc00', '--', 2.2, 'Buy Zone (historical)')
plot_regime(hist_df, 'consolidation', '#1e90ff', ':', 2.0, 'Consolidation (historical)')
plot_regime(hist_df, 'ath', '#e07000', '-', 2.2, 'ATH/Sell (historical)')

plot_regime(fut_df, 'buy', 'limegreen', '--', 3.5, 'Buy Zone (forecast)')
plot_regime(fut_df, 'consolidation', 'dodgerblue', ':', 2.5, 'Consolidation (forecast)')
plot_regime(fut_df, 'ath', 'darkorange', '-', 3.2, 'ATH/Sell (forecast)')

# Pure trend
ax.plot(all_dates, all_trend, color='navy', linestyle=':', linewidth=1.6, alpha=0.7, label='Pure Log Trend')

# Present
ax.axvline(df['timestamp'].max(), color='gray', linestyle=':', linewidth=2.8, label='Present')

# Halvings
for d in ['2012-11-28','2016-07-09','2020-05-11','2024-04-19','2028-04-20','2032-04-20']:
    ax.axvline(pd.to_datetime(d), color='purple', linestyle='--', alpha=0.5, linewidth=1.2)

# **CAGR blocks at bottom** (this was missing)
halving_list = [('2012-11-28','2012'), ('2016-07-09','2016'), ('2020-05-11','2020'),
                ('2024-04-19','2024'), ('2028-04-20','2028'), ('2032-04-20','2032')]
for i in range(len(halving_list)-1):
    s = pd.to_datetime(halving_list[i][0])
    e = pd.to_datetime(halving_list[i+1][0])
    t1 = (s - genesis).total_seconds() / (24*3600*365.25)
    t2 = (e - genesis).total_seconds() / (24*3600*365.25)
    p1 = 10 ** (a + b * np.log(t1))
    p2 = 10 ** (a + b * np.log(t2))
    cagr = (p2 / p1) ** (1 / (t2 - t1)) - 1
    mid = s + (e - s)/2
    ax.text(mid, 800, f"{halving_list[i][1]}–{halving_list[i+1][1]}\n{cagr*100:.1f}% CAGR",
            ha='center', va='bottom', fontsize=8.5, color='purple', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9))

ax.set_yscale('log')
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x/1e6:.1f}M' if x >= 1e6 else f'${x/1e3:.0f}K' if x >= 1e3 else f'${x:.0f}'))

ax.set_xlim(chart_start, chart_end)
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

ax.set_title('Bitcoin Damped-Oscillator Model — Regime Analysis & Long-Term Forecast (2012–2040)', fontsize=15)
ax.set_xlabel('Year')
ax.set_ylabel('Bitcoin Price (USD, log scale)')

ax.legend(loc='lower right', fontsize=9, ncol=2, framealpha=0.95)
ax.grid(True, which='both', linestyle='--', alpha=0.25)

# Equation
eq = r'$\log_{10}(P(t)) = a + b \cdot \ln(t) + A \cdot e^{-\gamma t} \cdot \sin(\omega t + \phi)$'
ax.text(0.02, 0.96, eq, transform=ax.transAxes, fontsize=11, va='top', ha='left',
        bbox=dict(boxstyle="round,pad=0.8", facecolor="white", alpha=0.95))

st.pyplot(fig, use_container_width=True)

st.caption(f"Last data: {df['timestamp'].max().date()} | Cycle
