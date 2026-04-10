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
st.caption("Live daily update • Buy / Consolidation / ATH Zones")

# Data
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

genesis = pd.to_datetime('2009-01-01')
df['t'] = (df['timestamp'] - genesis).dt.total_seconds() / (24 * 3600 * 365.25)
df['log_price'] = np.log10(df['price'])

def btc_damped_osc(t, a, b, A, gamma, omega, phi):
    return a + b * np.log(t) + A * np.exp(-gamma * t) * np.sin(omega * t + phi)

popt, _ = curve_fit(btc_damped_osc, df['t'], df['log_price'], p0=[-18, 1.15, 0.5, 0.05, 1.57, 0], maxfev=5000)
a, b, A, gamma, omega, phi = popt

# Unified timeline 2012-2040
chart_start = pd.to_datetime('2012-01-01')
chart_end = pd.to_datetime('2040-12-31')
all_dates = pd.date_range(chart_start, chart_end, freq='D')
all_t = ((all_dates - genesis).total_seconds() / (24*3600)) / 365.25

all_pred_price = 10 ** btc_damped_osc(all_t, *popt)
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

# Chart
fig, ax = plt.subplots(figsize=(22, 12))

# Background shading
regime_colors = {'buy': '#d4f4d4', 'consolidation': '#e0f0ff', 'ath': '#ffe6cc'}
for regime, color in regime_colors.items():
    mask = all_df['regime'] == regime
    ax.fill_between(all_df['timestamp'], 5, all_df['pred_price']*1.5, where=mask, color=color, alpha=0.6, zorder=1)

# Actual price
ax.plot(df['timestamp'], df['price'], color='crimson', linewidth=2.2, label='Actual BTC Price', zorder=6)

# Regime lines
def plot_regime(df_in, regime, color, ls, lw, label):
    mask = df_in['regime'] == regime
    ax.plot(df_in['timestamp'][mask], df_in['pred_price'][mask], color=color, linestyle=ls, linewidth=lw, label=label, zorder=5)

hist = all_df[all_df['is_hist']]
fut = all_df[~all_df['is_hist']]

plot_regime(hist, 'buy', '#00aa00', '--', 2.2, 'Buy Zone — historical')
plot_regime(hist, 'consolidation', '#1e90ff', ':', 2.0, 'Consolidation — historical')
plot_regime(hist, 'ath', '#e07000', '-', 2.2, 'ATH/Sell — historical')

plot_regime(fut, 'buy', 'limegreen', '--', 3.5, 'Buy Zone — forecast')
plot_regime(fut, 'consolidation', 'dodgerblue', ':', 2.5, 'Consolidation — forecast')
plot_regime(fut, 'ath', 'darkorange', '-', 3.2, 'ATH/Sell — forecast')

ax.plot(all_dates, all_trend, color='navy', linestyle=':', linewidth=1.8, alpha=0.7, label='Pure Log Trend')
