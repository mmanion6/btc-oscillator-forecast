import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import yfinance as yf

st.set_page_config(page_title="BTC Forecast", layout="wide", initial_sidebar_state="collapsed")

hide = """<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}</style>"""
st.markdown(hide, unsafe_allow_html=True)

st.title("Bitcoin Damped-Oscillator Model — Regime Analysis & Long-Term Forecast (2012–2040)")
st.caption("Live daily update • Buy / Consolidation / ATH Zones")

# Load and clean data
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

# Unified timeline
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

future_df = all_df[~all_df['is_hist']].copy()

# ====================== CHART ======================
fig, ax = plt.subplots(figsize=(20, 11))

# Background shading
for regime, (color, alpha) in {'buy': ('limegreen', 0.12), 'consolidation': ('dodgerblue', 0.08
