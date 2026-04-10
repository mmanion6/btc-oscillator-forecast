import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import yfinance as yf
from datetime import datetime, timedelta

st.set_page_config(page_title="BTC Damped Oscillator Forecast", layout="wide")
st.title("🚀 Bitcoin Damped-Oscillator Model + 10-Year Forecast")
st.caption("Live daily update • Powered by your original data + real-time price")

# ====================== DATA PULL ======================
@st.cache_data(ttl=86400)  # cache for 24 hours
def load_data():
    # Your original CSV
    df = pd.read_csv('BTC_All_graph_coinmarketcap.csv', sep=';')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Append today's latest price
    today = yf.download('BTC-USD', period='2d', interval='1d')['Close']
    latest_price = today.iloc[-1]
    latest_date = today.index[-1].date()
    
    # Only add if newer than last row
    if df['timestamp'].max().date() < latest_date:
        new_row = pd.DataFrame({
            'timestamp': [pd.Timestamp(latest_date)],
            'price': [latest_price]
        })
        df = pd.concat([df, new_row], ignore_index=True)
    
    return df

df = load_data()

# ====================== MODEL FIT ======================
genesis = pd.to_datetime('2009-01-01')
df['t_days'] = (df['timestamp'] - genesis).dt.total_seconds() / (24 * 3600)
df['t'] = df['t_days'] / 365.25
df['log_price'] = np.log10(df['price'])

def btc_damped_osc(t, a, b, A, gamma, omega, phi):
    trend = a + b * np.log(t)
    osc = A * np.exp(-gamma * t) * np.sin(omega * t + phi)
    return trend + osc

t_data = df['t'].values
y_data = df['log_price'].values

p0 = [-18.0, 1.15, 0.5, 0.05, 1.57, 0.0]
popt, _ = curve_fit(btc_damped_osc, t_data, y_data, p0=p0, maxfev=5000)
a, b, A, gamma, omega, phi = popt

df['pred_log'] = btc_damped_osc(t_data, *popt)
df['pred_price'] = 10 ** df['pred_log']

last_date = df['timestamp'].max()

# Future predictions
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=7), periods=520, freq='7D')
future_t = ((future_dates - genesis).total_seconds() / (24 * 3600)) / 365.25
future_pred_price = 10 ** btc_damped_osc(future_t, *popt)
future_df = pd.DataFrame({'timestamp': future_dates, 'pred_price': future_pred_price})

# ====================== HALVING CAGRs ======================
halving_list = [
    ('2012-11-28', '2012'), ('2016-07-09', '2016'), ('2020-05-11', '2020'),
    ('2024-04-19', '2024'), ('2028-04-20', '2028'), ('2032-04-20', '2032')
]

cagr_annotations = []
for i in range(len(halving_list)-1):
    start_str, label1 = halving_list[i]
    end_str, label2 = halving_list[i+1]
    t1 = (pd.to_datetime(start_str) - genesis).total_seconds() / (24*3600*365.25)
    t2 = (pd.to_datetime(end_str) - genesis).total_seconds() / (24*3600*365.25)
    
    logp1 = a + b * np.log(t1)
    logp2 = a + b * np.log(t2)
    price1 = 10 ** logp1
    price2 = 10 ** logp2
    years = t2 - t1
    cagr = (price2 / price1) ** (1/years) - 1
    
    mid_date = pd.to_datetime(start_str) + (pd.to_datetime(end_str) - pd.to_datetime(start_str))/2
    cagr_annotations.append((mid_date, f"{label1}–{label2}\n{cagr*100:.1f}% CAGR"))

# ====================== CHART ======================
fig, ax = plt.subplots(figsize=(16, 9))

ax.plot(df['timestamp'], df['price'], label='Actual BTC Price', color='red', linewidth=1.8)
ax.plot(df['timestamp'], df['pred_price'], label='Damped Oscillator Fit', color='blue', linestyle='--', linewidth=2)
ax.plot(future_df['timestamp'], future_df['pred_price'], label='Future Prediction', color='green', linewidth=3)

# Halvings
for d_str, _ in halving_list:
    dt = pd.to_datetime(d_str)
    if dt <= future_df['timestamp'].max():
        ax.axvline(dt, color='purple', linestyle='--', alpha=0.65, linewidth=1.5)

ax.axvline(last_date, color='gray', linestyle=':', linewidth=2.5, label='Present')

# Orange dots + labels
special_dates = [('2027-03-28', 'Mar 28, 2027'), ('2028-10-20', 'Oct 20, 2028')]
for date_str, short_label in special_dates:
    dt = pd.to_datetime(date_str)
    if dt > last_date:
        idx = np.argmin(np.abs(future_df['timestamp'] - dt))
        price = future_df['pred_price'].iloc[idx]
        ax.plot(dt, price, 'o', color='orange', markersize=11, markeredgecolor='darkred', markeredgewidth=2)
        ax.text(dt, price * 1.12, f"{short_label}\n${price:,.0f}", 
                ha='center', va='bottom', fontsize=9.5, color='darkorange', rotation=90,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85))

# New peaks (simplified)
peak_dates = pd.date_range(last_date + pd.Timedelta(days=30), '2036-12-31', freq='D')
peak_prices = 10 ** btc_damped_osc(((peak_dates - genesis).total_seconds() / (24*3600))/365.25, *popt)
peaks_idx, _ = find_peaks(peak_prices, prominence=8000, distance=180)
current_ath = df['price'].max()
for idx in peaks_idx:
    price = peak_prices[idx]
    if price > current_ath:
        dt = peak_dates[idx]
        idx_f = np.argmin(np.abs(future_df['timestamp'] - dt))
        p = future_df['pred_price'].iloc[idx_f]
        ax.plot(dt, p, 'o', color='orange', markersize=10, markeredgecolor='black')
        ax.text(dt, p * 1.15, f"{dt.date().strftime('%b %d, %Y')}\n${round(p):,}", 
                ha='center', va='bottom', fontsize=9, color='darkorange', rotation=90,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

ax.set_yscale('log')
def usd_formatter(x, pos):
    if x >= 1_000_000: return f'${x/1_000_000:.1f}M'
    elif x >= 10_000: return f'${x/1_000:.0f}K'
    return f'${x:.0f}'
ax.yaxis.set_major_formatter(FuncFormatter(usd_formatter))

ax.set_title('Bitcoin Damped-Oscillator Model + Long-Term Forecast', fontsize=16, pad=20)
ax.set_xlabel('Date')
ax.set_ylabel('Bitcoin Price (USD)')
ax.legend(fontsize=10.5, loc='lower right')

# Equation (top-left)
equation = r'$\log_{10}(P(t)) = a + b \cdot \ln(t) + A \cdot e^{-\gamma t} \cdot \sin(\omega t + \phi)$'
ax.text(0.02, 0.96, equation, transform=ax.transAxes, fontsize=14, va='top', ha='left',
        bbox=dict(boxstyle="round,pad=0.8", facecolor="white", alpha=0.95, edgecolor="#1f77b4"))

# CAGRs at bottom
for mid_date, text in cagr_annotations:
    ax.text(mid_date, 800, text, ha='center', va='bottom', fontsize=9.5, color='purple', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor='purple'))

ax.grid(True, which='both', linestyle='--', alpha=0.35)

st.pyplot(fig)

# Footer info
st.caption(f"Last data point: {last_date.date()} | Model refitted daily | Cycle period ≈ {2*np.pi/omega:.2f} years")
