import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import yfinance as yf
from datetime import datetime

# ====================== EMBED OPTIMIZATION ======================
st.set_page_config(page_title="BTC Oscillator Forecast", layout="wide", initial_sidebar_state="collapsed")

hide_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stApp {padding-top: 0.5rem;}
    </style>
"""
st.markdown(hide_style, unsafe_allow_html=True)

st.title("Bitcoin Damped-Oscillator Model — Regime Analysis & Forecast")
st.caption("Live daily update • 2012–2040 • Buy/Consolidation/ATH Zones")

# ====================== DATA LOADING & CLEANING ======================
@st.cache_data(ttl=86400)
def load_data():
    df = pd.read_csv('BTC_All_graph_coinmarketcap.csv', sep=';')
    
    # Clean price column
    df['price'] = df['price'].astype(str).str.replace(',', '').str.strip()
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df = df.dropna(subset=['price']).reset_index(drop=True)
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    
    # Append latest price
    try:
        today = yf.download('BTC-USD', period='2d', interval='1d', progress=False)['Close']
        latest_price = float(today.iloc[-1])
        latest_date = today.index[-1].date()
        
        if df['timestamp'].max().date() < latest_date:
            new_row = pd.DataFrame({'timestamp': [pd.Timestamp(latest_date)], 'price': [latest_price]})
            df = pd.concat([df, new_row], ignore_index=True)
    except:
        pass
    
    return df

df = load_data()

# ====================== MODEL FIT ======================
genesis = pd.to_datetime('2009-01-01')
df['t_days'] = (df['timestamp'] - genesis).dt.total_seconds() / (24 * 3600)
df['t'] = df['t_days'] / 365.25
df['log_price'] = np.log10(df['price'])

def btc_damped_osc(t, a, b, A, gamma, omega, phi):
    return a + b * np.log(t) + A * np.exp(-gamma * t) * np.sin(omega * t + phi)

def osc_component(t, A, gamma, omega, phi):
    return A * np.exp(-gamma * t) * np.sin(omega * t + phi)

t_data = df['t'].values
y_data = df['log_price'].values
p0 = [-18.0, 1.15, 0.5, 0.05, 1.57, 0.0]
popt, _ = curve_fit(btc_damped_osc, t_data, y_data, p0=p0, maxfev=5000)
a, b, A, gamma, omega, phi = popt

df['pred_price'] = 10 ** btc_damped_osc(df['t'], *popt)
last_date = df['timestamp'].max()

# ====================== UNIFIED SERIES 2012–2040 ======================
chart_start = pd.to_datetime('2012-01-01')
chart_end   = pd.to_datetime('2040-12-31')

all_dates = pd.date_range(chart_start, chart_end, freq='D')
all_t = ((all_dates - genesis).total_seconds() / (24*3600)) / 365.25

all_pred_price = 10 ** btc_damped_osc(all_t, *popt)
all_osc = osc_component(all_t, A, gamma, omega, phi)
all_trend_price = 10 ** (a + b * np.log(all_t))
all_envelope = A * np.exp(-gamma * all_t)

CONSOL_RATIO = 0.20
all_consol_thresh = CONSOL_RATIO * np.abs(all_envelope)

all_df = pd.DataFrame({
    'timestamp': all_dates,
    'pred_price': all_pred_price,
    'osc': all_osc,
    'trend_price': all_trend_price,
    'consol_thresh': all_consol_thresh,
    'is_hist': all_dates <= last_date
})

all_df['regime'] = np.where(all_df['osc'] < -all_df['consol_thresh'], 'buy',
                   np.where(all_df['osc'] > all_df['consol_thresh'], 'ath', 'consolidation'))

future_df = all_df[~all_df['is_hist']].copy().reset_index(drop=True)

# ====================== 3-MONTH REGIME TABLE ======================
st.subheader("3-Month Cycle Regime Outlook (2026–2040)")
analysis_start = pd.to_datetime('2026-04-01')
analysis_df = all_df[all_df['timestamp'] >= analysis_start].copy()

period_start = analysis_start
table_data = []

while period_start <= pd.to_datetime('2040-10-01'):
    period_end = period_start + pd.DateOffset(months=3) - pd.Timedelta(days=1)
    chunk = analysis_df[(analysis_df['timestamp'] >= period_start) & 
                        (analysis_df['timestamp'] <= period_end)]
    if len(chunk) == 0:
        period_start += pd.DateOffset(months=3)
        continue

    dominant = chunk['regime'].value_counts().idxmax()
    regime_label = {'buy': '✅ BUY ZONE', 'consolidation': '⏸ CONSOLIDATION', 'ath': '🔴 ATH / SELL'}[dominant]
    
    table_data.append({
        "Period": f"{period_start.strftime('%b %Y')} – {period_end.strftime('%b %Y')}",
        "Regime": regime_label,
        "Min Price": f"${chunk['pred_price'].min():,.0f}",
        "Max Price": f"${chunk['pred_price'].max():,.0f}",
        "Avg Price": f"${chunk['pred_price'].mean():,.0f}"
    })
    
    period_start += pd.DateOffset(months=3)

st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

# ====================== CHART ======================
fig, ax = plt.subplots(figsize=(20, 11))

# Background shading
shade_map = {'buy': ('limegreen', 0.09), 'consolidation': ('dodgerblue', 0.07), 'ath': ('orange', 0.08)}
for regime, (color, alpha) in shade_map.items():
    mask = all_df['regime'] == regime
    ax.fill_between(all_df['timestamp'], all_df['pred_price']*0.92, all_df['pred_price']*1.08,
                    where=mask, color=color, alpha=alpha, zorder=1)

# Actual price
ax.plot(df['timestamp'], df['price'], color='crimson', linewidth=1.8, label='Actual BTC Price', zorder=6)

# Regime-colored forecast
def plot_regime(df_in, regime, color, ls, lw, label):
    mask = df_in['regime'] == regime
    ax.plot(df_in['timestamp'][mask], df_in['pred_price'][mask],
            color=color, linestyle=ls, linewidth=lw, label=label, zorder=5)

plot_regime(all_df[all_df['is_hist']], 'buy', '#00aa00', '--', 2.2, 'Buy Zone (historical)')
plot_regime(all_df[all_df['is_hist']], 'consolidation', '#1e90ff', ':', 2.0, 'Consolidation (historical)')
plot_regime(all_df[all_df['is_hist']], 'ath', '#e07000', '-', 2.2, 'ATH/Sell (historical)')

plot_regime(future_df, 'buy', 'limegreen', '--', 3.5, 'Buy Zone (forecast)')
plot_regime(future_df, 'consolidation', 'dodgerblue', ':', 2.5, 'Consolidation (forecast)')
plot_regime(future_df, 'ath', 'darkorange', '-', 3.0, 'ATH/Sell (forecast)')

# Pure trend
ax.plot(all_dates, all_trend_price, color='navy', linestyle=':', linewidth=1.6, alpha=0.65, label='Pure Log Trend')

# Present day
ax.axvline(last_date, color='gray', linestyle=':', linewidth=2.8, label='Present')

# Halvings (add more if needed)
halvings = ['2012-11-28','2016-07-09','2020-05-11','2024-04-19','2028-04-20','2032-04-20']
for d in halvings:
    dt = pd.to_datetime(d)
    ax.axvline(dt, color='purple', linestyle='--', alpha=0.5, linewidth=1.2)

ax.set_yscale('log')
def usd_formatter(x, pos):
    if x >= 1_000_000: return f'${x/1_000_000:.1f}M'
    if x >= 10_000:    return f'${x/1_000:.0f}K'
    return f'${x:.0f}'
ax.yaxis.set_major_formatter(FuncFormatter(usd_formatter))

ax.set_xlim(chart_start, chart_end)
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

ax.set_title('Bitcoin Damped-Oscillator Model — Regime Analysis & Long-Term Forecast (2012–2040)', fontsize=16, pad=20)
ax.set_xlabel('Year')
ax.set_ylabel('Bitcoin Price (USD, log scale)')

ax.legend(loc='lower right', fontsize=9, ncol=2, framealpha=0.95)
ax.grid(True, which='both', linestyle='--', alpha=0.3)

# Equation
equation = r'$\log_{10}(P(t)) = a + b \cdot \ln(t) + A \cdot e^{-\gamma t} \cdot \sin(\omega t + \phi)$'
ax.text(0.02, 0.96, equation, transform=ax.transAxes, fontsize=11.5, va='top', ha='left',
        bbox=dict(boxstyle="round,pad=0.8", facecolor="white", alpha=0.95, edgecolor="#1f77b4"))

st.pyplot(fig, use_container_width=True)

st.caption(f"Last data: {last_date.date()} | Cycle length ≈ {2*np.pi/omega:.2f} years | Updated daily")
