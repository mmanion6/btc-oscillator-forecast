import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates

# ==================== LOAD DATA & FIT MODEL ====================
df = pd.read_csv('BTC_All_graph_coinmarketcap.csv', sep=';')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)
genesis = pd.to_datetime('2009-01-01')
df['t_days'] = (df['timestamp'] - genesis).dt.total_seconds() / (24 * 3600)
df['t'] = df['t_days'] / 365.25
df['log_price'] = np.log10(df['price'])

def btc_damped_osc(t, a, b, A, gamma, omega, phi):
    trend = a + b * np.log(t)
    osc   = A * np.exp(-gamma * t) * np.sin(omega * t + phi)
    return trend + osc

def osc_component(t, A, gamma, omega, phi):
    return A * np.exp(-gamma * t) * np.sin(omega * t + phi)

t_data = df['t'].values
y_data = df['log_price'].values
p0 = [-18.0, 1.15, 0.5, 0.05, 1.57, 0.0]
popt, pcov = curve_fit(btc_damped_osc, t_data, y_data, p0=p0, maxfev=5000)
a, b, A, gamma, omega, phi = popt

df['pred_log']   = btc_damped_osc(t_data, *popt)
df['pred_price'] = 10 ** df['pred_log']
last_date = df['timestamp'].max()

# ==================== BUILD UNIFIED DAILY TIME SERIES: 2012-01-01 → 2040-12-31 ====================
chart_start = pd.to_datetime('2012-01-01')
chart_end   = pd.to_datetime('2040-12-31')

all_dates = pd.date_range(start=chart_start, end=chart_end, freq='D')
all_t     = ((all_dates - genesis).total_seconds() / (24 * 3600)) / 365.25
all_pred_log   = btc_damped_osc(all_t, *popt)
all_pred_price = 10 ** all_pred_log
all_osc        = osc_component(all_t, A, gamma, omega, phi)
all_trend_log  = a + b * np.log(all_t)
all_trend_price= 10 ** all_trend_log
all_envelope   = A * np.exp(-gamma * all_t)
CONSOL_RATIO   = 0.20
all_consol_thresh = CONSOL_RATIO * np.abs(all_envelope)

all_df = pd.DataFrame({
    'timestamp':   all_dates,
    'pred_price':  all_pred_price,
    'pred_log':    all_pred_log,
    'osc':         all_osc,
    'trend_price': all_trend_price,
    'envelope':    all_envelope,
    'consol_thresh': all_consol_thresh,
    'is_hist':     all_dates <= last_date
})
all_df['regime'] = np.where(
    all_df['osc'] < -all_df['consol_thresh'], 'buy',
    np.where(all_df['osc'] > all_df['consol_thresh'], 'ath', 'consolidation')
)
all_df = all_df.reset_index(drop=True)

# ==================== FUTURE PREDICTIONS ====================
future_df = all_df[all_df['timestamp'] > last_date].copy().reset_index(drop=True)

# ==================== HISTORICAL ATHs & CYCLE LOWS (actual price data) ====================
hist_chart = df[df['timestamp'] >= chart_start].copy().reset_index(drop=True)

# Real ATHs on actual price
hist_peaks_idx, _ = find_peaks(
    hist_chart['price'].values,
    prominence=hist_chart['price'].values.max() * 0.10,
    distance=180
)
hist_aths = hist_chart.iloc[hist_peaks_idx][['timestamp', 'price']].copy()

# Real cycle lows on actual price (troughs)
hist_trough_idx, _ = find_peaks(
    -hist_chart['price'].values,
    prominence=hist_chart['price'].values.max() * 0.02,
    distance=180
)
hist_lows = hist_chart.iloc[hist_trough_idx][['timestamp', 'price']].copy()
# Keep only meaningful troughs (price < 80% of prior ATH suggests genuine bear low)
hist_lows = hist_lows[hist_lows['price'] < hist_chart['price'].max() * 0.50].copy()

# ==================== FUTURE ATHs & CYCLE LOWS (model predicted) ====================
analysis_start = pd.to_datetime('2026-04-08')
fut_analysis = future_df[future_df['timestamp'] >= analysis_start].copy().reset_index(drop=True)

fut_peaks_idx, _ = find_peaks(
    fut_analysis['osc'].values,
    prominence=0.05,
    distance=200
)
fut_aths = fut_analysis.iloc[fut_peaks_idx][['timestamp', 'pred_price']].copy()

fut_trough_idx, _ = find_peaks(
    -fut_analysis['osc'].values,
    prominence=0.05,
    distance=200
)
fut_lows = fut_analysis.iloc[fut_trough_idx][['timestamp', 'pred_price']].copy()

# ==================== 3-MONTH TABLE ====================
analysis_df = all_df[all_df['timestamp'] >= analysis_start].copy()

print("\n" + "="*118)
print("BITCOIN CYCLE TABLE — 3-Month Windows (Apr 2026 – Dec 2040)")
print("="*118)
print(f"{'Period':<24} {'Start':<12} {'End':<12} {'Regime':<18} "
      f"{'Min Price':>13} {'Max Price':>13} {'Avg Price':>13} {'Osc Dev':>10}")
print("-"*118)

period_start = pd.to_datetime('2026-04-01')
table_rows   = []

while period_start <= pd.to_datetime('2040-10-01'):
    period_end = period_start + pd.DateOffset(months=3) - pd.Timedelta(days=1)
    mask  = (analysis_df['timestamp'] >= period_start) & (analysis_df['timestamp'] <= period_end)
    chunk = analysis_df[mask]
    if len(chunk) == 0:
        period_start += pd.DateOffset(months=3)
        continue

    dominant_regime = chunk['regime'].value_counts().idxmax()
    min_price = chunk['pred_price'].min()
    max_price = chunk['pred_price'].max()
    avg_price = chunk['pred_price'].mean()
    avg_osc   = chunk['osc'].mean()

    ath_in_window = fut_aths[
        (fut_aths['timestamp'] >= period_start) &
        (fut_aths['timestamp'] <= period_end)
    ]
    ath_flag = f"  ★ ATH ~${ath_in_window['pred_price'].values[0]:,.0f}" if len(ath_in_window) > 0 else ''

    if dominant_regime == 'buy':
        regime_label = '✅ BUY ZONE'
    elif dominant_regime == 'consolidation':
        regime_label = '⏸  CONSOLIDATION'
    else:
        regime_label = '🔴 ATH / SELL'

    period_label = f"{period_start.strftime('%b %Y')} – {period_end.strftime('%b %Y')}"
    row = dict(period=period_label,
               start=period_start.strftime('%Y-%m-%d'),
               end=period_end.strftime('%Y-%m-%d'),
               regime=dominant_regime,
               regime_label=regime_label,
               min_price=min_price, max_price=max_price, avg_price=avg_price,
               avg_osc=avg_osc, ath_flag=ath_flag)
    table_rows.append(row)

    osc_str = f"{'▼' if avg_osc < 0 else '▲'} {abs(avg_osc)*100:.2f}%"
    print(f"{period_label:<24} {period_start.strftime('%Y-%m-%d'):<12} {period_end.strftime('%Y-%m-%d'):<12} "
          f"{regime_label:<18} ${min_price:>11,.0f} ${max_price:>11,.0f} ${avg_price:>11,.0f} "
          f"{osc_str:>10}{ath_flag}")
    period_start += pd.DateOffset(months=3)

print("="*118)
print("Regime: ✅ BUY = below trend  |  ⏸ CONSOLIDATION = near trend  |  🔴 ATH/SELL = above trend")
print(f"Model: log₁₀(P) = {a:.4f} + {b:.4f}·ln(t) + {A:.4f}·e^(-{gamma:.4f}t)·sin({omega:.4f}t + {phi:.4f})")

# ==================== HALVING DATES & CAGR ====================
halving_list = [
    ('2012-11-28', '2012 Halving'),
    ('2016-07-09', '2016 Halving'),
    ('2020-05-11', '2020 Halving'),
    ('2024-04-19', '2024 Halving'),
    ('2028-04-20', '2028 Halving'),
    ('2032-04-20', '2032 Halving'),
    ('2036-04-19', '2036 Halving'),
    ('2040-04-19', '2040 Halving'),
]

print("\n" + "="*72)
print("CAGR PER HALVING CYCLE (pure log trend, oscillation removed)")
print("="*72)
cagr_annotations = []
for i in range(len(halving_list) - 1):
    start_str, label1 = halving_list[i]
    end_str,   label2 = halving_list[i+1]
    t1 = (pd.to_datetime(start_str) - genesis).total_seconds() / (24*3600*365.25)
    t2 = (pd.to_datetime(end_str)   - genesis).total_seconds() / (24*3600*365.25)
    p1 = 10 ** (a + b * np.log(t1))
    p2 = 10 ** (a + b * np.log(t2))
    cagr = (p2 / p1) ** (1 / (t2 - t1)) - 1
    cycle_name = f"{label1[:4]}–{label2[:4]}"
    print(f"  {cycle_name}: {cagr*100:5.1f}% CAGR   (${p1:>12,.0f}  →  ${p2:>12,.0f})")
    mid_date = pd.to_datetime(start_str) + (pd.to_datetime(end_str) - pd.to_datetime(start_str)) / 2
    cagr_annotations.append((mid_date, f"{cycle_name}\n{cagr*100:.1f}%"))

# ==================== CHART ====================
fig, ax = plt.subplots(figsize=(22, 12))

# --- Helper: plot contiguous segments of a given regime ---
def plot_regime_segs(df_in, regime_val, color, linestyle, linewidth, zorder, label=None):
    in_seg, seg_start = False, None
    first = True
    for pos in range(len(df_in)):
        row = df_in.iloc[pos]
        if row['regime'] == regime_val and not in_seg:
            seg_start, in_seg = pos, True
        elif row['regime'] != regime_val and in_seg:
            seg = df_in.iloc[seg_start:pos + 1]
            lbl = label if first else '_nolegend_'
            ax.plot(seg['timestamp'], seg['pred_price'],
                    color=color, linestyle=linestyle, linewidth=linewidth, zorder=zorder, label=lbl)
            in_seg, first = False, False
    if in_seg:
        seg = df_in.iloc[seg_start:]
        lbl = label if first else '_nolegend_'
        ax.plot(seg['timestamp'], seg['pred_price'],
                color=color, linestyle=linestyle, linewidth=linewidth, zorder=zorder, label=lbl)

# --- Actual price (red) ---
hist_chart_plot = df[df['timestamp'] >= chart_start]
ax.plot(hist_chart_plot['timestamp'], hist_chart_plot['price'],
        color='crimson', linewidth=1.6, zorder=3, label='Actual BTC Price')

# --- Regime-colored model fit: HISTORICAL ---
hist_model_df = all_df[all_df['timestamp'] <= last_date].copy().reset_index(drop=True)
plot_regime_segs(hist_model_df, 'buy',          '#00aa00', '--', 2.2, 4, label='Buy Zone (model fit)')
plot_regime_segs(hist_model_df, 'consolidation','#1e90ff', ':',  2.0, 4, label='Consolidation (model fit)')
plot_regime_segs(hist_model_df, 'ath',          '#e07000', '-',  2.2, 4, label='ATH/Sell Zone (model fit)')

# --- Regime-colored model fit: FUTURE ---
plot_regime_segs(future_df, 'buy',           'limegreen',  '--', 3.5, 5, label='Buy Zone (forecast)')
plot_regime_segs(future_df, 'consolidation', 'dodgerblue', ':',  2.5, 5, label='Consolidation (forecast)')
plot_regime_segs(future_df, 'ath',           'darkorange', '-',  3.0, 5, label='ATH/Sell Zone (forecast)')

# --- Pure log trend ---
ax.plot(all_dates, all_trend_price,
        color='navy', linestyle=':', linewidth=1.4, alpha=0.55, zorder=2, label='Pure Log Trend')

# --- Background regime shading (full 2012–2040) ---
# Build shading from all_df in contiguous blocks per regime
shade_colors = {'buy': 'limegreen', 'consolidation': 'dodgerblue', 'ath': 'orange'}
shade_alpha  = {'buy': 0.07,        'consolidation': 0.06,          'ath': 0.06}
in_shade, shade_regime, shade_s = False, None, None
for pos in range(len(all_df)):
    row = all_df.iloc[pos]
    r = row['regime']
    if not in_shade:
        shade_regime, shade_s, in_shade = r, row['timestamp'], True
    elif r != shade_regime:
        ax.axvspan(shade_s, row['timestamp'],
                   alpha=shade_alpha[shade_regime], color=shade_colors[shade_regime], zorder=1)
        shade_regime, shade_s = r, row['timestamp']
if in_shade:
    ax.axvspan(shade_s, all_dates[-1],
               alpha=shade_alpha[shade_regime], color=shade_colors[shade_regime], zorder=1)

# --- Historical ATH markers (actual price) ---
for _, r in hist_aths.iterrows():
    ax.plot(r['timestamp'], r['price'],
            '*', color='gold', markersize=15, markeredgecolor='darkred',
            markeredgewidth=1.5, zorder=7)
    # Format price
    p = r['price']
    p_str = f"${p/1e6:.2f}M" if p >= 1e6 else f"${p/1e3:.0f}K" if p >= 1e4 else f"${p:,.0f}"
    ax.annotate(f"ATH\n{p_str}",
                xy=(r['timestamp'], r['price']),
                xytext=(0, 28), textcoords='offset points',
                ha='center', va='bottom', fontsize=8, color='darkred', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='darkred', lw=1.0),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                          alpha=0.92, edgecolor='darkred'), zorder=8)

# --- Historical Cycle Low markers (actual price) ---
for _, r in hist_lows.iterrows():
    ax.plot(r['timestamp'], r['price'],
            'v', color='limegreen', markersize=10, markeredgecolor='darkgreen',
            markeredgewidth=1.4, zorder=7)
    p = r['price']
    p_str = f"${p/1e3:.1f}K" if p >= 1e3 else f"${p:,.0f}"
    ax.annotate(f"Cycle Low\n{p_str}",
                xy=(r['timestamp'], r['price']),
                xytext=(0, -30), textcoords='offset points',
                ha='center', va='top', fontsize=7.5, color='darkgreen', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1.0),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="honeydew",
                          alpha=0.92, edgecolor='darkgreen'), zorder=8)

# --- Future ATH markers (model predicted) ---
for _, r in fut_aths.iterrows():
    ax.plot(r['timestamp'], r['pred_price'],
            '*', color='gold', markersize=15, markeredgecolor='darkred',
            markeredgewidth=1.5, zorder=7)
    p = r['pred_price']
    p_str = f"${p/1e6:.2f}M" if p >= 1e6 else f"${p/1e3:.0f}K"
    ax.annotate(f"Pred ATH\n{p_str}",
                xy=(r['timestamp'], r['pred_price']),
                xytext=(0, 28), textcoords='offset points',
                ha='center', va='bottom', fontsize=8, color='darkred', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='darkred', lw=1.0),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                          alpha=0.92, edgecolor='darkred'), zorder=8)

# --- Future Cycle Low markers ---
for _, r in fut_lows.iterrows():
    ax.plot(r['timestamp'], r['pred_price'],
            'v', color='limegreen', markersize=10, markeredgecolor='darkgreen',
            markeredgewidth=1.4, zorder=7)
    p = r['pred_price']
    p_str = f"${p/1e6:.2f}M" if p >= 1e6 else f"${p/1e3:.0f}K"
    ax.annotate(f"Pred Low\n{p_str}",
                xy=(r['timestamp'], r['pred_price']),
                xytext=(0, -30), textcoords='offset points',
                ha='center', va='top', fontsize=7.5, color='darkgreen', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1.0),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="honeydew",
                          alpha=0.92, edgecolor='darkgreen'), zorder=8)

# --- Halving vertical lines ---
for d_str, lbl in halving_list:
    dt = pd.to_datetime(d_str)
    if chart_start <= dt <= chart_end:
        ax.axvline(dt, color='purple', linestyle='--', alpha=0.55, linewidth=1.3, zorder=3)
        ax.text(dt, 0.0, lbl[:4],
                color='purple', fontsize=7.5, rotation=90,
                va='bottom', ha='right', alpha=0.85, transform=ax.get_xaxis_transform())

# --- Present day ---
ax.axvline(last_date, color='gray', linestyle=':', linewidth=2.2, zorder=6, label='Present')

# --- CAGR annotations (all cycles, placed just above x-axis) ---
for mid_date, text in cagr_annotations:
    if chart_start <= mid_date <= chart_end:
        ax.text(mid_date, 0.02, text,
                ha='center', va='bottom', fontsize=7.8, color='purple', fontweight='bold',
                transform=ax.get_xaxis_transform(),
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          alpha=0.88, edgecolor='purple'))

# --- Axes & formatting ---
ax.set_yscale('log')

def usd_formatter(x, pos):
    if x >= 1_000_000: return f'${x/1_000_000:.1f}M'
    if x >= 10_000:    return f'${x/1_000:.0f}K'
    return f'${x:.0f}'

ax.yaxis.set_major_formatter(FuncFormatter(usd_formatter))
ax.set_xlim(chart_start, chart_end)
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# --- Equation box ---
equation = (r'$\log_{10}(P(t)) = a + b \cdot \ln(t) + A \cdot e^{-\gamma t} \cdot \sin(\omega t + \phi)$'
            f'\n$a={a:.3f},\\ b={b:.3f},\\ A={A:.3f},\\ \\gamma={gamma:.4f},\\ \\omega={omega:.4f},\\ \\phi={phi:.4f}$')
ax.text(0.01, 0.975, equation, transform=ax.transAxes,
        fontsize=10.5, va='top', ha='left', family='monospace',
        bbox=dict(boxstyle="round,pad=0.7", facecolor="white", alpha=0.95, edgecolor="#1f77b4"))

# --- Legend ---
legend_handles = [
    plt.Line2D([0], [0], color='crimson',    linewidth=1.6,                  label='Actual BTC Price'),
    # Historical fit
    plt.Line2D([0], [0], color='#00aa00',    linewidth=2.2, linestyle='--',  label='Buy Zone — historical fit'),
    plt.Line2D([0], [0], color='#1e90ff',    linewidth=2.0, linestyle=':',   label='Consolidation — historical fit'),
    plt.Line2D([0], [0], color='#e07000',    linewidth=2.2,                  label='ATH/Sell — historical fit'),
    # Future forecast
    plt.Line2D([0], [0], color='limegreen',  linewidth=3.5, linestyle='--',  label='Buy Zone — forecast'),
    plt.Line2D([0], [0], color='dodgerblue', linewidth=2.5, linestyle=':',   label='Consolidation — forecast'),
    plt.Line2D([0], [0], color='darkorange', linewidth=3.0,                  label='ATH/Sell — forecast'),
    plt.Line2D([0], [0], color='navy',       linewidth=1.4, linestyle=':',   label='Pure Log Trend', alpha=0.6),
    plt.Line2D([0], [0], color='gray',       linewidth=2.2, linestyle=':',   label='Present'),
    plt.Line2D([0], [0], marker='*',         color='gold',  markersize=12,   markeredgecolor='darkred',  linewidth=0, label='ATH (actual / predicted)'),
    plt.Line2D([0], [0], marker='v',         color='limegreen', markersize=9, markeredgecolor='darkgreen', linewidth=0, label='Cycle Low (actual / predicted)'),
]
ax.legend(handles=legend_handles, fontsize=8.5, loc='lower right', framealpha=0.95, ncol=2)

ax.set_title('Bitcoin Damped-Oscillator Model — Full Cycle History & Forecast (2012–2040)',
             fontsize=15, pad=16)
ax.set_xlabel('Year', fontsize=11)
ax.set_ylabel('Bitcoin Price (USD, log scale)', fontsize=11)
ax.grid(True, which='both', linestyle='--', alpha=0.28)

plt.tight_layout()
plt.savefig('btc_buying_opportunities.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nChart saved to btc_buying_opportunities.png")
