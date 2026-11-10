"""
Bitcoin Power-Law Pure Oscillator Model (2012 to 2040)

    log10(P(t)) = a + b * log10(t) + amp * sin(2*pi*t / period + phi)

A pure, non-decaying sinusoid around a log-log power-law trend. No staircase,
no damping, no hard ceiling (an optional $10T market-cap cap can be toggled).

Run with:
    streamlit run app.py
"""

from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.optimize import curve_fit

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GENESIS = pd.Timestamp("2009-01-03")
DAYS_PER_YEAR = 365.25
EPS = 1e-6

CIRCULATING_SUPPLY = 20_000_000
MARKET_CAP_CEILING = 10e12
PRICE_CEILING = MARKET_CAP_CEILING / CIRCULATING_SUPPLY   # ~$500,000 per BTC

PROJECTION_START = pd.Timestamp("2012-01-01")
PROJECTION_END = pd.Timestamp("2040-12-31")

# Parameters: a, b, amp, period, phi
PARAM_NAMES = ["a", "b", "amp", "period", "phi"]
LOWER_BOUNDS = [-8.0, 0.1, 0.0, 3.0, -np.pi]
UPPER_BOUNDS = [8.0, 8.0, 2.0, 5.0, np.pi]
P0 = [-2.0, 5.5, 0.4, 4.0, 0.0]

ACTUAL_COLOR = "#DC143C"
MODEL_COLOR = "black"
TREND_COLOR = "#555555"
SUPPORT_COLOR = "grey"
CAGR_COLOR = "#6a0dad"
ARROW_COLOR = "#00aa00"
CONSOLIDATION_COLOR = "#4a90d9"
EXPANSION_COLOR = "#ff8c00"

# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------
def dates_to_t(index) -> np.ndarray:
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    delta_days = (idx - GENESIS) / pd.Timedelta(days=1)
    t = np.asarray(delta_days, dtype=float) / DAYS_PER_YEAR
    return np.maximum(t, EPS)


def t_to_date(t: float) -> pd.Timestamp:
    return GENESIS + pd.to_timedelta(float(t) * DAYS_PER_YEAR, unit="D")


# ---------------------------------------------------------------------------
# Model mathematics
# ---------------------------------------------------------------------------
def trend_log10(t, a, b):
    t = np.maximum(np.asarray(t, dtype=float), EPS)
    return a + b * np.log10(t)


def oscillator(t, amp, period, phi):
    t = np.asarray(t, dtype=float)
    return amp * np.sin(2.0 * np.pi * t / period + phi)


def model_log10(t, a, b, amp, period, phi):
    """Pure oscillator: power-law trend plus a constant-amplitude sinusoid."""
    return trend_log10(t, a, b) + oscillator(t, amp, period, phi)


def model_price(t, params, cap: bool = False) -> np.ndarray:
    p = 10.0 ** model_log10(t, *params)
    if cap:
        p = np.minimum(p, PRICE_CEILING)
    return p


def trend_price(t, params, cap: bool = False) -> np.ndarray:
    p = 10.0 ** trend_log10(t, params[0], params[1])
    if cap:
        p = np.minimum(p, PRICE_CEILING)
    return p


# ---------------------------------------------------------------------------
# Data ingestion
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def load_btc_history() -> pd.DataFrame:
    df = yf.download("BTC-USD", period="max", interval="1d",
                     auto_adjust=True, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if "Close" not in df.columns:
        return pd.DataFrame()

    out = df[["Close"]].copy()
    out["Close"] = pd.to_numeric(out["Close"], errors="coerce")
    out = out.dropna()
    out = out[out["Close"] > 0]
    out.index = pd.to_datetime(out.index)
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


# ---------------------------------------------------------------------------
# Curve fitting
# ---------------------------------------------------------------------------
def fit_model(t: np.ndarray, log_price: np.ndarray):
    """
    Fit the pure oscillator. The phase parameter makes the loss surface
    multimodal, so we seed several phase offsets and keep the best fit.
    """
    best = None
    best_sse = np.inf
    errors = []

    for phi0 in np.linspace(-np.pi, np.pi, 8, endpoint=False):
        p0 = list(P0)
        p0[4] = float(phi0)
        try:
            popt, _ = curve_fit(
                model_log10, t, log_price, p0=p0,
                bounds=(LOWER_BOUNDS, UPPER_BOUNDS), maxfev=20000,
            )
            if not np.all(np.isfinite(popt)):
                continue
            sse = float(np.sum((log_price - model_log10(t, *popt)) ** 2))
            if sse < best_sse:
                best_sse, best = sse, popt
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))

    if best is None:
        msg = "Curve fit failed; using default parameters."
        if errors:
            msg += f" Last error: {errors[-1]}"
        return np.array(P0, dtype=float), False, msg
    return best, True, "Curve fit converged."


def fit_quality(t, log_price, params) -> dict:
    pred = model_log10(t, *params)
    resid = log_price - pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((log_price - np.mean(log_price)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"r2": r2, "rmse_log10": float(np.sqrt(np.mean(resid ** 2)))}


# ---------------------------------------------------------------------------
# Phase / epoch construction
# ---------------------------------------------------------------------------
def upward_crossings(params, t_start: float, t_end: float) -> np.ndarray:
    """
    Times where the oscillator crosses zero going up, i.e. price moves from
    below trend to above trend. This is the steepest point of the ascent and
    the natural kickoff of each expansion.

    sin(2*pi*t/period + phi) = 0 with positive slope  =>  2*pi*t/period + phi = 2*pi*k
    """
    period, phi = float(params[3]), float(params[4])
    k_lo = int(np.floor((2 * np.pi * t_start / period + phi) / (2 * np.pi))) - 1
    k_hi = int(np.ceil((2 * np.pi * t_end / period + phi) / (2 * np.pi))) + 1
    ks = np.arange(k_lo, k_hi + 1)
    return (2 * np.pi * ks - phi) * period / (2 * np.pi)


def build_phases(params, t_start: float, t_end: float, expansion_frac: float) -> list:
    """
    Expansion epochs are centred on each upward zero crossing of the oscillator
    and span `expansion_frac` of the period. Consolidation fills the remainder.
    """
    period = float(params[3])
    half_exp = expansion_frac * period / 2.0
    phases = []
    for tc in upward_crossings(params, t_start, t_end):
        e_s, e_e = tc - half_exp, tc + half_exp
        phases.append(("Expansion", e_s, e_e))
        phases.append(("Consolidation", e_e, e_s + period))

    clipped = []
    for name, s, e in phases:
        s_c, e_c = max(s, t_start), min(e, t_end)
        if e_c - s_c > 1e-6:
            clipped.append((name, s_c, e_c))
    clipped.sort(key=lambda x: x[1])
    return clipped


def peaks_and_troughs(params, t_start: float, t_end: float):
    """Model peak (oscillator = +amp) and trough (oscillator = -amp) times."""
    period, phi = float(params[3]), float(params[4])
    crossings = upward_crossings(params, t_start - period, t_end + period)
    peaks = crossings + period / 4.0
    troughs = crossings + 3.0 * period / 4.0
    peaks = peaks[(peaks >= t_start) & (peaks <= t_end)]
    troughs = troughs[(troughs >= t_start) & (troughs <= t_end)]
    return peaks, troughs


def annualized_cagr(p0: float, p1: float, years: float) -> float:
    if p0 <= 0 or years <= 0:
        return float("nan")
    return (p1 / p0) ** (1.0 / years) - 1.0


def fmt_price(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    if p >= 1_000_000:
        return f"${p / 1e6:,.2f}M"
    if p >= 1000:
        return f"${p:,.0f}"
    if p >= 1:
        return f"${p:,.2f}"
    return f"${p:,.4f}"


def currency_tick(v, _pos=None) -> str:
    if v >= 1_000_000:
        return f"${v / 1e6:g}M"
    if v >= 1:
        return f"${v:,.0f}"
    return f"${v:g}"


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def build_figure(hist, params, support_discount, phases, apply_cap, show_trend, show_extrema):
    proj_dates = pd.date_range(PROJECTION_START, PROJECTION_END, freq="D")
    t_proj = dates_to_t(proj_dates)
    p_model = model_price(t_proj, params, apply_cap)
    p_trend = trend_price(t_proj, params, apply_cap)
    p_support = p_model * (1.0 - support_discount)

    fig, ax = plt.subplots(figsize=(16, 9), dpi=110)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for name, s, e in phases:
        color = EXPANSION_COLOR if name == "Expansion" else CONSOLIDATION_COLOR
        alpha = 0.18 if name == "Expansion" else 0.07
        ax.axvspan(t_to_date(s), t_to_date(e), color=color, alpha=alpha, lw=0, zorder=0)

    ax.fill_between(proj_dates, p_support, p_model, color=SUPPORT_COLOR, alpha=0.15, lw=0,
                    zorder=1, label=f"Support channel ({int(support_discount * 100)}% below model)")

    if show_trend:
        ax.plot(proj_dates, p_trend, linestyle="-", color=TREND_COLOR, linewidth=0.9,
                alpha=0.8, zorder=2, label="Power-law trend (a + b log10 t)")

    ax.plot(proj_dates, p_model, linestyle="--", color=MODEL_COLOR, linewidth=1.2,
            zorder=3, label="Power-law pure oscillator")

    ax.plot(hist.index, hist["Close"].values, color=ACTUAL_COLOR, linewidth=1.5,
            zorder=4, label="BTC-USD (actual)")

    if apply_cap:
        ax.axhline(PRICE_CEILING, color="#444444", linestyle=":", linewidth=1.0, zorder=2)
        ax.text(PROJECTION_START, PRICE_CEILING * 1.15,
                f"$10T market-cap cap (~{fmt_price(PRICE_CEILING)}/BTC)",
                fontsize=9, color="#444444", va="bottom", ha="left")

    t_start, t_end = float(t_proj[0]), float(t_proj[-1])

    # Peak / trough markers
    if show_extrema:
        peaks, troughs = peaks_and_troughs(params, t_start, t_end)
        if len(peaks):
            ax.plot([t_to_date(x) for x in peaks], model_price(peaks, params, apply_cap),
                    linestyle="none", marker="v", markersize=7, color="#b22222", zorder=6,
                    label="Model cycle peak")
        if len(troughs):
            ax.plot([t_to_date(x) for x in troughs], model_price(troughs, params, apply_cap),
                    linestyle="none", marker="o", markersize=6, markerfacecolor="none",
                    markeredgecolor="#1f77b4", zorder=6, label="Model cycle trough")

    # CAGR markers and kickoff arrows
    for name, s, e in phases:
        years = e - s
        if years < 0.25:
            continue
        p_s = float(model_price(s, params, apply_cap))
        p_e = float(model_price(e, params, apply_cap))
        cagr = annualized_cagr(p_s, p_e, years)
        mid = (s + e) / 2.0
        seg = np.linspace(s, e, 60)
        y_top = float(np.max(model_price(seg, params, apply_cap))) * (2.6 if name == "Consolidation" else 4.0)
        if np.isfinite(cagr):
            ax.text(t_to_date(mid), y_top, f"{cagr * 100:+.0f}% CAGR",
                    color=CAGR_COLOR, fontsize=9, fontweight="bold",
                    ha="center", va="bottom", zorder=7)

        if name == "Expansion":
            kickoff_date = t_to_date(s)
            ax.plot([kickoff_date], [p_s], marker="^", markersize=14, linestyle="none",
                    markerfacecolor="none", markeredgecolor=ARROW_COLOR, markeredgewidth=1.8,
                    zorder=7)
            ax.annotate(
                f"{kickoff_date.strftime('%b %Y')}\ntarget {fmt_price(p_e)}",
                xy=(kickoff_date, p_s), xytext=(10, -42), textcoords="offset points",
                fontsize=8, color=ARROW_COLOR, ha="left", va="top", zorder=7,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=ARROW_COLOR, alpha=0.85, lw=0.8),
            )

    ax.set_yscale("log")
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=14))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(currency_tick))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())

    y_min = max(1.0, float(hist["Close"].min()) / 3.0)
    y_max = float(np.nanmax(p_model)) * 30
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(PROJECTION_START, PROJECTION_END)

    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))

    ax.grid(True, which="major", axis="y", color="#dddddd", linewidth=0.8)
    ax.grid(True, which="minor", axis="y", color="#f0f0f0", linewidth=0.5)
    ax.grid(True, which="major", axis="x", color="#eeeeee", linewidth=0.6)

    ax.set_title("Bitcoin Power-Law Pure Oscillator (2012 to 2040)",
                 fontsize=15, fontweight="bold", pad=14)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("BTC-USD (log scale)", fontsize=11)

    handles, labels = ax.get_legend_handles_labels()
    handles += [
        Patch(facecolor=CONSOLIDATION_COLOR, alpha=0.25, label="Consolidation epoch"),
        Patch(facecolor=EXPANSION_COLOR, alpha=0.4, label="Expansion epoch"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="none",
               markeredgecolor=ARROW_COLOR, markersize=11, markeredgewidth=1.6,
               label="Expansion kickoff (crosses above trend)"),
        Line2D([0], [0], color=CAGR_COLOR, lw=0, marker="$\\%$", markersize=9,
               label="Annualized CAGR"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=9, framealpha=0.92)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    return fig


def build_phase_table(params, phases, apply_cap) -> pd.DataFrame:
    rows = []
    for name, s, e in phases:
        years = e - s
        p_s = float(model_price(s, params, apply_cap))
        p_e = float(model_price(e, params, apply_cap))
        cagr = annualized_cagr(p_s, p_e, years)
        rows.append({
            "Phase": name,
            "Start Date": t_to_date(s).strftime("%Y-%m-%d"),
            "End Date": t_to_date(e).strftime("%Y-%m-%d"),
            "Duration (yrs)": f"{years:.2f}",
            "Start Price": fmt_price(p_s),
            "End Price": fmt_price(p_e),
            "Annualized CAGR": f"{cagr * 100:+.1f}%" if np.isfinite(cagr) else "n/a",
        })
    return pd.DataFrame(rows)


def build_extrema_table(params, t_start, t_end, apply_cap) -> pd.DataFrame:
    peaks, troughs = peaks_and_troughs(params, t_start, t_end)
    rows = []
    for x in peaks:
        rows.append({"Type": "Peak", "Date": t_to_date(x).strftime("%Y-%m-%d"),
                     "Model Price": fmt_price(float(model_price(x, params, apply_cap))),
                     "Trend Price": fmt_price(float(trend_price(x, params, apply_cap)))})
    for x in troughs:
        rows.append({"Type": "Trough", "Date": t_to_date(x).strftime("%Y-%m-%d"),
                     "Model Price": fmt_price(float(model_price(x, params, apply_cap))),
                     "Trend Price": fmt_price(float(trend_price(x, params, apply_cap)))})
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Date").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="BTC Power-Law Oscillator", page_icon="₿", layout="wide")

    st.markdown(
        """
        <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Bitcoin Power-Law Pure Oscillator")
    st.caption(
        "Log-log power-law trend with a constant-amplitude sinusoid. No damping, no staircase, "
        "no hard ceiling by default. Data: Yahoo Finance (BTC-USD, daily)."
    )

    with st.spinner("Fetching BTC-USD history from Yahoo Finance..."):
        try:
            hist = load_btc_history()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Data download failed: {exc}")
            st.stop()

    if hist is None or hist.empty or len(hist) < 100:
        st.error("No usable BTC-USD price data was returned. Check network access and try again.")
        st.stop()

    t_hist = dates_to_t(hist.index)
    y_hist = np.log10(hist["Close"].values.astype(float))
    mask = np.isfinite(y_hist)
    if not mask.all():
        hist, t_hist, y_hist = hist[mask], t_hist[mask], y_hist[mask]
        if hist.empty:
            st.error("All price observations were invalid after cleaning.")
            st.stop()

    fitted, fit_ok, fit_msg = fit_model(t_hist, y_hist)

    # ---- Sidebar ----
    st.sidebar.header("Model Controls")
    use_fitted = st.sidebar.checkbox("Use fitted parameters", value=True)

    if use_fitted:
        params = np.array(fitted, dtype=float)
    else:
        st.sidebar.markdown("Manual overrides:")
        labels = {
            "a": "a (trend intercept)",
            "b": "b (power-law exponent)",
            "amp": "amp (oscillator amplitude, log10)",
            "period": "period (years)",
            "phi": "phi (phase, radians)",
        }
        params = []
        for i, name in enumerate(PARAM_NAMES):
            lo, hi = float(LOWER_BOUNDS[i]), float(UPPER_BOUNDS[i])
            default = float(np.clip(fitted[i], lo, hi))
            params.append(st.sidebar.slider(labels[name], lo, hi, default, step=(hi - lo) / 400.0))
        params = np.array(params, dtype=float)

    st.sidebar.markdown("---")
    support_discount = st.sidebar.slider("Support channel discount", 0.05, 0.60, 0.30, 0.05, format="%.2f")
    expansion_frac = st.sidebar.slider("Expansion share of cycle", 0.10, 0.50, 0.20, 0.05, format="%.2f")
    show_trend = st.sidebar.checkbox("Show bare power-law trend", value=True)
    show_extrema = st.sidebar.checkbox("Mark model peaks and troughs", value=True)
    apply_cap = st.sidebar.checkbox("Apply $10T market-cap cap", value=False)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Fitted parameters**")
    st.sidebar.dataframe(
        pd.DataFrame({"param": PARAM_NAMES, "value": [f"{v:.4f}" for v in fitted]}),
        hide_index=True,
    )
    if fit_ok:
        st.sidebar.success(fit_msg)
    else:
        st.sidebar.warning(fit_msg)

    # ---- Metrics ----
    q = fit_quality(t_hist, y_hist, params)
    last_date = hist.index[-1]
    last_price = float(hist["Close"].iloc[-1])
    t_now = float(t_hist[-1])
    model_now = float(model_price(t_now, params, apply_cap))
    trend_now = float(trend_price(t_now, params, apply_cap))
    osc_now = float(oscillator(t_now, params[2], params[3], params[4]))
    deviation = last_price / model_now - 1.0
    cycle_pos = (np.mod(2 * np.pi * t_now / params[3] + params[4], 2 * np.pi)) / (2 * np.pi)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Last close", fmt_price(last_price), last_date.strftime("%Y-%m-%d"))
    c2.metric("Model value", fmt_price(model_now), f"{deviation * 100:+.1f}% actual vs model")
    c3.metric("Trend value", fmt_price(trend_now), f"osc {osc_now:+.2f} log10 ({10 ** osc_now:.2f}x)")
    c4.metric("Cycle position", f"{cycle_pos * 100:.0f}%", f"period {params[3]:.2f} yrs")
    c5.metric("Fit R² (log10)", f"{q['r2']:.3f}")
    c6.metric("Fit RMSE (log10)", f"{q['rmse_log10']:.3f}")

    # ---- Chart ----
    t_start = float(dates_to_t([PROJECTION_START])[0])
    t_end = float(dates_to_t([PROJECTION_END])[0])
    phases = build_phases(params, t_start, t_end, expansion_frac)

    fig = build_figure(hist, params, support_discount, phases, apply_cap, show_trend, show_extrema)
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

    # ---- Tables ----
    st.subheader("Audited Macro Cycles")
    st.markdown(
        "Expansion epochs are centred on the point where the oscillator crosses from below trend "
        "to above trend (steepest ascent). Consolidation fills the rest of each period. "
        "Prices are model values, not actuals."
    )
    st.dataframe(build_phase_table(params, phases, apply_cap), hide_index=True)

    st.subheader("Model Peaks and Troughs")
    st.dataframe(build_extrema_table(params, t_start, t_end, apply_cap), hide_index=True)

    with st.expander("Model definition"):
        st.latex(r"\log_{10} P(t) = a + b \cdot \log_{10}(t) + \text{amp} \cdot \sin\!\left(\frac{2\pi t}{\text{period}} + \varphi\right)")
        st.markdown(
            f"- $t$ is fractional years since Genesis ({GENESIS.date()}), floored at {EPS:g}.\n"
            "- The sinusoid has constant amplitude: no decay term, so every cycle overshoots and "
            "undershoots the trend by the same multiple (10^amp).\n"
            "- Peaks occur a quarter period after each upward trend crossing; troughs three quarters after.\n"
            f"- Optional cap: $10T / {CIRCULATING_SUPPLY:,} BTC = {fmt_price(PRICE_CEILING)} per BTC "
            "(off by default).\n"
            f"- Fit bounds: lower {[round(float(x), 3) for x in LOWER_BOUNDS]}, "
            f"upper {[round(float(x), 3) for x in UPPER_BOUNDS]}."
        )

    st.caption(
        f"Rendered {datetime.now().strftime('%Y-%m-%d %H:%M')}. "
        "Curve-fitting exercise for illustration, not investment advice."
    )


if __name__ == "__main__":
    main()
