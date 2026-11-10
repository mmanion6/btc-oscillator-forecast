"""
Bitcoin Power-Law Underdamped Staircase & Macro Oscillator Model (2012 to 2040)

Run with:
    streamlit run app.py

Dependencies:
    pip install streamlit matplotlib numpy pandas scipy yfinance
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

CIRCULATING_SUPPLY = 20_000_000          # ~20M BTC
MARKET_CAP_CEILING = 10e12               # $10 Trillion hard cap
PRICE_CEILING = MARKET_CAP_CEILING / CIRCULATING_SUPPLY   # ~$500,000 per BTC
LOG_CEILING = np.log10(PRICE_CEILING)

PROJECTION_START = pd.Timestamp("2012-01-01")
PROJECTION_END = pd.Timestamp("2040-12-31")

PARAM_NAMES = ["a", "b", "step_h", "step_f", "sharp", "amp", "decay"]
LOWER_BOUNDS = [-5, 0.1, 0.1, 3.5, 10, -2, 0]
UPPER_BOUNDS = [5, 5, 2, 5.0, 60, 2, 5]
P0 = [-1.0, 3.5, 0.8, 4.0, 20.0, 0.3, 0.5]

ACTUAL_COLOR = "#DC143C"
MODEL_COLOR = "black"
SUPPORT_COLOR = "grey"
CAGR_COLOR = "#6a0dad"
ARROW_COLOR = "#00aa00"
CONSOLIDATION_COLOR = "#4a90d9"
EXPANSION_COLOR = "#ff8c00"

# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------
def dates_to_t(index: pd.DatetimeIndex) -> np.ndarray:
    """Convert datetimes to fractional years since Genesis with epsilon offset."""
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    delta_days = (idx - GENESIS) / pd.Timedelta(days=1)
    t = np.asarray(delta_days, dtype=float) / DAYS_PER_YEAR
    return np.maximum(t, EPS)


def t_to_date(t: float) -> pd.Timestamp:
    """Convert fractional years since Genesis back to a Timestamp."""
    return GENESIS + pd.to_timedelta(float(t) * DAYS_PER_YEAR, unit="D")


# ---------------------------------------------------------------------------
# Model mathematics
# ---------------------------------------------------------------------------
def model_log10(t, a, b, step_h, step_f, sharp, amp, decay):
    """
    log10(P(t)) = min( Trend + Staircase + Wobble, log10(CEILING) )

    Trend     = a + b * log10(t)
    Staircase = step_h * ( floor(t / step_f) + sigmoid(sharp * (phase - step_f / 2)) )
    Wobble    = amp * exp(-decay * phase) * sin(2 * pi * phase / step_f)
    phase     = t mod step_f
    """
    t = np.maximum(np.asarray(t, dtype=float), EPS)
    phase = np.mod(t, step_f)

    trend = a + b * np.log10(t)

    sig_arg = np.clip(-sharp * (phase - step_f / 2.0), -500.0, 500.0)
    staircase = step_h * (np.floor(t / step_f) + 1.0 / (1.0 + np.exp(sig_arg)))

    wobble = amp * np.exp(-decay * phase) * np.sin(2.0 * np.pi * phase / step_f)

    return np.minimum(trend + staircase + wobble, LOG_CEILING)


def model_price(t, params) -> np.ndarray:
    return 10.0 ** model_log10(t, *params)


# ---------------------------------------------------------------------------
# Data ingestion
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def load_btc_history() -> pd.DataFrame:
    df = yf.download(
        "BTC-USD",
        period="max",
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if df is None or df.empty:
        return pd.DataFrame()

    # Strip multi-index column layers dynamically
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
    """Fit the staircase model. Returns (params, success_flag, message)."""
    try:
        popt, _ = curve_fit(
            model_log10,
            t,
            log_price,
            p0=P0,
            bounds=(LOWER_BOUNDS, UPPER_BOUNDS),
            maxfev=20000,
        )
        if not np.all(np.isfinite(popt)):
            raise ValueError("Non-finite parameters returned by optimizer.")
        return popt, True, "Curve fit converged."
    except Exception as exc:  # noqa: BLE001
        return np.array(P0, dtype=float), False, f"Curve fit failed ({exc}); using default parameters."


def fit_quality(t: np.ndarray, log_price: np.ndarray, params) -> dict:
    pred = model_log10(t, *params)
    resid = log_price - pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((log_price - np.mean(log_price)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    return {"r2": r2, "rmse_log10": rmse}


# ---------------------------------------------------------------------------
# Phase / epoch construction
# ---------------------------------------------------------------------------
def build_phases(params, t_start: float, t_end: float, expansion_frac: float) -> list:
    """
    Construct alternating Consolidation / Expansion epochs.

    The Expansion epoch is centred on the sigmoid breakout (phase = step_f / 2)
    and spans `expansion_frac` of the step period. Consolidation fills the rest
    and wraps across the cycle boundary.
    """
    step_f = float(params[3])
    half_exp = expansion_frac * step_f / 2.0

    phases = []
    k_min = int(np.floor(t_start / step_f)) - 1
    k_max = int(np.ceil(t_end / step_f)) + 1

    for k in range(k_min, k_max + 1):
        centre = k * step_f + step_f / 2.0
        exp_start = centre - half_exp
        exp_end = centre + half_exp
        next_exp_start = exp_start + step_f

        phases.append(("Expansion", exp_start, exp_end))
        phases.append(("Consolidation", exp_end, next_exp_start))

    # Clip to plotting window, drop empties, sort
    clipped = []
    for name, s, e in phases:
        s_c, e_c = max(s, t_start), min(e, t_end)
        if e_c - s_c > 1e-6:
            clipped.append((name, s_c, e_c))
    clipped.sort(key=lambda x: x[1])
    return clipped


def annualized_cagr(p0: float, p1: float, years: float) -> float:
    if p0 <= 0 or years <= 0:
        return float("nan")
    return (p1 / p0) ** (1.0 / years) - 1.0


def fmt_price(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    if p >= 1000:
        return f"${p:,.0f}"
    if p >= 1:
        return f"${p:,.2f}"
    return f"${p:,.4f}"


def currency_tick(v, _pos=None) -> str:
    if v >= 1:
        return f"${v:,.0f}"
    return f"${v:g}"


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def build_figure(hist: pd.DataFrame, params, support_discount: float, phases: list):
    proj_dates = pd.date_range(PROJECTION_START, PROJECTION_END, freq="D")
    t_proj = dates_to_t(proj_dates)
    p_proj = model_price(t_proj, params)
    p_support = p_proj * (1.0 - support_discount)

    fig, ax = plt.subplots(figsize=(16, 9), dpi=110)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Phase shading
    for name, s, e in phases:
        color = EXPANSION_COLOR if name == "Expansion" else CONSOLIDATION_COLOR
        alpha = 0.18 if name == "Expansion" else 0.07
        ax.axvspan(t_to_date(s), t_to_date(e), color=color, alpha=alpha, lw=0, zorder=0)

    # Support channel
    ax.fill_between(
        proj_dates, p_support, p_proj,
        color=SUPPORT_COLOR, alpha=0.15, lw=0, zorder=1,
        label=f"Support channel ({int(support_discount * 100)}% below model)",
    )

    # Model line
    ax.plot(proj_dates, p_proj, linestyle="--", color=MODEL_COLOR, linewidth=1.2,
            zorder=3, label="Power-law staircase model")

    # Actual price
    ax.plot(hist.index, hist["Close"].values, color=ACTUAL_COLOR, linewidth=1.5,
            zorder=4, label="BTC-USD (actual)")

    # Ceiling
    ax.axhline(PRICE_CEILING, color="#444444", linestyle=":", linewidth=1.0, zorder=2)
    ax.text(PROJECTION_START, PRICE_CEILING * 1.15,
            f"$10T market-cap ceiling (~{fmt_price(PRICE_CEILING)}/BTC)",
            fontsize=9, color="#444444", va="bottom", ha="left")

    # CAGR markers and kickoff arrows
    for name, s, e in phases:
        years = e - s
        if years < 0.25:
            continue
        p_s = float(model_price(s, params))
        p_e = float(model_price(e, params))
        cagr = annualized_cagr(p_s, p_e, years)
        mid = (s + e) / 2.0
        seg_t = np.linspace(s, e, 50)
        y_top = float(np.max(model_price(seg_t, params))) * (2.6 if name == "Consolidation" else 4.0)
        y_top = min(y_top, PRICE_CEILING * 12)
        if np.isfinite(cagr):
            ax.text(t_to_date(mid), y_top, f"{cagr * 100:+.0f}% CAGR",
                    color=CAGR_COLOR, fontsize=9, fontweight="bold",
                    ha="center", va="bottom", zorder=6)

        if name == "Expansion":
            kickoff_date = t_to_date(s)
            ax.plot([kickoff_date], [p_s], marker="^", markersize=14, linestyle="none",
                    markerfacecolor="none", markeredgecolor=ARROW_COLOR, markeredgewidth=1.8,
                    zorder=7)
            ax.annotate(
                f"{kickoff_date.strftime('%b %Y')}\ntarget {fmt_price(p_e)}",
                xy=(kickoff_date, p_s),
                xytext=(10, -42), textcoords="offset points",
                fontsize=8, color=ARROW_COLOR, ha="left", va="top", zorder=7,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=ARROW_COLOR, alpha=0.85, lw=0.8),
            )

    # Axes formatting
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=12))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(currency_tick))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())

    y_min = max(1.0, float(hist["Close"].min()) / 3.0)
    ax.set_ylim(y_min, PRICE_CEILING * 25)
    ax.set_xlim(PROJECTION_START, PROJECTION_END)

    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))

    ax.grid(True, which="major", axis="y", color="#dddddd", linewidth=0.8)
    ax.grid(True, which="minor", axis="y", color="#f0f0f0", linewidth=0.5)
    ax.grid(True, which="major", axis="x", color="#eeeeee", linewidth=0.6)

    ax.set_title("Bitcoin Power-Law Underdamped Staircase & Macro Oscillator (2012 to 2040)",
                 fontsize=15, fontweight="bold", pad=14)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("BTC-USD (log scale)", fontsize=11)

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    handles += [
        Patch(facecolor=CONSOLIDATION_COLOR, alpha=0.25, label="Consolidation epoch"),
        Patch(facecolor=EXPANSION_COLOR, alpha=0.4, label="Expansion epoch"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="none",
               markeredgecolor=ARROW_COLOR, markersize=11, markeredgewidth=1.6,
               label="Expansion kickoff"),
        Line2D([0], [0], color=CAGR_COLOR, lw=0, marker="$\\%$", markersize=9,
               label="Annualized CAGR"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=9, framealpha=0.92)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    return fig


def build_phase_table(params, phases: list) -> pd.DataFrame:
    rows = []
    for name, s, e in phases:
        years = e - s
        p_s = float(model_price(s, params))
        p_e = float(model_price(e, params))
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


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="BTC Power-Law Staircase Model",
        page_icon="₿",
        layout="wide",
    )

    st.markdown(
        """
        <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
            .metric-note {color: #666; font-size: 0.85rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Bitcoin Power-Law Underdamped Staircase & Macro Oscillator")
    st.caption(
        "Log-log power-law trend + sigmoid halving staircase + underdamped decay wobble, "
        "capped at a $10T market cap. Data: Yahoo Finance (BTC-USD, daily)."
    )

    # ---- Data ----
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

    if not np.all(np.isfinite(y_hist)):
        mask = np.isfinite(y_hist)
        hist, t_hist, y_hist = hist[mask], t_hist[mask], y_hist[mask]
        if hist.empty:
            st.error("All price observations were invalid after cleaning.")
            st.stop()

    # ---- Fit ----
    fitted, fit_ok, fit_msg = fit_model(t_hist, y_hist)

    # ---- Sidebar ----
    st.sidebar.header("Model Controls")
    use_fitted = st.sidebar.checkbox("Use fitted parameters", value=True)

    if use_fitted:
        params = np.array(fitted, dtype=float)
    else:
        st.sidebar.markdown("Manual overrides (bounded to fit ranges):")
        params = []
        labels = {
            "a": "a (trend intercept)",
            "b": "b (power-law exponent)",
            "step_h": "step_h (log10 step height)",
            "step_f": "step_f (step period, yrs)",
            "sharp": "sharp (breakout sharpness)",
            "amp": "amp (wobble amplitude)",
            "decay": "decay (wobble decay)",
        }
        for i, name in enumerate(PARAM_NAMES):
            lo, hi = float(LOWER_BOUNDS[i]), float(UPPER_BOUNDS[i])
            default = float(np.clip(fitted[i], lo, hi))
            params.append(st.sidebar.slider(labels[name], lo, hi, default, step=(hi - lo) / 200.0))
        params = np.array(params, dtype=float)

    st.sidebar.markdown("---")
    support_discount = st.sidebar.slider("Support channel discount", 0.05, 0.60, 0.30, 0.05,
                                         format="%.2f")
    expansion_frac = st.sidebar.slider("Expansion share of cycle", 0.10, 0.40, 0.20, 0.05,
                                       format="%.2f")

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
    model_now = float(model_price(t_hist[-1], params))
    deviation = last_price / model_now - 1.0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Last close", fmt_price(last_price), last_date.strftime("%Y-%m-%d"))
    c2.metric("Model fair value", fmt_price(model_now), f"{deviation * 100:+.1f}% actual vs model")
    c3.metric("Support floor", fmt_price(model_now * (1 - support_discount)))
    c4.metric("Fit R² (log10)", f"{q['r2']:.3f}")
    c5.metric("Fit RMSE (log10)", f"{q['rmse_log10']:.3f}")

    # ---- Chart ----
    t_start = float(dates_to_t(pd.DatetimeIndex([PROJECTION_START]))[0])
    t_end = float(dates_to_t(pd.DatetimeIndex([PROJECTION_END]))[0])
    phases = build_phases(params, t_start, t_end, expansion_frac)

    fig = build_figure(hist, params, support_discount, phases)
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

    # ---- Phase table ----
    st.subheader("Audited Macro Cycles")
    st.markdown(
        "Each row is one epoch of the model. Expansion epochs are centred on the sigmoid "
        "breakout of the staircase; consolidation epochs fill the remainder of each step period. "
        "Prices are model values, not actuals."
    )
    table = build_phase_table(params, phases)
    st.dataframe(table, hide_index=True)

    # ---- Model definition ----
    with st.expander("Model definition"):
        st.latex(r"\log_{10} P(t) = \min\Big(\text{Trend}(t) + \text{Staircase}(t) + \text{Wobble}(t),\ \log_{10}(\text{CEILING})\Big)")
        st.latex(r"\text{Trend}(t) = a + b \cdot \log_{10}(t)")
        st.latex(r"\text{Staircase}(t) = \text{step\_h}\left(\left\lfloor \tfrac{t}{\text{step\_f}} \right\rfloor + \frac{1}{1 + e^{-\text{sharp}\,(t \bmod \text{step\_f} - \text{step\_f}/2)}}\right)")
        st.latex(r"\text{Wobble}(t) = \text{amp} \cdot e^{-\text{decay}\,(t \bmod \text{step\_f})} \cdot \sin\!\left(2\pi \frac{t \bmod \text{step\_f}}{\text{step\_f}}\right)")
        st.markdown(
            f"- $t$ is fractional years since Genesis ({GENESIS.date()}), floored at {EPS:g}.\n"
            f"- CEILING = $10T / {CIRCULATING_SUPPLY:,} BTC = {fmt_price(PRICE_CEILING)} per BTC.\n"
            f"- Fit bounds: lower {LOWER_BOUNDS}, upper {UPPER_BOUNDS}."
        )

    st.caption(
        f"Rendered {datetime.now().strftime('%Y-%m-%d %H:%M')}. "
        "This is a curve-fitting exercise for illustration, not investment advice."
    )


if __name__ == "__main__":
    main()
