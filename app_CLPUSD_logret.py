"""
CLP/USD Forecast — Streamlit App
Loads the artifact saved from Regresion_CLPUSD.ipynb (log-returns framework).

Usage:
    streamlit run "C:\\Users\\fsalgado\\Python Data\\app_CLPUSD_logret.py"
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import streamlit as st

# ── MUST be the very first Streamlit call ────────────────────────────────────
st.set_page_config(page_title="CLP/USD Forecast", layout="wide")

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# ── Custom transformers ───────────────────────────────────────────────────────
from clp_transformers import (
    LogReturnTransformer,
    MonthlyDiffTransformer,
    ForwardFillTransformer,
    SimpleImputerModel,
    TargetPreprocessorLogReturn,
)


# ── Config ───────────────────────────────────────────────────────────────────
# Relative path — works locally and on Streamlit Cloud
ARTIFACT_PATH = Path(__file__).parent / "model_CLPUSD_logret.pkl"
N_MONTHS      = 12
LOOKBACK_HIST = 3   # years in the scenario editor chart
LOOKBACK_REF  = 5   # years in the historical reference section


# ── Load artifact ─────────────────────────────────────────────────────────────
# Robust loader: injects proxy modules for ALL possible module names the pkl
# might use for our custom classes (__main__, transformers, clp_transformers),
# so it works regardless of which name was stamped at save time.
@st.cache_resource
def load_artifact():
    import sys, types
    import clp_transformers as _ct

    _CLS = {
        'LogReturnTransformer':        _ct.LogReturnTransformer,
        'MonthlyDiffTransformer':      _ct.MonthlyDiffTransformer,
        'ForwardFillTransformer':      _ct.ForwardFillTransformer,
        'SimpleImputerModel':          _ct.SimpleImputerModel,
        'TargetPreprocessorLogReturn': _ct.TargetPreprocessorLogReturn,
    }
    # Temporarily inject proxy modules so pickle resolves any module name
    _saved = {}
    for _mod_name in ('__main__', 'transformers', 'clp_transformers'):
        _saved[_mod_name] = sys.modules.get(_mod_name)
        _proxy = types.ModuleType(_mod_name)
        for _n, _c in _CLS.items():
            setattr(_proxy, _n, _c)
        sys.modules[_mod_name] = _proxy
    try:
        return joblib.load(ARTIFACT_PATH)
    finally:
        for _mod_name, _orig in _saved.items():
            if _orig is None:
                sys.modules.pop(_mod_name, None)
            else:
                sys.modules[_mod_name] = _orig

art                     = load_artifact()
model                   = art["model"]
final_selected_features = art["final_selected_features"]
raw_col_list            = art["raw_col_list"]
hist_data: pd.DataFrame = art["hist_data"]
last_raw: dict          = art["last_raw"]
last_fx: float          = art["last_known_price"]
last_date: pd.Timestamp = pd.Timestamp(art["last_date"])
metrics: dict           = art["metrics"]

_raw_map = {f: f.replace("logret_", "").replace("diff_", "") for f in final_selected_features}

# Friendly labels
_AUTO_LABELS = {
    "Precio_Cobre":    "Copper (USD/lb)",
    "Dolar_index":     "USD Index (DXY)",
    "VIX":             "VIX",
    "Chile_CDS":       "Chile CDS (bp)",
    "USA_Swap_Inf_1Y": "US 1Y Inf. Swap (%)",
    "USA_Swap_Inf_2Y": "US 2Y Inf. Swap (%)",
    "Chile_Bond_1Y":   "Chile 1Y Bond (%)",
    "Chile_Bond_10Y":  "Chile 10Y Bond (%)",
}
LABELS = {rc: _AUTO_LABELS.get(rc, rc) for rc in raw_col_list}

# Variable descriptions for the info panel
_VAR_DESC = {
    "Precio_Cobre":    "Global copper spot price (USD/lb). Key driver of Chilean export revenue — Chile is the world's largest copper producer. **Higher copper → stronger CLP** (lower USD/CLP). Applied as monthly log-return.",
    "Dolar_index":     "DXY — ICE broad measure of USD strength against a basket of major currencies. **Higher DXY → weaker CLP** (higher USD/CLP). Applied as monthly log-return.",
    "VIX":             "CBOE Volatility Index, market's expectation of 30-day S&P 500 volatility. Proxy for global risk aversion. **Higher VIX → risk-off, EM capital outflows → weaker CLP**. Applied as monthly log-return.",
    "Chile_CDS":       "Chile 5-year Credit Default Swap spread (basis points). Measures sovereign credit risk and country-specific uncertainty. **Higher CDS → higher perceived country risk → weaker CLP**. Applied as monthly difference.",
    "USA_Swap_Inf_1Y": "US 1-year inflation swap rate. Reflects market's short-term US inflation expectations. **Higher → Fed tightening expectations → stronger USD → weaker CLP**. Applied as monthly difference.",
    "USA_Swap_Inf_2Y": "US 2-year inflation swap rate. Reflects market's medium-term US inflation expectations and Fed rate-path pricing. **Higher → tighter Fed expectations → stronger USD → weaker CLP**. Applied as monthly difference.",
    "Chile_Bond_1Y":   "Chilean 1-year central government bond yield. Higher short-term local rates can attract foreign capital inflows, supporting the CLP. Applied as monthly difference.",
    "Chile_Bond_10Y":  "Chilean 10-year central government bond yield. Reflects long-term country risk premium, inflation outlook, and monetary conditions. Applied as monthly difference.",
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def compute_features(prev_raw: dict, curr_raw: dict) -> np.ndarray:
    row = {
        feat: (np.log(curr_raw[rc] / prev_raw[rc]) if feat.startswith("logret_") else curr_raw[rc] - prev_raw[rc])
        for feat, rc in _raw_map.items()
    }
    return np.array([[row[f] for f in final_selected_features]])


def run_forecast(scenario_values: list) -> list:
    fx_path, prev_raw, prev_fx = [], last_raw.copy(), last_fx
    for curr_raw in scenario_values:
        log_ret = model.predict(compute_features(prev_raw, curr_raw))[0]
        fx = prev_fx * np.exp(log_ret)
        fx_path.append(fx)
        prev_raw, prev_fx = curr_raw, fx
    return fx_path


def _slider_params(rc: str):
    s = hist_data[rc].dropna()
    mu, sd = s.mean(), s.std()
    lo = min(float(s.min()), mu - 4 * sd)
    hi = max(float(s.max()), mu + 4 * sd)
    raw_step = (hi - lo) / 500 if (hi - lo) > 0 else 0.01
    mag = 10 ** np.floor(np.log10(abs(raw_step))) if raw_step > 0 else 0.01
    step = max(round(raw_step / mag) * mag, 1e-6)
    fmt = "%.0f" if hi > 50 else ("%.2f" if hi > 1 else "%.4f")
    return float(lo), float(hi), float(step), fmt


# ── Pre-compute slider config ─────────────────────────────────────────────────
slider_cfg = {rc: _slider_params(rc) for rc in raw_col_list}

# ── Session-state defaults ────────────────────────────────────────────────────
for rc in raw_col_list:
    for m in range(N_MONTHS):
        key = f"sv_{rc}_{m}"
        if key not in st.session_state:
            st.session_state[key] = float(last_raw[rc])

# ── Future month labels ───────────────────────────────────────────────────────
future_months = [last_date + pd.DateOffset(months=m) for m in range(1, N_MONTHS + 1)]
month_labels  = [d.strftime("%b %Y") for d in future_months]


# ═══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.title("CLP/USD Forecast — Scenario Analysis")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Anchor FX",   f"{last_fx:.2f} CLP/USD")
c2.metric("Anchor date", str(last_date.date()))
c3.metric("Test R²",     f"{metrics['test_r2']:.4f}")
c4.metric("Test RMSE",   f"{metrics['test_rmse_fx']:.2f} CLP/USD")

st.divider()


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL DESCRIPTION
# ═══════════════════════════════════════════════════════════════════════════════
with st.expander("📖  About the Model", expanded=True):
    st.markdown("""
### CLP/USD Monthly Log-Return Model

This app uses an **ensemble model (Voting Regressor: Random Forest + Ridge, weights 1:3)**
trained on **monthly data** to forecast the Chilean Peso / US Dollar exchange rate.

**Methodology**
- **Target variable**: Monthly log-return of CLP/USD — $\\log(FX_t / FX_{t-1})$
- **Prediction equation**: $FX_t = FX_{t-1} \\times e^{\\hat{y}_t}$
- **Ensemble**: Ridge regression (75% weight) + Random Forest (25% weight), blended via soft voting on standardized log-return features
- **Feature engineering**: Price/index series → monthly log-return $\\Delta\\log(X)$; Rate/spread series → monthly difference $\\Delta X = X_t - X_{t-1}$
- **Scenario usage**: Enter the **raw level** of each variable for each future month. The app computes the correct transformation automatically.
- **Training**: Time-series cross-validation (expanding window); hold-out test set reserved for final evaluation.
""")

    st.markdown("---")
    st.markdown("**Input Variables**")

    # Display variable cards in a 2-column grid
    desc_cols = st.columns(2)
    for i, rc in enumerate(raw_col_list):
        feat = next(f for f in final_selected_features if _raw_map[f] == rc)
        transform_tag = "log-return" if feat.startswith("logret_") else "monthly diff"
        s = hist_data[rc].dropna()
        mu, sd = s.mean(), s.std()
        with desc_cols[i % 2]:
            st.markdown(
                f"**{LABELS[rc]}** &nbsp; `{transform_tag}`  \n"
                f"*Current:* `{last_raw[rc]:.3g}` &nbsp;|&nbsp; "
                f"*Hist. avg:* `{mu:.3g}` &nbsp;|&nbsp; "
                f"*±1σ:* `[{mu - sd:.3g}, {mu + sd:.3g}]`  \n"
                f"{_VAR_DESC.get(rc, '')}"
            )

st.divider()


# ═══════════════════════════════════════════════════════════════════════════════
#  SCENARIO EDITOR  —  tabs × variable
#  Layout: chart (full width) → sliders in 2 columns of 6  (matches notebook)
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("Scenario Editor")
st.caption("Select a variable, set your 12-month path with the sliders, and the forecast updates instantly.")

cutoff_chart = last_date - pd.DateOffset(years=LOOKBACK_HIST)
tabs = st.tabs([LABELS[rc] for rc in raw_col_list])

for tab, rc in zip(tabs, raw_col_list):
    with tab:
        lo, hi, step, fmt = slider_cfg[rc]

        # ── Chart (full width) ────────────────────────────────────────────────
        hist_win = hist_data[hist_data.index >= cutoff_chart][rc].dropna()
        mu  = hist_data[rc].mean()
        sd  = hist_data[rc].std()

        # Read current slider values to draw scenario path
        month_vals = [st.session_state[f"sv_{rc}_{m}"] for m in range(N_MONTHS)]

        fig = go.Figure()

        # ±1σ band (filled area)
        fig.add_trace(go.Scatter(
            x=list(hist_win.index) + list(hist_win.index[::-1]),
            y=([mu + sd] * len(hist_win)) + ([mu - sd] * len(hist_win)),
            fill='toself',
            fillcolor='rgba(100, 149, 237, 0.13)',
            line=dict(color='rgba(0,0,0,0)'),
            name='±1σ band',
            hoverinfo='skip',
        ))

        # Mean line
        fig.add_trace(go.Scatter(
            x=hist_win.index, y=[mu] * len(hist_win),
            mode='lines', line=dict(color='gray', dash='dot', width=1),
            name='Mean', hoverinfo='skip',
        ))

        # Historical line
        fig.add_trace(go.Scatter(
            x=hist_win.index, y=hist_win.values,
            mode='lines', name='Historical',
            line=dict(color='steelblue', width=2.5),
            hovertemplate='%{x|%b %Y}: %{y:.3f}<extra></extra>',
        ))

        # Anchor marker
        fig.add_trace(go.Scatter(
            x=[last_date], y=[last_raw[rc]],
            mode='markers', name='Anchor',
            marker=dict(symbol='diamond', size=12, color='crimson',
                        line=dict(color='darkred', width=1)),
            hovertemplate='Anchor %{x|%b %Y}: %{y:.3f}<extra></extra>',
        ))

        # Scenario path (orange dots — same as notebook)
        fig.add_trace(go.Scatter(
            x=future_months, y=month_vals,
            mode='lines+markers', name='Scenario',
            line=dict(color='darkorange', width=2.5, dash='dot'),
            marker=dict(size=10, color='darkorange'),
            hovertemplate='%{x|%b %Y}: %{y:.3f}<extra></extra>',
        ))

        fig.update_layout(
            height=340,
            margin=dict(t=15, b=15, l=10, r=10),
            legend=dict(orientation='h', y=1.07, x=0),
            yaxis_title=LABELS[rc],
            xaxis_title='',
            hovermode='x unified',
            template='plotly_white',
        )

        st.plotly_chart(fig, use_container_width=True)

        # ── Reset button ──────────────────────────────────────────────────────
        btn_cols = st.columns([1, 4])
        with btn_cols[0]:
            if st.button("↺  Reset to flat", key=f"reset_{rc}", use_container_width=True):
                for m in range(N_MONTHS):
                    st.session_state[f"sv_{rc}_{m}"] = float(last_raw[rc])
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Sliders: 2 columns × 6 rows  (mirrors notebook HBox layout) ──────
        col_left, col_right = st.columns(2, gap="large")

        with col_left:
            st.markdown("**Months 1 – 6**")
            for m_idx in range(6):
                st.slider(
                    label     = month_labels[m_idx],
                    min_value = lo,
                    max_value = hi,
                    step      = step,
                    format    = fmt,
                    key       = f"sv_{rc}_{m_idx}",
                )

        with col_right:
            st.markdown("**Months 7 – 12**")
            for m_idx in range(6, 12):
                st.slider(
                    label     = month_labels[m_idx],
                    min_value = lo,
                    max_value = hi,
                    step      = step,
                    format    = fmt,
                    key       = f"sv_{rc}_{m_idx}",
                )


# ═══════════════════════════════════════════════════════════════════════════════
#  BUILD SCENARIO + RUN FORECAST
# ═══════════════════════════════════════════════════════════════════════════════
scenario_values = [
    {rc: st.session_state[f"sv_{rc}_{m}"] for rc in raw_col_list}
    for m in range(N_MONTHS)
]
fx_forecast = run_forecast(scenario_values)

st.divider()


# ═══════════════════════════════════════════════════════════════════════════════
#  12-MONTH CLP/USD FORECAST
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("12-Month CLP/USD Forecast")

fig_fx = go.Figure()

fig_fx.add_trace(go.Scatter(
    x=[last_date], y=[last_fx],
    mode='markers', name='Anchor',
    marker=dict(symbol='diamond', size=13, color='crimson',
                line=dict(color='darkred', width=1.5)),
    hovertemplate=f'Anchor {last_date.strftime("%b %Y")}: {{y:.2f}} CLP/USD<extra></extra>',
))

fig_fx.add_trace(go.Scatter(
    x=future_months, y=fx_forecast,
    mode='lines+markers', name='Forecast',
    line=dict(color='darkorange', width=3, dash='dot'),
    marker=dict(size=9, color='darkorange', line=dict(color='saddlebrown', width=1.5)),
    hovertemplate='%{x|%b %Y}: %{y:.2f} CLP/USD<extra></extra>',
))

fig_fx.update_layout(
    height=380,
    yaxis_title='CLP/USD',
    xaxis_title='',
    legend=dict(orientation='h', y=1.07),
    margin=dict(t=15, b=20, l=10, r=10),
    hovermode='x unified',
    template='plotly_white',
)

st.plotly_chart(fig_fx, use_container_width=True)

# Forecast table
df_table = pd.DataFrame({
    "Month"       : month_labels,
    "CLP/USD"     : [f"{v:.2f}" for v in fx_forecast],
    "Δ vs anchor" : [f"{v - last_fx:+.2f}  ({(v / last_fx - 1) * 100:+.1f}%)" for v in fx_forecast],
})
st.dataframe(df_table, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  HISTORICAL REFERENCE  (collapsible — last 5 years, 2×3 grid)
# ═══════════════════════════════════════════════════════════════════════════════
with st.expander("📊  Historical Reference — last 5 years", expanded=False):
    cutoff_ref = last_date - pd.DateOffset(years=LOOKBACK_REF)
    hw5 = hist_data[hist_data.index >= cutoff_ref]

    _COLORS = ['steelblue', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c']

    fig_ref = make_subplots(
        rows=2, cols=3,
        subplot_titles=[LABELS[rc] for rc in raw_col_list],
        vertical_spacing=0.18,
        horizontal_spacing=0.08,
    )
    for i, rc in enumerate(raw_col_list):
        r, c_idx = divmod(i, 3)
        s  = hw5[rc].dropna()
        mu = hist_data[rc].mean()
        sd = hist_data[rc].std()

        # ±1σ filled band
        fig_ref.add_trace(go.Scatter(
            x=list(s.index) + list(s.index[::-1]),
            y=([mu + sd] * len(s)) + ([mu - sd] * len(s)),
            fill='toself', fillcolor='rgba(100,149,237,0.13)',
            line=dict(color='rgba(0,0,0,0)'),
            name='±1σ', showlegend=(i == 0), legendgroup='band',
            hoverinfo='skip'),
            row=r + 1, col=c_idx + 1)

        # Mean
        fig_ref.add_trace(go.Scatter(
            x=s.index, y=[mu] * len(s),
            mode='lines', line=dict(color='gray', dash='dot', width=1),
            name='Mean', showlegend=(i == 0), legendgroup='mean',
            hoverinfo='skip'),
            row=r + 1, col=c_idx + 1)

        # Historical line
        fig_ref.add_trace(go.Scatter(
            x=s.index, y=s.values, mode='lines',
            line=dict(color=_COLORS[i % len(_COLORS)], width=1.8),
            showlegend=False,
            hovertemplate='%{x|%b %Y}: %{y:.3g}<extra></extra>'),
            row=r + 1, col=c_idx + 1)

        # Anchor
        fig_ref.add_trace(go.Scatter(
            x=[last_date], y=[last_raw[rc]], mode='markers',
            marker=dict(symbol='diamond', size=9, color='crimson'),
            name='Anchor', showlegend=(i == 0), legendgroup='anchor'),
            row=r + 1, col=c_idx + 1)

    fig_ref.update_layout(
        height=700,
        margin=dict(t=50, b=10),
        template='plotly_white',
        legend=dict(orientation='h', y=1.01, x=1, xanchor='right'),
    )
    st.plotly_chart(fig_ref, use_container_width=True)


st.caption(
    f"Model: {art.get('framework', 'log-returns')}  ·  "
    f"Features: {', '.join(final_selected_features)}"
)
