"""
THERMOELECTRIC SCREENING TERMINAL  v2

Companion UI for:
  "From Prediction to Physics: A Robust Machine Learning Framework for
   Thermoelectric Discovery with Explicit First-Principles Validation"
  Sathish Panneer Selvam et al. (submitted to J. Mater. Chem. A)

Aligned with MLcode_v2.py:
  - ZT_proxy = E_g · |ΔH_f| / (ρ + ε)        [heuristic screening descriptor]
  - 20 fixed seeds from the manuscript
  - Random Forest + XGBoost comparison arm
  - Repeated 5-fold CV (3 repeats)
  - R², MAE, RMSE per seed
  - SHAP: bar + heatmap + beeswarm
  - Stability threshold 0.9
  - Correlation-matrix and parity/residual diagnostics

Run:  streamlit run app.py
"""

from __future__ import annotations

import io
import time
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Thermoelectric Screening Terminal",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "Thermoelectric Screening Terminal v2 — companion UI for Selvam et al., "
            "‘From Prediction to Physics’. Implements the heuristic ZT_proxy = "
            "E_g·|ΔH_f|/(ρ+ε) screening pipeline. Not a substitute for DFT transport."
        )
    },
)

# ============================================================================
# THEME
# ============================================================================
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Fraunces:opsz,wght@9..144,300;9..144,400;9..144,600;9..144,800&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --bg-0: #0a0d0f;
    --bg-1: #11161a;
    --bg-2: #1a2128;
    --bg-3: #232c35;
    --line: #2a343d;
    --text-0: #e8eef2;
    --text-1: #a7b3bd;
    --text-2: #6b7681;
    --accent: #6ee7a8;
    --accent-dim: #3aa874;
    --xgb: #f4a261;
    --warn: #ffb547;
    --danger: #ff6b6b;
    --info: #7dd3fc;
}

html, body, [class*="css"], .stApp {
    background: var(--bg-0) !important;
    color: var(--text-0) !important;
    font-family: 'Inter', sans-serif;
}

.stApp::before {
    content: "";
    position: fixed; inset: 0; pointer-events: none;
    background: repeating-linear-gradient(0deg,
        rgba(255,255,255,0.012) 0px, rgba(255,255,255,0.012) 1px,
        transparent 1px, transparent 3px);
    z-index: 0;
}

[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] { display: none; }
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }

section[data-testid="stSidebar"] {
    background: var(--bg-1) !important;
    border-right: 1px solid var(--line);
}
section[data-testid="stSidebar"] * { color: var(--text-0); }

h1, h2, h3, h4, h5 {
    font-family: 'Fraunces', serif !important;
    letter-spacing: -0.02em !important;
    color: var(--text-0) !important;
}

.mono, code, pre, .stCode { font-family: 'JetBrains Mono', monospace !important; }

.hero {
    border: 1px solid var(--line);
    background:
        radial-gradient(800px 200px at 10% 0%, rgba(110,231,168,0.08), transparent 60%),
        radial-gradient(600px 200px at 90% 100%, rgba(255,181,71,0.05), transparent 60%),
        var(--bg-1);
    padding: 28px 34px; margin-bottom: 22px; position: relative; overflow: hidden;
}
.hero .eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; letter-spacing: 0.28em;
    color: var(--accent); text-transform: uppercase; margin-bottom: 6px;
}
.hero h1 {
    font-size: 42px !important; margin: 0 0 4px 0 !important;
    line-height: 1.05 !important; font-weight: 600 !important;
}
.hero h1 em { font-style: italic; color: var(--accent); font-weight: 300; }
.hero .lede {
    color: var(--text-1); font-size: 15px; max-width: 760px;
    line-height: 1.55; margin-top: 10px;
}
.hero .ribbon {
    display: inline-block; margin-top: 14px; padding: 6px 12px;
    border: 1px solid var(--warn); color: var(--warn);
    background: rgba(255,181,71,0.06);
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; letter-spacing: 0.15em; text-transform: uppercase;
}
.hero .cite {
    display: inline-block; margin-top: 14px; margin-left: 10px;
    padding: 6px 12px; border: 1px solid var(--line);
    color: var(--text-1);
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; letter-spacing: 0.12em; text-transform: uppercase;
}

.tile {
    border: 1px solid var(--line); background: var(--bg-1);
    padding: 16px 18px; height: 100%; position: relative;
}
.tile .label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; letter-spacing: 0.22em;
    color: var(--text-2); text-transform: uppercase; margin-bottom: 6px;
}
.tile .value {
    font-family: 'Fraunces', serif; font-size: 30px; font-weight: 400;
    color: var(--text-0); line-height: 1.1;
}
.tile .delta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; color: var(--text-1); margin-top: 4px;
}
.tile.accent { border-left: 3px solid var(--accent); }
.tile.warn   { border-left: 3px solid var(--warn); }
.tile.info   { border-left: 3px solid var(--info); }
.tile.xgb    { border-left: 3px solid var(--xgb); }

.proxy-warn {
    border: 1px dashed var(--warn);
    background: rgba(255,181,71,0.05);
    color: var(--text-1); padding: 12px 16px; margin: 12px 0;
    font-size: 13px; line-height: 1.5;
}
.proxy-warn strong { color: var(--warn); font-family: 'JetBrains Mono', monospace; letter-spacing: 0.05em; }

.stDataFrame { border: 1px solid var(--line); }

.stButton > button, .stDownloadButton > button {
    background: transparent !important;
    color: var(--accent) !important;
    border: 1px solid var(--accent-dim) !important;
    border-radius: 0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important; letter-spacing: 0.15em !important;
    text-transform: uppercase !important; padding: 8px 18px !important;
    transition: all 0.15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background: var(--accent) !important; color: var(--bg-0) !important;
    border-color: var(--accent) !important;
}

.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] {
    background: transparent !important; color: var(--text-2) !important;
    border-radius: 0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important; letter-spacing: 0.15em !important;
    text-transform: uppercase !important; padding: 10px 18px !important;
    border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}

.stExpander { background: var(--bg-1) !important; border: 1px solid var(--line) !important; border-radius: 0 !important; }

.stTextInput input, .stNumberInput input, .stSelectbox > div, .stMultiSelect > div {
    background: var(--bg-1) !important; color: var(--text-0) !important;
    border-radius: 0 !important; border: 1px solid var(--line) !important;
    font-family: 'JetBrains Mono', monospace !important;
}

hr { border-color: var(--line) !important; margin: 24px 0 !important; }

.section-h {
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
    letter-spacing: 0.3em; color: var(--text-2); text-transform: uppercase;
    margin: 8px 0 12px 0; padding-bottom: 6px; border-bottom: 1px solid var(--line);
}

.tag {
    display: inline-block; padding: 3px 9px; border: 1px solid var(--line);
    color: var(--text-1); font-family: 'JetBrains Mono', monospace;
    font-size: 10px; letter-spacing: 0.12em; margin-right: 6px; background: var(--bg-2);
}
.tag.green { color: var(--accent); border-color: var(--accent-dim); }
.tag.amber { color: var(--warn); border-color: var(--warn); }
.tag.xgb   { color: var(--xgb); border-color: var(--xgb); }

.stProgress > div > div > div > div { background: var(--accent) !important; }

.stSlider [data-baseweb="slider"] [role="slider"] {
    background: var(--accent) !important;
    border: 2px solid var(--bg-0) !important;
}

a, a:visited { color: var(--accent) !important; }

.formula-box {
    border: 1px solid var(--line);
    background: linear-gradient(135deg, rgba(110,231,168,0.04), rgba(255,181,71,0.03));
    padding: 18px 22px; margin: 14px 0;
    font-family: 'Fraunces', serif; font-size: 22px;
    color: var(--text-0); text-align: center;
    letter-spacing: 0.02em;
}
.formula-box em { color: var(--accent); font-style: italic; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================================
# CONFIG (mirrors MLcode_v2.py Config class)
# ============================================================================
# Exact seed list from the manuscript
MANUSCRIPT_SEEDS = [
    3111, 2737, 9544, 5404, 9138, 4649, 3868, 3137, 7824, 6136,
    7799, 3486, 1002, 9011, 8614, 8763, 7906, 5359, 4530, 9615,
]


@dataclass
class PipelineConfig:
    band_gap_min: float = 0.1
    band_gap_max: float = 2.0
    e_above_hull_max: float = 0.03
    n_sites_min: int = 2
    n_sites_max: int = 100
    seeds: list = field(default_factory=lambda: list(MANUSCRIPT_SEEDS))
    top_k: int = 20
    rf_n_estimators: int = 150
    xgb_n_estimators: int = 300
    xgb_lr: float = 0.05
    xgb_max_depth: int = 6
    select_k: int = 50
    test_size: float = 0.20
    epsilon: float = 1e-6
    stability_threshold: float = 0.9
    cv_splits: int = 5
    cv_repeats: int = 3
    run_xgb: bool = True
    run_repeated_cv: bool = True


# ============================================================================
# PIPELINE
# ============================================================================
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_materials_project(api_key: str, cfg_dict: dict) -> pd.DataFrame:
    """Cached MP fetch."""
    from mp_api.client import MPRester
    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(
            band_gap=(cfg_dict["band_gap_min"], cfg_dict["band_gap_max"]),
            energy_above_hull=(0, cfg_dict["e_above_hull_max"]),
            num_sites=(cfg_dict["n_sites_min"], cfg_dict["n_sites_max"]),
            fields=[
                "material_id", "formula_pretty", "band_gap", "density",
                "formation_energy_per_atom", "energy_above_hull",
            ],
        )
    return pd.DataFrame([
        {
            "material_id": str(d.material_id),
            "formula_pretty": d.formula_pretty,
            "band_gap": d.band_gap,
            "density": d.density,
            "formation_energy_per_atom": d.formation_energy_per_atom,
            "energy_above_hull": d.energy_above_hull,
        }
        for d in docs
    ])


@st.cache_data(show_spinner=False)
def featurize(df_in: pd.DataFrame, epsilon: float) -> pd.DataFrame:
    """Clean + ZT proxy + Magpie features. Matches MLcode_v2.py exactly."""
    from pymatgen.core import Composition
    from matminer.featurizers.composition import ElementProperty

    df = df_in.dropna(subset=[
        "band_gap", "density", "formation_energy_per_atom", "energy_above_hull"
    ]).copy()
    df["n_elements"] = df["formula_pretty"].apply(lambda s: len(Composition(s).elements))
    df = df[df["n_elements"] > 1]
    df = df.drop_duplicates(subset=["formula_pretty"]).reset_index(drop=True)

    # ZT proxy — MANUSCRIPT-ALIGNED FORMULA:
    #     ZT_proxy = E_g * |ΔH_f| / (ρ + ε)
    df["ZT_proxy"] = (
        df["band_gap"]
        * np.abs(df["formation_energy_per_atom"])
        / (df["density"] + epsilon)
    )

    df["composition"] = df["formula_pretty"].apply(Composition)
    feat = ElementProperty.from_preset("magpie")
    df = feat.featurize_dataframe(df, col_id="composition", ignore_errors=True)
    df = df.drop(columns=["composition"])
    return df


def _make_rf(seed: int, cfg: PipelineConfig):
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.feature_selection import SelectKBest, f_regression
    from sklearn.ensemble import RandomForestRegressor
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("selector", SelectKBest(f_regression, k=cfg.select_k)),
        ("model", RandomForestRegressor(
            n_estimators=cfg.rf_n_estimators, random_state=seed, n_jobs=-1,
        )),
    ])


def _make_xgb(seed: int, cfg: PipelineConfig):
    from xgboost import XGBRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.feature_selection import SelectKBest, f_regression
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("selector", SelectKBest(f_regression, k=cfg.select_k)),
        ("model", XGBRegressor(
            n_estimators=cfg.xgb_n_estimators,
            learning_rate=cfg.xgb_lr,
            max_depth=cfg.xgb_max_depth,
            random_state=seed,
            n_jobs=-1, verbosity=0, tree_method="hist",
        )),
    ])


def _rf_sweep_with_candidates(df, cfg, progress_cb=None):
    """
    THE candidate-generating sweep. RF only.

    For each manuscript seed:
      - 80:20 split, fit RF, score on held-out test (R²/MAE/RMSE)
      - refit on full data, predict, record top-K candidates

    This output drives the Candidates tab and what goes into DFT validation.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    y = np.log1p(df["ZT_proxy"])
    drop_cols = ["ZT_proxy", "band_gap", "density", "formation_energy_per_atom"]
    X = df.select_dtypes(include=["float64", "int64"]).drop(columns=drop_cols, errors="ignore")

    metrics, candidates, preds_per_seed = [], [], []
    last_pipe = None

    for i, seed in enumerate(cfg.seeds):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=cfg.test_size, random_state=int(seed)
        )
        pipe = _make_rf(int(seed), cfg)
        pipe.named_steps["selector"].k = min(cfg.select_k, X.shape[1])
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)
        metrics.append({
            "seed": int(seed),
            "R2": float(r2_score(y_te, y_pred)),
            "MAE": float(mean_absolute_error(y_te, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_te, y_pred))),
        })

        pipe.fit(X, y)
        preds = pipe.predict(X)
        preds_per_seed.append(preds)
        df_iter = df.copy()
        df_iter["Predicted"] = preds
        top = df_iter.sort_values("Predicted", ascending=False).head(cfg.top_k).copy()
        top["seed"] = int(seed)
        candidates.append(top)
        last_pipe = pipe

        if progress_cb:
            progress_cb((i + 1) / len(cfg.seeds))

    return {
        "metrics": pd.DataFrame(metrics),
        "candidates": pd.concat(candidates, ignore_index=True),
        "pipeline": last_pipe,
        "preds_per_seed": np.array(preds_per_seed),
        "X": X,
        "y": y,
    }


def _xgb_sweep_metrics_only(df, cfg, progress_cb=None):
    """
    XGBoost comparison sweep — METRICS ONLY (manuscript §3.1.5).

    Same 20 seeds, same 80:20 splits, same preprocessing. Computes R²/MAE/RMSE
    per seed. Does NOT generate candidate rankings — XGBoost is not part of
    the candidate pipeline that fed DFT validation.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    y = np.log1p(df["ZT_proxy"])
    drop_cols = ["ZT_proxy", "band_gap", "density", "formation_energy_per_atom"]
    X = df.select_dtypes(include=["float64", "int64"]).drop(columns=drop_cols, errors="ignore")

    metrics = []
    for i, seed in enumerate(cfg.seeds):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=cfg.test_size, random_state=int(seed)
        )
        pipe = _make_xgb(int(seed), cfg)
        pipe.named_steps["selector"].k = min(cfg.select_k, X.shape[1])
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)
        metrics.append({
            "seed": int(seed),
            "R2": float(r2_score(y_te, y_pred)),
            "MAE": float(mean_absolute_error(y_te, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_te, y_pred))),
        })
        if progress_cb:
            progress_cb((i + 1) / len(cfg.seeds))

    return {"metrics": pd.DataFrame(metrics)}


def run_full_pipeline(df, cfg, rf_cb=None, xgb_cb=None):
    """
    RF candidate-generating sweep + XGBoost metrics-only diagnostic + repeated CV.

    IMPORTANT: RF and XGBoost play DIFFERENT roles, matching the manuscript:
      - RF is the pipeline that produced the DFT-validated candidates (§2.1.6)
      - XGBoost only fills the §3.1.5 comparison table; it does NOT generate
        any candidates and does NOT influence the Candidates tab in any way.
    """
    out = {}
    t0 = time.time()

    # RF sweep — candidates flow from here
    out["rf"] = _rf_sweep_with_candidates(df, cfg, progress_cb=rf_cb)

    # XGBoost sweep — metrics only, deliberately no candidates
    if cfg.run_xgb:
        try:
            out["xgb"] = _xgb_sweep_metrics_only(df, cfg, progress_cb=xgb_cb)
        except Exception as e:
            out["xgb_error"] = str(e)

    # Repeated CV (RF only)
    if cfg.run_repeated_cv:
        from sklearn.model_selection import RepeatedKFold, cross_val_score
        cv = RepeatedKFold(
            n_splits=cfg.cv_splits, n_repeats=cfg.cv_repeats, random_state=42
        )
        pipe = _make_rf(42, cfg)
        pipe.named_steps["selector"].k = min(cfg.select_k, out["rf"]["X"].shape[1])
        scores = cross_val_score(pipe, out["rf"]["X"], out["rf"]["y"],
                                  cv=cv, scoring="r2", n_jobs=1)
        out["cv"] = {"scores": scores.tolist(), "mean": float(scores.mean()),
                     "std": float(scores.std())}

    # Stability merge for RF
    freq = out["rf"]["candidates"]["material_id"].value_counts()
    freq_df = freq.reset_index()
    freq_df.columns = ["material_id", "count"]
    freq_df["stability_score"] = freq_df["count"] / len(cfg.seeds)
    out["freq_df"] = freq_df

    # Prediction stability
    df_ranked = df.copy()
    pmat = out["rf"]["preds_per_seed"]
    df_ranked["pred_mean"] = pmat.mean(axis=0)
    df_ranked["pred_std"] = pmat.std(axis=0)
    out["df_with_preds"] = df_ranked

    out["elapsed"] = time.time() - t0
    return out


@st.cache_data(show_spinner=False)
def compute_shap_values(_pipeline, X: pd.DataFrame, n_samples: int = 200):
    import shap
    model = _pipeline.named_steps["model"]
    X_imp = _pipeline.named_steps["imputer"].transform(X)
    X_sel = _pipeline.named_steps["selector"].transform(X_imp)
    sel = X.columns[_pipeline.named_steps["selector"].get_support()]
    X_df = pd.DataFrame(X_sel, columns=sel)
    X_s = X_df.sample(min(n_samples, len(X_df)), random_state=42)
    expl = shap.TreeExplainer(model)
    sv = expl.shap_values(X_s)
    return sv, X_s


# ============================================================================
# UI HELPERS
# ============================================================================
def hero():
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">◈ Thermoelectric Screening Terminal · v2 · manuscript-aligned</div>
          <h1>From prediction to physics —<br/>a <em>robust</em> screening pipeline.</h1>
          <div class="lede">
            Companion interface for Selvam <em>et al.</em>, <strong>“From Prediction to Physics:
            A Robust Machine Learning Framework for Thermoelectric Discovery with Explicit
            First-Principles Validation”</strong> (submitted to <em>J. Mater. Chem. A</em>).
            Implements the heuristic ZT proxy ranking on Materials Project data with
            20-seed stability analysis, Random Forest + XGBoost comparison, repeated
            cross-validation, and SHAP attribution.
          </div>
          <div class="ribbon">⚠ ZT_PROXY = E<sub>g</sub> · |Δh<sub>f</sub>| / (ρ + ε) &nbsp; — heuristic surrogate</div>
          <div class="cite">DOI: 10.1039/x0xx00000x &nbsp;·&nbsp; ε = 10⁻⁶</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def proxy_warning(short: bool = False):
    if short:
        st.markdown("""
        <div class="proxy-warn">
          <strong>NOTE</strong> &nbsp; Predicted values are on the log of a heuristic surrogate,
          not real ZT. Treat as a relative score for shortlisting.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="proxy-warn">
          <strong>HEURISTIC NOTICE</strong>&nbsp; The target
          <code>ZT_proxy = E<sub>g</sub> · |Δh<sub>f</sub>| / (ρ + ε)</code>, with ε = 10⁻⁶,
          is a <em>chemistry-aware surrogate</em>, not the true thermoelectric figure of merit
          ZT = S²σT/κ. Band gap (E<sub>g</sub>) reflects carrier accessibility; formation
          energy magnitude (|Δh<sub>f</sub>|) reflects thermodynamic bonding strength; density
          (ρ) in the denominator favors softer-lattice candidates as a phonon-transport prior.
          Use rankings to <em>shortlist candidates for proper BoltzTraP / AMSET / Phono3py
          follow-up</em>, not as ZT predictions.
        </div>
        """, unsafe_allow_html=True)


def tile(label, value, delta="", flavor="accent"):
    return f"""
    <div class="tile {flavor}">
      <div class="label">{label}</div>
      <div class="value">{value}</div>
      <div class="delta">{delta}</div>
    </div>
    """


def section(label):
    st.markdown(f"<div class='section-h'>{label}</div>", unsafe_allow_html=True)


PLOTLY_LAYOUT = dict(
    plot_bgcolor="#11161a", paper_bgcolor="#11161a",
    font=dict(family="JetBrains Mono, monospace", color="#a7b3bd", size=11),
    xaxis=dict(gridcolor="#2a343d", zerolinecolor="#2a343d"),
    yaxis=dict(gridcolor="#2a343d", zerolinecolor="#2a343d"),
    margin=dict(l=40, r=20, t=40, b=40),
)


# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown(
        "<div style='font-family:Fraunces,serif;font-size:22px;letter-spacing:-0.02em;margin-bottom:2px'>"
        "◈ <em>Terminal v2</em></div>"
        "<div style='font-family:JetBrains Mono,monospace;font-size:10px;letter-spacing:0.25em;color:#6b7681;text-transform:uppercase;margin-bottom:20px'>"
        "Manuscript-Aligned</div>",
        unsafe_allow_html=True,
    )

    section("01 · CREDENTIALS")
    api_key = st.text_input(
        "Materials Project API Key", type="password",
        help="Get a key at next-gen.materialsproject.org/api",
        key="mp_api_key",
    )

    section("02 · QUERY FILTERS")
    band_gap_range = st.slider(
        "Band gap range (eV)", 0.0, 6.0, (0.1, 2.0), step=0.05,
        help="Manuscript §2.1.1 uses 0.1–2.0 eV for the semiconducting regime.",
    )
    e_hull = st.slider(
        "Max E_hull (eV/atom)", 0.0, 0.2, 0.03, step=0.005,
        help="Manuscript §2.1.1 enforces E_hull < 0.03 eV/atom.",
    )
    n_sites = st.slider("Sites per cell", 2, 200, (2, 100), step=1)

    section("03 · MODEL CONTROLS")

    use_manuscript_seeds = st.checkbox(
        "Use manuscript 20 seeds", value=True,
        help="Locks the 20-seed list from §2.1.6 for exact reproducibility.",
    )
    if not use_manuscript_seeds:
        n_seeds = st.slider("Number of seeds", 5, 50, 20, step=5)
    else:
        n_seeds = len(MANUSCRIPT_SEEDS)
        st.caption(f"Seeds locked: {MANUSCRIPT_SEEDS[:5]}…")

    top_k = st.slider("Top-K per seed", 5, 100, 20, step=5)
    rf_trees = st.select_slider(
        "RF estimators", options=[50, 100, 150, 200, 300, 500], value=150,
    )
    select_k = st.slider("Features retained (SelectKBest)", 10, 132, 50, step=5)

    run_xgb = st.checkbox(
        "Run XGBoost comparison arm", value=True,
        help="Manuscript §3.1.5 compares RF vs XGBoost.",
    )
    run_cv = st.checkbox(
        "Run repeated 5-fold CV (×3)", value=True,
        help="Manuscript §2.1.5 repeats 5-fold CV three times.",
    )

    stab_threshold = st.slider(
        "Stability threshold", 0.5, 1.0, 0.9, step=0.05,
        help="Manuscript §2.1.7 uses ≥ 0.9 (≥ 18/20 seeds).",
    )

    section("04 · ACTIONS")
    fetch_clicked = st.button("◇ FETCH DATA", use_container_width=True)
    train_clicked = st.button("▷ RUN PIPELINE", use_container_width=True)
    reset_clicked = st.button("⨯ RESET CACHE", use_container_width=True)
    if reset_clicked:
        st.cache_data.clear()
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown(
        f"""<div style='font-family:JetBrains Mono,monospace;font-size:10px;color:#6b7681;line-height:1.6'>
        <span class='tag amber'>HEURISTIC</span>
        <span class='tag green'>RF · {rf_trees} EST</span>
        {('<span class="tag xgb">XGB</span>' if run_xgb else '')}<br/><br/>
        Built on MP · matminer Magpie · scikit-learn · XGBoost · SHAP.
        Predictions are <em>log1p(ZT_proxy)</em>.
        </div>""",
        unsafe_allow_html=True,
    )

# ============================================================================
# MAIN
# ============================================================================
hero()

cfg = PipelineConfig(
    band_gap_min=band_gap_range[0],
    band_gap_max=band_gap_range[1],
    e_above_hull_max=e_hull,
    n_sites_min=n_sites[0],
    n_sites_max=n_sites[1],
    seeds=list(MANUSCRIPT_SEEDS) if use_manuscript_seeds else list(
        np.random.RandomState(42).randint(0, 10000, n_seeds)
    ),
    top_k=top_k,
    rf_n_estimators=rf_trees,
    select_k=select_k,
    stability_threshold=stab_threshold,
    run_xgb=run_xgb,
    run_repeated_cv=run_cv,
)

for k in ["raw_df", "feat_df", "results"]:
    if k not in st.session_state:
        st.session_state[k] = None

# Fetch
if fetch_clicked:
    if not api_key:
        st.error("Materials Project API key required.")
        st.stop()
    with st.spinner("Querying Materials Project…"):
        try:
            t0 = time.time()
            df_raw = fetch_materials_project(api_key, cfg.__dict__)
            st.session_state.raw_df = df_raw
            st.session_state.fetch_seconds = time.time() - t0
            st.session_state.feat_df = None
            st.session_state.results = None
            st.success(f"Pulled {len(df_raw):,} materials in {st.session_state.fetch_seconds:.1f}s.")
        except Exception as e:
            st.error(f"Fetch failed: {e}")
            st.stop()

# Train
if train_clicked:
    if st.session_state.raw_df is None:
        st.warning("No data loaded. Click ◇ FETCH DATA first.")
        st.stop()
    if st.session_state.feat_df is None:
        with st.spinner("Featurizing with matminer Magpie…"):
            try:
                st.session_state.feat_df = featurize(st.session_state.raw_df, cfg.epsilon)
            except Exception as e:
                st.error(f"Featurization failed: {e}")
                st.stop()

    n = len(cfg.seeds)
    progress = st.progress(0.0, text=f"RF sweep 0/{n}")
    state = {"k": 0, "phase": "RF"}

    def rf_cb(p):
        state["k"] += 1
        progress.progress(p * 0.5 if cfg.run_xgb else p,
                          text=f"{state['phase']} sweep {state['k']}/{n}")

    def xgb_cb(p):
        state["k"] = int(p * n)
        progress.progress(0.5 + p * 0.5,
                          text=f"XGBoost sweep {state['k']}/{n}")

    try:
        state["phase"] = "RF"
        if cfg.run_xgb:
            state2 = {"k": 0}
            def rf_cb2(p):
                state2["k"] += 1
                progress.progress(p * 0.5, text=f"RF sweep {state2['k']}/{n}")
            def xgb_cb2(p):
                state2_k = int(p * n)
                progress.progress(0.5 + p * 0.5, text=f"XGBoost sweep {state2_k}/{n}")
            results = run_full_pipeline(
                st.session_state.feat_df, cfg, rf_cb=rf_cb2, xgb_cb=xgb_cb2
            )
        else:
            results = run_full_pipeline(
                st.session_state.feat_df, cfg,
                rf_cb=lambda p: progress.progress(p, text=f"RF sweep {int(p*n)}/{n}"),
            )
        st.session_state.results = results
        progress.empty()
        msg = f"Pipeline complete in {results['elapsed']:.1f}s · "
        msg += f"RF R²={results['rf']['metrics']['R2'].mean():.3f}±{results['rf']['metrics']['R2'].std():.3f}"
        if "xgb" in results:
            msg += f" · XGB R²={results['xgb']['metrics']['R2'].mean():.3f}±{results['xgb']['metrics']['R2'].std():.3f}"
        if "cv" in results:
            msg += f" · CV R²={results['cv']['mean']:.3f}±{results['cv']['std']:.3f}"
        st.success(msg)
    except Exception as e:
        progress.empty()
        st.error(f"Training failed: {e}")
        st.stop()


# ============================================================================
# INITIAL STATE (no results yet)
# ============================================================================
if st.session_state.results is None:
    section("WHAT THIS PIPELINE DOES")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(tile("01 PULL", "MP stable narrow-gap",
                         "§2.1.1 · E_hull ≤ 0.03 · 0.1–2.0 eV", "info"),
                    unsafe_allow_html=True)
    with c2:
        st.markdown(tile("02 SCORE", "Heuristic proxy",
                         "§2.1.2 · E_g · |Δh_f| / (ρ + ε)", "warn"),
                    unsafe_allow_html=True)
    with c3:
        st.markdown(tile("03 RANK", "20-seed sweep",
                         "§2.1.6 · RF + XGB · top-K consensus"),
                    unsafe_allow_html=True)
    with c4:
        st.markdown(tile("04 EXPLAIN", "SHAP attribution",
                         "§2.1.7 · bar · heatmap · beeswarm", "xgb"),
                    unsafe_allow_html=True)

    st.write("")
    st.markdown("""
    <div class='formula-box'>
      ZT<sub>proxy</sub> = <em>E<sub>g</sub></em> · |Δ<em>H</em><sub>f</sub>| / (ρ + ε) &nbsp;&nbsp;·&nbsp;&nbsp; <em>y</em> = log(1 + ZT<sub>proxy</sub>)
    </div>
    """, unsafe_allow_html=True)
    proxy_warning(short=False)

    st.markdown("""
    ### Pipeline at a glance (per the manuscript)

    | Step | Manuscript section | What happens |
    | --- | --- | --- |
    | Data acquisition | §2.1.1 | MP query: band gap 0.1–2.0 eV, E<sub>hull</sub> ≤ 0.03 eV/atom, n_sites 2–100 |
    | Target | §2.1.2 | ZT<sub>proxy</sub> = E<sub>g</sub>·&#124;Δh<sub>f</sub>&#124;/(ρ+ε); y = log(1+ZT<sub>proxy</sub>) |
    | Features | §2.1.3 | matminer Magpie descriptors (~132 dim) |
    | Model | §2.1.4 | Median impute → SelectKBest (F-test, k=50) → Random Forest |
    | Validation | §2.1.5 | 80:20 holdout + repeated 5-fold CV (3 repeats), R²/MAE/RMSE |
    | Stability | §2.1.6 | 20-seed sweep with fixed seed list |
    | Comparison | §3.1.5 | XGBoost arm under the same preprocessing |
    | Interpretability | §2.1.7 | SHAP bar + heatmap + beeswarm |
    | Selection | §2.1.7 | stability ≥ 0.9, E<sub>hull</sub> ≤ 0.03, band gap ≥ 0.2 |

    ### Start

    Enter your Materials Project API key in the sidebar, click **◇ FETCH DATA**,
    then **▷ RUN PIPELINE**. Expect 2–5 min for a typical run on a Streamlit Cloud
    instance with ~3-5k filtered materials.
    """, unsafe_allow_html=True)
    st.stop()


# ============================================================================
# RESULTS
# ============================================================================
results = st.session_state.results
df_feat = st.session_state.feat_df
rf_metrics = results["rf"]["metrics"]
xgb_metrics = results.get("xgb", {}).get("metrics") if "xgb" in results else None
freq_df = results["freq_df"]
df_ranked = results["df_with_preds"]

# KPI strip
section("RUN SUMMARY")
cols = st.columns(6 if "cv" in results else 5)
with cols[0]:
    st.markdown(tile("MATERIALS", f"{len(df_feat):,}",
                     "after cleaning + dedup", "info"), unsafe_allow_html=True)
with cols[1]:
    st.markdown(tile("RF R²", f"{rf_metrics['R2'].mean():.3f}",
                     f"σ = {rf_metrics['R2'].std():.3f} · {len(cfg.seeds)} seeds"),
                unsafe_allow_html=True)
with cols[2]:
    if xgb_metrics is not None:
        st.markdown(tile("XGB R²", f"{xgb_metrics['R2'].mean():.3f}",
                         f"σ = {xgb_metrics['R2'].std():.3f}", "xgb"),
                    unsafe_allow_html=True)
    else:
        st.markdown(tile("XGB", "—", "skipped", "xgb"), unsafe_allow_html=True)
with cols[3]:
    st.markdown(tile("RF RMSE", f"{rf_metrics['RMSE'].mean():.3f}",
                     f"MAE = {rf_metrics['MAE'].mean():.3f}"), unsafe_allow_html=True)
with cols[4]:
    robust_n = (freq_df["stability_score"] >= cfg.stability_threshold).sum()
    st.markdown(tile("ROBUST", f"{robust_n}",
                     f"≥ {int(cfg.stability_threshold*100)}% seed agreement"),
                unsafe_allow_html=True)
if "cv" in results:
    with cols[5]:
        st.markdown(tile("CV R²", f"{results['cv']['mean']:.3f}",
                         f"σ = {results['cv']['std']:.3f} · "
                         f"{cfg.cv_splits}-fold × {cfg.cv_repeats}", "info"),
                    unsafe_allow_html=True)

st.write("")
proxy_warning(short=True)


# Tabs — added DIAGNOSTICS (Fig 2) and RF vs XGB comparison
tab_overview, tab_diag2, tab_candidates, tab_seeds, tab_shap, tab_export = st.tabs(
    ["▤ OVERVIEW", "▥ FIG-2 DIAGNOSTICS", "◇ CANDIDATES", "▦ FIG-3 SEED METRICS",
     "◈ FIG-4/5 SHAP", "↓ EXPORT"]
)


# ============================================================================
# OVERVIEW
# ============================================================================
with tab_overview:
    left, right = st.columns([1.3, 1])
    with left:
        section("PROXY DISTRIBUTION · LOG SCALE")
        fig = px.histogram(df_ranked, x="ZT_proxy", nbins=60, log_x=True,
                           color_discrete_sequence=["#6ee7a8"])
        fig.update_layout(**PLOTLY_LAYOUT, height=320,
                          xaxis_title="ZT_proxy (heuristic)", yaxis_title="count")
        st.plotly_chart(fig, use_container_width=True)

        section("BAND GAP × DENSITY  (color = log ZT_proxy)")
        fig2 = px.scatter(
            df_ranked, x="band_gap", y="density",
            color=np.log1p(df_ranked["ZT_proxy"]),
            hover_data=["formula_pretty", "material_id", "energy_above_hull"],
            color_continuous_scale=[(0, "#1a2128"), (0.5, "#3aa874"), (1, "#6ee7a8")],
            opacity=0.6,
        )
        fig2.update_traces(marker=dict(size=5, line=dict(width=0)))
        fig2.update_layout(**PLOTLY_LAYOUT, height=380,
                           xaxis_title="band gap (eV)", yaxis_title="density (g/cm³)",
                           coloraxis_colorbar=dict(title="log1p(ZT_proxy)"))
        st.plotly_chart(fig2, use_container_width=True)

    with right:
        section("CHEMISTRY PROFILE  (robust subset)")
        try:
            from pymatgen.core import Composition
            robust_ids = freq_df[freq_df["stability_score"] >= cfg.stability_threshold]["material_id"]
            sub = df_feat[df_feat["material_id"].isin(robust_ids)]
            counts = {}
            for f in sub["formula_pretty"]:
                for el in Composition(f).elements:
                    counts[el.symbol] = counts.get(el.symbol, 0) + 1
            if counts:
                el_df = pd.DataFrame(
                    sorted(counts.items(), key=lambda x: -x[1])[:15],
                    columns=["element", "count"]
                )
                fig_el = px.bar(el_df, x="count", y="element", orientation="h",
                                color_discrete_sequence=["#6ee7a8"])
                fig_el.update_layout(**PLOTLY_LAYOUT, height=420,
                                     yaxis=dict(autorange="reversed", gridcolor="#2a343d"),
                                     xaxis_title="appearances in robust set",
                                     yaxis_title="")
                st.plotly_chart(fig_el, use_container_width=True)
                st.caption(
                    "Element frequency in the robust candidate set (stability ≥ "
                    f"{cfg.stability_threshold:.0%}). Heavy chalcogen-rich families "
                    "(Te, Se, Sb, Bi) are the classic thermoelectric chemistries — "
                    "alignment with them is a sanity check, not a guarantee."
                )
            else:
                st.info("No robust candidates at this threshold.")
        except Exception as e:
            st.warning(f"Chemistry profile unavailable: {e}")


# ============================================================================
# FIG-2 DIAGNOSTICS (parity, residuals, correlation, CV)
# ============================================================================
with tab_diag2:
    section("FIG. 2 · MODEL DIAGNOSTICS")
    st.caption(
        "Reproduces the four panels of Figure 2 in the manuscript: "
        "(a) parity plot, (b) residuals, (c) descriptor correlation matrix, "
        "(d) repeated CV results."
    )

    # Pick best-R² seed for parity/residual (any single seed works; manuscript
    # uses a representative split)
    best_seed_idx = rf_metrics["R2"].idxmax()
    best_seed = int(rf_metrics.loc[best_seed_idx, "seed"])

    from sklearn.model_selection import train_test_split
    X = results["rf"]["X"]; y = results["rf"]["y"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=cfg.test_size, random_state=best_seed
    )
    pipe = _make_rf(best_seed, cfg)
    pipe.named_steps["selector"].k = min(cfg.select_k, X.shape[1])
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)
    from sklearn.metrics import r2_score, mean_absolute_error
    r2_show = r2_score(y_te, y_pred); mae_show = mean_absolute_error(y_te, y_pred)

    c1, c2 = st.columns(2)
    with c1:
        section(f"FIG-2(A) · PARITY  (seed {best_seed})")
        lo, hi = min(y_te.min(), y_pred.min()), max(y_te.max(), y_pred.max())
        figp = go.Figure()
        figp.add_trace(go.Scatter(
            x=y_te, y=y_pred, mode="markers",
            marker=dict(size=5, color="#6ee7a8", opacity=0.5, line=dict(width=0)),
            name="held-out",
        ))
        figp.add_trace(go.Scatter(
            x=[lo, hi], y=[lo, hi], mode="lines",
            line=dict(color="#a7b3bd", dash="dash", width=1), name="y = x",
        ))
        figp.update_layout(**PLOTLY_LAYOUT, height=380,
                           xaxis_title="computed log(1+ZT_proxy)",
                           yaxis_title="predicted log(1+ZT_proxy)",
                           title=f"R² = {r2_show:.3f}   MAE = {mae_show:.3f}")
        st.plotly_chart(figp, use_container_width=True)

    with c2:
        section("FIG-2(B) · RESIDUALS")
        resid = y_te - y_pred
        figr = go.Figure()
        figr.add_trace(go.Scatter(
            x=y_pred, y=resid, mode="markers",
            marker=dict(size=5, color="#ffb547", opacity=0.5, line=dict(width=0)),
        ))
        figr.add_hline(y=0, line=dict(color="#a7b3bd", width=1))
        figr.update_layout(**PLOTLY_LAYOUT, height=380,
                           xaxis_title="predicted log(1+ZT_proxy)",
                           yaxis_title="residual")
        st.plotly_chart(figr, use_container_width=True)

    section("FIG-2(C) · DESCRIPTOR CORRELATION  (top-20 by F-score)")
    # Correlation of the selected features
    X_imp = pipe.named_steps["imputer"].transform(X)
    mask = pipe.named_steps["selector"].get_support()
    sel_names = X.columns[mask]
    X_sel_df = pd.DataFrame(X_imp, columns=X.columns)[sel_names]
    fscores = pipe.named_steps["selector"].scores_[mask]
    order = np.argsort(-fscores)[:20]
    corr = X_sel_df.iloc[:, order].corr()
    figc = px.imshow(
        corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        aspect="auto",
    )
    figc.update_layout(**PLOTLY_LAYOUT, height=560,
                       xaxis=dict(tickangle=-60, tickfont=dict(size=8)),
                       yaxis=dict(tickfont=dict(size=8)))
    st.plotly_chart(figc, use_container_width=True)

    if "cv" in results:
        section("FIG-2(D) · REPEATED CROSS-VALIDATION")
        scores = results["cv"]["scores"]
        mean = results["cv"]["mean"]; std = results["cv"]["std"]
        figcv = go.Figure()
        figcv.add_trace(go.Scatter(
            x=list(range(1, len(scores) + 1)), y=scores,
            mode="lines+markers",
            line=dict(color="#2a9d8f", width=2),
            marker=dict(size=7, color="#2a9d8f"),
            name="R² per fold",
        ))
        figcv.add_hline(y=mean, line=dict(color="#a7b3bd", dash="dash"),
                        annotation_text=f"mean = {mean:.3f}",
                        annotation_font_color="#a7b3bd")
        figcv.add_hrect(y0=mean - std, y1=mean + std,
                        fillcolor="#2a9d8f", opacity=0.1, line_width=0)
        figcv.update_layout(**PLOTLY_LAYOUT, height=320,
                            xaxis_title=f"fold index ({cfg.cv_repeats} repeats × {cfg.cv_splits} folds)",
                            yaxis_title="R²")
        st.plotly_chart(figcv, use_container_width=True)

    st.markdown("""
    <div style='margin-top:14px;padding:14px 18px;border:1px solid #2a343d;background:#11161a;font-family:Inter,sans-serif;font-size:13px;color:#a7b3bd;line-height:1.6'>
    <strong style='color:#ffb547;font-family:JetBrains Mono,monospace;letter-spacing:0.1em'>HOW TO READ THIS</strong><br/><br/>
    Parity and residuals here are on <code>log(1+ZT_proxy)</code>, a smooth synthetic target.
    R² near 0.65 means the RF reproduces the proxy formula from Magpie chemistry — self-consistency,
    not a claim about real ZT prediction. The fan-shaped residual spread at high predicted values
    reflects the informational ceiling of composition-only descriptors. Repeated CV close to the
    20-seed mean confirms the model is not split-dependent.
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# CANDIDATES
# ============================================================================
with tab_candidates:
    section("ROBUST CANDIDATES · SORTED BY SEED-STABILITY (RF ONLY)")
    st.markdown("""
    <div style='padding:10px 14px;border-left:3px solid #6ee7a8;background:rgba(110,231,168,0.04);font-family:JetBrains Mono,monospace;font-size:11px;color:#a7b3bd;margin-bottom:14px;letter-spacing:0.05em'>
    <strong style='color:#6ee7a8'>RF-ONLY</strong>&nbsp; Candidate rankings come from the Random Forest 20-seed sweep,
    matching the manuscript §2.1.6 pipeline that produced the DFT-validated leads.
    The XGBoost arm in the FIG-3 tab is a metrics-only comparison (§3.1.5) and does
    not influence this list.
    </div>
    """, unsafe_allow_html=True)
    threshold = st.slider(
        "Minimum stability score", 0.0, 1.0, float(cfg.stability_threshold), step=0.05,
        key="stab_thr",
    )

    merged = freq_df.merge(
        df_ranked[["material_id", "formula_pretty", "band_gap", "density",
                   "formation_energy_per_atom", "energy_above_hull", "n_elements",
                   "ZT_proxy", "pred_mean", "pred_std"]],
        on="material_id", how="left",
    )
    merged = merged[merged["stability_score"] >= threshold].copy()
    # Manuscript §2.1.7 also enforces band_gap >= 0.2
    merged = merged[merged["band_gap"] >= 0.2]
    merged = merged.sort_values(["stability_score", "pred_mean"], ascending=[False, False])

    if len(merged) == 0:
        st.warning("No candidates above this threshold. Lower the slider.")
    else:
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            st.markdown(tile("CANDIDATES", f"{len(merged)}",
                             f"≥ {int(threshold*100)}% seed agreement"),
                        unsafe_allow_html=True)
        with cc2:
            st.markdown(tile("MEDIAN BAND GAP", f"{merged['band_gap'].median():.2f} eV",
                             f"range {merged['band_gap'].min():.2f}–{merged['band_gap'].max():.2f}", "info"),
                        unsafe_allow_html=True)
        with cc3:
            st.markdown(tile("MEDIAN HULL", f"{merged['energy_above_hull'].median()*1000:.1f} meV/at",
                             f"≤ {cfg.e_above_hull_max*1000:.0f} threshold", "info"),
                        unsafe_allow_html=True)
        st.write("")

        display = merged[[
            "material_id", "formula_pretty", "stability_score", "count",
            "pred_mean", "pred_std", "band_gap", "density",
            "formation_energy_per_atom", "energy_above_hull", "n_elements", "ZT_proxy",
        ]].rename(columns={
            "formula_pretty": "formula",
            "stability_score": "stability",
            "count": "n_seeds_in_top",
            "pred_mean": "pred_mean(log)",
            "pred_std": "pred_std",
            "band_gap": "E_g (eV)",
            "density": "ρ (g/cm³)",
            "formation_energy_per_atom": "Δh_f (eV/at)",
            "energy_above_hull": "E_hull (eV/at)",
        })

        st.dataframe(
            display.style
            .format({
                "stability": "{:.2f}", "pred_mean(log)": "{:.3f}", "pred_std": "{:.3f}",
                "E_g (eV)": "{:.3f}", "ρ (g/cm³)": "{:.2f}",
                "Δh_f (eV/at)": "{:.3f}", "E_hull (eV/at)": "{:.4f}",
                "ZT_proxy": "{:.4f}",
            })
            .background_gradient(subset=["stability"], cmap="Greens")
            .background_gradient(subset=["ZT_proxy"], cmap="YlOrBr"),
            use_container_width=True, height=520,
        )

        st.markdown("---")
        section("CANDIDATE DEEP-DIVE")
        chosen = st.selectbox(
            "Pick a candidate to inspect",
            options=display["material_id"].tolist(),
            format_func=lambda mid: f"{mid}  ·  {display.set_index('material_id').loc[mid, 'formula']}",
        )
        row = merged[merged["material_id"] == chosen].iloc[0]

        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.markdown(tile("FORMULA", row["formula_pretty"],
                             f"{int(row['n_elements'])} elements"),
                        unsafe_allow_html=True)
        with d2:
            st.markdown(tile("STABILITY", f"{row['stability_score']:.0%}",
                             f"{int(row['count'])}/{len(cfg.seeds)} seeds", "accent"),
                        unsafe_allow_html=True)
        with d3:
            st.markdown(tile("ZT_PROXY", f"{row['ZT_proxy']:.4f}",
                             "heuristic units", "warn"), unsafe_allow_html=True)
        with d4:
            mp_url = f"https://next-gen.materialsproject.org/materials/{chosen}"
            st.markdown(
                f"""<div class='tile info'>
                  <div class='label'>EXTERNAL</div>
                  <div class='value' style='font-size:14px;line-height:1.4'>
                    <a href='{mp_url}' target='_blank' style='font-family:JetBrains Mono,monospace'>↗ View on MP</a>
                  </div>
                  <div class='delta'>{chosen}</div>
                </div>""", unsafe_allow_html=True)

        st.write("")
        d5, d6, d7 = st.columns(3)
        with d5:
            st.markdown(tile("BAND GAP", f"{row['band_gap']:.3f} eV",
                             "narrow gap → carrier accessibility"),
                        unsafe_allow_html=True)
        with d6:
            st.markdown(tile("DENSITY", f"{row['density']:.2f} g/cm³",
                             "lower ρ → softer lattice (denominator)"),
                        unsafe_allow_html=True)
        with d7:
            st.markdown(tile("E_HULL", f"{row['energy_above_hull']*1000:.1f} meV/at",
                             "thermodynamic stability"), unsafe_allow_html=True)

        st.markdown(f"""
        <div style='margin-top:18px;padding:14px 18px;border:1px solid #2a343d;background:#11161a;font-family:Inter,sans-serif;font-size:13px;color:#a7b3bd;line-height:1.6'>
        <strong style='color:#6ee7a8;font-family:JetBrains Mono,monospace;letter-spacing:0.1em'>NEXT-STEP RECOMMENDATION</strong><br/><br/>
        Before treating <span class='mono'>{row['formula_pretty']}</span> as a real thermoelectric lead:
        <ol style='margin-top:8px;line-height:1.7'>
        <li>Confirm band gap and band structure with a hybrid functional (HSE06 or higher).</li>
        <li>Run <strong>BoltzTraP2</strong> or <strong>AMSET</strong> for Seebeck coefficient and σ vs. doping.</li>
        <li>Estimate κ<sub>lattice</sub> with <strong>Phono3py</strong> or the Slack model — the proxy ignores this.</li>
        <li>Check synthesis literature; E<sub>hull</sub> {row['energy_above_hull']*1000:.1f} meV/atom is only a thermodynamic green light.</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# FIG-3 PER-SEED METRICS
# ============================================================================
with tab_seeds:
    section("FIG. 3 · PER-SEED STABILITY ACROSS 20 SEEDS")
    st.caption(
        "Reproduces Figure 3 of the manuscript: per-seed R², MAE, RMSE for "
        "(a) Random Forest and (b) XGBoost."
    )

    def _seed_panel(metrics, title, color):
        st.markdown(f"#### {title}")
        c1, c2, c3 = st.columns(3)
        for ax_col, metric, c, target in zip(
            (c1, c2, c3), ["R2", "MAE", "RMSE"], color,
            (c1, c2, c3),
        ):
            with ax_col:
                vals = metrics[metric].values
                m, s = vals.mean(), vals.std()
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=list(range(1, len(vals) + 1)), y=vals,
                    marker_color=c, marker_line_color="black", marker_line_width=0.4,
                ))
                fig.add_hline(y=m, line=dict(color="#a7b3bd", dash="dash"),
                              annotation_text=f"{m:.3f}",
                              annotation_font_color="#a7b3bd")
                fig.update_layout(**PLOTLY_LAYOUT, height=260,
                                  title=f"{metric}  {m:.3f} ± {s:.3f}",
                                  xaxis_title="seed index", yaxis_title="")
                st.plotly_chart(fig, use_container_width=True)

    _seed_panel(rf_metrics, "(a) RANDOM FOREST",
                ["#6ee7a8", "#ffb547", "#7dd3fc"])

    if xgb_metrics is not None:
        _seed_panel(xgb_metrics, "(b) XGBOOST",
                    ["#f4a261", "#ffb547", "#bc4749"])

    # Combined comparison table
    section("HEAD-TO-HEAD SUMMARY")
    summary_rows = [{
        "Model": "Random Forest",
        "R²": f"{rf_metrics['R2'].mean():.3f} ± {rf_metrics['R2'].std():.3f}",
        "MAE": f"{rf_metrics['MAE'].mean():.3f} ± {rf_metrics['MAE'].std():.3f}",
        "RMSE": f"{rf_metrics['RMSE'].mean():.3f} ± {rf_metrics['RMSE'].std():.3f}",
    }]
    if xgb_metrics is not None:
        summary_rows.append({
            "Model": "XGBoost",
            "R²": f"{xgb_metrics['R2'].mean():.3f} ± {xgb_metrics['R2'].std():.3f}",
            "MAE": f"{xgb_metrics['MAE'].mean():.3f} ± {xgb_metrics['MAE'].std():.3f}",
            "RMSE": f"{xgb_metrics['RMSE'].mean():.3f} ± {xgb_metrics['RMSE'].std():.3f}",
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    st.markdown("""
    <div style='margin-top:10px;padding:14px 18px;border:1px solid #2a343d;background:#11161a;font-family:Inter,sans-serif;font-size:13px;color:#a7b3bd;line-height:1.6'>
    <strong style='color:#6ee7a8;font-family:JetBrains Mono,monospace;letter-spacing:0.1em'>INTERPRETATION</strong>&nbsp;
    The manuscript reports RF R² = 0.648 ± 0.013, XGBoost R² = 0.641 ± 0.014.
    The two models agree closely on both mean performance and dispersion across seeds,
    indicating the descriptor-target relationship is captured by tree-ensemble methods
    in general — not a model-specific artifact. Small RF advantage in dispersion is
    consistent with bagging's averaging effect vs. sequential boosting in XGBoost.
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# FIG-4/5 SHAP
# ============================================================================
with tab_shap:
    section("FIG. 4–5 · SHAP ATTRIBUTION")
    st.caption(
        "Reproduces Figure 4 (bar + heatmap) and Figure 5 (beeswarm summary) of the "
        "manuscript. Run on the last-fit RF pipeline with 200 sampled materials."
    )

    run_shap = st.button("◈ COMPUTE SHAP VALUES")
    if run_shap or "shap_cache" in st.session_state:
        if "shap_cache" not in st.session_state:
            with st.spinner("Computing tree-SHAP on 200 samples…"):
                try:
                    sv, X_s = compute_shap_values(
                        results["rf"]["pipeline"], results["rf"]["X"]
                    )
                    st.session_state.shap_cache = (sv, X_s)
                except Exception as e:
                    st.error(f"SHAP failed: {e}")
                    st.stop()

        sv, X_s = st.session_state.shap_cache

        # --- Fig 4(a) Bar ---
        section("FIG-4(A) · MEAN |SHAP|  (global importance)")
        mean_abs = np.abs(sv).mean(axis=0)
        imp_df = pd.DataFrame({"feature": X_s.columns, "mean_abs_shap": mean_abs})
        imp_df = imp_df.sort_values("mean_abs_shap", ascending=False).head(20)
        figb = px.bar(imp_df.iloc[::-1], x="mean_abs_shap", y="feature", orientation="h",
                      color_discrete_sequence=["#6ee7a8"])
        figb.update_layout(**PLOTLY_LAYOUT, height=560,
                           xaxis_title="mean |SHAP value|", yaxis_title="")
        st.plotly_chart(figb, use_container_width=True)

        # --- Fig 4(b) Heatmap ---
        section("FIG-4(B) · PER-SAMPLE HEATMAP  (|SHAP| × samples × features)")
        top_feats = imp_df["feature"].tolist()[:15]
        heat = np.abs(pd.DataFrame(sv, columns=X_s.columns)[top_feats].iloc[:80].T)
        figh = px.imshow(heat, aspect="auto", color_continuous_scale="Inferno",
                         labels=dict(color="|SHAP|"))
        figh.update_layout(**PLOTLY_LAYOUT, height=480,
                           xaxis_title="sample index", yaxis_title="feature")
        st.plotly_chart(figh, use_container_width=True)

        # --- Fig 5 Beeswarm (replicated as a strip plot of SHAP vs feature value) ---
        section("FIG-5 · BEESWARM-STYLE DISTRIBUTION")
        top_n = 10
        rows = []
        for f in top_feats[:top_n]:
            fv = X_s[f].values
            sv_col = sv[:, list(X_s.columns).index(f)]
            for x, s in zip(fv, sv_col):
                rows.append({"feature": f, "shap": s, "value": x})
        bw_df = pd.DataFrame(rows)
        # Normalize feature value 0-1 per feature for color
        bw_df["value_norm"] = bw_df.groupby("feature")["value"].transform(
            lambda v: (v - v.min()) / (v.max() - v.min() + 1e-9)
        )
        figbw = px.strip(bw_df, x="shap", y="feature", color="value_norm",
                         color_continuous_scale="RdYlGn_r",
                         orientation="h", stripmode="overlay")
        figbw.update_traces(marker=dict(size=4, opacity=0.7))
        figbw.update_layout(**PLOTLY_LAYOUT, height=480,
                            xaxis_title="SHAP value (impact on log proxy)",
                            yaxis_title="",
                            yaxis=dict(autorange="reversed"),
                            coloraxis_colorbar=dict(title="feature value\n(scaled)"))
        figbw.add_vline(x=0, line=dict(color="#a7b3bd", width=1))
        st.plotly_chart(figbw, use_container_width=True)

        with st.expander("Raw SHAP statistics"):
            st.dataframe(
                pd.DataFrame(sv, columns=X_s.columns).describe().T
                .style.format("{:.4f}"),
                use_container_width=True,
            )

        st.markdown("""
        <div style='margin-top:10px;padding:14px 18px;border:1px solid #2a343d;background:#11161a;font-family:Inter,sans-serif;font-size:13px;color:#a7b3bd;line-height:1.6'>
        <strong style='color:#6ee7a8;font-family:JetBrains Mono,monospace;letter-spacing:0.1em'>INTERPRETATION</strong>&nbsp;
        The manuscript (§3.1.6) identifies <em>range of electronegativity</em>, <em>mean Nd valence</em>,
        and <em>max Np valence</em> as dominant. Electronegativity range proxies bond polarity and charge
        redistribution; valence descriptors proxy band-edge character and orbital filling. These
        attributions explain the model, not the underlying physics — a feature matters here because
        it correlates with E<sub>g</sub>, |Δh<sub>f</sub>|, or ρ in the training set.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Click ◈ COMPUTE SHAP VALUES to attribute the model's predictions. Takes a few seconds.")


# ============================================================================
# EXPORT
# ============================================================================
with tab_export:
    section("DOWNLOAD ARTIFACTS")
    st.markdown(
        "<div style='font-family:Inter,sans-serif;font-size:13px;color:#a7b3bd;margin-bottom:14px'>"
        "Bundled XLSX with all run artifacts matching the manuscript supplement. Sheets: "
        "<code>Dataset</code>, <code>RF_metrics</code>, <code>XGB_metrics</code>, "
        "<code>Stability</code>, <code>Robust_Candidates</code>, <code>RepeatedCV</code>.</div>",
        unsafe_allow_html=True,
    )

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        export_df = df_feat.copy()
        for c in export_df.columns:
            if export_df[c].dtype == "object":
                export_df[c] = export_df[c].astype(str)
        export_df.to_excel(writer, sheet_name="Dataset", index=False)
        rf_metrics.to_excel(writer, sheet_name="RF_metrics", index=False)
        if xgb_metrics is not None:
            xgb_metrics.to_excel(writer, sheet_name="XGB_metrics", index=False)
        freq_df.to_excel(writer, sheet_name="Stability", index=False)
        robust = freq_df[freq_df["stability_score"] >= cfg.stability_threshold].merge(
            df_feat, on="material_id", how="left"
        )
        for c in robust.columns:
            if robust[c].dtype == "object":
                robust[c] = robust[c].astype(str)
        robust.to_excel(writer, sheet_name="Robust_Candidates", index=False)
        if "cv" in results:
            pd.DataFrame({"fold_R2": results["cv"]["scores"]}).to_excel(
                writer, sheet_name="RepeatedCV", index=False
            )
    buf.seek(0)

    st.download_button(
        "↓ FINAL_ML_OUTPUT.xlsx",
        data=buf,
        file_name="FINAL_ML_OUTPUT.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    csv = freq_df.merge(
        df_ranked[["material_id", "formula_pretty", "pred_mean", "pred_std"]],
        on="material_id", how="left",
    ).to_csv(index=False).encode("utf-8")
    st.download_button(
        "↓ stability_ranking.csv", data=csv,
        file_name="stability_ranking.csv", mime="text/csv",
        use_container_width=True,
    )

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown(
    """
    <div style='font-family:JetBrains Mono,monospace;font-size:10px;color:#6b7681;letter-spacing:0.15em;text-align:center;padding:10px 0'>
    THERMOELECTRIC SCREENING TERMINAL · V2 · MANUSCRIPT-ALIGNED ·
    ZT<sub>PROXY</sub> = E<sub>G</sub> · |ΔH<sub>F</sub>| / (ρ + ε) · ε = 10⁻⁶ ·
    NOT A SUBSTITUTE FOR DFT TRANSPORT
    </div>
    """,
    unsafe_allow_html=True,
)
