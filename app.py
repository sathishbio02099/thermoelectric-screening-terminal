"""
THERMOELECTRIC SCREENING TERMINAL
A Streamlit interface around a Materials-Project + Random-Forest pipeline that
ranks materials by a *heuristic* ZT proxy. NOT a true ZT predictor — this is a
discovery-stage screening tool.

Run:  streamlit run app.py
"""

from __future__ import annotations

import io
import time
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# PAGE CONFIG  (must be first Streamlit call)
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Thermoelectric Screening Terminal",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "Thermoelectric Screening Terminal — a heuristic ML pipeline that "
            "ranks Materials Project entries by a ZT *proxy*. Not a substitute "
            "for DFT transport calculations or experimental measurement."
        )
    },
)

# ----------------------------------------------------------------------------
# THEME / CSS
# ----------------------------------------------------------------------------
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
    --accent: #6ee7a8;       /* phosphor green */
    --accent-dim: #3aa874;
    --warn: #ffb547;         /* amber for ZT-proxy disclaimers */
    --danger: #ff6b6b;
    --info: #7dd3fc;
}

html, body, [class*="css"], .stApp {
    background: var(--bg-0) !important;
    color: var(--text-0) !important;
    font-family: 'Inter', sans-serif;
}

/* subtle scan-line texture */
.stApp::before {
    content: "";
    position: fixed; inset: 0;
    pointer-events: none;
    background:
        repeating-linear-gradient(
            0deg,
            rgba(255,255,255,0.012) 0px,
            rgba(255,255,255,0.012) 1px,
            transparent 1px,
            transparent 3px
        );
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

.mono, code, pre, .stCode {
    font-family: 'JetBrains Mono', monospace !important;
}

/* hero band */
.hero {
    border: 1px solid var(--line);
    background:
        radial-gradient(800px 200px at 10% 0%, rgba(110,231,168,0.08), transparent 60%),
        radial-gradient(600px 200px at 90% 100%, rgba(255,181,71,0.05), transparent 60%),
        var(--bg-1);
    padding: 28px 34px;
    margin-bottom: 22px;
    position: relative;
    overflow: hidden;
}
.hero .eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.28em;
    color: var(--accent);
    text-transform: uppercase;
    margin-bottom: 6px;
}
.hero h1 {
    font-size: 42px !important;
    margin: 0 0 4px 0 !important;
    line-height: 1.05 !important;
    font-weight: 600 !important;
}
.hero h1 em {
    font-style: italic;
    color: var(--accent);
    font-weight: 300;
}
.hero .lede {
    color: var(--text-1);
    font-size: 15px;
    max-width: 760px;
    line-height: 1.55;
    margin-top: 10px;
}
.hero .ribbon {
    display: inline-block;
    margin-top: 14px;
    padding: 6px 12px;
    border: 1px solid var(--warn);
    color: var(--warn);
    background: rgba(255,181,71,0.06);
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
}

/* metric tiles */
.tile {
    border: 1px solid var(--line);
    background: var(--bg-1);
    padding: 16px 18px;
    height: 100%;
    position: relative;
}
.tile .label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.22em;
    color: var(--text-2);
    text-transform: uppercase;
    margin-bottom: 6px;
}
.tile .value {
    font-family: 'Fraunces', serif;
    font-size: 30px;
    font-weight: 400;
    color: var(--text-0);
    line-height: 1.1;
}
.tile .delta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--text-1);
    margin-top: 4px;
}
.tile.accent { border-left: 3px solid var(--accent); }
.tile.warn   { border-left: 3px solid var(--warn); }
.tile.info   { border-left: 3px solid var(--info); }

/* honest-disclaimer banner */
.proxy-warn {
    border: 1px dashed var(--warn);
    background: rgba(255,181,71,0.05);
    color: var(--text-1);
    padding: 12px 16px;
    margin: 12px 0;
    font-size: 13px;
    line-height: 1.5;
}
.proxy-warn strong { color: var(--warn); font-family: 'JetBrains Mono', monospace; letter-spacing: 0.05em; }

/* dataframe styling */
.stDataFrame { border: 1px solid var(--line); }

/* buttons */
.stButton > button, .stDownloadButton > button {
    background: transparent !important;
    color: var(--accent) !important;
    border: 1px solid var(--accent-dim) !important;
    border-radius: 0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    padding: 8px 18px !important;
    transition: all 0.15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background: var(--accent) !important;
    color: var(--bg-0) !important;
    border-color: var(--accent) !important;
}

/* tabs */
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-2) !important;
    border-radius: 0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    padding: 10px 18px !important;
    border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}

/* expander */
.stExpander {
    background: var(--bg-1) !important;
    border: 1px solid var(--line) !important;
    border-radius: 0 !important;
}

/* inputs */
.stTextInput input, .stNumberInput input, .stSelectbox > div, .stMultiSelect > div {
    background: var(--bg-1) !important;
    color: var(--text-0) !important;
    border-radius: 0 !important;
    border: 1px solid var(--line) !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* divider */
hr { border-color: var(--line) !important; margin: 24px 0 !important; }

/* section heading */
.section-h {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.3em;
    color: var(--text-2);
    text-transform: uppercase;
    margin: 8px 0 12px 0;
    padding-bottom: 6px;
    border-bottom: 1px solid var(--line);
}

/* tag pill */
.tag {
    display: inline-block;
    padding: 3px 9px;
    border: 1px solid var(--line);
    color: var(--text-1);
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.12em;
    margin-right: 6px;
    background: var(--bg-2);
}
.tag.green { color: var(--accent); border-color: var(--accent-dim); }
.tag.amber { color: var(--warn); border-color: var(--warn); }

/* progress bar */
.stProgress > div > div > div > div { background: var(--accent) !important; }

/* sliders */
.stSlider [data-baseweb="slider"] [role="slider"] {
    background: var(--accent) !important;
    border: 2px solid var(--bg-0) !important;
}

/* link color */
a, a:visited { color: var(--accent) !important; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# PIPELINE  (refactored from MLcode.py, with caching)
# ----------------------------------------------------------------------------
@dataclass
class PipelineConfig:
    band_gap_min: float = 0.1
    band_gap_max: float = 2.0
    e_above_hull_max: float = 0.03
    n_sites_min: int = 2
    n_sites_max: int = 100
    n_seeds: int = 20
    top_k: int = 20
    rf_n_estimators: int = 150
    select_k: int = 50
    test_size: float = 0.2


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_materials_project(api_key: str, cfg: PipelineConfig) -> pd.DataFrame:
    """Fetch from MP. Cached per (api_key + cfg)."""
    from mp_api.client import MPRester
    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(
            band_gap=(cfg.band_gap_min, cfg.band_gap_max),
            energy_above_hull=(0, cfg.e_above_hull_max),
            num_sites=(cfg.n_sites_min, cfg.n_sites_max),
            fields=[
                "material_id",
                "formula_pretty",
                "band_gap",
                "density",
                "formation_energy_per_atom",
                "energy_above_hull",
            ],
        )
    rows = []
    for d in docs:
        rows.append(
            {
                "material_id": str(d.material_id),
                "formula_pretty": d.formula_pretty,
                "band_gap": d.band_gap,
                "density": d.density,
                "formation_energy_per_atom": d.formation_energy_per_atom,
                "energy_above_hull": d.energy_above_hull,
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def featurize(df_in: pd.DataFrame) -> pd.DataFrame:
    from pymatgen.core import Composition
    from matminer.featurizers.composition import ElementProperty

    df = df_in.copy()
    df["n_elements"] = df["formula_pretty"].apply(lambda x: len(Composition(x).elements))
    df = df[df["n_elements"] > 1]
    df = df.drop_duplicates(subset=["formula_pretty"]).reset_index(drop=True)

    # ZT proxy — HEURISTIC ONLY
    df["ZT_proxy"] = (
        df["band_gap"]
        * np.abs(df["formation_energy_per_atom"])
        / (df["density"] + 1e-6)
    )

    df["composition"] = df["formula_pretty"].apply(Composition)
    feat = ElementProperty.from_preset("magpie")
    df = feat.featurize_dataframe(df, col_id="composition", ignore_errors=True)
    df = df.drop(columns=["composition"])
    return df


def run_stability_analysis(
    df: pd.DataFrame, cfg: PipelineConfig, progress_callback=None
) -> dict:
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.feature_selection import SelectKBest, f_regression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score, mean_absolute_error

    y = np.log1p(df["ZT_proxy"])
    drop_cols = ["ZT_proxy", "band_gap", "density", "formation_energy_per_atom"]
    X = df.select_dtypes(include=["float64", "int64"]).drop(columns=drop_cols, errors="ignore")

    rng = np.random.RandomState(42)
    seeds = rng.randint(0, 10000, cfg.n_seeds)

    metrics = []
    all_candidates = []
    last_pipeline = None
    df_ranked_master = df.copy()
    predictions_per_seed = []  # for prediction stability stats

    for i, seed in enumerate(seeds):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=cfg.test_size, random_state=int(seed)
        )
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "selector",
                    SelectKBest(f_regression, k=min(cfg.select_k, X.shape[1])),
                ),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=cfg.rf_n_estimators,
                        random_state=int(seed),
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        metrics.append(
            {
                "seed": int(seed),
                "R2": float(r2_score(y_test, y_pred)),
                "MAE": float(mean_absolute_error(y_test, y_pred)),
            }
        )

        # fit-full for ranking (matches MLcode.py behavior)
        pipe.fit(X, y)
        preds_full = pipe.predict(X)
        df_iter = df.copy()
        df_iter["Predicted"] = preds_full
        predictions_per_seed.append(preds_full)
        top = df_iter.sort_values("Predicted", ascending=False).head(cfg.top_k).copy()
        top["seed"] = int(seed)
        all_candidates.append(top)
        last_pipeline = pipe

        if progress_callback:
            progress_callback((i + 1) / cfg.n_seeds)

    all_candidates_df = pd.concat(all_candidates, ignore_index=True)
    freq = all_candidates_df["material_id"].value_counts()
    freq_df = freq.reset_index()
    freq_df.columns = ["material_id", "count"]
    freq_df["stability_score"] = freq_df["count"] / cfg.n_seeds

    # prediction-level stability
    pred_matrix = np.array(predictions_per_seed)  # (n_seeds, n_samples)
    df_ranked_master["pred_mean"] = pred_matrix.mean(axis=0)
    df_ranked_master["pred_std"] = pred_matrix.std(axis=0)

    return {
        "metrics": pd.DataFrame(metrics),
        "freq_df": freq_df,
        "all_candidates_df": all_candidates_df,
        "df_with_preds": df_ranked_master,
        "pipeline": last_pipeline,
        "X": X,
        "y": y,
    }


@st.cache_data(show_spinner=False)
def compute_shap_values(_pipeline, X: pd.DataFrame, n_samples: int = 200):
    import shap

    model = _pipeline.named_steps["model"]
    X_imp = _pipeline.named_steps["imputer"].transform(X)
    X_sel = _pipeline.named_steps["selector"].transform(X_imp)
    selected = X.columns[_pipeline.named_steps["selector"].get_support()]
    X_df = pd.DataFrame(X_sel, columns=selected)
    X_sample = X_df.sample(min(n_samples, len(X_df)), random_state=42)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_sample)
    return sv, X_sample


# ----------------------------------------------------------------------------
# UI HELPERS
# ----------------------------------------------------------------------------
def hero():
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">◈ Thermoelectric Screening Terminal · v1.0</div>
          <h1>Ranking materials by a <em>heuristic</em><br/>ZT proxy, not by ZT.</h1>
          <div class="lede">
            A discovery-stage funnel built on the Materials Project. We pull stable, narrow-gap
            crystals, score them with a chemistry-aware surrogate, and rank candidates by an
            <strong style="color:var(--warn)">explicitly-defined proxy</strong> &mdash; not by measured or
            DFT-computed figure of merit. Use this to <em>prioritize what to compute or
            synthesize next</em>, not to declare a winner.
          </div>
          <div class="ribbon">⚠ ZT_PROXY = E<sub>g</sub> · |Δh<sub>f</sub>| / ρ &nbsp; — heuristic surrogate</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def proxy_warning(short: bool = False):
    if short:
        st.markdown(
            """
            <div class="proxy-warn">
              <strong>NOTE</strong> &nbsp; Predicted values are on the log of a heuristic surrogate,
              not real ZT. Treat as a relative score for shortlisting.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="proxy-warn">
              <strong>HEURISTIC NOTICE</strong>&nbsp; The target <code>ZT_proxy = E<sub>g</sub> · |Δh<sub>f</sub>| / ρ</code>
              is a <em>chemistry-aware surrogate</em>, not the true thermoelectric figure of merit
              ZT = S²σT/κ. It rewards narrow-gap, strongly-bonded, low-density crystals — a reasonable
              <em>prior</em> for semiconducting thermoelectrics, but it does not encode the Seebeck
              coefficient (S), electrical conductivity (σ), or lattice thermal conductivity (κ<sub>L</sub>).
              Use these rankings as a <em>screening filter</em> to choose candidates for proper
              BoltzTraP / AMSET / experimental follow-up.
            </div>
            """,
            unsafe_allow_html=True,
        )


def tile(label: str, value: str, delta: str = "", flavor: str = "accent"):
    return f"""
    <div class="tile {flavor}">
      <div class="label">{label}</div>
      <div class="value">{value}</div>
      <div class="delta">{delta}</div>
    </div>
    """


def section(label: str):
    st.markdown(f"<div class='section-h'>{label}</div>", unsafe_allow_html=True)


# Plotly theme
PLOTLY_LAYOUT = dict(
    plot_bgcolor="#11161a",
    paper_bgcolor="#11161a",
    font=dict(family="JetBrains Mono, monospace", color="#a7b3bd", size=11),
    xaxis=dict(gridcolor="#2a343d", zerolinecolor="#2a343d"),
    yaxis=dict(gridcolor="#2a343d", zerolinecolor="#2a343d"),
    margin=dict(l=40, r=20, t=40, b=40),
)


# ----------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        "<div style='font-family:Fraunces,serif;font-size:22px;letter-spacing:-0.02em;margin-bottom:2px'>"
        "◈ <em>Terminal</em></div>"
        "<div style='font-family:JetBrains Mono,monospace;font-size:10px;letter-spacing:0.25em;color:#6b7681;text-transform:uppercase;margin-bottom:20px'>"
        "Materials Project · ML Screen</div>",
        unsafe_allow_html=True,
    )

    section("01 · CREDENTIALS")
    api_key = st.text_input(
        "Materials Project API Key",
        type="password",
        help="Get a key at https://next-gen.materialsproject.org/api",
        key="mp_api_key",
    )

    section("02 · QUERY FILTERS")
    band_gap_range = st.slider(
        "Band gap range (eV)",
        0.0, 6.0, (0.1, 2.0), step=0.05,
        help="Thermoelectrics favor narrow gaps. Default 0.1–2.0 eV.",
    )
    e_hull = st.slider(
        "Max energy above hull (eV/atom)",
        0.0, 0.2, 0.03, step=0.005,
        help="Lower = more thermodynamically stable. 0.03 eV/atom is a common stability cutoff.",
    )
    n_sites = st.slider(
        "Sites per unit cell",
        2, 200, (2, 100), step=1,
        help="Small cells run faster downstream; very large cells often have complex behavior.",
    )

    section("03 · MODEL CONTROLS")
    n_seeds = st.slider("Stability seeds", 5, 50, 20, step=5)
    top_k = st.slider("Top-K per seed", 5, 100, 20, step=5)
    rf_trees = st.select_slider(
        "RF estimators",
        options=[50, 100, 150, 200, 300, 500],
        value=150,
    )
    select_k = st.slider("Features to retain (SelectKBest)", 10, 132, 50, step=5)

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
        """<div style='font-family:JetBrains Mono,monospace;font-size:10px;color:#6b7681;line-height:1.6'>
        <span class='tag amber'>HEURISTIC</span> <span class='tag'>RF · 150 EST</span><br/><br/>
        Built on Materials Project · matminer Magpie features · scikit-learn · SHAP.
        Predictions are <em>log1p(ZT_proxy)</em>, ranks are relative.
        </div>""",
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------------------------
# MAIN PAGE
# ----------------------------------------------------------------------------
hero()

cfg = PipelineConfig(
    band_gap_min=band_gap_range[0],
    band_gap_max=band_gap_range[1],
    e_above_hull_max=e_hull,
    n_sites_min=n_sites[0],
    n_sites_max=n_sites[1],
    n_seeds=n_seeds,
    top_k=top_k,
    rf_n_estimators=rf_trees,
    select_k=select_k,
)

# state init
for k in ["raw_df", "feat_df", "results"]:
    if k not in st.session_state:
        st.session_state[k] = None

# fetch
if fetch_clicked:
    if not api_key:
        st.error("Materials Project API key required. Enter it in the sidebar.")
        st.stop()
    with st.spinner("Querying Materials Project…"):
        try:
            t0 = time.time()
            df_raw = fetch_materials_project(api_key, cfg)
            st.session_state.raw_df = df_raw
            st.session_state.fetch_seconds = time.time() - t0
            st.session_state.feat_df = None
            st.session_state.results = None
            st.success(f"Pulled {len(df_raw):,} materials from MP in {st.session_state.fetch_seconds:.1f}s.")
        except Exception as e:
            st.error(f"Fetch failed: {e}")
            st.stop()

# train
if train_clicked:
    if st.session_state.raw_df is None:
        st.warning("No data loaded. Click ◇ FETCH DATA first.")
        st.stop()
    if st.session_state.feat_df is None:
        with st.spinner("Featurizing (matminer Magpie)…"):
            try:
                st.session_state.feat_df = featurize(st.session_state.raw_df)
            except Exception as e:
                st.error(f"Featurization failed: {e}")
                st.stop()

    progress = st.progress(0.0, text="Stability sweep 0/{}".format(cfg.n_seeds))
    counter = {"n": 0}

    def update(p):
        counter["n"] += 1
        progress.progress(p, text=f"Stability sweep {counter['n']}/{cfg.n_seeds} · R²/MAE per seed…")

    try:
        t0 = time.time()
        results = run_stability_analysis(st.session_state.feat_df, cfg, progress_callback=update)
        results["elapsed"] = time.time() - t0
        st.session_state.results = results
        progress.empty()
        st.success(f"Pipeline complete in {results['elapsed']:.1f}s · {cfg.n_seeds} seeds · "
                   f"mean R²={results['metrics']['R2'].mean():.3f}")
    except Exception as e:
        progress.empty()
        st.error(f"Training failed: {e}")
        st.stop()


# ============================================================================
# OUTPUT
# ============================================================================
if st.session_state.results is None:
    # initial state — explain what's about to happen
    section("WHAT THIS TOOL DOES")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(tile("01 PULL", "Stable narrow-gap crystals",
                         "Materials Project · E_hull, band gap, density filters", "info"),
                    unsafe_allow_html=True)
    with c2:
        st.markdown(tile("02 SCORE", "Heuristic ZT proxy",
                         "E_g · |Δh_f| / ρ · log-transformed", "warn"),
                    unsafe_allow_html=True)
    with c3:
        st.markdown(tile("03 RANK", "Stability across N seeds",
                         "Random Forest · matminer Magpie · SHAP", "accent"),
                    unsafe_allow_html=True)
    st.write("")
    proxy_warning(short=False)

    st.markdown("""
    ### How to use this terminal

    1. **Enter your Materials Project API key** in the sidebar (free at next-gen.materialsproject.org).
    2. **Tune query filters** — narrow gaps (0.1–2.0 eV) and hull-stable (≤0.03 eV/atom) are sensible defaults for thermoelectric screening.
    3. Click **◇ FETCH DATA** to pull the chemical search space (cached for 1 hour).
    4. Click **▷ RUN PIPELINE** to featurize with Magpie descriptors and run the N-seed stability sweep.
    5. Explore four output tabs: *Overview · Candidates · Diagnostics · SHAP*.

    ### Why a *proxy*, not real ZT?

    True ZT = S²σT/κ requires Seebeck coefficient, electrical conductivity, and total thermal conductivity —
    quantities that demand DFT transport calculations (BoltzTraP/AMSET) or measurements, neither cheap nor
    available in bulk on Materials Project. The proxy used here captures three rough priors:

    - **E<sub>g</sub>** &nbsp;→ narrow gaps favor good Seebeck × σ trade-off (Mott)
    - **|Δh<sub>f</sub>|** &nbsp;→ strong bonding correlates with low κ<sub>lattice</sub> in many systems
    - **1/ρ** &nbsp;→ lower density correlates with softer phonon spectra (very rough)

    None of these *predict* ZT. Together they form a **filtering prior** that lets you cut a 50k-material
    space down to ~20 candidates worth proper transport calculations. That is what this terminal is for.
    """, unsafe_allow_html=True)
    st.stop()


# ============================================================================
# RESULTS LOADED
# ============================================================================
results = st.session_state.results
df_feat = st.session_state.feat_df
metrics_df = results["metrics"]
freq_df = results["freq_df"]
df_ranked = results["df_with_preds"]

# ---- KPI strip ----
section("RUN SUMMARY")
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(tile("MATERIALS", f"{len(df_feat):,}",
                     f"after dedup · n_elements > 1", "info"),
                unsafe_allow_html=True)
with c2:
    st.markdown(tile("MEAN R²", f"{metrics_df['R2'].mean():.3f}",
                     f"σ = {metrics_df['R2'].std():.3f} · {cfg.n_seeds} seeds", "accent"),
                unsafe_allow_html=True)
with c3:
    st.markdown(tile("MEAN MAE", f"{metrics_df['MAE'].mean():.3f}",
                     f"log1p(ZT_proxy) units", "accent"),
                unsafe_allow_html=True)
with c4:
    robust_n = (freq_df["stability_score"] >= 0.8).sum()
    st.markdown(tile("ROBUST", f"{robust_n}",
                     f"≥ 80% of seeds · top-{cfg.top_k}", "accent"),
                unsafe_allow_html=True)
with c5:
    st.markdown(tile("RUNTIME", f"{results.get('elapsed',0):.1f}s",
                     f"{cfg.n_seeds} seeds · {cfg.rf_n_estimators} trees", "warn"),
                unsafe_allow_html=True)

st.write("")
proxy_warning(short=True)


# ---- tabs ----
tab_overview, tab_candidates, tab_diag, tab_shap, tab_export = st.tabs(
    ["▤ OVERVIEW", "◇ CANDIDATES", "▥ DIAGNOSTICS", "◈ SHAP", "↓ EXPORT"]
)

# ============================================================================
# OVERVIEW
# ============================================================================
with tab_overview:
    left, right = st.columns([1.3, 1])

    with left:
        section("PROXY DISTRIBUTION · LOG SCALE")
        fig = px.histogram(
            df_ranked, x="ZT_proxy", nbins=60, log_x=True,
            color_discrete_sequence=["#6ee7a8"],
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=320,
                          xaxis_title="ZT_proxy  (heuristic)",
                          yaxis_title="count")
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
                           xaxis_title="band gap (eV)",
                           yaxis_title="density (g/cm³)",
                           coloraxis_colorbar=dict(title="log1p(ZT_proxy)"))
        st.plotly_chart(fig2, use_container_width=True)

    with right:
        section("CHEMISTRY PROFILE")
        # element frequency in top-K consensus
        try:
            from pymatgen.core import Composition
            robust_ids = freq_df[freq_df["stability_score"] >= 0.5]["material_id"]
            robust_subset = df_feat[df_feat["material_id"].isin(robust_ids)]
            element_counts = {}
            for f in robust_subset["formula_pretty"]:
                for el in Composition(f).elements:
                    element_counts[el.symbol] = element_counts.get(el.symbol, 0) + 1
            if element_counts:
                el_df = pd.DataFrame(
                    sorted(element_counts.items(), key=lambda x: -x[1])[:15],
                    columns=["element", "count"],
                )
                fig_el = px.bar(el_df, x="count", y="element", orientation="h",
                                color_discrete_sequence=["#6ee7a8"])
                fig_el.update_layout(**PLOTLY_LAYOUT, height=420,
                                     yaxis=dict(autorange="reversed", gridcolor="#2a343d"),
                                     xaxis_title="appearances in stable top set",
                                     yaxis_title="")
                st.plotly_chart(fig_el, use_container_width=True)
                st.caption(
                    "Element frequency within candidates appearing in ≥50% of seeds. "
                    "Heavy/chalcogen-rich families (Te, Se, Sb, Bi) are the classic thermoelectric "
                    "chemistries — proxy alignment with them is a sanity check, not a guarantee."
                )
            else:
                st.info("No robust candidates yet at ≥50% threshold.")
        except Exception as e:
            st.warning(f"Chemistry profile unavailable: {e}")

    st.markdown("---")
    section("INTERPRETATION GUIDE")
    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown("""
        **Read R² with care.** R² here is on `log1p(ZT_proxy)`, a smooth synthetic target.
        High R² (~0.9+) means the RF learned the proxy *formula*, not the underlying physics.
        Use R² to confirm the model isn't broken; don't read it as scientific accuracy.
        """)
    with g2:
        st.markdown("""
        **Stability > top rank.** A material that appears in the top-K of 19/20 seeds is a
        far stronger lead than a material that scored #1 once. Sort by stability_score, not
        by single-run predicted value.
        """)
    with g3:
        st.markdown("""
        **What's missing.** No κ_lattice, no anharmonicity, no carrier concentration, no
        doping optimization. Anything you ship from this list needs proper transport
        (BoltzTraP / AMSET / Phono3py) before claiming a ZT number.
        """)


# ============================================================================
# CANDIDATES
# ============================================================================
with tab_candidates:
    section("ROBUST CANDIDATES · SORTED BY SEED-STABILITY")

    threshold = st.slider(
        "Minimum stability score (fraction of seeds in which candidate appeared in top-K)",
        0.0, 1.0, 0.8, step=0.05, key="stab_thr",
    )

    merged = freq_df.merge(
        df_ranked[["material_id", "formula_pretty", "band_gap", "density",
                   "formation_energy_per_atom", "energy_above_hull", "n_elements",
                   "ZT_proxy", "pred_mean", "pred_std"]],
        on="material_id", how="left",
    )
    merged = merged[merged["stability_score"] >= threshold].copy()
    merged = merged.sort_values(["stability_score", "pred_mean"], ascending=[False, False])

    if len(merged) == 0:
        st.warning("No candidates above this threshold. Lower the slider.")
    else:
        # KPI row for candidates
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            st.markdown(tile("CANDIDATES", f"{len(merged)}",
                             f"≥ {int(threshold*100)}% seed agreement"), unsafe_allow_html=True)
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
                "stability": "{:.2f}",
                "pred_mean(log)": "{:.3f}",
                "pred_std": "{:.3f}",
                "E_g (eV)": "{:.3f}",
                "ρ (g/cm³)": "{:.2f}",
                "Δh_f (eV/at)": "{:.3f}",
                "E_hull (eV/at)": "{:.4f}",
                "ZT_proxy": "{:.4f}",
            })
            .background_gradient(subset=["stability"], cmap="Greens")
            .background_gradient(subset=["ZT_proxy"], cmap="YlOrBr"),
            use_container_width=True,
            height=520,
        )

        st.markdown("""
        <div style='font-family:Inter,sans-serif;font-size:13px;color:#a7b3bd;margin-top:10px;line-height:1.55'>
        Each row links a Materials Project ID to its proxy score, prediction stability across seeds,
        and the three physical inputs feeding the heuristic. <strong style='color:#ffb547'>pred_std</strong>
        is the across-seed standard deviation of the predicted log-proxy — small values mean the model
        agrees on this candidate regardless of train/test split.
        </div>
        """, unsafe_allow_html=True)

        # Detail card for selected material
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
                             f"{int(row['n_elements'])} elements"), unsafe_allow_html=True)
        with d2:
            st.markdown(tile("STABILITY", f"{row['stability_score']:.0%}",
                             f"{int(row['count'])}/{cfg.n_seeds} seeds", "accent"),
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
                             "narrower → better Seebeck/σ tradeoff (rough)"), unsafe_allow_html=True)
        with d6:
            st.markdown(tile("DENSITY", f"{row['density']:.2f} g/cm³",
                             "heavier elements → lower κ_lattice (rough)"), unsafe_allow_html=True)
        with d7:
            st.markdown(tile("E_HULL", f"{row['energy_above_hull']*1000:.1f} meV/at",
                             "thermodynamic stability"), unsafe_allow_html=True)

        st.markdown(f"""
        <div style='margin-top:18px;padding:14px 18px;border:1px solid #2a343d;background:#11161a;font-family:Inter,sans-serif;font-size:13px;color:#a7b3bd;line-height:1.6'>
        <strong style='color:#6ee7a8;font-family:JetBrains Mono,monospace;letter-spacing:0.1em'>NEXT-STEP RECOMMENDATION</strong><br/><br/>
        Before treating <span class='mono'>{row['formula_pretty']}</span> as a real thermoelectric lead:
        <ol style='margin-top:8px;line-height:1.7'>
        <li>Confirm the band gap and band structure with a hybrid functional (HSE06 or higher).</li>
        <li>Run <strong>BoltzTraP2</strong> or <strong>AMSET</strong> for the Seebeck coefficient and electronic conductivity vs. doping.</li>
        <li>Estimate κ<sub>lattice</sub> with <strong>Phono3py</strong> or the Slack model — the proxy completely ignores this.</li>
        <li>Check the synthesis literature for known phases of {row['formula_pretty']}; an E<sub>hull</sub> of {row['energy_above_hull']*1000:.1f} meV/atom is only a thermodynamic green light.</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# DIAGNOSTICS
# ============================================================================
with tab_diag:
    section("PER-SEED METRICS")
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Box(
            y=metrics_df["R2"], name="R²", marker_color="#6ee7a8",
            boxmean="sd",
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=320, title="R² distribution across seeds",
                          showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Box(
            y=metrics_df["MAE"], name="MAE", marker_color="#ffb547",
            boxmean="sd",
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=320, title="MAE distribution across seeds",
                          showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    section("R² PER SEED · TIMELINE")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(metrics_df))), y=metrics_df["R2"],
        mode="lines+markers",
        line=dict(color="#6ee7a8", width=2),
        marker=dict(size=8, color="#6ee7a8"),
        name="R²",
    ))
    fig.add_hline(y=metrics_df["R2"].mean(), line_dash="dash",
                  line_color="#a7b3bd",
                  annotation_text=f"mean = {metrics_df['R2'].mean():.3f}",
                  annotation_font_color="#a7b3bd")
    fig.update_layout(**PLOTLY_LAYOUT, height=300,
                      xaxis_title="seed index", yaxis_title="R²")
    st.plotly_chart(fig, use_container_width=True)

    section("RAW METRICS TABLE")
    st.dataframe(
        metrics_df.style.format({"R2": "{:.4f}", "MAE": "{:.4f}"})
                       .background_gradient(subset=["R2"], cmap="Greens")
                       .background_gradient(subset=["MAE"], cmap="OrRd"),
        use_container_width=True,
    )

    st.markdown(f"""
    <div style='margin-top:14px;padding:14px 18px;border:1px solid #2a343d;background:#11161a;font-family:Inter,sans-serif;font-size:13px;color:#a7b3bd;line-height:1.6'>
    <strong style='color:#ffb547;font-family:JetBrains Mono,monospace;letter-spacing:0.1em'>HOW TO READ THIS</strong><br/><br/>
    The model is learning <code>log1p(ZT_proxy)</code>, which is itself a smooth function of three Materials Project columns.
    A mean R² near 1 means the random forest has rediscovered the proxy formula from chemical descriptors — that is a
    <em>self-consistency check</em>, not evidence about real thermoelectric performance. The point of this run is the
    <strong>spread</strong> across seeds (σ = {metrics_df['R2'].std():.3f}) and the <strong>seed-agreement on top candidates</strong>, not absolute R².
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# SHAP
# ============================================================================
with tab_shap:
    section("FEATURE ATTRIBUTION · TREE-SHAP")
    st.markdown(
        "<div style='font-family:Inter,sans-serif;font-size:13px;color:#a7b3bd;margin-bottom:14px'>"
        "Shows which Magpie chemistry descriptors most influence the RF's <em>proxy</em> prediction. "
        "These attributions explain the model, not the underlying physics — a feature that matters here "
        "matters because it correlates with E<sub>g</sub>, |Δh<sub>f</sub>|, or 1/ρ in the training set.</div>",
        unsafe_allow_html=True,
    )

    run_shap = st.button("◈ COMPUTE SHAP VALUES")
    if run_shap or "shap_cache" in st.session_state:
        if "shap_cache" not in st.session_state:
            with st.spinner("Computing tree-SHAP on 200 samples…"):
                try:
                    sv, X_sample = compute_shap_values(results["pipeline"], results["X"])
                    st.session_state.shap_cache = (sv, X_sample)
                except Exception as e:
                    st.error(f"SHAP failed: {e}")
                    st.stop()

        sv, X_sample = st.session_state.shap_cache
        mean_abs = np.abs(sv).mean(axis=0)
        imp_df = pd.DataFrame({"feature": X_sample.columns, "mean_abs_shap": mean_abs})
        imp_df = imp_df.sort_values("mean_abs_shap", ascending=False).head(20)

        fig = px.bar(
            imp_df.iloc[::-1], x="mean_abs_shap", y="feature", orientation="h",
            color_discrete_sequence=["#6ee7a8"],
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=560,
                          xaxis_title="mean |SHAP value| (impact on log proxy)",
                          yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Show full SHAP values matrix"):
            st.dataframe(
                pd.DataFrame(sv, columns=X_sample.columns).describe().T
                .style.format("{:.4f}"),
                use_container_width=True,
            )
    else:
        st.info("Click ◈ COMPUTE SHAP VALUES to attribute the model's predictions. Takes a few seconds.")


# ============================================================================
# EXPORT
# ============================================================================
with tab_export:
    section("DOWNLOAD ARTIFACTS")
    st.markdown(
        "<div style='font-family:Inter,sans-serif;font-size:13px;color:#a7b3bd;margin-bottom:14px'>"
        "Bundled XLSX with all run artifacts. Sheets: <code>Dataset</code>, <code>Metrics</code>, "
        "<code>Stability</code>, <code>Robust_Candidates</code>.</div>",
        unsafe_allow_html=True,
    )

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        # drop unhashable columns before export
        export_df = df_feat.copy()
        for c in export_df.columns:
            if export_df[c].dtype == "object":
                export_df[c] = export_df[c].astype(str)
        export_df.to_excel(writer, sheet_name="Dataset", index=False)
        metrics_df.to_excel(writer, sheet_name="Metrics", index=False)
        freq_df.to_excel(writer, sheet_name="Stability", index=False)
        robust = freq_df[freq_df["stability_score"] >= 0.8].merge(
            df_feat, on="material_id", how="left"
        )
        for c in robust.columns:
            if robust[c].dtype == "object":
                robust[c] = robust[c].astype(str)
        robust.to_excel(writer, sheet_name="Robust_Candidates", index=False)
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
        "↓ stability_ranking.csv",
        data=csv,
        file_name="stability_ranking.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ----------------------------------------------------------------------------
# FOOTER
# ----------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style='font-family:JetBrains Mono,monospace;font-size:10px;color:#6b7681;letter-spacing:0.15em;text-align:center;padding:10px 0'>
    THERMOELECTRIC SCREENING TERMINAL · HEURISTIC ZT PROXY · NOT A SUBSTITUTE FOR DFT TRANSPORT OR EXPERIMENT
    </div>
    """,
    unsafe_allow_html=True,
)
