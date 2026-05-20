"""
THERMOELECTRIC SCREENING TERMINAL  v2.1 (with CIF prediction)

Companion UI for:
  "From Prediction to Physics: A Robust Machine Learning Framework for
   Thermoelectric Discovery with Explicit First-Principles Validation"
  Sathish Panneer Selvam et al. (submitted to J. Mater. Chem. A)

NEW in v2.1:
  - "Predict Your Material" tab: upload CIF + enter properties → get ZT proxy prediction
  - Memory-optimized for Streamlit Cloud (stratified subsampling, batched featurization)
  - Explicit RF-only candidate pipeline (XGBoost is metrics-only comparison)

Formula: ZT_proxy = E_g · |ΔH_f| / (ρ + ε), ε = 10⁻⁶
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
)

# ============================================================================
# THEME
# ============================================================================
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Fraunces:opsz,wght@9..144,300;9..144,400;9..144,600&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --bg-0: #0a0d0f; --bg-1: #11161a; --bg-2: #1a2128; --line: #2a343d;
    --text-0: #e8eef2; --text-1: #a7b3bd; --text-2: #6b7681;
    --accent: #6ee7a8; --accent-dim: #3aa874; --xgb: #f4a261;
    --warn: #ffb547; --danger: #ff6b6b; --info: #7dd3fc;
}

html, body, [class*="css"], .stApp {
    background: var(--bg-0) !important;
    color: var(--text-0) !important;
    font-family: 'Inter', sans-serif;
}

.stApp::before {
    content: ""; position: fixed; inset: 0; pointer-events: none;
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

h1, h2, h3, h4, h5 {
    font-family: 'Fraunces', serif !important;
    letter-spacing: -0.02em !important;
    color: var(--text-0) !important;
}

code, pre { font-family: 'JetBrains Mono', monospace !important; }

.hero {
    border: 1px solid var(--line);
    background: radial-gradient(800px 200px at 10% 0%, rgba(110,231,168,0.08), transparent 60%),
                radial-gradient(600px 200px at 90% 100%, rgba(255,181,71,0.05), transparent 60%),
                var(--bg-1);
    padding: 28px 34px; margin-bottom: 22px; position: relative; overflow: hidden;
}
.hero .eyebrow {
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
    letter-spacing: 0.28em; color: var(--accent);
    text-transform: uppercase; margin-bottom: 6px;
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

.tile {
    border: 1px solid var(--line); background: var(--bg-1);
    padding: 16px 18px; height: 100%; position: relative;
}
.tile .label {
    font-family: 'JetBrains Mono', monospace; font-size: 10px;
    letter-spacing: 0.22em; color: var(--text-2);
    text-transform: uppercase; margin-bottom: 6px;
}
.tile .value {
    font-family: 'Fraunces', serif; font-size: 30px;
    font-weight: 400; color: var(--text-0); line-height: 1.1;
}
.tile .delta {
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
    color: var(--text-1); margin-top: 4px;
}
.tile.accent { border-left: 3px solid var(--accent); }
.tile.warn   { border-left: 3px solid var(--warn); }
.tile.info   { border-left: 3px solid var(--info); }

.proxy-warn {
    border: 1px dashed var(--warn); background: rgba(255,181,71,0.05);
    color: var(--text-1); padding: 12px 16px; margin: 12px 0;
    font-size: 13px; line-height: 1.5;
}
.proxy-warn strong {
    color: var(--warn); font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.05em;
}

.section-h {
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
    letter-spacing: 0.3em; color: var(--text-2); text-transform: uppercase;
    margin: 8px 0 12px 0; padding-bottom: 6px; border-bottom: 1px solid var(--line);
}

.stButton > button, .stDownloadButton > button {
    background: transparent !important; color: var(--accent) !important;
    border: 1px solid var(--accent-dim) !important; border-radius: 0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important; letter-spacing: 0.15em !important;
    text-transform: uppercase !important; padding: 8px 18px !important;
    transition: all 0.15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background: var(--accent) !important; color: var(--bg-0) !important;
}

.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] {
    background: transparent !important; color: var(--text-2) !important;
    border-radius: 0 !important; font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important; letter-spacing: 0.15em !important;
    text-transform: uppercase !important; padding: 10px 18px !important;
    border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important; border-bottom: 2px solid var(--accent) !important;
}

.predict-box {
    border: 1px solid var(--accent-dim);
    background: linear-gradient(135deg, rgba(110,231,168,0.06), rgba(110,231,168,0.02));
    padding: 20px 24px; margin: 16px 0;
}
.predict-box h3 {
    font-size: 26px !important; margin-bottom: 8px !important;
    color: var(--accent) !important;
}
.predict-box .caption {
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
    color: var(--text-1); margin-bottom: 16px;
}

a, a:visited { color: var(--accent) !important; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================================
# CONFIG
# ============================================================================
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
    select_k: int = 50
    test_size: float = 0.20
    epsilon: float = 1e-6
    stability_threshold: float = 0.9
    cv_splits: int = 5
    cv_repeats: int = 3
    run_xgb: bool = False  # default off to save memory
    run_repeated_cv: bool = False  # default off to save time
    max_rows_cloud: int = 5000
    batch_size_cloud: int = 500


# ============================================================================
# DATA PIPELINE
# ============================================================================
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_materials_project(api_key: str, cfg_dict: dict) -> pd.DataFrame:
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
def featurize_batch(df_in: pd.DataFrame, epsilon: float,
                    max_rows: int = 0, batch_size: int = 500) -> pd.DataFrame:
    """
    Clean + ZT proxy + Magpie features. Memory-optimized for cloud.
    """
    from pymatgen.core import Composition
    from matminer.featurizers.composition import ElementProperty
    import gc

    df = df_in.dropna(subset=[
        "band_gap", "density", "formation_energy_per_atom", "energy_above_hull"
    ]).copy()
    df["n_elements"] = df["formula_pretty"].apply(lambda s: len(Composition(s).elements))
    df = df[df["n_elements"] > 1].drop_duplicates(subset=["formula_pretty"]).reset_index(drop=True)

    # ZT proxy
    df["ZT_proxy"] = (
        df["band_gap"] * np.abs(df["formation_energy_per_atom"]) / (df["density"] + epsilon)
    )

    # Stratified subsample
    if max_rows > 0 and len(df) > max_rows:
        df["_q"] = pd.qcut(df["ZT_proxy"], q=10, labels=False, duplicates="drop")
        n_per_q = max(1, max_rows // df["_q"].nunique())
        df = (
            df.groupby("_q", group_keys=False)
              .apply(lambda g: g.sample(min(len(g), n_per_q), random_state=42))
              .drop(columns="_q").reset_index(drop=True)
        )

    df["composition"] = df["formula_pretty"].apply(Composition)
    feat = ElementProperty.from_preset("magpie")

    # Batch processing
    if batch_size > 0 and len(df) > batch_size:
        chunks = []
        for start in range(0, len(df), batch_size):
            chunk = df.iloc[start:start + batch_size].copy()
            chunk = feat.featurize_dataframe(chunk, col_id="composition", ignore_errors=True)
            chunks.append(chunk)
            gc.collect()
        df = pd.concat(chunks, ignore_index=True)
    else:
        df = feat.featurize_dataframe(df, col_id="composition", ignore_errors=True)

    df = df.drop(columns=["composition"])
    gc.collect()
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


def _rf_sweep_with_candidates(df, cfg, progress_cb=None):
    """RF 20-seed sweep — THE candidate-generating pipeline."""
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


def run_pipeline_simple(df, cfg, progress_cb=None):
    """Simplified: just RF sweep, no XGBoost or CV to save memory/time."""
    out = {}
    t0 = time.time()
    out["rf"] = _rf_sweep_with_candidates(df, cfg, progress_cb=progress_cb)

    freq = out["rf"]["candidates"]["material_id"].value_counts()
    freq_df = freq.reset_index()
    freq_df.columns = ["material_id", "count"]
    freq_df["stability_score"] = freq_df["count"] / len(cfg.seeds)
    out["freq_df"] = freq_df

    df_ranked = df.copy()
    pmat = out["rf"]["preds_per_seed"]
    df_ranked["pred_mean"] = pmat.mean(axis=0)
    df_ranked["pred_std"] = pmat.std(axis=0)
    out["df_with_preds"] = df_ranked
    out["elapsed"] = time.time() - t0
    return out


# ============================================================================
# CIF PREDICTION UTILITIES
# ============================================================================
def parse_cif_to_formula(cif_bytes: bytes) -> str:
    """Extract chemical formula from CIF file."""
    from pymatgen.io.cif import CifParser
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".cif") as tmp:
        tmp.write(cif_bytes)
        tmp.flush()
        parser = CifParser(tmp.name)
        struct = parser.get_structures()[0]
        return struct.composition.reduced_formula


def featurize_single_composition(formula: str) -> pd.DataFrame:
    """
    Featurize a single composition with Magpie descriptors.
    Returns 1-row DataFrame with 132 features.
    """
    from pymatgen.core import Composition
    from matminer.featurizers.composition import ElementProperty

    comp = Composition(formula)
    feat = ElementProperty.from_preset("magpie")
    row = {"composition": comp}
    df_in = pd.DataFrame([row])
    df_out = feat.featurize_dataframe(df_in, col_id="composition", ignore_errors=True)
    return df_out.drop(columns=["composition"])


def predict_single_material(
    formula: str, band_gap: float, form_energy: float, density: float,
    epsilon: float, trained_pipeline
) -> dict:
    """
    Given a formula + properties, compute ZT_proxy, featurize, predict with trained model.
    Returns dict with proxy, prediction, and model's X input.
    """
    # Compute heuristic proxy
    zt_proxy = (band_gap * abs(form_energy)) / (density + epsilon)

    # Featurize
    X_new = featurize_single_composition(formula)

    # Predict log1p(proxy)
    y_pred_log = trained_pipeline.predict(X_new)[0]

    return {
        "formula": formula,
        "ZT_proxy": zt_proxy,
        "predicted_log": float(y_pred_log),
        "X": X_new,
    }


# ============================================================================
# UI HELPERS
# ============================================================================
def hero():
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">◈ Thermoelectric Screening Terminal · v2.1 · with CIF prediction</div>
          <h1>From prediction to physics —<br/>a <em>robust</em> screening pipeline.</h1>
          <div class="lede">
            Companion interface for Selvam <em>et al.</em> Implements the heuristic ZT proxy
            ranking on Materials Project data with 20-seed stability analysis.
            <strong>NEW:</strong> Upload your own CIF file and get a ZT proxy prediction from
            the trained model.
          </div>
          <div class="ribbon">⚠ ZT_PROXY = E<sub>g</sub> · |Δh<sub>f</sub>| / (ρ + ε) — heuristic surrogate</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(label):
    st.markdown(f"<div class='section-h'>{label}</div>", unsafe_allow_html=True)


def tile(label, value, delta="", flavor="accent"):
    return f"""
    <div class="tile {flavor}">
      <div class="label">{label}</div>
      <div class="value">{value}</div>
      <div class="delta">{delta}</div>
    </div>
    """


# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown(
        "<div style='font-family:Fraunces,serif;font-size:22px;letter-spacing:-0.02em;margin-bottom:2px'>"
        "◈ <em>Terminal v2.1</em></div>"
        "<div style='font-family:JetBrains Mono,monospace;font-size:10px;letter-spacing:0.25em;color:#6b7681;text-transform:uppercase;margin-bottom:20px'>"
        "CIF Prediction Enabled</div>",
        unsafe_allow_html=True,
    )

    section("01 · CREDENTIALS")
    api_key = st.text_input(
        "Materials Project API Key", type="password",
        help="Get a key at next-gen.materialsproject.org/api",
    )

    section("02 · QUERY FILTERS")
    band_gap_range = st.slider(
        "Band gap range (eV)", 0.0, 6.0, (0.1, 2.0), step=0.05,
    )
    e_hull = st.slider(
        "Max E_hull (eV/atom)", 0.0, 0.2, 0.03, step=0.005,
    )
    n_sites = st.slider("Sites per cell", 2, 200, (2, 100), step=1)

    section("03 · CLOUD MEMORY LIMITS")
    st.caption("Streamlit Cloud free tier has ~1 GB RAM. Subsample to stay under.")
    max_rows = st.select_slider(
        "Max materials to featurize",
        options=[1000, 2000, 3000, 5000, 8000, 0],
        value=3000,
        format_func=lambda v: "no limit" if v == 0 else f"{v:,}",
        help="Smaller = safer. The manuscript's full ~70k run requires ≥4 GB RAM (run MLcode_v2.py locally).",
    )
    batch_size = st.select_slider(
        "Featurization batch size",
        options=[100, 250, 500, 1000],
        value=500,
    )

    section("04 · MODEL CONTROLS")
    use_manuscript_seeds = st.checkbox("Use manuscript 20 seeds", value=True)
    if not use_manuscript_seeds:
        n_seeds = st.slider("Number of seeds", 5, 50, 20, step=5)
    else:
        n_seeds = len(MANUSCRIPT_SEEDS)

    top_k = st.slider("Top-K per seed", 5, 100, 20, step=5)
    rf_trees = st.select_slider(
        "RF estimators", options=[50, 100, 150, 200], value=150,
    )
    select_k = st.slider("Features retained", 10, 132, 50, step=5)
    stab_threshold = st.slider("Stability threshold", 0.5, 1.0, 0.9, step=0.05)

    section("05 · ACTIONS")
    fetch_clicked = st.button("◇ FETCH DATA", use_container_width=True)
    train_clicked = st.button("▷ RUN PIPELINE", use_container_width=True)
    reset_clicked = st.button("⨯ RESET CACHE", use_container_width=True)
    if reset_clicked:
        st.cache_data.clear()
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()


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
    max_rows_cloud=max_rows,
    batch_size_cloud=batch_size,
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
            df_raw = fetch_materials_project(api_key, cfg.__dict__)
            st.session_state.raw_df = df_raw
            st.session_state.feat_df = None
            st.session_state.results = None
            st.success(f"Pulled {len(df_raw):,} materials from MP.")
        except Exception as e:
            st.error(f"Fetch failed: {e}")
            st.stop()

# Train
if train_clicked:
    if st.session_state.raw_df is None:
        st.warning("No data loaded. Click ◇ FETCH DATA first.")
        st.stop()

    if st.session_state.feat_df is None:
        n_in = len(st.session_state.raw_df)
        n_target = min(n_in, max_rows) if max_rows > 0 else n_in
        with st.spinner(f"Featurizing {n_target:,} materials in batches of {batch_size}…"):
            try:
                st.session_state.feat_df = featurize_batch(
                    st.session_state.raw_df, cfg.epsilon,
                    max_rows=max_rows, batch_size=batch_size,
                )
                st.info(f"✓ Featurized {len(st.session_state.feat_df):,} materials.")
            except Exception as e:
                err_str = str(e).lower()
                if any(x in err_str for x in ["connection reset", "memory", "killed", "errno 104"]):
                    st.error(
                        f"**Featurization failed (out of memory on Streamlit Cloud).**\n\n"
                        f"Streamlit Cloud free tier has ~1 GB RAM. You tried {n_target:,} materials.\n\n"
                        f"**Fix:** Lower 'Max materials to featurize' to 2000 or 1000, or tighten query filters.\n\n"
                        f"For the full ~70k manuscript run, use `MLcode_v2.py` locally.\n\n"
                        f"Raw error: `{e}`"
                    )
                else:
                    st.error(f"Featurization failed: {e}")
                st.stop()

    progress = st.progress(0.0, text=f"RF sweep 0/{len(cfg.seeds)}")

    def cb(p):
        progress.progress(p, text=f"RF sweep {int(p*len(cfg.seeds))}/{len(cfg.seeds)}")

    try:
        results = run_pipeline_simple(st.session_state.feat_df, cfg, progress_cb=cb)
        st.session_state.results = results
        progress.empty()
        st.success(
            f"Pipeline complete in {results['elapsed']:.1f}s · "
            f"RF R²={results['rf']['metrics']['R2'].mean():.3f}±{results['rf']['metrics']['R2'].std():.3f}"
        )
    except Exception as e:
        progress.empty()
        st.error(f"Training failed: {e}")
        st.stop()


# ============================================================================
# OUTPUT — NEW TAB STRUCTURE
# ============================================================================
if st.session_state.results is None:
    section("WHAT THIS PIPELINE DOES")
    st.markdown("""
    1. **Pull** stable narrow-gap crystals from Materials Project (band gap 0.1–2.0 eV, E_hull ≤ 0.03 eV/atom)
    2. **Score** with the heuristic ZT proxy: `ZT_proxy = E_g · |ΔH_f| / (ρ + ε)`, ε = 10⁻⁶
    3. **Featurize** with matminer's Magpie elemental descriptors (~132 features)
    4. **Train** Random Forest on 20 seeds with 80:20 holdout, report R²/MAE/RMSE
    5. **Rank** candidates by stability across seeds (≥ 90% threshold)
    6. **NEW: Predict** on your own CIF files using the trained model

    ### Get started

    Enter your Materials Project API key in the sidebar, click **◇ FETCH DATA**, then **▷ RUN PIPELINE**.

    Once trained, use the **PREDICT YOUR MATERIAL** tab to upload a CIF and get a ZT proxy prediction.
    """)
    st.stop()

results = st.session_state.results
df_feat = st.session_state.feat_df
rf_metrics = results["rf"]["metrics"]
freq_df = results["freq_df"]
df_ranked = results["df_with_preds"]

# KPI strip
section("RUN SUMMARY")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(tile("MATERIALS", f"{len(df_feat):,}", "after cleaning"), unsafe_allow_html=True)
with c2:
    st.markdown(tile("RF R²", f"{rf_metrics['R2'].mean():.3f}",
                     f"σ = {rf_metrics['R2'].std():.3f}"), unsafe_allow_html=True)
with c3:
    st.markdown(tile("RF RMSE", f"{rf_metrics['RMSE'].mean():.3f}",
                     f"MAE = {rf_metrics['MAE'].mean():.3f}"), unsafe_allow_html=True)
with c4:
    robust_n = (freq_df["stability_score"] >= cfg.stability_threshold).sum()
    st.markdown(tile("ROBUST", f"{robust_n}",
                     f"≥ {int(cfg.stability_threshold*100)}% agreement"), unsafe_allow_html=True)


# Tabs
tab_predict, tab_candidates, tab_overview, tab_export = st.tabs([
    "🔮 PREDICT YOUR MATERIAL", "◇ CANDIDATES", "▤ OVERVIEW", "↓ EXPORT"
])


# ============================================================================
# TAB: PREDICT YOUR MATERIAL (NEW!)
# ============================================================================
with tab_predict:
    st.markdown("""
    <div class='predict-box'>
      <h3>🔮 Predict ZT Proxy for Your Own Material</h3>
      <div class='caption'>
        Upload a CIF file + enter band gap, formation energy, and density.
        The trained RF model will featurize your composition with the same Magpie
        descriptors and predict log(1 + ZT_proxy). This is a HEURISTIC screening
        score, not a real ZT prediction.
      </div>
    </div>
    """, unsafe_allow_html=True)

    section("01 · UPLOAD CIF OR ENTER FORMULA")
    upload_method = st.radio(
        "How would you like to specify your material?",
        ["Upload CIF file", "Manually enter formula"],
        horizontal=True,
    )

    formula_user = None
    if upload_method == "Upload CIF file":
        cif_file = st.file_uploader(
            "Upload a CIF file", type=["cif"],
            help="We'll extract the chemical formula automatically.",
        )
        if cif_file is not None:
            try:
                cif_bytes = cif_file.read()
                formula_user = parse_cif_to_formula(cif_bytes)
                st.success(f"Extracted formula: **{formula_user}**")
            except Exception as e:
                st.error(f"CIF parsing failed: {e}")
    else:
        formula_user = st.text_input(
            "Chemical formula (e.g., Bi2Te3, PbSe, Ag2Se)",
            help="Use standard pymatgen notation: capitalized element symbols + numbers.",
        )
        if formula_user:
            try:
                from pymatgen.core import Composition
                _ = Composition(formula_user)  # validate
                st.success(f"Formula validated: **{formula_user}**")
            except Exception as e:
                st.error(f"Invalid formula: {e}")
                formula_user = None

    if formula_user:
        section("02 · ENTER MATERIAL PROPERTIES")
        st.caption(
            "These are the inputs to the ZT proxy formula. "
            "Band gap and formation energy should come from DFT (PBE or HSE06). "
            "Density can be experimental or DFT-relaxed."
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            bg_user = st.number_input(
                "Band gap (eV)", min_value=0.0, max_value=10.0, value=1.0, step=0.05,
                help="E_g from DFT or experiment",
            )
        with col2:
            fe_user = st.number_input(
                "Formation energy (eV/atom)", min_value=-10.0, max_value=5.0, value=-0.5, step=0.05,
                help="ΔH_f per atom (usually negative for stable compounds)",
            )
        with col3:
            rho_user = st.number_input(
                "Density (g/cm³)", min_value=0.1, max_value=30.0, value=6.0, step=0.1,
                help="ρ from DFT or experiment",
            )

        section("03 · PREDICT")
        predict_button = st.button("◈ PREDICT ZT PROXY", use_container_width=True)

        if predict_button:
            with st.spinner("Featurizing and predicting…"):
                try:
                    pred = predict_single_material(
                        formula_user, bg_user, fe_user, rho_user,
                        cfg.epsilon, results["rf"]["pipeline"]
                    )

                    # Display results
                    st.markdown("---")
                    section("PREDICTION RESULT")

                    r1, r2, r3, r4 = st.columns(4)
                    with r1:
                        st.markdown(tile("FORMULA", pred["formula"], ""), unsafe_allow_html=True)
                    with r2:
                        st.markdown(tile("ZT_PROXY", f"{pred['ZT_proxy']:.4f}",
                                         "heuristic units", "warn"), unsafe_allow_html=True)
                    with r3:
                        st.markdown(tile("PREDICTED", f"{pred['predicted_log']:.3f}",
                                         "log(1+ZT_proxy)", "accent"), unsafe_allow_html=True)
                    with r4:
                        # Percentile in the ranked dataset
                        all_preds = results["rf"]["preds_per_seed"].mean(axis=0)
                        pct = (all_preds < pred["predicted_log"]).sum() / len(all_preds) * 100
                        st.markdown(tile("PERCENTILE", f"{pct:.1f}%",
                                         f"vs. {len(all_preds):,} MP materials", "info"),
                                    unsafe_allow_html=True)

                    st.markdown(f"""
                    <div class='proxy-warn'>
                    <strong>INTERPRETATION</strong>&nbsp; Your material's predicted log(1 + ZT_proxy) is
                    {pred['predicted_log']:.3f}, placing it at the <strong>{pct:.1f}th percentile</strong>
                    of the {len(all_preds):,} Materials Project entries in this run.
                    This is a <em>heuristic screening score</em>, not a real ZT.
                    The proxy rewards narrow band gap, strong bonding (large |ΔH_f|), and low density.
                    <br/><br/>
                    <strong>Next steps:</strong> If this score is promising (≥80th percentile), run
                    proper DFT transport (BoltzTraP2 / AMSET) to compute the real Seebeck coefficient,
                    electrical conductivity, and lattice thermal conductivity. Only then can you
                    calculate a real ZT.
                    </div>
                    """, unsafe_allow_html=True)

                    with st.expander("Show Magpie feature vector"):
                        st.dataframe(pred["X"], use_container_width=True)

                except Exception as e:
                    st.error(f"Prediction failed: {e}")


# ============================================================================
# TAB: CANDIDATES
# ============================================================================
with tab_candidates:
    section("ROBUST CANDIDATES · SORTED BY SEED-STABILITY (RF ONLY)")
    st.markdown("""
    <div style='padding:10px 14px;border-left:3px solid #6ee7a8;background:rgba(110,231,168,0.04);font-family:JetBrains Mono,monospace;font-size:11px;color:#a7b3bd;margin-bottom:14px'>
    <strong style='color:#6ee7a8'>RF-ONLY</strong> Candidate rankings come from the Random Forest
    20-seed sweep, matching the manuscript §2.1.6 pipeline.
    </div>
    """, unsafe_allow_html=True)

    threshold = st.slider(
        "Minimum stability score", 0.0, 1.0, float(cfg.stability_threshold), step=0.05,
    )

    merged = freq_df.merge(
        df_ranked[["material_id", "formula_pretty", "band_gap", "density",
                   "formation_energy_per_atom", "energy_above_hull", "n_elements",
                   "ZT_proxy", "pred_mean", "pred_std"]],
        on="material_id", how="left",
    )
    merged = merged[
        (merged["stability_score"] >= threshold) &
        (merged["band_gap"] >= 0.2)
    ].copy()
    merged = merged.sort_values(["stability_score", "pred_mean"], ascending=[False, False])

    if len(merged) == 0:
        st.warning("No candidates above this threshold.")
    else:
        st.markdown(f"**{len(merged)} candidates found.**")

        display = merged[[
            "material_id", "formula_pretty", "stability_score", "count",
            "pred_mean", "pred_std", "band_gap", "density",
            "formation_energy_per_atom", "energy_above_hull", "ZT_proxy",
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
            display.style.format({
                "stability": "{:.2f}", "pred_mean(log)": "{:.3f}", "pred_std": "{:.3f}",
                "E_g (eV)": "{:.3f}", "ρ (g/cm³)": "{:.2f}",
                "Δh_f (eV/at)": "{:.3f}", "E_hull (eV/at)": "{:.4f}",
                "ZT_proxy": "{:.4f}",
            }).background_gradient(subset=["stability"], cmap="Greens"),
            use_container_width=True, height=520,
        )


# ============================================================================
# TAB: OVERVIEW
# ============================================================================
with tab_overview:
    section("PROXY DISTRIBUTION")
    fig = px.histogram(df_ranked, x="ZT_proxy", nbins=60, log_x=True,
                       color_discrete_sequence=["#6ee7a8"])
    fig.update_layout(
        plot_bgcolor="#11161a", paper_bgcolor="#11161a",
        font=dict(family="JetBrains Mono", color="#a7b3bd"),
        xaxis=dict(gridcolor="#2a343d"), yaxis=dict(gridcolor="#2a343d"),
        height=320, xaxis_title="ZT_proxy", yaxis_title="count",
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# TAB: EXPORT
# ============================================================================
with tab_export:
    section("DOWNLOAD ARTIFACTS")
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        export_df = df_feat.copy()
        for c in export_df.columns:
            if export_df[c].dtype == "object":
                export_df[c] = export_df[c].astype(str)
        export_df.to_excel(writer, sheet_name="Dataset", index=False)
        rf_metrics.to_excel(writer, sheet_name="RF_metrics", index=False)
        freq_df.to_excel(writer, sheet_name="Stability", index=False)
        robust = freq_df[freq_df["stability_score"] >= cfg.stability_threshold].merge(
            df_feat, on="material_id", how="left"
        )
        for c in robust.columns:
            if robust[c].dtype == "object":
                robust[c] = robust[c].astype(str)
        robust.to_excel(writer, sheet_name="Robust_Candidates", index=False)
    buf.seek(0)

    st.download_button(
        "↓ FINAL_ML_OUTPUT.xlsx", data=buf,
        file_name="FINAL_ML_OUTPUT.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )


# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown(
    """
    <div style='font-family:JetBrains Mono,monospace;font-size:10px;color:#6b7681;letter-spacing:0.15em;text-align:center;padding:10px 0'>
    THERMOELECTRIC SCREENING TERMINAL · V2.1 · CIF PREDICTION ENABLED ·
    ZT<sub>PROXY</sub> = E<sub>G</sub> · |ΔH<sub>F</sub>| / (ρ + ε)
    </div>
    """,
    unsafe_allow_html=True,
)
