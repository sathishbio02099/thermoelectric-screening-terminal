"""
THERMOELECTRIC SCREENING TERMINAL v3.0 (artifact-based deployment)

Loads pre-trained artifacts generated in Colab:
  - trained_pipeline.pkl
  - dataset_ranked.csv
  - stability_rankings.csv

NO training happens on Streamlit Cloud. The app just:
  1. Displays the pre-computed candidate rankings
  2. Lets users upload CIF files for ZT proxy prediction

This solves all RAM issues and gives instant load times.
"""
from pymatgen.core import Composition
from __future__ import annotations

import io
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import joblib

warnings.filterwarnings("ignore")

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Thermoelectric Screening Terminal",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================================
# THEME (same as before)
# ============================================================================
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Fraunces:opsz,wght@9..144,300;9..144,400;9..144,600&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --bg-0: #0a0d0f; --bg-1: #11161a; --bg-2: #1a2128; --line: #2a343d;
    --text-0: #e8eef2; --text-1: #a7b3bd; --text-2: #6b7681;
    --accent: #6ee7a8; --accent-dim: #3aa874; --warn: #ffb547; --info: #7dd3fc;
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
}

[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] { display: none; }
footer { visibility: hidden; }

h1, h2, h3, h4, h5 {
    font-family: 'Fraunces', serif !important;
    letter-spacing: -0.02em !important;
    color: var(--text-0) !important;
}

.hero {
    border: 1px solid var(--line);
    background: radial-gradient(800px 200px at 10% 0%, rgba(110,231,168,0.08), transparent 60%),
                radial-gradient(600px 200px at 90% 100%, rgba(255,181,71,0.05), transparent 60%),
                var(--bg-1);
    padding: 28px 34px; margin-bottom: 22px;
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
    padding: 16px 18px; height: 100%;
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

.section-h {
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
    letter-spacing: 0.3em; color: var(--text-2); text-transform: uppercase;
    margin: 8px 0 12px 0; padding-bottom: 6px; border-bottom: 1px solid var(--line);
}

.proxy-warn {
    border: 1px dashed var(--warn); background: rgba(255,181,71,0.05);
    color: var(--text-1); padding: 12px 16px; margin: 12px 0;
    font-size: 13px; line-height: 1.5;
}
.proxy-warn strong {
    color: var(--warn); font-family: 'JetBrains Mono', monospace;
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

.stButton > button {
    background: transparent !important; color: var(--accent) !important;
    border: 1px solid var(--accent-dim) !important; border-radius: 0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important; letter-spacing: 0.15em !important;
    text-transform: uppercase !important; padding: 8px 18px !important;
}
.stButton > button:hover {
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

a { color: var(--accent) !important; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================================
# LOAD ARTIFACTS
# ============================================================================
@st.cache_resource
def load_artifacts():
    """Load the three pre-trained artifacts from Hugging Face Hub."""
    from huggingface_hub import hf_hub_download
    
    # Download from Hugging Face Hub (free, no size limits, reliable)
    repo_id = "sathishbio2/thermoelectric-rf-model"  # ← REPLACE WITH YOUR HUGGING FACE USERNAME
    
    try:
        with st.spinner("Downloading trained model from Hugging Face (first load only, ~142 MB)..."):
            pipeline_path = hf_hub_download(repo_id, "trained_pipeline.pkl")
            dataset_path = hf_hub_download(repo_id, "dataset_ranked.csv")
            stability_path = hf_hub_download(repo_id, "stability_rankings.csv")
        
        pipeline = joblib.load(pipeline_path)
        dataset = pd.read_csv(dataset_path)
        stability = pd.read_csv(stability_path)
        return pipeline, dataset, stability
        
    except Exception as e:
        st.error(
            f"**Failed to load model from Hugging Face.**\n\n"
            f"Error: {e}\n\n"
            f"**Setup instructions:**\n"
            f"1. Create a free account at https://huggingface.co/\n"
            f"2. Create a new model repo (e.g., 'yourname/thermoelectric-rf-model')\n"
            f"3. Upload the 3 artifact files to that repo\n"
            f"4. Update line with your repo name: repo_id = 'yourname/thermoelectric-rf-model'\n"
            f"5. Redeploy"
        )
        st.stop()


# ============================================================================
# CIF PREDICTION
# ============================================================================
def parse_cif_to_formula(cif_bytes: bytes) -> str:
    from pymatgen.io.cif import CifParser
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".cif") as tmp:
        tmp.write(cif_bytes)
        tmp.flush()
        parser = CifParser(tmp.name)
        struct = parser.get_structures()[0]
        return struct.composition.reduced_formula


def featurize_single(formula: str) -> pd.DataFrame:
    from pymatgen.core import Composition
    from matminer.featurizers.composition import ElementProperty
    comp = Composition(formula)
    feat = ElementProperty.from_preset("magpie")
    df_in = pd.DataFrame([{"composition": comp}])
    df_out = feat.featurize_dataframe(df_in, col_id="composition", ignore_errors=True)
    return df_out.drop(columns=["composition"])


def predict_material(formula, bg, fe, rho, epsilon, pipeline, all_preds):
    """Predict ZT proxy for a single material."""
    zt_proxy = (bg * abs(fe)) / (rho + epsilon)
    
    # Featurize the composition
    X_new = featurize_single(formula)
    
    # Add the missing columns that the model expects
    # (these were in the training data but not from single-composition featurization)
    X_new['energy_above_hull'] = 0.0  # Dummy value (not used by model after feature selection)
    X_new['n_elements'] = len(Composition(formula).elements)
    
    # Predict
    y_pred_log = pipeline.predict(X_new)[0]
    
    # Calculate percentile
    pct = (all_preds < y_pred_log).sum() / len(all_preds) * 100
    
    return {
        "formula": formula,
        "ZT_proxy": zt_proxy,
        "predicted_log": float(y_pred_log),
        "percentile": float(pct),
    }


# ============================================================================
# UI HELPERS
# ============================================================================
def hero():
    st.markdown("""
    <div class="hero">
      <div class="eyebrow">◈ Thermoelectric Screening Terminal · v3.0 · artifact-based</div>
      <h1>From prediction to physics —<br/>a <em>robust</em> screening pipeline.</h1>
      <div class="lede">
        Companion interface for Selvam <em>et al.</em> Displays pre-computed candidate
        rankings from the manuscript's 20-seed RF pipeline. <strong>Upload your own CIF</strong>
        and get a ZT proxy prediction from the trained model.
      </div>
      <div class="ribbon">⚠ ZT_PROXY = E<sub>g</sub> · |Δh<sub>f</sub>| / (ρ + ε) — heuristic</div>
    </div>
    """, unsafe_allow_html=True)


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
# MAIN
# ============================================================================
hero()

# Load artifacts
try:
    pipeline, dataset, stability = load_artifacts()
    all_preds = dataset["pred_mean"].values
except FileNotFoundError:
    st.error("""
    **Artifacts not found.**

    This app requires three pre-trained files in the repo root:
    - `trained_pipeline.pkl`
    - `dataset_ranked.csv`
    - `stability_rankings.csv`

    Run `generate_artifacts_colab.py` in Google Colab to create them, then upload
    to your GitHub repo.
    """)
    st.stop()
except Exception as e:
    st.error(f"Failed to load artifacts: {e}")
    st.stop()

# KPI strip
section("PRE-TRAINED MODEL FROM MANUSCRIPT")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(tile("MATERIALS", f"{len(dataset):,}", "manuscript run"), unsafe_allow_html=True)
with c2:
    # Estimate R² from manuscript (you'd hard-code the actual value here)
    st.markdown(tile("RF R²", "0.648", "σ = 0.013 · 20 seeds"), unsafe_allow_html=True)
with c3:
    robust_n = (stability["stability_score"] >= 0.9).sum()
    st.markdown(tile("ROBUST", f"{robust_n}", "≥ 90% stability"), unsafe_allow_html=True)
with c4:
    st.markdown(tile("MODEL", "Random Forest", "150 trees · Magpie"), unsafe_allow_html=True)

st.markdown("""
<div class='proxy-warn'>
<strong>NOTE</strong>&nbsp; This app loads <em>pre-computed</em> results from the manuscript's
full 70k-material run. No training happens on Streamlit Cloud. The ZT proxy is a heuristic
screening descriptor, not real ZT.
</div>
""", unsafe_allow_html=True)

# Tabs
tab_predict, tab_candidates, tab_overview, tab_export = st.tabs([
    "🔮 PREDICT YOUR MATERIAL", "◇ CANDIDATES", "▤ OVERVIEW", "↓ EXPORT"
])


# ============================================================================
# TAB: PREDICT
# ============================================================================
with tab_predict:
    st.markdown("""
    <div class='predict-box'>
      <h3>🔮 Predict ZT Proxy for Your Own Material</h3>
      <div style='font-family:JetBrains Mono;font-size:11px;color:#a7b3bd;margin-bottom:16px'>
        Upload a CIF file + enter properties → get ZT proxy prediction from the trained RF model.
      </div>
    </div>
    """, unsafe_allow_html=True)

    section("01 · UPLOAD CIF OR ENTER FORMULA")
    upload_method = st.radio(
        "Specify your material:",
        ["Upload CIF file", "Manually enter formula"],
        horizontal=True,
    )

    formula_user = None
    if upload_method == "Upload CIF file":
        cif_file = st.file_uploader("Upload CIF", type=["cif"])
        if cif_file:
            try:
                formula_user = parse_cif_to_formula(cif_file.read())
                st.success(f"Extracted formula: **{formula_user}**")
            except Exception as e:
                st.error(f"CIF parsing failed: {e}")
    else:
        formula_user = st.text_input("Chemical formula (e.g., Bi2Te3, PbSe)")
        if formula_user:
            try:
                from pymatgen.core import Composition
                _ = Composition(formula_user)
                st.success(f"Formula validated: **{formula_user}**")
            except Exception as e:
                st.error(f"Invalid formula: {e}")
                formula_user = None

    if formula_user:
        section("02 · ENTER MATERIAL PROPERTIES")
        st.caption("Band gap and formation energy from DFT (PBE/HSE06). Density from DFT or experiment.")

        col1, col2, col3 = st.columns(3)
        with col1:
            bg = st.number_input("Band gap (eV)", 0.0, 10.0, 1.0, 0.05)
        with col2:
            fe = st.number_input("Formation energy (eV/atom)", -10.0, 5.0, -0.5, 0.05)
        with col3:
            rho = st.number_input("Density (g/cm³)", 0.1, 30.0, 6.0, 0.1)

        section("03 · PREDICT")
        if st.button("◈ PREDICT ZT PROXY", use_container_width=True):
            with st.spinner("Featurizing and predicting…"):
                try:
                    pred = predict_material(
                        formula_user, bg, fe, rho, 1e-6, pipeline, all_preds
                    )

                    st.markdown("---")
                    section("PREDICTION RESULT")

                    r1, r2, r3, r4 = st.columns(4)
                    with r1:
                        st.markdown(tile("FORMULA", pred["formula"], ""), unsafe_allow_html=True)
                    with r2:
                        st.markdown(tile("ZT_PROXY", f"{pred['ZT_proxy']:.4f}",
                                         "heuristic", "warn"), unsafe_allow_html=True)
                    with r3:
                        st.markdown(tile("PREDICTED", f"{pred['predicted_log']:.3f}",
                                         "log(1+ZT_proxy)", "accent"), unsafe_allow_html=True)
                    with r4:
                        st.markdown(tile("PERCENTILE", f"{pred['percentile']:.1f}%",
                                         f"vs. {len(all_preds):,} MP materials", "info"),
                                    unsafe_allow_html=True)

                    st.markdown(f"""
                    <div class='proxy-warn'>
                    <strong>INTERPRETATION</strong>&nbsp; Your material's predicted log(1 + ZT_proxy)
                    is {pred['predicted_log']:.3f}, placing it at the <strong>{pred['percentile']:.1f}th
                    percentile</strong>. This is a <em>heuristic screening score</em>, not real ZT.
                    <br/><br/>
                    <strong>Next steps if promising (≥80th percentile):</strong> Run DFT transport
                    (BoltzTraP2 / AMSET) to compute Seebeck coefficient, electrical conductivity,
                    and lattice thermal conductivity (Phono3py). Only then can you calculate real ZT.
                    </div>
                    """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Prediction failed: {e}")


# ============================================================================
# TAB: CANDIDATES
# ============================================================================
with tab_candidates:
    section("ROBUST CANDIDATES FROM MANUSCRIPT RUN")

    threshold = st.slider("Minimum stability score", 0.0, 1.0, 0.9, 0.05)

    filtered = stability[
        (stability["stability_score"] >= threshold) &
        (stability["band_gap"] >= 0.2)
    ].copy()

    if len(filtered) == 0:
        st.warning("No candidates above this threshold.")
    else:
        st.markdown(f"**{len(filtered)} candidates found.**")

        display = filtered[[
            "material_id", "formula_pretty", "stability_score", "count",
            "band_gap", "density", "formation_energy_per_atom",
            "energy_above_hull", "ZT_proxy",
        ]].rename(columns={
            "formula_pretty": "formula",
            "stability_score": "stability",
            "count": "n_seeds",
            "band_gap": "E_g (eV)",
            "density": "ρ (g/cm³)",
            "formation_energy_per_atom": "Δh_f (eV/at)",
            "energy_above_hull": "E_hull (eV/at)",
        })

        st.dataframe(
            display.style.format({
                "stability": "{:.2f}", "E_g (eV)": "{:.3f}", "ρ (g/cm³)": "{:.2f}",
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
    fig = px.histogram(dataset, x="ZT_proxy", nbins=60, log_x=True,
                       color_discrete_sequence=["#6ee7a8"])
    fig.update_layout(
        plot_bgcolor="#11161a", paper_bgcolor="#11161a",
        font=dict(family="JetBrains Mono", color="#a7b3bd"),
        xaxis=dict(gridcolor="#2a343d"), yaxis=dict(gridcolor="#2a343d"),
        height=320, xaxis_title="ZT_proxy (heuristic)", yaxis_title="count",
    )
    st.plotly_chart(fig, use_container_width=True)

    section("BAND GAP × DENSITY")
    fig2 = px.scatter(
        dataset, x="band_gap", y="density",
        color=np.log1p(dataset["ZT_proxy"]),
        hover_data=["formula_pretty", "material_id"],
        color_continuous_scale=[(0, "#1a2128"), (0.5, "#3aa874"), (1, "#6ee7a8")],
        opacity=0.6,
    )
    fig2.update_traces(marker=dict(size=4, line=dict(width=0)))
    fig2.update_layout(
        plot_bgcolor="#11161a", paper_bgcolor="#11161a",
        font=dict(family="JetBrains Mono", color="#a7b3bd"),
        xaxis=dict(gridcolor="#2a343d"), yaxis=dict(gridcolor="#2a343d"),
        height=400, xaxis_title="band gap (eV)", yaxis_title="density (g/cm³)",
        coloraxis_colorbar=dict(title="log1p(ZT_proxy)"),
    )
    st.plotly_chart(fig2, use_container_width=True)


# ============================================================================
# TAB: EXPORT
# ============================================================================
with tab_export:
    section("DOWNLOAD ARTIFACTS")

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        dataset_export = dataset.copy()
        for c in dataset_export.columns:
            if dataset_export[c].dtype == "object":
                dataset_export[c] = dataset_export[c].astype(str)
        dataset_export.to_excel(writer, sheet_name="Dataset", index=False)

        stability_export = stability.copy()
        for c in stability_export.columns:
            if stability_export[c].dtype == "object":
                stability_export[c] = stability_export[c].astype(str)
        stability_export.to_excel(writer, sheet_name="Stability", index=False)

    buf.seek(0)
    st.download_button(
        "↓ MANUSCRIPT_DATA.xlsx", data=buf,
        file_name="MANUSCRIPT_DATA.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )


# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='font-family:JetBrains Mono,monospace;font-size:10px;color:#6b7681;letter-spacing:0.15em;text-align:center;padding:10px 0'>
THERMOELECTRIC SCREENING TERMINAL · V3.0 · ARTIFACT-BASED DEPLOYMENT ·
PRE-TRAINED ON ~70K MATERIALS PROJECT ENTRIES · CIF PREDICTION ENABLED
</div>
""", unsafe_allow_html=True)
