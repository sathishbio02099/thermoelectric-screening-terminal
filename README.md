# Thermoelectric Screening Terminal — v2 (manuscript-aligned)

Companion Streamlit interface for:

> Selvam *et al.*, "From Prediction to Physics: A Robust Machine Learning Framework for Thermoelectric Discovery with Explicit First-Principles Validation," submitted to *J. Mater. Chem. A*.

This v2 release replaces the original UI and is aligned exactly with `MLcode_v2.py` and the manuscript text.

---

## What's new vs. v1

| Capability | v1 | v2 |
| --- | --- | --- |
| Stability sweep seeds | 20 random | **20 fixed seeds from the manuscript** |
| Models | Random Forest only | **Random Forest + XGBoost comparison arm** |
| Metrics | R², MAE | **R², MAE, RMSE** |
| Cross-validation | none | **Repeated 5-fold × 3** |
| Diagnostic plots | none | **Parity, residuals, correlation matrix, CV chart** |
| SHAP | summary only | **bar + heatmap + beeswarm summary** |
| Stability threshold | 0.8 | **0.9** (matches §2.1.7) |
| Final candidate filter | — | **band gap ≥ 0.2 added** (matches §2.1.7) |
| Manuscript section cross-references | — | **Visible throughout the UI** |

---

## What the pipeline does

The full workflow from Section 2.1 of the manuscript:

1. **Pull** stable narrow-gap crystals from Materials Project (band gap 0.1–2.0 eV, E_hull ≤ 0.03 eV/atom, 2–100 sites/cell)
2. **Score** with the heuristic ZT proxy: `ZT_proxy = E_g · |ΔH_f| / (ρ + ε)`, ε = 10⁻⁶
3. **Featurize** with matminer's Magpie elemental descriptors (~132 features)
4. **Pipeline**: median imputation → SelectKBest (F-test, k=50) → Random Forest / XGBoost
5. **Validate**: 80:20 holdout + repeated 5-fold CV (3 repeats)
6. **Stabilize**: 20-seed sweep with the manuscript's fixed seed list
7. **Compare**: RF vs XGBoost under identical preprocessing
8. **Explain**: SHAP tree-explainer with bar, heatmap, and beeswarm
9. **Select**: stability ≥ 0.9, E_hull ≤ 0.03, band gap ≥ 0.2

## What the pipeline is *not*

The ZT proxy is a **heuristic screening descriptor**, not the true thermoelectric figure of merit ZT = S²σT/κ. It does not encode Seebeck coefficient, electrical conductivity, or lattice thermal conductivity. Use rankings to shortlist candidates for proper BoltzTraP / AMSET / Phono3py follow-up. The UI repeats this warning everywhere it matters.

---

## Run locally

```bash
git clone <your-repo>
cd thermoelectric-app
pip install -r requirements.txt
streamlit run app.py
```

Open the URL Streamlit prints (default `http://localhost:8501`). You'll need a free Materials Project API key from <https://next-gen.materialsproject.org/api>.

## Deploy to Streamlit Community Cloud

1. Push this folder to a public GitHub repo.
2. Go to <https://share.streamlit.io> and connect the repo.
3. Set the main file to `app.py`.
4. Deploy. Streamlit Cloud auto-installs from `requirements.txt`.

If the build fails on Python version mismatch with pymatgen, add a `runtime.txt` at the repo root containing just:

```
python-3.11
```

---

## File map

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit UI + manuscript-aligned pipeline |
| `requirements.txt` | Pinned Python dependencies including XGBoost |
| `.streamlit/config.toml` | Dark theme + server config |
| `README.md` | This file |
| `LICENSE` | MIT |

## Companion script

The standalone Python pipeline that this UI wraps is `MLcode_v2.py` — same formula, same seeds, same models, same metrics. Reviewers can run either; they produce equivalent results.

## License

MIT. The ZT proxy is a heuristic. Don't publish ZT claims from this tool without proper transport calculations.
