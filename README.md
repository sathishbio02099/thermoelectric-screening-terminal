# Thermoelectric Screening Terminal — v2.1 (with CIF prediction)

Companion Streamlit interface for:

> Selvam *et al.*, "From Prediction to Physics: A Robust Machine Learning Framework for Thermoelectric Discovery with Explicit First-Principles Validation," submitted to *J. Mater. Chem. A*.

##  What's new in v2.1

**🔮 PREDICT YOUR MATERIAL**

Upload your own CIF file (or enter a chemical formula manually) + input band gap, formation energy, and density → get a ZT proxy prediction from the trained Random Forest model. The app featurizes your composition with the same Magpie descriptors and tells you how your material ranks against the Materials Project dataset.

This feature makes the pipeline **interactive and reusable**. Reviewers and readers can test their own candidate materials against your trained model without running any code locally.

---

## What the pipeline does

1. **Pull** stable narrow-gap crystals from Materials Project (band gap 0.1–2.0 eV, E_hull ≤ 0.03 eV/atom, 2–100 sites/cell)
2. **Score** with the heuristic ZT proxy: `ZT_proxy = E_g · |ΔH_f| / (ρ + ε)`, ε = 10⁻⁶
3. **Featurize** with matminer's Magpie elemental descriptors (~132 features)
4. **Train** Random Forest on 20 manuscript seeds with 80:20 holdout, report R²/MAE/RMSE
5. **Rank** candidates by stability across seeds (≥ 90% threshold)
6. **Predict** on user-uploaded CIF files using the trained model

## What the pipeline is *not*

The ZT proxy is a **heuristic screening descriptor**, not the true thermoelectric figure of merit ZT = S²σT/κ. It does not encode Seebeck coefficient, electrical conductivity, or lattice thermal conductivity. Use rankings to shortlist candidates for proper BoltzTraP / AMSET / Phono3py follow-up.

---

## Run locally

```bash
git clone <your-repo>
cd thermoelectric-app
pip install -r requirements.txt
streamlit run app.py
```

You'll need a free Materials Project API key from <https://next-gen.materialsproject.org/api>.

## Deploy to Streamlit Community Cloud

1. Push this folder to a public GitHub repo.
2. Go to <https://share.streamlit.io> and connect the repo.
3. Set the main file to `app.py`.
4. Deploy.

The app is **memory-optimized for cloud deployment**. Defaults to processing 3,000 materials (vs. the manuscript's ~70,000) to stay under Streamlit Cloud's 1 GB RAM limit. Users can adjust this in the sidebar.

For the full ~70k-row manuscript run, use `MLcode_v2.py` locally on a machine with ≥4 GB RAM.

---

## File map

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit UI with manuscript-aligned pipeline + CIF prediction |
| `requirements.txt` | Pinned Python dependencies |
| `runtime.txt` | Python 3.11 for Streamlit Cloud |
| `.streamlit/config.toml` | Dark theme + server config |
| `README.md` | This file |
| `LICENSE` | MIT |

## Companion script

The standalone Python pipeline `MLcode_v2.py` (same formula, same seeds, same model) is in the parent repo for reviewers who want to run the full pipeline without the UI.

## Why this matters for publication

The **CIF prediction feature** transforms this from a "here's what we did" supplement into a **reusable tool**. Reviewers can upload their own thermoelectric candidates and see how they rank against your trained model. This interactive validation strengthens the manuscript's impact and demonstrates the pipeline's generalizability.

## License

MIT. The ZT proxy is a heuristic. Don't publish ZT claims from this tool without proper transport calculations.
