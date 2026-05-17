[README.md](https://github.com/user-attachments/files/27900958/README.md)
# Thermoelectric Screening Terminal

A Streamlit interface around a Materials-Project + Random-Forest pipeline that ranks crystalline materials by a **heuristic ZT proxy**. This is a discovery-stage funnel, not a true ZT predictor.

---

## What it does

1. **Pulls** stable, narrow-gap crystals from the Materials Project (configurable filters: band gap, energy above hull, sites per cell).
2. **Featurizes** with matminer's Magpie elemental descriptors (~132 features).
3. **Scores** each material with a chemistry-aware surrogate target: `ZT_proxy = E_g · |Δh_f| / ρ`.
4. **Trains** a Random Forest on `log1p(ZT_proxy)` across N random seeds (default 20) to assess ranking stability.
5. **Ranks** candidates by how often they appear in the top-K across seeds.
6. **Explains** the model with tree-SHAP attribution.

## What it is *not*

This tool does **not** predict the real thermoelectric figure of merit ZT = S²σT/κ. The proxy is a screening prior — it favors narrow-gap, strongly-bonded, low-density crystals because those are reasonable thermoelectric priors. It does not encode Seebeck coefficient, electronic conductivity, or lattice thermal conductivity. Anything you ship from this tool needs proper BoltzTraP/AMSET/Phono3py follow-up.

The UI repeats this warning everywhere it matters. Please keep that framing when sharing results.

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

For Hugging Face Spaces, Render, or Fly: any Python platform that runs `streamlit run app.py` with `requirements.txt` works. The Materials Project API key is entered at runtime by the user — no need to put it in env vars.

---

## File map

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit UI + refactored pipeline with caching |
| `requirements.txt` | Pinned Python dependencies |
| `.streamlit/config.toml` | Dark theme + server config |
| `README.md` | This file |

## Pipeline notes vs. original `MLcode.py`

- Same proxy formula, same RF + SelectKBest + Magpie features, same N-seed stability sweep.
- Adds prediction-level stability tracking (mean/std of predicted score per material across seeds).
- Uses `@st.cache_data` to avoid re-pulling MP and re-featurizing on every interaction.
- SHAP is on-demand (button) rather than always-run, to keep the page responsive.

## License

Use at your own risk. The proxy is a heuristic. Don't publish ZT claims from this tool without proper transport calculations.
