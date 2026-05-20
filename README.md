# Thermoelectric Screening Terminal v3.0 (artifact-based deployment)

**ZERO RAM ISSUES. INSTANT LOAD. FULL 70K MANUSCRIPT RUN.**

This version loads pre-trained artifacts instead of training on Streamlit Cloud.

---

## How this works

1. **Run `generate_artifacts_colab.py` in Google Colab** (10-20 min) → generates 3 files:
   - `trained_pipeline.pkl` (the fitted RF model)
   - `dataset_ranked.csv` (full dataset with predictions)
   - `stability_rankings.csv` (candidate stability scores)

2. **Upload those 3 files to this GitHub repo** (same directory as `app.py`)

3. **Deploy on Streamlit Cloud** → app loads the artifacts on startup, no training needed

4. **Users can:**
   - Browse the pre-computed candidate rankings
   - Upload their own CIF files and get ZT proxy predictions from the trained model

---

## Step-by-step deployment

### Step 1: Generate artifacts in Colab

1. Open Google Colab: <https://colab.research.google.com/>
2. Upload `generate_artifacts_colab.py` or paste the code into a new notebook
3. Run all cells
4. When prompted, enter your Materials Project API key
5. Wait ~10-20 minutes for the full run (processes ~70k materials)
6. Download the 3 output files:
   - `trained_pipeline.pkl` (~15 MB)
   - `dataset_ranked.csv` (~50 MB)
   - `stability_rankings.csv` (~100 KB)

### Step 2: Upload artifacts to GitHub

1. In your GitHub repo, click "Add file" → "Upload files"
2. Drag the 3 files into the upload area
3. Make sure they're in the **same directory** as `app.py` (repo root)
4. Commit the files

Your repo structure should look like:

```
.streamlit/
  config.toml
app.py
requirements.txt
runtime.txt
trained_pipeline.pkl        ← artifact 1
dataset_ranked.csv          ← artifact 2
stability_rankings.csv      ← artifact 3
README.md
LICENSE
.gitignore
```

### Step 3: Deploy on Streamlit Cloud

1. Go to <https://share.streamlit.io>
2. Sign in with GitHub
3. Click "New app"
4. Select your repo, branch `main`, main file `app.py`
5. Click "Deploy"

Streamlit Cloud will install dependencies and load the artifacts. First load takes ~1 minute (downloads the CSV files), then it's instant.

---

## What users see

**Tab 1: 🔮 PREDICT YOUR MATERIAL**
- Upload CIF or enter formula manually
- Enter band gap, formation energy, density
- Get ZT proxy prediction + percentile ranking vs. the full MP dataset

**Tab 2: ◇ CANDIDATES**
- Browse the pre-computed robust candidate list
- Filter by stability threshold
- See all manuscript results

**Tab 3: ▤ OVERVIEW**
- Proxy distribution histogram
- Band gap × density scatter plot

**Tab 4: ↓ EXPORT**
- Download the full dataset + stability rankings as Excel

---

## Why this is better than v2.1

| Feature | v2.1 (training on cloud) | v3.0 (artifact-based) |
| --- | --- | --- |
| RAM usage | 1-2 GB (hits cloud limit) | ~300 MB (artifacts only) |
| Load time | 5-10 min (featurizes 3k-5k) | ~10 sec (loads pickles) |
| Dataset size | 3k-5k (subsampled for RAM) | 70k (full manuscript run) |
| Reproducibility | Approximate (cloud subsample) | Exact (same artifacts as manuscript) |
| CIF prediction | ✅ | ✅ |

---

## Manuscript impact

Include this line in your **Data Availability** section:

> "An interactive Streamlit interface for exploring pre-computed candidate rankings and predicting ZT proxy values for user-uploaded CIF files is available at [your-deployed-URL]. The interface loads the trained Random Forest model from the manuscript's 20-seed stability sweep, enabling readers to test their own materials against our trained pipeline."

The artifact-based deployment is **publication-grade** because:
- ✅ Exact reproducibility (the artifacts ARE your manuscript run)
- ✅ No RAM limitations (cloud-friendly)
- ✅ Interactive validation (reviewers can test their own materials)
- ✅ Full dataset visible (all 70k materials, not a subsample)

---

## Troubleshooting

**"Artifacts not found" error on first load:**
- Make sure the 3 `.pkl` and `.csv` files are in the repo root, not in a subfolder
- Check that file names are exact: `trained_pipeline.pkl`, `dataset_ranked.csv`, `stability_rankings.csv`
- Try a hard refresh (Ctrl+Shift+R) or reboot the app in Streamlit Cloud

**CSV files too large for GitHub (>100 MB):**
- GitHub has a 100 MB file size limit
- If `dataset_ranked.csv` exceeds this, use Git LFS or host on Google Drive
- Update `app.py` line 115 to load from URL:
  ```python
  dataset = pd.read_csv("https://drive.google.com/uc?id=YOUR_FILE_ID")
  ```

**Colab artifact generation fails:**
- Check your MP API key is valid
- If timeout, reduce dataset size by tightening filters (band gap 0.5-1.5 eV)

---

## License

MIT. The ZT proxy is a heuristic. Don't publish ZT claims without proper transport calculations.
