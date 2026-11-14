# AgroVista — Seasonal Agriculture Intelligence Platform

AgroVista is a full-stack final-year project starter for seasonal agricultural performance analysis. It combines a responsive analytics dashboard, data-quality checks, a transparent Agricultural Performance Index (API), one-way ANOVA, regression model benchmarking, feature importance, and a yield scenario simulator.

> **Important data note:** `data/demo_agricultural_dataset.csv` contains 4,000 **synthetic demonstration records** generated for testing the application. It is not real farm data and must not be presented as empirical research. Import your actual dataset using **Import dataset** in the sidebar. All dataset-dependent metrics and models then recalculate from the uploaded CSV.

## Features
- Executive dashboard with yield, production, profit and composite API.
- Seasonal and crop performance comparisons.
- Water, irrigation, disease/pest risk and resource views.
- Data metadata, missing-cell and duplicate-row checks.
- One-way ANOVA for seasonal yield differences.
- Linear Regression, Random Forest and Gradient Boosting benchmark with MAE, RMSE and R².
- Permutation feature importance and held-out actual/predicted samples in the API response.
- Random Forest yield scenario estimator.
- **Deep-learning lab (PyTorch):** Deep MLP, Residual MLP, feature-vector 1D CNN and dense autoencoder anomaly scoring.
- CSV export of seasonal summary.
- Responsive UI and FastAPI backend.

## Run locally in VS Code (Windows)
1. Install **Python 3.10–3.12** and **Visual Studio Code**.
2. Extract the ZIP and open the `AgroVista` folder in VS Code.
3. Open **Terminal → New Terminal**.
4. Create and activate a virtual environment:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If Python 3.11 is not installed, use `py -3.12` or `py -3.10` instead. If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal and activate again.

5. Install dependencies:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

6. Start the application from the project root:

   ```powershell
   python -m uvicorn backend.main:app --reload
   ```

7. (Optional, for the Deep Learning Lab) install PyTorch: `pip install -r requirements-dl.txt`. For GPU/CUDA, choose the matching install command from https://pytorch.org/get-started/locally/.
8. Open **http://127.0.0.1:8000** in your browser. API documentation is at **http://127.0.0.1:8000/docs**.

## Upgrade: prepare a real India crop + weather dataset

The project now includes a reproducible data-preparation pipeline. The bundled CSV remains synthetic and is not used as evidence. The new pipeline is designed around official district/season/crop/year Area-Production-Yield (APY) records and NASA POWER weather aggregates.

1. Download official APY crop statistics from [data.gov.in](https://www.data.gov.in/catalog/district-wise-season-wise-crop-production-statistics-0), [DES](https://data.desagri.gov.in/website/crops-apy-report-web), or [UPAg](https://www.upag.gov.in/primary-estimate-report). Save the unmodified CSV as `data/raw/india_apy.csv`.
2. Read `docs/REAL_DATA_GUIDE.md` and verify source units.
3. Prepare APY rows:
   ```powershell
   python scripts/prepare_real_data.py --input data/raw/india_apy.csv --area-unit ha --production-unit tonnes
   ```
4. Copy `data/locations_template.csv` to `data/locations.csv`, replace the example rows with your chosen district coordinates, then install the data utility and download NASA POWER weather:
   ```powershell
   pip install -r requirements-data.txt
   python scripts/download_nasa_power_weather.py --locations data/locations.csv --start-year 2001 --end-year 2025
   python scripts/merge_apy_weather.py
   ```
5. Run a chronological holdout evaluation:
   ```powershell
   python scripts/evaluate_temporal.py --input data/processed/agrovista_real_enriched.csv
   ```
6. Start AgroVista and upload `data/processed/agrovista_real_enriched.csv` using **Import dataset**.

The pipeline does not fabricate soil, NPK, irrigation, profit, disease, or farm-level measurements. Read `docs/REAL_DATA_GUIDE.md` for spatial limitations, season definitions, data provenance, and research validation requirements.

## Upload your real dataset
Use **Import dataset** in the left sidebar and choose your CSV. The backend requires a yield target column named `Yield_Tonnes_Ha` or a recognized equivalent such as `Yield`. Other fields are detected using common aliases. For best coverage, use columns corresponding to:
- Season, Crop, State
- Yield_Tonnes_Ha, Production_Tonnes, Farm_Area_Ha
- Profit_INR, Revenue_INR, Cost_INR
- Water_Efficiency, Water_Usage_L
- Disease_Risk_Score, Pest_Risk_Score
- Rainfall_mm, Temperature_C, Humidity_Percent, Soil_pH
- Nitrogen_kg_ha, Phosphorus_kg_ha, Potassium_kg_ha
- Irrigation_Method

The uploaded CSV is held in application memory for the current server process; it is not written to disk. Restarting the server restores the included demo dataset. Do not upload sensitive personal information.

## Methodology notes
- **API formula:** `100 × (0.30Y + 0.25P + 0.20W + 0.25R)`, where Y, P and W are percentile-clipped min-max normalized yield, profit and water efficiency; R is inverse normalized average disease/pest risk. These weights are transparent project assumptions, not an official standard.
- **ANOVA:** tests whether average yield differs among season groups. A p-value below 0.05 is evidence against equal means, not proof of causation. Check assumptions and use a post-hoc test for pairwise comparisons.
- **ML:** uses an 80/20 holdout with random state 42. Derived outcomes (production, profit, revenue, water efficiency) and identifiers are excluded to reduce leakage. Review the feature set for leakage specific to your dataset before reporting results.
- **Prediction:** scenario inputs cover a subset of features; any other model inputs are imputed using dataset medians/modes. Predictions are estimates, not guarantees or causal claims.

## Project structure
```text
AgroVista/
├── backend/main.py
├── frontend/index.html
├── frontend/style.css
├── frontend/app.js
├── data/demo_agricultural_dataset.csv
├── data/raw/ and data/processed/
├── scripts/ (real-data preparation and temporal evaluation)
├── docs/
├── tests/
├── requirements.txt
└── README.md
```

## Before final submission
- Replace synthetic data with official APY/weather data and document its source/licence.
- Validate feature mappings and units against the dataset data dictionary.
- Add post-hoc testing, confidence intervals, residual diagnostics and model cross-validation.
- Add automated tests and capture screenshots of real results.
- Do not claim synthetic demo outputs as research findings.

## Deep-learning lab
Open the **Deep learning lab** navigation item and click **Train neural models**. It runs:
- Deep MLP and Residual MLP regressors for nonlinear tabular yield modelling.
- A 1D CNN over the encoded feature vector (experimental tabular use; not a temporal CNN).
- A dense autoencoder that scores unusual feature patterns using reconstruction error.

The neural models use an 80/20 holdout and report MAE, RMSE and R². They run on CPU by default and may take longer than classical ML. Install PyTorch separately with `pip install -r requirements-dl.txt`. The autoencoder's anomaly scores are not diagnoses of disease or crop failure.

**Sequence-model limitation:** LSTM/GRU should only be used when the data contains genuine chronological observations (for example, repeated monthly sensor readings for each farm). A randomly ordered cross-sectional table does not create a valid time series. Add a date/time field and farm-level sequence identifiers before claiming temporal forecasting.
