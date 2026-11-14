# AgroVista real-data upgrade: source and preparation guide

The bundled `demo_agricultural_dataset.csv` remains only a software demonstration dataset. It is not converted into or relabelled as real observations. Use the pipeline below to prepare an evidence-based Indian crop production + weather dataset.

## 1. Crop production: official Government of India APY data

Start with the Directorate of Economics & Statistics (DES), Ministry of Agriculture and Farmers Welfare:

- District-wise, season-wise crop production statistics (area and production by district, crop, season and year): https://www.data.gov.in/catalog/district-wise-season-wise-crop-production-statistics-0
- DES area, production and yield reports: https://data.desagri.gov.in/website/crops-apy-report-web
- UPAg reports include district-level complete APY and time-series reports: https://www.upag.gov.in/primary-estimate-report

Download the CSV from the official portal and save it as `data/raw/india_apy.csv`. Check the data dictionary and units. The preparation script supports common field names; if the portal's current CSV uses different labels, update the `ALIASES` mapping in `scripts/prepare_real_data.py`.

Example (from project root):

```powershell
python scripts/prepare_real_data.py --input data/raw/india_apy.csv --area-unit ha --production-unit tonnes
```

Use `--area-unit thousand_ha` or `--production-unit thousand_tonnes` only when the source metadata explicitly states those units. Yield is derived as tonnes divided by hectares when a yield column is absent. The script writes `data/processed/agrovista_real_apy.csv` and records source provenance.

## 2. Weather: NASA POWER

NASA POWER provides daily meteorological and solar time-series data through its Daily API: https://power.larc.nasa.gov/docs/services/api/temporal/daily/ . The pipeline downloads rainfall, mean/max/min temperature and relative humidity for each coordinate supplied by you.

1. Copy `data/locations_template.csv` to `data/locations.csv`.
2. Replace the example locations with the exact district centroids or representative coordinates you choose. Do not assume a city coordinate represents an entire district without documenting that approximation.
3. Run:

```powershell
python scripts/download_nasa_power_weather.py --locations data/locations.csv --start-year 2001 --end-year 2025
```

4. Join weather to APY records:

```powershell
python scripts/merge_apy_weather.py
```

The resulting `data/processed/agrovista_real_enriched.csv` can be uploaded using AgroVista's Import Dataset control. Check weather coverage and missing values before modelling. The current weather aggregation uses calendar-month groupings as a practical seasonal proxy; adjust season month windows to match the official crop-season definitions in your study region.

## 3. Soil and market data (optional next research extensions)

- ISRIC SoilGrids offers global soil-property maps (including pH, organic carbon, texture and total nitrogen) at 250 m resolution; its REST API is currently reported as paused, so use the documented WCS/WebDAV or Google Earth Engine access routes where suitable: https://docs.isric.org/globaldata/soilgrids/ .
- FAOSTAT provides international agricultural statistics and bulk/API access: https://www.fao.org/faostat/ .
- For Indian market prices, use official AGMARKNET / e-NAM or government price datasets and preserve market, commodity, grade, unit and date. Do not merge price data with APY without harmonising crop names, units and geography.

## 4. Research integrity and modelling

- Preserve original downloaded files under `data/raw/`; do not edit them manually.
- Keep source URL, download date, licence/terms, unit definitions and transformation steps in your project log.
- Crop APY is annual/seasonal administrative statistics, not farm-level sensor data. Do not claim it supports farm-level recommendations or individual-farm prediction.
- NASA POWER values are gridded estimates, not local weather-station observations. Document coordinate choices and spatial mismatch.
- Avoid random train/test splits when predicting future years. Use a chronological holdout (train on earlier years, test on later years) and, where feasible, hold out districts to test geographic generalisation.
- Never join on crop/year alone when state/district/season keys are available. Check duplicates and join coverage.
- Do not treat missing soil, irrigation, NPK, disease, profit or farm-level fields as zero. Keep them missing and exclude unsupported analyses.
- Report MAE, RMSE, R², sample size, target units, split strategy, confidence intervals where possible, and baseline comparisons. Results are not valid until run on the downloaded real data.
