# Retail Demand Forecasting & Inventory Optimization

An automated analytics pipeline for retail demand forecasting, built on the M5 Forecasting (Walmart) dataset. Covers the full path from raw data to an interactive forecasting dashboard: ETL, cloud data warehousing, dbt transformations, time-series forecasting (Prophet + LightGBM), and a Streamlit dashboard with what-if price scenario simulation.

## Project Overview

- Data source: M5 Forecasting dataset (Walmart historical sales, hierarchical by item/department/category/store/state, ~5 years of daily sales, calendar events, and pricing).
- Warehouse: Google BigQuery
- Transformation: dbt (staging and mart models)
- Forecasting: Facebook Prophet (per-item baseline) and LightGBM (multi-variable, feature-engineered model across many items at once)
- Dashboard: Streamlit, with store/item/model filters, an actual-vs-predicted forecast chart, stockout risk indicators, and a what-if price scenario slider

## Architecture

Raw CSVs (M5 dataset)
  -> Python cleaning and reshaping (wide to long format)
  -> Google BigQuery (raw tables)
  -> dbt (staging models -> fct_daily_sales -> weekly/monthly aggregations)
  -> Forecasting (Prophet + LightGBM) -> forecast_outputs table in BigQuery
  -> Streamlit dashboard (reads from forecast_outputs)

## Setup Instructions

### 1. Clone and set up the environment

git clone https://github.com/Sujitha-Anbazhagan/Retail-Demand-Forecasting-Inventory-Optimization.git
cd Retail-Demand-Forecasting-Inventory-Optimization
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

### 2. Get the dataset

The M5 dataset CSVs are not included in this repo (too large for git). Place the four CSVs (calendar.csv, sales_train_validation.csv, sell_prices.csv, sample_submission.csv) into m5_data/.

### 3. Set up Google Cloud and BigQuery

- Create a Google Cloud project and enable the BigQuery API
- Create a service account with BigQuery Admin role, download its JSON key
- Save the key outside the repo (e.g. ~/.gcp/key.json) and update the KEY_PATH and PROJECT_ID variables in the scripts to match your setup

### 4. Run the pipeline

python scripts/01_load_and_inspect.py
python scripts/02_clean_data.py
python scripts/03_upload_to_bigquery.py
python scripts/03c_upload_sales_chunked.py
cd retail_forecasting
dbt run
cd ..
python scripts/04_prophet_forecast.py
python scripts/05_lightgbm_forecast.py
python scripts/06_write_forecasts_to_bq.py

### 5. Launch the dashboard

streamlit run app/dashboard.py

Opens at http://localhost:8501

## Project Structure

scripts/              ETL, upload, and forecasting scripts
retail_forecasting/   dbt project (staging and mart models)
app/                  Streamlit dashboard
docs/                 Project notes and data dictionary
m5_data/              Raw and cleaned data (gitignored)
outputs/              Generated forecasts and plots (gitignored)

## Key Results

- LightGBM model achieved a test RMSE of approximately 12.8 on daily unit sales (average sales level approximately 44 units/day), with lag features (previous day, previous week) and day-of-week as the strongest predictors.
- Prophet provided per-item baseline forecasts with weekly and yearly seasonality for the top 5 highest-volume items.

## Notes and Limitations

- The dashboard's forecast data currently covers a limited subset of items (top sellers used during model development), not the full approximately 30,490 item-store catalog, to keep the project scope manageable within the timeline.
- The what-if price scenario uses an assumed price elasticity (-1.5) as an approximation rather than a live model re-run, for dashboard performance.

## Author

Sujitha Anbazhagan
