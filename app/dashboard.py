import streamlit as st
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account

KEY_PATH = r"C:\Users\Suji\.gcp\key.json"
PROJECT_ID = "steady-webbing-507608-i8"
DATASET_ID = "retail_forecasting_raw"
TABLE_ID = "forecast_outputs"

st.set_page_config(page_title="Retail Demand Forecasting", layout="wide")

@st.cache_resource
def get_bigquery_client():
    credentials = service_account.Credentials.from_service_account_file(KEY_PATH)
    return bigquery.Client(credentials=credentials, project=PROJECT_ID)

@st.cache_data(ttl=600)
def load_forecast_data():
    client = get_bigquery_client()
    query = f"""
        select *
        from `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`
        order by item_id, store_id, date
    """
    return client.query(query).to_dataframe()

st.title("Retail Demand Forecasting & Inventory Optimization")
st.caption("M5 Walmart dataset - Prophet + LightGBM forecasts")

with st.spinner("Loading forecast data from BigQuery..."):
    df = load_forecast_data()

st.success(f"Connected. Loaded {df.shape[0]:,} rows from BigQuery.")

st.subheader("Raw forecast data (preview)")
st.dataframe(df.head(20))

st.subheader("Quick stats")
col1, col2, col3 = st.columns(3)
col1.metric("Total rows", f"{df.shape[0]:,}")
col2.metric("Unique items", df["item_id"].nunique())
col3.metric("Models used", df["model"].nunique())
