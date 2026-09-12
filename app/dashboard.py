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

st.sidebar.header("Filters")

store_options = sorted(df["store_id"].unique())
selected_store = st.sidebar.selectbox("Store", store_options)

items_in_store = sorted(df[df["store_id"] == selected_store]["item_id"].unique())
selected_item = st.sidebar.selectbox("Item", items_in_store)

model_options = sorted(df["model"].unique())
selected_model = st.sidebar.selectbox("Model", model_options)

filtered = df[
    (df["store_id"] == selected_store)
    & (df["item_id"] == selected_item)
    & (df["model"] == selected_model)
].sort_values("date")

st.subheader(f"{selected_item} @ {selected_store} ({selected_model})")

if filtered.empty:
    st.warning("No data for this combination. Try a different filter selection.")
else:
    st.dataframe(filtered)

    st.subheader("Quick stats for selection")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", filtered.shape[0])
    col2.metric("Avg actual sales", round(filtered["actual_sales"].mean(), 2))
    col3.metric("Avg predicted sales", round(filtered["predicted_sales"].mean(), 2))
