import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from google.cloud import bigquery
from google.oauth2 import service_account

KEY_PATH = r"C:\Users\Suji\.gcp\key.json"
PROJECT_ID = "steady-webbing-507608-i8"
DATASET_ID = "retail_forecasting_raw"
TABLE_ID = "forecast_outputs"

STOCKOUT_RISK_RATIO = 1.3

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

st.subheader("Overview")

total_predicted = df["predicted_sales"].sum()
total_actual = df["actual_sales"].sum()
avg_ratio = (df["predicted_sales"] / df["actual_sales"].replace(0, pd.NA)).mean()

at_risk_count = (
    df.assign(ratio=df["predicted_sales"] / df["actual_sales"].replace(0, pd.NA))
    .query("ratio > @STOCKOUT_RISK_RATIO")["item_id"]
    .nunique()
)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Total predicted demand", f"{total_predicted:,.0f}")
kpi2.metric("Total actual sales (test period)", f"{total_actual:,.0f}")
kpi3.metric("Avg predicted/actual ratio", f"{avg_ratio:.2f}")
kpi4.metric("Items at stockout risk", at_risk_count, help=f"Items where predicted demand exceeds actual by more than {STOCKOUT_RISK_RATIO}x")

st.divider()

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

st.subheader(f"Detail view: {selected_item} @ {selected_store} ({selected_model})")

if filtered.empty:
    st.warning("No data for this combination. Try a different filter selection.")
elif len(filtered) < 2:
    st.info("Only a single summary data point is available for this selection "
            "(Prophet stores one averaged forecast per item, not daily values). "
            "Switch to lightgbm in the Model filter to see a full daily chart.")
    st.dataframe(filtered)
else:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(filtered["date"], filtered["actual_sales"], label="Actual", marker="o", markersize=3)
    ax.plot(filtered["date"], filtered["predicted_sales"], label="Predicted", marker="o", markersize=3, linestyle="--")
    ax.set_xlabel("Date")
    ax.set_ylabel("Units sold")
    ax.set_title(f"Actual vs Predicted Sales - {selected_item} @ {selected_store}")
    ax.legend()
    fig.autofmt_xdate()
    st.pyplot(fig)

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", filtered.shape[0])
    col2.metric("Avg actual sales", round(filtered["actual_sales"].mean(), 2))
    col3.metric("Avg predicted sales", round(filtered["predicted_sales"].mean(), 2))

    item_ratio = filtered["predicted_sales"].mean() / max(filtered["actual_sales"].mean(), 1)
    if item_ratio > STOCKOUT_RISK_RATIO:
        st.warning(f"Stockout risk: predicted demand is {item_ratio:.2f}x actual sales for this item.")
    else:
        st.success(f"No significant stockout risk detected (ratio: {item_ratio:.2f}x).")

    with st.expander("View raw data"):
        st.dataframe(filtered)

    st.divider()
    st.subheader("What-if: price scenario")
    st.caption("Simulate how a price change might affect predicted demand, using a standard price elasticity approximation (not a live model re-run).")

    price_change_pct = st.slider("Price change (%)", min_value=-30, max_value=30, value=0, step=5, help="Negative = price drop, positive = price increase")

    ASSUMED_ELASTICITY = -1.5

    baseline_demand = filtered["predicted_sales"].mean()
    demand_multiplier = 1 + (ASSUMED_ELASTICITY * (price_change_pct / 100))
    demand_multiplier = max(demand_multiplier, 0)
    adjusted_demand = baseline_demand * demand_multiplier

    wcol1, wcol2, wcol3 = st.columns(3)
    wcol1.metric("Baseline predicted demand", f"{baseline_demand:.1f}")
    wcol2.metric("Adjusted predicted demand", f"{adjusted_demand:.1f}", delta=f"{adjusted_demand - baseline_demand:+.1f}")
    wcol3.metric("Price change applied", f"{price_change_pct:+d}%")
