import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from google.cloud import bigquery
from google.oauth2 import service_account

PROJECT_ID = "steady-webbing-507608-i8"
DATASET_ID = "retail_forecasting_raw"
TABLE_ID = "forecast_outputs"
FUTURE_TABLE_ID = "future_forecasts"

STOCKOUT_RISK_RATIO = 1.3

st.set_page_config(page_title="Retail Demand Forecasting", layout="wide")
@st.cache_resource
def get_bigquery_client():
    try:
        has_secrets = "gcp_service_account" in st.secrets
    except Exception:
        has_secrets = False

    if has_secrets:
        credentials = service_account.Credentials.from_service_account_info(
            dict(st.secrets["gcp_service_account"])
        )
    else:
        KEY_PATH = r"C:\Users\Suji\.gcp\streamlit-reader-key.json"
        credentials = service_account.Credentials.from_service_account_file(KEY_PATH)
    return bigquery.Client(credentials=credentials, project=PROJECT_ID)

@st.cache_data(ttl=600)
def load_forecast_data():
    client = get_bigquery_client()
    query = f"""
        select * from `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`
        order by item_id, store_id, date
    """
    return client.query(query).to_dataframe()
 
 
@st.cache_data(ttl=600)
def load_future_forecast_data():
    client = get_bigquery_client()
    query = f"""
        select * from `{PROJECT_ID}.{DATASET_ID}.{FUTURE_TABLE_ID}`
        order by item_id, store_id, date
    """
    return client.query(query).to_dataframe()
 
 
st.title("Retail Demand Forecasting & Inventory Optimization")
st.caption("M5 Walmart dataset - Prophet + LightGBM forecasts")
 
with st.spinner("Loading forecast data from BigQuery..."):
    df = load_forecast_data()
    future_df = load_future_forecast_data()
 
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
kpi4.metric("Items at stockout risk", at_risk_count,
            help=f"Items where predicted demand exceeds actual by more than {STOCKOUT_RISK_RATIO}x")
 
st.divider()
 
# ---- Sidebar filters ----
st.sidebar.header("Filters")
 
category_options = ["All"] + sorted(
    df["item_id"].str.extract(r"^([A-Za-z]+_\d)")[0].dropna().unique().tolist()
)
selected_category = st.sidebar.selectbox("Category", category_options)
 
df_by_category = df if selected_category == "All" else df[df["item_id"].str.startswith(selected_category)]
 
store_options = sorted(df_by_category["store_id"].unique()) if not df_by_category.empty else sorted(df["store_id"].unique())
selected_store = st.sidebar.selectbox("Store", store_options)
 
items_in_store = sorted(df_by_category[df_by_category["store_id"] == selected_store]["item_id"].unique())
if not items_in_store:
    items_in_store = sorted(df[df["store_id"] == selected_store]["item_id"].unique())
selected_item = st.sidebar.selectbox("Item", items_in_store)
 
model_options = sorted(df["model"].unique())
selected_model = st.sidebar.selectbox("Model", model_options)
 
filtered = df[
    (df["store_id"] == selected_store)
    & (df["item_id"] == selected_item)
    & (df["model"] == selected_model)
].sort_values("date")
 
# ---- Historical detail view ----
st.subheader(f"Historical detail: {selected_item} @ {selected_store} ({selected_model})")
if filtered.empty:
    st.warning(
        "No data for this combination. Prophet's historical summary only covers "
        "5 specific item-store pairs (not every store for a given item). "
        "Try switching Model to 'lightgbm', which has broader coverage, or pick "
        "a different Store."
    )
elif len(filtered) < 2:
    st.info("Only a single summary data point is available for this selection "
            "(Prophet stores one averaged forecast per item here, not daily values). "
            "Switch to lightgbm in the Model filter to see a full daily chart, or "
            "check the Future Forecast section below for a true forward-looking Prophet view.")
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
    st.caption(
        "Simulate how a price change might affect predicted demand, using a "
        "standard price elasticity approximation (not a live model re-run)."
    )
    price_change_pct = st.slider("Price change (%)", min_value=-30, max_value=30, value=0, step=5,
                                  help="Negative = price drop, positive = price increase")
    ASSUMED_ELASTICITY = -1.5
    baseline_demand = filtered["predicted_sales"].mean()
    demand_multiplier = max(1 + (ASSUMED_ELASTICITY * (price_change_pct / 100)), 0)
    adjusted_demand = baseline_demand * demand_multiplier
 
    wcol1, wcol2, wcol3 = st.columns(3)
    wcol1.metric("Baseline predicted demand", f"{baseline_demand:.1f}")
    wcol2.metric("Adjusted predicted demand", f"{adjusted_demand:.1f}", delta=f"{adjusted_demand - baseline_demand:+.1f}")
    wcol3.metric("Price change applied", f"{price_change_pct:+d}%")
 
    if price_change_pct != 0:
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        bars = ax2.bar(["Baseline", "Adjusted"], [baseline_demand, adjusted_demand], color=["#4C72B0", "#DD8452"])
        ax2.set_ylabel("Predicted demand (avg units)")
        ax2.set_title(f"Demand impact of {price_change_pct:+d}% price change")
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f"{height:.1f}", (bar.get_x() + bar.get_width() / 2, height), ha="center", va="bottom")
        st.pyplot(fig2)
        pct_demand_change = (adjusted_demand - baseline_demand) / baseline_demand * 100 if baseline_demand else 0
        direction = "increase" if pct_demand_change > 0 else "decrease"
        st.caption(
            f"A {abs(price_change_pct)}% price {'drop' if price_change_pct < 0 else 'increase'} "
            f"is estimated to {direction} demand by {abs(pct_demand_change):.1f}%, "
            f"based on the assumed elasticity of {ASSUMED_ELASTICITY}."
        )
    else:
        st.caption("Move the slider above to see the projected demand impact.")
 
# ---- Genuine future forecast section ----
st.divider()
st.subheader("Future Forecast (next 30 days, beyond the dataset)")
st.caption(
    "Unlike the historical detail view above (which compares past predictions to "
    "known actuals), this section shows a genuine forward-looking Prophet forecast "
    "for dates that have not happened yet in the dataset. Available for the top 5 "
    "highest-volume items only."
)

future_item_store_pairs = set(zip(future_df["item_id"], future_df["store_id"]))

if (selected_item, selected_store) in future_item_store_pairs:
    future_filtered = future_df[
        (future_df["item_id"] == selected_item) & (future_df["store_id"] == selected_store)
    ].sort_values("date")
else:
    fallback = future_df[["item_id", "store_id"]].drop_duplicates().iloc[0]
    future_filtered = future_df[
        (future_df["item_id"] == fallback["item_id"]) & (future_df["store_id"] == fallback["store_id"])
    ].sort_values("date")
    st.info(
        f"No future forecast available for {selected_item} @ {selected_store} "
        f"(only specific top-5 item-store combinations have this). "
        f"Showing {fallback['item_id']} @ {fallback['store_id']} instead."
    )

if not future_filtered.empty:
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.plot(future_filtered["date"], future_filtered["predicted_sales"], label="Forecast", color="#55A868")
    ax3.fill_between(
        future_filtered["date"], future_filtered["predicted_lower"], future_filtered["predicted_upper"],
        alpha=0.2, color="#55A868", label="Confidence interval",
    )
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Predicted units sold")
    ax3.set_title("Next 30-day demand forecast")
    ax3.legend()
    fig3.autofmt_xdate()
    st.pyplot(fig3)
 
    fcol1, fcol2 = st.columns(2)
    fcol1.metric("Avg daily predicted demand (next 30 days)", f"{future_filtered['predicted_sales'].mean():.1f}")
    fcol2.metric("Total predicted demand (next 30 days)", f"{future_filtered['predicted_sales'].sum():.0f}")
 