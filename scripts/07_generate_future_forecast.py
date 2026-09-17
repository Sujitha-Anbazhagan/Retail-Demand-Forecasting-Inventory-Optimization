import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
from prophet import Prophet

KEY_PATH = r"C:\Users\Suji\.gcp\key.json"
PROJECT_ID = "steady-webbing-507608-i8"
DATASET_ID = "retail_forecasting"
RAW_DATASET_ID = "retail_forecasting_raw"
N_ITEMS = 5
FORECAST_DAYS = 30

credentials = service_account.Credentials.from_service_account_file(KEY_PATH)
client = bigquery.Client(credentials=credentials, project=PROJECT_ID)

print("Finding top-selling items...")
top_items_query = f"""
    select item_id, store_id, sum(sales) as total_sales
    from `{PROJECT_ID}.{DATASET_ID}.fct_daily_sales`
    group by item_id, store_id
    order by total_sales desc
    limit {N_ITEMS}
"""
top_items = client.query(top_items_query).to_dataframe()

all_future_rows = []

for _, row in top_items.iterrows():
    item_id = row["item_id"]
    store_id = row["store_id"]
    print(f"\nForecasting {item_id} @ {store_id} for next {FORECAST_DAYS} days...")

    query = f"""
        select date, sales
        from `{PROJECT_ID}.{DATASET_ID}.fct_daily_sales`
        where item_id = "{item_id}" and store_id = "{store_id}"
        order by date
    """
    hist = client.query(query).to_dataframe()
    hist = hist.rename(columns={"date": "ds", "sales": "y"})
    hist["ds"] = pd.to_datetime(hist["ds"])

    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.fit(hist)

    future = model.make_future_dataframe(periods=FORECAST_DAYS)
    forecast = model.predict(future)

    last_hist_date = hist["ds"].max()
    future_only = forecast[forecast["ds"] > last_hist_date][["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    future_only["item_id"] = item_id
    future_only["store_id"] = store_id
    future_only = future_only.rename(columns={"ds": "date", "yhat": "predicted_sales", "yhat_lower": "predicted_lower", "yhat_upper": "predicted_upper"})
    all_future_rows.append(future_only)

combined = pd.concat(all_future_rows, ignore_index=True)
combined["model"] = "prophet_future"
print(f"\nTotal future forecast rows: {combined.shape[0]:,}")

table_ref = f"{PROJECT_ID}.{RAW_DATASET_ID}.future_forecasts"
job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
job = client.load_table_from_dataframe(combined, table_ref, job_config=job_config)
job.result()
print(f"Uploaded to {table_ref}")
print("\nDone. This is a genuine forward-looking forecast beyond the dataset's last date.")
