import pandas as pd
import matplotlib.pyplot as plt
from google.cloud import bigquery
from google.oauth2 import service_account
from prophet import Prophet
import os

KEY_PATH = r"C:\Users\Suji\.gcp\key.json"
PROJECT_ID = "steady-webbing-507608-i8"
DATASET_ID = "retail_forecasting"
N_ITEMS = 5
FORECAST_DAYS = 28

OUT_DIR = "outputs/prophet_plots"
os.makedirs(OUT_DIR, exist_ok=True)

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
print(top_items)

results_summary = []

for _, row in top_items.iterrows():
    item_id = row["item_id"]
    store_id = row["store_id"]
    print(f"\nProcessing {item_id} @ {store_id}...")

    query = f"""
        select date, sales
        from `{PROJECT_ID}.{DATASET_ID}.fct_daily_sales`
        where item_id = "{item_id}" and store_id = "{store_id}"
        order by date
    """
    df = client.query(query).to_dataframe()

    df_prophet = df.rename(columns={"date": "ds", "sales": "y"})
    df_prophet["ds"] = pd.to_datetime(df_prophet["ds"])

    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.fit(df_prophet)

    future = model.make_future_dataframe(periods=FORECAST_DAYS)
    forecast = model.predict(future)

    fig = model.plot(forecast)
    plt.title(f"{item_id} @ {store_id} - {FORECAST_DAYS}-day forecast")
    plot_path = f"{OUT_DIR}/{item_id}_{store_id}.png"
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"Saved plot to {plot_path}")

    last_actual = df_prophet["y"].iloc[-1]
    next_28_avg_pred = forecast["yhat"].iloc[-FORECAST_DAYS:].mean()
    results_summary.append({
        "item_id": item_id,
        "store_id": store_id,
        "last_actual_sales": last_actual,
        "avg_predicted_next_28d": round(next_28_avg_pred, 2),
    })

summary_df = pd.DataFrame(results_summary)
print("\n=== Summary ===")
print(summary_df)
summary_df.to_csv(f"{OUT_DIR}/forecast_summary.csv", index=False)
print(f"\nSaved summary to {OUT_DIR}/forecast_summary.csv")
print("\nDone. Check outputs/prophet_plots/ for the forecast charts.")
