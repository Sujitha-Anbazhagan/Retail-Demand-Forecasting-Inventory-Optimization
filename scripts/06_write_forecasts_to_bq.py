import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account

KEY_PATH = r"C:\Users\Suji\.gcp\key.json"
PROJECT_ID = "steady-webbing-507608-i8"
DATASET_ID = "retail_forecasting_raw"
TABLE_ID = "forecast_outputs"

credentials = service_account.Credentials.from_service_account_file(KEY_PATH)
client = bigquery.Client(credentials=credentials, project=PROJECT_ID)

print("Loading LightGBM predictions...")
lgbm = pd.read_csv("outputs/lightgbm/lightgbm_predictions.csv")
lgbm["date"] = pd.to_datetime(lgbm["date"])
lgbm["model"] = "lightgbm"
lgbm = lgbm.rename(columns={"sales": "actual_sales", "predicted_sales": "predicted_sales"})
lgbm = lgbm[["item_id", "store_id", "date", "actual_sales", "predicted_sales", "model"]]
print(f"LightGBM rows: {lgbm.shape[0]:,}")

print("Loading Prophet forecast summary...")
prophet_summary = pd.read_csv("outputs/prophet_plots/forecast_summary.csv")
prophet_summary["model"] = "prophet"
prophet_summary = prophet_summary.rename(columns={"last_actual_sales": "actual_sales", "avg_predicted_next_28d": "predicted_sales"})
prophet_summary["date"] = pd.Timestamp.today().normalize()
prophet_summary = prophet_summary[["item_id", "store_id", "date", "actual_sales", "predicted_sales", "model"]]
print(f"Prophet rows: {prophet_summary.shape[0]:,}")

combined = pd.concat([lgbm, prophet_summary], ignore_index=True)
print(f"\nTotal combined rows: {combined.shape[0]:,}")

table_ref = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"
job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
job = client.load_table_from_dataframe(combined, table_ref, job_config=job_config)
job.result()

table = client.get_table(table_ref)
print(f"\nUploaded to {table_ref}: {table.num_rows:,} rows.")
print("\nDone. Your dashboard can now query this table directly.")
