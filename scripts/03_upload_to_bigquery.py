import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account

KEY_PATH = r"C:\Users\Suji\.gcp\key.json"
PROJECT_ID = "steady-webbing-507608-i8"
DATASET_ID = "retail_forecasting_raw"
LOCATION = "US"
CLEANED_DIR = "m5_data/cleaned"

credentials = service_account.Credentials.from_service_account_file(KEY_PATH)
client = bigquery.Client(credentials=credentials, project=PROJECT_ID, location=LOCATION)

dataset_ref = bigquery.Dataset(f"{PROJECT_ID}.{DATASET_ID}")
dataset_ref.location = LOCATION
client.create_dataset(dataset_ref, exists_ok=True)
print(f"Dataset {DATASET_ID} ready.")

def upload_table(df, table_name):
    table_id = f"{PROJECT_ID}.{DATASET_ID}.{table_name}"
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    print(f"\nUploading {table_name} ({df.shape[0]:,} rows)...")
    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()
    table = client.get_table(table_id)
    print(f"Done: {table_id} now has {table.num_rows:,} rows.")

calendar = pd.read_csv(f"{CLEANED_DIR}/calendar_clean.csv")
calendar["date"] = pd.to_datetime(calendar["date"])
upload_table(calendar, "calendar")

prices = pd.read_csv(f"{CLEANED_DIR}/prices_clean.csv")
upload_table(prices, "prices")

sales_long = pd.read_parquet(f"{CLEANED_DIR}/sales_long.parquet")
upload_table(sales_long, "sales_long")

print("\nAll tables uploaded. Check BigQuery console to confirm.")
