from google.cloud import bigquery
from google.oauth2 import service_account

KEY_PATH = r"C:\Users\Suji\.gcp\key.json"
PROJECT_ID = "steady-webbing-507608-i8"
DATASET_ID = "retail_forecasting_raw"
LOCATION = "US"
PARQUET_PATH = "m5_data/cleaned/sales_long.parquet"

credentials = service_account.Credentials.from_service_account_file(KEY_PATH)
client = bigquery.Client(credentials=credentials, project=PROJECT_ID, location=LOCATION)

table_id = f"{PROJECT_ID}.{DATASET_ID}.sales_long"

job_config = bigquery.LoadJobConfig(
    source_format=bigquery.SourceFormat.PARQUET,
    write_disposition="WRITE_TRUNCATE",
)

print(f"Uploading {PARQUET_PATH} directly to {table_id} (streaming from disk, no pandas)...")

with open(PARQUET_PATH, "rb") as source_file:
    job = client.load_table_from_file(source_file, table_id, job_config=job_config)

job.result()

table = client.get_table(table_id)
print(f"Done: {table_id} now has {table.num_rows:,} rows.")
