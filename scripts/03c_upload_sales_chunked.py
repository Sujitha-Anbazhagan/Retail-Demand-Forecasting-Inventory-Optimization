import pyarrow.parquet as pq
from google.cloud import bigquery
from google.oauth2 import service_account

KEY_PATH = r"C:\Users\Suji\.gcp\key.json"
PROJECT_ID = "steady-webbing-507608-i8"
DATASET_ID = "retail_forecasting_raw"
LOCATION = "US"
PARQUET_PATH = "m5_data/cleaned/sales_long.parquet"
CHUNK_SIZE = 2_000_000

credentials = service_account.Credentials.from_service_account_file(KEY_PATH)
client = bigquery.Client(credentials=credentials, project=PROJECT_ID, location=LOCATION)

table_id = f"{PROJECT_ID}.{DATASET_ID}.sales_long"

parquet_file = pq.ParquetFile(PARQUET_PATH)
total_rows = parquet_file.metadata.num_rows
print(f"Total rows to upload: {total_rows:,}")
print(f"Uploading in chunks of {CHUNK_SIZE:,} rows...\n")

rows_done = 0
first_chunk = True

for batch in parquet_file.iter_batches(batch_size=CHUNK_SIZE):
    df_chunk = batch.to_pandas()
    write_disposition = "WRITE_TRUNCATE" if first_chunk else "WRITE_APPEND"
    job_config = bigquery.LoadJobConfig(write_disposition=write_disposition)
    job = client.load_table_from_dataframe(df_chunk, table_id, job_config=job_config)
    job.result()
    rows_done += len(df_chunk)
    pct = rows_done / total_rows * 100
    print(f"  Uploaded {rows_done:,} / {total_rows:,} rows ({pct:.1f}%)")
    first_chunk = False

table = client.get_table(table_id)
print(f"\nDone: {table_id} now has {table.num_rows:,} rows.")
