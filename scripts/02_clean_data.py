import pandas as pd
import os

DATA_DIR = "m5_data"
OUT_DIR = "m5_data/cleaned"
os.makedirs(OUT_DIR, exist_ok=True)

print("Cleaning calendar.csv...")
calendar = pd.read_csv(f"{DATA_DIR}/calendar.csv")
calendar["date"] = pd.to_datetime(calendar["date"])

for col in ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]:
    calendar[col] = calendar[col].fillna("none")

int_cols = ["wm_yr_wk", "wday", "month", "year", "snap_CA", "snap_TX", "snap_WI"]
for col in int_cols:
    calendar[col] = pd.to_numeric(calendar[col], downcast="integer")

calendar.to_csv(f"{OUT_DIR}/calendar_clean.csv", index=False)
print(f"Calendar cleaned: {calendar.shape}")

print("\nCleaning sell_prices.csv...")
prices = pd.read_csv(f"{DATA_DIR}/sell_prices.csv")
prices["wm_yr_wk"] = pd.to_numeric(prices["wm_yr_wk"], downcast="integer")
prices["sell_price"] = pd.to_numeric(prices["sell_price"], downcast="float")
prices.to_csv(f"{OUT_DIR}/prices_clean.csv", index=False)
print(f"Prices cleaned: {prices.shape}")

print("\nMelting sales_train_validation.csv to long format (this may take a couple minutes)...")
sales = pd.read_csv(f"{DATA_DIR}/sales_train_validation.csv")

id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
day_cols = [c for c in sales.columns if c.startswith("d_")]

sales_long = sales.melt(
    id_vars=id_cols,
    value_vars=day_cols,
    var_name="d",
    value_name="sales"
)

sales_long["sales"] = pd.to_numeric(sales_long["sales"], downcast="integer")

print(f"Long format shape: {sales_long.shape}")

print("\nMerging with calendar to attach real dates...")
sales_long = sales_long.merge(
    calendar[["d", "date", "wm_yr_wk", "wday", "month", "year",
              "event_name_1", "event_type_1", "snap_CA", "snap_TX", "snap_WI"]],
    on="d",
    how="left"
)

out_path = f"{OUT_DIR}/sales_long.parquet"
sales_long.to_parquet(out_path, index=False)
print(f"\nSaved long-format sales to {out_path}")
print(f"Final shape: {sales_long.shape}")
print(f"\nSample:\n{sales_long.head()}")

print("\nDone. Ready for Day 6.")
