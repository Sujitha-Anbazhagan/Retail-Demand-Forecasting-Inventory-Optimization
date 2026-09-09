import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from google.cloud import bigquery
from google.oauth2 import service_account
import os

KEY_PATH = r"C:\Users\Suji\.gcp\key.json"
PROJECT_ID = "steady-webbing-507608-i8"
DATASET_ID = "retail_forecasting"
N_ITEMS = 50
FORECAST_DAYS = 28

OUT_DIR = "outputs/lightgbm"
os.makedirs(OUT_DIR, exist_ok=True)

credentials = service_account.Credentials.from_service_account_file(KEY_PATH)
client = bigquery.Client(credentials=credentials, project=PROJECT_ID)

print(f"Pulling data for top {N_ITEMS} item-store combos...")
query = f"""
    with top_combos as (
        select item_id, store_id
        from `{PROJECT_ID}.{DATASET_ID}.fct_daily_sales`
        group by item_id, store_id
        order by sum(sales) desc
        limit {N_ITEMS}
    )
    select f.*
    from `{PROJECT_ID}.{DATASET_ID}.fct_daily_sales` f
    join top_combos t
        on f.item_id = t.item_id and f.store_id = t.store_id
    order by f.item_id, f.store_id, f.date
"""
df = client.query(query).to_dataframe()
print(f"Pulled {df.shape[0]:,} rows.")

df["date"] = pd.to_datetime(df["date"])

print("Engineering features (lags, rolling averages, calendar)...")
df = df.sort_values(["item_id", "store_id", "date"])

for lag in [1, 7, 14, 28]:
    df[f"lag_{lag}"] = df.groupby(["item_id", "store_id"])["sales"].shift(lag)

for window in [7, 28]:
    df[f"rolling_mean_{window}"] = (
        df.groupby(["item_id", "store_id"])["sales"]
        .transform(lambda s: s.shift(1).rolling(window).mean())
    )

df["has_event"] = (df["event_name_1"] != "none").astype(int)

feature_cols = [
    "lag_1", "lag_7", "lag_14", "lag_28",
    "rolling_mean_7", "rolling_mean_28",
    "wday", "month", "sell_price", "has_event",
    "snap_CA", "snap_TX", "snap_WI",
]
df_model = df.dropna(subset=feature_cols + ["sales"]).copy()
print(f"Rows after dropping NaNs from lag features: {df_model.shape[0]:,}")

cutoff_date = df_model["date"].max() - pd.Timedelta(days=FORECAST_DAYS)
train = df_model[df_model["date"] <= cutoff_date]
test = df_model[df_model["date"] > cutoff_date]
print(f"Train rows: {train.shape[0]:,}, Test rows: {test.shape[0]:,}")

X_train, y_train = train[feature_cols], train["sales"]
X_test, y_test = test[feature_cols], test["sales"]

print("Training LightGBM model...")
model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=31, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
preds = np.clip(preds, 0, None)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print(f"\nTest RMSE: {rmse:.3f}")

test_results = test[["item_id", "store_id", "date", "sales"]].copy()
test_results["predicted_sales"] = preds
test_results.to_csv(f"{OUT_DIR}/lightgbm_predictions.csv", index=False)
print(f"Saved predictions to {OUT_DIR}/lightgbm_predictions.csv")

importance = pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
importance.to_csv(f"{OUT_DIR}/feature_importance.csv", index=False)
print(f"\nFeature importance:\n{importance}")

print("\nDone.")
