import pandas as pd

DATA_DIR = "m5_data"

print("Loading calendar.csv...")
calendar = pd.read_csv(f"{DATA_DIR}/calendar.csv")

print("Loading sell_prices.csv...")
prices = pd.read_csv(f"{DATA_DIR}/sell_prices.csv")

print("Loading sales_train_validation.csv (this is the big one, may take a minute)...")
sales = pd.read_csv(f"{DATA_DIR}/sales_train_validation.csv")

datasets = {"calendar": calendar, "prices": prices, "sales": sales}

for name, df in datasets.items():
    print(f"\n{'=' * 50}")
    print(f"{name.upper()}")
    print(f"{'=' * 50}")
    print(f"Shape: {df.shape}")
    print(f"\nColumns and dtypes:\n{df.dtypes}")
    print(f"\nMemory usage: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    if len(nulls) > 0:
        print(f"\nColumns with nulls:\n{nulls}")
    else:
        print("\nNo nulls found.")
    print(f"\nFirst 3 rows:\n{df.head(3)}")

print(f"\n{'=' * 50}")
print("SPECIFIC CHECKS")
print(f"{'=' * 50}")
print(f"\nCalendar date range: {calendar['date'].min()} to {calendar['date'].max()}")
print(f"Unique items in sales: {sales['item_id'].nunique()}")
print(f"Unique stores in sales: {sales['store_id'].nunique()}")
print(f"Unique states in sales: {sales['state_id'].nunique()}")
print(f"Unique item-store combos in prices: {prices[['item_id', 'store_id']].drop_duplicates().shape[0]}")
print("\nDone. Review the output above for any red flags before Day 5 cleaning.")
