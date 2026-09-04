# Week 1 Notes — Data Architecture & ETL

## Datasets
- calendar.csv: 1,969 rows, 14 cols. Date range 2011-01-29 to 2016-06-19.
- sell_prices.csv: 6,841,121 rows, 4 cols. No nulls.
- sales_train_validation.csv: 30,490 rows (item-store combos) x 1,919 cols (wide format, one col per day).

## Cleaning decisions
- calendar.date converted to proper datetime.
- event_name_1/2 and event_type_1/2 nulls filled with "none" (no event that day, not missing data).
- Numeric columns downcast to smaller int/float types to reduce memory.
- sales data melted from wide (1,913 day-columns) to long format: one row per item-store-day.
  Result: 58,327,370 rows.
- Long-format sales merged with calendar to attach real dates, day-of-week, month, year,
  event info, and SNAP flags.
- Saved as parquet instead of CSV due to size (~58M rows) - much faster to read/write.

## Data dictionary (key columns in sales_long.parquet)
- id: unique item-store-validation identifier
- item_id, dept_id, cat_id: product hierarchy
- store_id, state_id: location hierarchy
- d: day index (d_1 to d_1913)
- sales: units sold that day
- date: actual calendar date
- snap_CA/TX/WI: SNAP (food assistance) benefit indicator by state
- event_name_1/2, event_type_1/2: holiday/event info

## Next steps (Week 2)
- Load cleaned data into BigQuery
- Build dbt staging + mart models
- Create weekly/monthly aggregation views
