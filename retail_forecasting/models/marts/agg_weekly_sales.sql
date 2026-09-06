select
    item_id,
    dept_id,
    cat_id,
    store_id,
    state_id,
    date_trunc(date, week) as week_start,
    sum(sales) as total_sales,
    avg(sell_price) as avg_sell_price
from {{ ref("fct_daily_sales") }}
group by 1,2,3,4,5,6
