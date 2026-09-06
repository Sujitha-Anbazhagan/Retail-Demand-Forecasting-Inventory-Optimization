select
    s.id,
    s.item_id,
    s.dept_id,
    s.cat_id,
    s.store_id,
    s.state_id,
    s.date,
    s.sales,
    s.wday,
    s.month,
    s.year,
    s.event_name_1,
    s.event_type_1,
    s.snap_CA,
    s.snap_TX,
    s.snap_WI,
    p.sell_price
from {{ ref("stg_sales") }} s
left join {{ ref("stg_prices") }} p
    on s.item_id = p.item_id
    and s.store_id = p.store_id
    and s.wm_yr_wk = p.wm_yr_wk
