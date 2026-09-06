select
    store_id,
    item_id,
    wm_yr_wk,
    sell_price
from {{ source('raw', 'prices') }}
