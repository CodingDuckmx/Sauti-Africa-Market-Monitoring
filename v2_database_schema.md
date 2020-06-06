

Database schema v 0.2

Reproducable here:
https://dbdiagram.io/d




Table countries {
  id indexes
  country_code varchar(3)
  country_name varchar(99)
}

Table categories {
  id indexes
  category_name varchar(99)
}

Table currencies {
  id indexes
  currency_name varchar(99)
  currency_code varchar(3)
  is_in_uganda boolean
  is_in_kenya boolean
  is_in_congo boolean
  is_in_burundi boolean
  is_in_tanzania boolean
  is_in_south_sudan boolean
  is_in_rwanda boolean
  is_in_malawi boolean
}

Table sources {
  id indexes
  source_name varchar(99)
  is_in_uganda boolean
  is_in_kenya boolean
  is_in_congo boolean
  is_in_burundi boolean
  is_in_tanzania boolean
  is_in_south_sudan boolean
  is_in_rwanda boolean
  is_in_malawi boolean
}

Table markets {
  id indexes
  market_id varchar(99)
  market_name varchar(99)
  country_code varchar(3) [ref: > countries.country_code]
}
Table products {
  id indexes
  product_name varchar(99)
  category_id int [ref: > categories.id]
}

Table raw_table {
  id indexes
  product_name varchar(99) [ref:> products.product_name]
  market_id varchar(99) [ref: > markets.id]
  unit_scale varchar(32)
  source_id int [ref: > sources.id]
  currency_code var(3) [ref: > currencies.currency_code]
  date_price date
  retail_observed_price float8
  wholesale_observed_price float8
}



Table retail_prices {
  id indexes
  product_name varchar(99) [ref:> products.product_name]
  category_name var(99) [ref:> categories.category_name]
  market_id varchar(99) [ref: > markets.market_id]
  market_name varchar(99)
  country_code varchar(3)
  source_id int [ref: > sources.id]
  source_name varchar(99)
  unit_scale varchar(32)
  currency_code var(3) [ref: > currencies.currency_code]
  date_price date
  observed_price float8
  observed_class VARCHAR(9)
  class_method VAR(9)
  forecasted_price float4
  forecasted_class VARCHAR(9)
  forecasting_model VARCHAR(99)
  normal_band_limit float8
  stress_band_limit float8
  alert_band_limit float8
  stressness float8
  date_run_model DATE
}

Table wholesale_prices {
  id indexes
  product_name varchar(99) [ref:> products.product_name]
  category_name var(99) [ref:> categories.category_name]
  market_id varchar(99) [ref: > markets.market_id]
  market_name varchar(99)
  country_code varchar(3)
  source_id int [ref: > sources.id]
  source_name varchar(99)
  unit_scale varchar(32)
  currency_code var(3) [ref: > currencies.currency_code]
  date_price date
  observed_price float8
  observed_class VARCHAR(9)
  class_method VAR(9)
  forecasted_price float4
  forecasted_class VARCHAR(9)
  forecasting_model VARCHAR(99)
  normal_band_limit float8
  stress_band_limit float8
  alert_band_limit float8
  stressness float8
  date_run_model DATE
}

Table pulling_logs {
  id indexes
  pulled_date DATE
  final_id float4
}

Table error_logs {
  id indexes
  product_name varchar(99) 
  market_name varchar(99) 
  country_code varchar(3)
  unit_scale varchar(32)
  source_name varchar(99)
  currency_code var(3)
  date_price date
  retail_observed_price float8
  wholesale_observed_price float8
  mysql_db_id INT
  error_date DATE
  possible_error varchar(99)
}

Table qc_retail{
  product_name varchar(99) [ref:> products.product_name]
  market_id varchar(99) [ref: > markets.market_id]
  source_id int [ref: > sources.id]
  start_date DATE
  end_date DATE
  timeliness float4
  data_length float4
  completeness float4
  duplicates int
  mode_d int
}

Table qc_wholesale{
  product_name varchar(99) [ref:> products.product_name]
  market_id varchar(99) [ref: > markets.market_id]
  source_id int [ref: > sources.id]
  start_date DATE
  end_date DATE
  timeliness float4
  data_length float4
  completeness float4
  duplicates int
  mode_d int
}