import datetime
import numpy as np
import pandas as pd
import psycopg2

from scipy import stats
from sqlalchemy import create_engine

from statsmodels.tsa.stattools import adfuller

from v2_functions_and_classes import make_dictionaries_possible_dataframes, drop_stats_results_to_df


if __name__ == "__main__":
    

    product_market_pair, descriptions_retail, descriptions_wholesale = make_dictionaries_possible_dataframes()
    drop_stats_results_to_df(descriptions_retail, 'retail_stats')
    drop_stats_results_to_df(descriptions_wholesale, 'wholesale_stats')