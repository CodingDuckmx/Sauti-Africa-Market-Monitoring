import os
import pandas as pd
import psycopg2

from dotenv import load_dotenv, find_dotenv
from scipy import stats
from sklearn.linear_model import LinearRegression

from functions_and_classes import *

load_dotenv()

# First I will work on wholesale prices only.


def populate_pcwi_table():

    # What markets are vialable?

    pctwo_retail, pctwo_wholesale = possible_maize_markets_to_label()

    # print('Wholesale markets:')

    # for i in range(len(pctwo_wholesale)):

    #     product_name = pctwo_wholesale[i][1]
    #     market_id = pctwo_wholesale[i][2]
    #     source_id = pctwo_wholesale[i][3]
    #     currency_code = pctwo_wholesale[i][4]

    #     print(market_id)

    #     wholesale_clean_and_classify(product_name, market_id, source_id, currency_code)

    print('Retail markets:')

    for i in range(len(pctwo_wholesale)):

        product_name = pctwo_wholesale[i][1]
        market_id = pctwo_wholesale[i][2]
        source_id = pctwo_wholesale[i][3]
        currency_code = pctwo_wholesale[i][4]

        print(market_id)

        retail_clean_and_classify(product_name, market_id, source_id, currency_code)


if __name__ == "__main__":
    
    populate_pcwi_table()
   