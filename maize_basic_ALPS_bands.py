import os
import pandas as pd
import psycopg2

from dotenv import load_dotenv, find_dotenv
from scipy import stats
from sklearn.linear_model import LinearRegression

from functions_and_classes import *

load_dotenv()


def populate_product_wholesale_bands():


    # What markets are vialables?

    pctwo_retail, pctwo_wholesale = possible_maize_markets()


    markets_with_problems = []


    # First I will work on wholesale prices only.

    for i in range(len(pctwo_wholesale)):

        product_name = pctwo_wholesale[i][1]
        market_id = pctwo_wholesale[i][2]
        source_id = pctwo_wholesale[i][3]
        currency_code = pctwo_wholesale[i][4]

        print(market_id)

        market_with_problems = historic_ALPS_bands(product_name, market_id, source_id, currency_code)

        if market_with_problems:
            markets_with_problems.append(market_with_problems)


    print(markets_with_problems)




if __name__ == "__main__":

    populate_product_wholesale_bands() 

    # product_name = 'Maize'
    # market_id = 'Kampala : UGA'
    # source_id = 1
    # currency_code = 'UGX'

    # historic_ALPS_bands(product_name, market_id, source_id, currency_code)
