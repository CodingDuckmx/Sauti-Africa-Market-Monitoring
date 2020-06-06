import os
import pandas as pd
import psycopg2

from dotenv import load_dotenv, find_dotenv
from scipy import stats
from sklearn.linear_model import LinearRegression

from v2_dictionaries_and_lists import *
from v2_functions_and_classes import *


load_dotenv()

############################################################################################################

'''Verify the credentials before running deployment. '''

############################################################################################################


def populate_product_table_bands():


    # What markets are vialables?

    pctwo_retail, pctwo_wholesale, _, _ = possible_product_market_pairs()


    markets_with_problems = []


    for i in range(len(pctwo_wholesale)):

        product_name = pctwo_wholesale[i][1]
        market_id = pctwo_wholesale[i][2]
        source_id = pctwo_wholesale[i][3]
        currency_code = pctwo_wholesale[i][4]



        market_with_problems = product_ws_hist_ALPS_bands(product_name, market_id, source_id, currency_code)

        if market_with_problems:
            markets_with_problems.append(market_with_problems)

    for i in range(len(pctwo_retail)):

        product_name = pctwo_retail[i][1]
        market_id = pctwo_retail[i][2]
        source_id = pctwo_retail[i][3]
        currency_code = pctwo_retail[i][4]

        print(market_id)

        market_with_problems = product_retail_historic_ALPS_bands(product_name, market_id, source_id, currency_code)

        if market_with_problems:
            markets_with_problems.append(market_with_problems)





    # print(markets_with_problems)




if __name__ == "__main__":

    populate_product_table_bands() 
    