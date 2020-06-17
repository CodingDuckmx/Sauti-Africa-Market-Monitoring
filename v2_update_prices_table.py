import os
import pandas as pd
import psycopg2

from dotenv import load_dotenv, find_dotenv
from scipy import stats
from sklearn.linear_model import LinearRegression

from v2_dictionaries_and_lists import *
from v2_functions_and_classes import *

load_dotenv()

# First I will work on wholesale prices only.


def update_prices_table():


    # Stablishes connection with our db.

    connection = psycopg2.connect(user=os.environ.get('aws_db_user'),
                                    password=os.environ.get('aws_db_password'),
                                    host=os.environ.get('aws_db_host'),
                                    port=os.environ.get('aws_db_port'),
                                    database=os.environ.get('aws_db_name'))

    
    # Create the cursor.

    cursor = connection.cursor()


    # What markets are vialable?

    cursor.execute('''
                SELECT DISTINCT product_name, market_id, source_id, currency_code
                FROM wholesale_bands
                ''')

    wholesale_markets = cursor.fetchall()

    cursor.execute('''
                SELECT DISTINCT product_name, market_id, source_id, currency_code
                FROM retail_bands
                ''')

    retail_markets = cursor.fetchall()

    cursor.close()
    connection.close()

    for i in range(len(wholesale_markets)):

        product_name = wholesale_markets[i][0]
        market_id = wholesale_markets[i][1]
        source_id = wholesale_markets[i][2]
        currency_code = wholesale_markets[i][3]

        product_wholesale_clean_and_classify(product_name, market_id, source_id, currency_code)

    print('Retail markets:')

    for i in range(len(retail_markets)):

        product_name = retail_markets[i][0]
        market_id = retail_markets[i][1]
        source_id = retail_markets[i][2]
        currency_code = retail_markets[i][3]

        product_retail_clean_and_classify(product_name, market_id, source_id, currency_code)


if __name__ == "__main__":
    
    # product_retail_clean_and_classify('Morogoro Rice','Arusha : TZA',1,'KES')
    # product_wholesale_clean_and_classify('Morogoro Rice','Arusha : TZA',1,'KES')
    update_prices_table()