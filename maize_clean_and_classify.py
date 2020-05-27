import os
import pandas as pd
import psycopg2

from dotenv import load_dotenv, find_dotenv
from scipy import stats
from sklearn.linear_model import LinearRegression

from draft_functions import *

load_dotenv()

# First I will work on wholesale prices only.

def clean_and_classify(product_name, market_id, source_id, currency_code):

    data = None

    try:


        # Stablishes connection with our db.

        connection = psycopg2.connect(user=os.environ.get('eleph_db_user'),
                                      password=os.environ.get('eleph_db_password'),
                                      host=os.environ.get('eleph_db_host'),
                                      port=os.environ.get('eleph_db_port'),
                                      database=os.environ.get('eleph_db_name'))

        
        # Create the cursor.

        cursor = connection.cursor()

        cursor.execute('''
                        SELECT date_price, unit_scale, wholesale_observed_price
                        FROM maize_raw_info
                        WHERE product_name = %s
                        AND market_id = %s
                        AND source_id = %s
                        AND currency_code = %s
        ''', (product_name, market_id, source_id, currency_code))

        data = cursor.fetchall()

    except (Exception, psycopg2.Error) as error:
        print('Error pulling the data.')

    finally:

        if (connection):
            cursor.close()
            connection.close()


    if data:
        
        data = set_columns(data)
        metric, data = basic_cleanning(data)
        data = limit_2019_and_later(data)

        try:


        # Stablishes connection with our db.

            connection = psycopg2.connect(user=os.environ.get('aws_db_user'),
                                        password=os.environ.get('aws_db_password'),
                                        host=os.environ.get('aws_db_host'),
                                        port=os.environ.get('aws_db_port'),
                                        database=os.environ.get('aws_db_name'))

            
            # Create the cursor.

            cursor = connection.cursor()

            cursor.execute('''
                            SELECT date_price, normal_band_limit, stress_band_limit, alert_band_limit
                            FROM product_wholesale_bands
                            WHERE product_name = %s
                            AND market_id = %s
                            AND source_id = %s
                            AND currency_code = %s
            ''', (product_name, market_id, source_id, currency_code))

            bands = cursor.fetchall()

            #### We are assuming all data is in the same metric.####


        except (Exception, psycopg2.Error) as error:
            print('Error pulling the bands.')

        finally:

            if (connection):
                cursor.close()
                connection.close()


        bands = set_columns_bands_df(bands)

        classified = assign_classification(data,bands)

        classified = classified.values.tolist()
        

        # we will be dropping the classification values into the db.


        try:


            # Stablishes connection with our db.

            connection = psycopg2.connect(user=os.environ.get('aws_db_user'),
                                        password=os.environ.get('aws_db_password'),
                                        host=os.environ.get('aws_db_host'),
                                        port=os.environ.get('aws_db_port'),
                                        database=os.environ.get('aws_db_name'))

                
                # Create the cursor.

            cursor = connection.cursor()

            for j in range(len(classified)):

                vector = (product_name, market_id, source_id, currency_code, classified[j][0],
                        classified[j][2],classified[j][3],classified[j][4])

                print(vector)

                query_drop_classification_labels = '''
                                INSERT INTO product_clean_wholesale_info (
                                product_name,
                                market_id,
                                source_id,
                                currency_code,
                                date_price,
                                observed_price,
                                observed_class,
                                stressness
                )
                                VALUES(
                                    %s,
                                    %s,
                                    %s,
                                    %s,
                                    %s,
                                    %s,
                                    %s,
                                    %s
                                );
                '''

                cursor.execute(query_drop_classification_labels,vector)

                connection.commit()




        except (Exception, psycopg2.Error) as error:
            print('Error dropping the labels.')

        finally:

            if (connection):
                cursor.close()
                connection.close()





if __name__ == "__main__":
    
    # for testing propourses:
    product_name= 'Maize'
    market_id = 'Eldoret : KEN'
    source_id = '1'
    currency_code = 'KES'
    mode_price = 'wholesale_observed_price'

    clean_and_classify(product_name, market_id, source_id, currency_code)