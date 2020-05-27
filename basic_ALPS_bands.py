import os
import pandas as pd
import psycopg2

from dotenv import load_dotenv, find_dotenv
from scipy import stats
from sklearn.linear_model import LinearRegression

from draft_functions import *

load_dotenv()

# First I will work on wholesale prices only.

def historic_ALPS_bands(product_name, market_id, source_id, currency_code):

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

        l4y = last_four_year_truncate(data)

        metric, cfd = prepare_data_to_ALPS(l4y)

        stop_0, forecasted_prices = inmediate_forecast_ALPS_based(cfd)

        errorstable = pd.DataFrame(index=pd.date_range(cfd.loc[stop_0:].index[0],datetime.date(cfd.index[-1].year,cfd.index[-1].month + 1, 1), freq='MS'),
                                columns=['observed_wholesale_price','forecast']) 
        errorstable.iloc[:,0] = None
        errorstable.iloc[:-1,0] =  [x[0] for x in cfd.iloc[len(cfd.loc[:stop_0]):,:].values.tolist()]
        errorstable.iloc[:,1] =  forecasted_prices
        wfp_forecast = build_bands_wfp_forecast(errorstable)

        wfp_forecast = wfp_forecast.reset_index()
        
        try:

            
            # Stablishes connection with our db.

            connection = psycopg2.connect(user=os.environ.get('aws_db_user'),
                                        password=os.environ.get('aws_db_password'),
                                        host=os.environ.get('aws_db_host'),
                                        port=os.environ.get('aws_db_port'),
                                        database=os.environ.get('aws_db_name'))

            
            # Create the cursor.

            cursor = connection.cursor()


            for row in wfp_forecast.values.tolist():
                
                date_price = str(row[0].strftime("%Y-%m-%d"))
                date_run_model = str(datetime.date(datetime.datetime.today().year, datetime.datetime.today().month, datetime.datetime.today().day).strftime("%Y-%m-%d"))
                observed_price = row[1]
                observed_class = row[6]
                used_model =  'ALPS'
                normal_band_limit = round(row[7],4) 
                stress_band_limit = round(row[8],4)
                alert_band_limit = round(row[9],4)


                vector = (product_name,market_id,source_id,currency_code,date_price,
                          observed_price,observed_class,used_model,date_run_model,
                          normal_band_limit,stress_band_limit,alert_band_limit)

                query_insert_results ='''
                                    INSERT INTO product_wholesale_bands (
                                    product_name,
                                    market_id,
                                    source_id,
                                    currency_code,
                                    date_price,
                                    observed_price,
                                    observed_class,
                                    used_model,
                                    date_run_model,
                                    normal_band_limit,
                                    stress_band_limit,
                                    alert_band_limit
                                    )
                                    VALUES (
                                        %s,
                                        %s,
                                        %s,
                                        %s,
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

                cursor.execute(query_insert_results, vector)

                connection.commit()


        except (Exception, psycopg2.Error) as error:
            print('Error pulling the data.')

        finally:

            # for row in wfp_forecast.values.tolist():
               
            #     date_price = str(row[0].strftime("%Y-%m-%d"))
            #     date_run_model = str(datetime.date(datetime.datetime.today().year, datetime.datetime.today().month, datetime.datetime.today().day).strftime("%Y-%m-%d"))
            #     observed_price = row[1]
            #     observed_class = row[6]
            #     used_model =  'ALPS'
            #     normal_band_limit = round(row[7],4) 
            #     stress_band_limit = round(row[8],4)
            #     alert_band_limit = round(row[9],4)

            #     vector = (product_name,market_id,source_id,currency_code)#,date_price,
            #             #   observed_price,observed_class,used_model,date_run_model,
            #             #   normal_band_limit,stress_band_limit,alert_band_limit)

            #     print(vector)

            #     query_insert_results ='''
            #                         INSERT INTO product_clean_wholesale_info (
            #                         product_name,
            #                         market_id,
            #                         source_id,
            #                         currency_code
            #                         )
            #                         VALUES (
            #                             %s,
            #                             %s,
            #                             %s,
            #                             %s
            #                         );
            #     '''

            #     cursor.execute(query_insert_results, vector)
            #     connection.commit()


            if (connection):
                cursor.close()
                connection.close()





if __name__ == "__main__":

    # for testing propourses:
    product_name= 'Maize'
    market_id = 'Mulindi : RWA'
    source_id = '1'
    currency_code = 'KES'
    mode_price = 'wholesale_observed_price'

    historic_ALPS_bands(product_name, market_id, source_id, currency_code)