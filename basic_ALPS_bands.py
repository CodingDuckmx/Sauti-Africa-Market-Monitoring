import os
import pandas as pd
import psycopg2

from dotenv import load_dotenv, find_dotenv
from scipy import stats
from sklearn.linear_model import LinearRegression

from draft_functions import *

load_dotenv()

# First I will work on wholesale prices only.

def pull_product_info(product_name, market_id, source_id, currency_code):

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

        mode, cfd = prepare_data_to_ALPS(pd.DataFrame(data))

        forecasted_prices = []

        stop_0 = cfd.index[round(len(cfd) *.8)]
        baseset = cfd.loc[:stop_0].copy()

        for i in range(0,(len(cfd)- len(baseset))+1):
            
            # print(stop_0 + datetime.timedelta(weeks=i))
            

            # Baseline
            workset = cfd.loc[:stop_0 + datetime.timedelta(weeks=i)].copy()

            # In what week are we?
            workset['week'] = workset.index.week

            # Build the dummy variables for each week. 
            workset = workset.join(pd.get_dummies(workset['week']))
            workset = workset.drop(labels=['week'], axis=1)

            features = workset.columns[1:]
            target = workset.columns[0]

            X = workset[features]
            y = workset[target]

            reg = LinearRegression()

            reg = reg.fit(X,y)

            next_week = cfd.loc[stop_0 + datetime.timedelta(weeks=1)]

            raw_next_week = [next_week.values[0]] + [0 if i != next_week.name.week else 1 for i in range(53)]

            np.array(raw_next_week[1:]).reshape(1,-1)

            forecasted_prices.append(reg.predict(np.array(raw_next_week[1:]).reshape(1,-1))[0])

        errorstable = cfd.loc[stop_0:]
        errorstable['forecast'] = forecasted_prices
        
        # print(build_bands_wfp_forecast(errorstable))

        wfp_forecast = build_bands_wfp_forecast(errorstable).iloc[1:,:].reset_index()

        try:


            # Stablishes connection with our db.

            connection = psycopg2.connect(user=os.environ.get('eleph_db_user'),
                                        password=os.environ.get('eleph_db_password'),
                                        host=os.environ.get('eleph_db_host'),
                                        port=os.environ.get('eleph_db_port'),
                                        database=os.environ.get('eleph_db_name'))

            
            # Create the cursor.

            cursor = connection.cursor()

            # for row in wfp_forecast.values.tolist():
                
            #     vector = (product_name, market_id, source_id, currency_code,row[0].strftime('%Y-%m-%d'),
            #             row[1],row[6],row[2], 'ALPS', 
            #             datetime.date(datetime.datetime.today().year, datetime.datetime.today().month, datetime.datetime.today().day).strftime('%Y-%m-%d'),
            #             row[7], row[8], row[9])

            #     print(vector)

            #     query_insert_results ='''
            #                         INSERT INTO product_clean_wholesale_info (
            #                         product_name,
            #                         market_id,
            #                         source_id,
            #                         currency_code,
            #                         date_price,
            #                         observed_price,
            #                         observed_class,
            #                         used_model,
            #                         date_run_model,
            #                         normal_band_limit,
            #                         stress_band_limit,
            #                         alert_band_limit
            #                         )
            #                         VALUES (
            #                             %s,
            #                             %s,
            #                             %s,
            #                             %s,
            #                             %s,
            #                             %s,
            #                             %s,
            #                             %s,
            #                             %s,
            #                             %s,
            #                             %s,
            #                             %s
            #                         )
            #     '''

            #     cursor.execute(query_insert_results, vector)

            #     connection.commit()


        except (Exception, psycopg2.Error) as error:
            print('Error pulling the data.')

        finally:

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
                                    INSERT INTO product_clean_wholesale_info (
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


            if (connection):
                cursor.close()
                connection.close()





if __name__ == "__main__":

    # for testing propourses:
    product_name= 'Maize'
    market_id = 'Kampala : UGA'
    source_id = '1'
    currency_code = 'UGX'
    mode_price = 'wholesale_observed_price'

    pull_product_info(product_name, market_id, source_id, currency_code)