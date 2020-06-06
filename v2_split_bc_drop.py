import datetime
import numpy as np
import os
import pandas as pd
import psycopg2

from dotenv import load_dotenv, find_dotenv
from scipy import stats

from v2_functions_and_classes import Clean_and_classify_class

load_dotenv()

############################################################################################################

'''Verify the credentials before running deployment. '''

############################################################################################################


def split_basic_clean_drop():

    '''  
    Pulls the raw data, do a basic cleanning and drop the info in the correspondig table.
    '''
    try:

        # Stablishes connection with our db



        connection = psycopg2.connect(user=os.environ.get('aws_db_user'),
                                    password=os.environ.get('aws_db_password'),
                                    host=os.environ.get('aws_db_host'),
                                    port=os.environ.get('aws_db_port'),
                                    database=os.environ.get('aws_db_name'))

        # Create the cursor.

        cursor = connection.cursor()

        cursor.execute('''
                        SELECT DISTINCT product_name, market_id, source_id, currency_code
                        FROM raw_table;
                        ''')

        rows = cursor.fetchall()

        if rows:

            for row in rows:

                product_name = row[0]
                market_id = row[1]
                source_id = row[2]
                currency_code = row[3]

                cursor.execute('''
                                SELECT market_name, country_code
                                FROM markets
                                WHERE market_id = %s;
                ''',(market_id,))

                result = cursor.fetchall()
                market_name = result[0][0]
                country_code = result[0][1]


                cursor.execute('''
                                SELECT source_name
                                FROM sources
                                WHERE id = %s;
                ''',(source_id,))

                source_name = cursor.fetchall()[0][0]

                quoted_product_name = "'%s'" % row[0]
                quoted_market_id = "'%s'" % row[1]
                quoted_currency_code = "'%s'" % row[3]

                # For retail


                query_retail = '''SELECT date_price, unit_scale, retail_observed_price FROM raw_table WHERE product_name = {} AND market_id = {} AND source_id = {} AND currency_code = {}'''.format(quoted_product_name,quoted_market_id,source_id,quoted_currency_code)

                data = pd.read_sql_query(query_retail,connection)

                clean_class = Clean_and_classify_class()
                unit_scale, data = clean_class.basic_cleanning(data)

                if not data.empty:

                    data_dict = data.to_dict()

                    for i in range(len(data)):

                        vector = (product_name, market_id,market_name,country_code,unit_scale,source_id,source_name,currency_code,data_dict['date_price'][i],data_dict['retail_observed_price'][i])

                        cursor.execute('''
                                        SELECT id
                                        FROM retail_prices
                                        WHERE product_name = %s
                                        AND market_id = %s
                                        AND unit_scale = %s
                                        AND source_id = %s
                                        AND currency_code = %s
                                        AND date_price = %s
                                        AND observed_price =%s
                        ''', (product_name, market_id,unit_scale,source_id,currency_code,data_dict['date_price'][i],data_dict['retail_observed_price'][i]))

                        result = cursor.fetchall()

                        if not result:

                            query_insert_vector = '''
                                                    INSERT INTO retail_prices (
                                                    product_name,
                                                    market_id,
                                                    market_name,
                                                    country_code,
                                                    unit_scale,
                                                    source_id,
                                                    source_name,
                                                    currency_code,
                                                    date_price,
                                                    observed_price
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
                                                        %s
                                                    );
                            '''  

                            cursor.execute(query_insert_vector,vector)

                            connection.commit()

                # For wholesale

                query_wholesale = '''SELECT date_price, unit_scale, wholesale_observed_price FROM raw_table WHERE product_name = {} AND market_id = {} AND source_id = {} AND currency_code = {}'''.format(quoted_product_name,quoted_market_id,source_id,quoted_currency_code)

                data = pd.read_sql_query(query_wholesale,connection)

                clean_class = Clean_and_classify_class()
                unit_scale, data = clean_class.basic_cleanning(data)

                if not data.empty:

                    data_dict = data.to_dict()

                    for i in range(len(data)):

                        try:    # A problem was popping up with the combination df['product']=='Kilombero Rice') & (df['market'] == 'Arusha') & (df['country'] == 'TZA')

                            vector = (product_name, market_id,market_name,country_code,unit_scale,source_id,source_name,currency_code,data_dict['date_price'][i],data_dict['wholesale_observed_price'][i])

                            cursor.execute('''
                                            SELECT id
                                            FROM wholesale_prices
                                            WHERE product_name = %s
                                            AND market_id = %s
                                            AND unit_scale = %s
                                            AND source_id = %s
                                            AND currency_code = %s
                                            AND date_price = %s
                                            AND observed_price =%s
                            ''', (product_name, market_id,unit_scale,source_id,currency_code,data_dict['date_price'][i],data_dict['wholesale_observed_price'][i]))

                            result = cursor.fetchall()

                            if not result:

                                query_insert_vector = '''
                                                        INSERT INTO wholesale_prices (
                                                        product_name,
                                                        market_id,
                                                        market_name,
                                                        country_code,
                                                        unit_scale,
                                                        source_id,
                                                        source_name,
                                                        currency_code,
                                                        date_price,
                                                        observed_price
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
                                                            %s
                                                        );
                                '''  

                                cursor.execute(query_insert_vector,vector)

                                connection.commit()

                        except:

                            print('There has been a problem in this combination:')                        
                            print(product_name)
                            print(market_id)
                            print(market_name)
                            print(country_code)
                            print(unit_scale)
                            print(source_id)
                            print(source_name)
                            print(currency_code)

        # cursor.close()
        # connection.close()


    except (Exception, psycopg2.Error) as error:
        print('Error pulling possible combinations.')

    finally:

        if (connection):
            cursor.close()
            connection.close()
            print('Connection closed.')



if __name__ == "__main__":
    
    split_basic_clean_drop()