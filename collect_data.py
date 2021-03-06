import mysql.connector
import numpy as np
import os
import pandas as pd101
import psycopg2

from dotenv import load_dotenv, find_dotenv

from dictionaries import *


load_dotenv()


# Tables creation

def create_tables():

    ''' Creates the table if it doesn't exists already.'''

    try:

        # Stablishes connection with our db



        connection = psycopg2.connect(user=os.environ.get('db_user'),
                                    password=os.environ.get('db_password'),
                                    host=os.environ.get('db_host'),
                                    port=os.environ.get('db_port'),
                                    database=os.environ.get('db_name'))



        # Create the cursor.

        cursor = connection.cursor()


        query_country_table='''
                            CREATE TABLE IF NOT EXISTS countries (
                                id SERIAL NOT NULL,
                                country_code VARCHAR(3) NOT NULL UNIQUE,
                                country_name CHAR(99),
                                PRIMARY KEY(country_code)
                            );
        '''

        cursor.execute(query_country_table)
        connection.commit()

        query_market_country_table= '''
                            CREATE TABLE IF NOT EXISTS markets (
                                id SERIAL,
                                market_id VARCHAR(99),
                                market_name VARCHAR(99),
                                country_code VARCHAR(3) REFERENCES countries(country_code),
                                PRIMARY KEY (market_id)
                            );
        '''

        cursor.execute(query_market_country_table)
        connection.commit()

        query_currency_table = '''
                            CREATE TABLE IF NOT EXISTS currencies (
                                id SERIAL NOT NULL,
                                currency_name VARCHAR(99),
                                currency_code VARCHAR(3) NOT NULL UNIQUE,
                                is_in_uganda BOOLEAN,
                                is_in_kenya BOOLEAN,
                                is_in_congo BOOLEAN,
                                is_in_burundi BOOLEAN,
                                is_in_tanzania BOOLEAN,
                                is_in_south_sudan BOOLEAN,
                                is_in_rwanda BOOLEAN,
                                is_in_malawi BOOLEAN,
                                PRIMARY KEY(currency_code)
                            );
        '''

        cursor.execute(query_currency_table)
        connection.commit()

        query_sources_table = '''
                            CREATE TABLE IF NOT EXISTS sources (
                                id SERIAL NOT NULL,
                                source_name VARCHAR(99) NOT NULL,
                                is_in_uganda BOOLEAN,
                                is_in_kenya BOOLEAN,
                                is_in_congo BOOLEAN,
                                is_in_burundi BOOLEAN,
                                is_in_tanzania BOOLEAN,
                                is_in_south_sudan BOOLEAN,
                                is_in_rwanda BOOLEAN,
                                is_in_malawi BOOLEAN,
                                PRIMARY KEY (id)
                            );
        '''

        cursor.execute(query_sources_table)
        connection.commit()

        query_categories_table = '''
                            CREATE TABLE IF NOT EXISTS categories (
                                id SERIAL NOT NULL,
                                category_name VARCHAR(99) UNIQUE,
                                PRIMARY KEY(id)
                            );
        '''

        cursor.execute(query_categories_table)
        connection.commit()

        # To be discused. ####

        # query_products_table = '''
        #                     CREATE TABLE IF NOT EXISTS products (
        #                         id SERIAL NOT NULL UNIQUE,
        #                         product_name VARCHAR(99) UNIQUE,
        #                         PRIMARY KEY(id, product_name)
        #                     );
        # '''

        # cursor.execute(query_products_table)
        # connection.commit()


        query_product_category_pair_table = '''
                            CREATE TABLE IF NOT EXISTS prod_cat_pair (
                                id SERIAL NOT NULL,
                                product_name VARCHAR(99) UNIQUE,
                                category_id INT REFERENCES categories(id),
                                PRIMARY KEY(product_name)
                            );
        '''

        cursor.execute(query_product_category_pair_table)
        connection.commit()

        query_product_raw_info_table = '''
                            CREATE TABLE IF NOT EXISTS product_raw_info (
                                product_name VARCHAR(99) REFERENCES prod_cat_pair(product_name),
                                market_id VARCHAR(99) REFERENCES markets(market_id),
                                unit_scale VARCHAR(32),
                                source_id INT REFERENCES sources(id),
                                currency_code VARCHAR(3) REFERENCES currencies(currency_code),
                                date_price DATE,
                                retail_observed_price float4,
                                wholesale_observed_price float4
                            );
        '''

        cursor.execute(query_product_raw_info_table)
        connection.commit()


        query_product_clean_retail_info_table = '''
                            CREATE TABLE IF NOT EXISTS product_clean_retail_info (
                                product_name VARCHAR(99) REFERENCES prod_cat_pair(product_name),
                                market_id VARCHAR(99) REFERENCES markets(market_id),
                                source_id INT REFERENCES sources(id),
                                currency_code VARCHAR(3) REFERENCES currencies(currency_code),
                                date_price DATE,
                                observed_price float4,
                                observed_class VARCHAR(9),
                                forecasted_price_1 float4,
                                forecasted_class_1 VARCHAR(9),
                                forecasted_price_2 float4,
                                forecasted_class_2 VARCHAR(9),
                                forecasted_price_3 float4,
                                forecasted_class_3 VARCHAR(9),
                                forecasted_price_4 float4,
                                forecasted_class_4 VARCHAR(9),
                                used_model VARCHAR(99),
                                date_run_model DATE,
                                normal_band_limit float8,
                                stress_band_limit float8,
                                alert_band_limit float8,
                                stressness float8
                            );
        '''

        cursor.execute(query_product_clean_retail_info_table)
        connection.commit()

        query_product_clean_wholesale_info_table = '''
                            CREATE TABLE IF NOT EXISTS product_clean_wholesale_info (
                                product_name VARCHAR(99) REFERENCES prod_cat_pair(product_name),
                                market_id VARCHAR(99) REFERENCES markets(market_id),
                                source_id INT REFERENCES sources(id),
                                currency_code VARCHAR(3) REFERENCES currencies(currency_code),
                                date_price DATE,
                                observed_price float4,
                                observed_class VARCHAR(9),
                                forecasted_price_1 float4,
                                forecasted_class_1 VARCHAR(9),
                                forecasted_price_2 float4,
                                forecasted_class_2 VARCHAR(9),
                                forecasted_price_3 float4,
                                forecasted_class_3 VARCHAR(9),
                                forecasted_price_4 float4,
                                forecasted_class_4 VARCHAR(9),
                                used_model VARCHAR(99),
                                date_run_model DATE,
                                normal_band_limit float8,
                                stress_band_limit float8,
                                alert_band_limit float8,
                                stressness float8
                            );
        '''

        cursor.execute(query_product_clean_wholesale_info_table)
        connection.commit()

        return 'Success'

    except (Exception, psycopg2.Error) as error:
        print('Error verifying or creating the table.')

    finally:

        if (connection):
            cursor.close()
            connection.close()



def populate_basic_tables():

    '''  Populates the basic tables as countries, categories, currencies,
        cources, product names, markets.
    '''

    try:

        # Stablishes connection with our db



        connection = psycopg2.connect(user=os.environ.get('db_user'),
                                    password=os.environ.get('db_password'),
                                    host=os.environ.get('db_host'),
                                    port=os.environ.get('db_port'),
                                    database=os.environ.get('db_name'))

        print('DB Connection:', connection)

        # Create the cursor.

        cursor = connection.cursor()
        print('DB cursor:', cursor)    

        for country in list(countries.keys()):

            country_code = countries[country]['country_code']
            country_name = countries[country]['country_name']

            # Verfies if the country already exists.

            cursor.execute('''
                        SELECT country_code
                        FROM countries
                        WHERE country_code = %s   
            ''', (country_code,))

            country_exists = cursor.fetchall()

            if not country_exists:

                query_populate_countries = '''
                                    INSERT INTO countries (
                                       country_code,
                                       country_name 
                                    )
                                    VALUES (
                                        %s,
                                        %s
                                    );
                '''

                cursor.execute(query_populate_countries,(country_code, country_name))

                connection.commit()

            else:

                print(country_name, 'already in countries table.')


        for country in list(countries.keys()):

            country_code = countries[country]['country_code']
            country_name = countries[country]['country_name']
            
            for market in markets[country_code]:

                # This market id will prevent to duplicates in case
                # there's a market with the same name in other country.
                market_id = market + ' : ' + country_code

                # Verfies if the market already exists.
                cursor.execute('''
                            SELECT market_id
                            FROM markets
                            WHERE market_id = %s   
                ''', (market_id,))

                market_exists = cursor.fetchall()

                if not market_exists:
                    
                    query_populate_markets = '''
                                        INSERT INTO markets (
                                           market_id,
                                           market_name,
                                           country_code 
                                        )
                                        VALUES (
                                            %s,
                                            %s,
                                            %s
                                        );
                    '''

                    cursor.execute(query_populate_markets,(market_id, market, country_code))

                    connection.commit()

                else:

                    print(market, 'already in markets table for the country', country_name, '.')

        for currency in list(currencies.keys()):

            # Verfies if the currency already exists.
            cursor.execute('''
                        SELECT currency_code
                        FROM currencies
                        WHERE currency_code = %s
            ''', (currency,))

            currency_exists = cursor.fetchall()

            if not currency_exists:
              
                is_in_uganda = False
                is_in_kenya = False
                is_in_congo = False
                is_in_burundi = False
                is_in_tanzania = False
                is_in_south_sudan = False
                is_in_rwanda = False
                is_in_malawi = False


                if 'UGA' in currencies[currency]:

                    is_in_uganda = True

                else:

                    pass
                
                if 'KEN' in currencies[currency]:

                    is_in_kenya = True

                else:

                    pass


                if 'DRC' in currencies[currency]:

                    is_in_congo = True

                else:

                    pass

                if 'BDI' in currencies[currency]:

                    is_in_burundi = True

                else:

                    pass

                if 'TZA' in currencies[currency]:

                    is_in_tanzania = True

                else:

                    pass

                if 'SSD' in currencies[currency]:

                    is_in_south_sudan = True

                else:

                    pass

                if 'RWA' in currencies[currency]:

                    is_in_rwanda = True

                else:

                    pass

                if 'MWI' in currencies[currency]:

                    is_in_malawi = True

                else:

                    pass
                    
                query_populate_currencies = '''
                                    INSERT INTO currencies (
                                    currency_code,
                                    is_in_uganda,
                                    is_in_kenya,
                                    is_in_congo,
                                    is_in_burundi,
                                    is_in_tanzania,
                                    is_in_south_sudan,
                                    is_in_rwanda,
                                    is_in_malawi 
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
                                        %s
                                    );
                '''

                cursor.execute(query_populate_currencies,(currency,is_in_uganda,is_in_kenya,
                                    is_in_congo,is_in_burundi,is_in_tanzania,is_in_south_sudan,
                                    is_in_rwanda,is_in_malawi))

                connection.commit()

            else:

                print(currency, 'already in currencies table.')


        for source in list(sources.keys()):

            # Verfies if the currency already exists.
            cursor.execute('''
                        SELECT source_name
                        FROM sources
                        WHERE source_name = %s
            ''', (source,))

            source_exists = cursor.fetchall()

            if not source_exists:
              
                is_in_uganda = False
                is_in_kenya = False
                is_in_congo = False
                is_in_burundi = False
                is_in_tanzania = False
                is_in_south_sudan = False
                is_in_rwanda = False
                is_in_malawi = False


                if 'UGA' in sources[source]:

                    is_in_uganda = True

                else:

                    pass
                
                if 'KEN' in sources[source]:

                    is_in_kenya = True

                else:

                    pass


                if 'DRC' in sources[source]:

                    is_in_congo = True

                else:

                    pass

                if 'BDI' in sources[source]:

                    is_in_burundi = True

                else:

                    pass

                if 'TZA' in sources[source]:

                    is_in_tanzania = True

                else:

                    pass

                if 'SSD' in sources[source]:

                    is_in_south_sudan = True

                else:

                    pass

                if 'RWA' in sources[source]:

                    is_in_rwanda = True

                else:

                    pass

                if 'MWI' in sources[source]:

                    is_in_malawi = True

                else:

                    pass
                       
                query_populate_sources = '''
                                    INSERT INTO sources (
                                    source_name,
                                    is_in_uganda,
                                    is_in_kenya,
                                    is_in_congo,
                                    is_in_burundi,
                                    is_in_tanzania,
                                    is_in_south_sudan,
                                    is_in_rwanda,
                                    is_in_malawi 
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
                                        %s
                                    );
                '''

                cursor.execute(query_populate_sources,(source,is_in_uganda,is_in_kenya,
                                    is_in_congo,is_in_burundi,is_in_tanzania,is_in_south_sudan,
                                    is_in_rwanda,is_in_malawi))

                connection.commit()

            else:

                print(source, 'already in sources table.')


        for cat in prod_categories:

             # Verfies if the category already exists.

            cursor.execute('''
                        SELECT category_name
                        FROM categories
                        WHERE category_name = %s   
            ''', (cat,))

            category_exists = cursor.fetchall()



            if not category_exists:

                query_populate_categories = '''
                                    INSERT INTO categories (
                                       category_name 
                                    )
                                    VALUES (
                                        %s
                                    );
                '''

                cursor.execute(query_populate_categories,(cat,))

                connection.commit()

            else:

                print(cat, 'already in categories table.')


        for cat in list(products.keys()):           

            for product in products[cat]:

                # Verfies if the product already exists.
                cursor.execute('''
                            SELECT product_name
                            FROM prod_cat_pair
                            WHERE product_name = %s   
                ''', (product,))

                product_exists = cursor.fetchall()
                
                if not product_exists:
                    
                    cursor.execute('''
                                SELECT id
                                FROM categories
                                WHERE category_name = %s   
                    ''', (cat,))

                    category_id = cursor.fetchall()[0][0]

                    query_populate_prod_cat_pair = '''
                                        INSERT INTO prod_cat_pair (
                                           product_name,
                                           category_id 
                                        )
                                        VALUES (
                                            %s,
                                            %s
                                        );
                    '''

                    cursor.execute(query_populate_prod_cat_pair,(product, category_id))

                    connection.commit()

                else:

                    print(product, 'already in products table.')



    except (Exception, psycopg2.Error) as error:
        print('Error inserting or verifying the sources value.')

    finally:

        if (connection):
            cursor.close()
            connection.close()
            print('Connection closed.')


def populate_product_raw_table():

    '''  Pulls the raw data in puts it in our database.
    '''
    try:

        # Stablishes connection with our db



        connection = psycopg2.connect(user=os.environ.get('db_user'),
                                    password=os.environ.get('db_password'),
                                    host=os.environ.get('db_host'),
                                    port=os.environ.get('db_port'),
                                    database=os.environ.get('db_name'))

        # Create the cursor.

        cursor = connection.cursor()

        # Pulls the list of products.
        cursor.execute('''
                    SELECT product_name
                    FROM prod_cat_pair
        ''')

        products_list = [x[0] for x in cursor.fetchall()]

        for product in products_list:

            # Stablishes connection with Sauti's database.

            conn = mysql.connector.connect(user=os.environ.get('sauti_db_user'), password=os.environ.get('sauti_db_password'),host=os.environ.get('sauti_db_host'), database=os.environ.get('sauti_db_name'))

            cur = conn.cursor(dictionary=True)

            cur.execute('''
                    SELECT *
                    FROM platform_market_prices2
                    WHERE product = %s
            ''', (product,))

            rows = cur.fetchall()

            cur.close()
            conn.close()

            for row in rows:

                market = row['market'].lower().capitalize()
                country_code = row['country']
                unit_scale = row['unit']
                source = row['source']
                currency = row['currency']
                date_price = row['date'].strftime('%Y-%m-%d')
                retail_observed_price = row['retail']
                wholesale_observed_price = row['wholesale']

                # Verfies the market already exists in our database
                # and pull its market_id.

                cursor.execute('''
                        SELECT market_id
                        FROM markets
                        WHERE market_name = %s 
                        AND country_code = %s
                ''', (market,country_code))

                # TO DO Pop an error if the market_id does not exist.

                market_id = cursor.fetchall()[0][0]

                cursor.execute('''
                        SELECT id
                        FROM sources
                        WHERE source_name = %s
                ''', (source,))

                # TO DO Pop an error if the market_id does not exist.

                source_id = cursor.fetchall()[0][0]

                vector = (product,market_id,unit_scale,source_id, currency,date_price, retail_observed_price,wholesale_observed_price)

                cursor.execute('''
                        SELECT product_name
                        FROM product_raw_info
                        WHERE product_name = %s 
                        AND market_id = %s 
                        AND unit_scale = %s 
                        AND source_id = %s 
                        AND currency_code = %s 
                        AND date_price = %s 
                ''', (product,market_id,unit_scale,source_id, currency,date_price))                

                result = cursor.fetchall()

                if not result: 

                    query_insert_product_info = '''
                                        INSERT INTO product_raw_info (
                                        product_name,
                                        market_id,
                                        unit_scale,
                                        source_id,
                                        currency_code,
                                        date_price,
                                        retail_observed_price,
                                        wholesale_observed_price
                                        )
                                        VALUES (
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

                    cursor.execute(query_insert_product_info,vector)

                    connection.commit()

                else:

                    pass

                del result


    except (Exception, psycopg2.Error) as error:
        print('Error inserting or verifying the values.')

    finally:

        if (connection):
            cursor.close()
            connection.close()
            print('Connection closed.')











############################################################################

############################# DRAFT Section ################################

############################################################################



# Sources
# Uganda_sources = list(set(Uganda['source']))

# We have three sources in Uganda: 'EAGC-RATIN', 'Farmgain', 'InfoTrade'.

# Uganda_currencies = list(set(Uganda['currency']))

# Focusing on EAGC-RATIN first and KES currency only.

# Uganda_markets = list(set(Uganda['market']))

# Uganda_product_cat = list(set(Uganda['product_cat']))

# Uganda_products = list(set(Uganda['product']))



# Populate the dictionary with product category.

# general_dict = {cat:np.nan for cat in Uganda_product_cat}



# for market in Uganda_markets:
#     for prod_cat in list(set(Uganda[Uganda['market'] == market]['product_cat'])):
#         if list(set(Uganda[Uganda['market'] == market]['product_cat'])) != []:
#             general_dict[market] = {prod_cat:np.nan}


# for market in Uganda_markets:
#     for prod_category in Uganda_product_cat:
#         for product in list(set(Uganda[(Uganda['market'] == market) & (Uganda['product_cat'] == prod_category)]['product'])):
#             if list(set(Uganda[(Uganda['market'] == market) & (Uganda['product_cat'] == prod_category)]['product'])) != []:
#                 general_dict[market][prod_category] = {product: np.nan}


# for market in Uganda_markets[:1]:
#     for prod_category in Uganda_product_cat[:1]:
#         for product in Uganda_products[:10]:
#             if list(Uganda[(Uganda['market'] == market) & (Uganda['product_cat'] == prod_category) & (Uganda['product'] == product)]['id']) != []:
#                 dataframe = Uganda[(Uganda['market'] == market) & (Uganda['product_cat'] == prod_category) & (Uganda['product'] == product) & (Uganda['currency'] == 'KES')][['id','source', 'date', 'retail', 'wholesale', 'unit']]
#                 dataframe = dataframe.drop(labels=list(dataframe[dataframe.duplicated(['date'],keep='last')].index), axis=0)
#                 general_dict[market][prod_category][product] = dataframe.to_dict()
#                 del dataframe


# print(general_dict)



if __name__ == "__main__":
    # create_tables()
    # populate_basic_tables()
    populate_product_raw_table()