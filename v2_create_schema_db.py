import mysql.connector
import numpy as np
import os
import psycopg2

from dotenv import load_dotenv, find_dotenv

from v2_dictionaries_and_lists import *


load_dotenv()

############################################################################################################

'''Verify the credentials before running deployment. '''

############################################################################################################




# Tables creation

def create_tables():

    ''' Creates the table if it doesn't exists already.'''

    try:

        # Stablishes connection with our db



        connection = psycopg2.connect(user=os.environ.get('aws_db_user'),
                                    password=os.environ.get('aws_db_password'),
                                    host=os.environ.get('aws_db_host'),
                                    port=os.environ.get('aws_db_port'),
                                    database=os.environ.get('aws_db_name'))



        # Create the cursor.

        cursor = connection.cursor()


        query_countries_table='''
                            CREATE TABLE IF NOT EXISTS countries (
                                id SERIAL NOT NULL,
                                country_code VARCHAR(3) NOT NULL UNIQUE,
                                country_name CHAR(99),
                                PRIMARY KEY(country_code)
                            );
        '''

        cursor.execute(query_countries_table)
        connection.commit()

        query_markets_table= '''
                            CREATE TABLE IF NOT EXISTS markets (
                                id SERIAL NOT NULL,
                                market_id VARCHAR(99),
                                market_name VARCHAR(99),
                                country_code VARCHAR(3) REFERENCES countries(country_code),
                                PRIMARY KEY (market_id)
                            );
        '''

        cursor.execute(query_markets_table)
        connection.commit()

        query_currencies_table = '''
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

        cursor.execute(query_currencies_table)
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


        query_products_table = '''
                            CREATE TABLE IF NOT EXISTS products (
                                id SERIAL NOT NULL,
                                product_name VARCHAR(99) UNIQUE,
                                category_id INT REFERENCES categories(id),
                                PRIMARY KEY(product_name)
                            );
        '''

        cursor.execute(query_products_table)
        connection.commit()

        query_raw_table = '''
                            CREATE TABLE IF NOT EXISTS raw_table (
                                id SERIAL NOT NULL,
                                product_name VARCHAR(99) REFERENCES products(product_name),
                                market_id VARCHAR(99) REFERENCES markets(market_id),
                                unit_scale VARCHAR(32),
                                source_id INT REFERENCES sources(id),
                                currency_code VARCHAR(3) REFERENCES currencies(currency_code),
                                date_price DATE,
                                retail_observed_price float4,
                                wholesale_observed_price float4,
                                PRIMARY KEY(id)
                            );
        '''

        cursor.execute(query_raw_table)
        connection.commit()

        # query_bc_retail_prices_table = '''
        #                     CREATE TABLE IF NOT EXISTS bc_retail_prices (
        #                         id SERIAL NOT NULL,
        #                         product_name VARCHAR(99) REFERENCES products(product_name),
        #                         market_id VARCHAR(99) REFERENCES markets(market_id),
        #                         market_name VARCHAR(99),
        #                         country_code VARCHAR(3) REFERENCES countries(country_code),
        #                         unit_scale VARCHAR(32),
        #                         source_id INT REFERENCES sources(id),
        #                         source_name VARCHAR(99),
        #                         currency_code VARCHAR(3) REFERENCES currencies(currency_code),
        #                         date_price DATE,
        #                         observed_price float4,
        #                         PRIMARY KEY(id)
        #                     );
        # '''

        # cursor.execute(query_bc_retail_prices_table)
        # connection.commit()

        # query_bc_wholesale_prices_table = '''
        #                     CREATE TABLE IF NOT EXISTS bc_wholesale_prices (
        #                         id SERIAL NOT NULL,
        #                         product_name VARCHAR(99) REFERENCES products(product_name),
        #                         market_id VARCHAR(99) REFERENCES markets(market_id),
        #                         market_name VARCHAR(99),
        #                         country_code VARCHAR(3) REFERENCES countries(country_code),
        #                         unit_scale VARCHAR(32),
        #                         source_id INT REFERENCES sources(id),
        #                         source_name VARCHAR(99),
        #                         currency_code VARCHAR(3) REFERENCES currencies(currency_code),
        #                         date_price DATE,
        #                         observed_price float4,
        #                         PRIMARY KEY(id)
        #                     );
        # '''

        # cursor.execute(query_bc_wholesale_prices_table)
        # connection.commit()


        query_retail_prices_table = '''
                            CREATE TABLE IF NOT EXISTS retail_prices (
                                id SERIAL NOT NULL,
                                product_name VARCHAR(99) REFERENCES products(product_name),
                                market_id VARCHAR(99) REFERENCES markets(market_id),
                                market_name VARCHAR(99),
                                country_code VARCHAR(3) REFERENCES countries(country_code),
                                source_id INT REFERENCES sources(id),
                                source_name VARCHAR(99),
                                currency_code VARCHAR(3) REFERENCES currencies(currency_code),
                                unit_scale VARCHAR(32),
                                date_price DATE,
                                observed_price float4,
                                observed_class VARCHAR(9),
                                class_method VARCHAR(9),
                                forecasted_price float4,
                                forecasted_class VARCHAR(9),
                                forecasting_model VARCHAR(99),
                                normal_band_limit float8,
                                stress_band_limit float8,
                                alert_band_limit float8,
                                stressness float8,
                                date_run_model DATE,
                                PRIMARY KEY(id)
                            );
        '''

        cursor.execute(query_retail_prices_table)
        connection.commit()

        query_wholesale_prices_table = '''
                            CREATE TABLE IF NOT EXISTS wholesale_prices (
                                id SERIAL NOT NULL,
                                product_name VARCHAR(99) REFERENCES products(product_name),
                                market_id VARCHAR(99) REFERENCES markets(market_id),
                                market_name VARCHAR(99),
                                country_code VARCHAR(3) REFERENCES countries(country_code),
                                source_id INT REFERENCES sources(id),
                                source_name VARCHAR(99),
                                currency_code VARCHAR(3) REFERENCES currencies(currency_code),
                                unit_scale VARCHAR(32),
                                date_price DATE,
                                observed_price float4,
                                observed_class VARCHAR(9),
                                class_method VARCHAR(9),
                                forecasted_price float4,
                                forecasted_class VARCHAR(9),
                                forecasting_model VARCHAR(99),
                                normal_band_limit float8,
                                stress_band_limit float8,
                                alert_band_limit float8,
                                stressness float8,
                                date_run_model DATE,
                                PRIMARY KEY(id)
                            );
        '''

        cursor.execute(query_wholesale_prices_table)
        connection.commit()

        query_retail_bands_table = '''
                    CREATE TABLE IF NOT EXISTS retail_bands (
                        id SERIAL NOT NULL,
                        product_name VARCHAR(99) REFERENCES products(product_name),
                        market_id VARCHAR(99) REFERENCES markets(market_id),
                        source_id INT REFERENCES sources(id),
                        currency_code VARCHAR(3) REFERENCES currencies(currency_code),
                        unit_scale VARCHAR(32),
                        date_price DATE,
                        observed_price float4,
                        observed_class VARCHAR(9),
                        class_method VARCHAR(99),
                        normal_band_limit float8,
                        stress_band_limit float8,
                        alert_band_limit float8,
                        date_run_model DATE,
                        PRIMARY KEY(id)
                    );
        '''

        cursor.execute(query_retail_bands_table)
        connection.commit()

        query_wholesale_bands_table = '''
                    CREATE TABLE IF NOT EXISTS wholesale_bands (
                        id SERIAL NOT NULL,
                        product_name VARCHAR(99) REFERENCES products(product_name),
                        market_id VARCHAR(99) REFERENCES markets(market_id),
                        source_id INT REFERENCES sources(id),
                        currency_code VARCHAR(3) REFERENCES currencies(currency_code),
                        unit_scale VARCHAR(32),
                        date_price DATE,
                        observed_price float4,
                        observed_class VARCHAR(9),
                        class_method VARCHAR(99),
                        normal_band_limit float8,
                        stress_band_limit float8,
                        alert_band_limit float8,
                        date_run_model DATE,
                        PRIMARY KEY(id)
                    );
        '''

        cursor.execute(query_wholesale_bands_table)
        connection.commit()


        query_pulling_logs_table = '''
                    CREATE TABLE IF NOT EXISTS pulling_logs (
                        id SERIAL NOT NULL,
                        pulled_date DATE NOT NULL,
                        final_id float8 NOT NULL,
                        PRIMARY KEY(id)
                    );
        '''

        cursor.execute(query_pulling_logs_table)
        connection.commit()

        query_errors_logs_table = '''
                            CREATE TABLE IF NOT EXISTS error_logs (
                                id SERIAL NOT NULL,
                                product_name VARCHAR(99),
                                market_name VARCHAR(99),
                                country_code VARCHAR(3),
                                unit_scale VARCHAR(32),
                                source_name VARCHAR(99),
                                currency_code VARCHAR(3),
                                date_price DATE,
                                retail_observed_price float4,
                                wholesale_observed_price float4,
                                mysql_db_id INT,
                                error_date DATE,
                                possible_error VARCHAR(99),
                                PRIMARY KEY(id)
                            );
        '''

        cursor.execute(query_errors_logs_table)
        connection.commit()

        # query_qc_retail_table = '''
        #                     CREATE TABLE IF NOT EXISTS qc_retail (
        #                         id SERIAL NOT NULL,
        #                         product_name VARCHAR(99),
        #                         market_id VARCHAR(99),
        #                         source_id INT,
        #                         currency_code VARCHAR(3),
        #                         unit_scale VARCHAR(32),
        #                         start_date DATE,
        #                         end_date DATE,
        #                         timeliness float4,
        #                         data_length float4,
        #                         completeness float4,
        #                         duplicates INT,
        #                         mode_d INT,
        #                         data_points INT,
        #                         PRIMARY KEY(id)
        #                     );
        # '''

        # cursor.execute(query_qc_retail_table)
        # connection.commit()

    #     query_qc_wholesale_table = '''
    #                     CREATE TABLE IF NOT EXISTS qc_wholesale (
    #                         id SERIAL NOT NULL,
    #                         product_name VARCHAR(99),
    #                         market_id VARCHAR(99),
    #                         source_id INT,
    #                         currency_code VARCHAR(3),
    #                         unit_scale VARCHAR(32),
    #                         start_date DATE,
    #                         end_date DATE,
    #                         timeliness float4,
    #                         data_length float4,
    #                         completeness float4,
    #                         duplicates INT,
    #                         mode_d INT,
    #                         data_points INT,
    #                         PRIMARY KEY(id)
    #                     );
    # '''

    #     cursor.execute(query_qc_wholesale_table)
    #     connection.commit()


        # cursor.close()
        # connection.close()

        return 'Success'

    except (Exception, psycopg2.Error) as error:
        print('Error verifying or creating the table.')



    finally:

        if (connection):
            cursor.close()
            connection.close()



def populate_basic_tables():

    '''  Populates the basic tables as countries, categories, currencies,
        sources, product names, markets.
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
  

        for country in list(countries_dict.keys()):

            country_code = countries_dict[country]['country_code']
            country_name = countries_dict[country]['country_name']

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


        for country in list(countries_dict.keys()):

            country_code = countries_dict[country]['country_code']
            country_name = countries_dict[country]['country_name']
            
            for market in markets_country[country_code]:

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

        for currency in list(currencies_country.keys()):

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


                if 'UGA' in currencies_country[currency]:

                    is_in_uganda = True

                else:

                    pass
                
                if 'KEN' in currencies_country[currency]:

                    is_in_kenya = True

                else:

                    pass


                if 'DRC' in currencies_country[currency]:

                    is_in_congo = True

                else:

                    pass

                if 'BDI' in currencies_country[currency]:

                    is_in_burundi = True

                else:

                    pass

                if 'TZA' in currencies_country[currency]:

                    is_in_tanzania = True

                else:

                    pass

                if 'SSD' in currencies_country[currency]:

                    is_in_south_sudan = True

                else:

                    pass

                if 'RWA' in currencies_country[currency]:

                    is_in_rwanda = True

                else:

                    pass

                if 'MWI' in currencies_country[currency]:

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


        for source in list(sources_country.keys()):

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


                if 'UGA' in sources_country[source]:

                    is_in_uganda = True

                else:

                    pass
                
                if 'KEN' in sources_country[source]:

                    is_in_kenya = True

                else:

                    pass


                if 'DRC' in sources_country[source]:

                    is_in_congo = True

                else:

                    pass

                if 'BDI' in sources_country[source]:

                    is_in_burundi = True

                else:

                    pass

                if 'TZA' in sources_country[source]:

                    is_in_tanzania = True

                else:

                    pass

                if 'SSD' in sources_country[source]:

                    is_in_south_sudan = True

                else:

                    pass

                if 'RWA' in sources_country[source]:

                    is_in_rwanda = True

                else:

                    pass

                if 'MWI' in sources_country[source]:

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


        for cat in categories_list:

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
  

        for cat in list(cat_broken_down.keys()):           

            for product in cat_broken_down[cat]:

                # Verfies if the product already exists.
                cursor.execute('''
                            SELECT product_name
                            FROM products
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

                    query_populate_products = '''
                                        INSERT INTO products (
                                            product_name,
                                            category_id 
                                        )
                                        VALUES (
                                            %s,
                                            %s
                                        );
                    '''

                    cursor.execute(query_populate_products,(product, category_id))

                    connection.commit()

                else:

                    print(product, 'already in products table.')

        # cursor.close()
        # connection.close()


    except (Exception, psycopg2.Error) as error:
        print('Error inserting or verifying the sources value.')

    finally:

        if (connection):
            cursor.close()
            connection.close()
            print('Connection closed.')




if __name__ == "__main__":
    create_tables()
    populate_basic_tables()