import os
import pandas as pd
import psycopg2

from dotenv import load_dotenv, find_dotenv


load_dotenv()

# First I will work on wholesale prices only.

def populate_maize_table():

    # pull the data from its origin
    
    prices = pd.read_csv('eac-ratin.csv')
    prices.columns = [x.lower() for x in list(prices.columns)]
    
    # drop rows with product not maize

    prices = prices.drop(labels=list(prices[prices['product'] == 'Product'].index), axis=0)
    prices = prices.reset_index(drop=True)

    # Replace countries by its code.

    prices[['country']] = prices[['country']].replace({'Burundi':'BDI','Kenya':'KEN',
                                            'Rwanda':'RWA', 'South Sudan':'SSD',
                                            'Tanzania': 'TZA', 'Uganda': 'UGA'})

    # Set market_id as <<market : country code>>.

    prices['market_id'] = prices['market'] + ' : ' + prices['country']
    
    # We don't have source_id for this csv, so I'll asume all of them as from source 1.

    prices['source_id'] = 1

    # Set unit scale as mt.

    prices['unit_scale'] = 'mt'

    # Drop currencies not in our actual db.

    prices = prices.drop(labels=list(prices[(prices['currency'] == 'BIF') |
                                    (prices['currency'] == 'SSD') | 
                                    (prices['currency'] == 'CDF')].index), axis=0)
    prices.reset_index(drop=True)

    prices = prices.rename({'product':'product_name','currency':'currency_code',
                 'date':'date_price','retail (mt)':'retail_observed_price',
                 'wholesale (mt)': 'wholeale_observed_price'}, axis=1)

    prices = prices[['product_name','market_id','unit_scale','source_id','currency_code',
                    'date_price','retail_observed_price','wholeale_observed_price']]
    
    prices['date_price'] = pd.to_datetime(prices['date_price'])

    prices[['retail_observed_price','wholeale_observed_price']] = prices[['retail_observed_price','wholeale_observed_price']].astype(float)


    # I'll segment the commits, in order to prevent the collapse of the connection.

    list_of_markets = list(set(prices['market_id']))  


    list_of_prices = []
    for i in range(len(list_of_markets)):

        list_of_prices.append(prices[prices['market_id'] == list_of_markets[i]].values.tolist())


    for j in range(len(list_of_prices)):


        try:


            # Stablishes connection with our db.

            connection = psycopg2.connect(user=os.environ.get('aws_db_user'),
                                        password=os.environ.get('aws_db_password'),
                                        host=os.environ.get('aws_db_host'),
                                        port=os.environ.get('aws_db_port'),
                                        database=os.environ.get('aws_db_name'))

            
            # Create the cursor.

            cursor = connection.cursor()

            for k in range(len(list_of_prices[j])):

                vector = (list_of_prices[j][k][0],list_of_prices[j][k][1],list_of_prices[j][k][2],
                        list_of_prices[j][k][3],list_of_prices[j][k][4],list_of_prices[j][k][5],
                        list_of_prices[j][k][6],list_of_prices[j][k][7])

                query_populate_maize_table = '''
                                INSERT INTO maize_raw_info (
                                product_name,
                                market_id,
                                unit_scale,
                                source_id,
                                currency_code,
                                date_price,
                                retail_observed_price,
                                wholesale_observed_price)
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

                cursor.execute(query_populate_maize_table,vector)

                connection.commit()

        

        except (Exception, psycopg2.Error) as error:
            print('Error dropping the data.')
            print(list_of_prices[j][0])

        finally:

            if (connection):
                cursor.close()
                connection.close()


if __name__ == "__main__":

    populate_maize_table()
