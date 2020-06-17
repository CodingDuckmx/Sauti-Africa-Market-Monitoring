import copy
import datetime
import numpy as np
import os
import pandas as pd
import psycopg2

from dotenv import load_dotenv, find_dotenv
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
from sqlalchemy import create_engine

load_dotenv()

class Clean_and_classify_class:
    def __init__(self):
        self.description = ''' This class groups the main functions that are used to clean, 
                                prepare the data, build  the ALPS bands and also label the
                                product prices.'''

    def set_columns(self,data):
        '''  Builds a dataframe with the raw data comming from the db. '''

        data = pd.DataFrame(data)
        data = data.rename(columns={0:'date_price',1:'unit_scale',2:'observed_price'})

        return data


    def basic_cleanning(self,df):
        
        ''' 
        Removes duplicates in dates column. 
        Verify unique unit scale.
        Try to correct typos.

        Returns the metric and the dataframe with the basic cleaned data.
        '''

        cfd = df.copy()    

        # Set dates into date format.

        cfd['date_price'] =  pd.to_datetime(cfd['date_price'])

        # Remove duplicates in dates column.

        cfd = cfd.sort_values(by=['date_price',cfd.columns[-1]])

        drop_index = list(cfd[cfd.duplicated(['date_price'], keep='first')].index)

        cfd = cfd.drop(labels=drop_index, axis=0).reset_index(drop=True)

        # Drop values with prices zero.

        drop_index = list(cfd[cfd.iloc[:,-1] == 0].index)

        cfd = cfd.drop(labels=drop_index, axis=0).reset_index(drop=True) 

        if cfd.empty:

            return None, cfd
        
        else:

            # Consider the mode of unit scale as the one.

            metric = stats.mode(cfd.iloc[:,1])[0][0]

            discording_scale = list(cfd[cfd['unit_scale'] != metric].index)

            if discording_scale:

                cfd = cfd.drop(labels=discording_scale, axis=0).reset_index(drop=True)  
            
            # Try to correct typos that seems to be missing a decimal point.
            
            if 9 <= cfd.describe().T['max'].values[0] / cfd.describe().T['75%'].values[0] <= 11:
                
                Q99 = cfd.quantile(0.99).values[0]
                selected_indices = list(cfd[cfd.iloc[:,-1] > Q99].index)
                for i in selected_indices:
                    cfd.iloc[i,-1] = cfd.iloc[i,-1] / 10
            
            # Drop typos we can't solve.
            
            drop_index = cfd[(cfd.iloc[:,-1].rolling(window=1, min_periods= 1).max() / cfd.iloc[:,-1].rolling(window=30, min_periods= 10).quantile(0.75) ) > 5].index
            
            cfd = cfd.drop(labels=drop_index, axis=0).reset_index(drop=True)  
            
            # TODO the following not sure if they apport meaningful results, because the two lines before this are very aggresive. 

            
            Q01 = cfd.iloc[:,-1].quantile(.1)
            
            if cfd.describe().T['min'].values[0] < Q01:              

                drop_index = list(cfd[cfd.iloc[:,-1] < Q01].index)

                cfd = cfd.drop(labels=drop_index, axis=0).reset_index(drop=True)
    
            # Drop outliers.

            if (cfd.describe().T['max'].values[0]) > (cfd.describe().T['75%'].values[0] + cfd.describe().T['std'].values[0]):

                # First round

                z = np.abs(stats.zscore(cfd.iloc[:,-1], nan_policy='omit'))

                drop_index = list(np.where(z>4)[0])

                cfd = cfd.drop(labels=drop_index, axis=0).reset_index(drop=True)

            if (cfd.describe().T['max'].values[0]) > (cfd.describe().T['75%'].values[0] + cfd.describe().T['std'].values[0]):                
                
                # Second round.

                z = np.abs(stats.zscore(cfd.iloc[:,-1], nan_policy='omit'))

                drop_index = list(np.where(z>5)[0])

                cfd = cfd.drop(labels=drop_index, axis=0).reset_index(drop=True)            
                                
            return metric, cfd
    
    def limit_2019_and_later(self,df):

        ''' 
        Limit the info to the 2020 or later and assigns its month, so the price could be compared with the bands.
        '''


        df = df[df['date_price'] > pd.to_datetime('2018-12-31')]
        df['date_price'] = df['date_price'].astype('datetime64')
        df['month'] = [str(df.iloc[i,0])[:8] + '01' for i in range(len(df))]
        df = df.reset_index(drop=True)

        return df


    def prepare_data_to_ALPS(self,df):
    
        ''' 
        Make a dataframe with the last Sunday before the dates of the input dataframe, and the saturday of the last week in within the dates.
        Then Merge both dataframes to have one with all the possible weeks within the dates of the original dataframe.
        Interpolate the missing values.
        '''      
        
        cfd = df.copy()
        

        # Turn the dataframe into a calendar.

        if cfd['date_price'].min().day == 1:
            start = cfd['date_price'].min()
        else:
            start = cfd['date_price'].min() - datetime.timedelta(days=cfd['date_price'].min().day + 1)
        if cfd['date_price'].max().day >= 28:
            end = cfd['date_price'].max()
        else:
            end = cfd['date_price'].max() - datetime.timedelta(days=cfd['date_price'].max().day +1)

        dummy = pd.DataFrame()
        dummy['date_price'] = pd.date_range(start=start, end=end)
        dummy = dummy.set_index('date_price')
        cfd = cfd.set_index('date_price')
        cfd = dummy.merge(cfd,how='outer',left_index=True, right_index=True)
        del dummy


        cfd['max_price_30days'] = cfd.iloc[:,-1].rolling(window=30,min_periods=1).max()

        cfd['max_price_30days'] = cfd['max_price_30days'].shift(-1)

        cfd = cfd[cfd.index.day == 1]

        cfd = cfd[['max_price_30days']].interpolate()

        cfd = cfd.dropna()

        return cfd
    
    def inmediate_forecast_ALPS_based(self,df):

        '''
        Takes the prices and the prediction for the next month, for the last 
        two years, using a basic linear regression, taking for variables, 
        the month in which the price was taken.
        '''
               
        forecasted_prices = []

        basesetyear = df.index.max().year - 2

        stop_0 = datetime.date(year=basesetyear,month=12,day=31)

        baseset = df.iloc[:len(df.loc[:stop_0]),:].copy()   

        # For all the past months:
        for i in range(len(df)-len(baseset)):

            workset = df.iloc[:len(df.loc[:stop_0]) + i,:].copy()

            # What month are we?
            
            workset['month'] = workset.index.month

            # Build dummy variables for the months.

            dummies_df = pd.get_dummies(workset['month'])
            dummies_df = dummies_df.T.reindex(range(1,13)).T.fillna(0)

            workset = workset.join(dummies_df)
            workset = workset.drop(labels=['month'], axis=1)
            
            features = workset.columns[1:]
            target = workset.columns[0]

            X = workset[features]
            y = workset[target]

            reg = LinearRegression()
                       
            reg = reg.fit(X,y)

            next_month = df.iloc[len(df.loc[:stop_0]) + i,:].name

            raw_next_month = [0 if j != next_month.month else 1 for j in range(1,13)]

            next_month_array = np.array(raw_next_month).reshape(1,-1)
        
            forecasted_prices.append(reg.predict(next_month_array)[0])
        
        # For the current month.

        raw_next_month = [0 if j != next_month.month + 1 else 1 for j in range(1,13)]

        next_month_array = np.array(raw_next_month).reshape(1,-1)

        forecasted_prices.append(reg.predict(next_month_array)[0])    

        return stop_0, forecasted_prices
           
    
    def build_bands_wfp_forecast(self,df, stop_0, forecasted_prices):

        ''' 
        Takes the forecasted prices and build a dataframe with the ALPS bands,
        and calculates the stressness of them.
        '''
        
        errorstable = pd.DataFrame(index=pd.date_range(df.loc[stop_0:].index[0],datetime.date(df.index[-1].year,df.index[-1].month + 1, 1), freq='MS'),
                        columns=['observed_price','forecast']) 
        errorstable.iloc[:,0] = None
        errorstable.iloc[:-1,0] =  [x[0] for x in df.iloc[len(df.loc[:stop_0]):,:].values.tolist()]
        errorstable.iloc[:,1] =  forecasted_prices
        
        errorstable['residuals'] = errorstable.iloc[:,0] - errorstable['forecast']
        errorstable['cum_residual_std'] = [np.std(errorstable.iloc[:i,2]) for i in range(1,len(errorstable)+1)]
        errorstable['ALPS'] = [None] + list(errorstable.iloc[1:,2]  / errorstable.iloc[1:,3])
        errorstable['Price Status'] = None
        errorstable['Stressness'] = None
  
        errorstable['normal_limit'] = errorstable['forecast'] + 0.25 * errorstable['cum_residual_std']
        errorstable['stress_limit'] = errorstable['forecast'] + errorstable['cum_residual_std']
        errorstable['alert_limit'] = errorstable['forecast'] + 2 * errorstable['cum_residual_std']

        for date in range(len(errorstable)-1):

            if errorstable.iloc[date,4] < 0.25:
                errorstable.iloc[date,5] = 'Normal'
                errorstable.iloc[date,6] =  errorstable.iloc[date,0] / errorstable.iloc[date,7]
                
            elif errorstable.iloc[date,4] < 1:
                errorstable.iloc[date,5] = 'Stress'
                errorstable.iloc[date,6] =  errorstable.iloc[date,0] / errorstable.iloc[date,8]
                
            elif errorstable.iloc[date,4] < 2:
                errorstable.iloc[date,5] = 'Alert'
                errorstable.iloc[date,6] =  errorstable.iloc[date,0] / errorstable.iloc[date,9]
                
            else:
                errorstable.iloc[date,5] = 'Crisis'
                errorstable.iloc[date,6] =  errorstable.iloc[date,0] / errorstable.iloc[date,9]

        mae = mean_squared_error(errorstable.iloc[:-1,0],errorstable.iloc[:-1,1])
                
        return errorstable, mae

    def set_columns_bands_df(self,bands):

        '''
        Builds a dataframe from the raw data for the bands, from the db.
        '''

        bands= pd.DataFrame(bands)
        bands = bands.rename(columns={0:'date_price',1:'normal_band_limit',2:'stress_band_limit',3:'alert_band_limit', 4:'class_method'})
        
        return bands 

    def assign_classification(self,data,bands):

        '''
        Combine the data from the prices and the bands to classify the price in its status.
        '''

        results = data.copy()

        results['Observed_class'] = None
        results['Stressness'] = None
        results['class_method'] = bands.iloc[0,-1]

        for i in range(len(results)):

            bands_limits = bands[bands['date_price'] == datetime.date.fromisoformat(data.iloc[i,3])]

            if results.iloc[i,2] < bands_limits.iloc[0,1]:

                results.iloc[i,4] = 'Normal'
                results.iloc[i,5] = results.iloc[i,2] / bands_limits.iloc[0,1]

            elif results.iloc[i,2] < bands_limits.iloc[0,2]:

                results.iloc[i,4] = 'Stress'
                results.iloc[i,5] = results.iloc[i,2] / bands_limits.iloc[0,2]
            
            elif results.iloc[i,2] < bands_limits.iloc[0,3]:

                results.iloc[i,4] = 'Alert'
                results.iloc[i,5] = results.iloc[i,2] / bands_limits.iloc[0,3]

            else:

                results.iloc[i,4] = 'Crisis'
                results.iloc[i,5] = results.iloc[i,2] / bands_limits.iloc[0,3]

        results = results.drop(labels=['month'], axis=1)

        return results

    
  
    def run_build_bands(self,data):

        '''
        A method that runs in a line the methods required.
        '''
        
        metric, cleaned = self.basic_cleanning(data)
        try:
            stop_0, forecasted_prices = self.inmediate_forecast_ALPS_based(self.prepare_data_to_ALPS(cleaned))
            result, mae = self.build_bands_wfp_forecast(self.prepare_data_to_ALPS(cleaned),stop_0,forecasted_prices)

            return metric, stop_0, result, mae
        
        except:

            return None, None, None, None

def possible_product_market_pairs():
    '''
    Pulls the data from the table raw_table and stablishes a set of product/market pair
    might be worth to work with.
    It makes a dictionary with dataframes for all combination possibles and also returns a list
    of the 'worth ones' to build the stress bands.
    '''

    try:

        # Stablishes connection with our db.

        connection = psycopg2.connect(user=os.environ.get('aws_db_user'),
                                        password=os.environ.get('aws_db_password'),
                                        host=os.environ.get('aws_db_host'),
                                        port=os.environ.get('aws_db_port'),
                                        database=os.environ.get('aws_db_name'))

        query_retail = '''
                SELECT *
                FROM retail_stats
                WHERE data_length > 730
                AND max_date >= '2019-01-01'::date
                AND std > 0
                '''

        query_wholesale = '''
                SELECT *
                FROM wholesale_stats
                WHERE data_length > 730
                AND max_date >= '2019-01-01'::date
                AND std > 0
                '''

        retail_df = pd.read_sql(query_retail, con=connection)
        retail_df = retail_df.drop(labels='index', axis=1)

        wholesale_df = pd.read_sql(query_wholesale, con=connection)
        wholesale_df = wholesale_df.drop(labels='index', axis=1)

        retail_df['stationary'] = None
        for i in range(len(retail_df)):
                if retail_df.iloc[i,-13] > retail_df.iloc[i,-17]:
                    retail_df.iloc[i,-1] = True
                else:
                    retail_df.iloc[i,-1] = False

        strong_candidates_retail = retail_df[(retail_df['data_length'] > 365*3) & (retail_df['stationary'] == True)]
        retail_df = retail_df.drop(labels=list(strong_candidates_retail.index), axis=0)

        weak_candidates_retail = retail_df.copy()

        wholesale_df['stationary'] = None
        for i in range(len(wholesale_df)):
                if wholesale_df.iloc[i,-13] > wholesale_df.iloc[i,-17]:
                    wholesale_df.iloc[i,-1] = True
                else:
                    wholesale_df.iloc[i,-1] = False

        strong_candidates_wholesale = wholesale_df[(wholesale_df['data_length'] > 365*3) & (wholesale_df['stationary'] == True)]
        wholesale_df = wholesale_df.drop(labels=list(strong_candidates_wholesale.index), axis=0)

        weak_candidates_wholesale = wholesale_df.copy()

        strong_candidates_retail = strong_candidates_retail[['product_name','market_id','source_id','currency_code']].values.tolist()
        weak_candidates_retail = weak_candidates_retail[['product_name','market_id','source_id','currency_code']].values.tolist()
        strong_candidates_wholesale = strong_candidates_wholesale[['product_name','market_id','source_id','currency_code']].values.tolist()
        weak_candidates_wholesale = weak_candidates_wholesale[['product_name','market_id','source_id','currency_code']].values.tolist()

                
        return strong_candidates_retail, weak_candidates_retail, strong_candidates_wholesale, weak_candidates_wholesale


    except (Exception, psycopg2.Error) as error:
        print('Error pulling the data or forming the dictionary.')

    finally:

        if (connection):
            connection.close()


def product_ws_hist_ALPS_bands(product_name, market_id, source_id, currency_code,model_name):
    '''
    Builds the wholesale historic ALPS bands.
    '''

    data = None
    market_with_problems = []

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
                        SELECT date_price, unit_scale, wholesale_observed_price
                        FROM raw_table
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

        # Clean, prepare the data, build  the ALPS bands.

        clean_class = Clean_and_classify_class()

        metric, stop_0, wfp_forecast, mse = clean_class.run_build_bands(data)

        if metric:

            # If the bands were built, this code will be run to drop the info in the db.

            wfp_forecast = wfp_forecast.reset_index()
            
            # try:

                
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
                if observed_price:
                    observed_price = round(row[1],4)
                observed_class = row[6]
                class_method =  model_name
                normal_band_limit = round(row[8],4) 
                stress_band_limit = round(row[9],4)
                alert_band_limit = round(row[10],4)

                vector = (product_name,market_id,source_id,currency_code,metric,date_price,
                            observed_price,observed_class,class_method,date_run_model,
                            normal_band_limit,stress_band_limit,alert_band_limit)

                # try:

                cursor.execute('''
                                            DELETE FROM wholesale_bands
                                            WHERE product_name = %s
                                            AND market_id = %s
                                            AND unit_scale = %s
                                            AND source_id = %s
                                            AND currency_code = %s
                                            AND date_price = %s
                                            AND observed_price IS NULL
                            ''', (product_name, market_id,metric,source_id,currency_code,date_price))
             
                # except:
                #     pass
                
                cursor.execute('''
                                SELECT id
                                FROM wholesale_bands
                                WHERE product_name = %s
                                AND market_id = %s
                                AND source_id = %s
                                AND currency_code = %s
                                AND unit_scale = %s
                                AND date_price = %s
                                AND observed_price = %s
                                AND observed_class = %s
                                AND class_method = %s
                                AND normal_band_limit = %s
                                AND stress_band_limit = %s
                                AND alert_band_limit = %s
                            ''', (product_name,market_id,source_id,currency_code,metric,date_price,
                            observed_price,observed_class,class_method,
                            normal_band_limit,stress_band_limit,alert_band_limit))

                result = cursor.fetchall()

                print(result)

                if not result:

                    print(vector)

                    query_insert_results ='''
                                        INSERT INTO wholesale_bands (
                                        product_name,
                                        market_id,
                                        source_id,
                                        currency_code,
                                        unit_scale,
                                        date_price,
                                        observed_price,
                                        observed_class,
                                        class_method,
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
                                            %s,
                                            %s
                                        );
                    '''

                    cursor.execute(query_insert_results, vector)

                    connection.commit()

            connection.close()
        
        else:

            print('The combination:',product_name, market_id, source_id, currency_code, 'has problems.')
            market_with_problems.append((product_name, market_id, source_id, currency_code))


        return market_with_problems



def product_retail_historic_ALPS_bands(product_name, market_id, source_id, currency_code,model_name):
    '''
    Builds the retail historic ALPS bands.
    '''

    data = None
    market_with_problems = []

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
                        SELECT date_price, unit_scale, wholesale_observed_price
                        FROM raw_table
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

        # Clean, prepare the data, build  the ALPS bands.

        clean_class = Clean_and_classify_class()

        metric, stop_0, wfp_forecast, mse = clean_class.run_build_bands(data)

        if metric:

            # If the bands were built, this code will be run to drop the info in the db.

            wfp_forecast = wfp_forecast.reset_index()
            
            # try:

                
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
                used_band_model =  model_name
                normal_band_limit = round(row[8],4) 
                stress_band_limit = round(row[9],4)
                alert_band_limit = round(row[10],4)

                vector = (product_name,market_id,source_id,currency_code,metric,date_price,
                            observed_price,observed_class,used_band_model,date_run_model,
                            normal_band_limit,stress_band_limit,alert_band_limit)

                query_insert_results ='''
                                    INSERT INTO retail_bands (
                                    product_name,
                                    market_id,
                                    source_id,
                                    currency_code,
                                    unit_scale,
                                    date_price,
                                    observed_price,
                                    observed_class,
                                    class_method,
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
                                        %s,
                                        %s
                                    );
                '''

                cursor.execute(query_insert_results, vector)

                connection.commit()

            connection.close()
        
        else:

            print('The combination:',product_name, market_id, source_id, currency_code, 'has problems.')
            market_with_problems.append((product_name, market_id, source_id, currency_code))


        return market_with_problems


def make_dictionaries_possible_dataframes():

    connection = psycopg2.connect(user=os.environ.get('aws_db_user'),
                            password=os.environ.get('aws_db_password'),
                            host=os.environ.get('aws_db_host'),
                            port=os.environ.get('aws_db_port'),
                            database=os.environ.get('aws_db_name'))

    query = '''
    SELECT *
    FROM raw_table
    '''

    pulled_info = pd.read_sql(query, con=connection)

    pulled_info = pulled_info.drop(labels=['id'], axis=1)

    connection.close()

    product_list = list(set(pulled_info['product_name']))

    df = pulled_info.copy()
    product_market_pair = []
    total_count = 1
    prod_dict = {product:np.nan for product in product_list}
    descriptions_retail = {i:np.nan for i in range(1,len(df)+1)}
    descriptions_wholesale = {i:np.nan for i in range(1,len(df)+1)}
    for product in product_list:
        available_markets = list(set(df[df['product_name'] == product]['market_id']))
        prod_dict[product] = {market:np.nan for market in available_markets}
        for market in available_markets:
            available_sources = list(set(df[(df['product_name'] == product) & (df['market_id'] == market)]['source_id']))
            prod_dict[product][market] = {source:np.nan for source in available_sources}
            for source in available_sources:
                available_currencies = list(set(df[(df['product_name'] == product) & (df['market_id'] == market) & (df['source_id'] == source)]['currency_code']))
                prod_dict[product][market][source] = {currency:np.nan for currency in available_currencies}
                for currency in available_currencies:
                    prod_dict[product][market][source][currency] = {'retail_observed_price':np.nan, 'wholesale_observed_price':np.nan}
                    prod_dict[product][market][source][currency]['retail_observed_price'] = {'shape':np.nan, 'info':np.nan}
                    prod_dict[product][market][source][currency]['wholesale_observed_price'] = {'shape':np.nan, 'info':np.nan}

                    prod_dict[product][market][source][currency]['retail_observed_price']['shape'] = df[(df['product_name'] == product) & (df['market_id'] == market) & (df['source_id'] == source) & (df['currency_code'] == currency)][['date_price','unit_scale','retail_observed_price']].shape
                    prod_dict[product][market][source][currency]['retail_observed_price']['info'] = df[(df['product_name'] == product) & (df['market_id'] == market) & (df['source_id'] == source) & (df['currency_code'] == currency)][['date_price','unit_scale','retail_observed_price']]
                    product_market_pair.append((product, market, source, currency))

                    temporal_df = prod_dict[product][market][source][currency]['retail_observed_price']['info'].copy()
                    
                    if len(temporal_df) > 20:
                        result_adft = adfuller(temporal_df.iloc[:,-1])
                        stationary_results = [round(result_adft[0],4), round(result_adft[1],4),result_adft[4]['1%'], result_adft[4]['5%'], result_adft[4]['10%'] ]
                    else:
                        stationary_results = [None, None, None, None, None]
                    
                    descriptions_retail[total_count] = [product,market, source, currency, temporal_df['date_price'].min(),temporal_df['date_price'].max(), stats.mode(np.diff(temporal_df['date_price'].sort_values()))[0]] +  temporal_df.describe().T.values.tolist() + stationary_results

                                       
                    del temporal_df
                    
                    prod_dict[product][market][source][currency]['wholesale_observed_price']['shape'] = df[(df['product_name'] == product) & (df['market_id'] == market) & (df['source_id'] == source) & (df['currency_code'] == currency)][['date_price','unit_scale','wholesale_observed_price']].shape
                    prod_dict[product][market][source][currency]['wholesale_observed_price']['info'] = df[(df['product_name'] == product) & (df['market_id'] == market) & (df['source_id'] == source) & (df['currency_code'] == currency)][['date_price','unit_scale','wholesale_observed_price']]

                    temporal_df = prod_dict[product][market][source][currency]['wholesale_observed_price']['info'].copy()
                    
                    if len(temporal_df) > 20:
                        result_adft = adfuller(temporal_df.iloc[:,-1])
                        stationary_results = [round(result_adft[0],4), round(result_adft[1],4),result_adft[4]['1%'], result_adft[4]['5%'], result_adft[4]['10%'] ]
                    else:
                        stationary_results = [None, None, None, None, None]
                    
                    descriptions_wholesale[total_count] = [product,market, source, currency, temporal_df['date_price'].min(),temporal_df['date_price'].max(), stats.mode(np.diff(temporal_df['date_price'].sort_values()))[0]] + temporal_df.describe().T.values.tolist() + stationary_results

                    total_count +=1
                    
                    del temporal_df

    return product_market_pair, descriptions_retail, descriptions_wholesale

def drop_stats_results_to_df(description_df,tablename):

    df1 = pd.DataFrame.from_dict(description_df).T.dropna().rename(columns={0:'product_name',1:'market_id', 2: 'source_id', 3:'currency_code',4:'min_date',5:'max_date',6:'erase',7:'stats',8:'ADF Statistic',9:'p-value',10:'Critical value 0.01',11:'Critical value 0.05',12:'Critical value 0.1'}).reset_index(drop=True)
    df2 = pd.DataFrame(df1['erase'].tolist(), index=df1.index).rename(columns={0:'mode_dispersion'})
    df3 = pd.DataFrame(df1['stats'].tolist(), index=df1.index).rename(columns={0:'data_points', 1:' mean', 2:'std', 3:'min',4:'0.25',5:'0.50',6:'0.75',7:'max'})

    summary_df = pd.concat([df1,df2,df3], axis=1).drop(labels=['erase','stats'], axis=1)
    
    summary_df['data_length'] = (summary_df['max_date'] - summary_df['min_date'] + datetime.timedelta(days=1)).dt.days
    summary_df['completeness'] = summary_df['data_points'] / summary_df['data_length']
    summary_df['mode_dispersion'] = summary_df['mode_dispersion'].fillna(pd.Timedelta(seconds=0))
    summary_df['mode_dispersion'] = summary_df['mode_dispersion'].dt.days
    summary_df = summary_df.replace({float('inf'):0, np.nan:0})
    summary_df = summary_df.sort_values(by=['data_length','completeness','max_date'], ascending=False)


    db_URI = 'postgresql://' + os.environ.get('aws_db_user') + ':' + os.environ.get('aws_db_password') + '@' + os.environ.get('aws_db_host') + '/' + os.environ.get('aws_db_name')
    engine = create_engine(db_URI)
    conn = engine.connect()

    summary_df.to_sql(tablename, con=conn, if_exists='replace', chunksize=100)


    conn.close()


def product_retail_clean_and_classify(product_name, market_id, source_id, currency_code):

    '''
    Pulls the data from the raw and compare it with the bands, to classify
    the stress level of the price.
    '''

    data = None

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
                        SELECT date_price, unit_scale, retail_observed_price
                        FROM raw_table
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
        

        clean_class = Clean_and_classify_class()
        data = clean_class.set_columns(data)
        metric, cleaned = clean_class.basic_cleanning(data)
        data = clean_class.limit_2019_and_later(cleaned)

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
                            SELECT date_price, normal_band_limit, stress_band_limit, alert_band_limit, class_method
                            FROM retail_bands
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


        bands = clean_class.set_columns_bands_df(bands)

        classified = clean_class.assign_classification(data,bands)

        classified['date_price'] = pd.to_datetime(classified['date_price'])
        classified['Stressness'] = classified['Stressness'].astype(float)

        classified = classified.values.tolist()
        
        for row in classified:

            date_price = str(row[0].strftime("%Y-%m-%d"))
            unit_scale = row[1]
            observed_price = row[2]
            observed_class = row[3]
            stressness = row[4]
            class_method = row[5]

            # we will be dropping the classification values into the db.

            # try:


            # Stablishes connection with our db.

            connection = psycopg2.connect(user=os.environ.get('aws_db_user'),
                                        password=os.environ.get('aws_db_password'),
                                        host=os.environ.get('aws_db_host'),
                                        port=os.environ.get('aws_db_port'),
                                        database=os.environ.get('aws_db_name'))

                
                # Create the cursor.

            cursor = connection.cursor()

            cursor.execute('''
                            SELECT id
                            FROM retail_prices
                            WHERE product_name = %s
                            AND market_id = %s
                            AND source_id = %s
                            AND currency_code = %s
                            AND unit_scale = %s
                            AND date_price = %s
                            AND observed_price = %s
                        ''', (product_name,market_id,source_id,currency_code,unit_scale,
                        date_price, observed_price))

            row_id = cursor.fetchall()

            if row_id:

                row_id = row_id[0][0]

                cursor.execute('''
                                UPDATE retail_prices
                                SET observed_class = %s,
                                class_method = %s,
                                stressness = %s
                                WHERE id = %s
                ''', (observed_class, class_method, stressness, row_id))

                connection.commit()


        connection.close()        

 



        # except (Exception, psycopg2.Error) as error:
        #     print('Error dropping the labels.')

        # finally:

        #     if (connection):
        #         cursor.close()
        #         connection.close()

def product_wholesale_clean_and_classify(product_name, market_id, source_id, currency_code):

    '''
    Pulls the data from the raw and compare it with the bands, to classify
    the stress level of the price.
    '''

    data = None

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
                        SELECT date_price, unit_scale, wholesale_observed_price
                        FROM raw_table
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
        

        clean_class = Clean_and_classify_class()
        data = clean_class.set_columns(data)
        _, cleaned = clean_class.basic_cleanning(data)
        data = clean_class.limit_2019_and_later(cleaned)

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
                            SELECT date_price, normal_band_limit, stress_band_limit, alert_band_limit, class_method
                            FROM wholesale_bands
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


        bands = clean_class.set_columns_bands_df(bands)

        classified = clean_class.assign_classification(data,bands)

        classified['date_price'] = pd.to_datetime(classified['date_price'])
        classified['Stressness'] = classified['Stressness'].astype(float)

        classified = classified.values.tolist()
        
        for row in classified:

            date_price = str(row[0].strftime("%Y-%m-%d"))
            unit_scale = row[1]
            observed_price = row[2]
            observed_class = row[3]
            stressness = row[4]
            class_method = row[5]

            # we will be dropping the classification values into the db.

            # try:


            # Stablishes connection with our db.

            connection = psycopg2.connect(user=os.environ.get('aws_db_user'),
                                        password=os.environ.get('aws_db_password'),
                                        host=os.environ.get('aws_db_host'),
                                        port=os.environ.get('aws_db_port'),
                                        database=os.environ.get('aws_db_name'))

                
                # Create the cursor.

            cursor = connection.cursor()

            cursor.execute('''
                            SELECT id
                            FROM wholesale_prices
                            WHERE product_name = %s
                            AND market_id = %s
                            AND source_id = %s
                            AND currency_code = %s
                            AND unit_scale = %s
                            AND date_price = %s
                            AND observed_price = %s
                        ''', (product_name,market_id,source_id,currency_code,unit_scale,
                        date_price, observed_price))

            row_id = cursor.fetchall()

            if row_id:

                row_id = row_id[0][0]

                cursor.execute('''
                                UPDATE wholesale_prices
                                SET observed_class = %s,
                                class_method = %s,
                                stressness = %s
                                WHERE id = %s
                ''', (observed_class, class_method, stressness, row_id))

                connection.commit()


        connection.close()        

 



        # except (Exception, psycopg2.Error) as error:
        #     print('Error dropping the labels.')

        # finally:

        #     if (connection):
        #         cursor.close()
        #         connection.close()



#########################################################

################## Jing's Classes #######################

#########################################################


class dbConnect:
    """connect to database for read and write table"""
    def __init__(self, name='wholesale_observed_price'):
        self.name = name
        self.df = [] # create an empy dataframe
    
    def read_stakeholder_db(self):
        """read data from specific table in stakeholder's db"""
        db_URI = 'mysql+pymysql://' + os.environ.get('stakeholder_db_user') + ':' + \
            os.environ.get('stakeholder_db_password') + '@' + os.environ.get('stakeholder_db_host') + '/' + os.environ.get('stakeholder_db_name')
        engine = create_engine(db_URI)
        conn = engine.connect()
        tablename = "platform_market_id_prices2"
        query_statement = "SELECT * FROM "+ tablename 
        data = pd.read_sql(query_statement, con=conn)   
        conn.close()     
        return data

    def read_raw_table(self):
      """read the raw_table data from our db"""
      db_URI = 'postgresql://' + os.environ.get('aws_db_user') + ':' + os.environ.get('aws_db_password') + '@' + os.environ.get('aws_db_host') + '/' + os.environ.get('aws_db_name')
      engine = create_engine(db_URI)
      conn = engine.connect()
      tablename = "raw_table"
      query_statement = "SELECT * FROM "+ tablename 
      data = pd.read_sql(query_statement, con=conn)    
      conn.close()
      return data


    def read_analytical_db(self, tablename):
        """read AWS analytical db """
        db_URI = 'postgresql://' + os.environ.get('aws_db_user') + ':' + os.environ.get('aws_db_password') + '@' + os.environ.get('aws_db_host') + '/' + os.environ.get('aws_db_name')
        engine = create_engine(db_URI)
        conn = engine.connect()
        query_statement = "SELECT * FROM " + tablename 
        data = pd.read_sql(query_statement, con=conn)
        conn.close()
        return data

    def populate_analytical_db(self, df, tablename):
        """populate AWS analytical db with df and tablename """
        db_URI = 'postgresql://' + os.environ.get('aws_db_user') + ':' + os.environ.get('aws_db_password') + '@' + os.environ.get('aws_db_host') + '/' + os.environ.get('aws_db_name')
        engine = create_engine(db_URI)
        conn = engine.connect()
        
        df.to_sql(tablename, con=conn, if_exists='replace', index=False, chunksize=100)

        conn.close()
       
    def migrate_analyticalDB(self):
        """read/add newly added data only"""
        pass #raw = read_stakeholderDB()


class DataCleaning:
    """ method to clean data, apply to the whole data set (mixed time series)"""
    def __init__(self):
        pass
        
    def read_data(self, data=None):
        if data is None:
            print("Warning: No data provided")
        if (isinstance(data, pd.DataFrame) == False):
            print( "Input should be a dataframe!")
        else:
            df = pd.DataFrame(data)
            print('Data is fed to class object.')
        return df
              
    def remove_zeros(self, data):
        """clean all invalid entries
        cost cannot be 0, replace zeros with NaN"""
        df = data.copy()
        cols = ['wholesale_observed_price', 'retail_observed_price']
        
        df[cols] = df[cols].replace({0: np.nan})
        if np.prod(df['wholesale_observed_price'] != 0):
            print('All zero values has been replaced with NaN successfully')
        else:
            print('Zero to NaN process not complete.')
        return df    
   

    def convert_dtypes(self, data):
        """change each column to desired data type"""
        df = data.copy()
        # # change date to datetime
        # df['date_price'] = pd.to_datetime(df['date_price'])

        # change num dtype to float
        df['wholesale_observed_price'] = df['wholesale_observed_price'].astype('float')
        df['retail_observed_price'] = df['retail_observed_price'].astype('float')
      
        # change text col to categorical
        str_cols = ['market_id', 'product_name', 'currency_code']
        for item in str_cols:
            df[item] = df[item].astype('category')
        
        print('Data type converted. Numericals converted to float, date to datatime type, and non-numericals to category.')
       
        return df
     

class DataQualityCheck:
    """contain methods for quality check for one time series"""
    def __init__(self):
        pass
    
    def read_data(self, data=None):
        if data is None:
            print("Warning: No data provided")
        if (isinstance(data, pd.Series) == False) & (isinstance(data.index, pd.DatetimeIndex)):
            print("Data needs to be pandas series with datetime index!")
        else:
            df = pd.Series(data)
        return df
        
    def remove_duplicates(self, df):
        """remove duplicated rows, keep the first"""
        y = df.copy()
        rows_rm = y.index.duplicated(keep='first')
        if np.sum(rows_rm):
            y = y[~rows_rm]
        return y
        
    def remove_outliers(self, df):
        """remove outliers from a series"""        
        y = df.copy()
        lower_bound, upper_bound = y.quantile(.05), y.quantile(.95)
        
        y = y[y.iloc[:, 0].between(lower_bound[0], upper_bound[0])]
        return y

    def day_by_day(self, df):
        """construct time frame and create augumented time series"""
        y = df.copy()
        
        START, END = y.index.min(), y.index.max()        
        # construct a time frame from start to end
        date_range = pd.date_range(start=START, end=END, freq='D')
        time_df = pd.DataFrame([], index=date_range)
        # this is time series framed in the complete day-by-day timeframe
        y_t = time_df.merge(y, how='left', left_index=True, right_index=True)
        return y_t

    def generate_QC(self, df, figure_output=0):
        """ 
        Input:  y: time series with sorted time index
        Output: time series data quality metrics
            start, end, timeliness, data_length, completeness, duplicates, mode_D
            start: start of time series
            end: end of time seires
            timeliness: gap between the end of time seires and today, days. 0 means sampling is up to today, 30 means the most recent data was sampled 30 days ago.
            data_length: length of available data in terms of days
            completeness: not NaN/total data in a complete day-by-day time frame, 0 means all data are not valid, 1 means data is completed on 
            duplicates: number of data sampled on same date, 0: no duplicates, 10: 10 data were sampled on a same date
            mode_D: the most frequent sampling interval in time series, days, this is important for determing forecast resolution
        """
        y = df.copy()
        y1 = self.remove_duplicates(y)
        y2 = self.remove_outliers(y)

        if y2.empty:
            # e.g., special case of two datapoint, all data will be considered outlier
            y = y1
        else:
            y = y2
            # construct time frame and create augumented time series
        START, END = y.index.min(), y.index.max()
        TIMELINESS = (datetime.date.today()-END).days
        
        # this is time series framed in the complete day-by-day timeframe
        y_t = self.day_by_day(y) 
        
        # completeness
        L = len(y_t)
        L_nan = y_t.isnull().sum()
        COMPLETENESS = (1-L_nan/L)[0]
        COMPLETENESS = round(COMPLETENESS, 3)
        DATA_LEN = L

        if COMPLETENESS == 0 | DATA_LEN == 1:
            # no data or 1 datum
            DUPLICATES = np.nan
            MODE_D = np.nan

        else:
            # some data exist
            timediff = pd.DataFrame(np.diff(y.index.values), columns=['D'])
            x = timediff['D'].value_counts()
            x.index = x.index.astype(str)
            # x is value counts of differences between all adjecent sampling dates for one time series

            if x.empty:
                # only one data available, keep row for future data addition
                DUPLICATES = 0
                MODE_D = 0

            elif any(x.index == '0 days') | len(x) == 1:
                # duplicates exists, and all data occur on the same date
                DUPLICATES = x[0]
                MODE_D = 0

            elif any(x.index == '0 days') | len(x) > 1:
                # duplicates exists and data not equally spaced
                DUPLICATES = x[0]
                MODE_D = x[~(x.index == '0 days')].index[0]

            else:  # elif ('0 days' not in x.index):
                # no duplication
                DUPLICATES = 0
                MODE_D = x.index[0]

        # START = str(START.date())
        # END = str(END.date())
        QC_i = [START, END, TIMELINESS,
                DATA_LEN, COMPLETENESS, DUPLICATES, MODE_D]

        if figure_output == 1:
            # a small plot indicating sampling scheme
            ax = sns.heatmap(y_t.isnull(), cbar=False)
            plt.show()

        return QC_i