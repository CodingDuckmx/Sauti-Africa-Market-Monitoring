import datetime
import numpy as np
import os
import pandas as pd
import psycopg2

from dotenv import load_dotenv, find_dotenv
from scipy import stats
from sklearn.linear_model import LinearRegression

load_dotenv()


def possible_maize_markets():


    try:

        # Stablishes connection with our db.

        connection = psycopg2.connect(user=os.environ.get('aws_db_user'),
                                        password=os.environ.get('aws_db_password'),
                                        host=os.environ.get('aws_db_host'),
                                        port=os.environ.get('aws_db_port'),
                                        database=os.environ.get('aws_db_name'))


        # Create the cursor.

        query = '''
                SELECT *
                FROM maize_raw_info
                '''

        all_ws = pd.read_sql(query, con=connection)

        connection.close()

        pctwo_retail = []
        pctwo_wholesale = []
        total_count = 1
        useful_count = 1
        products = ['Maize']
        df = all_ws.copy()
        prod_dict = {product:np.nan for product in products}
        for product in products:
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
                        total_count +=1
                        if (prod_dict[product][market][source][currency]['retail_observed_price']['info']['date_price'].min() < datetime.date(2015,12,31)) & (prod_dict[product][market][source][currency]['retail_observed_price']['info']['date_price'].max() > datetime.date(2020, 1, 1)):
                            pctwo_retail.append(('product_'+ str(useful_count), product, market, source, currency,'retail'))
                            useful_count +=1
                        
                        prod_dict[product][market][source][currency]['wholesale_observed_price']['shape'] = df[(df['product_name'] == product) & (df['market_id'] == market) & (df['source_id'] == source) & (df['currency_code'] == currency)][['date_price','unit_scale','wholesale_observed_price']].shape
                        prod_dict[product][market][source][currency]['wholesale_observed_price']['info'] = df[(df['product_name'] == product) & (df['market_id'] == market) & (df['source_id'] == source) & (df['currency_code'] == currency)][['date_price','unit_scale','wholesale_observed_price']]
                        if (prod_dict[product][market][source][currency]['wholesale_observed_price']['info']['date_price'].min() < datetime.date(2015,12,31)) & (prod_dict[product][market][source][currency]['wholesale_observed_price']['info']['date_price'].max() > datetime.date(2020, 1, 1)):
                            pctwo_wholesale.append(('product_'+ str(useful_count), product, market, source, currency,'wholesale'))
                            useful_count +=1
        
        
        return pctwo_retail, pctwo_wholesale


    except (Exception, psycopg2.Error) as error:
        print('Error pulling the data or forming the dictionary.')

    finally:

        if (connection):
            connection.close()

class Maize_clean_and_classify_class:
    def __init__(self):
        pass
        
    def set_columns(self,data):

        data = pd.DataFrame(data)
        data = data.rename(columns={0:'date_price',1:'unit_scale',2:'observed_price'})

        return data
    
    
    def last_four_year_truncate(self,df):
      
        start_point = df['date_price'].max() - datetime.timedelta(weeks=212)

        l4y = df[df['date_price'] >= start_point].copy()

        return l4y
    
    def basic_cleanning(self,df):
        
        ''' 
        Removes duplicates in dates column. 
        Verify unique unit scale.
        Try to correct typos.

        Returns the metric and the dataframe with the basic cleaned data.
        '''

        cfd = df.copy()    

        # Remove duplicates in dates column.

        drop_index = list(cfd[cfd.duplicated(['date_price'], keep='first')].index)

        cfd = cfd.drop(labels=drop_index, axis=0).reset_index(drop=True)

        # Consider the mode of unit scale as the one.

        metric = stats.mode(cfd.iloc[:,1])[0][0]

        discording_scale = list(cfd[cfd['unit_scale'] != metric].index)

        if discording_scale:

            cfd = cfd.drop(labels=discording_scale, axis=0).reset_index(drop=True)  

        # Drop outliers - the first round will face typos, the seconds truly outliers.

        z = np.abs(stats.zscore(cfd.iloc[:,-1], nan_policy='omit'))

        drop_index = list(np.where(z>4)[0])   

        cfd = cfd.drop(labels=drop_index, axis=0).reset_index(drop=True)

        # Second round.

        z = np.abs(stats.zscore(cfd.iloc[:,-1], nan_policy='omit'))

        drop_index = list(np.where(z>5)[0])

        cfd = cfd.drop(labels=drop_index, axis=0).reset_index(drop=True)  

        # Drop values with prices zero.

        drop_index = list(cfd[cfd.iloc[:,-1] == 0].index)

        cfd = cfd.drop(labels=drop_index, axis=0).reset_index(drop=True) 

        return metric, cfd
    
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
        
        errorstable = pd.DataFrame(index=pd.date_range(df.loc[stop_0:].index[0],datetime.date(df.index[-1].year,df.index[-1].month + 1, 1), freq='MS'),
                        columns=['observed_wholesale_price','forecast']) 
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

        return errorstable

    def run_class_colab(self,df):
        
        metric, cleaned = self.basic_cleanning(self.last_four_year_truncate(df))
        stop_0, forecasted_prices = self.inmediate_forecast_ALPS_based(self.prepare_data_to_ALPS(cleaned))
        result = self.build_bands_wfp_forecast(self.prepare_data_to_ALPS(cleaned),stop_0,forecasted_prices)
        
        return metric, stop_0, result
    
    def run_class(self,data):
        
        df = self.set_columns(data)
        metric, cleaned = self.basic_cleanning(self.last_four_year_truncate(df))
        try:
            stop_0, forecasted_prices = self.inmediate_forecast_ALPS_based(self.prepare_data_to_ALPS(cleaned))
            result = self.build_bands_wfp_forecast(self.prepare_data_to_ALPS(cleaned),stop_0,forecasted_prices)

            return metric, stop_0, result
        
        except:

            return None, None, None
     



def historic_ALPS_bands(product_name, market_id, source_id, currency_code):

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

        maize_class = Maize_clean_and_classify_class()
        # data = maize_class.set_columns(data)
        # metric, cleaned = maize_class.basic_cleanning(maize_class.last_four_year_truncate(data))
        # stop_0, forecasted_prices = maize_class.inmediate_forecast_ALPS_based(maize_class.prepare_data_to_ALPS(cleaned))
        # wfp_forecast = maize_class.build_bands_wfp_forecast(maize_class.prepare_data_to_ALPS(cleaned),stop_0, forecasted_prices)
        metric, stop_0, wfp_forecast = maize_class.run_class(data)

        if metric:


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
                used_band_model =  'ALPS'
                normal_band_limit = round(row[8],4) 
                stress_band_limit = round(row[9],4)
                alert_band_limit = round(row[10],4)

                vector = (product_name,market_id,source_id,currency_code,date_price,
                            observed_price,observed_class,used_band_model,date_run_model,
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
                                    used_band_model,
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

            connection.close()
        
        else:

            print('The combination:',product_name, market_id, source_id, currency_code, 'has problems.')
            market_with_problems.append((product_name, market_id, source_id, currency_code))
        #     pass


        return market_with_problems

        # except (Exception, psycopg2.Error) as error:
        #     print('Error dropping the data.')

        # finally:


        #     if (connection):
        #         cursor.close()
        #         connection.close()





# if __name__ == "__main__":

#     # pctwo_retail, pctwo_wholesale = possible_maize_markets()
#     # print(pctwo_retail)