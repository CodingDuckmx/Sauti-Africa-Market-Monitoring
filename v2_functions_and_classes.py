import datetime
import numpy as np
import os
import pandas as pd
import psycopg2

from dotenv import load_dotenv, find_dotenv
from scipy import stats
from sklearn.linear_model import LinearRegression

load_dotenv()

class Clean_and_classify_class:
    def __init__(self):
        self.description = ''' This class groups the main functions that are used to clean, 
                                prepare the data, build  the ALPS bands and also label the
                                maize prices.'''
          
    def last_four_year_truncate(self,df):
        ''' Trunks the data for the last four years'''

        # Verify the date exists.

        try:

            start_point = df['date_price'].max() - datetime.timedelta(weeks=212)

            l4y = df[df['date_price'] >= start_point].copy()

            return l4y

        except:

            return pd.DataFrame()

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
                
                Q95 = cfd.quantile(0.95).values[0]
                selected_indices = list(cfd[cfd.iloc[:,-1] > Q95].index)
                for i in selected_indices:
                    cfd.iloc[i,-1] = cfd.iloc[i,-1] / 10
                    
    
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


        df = df[df['date_price'] > datetime.date(2018,12,31)]
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
        bands = bands.rename(columns={0:'date_price',1:'normal_band_limit',2:'stress_band_limit',3:'alert_band_limit'})

        return bands 

    def assign_classification(self,data,bands):

        '''
        Combine the data from the prices and the bands to classify the price in its status.
        '''

        results = data.copy()

        results['Observed_class'] = None
        results['Stressness'] = None

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
        
        metric, cleaned = self.basic_cleanning(self.last_four_year_truncate(data))
        try:
            stop_0, forecasted_prices = self.inmediate_forecast_ALPS_based(self.prepare_data_to_ALPS(cleaned))
            result, mae = self.build_bands_wfp_forecast(self.prepare_data_to_ALPS(cleaned),stop_0,forecasted_prices)

            return metric, stop_0, result, mae
        
        except:

            return None, None, None, None