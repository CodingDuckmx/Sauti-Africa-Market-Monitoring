
import datetime
import numpy as np
import pandas as pd

from scipy import stats
from sklearn.linear_model import LinearRegression

def set_columns(data):

    data = pd.DataFrame(data)
    data = data.rename(columns={0:'date_price',1:'unit_scale',2:'observed_price'})

    return data

def set_columns_bands_df(bands):

    bands= pd.DataFrame(bands)
    bands = bands.rename(columns={0:'date_price',1:'normal_band_limit',2:'stress_band_limit',3:'alert_band_limit'})

    return bands 

def last_four_year_truncate(dataframe):
      
    start_point = dataframe['date_price'].max() - datetime.timedelta(weeks=212)
    
    l4y = dataframe[dataframe['date_price'] >= start_point].copy()
    
    return l4y


def limit_2019_and_later(dataframe):

    ''' 
    Limit the info to the 2020 or later and assigns its month, so the price could be compared with the bands.
    '''


    dataframe = dataframe[dataframe['date_price'] > datetime.date(2018,12,31)]
    dataframe['date_price'] = dataframe['date_price'].astype('datetime64')
    dataframe['month'] = [str(dataframe.iloc[i,0])[:8] + '01' for i in range(len(dataframe))]
    dataframe = dataframe.reset_index(drop=True)

    return dataframe



def basic_cleanning(dataframe):

    ''' 
    Removes duplicates in dates column. 
    Verify unique unit scale.
    Try to correct typos.

    Returns the metric and the dataframe with the basic cleaned data.
    '''

    cfd = dataframe.copy()    
   
    # Remove duplicates in dates column.
    
    drop_index = list(cfd[cfd.duplicated(['date_price'], keep='first')].index)
    
    cfd = cfd.drop(labels=drop_index, axis=0).reset_index(drop=True)

    # Consider the mode of unit scale as the one.
    
    mode = stats.mode(cfd.iloc[:,1])[0][0]
    
    discording_scale = list(cfd[cfd['unit_scale'] != mode].index)
    
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

    return mode, cfd


def prepare_data_to_ALPS(dataframe):
    
    ''' 
    
    Make a dataframe with the last Sunday before the dates of the input dataframe, and the saturday of the last week in within the dates.
    Then Merge both dataframes to have one with all the possible weeks within the dates of the original dataframe.
    Interpolate the missing values.
    Returns the metric and the dataframe with the values.
    '''
    

    mode, cfd = basic_cleanning(dataframe)
    

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

    return mode, cfd


def inmediate_forecast_ALPS_based(dataframe):
    
    forecasted_prices = []
    
    basesetyear = dataframe.index.max().year - 2
    
    stop_0 = datetime.date(year=basesetyear,month=12,day=31)
    
    baseset = dataframe.iloc[:len(dataframe.loc[:stop_0]),:].copy()   
   

    # For all the past months:
    for i in range(len(dataframe)-len(baseset)):

        workset = dataframe.iloc[:len(dataframe.loc[:stop_0]) + i,:].copy()

        # What month are we?

        workset['month'] = workset.index.month

        # Build dummy variables for the months.

        workset = workset.join(pd.get_dummies(workset['month']))
        workset = workset.drop(labels=['month'], axis=1)

        features = workset.columns[1:]
        target = workset.columns[0]

        X = workset[features]
        y = workset[target]

        reg = LinearRegression()

        reg = reg.fit(X,y)

        next_month = dataframe.iloc[len(dataframe.loc[:stop_0]) + i,:].name

        raw_next_month = [0 if j != next_month.month else 1 for j in range(1,13)]

        next_month_array = np.array(raw_next_month).reshape(1,-1)

        forecasted_prices.append(reg.predict(next_month_array)[0])
        
    # For the current month.
    
    raw_next_month = [0 if j != next_month.month + 1 else 1 for j in range(1,13)]
    
    next_month_array = np.array(raw_next_month).reshape(1,-1)
    
    forecasted_prices.append(reg.predict(next_month_array)[0])    
    
    return stop_0, forecasted_prices
    



def build_bands_wfp_forecast(dataframe):
       
    dataframe['residuals'] = dataframe.iloc[:,0] - dataframe['forecast']
    dataframe['cum_residual_std'] = [np.std(dataframe.iloc[:i,2]) for i in range(1,len(dataframe)+1)]
    dataframe['ALPS'] = [None] + list(dataframe.iloc[1:,2]  / dataframe.iloc[1:,3])
    dataframe['Price Status'] = None
       
    for date in range(len(dataframe)-1):

        if dataframe.iloc[date,4] < 0.25:
            dataframe.iloc[date,5] = 'Normal'
        elif dataframe.iloc[date,4] < 1:
            dataframe.iloc[date,5] = 'Stress'
        elif dataframe.iloc[date,4] < 2:
            dataframe.iloc[date,5] = 'Alert'
        else:
            dataframe.iloc[date,5] = 'Crisis'
       
    dataframe['normal_limit'] = dataframe['forecast'] + 0.25 * dataframe['cum_residual_std']
    dataframe['stress_limit'] = dataframe['forecast'] + dataframe['cum_residual_std']
    dataframe['alert_limit'] = dataframe['forecast'] + 2 * dataframe['cum_residual_std']
    
       
    return dataframe



def assign_classification(data,bands):

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

