import time
from datetime import datetime, date
from sqlalchemy import create_engine
import psycopg2
# import pymysql
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy import create_engine

from v2_functions_and_classes import dbConnect, DataCleaning, DataQualityCheck


def create_qc_tables(sale_type):

    # User input: sale type 
    sale_type = sale_type

    ## read data from stakeholder's database
    # instantiate db class
    db_c = dbConnect()
    raw_data = db_c.read_raw_table()
    # print('data retrieved for local analysis')

    ## Clean data
    dc = DataCleaning()
    df = raw_data.copy()
    df  = dc.read_data(df)
    df = dc.remove_zeros(df)
    df = dc.convert_dtypes(df)
    # print('data cleaned')

    # All product list
    PRODUCT_LIST = df['product_name'].unique().tolist()
    # All market list
    MARKET_LIST = df['market_id'].unique().tolist()
    # All source list
    SOURCE_LIST = df['source_id'].unique().tolist()

    # prepare table for data quality dimension
    col_names = ['market_id', 'product_name', 'source_id', 'start', 'end', 'timeliness', 'data_length', 'completeness', 'duplicates', 'mode_D']

    m = len(MARKET_LIST)*len(PRODUCT_LIST)*len(SOURCE_LIST)
    # n = len(col_names)
    # print(f'Anticipate qc talbe size is {m*n} entries')

    start_time = time.time()
    ## Generate quality table with specified data quality dimensions

    # instantiate qc class
    qc = DataQualityCheck()

    # initialize QC table
    QC = [[] for _ in range(m)] 
    i = 0
    for MARKET in MARKET_LIST:
        for PRODUCT in PRODUCT_LIST:
            for SOURCE in SOURCE_LIST:
                # print(MARKET, PRODUCT, SOURCE)
                # apply filters
                cond1 = (df['product_name']==PRODUCT)
                cond2 = (df['source_id']==SOURCE)
                cond3 = (df['market_id']==MARKET)
                            
                subset = df[cond1 & cond2 & cond3].sort_values(by='date_price', ascending=True).set_index('date_price')
                
                # this is the sale time series
                sale = subset[[sale_type]] 
                
                if sale.empty:
                    break
                
                if len(sale)==sale.isnull().sum().values[0]:
                    break
                    
                else:
                    QC_i = qc.generate_QC(sale, figure_output=0)
                    QC[i] = [MARKET, PRODUCT, SOURCE] + QC_i
                    i = i+1

    # write to DQ dataframe
    QC_df = pd.DataFrame(columns=col_names, data = QC)

    # remove valid data rows but containing NaN in mode_D and duplicates
    QC_df = QC_df[~(QC_df['duplicates'].isnull() | QC_df['mode_D'].isnull())]

    # add id column
    QC_df['qc_id'] = QC_df.index
    cols = QC_df.columns.tolist()
    # move id to first column location
    cols = [cols[-1]] + cols[:-1]
    QC_df = QC_df[cols]

    # add an other col to get valid datapoint
    QC_df['data_points']=round(QC_df['data_length']*QC_df['completeness']-QC_df['duplicates'], 0)

    # convert mode_D to float
    QC_df['mode_D'] = QC_df['mode_D'].astype(str).str.rstrip(' days').astype(float)

    # transfer num columns to minmax scale
    cat_cols = ['market_id', 'product_name', 'source_id']
    num_cols = ['timeliness', 'data_length', 'completeness', 'duplicates', 'mode_D','data_points']

    cat_vars = QC_df[cat_cols]
    num_vars = QC_df[num_cols]

    column_trans = ColumnTransformer(
        [('scaled_numeric', MinMaxScaler(), num_cols), 
        ],
        remainder="drop",
    )

    X = column_trans.fit_transform(num_vars)

    # tdf: transformed df 
    tdf = QC_df.copy()
    tdf[num_cols]=X

    # Rank the data for ML candidate using data quality index (DQI)
    # Higher DQI, better data quality
    # DQI based on six data quality dimensions:
    D1, W1 = tdf['data_length'], 0.6
    D2, W2 = tdf['completeness'], 0.3
    D3, W3 = 1-tdf['mode_D'], 0.9
    D4, W4 = 1-tdf['timeliness'], 0.9
    D5, W5 = 1-tdf['duplicates'], 0.3
    D6, W6 = tdf['data_points'], 0.9

    QC_df['DQI'] = D1*W1 + D2*W2 + D3*W3 + D4*W4 + D5*W5 + D6*W6

    QC_df['DQI_cat']=pd.qcut(QC_df['DQI'], [0, .25, .5, .75, 0.9, 1.], labels = [ "poor", "medium", "fair", "good", "great"])



    # populate database with new qc table
    tablename = 'qc_' + sale_type
    db_c.populate_analytical_db(QC_df, tablename)

    elapsed_time = time.time()-start_time
    print(f"--- QC table generated for {sale_type}  and successfually initiated in AWS db. Elapsed time ={elapsed_time/60} minutes! ---" )



if __name__ == "__main__":
    
    create_qc_tables('retail_observed_price')
    create_qc_tables('wholesale_observed_price')

