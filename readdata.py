# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 15:49:33 2023

@author: askarzam
"""

import pandas as pd
import numpy as np



def readdata (d1,d2,d3,d4,d5):
    # N= number of datasets =5
    
    dataset= ['Ne1', 'Ne2','Ne3','Br1','Br3']

    Ne1 = pd.read_csv(d1,header=2)
    Ne2 = pd.read_csv(d2,header=2)
    Ne3 = pd.read_csv(d3,header=2)
    
    Br1 = pd.read_csv(d4, header = [0], parse_dates = ['TIMESTAMP_START', 'TIMESTAMP_END'])
    Br3 = pd.read_csv(d5, header = [0], parse_dates = ['TIMESTAMP_START', 'TIMESTAMP_END'])

    
    # we have some missing values in most of the variables in couple first years, so I decided to just derop those years
    # and start from 2010
    
    #find the index of 01/01/2010
    #print(Ne1[Ne1['TIMESTAMP_START']==201001010000].index.values)
    
    # Column Renaming and Extraction: Specific columns from each dataset are extracted 
    # and renamed for consistency. This ensures easier readability and further processing.
    
    
    #dropped those years for all datasets and start from 2010
    Variables_Ne1 = Ne1[['TIMESTAMP_START', 'TA_PI_F_1_1_1', 'RH_PI_F_1_1_1', 'P_PI_F_1_1_1', 'TS_PI_F_1_1_1',
                     'PPFD_IN_PI_F_1_1_1', 'NETRAD_PI_F_1_1_1', 'WS_1_2_1', \
                         'PA_PI_F_1_1_1', 'SWC_PI_F_1_1_1','FC_1_1_1']].iloc[78888:,:]
    Variables_Ne2 = Ne2[['TIMESTAMP_START', 'TA_PI_F_1_1_1', 'RH_PI_F_1_1_1', 'P_PI_F_1_1_1', 'TS_PI_F_1_1_1',
                     'PPFD_IN_PI_F_1_1_1', 'NETRAD_PI_F_1_1_1', 'WS_1_2_1', \
                         'PA_PI_F_1_1_1', 'SWC_PI_F_1_1_1','FC_1_1_1']].iloc[78888:,:]
    
    
    Variables_Ne3 = Ne3[['TIMESTAMP_START', 'TA_PI_F_1_1_1', 'RH_PI_F_1_1_1', 'P_PI_F_1_1_1', 'TS_PI_F_1_1_1',
                     'PPFD_IN_PI_F_1_1_1', 'NETRAD_PI_F_1_1_1', 'WS_1_2_1', \
                         'PA_PI_F_1_1_1', 'SWC_PI_F_1_1_1','FC_1_1_1']].iloc[78888:,:]
        
        

    
    
    Variables_Ne1 = Variables_Ne1.rename(columns={'TIMESTAMP_START': 'Time',\
                                                      'TA_PI_F_1_1_1': 'Ta','RH_PI_F_1_1_1': 'RH','P_PI_F_1_1_1': 'P',\
                                                     'TS_PI_F_1_1_1': 'TS','PPFD_IN_PI_F_1_1_1': 'PPFD','NETRAD_PI_F_1_1_1': 'NETRAD',\
                                                      'WS_1_2_1': 'WS','PA_PI_F_1_1_1': 'Pa',\
                                                      'SWC_PI_F_1_1_1': 'SWC','FC_1_1_1': 'Fc'})
    Variables_Ne2 = Variables_Ne2.rename(columns={'TIMESTAMP_START': 'Time',\
                                                      'TA_PI_F_1_1_1': 'Ta','RH_PI_F_1_1_1': 'RH','P_PI_F_1_1_1': 'P',\
                                                     'TS_PI_F_1_1_1': 'TS','PPFD_IN_PI_F_1_1_1': 'PPFD','NETRAD_PI_F_1_1_1': 'NETRAD',\
                                                      'WS_1_2_1': 'WS','PA_PI_F_1_1_1': 'Pa',\
                                                      'SWC_PI_F_1_1_1': 'SWC','FC_1_1_1': 'Fc'})
    
    Variables_Ne3 = Variables_Ne3.rename(columns={'TIMESTAMP_START': 'Time',\
                                                      'TA_PI_F_1_1_1': 'Ta','RH_PI_F_1_1_1': 'RH','P_PI_F_1_1_1': 'P',\
                                                     'TS_PI_F_1_1_1': 'TS','PPFD_IN_PI_F_1_1_1': 'PPFD','NETRAD_PI_F_1_1_1': 'NETRAD',\
                                                      'WS_1_2_1': 'WS','PA_PI_F_1_1_1': 'Pa',\
                                                      'SWC_PI_F_1_1_1': 'SWC','FC_1_1_1': 'Fc'})
        
        
        
    Variables_Br1 = Br1[['TIMESTAMP_START', 'TA', 'RH', 'P', 'TS_1', 'PPFD_IN', 'NETRAD', \
                         'WS', 'PA', 'SWC_1','FC']].iloc[:119700,:]
    Variables_Br3 = Br3[['TIMESTAMP_START', 'TA', 'RH', 'P', 'TS_1', 'PPFD_IN', 'NETRAD', \
                         'WS', 'PA', 'SWC_1','FC']].iloc[:119700,:]
    
    
    
    
    Variables_Br1 = Variables_Br1.rename(columns={'TIMESTAMP_START': 'Time','TA': 'Ta','RH': 'RH',\
                                                           'P': 'P', 'TS_1': 'TS','PPFD_IN': 'PPFD','NETRAD': 'NETRAD',\
                                                           'WS': 'WS','PA': 'Pa','SWC_1': 'SWC','FC': 'Fc'})
                                                  
                                                  
    Variables_Br3 = Variables_Br3.rename(columns={'TIMESTAMP_START': 'Time','TA': 'Ta','RH': 'RH',\
                                                           'P': 'P', 'TS_1': 'TS','PPFD_IN': 'PPFD','NETRAD': 'NETRAD',\
                                                           'WS': 'WS','PA': 'Pa','SWC_1': 'SWC','FC': 'Fc'})
        
    
    Variables_Ne1['Time']=  pd.to_datetime(Variables_Ne1['Time'], format='%Y%m%d%H%M')
    Variables_Ne2['Time']=  pd.to_datetime(Variables_Ne2['Time'], format='%Y%m%d%H%M')
    Variables_Ne3['Time']=  pd.to_datetime(Variables_Ne3['Time'], format='%Y%m%d%H%M')

    Variables_Br1['Time']=  pd.to_datetime(Variables_Br1['Time'])
    Variables_Br3['Time']=  pd.to_datetime(Variables_Br3['Time'])


    
    Variables_Ne1=Variables_Ne1.set_index('Time') 
    Variables_Ne2=Variables_Ne2.set_index('Time') 
    Variables_Ne3=Variables_Ne3.set_index('Time') 
    Variables_Br1=Variables_Br1.set_index('Time') 
    Variables_Br3=Variables_Br3.set_index('Time') 

    label= Variables_Ne1.columns
    
    Variables_Br1 = Variables_Br1.resample('H').mean()
    Variables_Br3 = Variables_Br3.resample('H').mean()
    
    # Handling Missing Values: 
    
    
    Variables_Ne1 = Variables_Ne1.replace(-9999, np.NaN)
    Variables_Ne2 = Variables_Ne2.replace(-9999, np.NaN)
    Variables_Ne3 = Variables_Ne3.replace(-9999, np.NaN)
    
    
    # replaced nan values with zero and mean and median:
    Variables_Ne1['Fc'] = Variables_Ne1['Fc'].fillna(Variables_Ne1['Fc'].mean())
    # replaced median for missing values for wind speed and NETRAD:
    Variables_Ne1['WS'] = Variables_Ne1['WS'].fillna(Variables_Ne1['WS'].median())
    Variables_Ne1['NETRAD'] = Variables_Ne1['NETRAD'].fillna(Variables_Ne1['NETRAD'].median())


    # replaced nan values with zero and mean and median:
    Variables_Ne2['Fc'] = Variables_Ne2['Fc'].fillna(Variables_Ne2['Fc'].mean())
    Variables_Ne2['SWC'] = Variables_Ne2['SWC'].fillna(Variables_Ne2['SWC'].mean())
    Variables_Ne2['WS'] = Variables_Ne2['WS'].fillna(Variables_Ne2['WS'].median())
    Variables_Ne2['NETRAD'] = Variables_Ne2['NETRAD'].fillna(Variables_Ne2['NETRAD'].median())
    
    
    # replaced nan values with mean and median:
    Variables_Ne3['Fc'] = Variables_Ne3['Fc'].fillna(Variables_Ne3['Fc'].mean())
    Variables_Ne3['SWC'] = Variables_Ne3['SWC'].fillna(Variables_Ne3['SWC'].mean())
    Variables_Ne3['Ta'] = Variables_Ne3['Ta'].fillna(Variables_Ne3['Ta'].mean())
    Variables_Ne3['TS'] = Variables_Ne3['TS'].fillna(Variables_Ne3['TS'].mean())
    Variables_Ne3['Pa'] = Variables_Ne3['Pa'].fillna(Variables_Ne3['Pa'].mean())
    Variables_Ne3['WS'] = Variables_Ne3['WS'].fillna(Variables_Ne3['WS'].median())
    Variables_Ne3['P'] = Variables_Ne3['P'].fillna(Variables_Ne3['P'].median())
    Variables_Ne3['PPFD'] = Variables_Ne3['PPFD'].fillna(Variables_Ne3['PPFD'].median())
    Variables_Ne3['NETRAD'] = Variables_Ne3['NETRAD'].fillna(Variables_Ne3['NETRAD'].median())
    Variables_Ne3['RH'] = Variables_Ne3['RH'].fillna(Variables_Ne3['RH'].median())
    

    
    Variables_Br1 = Variables_Br1.replace(-9999, np.NaN)
    Variables_Br3 = Variables_Br3.replace(-9999, np.NaN)

    # remove outlier
    Variables_Br1['Ta'][(Variables_Br1['Ta']<-50)]=np.NaN
    Variables_Br1['RH'][(Variables_Br1['RH']<0)]=np.NaN
    Variables_Br1['P'][(Variables_Br1['P']<0)]=np.NaN
    Variables_Br1['TS'][(Variables_Br1['TS']<-50)]=np.NaN
    Variables_Br1['PPFD'][(Variables_Br1['PPFD']<0)]=np.NaN
    Variables_Br1['NETRAD'][(Variables_Br1['NETRAD']<-100)]=np.NaN
    Variables_Br1['WS'][(Variables_Br1['WS']<-100)]=np.NaN
    Variables_Br1['Pa'][(Variables_Br1['Pa']<0)]=np.NaN
    Variables_Br1['SWC'][(Variables_Br1['SWC']<0)]=np.NaN
    Variables_Br1['Fc'][(Variables_Br1['Fc']<-50)]=np.NaN

    
    # replaced nan values with mean and median:
    Variables_Br1['Fc'] = Variables_Br1['Fc'].fillna(Variables_Br1['Fc'].mean())
    Variables_Br1['SWC'] = Variables_Br1['SWC'].fillna(Variables_Br1['SWC'].mean())
    Variables_Br1['Ta'] = Variables_Br1['Ta'].fillna(Variables_Br1['Ta'].mean())
    Variables_Br1['TS'] = Variables_Br1['TS'].fillna(Variables_Br1['TS'].mean())
    Variables_Br1['Pa'] = Variables_Br1['Pa'].fillna(Variables_Br1['Pa'].mean())
    Variables_Br1['WS'] = Variables_Br1['WS'].fillna(Variables_Br1['WS'].median())
    Variables_Br1['P'] = Variables_Br1['P'].fillna(Variables_Br1['P'].median())
    Variables_Br1['NETRAD'] = Variables_Br1['NETRAD'].fillna(Variables_Br1['NETRAD'].median())
    Variables_Br1['RH'] = Variables_Br1['RH'].fillna(Variables_Br1['RH'].median())
    

    # remove outlier
    Variables_Br3['Ta'][(Variables_Br3['Ta']<-50)]=np.NaN
    Variables_Br3['RH'][(Variables_Br3['RH']<0)]=np.NaN
    Variables_Br3['P'][(Variables_Br3['P']<0)]=np.NaN
    Variables_Br3['TS'][(Variables_Br3['TS']<-50)]=np.NaN
    Variables_Br3['PPFD'][(Variables_Br3['PPFD']<0)]=np.NaN
    Variables_Br3['NETRAD'][(Variables_Br3['NETRAD']<-100)]=np.NaN
    Variables_Br3['WS'][(Variables_Br3['WS']<-100)]=np.NaN
    Variables_Br3['Pa'][(Variables_Br3['Pa']<0)]=np.NaN
    Variables_Br3['SWC'][(Variables_Br3['SWC']<0)]=np.NaN
    Variables_Br3['Fc'][(Variables_Br3['Fc']<-50)]=np.NaN

    
    # replaced nan values with mean and median:
    Variables_Br3['Fc'] = Variables_Br3['Fc'].fillna(Variables_Br3['Fc'].mean())
    Variables_Br3['SWC'] = Variables_Br3['SWC'].fillna(Variables_Br3['SWC'].mean())
    Variables_Br3['Ta'] = Variables_Br3['Ta'].fillna(Variables_Br3['Ta'].mean())
    Variables_Br3['TS'] = Variables_Br3['TS'].fillna(Variables_Br3['TS'].mean())
    Variables_Br3['Pa'] = Variables_Br3['Pa'].fillna(Variables_Br3['Pa'].mean())
    Variables_Br3['WS'] = Variables_Br3['WS'].fillna(Variables_Br3['WS'].median())
    Variables_Br3['P'] = Variables_Br3['P'].fillna(Variables_Br3['P'].median())
    Variables_Br3['NETRAD'] = Variables_Br3['NETRAD'].fillna(Variables_Br3['NETRAD'].median())
    Variables_Br3['RH'] = Variables_Br3['RH'].fillna(Variables_Br3['RH'].median())
    
    

    # estimate missing PPFD_IN values based on NETRAD values using a linear regression model
    from sklearn.linear_model import LinearRegression
    
    # Filter rows where both PPFD_IN and NETRAD are not NaN
    train_Br1 = Variables_Br1.dropna()
    train_Br3 = Variables_Br3.dropna()
    
    # Fit a linear regression model for each dataset
    model_Br1 = LinearRegression().fit(train_Br1[['NETRAD']], train_Br1['PPFD'])
    model_Br3 = LinearRegression().fit(train_Br3[['NETRAD']], train_Br3['PPFD'])
    
    # Now, we'll replace missing PPFD_IN values with the estimates from our models
    for df, model in zip([Variables_Br1, Variables_Br3], [model_Br1, model_Br3]):
        missing_ppfd = df['PPFD'].isna()
        df.loc[missing_ppfd, 'PPFD'] = model.predict(df.loc[missing_ppfd, ['NETRAD']])
        
    
    # list of data frames
    Variables = [Variables_Ne1, Variables_Ne2, Variables_Ne3,Variables_Br1, Variables_Br3]
    
    # dictionary to save data sets
    frames={} 
    
    for key, value in enumerate(Variables):    
        frames[key] = value # assigning data frame from list to key in dictionary

        print(dataset[key])
        print(frames[key], "\n")


    return frames, label