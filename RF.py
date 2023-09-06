# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 15:31:31 2023

@author: askarzam
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

dataset= ['Ne1', 'Ne2','Ne3','Br1','Br3','GC']

def RF_l(frames,method):
    
    y_l={}
    X_l={}
    y_predict_RF_l={}
    prediction_RF_DS_l={}
    
    
    for i in frames.keys():
        
        print(dataset[i])

        if method == 'MinMax':
            scaler = MinMaxScaler() 
            data_Norm = scaler.fit_transform(frames[i])
            data_Norm= pd.DataFrame(data_Norm, columns=['Ta', 'RH', 'P', 'TS', 'PPFD','NETRAD', 'WS', 'Pa','SWC', 'Fc'],index=frames[i].index)
        
        if method == 'Quan':
        #IQR = 75th quantile — 25th quantile    
        #X_scaled = (X — X.median) / IQR
            scaler = RobustScaler() 
            data_Norm = scaler.fit_transform(frames[i])
            data_Norm= pd.DataFrame(data_Norm, columns=[ 'Ta', 'RH', 'P', 'TS', 'PPFD','NETRAD', 'WS', 'Pa','SWC', 'Fc'],index=frames[i].index)
        
    
        # defining feature matrix(X) and response vector(y)
        y_l[i]= data_Norm['Fc']
        X_l[i]= data_Norm.iloc[:,0:9]
    
        # splitting X and y into training and testing sets
        X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X_l[i], y_l[i], test_size=0.2)

        # Training a RandomForestRegressor
        rfr = RandomForestRegressor(n_estimators=100, max_features='sqrt' , random_state=42)
    
        rfr.fit(X_train_l, y_train_l)
        # model accuracy score
        score = rfr.score(X_train_l, y_train_l)
        print("R-squared:", score) 
        #Predicting and accuracy check
        y_predict_RF_l[i] = rfr.predict(X_test_l)
        
        
        # Evaluating a RandomForestRegressor
        print('Mean Absolute Error:', mean_absolute_error(y_test_l, y_predict_RF_l[i] ))
        print('Mean Squared Error:', mean_squared_error(y_test_l, y_predict_RF_l[i] ))
        print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test_l, y_predict_RF_l[i] )))
        print('\n')

        #Predicting for entire dataset
        prediction_RF_DS_l[i] = rfr.predict(X_l[i])
        
        # add observation and modeled Fc to X
        X_l[i]['Fc_obs']=y_l[i]
        X_l[i]['Fc_model']=prediction_RF_DS_l[i]
        

        # Extract hour of day from datetime index
        hour = X_l[i].index.hour
        
        # Calculate the mean value for each hour of day
        diurnal_cycle = X_l[i].groupby(hour).mean()
        
        # Plot the diurnal cycle
        fig = diurnal_cycle.iloc[:,9:].plot()
        plt.xlabel('Hour of day')
        plt.ylabel('Normalized Fc')
        plt.title(dataset[i])
        plt.show()
        
             
    return X_l, fig
 
    
def RF_r(frames,method):
    y={}
    X={}
    X_train={}
    X_test={}
    y_train={}
    y_test={}

    for i in frames.keys():

        if method == 'MinMax':
            scaler = MinMaxScaler() 
            data_Norm = scaler.fit_transform(frames[i])
            data_Norm= pd.DataFrame(data_Norm, columns=['Ta', 'RH', 'P', 'TS', 'PPFD','NETRAD', 'WS', 'Pa','SWC', 'Fc'],index=frames[i].index)
        
        if method == 'Quan':
        #IQR = 75th quantile — 25th quantile    
        #X_scaled = (X — X.median) / IQR
            scaler = RobustScaler() 
            data_Norm = scaler.fit_transform(frames[i])
            data_Norm= pd.DataFrame(data_Norm, columns=[ 'Ta', 'RH', 'P', 'TS', 'PPFD','NETRAD', 'WS', 'Pa','SWC', 'Fc'],index=frames[i].index)
        
    
        # defining feature matrix(X) and response vector(y)
        y[i]= data_Norm['Fc']
        X[i]= data_Norm.iloc[:,0:9]
    
        # splitting X and y into training and testing sets
        X_train[i], X_test[i], y_train[i], y_test[i] = train_test_split(X[i], y[i], test_size=0.2)

    
    X_train= (pd.concat({k:pd.DataFrame(v).T for k, v in X_train.items()}, axis=1).T).values.tolist()
    
    X_test= (pd.concat({k:pd.DataFrame(v).T for k, v in X_test.items()}, axis=1).T).values.tolist()
    
    y_train= (pd.concat({k:pd.DataFrame(v).T for k, v in y_train.items()}, axis=1).T).values.tolist()
    
    y_test= (pd.concat({k:pd.DataFrame(v).T for k, v in y_test.items()}, axis=1).T).values.tolist()


    # Training a RandomForestRegressor
    rfr = RandomForestRegressor(n_estimators=100, max_features='sqrt' , random_state=42)

    # train the model using the training sets
    rfr.fit(X_train, y_train)
    
    # model accuracy score
    score = rfr.score(X_train, y_train)
    print("R-squared:", score)

    #Predicting and accuracy check
    y_pred= rfr.predict(X_test)
    
    # Evaluating a RandomForestRegressor
    print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', mean_squared_error(y_test, y_pred ))
    print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred )))
    print('\n')

    
    X= pd.concat({k:pd.DataFrame(v).T for k, v in X.items()}, axis=1).T
    y= pd.concat({k:pd.DataFrame(v).T for k, v in y.items()}, axis=1).T
    
    # Predict using the best model for each dataset
    y_predict = rfr.predict(X)

    # add observation and modeled Fc to X
    X['Fc_obs']=y
    X['Fc_model']=y_predict
    X_r = X
    
    for i in frames.keys():
        # Extract hour of day from datetime index
        hour = X_r.loc[i].index.hour
    
        # Calculate the mean value for each hour of day
        diurnal_cycle = X_r.loc[i].groupby(hour).mean()
    
        # Plot the diurnal cycle
        fig = diurnal_cycle.iloc[:,9:].plot()
        plt.xlabel('Hour of day')
        plt.ylabel('Normalized Fc')
        plt.title(dataset[i])
        plt.show()
       
    return X_r, fig



def RF(frames):
    y={}
    X={}
    
    X_train={}
    X_test={}
    y_train={}
    y_test={}

    for i in frames.keys():

        scaler = MinMaxScaler() 
        data_Norm = scaler.fit_transform(frames[i])
        data_Norm= pd.DataFrame(data_Norm, columns=['Ta', 'RH', 'P', 'TS', 'PPFD','NETRAD', 'WS', 'Pa','SWC', 'Fc'],index=frames[i].index)

    
        # defining feature matrix(X) and response vector(y)
        y[i]= data_Norm['Fc']
        X[i]= data_Norm.iloc[:,0:9]
    
        # splitting X and y into training and testing sets
        X_train[i], X_test[i], y_train[i], y_test[i] = train_test_split(X[i], y[i], test_size=0.8)

    
    X_train= (pd.concat({k:pd.DataFrame(v).T for k, v in X_train.items()}, axis=1).T).values.tolist()
    
    X_test= (pd.concat({k:pd.DataFrame(v).T for k, v in X_test.items()}, axis=1).T).values.tolist()
    
    y_train= (pd.concat({k:pd.DataFrame(v).T for k, v in y_train.items()}, axis=1).T).values.tolist()
    
    y_test= (pd.concat({k:pd.DataFrame(v).T for k, v in y_test.items()}, axis=1).T).values.tolist()


    # Create a RandomForestRegressor
    rfr = RandomForestRegressor(random_state=42)

    # Define the hyperparameters and their possible values
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 6, 9],
        'max_features': ['auto', 'sqrt', 'log2']
    }

    # Use GridSearchCV with the Random Forest Regressor
    grid_search = GridSearchCV(estimator=rfr, param_grid=param_grid, cv=3, n_jobs=-1)

    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)

    # Get the best parameters
    best_params = grid_search.best_params_
    print("Best Parameters:", best_params)

    # Get the best estimator
    best_model = grid_search.best_estimator_

    # model accuracy score
    score = best_model.score(X_train, y_train)
    print("R-squared:", score)

