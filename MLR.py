# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 13:36:00 2023

@author: askarzam


"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,RobustScaler
from sklearn import linear_model
from sklearn.model_selection import train_test_split


dataset= ['Ne1', 'Ne2','Ne3','Br1','Br3','GC']  
      
def MLR_l(frames,method):
    
    y_l={}
    X_l={}
    y_predict_Linear_l={}
    
    
    for i in frames.keys():
        
        print(dataset[i])

        if method == 'MinMax':
            scaler = MinMaxScaler() 
            data_Norm = scaler.fit_transform(frames[i])
            data_Norm= pd.DataFrame(data_Norm, columns=['Ta', 'RH', 'P', 'TS', 'PPFD','NETRAD', 'WS', 'Pa','SWC', 'Fc'],\
                                    index=frames[i].index)
        
        if method == 'Quan':
        #IQR = 75th quantile — 25th quantile    
        #X_scaled = (X — X.median) / IQR
            scaler = RobustScaler() 
            data_Norm = scaler.fit_transform(frames[i])
            data_Norm= pd.DataFrame(data_Norm, columns=['Ta', 'RH', 'P', 'TS', 'PPFD','NETRAD', 'WS', 'Pa','SWC', 'Fc'],\
                                    index=frames[i].index)
        
    
        # defining feature matrix(X) and response vector(y)
        y_l[i]= data_Norm['Fc']
        X_l[i]= data_Norm.iloc[:,0:9]
    
        # splitting X and y into training and testing sets
        X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X_l[i], y_l[i], test_size=0.2)
    
        # create linear regression object
        reg = linear_model.LinearRegression()
    
        # train the model using the training sets
        reg.fit(X_train_l, y_train_l)
    
        # regression coefficients
        print('Coefficients: ', reg.coef_)
    
        # variance score: 1 means perfect prediction
        print('Test Variance Score: {}'.format(reg.score(X_test_l, y_test_l)))
        
        
        y_predict_Linear_l[i]= reg.predict(X_l[i])
        error_l= y_l[i]-y_predict_Linear_l[i]
        print('error y-y_predic: ',np.sum(error_l))
        var_y_l = np.var(y_predict_Linear_l[i]) # variance
        print('varianceof prediction: ', var_y_l)
        sd_y_l= np.std(y_predict_Linear_l[i]) # standard deviation, the average error associated with the mean
        print('standard deviation of prediction: ', sd_y_l) 
        
        # add observation and modeled Fc to X
        X_l[i]['Fc_obs']=y_l[i]
        X_l[i]['Fc_model']=y_predict_Linear_l[i]

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

def MLR_r(frames,method):
    
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
            data_Norm= pd.DataFrame(data_Norm, columns=[ 'Ta', 'RH', 'P', 'TS', 'PPFD','NETRAD', 'WS', 'Pa','SWC', 'Fc'],index=frames[i].index)
        
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
    
    X= pd.concat({k:pd.DataFrame(v).T for k, v in X.items()}, axis=1).T
    y= pd.concat({k:pd.DataFrame(v).T for k, v in y.items()}, axis=1).T
    
    X_train= (pd.concat({k:pd.DataFrame(v).T for k, v in X_train.items()}, axis=1).T).values.tolist()
    
    X_test= (pd.concat({k:pd.DataFrame(v).T for k, v in X_test.items()}, axis=1).T).values.tolist()
    
    y_train= (pd.concat({k:pd.DataFrame(v).T for k, v in y_train.items()}, axis=1).T).values.tolist()
    
    y_test= (pd.concat({k:pd.DataFrame(v).T for k, v in y_test.items()}, axis=1).T).values.tolist()
    
    # create linear regression object
    reg = linear_model.LinearRegression()
    
    # train the model using the training sets
    reg.fit(X_train, y_train)
    
    # regression coefficients
    print('Coefficients: ', reg.coef_)
    
    # variance score: 1 means perfect prediction
    print('Test Variance Score: {}'.format(reg.score(X_test, y_test)))
     
    y_predict_Linear= reg.predict(X)
    error= y-y_predict_Linear
    print('error y-y_predic: ',np.sum(error))
    var_y = np.var(y_predict_Linear) # variance
    print('varianceof prediction: ', var_y)
    sd_y= np.std(y_predict_Linear) # standard deviation, the average error associated with the mean
    print('standard deviation of prediction: ', sd_y) 
    
    # add observation and modeled Fc to X
    X['Fc_obs']=y
    X['Fc_model']=y_predict_Linear
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
