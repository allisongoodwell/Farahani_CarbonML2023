# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 16:11:31 2023

@author: askarzam
"""


# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm.notebook import trange
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

class MyDataset(Dataset):
    def __init__(self,X,y,sequence_length=12):

        self.sequence_length = sequence_length #Length of LSTM
        self.y = y #Target variable (Last column)
        self.X = X #Input Variables

    
    def __len__(self):
        return (self.X.shape[0])
    
    def __getitem__(self,i):
        
        if i < self.sequence_length:
            
            X_part_b = self.X[:i+1]
            X_part_a = np.repeat([self.X[0]],self.sequence_length - len(X_part_b),axis = 0) #padding 
            
            y_part_b = self.y[:i+1]
            y_part_a = np.repeat(self.y[0],self.sequence_length - len(y_part_b),axis = 0) #padding
            
            inputs = np.array(np.concatenate((X_part_a,X_part_b),axis = 0))
            targets = np.array(np.concatenate((y_part_a,y_part_b),axis = 0))
            
        else:
            inputs = np.array(self.X[i-self.sequence_length+1:i+1])
            targets = np.array(self.y[i-self.sequence_length+1:i+1])
            
        return torch.from_numpy(inputs).float(),torch.from_numpy(targets).float() 
        #conversion required for Pytorch (everything as floats)
    
    
    
class LSTM(nn.Module):
    def __init__(self, input_size,hidden_size,num_layers = 1):
        super(LSTM, self).__init__()
        
        self.input_size = input_size #number of features
        self.hidden_size = hidden_size
        self.num_layers = num_layers #number of stacked LSTM
        
        #describe the architecture
        self.rnn = nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,batch_first=True)
        
        self.fc = nn.Linear(hidden_size,1) 
        
    def forward(self,x):
        #forward pass
        batch_size = x.shape[0]
        #  initialize a hidden state and cell state for the LSTM as this is the first cell.
        
        h0 = torch.zeros(self.num_layers,batch_size,self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.num_layers,batch_size,self.hidden_size).requires_grad_()
        
        # The hidden state and cell state is stored in a tuple with the format (hidden_state, cell_state)
        out,(h_out,c_out) = self.rnn(x,(h0,c0)) 
        #out:hidden for each time step(sequence length) for final layer(# stacked lstm) 
        #h_out,c_out: hidden and cell state for final time step for each layer
        
        out = self.fc(out) 
        
        out = torch.reshape(out,x.shape[:2])
        
        return out
    
    
def train_model(data_loader,model,loss_function,optimizer,device):
    
    num_batches = len(data_loader)
    
    total_loss = 0
    model.to(device)
    model.train()
    
    for X,y in data_loader:
        X,y = X.to(device),y.to(device)
        y_hat = model(X)
        # print(y_hat,y)
        loss = loss_function(y_hat,y) #Confirm how to calculate loss for many-to-many LSTMs

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / num_batches
    print(f'Train loss: {avg_loss}')
    
    return avg_loss

def test_model(data_loader,model,loss_function,device):
    
    num_batches = len(data_loader)
    
    total_loss = 0
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        for X,y in data_loader:
            X,y = X.to(device),y.to(device)
            y_hat = model(X)
            
            loss = loss_function(y_hat,y)

            total_loss += loss.item()

    avg_loss = total_loss / num_batches
    print(f'Test loss: {avg_loss}')
    print('\n------------------------------')
    return avg_loss


def prediction(data_loader,model,device):
    num_batches = len(data_loader)
    
    total_loss = 0
    model.to(device)
    model.eval()

    output = torch.tensor([],device=device)
    
    with torch.no_grad():
        for X,y in data_loader:
            
            X,y = X.to(device),y.to(device)
            y_hat = model(X)
            
            output = torch.cat((output,y_hat[:,-1]),0)
    
    return output

dataset= ['Ne1', 'Ne2','Ne3','Br1','Br3','GC']
def LSTM_l(frames,method):
    
    y_l={}
    X_l={}
    y_predict_LSTM_l={}
    prediction_LSTM_DS_l={}
    data_scaled_l={}
    

    sequence_length = 12
    batch_size = 128
    hidden_size = 20
    num_layers = 3
    epoch = 50
    learning_rate = 1e-3 
    
    
    for i in frames.keys():
        
        print(dataset[i])

        
        if method == 'Standard':
            scaler = StandardScaler()
            data_Norm  = scaler.fit_transform(frames[i])
            data_Norm= pd.DataFrame(data_Norm,columns=['Ta', 'RH', 'P', 'TS', 'PPFD','NETRAD', 'WS', 'Pa','SWC', 'Fc'],index=frames[i].index)
        
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
        
    
        # we can choose one of the above scaling medthod:
        data_scaled_l[i]=data_Norm
    
    
        y_l[i]= np.array(data_scaled_l[i]['Fc'])
        X_l[i]= np.array(data_scaled_l[i].iloc[:,0:9])
    
        # splitting X and y into training and testing sets
        X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X_l[i], y_l[i], test_size=0.3)

        TrainDataset_l = MyDataset(X_train_l,y_train_l,sequence_length) 
        TestDataset_l = MyDataset(X_test_l,y_test_l,sequence_length)
    
    
        TrainLoader_l = DataLoader(TrainDataset_l,batch_size=batch_size, shuffle=True) #DataLoader builtin pytorch
        TestLoader_l = DataLoader(TestDataset_l,batch_size= batch_size, shuffle=True)
    
        # input_size = (batch_size,sequence_length,10)
    
    
        model_l = LSTM(9,hidden_size=hidden_size,num_layers=num_layers)
        loss_function_l = nn.MSELoss()
        optimizer_l = torch.optim.Adam(model_l.parameters(),lr=learning_rate)
    
        train_loss_l = []
        test_loss_l = []
        for j in trange(epoch):
            print()
            print(f'Epoch {j}\n--------')
    
            device = torch.device('cpu')
            train_loss_l.append(train_model(TrainLoader_l,model_l,loss_function_l,optimizer_l,device))
            test_loss_l.append(test_model(TestLoader_l,model_l,loss_function_l,device))
            
        y_predict_LSTM_l[i] = prediction(TestLoader_l,model_l,device).cpu().numpy()

    
        #Predicting for entire dataset
        Dataset_l = MyDataset(X_l[i],y_l[i],sequence_length) 
        DSLoader_l = DataLoader(Dataset_l,batch_size=batch_size, shuffle=False)
        
        prediction_LSTM_DS_l[i] = prediction(DSLoader_l,model_l,device).cpu().numpy()
        
        X_l[i]= pd.DataFrame(X_l[i], columns=['Ta', 'RH', 'P', 'TS', 'PPFD','NETRAD', 'WS', 'Pa','SWC'],index=frames[i].index)
        # add observation and modeled Fc to X
        X_l[i]['Fc_obs']=y_l[i]
        X_l[i]['Fc_model']=prediction_LSTM_DS_l[i]
        
        # Evaluating a LSTM
        print('MAE for entire dataset:', mean_absolute_error(y_l[i], prediction_LSTM_DS_l[i] ))
        print('MSE for entire dataset:', mean_squared_error(y_l[i], prediction_LSTM_DS_l[i] ))
        print('RMSE for entire dataset:', np.sqrt(mean_squared_error(y_l[i], prediction_LSTM_DS_l[i] )))
        print('\n-------------------------------')
        
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




def LSTM_r(frames,method):
    y={}
    X_df={}
    X_train={}
    X_test={}
    y_train={}
    y_test={}
    
    sequence_length = 12
    batch_size = 128
    hidden_size = 20
    num_layers = 3
    epoch = 50
    learning_rate = 1e-3 

    for i in frames.keys():
        if method == 'Standard':
            scaler = StandardScaler()
            data_Norm  = scaler.fit_transform(frames[i])
            data_Norm= pd.DataFrame(data_Norm,columns=['Ta', 'RH', 'P', 'TS', 'PPFD','NETRAD', 'WS', 'Pa','SWC', 'Fc'],index=frames[i].index)
        
        if method == 'MinMax':
            scaler = MinMaxScaler() 
            data_Norm = scaler.fit_transform(frames[i])
            data_Norm= pd.DataFrame(data_Norm, columns=['Ta', 'RH', 'P', 'TS', 'PPFD','NETRAD', 'WS', 'Pa','SWC', 'Fc'],index=frames[i].index)
        
        if method == 'Quan':
        #IQR = 75th quantile — 25th quantile    
        #X_scaled = (X — X.median) / IQR
            scaler = RobustScaler() 
            data_Norm = scaler.fit_transform(frames[i])
            data_Norm= pd.DataFrame(data_Norm, columns=['Ta', 'RH', 'P', 'TS', 'PPFD','NETRAD', 'WS', 'Pa','SWC', 'Fc'],index=frames[i].index)
        
    
        # defining feature matrix(X) and response vector(y)
        y[i]= data_Norm['Fc']
        X_df[i]= data_Norm.iloc[:,0:9]
    
        # splitting X and y into training and testing sets
        X_train[i], X_test[i], y_train[i], y_test[i] = train_test_split(X_df[i], y[i], test_size=0.3)
    
    X_df= pd.concat({k:pd.DataFrame(v).T for k, v in X_df.items()}, axis=1).T
    y= pd.concat({k:pd.DataFrame(v).T for k, v in y.items()}, axis=1).T
    
    X_train= (pd.concat({k:pd.DataFrame(v).T for k, v in X_train.items()}, axis=1).T).values.tolist()
    
    X_test= (pd.concat({k:pd.DataFrame(v).T for k, v in X_test.items()}, axis=1).T).values.tolist()
    
    y_train= (pd.concat({k:pd.DataFrame(v).T for k, v in y_train.items()}, axis=1).T).values.tolist()
    
    y_test= (pd.concat({k:pd.DataFrame(v).T for k, v in y_test.items()}, axis=1).T).values.tolist()
    
    # convert to array
    X= np.array(X_df)
    y= (np.array(y).T)[0]
    
    X_train= np.array(X_train)
    X_test= np.array(X_test)
    y_train= (np.array(y_train).T)[0]
    y_test= (np.array(y_test).T)[0]
    


    TrainDataset = MyDataset(X_train,y_train,sequence_length) 
    TestDataset = MyDataset(X_test,y_test,sequence_length)


    TrainLoader= DataLoader(TrainDataset,batch_size=batch_size, shuffle=True) #DataLoader builtin pytorch
    TestLoader = DataLoader(TestDataset,batch_size= batch_size, shuffle=True)

    #input_size = (batch_size,sequence_length,10)


    model = LSTM(9,hidden_size=hidden_size,num_layers=num_layers)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

    train_loss = []
    test_loss = []
    for j in trange(epoch):
        print()
        print(f'Epoch {j}\n--------')

        device = torch.device('cpu')
        train_loss.append(train_model(TrainLoader,model,loss_function,optimizer,device))
        test_loss.append(test_model(TestLoader,model,loss_function,device))
        
        
    #y_predict_LSTM = prediction(TestLoader,model,device).cpu().numpy()


    #Predicting for entire dataset
    Dataset= MyDataset(X,y,sequence_length) 
    DSLoader= DataLoader(Dataset,batch_size=batch_size, shuffle=False)
    
    prediction_LSTM_DS= prediction(DSLoader,model,device).cpu().numpy()
    
    
    
    # add observation and modeled Fc to X
    X_df['Fc_obs']=y
    X_df['Fc_model']=prediction_LSTM_DS
    X_r = X_df
    
    
    print('MAE for entire dataset:', mean_absolute_error(y, prediction_LSTM_DS))
    print('MSE for entire dataset:', mean_squared_error(y, prediction_LSTM_DS ))
    print('RMSE for entire dataset:', np.sqrt(mean_squared_error(y, prediction_LSTM_DS)))
    
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

