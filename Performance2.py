# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 13:12:03 2023

@author: askarzam

"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import allfunctions as af
import Performance as per
import numpy as np
from collections import defaultdict
import joblib




def kge2(observed, simulated):
    """
    The observed and simulated inputs should be arrays of the same length, representing the observed
    and simulated time series data.

    """
    observed_mean = observed.mean()
    simulated_mean = simulated.mean()
    covariance = ((observed - observed_mean) * (simulated - simulated_mean)).mean()
    observed_stdev = observed.std()
    simulated_stdev = simulated.std()
    correlation = covariance / (observed_stdev * simulated_stdev)
    alpha = simulated_stdev / observed_stdev
    beta = simulated_mean / observed_mean
    kge = 1 - ((correlation - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2) ** 0.5
    return kge

def NSE(observed, predicted):
    """
    Calculates the Nash-Sutcliffe efficiency (NSE) between observed and predicted values.
    """
    mean_obs = sum(observed) / len(observed)
    numerator = sum([(observed[i] - predicted[i])**2 for i in range(len(observed))])
    denominator = sum([(observed[i] - mean_obs)**2 for i in range(len(observed))])
    NSE = 1 - numerator / denominator
    return NSE

colors = ['#008000', '#8CEF74', '#7EF2C7','#0000FF','#8080FF','#9e1b42']

label = ['CO2', 'Ta', 'RH', 'P', 'TS', 'PPFD', 'NETRAD', 'WS', 'Pa', 'SWC']
dataset=['Ne1', 'Ne2','Ne3','Br1','Br3']
    
def AfMI(X, Hz,Hz_predicted,bins_model):
    

    bins_obs =100

    MI_y=defaultdict(list)
    MI_y_predicted=defaultdict(list)
    
    MI_y_N=defaultdict(list)
    MI_y_predicted_N=defaultdict(list)
    
    for k in X.keys():
        mi_y=np.zeros(len(X[k].columns)-2)
        mi_y_p=np.zeros(len(X[k].columns)-2)
    
        for i in range(0,len(X[k].columns)-2):
            mi_y[i] = af.mutual_information(X[k], label[i],  'Fc_obs',  bins_obs)
            mi_y_p[i] = af.mutual_information(X[k], label[i],  'Fc_model',  int(bins_model[k]))
            
        MI_y[k].append(mi_y)
        MI_y_predicted[k].append(mi_y_p)
        
        # Normalized MI by : MI( Xs, Fc)/H(Fc)
        MI_y_N[k]= MI_y[k]/Hz[k]
        MI_y_predicted_N[k]= MI_y_predicted[k]/Hz_predicted[k]
        
    for i in X.keys():
            
        MI_y_N[i]= np.reshape(MI_y_N[i],len(X[k].columns)-2)    
        MI_y_predicted_N[i]= np.reshape(MI_y_predicted_N[i],len(X[k].columns)-2)    
    
  

    # Af,MI = (observed MI- predicted MI)/ Observed MI
    # obs> pred : positive values: random model
    # obs< pred : negative values: deterministic model

    MI_dif= pd.DataFrame({key: (MI_y_N[key] - MI_y_predicted_N[key])/MI_y_N[key] for key in X}, index=label)
    
    MI_dif.plot(kind='bar', align='center', width=0.8, color = colors)
    
    plt.legend(['Ne1', 'Ne2','Ne3','Br1','Br3'])
    plt.xlabel("Driving source")
    plt.ylabel("Af,MI ",fontweight='bold')
    plt.title('The single source level')
    Af_MI = plt.figure()
    plt.show()    

    return Af_MI

def calc_information(X, label, bins_obs, bins_model):
    itot = np.zeros((len(label), len(label), 2))
    u1 = np.zeros((len(label), len(label), 2))
    u2 = np.zeros((len(label), len(label), 2))
    r = np.zeros((len(label), len(label), 2))
    s = np.zeros((len(label), len(label), 2))
    
    for i in range(0, len(label)):
        for j in range(0, len(label)):
            obs_output = af.information_partitioning(X, label[i], label[j], 'Fc_obs', bins_obs)
            model_output = af.information_partitioning(X, label[i], label[j], 'Fc_model', int(bins_model))
    
            itot[i][j][0], u1[i][j][0], u2[i][j][0], r[i][j][0], s[i][j][0] = obs_output
            itot[i][j][1], u1[i][j][1], u2[i][j][1], r[i][j][1], s[i][j][1] = model_output
    
    
    return itot, u1, u2, r, s


def PIF(X,bins_model):
    
    vmin = 0
    vmax = 1

    bins_obs = 100
    
    Itot = {}
    R = {}
    S = {}
    U1 = {}
    U2 = {}
    
    num_cores = joblib.cpu_count()
    results = joblib.Parallel(n_jobs=num_cores)(joblib.delayed(per.calc_information)(X[k], label, bins_obs, bins_model[k]) for k in X.keys())
    
    for k, result in zip(X.keys(), results):
        Itot[k] = result[0]
        U1[k] = result[1]
        U2[k] = result[2]
        R[k] = result[3]
        S[k] = result[4]
        
        
    for i in X.keys():
    
        fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(25, 10))
        mask_S = np.triu(np.ones_like(S[i][:,:,0]))
        sns.heatmap(S[i][:,:,0], xticklabels=label, yticklabels=label,fmt='.2f',cmap='RdYlGn', vmin=vmin, vmax=vmax, cbar=True, center=0, annot=True, mask=mask_S, ax=axs[0,0], annot_kws={"fontsize": 10})
        axs[0,0].set_title(dataset[i]+': S_Observation', fontsize=14, fontweight='bold')
        sns.heatmap(S[i][:,:,1], xticklabels=label, yticklabels=label,fmt='.2f',cmap='RdYlGn',vmin=vmin, vmax=vmax, cbar=True, center=0, annot=True, mask=mask_S, ax=axs[1,0], annot_kws={"fontsize": 10})
        axs[1,0].set_title(dataset[i]+': S_Model', fontsize=14, fontweight='bold')
        
        mask_R = np.triu(np.ones_like(R[i][:,:,0]))
        sns.heatmap(R[i][:,:,0], xticklabels=label, yticklabels=label,fmt='.2f',cmap='RdYlGn',vmin=vmin, vmax=vmax, cbar=True, center=0, annot=True, mask=mask_R, ax=axs[0,1], annot_kws={"fontsize": 10})
        axs[0,1].set_title(dataset[i]+': R_Observation', fontsize=14, fontweight='bold')
        sns.heatmap(R[i][:,:,1], xticklabels=label, yticklabels=label,fmt='.2f',cmap='RdYlGn',vmin=vmin, vmax=vmax, cbar=True, center=0, annot=True, mask=mask_R, ax=axs[1,1], annot_kws={"fontsize": 10})
        axs[1,1].set_title(dataset[i]+': R_Model', fontsize=14, fontweight='bold')
    
        mask_U = np.triu(np.ones_like(U1[i][:,:,0]))
        sns.heatmap(U1[i][:,:,0], xticklabels=label, yticklabels=label,fmt='.2f', cmap='RdYlGn',vmin=vmin, vmax=vmax, cbar=True, center=0, annot=True, mask=mask_U, ax=axs[0,2], annot_kws={"fontsize": 10})
        axs[0,2].set_title(dataset[i]+': U1_Observation', fontsize=14, fontweight='bold')
        sns.heatmap(U1[i][:,:,1], xticklabels=label, yticklabels=label,fmt='.2f', cmap='RdYlGn',vmin=vmin, vmax=vmax, cbar=True, center=0, annot=True, mask=mask_U, ax=axs[1,2], annot_kws={"fontsize": 10})
        axs[1,2].set_title(dataset[i]+': U1_Model', fontsize=14, fontweight='bold')
    
            
        mask_U = np.triu(np.ones_like(U2[i][:,:,0]))
        sns.heatmap(U2[i][:,:,0], xticklabels=label, yticklabels=label,fmt='.2f', cmap='RdYlGn',vmin=vmin, vmax=vmax, cbar=True, center=0, annot=True, mask=mask_U, ax=axs[0,3], annot_kws={"fontsize": 10})
        axs[0,3].set_title(dataset[i]+': U2_Observation', fontsize=14, fontweight='bold')
        sns.heatmap(U2[i][:,:,1], xticklabels=label, yticklabels=label,fmt='.2f', cmap='RdYlGn',vmin=vmin, vmax=vmax, cbar=True, center=0, annot=True, mask=mask_U, ax=axs[1,3], annot_kws={"fontsize": 10})
        axs[1,3].set_title(dataset[i]+': U2_Model', fontsize=14, fontweight='bold')
        
        mask_I = np.triu(np.ones_like(Itot[i][:,:,0]))
        sns.heatmap(Itot[i][:,:,0], xticklabels=label, yticklabels=label, fmt='.2f',cmap='RdYlGn', center=0, annot=True, mask=mask_I, annot_kws={"fontsize": 10}, ax=axs[0,4], vmin=vmin, vmax=vmax, cbar=True, cbar_kws={"label": "PIF"})
        axs[0,4].set_title(dataset[i]+': Itot_Observation', fontsize=14, fontweight='bold')
        sns.heatmap(Itot[i][:,:,1], xticklabels=label, yticklabels=label, fmt='.2f',cmap='RdYlGn', center=0, annot=True, mask=mask_I, annot_kws={"fontsize": 10}, ax=axs[1,4], vmin=vmin, vmax=vmax, cbar=True, cbar_kws={"label": "PIF"})
        axs[1, 4].set_title(dataset[i]+': Itot_Model', fontsize=14, fontweight='bold')
        fig.tight_layout()
        plt.show()
        
    return fig,Itot,S,R,U1,U2
        

def Af_PID(X,S,R,U1,U2):
    #Af,S= S_model- S_obs: + overestimate , _ underestimate

    AfS=defaultdict(list)
    AfR=defaultdict(list)
    AfU1=defaultdict(list)
    AfU2=defaultdict(list)
    AfIpart =defaultdict(list)
    
    for i in X.keys():
        AfS[i]=S[i][:,:,1]-S[i][:,:,0]
        AfR[i]=R[i][:,:,1]-R[i][:,:,0]
        AfU1[i]=U1[i][:,:,1]-U1[i][:,:,0]
        AfU2[i]=U2[i][:,:,1]-U2[i][:,:,0]
        AfIpart[i]= abs(AfS[i])+abs(AfR[i])+abs(AfU1[i])+abs(AfU2[i])
    
        fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(25, 5))
        
        mask_S = np.triu(np.ones_like(AfS[i]))
        sns.heatmap(AfS[i], xticklabels=label, yticklabels=label,\
                        fmt='.2f',cmap='RdYlGn', center=0, annot=True, mask=mask_S, ax=axs[0], annot_kws={"fontsize": 10})
        axs[0].set_title(dataset[i]+': AfS')
     
    
        mask_R = np.triu(np.ones_like(AfR[i]))
        sns.heatmap(AfR[i], xticklabels=label, yticklabels=label,\
                        fmt='.2f',cmap='RdYlGn', center=0, annot=True, mask=mask_R, ax=axs[1], annot_kws={"fontsize": 10})
        axs[1].set_title(dataset[i]+': AfR')
    
    
        mask_U = np.triu(np.ones_like(AfU1[i]))
        sns.heatmap(AfU1[i], xticklabels=label, yticklabels=label,\
                        fmt='.2f', cmap='RdYlGn', center=0, annot=True, mask=mask_U, ax=axs[2], annot_kws={"fontsize": 10})
        axs[2].set_title(dataset[i]+': AfU1')
    
            
        sns.heatmap(AfU2[i], xticklabels=label, yticklabels=label,\
                        fmt='.2f', cmap='RdYlGn', center=0, annot=True, mask=mask_U, ax=axs[3], annot_kws={"fontsize": 10})
        axs[3].set_title(dataset[i]+': AfU2')

        
        mask_I = np.triu(np.ones_like(AfIpart[i]))
        sns.heatmap(AfIpart[i], xticklabels=label, yticklabels=label,\
                    fmt='.2f', cmap='RdYlGn', center=0, annot=True, mask=mask_I, ax=axs[4], annot_kws={"fontsize": 10})
        axs[4].set_title(dataset[i]+': AfIpart')


    return fig,AfS,AfR,AfU1,AfU2,AfIpart



def Af_Model(X,AfS,AfR,AfU1,AfU2,AfIpart,model_name):  
    Af_Model =defaultdict(list)

    for i in X.keys():
        AfS[i]=(np.triu(AfS[i], k=1)).reshape(1,100)
        AfR[i]=(np.triu(AfR[i], k=1)).reshape(1,100)
        AfU1[i]=(np.triu(AfU1[i], k=1)).reshape(1,100)
        AfU2[i]=(np.triu(AfU2[i], k=1)).reshape(1,100)
        AfIpart[i]=(np.triu(AfIpart[i], k=1)).reshape(1,100)
        
        Af_Model[i]= pd.DataFrame([AfS[i][np.nonzero(AfS[i])], AfR[i][np.nonzero(AfR[i])],AfU1[i][np.nonzero(AfU1[i])],\
              AfU2[i][np.nonzero(AfU2[i])],AfIpart[i][np.nonzero(AfIpart[i])]])
            
    Af_M= pd.DataFrame({key: Af_Model[key].sum(axis = 1)/Af_Model[0].shape[1] for key in X})

    Af_M.rename(index={0: "AfS_tot", 1: "AfR_tot", 2: "AfU1_tot",3: "AfU2_tot", 4:"AfIpart_tot"})
    Af_M.rename(columns={0: "Ne1", 1: "Ne2",2: "Ne3", 3: "Br1", 4: "Br2", 5:"Regional"})
    
        
    Af_M.plot(kind='bar', align='center', width=0.8, color = colors)
    X_axis = np.arange(5)
    plt.legend(dataset)
    plt.ylabel("Af,tot ",fontweight='bold')
    
    plt.xticks(X_axis, ["AfS_tot", "AfR_tot", \
                        "AfU1_tot","AfU2_tot", "AfIpart_tot"], rotation=0)
    plt.title('The model level: '+ model_name)
    Af_Model_fig = plt.figure()
    plt.show()
    
    return Af_M, Af_Model_fig
    
    