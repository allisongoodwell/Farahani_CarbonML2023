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


label = ['Ta', 'RH', 'P', 'TS', 'PPFD', 'NETRAD', 'WS', 'Pa', 'SWC']
dataset= ['Ne1', 'Ne2','Ne3','Br1','Br3','GC']


def KGE(observed, simulated):
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
    KGE = 1 - ((correlation - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2) ** 0.5
    
    return KGE, correlation, alpha, beta

def NSE(observed, predicted):
    """
    Calculates the Nash-Sutcliffe efficiency (NSE) between observed and predicted values.
    """
    mean_obs = sum(observed) / len(observed)
    numerator = sum([(observed[i] - predicted[i])**2 for i in range(len(observed))])
    denominator = sum([(observed[i] - mean_obs)**2 for i in range(len(observed))])
    NSE = 1 - numerator / denominator
    
    return NSE


def Entropy_LR(X_l,X_r):

    bin_min = X_l[0]['Fc_obs'].min()
    bin_max = X_l[0]['Fc_obs'].max()
    bins_obs =100
    rang = [np.linspace(bin_min, bin_max, bins_obs)]
    
    Hz = np.zeros([len(X_l.keys()),3]) # 0:obs  1:local, 2: regional
    
    for i in range(len(X_l.keys())):

        obs_output = af.shannon_entropy(X_l[i]['Fc_obs'], rang)/np.log2(bins_obs)
        model_output_l= af.shannon_entropy(X_l[i]['Fc_model'], rang)/np.log2(bins_obs)
        model_output_r = af.shannon_entropy(X_r[i]['Fc_model'], rang)/np.log2(bins_obs)

        Hz[i][0] = obs_output
        Hz[i][1] = model_output_l
        Hz[i][2] = model_output_r
        
    return Hz




def MI_LR(X_l,X_r,bins_model_l,bins_model_r):
    """
    Parameters
    ----------
    X_l : Local model.
    X_r : Reginoal model.
    bins_model_l : number of bins for local model
    bins_model_r : number of bins for regional model

    Returns
    -------
    MI : Mutual information of each driving source and carbon flux for observation, local and regional models
    """
    bins_obs =100
    
    bin_min = X_l[0]['Fc_obs'].min()
    bin_max = X_l[0]['Fc_obs'].max()
    bins_obs =100
    rang = [np.linspace(bin_min, bin_max, bins_obs)]
    
    Hz = np.zeros([len(X_l.keys()),3]) # 0:obs  1:local, 2: regional
    
    for i in range(len(X_l.keys())):

        obs_output = af.shannon_entropy(X_l[i]['Fc_obs'], rang)
        model_output_l= af.shannon_entropy(X_l[i]['Fc_model'], rang)
        model_output_r = af.shannon_entropy(X_r[i]['Fc_model'], rang)

        Hz[i][0] = obs_output
        Hz[i][1] = model_output_l
        Hz[i][2] = model_output_r
        
        
        
    
    MI=defaultdict(list)
    MI_N=defaultdict(list)

    for k in X_l.keys():
        
        mi=np.zeros([len(X_l[k].columns)-2,3]) # 0:obs  1:local, 2: regional

        for i in range(0,len(X_l[k].columns)-2):
            mi[i][0] = af.mutual_information(X_l[k], label[i],  'Fc_obs',  bins_obs)
            mi[i][1] = af.mutual_information(X_l[k], label[i],  'Fc_model',  int(bins_model_l[k]))
            mi[i][2] = af.mutual_information(X_r[k], label[i],  'Fc_model',  int(bins_model_r[k]))
            
        MI[k].append(mi)
        
        # Normalized MI by : MI( Xs, Fc)/H(Fc)
        MI_N[k] = MI[k]/Hz[k]
        
    for i in X_l.keys():

        MI[i]= np.reshape(MI[i],[len(X_l[k].columns)-2,3]) 
        MI_N[i]= np.reshape(MI_N[i],[len(X_l[k].columns)-2,3]) 

    return MI, MI_N


def AfMI(X_l, Hz, MI):
    
    """
    Af,MI = (observed MI- predicted MI)/ Observed MI
    obs> pred : positive values: random model
    obs< pred : negative values: deterministic model

    """
    MI_N = defaultdict(list)
    Af_MI = defaultdict(list)

    for k in X_l.keys():
        # Normalized MI by : MI( Xs, Fc)/H(Fc)
        MI_N[k] = pd.DataFrame(MI[k]/Hz[k])
        
        for i in range(1,3):
            eq = lambda x: (x[0]-x[i]) / x[0] if x[0] != 0 else 0 # To avoid zero division error

            # Apply the lambda function to all values in the DataFrame using the .apply() method
            Af_MI[k].append(MI_N[k].apply(eq, axis=1))
                        
    return Af_MI


def calc_information(X_l,X_r, label, bins_obs, bins_model_l,bins_model_r):
    
    """
    Calculate information partitioning metrics and return them as numpy arrays.
    """
    itot = np.zeros((len(label), len(label), 3))
    u1 = np.zeros((len(label), len(label), 3))
    u2 = np.zeros((len(label), len(label), 3))
    r = np.zeros((len(label), len(label), 3))
    s = np.zeros((len(label), len(label), 3))
    
    for i in range(0, len(label)):
        for j in range(0, len(label)):
            obs_output = af.information_partitioning(X_l, label[i], label[j], 'Fc_obs', bins_obs)
            model_output_l = af.information_partitioning(X_l, label[i], label[j], 'Fc_model', int(bins_model_l))
            model_output_r = af.information_partitioning(X_r, label[i], label[j], 'Fc_model', int(bins_model_r))
    
            itot[i][j][0], u1[i][j][0], u2[i][j][0], r[i][j][0], s[i][j][0] = obs_output
            itot[i][j][1], u1[i][j][1], u2[i][j][1], r[i][j][1], s[i][j][1] = model_output_l
            itot[i][j][2], u1[i][j][2], u2[i][j][2], r[i][j][2], s[i][j][2] = model_output_r

      
    return itot, u1+u2, r, s

def heatmap(ax, data, mask, vmin, vmax, title, label, fmt='.2f', cmap='RdYlGn', center=0, cbar=True, cbar_kws=None):
    """
    Helper function to plot a heatmap using seaborn.
    """
    sns.heatmap(data, xticklabels=label, yticklabels=label, fmt=fmt, cmap=cmap, vmin=vmin, vmax=vmax, center=center,
                cbar=cbar, annot=True, mask=mask, ax=ax, annot_kws={"fontsize": 9}, cbar_kws=cbar_kws)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    

def PIF(X_l,X_r,bins_model_l,bins_model_r):
    
    vmin = 0
    vmax = 1
    bins_obs = 100
    
    Itot = {}
    R = {}
    S = {}
    U = {}

    
    num_cores = joblib.cpu_count()
    results = joblib.Parallel(n_jobs=num_cores)(joblib.delayed(per.calc_information)(X_l[k],X_r[k], label, bins_obs, bins_model_l[k], bins_model_r[k]) for k in X_l.keys())
    
    for k, result in zip(X_l.keys(), results):
        Itot[k] = result[0]
        U[k] = result[1]
        R[k] = result[2]
        S[k] = result[3]
        
    for i in X_l.keys():

        fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(18, 8))
        
        mask_S = np.triu(np.ones_like(S[i][:,:,0]))
        heatmap(axs[0,0], S[i][:,:,0], mask_S, vmin, vmax, f'{dataset[i]}: S_Observation', label)
        heatmap(axs[1,0], S[i][:,:,1], mask_S, vmin, vmax, f'{dataset[i]}: S_Model_loc', label)
        heatmap(axs[2,0], S[i][:,:,2], mask_S, vmin, vmax, f'{dataset[i]}: S_Model_Reg', label)
                
        mask_R = np.triu(np.ones_like(R[i][:,:,0]))
        heatmap(axs[0,1], R[i][:,:,0], mask_R, vmin, vmax, f'{dataset[i]}: R_Observation', label)
        heatmap(axs[1,1], R[i][:,:,1], mask_R, vmin, vmax, f'{dataset[i]}: R_Model_loc', label)
        heatmap(axs[2,1], R[i][:,:,2], mask_R, vmin, vmax, f'{dataset[i]}: R_Model_Reg', label)
        
        mask_U = np.triu(np.ones_like(U[i][:,:,0]))
        heatmap(axs[0,2], U[i][:,:,0], mask_U, vmin, vmax, f'{dataset[i]}: U_Observation', label)
        heatmap(axs[1,2], U[i][:,:,1], mask_U, vmin, vmax, f'{dataset[i]}: U_Model_loc', label)
        heatmap(axs[2,2], U[i][:,:,2], mask_U, vmin, vmax, f'{dataset[i]}: U_Model_Reg', label)
   
        mask_I = np.triu(np.ones_like(Itot[i][:,:,0]))
        heatmap(axs[0,3], Itot[i][:,:,0], mask_I, vmin, vmax, f'{dataset[i]}: Itot_Observation', label, cbar_kws={"label": "PIF"})
        heatmap(axs[1,3], Itot[i][:,:,1], mask_I, vmin, vmax, f'{dataset[i]}: Itot_Model_loc', label, cbar_kws={"label": "PIF"})
        heatmap(axs[2,3], Itot[i][:,:,2], mask_I, vmin, vmax, f'{dataset[i]}: Itot_Model_Reg', label, cbar_kws={"label": "PIF"})
        
        fig.tight_layout()
        plt.show()
        
        
    return fig,Itot,S,R,U


from matplotlib.colors import Normalize

def Af_PID(X, S, R, U):
    #Af,S= S_model- S_obs: + overestimate , _ underestimate


    Af_l = {'Af,S': defaultdict(list), 'Af,R': defaultdict(list), 'Af,U': defaultdict(list), 'Af,Ipart': defaultdict(list)}
    Af_r = {'Af,S': defaultdict(list), 'Af,R': defaultdict(list), 'Af,U': defaultdict(list), 'Af,Ipart': defaultdict(list)}
    
    for i in X.keys():
        for var, dic in zip([S, R, U], ['Af,S', 'Af,R', 'Af,U']):
            Af_l[dic][i] = var[i][:,:,1] - var[i][:,:,0]
            Af_r[dic][i] = var[i][:,:,2] - var[i][:,:,0]

        Af_l['Af,Ipart'][i] = sum(abs(Af_l[dic][i]) for dic in ['Af,S', 'Af,R', 'Af,U'])
        Af_r['Af,Ipart'][i] = sum(abs(Af_r[dic][i]) for dic in ['Af,S', 'Af,R', 'Af,U'])

    # Calculate global vmin and vmax
    vmin_l = min(np.min(Af_l[dic][i]) for dic in ['Af,S', 'Af,R', 'Af,U', 'Af,Ipart'] for i in X.keys())
    vmax_l = max(np.max(Af_l[dic][i]) for dic in ['Af,S', 'Af,R', 'Af,U', 'Af,Ipart'] for i in X.keys())
    vmin_r = min(np.min(Af_r[dic][i]) for dic in ['Af,S', 'Af,R', 'Af,U', 'Af,Ipart'] for i in X.keys())
    vmax_r = max(np.max(Af_r[dic][i]) for dic in ['Af,S', 'Af,R', 'Af,U', 'Af,Ipart'] for i in X.keys())


    for i in X.keys():
        fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(24, 12), layout='constrained')
        
        for j, dic in enumerate(['Af,S', 'Af,R', 'Af,U', 'Af,Ipart']):
            for row, af_dic, vmin, vmax, exp in zip([0, 1], [Af_l, Af_r], [vmin_l, vmin_r], [vmax_l, vmax_r], ['_Loc', '_Reg']):
                mask = np.triu(np.ones_like(af_dic[dic][i]))
                sns.heatmap(af_dic[dic][i], xticklabels=label, yticklabels=label, cbar=True,\
                            fmt='.2f',cmap='RdYlGn', center=0, annot=True, mask=mask, ax=axs[row, j], annot_kws={"fontsize": 11}, vmin=vmin, vmax=vmax)
                axs[row, j].set_title(f"{dataset[i]}: {dic}{exp}", fontsize=14, fontweight='bold')
                axs[row, j].tick_params(axis='both', which='major', labelsize=12)
                
                
        
    return fig, Af_l, Af_r


def Af_PID_2(X, S, R, U):
    #Af,S= S_model- S_obs: + overestimate , _ underestimate


    Af_l = {'Af,S': defaultdict(list), 'Af,R': defaultdict(list), 'Af,U': defaultdict(list)}
    Af_r = {'Af,S': defaultdict(list), 'Af,R': defaultdict(list), 'Af,U': defaultdict(list)}
    
    for i in X.keys():
        for var, dic in zip([S, R, U], ['Af,S', 'Af,R', 'Af,U']):
            Af_l[dic][i] = var[i][:,:,1] - var[i][:,:,0]
            Af_r[dic][i] = var[i][:,:,2] - var[i][:,:,0]


    # Calculate global vmin and vmax
    vmin_l = min(np.min(Af_l[dic][i]) for dic in ['Af,S', 'Af,R', 'Af,U'] for i in X.keys())
    vmax_l = max(np.max(Af_l[dic][i]) for dic in ['Af,S', 'Af,R', 'Af,U'] for i in X.keys())
    vmin_r = min(np.min(Af_r[dic][i]) for dic in ['Af,S', 'Af,R', 'Af,U'] for i in X.keys())
    vmax_r = max(np.max(Af_r[dic][i]) for dic in ['Af,S', 'Af,R', 'Af,U'] for i in X.keys())
    
    # Create a normalized colormap
    norm_l = Normalize(vmin=vmin_l, vmax=vmax_l)

    for i in X.keys():
        fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(18, 12), layout='constrained')
        
        for j, dic in enumerate(['Af,S', 'Af,R', 'Af,U']):
            for row, af_dic, vmin, vmax, exp in zip([0, 1], [Af_l, Af_r], [vmin_l, vmin_r], [vmax_l, vmax_r], ['_Loc', '_Reg']):
                mask = np.triu(np.ones_like(af_dic[dic][i]))
                sns.heatmap(af_dic[dic][i], xticklabels=label, yticklabels=label, cbar=False,\
                            fmt='.2f',cmap='RdYlGn', center=0, annot=True, mask=mask, ax=axs[row, j], annot_kws={"fontsize": 11}, vmin=vmin, vmax=vmax)
                axs[row, j].set_title(f"{dataset[i]}: {dic}{exp}", fontsize=14, fontweight='bold')
                axs[row, j].tick_params(axis='both', which='major', labelsize=12)
                
         # Add a single colorbar
        fig.colorbar(plt.cm.ScalarMappable(norm=norm_l, cmap='RdYlGn'), ax=axs[0,2], location="right")
                
        
    return fig, Af_l, Af_r



def Af_Model(X,Af1,model_name,exp):  
    
    Af_Model =defaultdict(list)

    AfS=Af1['Af,S']
    AfR=Af1['Af,R']
    AfU=Af1['Af,U']
    AfIpart=Af1['Af,Ipart']
    
    for i in X.keys():
        AfS[i]=(np.triu(AfS[i], k=1)).reshape(1,81)  #  replace with . flatten()
        AfR[i]=(np.triu(AfR[i], k=1)).reshape(1,81)
        AfU[i]=(np.triu(AfU[i], k=1)).reshape(1,81)
        AfIpart[i]=(np.triu(AfIpart[i], k=1)).reshape(1,81)
        
        Af_Model[i]= pd.DataFrame([AfS[i][np.nonzero(AfS[i])], AfR[i][np.nonzero(AfR[i])],\
                                   AfU[i][np.nonzero(AfU[i])],AfIpart[i][np.nonzero(AfIpart[i])]])
            
            
    Af_M_1 = pd.DataFrame({key: (1 - np.abs([Af_Model[key].loc[i] for i in [0, 1, 2]])).sum(axis=1)\
                           / Af_Model[0].shape[1] for key in X})
    Af_M_2 = pd.DataFrame({key: (2 - Af_Model[key].loc[3]).sum()/ Af_Model[0].shape[1] for key in X}, index=[3])


    if exp=='L':

        Af_M_1= Af_M_1.rename(columns={0: "Ne1", 1: "Ne2",2: "Ne3", 3: "Br1", 4: "Br3", 5: "GC"},\
                    index={0: "$A_{f,S,tot}$", 1: "$A_{f,R,tot}$", 2: "$A_{f,U,tot}$"})
        Af_M_2= Af_M_2.rename(columns={0: "Ne1", 1: "Ne2",2: "Ne3", 3: "Br1", 4: "Br3", 5: "GC"},\
                    index={3: "$A_{f,Ipart,tot}$"})
    else:
        
        Af_M_1= Af_M_1.rename(columns={0: "Ne1_r", 1: "Ne2_r",2: "Ne3_r", 3: "Br1_r", 4: "Br3_r", 5: "GC_r"},\
                    index={0: "$A_{f,S,tot}$", 1: "$A_{f,R,tot}$", 2: "$A_{f,U,tot}$"})
        Af_M_2= Af_M_2.rename(columns={0: "Ne1_r", 1: "Ne2_r",2: "Ne3_r", 3: "Br1_r", 4: "Br3_r", 5: "GC_r"},\
                    index={3:"$A_{f,Ipart,tot}$"})

        
    Af_M = Af_M_1.append(Af_M_2, ignore_index = False)
        

        
    Af_M.plot(kind='bar', align='center', width=0.8, color = ['#008000', '#8CEF74', '#7EF2C7','#0000FF','#8080FF','black'])
    X_axis = np.arange(4)
    plt.legend(dataset)
    plt.ylabel("$A_{f,tot}$ ",fontweight='bold')
    plt.xticks(X_axis, ["$A_{f,S,tot}$", "$A_{f,R,tot}$", "$A_{f,U,tot}$", "$A_{f,Ipart,tot}$"], rotation=0)
    plt.title('The model level: '+ model_name)
    Af_Model_fig = plt.figure()

    plt.show()
    
    return Af_M.T, Af_Model_fig




    
    