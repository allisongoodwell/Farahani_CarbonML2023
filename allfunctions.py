# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 14:52:16 2023

@author: askarzam

"""
import pandas as pd
import numpy as np
import random 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler


def shannon_entropy(x, bins):
    c = np.histogramdd(x, bins)[0]
    p = c / np.sum(c)
    p = p[p > 0]
    h =  - np.sum(p * np.log2(p))
    
    return h

def mutual_information(dfi, source, target, bins, reshuffle=0,ntests=0):
    x = dfi[source].values
    y = dfi[target].values
    
    if reshuffle == 1:
        mishuff=[]
        for i in range(0,ntests+1):
            random.shuffle(x)
            random.shuffle(y)
            
            H_x = shannon_entropy([x], [bins])
            H_y = shannon_entropy([y], [bins])
            H_xy = shannon_entropy([x, y], [bins, bins])
            
            mishuff.append(H_x + H_y - H_xy)
        
        MI_crit = np.mean(mishuff)+ 3*np.std(mishuff)
        return MI_crit
            
    else:        
            
        H_x = shannon_entropy([x], [bins])
        H_y = shannon_entropy([y], [bins])
        H_xy = shannon_entropy([x, y], [bins, bins])
    
        return H_x + H_y - H_xy

def conditional_mutual_information(dfi, source, target, condition, bins, reshuffle=0):
    x = dfi[source].values
    y = dfi[condition].values
    z = dfi[target].values
    if reshuffle == 1:
        random.shuffle(x)
        random.shuffle(y)
        random.shuffle(z)

    H_y = shannon_entropy([y],[bins])
    H_z = shannon_entropy([z],[bins])
    H_xy = shannon_entropy([x, y],[bins, bins])
    H_zy = shannon_entropy([z, y],[bins, bins])
    H_xyz = shannon_entropy([x, y, z],[bins, bins, bins])
    
    return H_xy + H_zy - H_y - H_xyz


# I(Xs1,Xtar|Xs2)- I (Sx2;Xtar)
def interaction_information(mi_c, mi):
    i = mi_c - mi
    return i

#new formulation of redundancy: normalize I between sources (I_x1x2) 
#multiply R by normalized I (to decrease I when sources independent)
#R = R .* normalized_source_dependency; if I(S1;S2)=0, R=0

def normalized_source_dependency(mi_s1_s2, H_s1, H_s2):
    i = mi_s1_s2 / np.min([H_s1, H_s2])
    
    return i

#Account for source correlation and keep redundancy within bounds
#I_x1y + I_x2y - I_tot < R < min[I_x1y,I_x2y]
def redundant_information_bounds(mi_s1_tar, mi_s2_tar, interaction_info):
    r_mmi = np.min([mi_s1_tar, mi_s2_tar])
    r_min = np.max([0, - interaction_info])
    
    return r_mmi, r_min

def rescaled_redundant_information(mi_s1_s2, H_s1, H_s2, mi_s1_tar, mi_s2_tar, interaction_info):
    norm_s_dependency = normalized_source_dependency(mi_s1_s2, H_s1, H_s2)
    r_mmi, r_min = redundant_information_bounds(mi_s1_tar, mi_s2_tar, interaction_info)
    
    return r_min + norm_s_dependency * (r_mmi - r_min)

def information_partitioning(df, source_1, source_2, target, bins, reshuffle=0):
    if reshuffle == 1:
        x = df[source_1].values
        y = df[source_2].values
        z = df[target].values
        random.shuffle(x)
        random.shuffle(y)
        random.shuffle(z)
        df[source_1] = x
        df[source_2] = y
        df[target] = z
    else:
        df[source_1] = df[source_1].values
        df[source_2] = df[source_2].values
        df[target] = df[target].values

    H_s1 = shannon_entropy(df[source_1].values, [bins]) #Hx1
    H_s2 = shannon_entropy(df[source_2].values, [bins]) #Hx2
    mi_s1_s2 = mutual_information(df, source_1, source_2, bins) #I_x1x2
    mi_s1_tar = mutual_information(df, source_1, target, bins) #I_x1y
    mi_s2_tar = mutual_information(df, source_2, target, bins) #I_x2y
    mi_s1_tar_cs2 = conditional_mutual_information(df, source_1, target, source_2, bins) #I(Xs1,Xtar|Xs2)
    interaction_info = interaction_information(mi_s1_tar_cs2, mi_s1_tar) #S-R

    redundant = rescaled_redundant_information(mi_s1_s2, H_s1, H_s2, mi_s1_tar, mi_s2_tar, interaction_info)
    unique_s1 = mi_s1_tar - redundant
    unique_s2 = mi_s2_tar - redundant
    synergistic = interaction_info + redundant
    total_information = unique_s1 + unique_s2 + redundant + synergistic
    
    return total_information, unique_s1/total_information, unique_s2/total_information, redundant/total_information, synergistic/total_information



# bins sizes for obseravtion and model data
def BinNumber_model(obs,model,bins_obs):

    range_obs= np.max(obs)-np.min(obs)
    range_model= np.max(model)-np.min(model)
    bins_model= np.ceil((range_model*bins_obs)/range_obs)
    return bins_model


def Normalazied(frames,method):
    data_Norm={}
    for i in frames.keys():
    
        if method == 'Standard':
            scaler = StandardScaler()
            data_Norm[i]  = scaler.fit_transform(frames[i])
            data_Norm[i]= pd.DataFrame(data_Norm[i],columns=['Ta', 'RH', 'P', 'TS', 'PPFD','NETRAD', 'WS', 'Pa','SWC', 'Fc'])
        
        if method == 'MinMax':
            scaler = MinMaxScaler() 
            data_Norm[i] = scaler.fit_transform(frames[i])
            data_Norm[i]= pd.DataFrame(data_Norm[i], columns=['Ta', 'RH', 'P', 'TS', 'PPFD','NETRAD', 'WS', 'Pa','SWC', 'Fc'])
        
        if method == 'Quan':
        #IQR = 75th quantile — 25th quantile    
        #X_scaled = (X — X.median) / IQR
            scaler = RobustScaler() 
            data_Norm[i] = scaler.fit_transform(frames[i])
            data_Norm[i]= pd.DataFrame(data_Norm[i], columns=['Ta', 'RH', 'P', 'TS', 'PPFD','NETRAD', 'WS', 'Pa','SWC', 'Fc'])
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
    return data_Norm

