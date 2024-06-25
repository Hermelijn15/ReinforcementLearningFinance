# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 22:54:23 2024

@author: Joost
"""

import pandas as pd
import numpy as np


def add_noise(time_series_dataframe, noise_strength=0.001):
    
    time_series_dataframe = time_series_dataframe.copy()
    
    nr_data_points = time_series_dataframe.shape[0]
    for column in time_series_dataframe.columns:
        noise_factor = 1 +  np.random.normal(0, noise_strength  ,nr_data_points)
        time_series_dataframe[column] *= noise_factor
    return time_series_dataframe


def next_state_conversion(pandas_time_series, indices,  features = {"Close": [-1]}, price_label="Close"):
    
    result = pd.DataFrame()
        
    
    for column_name in features:
        if(column_name=="portfolio value"):

            result["portfolio value"] = pandas_time_series["portfolio value"]
            continue

        
        time_steps = features[column_name]
        current_values = pandas_time_series[column_name]
        for i in time_steps:
            result[column_name + " offset " + str(i)] = (current_values/current_values.shift(-i) - 1)
     
    next_result = result.shift(-1)   
     
    return next_result.loc[indices]
    

def state_conversion(pandas_time_series, features = {"Close": [-1]}, price_label="Close"):
    '''

    Parameters
    ----------
    pandas_time_series : Pandas dataframe containing a timeseries of all the data
    features : which features to use, all normalised to the current value of that feature

    Returns
    -------
    states : return a dataframe of states, ready to be used for training or evaluating
    next_state : 
    price_change :  the price change towards the next state, to be used in calculating the reward

    '''
    result = pd.DataFrame()

    
    for column_name in features:
        if(column_name=="portfolio value"):

           result["portfolio value"] = pandas_time_series["portfolio value"]
           continue

        
        time_steps = features[column_name]
        current_values = pandas_time_series[column_name]
        for i in time_steps:
            result[column_name + " offset " + str(i)] = (current_values/current_values.shift(-i) - 1)
            
    prices = pandas_time_series[price_label]
    result["price change"] = (prices -prices.shift(1)).shift(-1)/prices
    #if you bought how much would you have made until the next step (a percentage)
    
    #drop nan rows
    result = result.dropna()
    
    states = result.drop(columns=["price change"])
    price_change = result["price change"]
    
    return states, price_change, states.index
    
    
    
    