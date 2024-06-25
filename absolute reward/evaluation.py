# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 14:51:08 2024

@author: Joost
"""

import library.q_network as q_network
import library.rewards as rewards
import library.price_data_generators as pdg
import library.utils as utils

import tensorflow as tf
from tensorflow import keras
import keras.optimizers as optimizers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from matplotlib.ticker import MaxNLocator

def action_allocation(action, output_dim, max_allocation=1):
    '''
    takes the action and converts it (linearly) to an allocation

    Parameters
    ----------
    action : array of actions
    output_dim : possible number of actions

    Returns
    -------
    allocation : actions converted to allocation

    '''
    
    allocation = (action/(output_dim - 1) ) * max_allocation
    
    return allocation

def evaluate(nn, data, features, fig, risk_free_rate=0, plot_stock=True, label_extension = ""):
    
    #data["portfolio value"] = 100
    
    initial_value = 100
    
    output_dim = nn.layers[-1].output.shape[1]
    
    states, price_change, indices = utils.state_conversion(data, features)
    actions = q_network.choose_action(nn, states, 0, output_dim)
    allocations = action_allocation(actions, output_dim, max_allocation=1)
    #r = rewards.percentage_reward(price_change, allocations, risk_free_rate=risk_free_rate)
    #print("average reward : " , np.mean(r))

    initial_value = data.iloc[0]["Close"]

    value  = pd.Series(initial_value, index=data.index, dtype=float, name="value")
    value = pd.DataFrame(value)
    

    factor = 1 + price_change * allocations + (1-allocations)*risk_free_rate

    cumulative_product = initial_value * np.cumprod(factor)
    value.loc[indices, 'value'] = cumulative_product
    


    plt.figure(fig.number)
    
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    
    plt.plot(value[:-1]['value'], label ="portfolio value " + label_extension )
    
    if plot_stock:
        plt.plot(data["Close"], label = "stock price")
    plt.legend()
    plt.xlabel("timestep")
    plt.ylabel("value")
    
    
    #print("average reward : " , np.mean(r))
    print("average price change : ",  np.mean(price_change))
    print("stand deviation pric change : " , np.std(price_change))
    
    #kelly criterion only works if we do not pick up on patterns!
    print("kelly criterion allocation : " , (np.mean(price_change)-risk_free_rate)/(np.std(price_change)**2))
    print("average allocation : " , np.mean(allocations))
    
    
    plt.figure()
    plt.scatter(states["Close offset -1"] /60, action_allocation(actions, 5)) #divide by 60 to go back to non normalised
    plt.xlabel("previous change")
    plt.ylabel("allocation")
  



if __name__ == "__main__":
    


                    
    features = {"Close" :[-1],"portfolio value": [0]}
    risk_free_rate=0.0001
                    
    folder = "result_random"


    model_name = "model.keras"
        
    model = tf.keras.models.load_model(folder + "/" + model_name)
    output_dim = model.layers[-1].output.shape[1]
    

    data_fig = plt.figure(1)
    plt.figure(data_fig.number)
    train_data = pd.read_csv(folder + "/training_data.csv", index_col=0)
    test_data = pd.read_csv(folder + "/test_data.csv", index_col=0)
    
    data_fig = plt.figure(1)
    plt.figure(data_fig.number)
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    plt.plot(train_data["Close"], label="training data")
    plt.plot(test_data["Close"], label="testing data")
    plt.legend()
    plt.xlabel("timestep")
    plt.ylabel("value")
        
    
    
    train_figure = plt.figure(2)
    test_figure = plt.figure(3)
    
    evaluate(model, train_data, features, train_figure, risk_free_rate=risk_free_rate, plot_stock=True)
    evaluate(model, test_data, features, test_figure,  risk_free_rate=risk_free_rate, plot_stock=True)
    

    
    
    
    
    
    
    
    
    






