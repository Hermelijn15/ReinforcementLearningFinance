# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 10:58:19 2024

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
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from keras.optimizers import RMSprop




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
    
def decaying_epsilon(x, x_max):
    # Initial value
    a = 1.0
    
    # Final value
    final_value = 0.05
    
    # Solve for b using the final condition
    b = -np.log(final_value / a) / x_max
    
    # Exponential decay function
    return a * np.exp(-b * x)


def step(nn, history, epsilon, risk_free_rate, n_actions, features, max_allocation=1):
    #history contains a column of portfolio values!

    states, price_change , indices = utils.state_conversion(history, features)
    print(np.mean(price_change))

    actions = q_network.choose_action(nn, states, epsilon, n_actions)

    allocations = action_allocation(actions, n_actions, max_allocation=max_allocation)

    #set the portfolio value of updated history equal to the previous value plus a change
    updated_history = history.copy()

    earnings_factor =  (1+(price_change * allocations + (1-allocations)*risk_free_rate ))

    starting_value = history["portfolio value"].loc[indices]
    new_value = (history["portfolio value"].loc[indices] * earnings_factor)
    reward = new_value - starting_value

    updated_history["portfolio value"] = new_value.shift(1)


    updated_history["portfolio value"] = updated_history["portfolio value"].fillna(100)
     
    new_states = utils.next_state_conversion(updated_history, indices, features = features)
    
    return states, new_states, actions, reward, updated_history

#%% Training    
def train(#make sure the the columns used in features are present, aswell as a column called "Close" which is the price at which we can execute trades
        stock_data,
        folder_path = 'outputs',
        optimizer = optimizers.Adam(),
        features = {"Close": [-1],
            "portfolio value": [0]},
          output_dim = 5,
          hidden_layers = np.array([3,3]),
          gamma = 0.9,
          nr_steps = 50,
          test_fraction = 0.15,
          risk_free_rate = 0,
          max_allocation = 1,
          batch_size = 50
          ):
    


    # Check if the folder exists
    if not os.path.exists(folder_path):
        # Create the folder
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")

    print("starting")
    

    #neural network
    input_dim = 0
    for value in features.values():
        input_dim += len(value)
        
 
    #setup q_network (neural network)
    nn = q_network.create_q_network(input_dim, output_dim, hidden_layers, optimizer=optimizer)
    
    
    #test train split
    length_data = len(stock_data)

    cutoff = int((1-test_fraction) * length_data)
    
    history = stock_data.iloc[:cutoff]
    test_data = stock_data.iloc[cutoff:]    
    history.to_csv(folder_path + "/training_data.csv")
    test_data.to_csv(folder_path + "/test_data.csv")
    
    
    #training
    avg_step_reward = np.zeros(nr_steps)
    avg_qvalue = np.zeros((nr_steps,output_dim))
    avg_qtargets = np.zeros((nr_steps,output_dim))

    for s in range(nr_steps):
        epsilon = decaying_epsilon(s, nr_steps-1)
        
        
        states, new_state, actions, r , history = step(nn, history, epsilon, risk_free_rate, output_dim, features=features, max_allocation=max_allocation)
        
        q_targets, q_values = q_network.train_q_network(nn, states, actions, r, new_state, gamma, batch_size=batch_size)
        
        #reset portfolio value's
        if s%5==0:
            history.loc[:,"portfolio value"] = np.clip(np.random.normal(100,1,len(history))  , 0, None)
        
        avg_reward = np.mean(r)
        print("----------")
        print(s)
        print("average reward " , avg_reward)
        print("q values ", q_values)
        
        avg_step_reward[s] = avg_reward
        avg_qvalue[s,:] = q_values
        avg_qtargets[s,:] = q_targets
        
    
    
    #save the model
    
    plt.figure()
    plt.plot(avg_step_reward)
    plt.xlabel("training episode")
    plt.ylabel("average reward")
    plt.title("average reward during training")
    plt.savefig(folder_path + "/" +  "training_rewards.png", bbox_inches='tight')
    
    
    plt.figure()
    plt.plot(avg_qvalue,ls='-' , label=["value 1", "value 2", "value 3", "value 4", "value 5"])
    #plt.plot(avg_qtargets,ls='--' , label=["target 1", "target 2", "target 3", "target 4", "target 5"])
    plt.title("evolution of Q predictions and targets for each action")
    plt.xlabel("training episode")
    plt.ylabel("Q")
    plt.legend()
    plt.savefig(folder_path + "/" +  "Q_evolution.png", bbox_inches='tight')
    
    nn.save(folder_path + "/model.keras")


def random_prizes(size, batch_size=50, learning_rate=0.01, risk_free_rate=0.0001, mean=0, std=1, gamma=0.9, folder = "random", features = {"Close": [-1],"portfolio value": [0]}):
    
    stock_data = pdg.random(size, mean, std)    
    #add random initial portfolio values
    stock_data.loc[:,"portfolio value"] = np.clip(np.random.normal(100,1,size)  , 0, None)
    
    
    output_dim = 5
    hidden_layers = np.array([3,3])
    gamma = gamma
    nr_steps = 1000
    

    test_fraction = 0.15
    max_allocation = 1
    
    
    optimizer = keras.optimizers.SGD(
                        learning_rate=learning_rate,
                        momentum=0.0
                        )
    
    
    train(stock_data,
          folder_path=folder,
          optimizer= optimizer,
          features=features,
          output_dim=output_dim,
          hidden_layers=hidden_layers,
          gamma=gamma,
          nr_steps=nr_steps,
          test_fraction=test_fraction,
          risk_free_rate=risk_free_rate,
          max_allocation=max_allocation,
          batch_size=batch_size
          )



if __name__ == "__main__":
    
    #random_prizes(size = 20000, risk_free_rate=0, mean=0.002,std=2, gamma =0.99, folder = "random_mean=0.001_std_2_rf=0")
    
   
   
    batch_size = 640
    learning_rate = 0.001

    features = {"Close" : [-1], "portfolio value": [0]}
    
    random_prizes(size=200000, risk_free_rate=0.0001, mean=0.015, std=2, gamma =0.95, batch_size=batch_size , learning_rate = learning_rate , folder = "result_random_2",  features=features )
    
  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


    
    
    

    
    
        
    
    