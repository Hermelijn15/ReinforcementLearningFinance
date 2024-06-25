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


def run_episode(nn, data_frame, gamma, epsilon, n_actions, features, data_multiplications = 1, noise_strength=0.0, price_label="Close", risk_free_rate=0, shuffle=True, batch_size = 1000, max_allocation=1):
    '''

    Parameters
    ----------
    data_frame : Dataframe containing timeseries data
    gamma : discount rate
    epsilon : epsilon for epsilon greedy strategy
    features : dictionary of which features to use, all relative to the current value (percentage change)
    price_label : The collumn holding the price datay, for which we can buy, by default this is close.
    if you use open you have to be carefull with using other columns such as volumen. These 
    would then contain information of the future

    Returns
    -------
    average_reward : average reward per action
    q_targets : 
    q_values : 

    '''
    all_states = pd.DataFrame()
    all_next_states = pd.DataFrame()
    all_actions = np.array([], dtype=int)
    all_rewards = pd.Series()
    
    
    
    for _ in range(data_multiplications):
        modified_data_frame = utils.add_noise(data_frame, noise_strength=noise_strength)
        
        states, price_change, indices = utils.state_conversion(modified_data_frame, features)
        
        actions = q_network.choose_action(nn, states, epsilon, n_actions)
        allocations = action_allocation(actions, n_actions, max_allocation=max_allocation)
        r = rewards.percentage_reward(price_change, allocations, risk_free_rate=risk_free_rate)
        

        next_states = utils.next_state_conversion(modified_data_frame, indices, features=features)
        
        all_states = pd.concat([all_states, states])
        all_next_states = pd.concat([all_next_states, next_states])
        all_actions = np.append(all_actions, actions)
        if all_rewards.empty:
            all_rewards = r
        else:
            all_rewards = pd.concat([all_rewards, r])
    
    q_targets, q_values = q_network.train_q_network(nn, all_states, all_actions, all_rewards, all_next_states, gamma, batch_size)
    
    print("action distribution : ")
    print(np.bincount(actions))
    
    return np.mean(r), q_targets, q_values

def evaluate(nn, data, features, risk_free_rate=0):
    
    output_dim = nn.layers[-1].output.shape[1]
    
    states, price_change, indices = utils.state_conversion(data, features)
    actions = q_network.choose_action(nn, states, 0, output_dim)
    allocations = action_allocation(actions, output_dim, max_allocation=1)
    r = rewards.percentage_reward(price_change, allocations, risk_free_rate=risk_free_rate)
    print("average reward : " , np.mean(r))
    print("average price change : ",  np.mean(price_change))



#%% Training    
def train(#make sure the the columns used in features are present, aswell as a column called "Close" which is the price at which we can execute trades
        stock_data,
        folder_path = 'outputs',
        optimizer = optimizers.Adam(),
        features = {"Close": [-1,-2,-3,-4,-5],
            "Volume": [-1,-2,-3,-4,-5]},
          output_dim = 5,
          hidden_layers = np.array([16,16,16]),
          gamma = 0.9,
          nr_episodes = 5000,
          data_multiplications = 1,
          noise_strength=0.0,
          test_fraction = 0.15,
          risk_free_rate = 0,
          max_allocation = 1,
          batch_size=1000
          ):
    


    # Check if the folder exists
    if not os.path.exists(folder_path):
        # Create the folder
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
        print("should already have been there!!!!")

    print("starting")
    

    #neural network
    input_dim = 0
    for value in features.values():
        input_dim += len(value)
        
 
    #setup q_network (neural network)
    nn = q_network.create_q_network(input_dim, output_dim, hidden_layers, optimizer=optimizer)


#%% traning loops


    #reminder some of the lenght of the episode is lost due to having to look back/forwad (last state doesnt have a next otherwise)
    #stock_data = pdg.fetch_and_save_data("^GSPC")  #generate outside of loop to only have 1 timeseries for training
    
    length_data = len(stock_data)
    
    
    cutoff = int((1-test_fraction) * length_data)
    
    history = stock_data.iloc[:cutoff]
    test_data = stock_data.iloc[cutoff:]    
    history.to_csv(folder_path + "/training_data.csv")
    test_data.to_csv(folder_path + "/test_data.csv")
    
    
    avg_episode_rewards = np.zeros(nr_episodes)
    avg_qvalue = np.zeros((nr_episodes,output_dim))
    avg_qtargets = np.zeros((nr_episodes,output_dim))
    
    
    for episode in range(nr_episodes):
        print("running episode " , episode)
        epsilon = decaying_epsilon(episode, nr_episodes-1)
        
        avg_reward, q_targets, q_values = run_episode(nn, history, gamma, epsilon, output_dim, features, data_multiplications=data_multiplications, risk_free_rate=risk_free_rate, noise_strength=noise_strength, batch_size=batch_size)
        
        print("average reward " , avg_reward)
        print("q values ", q_values)
        avg_episode_rewards[episode] = avg_reward
        avg_qvalue[episode,:] = q_values
        avg_qtargets[episode,:] = q_targets
    
    
    plt.figure()
    plt.plot(avg_episode_rewards)
    plt.xlabel("training episode")
    plt.ylabel("average reward")
    plt.title("average reward during training")
    plt.savefig(folder_path + "/" +  "training_rewards.png", bbox_inches='tight')
    
    
    plt.figure()
    plt.plot(avg_qvalue,ls='-' , label=["value 1", "value 2", "value 3", "value 4", "value 5"])
    plt.plot(avg_qtargets,ls='--' , label=["target 1", "target 2", "target 3", "target 4", "target 5"])
    plt.title("evolution of Q predictions and targets for each action")
    plt.xlabel("training episode")
    plt.ylabel("Q")
    plt.legend()
    plt.savefig(folder_path + "/" +  "Q_evolution.png", bbox_inches='tight')
    
    
    #save the model
    nn.save(folder_path + "/agent.keras")


#%% load model
def evaluate_model(folder,
                   features = {"Close": [-1,-2,-3,-4,-5],"Volume": [-1,-2,-3,-4,-5]},
                   risk_free_rate = 0
                   ):
    
    test_data_file = folder + "/test_data.csv"
    file_name = folder + "/agent.keras"
    
    data = pd.read_csv(test_data_file)
    nn = tf.keras.models.load_model(file_name)
    
    output_dim = nn.layers[-1].output.shape[1]
    
    states, price_change, indices = utils.state_conversion(data, features)
    actions = q_network.choose_action(nn, states, 0, output_dim)
    allocations = action_allocation(actions, output_dim, max_allocation=1)
    r = rewards.percentage_reward(price_change, allocations, risk_free_rate=risk_free_rate)
    print("average reward : " , np.mean(r))
    print("average price change : ",  np.mean(price_change))
        




def random_prizes(size, batch_size=50, learning_rate=0.01, risk_free_rate=0.0001, mean=0, std=1, folder = "random", features = {"Close": [-1]}):
    stock_data = pdg.random(size, mean, std)
    output_dim = 5
    hidden_layers = np.array([3,3])
    gamma = 0.9
    nr_episodes = 500
    data_multiplications = 1
    noise_strength = 0

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
          nr_episodes=nr_episodes,
          data_multiplications=data_multiplications,
          noise_strength=noise_strength,
          test_fraction=test_fraction,
          risk_free_rate=risk_free_rate,
          max_allocation=max_allocation,
          batch_size=batch_size
          )
    
    


if __name__ == "__main__":
    
    random_prizes(size = 20000, risk_free_rate=0.0001, mean=0.05,std=5, folder = "random_mean=0.05_std_5_rf=0.0001")
    
    random_prizes(size = 20000, risk_free_rate=0.0001, mean=0.1,std=1, folder = "random_mean=0.001_std_1_rf=0.0001")
    
    
    
    
    
    
    


    
    
    

    
    
        
    
    