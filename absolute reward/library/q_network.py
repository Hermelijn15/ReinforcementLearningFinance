# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 11:00:38 2024

@author: Joost
"""

from tensorflow.keras.layers import Dense, Input, Activation
from tensorflow.keras.models import Sequential
import tensorflow as tf
import numpy as np
import keras.optimizers as optimizers



def create_q_network(input_dim, n_actions, hidden_layers, optimizer=optimizers.Adam()):
    '''
    Initialise a neural network

    Parameters
    ----------
    input_dim : the size of the input state
    n_actions : number of possible actions (output state size)
    hidden_layers : how many which size hidden layers to use, 2d np array
    learning_rate : learning rate to implement in the model

    Returns
    -------
    model : neural network based on inputs

    '''
    
    model = Sequential()
    
    #create input layer
    model.add(Input(shape=(input_dim,)))

    #create hidden layers
    for hidden_layer in hidden_layers:
        model.add(Dense(hidden_layer)) #add aditional hidden layers
        model.add(Activation('relu'))
    
    #create output layer
    model.add(Dense(n_actions, activation='linear'))
    
    model.compile(loss='mse', optimizer=optimizer)
    
    return model


def train_q_network(q_network, states, actions, rewards, next_states, gamma, batch_size, epochs=1):
    '''
    Train a neural network

    Parameters
    ----------
    q_network : the neural network to train
    states : initial states
    actions : taken actions
    rewards : reward base on state and taken action
    next_states : state after taking action
    gamma : discount rate
    batch_size :  how to split up the input during fitting
    epochs : number of training epochs used during fitting

    Returns
    -------
    None.

    '''
    states = states.to_numpy()
    #actions= actions.to_numpy()
    rewards = rewards.to_numpy()
    next_states = next_states.to_numpy()
    
    q_values = q_network.predict_on_batch(states)  #old q values
    
    next_q_values = q_network.predict_on_batch(next_states) # q values in the next state (after the action)
    
    max_next_q_values = np.max(next_q_values, axis=1)
    
    
    updated_values =  rewards + gamma * max_next_q_values
    
    targets = q_values.copy() #the original q_values 
    targets[np.arange(len(actions)), actions] = updated_values #update the q values corresponding to the action taken

    # Train the Q-network on the entire batch at once (no mini batching)
    
    q_network.fit(states, targets, batch_size=batch_size, epochs=epochs, verbose=1, shuffle=True)
    
    return np.mean(targets, axis=0), np.mean(q_values, axis=0)
    
def choose_action(q_network, states, epsilon, n_actions):
    '''
    Let the q_network choose an action

    Parameters
    ----------
    q_network : neural network to use
    states : input states to choose actions for
    epsilon : chance to return a random action instead
    n_actions : output diminesion/number of possible actions

    Returns
    -------
    actions : an array of choosen actions for each state

    '''
    states = states.to_numpy()

    # Predict Q-values for all states in one batch operation
    q_values = q_network.predict_on_batch(states)

    # Generate random numbers for epsilon-greedy decisions
    random_numbers = np.random.rand(states.shape[0])

    
    greedy_actions = np.argmax(q_values, axis=1)
    random_actions = np.random.randint(n_actions, size=states.shape[0])
   
    actions = np.where(random_numbers < epsilon, random_actions, greedy_actions) 
   
    return actions