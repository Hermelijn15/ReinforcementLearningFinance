# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 11:01:05 2024

@author: Joost
"""

import numpy as np
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os


'''
Rewrite to return pandas dataframes
'''

def random(size, mean, std, initial_value = 100):
    '''
    Generate an array of price data, randomly moving at each step

    Parameters
    ----------
    size : length of output
    mean : mean gain of each step
    std : std gain each step
    initial_value : initial value of the stock, default 100

    Returns
    -------
    values : price array of length 'size'

    '''
    
    changes = np.random.normal(mean,std,size-1)

    factors = 1 + changes/100
    
    # Compute the cumulative product to get the value changes
    values = np.cumprod(factors)

    # Insert the initial value at the beginning and scale the cumulative product
    values = np.insert(values, 0, 1) * initial_value
    
    #convert to dataframe
    values = pd.DataFrame(values, columns=["Close"])
    return values
    
    
    
def reverting_prices(size, mean, std, initial_value = 100, reversion_strength = 0.1):
    '''
    Generate an array of price data, having a bigger chance to go up if you just went down
    and visa versa 

    Parameters
    ----------
    size : length of output
    mean : mean gain of each step
    std : std gain each step
    initial_value : initial value of the stock, default 100
    reversion_strength : strength of the added change based on previous step

    Returns
    -------
    values : price array of length 'size'

    '''
    
    changes = np.random.normal(mean,std,size-1)

    for i in range(1,size-1):
        #add a correction based on the last number, skipping the first
        changes[i] -= changes[i-1] * reversion_strength
        
    factors = 1 + changes/100
    
    # Compute the cumulative product to get the value changes
    values = np.cumprod(factors)

    # Insert the initial value at the beginning and scale the cumulative product
    values = np.insert(values, 0, 1) * initial_value
    values = pd.DataFrame(values, columns=["Close"])
    return values

def load_historical_data(filepath):
    '''
    Load historical price data from a CSV file.

    Parameters
    ----------
    filepath : str
        Path to the CSV file containing historical price data.

    Returns
    -------
    prices : np.array
        Array of historical prices.
    additional_features : np.array
        Array of additional features (like volume, moving averages).
    '''
    data = pd.read_csv(filepath)

    return data

def fetch_and_save_data(ticker, start_date = "1950-1-1", end_date = "2024-6-18", interval="1d"):
    
    # Fetch stock data
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    
    # Remove timezone information to avoid Excel compatibility issues
    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)
    
    # Create the datasets directory if it doesn't exist
    directory = 'datasets'
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Save the data to a CSV file in the datasets directory
    csv_filename = os.path.join(directory, f"{ticker}_daily_data.csv")
    data.to_csv(csv_filename)
    
    print(f"Data saved to {csv_filename}")
    return data