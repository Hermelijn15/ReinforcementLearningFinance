# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 11:00:37 2024

@author: Joost
"""

def absolute_reward(price_change, allocation, portfolio_value, risk_free_rate = 0):
    
    
    new_value = portfolio_value * (price_change * allocation  + (1-allocation) * risk_free_rate )
    
    reward = new_value - portfolio_value
    
    return reward, portfolio_value
    
    
def percentage_reward(price_change, allocation, risk_free_rate = 0, multiplier=100):
    reward = ( price_change * allocation  + (1-allocation) * risk_free_rate )
    
    return reward*multiplier

def adjusted_percentage_reward(price_change, allocation, risk_free_rate = 0):
    
    #adjusts for the fact that going down 10% requires a more then 10% gain to recover
    
    x =  price_change * allocation  + (1-allocation) * risk_free_rate 
    
    reward = 1- (1/(1-x))
    
    return reward