import pandas as pd
import statsmodels.formula.api as smf
from linearmodels import OLS, IV2SLS 
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.optimize import minimize

# import files
asset_1 = pd.read_csv('FB.csv', index_col = 'Date')
asset_2 = pd.read_csv('VZ.csv', index_col = 'Date')
asset_3 = pd.read_csv('CRM.csv', index_col = 'Date')
risk_free = pd.read_csv('TNX.csv', index_col = 'Date')

# select only adjusted close
asset_1 = asset_1[['Adj Close']]
asset_2 = asset_2[['Adj Close']]
asset_3 = asset_3[['Adj Close']]
risk_free = risk_free[['Adj Close']]

# asset array
assets = [asset_1, asset_2, asset_3, risk_free]

# portfolio data frame
portfolio = pd.concat(assets, axis=1)
portfolio.columns = ['asset_1', 'asset_2', 'asset_3', 'risk_free']


# MONTE CARLO SIM:
#######################################################################
# log daily returns
log_returns = np.log(portfolio/portfolio.shift(1))

# weights
weights = np.array(np.random.random(4))
weights = weights/np.sum(weights)

# expected returns
exp_ret = np.sum(log_returns.mean()*weights)*252
# risk free rate
rf = float((np.log(risk_free/risk_free.shift(1))).mean()*252)

# expected volatility
exp_vol = np.sqrt(np.dot(weights.T,np.dot(log_returns.cov()*252, weights)))

# sharpe ratio
SR = exp_ret-rf/exp_vol

# set up portfolio optimization variables
# number of tests
number_of_portfolios = 5000
# array to save all weights
all_weights = np.zeros((number_of_portfolios, len(portfolio.columns)))
# array to hold all returns
ret_arr = np.zeros(number_of_portfolios)
# array to hold all volatility measurements
vol_arr = np.zeros(number_of_portfolios)
# array to hold all sharpe ratios
sharpe_arr = np.zeros(number_of_portfolios)

# for loop to iterate
for ind in range(number_of_portfolios):
    #weights
    weights = np.array(np.random.random(4))
    weights = weights/np.sum(weights)

    # save weight in array
    all_weights[ind,:] = weights

    # expected return
    ret_arr[ind] = np.sum((log_returns.mean()*weights)*252)

    # expected volatility
    vol_arr[ind] = np.sqrt(np.dot(weights.T,np.dot(log_returns.cov()*252, weights)))

    # Sharpe Ratio
    sharpe_arr[ind] = ret_arr[ind]-rf/vol_arr[ind]

# extract optimal sharpe and weights
max_sharpe_ratio = sharpe_arr.max()
max_sharpe_ratio_return = ret_arr[sharpe_arr.argmax()]
max_sharpe_ratio_volatility = vol_arr[sharpe_arr.argmax()]
optimal_weights = all_weights[sharpe_arr.argmax(),:]

# plot the data
plt.figure(figsize=(12,8))
plt.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='plasma')
plt.colorbar(label = 'Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')
# add max sharpe ratio point from monte carlo sim
plt.scatter(max_sharpe_ratio_volatility, max_sharpe_ratio_return, c='red', s=50, edgecolors='black')

#######################################################################


# Machine Learning Optimization

# this function gets weights and return array of returns, volatility, and Sharpe Ratio
# weights are np array
# calculate return, volatility, and Sharpe,
# return
def get_ret_vol_sr(weights):
    weights = np.array(weights)
    ret = np.sum(log_returns.mean() * weights)*252
    vol = np.sqrt(np.dot(weights.T,np.dot(log_returns.cov()*252,weights)))
    sr = ret-rf/vol
    return np.array([ret, vol, sr])

# this function takes weights and returns second index of above sharpe ratio
def neg_sharpe(weights):
    return (get_ret_vol_sr(weights)[2] * -1)

# this function will check to make sure the weights add up to 1, if not it will return the error
def check_sum(weights):
    return (np.sum(weights) - 1)

# create constraint variable
# create weight boundaries 0 and 1 are min max respectivley
# create initial weight guess as starting point
cons = ({'type':'eq','fun':check_sum})
bounds = ((0,1),(0,1),(0,1),(0,1))
init_guess = [0.25, 0.25, 0.25, 0.25]

# pass all out arguments to minimize function
opt_results = minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=cons)


print(opt_results)
print(get_ret_vol_sr(opt_results.x))
print(max_sharpe_ratio)

# add max sharpe ratio point from minimization sim
plt.scatter(get_ret_vol_sr(opt_results.x)[1], get_ret_vol_sr(opt_results.x)[0], c='green', s=50, edgecolors='black')
plt.show()






























