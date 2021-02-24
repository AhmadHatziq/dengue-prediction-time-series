# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 18:16:30 2020

@author: Ahmad
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.options.display.max_columns = None
pd.options.display.max_rows = None

# Import dengue cases data
filepath = "<INSERT FILE PATH>"
dengue_df = pd.read_csv(filepath)
dengue_df["date"] = pd.to_datetime(dengue_df["date"], format = '%d/%m/%Y') # Converts the column to a datetime
dengue_df.set_index("date", inplace = True)
dengue_df = dengue_df.asfreq('W-MON')
# Observe that only 2 values are missing. Fill via interpolation
dengue_df = dengue_df.interpolate()

# Do an initial plot
dengue_df.plot()

# Split the data into test and training sets. Plot the data
# Last 2 months = Test set
train_dengue = dengue_df[:'2019-10-01']
test_dengue = dengue_df['2019-10-01':]
train_dengue = train_dengue.rename(columns={"cases": "train_data"})
test_dengue = test_dengue.rename(columns={"cases": "test_data"})
combined = pd.merge(test_dengue, train_dengue, how='outer', left_index=True, right_index=True)
combined.plot()

# Create a persistence model ==================================================
from pandas import DataFrame
from pandas import concat
from matplotlib import pyplot
values = DataFrame(dengue_df.cases)
dataframe = concat([values.shift(1), values], axis=1)
dataframe.columns = ['t-1', 't+1']
print(dataframe.head(5))

# split into train and test sets. train is the first 405 percent of total data
X = dataframe.values
train_size = 405
train, test = X[1:train_size], X[train_size:]
train_X, train_y = train[:,0], train[:,1]
test_X, test_y = test[:,0], test[:,1]

# persistence model
def model_persistence(x):
	return x
 
# Get the test MSE using the persistence model
from sklearn.metrics import mean_squared_error
predictions = list()
for x in test_X:
	yhat = model_persistence(x)
	predictions.append(yhat)
test_score = mean_squared_error(test_y, predictions)
print('Test MSE: %.3f' % test_score)

# plot predictions and expected results for persistence model
# pyplot.plot(train_y)
pyplot.plot([None for i in train_y] + [x for x in test_y], label = 'Test data')
pyplot.plot([None for i in train_y] + [x for x in predictions], label = 'Predictions')
pyplot.title('Persistence model on test set')
pyplot.legend()
pyplot.show()

# Start the Box Jenkins Methodology ===========================================
# Make the training data stationary
from statsmodels.tsa.stattools import adfuller
results = adfuller(train_dengue['cases'])
print(results)
print('p-value is: ', results[1])
# As the p-value is <0.05, we conclude that the original train data is stationary since
# we reject the null hypothesis

# Note to self: Take differenced value as there must be seasonality. There should be a better p-value

# Plot the PACF and ACF of the train data
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(train_dengue, lags = 100, alpha = 0.05, zero = False)

from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(train_dengue, lags = 100, alpha = 0.05, zero = False)

# As the ACF plot tails off and PACF cuts off after lag 1, it is suggested that
# a AR(p) model be used, where p = 1

# Creating an AR(1) model =====================================================
from statsmodels.tsa.statespace.sarimax import SARIMAX
ar_1_model = SARIMAX(train_dengue, order = (1,0,0))
results = ar_1_model.fit()
print(results.summary())

'''
                               SARIMAX Results                                
==============================================================================
Dep. Variable:                  cases   No. Observations:                  387
Model:               SARIMAX(1, 0, 0)   Log Likelihood               -2014.619
Date:                Sat, 28 Mar 2020   AIC                           4033.238
Time:                        14:46:35   BIC                           4041.155
Sample:                    01-02-2012   HQIC                          4036.377
                         - 05-27-2019                                         
Covariance Type:                  opg                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
ar.L1          0.9866      0.005    197.701      0.000       0.977       0.996
sigma2      1927.6206     82.849     23.267      0.000    1765.240    2090.002
===================================================================================
Ljung-Box (Q):                       59.96   Jarque-Bera (JB):               221.45
Prob(Q):                              0.02   Prob(JB):                         0.00
Heteroskedasticity (H):               0.18   Skew:                             0.19
Prob(H) (two-sided):                  0.00   Kurtosis:                         6.69
===================================================================================
'''
# above prob(Q) and prob(JB) are less than 0.05. Error residuals are correlated and not normal

# Plot the error residuals
results.plot_diagnostics()
plt.show()
# Model seems inadequate

# View in-sample predictions
forecast = results.get_prediction(start = -50)
mean_forecast = forecast.predicted_mean
confidence_intervals = forecast.conf_int()
lower_limits = confidence_intervals.loc[:,'lower cases']
upper_limits = confidence_intervals.loc[:,'upper cases']

plt.figure()
plt.plot(train_dengue.index, train_dengue['cases'], label = 'observed')
plt.plot(mean_forecast.index, mean_forecast, color = 'r', label = 'forecast')
plt.fill_between(lower_limits.index, lower_limits, upper_limits, color = 'pink')
plt.show()

# Get the test error
test_forecast = results.get_forecast(steps = len(test_dengue))
mean_forecast = test_forecast.predicted_mean

confidence_intervals = test_forecast.conf_int()
lower_limits = confidence_intervals.loc[:,'lower cases']
upper_limits = confidence_intervals.loc[:,'upper cases']

plt.figure()
plt.plot(train_dengue.index, train_dengue['cases'], label = 'observed')
plt.plot(mean_forecast.index, mean_forecast, color = 'r', label = 'forecast')
plt.fill_between(lower_limits.index, lower_limits, upper_limits, color = 'pink')
plt.plot(test_dengue.index, test_dengue['cases'], label = 'test data')
plt.legend()
plt.show()

# To do: Get MSE from above model


# Need to create another model

# Difference by one ===========================================================
train_first_diff = train_dengue.diff().dropna()

# Perform DF test on difference data
results = adfuller(train_first_diff['cases'])
print(results)
print('p-value is: ', results[1]) # p-value is:  2.329423623277005e-15

# Data appears to be more stationary

# View ACF and PACF of first order differenced data
plot_acf(train_first_diff, lags = 100, alpha = 0.05, zero = False)
# acf has a high peak approx every 24 lags

plot_pacf(train_first_diff, lags = 100, alpha = 0.05, zero = False)

# Take the cycle as 24
decomp_results = seasonal_decompose(train_first_diff['cases'], freq = 24)
decomp_results.plot()

df_rolling_mean = train_first_diff - train_first_diff.rolling(24).mean()
df_rolling_mean = df_rolling_mean.dropna()
plot_acf(df_rolling_mean, lags = [24,48,72, 99,120,144,168,192,216], zero = False)
plot_acf(df_rolling_mean, lags = [24,48,72, 99,120,144,168,192,216], alpha = 0.05, zero = False)
# Hence, a model with sesonal period 24 seems to fit. 

# Fit a SARIMA model (0,1,0)(0,1,1,)24
model = SARIMAX(train_dengue, order = (1,0,1), seasonal_order = (0,1,1,24))
results = model.fit()
print(results.summary())
results.plot_diagnostics()
plt.show()

# Difference by 2
train_second_diff = train_dengue.diff().diff().dropna()

# Perform DF test on difference data
results = adfuller(train_second_diff['cases'])
print(results)
print('p-value is: ', results[1]) # p-value is:  5.282480442206798e-20

# View ACF and PACF of 2nd order differenced data
plot_acf(train_second_diff, lags = 150, alpha = 0.05, zero = False)

plot_pacf(train_second_diff, lags = 150, alpha = 0.05, zero = False)


# Seasonal decomposition using statsmodels
from statsmodels.tsa.seasonal import seasonal_decompose
decomp_results = seasonal_decompose(train_dengue['cases'], freq = 60)
decomp_results.plot()

# Finding the seasonality by using rolling mean
df_rolling_mean = train_dengue - train_dengue.rolling(52).mean()
df_rolling_mean = df_rolling_mean.dropna()
plot_acf(df_rolling_mean, lags = 150, zero = False)

# use pmarina
import pmdarima as pm
results = pm.auto_arima(train_dengue)
results.summary()
results.plot_dignostics()

model2 = pm.auto_arima(train_dengue,
                      d=1,
                      seasonal=True,
                      trend= 'c',
                 	  max_p=2, max_q=2, max_P = 3, max_Q = 3,
                      trace=True,
                      error_action='ignore',
                      suppress_warnings=True) 

model3 = pm.auto_arima(train_dengue,
                      d=1,
                      seasonal=False,
                 	  max_p=2, max_q=2, max_P = 3, max_Q = 3,
                      trace=True,
                      error_action='ignore',
                      suppress_warnings=True) 

# Best model is ARIMA(2,1,2)
model = SARIMAX(train_dengue, order = (2,1,2))
results = model.fit()
print(results.summary())
results.plot_diagnostics()
plt.show()


# See predictions on test set
test_forecast = results.get_forecast(steps = len(test_dengue))
mean_forecast = test_forecast.predicted_mean

confidence_intervals = test_forecast.conf_int()
lower_limits = confidence_intervals.loc[:,'lower cases']
upper_limits = confidence_intervals.loc[:,'upper cases']

plt.figure()
# plt.plot(train_dengue.index, train_dengue['cases'], label = 'observed')
plt.plot(mean_forecast.index, mean_forecast, color = 'r', label = 'forecast')
plt.fill_between(lower_limits.index, lower_limits, upper_limits, color = 'pink')
plt.plot(test_dengue.index, test_dengue['cases'], label = 'test data')
plt.legend()
plt.show()

# Get test MSE
test_score = mean_squared_error(test_y, mean_forecast)
print('Test score for ARIMA(2,1,2) model: ', test_score)

model4 = pm.auto_arima(train_dengue,
                      d=2,
                      seasonal=True,
                      trend= 'c',
                 	  max_p=2, max_q=2, max_P = 3, max_Q = 3, 
                      trace=True,
                      error_action='ignore',
                      suppress_warnings=True) 

# Best model from auto arima is (2,2,1)
model = SARIMAX(train_dengue, order = (2,2,1))
results = model.fit()
print(results.summary())
results.plot_diagnostics()
plt.show()

# Get best model for d = 3
ACFmodel4 = pm.auto_arima(train_dengue,
                      d=2,
                      seasonal=False,
                      trend= 'c',
                 	  max_p=2, max_q=2, max_P = 3, max_Q = 3, 
                      trace=True,
                      error_action='ignore',
                      suppress_warnings=True) 

# Best model when m is set at 24 is (2,2,1)(1,0,1,24)
model4 = pm.auto_arima(train_dengue,
                      d=2,
                      seasonal=True,
                      trend= 'c',
                 	  max_p=2, max_q=2, max_P = 3, max_Q = 3, m = 24,
                      trace=True,
                      error_action='ignore',
                      suppress_warnings=True) 

# Iterate through to find the best period. m
# From plot, suspect that m is between 20 and 35.
aic_scores = []
for m in range(20, 36):
    seasonal_model = pm.auto_arima(train_dengue, d = 2, seasonal = True, 
                                   max_p = 3, max_q = 3, max_P = 3, max_Q = 3, 
                                   m = m, error_action = 'ignore', suppress_warnings = True)
    aic_scores.append(seasonal_model.aic())
# Best model is with 24 m. aic scores is stored on notepad ++
    
# Construct the best seasonal model using pm auto arima
# Find out what params does the model use with d = 1
best_seasonal_model = pm.auto_arima(train_dengue, d = 2, seasonal = True, 
                                   max_p = 3, max_q = 3, max_P = 3, max_Q = 3, 
                                   m = 24, error_action = 'ignore', suppress_warnings = True)
# Best AIC is 4012.606241098982
# Model used is (3,2,1) (1,0,1,24)

# See if the same aic is obtained by setting d = 1
best_seasonal_model_2 = pm.auto_arima(train_dengue, d = 1, seasonal = True, 
                                   max_p = 3, max_q = 3, max_P = 3, max_Q = 3, 
                                   m = 24, error_action = 'ignore', suppress_warnings = True)
# AIC for (2,1,2)(1,0,1,24) is 4011

# Set m = 23 for d = 1
best_seasonal_model_3 = pm.auto_arima(train_dengue, d = 1, seasonal = True, 
                                   max_p = 3, max_q = 3, max_P = 3, max_Q = 3, 
                                   m = 23, error_action = 'ignore', suppress_warnings = True)
# Best model is (2,1,2) (0,0,0,23)

# Set m = 52 as time period is in weeks
best_seasonal_model_4 = pm.auto_arima(train_dengue,  seasonal = True, 
                                  
                                   m = 52, error_action = 'ignore', suppress_warnings = True)

# Best model for m = 52 is (2,1,2) (1,0,1,52)
 
# Get best model with seasonal + false
best_non_seasonal_model_1 = pm.auto_arima(train_dengue,  seasonal = False, 
                                         
                                          max_d = 5, max_P = 5, max_Q = 5,
                                          max_D = 5,
                                   error_action = 'ignore', suppress_warnings = True)

best_seasonal_model_52 = pm.auto_arima(train_dengue,  seasonal = True, 
                                          m = 52,
                                          max_d = 5, max_P = 5, max_Q = 5,
                                          max_D = 5,
                                          error_action = 'ignore', suppress_warnings = True,
                                          trace = True)

# Build a (2,1,2) (1,0,0,52) model. 
model = SARIMAX(train_dengue, order = (2,1,2), seasonal_order = (1,0,0,52))
results = model.fit()
print(results.summary())
results.plot_diagnostics()
plt.show()

# Plot lags at order 52
lags = [52, 104,156,208, 260, 312, 364 ]
plot_acf(train_dengue, lags = lags, alpha = 0.05, zero = False)

from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(train_dengue, lags = lags, alpha = 0.05, zero = False)

# Find a model when d = 1, m = 25
best_seasonal_model_5 = pm.auto_arima(train_dengue,  seasonal = True, 
                                          d = 1, m = 25,
                                          max_d = 5, max_P = 5, max_Q = 5,
                                          max_D = 5,
                                          error_action = 'ignore', suppress_warnings = True,
                                          trace = True)
# Train the model (2,1,2) (0,0,1,25)
model = SARIMAX(train_dengue, order = (2,1,2), seasonal_order = (0,0,1,25))
results = model.fit()
print(results.summary())
results.plot_diagnostics()
plt.show()

# See predictions on test set
test_forecast = results.get_forecast(steps = len(test_dengue))
mean_forecast = test_forecast.predicted_mean

confidence_intervals = test_forecast.conf_int()
lower_limits = confidence_intervals.loc[:,'lower cases']
upper_limits = confidence_intervals.loc[:,'upper cases']

plt.figure()
# plt.plot(train_dengue.index, train_dengue['cases'], label = 'observed')
plt.plot(mean_forecast.index, mean_forecast, color = 'r', label = 'forecast')
plt.fill_between(lower_limits.index, lower_limits, upper_limits, color = 'pink')
plt.plot(test_dengue.index, test_dengue['cases'], label = 'test data')
plt.legend()
plt.show()

test_score = mean_squared_error(test_y, mean_forecast)
print('Test score for ARIMA(2,1,2) model: ', test_score)
