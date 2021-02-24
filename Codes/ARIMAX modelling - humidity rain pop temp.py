# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 00:03:15 2020

@author: Ahmad
"""

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
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

# Prepare test set for MSE calculation
from pandas import DataFrame
from pandas import concat
from matplotlib import pyplot
values = DataFrame(dengue_df.cases)
dataframe = concat([values.shift(1), values], axis=1)
dataframe.columns = ['t-1', 't+1']
X = dataframe.values
train_size = 405
train, test = X[1:train_size], X[train_size:]
train_X, train_y = train[:,0], train[:,1]
test_X, test_y = test[:,0], test[:,1]

# SARIMAX model for humidity ==================================================
folder = "<INSERT FILE PATH>"
humid = folder + r"\humidity.csv"
humid_df = pd.read_csv(humid)
humid_df.set_index("date", inplace = True)

dengue = dengue_df.copy()
dengue = pd.merge(dengue, humid_df, how='inner', left_index=True, right_index=True)
dengue['mean_rh'] = 6 * dengue['mean_rh'] 

train_dengue = dengue[:'2019-10-01']
test_dengue = dengue['2019-10-01':]

model = SARIMAX(train_dengue['cases'],
                exog = train_dengue['mean_rh'], 
                order = (2,1,2), 
                seasonal_order = (1,0,1,52),
                )
results = model.fit()
print(results.summary())
results.plot_diagnostics()
plt.show()

test_forecast = results.get_forecast(steps = len(test_dengue), 
                                     exog = test_dengue['mean_rh'] )
mean_forecast = test_forecast.predicted_mean
confidence_intervals = test_forecast.conf_int()
lower_limits = confidence_intervals.loc[:,'lower cases']
upper_limits = confidence_intervals.loc[:,'upper cases']
test_score = mean_squared_error(test_y, mean_forecast)
print('Test score for model: ', test_score) # 4537.599915560763

# SARIMAX model for population ==================================================
folder = "<INSERT FILE PATH>"
pop = folder + r"\population.csv"
pop_df = pd.read_csv(pop)
pop_df.set_index("date", inplace = True)

dengue = dengue_df.copy()
dengue = pd.merge(dengue, pop_df, how='inner', left_index=True, right_index=True)
dengue['population'] = dengue['population'] / 10000

train_dengue = dengue[:'2019-10-01']
test_dengue = dengue['2019-10-01':]

model = SARIMAX(train_dengue['cases'],
                exog = train_dengue['population'], 
                order = (2,1,2), 
                seasonal_order = (1,0,1,52),
                )
results = model.fit()
print(results.summary())
results.plot_diagnostics()
plt.show()

test_forecast = results.get_forecast(steps = len(test_dengue), 
                                     exog = test_dengue['population'] )
mean_forecast = test_forecast.predicted_mean
confidence_intervals = test_forecast.conf_int()
lower_limits = confidence_intervals.loc[:,'lower cases']
upper_limits = confidence_intervals.loc[:,'upper cases']
test_score = mean_squared_error(test_y, mean_forecast)
print('Test score for model: ', test_score) # Unable to converge

# SARIMAX model for temperature, ==================================================
folder = "<INSERT FILE PATH>"
temp = folder + r"\temperature.csv"
temp_df = pd.read_csv(temp)
temp_df.set_index("dateTime", inplace = True)

dengue = dengue_df.copy()
dengue = pd.merge(dengue, temp_df['max_temperature'], how='inner', left_index=True, right_index=True)

# NTS: get TEST MSE FOR max


# SARIMAX model for temperature, mean ==================================================
dengue = dengue_df.copy()
dengue = pd.merge(dengue, temp_df['mean_temp'], how='inner', left_index=True, right_index=True)
dengue['mean_temp'] = dengue['mean_temp'] * 8

train_dengue = dengue[:'2019-10-01']
test_dengue = dengue['2019-10-01':]

model = SARIMAX(train_dengue['cases'],
                exog = train_dengue['mean_temp'], 
                order = (2,1,2), 
                seasonal_order = (1,0,1,52),
                )
results = model.fit()
print(results.summary())
results.plot_diagnostics()
plt.show()

test_forecast = results.get_forecast(steps = len(test_dengue), 
                                     exog = test_dengue['mean_temp'] )
mean_forecast = test_forecast.predicted_mean
confidence_intervals = test_forecast.conf_int()
lower_limits = confidence_intervals.loc[:,'lower cases']
upper_limits = confidence_intervals.loc[:,'upper cases']
test_score = mean_squared_error(test_y, mean_forecast)
print('Test score for model: ', test_score) # 3592.65905978112

# SARIMAX model for temperature, max ==================================================
dengue = dengue_df.copy()
dengue = pd.merge(dengue, temp_df['max_temperature'], how='inner', left_index=True, right_index=True)
dengue['max_temperature'] = dengue['max_temperature'] * 8

train_dengue = dengue[:'2019-10-01']
test_dengue = dengue['2019-10-01':]

model = SARIMAX(train_dengue['cases'],
                exog = train_dengue['max_temperature'], 
                order = (2,1,2), 
                seasonal_order = (1,0,1,52),
                )
results = model.fit()
print(results.summary())
results.plot_diagnostics()
plt.show()

test_forecast = results.get_forecast(steps = len(test_dengue), 
                                     exog = test_dengue['max_temperature'] )
mean_forecast = test_forecast.predicted_mean
confidence_intervals = test_forecast.conf_int()
lower_limits = confidence_intervals.loc[:,'lower cases']
upper_limits = confidence_intervals.loc[:,'upper cases']
test_score = mean_squared_error(test_y, mean_forecast)
print('Test score for model: ', test_score) # 3028.137076628815

# SARIMAX model for temperature, temp_extremes_min ==================================================
dengue = dengue_df.copy()
dengue = pd.merge(dengue, temp_df['temp_extremes_min'], how='inner', left_index=True, right_index=True)
dengue['temp_extremes_min'] = dengue['temp_extremes_min'] * 8

train_dengue = dengue[:'2019-10-01']
test_dengue = dengue['2019-10-01':]

model = SARIMAX(train_dengue['cases'],
                exog = train_dengue['temp_extremes_min'], 
                order = (2,1,2), 
                seasonal_order = (1,0,1,52),
                )
results = model.fit()
print(results.summary())
results.plot_diagnostics()
plt.show()

test_forecast = results.get_forecast(steps = len(test_dengue), 
                                     exog = test_dengue['temp_extremes_min'] )
mean_forecast = test_forecast.predicted_mean
confidence_intervals = test_forecast.conf_int()
lower_limits = confidence_intervals.loc[:,'lower cases']
upper_limits = confidence_intervals.loc[:,'upper cases']
test_score = mean_squared_error(test_y, mean_forecast)
print('Test score for model: ', test_score) # 4308.704501498803

# SARIMAX model for temperature, temp_mean_daily_max ==================================================
dengue = dengue_df.copy()
dengue = pd.merge(dengue, temp_df['temp_mean_daily_max'], how='inner', left_index=True, right_index=True)
dengue['temp_mean_daily_max'] = dengue['temp_mean_daily_max'] * 8

train_dengue = dengue[:'2019-10-01']
test_dengue = dengue['2019-10-01':]

model = SARIMAX(train_dengue['cases'],
                exog = train_dengue['temp_mean_daily_max'], 
                order = (2,1,2), 
                seasonal_order = (1,0,1,52),
                )
results = model.fit()
print(results.summary())
results.plot_diagnostics()
plt.show()

test_forecast = results.get_forecast(steps = len(test_dengue), 
                                     exog = test_dengue['temp_mean_daily_max'] )
mean_forecast = test_forecast.predicted_mean
confidence_intervals = test_forecast.conf_int()
lower_limits = confidence_intervals.loc[:,'lower cases']
upper_limits = confidence_intervals.loc[:,'upper cases']
test_score = mean_squared_error(test_y, mean_forecast)
print('Test score for model: ', test_score) # 2510.6506676868466

# SARIMAX model for temperature, temp_mean_daily_min ==================================================
dengue = dengue_df.copy()
dengue = pd.merge(dengue, temp_df['temp_mean_daily_min'], how='inner', left_index=True, right_index=True)
dengue['temp_mean_daily_min'] = dengue['temp_mean_daily_min'] * 8

train_dengue = dengue[:'2019-10-01']
test_dengue = dengue['2019-10-01':]

model = SARIMAX(train_dengue['cases'],
                exog = train_dengue['temp_mean_daily_min'], 
                order = (2,1,2), 
                seasonal_order = (1,0,1,52),
                )
results = model.fit()
print(results.summary())
results.plot_diagnostics()
plt.show()

test_forecast = results.get_forecast(steps = len(test_dengue), 
                                     exog = test_dengue['temp_mean_daily_min'] )
mean_forecast = test_forecast.predicted_mean
confidence_intervals = test_forecast.conf_int()
lower_limits = confidence_intervals.loc[:,'lower cases']
upper_limits = confidence_intervals.loc[:,'upper cases']
test_score = mean_squared_error(test_y, mean_forecast)
print('Test score for model: ', test_score) # 4679.87674865056




