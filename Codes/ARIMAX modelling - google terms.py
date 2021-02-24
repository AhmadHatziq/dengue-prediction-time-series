# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 21:33:43 2020

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
filepath = r"<INSERT FILE PATH>"
dengue_df = pd.read_csv(filepath)
dengue_df["date"] = pd.to_datetime(dengue_df["date"], format = '%d/%m/%Y') # Converts the column to a datetime
dengue_df.set_index("date", inplace = True)
dengue_df = dengue_df.asfreq('W-MON')
# Observe that only 2 values are missing. Fill via interpolation
dengue_df = dengue_df.interpolate()

# Split data into test and train set.
# Run this everytime want to split after adding new exog var
train_dengue = dengue[:'2019-10-01']
test_dengue = dengue['2019-10-01':]

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

# Read in dengue search term
folder = "<INSERT FILE PATH>"
dengue_google = folder + r"\google-search-term-dengue.csv"
google_dengue_df = pd.read_csv(dengue_google)
google_dengue_df.set_index("date", inplace = True)

# Combine with dengue df
dengue_df = pd.merge(dengue_df, google_dengue_df, how='inner', left_index=True, right_index=True)
# Scale up the data
dengue_df['google_search_for_dengue'] = 8 * dengue_df['google_search_for_dengue']
# Plot to see correlation

plt.title('Before scaling')
dengue_df.plot()

best_seasonal_model_6 = pm.auto_arima(train_dengue['cases'],  seasonal = True, 
                                      exogeneous = train_dengue['google_search_for_dengue']
                                          ,max_d = 5, max_P = 5, max_Q = 5,
                                          max_D = 5,
                                          error_action = 'ignore', suppress_warnings = True)

# Train and manually insert the data
train_dengue.plot()

# Train SARIMAX 2 2 1 with google dnegue search
model = SARIMAX(train_dengue['cases'], order = (2,2,1),
                exog = train_dengue['google_search_for_dengue'])
results = model.fit()
print(results.summary())
results.plot_diagnostics()
plt.show()

# Get test MSE
from sklearn.metrics import mean_squared_error
test_forecast = results.get_forecast(steps = len(test_dengue), 
                                     exog = test_dengue['google_search_for_dengue'] )
mean_forecast = test_forecast.predicted_mean

confidence_intervals = test_forecast.conf_int()
lower_limits = confidence_intervals.loc[:,'lower cases']
upper_limits = confidence_intervals.loc[:,'upper cases']
test_score = mean_squared_error(test_y, mean_forecast)
print('Test score for model: ', test_score) # 3179.371030549788

# Train SARIMAX 2 2 1 with google eyepain search ==================================
# Read in dengue search term
folder = "<INSERT FILE PATH>"
eye_google = folder + r"\google-search-term-eye-pain.csv"
eye_df = pd.read_csv(eye_google)
eye_df.set_index("date", inplace = True)
# Combine with dengue df
dengue_df = pd.merge(dengue_df, eye_df, how='inner', left_index=True, right_index=True)
dengue_df['google_search_for_eye_pain'] = dengue_df['google_search_for_eye_pain'] * 8

# Train SARIMAX 2 2 1 with google eye pain search
model = SARIMAX(train_dengue['cases'], order = (2,2,1),
                exog = train_dengue['google_search_for_eye_pain'])
results = model.fit()
print(results.summary())
results.plot_diagnostics()
plt.show()

# Get TEST MSE
test_forecast = results.get_forecast(steps = len(test_dengue), 
                                     exog = test_dengue['google_search_for_eye_pain'] )
mean_forecast = test_forecast.predicted_mean

confidence_intervals = test_forecast.conf_int()
lower_limits = confidence_intervals.loc[:,'lower cases']
upper_limits = confidence_intervals.loc[:,'upper cases']
test_score = mean_squared_error(test_y, mean_forecast)
print('Test score for model: ', test_score) # 3032.1829095135868


# Train SARIMAX 2 2 1 with google FEVER search ==================================
folder = "<INSERT FILE PATH>"
fever_google = folder + r"\google-search-term-fever.csv"
fever_df = pd.read_csv(fever_google)
fever_df.set_index("date", inplace = True)

dengue = dengue_df.copy()

# Merge and plot to see how to scale
dengue = pd.merge(dengue, fever_df, how='inner', left_index=True, right_index=True)

dengue['google_search_for_fever'] = dengue['google_search_for_fever'] * 8

train_dengue = dengue[:'2019-10-01']
test_dengue = dengue['2019-10-01':]

# Train model
model = SARIMAX(train_dengue['cases'], order = (2,2,1),
                exog = train_dengue['google_search_for_fever'])
results = model.fit()
print(results.summary())
results.plot_diagnostics()
plt.show()

# Get test mse
test_forecast = results.get_forecast(steps = len(test_dengue), 
                                     exog = test_dengue['google_search_for_fever'] )
mean_forecast = test_forecast.predicted_mean
confidence_intervals = test_forecast.conf_int()
lower_limits = confidence_intervals.loc[:,'lower cases']
upper_limits = confidence_intervals.loc[:,'upper cases']
test_score = mean_squared_error(test_y, mean_forecast)
print('Test score for model: ', test_score) # 910.7622920660885 *** Better than persistence OMG

# Plot on test data
plt.plot(mean_forecast.index, mean_forecast, color = 'r', label = 'forecast')
plt.fill_between(lower_limits.index, lower_limits, upper_limits, color = 'pink')
plt.plot(test_dengue.index, test_dengue['cases'], label = 'test data')
plt.legend()
plt.show()

# See if the SARIMAX model performs best with this exog var
model = SARIMAX(train_dengue['cases'],
                exog = train_dengue['google_search_for_fever'], 
                order = (2,1,2), 
                seasonal_order = (1,0,1,52),
                )
results = model.fit()

# Get test mse for sarimax
test_forecast = results.get_forecast(steps = len(test_dengue), 
                                     exog = test_dengue['google_search_for_fever'] )
mean_forecast = test_forecast.predicted_mean
confidence_intervals = test_forecast.conf_int()
lower_limits = confidence_intervals.loc[:,'lower cases']
upper_limits = confidence_intervals.loc[:,'upper cases']
test_score = mean_squared_error(test_y, mean_forecast)
print('Test score for model: ', test_score) # 866.135623699692


# Train SARIMAX 2 2 1 with google headache search ==================================
folder = "<INSERT FILE PATH>"
headache_google = folder + r"\google-search-term-headache.csv"
heacache_df = pd.read_csv(headache_google)
heacache_df.set_index("date", inplace = True)

# Import a fresh copy of dengue cases
dengue = dengue_df.copy()

# Merge to see how to scale up
dengue = pd.merge(dengue, heacache_df, how='inner', left_index=True, right_index=True)
dengue['google_search_for_headache'] = dengue['google_search_for_headache'] * 8

# Reobtain test and train, with exog
train_dengue = dengue[:'2019-10-01']
test_dengue = dengue['2019-10-01':]

# Train model
model = SARIMAX(train_dengue['cases'], order = (2,2,1),
                exog = train_dengue['google_search_for_headache'])
results = model.fit()
print(results.summary())
results.plot_diagnostics()
plt.show()

# Get test mse
test_forecast = results.get_forecast(steps = len(test_dengue), 
                                     exog = test_dengue['google_search_for_headache'] )
mean_forecast = test_forecast.predicted_mean
confidence_intervals = test_forecast.conf_int()
lower_limits = confidence_intervals.loc[:,'lower cases']
upper_limits = confidence_intervals.loc[:,'upper cases']
test_score = mean_squared_error(test_y, mean_forecast)
print('Test score for model: ', test_score) # 3752.9666107295016


# Train SARIMAX 2 2 1 with google joint pain search ==================================
folder = "<INSERT FILE PATH>"
joint_google = folder + r"\google-search-term-joint-pain.csv"
joint_df = pd.read_csv(joint_google)
joint_df.set_index("date", inplace = True)

# Import a fresh copy of dengue cases
dengue = dengue_df.copy()

# Merge to see how to scale up
dengue = pd.merge(dengue, joint_df, how='inner', left_index=True, right_index=True)
dengue['google_search_for_joint_pain'] = dengue['google_search_for_joint_pain'] * 8

# Reobtain test and train, with exog
train_dengue = dengue[:'2019-10-01']
test_dengue = dengue['2019-10-01':]

# Train model
model = SARIMAX(train_dengue['cases'], order = (2,2,1),
                exog = train_dengue['google_search_for_joint_pain'])
results = model.fit()
print(results.summary())
results.plot_diagnostics()
plt.show()

# Get test mse
test_forecast = results.get_forecast(steps = len(test_dengue), 
                                     exog = test_dengue['google_search_for_joint_pain'] )
mean_forecast = test_forecast.predicted_mean
confidence_intervals = test_forecast.conf_int()
lower_limits = confidence_intervals.loc[:,'lower cases']
upper_limits = confidence_intervals.loc[:,'upper cases']
test_score = mean_squared_error(test_y, mean_forecast)
print('Test score for model: ', test_score) # 3760.5012657320462

# Train SARIMAX 2 2 1 with google nausea search ==================================
folder = "<INSERT FILE PATH>"
nausea_google = folder + r"\google-search-term-nausea.csv"
nausea_df = pd.read_csv(nausea_google)
nausea_df.set_index("date", inplace = True)

# Import a fresh copy of dengue cases
dengue = dengue_df.copy()

# Merge to see how to scale up
dengue = pd.merge(dengue, nausea_df, how='inner', left_index=True, right_index=True)
dengue['google_search_for_nausea'] = dengue['google_search_for_nausea'] * 8

# Reobtain test and train, with exog
train_dengue = dengue[:'2019-10-01']
test_dengue = dengue['2019-10-01':]

# Train model
model = SARIMAX(train_dengue['cases'], order = (2,2,1),
                exog = train_dengue['google_search_for_nausea'])
results = model.fit()
print(results.summary())
results.plot_diagnostics()
plt.show()

# Get test mse
test_forecast = results.get_forecast(steps = len(test_dengue), 
                                     exog = test_dengue['google_search_for_nausea'] )
mean_forecast = test_forecast.predicted_mean
confidence_intervals = test_forecast.conf_int()
lower_limits = confidence_intervals.loc[:,'lower cases']
upper_limits = confidence_intervals.loc[:,'upper cases']
test_score = mean_squared_error(test_y, mean_forecast)
print('Test score for model: ', test_score) # 4491.370692494964

# Train SARIMAX 2 2 1 with google rash search ==================================
folder = "<INSERT FILE PATH>"
rash_google = folder + r"\google-search-term-rash.csv"
rash_df = pd.read_csv(rash_google)
rash_df.set_index("date", inplace = True)

# Import a fresh copy of dengue cases
dengue = dengue_df.copy()

# Merge to see how to scale up
dengue = pd.merge(dengue, rash_df, how='inner', left_index=True, right_index=True)
dengue['google_search_for_rash'] = dengue['google_search_for_rash'] * 8

# Reobtain test and train, with exog
train_dengue = dengue[:'2019-10-01']
test_dengue = dengue['2019-10-01':]

# Train model
model = SARIMAX(train_dengue['cases'], order = (2,2,1),
                exog = train_dengue['google_search_for_rash'])
results = model.fit()
print(results.summary())
results.plot_diagnostics()
plt.show()

# Get test mse
test_forecast = results.get_forecast(steps = len(test_dengue), 
                                     exog = test_dengue['google_search_for_rash'] )
mean_forecast = test_forecast.predicted_mean
confidence_intervals = test_forecast.conf_int()
lower_limits = confidence_intervals.loc[:,'lower cases']
upper_limits = confidence_intervals.loc[:,'upper cases']
test_score = mean_squared_error(test_y, mean_forecast)
print('Test score for model: ', test_score) # 3248.338127549261





