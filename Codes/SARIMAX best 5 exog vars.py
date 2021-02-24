# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 20:37:50 2020

@author: Ahmad
"""

# Best model with 5 exog var + SARIMA(2,1,2) (0,0,1,25)
# Fever search, temp_mean_daily_max changi, Max temp changi, maximum_rainfall_in_a_day changi
# and Rash search

# Load dengue cases
filepath = "<INSERT FILE PATH>"
dengue_df = pd.read_csv(filepath)
dengue_df["date"] = pd.to_datetime(dengue_df["date"], format = '%d/%m/%Y') # Converts the column to a datetime
dengue_df.set_index("date", inplace = True)
dengue_df = dengue_df.asfreq('W-MON')
dengue_df = dengue_df.interpolate()

# Load fever
folder = "<INSERT FILE PATH>"
fever_google = folder + r"\google-search-term-fever.csv"
fever_df = pd.read_csv(fever_google)
fever_df.set_index("date", inplace = True)
dengue_df = pd.merge(dengue_df, fever_df, how='inner', left_index=True, right_index=True)
dengue_df['google_search_for_fever'] = dengue_df['google_search_for_fever'] * 8

# Load changi rainfall
folder = "<INSERT FILE PATH>"
rain = folder + r"\rain_changi.csv"
rain_df = pd.read_csv(rain)
rain_df.set_index("dateTime", inplace = True)
dengue_df = pd.merge(dengue_df, rain_df, how='inner', left_index=True, right_index=True)
dengue_df['maximum_rainfall_in_a_day'] = 3 * dengue_df['maximum_rainfall_in_a_day']
dengue_df['no_of_rainy_days'] = 3 * dengue_df['no_of_rainy_days']

# Load changi temp
folder = "<INSERT FILE PATH>"
temp = folder + r"\temperature.csv"
temp_df = pd.read_csv(temp)
temp_df.set_index("dateTime", inplace = True)
dengue_df = pd.merge(dengue_df, temp_df['max_temperature'], how='inner', left_index=True, right_index=True)
dengue_df['max_temperature'] = dengue_df['max_temperature'] * 8
dengue_df = pd.merge(dengue_df, temp_df['temp_mean_daily_max'], how='inner', left_index=True, right_index=True)
dengue_df['temp_mean_daily_max'] = dengue_df['temp_mean_daily_max'] * 8

# Load rash
folder = "<INSERT FILE PATH>"
rash_google = folder + r"\google-search-term-rash.csv"
rash_df = pd.read_csv(rash_google)
rash_df.set_index("date", inplace = True)
dengue_df = pd.merge(dengue_df, rash_df, how='inner', left_index=True, right_index=True)
dengue_df['google_search_for_rash'] = dengue_df['google_search_for_rash'] * 8


# Get Test MSE with all 5 exog var
train_dengue = dengue_df[:'2019-10-01']
test_dengue = dengue_df['2019-10-01':]
model = SARIMAX(train_dengue['cases'], order = (2,1,2), seasonal_order = (0,0,1,25),
                exog = train_dengue[['google_search_for_fever', 'google_search_for_rash', 
                                    'maximum_rainfall_in_a_day', 'max_temperature', 'temp_mean_daily_max']])
results = model.fit(maxiter=2000)
test_forecast = results.get_forecast(steps = len(test_dengue), 
                                     exog = test_dengue[['google_search_for_fever', 'google_search_for_rash', 
                                    'maximum_rainfall_in_a_day', 'max_temperature', 'temp_mean_daily_max']] )
mean_forecast = test_forecast.predicted_mean
test_score = mean_squared_error(test_y, mean_forecast)
print('Test score for model: ', test_score) # 1000.8107583822025
# aic = 4204

# Fit with just fever search
train_dengue = dengue_df[:'2019-10-01']
test_dengue = dengue_df['2019-10-01':]
model = SARIMAX(train_dengue['cases'], order = (2,1,2), seasonal_order = (0,0,1,25),
                exog = train_dengue['google_search_for_fever'])
results = model.fit(maxiter=2000)
test_forecast = results.get_forecast(steps = len(test_dengue), 
                                     exog = test_dengue['google_search_for_fever'] )
mean_forecast = test_forecast.predicted_mean
test_score = mean_squared_error(test_y, mean_forecast)
print('Test score for model: ', test_score) # aic = 4200.428

# Fit with fever and max rainfall - best model
train_dengue = dengue_df[:'2019-10-01']
test_dengue = dengue_df['2019-10-01':]
model = SARIMAX(train_dengue['cases'], order = (2,1,2), seasonal_order = (0,0,1,25),
                exog = train_dengue[['google_search_for_fever', 'maximum_rainfall_in_a_day']])
results = model.fit(maxiter=2000)
test_forecast = results.get_forecast(steps = len(test_dengue), 
                                     exog = test_dengue[['google_search_for_fever', 'maximum_rainfall_in_a_day']] )
mean_forecast = test_forecast.predicted_mean
test_score = mean_squared_error(test_y, mean_forecast)
print('Test score for model: ', test_score) # mse = 833.629463792743

# Get out of sample predictions
plt.figure()
# plt.plot(train_dengue.index, train_dengue['cases'], label = 'observed')
plt.plot(mean_forecast.index, mean_forecast, color = 'r', label = 'forecast')
plt.fill_between(lower_limits.index, lower_limits, upper_limits, color = 'pink')
plt.plot(test_dengue.index, test_dengue['cases'], label = 'test data')
plt.legend()
plt.show()

# Get in sample predictions
forecast = results.get_prediction(start = -50)
mean_forecast = forecast.predicted_mean
confidence_intervals = forecast.conf_int()
lower_limits = confidence_intervals['lower cases']
upper_limits = confidence_intervals['upper cases']

plt.figure()
plt.plot(mean_forecast.index, mean_forecast, color = 'r', label = 'forecast')
plt.fill_between(lower_limits.index, lower_limits, upper_limits, color = 'pink')
plt.plot(train_dengue.index, train_dengue['cases'], label = 'train data')
