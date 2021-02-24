# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 18:28:42 2020

@author: Ahmad
"""

# From previous modelling, the SARIMAX model is of (2,1,2) (0, 0, 1, 25)

# Load all the data into a single dataframe ===================================

# Load dengue cases
filepath = "<INSERT FILE PATH>"
dengue_df = pd.read_csv(filepath)
dengue_df["date"] = pd.to_datetime(dengue_df["date"], format = '%d/%m/%Y') # Converts the column to a datetime
dengue_df.set_index("date", inplace = True)
dengue_df = dengue_df.asfreq('W-MON')
dengue_df = dengue_df.interpolate()

# Load dengue search term
folder = "<INSERT FILE PATH>"
dengue_google = folder + r"\google-search-term-dengue.csv"
google_dengue_df = pd.read_csv(dengue_google)
google_dengue_df.set_index("date", inplace = True)
dengue_df = pd.merge(dengue_df, google_dengue_df, how='inner', left_index=True, right_index=True)
dengue_df['google_search_for_dengue'] = 8 * dengue_df['google_search_for_dengue']

# Load eye pain search term
folder = "<INSERT FILE PATH>"
eye_google = folder + r"\google-search-term-eye-pain.csv"
eye_df = pd.read_csv(eye_google)
eye_df.set_index("date", inplace = True)
dengue_df = pd.merge(dengue_df, eye_df, how='inner', left_index=True, right_index=True)
dengue_df['google_search_for_eye_pain'] = dengue_df['google_search_for_eye_pain'] * 8

# Load fever search term
folder = "<INSERT FILE PATH>"
fever_google = folder + r"\google-search-term-fever.csv"
fever_df = pd.read_csv(fever_google)
fever_df.set_index("date", inplace = True)
dengue_df = pd.merge(dengue_df, fever_df, how='inner', left_index=True, right_index=True)
dengue_df['google_search_for_fever'] = dengue_df['google_search_for_fever'] * 8

# Load headache search term
folder = "<INSERT FILE PATH>"
headache_google = folder + r"\google-search-term-headache.csv"
heacache_df = pd.read_csv(headache_google)
heacache_df.set_index("date", inplace = True)
dengue_df = pd.merge(dengue_df, heacache_df, how='inner', left_index=True, right_index=True)
dengue_df['google_search_for_headache'] = dengue_df['google_search_for_headache'] * 8

# Load joint search term
folder = "<INSERT FILE PATH>"
joint_google = folder + r"\google-search-term-joint-pain.csv"
joint_df = pd.read_csv(joint_google)
joint_df.set_index("date", inplace = True)
dengue_df = pd.merge(dengue_df, joint_df, how='inner', left_index=True, right_index=True)
dengue_df['google_search_for_joint_pain'] = dengue_df['google_search_for_joint_pain'] * 8

# Load nausea search term
folder = "<INSERT FILE PATH>"
nausea_google = folder + r"\google-search-term-nausea.csv"
nausea_df = pd.read_csv(nausea_google)
nausea_df.set_index("date", inplace = True)
dengue_df = pd.merge(dengue_df, nausea_df, how='inner', left_index=True, right_index=True)
dengue_df['google_search_for_nausea'] = dengue_df['google_search_for_nausea'] * 8

# Load rash search term
folder = "<INSERT FILE PATH>"
rash_google = folder + r"\google-search-term-rash.csv"
rash_df = pd.read_csv(rash_google)
rash_df.set_index("date", inplace = True)
dengue_df = pd.merge(dengue_df, rash_df, how='inner', left_index=True, right_index=True)
dengue_df['google_search_for_rash'] = dengue_df['google_search_for_rash'] * 8

# Reobtain test and train, with exog
train_dengue = dengue_df[:'2019-10-01']
test_dengue = dengue_df['2019-10-01':]

# Obtain Test MSE with dengue search term
model = SARIMAX(train_dengue['cases'], order = (2,1,2), seasonal_order = (0,0,1,25),
                exog = train_dengue['google_search_for_dengue'])
results = model.fit()
test_forecast = results.get_forecast(steps = len(test_dengue), 
                                     exog = test_dengue['google_search_for_dengue'] )
mean_forecast = test_forecast.predicted_mean
test_score = mean_squared_error(test_y, mean_forecast)
print('Test score for model: ', test_score) # 2594.847358382291

# Obtain Test MSE with eye pain search term
model = SARIMAX(train_dengue['cases'], order = (2,1,2), seasonal_order = (0,0,1,25),
                exog = train_dengue['google_search_for_eye_pain'])
results = model.fit()
test_forecast = results.get_forecast(steps = len(test_dengue), 
                                     exog = test_dengue['google_search_for_eye_pain'] )
mean_forecast = test_forecast.predicted_mean
test_score = mean_squared_error(test_y, mean_forecast)
print('Test score for model: ', test_score) # 2561.123533746704

# Obtain Test MSE with fever search term
model = SARIMAX(train_dengue['cases'], order = (2,1,2), seasonal_order = (0,0,1,25),
                exog = train_dengue['google_search_for_fever'])
results = model.fit()
test_forecast = results.get_forecast(steps = len(test_dengue), 
                                     exog = test_dengue['google_search_for_fever'] )
mean_forecast = test_forecast.predicted_mean
test_score = mean_squared_error(test_y, mean_forecast)
print('Test score for model: ', test_score) # 779.07898906797

# Obtain Test MSE with headache search term
model = SARIMAX(train_dengue['cases'], order = (2,1,2), seasonal_order = (0,0,1,25),
                exog = train_dengue['google_search_for_headache'])
results = model.fit()
test_forecast = results.get_forecast(steps = len(test_dengue), 
                                     exog = test_dengue['google_search_for_headache'] )
mean_forecast = test_forecast.predicted_mean
test_score = mean_squared_error(test_y, mean_forecast)
print('Test score for model: ', test_score) # 2725.543790238676

# Obtain Test MSE with joint search term
model = SARIMAX(train_dengue['cases'], order = (2,1,2), seasonal_order = (0,0,1,25),
                exog = train_dengue['google_search_for_joint_pain'])
results = model.fit()
test_forecast = results.get_forecast(steps = len(test_dengue), 
                                     exog = test_dengue['google_search_for_joint_pain'] )
mean_forecast = test_forecast.predicted_mean
test_score = mean_squared_error(test_y, mean_forecast)
print('Test score for model: ', test_score) # 2936.5832400065747

# Obtain Test MSE with nausea search term
model = SARIMAX(train_dengue['cases'], order = (2,1,2), seasonal_order = (0,0,1,25),
                exog = train_dengue['google_search_for_nausea'])
results = model.fit()
test_forecast = results.get_forecast(steps = len(test_dengue), 
                                     exog = test_dengue['google_search_for_nausea'] )
mean_forecast = test_forecast.predicted_mean
test_score = mean_squared_error(test_y, mean_forecast)
print('Test score for model: ', test_score) # 3522.187578224679

# Obtain Test MSE with rash search term
model = SARIMAX(train_dengue['cases'], order = (2,1,2), seasonal_order = (0,0,1,25),
                exog = train_dengue['google_search_for_rash'])
results = model.fit()
test_forecast = results.get_forecast(steps = len(test_dengue), 
                                     exog = test_dengue['google_search_for_rash'] )
mean_forecast = test_forecast.predicted_mean
test_score = mean_squared_error(test_y, mean_forecast)
print('Test score for model: ', test_score) # 2095.7111401730426

# Train the other non google terms ============================================

# Load population term
folder = "<INSERT FILE PATH>"
population = folder + r"\population.csv"
pop_df = pd.read_csv(population)
pop_df.set_index("date", inplace = True)
pop_df = pop_df.replace({',':''},regex=True).apply(pd.to_numeric,1)
dengue_df = pd.merge(dengue_df, pop_df, how='inner', left_index=True, right_index=True)
dengue_df['population'] = dengue_df['population'] / 10000

# Get Test MSE for population
train_dengue = dengue_df[:'2019-10-01']
test_dengue = dengue_df['2019-10-01':]
model = SARIMAX(train_dengue['cases'], order = (2,1,2), seasonal_order = (0,0,1,25),
                exog = train_dengue['population'])
results = model.fit(method = 'nm', maxiter=200)
test_forecast = results.get_forecast(steps = len(test_dengue), 
                                     exog = test_dengue['population'] )
mean_forecast = test_forecast.predicted_mean
test_score = mean_squared_error(test_y, mean_forecast)
print('Test score for model: ', test_score) # 2998.9872467581586

# Load humidity term
folder = "<INSERT FILE PATH>"
humid = folder + r"\humidity.csv"
humid_df = pd.read_csv(humid)
humid_df.set_index("date", inplace = True)
dengue_df = pd.merge(dengue_df, humid_df, how='inner', left_index=True, right_index=True)
dengue_df['mean_rh'] = 6 * dengue_df['mean_rh']

# Get Test MSE for humidity
train_dengue = dengue_df[:'2019-10-01']
test_dengue = dengue_df['2019-10-01':]
model = SARIMAX(train_dengue['cases'], order = (2,1,2), seasonal_order = (0,0,1,25),
                exog = train_dengue['mean_rh'])
results = model.fit(method = 'nm', maxiter=200)
test_forecast = results.get_forecast(steps = len(test_dengue), 
                                     exog = test_dengue['mean_rh'] )
mean_forecast = test_forecast.predicted_mean
test_score = mean_squared_error(test_y, mean_forecast)
print('Test score for model: ', test_score) #  3202.944198785054

# Load temperature term
folder = "<INSERT FILE PATH>"
temp = folder + r"\temperature.csv"
temp_df = pd.read_csv(temp)
temp_df.set_index("dateTime", inplace = True)
dengue_df = pd.merge(dengue_df, temp_df['mean_temp'], how='inner', left_index=True, right_index=True)
dengue_df['mean_temp'] = dengue_df['mean_temp'] * 8

# Get Test MSE for mean temp
train_dengue = dengue_df[:'2019-10-01']
test_dengue = dengue_df['2019-10-01':]
model = SARIMAX(train_dengue['cases'], order = (2,1,2), seasonal_order = (0,0,1,25),
                exog = train_dengue['mean_temp'])
results = model.fit(maxiter=2000)
test_forecast = results.get_forecast(steps = len(test_dengue), 
                                     exog = test_dengue['mean_temp'] )
mean_forecast = test_forecast.predicted_mean
test_score = mean_squared_error(test_y, mean_forecast)
print('Test score for model: ', test_score) # 2463.66153468217

# Get Test MSE for max_temperature temp
dengue_df = pd.merge(dengue_df, temp_df['max_temperature'], how='inner', left_index=True, right_index=True)
dengue_df['max_temperature'] = dengue_df['max_temperature'] * 8
train_dengue = dengue_df[:'2019-10-01']
test_dengue = dengue_df['2019-10-01':]
model = SARIMAX(train_dengue['cases'], order = (2,1,2), seasonal_order = (0,0,1,25),
                exog = train_dengue['max_temperature'])
results = model.fit(maxiter=2000)
test_forecast = results.get_forecast(steps = len(test_dengue), 
                                     exog = test_dengue['max_temperature'] )
mean_forecast = test_forecast.predicted_mean
test_score = mean_squared_error(test_y, mean_forecast)
print('Test score for model: ', test_score) # 1924.3619191086284

# Get Test MSE for temp_extremes_min temp
dengue_df['temp_extremes_min'] = dengue_df['temp_extremes_min'] * 8
train_dengue = dengue_df[:'2019-10-01']
test_dengue = dengue_df['2019-10-01':]
model = SARIMAX(train_dengue['cases'], order = (2,1,2), seasonal_order = (0,0,1,25),
                exog = train_dengue['temp_extremes_min'])
results = model.fit(maxiter=2000)
test_forecast = results.get_forecast(steps = len(test_dengue), 
                                     exog = test_dengue['temp_extremes_min'] )
mean_forecast = test_forecast.predicted_mean
test_score = mean_squared_error(test_y, mean_forecast)
print('Test score for model: ', test_score) # 2632.473257205194

# Get Test MSE for temp_mean_daily_max temp
dengue_df = pd.merge(dengue_df, temp_df['temp_mean_daily_max'], how='inner', left_index=True, right_index=True)
dengue_df['temp_mean_daily_max'] = dengue_df['temp_mean_daily_max'] * 8
train_dengue = dengue_df[:'2019-10-01']
test_dengue = dengue_df['2019-10-01':]
model = SARIMAX(train_dengue['cases'], order = (2,1,2), seasonal_order = (0,0,1,25),
                exog = train_dengue['temp_mean_daily_max'])
results = model.fit(maxiter=2000)
test_forecast = results.get_forecast(steps = len(test_dengue), 
                                     exog = test_dengue['temp_mean_daily_max'] )
mean_forecast = test_forecast.predicted_mean
test_score = mean_squared_error(test_y, mean_forecast)
print('Test score for model: ', test_score) # 1756.0875507878227

# Get Test MSE for temp_mean_daily_min temp
dengue_df = pd.merge(dengue_df, temp_df['temp_mean_daily_min'], how='inner', left_index=True, right_index=True)
dengue_df['temp_mean_daily_min'] = dengue_df['temp_mean_daily_min'] * 8
train_dengue = dengue_df[:'2019-10-01']
test_dengue = dengue_df['2019-10-01':]
model = SARIMAX(train_dengue['cases'], order = (2,1,2), seasonal_order = (0,0,1,25),
                exog = train_dengue['temp_mean_daily_min'])
results = model.fit(maxiter=2000)
test_forecast = results.get_forecast(steps = len(test_dengue), 
                                     exog = test_dengue['temp_mean_daily_min'] )
mean_forecast = test_forecast.predicted_mean
test_score = mean_squared_error(test_y, mean_forecast)
print('Test score for model: ', test_score) # 3014.9711756937754

# Load rainfall data
folder = "<INSERT FILE PATH>"
rain = folder + r"\rain_changi.csv"
rain_df = pd.read_csv(rain)
rain_df.set_index("dateTime", inplace = True)
dengue_df = pd.merge(dengue_df, rain_df, how='inner', left_index=True, right_index=True)
dengue_df['maximum_rainfall_in_a_day'] = 3 * dengue_df['maximum_rainfall_in_a_day']
dengue_df['no_of_rainy_days'] = 3 * dengue_df['no_of_rainy_days']

# Get test mse for no_of_rainy_days
train_dengue = dengue_df[:'2019-10-01']
test_dengue = dengue_df['2019-10-01':]
model = SARIMAX(train_dengue['cases'], order = (2,1,2), seasonal_order = (0,0,1,25),
                exog = train_dengue['no_of_rainy_days'])
results = model.fit(maxiter=2000)
test_forecast = results.get_forecast(steps = len(test_dengue), 
                                     exog = test_dengue['no_of_rainy_days'] )
mean_forecast = test_forecast.predicted_mean
test_score = mean_squared_error(test_y, mean_forecast)
print('Test score for model: ', test_score) # 2822.9660622743872


# Get test mse for total_rainfall
train_dengue = dengue_df[:'2019-10-01']
test_dengue = dengue_df['2019-10-01':]
model = SARIMAX(train_dengue['cases'], order = (2,1,2), seasonal_order = (0,0,1,25),
                exog = train_dengue['total_rainfall'])
results = model.fit(maxiter=2000)
test_forecast = results.get_forecast(steps = len(test_dengue), 
                                     exog = test_dengue['total_rainfall'] )
mean_forecast = test_forecast.predicted_mean
test_score = mean_squared_error(test_y, mean_forecast)
print('Test score for model: ', test_score) # 3283.6915512723026


# Get test mse for maximum_rainfall_in_a_day
train_dengue = dengue_df[:'2019-10-01']
test_dengue = dengue_df['2019-10-01':]
model = SARIMAX(train_dengue['cases'], order = (2,1,2), seasonal_order = (0,0,1,25),
                exog = train_dengue['maximum_rainfall_in_a_day'])
results = model.fit(maxiter=2000)
test_forecast = results.get_forecast(steps = len(test_dengue), 
                                     exog = test_dengue['maximum_rainfall_in_a_day'] )
mean_forecast = test_forecast.predicted_mean
test_score = mean_squared_error(test_y, mean_forecast)
print('Test score for model: ', test_score) # 2006.3287604477339

# Load weather station data
folder = "<INSERT FILE PATH>"
weather = folder + r"\weather-station.csv"
weather_df = pd.read_csv(weather)
weather_df.set_index("dateTime", inplace = True)
dengue_df = pd.merge(dengue_df, weather_df, how='inner', left_index=True, right_index=True)

dengue_df['avg_daily_rainfall'] = dengue_df['avg_daily_rainfall'] * 8
dengue_df['avg_highest_30_min_rainfall'] = dengue_df['avg_highest_30_min_rainfall'] * 8
dengue_df['avg_highest_60_min_rainfall'] = dengue_df['avg_highest_60_min_rainfall'] * 8
dengue_df['avg_highest_120_min_rainfall'] = dengue_df['avg_highest_120_min_rainfall'] * 8
dengue_df['avg_mean_temp'] = dengue_df['avg_mean_temp'] * 8
dengue_df['avg_max_temp'] = dengue_df['avg_max_temp'] * 8
dengue_df['avg_min_temp'] = dengue_df['avg_min_temp'] * 8
dengue_df['avg_wind_speed'] = dengue_df['avg_wind_speed'] * 8
dengue_df['avg_max_wind_speed'] = dengue_df['avg_max_wind_speed'] * 8

# Get test mse for avg_daily_rainfall
train_dengue = dengue_df[:'2019-10-01']
test_dengue = dengue_df['2019-10-01':]
model = SARIMAX(train_dengue['cases'], order = (2,1,2), seasonal_order = (0,0,1,25),
                exog = train_dengue['avg_daily_rainfall'])
results = model.fit(maxiter=2000)
test_forecast = results.get_forecast(steps = len(test_dengue), 
                                     exog = test_dengue['avg_daily_rainfall'] )
mean_forecast = test_forecast.predicted_mean
test_score = mean_squared_error(test_y, mean_forecast)
print('Test score for model: ', test_score) # 2777.5573492494404

# Get test mse for avg_highest_30_min_rainfall
train_dengue = dengue_df['2016-01-01':'2019-10-01']
test_dengue = dengue_df['2019-10-01':]
model = SARIMAX(train_dengue['cases'], order = (2,1,2), seasonal_order = (0,0,1,25),
                exog = train_dengue['avg_highest_30_min_rainfall'])
results = model.fit(maxiter=2000)
test_forecast = results.get_forecast(steps = len(test_dengue), 
                                     exog = test_dengue['avg_highest_30_min_rainfall'] )
mean_forecast = test_forecast.predicted_mean
test_score = mean_squared_error(test_y, mean_forecast)
print('Test score for model: ', test_score) # 22562.84768366746

# Get test mse for avg_highest_60_min_rainfall
train_dengue = dengue_df['2016-01-01':'2019-10-01']
test_dengue = dengue_df['2019-10-01':]
model = SARIMAX(train_dengue['cases'], order = (2,1,2), seasonal_order = (0,0,1,25),
                exog = train_dengue['avg_highest_60_min_rainfall'])
results = model.fit(maxiter=2000)
test_forecast = results.get_forecast(steps = len(test_dengue), 
                                     exog = test_dengue['avg_highest_60_min_rainfall'] )
mean_forecast = test_forecast.predicted_mean
test_score = mean_squared_error(test_y, mean_forecast)
print('Test score for model: ', test_score) # 22801.728766219305

# Get test mse for avg_highest_120_min_rainfall
train_dengue = dengue_df['2016-01-01':'2019-10-01']
test_dengue = dengue_df['2019-10-01':]
model = SARIMAX(train_dengue['cases'], order = (2,1,2), seasonal_order = (0,0,1,25),
                exog = train_dengue['avg_highest_120_min_rainfall'])
results = model.fit(maxiter=2000)
test_forecast = results.get_forecast(steps = len(test_dengue), 
                                     exog = test_dengue['avg_highest_120_min_rainfall'] )
mean_forecast = test_forecast.predicted_mean
test_score = mean_squared_error(test_y, mean_forecast)
print('Test score for model: ', test_score) # 22699.507670343028

# Get test mse for avg_mean_temp
train_dengue = dengue_df[:'2019-10-01']
test_dengue = dengue_df['2019-10-01':]
model = SARIMAX(train_dengue['cases'], order = (2,1,2), seasonal_order = (0,0,1,25),
                exog = train_dengue['avg_mean_temp'])
results = model.fit(maxiter=2000)
test_forecast = results.get_forecast(steps = len(test_dengue), 
                                     exog = test_dengue['avg_mean_temp'] )
mean_forecast = test_forecast.predicted_mean
test_score = mean_squared_error(test_y, mean_forecast)
print('Test score for model: ', test_score) # 2831.6144760930765

# Get test mse for avg_max_temp
train_dengue = dengue_df[:'2019-10-01']
test_dengue = dengue_df['2019-10-01':]
model = SARIMAX(train_dengue['cases'], order = (2,1,2), seasonal_order = (0,0,1,25),
                exog = train_dengue['avg_max_temp'])
results = model.fit(maxiter=2000)
test_forecast = results.get_forecast(steps = len(test_dengue), 
                                     exog = test_dengue['avg_max_temp'] )
mean_forecast = test_forecast.predicted_mean
test_score = mean_squared_error(test_y, mean_forecast)
print('Test score for model: ', test_score) # 2882.660052859978

# Get test mse for avg_min_temp
train_dengue = dengue_df[:'2019-10-01']
test_dengue = dengue_df['2019-10-01':]
model = SARIMAX(train_dengue['cases'], order = (2,1,2), seasonal_order = (0,0,1,25),
                exog = train_dengue['avg_min_temp'])
results = model.fit(maxiter=2000)
test_forecast = results.get_forecast(steps = len(test_dengue), 
                                     exog = test_dengue['avg_min_temp'] )
mean_forecast = test_forecast.predicted_mean
test_score = mean_squared_error(test_y, mean_forecast)
print('Test score for model: ', test_score) # 2860.110550918022

# Get test mse for avg_wind_speed
train_dengue = dengue_df[:'2019-10-01']
test_dengue = dengue_df['2019-10-01':]
model = SARIMAX(train_dengue['cases'], order = (2,1,2), seasonal_order = (0,0,1,25),
                exog = train_dengue['avg_wind_speed'])
results = model.fit(maxiter=2000)
test_forecast = results.get_forecast(steps = len(test_dengue), 
                                     exog = test_dengue['avg_wind_speed'] )
mean_forecast = test_forecast.predicted_mean
test_score = mean_squared_error(test_y, mean_forecast)
print('Test score for model: ', test_score) # 2851.1418767047785

# Get test mse for avg_max_wind_speed
train_dengue = dengue_df[:'2019-10-01']
test_dengue = dengue_df['2019-10-01':]
model = SARIMAX(train_dengue['cases'], order = (2,1,2), seasonal_order = (0,0,1,25),
                exog = train_dengue['avg_max_wind_speed'])
results = model.fit(maxiter=2000)
test_forecast = results.get_forecast(steps = len(test_dengue), 
                                     exog = test_dengue['avg_max_wind_speed'] )
mean_forecast = test_forecast.predicted_mean
test_score = mean_squared_error(test_y, mean_forecast)
print('Test score for model: ', test_score) # 2845.354213369346











