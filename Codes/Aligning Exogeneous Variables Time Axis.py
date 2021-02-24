# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 10:06:33 2020

@author: Ahmad
"""

# Import dengue cases data
filepath = "<INSERT FILE PATH>"
dengue_df = pd.read_csv(filepath)
dengue_df["date"] = pd.to_datetime(dengue_df["date"], format = '%d/%m/%Y') # Converts the column to a datetime
dengue_df.set_index("date", inplace = True)
dengue_df = dengue_df.asfreq('W-MON')
# Observe that only 2 values are missing. Fill via interpolation
dengue_df = dengue_df.interpolate()

# Get the dengue index
index = dengue_df.index

# Align google trend search term for "dengue" =================================
folder = "<INSERT FILE PATH>"
dengue_search = folder + r"\dengue_timeline.csv"
dengue_search_df = pd.read_csv(dengue_search)
dengue_search_df["date"] = pd.to_datetime(dengue_search_df["Month"], format = '%Y-%m')
dengue_search_df = dengue_search_df.drop(['Month'], axis = 1)
dengue_search_df.set_index("date", inplace = True)
dengue_search_df = dengue_search_df.asfreq('D') # Set index frequency to 'day'
dengue_search_df = dengue_search_df.interpolate(method = 'linear')
dengue_search_df = dengue_search_df.asfreq('W-MON')
dengue_search_df = dengue_search_df.rename(columns={"dengue: (Singapore)": "google_search_for_dengue"})

# Verify that the axis has been properly alligned  by doing merging and plotting with dengue cases data
source = dengue_df.copy()
source = pd.merge(source, dengue_search_df, how='inner', left_index=True, right_index=True)

# Save the time transformed dengue search data
save_path = r"<INSERT FILE PATH>"
dengue_search_df.to_csv(save_path)

# Align google trend search term for "eye-pain" ===============================
folder = "<INSERT FILE PATH>"
eye_pain_location = folder + r"\eye_pain_timeline.csv"
eye_df = pd.read_csv(eye_pain_location)
eye_df["date"] = pd.to_datetime(eye_df["Month"], format = '%Y-%m')
eye_df = eye_df.drop(['Month'], axis = 1)
eye_df = eye_df.rename(columns={"eye pain: (Singapore)": "google_search_for_eye_pain"})
eye_df.set_index("date", inplace = True)
eye_df = eye_df.asfreq('D') 
eye_df = eye_df.interpolate(method = 'linear')
eye_df = eye_df.asfreq('W-MON')

source = dengue_df.copy()
source = pd.merge(source, eye_df, how='inner', left_index=True, right_index=True)
source.plot()

save_path = "<INSERT FILE PATH>"
eye_df.to_csv(save_path)

# Align google trend search term for 'fever' ==================================
folder = "<INSERT FILE PATH>"
fever_location = folder + r"\fever_timeline.csv"
fever_df = pd.read_csv(fever_location)
fever_df["date"] = pd.to_datetime(fever_df["Month"], format = '%Y-%m')
fever_df = fever_df.drop(['Month'], axis = 1)
fever_df = fever_df.rename(columns={"fever: (Singapore)": "google_search_for_fever"})
fever_df.set_index("date", inplace = True)
fever_df = fever_df.asfreq('D') 
fever_df = fever_df.interpolate(method = 'linear')
fever_df = fever_df.asfreq('W-MON')

source = dengue_df.copy()
source = pd.merge(source, fever_df, how='inner', left_index=True, right_index=True)
source.plot()

save_path = "<INSERT FILE PATH>"
fever_df.to_csv(save_path)

# Align google trend search term for 'headache' ===============================
folder = "<INSERT FILE PATH>"
headache_location = folder + r"\headache_timeline.csv"
headache_df = pd.read_csv(headache_location)
headache_df["date"] = pd.to_datetime(headache_df["Month"], format = '%Y-%m')
headache_df = headache_df.drop(['Month'], axis = 1)
headache_df = headache_df.rename(columns={"headache: (Singapore)": "google_search_for_headache"})
headache_df.set_index("date", inplace = True)
headache_df = headache_df.asfreq('D') 
headache_df = headache_df.interpolate(method = 'linear')
headache_df = headache_df.asfreq('W-MON')

source = dengue_df.copy()
source = pd.merge(source, headache_df, how='inner', left_index=True, right_index=True)
source.plot()

save_path = "<INSERT FILE PATH>"
headache_df.to_csv(save_path)

# Align google trend search term for 'joint-pain' ============================
folder = "<INSERT FILE PATH>"
joint_location = folder + r"\joint_pain_timeline.csv"
joint_df = pd.read_csv(joint_location)
joint_df["date"] = pd.to_datetime(joint_df["Month"], format = '%Y-%m')
joint_df = joint_df.drop(['Month'], axis = 1)
joint_df = joint_df.rename(columns={"joint pain: (Singapore)": "google_search_for_joint_pain"})
joint_df.set_index("date", inplace = True)
joint_df = joint_df.asfreq('D') 
joint_df = joint_df.interpolate(method = 'linear')
joint_df = joint_df.asfreq('W-MON')

source = dengue_df.copy()
source = pd.merge(source, joint_df, how='inner', left_index=True, right_index=True)
source.plot()

save_path = "<INSERT FILE PATH>"
joint_df.to_csv(save_path)

# Align google trend search term for 'nausea' =================================
folder = "<INSERT FILE PATH>"
nausea_location = folder + r"\nausea_timeline.csv"
nausea_df = pd.read_csv(nausea_location)
nausea_df["date"] = pd.to_datetime(nausea_df["Month"], format = '%Y-%m')
nausea_df = nausea_df.drop(['Month'], axis = 1)
nausea_df = nausea_df.rename(columns={"nausea: (Singapore)": "google_search_for_nausea"})
nausea_df.set_index("date", inplace = True)
nausea_df = nausea_df.asfreq('D') 
nausea_df = nausea_df.interpolate(method = 'linear')
nausea_df = nausea_df.asfreq('W-MON')

source = dengue_df.copy()
source = pd.merge(source, nausea_df, how='inner', left_index=True, right_index=True)
source.plot()

save_path = "<INSERT FILE PATH>"
nausea_df.to_csv(save_path)

# Align google trend search term for 'rash' ===================================
folder = "<INSERT FILE PATH>"
rash_location = folder + r"\rash_timeline.csv"
rash_df = pd.read_csv(rash_location)
rash_df["date"] = pd.to_datetime(rash_df["Month"], format = '%Y-%m')
rash_df = rash_df.drop(['Month'], axis = 1)
rash_df = rash_df.rename(columns={"rash: (Singapore)": "google_search_for_rash"})
rash_df.set_index("date", inplace = True)
rash_df = rash_df.asfreq('D') 
rash_df = rash_df.interpolate(method = 'linear')
rash_df = rash_df.asfreq('W-MON')

source = dengue_df.copy()
source = pd.merge(source, rash_df, how='inner', left_index=True, right_index=True)
source.plot()

save_path = r"<INSERT FILE PATH>"
rash_df.to_csv(save_path)

# Align SG Population data ====================================================
folder = "<INSERT FILE PATH>"
population_location = folder + r"\population_from_datagov_by_year.csv"
pop_df = pd.read_csv(population_location)
pop_df["date"] = pd.to_datetime(pop_df["year"], format = '%Y') # Original index is in years
pop_df = pop_df.drop(['year'], axis = 1)
pop_df.set_index("date", inplace = True)
pop_df = pop_df.replace({',':''},regex=True).apply(pd.to_numeric,1)
pop_df = pop_df.asfreq('D') # Introduce an index by days
pop_df = pop_df.interpolate(method = 'linear') # Do the linear interpolation
pop_df = pop_df.asfreq('W-MON') # Set the index to be aligned with the dengue data

source = dengue_df.copy()
source = pd.merge(source, pop_df, how='inner', left_index=True, right_index=True)
source.plot()

save_path = "<INSERT FILE PATH>"
pop_df.to_csv(save_path)

# Align rainfall data =========================================================
folder = "<INSERT FILE PATH>"
rain_location = folder + r"\rainfall_from_datagov_by_month.csv"
rain_df = pd.read_csv(rain_location)
rain_df["dateTime"] = pd.to_datetime(rain_df["date"], format = '%Y-%m-%d')
rain_df = rain_df.drop(['date'], axis = 1)
rain_df.set_index("dateTime", inplace = True)
rain_df = rain_df.asfreq('D') 
rain_df = rain_df.interpolate(method = 'linear')
rain_df = rain_df.asfreq('W-MON')

source = dengue_df.copy()
source = pd.merge(source, rain_df, how='inner', left_index=True, right_index=True)
source.plot()

save_path = "<INSERT FILE PATH>"
rain_df.to_csv(save_path)


# Align relative humidity data ================================================
folder = "<INSERT FILE PATH>"
humid_location = folder + r"\relative_humidity_from_datagov_by_month.csv"
humid_df = pd.read_csv(humid_location)
humid_df["date"] = pd.to_datetime(humid_df["month"], format = '%Y-%m')
humid_df = humid_df.drop(['month'], axis = 1)
humid_df.set_index("date", inplace = True)
humid_df = humid_df.asfreq('D') 
humid_df = humid_df.interpolate(method = 'linear')
humid_df = humid_df.asfreq('W-MON')

source = dengue_df.copy()
source = pd.merge(source, humid_df, how='inner', left_index=True, right_index=True)
source.plot()

save_path = "<INSERT FILE PATH>"
humid_df.to_csv(save_path)

# Align temperature data ======================================================
folder = "<INSERT FILE PATH>"
temp_location = folder + r"\temperature_from_datagov_by_month.csv"
temp_df = pd.read_csv(temp_location)
temp_df["dateTime"] = pd.to_datetime(temp_df["date"], format = '%Y-%m-%d')
temp_df = temp_df.drop(['date'], axis = 1)
temp_df.set_index("dateTime", inplace = True)
temp_df = temp_df.asfreq('D') 
temp_df = temp_df.interpolate(method = 'linear')
temp_df = temp_df.asfreq('W-MON')

source = dengue_df.copy()
source = pd.merge(source, temp_df, how='inner', left_index=True, right_index=True)
source.plot()

save_path = "<INSERT FILE PATH>"
temp_df.to_csv(save_path)

# Align weather station data ==================================================
folder = "<INSERT FILE PATH>"
weather_location = folder + r"\cleaned_weather_station_by_week.csv"
weather_df = pd.read_csv(weather_location)

weather_df['weekNum'] = ''
weather_df['yearNum'] = ''
for index, row in weather_df.iterrows():
    date_string = str(row['week'])
    week_num = int(date_string[4:6])
    year_num = int(date_string[0:4])
    
    weather_df.at[index, 'weekNum'] = week_num
    weather_df.at[index, 'yearNum'] = year_num
    
weather_df['dateTime'] = pd.to_datetime(weather_df.yearNum.astype(str), format='%Y') + \
             pd.to_timedelta(weather_df.weekNum.mul(7).astype(str) + ' days')
weather_df = weather_df.drop(['Unnamed: 0', 'week', 'weekNum', 'yearNum'], axis = 1)
weather_df.set_index("dateTime", inplace = True)
weather_df = weather_df.asfreq('D') 
weather_df = weather_df.interpolate(method = 'linear')
weather_df = weather_df.asfreq('W-MON')

source = dengue_df.copy()
source = pd.merge(source, weather_df, how='inner', left_index=True, right_index=True)
source.plot()

save_path = "<INSERT FILE PATH>"
weather_df.to_csv(save_path)






