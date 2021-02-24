# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 15:38:39 2020

Code to clean web-scrapped weather station data

@author: Ahmad
"""

import numpy as np
import pandas as pd
pd.options.display.max_columns = None
pd.options.display.max_rows = None

folder = "<INSERT FILE PATH>"
filepath = folder + r"\cleaned_scrapped.csv"

scrapped_df = pd.read_csv(filepath)

# Generate a YYYYMMDD Column
scrapped_df['date_string'] = "" 
for index, row in scrapped_df.iterrows():
    month_num = row['Month']
    day_num = row['Day']
    year_num = row['Year']
    
    month_string = ""
    if month_num < 10:
        month_string = "0" + str(month_num)
    else:
        month_string = str(month_num)
        
    day_string = ""
    if day_num < 10:
        day_string = "0" + str(day_num)
    else:
        day_string = str(day_num)
        
    date_string = str(year_num) + month_string + day_string
    
    scrapped_df.at[index, 'date_string'] = date_string
    
# Create a datetime column
scrapped_df['DateTime'] = pd.to_datetime(scrapped_df["date_string"])
scrapped_df = scrapped_df.drop(['date_string'], axis=1)

# Save the cleaned data for manual inspection
save_path = "<INSERT FILE PATH>"
scrapped_df.to_csv(save_path)

# Drop previous date columns
scrapped_df = scrapped_df.drop(['Year', 'Month', 'Day'], axis=1)

# Create a column detailing year + week
scrapped_df['year-week'] = ""
for index, row in scrapped_df.iterrows():
    week_num = row['DateTime'].week
    year_num = row['DateTime'].year
    
    week_string = ""
    if week_num < 10:
        week_string = '0' + str(week_num)
    else:
        week_string = str(week_num)
        
    year_week_string = str(year_num) + week_string
    scrapped_df.at[index, 'year-week'] = year_week_string
    
# Iterate through all weeks
weeks = scrapped_df['year-week'].unique()

week_data = pd.DataFrame(columns = ['week', 'avg_daily_rainfall', 'avg_highest_30_min_rainfall', 
                                    'avg_highest_60_min_rainfall', 'avg_highest_120_min_rainfall', 
                                    'avg_mean_temp', 'avg_max_temp', 'avg_min_temp', 'avg_wind_speed', 
                                    'avg_max_wind_speed'])

index = -1
for week in weeks:
    index = index + 1
    week_df = scrapped_df[scrapped_df['year-week'] == week]
    
    avg_daily_rainfall = week_df['Daily Rainfall Total (mm)'].mean()
    avg_highest_30_min_rainfall = week_df['Highest 30 Min Rainfall (mm)'].mean()
    avg_highest_60_min_rainfall = week_df['Highest 60 Min Rainfall (mm)'].mean()
    avg_highest_120_min_rainfall = week_df['Highest 120 Min Rainfall (mm)'].mean()
    avg_mean_temp = week_df['Mean Temperature'].mean()
    avg_max_temp = week_df['Maximum Temperature'].mean()
    avg_min_temp = week_df['Minimum Temperature'].mean()
    avg_wind_speed = week_df['Mean Wind Speed (km/h)'].mean()
    avg_max_wind_speed = week_df['Max Wind Speed (km/h)'].mean()
    
    # Insert the data into the dataframe
    week_data.at[index, 'week'] = week
    week_data.at[index, 'avg_daily_rainfall'] = avg_daily_rainfall
    week_data.at[index, 'avg_highest_30_min_rainfall'] = avg_highest_30_min_rainfall
    week_data.at[index, 'avg_highest_60_min_rainfall'] = avg_highest_60_min_rainfall
    week_data.at[index, 'avg_highest_120_min_rainfall'] = avg_highest_120_min_rainfall
    week_data.at[index, 'avg_mean_temp'] = avg_mean_temp
    week_data.at[index, 'avg_max_temp'] = avg_max_temp
    week_data.at[index, 'avg_min_temp'] = avg_min_temp
    week_data.at[index, 'avg_wind_speed'] = avg_wind_speed
    week_data.at[index, 'avg_max_wind_speed'] = avg_max_wind_speed
    
    
# Save the data to file
save_path = "<INSERT FILE PATH>"
week_data.to_csv(save_path)   
