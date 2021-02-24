# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 19:16:10 2020

This file aims to extract the 3 rainfall data files and combine them into
a single dataframe and subsequently saves the cleaned dataframe to a new csv

@author: Ahmad
"""

import pandas as pd
import numpy as np

# Set filepaths of data csv files
rainfall_highest_filepath = "<INSERT FILE PATH>"
rainfall_number_rainy_days_filepath = "<INSERT FILE PATH>"
rainfall_total_filepath = "<INSERT FILE PATH>"

# Import the data from csv to dataframes
rain_highest_df = pd.read_csv(rainfall_highest_filepath)
rain_days_df = pd.read_csv(rainfall_number_rainy_days_filepath)
rain_total_df = pd.read_csv(rainfall_total_filepath)

# Note that all the datasets have the same length
print("Highest rain length: ", len(rain_highest_df))
print("Rain days count length: ", len(rain_days_df) )
print("Total rain length: ", len(rain_total_df) )

# Create a datetime index for each dataframe

rain_highest_df["date"] = pd.to_datetime(rain_highest_df["month"])
rain_highest_df.set_index("date", inplace = True)
rain_highest_df = rain_highest_df.drop("month", axis = 1)

rain_days_df["date"] = pd.to_datetime(rain_days_df["month"])
rain_days_df.set_index("date", inplace = True)
rain_days_df = rain_days_df.drop("month", axis = 1)

rain_total_df["date"] = pd.to_datetime(rain_total_df["month"])
rain_total_df.set_index("date", inplace = True)
rain_total_df = rain_total_df.drop("month", axis = 1)

# Find if there are any nans
print("Rain total, number on NaNs: ", rain_total_df.isnull().sum().sum() )
print("Rain highest, number on NaNs: ", rain_highest_df.isnull().sum().sum() )
print("Rain days, number on NaNs: ", rain_days_df.isnull().sum().sum() )

# Join the dataframes into a single datframe
rainfall_df = rain_highest_df.copy()
rainfall_df = pd.merge(rainfall_df, rain_days_df, how='inner', left_index=True, right_index=True)
rainfall_df = pd.merge(rainfall_df, rain_total_df, how='inner', left_index=True, right_index=True)

# Save the cleaned data to a csv file
save_path = "<INSERT FILE PATH>"
rainfall_df.to_csv(save_path)

