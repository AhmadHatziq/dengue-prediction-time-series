# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 22:32:36 2020

This code will combine the temperature data files from Data Gov into a single dataframe and save
into file

@author: Ahmad
"""
import pandas as pd
import numpy as np

# File paths to csv files
folder = "<INSERT FILE PATH>"
extreme_max = folder + r"\surface-air-temperature-monthly-absolute-extreme-maximum.csv"
extreme_min = folder + r"\surface-air-temperature-monthly-absolute-extreme-minimum.csv"
mean = folder + r"\surface-air-temperature-monthly-mean.csv"
avg_max = folder + r"\surface-air-temperature-monthly-mean-daily-maximum.csv"
avg_min = folder + r"\surface-air-temperature-monthly-mean-daily-minimum.csv"

# Read in the csv files
extreme_max_df = pd.read_csv(extreme_max)
extreme_min_df = pd.read_csv(extreme_min)
mean_df = pd.read_csv(mean)
avg_max_df = pd.read_csv(avg_max)
avg_min_df = pd.read_csv(avg_min)

# Note that all the datasets have the same length
print("extreme_max_df length: ", len(extreme_max_df))
print("extreme_min_df length: ", len(extreme_min_df) )
print("mean_df length: ", len(mean_df) )
print("avg_max_df length: ", len(avg_max_df) )
print("avg_min_df length: ", len(avg_min_df) )

# Create a datetime index for each dataframe

extreme_max_df["date"] = pd.to_datetime(extreme_max_df["month"])
extreme_max_df.set_index("date", inplace = True)
extreme_max_df = extreme_max_df.drop("month", axis = 1)

extreme_min_df["date"] = pd.to_datetime(extreme_min_df["month"])
extreme_min_df.set_index("date", inplace = True)
extreme_min_df = extreme_min_df.drop("month", axis = 1)

mean_df["date"] = pd.to_datetime(mean_df["month"])
mean_df.set_index("date", inplace = True)
mean_df = mean_df.drop("month", axis = 1)

avg_max_df["date"] = pd.to_datetime(avg_max_df["month"])
avg_max_df.set_index("date", inplace = True)
avg_max_df = avg_max_df.drop("month", axis = 1)

avg_min_df["date"] = pd.to_datetime(avg_min_df["month"])
avg_min_df.set_index("date", inplace = True)
avg_min_df = avg_min_df.drop("month", axis = 1)

# Find if there are any nans
print("extreme_max_df, number on NaNs: ", extreme_max_df.isnull().sum().sum() )
print("extreme_min_df, number on NaNs: ", extreme_min_df.isnull().sum().sum() )
print("mean_df, number on NaNs: ", mean_df.isnull().sum().sum() )
print("avg_max_df, number on NaNs: ", avg_max_df.isnull().sum().sum() )
print("avg_min_df, number on NaNs: ", avg_min_df.isnull().sum().sum() )

# Join the dataframes into a single datframe
temp_df = extreme_max_df.copy()
temp_df = pd.merge(temp_df, extreme_min_df, how='inner', left_index=True, right_index=True)
temp_df = pd.merge(temp_df, mean_df, how='inner', left_index=True, right_index=True)
temp_df = pd.merge(temp_df, avg_max_df, how='inner', left_index=True, right_index=True)
temp_df = pd.merge(temp_df, avg_min_df, how='inner', left_index=True, right_index=True)

# Save the cleaned data to a csv file
save_path = "<INSERT FILE PATH>"
temp_df.to_csv(save_path)
