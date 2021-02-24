# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 13:49:15 2020

@author: Ahmad
"""

# Imports and set settings
import os
import glob
import seaborn as sns; sns.set()
from datetime import datetime
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import statsmodels.api as sm
pd.options.display.max_columns = None
pd.options.display.max_rows = None
plt.rcParams['figure.figsize'] = (16.0, 10)

# Set folder directory and file extension
folder = "<INSERT FILE PATH>"
file_ext = r"\*.csv"

# Store csv files in a list
csv_files = []
for filename in glob.glob(folder + file_ext ):
    csv_files.append(filename)

# Extract the rows from each year and convert to a single dataframe
rows_list = []
# Append data to dataframe
for i in range(len(csv_files)):
    # Extract data from each year
    year_cases = pd.read_csv(csv_files[i])
    year = int((os.path.basename(csv_files[i])).split(".")[0]) 
    
    
    # Create a row entry for each row
    for index, row in year_cases.iterrows():
        row_dict = {}
        row_dict['week'] = row["Epidemiology Wk"]
        row_dict['cases'] = row['dengue']
        row_dict['year'] = year
        rows_list.append(row_dict)
     
dengue_cases = pd.DataFrame(rows_list)

# Create a date-time column
dengue_cases['date'] = ""
for index, row in dengue_cases.iterrows():
    year = row['year']
    week = row['week']
    date_string = str(year) + "-W" + str(week)
    date =  datetime.datetime.strptime(date_string + '-1', "%Y-W%W-%w")
    dengue_cases.at[index, 'date'] = date

# Set the index as the year
dengue_cases.set_index("date", inplace = True)
dengue_cases.drop("week",axis = 1, inplace = True)
dengue_cases.drop("year",axis = 1, inplace = True)

# Get overall plot of cases
dengue_cases.plot(legend = True, label = 'Dengue Cases',title = 'Dengue Cases from 2012 to 2020')

# Super impose moving average
dengue_cases.plot(legend = True, label = 'Dengue Cases',title = 'Dengue Cases with Moving Averages')
dengue_cases.rolling(window = 7 * 30).mean()["cases"].plot(label = "30 Week Moving Average", legend = True)
dengue_cases.rolling(window = 7 * 10).mean()["cases"].plot(label = "10 Week Moving Average", legend = True)

# Save the dengue data
save_file = "<INSERT FILE PATH>"
dengue_cases.to_csv(save_file)

# Test if the csv file is valid
dengue_csv = "<INSERT FILE PATH>"
dengue_cases = pd.read_csv(dengue_csv)
dengue_cases["date"] = pd.to_datetime(dengue_cases["date"]) # Converts the column to a datetime
dengue_cases.set_index("date", inplace = True)

