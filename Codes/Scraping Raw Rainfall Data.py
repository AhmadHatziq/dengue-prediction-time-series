# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 16:41:09 2020

@author: Ahmad
"""

import requests
import pandas as pd
import numpy as np

# Test on a single working station
csv_file_path = "<INSERT FILE PATH>"
working_url = "http://www.weather.gov.sg/files/dailydata/DAILYDATA_S24_201911.csv"
test_req = requests.get(working_url, allow_redirects = True)
open(csv_file_path, 'wb').write(test_req.content)

# Sample working URL: http://www.weather.gov.sg/files/dailydata/DAILYDATA_S24_201911.csv
# Sample Not working: http://www.weather.gov.sg/files/dailydata/DAILYDATA_S21_201912.csv
working_url = "http://www.weather.gov.sg/files/dailydata/DAILYDATA_S24_201911.csv"

# Test for a single Station 24 ========================================================================
stations = [24]

# Initalize an empty dataframe to store the data
scrapped_df = pd.DataFrame(columns = ['Station', 'Year', 'Month', 'Day', 'Daily Rainfall Total (mm)',
       'Highest 30 Min Rainfall (mm)', 'Highest 60 Min Rainfall (mm)',
       'Highest 120 Min Rainfall (mm)', 'Mean Temperature (°C)',
       'Maximum Temperature (°C)', 'Minimum Temperature (°C)',
       'Mean Wind Speed (km/h)', 'Max Wind Speed (km/h)'])

csv_file_path = r"<INSERT FILE PATH>"
for station_num in stations: 
    
    # Parse the station number to be 2 digits
    station_string = ""
    if station_num < 10:
        station_string = "%02d" % station_num
    elif station >= 10:
        station_string = str(station_num)
        
    # print(station_string)
    
    
    for year_num in range(2018, 2020):
        for month_num in range(1, 13):
            
            # Parse the month number to be 2 digits
            month_string = ""
            if month_num < 10:
                month_string = "%02d" % month_num
            elif month_num >= 10:
                month_string = str(month_num)
            
            # print(str(year_num) + month_string)
                
            # Construct the csv url
            csv_url = "http://www.weather.gov.sg/files/dailydata/DAILYDATA_S" + station_string
            csv_url = csv_url + "_" + str(year_num) + month_string + ".csv"
            
            # Access the url
            request_object = requests.get(csv_url, allow_redirects=True)
            
            http_code = request_object.status_code
            
            # HTTP 200 is code OK
            if http_code == 200:
                # Write the data to a csv file path
                open(csv_file_path, 'wb').write(request_object.content)
                scrapped_df = scrapped_df.append(pd.read_csv(csv_file_path, encoding = "ISO-8859-1"))
                
    


# Access the csv and attempt to pull the data via the http address for stations 1 - 30 ============================

# Initalize an empty dataframe to store the data
scrapped_df = pd.DataFrame(columns = ['Station', 'Year', 'Month', 'Day', 'Daily Rainfall Total (mm)',
       'Highest 30 Min Rainfall (mm)', 'Highest 60 Min Rainfall (mm)',
       'Highest 120 Min Rainfall (mm)', 'Mean Temperature (°C)',
       'Maximum Temperature (°C)', 'Minimum Temperature (°C)',
       'Mean Wind Speed (km/h)', 'Max Wind Speed (km/h)'])

csv_file_path = "<INSERT FILE PATH>"
stations = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
success_log = []

for station_num in stations: 
    
    # Parse the station number to be 2 digits
    station_string = ""
    if station_num < 10:
        station_string = "%02d" % station_num
    elif station >= 10:
        station_string = str(station_num)
        
    # print(station_string)
    
    for year_num in range(2000, 2020):
        for month_num in range(1, 13):
            
            # Parse the month number to be 2 digits
            month_string = ""
            if month_num < 10:
                month_string = "%02d" % month_num
            elif month_num >= 10:
                month_string = str(month_num)
            
            # print(str(year_num) + month_string)
                
            # Construct the csv url
            csv_url = "http://www.weather.gov.sg/files/dailydata/DAILYDATA_S" + station_string
            csv_url = csv_url + "_" + str(year_num) + month_string + ".csv"
            
            # Access the url
            request_object = requests.get(csv_url, allow_redirects=True)
            
            http_code = request_object.status_code
            
            # HTTP 200 is code OK
            if http_code == 200:
                # Write the data to a csv file path
                open(csv_file_path, 'wb').write(request_object.content)
                scrapped_df = scrapped_df.append(pd.read_csv(csv_file_path, encoding = "ISO-8859-1"))
                success_log.append(csv_url)
    
copy = scrapped_df.copy()

# Save the cleaned data to a csv file
save_path = "<INSERT FILE PATH>"
scrapped_df.to_csv(save_path)

# Save the successfull https to a file
dfObj = pd.DataFrame(success_log) 
save_path = "<INSERT FILE PATH>"
dfObj.to_csv(save_path)


# ======================= End of section of scraping stations 1 - 30 ==================================

# Access the csv and attempt to pull the data via the http address for stations 31 - 99 ============================
station_list = list(range(31, 100))

scrapped_df_31_to_100 = pd.DataFrame(columns = ['Station', 'Year', 'Month', 'Day', 'Daily Rainfall Total (mm)',
       'Highest 30 Min Rainfall (mm)', 'Highest 60 Min Rainfall (mm)',
       'Highest 120 Min Rainfall (mm)', 'Mean Temperature (°C)',
       'Maximum Temperature (°C)', 'Minimum Temperature (°C)',
       'Mean Wind Speed (km/h)', 'Max Wind Speed (km/h)'])

csv_file_path = "<INSERT FILE PATH>"
success_log_v2 = []

for station_num in station_list: 
    
    # Parse the station number to be 2 digits
    station_string = ""
    if station_num < 10:
        station_string = "%02d" % station_num
    elif station >= 10:
        station_string = str(station_num)
        
    # print(station_string)
    
    for year_num in range(2000, 2020):
        for month_num in range(1, 13):
            
            # Parse the month number to be 2 digits
            month_string = ""
            if month_num < 10:
                month_string = "%02d" % month_num
            elif month_num >= 10:
                month_string = str(month_num)
            
            # print(str(year_num) + month_string)
                
            # Construct the csv url
            csv_url = "http://www.weather.gov.sg/files/dailydata/DAILYDATA_S" + station_string
            csv_url = csv_url + "_" + str(year_num) + month_string + ".csv"
            
            # Access the url
            request_object = requests.get(csv_url, allow_redirects=True)
            
            http_code = request_object.status_code
            
            # HTTP 200 is code OK
            if http_code == 200:
                # Write the data to a csv file path
                open(csv_file_path, 'wb').write(request_object.content)
                scrapped_df_31_to_100 = scrapped_df_31_to_100.append(pd.read_csv(csv_file_path, encoding = "ISO-8859-1"))
                success_log_v2.append(csv_url)
                
    print("Completed station: ", station_num)
    
copy = scrapped_df_31_to_100.copy()

# Save the cleaned data to a csv file
save_path = r"<INSERT FILE PATH>"
scrapped_df_31_to_100.to_csv(save_path)

# Save the successfull https to a file
dfObj = pd.DataFrame(success_log_v2) 
save_path = r"<INSERT FILE PATH>"
dfObj.to_csv(save_path)


# Combine both dfs into one and save
scrapped_total = scrapped_df_31_to_100.copy()
scrapped_total = scrapped_total.append(scrapped_df)
save_path = r"<INSERT FILE PATH>"
scrapped_total.to_csv(save_path)

# Combine both success logs into one and save
dfObj_1 = pd.DataFrame(success_log)
dfObj_2 = pd.DataFrame(success_log_v2)
dfObj_2 = dfObj_2.append(dfObj_1)
save_path = r"<INSERT FILE PATH>"
dfObj_2.to_csv(save_path)

# Clean the success df
sample_url = "http://www.weather.gov.sg/files/dailydata/DAILYDATA_S29_201712.csv"

dfObj_2['Station'] = ""
dfObj_2['Station_String'] = ""
dfObj_2['Month'] = ""
dfObj_2['Year'] = ""
for index, row in dfObj_2.iterrows():
    http_string = row[0]
    string_array  = http_string.split('/')
    final_string = string_array[5]
    string_array = final_string.split('.')
    final_string = string_array[0]
    string_array = final_string.split('_')
    
    station_string = string_array[1]
    date_string = string_array[2]
    
    year_string = date_string[-2:]
    month_string = date_string[0:4]
    
    year_num = int(year_string)
    month_num = int(month_string)
    station_num = int(station_string[1:3])
    
    dfObj_2.at[index, 'Station'] = station_num
    dfObj_2.at[index, 'Station_String'] = station_string
    dfObj_2.at[index, 'Month'] = month_num
    dfObj_2.at[index, 'Year'] = year_num

dfObj_2 = dfObj_2.rename(columns={0: "http_string", "Station": "Station_ID", "Station_String": "Station_String", "Month": "Month", "Year": "Year"})

# Save the dataframe

save_path = r"<INSERT FILE PATH>"
dfObj_2.to_csv(save_path)











