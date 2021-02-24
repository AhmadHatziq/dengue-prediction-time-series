# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 09:43:21 2020

@author: Ahmad
"""

population_file_path = r"<INSERT FILE PATH>"
pop_df = pd.read_csv(population_file_path)
pop_df["Year"] = pd.to_datetime(pop_df["year"], format = '%Y')
pop_df = pop_df.drop(['year'], axis = 1)
pop_df.set_index("Year", inplace = True)
pop_df = pop_df.replace({',':''},regex=True).apply(pd.to_numeric,1)

ax = plt.gca()

plt.title('SG Population')
plt.ylabel('Population persons')
pop_df.plot()

folder = "<INSERT FILE PATH>"
rain_location = folder + r"\rainfall_from_datagov_by_month.csv"
rain_df = pd.read_csv(rain_location)
rain_df["Year"] = pd.to_datetime(rain_df["date"], format = '%Y-%m-%d')
rain_df = rain_df.drop(['date'], axis = 1)
rain_df.set_index("Year", inplace = True)

plt.title('No. of Rainfall Days per Month from Changi Station')
plt.ylabel('No. of Days')
rain_df['no_of_rainy_days'].plot()

plt.title('Total Rainfall per Month from Changi Station')
plt.ylabel('Rainfall (mm)')
rain_df['total_rainfall'].plot()


humidity_filepath = r"<INSERT FILE PATH>"
humid_df = pd.read_csv(humidity_filepath)
humid_df["Year"] = pd.to_datetime(humid_df["month"], format = '%Y-%m')
humid_df = humid_df.drop(['month'], axis = 1)
humid_df.set_index("Year", inplace = True)

plt.title('Relative Humidity from Changi Climate Station')
plt.ylabel('Relative humidity')
humid_df['mean_rh'].plot()


folder = "<INSERT FILE PATH>"
dengue_search = folder + r"\dengue_timeline.csv"
dengue_search_df = pd.read_csv(dengue_search)
dengue_search_df["date"] = pd.to_datetime(dengue_search_df["Month"], format = '%Y-%m')
dengue_search_df = dengue_search_df.drop(['Month'], axis = 1)
dengue_search_df.set_index("date", inplace = True)

eye_pain = folder + r"\eye_pain_timeline.csv"
eye_df = pd.read_csv(eye_pain)
eye_df["date"] = pd.to_datetime(eye_df["Month"], format = '%Y-%m')
eye_df = eye_df.drop(['Month'], axis = 1)
eye_df.set_index("date", inplace = True)

fever = folder + r"\fever_timeline.csv"
fever_df = pd.read_csv(fever)
fever_df["date"] = pd.to_datetime(fever_df["Month"], format = '%Y-%m')
fever_df = fever_df.drop(['Month'], axis = 1)
fever_df.set_index("date", inplace = True)

headache = folder + r"\headache_timeline.csv"
headache_df = pd.read_csv(headache)
headache_df["date"] = pd.to_datetime(headache_df["Month"], format = '%Y-%m')
headache_df = headache_df.drop(['Month'], axis = 1)
headache_df.set_index("date", inplace = True)

google = dengue_search_df.copy()
google = pd.merge(google, eye_df, how='inner', left_index=True, right_index=True)
google = pd.merge(google, fever_df, how='inner', left_index=True, right_index=True)
google = pd.merge(google, headache_df, how='inner', left_index=True, right_index=True)


nausea = folder + r"\nausea_timeline.csv"
nausea_df = pd.read_csv(nausea)
nausea_df["date"] = pd.to_datetime(nausea_df["Month"], format = '%Y-%m')
nausea_df = nausea_df.drop(['Month'], axis = 1)
nausea_df.set_index("date", inplace = True)

joint = folder + r"\joint_pain_timeline.csv"
joint_df = pd.read_csv(joint)
joint_df["date"] = pd.to_datetime(joint_df["Month"], format = '%Y-%m')
joint_df = joint_df.drop(['Month'], axis = 1)
joint_df.set_index("date", inplace = True)

rash = folder + r"\rash_timeline.csv"
rash_df = pd.read_csv(rash)
rash_df["date"] = pd.to_datetime(rash_df["Month"], format = '%Y-%m')
rash_df = rash_df.drop(['Month'], axis = 1)
rash_df.set_index("date", inplace = True)

google = nausea_df.copy()
google = pd.merge(google, joint_df, how='inner', left_index=True, right_index=True)
google = pd.merge(google, rash_df, how='inner', left_index=True, right_index=True)


filepath = "<INSERT FILE PATH>"
df = pd.read_csv(filepath)
df = df.iloc[675: 700]

import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import table  # EDIT: see deprecation warnings below

ax = plt.subplot(111, frame_on=False) # no visible frame
ax.xaxis.set_visible(False)  # hide the x axis
ax.yaxis.set_visible(False)  # hide the y axis

table(ax, df)  # where df is your data frame




