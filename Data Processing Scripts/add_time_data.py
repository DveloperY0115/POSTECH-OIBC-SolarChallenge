import pandas as pd
from datetime import datetime

# load original data
data = pd.read_csv('Weather-Energy_rev_data.csv')
dataset = data.loc[:,['temperature', 'wind_speed(m/s)', 'humidity(%)', 'solar_radiation(MJ/m^2)','energy']]
dataset.index = data.loc[:, 'time']

# get the datetime object for index
days = []
for date_time_str in dataset.index:
    days.append(datetime.strptime(date_time_str, '%Y-%m-%d %H:%M'))

hours = []
for day in days:
    hours.append(day.hour)

# add hour column
dataset['hour'] = hours

last = datetime(2018,12,31)

# calculate the day of year using last
day_of_years = []
for day in days:
    doy = day - last
    doy = doy.days
    if(doy > 365):
        doy -= 365
    day_of_years.append(doy)
    
# add day column
dataset['day'] = day_of_years
cols = dataset.columns.to_list()
cols = cols[-2:] + cols[:-2]
dataset = dataset[cols]






