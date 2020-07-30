import pandas as pd
from datetime import datetime

# load original data
data = pd.read_csv('../Data/ProductionData/Weather-Energy_rev_time_hourly_data.csv')
dataset = data.loc[:,['temperature', 'wind_speed(m/s)', 'humidity(%)', 'solar_radiation(MJ/m^2)','energy']]
dataset.index = data.loc[:, 'time']

whole_data = pd.read_csv('../Data/ProductionData/Weather-Energy_rev_time_hourly_data.csv')
whole_dataset = whole_data[['day', 'hour', 'temperature', 'wind_speed(m/s)', 'humidity(%)', 'solar_radiation(MJ/m^2)','energy', 'time']]

whole_dataset

data = pd.read_csv('../Data/ProductionData/KMA_July_hourly.csv')

data

dataset = data[['temperature', 'wind_speed(m/s)', 'humidity(%)', 'solar_radiation(MJ/m^2)','energy']]

# get the datetime object for index
days = []
for i in range(len(dataset)):
    days.append(datetime.strptime(data['time'][i], '%Y.%m.%d %H:%M'))

hours = []
for day in days:
    hours.append(day.hour)

# add hour column
dataset['hour'] = hours

last = datetime(2018,12,31)

# add day column
dataset['day'] = day_of_years
cols = dataset.columns.to_list()
cols = cols[-2:] + cols[:-2]
dataset = dataset[cols]

# calculate the day of year using last
day_of_years = []
for day in days:
    doy = day - last
    doy = doy.days
    if(doy > 365):
        doy -= 365
    day_of_years.append(doy)

cols = dataset.columns.tolist()

cols

# +
cols.remove('day')
cols.remove('hour')


cols.insert(0, 'day')
cols.insert(1, 'hour')
# -

dataset = dataset[cols]

whole_dataset

dataset

dates = list()

for i in range(len(dataset)):
    temp_date = datetime.strptime(whole_dataset['time'][i], '%Y.%m.%d %H:%M')
    
    dates.append(datetime.strftime(temp_date, '%Y-%m-%d %H:%M'))


dates

whole_dataset['time'] = dates

whole_dataset

whole_dataset.to_csv('../Data/ProductionData/Weather-Energy_rev_time_hourly_data.csv')

dataset

data_including_July = pd.concat([whole_dataset, dataset])

data_including_July

data_including_July = pd.read_csv('../Data/ProductionData/Weather-Energy_rev_time_hourly_data.csv')

data_including_July = data_including_July.reset_index()

data_including_July.to_csv('../Data/ProductionData/Weather-Energy_rev_time_hourly_data.csv')


