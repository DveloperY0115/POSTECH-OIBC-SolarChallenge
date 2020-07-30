# -*- coding: utf-8 -*-
# %%
import pandas as pd
import matplotlib.pyplot as plt

# %%
KMA_July = pd.read_csv('../Data/WeatherData/2020/KMA_2020_07.csv', encoding='CP949')

# %%
energy_July = pd.read_csv('../Data/ProductionData/Solar_PV_July.csv')

# %%
energy_July

# %%
hourly_energy = list()

# %%
for i in range(0, len(energy_July), 4):
    hourly_energy.append(energy_July['energy'][i:i+4].sum())

# %%
del hourly_energy[-21:]

# %%
len(hourly_energy)

# %%

# %%
KMA_July

# %%
KMA_July.info()

# %%
KMA_July.fillna('bfill')

# %%
data = KMA_July

# %%
time = data['일시']
time

# %%
cols = KMA_July.columns.tolist()
cols

# %%
cols.remove('지점')
cols.remove('일시')
cols.remove('누적강수량(mm)')
cols.remove('풍향(deg)')
cols.remove('현지기압(hPa)')
cols.remove('해면기압(hPa)')
cols.remove('일조(Sec)')

# %%
data = data[cols]

# %%
data

# %%
time = time.to_numpy()

# %%
total = data.isnull().sum().sort_values(ascending=False)
percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(25)

# %%
new_time = list()
for i in range(60, len(data), 60):
        time_temp = time[i-1]
        new_time.append(time_temp)

# %%
new_time

# %%
len(new_time)

# %%
new_data = {}

for col in cols:
    new_data[col] = list()
new_data['time'] = list()

# %%
new_data

# %%
new_data['time'] = new_time

# %%
temp = list()
wind_speed = list()
humid = list()
solar_rad = list()

# %%
for i in range(0, len(data)-60, 60):
    slice = data.iloc[i:i+60]
    hourly_mean = slice.mean(axis=0)
    hourly_mean_list = hourly_mean.tolist()
    new_data['기온(°C)'].append(hourly_mean_list[0])
    new_data['풍속(m/s)'].append(hourly_mean_list[1])
    new_data['습도(%)'].append(hourly_mean_list[2])
    new_data['일사(MJ/m^2)'].append(slice.iloc[-1, 3])

# %%
keys = new_data.keys()

# %%
for key in keys:
    print(str(key) + ':' + str(len(new_data[key])))
    print('=============='+ str(key) + '=================')
    print('The first five elements of ' + str(key))
    print(new_data[key][:5])
    print()
    print('The last five elements of ' + str(key))
    print(new_data[key][-5:])
    print()

# %%
print(new_time[:5])
print(new_time[-5:])

# %%
new_data['energy'] = hourly_energy

# %%
new_dataset = pd.DataFrame(new_data)

# %%
plt.plot(new_dataset['일사(MJ/m^2)'].to_numpy(), label='일사량(MJ/m^2)')

# %%
for i in range(len(new_dataset)-1, 0, -1):
    diff = new_dataset.iloc[i, 3] - new_dataset.iloc[i-1, 3]
    print(diff)
    if (diff >= 0):
        new_dataset.iloc[i, 3] = diff

# %%
new_dataset['일사(MJ/m^2)'].tail(19)

# %%
new_dataset.to_csv('../Data/WeatherData/KMA_July_hourly.csv')

# %%
test = pd.read_csv('../Data/WeatherData/KMA_July_hourly.csv')

# %%
test

# %%
