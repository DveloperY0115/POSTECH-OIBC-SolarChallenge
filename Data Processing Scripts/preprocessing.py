import pandas as pd
import datetime
from dateutil import tz

PROJECT_PATH = '../../../'
WEATHER_DATA_PATH = 'Data/WeatherData/'
PRODUCTION_DATA_PATH = 'Data/ProductionData/'

# Load all the KMA files and make it into one dataframe
KMA_2019 = pd.read_csv(PROJECT_PATH + WEATHER_DATA_PATH + '2019/KMA_2019_07.csv', encoding='CP949')
KMA_2020 = pd.read_csv(PROJECT_PATH + WEATHER_DATA_PATH + '2020/KMA_2020_01.csv', encoding='CP949')

for i in range(8,13):
       if(i<10):
              temp_KMA = pd.read_csv(PROJECT_PATH + WEATHER_DATA_PATH + '2019/KMA_2019_0'+ str(i) +'.csv', encoding='CP949')
       else:
              temp_KMA = pd.read_csv(PROJECT_PATH + WEATHER_DATA_PATH + '2019/KMA_2019_' + str(i) + '.csv',encoding='CP949')
       KMA_2019 = pd.concat([KMA_2019,temp_KMA])

for i in range(2,7):
    temp_KMA = pd.read_csv(PROJECT_PATH + WEATHER_DATA_PATH + '2020/KMA_2020_0'+ str(i) +'.csv', encoding='CP949')
    KMA_2020 = pd.concat([KMA_2020,temp_KMA])

KMA_year = pd.concat([KMA_2019, KMA_2020])

# Add a row in Submission file
# Change the utc time format of "SolarPV_Elec_Problem.csv" and make a new file called "local_SolarPV.csv"
def utc_to_local():
       SolarPV = pd.read_csv(PROJECT_PATH + PRODUCTION_DATA_PATH + 'SolarPV_Elec_Problem.csv', encoding='CP949')
       # fill nan values to -1
       # print(SolarPV.fillna(-1))
       for index, row in SolarPV.iterrows():
              utc_time = row['time'].split('+')[0]
              date_time_obj = datetime.datetime.strptime(utc_time, '%Y-%m-%dT%H:%M:%S')
              # datetime objects are naive as default: have to change the timezone
              utc_time = date_time_obj.replace(tzinfo=tz.tzutc())
              local_time = utc_time.replace(tzinfo=tz.tzlocal())
              # datetime object to string like KMA data
              SolarPV.loc[index, 'time'] = local_time.strftime("%Y-%m-%d %I:%M:%S %p")
       SolarPV.to_csv(PROJECT_PATH + PRODUCTION_DATA_PATH + 'local_SolarPV.csv')


# Concatenate KMA_year data and local_SolarPV and make a new file called "Weather-Energy_data.csv"
def normalize_KMA_overwrite_EnergyData():
    local_SolarPV = pd.read_csv(PROJECT_PATH + PRODUCTION_DATA_PATH + 'local_SolarPV.csv', encoding='CP949')

    for col in list(KMA_year.columns):
        if (col == '일시' or col == '지점'): continue
        local_SolarPV[col] = 0

    # re-ordering
    cols = KMA_year.columns.tolist()
    cols.remove('지점')
    cols.remove('일시')
    cols.insert(0, 'time')
    cols.append('energy')
    local_SolarPV = local_SolarPV[cols]

    for index, row in local_SolarPV.iterrows():
        if index == 0:
            continue

        if index % 1000 == 0:
            print("%i-th data processed just now" % index)

        local_time = row['time']
        date_time_obj = datetime.datetime.strptime(local_time, '%Y-%m-%d %I:%M:%S %p')
        start_time = date_time_obj - datetime.timedelta(minutes=15)
        temp_list = []
        for i in range(0, 15):
            str_time = (start_time + datetime.timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M")
            temp_list.append(str_time)
        temp_df = KMA_year[KMA_year['일시'].isin(temp_list)]

        cols = list(temp_df.columns)
        cols.remove('지점')
        cols.remove('일시')

        for col in cols:
            local_SolarPV.loc[index, col] = temp_df.mean()[col]

    local_SolarPV.to_csv(PROJECT_PATH + PRODUCTION_DATA_PATH + 'Weather-Energy_data.csv')
