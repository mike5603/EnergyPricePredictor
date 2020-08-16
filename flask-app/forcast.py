from retrying import retry
import simplejson as json
import requests
import pandas as pd
import calendar

def dateFormat(date):
    return date[:25]

def avgTemp(df):
    df['date'] = df['validTime'].map(dateFormat)
    df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True).dt.date
    TAVG = df.groupby('date').mean()
    TAVG = TAVG*1.8+32
    return TAVG.squeeze()
def maxTemp(df):
    df['date'] = df['validTime'].map(dateFormat)
    df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True).dt.date
    TMAX = df[['value','date']].groupby('date').max()
    TMAX = TMAX*1.8+32
    return TMAX.squeeze()
def minTemp(df):
    df['date'] = df['validTime'].map(dateFormat)
    df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True).dt.date
    TMIN = df[['value','date']].groupby('date').min()
    TMIN = TMIN*1.8+32
    return TMIN.squeeze()
def avgWind(df):
    df['date'] = df['validTime'].map(dateFormat)
    df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True).dt.date
    AWND = df.groupby('date').mean()
    AWND = AWND*0.621371
    return AWND.squeeze()
def precip(df):
    df['date'] = df['validTime'].map(dateFormat)
    df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True).dt.date
    PRCP = df[['value','date']].groupby('date').sum()    
    PRCP = PRCP*0.0393701
    return PRCP.squeeze()

@retry(stop_max_attempt_number=10,wait_fixed=2000)
def getProperties(i,weather):
    Cities = ['Redding','Sacremento','Fresno','Bakersfield','Los Angeles']
    offices = ['STO','STO','HNX','HNX','LOX']
    gridpoints = ['27,163','40,67','51,100','67,35','154,44']
    DataCatagories = ['AWND','PRCP','TAVG','TMAX','TMIN']
    response = requests.get('https://api.weather.gov/gridpoints/{}/{}'.format(offices[i],gridpoints[i]))
    data = json.loads(response.text)
    weather['{} AWND'.format(Cities[i])] = avgWind(pd.DataFrame(data['properties']['windSpeed']['values']))
    weather['{} PRCP'.format(Cities[i])] = precip(pd.DataFrame(data['properties']['quantitativePrecipitation']['values']))
    weather['{} TAVG'.format(Cities[i])] = avgTemp(pd.DataFrame(data['properties']['temperature']['values']))
    weather['{} TMAX'.format(Cities[i])] = maxTemp(pd.DataFrame(data['properties']['temperature']['values']))
    weather['{} TMIN'.format(Cities[i])] = minTemp(pd.DataFrame(data['properties']['temperature']['values']))
    return weather

def getWeather():
    weather=pd.DataFrame()
    for i in range(5):
        weather = getProperties(i,weather)
    return weather

def MonthToString(month):
    return calendar.month_name[month]

def expandInput(X_predict,region):
    X_predict.index = pd.to_datetime(X_predict.index)
    X_predict['Day of Month'] = X_predict.index.get_level_values('date').day
    X_predict['Month'] = X_predict.index.get_level_values('date').month
    X_predict['Year'] = X_predict.index.get_level_values('date').year
    X_predict['Day of Week'] = X_predict.index.get_level_values('date').weekday
    DOW = {0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday',4:'Friday',5:'Saturday',6:'Sunday'}
    X_predict['Day Of Week String'] = X_predict['Day of Week'].map(DOW)
    X_predict['Month Name'] = X_predict['Month'].map(MonthToString)
    X_predict['region'] = region
    capacity = pd.read_pickle('./Power_Plant_Capacities.pkl')
    X_predict['Hydro Capacity'] = capacity['Hydro Capacity'][2019,12]
    X_predict['Solar Capacity'] = capacity['Solar Capacity'][2019,12]
    X_predict['Gas Capacity'] = capacity['Gas Capacity'][2019,12]
    X_predict['Wind Capacity'] = capacity['Wind Capacity'][2019,12]
    X_predict['Geothermal Capacity'] = capacity['Geothermal Capacity'][2019,12]
    X_predict['Nuclear Capacity'] = capacity['Nuclear Capacity'][2019,12]
    X_predict['Coal Capacity'] = capacity['Coal Capacity'][2019,12]
    X_predict['Biomass Capacity'] = capacity['Biomass Capacity'][2019,12]
    X_predict = X_predict.drop(['Day of Month','Month','Year','Day of Week'],axis=1)
    cols = ['region', 'Day Of Week String', 'Month Name', 'Redding AWND',
       'Redding PRCP', 'Redding TAVG', 'Redding TMAX', 'Redding TMIN',
       'Sacremento AWND', 'Sacremento PRCP', 'Sacremento TAVG',
       'Sacremento TMAX', 'Sacremento TMIN', 'Fresno AWND', 'Fresno PRCP',
       'Fresno TAVG', 'Fresno TMAX', 'Fresno TMIN', 'Bakersfield AWND',
       'Bakersfield PRCP', 'Bakersfield TAVG', 'Bakersfield TMAX',
       'Bakersfield TMIN', 'Los Angeles AWND', 'Los Angeles PRCP',
       'Los Angeles TAVG', 'Los Angeles TMAX', 'Los Angeles TMIN',
       'Hydro Capacity', 'Solar Capacity', 'Gas Capacity', 'Wind Capacity',
       'Geothermal Capacity', 'Nuclear Capacity', 'Coal Capacity',
       'Biomass Capacity']
    X_predict = X_predict[cols]
    return X_predict

def getInput(region):
    weather = getWeather()
    X_predict = expandInput(weather,region)
    return X_predict
