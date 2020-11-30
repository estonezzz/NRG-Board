<<<<<<< HEAD
from flask import Flask, render_template, request, redirect, jsonify, url_for
import os
from celery import Celery
import redis
import gc

import re
=======
from flask import Flask, render_template, request, redirect
from datetime import datetime, timedelta
import numpy as np
>>>>>>> parent of a51c5e8... celery
import requests
import time
from datetime import datetime, timedelta

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
import numpy as np
import scipy
from pmdarima import arima

from bokeh.plotting import figure, ColumnDataSource, output_file, show
from bokeh.embed import components
from bokeh.layouts import gridplot
from bokeh.models import DatetimeTickFormatter

app = Flask(__name__)
app.vars = {}

<<<<<<< HEAD
def make_celery(app):
    celery = Celery(
        app.import_name,
        backend=app.config['CELERY_RESULT_BACKEND'],
        broker=app.config['CELERY_BROKER_URL']
    )
    #celery.conf.update(app.config)

    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery
    
app.config.update(
    CELERY_BROKER_URL = os.environ.get('REDIS_URL','redis://localhost:6379/0'),
    CELERY_RESULT_BACKEND = os.environ.get('REDIS_URL','redis://localhost:6379/0')
    #CELERY_BROKER_URL = os.environ['REDIS_URL'],
    #CELERY_RESULT_BACKEND = os.environ['REDIS_URL']
)
celery = make_celery(app)


=======
>>>>>>> parent of a51c5e8... celery
def lat_long(Name):
    locations = pd.read_csv(r"static/locations.csv")
    lat = locations.loc[locations[locations['Acronym'] == Name].index,'Lat'].values[0]
    long = locations.loc[locations[locations['Acronym'] == Name].index,'Long'].values[0]
    return lat, long
    
def station(Name):
    locations = pd.read_csv(r"static/locations.csv")
    station = locations.loc[locations[locations['Acronym'] == Name].index,'Stations'].values[0]
    return station
    
# get weather history data from noaa.gov, based on station IDs. Station IDs being assigned in Load_Save_CSV.ipynb notebook and saved in EIA930_Reference_Tables.csv file
#function returns dataframe indexed by Datetime and 4 columns: TMAX, TMIN, SNOW, PRCP
def grab_weather(station):
    today = datetime.today().strftime ('%Y-%m-%d')
    url = 'https://www.ncei.noaa.gov/access/services/data/v1?dataset=daily-summaries&dataTypes=TMIN,TMAX,SNOW,PRCP&stations='+ station +'&startDate=2015-07-01&endDate='+today+'&format=json'
    r = requests.get(url)    
    df = pd.DataFrame(r.json()).drop('STATION', axis=1)
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.set_index('DATE')
    df[['SNOW','PRCP']] = df[['SNOW','PRCP']].fillna('0')
    df[['TMIN','TMAX']] = df[['TMIN','TMAX']].fillna(method='pad')
    df = df.astype(int)
    return df

#get previous 2 days of weather from openweather service bc noaa.gov doesn't have this data    
def weather_days(lat, long, days_ = 2):
    days,TMIN,TMAX,SNOW,PRCP = [],[],[],[],[]
    today = datetime.today().strftime ('%Y-%m-%d')
    API_key = '32fbc02006fd2d5d57552e2c0691c1b8'
    time_ = datetime.now()
    for i in range(days_):
        timestamp = str(int(time.time() - timedelta(days = days_, hours = time_.hour - 12, minutes = time_.minute).total_seconds()) + i * 86400)
        url = 'http://api.openweathermap.org/data/2.5/onecall/timemachine?lat='+str(lat)+'&lon='+str(long)+'&dt='+timestamp+'&appid='+API_key
        r = requests.get(url)
        days.append(pd.to_datetime((datetime.today()-timedelta(days = days_ - i)).date()))
        TMIN.append(r.json()['current']['temp'])
        TMAX.append(r.json()['current']['temp'])
        if str.lower(r.json()['current']['weather'][0]['main']) =='rain':
            PRCP.append(r.json()['current']['weather'][0]['rain']/254)
            SNOW.append(0.0)
        elif str.lower(r.json()['current']['weather'][0]['main']) =='snow': 
            SNOW.append(r.json()['daily'][0]['snow']/254)
            PRCP.append(r.json()['daily'][0]['snow']/254)
        else:
            SNOW.append(0.)
            PRCP.append(0.)
    df = pd.DataFrame({'Date': days, 'TMAX': TMAX,'TMIN': TMIN,'PRCP': PRCP,'SNOW': SNOW}).set_index('Date')
    return df
    
# get data from EIA database; API key is needed; codes for data can be checked on EIA website https://www.eia.gov/opendata/qb.php?category=2122628  
#function returns dataframe with data requested indexed by Datetime
def grab_EIA_data(Name,data='ALL.D.HL'):
    API_key = '7abb604f0b0ec946e5de1f4dd66fbe6a'
    url = 'http://api.eia.gov/series/?api_key=' + API_key + '&series_id=EBA.'+Name+'-'+data
    r = requests.get(url)
    '''data_df = pd.DataFrame(r.json()['series'][0]['data'],columns = ['Date','Demand'])
    data_df.Date = pd.to_datetime( data_df.Date.str.split('-').apply(lambda x: x[0].replace('T',' ')+':00'))
    data_df = data_df.set_index('Date')'''
    return r.json()
    
def json_to_df(json,column_name = 'Demand'):
    data_df = pd.DataFrame(json['series'][0]['data'],columns = ['Date',column_name])
    data_df.Date = pd.to_datetime( data_df.Date.str.split('-').apply(lambda x: x[0].replace('T',' ')+':00'))
    data_df = data_df.set_index('Date')
    return data_df
    

def weather_forecast(Name, days = 3): # forecast for today, tomorrow and day after tomorrow
    lat, long = lat_long(Name)
    dates,TMIN,TMAX,SNOW,PRCP = [],[],[],[],[] 
    API_key = '32fbc02006fd2d5d57552e2c0691c1b8'
    url = 'https://api.openweathermap.org/data/2.5/onecall?lat=' + str(lat) + '&lon='+ str(long) + '&exclude=hourly,minutely&appid='+ API_key
    r = requests.get(url)
    TMIN = [r.json()['daily'][i]['temp']['min'] for i in range(days)]
    TMAX = [r.json()['daily'][i]['temp']['max'] for i in range(days)]
    for i in range (days):
        dates.append(pd.to_datetime((datetime.today()+timedelta(days = i)).date()))
        if str.lower(r.json()['daily'][i]['weather'][0]['main']) =='rain':
            PRCP.append(r.json()['daily'][i]['rain'])
            SNOW.append(0.0)
        elif str.lower(r.json()['daily'][i]['weather'][0]['main']) =='snow': 
            SNOW.append(r.json()['daily'][i]['snow'])
            PRCP.append(r.json()['daily'][i]['snow'])
        else:
            SNOW.append(0.0)
            PRCP.append(0.0)
    df = pd.DataFrame({'Date': dates, 'TMAX': TMAX,'TMIN': TMIN,'PRCP': PRCP,'SNOW': SNOW}).set_index('Date')
    return df
    
<<<<<<< HEAD
def weather_transform(weather):
    #weather = weather_.copy()
    # temp is in 1/10 degC; convert to degC
=======
def weather_transform(weather_):
    weather = weather_.copy()
>>>>>>> parent of a51c5e8... celery
    weather['TMIN'] /= 10
    weather['TMAX'] /= 10
    weather['TAVG'] = (weather['TMAX'] + weather['TMIN']) / 2
    weather['TAVG^2'] = weather['TAVG']**2
    # precip is in 1/10 mm; convert to inches
    weather['PRCP'] /= 254
    weather['SNOW'] /= 254
    weather['dry day'] = (weather['PRCP'] == 0).astype(int)
    weather['snowfall'] = (weather['SNOW'] != 0).astype(int)
    weather = weather.drop(['TMIN','TMAX','SNOW'], axis = 1)
    return weather
    
def fourier(period, X, full_data_index_0):
    Y = X.copy()
    dt = (Y.index - full_data_index_0).days * 2 * np.pi / period
    Y.loc[:,'sin_T_'+str(365//period)] = np.sin(dt)
    Y.loc[:,'cos_T_'+str(365//period)] = np.cos(dt)
    return Y
    
def day_of_week(df_daily):
    #df_daily = df.copy()
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    for i, day in enumerate(days):
        df_daily[day] = (df_daily.index.dayofweek == i).astype(float)
    return df_daily
    
<<<<<<< HEAD
def rolling_year(df_daily,full_data_index_0):
    #df_daily = df.copy()
    df_daily['annual'] = (df_daily.index - full_data_index_0).days / 365
=======
def rolling_year(df):
    df_daily = df.copy()
    df_daily['annual'] = (df_daily.index - df_daily.index[0]).days / 365
>>>>>>> parent of a51c5e8... celery
    return df_daily
    
def holidays(df_daily):
    #df_daily = df.copy()
    current_year = datetime.today().year
    cal = USFederalHolidayCalendar()
<<<<<<< HEAD
    holidays = cal.holidays('2015', str(current_year+1))
    df_daily = df_daily.join(pd.Series(1, index=holidays, name='holiday'))
    df_daily['holiday'].fillna(0, inplace=True)
=======
    holidays = cal.holidays('2015', str(current_year))
    df_daily_ = df_daily.join(pd.Series(1, index=holidays, name='holiday'))
    df_daily_['holiday'].fillna(0, inplace=True)
>>>>>>> parent of a51c5e8... celery
    return df_daily
    
    
def hours_of_daylight(date,latitude, axis=23.44):
    """Compute the hours of daylight for the given date"""
    days = (date - pd.to_datetime('2000-12-21')).days
    m = (1. - np.tan(np.radians(latitude))* np.tan(np.radians(axis) * np.cos(days * 2 * np.pi / 365.25)))
    return 24. * np.degrees(np.arccos(1 - np.clip(m, 0, 2))) / 180.

def daylight(df_daily,Name):
    #df_daily = df.copy()
    latitude = lat_long(Name)[0]
    df_daily['daylight_hrs'] = list(map(hours_of_daylight, df_daily.index,[latitude]*len(df_daily.index)))
    return df_daily

def group_by_time(df_hourly,column1 = 'Demand',column2 = 'Net_Generation'):
    weekend = np.where(df_hourly.index.weekday < 5, 'Weekday', 'Weekend')
    return df_hourly.groupby([weekend, df_hourly.index.time]).mean()[[column1,column2]]
    
def var_composition(df,Name, full_data_index_0):
    #df = df_.copy()
    df = daylight(df,Name)
    df = holidays(df)
    df = rolling_year(df)
    df = day_of_week(df)
    df = fourier(365, df,full_data_index_0 )
    return df
    
<<<<<<< HEAD
def simple_imputation(df, column_name='Demand', shift = 7):
    #df = df_.copy()
    median = df[column_name].median() # median help to remove big positive outliers, it is not skewed by them
    df.loc[df[df[column_name] < 0].index,column_name] = 0 # remove negative values   
    df.loc[df[df[column_name] > 2*median].index,column_name] = 0
    avg = df[column_name].mean() 
    std = df[column_name].std()
    '''All outliers being imputed with 7-days-ago values, that way it is not significantly distorted '''
    df.loc[df[(df[column_name] < avg - 3*std)|(df[column_name] > avg + 3*std)].index,column_name]= df.shift(shift).loc[df[(df[column_name] < avg - 3*std)|(df[column_name] > avg + 3*std)].index,column_name]
    return df
  
def model_preprocess(Name,df_daily_):
    lat, long = lat_long(Name)   
    df_daily = simple_imputation(df_daily_) # simple imputation on 'Demand column by default'
    weather = weather_transform(grab_weather(station(Name)))# weather taken from gov and transformed  
    
    weather_add = weather_transform(weather_days(lat, long))# weather from Openweather data 2 days back by default and up to 5 days
    
    ind = weather.index[-1] + timedelta(days=1)
    weather_tr = pd.concat([weather,weather_add.loc[ind:,:]])# 
    df_daily = df_daily.join(weather_tr, how = 'left')
    df_daily = df_daily.fillna(method = 'pad')
    if df_daily.index[-1].date() == datetime.today().date():
        df_daily = df_daily.drop(df_daily.index[-1],axis = 0)# if last row is todays date -> drop it
    df_daily = var_composition(df_daily,Name, df_daily.index[0])
    return df_daily

@celery.task(bind=True,name='model_arima')
def Model(self,Name):
    model_return = {}
    lat, long = lat_long(Name)
    
    self.update_state(state='PROGRESS',
                          meta={'current': 0, 'total': 6,
                                'status': 'Downloading and preprocessing Demand data'})
    model_return['df_hourly_d'] = grab_EIA_data(Name,data='ALL.D.HL')
 
    self.update_state(state='PROGRESS',
                          meta={'current': 1, 'total': 6,
                                'status': 'Downloading and preprocessing Net Generation data'})
    model_return['df_hourly_s'] = grab_EIA_data(Name,data='ALL.NG.HL')
    self.update_state(state='PROGRESS',
                          meta={'current': 2, 'total': 6,
                                'status': 'Downloading and preprocessing Demand Forecast data'})
    model_return['df_hourly_df'] = grab_EIA_data(Name,data='ALL.DF.HL')
    self.update_state(state='PROGRESS',
                          meta={'current': 3, 'total': 6,
                                'status': 'Building predictive Model'})
    EIA_data = model_preprocess(Name,json_to_df(model_return['df_hourly_d']).resample("D").sum()) 
    self.update_state(state='PROGRESS',
                          meta={'current': 4, 'total': 6,
                                'status': 'Fitting Model'})
    model = arima.ARIMA((5,1,1), seasonal=False, suppress_warnings=True) 
    model.fit(EIA_data['Demand'], EIA_data.drop('Demand',axis = 1))
    self.update_state(state='PROGRESS',
                          meta={'current': 5, 'total': 6,
                                'status': 'Calculating predictions'})
    X_ex = var_composition(weather_transform(weather_forecast(Name)),Name,EIA_data.index[0])
    model_return['predictions'] = list(model.predict(n_periods=3, exogenous = X_ex, return_conf_int=False))
    self.update_state(state='PROGRESS',
                          meta={'current': 6, 'total': 6,
                                'status': 'Predictions calculated'})
    gc.collect()                       
    return {'current': 100, 'total': 100, 'status': '', 'result': model_return}

def plotting_datetime(df,title,h=500,w=800,tools='pan,wheel_zoom,save,reset'):
    
    p = figure(x_axis_type="datetime", title=title, plot_height=h, plot_width=w, sizing_mode='scale_both', tools = tools, background_fill_color="#fafafa")
    p.xgrid.grid_line_color=None
    p.ygrid.grid_line_alpha=0.5
    p.yaxis.axis_label = "MegaWatthours"

    pl_data = ColumnDataSource(data=dict(index = [], Demand = [], Net_Generation = [], Date = []))
    pl_data.data = pl_data.from_df(df)
    p.line(pl_data.data['Date'], pl_data.data['Demand'], line_width=2, alpha=0.8,legend = 'Demand')
    p.line(pl_data.data['Date'], pl_data.data['Net_Generation'], line_color = 'red',line_width=2, alpha=0.8,legend = 'Net_Generation')
    p.legend.location = "bottom_left"
    p.yaxis.formatter.use_scientific = False
    return p


def plotting_an(df,df_daily,title1,title2,title3,h=500,w=800,tools='pan,wheel_zoom,save,reset'):
    by_weekday = df_daily.groupby(df_daily.index.dayofweek).mean()[['Demand','Net_Generation']]
    weekdays = ['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun']
    by_weekday.index = weekdays
       
    p1 = figure(x_axis_type="datetime", title=title1,sizing_mode='scale_both',plot_width=w, plot_height=h,   tools = tools,background_fill_color="#fafafa" )
    p1.xgrid.grid_line_color=None
    p1.ygrid.grid_line_alpha=0.5
    p1.yaxis.axis_label = "MegaWatthours"
    p2 = figure(x_axis_type="datetime", title=title2,sizing_mode='scale_both', plot_width=w, plot_height=h,  tools = tools,background_fill_color="#fafafa" )
    p2.xgrid.grid_line_color=None
    p2.ygrid.grid_line_alpha=0.5
    p2.yaxis.axis_label = "MegaWatthours"
    p3 = figure(x_range=weekdays, title=title3, plot_height=h, plot_width=w, sizing_mode='scale_both', tools = tools, background_fill_color="#fafafa")
    p3.xgrid.grid_line_color=None
    p3.ygrid.grid_line_alpha=0.5
    p3.yaxis.axis_label = "MegaWatthours"
   
    pl_data1 = ColumnDataSource(data=dict(index = [], Demand = [], Net_Generation = [], Date = []))
    pl_data1.data = pl_data1.from_df(df.loc[('Weekday',slice(None))])
    pl_data2 = ColumnDataSource(data=dict(index = [], Demand = [], Net_Generation = [], Date = []))
    pl_data2.data = pl_data2.from_df(df.loc[('Weekend',slice(None))])
    
    p1.line(pl_data1.data['index'], pl_data1.data['Demand'], line_width=2, alpha=0.8,legend = 'Demand',line_dash = 'dotted')
    p1.line(pl_data1.data['index'], pl_data1.data['Net_Generation'], line_color = 'red',line_width=2, alpha=0.8,legend = 'Net_Generation', line_dash = 'dotdash')
    
    p2.line(pl_data2.data['index'], pl_data2.data['Demand'], line_width=2, alpha=0.8,legend = 'Demand',line_dash = 'dotted')
    p2.line(pl_data2.data['index'], pl_data2.data['Net_Generation'], line_color = 'red',line_width=2, alpha=0.8,legend = 'Net_Generation', line_dash = 'dotdash')
    p1.legend.location = "top_left"
    p2.legend.location = "top_left"
    pl_data3 = ColumnDataSource(data=dict(index = [], Demand = [], Net_Generation = [], Date = []))
    pl_data3.data = pl_data3.from_df(by_weekday)
    p3.line(weekdays, pl_data3.data['Demand'], line_width=2, alpha=0.8,legend = 'Demand',line_dash = 'dotted')
    p3.line(weekdays, pl_data3.data['Net_Generation'], line_color = 'red',line_width=2, alpha=0.8,legend = 'Net_Generation', line_dash = 'dotdash')
    p3.legend.location = "bottom_left"
    
    p1.yaxis.formatter.use_scientific = False
    p2.yaxis.formatter.use_scientific = False
    p3.yaxis.formatter.use_scientific = False
    
    p1.xaxis.formatter = DatetimeTickFormatter(hours = ['%H:%M'],days = ['%H:%M'])
    p2.xaxis.formatter = DatetimeTickFormatter(hours = ['%H:%M'],days = ['%H:%M'])
    
    return gridplot([[p3, None], [p1, p2]], sizing_mode='scale_both',plot_width=800, plot_height=500)

def is_forecast(df_hourly,df):
    if df_hourly.index[0].date() == (datetime.today() + timedelta (days = 1)).date():
        return int(df['Demand_Forecast'][-2])
    else:
        return "No Authority-based forecast available"
    
def rolling_z(x, window):
    roll_mean = x.rolling(window=window).mean().shift(1) # Don't include current data
    roll_std = x.rolling(window=window).std().shift(1)
    return (x - roll_mean) / roll_std    
    
def z_score(residuals,window):
    roll_mean = residuals[-window:].mean()
    roll_std = residuals[-window:].std()
    return roll_mean, roll_std
    
def make_hist(title, hist, edges, x, pdf):
    p = figure(title=title, tools='pan,wheel_zoom,save,reset', background_fill_color="#fafafa",x_range=[-5,5])
    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
           fill_color="navy", line_color="white", alpha=0.5)
    p.line(x, pdf, line_color="#ff8888", line_width=4, alpha=0.7, legend_label="PDF")
    #p.line(x, cdf, line_color="orange", line_width=2, alpha=0.7, legend_label="CDF")

    p.y_range.start = 0
    p.legend.location = "center_right"
    p.legend.background_fill_color = "#fefefe"
    p.xaxis.axis_label = 'x'
    p.yaxis.axis_label = 'Pr(x)'
    p.grid.grid_line_color="white"
    return p
=======
def model_preprocess(Name):
    lat, long = lat_long(Name)
    df_daily = grab_EIA_data(Name,data='ALL.D.HL').resample("D").sum()# Demand data from EIA summed up by day
    weather = weather_transform(grab_weather(station(Name)))# weather taken from gov and transformed
    weather_add = weather_transform(weather_days(lat, long, days_ = 2))# weather from Openweather data 2 days back by defqult and up to 4 days
    weather_tr = pd.concat([weather,weather_add])# 
    df_daily_j = df_daily.join(weather_tr, how = 'left')
    if df_daily_j.index[-1].date() == datetime.today().date():
        df_daily_j = df_daily_j.drop(df_daily_j.index[-1],axis = 0)# if last row is todays date -> drop it
    df_daily_j = var_composition(df_daily_j,Name, df_daily_j.index[0])
    return df_daily_j
    
    
>>>>>>> parent of a51c5e8... celery
    
@app.route('/', methods=['GET','POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html', message = "")
    else:
        return redirect(url_for('index'))

@app.route('/modeling', methods=['POST'])
def modeling():    
    app.vars['regions'] = request.form['regions'].strip()
<<<<<<< HEAD
    app.vars['authority'] = request.form['authority'].split('-')[0].strip()
    app.vars['flag'] = request.form['flag'].strip()
    if app.vars['flag'] == 'true':
        task = Model.apply_async(args=[app.vars['authority']])
    else:
        task = Model.apply_async(args=[app.vars['regions']])
    
    return (jsonify({"Location": url_for("taskstatus", task_id=task.id,_external = True)}),202)
    
    #return jsonify({'Location': url_for('taskstatus',task_id = task.id)})
                                                  
@app.route('/status/<task_id>', methods=['GET'])
def taskstatus(task_id):
    task = Model.AsyncResult(task_id)
    if task.state == 'PENDING':
        # job did not start yet
        response = {
            'state': task.state,
            'current': 0,
            'total': 1,
            'status': 'Waiting for Response...'
        }
    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'current': task.info.get('current', 0),
            'total': task.info.get('total', 1),
            'status': task.info.get('status', '')
        }
        if 'result' in task.info:
            response['result'] = task.info['result']
    else:
        # something went wrong in the background job
        response = {
            'state': task.state,
            'current': 1,
            'total': 1,
            'status': str(task.info),  # this is the exception raised
        }
    return jsonify(response)

@app.route('/chart',methods=['GET','POST'])
def chart():
    content = request.get_json()
    app.vars['predictions'] = content['predictions']
    app.vars['df_hourly_d'] = simple_imputation(json_to_df(content['df_hourly_d']),shift = 168)
    app.vars['df_hourly_s'] = simple_imputation(json_to_df(content['df_hourly_s'],column_name = 'Net_Generation'),column_name='Net_Generation',shift = 168)
    app.vars['df_hourly_df'] = simple_imputation(json_to_df(content['df_hourly_df'],column_name = 'Demand_Forecast'),column_name='Demand_Forecast',shift = 168)
    auth_name = re.findall('(?<=EBA\.)(\w+)(?=-)',content['df_hourly_d']['request']['series_id'])[0]
    
    app.vars['predictions'] = [str((datetime.today()+timedelta(days=i)).date()) + " " + str(int(x)) for i,x in enumerate(app.vars['predictions'])]
    
    df_hourly = app.vars['df_hourly_d'].join(app.vars['df_hourly_s']).join(app.vars['df_hourly_df']).dropna()
    demand_f = is_forecast(app.vars['df_hourly_df'],df_hourly.resample('D').sum())# forecasted demand from authority
    by_time = group_by_time(df_hourly)
    
    df_daily = simple_imputation(app.vars['df_hourly_d'].resample('D').sum()).join(simple_imputation(app.vars['df_hourly_s'].resample('D').sum(),column_name='Net_Generation')).dropna()[:-1]
    
    plot = plotting_datetime(df_daily, f"{auth_name} Electricity Demand VS Net Generation",h=200,w=800)
    script, div = components(plot)
    plot1 = plotting_an(by_time,df_daily,"WEEKDAYS: Electricity Demand VS Net Generation","WEEKENDS: Electricity Demand VS Net Generation",f"{auth_name} Electricity Demand VS Net Generation by days of week",h=300,w=400) 
    script1, div1 = components(plot1)
       
    window = 50 
    mu, sigma = 0, 0.8
    df_ddf = df_hourly.resample('D').sum()
    residuals = df_ddf['Demand_Forecast'] - df_ddf['Demand']
    residuals_norm = rolling_z(residuals,window).dropna()
    
    if isinstance(demand_f,int):
        z_score_v = round((demand_f - int(content['predictions'][0]) - z_score(residuals,window)[0])/z_score(residuals,window)[1],2)
        if z_score_v > 2.5:
            comment = "Z-Score is too high, possible overproduction"
        elif z_score_v <= 2.5 and z_score_v >= -2.5:
            comment = "Normal Operations"
        else:
            comment = "Z-Score is too low, possible blackouts"
    else:
        z_score_v = "NA"
        comment = "No Authority-based forecast available"
        
    hist, edges = np.histogram(residuals_norm, density=True, bins=100)

    x = np.linspace(-5, 5, 1000)
    pdf = 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-(x-mu)**2 / (2*sigma**2))
    #cdf = (1+scipy.special.erf((x-mu)/np.sqrt(2*sigma**2)))/2
    histogram = make_hist('Historical Distribution of Residuals of Demand Forecast and Actual Demand',hist, edges, x, pdf)
    script2, div2 = components(histogram)
    return render_template('chart.html', comment = comment,z_score = z_score_v, demand = demand_f, vars_ = app.vars['predictions'], the_div=div, the_script=script,the_div1=div1, the_script1=script1,the_div2=div2, the_script2=script2)  

=======
    app.vars['authority'] = request.form['authority'].split('-')[0].strip() 
    EIA_data = model_preprocess(app.vars['authority']) 
    model = arima.ARIMA((5,1,1), seasonal=False, suppress_warnings=True)    
    model.fit(EIA_data['Demand'], EIA_data.drop('Demand',axis = 1))
    X_ex = var_composition(weather_transform(weather_forecast(app.vars['authority'])),app.vars['authority'],EIA_data.index[0])
    predictions = model.predict(n_periods=3, exogenous = X_ex, return_conf_int=False)
    return render_template('chart.html', vars_ = predictions)
>>>>>>> parent of a51c5e8... celery
    
@app.route('/about')
def about():
  return render_template('about.html')
  
if __name__ == '__main__':
  app.run(port=33507,debug = True)
