from flask import Flask, render_template, request, redirect
import pandas as pd
from bokeh.plotting import figure
import datetime
from bokeh.embed import components
from bokeh.resources import CDN
from bokeh.models import HoverTool,Range1d,LinearAxis
import os
from datetime import date
from forcast import getInput
import pickle
from joblib import dump, load
from classes import Composite,FourierTransformer

app = Flask(__name__)
app.vars = {}

def dateRange():
  today = datetime.datetime.date(datetime.datetime.now())
  start = today.replace(year=2018)
  end = start + datetime.timedelta(days=14)
  return start,end

def readData(start,end):
  region = app.vars['region']
  Real_Time = pd.read_pickle('./realtime.pkl')
  Day_Ahead = pd.read_pickle('./dayahead.pkl')
  Data = pd.DataFrame()
  Data['Real_Time']  = Real_Time[region]['mean'][start:end]
  Data['Day_Ahead'] = Day_Ahead[region]['mean'][start:end]
  Data['Diff'] = Data['Real_Time']-Data['Day_Ahead']
  Data['Date_str'] = Data.index.strftime('%Y-%m-%d')                                                                                                                                                          
  Data = Data.fillna(method='ffill')
  return Data

def plot(Data):
  p=figure(x_axis_type='datetime',title='Average Daily Price for {}'.format(app.vars['region']))
  p.line(x='date',y='Real_Time',source = Data, legend_label='Real Time',line_width=2)
  p.line(x='date',y= 'Day_Ahead',source=Data,line_color='red',legend_label='Day Ahead',line_width=2)
  p.yaxis.axis_label='Average Daily Price'
  p.xaxis.axis_label='Date'
  p.title.text_font_size='14pt'
  p.yaxis.axis_label_text_font_size='14pt'
  p.yaxis.major_label_text_font_size='12pt'
  p.xaxis.axis_label='Date'
  p.xaxis.axis_label_text_font_size='14pt'
  p.xaxis.major_label_text_font_size='12pt'
  p.legend.label_text_font_size='14pt'
  p.add_tools(HoverTool(tooltips=[('Date','@Date_str'),('Real Time','@Real_Time'),('Day Ahead','@Day_Ahead'),('Difference','@Diff')]))
  return p

def predict(Data):
  realTime = load('realtime_model_Fourier.pkl')
  dayAhead = load('dayahead_model_Fourier.pkl')
  realTimePredict = realTime.predict(Data)
  dayAheadPredict = dayAhead.predict(Data)
  Data['Real_Time'] = realTimePredict
  Data['Day_Ahead'] = dayAheadPredict
  Data['Diff'] = Data['Real_Time']-Data['Day_Ahead']
  Data['Date_str'] = Data.index.strftime('%Y-%m-%d')
  return Data

@app.route('/',methods=['GET','POST'])
def index():
  if request.method=='GET':
    return render_template('index.html')
  app.vars['region'] = request.form['region']
  start,end = dateRange()
  Data = getInput(app.vars['region'])
  Data = predict(Data)
  p = plot(Data)
  script,div = components(p)
  return render_template('plot.html',script=script,div=div)

@app.route('/about')
def about():
  return render_template('about.html')

@app.route('/Features')
def features():
  return render_template('features.html')

@app.route('/Data')
def data():
  return render_template('data.html')

if __name__ == '__main__':
  app.run(port=33507)
