#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 15:52:51 2017

@author: maverick
"""

import h5py
import numpy as np
import tables
import pandas as pd
import os,sys, time
from datetime import datetime, timedelta, date
from dateutil import parser
from joblib import Parallel, delayed
import multiprocessing

#PVLIB Stuff
import itertools
import matplotlib.pyplot as plt
import pvlib
from pvlib import clearsky, atmosphere
from pvlib.location import Location




def create_timestamp(df):
    val = datetime.combine(df.date, df.time)
    return val

    

    
# method to loop over a date range and yields a set of dates
def date_range(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        start = start_date
        end = start_date + timedelta(1)
        start_date = end
        yield {'start':start.strftime("%Y-%m-%d"), 'end': end.strftime("%Y-%m-%d") }


# retreives clearsky irradiance timeseries data for the given period        
#frequency = 1D, 1min, 5s, 5000ms    
#Timezones - CET, Europe/Zurich,    
def get_GHI_data(period):
    print period["start"] +","+ period["end"]
#    tus = Location(43.5354, 11.4814, 'CET', 308, 'Cavriglia')
    tus = Location(47.1642, 6.9903, 'CET', 1278, 'Mont_Soleil')
    times = pd.DatetimeIndex(start= period["start"], end= period["end"], freq='1s', tz=tus.tz)
    cs = tus.get_clearsky(times)  # ineichen with climatology table by default
    return cs
               

# to retrieve GHI data in parallel and concat them to form a larger data frame 
def build_irradiance_dataframe(start_date, end_date): #eg: format  'yyyy-mm-dd'

    start_date = parser.parse(start_date)
    end_date = parser.parse(end_date)
    
    ghi_list =Parallel(n_jobs=6)(delayed(get_GHI_data)(period) for period in date_range(start_date, end_date))
    ghi_data = pd.concat(ghi_list) #merges individual df's into one
    ghi_data = ghi_data.sort_index()
    
    return ghi_data

    

    
    
if __name__=="__main__":
    
    ghi_series_data = build_irradiance_dataframe('2015-07-01', '2016-06-01')
    ghi_series_data.to_hdf('/home/pdinesh/ghi_MS.h5','df',mode='w',format='table',data_columns=True)
#    ghi_series_data.to_hdf('/home/maverick/Desktop/ghi_cavriglia.h5','df',mode='w',format='table',data_columns=True)