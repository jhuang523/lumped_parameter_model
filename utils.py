import pandas as pd
import numpy as np
import datetime 
import importlib
def convert_df_to_date_time(df, date_col):
    df[date_col] = pd.to_datetime(df[date_col])
    return df

def convert_hourly_to_daily(df, date_col, agg = 'median'):
    non_date_cols = list(df.columns)
    non_date_cols.remove(date_col)
    return convert_df_to_date_time(df.groupby(df[date_col].dt.date)[non_date_cols].agg(agg).reset_index(), date_col)

def convert_daily_to_hourly(df, date_col, interp_method = 'constant'):
    #TODO: add other methods for interpolating hourly data
    hourly_timestamps = (df[date_col].repeat(24) + pd.to_timedelta(np.tile(np.arange(24), len(df[date_col])), unit = 'h')).reset_index(drop=True)
    if interp_method == 'constant':
        hourly_data = df.loc[df.index.repeat(24)].reset_index(drop=True)
    elif interp_method == 'uniform':
        hourly_data = df.drop(date_col, axis =1).loc[df.index.repeat(24)].reset_index(drop=True)/24
    hourly_data[date_col] = hourly_timestamps
    return hourly_data
def index_to_date(df, date_col = 'DATE'):
    """sets index to date col"""
    df = convert_df_to_date_time(df, date_col)
    df.index = df[date_col]
    return df.drop(date_col, index = 1)
def add_to_timestamp(t, dt, unit = 'hr'):
    if unit == 'hr':
        return t + datetime.timedelta(hours = dt)  
    if unit == 'day':
        return t + datetime.timedelta(days = dt)
    if unit == 'sec':
        return t + datetime.timedelta(seconds = dt)
    else:
        print("can't parse unit, try again")
        return t
def subtract_timestamps(t1, t2, unit = 'hr'):
    """Take 2 Timestamps and returns the difference as a float"""
    diff = (t1 - t2).total_seconds()
    if unit == 'hr':
        return diff / 3600
    if unit == 'day':
        return diff / 86400 
    else:
        return diff 

def reload_module(module):
    importlib.reload(module)