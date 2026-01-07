# Libraries
import pandas as pd                     # data handling
import numpy as np                      # numerical computations
import networkx as nx                   # graph handling
import matplotlib.pyplot as plt         # plotting
import geopandas as gpd                 # geospatial data handling
import  datetime
from Graph_class import Graph


def build_adjacency_matrix(df,start_end_months,weekdays,t_begin,
                          duration,mat_selecter=2):
    """
        Build a 263x263 adjacency matrix the time frame given as arguments.

        Parameters:
            df: pandas dataframe
                the dataframe that contains all the taxi trips
            start_end_months: list[int]
                a list of two ints from 1-12 corresponding to the months of the year
            weekdays: np.array[int]
                a list of up to 7 ints from 0-6 corresponding to the weekdays
            t_begin: dt.time
                the start of the time windows
            duration: int
                the duration of the time windows (can overflow to the next day from t_begin)
            mat_selecter: int
                select the type of symmetrizing the adjacency matrix: 1 (A+A.T), 2 (mirror lower half onto upper half), 3 (mirror upper half onto lower half)



    
    """        
    location_ids = np.arange(1,264, dtype = float)
    
    start_month=start_end_months[0]
    end_month=start_end_months[1]
    # wrap around if the time window goes beyond one year 

    
    start_of_season=f'2018-{start_month:02d}-01 00:00:00'
    end_of_season=f'2018-{(end_month)%12+1:02d}-01 00:00:00'

    endmicroseconds=t_begin.microsecond+duration.microsecond
    endseconds=t_begin.second+duration.second+endmicroseconds//int(1e6)
    endminutes=t_begin.minute+duration.minute+endseconds//60
    endhour=(t_begin.hour+duration.hour+endminutes//60)
    t_end = t_begin.replace(hour=endhour%24).replace(minute=endminutes%60).replace(second=endseconds%60).replace(microsecond=endmicroseconds%int(1e6))


    if start_month>end_month:        
        season_mask = (
                        ((df['tpep_pickup_datetime'] >= start_of_season) &
                        (df['tpep_pickup_datetime'] <=  '2019-01-01'))
                        |
                        ((df['tpep_pickup_datetime'] >= '2018-01-01') &
                        (df['tpep_pickup_datetime'] <=  end_of_season))
                    )
    elif end_month==12:
        season_mask = (df['tpep_pickup_datetime'] >= start_of_season) & (df['tpep_pickup_datetime'] <'2019-01-01')
    else:
        season_mask = (df['tpep_pickup_datetime'] >= start_of_season) & (df['tpep_pickup_datetime'] < end_of_season)
    
    if endhour<24:
        day_mask1 = df["tpep_pickup_datetime"].dt.weekday.isin( weekdays)
        day_mask2 = (df["tpep_pickup_datetime"].dt.time >= t_begin)&(df["tpep_pickup_datetime"].dt.time < t_end)
        day_mask=day_mask1&day_mask2

    else:
        day_mask1=((df["tpep_pickup_datetime"].dt.weekday.isin( weekdays)) & (df["tpep_pickup_datetime"].dt.time >= t_begin))
        nextdays=(weekdays+1)%7
        day_mask2=(df["tpep_pickup_datetime"].dt.weekday.isin( nextdays) & (df["tpep_pickup_datetime"].dt.time < t_end))
        day_mask=day_mask1|day_mask2
        
    filtered_df= df.loc[season_mask&day_mask]
            

    group_time_df = filtered_df.groupby(["PULocationID", "DOLocationID"]).sum('passenger_count').unstack(fill_value=0)
    
    group_time_df= group_time_df['passenger_count']
    group_time_df = group_time_df.reindex(index=location_ids, columns=location_ids, fill_value=0)
    
    group_time_matrix=group_time_df.values.astype(int)

    if mat_selecter==1:
        group_matrix_sym = (group_time_matrix + group_time_matrix.T) * 0.5
    elif mat_selecter==2:
        group_matrix_sym = (np.triu(group_time_matrix) +np.triu(group_time_matrix,1).T)

    elif mat_selecter==3:
        group_matrix_sym = (np.tril( group_time_matrix) +np.tril( group_time_matrix,-1).T)
    else:
        raise KeyError("mat_selecter must be within 1-3")
    group_matrix_sym = group_matrix_sym / np.max(group_matrix_sym)
    
    node_values1=group_matrix_sym.sum(axis=1)
    
    return group_matrix_sym,node_values1

def get_specific_data(df,start_date, end_date,mat_selecter):
    """get the specific data from the data frame starting at start_date end ending at end_date.
    Both arguments are strings of the form: "2018-10-20 06:00:00"
    """
    location_ids = np.arange(1,264, dtype = float)
    mask = (df['tpep_pickup_datetime'] >= start_date) & (df['tpep_pickup_datetime'] <= end_date)
    filtered_df = df.loc[mask]
    group_df = filtered_df.groupby(["PULocationID", "DOLocationID"]).sum('passenger_count').unstack(fill_value=0)
    group_df = group_df['passenger_count']
    group_df = group_df.reindex(index=location_ids, columns=location_ids, fill_value=0)
    group_matrix=group_df.values.astype(int)

    if mat_selecter==1:
        group_matrix_sym = (group_matrix + group_matrix.T) * 0.5
    if mat_selecter==2:
        group_matrix_sym = (np.triu(group_matrix) +np.triu(group_matrix,1).T)

    if mat_selecter==3:
        group_matrix_sym = (np.tril( group_matrix) +np.tril( group_matrix,-1).T)
    group_matrix_sym = group_matrix_sym / np.max(group_matrix_sym)

    return group_matrix_sym