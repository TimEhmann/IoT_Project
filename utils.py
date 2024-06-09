import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
from copy import deepcopy

def prepare_data_for_plot(df: pd.DataFrame, clean_data: bool=True) -> pd.DataFrame:
    """
    args:   df: pd.DataFrame
            clean_data: bool

    returns: pd.DataFrame
    """
    df_cpy = deepcopy(df)
    df_cpy['date_time'] = pd.to_datetime(df_cpy['date_time'])
    if clean_data:
        # in room am005, CO2 data starting 2023-07-05 is constantly 10k. Drop All Data >= 10k
        df_cpy = df_cpy[df_cpy['CO2'] < 10000]
        # drop all points where hum > 100
        df_cpy = df_cpy[df_cpy['hum'] <= 100]
        # drop all points where tmp > 100. Is 45°C realistic tho?
        df_cpy = df_cpy[df_cpy['tmp'] <= 100]
        # drop all points where VOC > 4000
        df_cpy = df_cpy[df_cpy['VOC'] <= 4000]
        # I dont know what to do with the vis data. Unclear so far.


    return df_cpy

def plot_figure(df: pd.DataFrame, x_feature: str='date_time', y_feature: str='CO2', x_title: str=None, y_title: str=None, mode: str='lines+markers'):
    """
    args:   df: pd.DataFrame
            x_feature: str
            y_feature: str
            x_title: str
            y_title: str
            mode: str

    returns: go.Figure
    """
    feature_title_dictionary = {
        'date_time': 'Time',
        'tmp': 'Temperature in °C',
        'hum': 'Humidity in %',
        'CO2': 'CO2 in ppm',
        'VOC': 'VOC in ppb',
        'vis': 'Visibility? Maybe Raw Bit Format?',
    }
    x_title = feature_title_dictionary[x_feature] if x_feature in feature_title_dictionary and x_title is not None else x_feature
    y_title = feature_title_dictionary[y_feature] if y_feature in feature_title_dictionary and y_title is not None else y_feature
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[x_feature], y=df[y_feature], mode=mode))
    fig.update_layout(xaxis_title=x_title, yaxis_title=y_title)

    return fig

