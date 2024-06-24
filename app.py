import streamlit as st
import pandas as pd
import numpy as np
import pyvista as pv
import os
import utils
from stpyvista import stpyvista
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
    )
st.title("Dashboard Building AM")

# Read Data and gather directory information
cwd = os.getcwd()
st.sidebar.header("Dashboard Building AM")
selected_building = st.sidebar.selectbox(label= "Select Building", options= ['am', 'a'])
directory = cwd +f'/hka-aqm-{selected_building}/'
files = [f.removeprefix('._') for f in os.listdir(directory)]
rooms = sorted(set([f.split("_")[0].removeprefix('hka-aqm-') for f in files]))
available_dates_per_room = {room: sorted(set([f.split("_", 1)[1].split(".")[0] for f in files if f.split("_")[0].removeprefix('hka-aqm-') == room])) for room in rooms}

# Sidebar
input_device = st.sidebar.selectbox(label= "Select Room", options= rooms)
available_dates_per_room[input_device].sort()
min_date = datetime.strptime(available_dates_per_room[input_device][0], "%Y_%m_%d")
max_date = datetime.strptime(available_dates_per_room[input_device][-1], "%Y_%m_%d")
input_date = st.sidebar.date_input(label= "Select Date", value= min_date, min_value= min_date, max_value= max_date)

feature_data = st.sidebar.selectbox(label= "Feature", options= ["Temperature", "Humidity", "CO2", "VOC", "Visibility"])
feature_name_to_feature = {"Temperature": "tmp", "Humidity": "hum", "CO2": "CO2", "VOC": "VOC", "Visibility": "vis"}
selected_feature = feature_name_to_feature[feature_data]

aggregation_level_selection = st.sidebar.selectbox(label= "Aggregation Level", options= ["15min", "30min", "60min"])
aggregation_level_selection_to_aggregation_level = {"60min": 'hour', "30min": 'half_hour', "15min": 'quarter_hour'}
frequency = aggregation_level_selection.replace('min', 'T')
aggregation_level = aggregation_level_selection_to_aggregation_level[aggregation_level_selection]

clean_data = st.sidebar.checkbox(label= "Clean Data", value= True)

points_to_forecast = st.sidebar.number_input(label= "Points to Forecast", value= 1, min_value= 1, max_value= 10, step= 1)

models = ["Transformer", "LSTM"]
selected_models = []
with st.sidebar.expander("Selected Models"):
    for model in models:
        if st.checkbox(model):
            selected_models.append(model)
# model_name_to_internal_name = {'Transformer': 'transformer_multivariate', 'LSTM': 'lstm_multivariate'}
# selected_models = [model_name_to_internal_name[model] for model in selected_models]

# Room data for the selected room
df_room = pd.concat([pd.read_csv(directory + f"hka-aqm-{input_device}_{date}.dat", skiprows=1, sep=';', engine='python') for date in available_dates_per_room[input_device]])
df_room = utils.clean_df(df_room, clean_data)

# check if file for the selected date exists
data_exists = os.path.exists(directory + f"hka-aqm-{input_device}_{str(input_date).replace('-', '_')}.dat")
if not data_exists:
    st.markdown("## No data available")
    st.markdown(f"## Overview of available dates for {input_device}")
    st.markdown("### Average CO2 in ppm per available date")
    st.plotly_chart(utils.plot_available_data(df_room), use_container_width=True)
    st.stop()

df_room_date = df_room[df_room['date_time'].dt.strftime("%Y-%m-%d") == input_date.strftime("%Y-%m-%d")]
df_room_date['date_time_rounded'] = df_room_date['date_time'].dt.round(frequency)

c1, c2 = st.columns([1, 2])

with c1:
    container = st.container(border=True)
    with container:
        st.markdown("## Placeholder for building model")


with c2:
    kpi_tmp, kpi_hum, kpi_co2 = st.columns(3)
    kpi_voc, kpi_vis, placeholder = st.columns(3)

    with kpi_tmp:
        container = st.container(border=True)
        with container:
            st.metric("~Temperature", f"{round(df_room_date['tmp'].mean(), 2)} °C")

    with kpi_hum:
        container = st.container(border=True)
        with container:
            st.metric("~Humidity", f"{round(df_room_date['hum'].mean(), 2)} %")

    with kpi_co2:
        container = st.container(border=True)
        with container:
            st.metric("~CO2", f"{int(df_room_date['CO2'].mean())} ppm")
    
    with kpi_voc:
        container = st.container(border=True)
        with container:
            st.metric("~VOC", f"{int(df_room_date['VOC'].mean())} ppb")
    
    with kpi_vis:
        container = st.container(border=True)
        with container:
            st.metric("~Visibility", f"{int(df_room_date['vis'].mean())}")

device = utils.get_device()
window_size = 20
if 'LSTM' in selected_models:
    df_room_date_pred_LSTM = utils.predict_data_multivariate_LSTM(selected_room=input_device, start_time=input_date, y_feature=selected_feature, aggregation_level=aggregation_level, prediction_count=points_to_forecast, window_size=window_size, device=device, clean_data=clean_data)
    df_room_date = pd.merge(df_room_date, df_room_date_pred_LSTM, on='date_time_rounded', how='left', suffixes=('', '_pred_LSTM'))
if 'Transformer' in selected_models:
    df_room_date_pred_Transformer = utils.predict_data_multivariate_transformer(selected_room=input_device, start_time=input_date, y_feature=selected_feature, aggregation_level=aggregation_level, prediction_count=points_to_forecast, window_size=window_size, device=device, clean_data=clean_data)
    df_room_date = pd.merge(df_room_date, df_room_date_pred_Transformer, on='date_time_rounded', how='left', suffixes=('', '_pred_Transformer'))


st.markdown("## Detailed Data View")

pred_columns = [column for column in df_room_date.columns if '_pred' in column]
if df_room_date[pred_columns].isnull().values.all():
    # show message that theres not enough context data to predict
    st.markdown("<div style='color:red; border: 1px solid red; padding: 10px;'>Not enough context data available to calculate a forecast, please select a data with more available data</div>", unsafe_allow_html=True)
st.plotly_chart(utils.plot_figure(df_room_date, y_feature=selected_feature), use_container_width=True)
st.markdown(f"## Overview of entire available {feature_data} data for {input_device}")
st.plotly_chart(utils.plot_figure(df_room, y_feature=selected_feature, mode="markers"), use_container_width=True)


# Detailed data view
st.markdown("## Detailed Data View") 
st.dataframe(df_room_date)