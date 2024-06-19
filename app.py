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
# only sort the values for the selected decive for run time
available_dates_per_room[input_device].sort()
min_date = datetime.strptime(available_dates_per_room[input_device][0], "%Y_%m_%d")
max_date = datetime.strptime(available_dates_per_room[input_device][-1], "%Y_%m_%d")
input_date = st.sidebar.date_input(label= "Select Date", value= min_date, min_value= min_date, max_value= max_date)
feature_data = st.sidebar.selectbox(label= "Feature", options= ["Temperature", "Humidity", "CO2", "VOC", "Visibility"])
feature_name_to_feature = {"Temperature": "tmp", "Humidity": "hum", "CO2": "CO2", "VOC": "VOC", "Visibility": "vis"}
selected_feature = feature_name_to_feature[feature_data]
aggregation_level_selection = st.sidebar.selectbox(label= "Aggregation Level", options= ["60min", "30min", "15min"])
aggregation_level_selection_to_aggregation_level = {"60min": 'hour', "30min": 'half_hour', "15min": 'quarter_hour'}
frequency = aggregation_level_selection.replace('min', 'T')
aggregation_level = aggregation_level_selection_to_aggregation_level[aggregation_level_selection]
clean_data = st.sidebar.checkbox(label= "Clean Data", value= True)
# button to increase and decrese value of points_to_forecast
points_to_forecast = st.sidebar.number_input(label= "Points to Forecast", value= 1, min_value= 1, max_value= 10, step= 1)

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
print(df_room_date)

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
#model = utils.load_model(selected_feature, device=device)
#scaler = utils.load_scaler(selected_feature)
#df_room_pred = utils.predict_data(model, scaler, df_room, device, y_feature=selected_feature)
#df_room_date_pred = df_room_pred[df_room_pred['date_time'].dt.strftime("%Y-%m-%d") == input_date.strftime("%Y-%m-%d")]
df_room_date_pred = utils.predict_data_multivariate_transformer(selected_room=input_device, start_time=input_date, y_feature=selected_feature, aggregation_level=aggregation_level, prediction_count=points_to_forecast)
# merge with df_room_date on 'date_time_rounded'
df_for_plot = pd.merge(df_room_date, df_room_date_pred, on='date_time_rounded', how='left', suffixes=('', '_pred'))
print(df_for_plot.columns)

st.markdown("## Detailed Data View")
st.plotly_chart(utils.plot_figure(df_for_plot, y_feature=selected_feature), use_container_width=True)
st.markdown(f"## Overview of entire available {feature_data} data for {input_device}")
st.plotly_chart(utils.plot_figure(df_room, y_feature=selected_feature, mode="markers"), use_container_width=True)

"""
tab_tmp, tab_hum, tab_co2, tab_voc, tab_vis = st.tabs(["Temperature", "Humidity", "CO2", "VOC", "Visibility"])

with tab_tmp:
    st.markdown("### Temperature in °C")
    st.plotly_chart(utils.plot_figure(df_room_pred, y_feature="tmp"))
    st.markdown(f"## Overview of entire available temperature data for {input_device}")
    st.plotly_chart(utils.plot_figure(df_room, y_feature="tmp", mode="markers"), use_container_width=True)

with tab_hum:
    st.markdown("### Humidity in %")
    st.plotly_chart(utils.plot_figure(df_room_pred, y_feature="hum"))
    st.markdown(f"## Overview of entire available humidity data for {input_device}")
    st.plotly_chart(utils.plot_figure(df_room, y_feature="hum", mode="markers"), use_container_width=True)

with tab_co2:
    st.markdown("### CO2 in ppm")
    st.plotly_chart(utils.plot_figure(df_room_pred, y_feature="CO2"))
    st.markdown(f"## Overview of entire available CO2 data for {input_device}")
    st.plotly_chart(utils.plot_figure(df_room, y_feature="CO2", mode="markers"), use_container_width=True)

with tab_voc:
    st.markdown("### VOC in ppb")
    st.plotly_chart(utils.plot_figure(df_room_pred, y_feature="VOC"))
    st.markdown(f"## Overview of entire available VOC data for {input_device}")
    st.plotly_chart(utils.plot_figure(df_room, y_feature="VOC", mode="markers"), use_container_width=True)

with tab_vis:
    st.markdown("### Visibility? Maybe Raw Bit Format?")
    st.plotly_chart(utils.plot_figure(df_room_pred, y_feature="vis"))
    st.markdown(f"## Overview of entire available vis data for {input_device}")
    st.plotly_chart(utils.plot_figure(df_room, y_feature="vis", mode="markers"), use_container_width=True)
"""


# Detailed data view
st.markdown("## Detailed Data View") 
st.dataframe(df_room_date_pred)