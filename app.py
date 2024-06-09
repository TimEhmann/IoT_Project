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
directory = cwd +'/hka-aqm-am/'
files = [f for f in os.listdir(directory)]
rooms = sorted(set([f.split("_")[0].removeprefix('hka-aqm-') for f in files]))
available_dates_per_room = {room: [f.split("_", 1)[1].split(".")[0] for f in files if f.split("_")[0].removeprefix('hka-aqm-') == room] for room in rooms}

# Sidebar
st.sidebar.header("Dashboard Building AM")
input_device = st.sidebar.selectbox(label= "Select Room", options= rooms)
# only sort the values for the selected decive for run time
available_dates_per_room[input_device].sort()
min_date = datetime.strptime(available_dates_per_room[input_device][0], "%Y_%m_%d")
max_date = datetime.strptime(available_dates_per_room[input_device][-1], "%Y_%m_%d")
input_date = st.sidebar.date_input(label= "Select Date", value= min_date, min_value= min_date, max_value= max_date)
clean_data = st.sidebar.checkbox(label= "Clean Data", value= True)

# Room data for the selected day
df_room_date = pd.read_csv(directory + f"hka-aqm-{input_device}_{str(input_date).replace('-', '_')}.dat", skiprows=1, sep=';', engine='python')
df_room_date = utils.prepare_data_for_plot(df_room_date, clean_data)
df_room = pd.concat([pd.read_csv(directory + f"hka-aqm-{input_device}_{date}.dat", skiprows=1, sep=';', engine='python') for date in available_dates_per_room[input_device]])
df_room = utils.prepare_data_for_plot(df_room, clean_data)

c1, c2 = st.columns([1, 2])

with c1:
    container = st.container(border=True)
    with container:
        st.markdown("## Placeholder for building model")


with c2:
    # Your existing code here
    try:
        kpi_tmp, kpi_hum, kpi_co2 = st.columns(3)
        kpi_voc, kpi_vis, placeholder = st.columns(3)

        with kpi_tmp:
            container = st.container(border=True)
            with container:
                st.metric("~Temperature", f"{df_room_date['tmp'].mean().round(2)} °C")

        with kpi_hum:
            container = st.container(border=True)
            with container:
                st.metric("~Humidity", f"{df_room_date['hum'].mean().round(2)} %")

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

    except AttributeError:
        st.markdown("## No data available")
        with st.expander("See explanation"):
            st.write(f"No data available for the selected room {input_device} and room {input_date}. To see what data is available, look 'Gaps in the data'.")              


tab_tmp, tab_hum, tab_co2, tab_voc, tab_vis = st.tabs(["Temperature", "Humidity", "CO2", "VOC", "Visibility"])

with tab_tmp:
    st.markdown("### Temperature in °C")
    st.plotly_chart(utils.plot_figure(df_room_date, y_feature="tmp"))

with tab_hum:
    st.markdown("### Humidity in %")
    st.plotly_chart(utils.plot_figure(df_room_date, y_feature="hum"))

with tab_co2:
    st.markdown("### CO2 in ppm")
    st.plotly_chart(utils.plot_figure(df_room_date, y_feature="CO2"))

with tab_voc:
    st.markdown("### VOC in ppb")
    st.plotly_chart(utils.plot_figure(df_room_date, y_feature="VOC"))

with tab_vis:
    st.markdown("### Visibility? Maybe Raw Bit Format?")
    st.plotly_chart(utils.plot_figure(df_room_date, y_feature="vis"))


st.markdown("## Overview of room data")
st.markdown("### Full CO2 in ppm Data")
st.plotly_chart(utils.plot_figure(df_room, y_feature="CO2", mode="markers"), use_container_width=True)


# Detailed data view
st.markdown("## Detailed Data View") 
st.dataframe(df_room_date)