import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import utils
from datetime import datetime
from copy import deepcopy

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
selected_building = st.sidebar.selectbox(label= "Select Building", options= ['am', 'f'])
df_full_data = pd.read_parquet(f'data/hka-aqm-{selected_building}-combined-RAW.parquet')
df_full_data['device_id'] = df_full_data['device_id'].str.strip()
df_full_data['date_time'] = pd.to_datetime(df_full_data['date_time'])
rooms = sorted(set([f.removeprefix('hka-aqm-') for f in df_full_data['device_id'].unique()]))
available_dates_per_room = {room: sorted(set(df_full_data[df_full_data['device_id'] == f'hka-aqm-{room}']['date_time'].dt.strftime("%Y_%m_%d"))) for room in rooms}

# Sidebar
input_device = st.sidebar.selectbox(label= "Select Room", options= rooms)
available_dates_per_room[input_device].sort()
min_date = datetime.strptime(available_dates_per_room[input_device][0], "%Y_%m_%d")
max_date = datetime.strptime(available_dates_per_room[input_device][-1], "%Y_%m_%d")
input_date = st.sidebar.date_input(label= "Select Date", value= min_date, min_value= min_date, max_value= max_date)

clean_data = st.sidebar.checkbox(label= "Clean Data", value= True)

# Room data for the selected room
df_full_data = df_full_data.drop(columns=['channel_rssi', 'channel_index'])
df_full_data['device_id'] = df_full_data['device_id'].str.strip()
df_room = deepcopy(df_full_data[df_full_data['device_id'] == f'hka-aqm-{input_device}'])
df_room = utils.clean_df(df_room, clean_data)
df_full_data = utils.clean_df(df_full_data, clean_data)

# check if data exists for the selected room and date
data_exists = df_room[df_room['device_id'] == f'hka-aqm-{input_device}']['date_time'].dt.strftime("%Y_%m_%d").str.contains(input_date.strftime("%Y_%m_%d")).any()
if not data_exists:
    st.markdown("## No data available")
    st.markdown(f"## Overview of available dates for {input_device}")
    st.markdown("### Average CO2 in ppm per available date")
    st.plotly_chart(utils.plot_available_data(df_room), use_container_width=True)
    st.stop()

# MAIN CONTENT

# Create the tabs
if 'active_tab' not in st.session_state:
    st.session_state['active_tab'] = "Data Analysis"

tab1, tab2 = st.tabs(["Data Analysis", "Data Forecast"])

# Update the active tab based on the user selection
if tab1:
    st.session_state['active_tab'] = "Data Analysis"
elif tab2:
    st.session_state['active_tab'] = "Data Forecast"

# Data Analysis
with tab1:
    # AGGREGATED VALUES ON DAY AND HOUR LEVEL
    st.markdown(f"## Correlation Analysis for {input_device}")
    st.markdown(f"#### Select the feature you are interested in. The data is aggregated on a daily basis by taking the mean value.")
    df_room_agg_day = df_room.groupby(df_room['date_time'].dt.date).mean().reset_index()
    df_room_agg_hour = df_room.groupby(df_room['date_time'].dt.hour).mean().reset_index()
    df_room_agg_day.sort_values(by='date_time', inplace=True)
    df_room_agg_hour.sort_values(by='date_time', inplace=True)
    features = ['Temperature', 'Humidity', 'CO2', 'VOC', 'Brightness',
       'WIFI', 'BLE', 'rssi', 'snr',
       'channel_index', 'spreading_factor', 'bandwidth']
    features_to_feature = {'Temperature': 'tmp', 'Humidity': 'hum', 'CO2': 'CO2', 'VOC': 'VOC', 'Brightness': 'vis'}
    selected_features = []
    selected_features = st.multiselect("Selected Features", features, default=['CO2'])
    selected_features = [features_to_feature.get(feature, feature) for feature in selected_features]
    print(selected_features)
    st.plotly_chart(utils.plot_figure(df_room_agg_day, y_feature=selected_features, mode='lines+markers'), use_container_width=True)
    col1, col2 = st.columns([5,1])
    with col2:
        #### IS THIS EVEN NEEDED??? WE CAN JUST USE ALL AND SELECT WEEKDAYS VIA PLOTLY PLOT
        weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        selected_weekdays = weekdays
        selected_feature_weekday = st.selectbox(label= "Select Feature", options= features, key='weekday_feature')
        selected_feature_weekday = features_to_feature.get(selected_feature_weekday, selected_feature_weekday)
        to_zero = st.checkbox(label= "Set y-axis to zero", value= False, key='to_zero_weekday')

    with col1:
        fig = go.Figure()
        for weekday in selected_weekdays:
            df_weekday = df_room[df_room['date_time'].dt.day_name() == weekday]
            df_room_agg_hour_weekday = df_weekday.groupby(df_weekday['date_time'].dt.hour).mean().reset_index()
            # fig.add_trace(go.Scatter(x=df_room_agg_hour_weekday['date_time'], y=df_room_agg_hour_weekday[selected_features[0]], mode='lines+markers', name=weekday))
            fig = utils.plot_figure(df_room_agg_hour_weekday, y_feature=selected_feature_weekday, mode='lines+markers', fig=fig, name=" " + weekday, to_zero=to_zero)

        fig.update_layout(title=f"Average values of {selected_feature_weekday} per hour for selected weekdays in {input_device}", xaxis_title="Hour")
        st.plotly_chart(fig, use_container_width=True)
        # st.plotly_chart(utils.plot_figure(df_room_agg_hour_weekday, y_feature=selected_features, mode='lines+markers'), use_container_width=True)

    ## CORRELATION MATRIX
    correlation_matrix = df_room.drop(columns=['bandwidth']).corr()
    fig = px.imshow(correlation_matrix,
                    text_auto=".2f",
                    labels=dict(x="Feature", y="Feature", color="Correlation"),
                    x=correlation_matrix.columns,
                    y=correlation_matrix.columns,
                    color_continuous_scale='RdBu_r')  # 'RdBu_r' is a good choice for correlation as it centers around zero
    fig.update_layout(
        title="Correlation Matrix",
        width=1800,  # Adjust width as needed
        height=800,  # Adjust height as needed
        margin=dict(t=50, l=10, b=10, r=10)  # Tightens the margin around the plot
    )
    st.plotly_chart(fig, use_container_width=True)

    ## SEMESTER DIFFERENCES

    st.markdown("## Semester Differences")
    df_full_data['semester'] = 'WS22/23'
    df_full_data.loc[df_full_data['date_time'] >= '2023-03-01', 'semester'] = 'SS23'
    df_full_data.loc[df_full_data['date_time'] >= '2023-09-01', 'semester'] = 'WS23/24'

    col1, col2 = st.columns([5,1])
    with col2:
        features = ['Temperature', 'Humidity', 'CO2', 'VOC', 'Brightness',
            'WIFI', 'BLE', 'rssi', 'snr',
            'channel_index', 'spreading_factor', 'bandwidth']
        features_to_feature = {'Temperature': 'tmp', 'Humidity': 'hum', 'CO2': 'CO2', 'VOC': 'VOC', 'Brightness': 'vis'}
        
        selected_feature_semester = st.selectbox(label= "Select Feature", options= features)
        selected_feature_semester = features_to_feature.get(selected_feature_semester, selected_feature_semester)

        weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday', 'Full Week']
        selected_weekday = st.selectbox(label= "Select Weekday", options= weekdays)

        aggregation_level_selection = st.selectbox(label= "Aggregation Level", options= ["15min", "30min", "60min"], key='semester')
        freq = aggregation_level_selection.replace('min', 'T')

        use_all_rooms = st.checkbox(label= "Use all rooms", value= False)

    with col1:
        fig = go.Figure()
        print(df_full_data.dtypes)
        for semester in df_full_data['semester'].unique():
            df_semester = df_full_data[df_full_data['semester'] == semester]
            if selected_weekday != 'Full Week':
                df_semester = df_semester[df_semester['date_time'].dt.day_name() == selected_weekday]
            if not use_all_rooms:
                df_semester = df_semester[df_semester['device_id'] == f'hka-aqm-{input_device}']
            data_points = len(df_semester)
            df_semester['date_time'] = df_semester['date_time'].dt.round(freq)
            df_room_agg_hour_semester = df_semester.groupby(df_semester['date_time'].dt.time).mean().reset_index()
            fig = utils.plot_figure(df_room_agg_hour_semester, y_feature=selected_feature_semester, mode='lines+markers', fig=fig, name=f' {semester} ({data_points})')

        fig.update_layout(title="Average values per hour for selected weekday by semester", xaxis_title="Time")
        st.plotly_chart(fig, use_container_width=True)
        # st.plotly_chart(utils.plot_figure(df_room_agg_hour_weekday, y_feature=selected_features, mode='lines+markers'), use_container_width=True)


    ## ROOM DIFFERENCES PER SEMESTER
    st.markdown("## Room differences per semester")
    st.markdown('#### Show the rooms with the highest and lowest average for a given feature')
    col1, col2 = st.columns([5,1])
    with col2:
        selected_semester = st.selectbox(label= "Select Semester", options= ['WS22/23', 'SS23', 'WS23/24'], key='winter_22_semester')

        features = ['Temperature', 'Humidity', 'CO2', 'VOC', 'Brightness', 'IR',
            'WIFI', 'BLE', 'rssi', 'snr',
            'channel_index', 'spreading_factor', 'bandwidth']
        features_to_feature = {'Temperature': 'tmp', 'Humidity': 'hum', 'CO2': 'CO2', 'VOC': 'VOC', 'Brightness': 'vis'}
        
        selected_feature_winter_22 = st.selectbox(label= "Select Feature", options= features, key='winter_22_feature')
        selected_feature_winter_22 = features_to_feature.get(selected_feature_winter_22, selected_feature_winter_22)

        weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday', 'Full Week']
        selected_weekday_winter_22 = st.selectbox(label= "Select Weekday", options= weekdays, key='winter_22_weekday')

        aggregation_level_selection = st.selectbox(label= "Aggregation Level", options= ["15min", "30min", "60min"], key='winter_22_aggregation_level')
        freq = aggregation_level_selection.replace('min', 'T')

    with col1:
        fig = go.Figure()
        semester = selected_semester
        df_winter_22 = df_full_data[df_full_data['semester'] == semester]
        if selected_weekday_winter_22 != 'Full Week':
            df_winter_22 = df_winter_22[df_winter_22['date_time'].dt.day_name() == selected_weekday_winter_22]

        lowest_avg_room = df_winter_22.groupby('device_id').mean().idxmin()[selected_feature_winter_22]
        highest_avg_room = df_winter_22.groupby('device_id').mean().idxmax()[selected_feature_winter_22]

        df_winter_22['date_time'] = df_winter_22['date_time'].dt.round(freq)
        df_room_highest = df_winter_22[df_winter_22['device_id'] == highest_avg_room]
        df_room_lowest = df_winter_22[df_winter_22['device_id'] == lowest_avg_room]

        data_points_total = len(df_winter_22)
        data_points_highest = len(df_room_highest)
        data_points_lowest = len(df_room_lowest)

        df_room_agg_highest = df_room_highest.groupby(df_room_highest['date_time'].dt.time).mean().reset_index()
        df_room_agg_semester = df_winter_22.groupby(df_winter_22['date_time'].dt.time).mean().reset_index()
        df_room_agg_lowest = df_room_lowest.groupby(df_room_lowest['date_time'].dt.time).mean().reset_index()

        fig = utils.plot_figure(df_room_agg_highest, y_feature=selected_feature_winter_22, mode='lines+markers', fig=fig, name=f' {highest_avg_room} (Highest)\n({data_points_highest})')
        fig = utils.plot_figure(df_room_agg_semester, y_feature=selected_feature_winter_22, mode='lines+markers', fig=fig, name=f' {semester} ({data_points_total})')
        fig = utils.plot_figure(df_room_agg_lowest, y_feature=selected_feature_winter_22, mode='lines+markers', fig=fig, name=f' {lowest_avg_room} (Lowest)\n({data_points_lowest}')

        fig.update_layout(title="Average values per hour for selected weekday", xaxis_title="Time")
        st.plotly_chart(fig, use_container_width=True)
        # st.plotly_chart(utils.plot_figure(df_room_agg_hour_weekday, y_feature=selected_features, mode='lines+markers'), use_container_width=True)


# Data Forecast
with tab2:

    c1, c2 = st.columns([1,1])

    with c1:
        container = st.container(border=True)
        with container:
            models = ["Transformer", "LSTM"]
            selected_models = []
            selected_models = st.multiselect("Selected Models", options=models, default=models)
            
            points_to_forecast = st.number_input(label= "Points to Forecast", value= 1, min_value= 1, max_value= 10, step= 1)
            
            feature_data = st.selectbox(label= "Feature", options= ["Temperature", "Humidity", "CO2", "VOC", "Brightness"])
            feature_name_to_feature = {"Temperature": "tmp", "Humidity": "hum", "CO2": "CO2", "VOC": "VOC", "Brightness": "vis"}
            selected_feature = feature_name_to_feature[feature_data]

            aggregation_level_selection = st.selectbox(label= "Aggregation Level", options= ["15min", "30min", "60min"])
            aggregation_level_selection_to_aggregation_level = {"60min": 'hour', "30min": 'half_hour', "15min": 'quarter_hour'}
            frequency = aggregation_level_selection.replace('min', 'T')
            aggregation_level = aggregation_level_selection_to_aggregation_level[aggregation_level_selection]
            
            df_room_date = deepcopy(df_room[df_room['date_time'].dt.strftime("%Y-%m-%d") == input_date.strftime("%Y-%m-%d")])
            df_room_date['date_time_rounded'] = df_room_date['date_time'].dt.round(frequency)

    with c2:
        kpi_tmp, kpi_hum = st.columns(2)
        kpi_co2, kpi_voc = st.columns(2)
        kpi_vis, placeholder = st.columns(2)

        with kpi_tmp:
            container = st.container(border=True)
            with container:
                st.metric("ØTemperature", f"{round(df_room_date['tmp'].mean(), 2)} °C")

        with kpi_hum:
            container = st.container(border=True)
            with container:
                st.metric("ØHumidity", f"{round(df_room_date['hum'].mean(), 2)} %")

        with kpi_co2:
            container = st.container(border=True)
            with container:
                st.metric("ØCO2", f"{int(df_room_date['CO2'].mean())} ppm")
        
        with kpi_voc:
            container = st.container(border=True)
            with container:
                st.metric("ØVOC", f"{int(df_room_date['VOC'].mean())} ppb")
        
        with kpi_vis:
            container = st.container(border=True)
            with container:
                st.metric("ØBrightness", f"{int(df_room_date['vis'].mean())}")

    device = utils.get_device()
    window_size = 20
    if 'LSTM' in selected_models:
        df_room_date_pred_LSTM = utils.predict_data_multivariate_LSTM(selected_room=input_device, start_time=input_date, y_feature=selected_feature, aggregation_level=aggregation_level, prediction_count=points_to_forecast, window_size=window_size, device=device, clean_data=clean_data,feature_count=25,selected_building=selected_building)
        df_room_date = pd.merge(df_room_date, df_room_date_pred_LSTM, on='date_time_rounded', how='left', suffixes=('', '_pred_LSTM'))
    if 'Transformer' in selected_models:
        df_room_date_pred_Transformer = utils.predict_data_multivariate_transformer(selected_room=input_device, start_time=input_date, y_feature=selected_feature, aggregation_level=aggregation_level, prediction_count=points_to_forecast, window_size=window_size, device=device, clean_data=clean_data,feature_count=25,selected_building=selected_building)
        df_room_date = pd.merge(df_room_date, df_room_date_pred_Transformer, on='date_time_rounded', how='left', suffixes=('', '_pred_Transformer'))
    
    print(df_room_date.columns)
    st.markdown("## Detailed Data View")

    pred_columns = [column for column in df_room_date.columns if '_pred' in column]
    if df_room_date[pred_columns].isnull().values.all():
        # show message that theres not enough context data to predict
        st.markdown("<div style='color:red; border: 1px solid red; padding: 10px;'>Not enough context data available to calculate a forecast, please select a date with more available data</div>", unsafe_allow_html=True)
    st.plotly_chart(utils.plot_figure(df_room_date, y_feature=[selected_feature]), use_container_width=True)
    st.markdown(f"## Overview of entire available {feature_data} data for {input_device}")
    st.plotly_chart(utils.plot_figure(df_room, y_feature=[selected_feature], mode="markers"), use_container_width=True)


    # Detailed data view
    st.markdown("## Detailed Data View") 
    st.dataframe(df_room_date)