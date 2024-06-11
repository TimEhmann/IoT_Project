import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from copy import deepcopy
from models import TransformerModel

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

def plot_available_data(df: pd.DataFrame):
    """
    args:   df: pd.DataFrame

    returns: go.Figure
    """
    df_cpy = deepcopy(df)
    df_cpy['date_time'] = pd.to_datetime(df_cpy['date_time'])
    # group by date and take average CO2
    df_cpy['date'] = df_cpy['date_time'].dt.date
    df_cpy = df_cpy.groupby('date').agg({'CO2': 'mean'}).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_cpy['date'], y=df_cpy['CO2'], mode='markers'))

    return fig

def get_data_for_transformer(df: pd.DataFrame, y_feature: str='CO2', window_size: int=10, aggregation_level: str = 'hourly') -> np.array:
    """
    args:   df: pd.DataFrame
            y_feature: str
            window_size: int
            aggregation_level: str, one of "hour", "half_hour", "quarter_hour"

    returns: np.array
    """
    # use data cleaning from prepare_data_for_plot function
    df_cpy = prepare_data_for_plot(df)

    if aggregation_level == "hour":
        df_cpy['date_time'] = df_cpy['date_time'].dt.round('60T')
    elif aggregation_level == "half_hour":
        df_cpy['date_time'] = df_cpy['date_time'].dt.floor('30T')
    elif aggregation_level == "quarter_hour":
        df_cpy['date_time'] = df_cpy['date_time'].dt.floor('15T')
    else:
        raise ValueError("Invalid aggregation_level. Please choose one of 'hour', 'half_hour', or 'quarter_hour'.")


    df_cpy = df_cpy[['device_id', 'date_time', y_feature]]

    df_cpy = df_cpy.groupby(['device_id', 'date_time']).mean().reset_index()
    
    # create 'consecutive_data_point' thats 1 if the previous data point is 1 hour before the current data point and device_id is the same, else 0
    time_delta = 3600 if aggregation_level == 'hour' else 1800 if aggregation_level == 'half_hour' else 900
    df_cpy['consecutive_data_point'] = (df_cpy['date_time'] - df_cpy['date_time'].shift(1)).dt.total_seconds() == time_delta
    df_cpy['consecutive_data_point'] = df_cpy['consecutive_data_point'].astype(int)
    
    # Identify changes and resets (when the value is '0' or there's a change in 'device_id')
    df_cpy['reset'] = (df_cpy['consecutive_data_point'] == 0) | (df_cpy['device_id'] != df_cpy['device_id'].shift(1))

    # Create a group identifier that increments every time a reset occurs
    df_cpy['group'] = df_cpy['reset'].cumsum()

    # Calculate cumulative sum of "1"s within each group
    df_cpy['consecutive_data_points'] = df_cpy.groupby(['device_id', 'group'])['consecutive_data_point'].cumsum() - df_cpy['consecutive_data_point']
    df_cpy['group_size'] = df_cpy.groupby(['device_id', 'group'])['consecutive_data_point'].transform('count')

    df_cpy = df_cpy[df_cpy['group_size'] > window_size]

    # You may want to drop the 'reset' and 'group' columns if they are no longer needed
    df_cpy.drop(['reset', 'consecutive_data_point', 'consecutive_data_points', 'group_size'], axis=1, inplace=True)
    
    threshold_date = df_cpy.sort_values('date_time', ascending=True)['date_time'].quantile(0.8)

    df_train = df_cpy[df_cpy['date_time'] < threshold_date]
    df_test = df_cpy[df_cpy['date_time'] >= threshold_date]

    # Create the scaler instance
    scaler = StandardScaler()

    # Fit on training data and transform both training and test data
    df_train['CO2_scaled'] = scaler.fit_transform(df_train[['CO2']])
    df_test['CO2_scaled'] = scaler.transform(df_test[['CO2']])

    def to_sequences_2(seq_size: int, obs: pd.DataFrame):
        x = []
        y = []
        for g_id in obs['group'].unique():
            group_df = obs[obs['group'] == g_id]
            CO2_values = group_df['CO2_scaled'].to_numpy().reshape(-1, 1).flatten().tolist()
            for i in range(len(CO2_values) - seq_size):
                window = CO2_values[i:(i + seq_size)]
                after_window = CO2_values[i + seq_size]
                x.append(window)
                y.append(after_window)
        return torch.tensor(x, dtype=torch.float32).view(-1, seq_size, 1), torch.tensor(y, dtype=torch.float32).view(-1, 1)

    x_train, y_train = to_sequences_2(window_size, df_train)
    x_test, y_test = to_sequences_2(window_size, df_test)

    print("Training data shape:", x_train.shape, y_train.shape)
    print("Testing data shape:", x_test.shape, y_test.shape)

    # Setup data loaders for batch
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    test_dataset = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_dataset, test_dataset, train_loader, test_loader, scaler, y_test

def train_transformer_model(device: torch.device, train_loader: DataLoader, test_loader: DataLoader, epochs: int=1000):
    model = TransformerModel().to(device)
    # Train the model
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)

    early_stop_count = 0
    min_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            x_batch, y_batch = batch
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in test_loader:
                x_batch, y_batch = batch
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                val_losses.append(loss.item())

        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= 5:
            print("Early stopping!")
            break
        print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss:.4f}")
    
    return model

def evaluate_transformer_model(device: torch.device, test_loader: DataLoader, model: nn.Module, scaler: StandardScaler, y_test: torch.Tensor):
    # Evaluation
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            x_batch, y_batch = batch
            x_batch = x_batch.to(device)
            outputs = model(x_batch)
            predictions.extend(outputs.squeeze().tolist())

    rmse = np.sqrt(np.mean((scaler.inverse_transform(np.array(predictions).reshape(-1, 1)) - scaler.inverse_transform(y_test.numpy().reshape(-1, 1)))**2))
    print(f"Score (RMSE): {rmse:.4f}")