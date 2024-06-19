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
from models import TransformerModel, LSTMModel, FCN
import joblib
import pickle
import holidays
from datetime import date, timedelta
import torch.optim as optim

def get_device() -> torch.device:
    has_mps = torch.backends.mps.is_built()
    return "mps" if has_mps else "cuda" if torch.cuda.is_available() else "cpu"

def get_batch_size() -> int:
    if get_device() == "mps":
        return 128
    else:
        return 1500

def get_list_of_special_dates():
    holiday_list = holidays.Germany(years=[2022, 2023, 2024])
    exam_dates = holidays.HolidayBase()

    # Date ranges for HKA holidays
    holiday_date_ranges = [
        ('2022-07-25', '2023-09-23'),
        ('2022-12-26', '2023-01-06'),
        ('2023-02-13', '2023-03-12'),
        ('2023-04-11', '2023-04-11'),
        ('2023-05-29', '2023-06-02'),
        ('2023-07-24', '2023-09-22'),
        ('2023-12-22', '2024-01-05'),
        ('2024-02-12', '2024-03-15')
    ]

    # Date ranges for HKA exams
    exam_date_ranges = [
        ('2023-01-23', '2023-02-11'),
        ('2023-07-03', '2023-07-21'),
        ('2024-01-22', '2024-02-09')
    ]

    def add_custom_holidays(ranges, holiday_obj):
        for start, end in ranges:
            start_date = date.fromisoformat(start)
            end_date = date.fromisoformat(end)
            current_date = start_date
            while current_date <= end_date:
                holiday_obj.append(current_date)
                current_date += timedelta(days=1)

    add_custom_holidays(holiday_date_ranges, holiday_list)
    add_custom_holidays(exam_date_ranges, exam_dates)

    return holiday_list, exam_dates

def save_model(model: nn.Module, model_name: str=None, model_path: str=None, y_feature: str='CO2'):
    if model_path is None:
        version = 1
        while os.path.isfile(f"models/{model_name}_{y_feature}_model_v{version}.pth"):
            version += 1
        
        model_path = f'models/{model_name}_{y_feature}_model_v{version}.pth'

    torch.save(model.state_dict(), model_path)

def save_scaler(scaler: StandardScaler, model_name: str=None, scaler_path: str=None, y_feature: str='CO2'):
    if scaler_path is None:
        version = 1
        while os.path.isfile(f'models/{model_name}_{y_feature}_scaler_v{version}.pth'):
            version += 1
        
        scaler_path = f'models/{model_name}_{y_feature}_scaler_v{version}.pth'
    
    torch.save(scaler, scaler_path)

def save_columns(df: pd.DataFrame, model_name: str=None, file_path: str=None, y_feature: str='CO2'):
    if file_path is None:
        version = 1
        while os.path.isfile(f'models/{model_name}_{y_feature}_columns_v{version}.pkl'):
            version += 1
        
        file_path = f'models/{model_name}_{y_feature}_columns_v{version}.csv'
    
    with open(file_path, 'wb') as f:
        pickle.dump(df.columns.tolist(), f)

def save_dataframe(df: pd.DataFrame, model_name: str=None, file_path: str=None):
    if file_path is None:
        version = 1
        while os.path.isfile(f'data/{model_name}_dataframe_v{version}.csv'):
            version += 1
        
        file_path = f'data/{model_name}_dataframe_v{version}.csv'
    
    df.to_csv(file_path, index=False)

def clean_df(df: pd.DataFrame, clean_data: bool=True) -> pd.DataFrame:
    """
    args:   df: pd.DataFrame
            clean_data: bool

    returns: pd.DataFrame
    """
    df_cpy = deepcopy(df)
    df_cpy['date_time'] = pd.to_datetime(df_cpy['date_time'])
    df_cpy['device_id'] = df_cpy['device_id'].str.strip()
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
    # add a trace for CO2_pred if it exists
    if f'{y_feature}_pred' in df.columns:
        fig.add_trace(go.Scatter(x=df['date_time'], y=df[f'{y_feature}_pred'], mode=mode, line=dict(color='red')))
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

def get_data_for_transformer(df: pd.DataFrame, y_feature: str='CO2', window_size: int=20, aggregation_level: str = 'quarter_hour', batch_size: int=get_batch_size(), clean_data: bool=True) -> np.array:
    """
    args:   df: pd.DataFrame
            y_feature: str
            window_size: int
            aggregation_level: str, one of "hour", "half_hour", "quarter_hour"

    returns: np.array
    """
    # use data cleaning from clean_data function
    df_cpy = clean_df(df, clean_data)

    if aggregation_level == "hour":
        df_cpy['date_time'] = df_cpy['date_time'].dt.round('60T')
    elif aggregation_level == "half_hour":
        df_cpy['date_time'] = df_cpy['date_time'].dt.round('30T')
    elif aggregation_level == "quarter_hour":
        df_cpy['date_time'] = df_cpy['date_time'].dt.round('15T')
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
    print('training data cutoff: ', threshold_date)

    df_train = df_cpy[df_cpy['date_time'] < threshold_date]
    df_test = df_cpy[df_cpy['date_time'] >= threshold_date]

    # Create the scaler instance
    scaler = StandardScaler()

    # Fit on training data and transform both training and test data
    df_train[f'{y_feature}_scaled'] = scaler.fit_transform(df_train[[y_feature]])
    df_test[f'{y_feature}_scaled'] = scaler.transform(df_test[[y_feature]])

    def to_sequences(seq_size: int, obs: pd.DataFrame):
        x = []
        y = []
        for g_id in obs['group'].unique():
            group_df = obs[obs['group'] == g_id]
            feature_values = group_df[f'{y_feature}_scaled'].to_numpy().reshape(-1, 1).flatten().tolist()
            for i in range(len(feature_values) - seq_size):
                window = feature_values[i:(i + seq_size)]
                after_window = feature_values[i + seq_size]
                x.append(window)
                y.append(after_window)
        return torch.tensor(x, dtype=torch.float32).view(-1, seq_size, 1), torch.tensor(y, dtype=torch.float32).view(-1, 1)

    x_train, y_train = to_sequences(window_size, df_train)
    x_test, y_test = to_sequences(window_size, df_test)

    print("Training data shape:", x_train.shape, y_train.shape)
    print("Testing data shape:", x_test.shape, y_test.shape)

    # Setup data loaders for batch
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    test_dataset = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    return train_dataset, test_dataset, train_loader, test_loader, scaler, y_test

def train_transformer_model(device, train_loader: DataLoader, test_loader: DataLoader, scaler: StandardScaler, epochs: int=1000, input_dim=None, d_model=64, nhead=4, num_layers=2, dropout=0.25, learning_rate: float=0.001, y_feature_scaler_index: int=0):
    if input_dim == None:
        input_dim = train_loader.dataset.tensors[0].shape[-1]
    model = TransformerModel(input_dim=input_dim, d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=dropout).to(device)
    # Train the model
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)

    early_stop_count = 0
    min_val_loss = float('inf')

    for epoch in range(epochs):
        print_for_epoch = False
        model.train()
        for batch in train_loader:
            x_batch, y_batch = batch
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
                if not print_for_epoch:
                    # Reshape and inverse transform only the 'CO2' outputs using the specific index
                    actual = y_batch.cpu().numpy().reshape(-1, 1)
                    predicted = outputs.cpu().numpy().reshape(-1, 1)
                    zeroes_for_scaler = np.zeros((actual.shape[0], input_dim))
                    
                    zeroes_for_scaler[:, y_feature_scaler_index] = actual.flatten()  # Insert CO2 values into the correct column
                    inverse_transformed = scaler.inverse_transform(zeroes_for_scaler)
                    actual_unscaled = inverse_transformed[:, y_feature_scaler_index]

                    zeroes_for_scaler[:, y_feature_scaler_index] = predicted.flatten()  # Insert CO2 values into the correct column
                    inverse_transformed = scaler.inverse_transform(zeroes_for_scaler)
                    predicted_unscaled = inverse_transformed[:, y_feature_scaler_index]

                    # print the predictions vs the actual values for the first batch scaled back to original values as a dataframe
                    print(pd.DataFrame({'actual': actual_unscaled, 'predicted': predicted_unscaled}))
                    print_for_epoch = True

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

def train_multivariate_model(device, train_loader: DataLoader, test_loader: DataLoader, scaler: StandardScaler, epochs: int=1000):
    model = LSTMModel(input_dim=next(iter(train_loader))[0].shape[-1], hidden_dim=100, num_layers=1, output_dim=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)

    early_stop_count = 0
    min_val_loss = float('inf')

    for epoch in range(epochs):
        print_for_epoch = False
        model.train()
        for batch in train_loader:
            x_batch, y_batch = batch
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            print(x_batch.shape)
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
                if not print_for_epoch:
                    # print the predictions vs the actual values for the first batch scaled back to original values as a dataframe
                    print(pd.DataFrame({'actual': scaler.inverse_transform(y_batch.cpu().numpy().reshape(-1, 1)).flatten(), 'predicted': scaler.inverse_transform(outputs.cpu().numpy().reshape(-1, 1)).flatten()}))
                    print_for_epoch = True

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

def train_fcn_model(device, train_loader: DataLoader, test_loader: DataLoader, scaler: StandardScaler, epochs: int=1000):
    model = FCN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)

    early_stop_count = 0
    min_val_loss = float('inf')

    for epoch in range(epochs):
        print_for_epoch = False
        model.train()
        for batch in train_loader:
            x_batch, y_batch = batch
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch.unsqueeze(1))  # Add unsqueeze to match target size
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
                loss = criterion(outputs, y_batch.unsqueeze(1))  # Add unsqueeze to match target size
                val_losses.append(loss.item())
                if not print_for_epoch:
                    # print the predictions vs the actual values for the first batch scaled back to original values as a dataframe
                    print(pd.DataFrame({'actual': scaler.inverse_transform(y_batch.cpu().numpy().reshape(-1, 1)).flatten(), 'predicted': scaler.inverse_transform(outputs.cpu().numpy().reshape(-1, 1)).flatten()}))
                    print_for_epoch = True

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

def evaluate_transformer_model(device, test_loader: DataLoader, model: nn.Module, scaler: StandardScaler, y_test: torch.Tensor, y_feature_scaler_index: int=0, input_dim: int=1):
    '''kind of pointless function, because its almost the exact same thing that happens after each epoch.
    But it can be used to evaluate a model at a later point again'''
    if input_dim == None:
        input_dim = test_loader.dataset.tensors[0].shape[-1]
    # Evaluation
    model.eval()
    predictions = []
    actual = []
    
    with torch.no_grad():
        for batch in test_loader:
            x_batch, y_batch = batch
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            # Reshape and inverse transform only the 'CO2' outputs using the specific index
            actual_batch = y_batch.cpu().numpy().reshape(-1, 1)
            predicted_batch = outputs.cpu().numpy().reshape(-1, 1)
            zeroes_for_scaler = np.zeros((actual_batch.shape[0], input_dim))
            
            zeroes_for_scaler[:, y_feature_scaler_index] = actual_batch.flatten()  # Insert CO2 values into the correct column
            inverse_transformed = scaler.inverse_transform(zeroes_for_scaler)
            actual_batch_unscaled = inverse_transformed[:, y_feature_scaler_index]

            zeroes_for_scaler[:, y_feature_scaler_index] = predicted_batch.flatten()  # Insert CO2 values into the correct column
            inverse_transformed = scaler.inverse_transform(zeroes_for_scaler)
            predicted_batch_unscaled = inverse_transformed[:, y_feature_scaler_index]

            predictions.extend(predicted_batch_unscaled)
            actual.extend(actual_batch_unscaled)

    # print dataframe of actual vs predicted with inverse transform
    print(pd.DataFrame({'actual': actual, 'predicted': predictions}))
    
    # print multiple accuracy messurements
    rmse = np.sqrt(np.mean((actual - predictions)**2))
    print(f"Score (RMSE): {rmse:.4f}")
    mae = np.mean(np.abs(actual - predictions))
    print(f"Score (MAE): {mae:.4f}")
    mape = np.mean(np.abs((actual - predictions) / actual)) * 100
    print(f"Score (MAPE): {mape:.4f}%")
    return rmse, mae, mape

def load_transformer_model(y_feature: str='CO2', model_name: str=None, model_path: str=None, device: torch.device=get_device(), input_dim: int=1) -> nn.Module:
    if model_path is None:
        version = 1
        while os.path.isfile(f"models/{model_name}_{y_feature}_model_v{version}.pth"):
            version += 1
        
        model_path = f"models/{model_name}_{y_feature}_model_v{version-1}.pth"
        print("loading latest model: " + model_path)
    else:
        print("loading:" + model_path)
    model = TransformerModel(input_dim=input_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model

def load_scaler(y_feature: str='CO2', model_name: str=None, scaler_path: str=None) -> StandardScaler:
    if scaler_path is None:
        version = 1
        while os.path.isfile(f"models/{model_name}_{y_feature}_scaler_v{version}.pth"):
            version += 1
        
        scaler_path = f"models/{model_name}_{y_feature}_scaler_v{version-1}.pth"
        print("loading latest scaler: " + scaler_path)
    else:
        print("loading:" + scaler_path)
    scaler = torch.load(scaler_path)

    return scaler

def load_columns(file_path: str) -> list:
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def load_dataframe(file_path: str, dataframe_name: str=None) -> pd.DataFrame:
    if file_path is None:
        version = 1
        while os.path.isfile(f'data/{dataframe_name}_v{version}.csv'):
            version += 1
        
        file_path = f'data/{dataframe_name}_dataframe_v{version-1}.csv'
    
    return pd.read_csv(file_path)

def get_data_for_prediction(df: pd.DataFrame, scaler: StandardScaler, clean_data: bool=True, window_size: int=20, aggregation_level: str='quarter_hour', y_feature: str='CO2') -> pd.DataFrame:
    """
    args:   df: pd.DataFrame
            clean_data: bool

    returns: pd.DataFrame
    """
    df_cpy = clean_df(df, clean_data)

    if aggregation_level == "hour":
        df_cpy['date_time_rounded'] = df_cpy['date_time'].dt.round('60T')
    elif aggregation_level == "half_hour":
        df_cpy['date_time_rounded'] = df_cpy['date_time'].dt.round('30T')
    elif aggregation_level == "quarter_hour":
        df_cpy['date_time_rounded'] = df_cpy['date_time'].dt.round('15T')
    else:
        raise ValueError("Invalid aggregation_level. Please choose one of 'hour', 'half_hour', or 'quarter_hour'.")


    df_help = df_cpy[['device_id', 'date_time_rounded', y_feature]]

    df_help = df_help.groupby(['device_id', 'date_time_rounded']).mean().reset_index()
    
    # create 'consecutive_data_point' thats 1 if the previous data point is 1 hour before the current data point and device_id is the same, else 0
    time_delta = 3600 if aggregation_level == 'hour' else 1800 if aggregation_level == 'half_hour' else 900
    df_help['consecutive_data_point'] = (df_help['date_time_rounded'] - df_help['date_time_rounded'].shift(1)).dt.total_seconds() == time_delta
    df_help['consecutive_data_point'] = df_help['consecutive_data_point'].astype(int)
    
    # Identify changes and resets (when the value is '0' or there's a change in 'device_id')
    df_help['reset'] = (df_help['consecutive_data_point'] == 0) | (df_help['device_id'] != df_help['device_id'].shift(1))

    # Create a group identifier that increments every time a reset occurs
    df_help['group'] = df_help['reset'].cumsum()

    # Calculate cumulative sum of "1"s within each group
    df_help['consecutive_data_points'] = df_help.groupby(['device_id', 'group'])['consecutive_data_point'].cumsum()
    df_help['group_size'] = df_help.groupby(['device_id', 'group'])['consecutive_data_point'].transform('count')

    df_help[f'{y_feature}_scaled'] = scaler.transform(df_help[[y_feature]])

    # add column that contains a list of the previous $window_size data points to df_help.
    df_help[f'{y_feature}_context'] = [df_help[f'{y_feature}_scaled'].values[max(i-20, 0):i] for i in range(df_help.shape[0])]
    df_help[f'{y_feature}_context_unscaled'] = [df_help[y_feature].values[max(i-20, 0):i] for i in range(df_help.shape[0])]

    # throw out all datapoints that are not predictable
    df_help = df_help[df_help['consecutive_data_points'] >= window_size]

    # join the column "{y_feature}_context" to df_cpy based on  date_time_rounded and device_id columns
    print("cpy: ", df_cpy.shape)
    print("help: ", df_help.shape)
    df_cpy = df_cpy.merge(df_help[['device_id', 'date_time_rounded', f'{y_feature}_context', f'{y_feature}_context_unscaled']], on=['device_id', 'date_time_rounded'], how='left')
    print("NaN count: ", df_cpy[f'{y_feature}_context'].isna().sum())
    return df_cpy

def predict_data(model: nn.Module, scaler: StandardScaler, df: pd.DataFrame, device: torch.device=get_device(), clean_data: bool=True, window_size: int=20, aggregation_level: str='quarter_hour', y_feature: str='CO2', batch_size: int=get_batch_size()) -> pd.DataFrame:
    """
    args:   device: torch.device
            model: nn.Module
            scaler: StandardScaler
            df: pd.DataFrame
            clean_data: bool
            window_size: int
            aggregation_level: str
            y_feature: str

    returns: pd.DataFrame
    """
    df_pred = get_data_for_prediction(df, scaler, clean_data, window_size, aggregation_level, y_feature)
    print(df_pred)
    print(df_pred.shape)
    # print count of NaN in CO2_context
    print(df_pred[f'{y_feature}_context'].isna().sum())
    valid_mask = df_pred[f'{y_feature}_context'].notna()

    context_size = window_size
    # Prepare the data for model input
    contexts = np.stack(df_pred.loc[valid_mask, f'{y_feature}_context'].values)
    contexts_tensor = torch.tensor(contexts, dtype=torch.float32).view(-1, context_size, 1)

    # Initialize a placeholder for predictions with NaN values
    df_pred[f'{y_feature}_pred'] = np.nan  # Add this column initially filled with NaN
    predictions = np.empty((contexts_tensor.shape[0], 1))

    # Make predictions
    dbg = 0
    model.eval()
    with torch.no_grad():
        print(contexts_tensor.shape)
        for i in range(0, contexts_tensor.shape[0], batch_size):
            if i//50000 > dbg:
                print(f"{i} rows processed out of {contexts_tensor.shape[0]}")
                dbg += 1
            batch = contexts_tensor[i:i+batch_size].to(device)
            batch_predictions = model(batch).cpu().numpy().squeeze()
            # Scale back predictions and insert them into the correct positions
            #df_pred.loc[valid_mask, 'CO2_pred'].iloc[i:i+batch_size] = scaler.inverse_transform(batch_predictions.reshape(-1, 1)).flatten()
            predictions[i:i+batch_size, :] = scaler.inverse_transform(batch_predictions.reshape(-1, 1))
    
    df_pred.loc[valid_mask, f'{y_feature}_pred'] = predictions.flatten()
    print("NaN count after prediction: ", df_pred[f'{y_feature}_pred'].isna().sum())
    
    return df_pred

def create_transformer_model_for_feature(df: pd.DataFrame, y_feature: str='CO2', aggregation_level: str='quarter_hour', device: torch.device=get_device() ,window_size: int=20, epochs: int=1000, clean_data: bool=True):
    """
    args:   df: pd.DataFrame
            y_feature: str
            aggregation_level: str
            window_size: int
            epochs: int
            clean_data: bool

    returns: nn.Module, StandardScaler
    """
    # Prepare the data for the model
    train_dataset, test_dataset, train_loader, test_loader, scaler, y_test = get_data_for_transformer(df, y_feature, window_size, aggregation_level, clean_data=clean_data)
    # Train the model
    model = train_transformer_model(device, train_loader, test_loader, scaler, epochs)
    # Evaluate the model
    evaluate_transformer_model(device, test_loader, model, scaler, y_test)
    # Save the model and the scaler
    save_model(model, y_feature=y_feature, model_name='transformer')
    save_scaler(scaler, y_feature=y_feature, model_name='transformer')

    return model, scaler

def get_full_training_dataset(df: pd.DataFrame, aggregation_level: str='half_hour', window_size: int=5, clean_data: bool=True):
    """
    args:   df: pd.DataFrame
            clean_data: bool

    returns: pd.DataFrame
    """
    # Clean data if requested
    if clean_data:
        df_cpy = clean_df(df, clean_data=clean_data)
    else:
        df_cpy = deepcopy(df)
        df_cpy['date_time'] = pd.to_datetime(df_cpy['date_time'])
    
    # Get list of holidays and exams at HKA and add features for them
    holiday_list, exam_dates = get_list_of_special_dates()
    df_cpy['isHoliday'] = df_cpy['date_time'].dt.date.isin(holiday_list).astype(int)
    df_cpy['isExamTime'] = df_cpy['date_time'].dt.date.isin(exam_dates).astype(int)

    # round date_time by half hour, group by date_time_rounded and device_id and take the mean of the other columns
    if aggregation_level == "hour":
        freq='60T'
        df_cpy['date_time_rounded'] = df_cpy['date_time'].dt.round('60T')
    elif aggregation_level == "half_hour":
        freq='30T'
        df_cpy['date_time_rounded'] = df_cpy['date_time'].dt.round('30T')
    elif aggregation_level == "quarter_hour":
        freq='15T'
        df_cpy['date_time_rounded'] = df_cpy['date_time'].dt.round('15T')
    else:
        raise ValueError("Invalid aggregation_level. Please choose one of 'hour', 'half_hour', or 'quarter_hour'.")
    
    # add time dependent information
    df_cpy['weekday'] = df_cpy['date_time_rounded'].dt.weekday
    df_cpy['month'] = df_cpy['date_time_rounded'].dt.month
    #df_cpy['hour'] = df_cpy['date_time_rounded'].dt.round('60T').dt.hour
    df_cpy['hour_sin'] = np.sin(2 * np.pi * (df_cpy['date_time_rounded'].dt.hour * 3600 + df_cpy['date_time'].dt.minute * 60) / 86400.0)
    df_cpy['semester'] = 'WS22/23'
    df_cpy.loc[df_cpy['date_time_rounded'] >= '2023-03-01', 'semester'] = 'SS23'
    df_cpy.loc[df_cpy['date_time_rounded'] >= '2023-09-01', 'semester'] = 'WS23/24'
    df_cpy = pd.get_dummies(df_cpy, columns=['semester'])

    df_grouped = df_cpy.groupby(['device_id', 'date_time_rounded']).mean().reset_index()
    df_grouped.set_index(['device_id', 'date_time_rounded'], inplace=True)

    # Exted the dataset to include all possible date_time_rounded values for each device_id so that we have a complete dataset.
    # Therefore, we can efficiently calculate the correct lag features and the next value.
    df_grouped.sort_index(inplace=True)
    all_times = pd.date_range(df_grouped.index.get_level_values(1).min(), df_grouped.index.get_level_values(1).max(), freq=freq)
    multi_index = pd.MultiIndex.from_product([df_grouped.index.get_level_values(0).unique(), all_times], names=['device_id', 'date_time_rounded'])
    df_grouped = df_grouped.reindex(multi_index)
    lag_features = ['tmp', 'hum', 'CO2', 'VOC', 'vis']
    for feature in lag_features:
        for lag in range(1, window_size+1):
            df_grouped[f'{feature}_lag_{lag}'] = df_grouped.groupby(level=0)[feature].shift(lag)
        df_grouped[f'{feature}_next'] = df_grouped.groupby(level=0)[feature].shift(-1)

    # Drop rows that were not present in the original dataset
    df_grouped.dropna(subset=lag_features, how='all', inplace=True)
    
    return df_grouped

def get_data_for_multivariate_forecast(df: pd.DataFrame, y_feature: str='CO2', window_size: int=5, aggregation_level: str = 'half_hour', batch_size: int=get_batch_size(), clean_data: bool=True, drop_columns: list=[]) -> np.array:
    """
    args:   df: pd.DataFrame
            y_feature: str
            window_size: int
            aggregation_level: str, one of "hour", "half_hour", "quarter_hour"
            batch_size: int
            clean_data: bool
            drop_columns: list

    returns: np.array
    """
    df_cpy = get_full_training_dataset(df, aggregation_level, window_size, clean_data)

    for column in drop_columns:
        df_cpy.drop(column, axis=1, inplace=True)
    for column in df_cpy.columns:
        if '_next' in column and y_feature not in column:
            df_cpy.drop(column, axis=1, inplace=True)

    # drop all rows where CO2_next is NaN
    df_cpy.dropna(subset=[f'{y_feature}_next'], inplace=True)

    # Create the scaler instance
    scaler = StandardScaler()

    # Fit on training data and transform both training and test data
    df_cpy[f'{y_feature}_next_scaled'] = scaler.fit_transform(df_cpy[[f'{y_feature}_next']])
    df_cpy.drop([f'{y_feature}_next'], axis=1, inplace=True)


    train_df, test_df = pd.DataFrame(), pd.DataFrame()
    for device_id, group in df_cpy.groupby(level=0):
        # Find the 75th percentile time
        percentile_75 = np.percentile(group.index.get_level_values(1), 75)
        
        # Convert numpy float timestamp back to datetime for comparison
        date_75 = pd.to_datetime(percentile_75, unit='ns')
        
        # Splitting the data
        train_data = group[group.index.get_level_values(1) <= date_75]
        test_data = group[group.index.get_level_values(1) > date_75]
        
        # Append to the main training and testing DataFrames
        train_df = train_df.append(train_data)
        test_df = test_df.append(test_data)


    x_train = train_df.drop([f'{y_feature}_next_scaled'], axis=1).values
    y_train = train_df[f'{y_feature}_next_scaled'].values
    x_test = test_df.drop([f'{y_feature}_next_scaled'], axis=1).values
    y_test = test_df[f'{y_feature}_next_scaled'].values

    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)


    print("Training data shape:", x_train.shape, y_train.shape)
    print("Testing data shape:", x_test.shape, y_test.shape)

    # Setup data loaders for batch
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    test_dataset = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    return train_dataset, test_dataset, train_loader, test_loader, scaler, y_test

def get_data_for_multivarate_sequential_forecast(df: pd.DataFrame, y_feature: str='CO2', window_size: int=5, aggregation_level: str = 'half_hour', batch_size: int=get_batch_size(), clean_data: bool=True, drop_columns: list=[]) -> np.array:
    """
    args:   df: pd.DataFrame
            y_feature: str
            window_size: int
            aggregation_level: str, one of "hour", "half_hour", "quarter_hour"

    returns: np.array
    """
    # use data cleaning from clean_data function
    df_cpy = clean_df(df, clean_data)

    # Get list of holidays and exams at HKA and add features for them
    holiday_list, exam_dates = get_list_of_special_dates()
    df_cpy['isHoliday'] = df_cpy['date_time'].dt.date.isin(holiday_list).astype(int)
    df_cpy['isExamTime'] = df_cpy['date_time'].dt.date.isin(exam_dates).astype(int)

    # round date_time by half hour, group by date_time_rounded and device_id and take the mean of the other columns
    if aggregation_level == "hour":
        freq='60T'
        df_cpy['date_time_rounded'] = df_cpy['date_time'].dt.round('60T')
    elif aggregation_level == "half_hour":
        freq='30T'
        df_cpy['date_time_rounded'] = df_cpy['date_time'].dt.round('30T')
    elif aggregation_level == "quarter_hour":
        freq='15T'
        df_cpy['date_time_rounded'] = df_cpy['date_time'].dt.round('15T')
    else:
        raise ValueError("Invalid aggregation_level. Please choose one of 'hour', 'half_hour', or 'quarter_hour'.")
    
    # add time dependent information
    df_cpy['weekday'] = df_cpy['date_time_rounded'].dt.weekday
    df_cpy['month'] = df_cpy['date_time_rounded'].dt.month
    df_cpy['hour_sin'] = np.sin(2 * np.pi * (df_cpy['date_time_rounded'].dt.hour * 3600 + df_cpy['date_time_rounded'].dt.minute * 60) / 86400.0)
    df_cpy['semester'] = 'WS22/23'
    df_cpy.loc[df_cpy['date_time_rounded'] >= '2023-03-01', 'semester'] = 'SS23'
    df_cpy.loc[df_cpy['date_time_rounded'] >= '2023-09-01', 'semester'] = 'WS23/24'
    df_cpy = pd.get_dummies(df_cpy, columns=['semester'])

    df_cpy = df_cpy.groupby(['device_id', 'date_time_rounded']).mean().reset_index()
    
    # create 'consecutive_data_point' thats 1 if the previous data point is 1 hour before the current data point and device_id is the same, else 0
    time_delta = 3600 if aggregation_level == 'hour' else 1800 if aggregation_level == 'half_hour' else 900
    df_cpy['consecutive_data_point'] = (df_cpy['date_time_rounded'] - df_cpy['date_time_rounded'].shift(1)).dt.total_seconds() == time_delta
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
    
    threshold_date = df_cpy.sort_values('date_time_rounded', ascending=True)['date_time_rounded'].quantile(0.8)
    print('training data cutoff: ', threshold_date)

    # one hout encode device_id
    df_cpy = pd.get_dummies(df_cpy, columns=['device_id'])
    df_train = df_cpy[df_cpy['date_time_rounded'] < threshold_date]
    df_test = df_cpy[df_cpy['date_time_rounded'] >= threshold_date]

    # Create the scaler instance
    scaler = StandardScaler()

    # Get the columns to scale
    columns_to_scale = [col for col in df_train.columns if col != 'date_time_rounded']

    # Fit on training data and transform both training and test data
    df_train[columns_to_scale] = scaler.fit_transform(df_train[columns_to_scale])
    df_test[columns_to_scale] = scaler.transform(df_test[columns_to_scale])
    y_feature_scaler_index = columns_to_scale.index('CO2')

    # drop unconvertible columns
    df_train.drop(['date_time_rounded'], axis=1, inplace=True)
    df_test.drop(['date_time_rounded'], axis=1, inplace=True)
    
    print(df_train.dtypes)

    def to_sequences(seq_size: int, obs: pd.DataFrame):
        x = []
        y = []
        for g_id in obs['group'].unique():
            group_df = obs[obs['group'] == g_id]
            feature_values = group_df[f'{y_feature}'].to_numpy().reshape(-1, 1).flatten().tolist()
            for i in range(len(group_df) - seq_size):
                window = group_df[i:(i + seq_size)]
                after_window = feature_values[i + seq_size]
                x.append(window.values)
                y.append(after_window)
        feature_count = x[0].shape[1]
        print("feature count: ", feature_count)
        return torch.tensor(np.array(x), dtype=torch.float32).view(-1, seq_size, feature_count), torch.tensor(y, dtype=torch.float32).view(-1, 1)

    x_train, y_train = to_sequences(window_size, df_train)
    x_test, y_test = to_sequences(window_size, df_test)

    print("Training data shape:", x_train.shape, y_train.shape)
    print("Testing data shape:", x_test.shape, y_test.shape)

    # Setup data loaders for batch
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    test_dataset = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    return train_dataset, test_dataset, train_loader, test_loader, scaler, y_test, df_cpy, y_feature_scaler_index

def create_multivariate_model_for_feature(df: pd.DataFrame, y_feature: str='CO2', aggregation_level: str='quarter_hour', device: torch.device=get_device() ,window_size: int=5, epochs: int=1000, clean_data: bool=True, drop_columns: list=[]):
    """
    args:   df: pd.DataFrame
            y_feature: str
            aggregation_level: str, one of "hour", "half_hour", "quarter_hour"
            window_size: int
            epochs: int
            clean_data: bool

    returns: nn.Module, StandardScaler
    """
    # Prepare the data for the model
    train_dataset, test_dataset, train_loader, test_loader, scaler, y_test = get_data_for_multivariate_forecast(df, y_feature, window_size, aggregation_level, clean_data=clean_data, drop_columns=drop_columns)
    # Train the model
    model = train_fcn_model(device, train_loader, test_loader, scaler, epochs)
    # Evaluate the model
    #evaluate_transformer_model(device, test_loader, model, scaler, y_test)
    # Save the model and the scaler
    #save_model(model, y_feature=y_feature)
    #save_scaler(scaler, y_feature=y_feature)

    return model, scaler

def create_multivariate_transformer_model_for_feature(df: pd.DataFrame, y_feature: str='CO2', aggregation_level: str='quarter_hour', device: torch.device=get_device(), window_size: int=20, batch_size: int =get_batch_size(), epochs: int=1000, clean_data: bool=True, input_dim: int=None, d_model: int=64, nhead: int=4, num_layers: int=2, dropout: float=0.25, learning_rate: float=0.001, drop_columns: list=[]):
    """
    args:   df: pd.DataFrame
            y_feature: str
            aggregation_level: str, one of "hour", "half_hour", "quarter_hour"
            window_size: int
            epochs: int
            clean_data: bool

    returns: nn.Module, StandardScaler
    """
    # Prepare the data for the model
    train_dataset, test_dataset, train_loader, test_loader, scaler, y_test, full_preprocessed_df_unscaled, y_feature_scaler_index = get_data_for_multivarate_sequential_forecast(df, y_feature, window_size, aggregation_level, clean_data=clean_data, batch_size=batch_size, drop_columns=drop_columns)
    save_dataframe(full_preprocessed_df_unscaled, model_name=f'transformer_multivariate_{aggregation_level}')
    # Train the model
    model = train_transformer_model(device, train_loader, test_loader, scaler, epochs, input_dim=input_dim, d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=dropout, learning_rate=learning_rate, y_feature_scaler_index=y_feature_scaler_index)
    
    # Save the model, scaler and used columns
    save_model(model, y_feature=y_feature, model_name=f'transformer_multivariate_{aggregation_level}')
    save_scaler(scaler, y_feature=y_feature, model_name=f'transformer_multivariate_{aggregation_level}')
    save_columns(full_preprocessed_df_unscaled, y_feature=y_feature, model_name=f'transformer_multivariate_{aggregation_level}')

    # Evaluate the model
    rmse, mae, mape = evaluate_transformer_model(device, test_loader, model, scaler, y_test, y_feature_scaler_index=y_feature_scaler_index, input_dim=input_dim)

    return model, scaler, rmse, mae, mape

def predict_data_multivariate_transformer(model_name: str='transformer_multivariate', device: torch.device=get_device(), clean_data: bool=True, window_size: int=20, aggregation_level: str='quarter_hour', y_feature: str='CO2', batch_size: int=get_batch_size(), start_time: np.datetime64=None, prediction_count: int=1, selected_room: str=None) -> pd.DataFrame:
    """
    args:   model_name: str
            scaler_name: str
            dataframe_name: str
            device: torch.device
            clean_data: bool
            window_size: int
            aggregation_level: str
            y_feature: str
            batch_size: int
            start_time: np.datetime64
            prediction_count: int

    returns: pd.DataFrame
    """

    full_model_name = model_name + '_' + aggregation_level

    df = load_dataframe(model_name=full_model_name)
    scaler = load_scaler(model_name=full_model_name)
    model = load_transformer_model(model_name=full_model_name, y_feature=y_feature, device=device, input_dim=df.shape[1])


    df = df[df[f'device_id_{selected_room}'] == 1]

    # create a series of <window_size> time points before the start_time. Depending on the aggregation_level, the time points are rounded to the nearest hour, half hour, or quarter hour.
    if start_time is None:
        start_time = df['date_time_rounded'].max()
    
    if aggregation_level == "hour":
        freq = '60T'
    elif aggregation_level == "half_hour":
        freq = '30T'
    elif aggregation_level == "quarter_hour":
        freq = '15T'
    else:
        raise ValueError("Invalid aggregation_level. Please choose one of 'hour', 'half_hour', or 'quarter_hour'.")
    
    start_time = pd.to_datetime(start_time).dt.round(freq)

    # select the <window_size> time points of the df before the start_time and after start_time - <window_size> * freq
    earliest_datapoint_in_window = start_time - window_size * pd.to_timedelta(freq)
    df = df[(df['date_time_rounded'] < start_time)].tail(window_size)






