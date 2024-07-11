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
import datetime
import torch.optim as optim
import warnings
import gzip
import shutil
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore", category=UserWarning, module='sklearn.base')

def get_device() -> torch.device:
    """
    returns device depending on macbook and pc
    return device
    """
    has_mps = torch.backends.mps.is_built()
    return "mps" if has_mps else "cuda" if torch.cuda.is_available() else "cpu"

def get_batch_size() -> int:
    """
    returns batch_size depending on macbook and pc
    return batch_size
    """
    if get_device() == "mps":
        return 128
    else:
        return 1500

def get_list_of_special_dates():
    """
    1. returns a holidays object with holidays in germany as well as holidays at HKA.
    2. returns a holidays object with the exam dates at HKA

    return holiday_list, exam_dates
    """
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

def save_model(model: nn.Module, model_name: str=None, model_path: str=None, y_feature: str='CO2', overwrite: bool = False):
    """
    saves model to path or figures out the path
    """
    compress = True
    file_ending = '.pth.gz' if compress else '.pth'
    if model_path is None:
        version = 1
        while os.path.isfile(f"models/{model_name}_{y_feature}_model_v{version}{file_ending}"):
            version += 1
        
        if overwrite and version > 1:
            version -= 1
        
        model_path = f'models/{model_name}_{y_feature}_model_v{version}{file_ending}'

    if compress:
        temp_path = model_path + '.tmp'
        torch.save(model, temp_path)
        with open(temp_path, 'rb') as f_in:
            with gzip.open(model_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    
        os.remove(temp_path)
    else:
        torch.save(model, model_path)

def save_scaler(scaler: StandardScaler, model_name: str=None, scaler_path: str=None, overwrite: bool = True):
    """
    saves model to path or figures out the path
    """
    if scaler_path is None:
        version = 1
        while os.path.isfile(f'models/{model_name}_scaler_v{version}.pth'):
            version += 1

        if overwrite and version > 1:
            version -= 1
        
        scaler_path = f'models/{model_name}_scaler_v{version}.pth'
    
    torch.save(scaler, scaler_path)

def save_columns(df: pd.DataFrame, model_name: str=None, file_path: str=None, y_feature: str='CO2', overwrite: bool = False):
    """
    saves columns to path or figures out the path
    """
    if file_path is None:
        version = 1
        while os.path.isfile(f'models/{model_name}_{y_feature}_columns_v{version}.pkl'):
            version += 1
        
        file_path = f'models/{model_name}_{y_feature}_columns_v{version}.csv'
    
    with open(file_path, 'wb') as f:
        pickle.dump(df.columns.tolist(), f)

def get_different_rows(source_df, new_df):
    """
    Returns just the rows from the new dataframe that differ from the source dataframe
    return changed_rows_df
    """
    for col in source_df.columns:
        if col in new_df.columns:
            if source_df[col].dtype != new_df[col].dtype:
                if source_df[col].dtype == 'datetime64[ns]':
                    new_df[col] = pd.to_datetime(new_df[col], errors='coerce')
                elif new_df[col].dtype == 'datetime64[ns]':
                    source_df[col] = pd.to_datetime(source_df[col], errors='coerce')
                else:
                    new_df[col] = new_df[col].astype(source_df[col].dtype)
    
    merged_df = source_df.merge(new_df, indicator=True, how='outer')
    changed_rows_df = merged_df[merged_df['_merge'] == 'right_only']
    return changed_rows_df.drop('_merge', axis=1)

def save_dataframe(df: pd.DataFrame, model_name: str=None, file_path: str=None, overwrite: bool=False):
    """
    saves dataframe to path or figures out the path
    """
    if file_path is None:
        version = 1
        while os.path.isfile(f'data/{model_name}_dataframe_v{version}.parquet'):
            version += 1

        # check if newest saved dataframe is equal to the one thats about to be saved
        if version > 1:
            latest_dataframe = pd.read_parquet(f'data/{model_name}_dataframe_v{version-1}.parquet')
            latest_dataframe['date_time_rounded'] = pd.to_datetime(latest_dataframe['date_time_rounded'])
            if 'group' in latest_dataframe.columns:
                latest_dataframe['group'] = latest_dataframe['group'].astype('int32')

            for col in latest_dataframe.select_dtypes(include=['int64']).columns:
                latest_dataframe[col] = latest_dataframe[col].astype('uint8')
            
            if latest_dataframe.columns.equals(df.columns):
                print("same columns")
            else:
                print(latest_dataframe.columns)
                print(df.columns)
            
            if latest_dataframe.dtypes.equals(df.dtypes):
                print("same dtypes")
            else:
                print(latest_dataframe.dtypes)
                print(df.dtypes)
            
            sin_and_cos_cols = [col for col in latest_dataframe.columns if '_sin' in col or '_cos' in col]
            
            if df.drop(columns=sin_and_cos_cols).equals(latest_dataframe.drop(columns=sin_and_cos_cols)):
                print("Dataframe is already saved.")
                return
            else:
                difs = get_different_rows(latest_dataframe, df)
                print(len(difs), " rows differ from the last saved dataframe.")
        
            if overwrite:
                version -= 1
        
        file_path = f'data/{model_name}_dataframe_v{version}.parquet'
    
    df.to_parquet(file_path, index=False)

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
        df_cpy = df_cpy[df_cpy['CO2'] < 2000]
        # drop all points where hum > 100
        df_cpy = df_cpy[df_cpy['hum'] <= 100]
        # drop all points where tmp > 100. Is 45°C realistic tho?
        df_cpy = df_cpy[df_cpy['tmp'] <= 100]
        # drop all points where VOC > 3000
        df_cpy = df_cpy[df_cpy['VOC'] <= 3000]
        # I dont know what to do with the vis data. Unclear so far.


    return df_cpy

def plot_figure(df: pd.DataFrame, x_feature: str='date_time', y_feature: list=['CO2'], x_title: str=None, y_title: str=None, mode: str='lines+markers', title=None, fig=None, name='Real Data', to_zero=True):
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
        'date_time_rounded': 'Time',
        'tmp': 'Temperature in °C',
        'hum': 'Humidity in %',
        'CO2': 'CO2 in ppm',
        'VOC': 'VOC in ppb',
        'vis': 'Brightness?',
    }
    if not isinstance(y_feature, list):
        y_feature = [y_feature]
    if fig is None:
        fig = go.Figure()
    x_title = feature_title_dictionary.get(x_feature, x_feature) if x_title is None else x_title

    if len(y_feature) == 1:
        fig.add_trace(go.Scatter(x=df[x_feature], y=df[y_feature[0]], mode=mode, name=name))
        y_title = feature_title_dictionary.get(y_feature[0], y_feature[0]) if y_title is None else y_title
        fig.update_layout(xaxis_title=x_title, yaxis_title=y_title)
    elif len(y_feature) == 2:
        y1_title = feature_title_dictionary.get(y_feature[0], y_feature[0])
        y2_title = feature_title_dictionary.get(y_feature[1], y_feature[1])
        
        fig.add_trace(go.Scatter(x=df[x_feature], y=df[y_feature[0]], mode=mode, name=y1_title + name, yaxis='y'))
        fig.add_trace(go.Scatter(x=df[x_feature], y=df[y_feature[1]], mode=mode, name=y2_title + name, yaxis='y2'))
        
        fig.update_layout(
            xaxis_title=x_title,
            yaxis=dict(title=y1_title),
            yaxis2=dict(title=y2_title, overlaying='y', side='right'),
            title=title
        )
    else:
        # Apply Min-Max scaling for more than two features
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[y_feature])
        scaled_df = pd.DataFrame(scaled_data, columns=y_feature)
        
        # Plotting each scaled y feature on the same axis
        for feature in y_feature:
            fig.add_trace(go.Scatter(x=df[x_feature], y=scaled_df[feature], mode=mode, name=feature_title_dictionary.get(feature, feature) + name))
        fig.update_layout(xaxis_title=x_title, yaxis_title='features scaled')
            
    # add a trace for CO2_pred if it exists
    x_feature_pred = 'date_time_rounded' if 'date_time_rounded' in df.columns else 'date_time'
    if f'{y_feature[0]}_pred_LSTM' in df.columns:
        if not df[f'{y_feature[0]}_pred_LSTM'].isna().all():
            fig.add_trace(go.Scatter(x=df[x_feature_pred], y=df[f'{y_feature[0]}_pred_LSTM'], mode=mode, line=dict(color='red'), name='LSTM'))
    if f'{y_feature[0]}_pred_Transformer' in df.columns:
        if not df[f'{y_feature[0]}_pred_Transformer'].isna().all():
            fig.add_trace(go.Scatter(x=df[x_feature_pred], y=df[f'{y_feature[0]}_pred_Transformer'], mode=mode, line=dict(color='Yellow'), name='Transformer'))
    if to_zero:
        fig.update_yaxes(rangemode="tozero")

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
            batch_size: int
            clean_data: bool

    return train_dataset, test_dataset, train_loader, test_loader, scaler, y_test
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

def train_transformer_model(device, train_loader: DataLoader, test_loader: DataLoader, scaler: StandardScaler, epochs: int=1000, input_dim=None, d_model=64, nhead=4, num_layers=2, dropout_pe: float=0.25, dropout_encoder: float=0.25, learning_rate: float=0.001, y_feature_scaler_index: int=0, num_devices=50, device_embedding_dim=4):
    """
    training transformer model. 
    Got reworked to use embedding for device_id to drastically reduce the dimensionality.
    Doesnt work anymore with one hot encoded data.

    return model, train_loss, val_loss
    """
    
    if input_dim == None:
        input_dim = train_loader.dataset.tensors[0].shape[-1]
    model = TransformerModel(input_dim=input_dim, d_model=d_model, nhead=nhead, num_layers=num_layers, dropout_pe=dropout_pe, dropout_encoder=dropout_encoder, num_devices=num_devices, device_embedding_dim=device_embedding_dim).to(device)
    # Train the model
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)

    early_stop_count = 0
    min_val_loss = float('inf')

    for epoch in range(epochs):
        print_for_epoch = False
        model.train()
        train_losses = []
        for batch in train_loader:
            x_batch, device_ids_batch, y_batch = batch
            x_batch, device_ids_batch, y_batch = x_batch.to(device), device_ids_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(x_batch, device_ids_batch)
            loss = criterion(outputs, y_batch)
            train_losses.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in test_loader:
                x_batch, device_ids_batch, y_batch = batch
                x_batch, device_ids_batch, y_batch = x_batch.to(device), device_ids_batch.to(device), y_batch.to(device)
                outputs = model(x_batch, device_ids_batch)
                loss = criterion(outputs, y_batch)
                val_losses.append(loss.item())
                if not print_for_epoch:
                    # Reshape and inverse transform only the 'y_feature' outputs using the specific index
                    actual = y_batch.cpu().numpy().reshape(-1, 1)
                    predicted = outputs.cpu().numpy().reshape(-1, 1)
                    zeroes_for_scaler = np.zeros((actual.shape[0], input_dim))
                    
                    zeroes_for_scaler[:, y_feature_scaler_index] = actual.flatten()  # Insert actual values into the correct column
                    inverse_transformed = scaler.inverse_transform(zeroes_for_scaler)
                    actual_unscaled = inverse_transformed[:, y_feature_scaler_index]

                    zeroes_for_scaler[:, y_feature_scaler_index] = predicted.flatten()  # Insert predicted values into the correct column
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
        train_loss = np.mean(train_losses)  # Calculate the average training loss
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {train_loss:.4f}")  # Print the average training loss
        print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss:.4f}")
    
    return model, train_loss, val_loss

def train_lstm_model(device, train_loader: DataLoader, test_loader: DataLoader, scaler: StandardScaler, epochs: int=1000, input_dim=None, hidden_dim=64, num_layers=2, dropout=0.2, learning_rate: float=0.001, y_feature_scaler_index: int=0, num_devices=50, device_embedding_dim=4):
    if input_dim is None:
        input_dim = train_loader.dataset.tensors[0].shape[-1]
    model = LSTMModel(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=1, num_devices=num_devices, device_embedding_dim=device_embedding_dim, dropout=dropout).to(device)

    # Train the model
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)

    early_stop_count = 0
    min_val_loss = float('inf')

    for epoch in range(epochs):
        print_for_epoch = False
        model.train()
        train_losses = []
        for batch in train_loader:
            x_batch, device_ids_batch, y_batch = batch
            x_batch, device_ids_batch, y_batch = x_batch.to(device), device_ids_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(x_batch, device_ids_batch)
            loss = criterion(outputs, y_batch)
            train_losses.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in test_loader:
                x_batch, device_ids_batch, y_batch = batch
                x_batch, device_ids_batch, y_batch = x_batch.to(device), device_ids_batch.to(device), y_batch.to(device)
                outputs = model(x_batch, device_ids_batch)
                loss = criterion(outputs, y_batch)
                val_losses.append(loss.item())
                if not print_for_epoch:
                    # Reshape and inverse transform only the 'y_feature' outputs using the specific index
                    actual = y_batch.cpu().numpy().reshape(-1, 1)
                    predicted = outputs.cpu().numpy().reshape(-1, 1)
                    zeroes_for_scaler = np.zeros((actual.shape[0], input_dim))
                    
                    zeroes_for_scaler[:, y_feature_scaler_index] = actual.flatten()  # Insert actual values into the correct column
                    inverse_transformed = scaler.inverse_transform(zeroes_for_scaler)
                    actual_unscaled = inverse_transformed[:, y_feature_scaler_index]

                    zeroes_for_scaler[:, y_feature_scaler_index] = predicted.flatten()  # Insert predicted values into the correct column
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
        train_loss = np.mean(train_losses)  # Calculate the average training loss
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {train_loss:.4f}")  # Print the average training loss
        print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss:.4f}")
    
    return model, train_loss, val_loss

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

def evaluate_transformer_model(device, test_loader: DataLoader, model: nn.Module, scaler: StandardScaler, y_test: torch.Tensor, y_feature_scaler_index: int=0, input_dim: int=1, window_size: int=20, y_feature: str='CO2',combined_loader: DataLoader=None, date_time_combined: list=None, device_ids_original_combined: list=None):
    '''
    kind of pointless function, because its almost the exact same thing that happens after each epoch.
    But it can be used to evaluate a model at a later point again.
    Also, it returns more model performance scores than the training loop.

    return rmse, mae, me, mape
    '''
    if input_dim == None:
        input_dim = test_loader.dataset.tensors[0].shape[-1]

    # Evaluation
    model.eval()
    predictions = []
    actual = []

    # Evaluate on combined_loader to get predictions for all data
    combined_predictions = []
    device_ids_combined_list = []
    
    with torch.no_grad():
        for batch in test_loader:
            x_batch, device_ids_batch, y_batch = batch
            x_batch, device_ids_batch, y_batch = x_batch.to(device), device_ids_batch.to(device), y_batch.to(device)
            outputs = model(x_batch, device_ids_batch)
            # Reshape and inverse transform only the 'y_feature' outputs using the specific index
            actual_batch = y_batch.cpu().numpy().reshape(-1, 1)
            predicted_batch = outputs.cpu().numpy().reshape(-1, 1)
            zeroes_for_scaler = np.zeros((actual_batch.shape[0], input_dim))
            
            zeroes_for_scaler[:, y_feature_scaler_index] = actual_batch.flatten()  # Insert actual values into the correct column
            inverse_transformed = scaler.inverse_transform(zeroes_for_scaler)
            actual_batch_unscaled = inverse_transformed[:, y_feature_scaler_index]

            zeroes_for_scaler[:, y_feature_scaler_index] = predicted_batch.flatten()  # Insert predicted values into the correct column
            inverse_transformed = scaler.inverse_transform(zeroes_for_scaler)
            predicted_batch_unscaled = inverse_transformed[:, y_feature_scaler_index]

            predictions.extend(predicted_batch_unscaled)
            actual.extend(actual_batch_unscaled)
        
        for batch in combined_loader:
            x_batch, device_ids_batch, y_batch = batch
            x_batch, device_ids_batch, y_batch = x_batch.to(device), device_ids_batch.to(device), y_batch.to(device)
            outputs = model(x_batch, device_ids_batch)
            # Reshape and inverse transform only the 'y_feature' outputs using the specific index
            predicted_batch = outputs.cpu().numpy().reshape(-1, 1)
            zeroes_for_scaler = np.zeros((predicted_batch.shape[0], input_dim))

            zeroes_for_scaler[:, y_feature_scaler_index] = predicted_batch.flatten()  # Insert predicted values into the correct column
            inverse_transformed = scaler.inverse_transform(zeroes_for_scaler)
            predicted_batch_unscaled = inverse_transformed[:, y_feature_scaler_index]

            combined_predictions.extend(predicted_batch_unscaled)
    
        # Create a DataFrame with date_time_combined and predictions
        prediction_df = pd.DataFrame({
            'date_time': date_time_combined,
            'device_id': device_ids_original_combined,
            f'{y_feature}_prediction_Transformer': combined_predictions
        })

        # Save the predictions to a CSV file
        prediction_df.to_csv(f'data/transformer_multivariate_quarter_hour_{input_dim+1}f_{window_size}ws_{y_feature}_predictions.csv', index=False)

        
        ### test for comparison for later, only works with quarter hour 26f data
        data_for_comparison = load_dataframe(model_name=f'quarter_hour_{input_dim+1}f_{window_size}ws')

        # data are the first 20 data points on 10-10-2022 for am001
        data_df = data_for_comparison.iloc[7:27].drop(columns=['device_id', 'date_time_rounded'])

        data_df_scaled = scaler.transform(data_df)

        input_data = torch.tensor(data_df_scaled, dtype=torch.float32).view(-1, 20, data_df_scaled.shape[1])
        device_ids = torch.tensor([0])

        predicted = model(input_data.to(device), device_ids.to(device))
        print('input_data:', input_data)
        print('predicted:', predicted)

        prediction = predicted.cpu().numpy().reshape(-1, 1)
        zeroes_for_scaler = np.zeros((1, input_dim))

        zeroes_for_scaler[:, y_feature_scaler_index] = prediction.flatten()  # Insert predicted values into the correct column
        inverse_transformed = scaler.inverse_transform(zeroes_for_scaler)
        predicted_unscaled = inverse_transformed[:, y_feature_scaler_index].round(2)
        print(predicted_unscaled)

        # save the last input for reproducability
        with open('data/last_batch.pkl', 'wb') as f:
            pickle.dump((x_batch.cpu().numpy(), device_ids_batch.cpu().numpy(), y_batch.cpu().numpy()), f)

        

    predictions = np.array(predictions)
    actual = np.array(actual)

    # print dataframe of actual vs predicted with inverse transform
    print(pd.DataFrame({'actual': actual, 'predicted': predictions}))
    
    # print multiple accuracy messurements
    rmse = np.sqrt(np.mean((actual - predictions)**2))
    print(f"Score (RMSE): {rmse:.4f}")
    mae = np.mean(np.abs(actual - predictions))
    print(f"Score (MAE): {mae:.4f}")
    me = np.mean(actual - predictions)
    print(f"Score (ME): {me:.4f}")
    mape = np.mean(np.abs((actual - predictions) / actual)) * 100
    print(f"Score (MAPE): {mape:.4f}%")
    return rmse, mae, me, mape

def evaluate_lstm_model(device, test_loader: DataLoader, model: nn.Module, scaler: StandardScaler, y_test: torch.Tensor, y_feature_scaler_index: int=0, input_dim: int=1, window_size: int=20, y_feature: str='CO2', combined_loader: DataLoader=None, date_time_combined: list=None, device_ids_original_combined: list=None):
    '''
    Evaluate the LSTM model and return various performance metrics.

    return rmse, mae, me, mape
    '''
    if input_dim is None:
        input_dim = test_loader.dataset.tensors[0].shape[-1]

    # Evaluation
    model.eval()
    predictions = []
    actual = []

    # Evaluate on combined_loader to get predictions for all data
    combined_predictions = []
    device_ids_combined_list = []
    
    with torch.no_grad():
        for batch in test_loader:
            x_batch, device_ids_batch, y_batch = batch
            x_batch, device_ids_batch, y_batch = x_batch.to(device), device_ids_batch.to(device), y_batch.to(device)
            outputs = model(x_batch, device_ids_batch)
            # Reshape and inverse transform only the 'y_feature' outputs using the specific index
            actual_batch = y_batch.cpu().numpy().reshape(-1, 1)
            predicted_batch = outputs.cpu().numpy().reshape(-1, 1)
            zeroes_for_scaler = np.zeros((actual_batch.shape[0], input_dim))
            
            zeroes_for_scaler[:, y_feature_scaler_index] = actual_batch.flatten()  # Insert actual values into the correct column
            inverse_transformed = scaler.inverse_transform(zeroes_for_scaler)
            actual_batch_unscaled = inverse_transformed[:, y_feature_scaler_index]

            zeroes_for_scaler[:, y_feature_scaler_index] = predicted_batch.flatten()  # Insert predicted values into the correct column
            inverse_transformed = scaler.inverse_transform(zeroes_for_scaler)
            predicted_batch_unscaled = inverse_transformed[:, y_feature_scaler_index]

            predictions.extend(predicted_batch_unscaled)
            actual.extend(actual_batch_unscaled)
        
        for batch in combined_loader:
            x_batch, device_ids_batch, y_batch = batch
            x_batch, device_ids_batch, y_batch = x_batch.to(device), device_ids_batch.to(device), y_batch.to(device)
            outputs = model(x_batch, device_ids_batch)
            # Reshape and inverse transform only the 'y_feature' outputs using the specific index
            predicted_batch = outputs.cpu().numpy().reshape(-1, 1)
            zeroes_for_scaler = np.zeros((predicted_batch.shape[0], input_dim))

            zeroes_for_scaler[:, y_feature_scaler_index] = predicted_batch.flatten()  # Insert predicted values into the correct column
            inverse_transformed = scaler.inverse_transform(zeroes_for_scaler)
            predicted_batch_unscaled = inverse_transformed[:, y_feature_scaler_index]

            combined_predictions.extend(predicted_batch_unscaled)
    
        # Create a DataFrame with date_time_combined and predictions
        prediction_df = pd.DataFrame({
            'date_time': date_time_combined,
            'device_id': device_ids_original_combined,
            f'{y_feature}_prediction_LSTM': combined_predictions
        })

        prediction_df['date_time'] = pd.to_datetime(prediction_df['date_time'])
        first_date = prediction_df['date_time'].iloc[0]
        second_date = prediction_df['date_time'].iloc[1]

        diff_in_min = int(pd.Timedelta(second_date - first_date).seconds / 60)


        # Save the predictions to a CSV file
        prediction_df.to_csv(f'data/lstm_multivariate_{diff_in_min}_{input_dim+1}f_{window_size}ws_{y_feature}_predictions.csv', index=False)

        
        ### test for comparison for later, only works with quarter hour 26f data
        data_for_comparison = load_dataframe(model_name=f'quarter_hour_{input_dim+1}f_{window_size}ws')

        # data are the first 20 data points on 10-10-2022 for am001
        data_df = data_for_comparison.iloc[7:27].drop(columns=['device_id', 'date_time_rounded'])

        data_df_scaled = scaler.transform(data_df)

        input_data = torch.tensor(data_df_scaled, dtype=torch.float32).view(-1, 20, data_df_scaled.shape[1])
        device_ids = torch.tensor([0])

        predicted = model(input_data.to(device), device_ids.to(device))
        print('input_data:', input_data)
        print('predicted:', predicted)

        prediction = predicted.cpu().numpy().reshape(-1, 1)
        zeroes_for_scaler = np.zeros((1, input_dim))

        zeroes_for_scaler[:, y_feature_scaler_index] = prediction.flatten()  # Insert predicted values into the correct column
        inverse_transformed = scaler.inverse_transform(zeroes_for_scaler)
        predicted_unscaled = inverse_transformed[:, y_feature_scaler_index].round(2)
        print(predicted_unscaled)

        # save the last input for reproducability
        with open('data/last_batch.pkl', 'wb') as f:
            pickle.dump((x_batch.cpu().numpy(), device_ids_batch.cpu().numpy(), y_batch.cpu().numpy()), f)

    predictions = np.array(predictions)
    actual = np.array(actual)

    # print dataframe of actual vs predicted with inverse transform
    print(pd.DataFrame({'actual': actual, 'predicted': predictions}))
    
    # print multiple accuracy measurements
    rmse = np.sqrt(np.mean((actual - predictions)**2))
    print(f"Score (RMSE): {rmse:.4f}")
    mae = np.mean(np.abs(actual - predictions))
    print(f"Score (MAE): {mae:.4f}")
    me = np.mean(actual - predictions)
    print(f"Score (ME): {me:.4f}")
    mape = np.mean(np.abs((actual - predictions) / actual)) * 100
    print(f"Score (MAPE): {mape:.4f}%")
    return rmse, mae, me, mape

def load_model(y_feature: str='CO2', model_name: str=None, model_path: str=None, device: torch.device=get_device()) -> nn.Module:
    """
    loads a transformer model using the model name or model_path.
    If model_path is set, it will load exactly that model.
    If model_path is not set, it will load the latest model for the combination of model_name and y_feature

    return model
    """
    compress = True
    file_ending = '.pth.gz' if compress else '.pth'
    
    if model_path is None:
        version = 1
        while os.path.isfile(f"models/{model_name}_{y_feature}_model_v{version}{file_ending}"):
            version += 1
        
        model_path = f"models/{model_name}_{y_feature}_model_v{version-1}{file_ending}"
        print("loading latest model: " + model_path)
    else:
        print("loading:" + model_path)
    
    if compress:
        temp_path = model_path + '.tmp'
        with gzip.open(model_path, 'rb') as f_in:
            with open(temp_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        model = torch.load(temp_path, map_location=device)
        
        os.remove(temp_path)
    else:
        model = torch.load(model_path, map_location=device)
    
    return model

def load_scaler(model_name: str=None, scaler_path: str=None) -> StandardScaler:
    """
    loads a scaler using the model name or scaler_path.
    If scaler_path is set, it will load exactly that model.
    If scaler_path is not set, it will load the latest scaler for the combination of model_name and y_feature

    return scaler
    """
    
    if scaler_path is None:
        version = 1
        while os.path.isfile(f"models/{model_name}_scaler_v{version}.pth"):
            version += 1
        
        scaler_path = f"models/{model_name}_scaler_v{version-1}.pth"
        print("loading latest scaler: " + scaler_path)
    else:
        print("loading:" + scaler_path)
    scaler = torch.load(scaler_path)

    return scaler

def load_columns(file_path: str) -> list:
    """
    loads the list of columns saved at <file_path>. Pointless function so far, probably should be deleted.

    return pickle.load(f)
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def load_dataframe(file_path: str=None, model_name: str=None) -> pd.DataFrame:
    """
    loads a saved dataframe using the model_name or file_path.
    If file_path is set, it will load exactly that model.
    If file_path is not set, it will load the latest dataframe for the model_name

    return pd.read_parquet(file_path)
    """
    if file_path is None:
        version = 1
        while os.path.isfile(f'data/{model_name}_dataframe_v{version}.parquet'):
            version += 1
        
        file_path = f'data/{model_name}_dataframe_v{version-1}.parquet'
        print("loading latest dataframe: " + file_path)

    return pd.read_parquet(file_path)

def get_data_for_prediction(df: pd.DataFrame, scaler: StandardScaler, clean_data: bool=True, window_size: int=20, aggregation_level: str='quarter_hour', y_feature: str='CO2') -> pd.DataFrame:
    """
    This function was used for predicting data with the v1 transformer (uni-variate). 
    Probably useless by now. I dont even know anymore what exact data is returned.
    I believe it takes df as input, checks for which rows its possible to predict data, and if possible, add the required <window_size>
    context datapoints in one column scaled, and one unscaled. I dont know how its used further than that right now.

    args:   df: pd.DataFrame
            clean_data: bool

    returns: pd.DataFrame

    return df_cpy
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
    takes a df as input and create a "predicted" point at every row where its possible.
    args:   device: torch.device
            model: nn.Module
            scaler: StandardScaler
            df: pd.DataFrame
            clean_data: bool
            window_size: int
            aggregation_level: str
            y_feature: str

    returns: pd.DataFrame

    return df+pred
    """
    df_pred = get_data_for_prediction(df, scaler, clean_data, window_size, aggregation_level, y_feature)
    print(df_pred)
    print(df_pred.shape)
    # print count of NaN in 'y_feature'_context
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
    Deprecated function.
    Full pipeline to create a univariate transformer model for <y_feature> using the other input.

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
    model, train_loss, val_loss = train_transformer_model(device, train_loader, test_loader, scaler, epochs)
    # Evaluate the model
    evaluate_transformer_model(device, test_loader, model, scaler, y_test)
    # Save the model and the scaler
    save_model(model, y_feature=y_feature, model_name='transformer')
    save_scaler(scaler, model_name='transformer')

    return model, scaler

def get_full_training_dataset(df: pd.DataFrame, aggregation_level: str='half_hour', window_size: int=5, clean_data: bool=True):
    """
    This function was used for creating a dataset for non-sequential models like MLPs.
    It works by adding lag features for the columns ['tmp', 'hum', 'CO2', 'VOC', 'vis'] because these seem to be the relecant context data.
    Deprecated by now because the model is absolute garbage.
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
    df_cpy['weekday_sin'] = df_cpy['date_time_rounded'].dt.weekday
    # df_cpy['month'] = df_cpy['date_time_rounded'].dt.month
    df_cpy['month_sin'] = np.sin(2 * np.pi * df_cpy['date_time_rounded'].dt.month / 12)
    df_cpy['month_cos'] = np.cos(2 * np.pi * df_cpy['date_time_rounded'].dt.month / 12)
    # df_cpy['hour'] = df_cpy['date_time_rounded'].dt.round('60T').dt.hour
    df_cpy['time_sin'] = np.sin(2 * np.pi * (df_cpy['date_time_rounded'].dt.hour * 3600 + df_cpy['date_time'].dt.minute * 60) / 86400.0)
    df_cpy['time_sin'] = np.cos(2 * np.pi * (df_cpy['date_time_rounded'].dt.hour * 3600 + df_cpy['date_time'].dt.minute * 60) / 86400.0)
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
    Creates a dataframe for multivariate-non-sequential-models.
    Deprecated by now because the models are absolute garbage.

    args:   df: pd.DataFrame
            y_feature: str
            window_size: int
            aggregation_level: str, one of "hour", "half_hour", "quarter_hour"
            batch_size: int
            clean_data: bool
            drop_columns: list

    return train_dataset, test_dataset, train_loader, test_loader, scaler, y_test
    """
    df_cpy = get_full_training_dataset(df, aggregation_level, window_size, clean_data)

    for column in drop_columns:
        df_cpy.drop(column, axis=1, inplace=True)
    for column in df_cpy.columns:
        if '_next' in column and y_feature not in column:
            df_cpy.drop(column, axis=1, inplace=True)

    # drop all rows where 'y_feature'_next is NaN
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

def get_data_for_multivarate_sequential_forecast(df: pd.DataFrame, y_feature: str='CO2', window_size: int=5, aggregation_level: str = 'half_hour', batch_size: int=get_batch_size(), clean_data: bool=True, drop_columns: list=[], extend_training_data: bool=True) -> np.array:
    """
    Creates the training and test data as well as the scaler and a dataframe with the full preprocessed data for multivariate-sequential-models.
    Currently the latest and best function to create the training and test data.
    Currently only uses complete data sets where a full <window_size> data points are available before another datapoint.
    This combination then is a training example. There is no filling if e.g. one data point would be missing because currently there is
    enough training data available anyways and we can use only high quality real training examples therefore.

    args:   df: pd.DataFrame
            y_feature: str
            window_size: int
            aggregation_level: str, one of "hour", "half_hour", "quarter_hour"
            aggregation_level: str
            batch_size: int
            clean_data: bool=True
            drop_columns: list
            extend_training_data: bool


    return train_dataset, test_dataset, train_loader, test_loader, scaler, y_test, full_preprocessed_df_unscaled, y_feature_scaler_index
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

    # encode cyclic features
    df_cpy['weekday_sin'] = np.sin(2 * np.pi * df_cpy['date_time_rounded'].dt.weekday / 7)
    df_cpy['weekday_cos'] = np.cos(2 * np.pi * df_cpy['date_time_rounded'].dt.weekday / 7)
    df_cpy['month_sin'] = np.sin(2 * np.pi * df_cpy['date_time_rounded'].dt.month / 12)
    df_cpy['month_cos'] = np.cos(2 * np.pi * df_cpy['date_time_rounded'].dt.month / 12)
    df_cpy['time_sin'] = np.sin(2 * np.pi * (df_cpy['date_time_rounded'].dt.hour * 3600 + df_cpy['date_time'].dt.minute * 60) / 86400.0)
    df_cpy['time_cos'] = np.cos(2 * np.pi * (df_cpy['date_time_rounded'].dt.hour * 3600 + df_cpy['date_time'].dt.minute * 60) / 86400.0)
    df_cpy['semester'] = 'WS22/23'
    df_cpy.loc[df_cpy['date_time_rounded'] >= '2023-03-01', 'semester'] = 'SS23'
    df_cpy.loc[df_cpy['date_time_rounded'] >= '2023-09-01', 'semester'] = 'WS23/24'
    df_cpy = pd.get_dummies(df_cpy, columns=['semester'])

    df_cpy = df_cpy.groupby(['device_id', 'date_time_rounded']).mean(numeric_only=True).reset_index()
    
    def fill_consecutive_nans(device_df: pd.DataFrame, window_size: int):
        max_consecutive_nans = window_size // 4
        device_df = device_df.sort_values('date_time_rounded')
        device_df['row_contains_no_nan'] = device_df.notna().all(axis=1).astype(int)
        device_df['group'] = device_df['row_contains_no_nan'].cumsum()
        reduced_df = deepcopy(device_df[device_df['row_contains_no_nan'] == False])
        nan_count_in_group = reduced_df.groupby('group')['row_contains_no_nan'].transform('count')
        device_df.loc[device_df.isna().any(axis=1), 'nan_count_in_group'] = nan_count_in_group.loc[device_df.isna().any(axis=1)]

        device_df['nan_count_in_group'] = device_df['nan_count_in_group'].fillna(0)
        device_df['device_id'] = device_df['device_id'].fillna(method='ffill')
        device_df = device_df[device_df['nan_count_in_group'] <= max_consecutive_nans]
        device_df.drop(columns=['row_contains_no_nan', 'group', 'nan_count_in_group'], inplace=True)
        device_df[[col for col in device_df.columns if col != 'date_time_rounded']] = device_df[[col for col in device_df.columns if col != 'date_time_rounded']].interpolate(method='linear', axis=0)
        
        return device_df

    
    if extend_training_data:
        df_cpy_extended = pd.DataFrame()
        
        for device_id in df_cpy['device_id'].unique():
            device_data = df_cpy[df_cpy['device_id'] == device_id]
            min_date = device_data['date_time_rounded'].min()
            max_date = device_data['date_time_rounded'].max()
            date_range = pd.date_range(start=min_date, end=max_date, freq=freq)
            device_df = pd.DataFrame(date_range, columns=['date_time_rounded'])
            merged_df = pd.merge(device_df, device_data, on='date_time_rounded', how='left')
            merged_df_filled = fill_consecutive_nans(merged_df, window_size)

            df_cpy_extended = pd.concat([df_cpy_extended, merged_df_filled], ignore_index=True)
        
        print(f'extended data shape from {df_cpy.shape} to {df_cpy_extended.shape}')

        df_cpy = deepcopy(df_cpy_extended)
    
    
    # the data is now in the state that it needs to be so that it can be used as the context data at later predictions
    full_preprocessed_df_unscaled = deepcopy(df_cpy)
    
    # create 'consecutive_data_point' thats 1 if the previous data point is <freq> before the current data point and device_id is the same, else 0
    df_cpy['consecutive_data_point'] = (df_cpy['date_time_rounded'] - df_cpy['date_time_rounded'].shift(1)).dt.total_seconds() == pd.to_timedelta(freq).total_seconds()
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

    df_cpy['device_id_codes'] = df_cpy['device_id'].astype('category').cat.codes

    df_train = deepcopy(df_cpy[df_cpy['date_time_rounded'] < threshold_date])
    df_test = deepcopy(df_cpy[df_cpy['date_time_rounded'] >= threshold_date])

    # Create the scaler instance
    scaler = StandardScaler()

    # Get the columns to scale
    columns_to_scale = [col for col in df_train.columns if col not in ['date_time_rounded', 'device_id', 'group', 'device_id_codes']]

    # Fit on training data and transform both training and test data
    df_train[columns_to_scale] = scaler.fit_transform(df_train[columns_to_scale])
    df_test[columns_to_scale] = scaler.transform(df_test[columns_to_scale])
    y_feature_scaler_index = columns_to_scale.index(y_feature)

    # drop unconvertible columns
    # df_train.drop(['date_time_rounded'], axis=1, inplace=True)
    # df_test.drop(['date_time_rounded'], axis=1, inplace=True)
    
    print(df_train.dtypes)

    def to_sequences(seq_size: int, obs: pd.DataFrame):
        x = []
        y = []
        time_points = []
        device_ids = []
        device_id_originals = []
        for g_id in obs['group'].unique():
            group_df = obs[obs['group'] == g_id]
            feature_values = group_df[f'{y_feature}'].tolist()
            for i in range(len(group_df) - seq_size):
                window = group_df[i:(i + seq_size)]
                after_window = feature_values[i + seq_size]
                x.append(window.drop(columns=['device_id', 'group', 'date_time_rounded', 'device_id_codes']).values)
                device_ids.append(window['device_id_codes'].values[-1])
                y.append(after_window)
                time_points.append(window['date_time_rounded'].values[-1])
                device_id_originals.append(window['device_id'].values[-1])
        feature_count = x[0].shape[1]
        return (torch.tensor(np.array(x), dtype=torch.float32).view(-1, seq_size, feature_count),
                torch.tensor(device_ids, dtype=torch.long),
                torch.tensor(y, dtype=torch.float32).view(-1, 1),
                time_points,
                device_id_originals)

    print("Creating sequences...")
    x_train, train_device_ids, y_train, time_points_train, device_id_originals_train = to_sequences(window_size, df_train)
    x_test, test_device_ids, y_test, time_points_test, device_id_originals_test = to_sequences(window_size, df_test)

    print("Training data shape:", x_train.shape, train_device_ids.shape,y_train.shape)
    print("Testing data shape:", x_test.shape, test_device_ids.shape, y_test.shape)

    # combining the datasets again and shuffle them. This is because we cant shuffle them before because of the context. Maybe we can, but im too layzy to think about it.
    x_combined = torch.cat((x_train, x_test), dim=0)
    device_ids_combined = torch.cat((train_device_ids, test_device_ids), dim=0)
    y_combined = torch.cat((y_train, y_test), dim=0)
    date_time_combined = time_points_train + time_points_test
    device_ids_original_combined = device_id_originals_train + device_id_originals_test

    # Shuffle the combined dataset
    combined_dataset = TensorDataset(x_combined, device_ids_combined, y_combined)
    combined_loader_unshuffled = DataLoader(combined_dataset, batch_size=batch_size, shuffle=False)
    combined_loader = DataLoader(combined_dataset, batch_size=len(combined_dataset), shuffle=True)

    # Get shuffled data from the loader
    for batch in combined_loader:
        x_combined_shuffled, device_ids_combined_shuffled, y_combined_shuffled = batch

    # Split back into training and test sets (80:20 split)
    split_index = int(0.8 * len(x_combined_shuffled))
    x_train_shuffled, x_test_shuffled = x_combined_shuffled[:split_index], x_combined_shuffled[split_index:]
    train_device_ids_shuffled, test_device_ids_shuffled = device_ids_combined_shuffled[:split_index], device_ids_combined_shuffled[split_index:]
    y_train_shuffled, y_test_shuffled = y_combined_shuffled[:split_index], y_combined_shuffled[split_index:]

    print("Shuffled Training data shape:", x_train_shuffled.shape, train_device_ids_shuffled.shape, y_train_shuffled.shape)
    print("Shuffled Testing data shape:", x_test_shuffled.shape, test_device_ids_shuffled.shape, y_test_shuffled.shape)

    # Setup data loaders for batch
    # train_dataset = TensorDataset(x_train, train_device_ids, y_train)
    train_dataset = TensorDataset(x_train_shuffled, train_device_ids_shuffled, y_train_shuffled)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)


    # test_dataset = TensorDataset(x_test, test_device_ids, y_test)
    test_dataset = TensorDataset(x_test_shuffled, test_device_ids_shuffled, y_test_shuffled)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    return train_dataset, test_dataset, train_loader, test_loader, scaler, y_test, full_preprocessed_df_unscaled, y_feature_scaler_index, combined_loader_unshuffled, date_time_combined, device_ids_original_combined

def create_multivariate_model_for_feature(df: pd.DataFrame, y_feature: str='CO2', aggregation_level: str='quarter_hour', device: torch.device=get_device() ,window_size: int=5, epochs: int=1000, clean_data: bool=True, drop_columns: list=[]):
    """
    Creates a multivariate-non-sequential-model for the selected feature.
    Deprecated because the model is absolute garbage.
    args:   df: pd.DataFrame
            y_feature: str
            aggregation_level: str, one of "hour", "half_hour", "quarter_hour"
            window_size: int
            epochs: int
            clean_data: bool

    returns: nn.Module, StandardScaler

    return model, scaler
    """
    # Prepare the data for the model
    train_dataset, test_dataset, train_loader, test_loader, scaler, y_test = get_data_for_multivariate_forecast(df, y_feature, window_size, aggregation_level, clean_data=clean_data, drop_columns=drop_columns)
    # Train the model
    model = train_fcn_model(device, train_loader, test_loader, scaler, epochs)

    return model, scaler

def create_multivariate_transformer_model_for_feature(df: pd.DataFrame, y_feature: str='CO2', aggregation_level: str='quarter_hour', device: torch.device=get_device(), window_size: int=20, batch_size: int =get_batch_size(), epochs: int=1000, clean_data: bool=True, input_dim: int=None, d_model: int=64, nhead: int=4, num_layers: int=2, dropout_pe: float=0.25, dropout_encoder: float=0.25, learning_rate: float=0.001, drop_columns: list=[], overwrite: bool=True,selected_building: str='am'):
    """
    full pipeline to create a multi-variate transformer model for a selected feature.
    Currently best model. Doesnt use one-hot encoding (except of Semester, which has only 3 values), but embeddings to reduce dimensionality.

    args:   df: pd.DataFrame
            y_feature: str
            aggregation_level: str, one of "hour", "half_hour", "quarter_hour"
            window_size: int
            epochs: int
            clean_data: bool

    returns: nn.Module, StandardScaler, rmse (float), mae (float), mape (float), train_loss (float), val_loss (float), num_features (int)
    """
    # Prepare the data for the model
    train_dataset, test_dataset, train_loader, test_loader, scaler, y_test, full_preprocessed_df_unscaled, y_feature_scaler_index, combined_loader, date_time_combined, device_ids_original_combined = get_data_for_multivarate_sequential_forecast(df, y_feature, window_size, aggregation_level, clean_data=clean_data, batch_size=batch_size, drop_columns=drop_columns)
    # Get the first batch of the training data
    data, devices, labels = next(iter(train_loader))

    # Get the number of features from the data
    num_features = data.shape[2] + 1
    model_name = f'{aggregation_level}_{num_features}f_{window_size}ws_{selected_building}'
    save_dataframe(full_preprocessed_df_unscaled, model_name=model_name, overwrite=overwrite)
    save_scaler(scaler, model_name=model_name, overwrite=overwrite)
    
    # Train the model
    num_unique_devices = len(full_preprocessed_df_unscaled['device_id'].unique())
    model, train_loss, val_loss = train_transformer_model(device, train_loader, test_loader, scaler, epochs, input_dim=input_dim, d_model=d_model, nhead=nhead, num_layers=num_layers, dropout_pe=dropout_pe, dropout_encoder=dropout_encoder, learning_rate=learning_rate, y_feature_scaler_index=y_feature_scaler_index, num_devices=num_unique_devices)
    
    # Save the model
    model_name = 'transformer_multivariate_' + model_name
    save_model(model, y_feature=y_feature, model_name=model_name)

    # Evaluate the model
    rmse, mae, me, mape = evaluate_transformer_model(device, test_loader, model, scaler, y_test, y_feature_scaler_index=y_feature_scaler_index, input_dim=input_dim, combined_loader=combined_loader, date_time_combined=date_time_combined, y_feature=y_feature, window_size=window_size, device_ids_original_combined=device_ids_original_combined)

    return model, scaler, rmse, mae, me, mape, train_loss, val_loss, num_features

def fill_data_for_prediction(df: pd.DataFrame):
    """
    Probably not required anymore because we fill the data more precisely when creating the dataframe.
    fills missing datapoints using intpolation. THe rest is filles with bfill and ffill
    args:   df: pd.DataFrame

    returns: pd.DataFrame
    """
    # Fill missing values by interpolation except date_time_rounded
    df[[col for col in df.columns if col != 'date_time_rounded']] = df[[col for col in df.columns if col != 'date_time_rounded']].interpolate(method='linear', axis=0)
    
    df = df.fillna(method='bfill')
    df = df.fillna(method='ffill')
    return df

def predict_data_multivariate_transformer(model_name: str='transformer_multivariate', device: torch.device=get_device(), clean_data: bool=True, window_size: int=20, aggregation_level: str='quarter_hour', y_feature: str='CO2', batch_size: int=get_batch_size(), start_time: np.datetime64=None, prediction_count: int=1, selected_room: str=None, feature_count: int=26, selected_building: str='am') -> pd.DataFrame:
    """
    args:   model_name: str
            scaler_name: str
            dataframe_name: str
            device: torch.device
            clean_data: bool
            window_size: int
            aggregation_level: str, one of "hour", "half_hour", "quarter_hour"
            y_feature: str
            batch_size: int
            start_time: np.datetime64
            prediction_count: int

    returns: pd.DataFrame
    """

    full_model_name = aggregation_level + '_' + str(feature_count) + 'f' + '_' + str(window_size) + 'ws' + '_' + selected_building
    
    df = load_dataframe(model_name=full_model_name)
    scaler = load_scaler(model_name=full_model_name)
    model = load_model(model_name=model_name + '_' + full_model_name, y_feature=y_feature, device=device)

    df['device_id_cat'] = df['device_id'].astype('category').cat.codes
    df = df[df[f'device_id'] == f'hka-aqm-{selected_room}']
    df['device_id'] = df['device_id_cat']
    df.drop(['device_id_cat'], axis=1, inplace=True)
    df['date_time_rounded'] = pd.to_datetime(df['date_time_rounded'])

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
    
    start_time = pd.to_datetime(start_time).round(freq)
    date_midnight = pd.to_datetime(start_time.date())
    print("start_time: ", start_time)
    print("date_midnight: ", date_midnight)

    df_context = df[(df['date_time_rounded'] < date_midnight)].tail(window_size)
    df_date = df[(df['date_time_rounded'].dt.date == start_time.date())]
    df = pd.concat([df_context, df_date], ignore_index=True)
    #df = df[(df['date_time_rounded'] < start_time)].tail(window_size)

    #required_timestamps = pd.date_range(end=start_time, periods=window_size+1, freq=freq, closed='left')
    # timestamps from date_midnight - window_size * freq to date_midnight+1day
    required_timestamps = pd.date_range(start=date_midnight - window_size * pd.to_timedelta(freq), end=date_midnight + pd.to_timedelta(1, unit='D'), freq=freq, closed='left')

    # Create a dataframe from these timestamps
    df_timestamps = pd.DataFrame(required_timestamps, columns=['date_time_rounded'])
    
    # Create a dataframe to save predictions in. Columns are date_time_rounded and y_feature
    prediction_timestamps = pd.date_range(start=date_midnight, end=date_midnight + pd.to_timedelta(1, unit='D'), freq=freq, closed='left')
    df_predictions = pd.DataFrame({'date_time_rounded': prediction_timestamps, f'{y_feature}': np.nan})


    # Perform a left join to find missing timestamps in the original dataframe
    result_df = df_timestamps.merge(df, on='date_time_rounded', how='left')

    # check if result_df has nan values
    
    if result_df.isnull().values.any() and (result_df.shape[0] - result_df.dropna().shape[0]) < 0.75 * result_df.shape[0]:
        pass
        # result_df = fill_data_for_prediction(result_df)
    if result_df.dropna().shape[0] < window_size:
        # return empty dataframe
        # print(f'there are {result_df.shape[0] - result_df.dropna().shape[0]} rows with missing values')
        print('we cant predict anything because too many values are missing')
        return df_predictions
    
    columns_to_scale = [col for col in result_df.columns if col not in ['date_time_rounded', 'device_id', 'group']]
    y_feature_scaler_index = columns_to_scale.index(y_feature)
    print(y_feature_scaler_index)
    model.eval()
    with torch.no_grad():
        # create rolling window of 20 datepoints everytime over result_df
        for j in range(result_df.shape[0] - window_size):
            df_subset = result_df.iloc[j:j+window_size]
            for i in range(prediction_count):
                device_ids = df_subset.tail(20)['device_id'].values[-1]
                cols_to_drop = ['date_time_rounded', 'group', 'device_id'] if 'group' in df_subset.columns else ['date_time_rounded', 'device_id']
                df_input = df_subset.tail(window_size).drop(cols_to_drop, axis=1)
                if not df_input.isna().values.any():
                    df_input = scaler.transform(df_input)
                    input_data = torch.tensor(df_input, dtype=torch.float32).view(-1, window_size, df_input.shape[1])
                    device_ids_tensor = torch.tensor(device_ids, dtype=torch.long).view(-1)

                    prediction = model(input_data.to(device), device_ids_tensor.to(device))
                    prediction = prediction.cpu().numpy().reshape(-1, 1)
                    zeroes_for_scaler = np.zeros((prediction.shape[0], len(columns_to_scale)))

                    zeroes_for_scaler[:, y_feature_scaler_index] = prediction.flatten()  # Insert predicted values into the correct column
                    inverse_transformed = scaler.inverse_transform(zeroes_for_scaler)
                    predicted_unscaled = inverse_transformed[:, y_feature_scaler_index].round(2)
                else:
                    predicted_unscaled = np.nan
                new_timestamp = df_subset['date_time_rounded'].max() + pd.to_timedelta(freq)
                if i == prediction_count-1:
                    df_predictions.loc[df_predictions['date_time_rounded'] == new_timestamp, f'{y_feature}'] = predicted_unscaled
                
                # following lines are required when we want to predict multiple values for the future, not just 1
                new_row = pd.DataFrame({
                    'date_time_rounded': [new_timestamp],
                    y_feature: [predicted_unscaled]
                    # Add other columns here if necessary, filling with NaN or default values
                })
                # add output in a new row of the y_feature column of result_df and remove first line
                df_subset = pd.concat([df_subset, new_row], ignore_index=True)
                df_subset.fillna(method='ffill', inplace=True)
    
    return df_predictions

def create_multivariate_lstm_model_for_feature(df: pd.DataFrame, y_feature: str='CO2', aggregation_level: str='quarter_hour', device: torch.device=get_device(), window_size: int=20, batch_size: int=get_batch_size(), epochs: int=1000, clean_data: bool=True, input_dim: int=None, hidden_dim: int=64, num_layers: int=2, dropout: float=0.25, learning_rate: float=0.001, drop_columns: list=[], overwrite: bool=True,selected_building: str='am'):
    """
    full pipeline to create a multi-variate LSTM model for a selected feature.

    args:   df: pd.DataFrame
            y_feature: str
            aggregation_level: str, one of "hour", "half_hour", "quarter_hour"
            window_size: int
            epochs: int
            clean_data: bool

    returns: nn.Module, StandardScaler, rmse (float), mae (float), mape (float), train_loss (float), val_loss (float), num_features (int)
    """
    # Prepare the data for the model
    train_dataset, test_dataset, train_loader, test_loader, scaler, y_test, full_preprocessed_df_unscaled, y_feature_scaler_index, combined_loader, date_time_combined, device_ids_original_combined = get_data_for_multivarate_sequential_forecast(df, y_feature, window_size, aggregation_level, clean_data=clean_data, batch_size=batch_size, drop_columns=drop_columns)
    # Get the first batch of the training data
    data, devices, labels = next(iter(train_loader))

    # Get the number of features from the data
    num_features = data.shape[2] + 1
    model_name = f'{aggregation_level}_{num_features}f_{window_size}ws_{selected_building}'
    save_dataframe(full_preprocessed_df_unscaled, model_name=model_name, overwrite=overwrite)
    save_scaler(scaler, model_name=model_name)
    
    # Train the model
    num_unique_devices = len(full_preprocessed_df_unscaled['device_id'].unique())
    model, train_loss, val_loss = train_lstm_model(device, train_loader, test_loader, scaler, epochs, input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout, learning_rate=learning_rate, y_feature_scaler_index=y_feature_scaler_index, num_devices=num_unique_devices, device_embedding_dim=4)
    
    # Save the model
    model_name = 'lstm_multivariate_' + model_name
    save_model(model, y_feature=y_feature, model_name=model_name)

    # Evaluate the model
    rmse, mae, me, mape = evaluate_lstm_model(device, test_loader, model, scaler, y_test, y_feature_scaler_index=y_feature_scaler_index, input_dim=input_dim, combined_loader=combined_loader, date_time_combined=date_time_combined, y_feature=y_feature, window_size=window_size, device_ids_original_combined=device_ids_original_combined)

    return model, scaler, rmse, mae, me, mape, train_loss, val_loss, num_features

def predict_data_multivariate_LSTM(model_name: str='lstm_multivariate', device: torch.device=get_device(), clean_data: bool=True, window_size: int=20, aggregation_level: str='quarter_hour', y_feature: str='CO2', batch_size: int=get_batch_size(), start_time: np.datetime64=None, prediction_count: int=1, selected_room: str=None, feature_count: int=26,selected_building: str='am') -> pd.DataFrame:
    """
    args:   model_name: str
            scaler_name: str
            dataframe_name: str
            device: torch.device
            clean_data: bool
            window_size: int
            aggregation_level: str, one of "hour", "half_hour", "quarter_hour"
            y_feature: str
            batch_size: int
            start_time: np.datetime64
            prediction_count: int

    returns: pd.DataFrame
    """

    full_model_name = aggregation_level + '_' + str(feature_count) + 'f' + '_' + str(window_size) + 'ws' + '_' + selected_building
    df = load_dataframe(model_name=full_model_name)
    scaler = load_scaler(model_name=full_model_name)
    model = load_model(model_name=model_name + '_' + full_model_name, y_feature=y_feature, device=device)

    df['device_id_cat'] = df['device_id'].astype('category').cat.codes
    df = df[df[f'device_id'] == f'hka-aqm-{selected_room}']
    df['device_id'] = df['device_id_cat']
    df.drop(['device_id_cat'], axis=1, inplace=True)
    df['date_time_rounded'] = pd.to_datetime(df['date_time_rounded'])

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
    
    start_time = pd.to_datetime(start_time).round(freq)
    date_midnight = pd.to_datetime(start_time.date())
    print("start_time: ", start_time)
    print("date_midnight: ", date_midnight)

    df_context = df[(df['date_time_rounded'] < date_midnight)].tail(window_size)
    df_date = df[(df['date_time_rounded'].dt.date == start_time.date())]
    df = pd.concat([df_context, df_date], ignore_index=True)

    # timestamps from date_midnight - window_size * freq to date_midnight+1day
    required_timestamps = pd.date_range(start=date_midnight - window_size * pd.to_timedelta(freq), end=date_midnight + pd.to_timedelta(1, unit='D'), freq=freq, closed='left')

    # Create a dataframe from these timestamps
    df_timestamps = pd.DataFrame(required_timestamps, columns=['date_time_rounded'])
    
    # Create a dataframe to save predictions in. Columns are date_time_rounded and y_feature
    prediction_timestamps = pd.date_range(start=date_midnight, end=date_midnight + pd.to_timedelta(1, unit='D'), freq=freq, closed='left')
    df_predictions = pd.DataFrame({'date_time_rounded': prediction_timestamps, f'{y_feature}': np.nan})

    # Perform a left join to find missing timestamps in the original dataframe
    result_df = df_timestamps.merge(df, on='date_time_rounded', how='left')

    if result_df.dropna().shape[0] < window_size:
        print('we cant predict anything because too many values are missing')
        return df_predictions
    
    columns_to_scale = [col for col in result_df.columns if col not in ['date_time_rounded', 'device_id', 'group']]
    y_feature_scaler_index = columns_to_scale.index(y_feature)
    model.eval()
    with torch.no_grad():
        # create rolling window of 20 datepoints everytime over result_df
        for j in range(result_df.shape[0] - window_size):
            df_subset = result_df.iloc[j:j+window_size]
            for i in range(prediction_count):
                device_ids = df_subset.tail(20)['device_id'].values[-1]
                cols_to_drop = ['date_time_rounded', 'group', 'device_id'] if 'group' in df_subset.columns else ['date_time_rounded', 'device_id']
                df_input = df_subset.tail(window_size).drop(cols_to_drop, axis=1)
                if not df_input.isna().values.any():
                    df_input = scaler.transform(df_input)
                    input_data = torch.tensor(df_input, dtype=torch.float32).view(-1, window_size, df_input.shape[1])
                    device_ids_tensor = torch.tensor(device_ids, dtype=torch.long).view(-1)

                    prediction = model(input_data.to(device), device_ids_tensor.to(device))
                    prediction = prediction.cpu().numpy().reshape(-1, 1)
                    zeroes_for_scaler = np.zeros((prediction.shape[0], len(columns_to_scale)))

                    zeroes_for_scaler[:, y_feature_scaler_index] = prediction.flatten()  # Insert predicted values into the correct column
                    inverse_transformed = scaler.inverse_transform(zeroes_for_scaler)
                    predicted_unscaled = inverse_transformed[:, y_feature_scaler_index].round(2)
                else:
                    predicted_unscaled = np.nan
                new_timestamp = df_subset['date_time_rounded'].max() + pd.to_timedelta(freq)
                if i == prediction_count-1:
                    df_predictions.loc[df_predictions['date_time_rounded'] == new_timestamp, f'{y_feature}'] = predicted_unscaled
                
                # following lines are required when we want to predict multiple values for the future, not just 1
                new_row = pd.DataFrame({
                    'date_time_rounded': [new_timestamp],
                    y_feature: [predicted_unscaled]
                    # Add other columns here if necessary, filling with NaN or default values
                })
                # add output in a new row of the y_feature column of result_df and remove first line
                df_subset = pd.concat([df_subset, new_row], ignore_index=True)
                df_subset.fillna(method='ffill', inplace=True)
    
    return df_predictions



