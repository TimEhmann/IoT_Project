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
import joblib

def get_device() -> torch.device:
    has_mps = torch.backends.mps.is_built()
    return "mps" if has_mps else "cuda" if torch.cuda.is_available() else "cpu"

def get_batch_size() -> int:
    if get_device() == "mps":
        return 128
    else:
        return 1500

def save_model(model: nn.Module, model_path: str=None, y_feature: str='CO2'):
    if model_path is None:
        model_path = f'models/transformer_model_{y_feature}_v1.pth'
    if os.path.isfile(model_path):
        version = 2
        while os.path.isfile(f'models/transformer_model_{y_feature}_v{version}.pth'):
            version += 1
        
        model_path = f'models/transformer_model_{y_feature}_v{version}.pth'

    torch.save(model.state_dict(), model_path)

def save_scaler(scaler: StandardScaler, scaler_path: str=None, y_feature: str='CO2'):
    if scaler_path is None:
        scaler_path = f'models/transformer_scaler_{y_feature}_v1.pth'
    if os.path.isfile(scaler_path):
        version = 2
        while os.path.isfile(f'models/transformer_scaler_{y_feature}_v{version}.pth'):
            version += 1
        
        scaler_path = f'models/transformer_scaler_{y_feature}_v{version}.pth'
    
    torch.save(scaler, scaler_path)

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
    # use data cleaning from prepare_data_for_plot function
    df_cpy = prepare_data_for_plot(df, clean_data)

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

def train_transformer_model(device, train_loader: DataLoader, test_loader: DataLoader, scaler: StandardScaler, epochs: int=1000):
    model = TransformerModel().to(device)
    # Train the model
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
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

def evaluate_transformer_model(device, test_loader: DataLoader, model: nn.Module, scaler: StandardScaler, y_test: torch.Tensor):
    # Evaluation
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            x_batch, y_batch = batch
            x_batch = x_batch.to(device)
            outputs = model(x_batch)
            predictions.extend(outputs.squeeze().tolist())

    # print dataframe of actual vs predicted with inverse transform
    print(pd.DataFrame({'actual': scaler.inverse_transform(y_test.cpu().numpy().reshape(-1, 1)).flatten(), 'predicted': scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()}))

    rmse = np.sqrt(np.mean((scaler.inverse_transform(np.array(predictions).reshape(-1, 1)) - scaler.inverse_transform(y_test.numpy().reshape(-1, 1)))**2))
    print(f"Score (RMSE): {rmse:.4f}")

def load_model(y_feature: str='CO2', model_path: str=None, device: torch.device=get_device()) -> nn.Module:
    if model_path is None:
        version = 1
        while os.path.isfile(f"models/transformer_model_{y_feature}_v{version}.pth"):
            version += 1
        
        model_path = f"models/transformer_model_{y_feature}_v{version-1}.pth"
        print("loading latest model: " + model_path)
    else:
        print("loading:" + model_path)
    model = TransformerModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model

def load_scaler(y_feature: str='CO2', scaler_path: str=None) -> StandardScaler:
    if scaler_path is None:
        version = 1
        while os.path.isfile(f"models/transformer_scaler_{y_feature}_v{version}.pth"):
            version += 1
        
        scaler_path = f"models/transformer_scaler_{y_feature}_v{version-1}.pth"
        print("loading latest scaler: " + scaler_path)
    else:
        print("loading:" + scaler_path)
    scaler = torch.load(scaler_path)

    return scaler

def get_data_for_prediction(df: pd.DataFrame, scaler: StandardScaler, clean_data: bool=True, window_size: int=20, aggregation_level: str='quarter_hour', y_feature: str='CO2') -> pd.DataFrame:
    """
    args:   df: pd.DataFrame
            clean_data: bool

    returns: pd.DataFrame
    """
    df_cpy = prepare_data_for_plot(df, clean_data)

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
    save_model(model, y_feature=y_feature)
    save_scaler(scaler, y_feature=y_feature)

    return model, scaler
