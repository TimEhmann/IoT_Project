import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
class TransformerModel(nn.Module):
    def __init__(self, input_dim=1, d_model=64, nhead=4, num_layers=2, dropout_pe=0.25, dropout_encoder=0.25, num_devices=1, device_embedding_dim=4):
        super(TransformerModel, self).__init__()

        self.device_embedding = nn.Embedding(num_devices, device_embedding_dim)
        self.encoder = nn.Linear(input_dim + device_embedding_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout_pe)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dropout=dropout_encoder)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder1 = nn.Linear(d_model, d_model//2)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.1)
        self.decoder2 = nn.Linear(d_model//2, 1)

    def forward(self, x, device_ids):
        device_embeds = self.device_embedding(device_ids)
        device_embeds = device_embeds.unsqueeze(1).expand(-1, x.size(1), -1)
        x = torch.cat((x, device_embeds), dim=-1)
        x = self.encoder(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.decoder1(x[:, -1, :])
        x = self.relu(x)
        x = self.dropout(x)
        x = self.decoder2(x)
        return x
    
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, num_devices, device_embedding_dim, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Device embedding layer
        self.device_embedding = nn.Embedding(num_devices, device_embedding_dim)
        
        # LSTM input dimensions need to account for the additional embedding dimension
        self.lstm = nn.LSTM(input_dim + device_embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, device_ids):
        device_embeds = self.device_embedding(device_ids)
        device_embeds = device_embeds.unsqueeze(1).expand(-1, x.size(1), -1)
        x = torch.cat((x, device_embeds), dim=-1)
        
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Take the last output
        return out

    
class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.fc1 = nn.Linear(47, 128)  # Input layer
        self.fc2 = nn.Linear(128, 128)  # Hidden layer
        self.fc3 = nn.Dropout(0.1)
        self.fc4 = nn.Linear(128, 128)  # Hidden layer
        self.fc5 = nn.Dropout(0.1)
        self.fc6 = nn.Linear(128, 64)
        self.fc7 = nn.Linear(64, 1)    # Output layer
        self.relu = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        x = self.relu(self.fc6(x))
        x = self.fc7(x)
        return x

