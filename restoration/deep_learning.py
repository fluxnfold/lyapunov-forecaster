import numpy as np
import pandas as pd
from typing import Tuple, Optional
import warnings

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Deep learning methods will not work.")

def get_device() -> 'torch.device':
  
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for device detection")
  
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

class TimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, mask: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.mask = torch.BoolTensor(mask)
  
    def __len__(self):
        return len(self.X)
  
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.mask[idx]

class CNN1DImputer(nn.Module):
  
  
    def __init__(self, input_dim: int = 1, hidden_dims: list = [64, 128, 64], 
                 kernel_size: int = 5):
        super(CNN1DImputer, self).__init__()
      
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Conv1d(in_dim, hidden_dim, kernel_size, 
                                   padding=kernel_size//2))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            in_dim = hidden_dim
        layers.append(nn.Conv1d(in_dim, input_dim, kernel_size=1))
        self.network = nn.Sequential(*layers)
  
    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.network(x)
        out = out.transpose(1, 2)
        return out

class LSTMImputer(nn.Module):
  
  
    def __init__(self, input_dim: int = 1, hidden_dim: int = 128, 
                 num_layers: int = 2, dropout: float = 0.2):
        super(LSTMImputer, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, input_dim)
  
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out)
        return out

def prepare_windows(
    data: np.ndarray,
    window_size: int = 50,
    stride: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    n_samples = data.shape[0]
    n_windows = (n_samples - window_size) // stride + 1
    windows = []
    indices = []
    for i in range(n_windows):
        start = i * stride
        end = start + window_size
        windows.append(data[start:end])
        indices.append(start)
  
    return np.array(windows), np.array(indices)

def cnn_imputation(
    df: pd.DataFrame,
    col: str = 'acc_x',
    window_size: int = 50,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    device: Optional[str] = None,
    verbose: bool = False
) -> pd.DataFrame:
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for CNN imputation")
    if device is None:
        dev = get_device()
    else:
        dev = torch.device(device)
    if verbose:
        print(f"CNN imputation using device: {dev}")
    df_copy = df.copy()
    signal = df_copy[col].values.copy()
    missing_mask = np.isnan(signal)
    if not missing_mask.any():
        return df_copy
    signal_filled = signal.copy()
    signal_filled[missing_mask] = 0
    windows, indices = prepare_windows(signal_filled, window_size)
    X_train = windows.copy()
    y_train = windows.copy()

    for i in range(len(X_train)):
        mask = np.random.rand(window_size, 1) < 0.2
        X_train[i][mask] = 0
    train_mask = np.zeros((len(X_train), window_size, 1), dtype=bool)
    dataset = TimeSeriesDataset(X_train, y_train, train_mask)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = CNN1DImputer(input_dim=1).to(dev)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y, _ in dataloader:
          
            batch_X = batch_X.to(dev)
            batch_y = batch_y.to(dev)
          
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
      
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader):.6f}")
    model.eval()
    with torch.no_grad():
        windows_tensor = torch.FloatTensor(windows).to(dev)
        predictions = model(windows_tensor).cpu().numpy()
    reconstructed = np.zeros_like(signal)
    counts = np.zeros_like(signal)
    for i, start_idx in enumerate(indices):
        end_idx = start_idx + window_size
        reconstructed[start_idx:end_idx] += predictions[i, :, 0]
        counts[start_idx:end_idx] += 1
    reconstructed = reconstructed / np.maximum(counts, 1)
    df_copy.loc[missing_mask, col] = reconstructed[missing_mask]
    return df_copy

def lstm_imputation(
    df: pd.DataFrame,
    col: str = 'acc_x',
    window_size: int = 50,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    hidden_dim: int = 128,
    device: Optional[str] = None,
    verbose: bool = False
) -> pd.DataFrame:
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for LSTM imputation")
    
    if device is None:
        dev = get_device()
    else:
        dev = torch.device(device)
    if verbose:
        print(f"LSTM imputation using device: {dev}")
    
    df_copy = df.copy()
    signal = df_copy[col].values.copy()
    missing_mask = np.isnan(signal)
    if not missing_mask.any():
        return df_copy
    signal_filled = signal.copy()
    signal_filled[missing_mask] = 0
    windows, indices = prepare_windows(signal_filled, window_size)
    X_train = windows.copy()
    y_train = windows.copy()
    for i in range(len(X_train)):
        mask = np.random.rand(window_size, 1) < 0.2
        X_train[i][mask] = 0
    train_mask = np.zeros((len(X_train), window_size, 1), dtype=bool)
    dataset = TimeSeriesDataset(X_train, y_train, train_mask)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = LSTMImputer(input_dim=1, hidden_dim=hidden_dim).to(dev)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y, _ in dataloader:
            batch_X = batch_X.to(dev)
            batch_y = batch_y.to(dev)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader):.6f}")
    model.eval()
    with torch.no_grad():
        windows_tensor = torch.FloatTensor(windows).to(dev)
        predictions = model(windows_tensor).cpu().numpy()
    reconstructed = np.zeros_like(signal)
    counts = np.zeros_like(signal)
    for i, start_idx in enumerate(indices):
        end_idx = start_idx + window_size
        reconstructed[start_idx:end_idx] += predictions[i, :, 0]
        counts[start_idx:end_idx] += 1
    reconstructed = reconstructed / np.maximum(counts, 1)
    df_copy.loc[missing_mask, col] = reconstructed[missing_mask]
    return df_copy
