import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List, Dict

class StockDataset(Dataset):
    def __init__(
        self, 
        features: torch.Tensor, 
        targets: torch.Tensor
    ):
        self.features = features
        self.targets = targets

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]

class StockDataLoader:
    def __init__(
        self, 
        data_path: str, 
        window_size: int = 30, 
        predict_days: int = 5,
        test_ratio: float = 0.2
    ):
        self.data_path = data_path
        self.window_size = window_size
        self.predict_days = predict_days
        self.test_ratio = test_ratio
        self.scaler = MinMaxScaler()

    def load_data(self) -> Dict[str, torch.Tensor]:
        df = pd.read_csv(self.data_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        features = df[['Open', 'High', 'Low', 'Close', 'Volume']].values
        features_scaled = self.scaler.fit_transform(features)

        X, y = self._create_sequences(features_scaled)
        
        train_size = int(len(X) * (1 - self.test_ratio))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        return {
            'X_train': torch.FloatTensor(X_train),
            'y_train': torch.FloatTensor(y_train),
            'X_test': torch.FloatTensor(X_test),
            'y_test': torch.FloatTensor(y_test)
        }

    def _create_sequences(
        self, 
        data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(len(data) - self.window_size - self.predict_days + 1):
            X.append(data[i:i+self.window_size])
            y.append(data[i+self.window_size:i+self.window_size+self.predict_days, 3])
        
        return np.array(X), np.array(y)

    def get_dataloader(
        self, 
        X: torch.Tensor, 
        y: torch.Tensor, 
        batch_size: int = 32, 
        shuffle: bool = True
    ) -> DataLoader:
        dataset = StockDataset(X, y)
        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            pin_memory=True
        )