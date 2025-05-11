import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class StockDataset(Dataset):
    def __init__(
        self, 
        data_path: str, 
        features: List[str], 
        target_column: str,
        window_size: int = 30,
        transform: Optional[callable] = None
    ):
        self.data_path = data_path
        self.features = features
        self.target_column = target_column
        self.window_size = window_size
        self.transform = transform
        
        try:
            self.load_data()
            self.prepare_sequences()
        except Exception as e:
            logger.error(f"Error loading stock data: {e}")
            raise
    
    def load_data(self):
        try:
            self.raw_data = pd.read_csv(self.data_path)
            self.raw_data.sort_values('date', inplace=True)
            self.raw_data.fillna(method='ffill', inplace=True)
        except FileNotFoundError:
            logger.error(f"Data file not found: {self.data_path}")
            raise
    
    def prepare_sequences(self):
        X, y = [], []
        for i in range(len(self.raw_data) - self.window_size):
            window = self.raw_data.iloc[i:i+self.window_size][self.features]
            target = self.raw_data.iloc[i+self.window_size][self.target_column]
            
            X.append(window.values)
            y.append(target)
        
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.transform:
            return self.transform(self.X[idx]), self.y[idx]
        return self.X[idx], self.y[idx]

def create_stock_dataloader(
    data_path: str,
    features: List[str],
    target_column: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    try:
        dataset = StockDataset(
            data_path=data_path, 
            features=features, 
            target_column=target_column
        )
        
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        logger.info(f"DataLoader created with {len(dataset)} samples")
        return dataloader
    
    except Exception as e:
        logger.error(f"Failed to create DataLoader: {e}")
        raise

def get_stock_data_stats(data_path: str) -> Dict:
    try:
        df = pd.read_csv(data_path)
        return {
            'total_samples': len(df),
            'date_range': {
                'start': df['date'].min(),
                'end': df['date'].max()
            },
            'feature_stats': df.describe().to_dict()
        }
    except Exception as e:
        logger.error(f"Error computing data statistics: {e}")
        return {}