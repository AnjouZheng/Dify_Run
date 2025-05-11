import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import logging
from typing import Dict, Any
from models.lstm_model import LSTMStockPredictor
from models.transformer_model import TransformerStockPredictor
from src.data_loader import StockDataset
from src.preprocess import preprocess_data
from src.feature_engineering import create_features
import yaml
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def select_model(model_type: str, input_dim: int, config: Dict[str, Any]):
    if model_type == 'lstm':
        return LSTMStockPredictor(input_dim, **config['lstm_params'])
    elif model_type == 'transformer':
        return TransformerStockPredictor(input_dim, **config['transformer_params'])
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def train_model(
    model: nn.Module, 
    train_loader: DataLoader, 
    val_loader: DataLoader, 
    config: Dict[str, Any]
) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    best_val_loss = float('inf')
    
    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                val_loss += criterion(outputs, batch_y).item()
            
            val_loss /= len(val_loader)
        
        logging.info(f'Epoch {epoch+1}/{config["epochs"]}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join('models/trained_models', f'{config["model_type"]}_best.pth'))

def main():
    config_path = 'config/model_config.yaml'
    config = load_config(config_path)
    
    raw_data = np.load('data/raw/stocks_raw.csv')
    processed_data = preprocess_data(raw_data)
    features = create_features(processed_data)
    
    train_dataset = StockDataset(features, train=True)
    val_dataset = StockDataset(features, train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    
    model = select_model(config['model_type'], features.shape[1], config)
    
    train_model(model, train_loader, val_loader, config)

if __name__ == "__main__":
    main()